"""向量化对局采样：N 局同时推进，神经网络决策合批一次前向。

核心思想（步进锁存调度）
------------------------
引擎内部零改动，只在调度层做文章：

1. 同时开 N 局，每局各自"快进"——教师决策、强制阶段内联执行，
   直到遇到网络座位的决策点（或终局）；
2. 把所有处于决策点的观测堆成一个 batch，**一次前向**拿到全部动作分布；
3. 逐局采样动作并 step 一步，回到 1。

实测收益（202K 参数模型，CPU）：
* 单条推理 0.86ms/样本 → batch=64 时 0.041ms/样本（21 倍）；
* profile 显示网络前向占一局总耗时约 53%，批量化几乎消灭这块；
* 配合 ``fan.py`` 的胡牌/听牌缓存（引擎热点，占约 27%），
  整体采集速度提升约 2-3 倍，多进程再乘约 1.4 倍。

对外提供：
* ``collect_vectorized`` —— 训练用采样（与 selfplay.collect_games 兼容）；
* ``collect_parallel``   —— 多进程包装；
* ``run_duplicate``      —— 复式赛（固定座次轮换），循环赛/评测用。
"""
from __future__ import annotations

import multiprocessing as mp
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from zhuoji.bots import TEACHER_STYLES, HeuristicBot  # noqa: E402
from zhuoji.encoder import encode, index_to_action  # noqa: E402

HEAD_NAMES = ["discard", "self_kong", "respond"]
MAX_STEPS = 4000
ACT_WIDTH = 28          # 动作头最大宽度（discard）
STYLES = list(TEACHER_STYLES.keys())


def _resolve_game_classes(rules: str):
    if rules == "dushan":
        from zhuoji.dushan import DushanConfig, DushanGame
        return DushanGame, DushanConfig
    from zhuoji import RulesConfig, ZhuojiGame
    return ZhuojiGame, RulesConfig


class _GameSlot:
    """一局在调度器里的状态。

    座位只有两种执行方式：
    * ``net`` —— 网络决策，进入批处理队列（``net_keys[p]`` 指定模型键，
      ``models`` 给出键到模型实例的映射，支持一局内多个不同网络）；
    * ``bot`` —— 启发式/随机决策，内联执行。
    """

    __slots__ = ("game", "gi", "seats", "bots", "net_keys", "models", "steps")

    def __init__(self, game, gi, seats, bots, net_keys, models):
        self.game = game
        self.gi = gi
        self.seats = seats        # list["net"|"bot"]
        self.bots = bots          # dict[p, bot]
        self.net_keys = net_keys  # list[p, key]
        self.models = models      # dict[key, model]
        self.steps = 0


def _make_seats(rng, teacher_prob, snapshots, past_prob):
    """决定一局四个座位的归属（与 selfplay._seat_config 同口径）。"""
    r = rng.random()
    bots: dict = {}
    net_keys = ["net"] * 4
    models: dict = {}
    if snapshots and r < past_prob:
        # 部分座位由历史快照执刀（league 对手池），其余主模型
        k = int(rng.integers(1, 3))
        idxs = set(rng.choice(4, size=k, replace=False).tolist())
        seats = ["net"] * 4
        for i in idxs:
            snap = snapshots[int(rng.integers(0, len(snapshots)))]
            key = f"snap{id(snap)}"
            net_keys[i] = key
            models[key] = snap
    elif r < past_prob + teacher_prob:
        k = int(rng.integers(1, 3))
        idxs = set(rng.choice(4, size=k, replace=False).tolist())
        seats = ["bot" if i in idxs else "net" for i in range(4)]
    else:
        seats = ["net"] * 4
    for i, kind in enumerate(seats):
        if kind == "bot":
            st = STYLES[int(rng.integers(0, len(STYLES)))]
            bots[i] = HeuristicBot(seed=int(rng.integers(1, 10 ** 9)),
                                   weights=TEACHER_STYLES[st])
    return seats, bots, net_keys, models


def _advance(slot: _GameSlot):
    """快进一局直到网络决策点。

    返回 ``(player, obs, model_key, model)``；终局/步数超限返回 ``None``。
    """
    game = slot.game
    while game.phase.value != "over":
        p = game.actor()
        if slot.seats[p] == "net":
            key = slot.net_keys[p]
            return p, encode(game, p), key, slot.models[key]
        game.step(slot.bots[p].choose(game))
        slot.steps += 1
        if slot.steps >= MAX_STEPS:
            break
    return None


def _sample_actions(logits_list: list, masks_list: list,
                    temperature: float, rng: np.random.Generator) -> np.ndarray:
    """按温度从各决策点采样动作索引（宽度不齐，先补到 ACT_WIDTH）。"""
    B = len(logits_list)
    lg = torch.full((B, ACT_WIDTH), -1e9)
    for i, lg_i in enumerate(logits_list):
        lg[i, :lg_i.shape[0]] = lg_i
    m = torch.from_numpy(np.stack(masks_list)).bool()
    lg[~m] = -1e9
    if temperature <= 1e-6:
        return lg.argmax(dim=-1).numpy()
    probs = torch.softmax(lg / temperature, dim=-1).numpy()
    probs = np.clip(probs, 0, None)
    probs /= probs.sum(axis=1, keepdims=True)
    return np.array([rng.choice(ACT_WIDTH, p=row) for row in probs])


def _batch_decide(items, temperature, rng):
    """对同模型的若干决策点做一次合批前向并采样。

    ``items``: list[(slot, p, obs, model)]。
    返回 list[(slot, p, obs, action_idx)]。
    """
    tiles = torch.from_numpy(
        np.stack([it[2].tiles for it in items]).astype(np.float32))
    glob = torch.from_numpy(
        np.stack([it[2].glob for it in items]).astype(np.float32))
    model = items[0][3]
    with torch.no_grad():
        out = model(tiles, glob)
    logits_list = [out[it[2].head][i][:it[2].mask.shape[0]]
                   for i, it in enumerate(items)]
    masks_list = [np.pad(it[2].mask.astype(np.float32),
                         (0, ACT_WIDTH - it[2].mask.shape[0]))
                  for it in items]
    acts = _sample_actions(logits_list, masks_list, temperature, rng)
    return [(it[0], it[1], it[2], int(a)) for it, a in zip(items, acts)]


def collect_vectorized(model, n_games: int, seed0: int, temperature: float,
                       reward_scale: float, teacher_prob: float,
                       threads: int = 4, snapshots=None, past_prob: float = 0.3,
                       game_cls=None, config_cls=None, want_traj: bool = True):
    """向量化采集 ``n_games`` 局。返回与 selfplay.collect_games 兼容的结果。

    ``want_traj=False`` 时只关心对局结果（评测用），不记录训练样本。
    """
    if game_cls is None:
        from zhuoji import RulesConfig, ZhuojiGame
        game_cls, config_cls = ZhuojiGame, RulesConfig
    torch.set_num_threads(threads)
    rng = np.random.default_rng(seed0)

    snapshots = list(snapshots or [])
    traj: list = []
    rewards = np.zeros((n_games, 4), dtype=np.float32)
    t0 = time.time()

    slots: list[_GameSlot] = []
    for gi in range(n_games):
        seats, bots, net_keys, models = _make_seats(rng, teacher_prob,
                                                    snapshots, past_prob)
        models.setdefault("net", model)   # 主模型也必须登记，否则 _advance 取不到
        game = game_cls(config_cls(), seed=int(seed0 + gi * 6151))
        slots.append(_GameSlot(game, gi, seats, bots, net_keys, models))

    active = list(range(n_games))
    while active:
        pend: dict = {}   # model_key -> list[(slot, p, obs, model)]
        nxt: list = []
        for gi in active:
            slot = slots[gi]
            r = _advance(slot)
            if r is None:
                rewards[gi] = [d / reward_scale for d in slot.game.score_delta]
            else:
                p, obs, key, mdl = r
                pend.setdefault(key, []).append((slot, p, obs, mdl))
                nxt.append(gi)
        active = nxt
        if not pend:
            break

        for key, items in pend.items():
            for slot, p, obs, idx in _batch_decide(items, temperature, rng):
                if want_traj and key == "net":
                    m = np.zeros(ACT_WIDTH, dtype=np.float32)
                    m[:obs.mask.shape[0]] = obs.mask.astype(np.float32)
                    traj.append((obs.tiles, obs.glob, m,
                                 HEAD_NAMES.index(obs.head), idx, p, slot.gi))
                slot.game.step(index_to_action(obs.head, idx, slot.game))
                slot.steps += 1

    dt = time.time() - t0
    return traj, rewards, dt


# ---------------------------------------------------------------------------
# 复式赛（固定座次轮换）
# ---------------------------------------------------------------------------
def _make_bot(kind: str, seed: int):
    """按风格名构造启发式/随机 bot（模块级函数，保证 spawn 可 pickle）。"""
    if kind == "random":
        from zhuoji.bots import RandomBot
        return RandomBot(seed=seed)
    return HeuristicBot(seed=seed, weights=TEACHER_STYLES[kind])


def run_duplicate(model_bank, seat_specs, deals: int, seed0: int,
                  game_cls, config_cls, threads: int = 2):
    """复式对局：``deals`` 副牌 × 4 座位轮换，网络决策按模型合批。

    ``model_bank``: dict[key -> model]；
    ``seat_specs``: 长度 4，每项 ``("net", key)`` 或 ``("bot", kind)``，
    ``kind`` 是启发式风格名（如 "balanced"）或 "random"——每副牌每座位
    用 ``_make_bot(kind, seed)`` 新建实例，保证确定性。

    返回 records 列表，每项 ``(r, shift, deltas4, rtype, winner, loser,
    is_tsumo, fan)``——座位为局部座位 0..3，策略归属由调用方按
    ``seat_specs[(s + shift) % 4]`` 还原（与 play_duplicate 同口径）。
    """
    torch.set_num_threads(threads)
    rng = np.random.default_rng(seed0)
    results: list = [None] * (deals * 4)
    # 分块推进：每块 16 副 × 4 轮换 = 64 局同时开（合批收益仍在），
    # 块结束即释放棋局对象，避免整桌几百局常驻内存。
    CHUNK = 16
    for c0 in range(0, deals, CHUNK):
        c1 = min(c0 + CHUNK, deals)
        slots: list[_GameSlot] = []
        metas: list = []
        for r in range(c0, c1):
            for shift in range(4):
                seed = int(seed0 + r * 1013 + shift)
                game = game_cls(config_cls(), seed=seed)
                seats = ["bot"] * 4
                bots: dict = {}
                net_keys = ["net"] * 4
                models: dict = {}
                for s in range(4):
                    spec = seat_specs[(s + shift) % 4]
                    if spec[0] == "net":
                        seats[s] = "net"
                        net_keys[s] = spec[1]
                        models[spec[1]] = model_bank[spec[1]]
                    else:
                        bots[s] = _make_bot(spec[1], seed * 4 + s)
                slots.append(_GameSlot(game, len(slots), seats, bots, net_keys, models))
                metas.append((r, shift))

        base = c0 * 4
        active = list(range(len(slots)))
        while active:
            pend: dict = {}
            nxt = []
            for gi in active:
                slot = slots[gi]
                adv = _advance(slot)
                if adv is None:
                    g = slot.game
                    res = getattr(g, "result", None) or {}
                    deltas = [float(d) for d in g.score_delta]
                    winner = res.get("winner")
                    loser = res.get("loser")
                    results[base + gi] = (metas[gi][0], metas[gi][1], deltas,
                                          res.get("type", "unknown"),
                                          -1 if winner is None else int(winner),
                                          -1 if loser is None else int(loser),
                                          bool(res.get("is_tsumo", False)),
                                          float(res.get("fan", 0.0) or 0.0))
                    slot.game = None          # 终局即释放
                    slot.bots = {}
                    slot.models = {}
                else:
                    p, obs, key, mdl = adv
                    pend.setdefault(key, []).append((slot, p, obs, mdl))
                    nxt.append(gi)
            active = nxt
            if not pend:
                break
            for key, items in pend.items():
                for slot, p, obs, idx in _batch_decide(items, 0.0, rng):
                    slot.game.step(index_to_action(obs.head, idx, slot.game))
                    slot.steps += 1
        del slots
    return [r for r in results if r is not None]


# ---------------------------------------------------------------------------
# 多进程包装（Windows 用 spawn，worker 必须是模块级函数、参数可 pickle）
# ---------------------------------------------------------------------------
def _build_net(nc, sd):
    from zhuoji.net import NetConfig, build_model
    m = build_model(NetConfig(nc["channels"], nc["blocks"], nc["hidden"]))
    m.load_state_dict(sd)
    return m.eval()


def _worker(args):
    (state_dict, nc, n_games, seed0, temperature, reward_scale, teacher_prob,
     past_prob, snapshots_sd, rules, want_traj) = args
    game_cls, config_cls = _resolve_game_classes(rules)
    torch.set_num_threads(2)
    model = _build_net(nc, state_dict)
    snaps = [_build_net(nc, sd) for sd in (snapshots_sd or [])]
    with torch.no_grad():
        traj, rewards, dt = collect_vectorized(
            model, n_games, seed0, temperature, reward_scale, teacher_prob,
            threads=2, snapshots=snaps, past_prob=past_prob,
            game_cls=game_cls, config_cls=config_cls, want_traj=want_traj)
    if want_traj and traj:
        tiles = np.stack([t[0] for t in traj])
        glob = np.stack([t[1] for t in traj])
        mask = np.stack([t[2] for t in traj])
        head = np.array([t[3] for t in traj], dtype=np.int64)
        act = np.array([t[4] for t in traj], dtype=np.int64)
        ply = np.array([t[5] for t in traj], dtype=np.int64)
        gi = np.array([t[6] for t in traj], dtype=np.int64)
        return tiles, glob, mask, head, act, ply, gi, rewards, dt
    return None, None, None, None, None, None, None, rewards, dt


def collect_parallel(state_dict, nc, n_games: int, seed0: int, temperature: float,
                     reward_scale: float, teacher_prob: float, workers: int = 4,
                     past_prob: float = 0.3, snapshots_sd=None, rules: str = "zhuoji",
                     want_traj: bool = True, pool=None):
    """多进程向量化采集：``n_games`` 均分给 ``workers`` 个进程。

    ``pool`` 传入已建好的 ``multiprocessing.Pool`` 可复用进程，
    避免训练循环里每轮重新 spawn（Windows 下每次约 1-2 秒开销）。
    """
    if workers <= 1:
        tiles, glob, mask, head, act, ply, gi, rewards, dt = _worker(
            (state_dict, nc, n_games, seed0, temperature, reward_scale,
             teacher_prob, past_prob, snapshots_sd or [], rules, want_traj))
        if tiles is not None:
            traj = [(t, g, m, int(h), int(a), int(p), int(x))
                    for t, g, m, h, a, p, x in zip(tiles, glob, mask, head, act, ply, gi)]
        else:
            traj = []
        return traj, rewards, dt
    per = [n_games // workers] * workers
    for i in range(n_games - sum(per)):
        per[i] += 1
    tasks = [(state_dict, nc, k, seed0 + w * 102871, temperature,
              reward_scale, teacher_prob, past_prob,
              snapshots_sd or [], rules, want_traj)
             for w, k in enumerate(per) if k > 0]
    ctx = mp.get_context("spawn")
    if pool is not None:
        results = pool.map(_worker, tasks)
    else:
        with ctx.Pool(processes=len(tasks)) as p:
            results = p.map(_worker, tasks)

    rewards = np.zeros((n_games, 4), dtype=np.float32)
    tiles_all, glob_all, mask_all = [], [], []
    head_all, act_all, ply_all, gi_all = [], [], [], []
    off = 0
    dt = 0.0
    for tiles, glob, mask, head, act, ply, gi, rw, wdt in results:
        k = rw.shape[0]
        rewards[off:off + k] = rw
        if tiles is not None:
            tiles_all.append(tiles)
            glob_all.append(glob)
            mask_all.append(mask)
            head_all.append(head)
            act_all.append(act)
            ply_all.append(ply)
            gi_all.append(gi + off)
        off += k
        dt = max(dt, wdt)   # 并行耗时取最慢 worker
    if tiles_all:
        traj = [(t, g, m, int(h), int(a), int(p), int(x))
                for t, g, m, h, a, p, x in zip(
                    np.concatenate(tiles_all), np.concatenate(glob_all),
                    np.concatenate(mask_all), np.concatenate(head_all),
                    np.concatenate(act_all), np.concatenate(ply_all),
                    np.concatenate(gi_all))]
    else:
        traj = []
    return traj, rewards, dt


def _table_worker(args):
    """循环赛 worker：跑若干张牌桌的复式赛，返回对局记录。"""
    (table_ids, seat_specs, sd_bank, nc, deals, seed0, rules) = args
    game_cls, config_cls = _resolve_game_classes(rules)
    bank = {k: _build_net(nc, sd) for k, sd in sd_bank.items()}
    torch.set_num_threads(2)
    out = []
    with torch.no_grad():
        for tid in table_ids:
            recs = run_duplicate(bank, seat_specs[tid], deals, seed0,
                                 game_cls, config_cls, threads=2)
            for rec in recs:
                out.append((tid,) + rec)
    return out
