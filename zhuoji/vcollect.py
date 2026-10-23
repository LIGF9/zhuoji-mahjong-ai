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
import torch.nn.functional as F

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

    __slots__ = ("game", "gi", "seats", "bots", "net_keys", "models", "steps",
                 "enc_version", "enc_packs")

    def __init__(self, game, gi, seats, bots, net_keys, models,
                 enc_version: int = 1, enc_packs=None):
        self.game = game
        self.gi = gi
        self.seats = seats        # list["net"|"bot"]
        self.bots = bots          # dict[p, bot]
        self.net_keys = net_keys  # list[p, key]
        self.models = models      # dict[key, model]
        self.steps = 0
        self.enc_version = int(enc_version)   # 状态编码器版本（P1 的 v1/v2）
        self.enc_packs = enc_packs            # 启用的 P1 特征包（None = 全开）


def _make_seats(rng, teacher_prob, snapshots, past_prob,
                master_prob: float = 0.0, master_key: str = "master",
                snap_temp: float = 0.0):
    """决定一局四个座位的归属（与 selfplay._seat_config 同口径）。

    M2-A · A5 对手池重构（2026-10-14）
    ---------------------------------
    旧版把 master 放在 ``snapshots`` 里：一旦池子里有 8 份自快照，master 作为
    **最强对手**只能分到 past_prob 分支的 1/9 出场率。现在 master 从快照池里**拿出来**，
    作为独立的一等对手分支，权重与自快照份数彻底解耦。

    ⚠️ 千万不要把"座位级独立权重"理解成「逐座位独立掷骰」—— 实测（2026-10-14 冒烟）
    那样做会把对手占位从 **0.53 座位/局** 拉到 **2.4 座位/局**（每局 4 个座位各自
    独立命中），主模型只剩 40% 座位，samples/轮 从 10 900 塌到 4 850，整段训练量
    腰斩 45%，而且完全不符合"只替换 1~2 个座位"的既有口径。
    正确做法是：**每局抽一次分支**（master / 自快照 / 教师 / 纯自弈），命中后
    只替换 1~2 个座位 —— 与旧版同量级。

    自快照加权（P2 · m2b1，2026-10-18）：
    * ``snap_temp=0``（默认）＝**线性新近度**：权重 1..n，越新越常出场（旧行为）。
    * ``snap_temp=T>0``＝**温度 softmax**：新近度归一化到 [0,1] 后取 exp(w/T)。
      T→∞ 均匀采样；T→0 只打最新一份；T=1 时最新:最老 ≈ e:1 —— 池扩到 12 份后
      线性加权会把老快照压到几乎不出场（1/78），温度加权保证「老而不同」的
      历史自己仍占可观出场率（多样性假设的本体）。
    """
    r = rng.random()
    mp = master_prob if master_prob > 0 else 0.0
    n_snap = len(snapshots)
    w = None
    if n_snap:
        if snap_temp and snap_temp > 0:
            rec = np.linspace(0.0, 1.0, n_snap, dtype=np.float64)  # 0=最老 1=最新
            w = np.exp(rec / float(snap_temp))
            w /= w.sum()
        else:
            w = np.arange(1, n_snap + 1, dtype=np.float64)   # 越新权重越大（旧）
            w /= w.sum()
    if mp > 0 and r < mp:
        kind = master_key
    elif n_snap and r < mp + past_prob:
        kind = "past"
    elif teacher_prob > 0 and r < mp + past_prob + teacher_prob:
        # 必须显式判 teacher_prob>0：否则快照池为空的那几轮，past 的区间会**漏进**
        # 教师分支，凭空造出一堆启发式 bot（冒烟实测：teacher_prob=0 却出现 0.445
        # 座位/局的 bot）。
        kind = "teacher"
    else:
        kind = "net"
    bots: dict = {}
    net_keys = ["net"] * 4
    models: dict = {}
    if kind != "net":
        k = int(rng.integers(1, 3))          # 替换 1~2 个座位（与历史口径一致）
        idxs = set(rng.choice(4, size=k, replace=False).tolist())
        for i in idxs:
            if kind == master_key:
                net_keys[i] = master_key
            elif kind == "past":
                snap = snapshots[int(rng.choice(n_snap, p=w))]
                key = f"snap{id(snap)}"
                net_keys[i] = key
                models[key] = snap
            else:
                st = STYLES[int(rng.integers(0, len(STYLES)))]
                bots[i] = HeuristicBot(seed=int(rng.integers(1, 10 ** 9)),
                                       weights=TEACHER_STYLES[st])
    seats = ["bot" if i in bots else "net" for i in range(4)]
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
            mdl = slot.models[key]
            # 版本与包掩码优先取模型自带属性：跨代对手池里 v1/v2 可以混坐，
            # 每个座位按自己的口径编码，不会因为主模型是 v2 就把 v1 快照喂错。
            ver = getattr(mdl, "encoder_version", slot.enc_version)
            packs = getattr(mdl, "encoder_packs", slot.enc_packs)
            return (p, encode(game, p, ver, packs), key, mdl)
        game.step(slot.bots[p].choose(game))
        slot.steps += 1
        if slot.steps >= MAX_STEPS:
            break
    return None


def _pack_logits(logits_list: list, masks_list: list) -> torch.Tensor:
    """把宽度不齐的各头 logits 补到 ACT_WIDTH，并把非法动作挖成 -1e9。"""
    B = len(logits_list)
    lg = torch.full((B, ACT_WIDTH), -1e9)
    for i, lg_i in enumerate(logits_list):
        lg[i, :lg_i.shape[0]] = lg_i
    m = torch.from_numpy(np.stack(masks_list)).bool()
    lg[~m] = -1e9
    return lg


def _sample_actions(logits_list: list, masks_list: list, temperature: float,
                    rng: np.random.Generator, explore_eps: float = 0.0
                    ) -> np.ndarray:
    """按温度从各决策点采样动作索引（宽度不齐，先补到 ACT_WIDTH）。

    ``explore_eps``（M2-A · A2）把温度采样与**均匀合法动作**做凸混合：

        π_sample = (1 − ε)·softmax(logits/τ) + ε·U(legal)

    为什么需要它：原配方下策略越练越尖，被采样的动作几乎总在 π 的众数附近，
    ``logπ(a)`` 逼近 0 → 策略梯度 ``−logπ·A`` 的幅值自我熄火（D3 的直接病灶）。
    ε 混合保证**永远有一部分探索性动作被采样**，它们的 logπ 远离极值，
    梯度不会随策略变尖而消失。退火（3% → 1%）让后期逐步收敛回贪心行为。

    注意：这是**行为策略**的改动，训练目标里的 logπ 仍取 π 自己的对数概率
    （不做重要性修正）—— 目的正是让"低概率但被访问到的动作"贡献梯度，
    符合 Kickstarting/Reincarnating RL 里"先覆盖、后退火"的做法。
    """
    lg = _pack_logits(logits_list, masks_list)
    if temperature <= 1e-6:
        return lg.argmax(dim=-1).numpy()
    probs = torch.softmax(lg / temperature, dim=-1).numpy()
    probs = np.clip(probs, 0, None)
    probs /= probs.sum(axis=1, keepdims=True)
    if explore_eps > 0:
        m = np.stack(masks_list).astype(np.float64)
        cnt = m.sum(axis=1, keepdims=True)
        unif = np.divide(m, cnt, out=np.zeros_like(m), where=cnt > 0)
        probs = (1.0 - explore_eps) * probs + explore_eps * unif
        probs /= probs.sum(axis=1, keepdims=True)
    return np.array([rng.choice(ACT_WIDTH, p=row) for row in probs])


def _batch_decide(items, temperature, rng, want_argmax: bool = False,
                  explore_eps: float = 0.0):
    """对同模型的若干决策点做一次合批前向并采样。

    ``items``: list[(slot, p, obs, model)]。
    返回 list[(slot, p, obs, action_idx, argmax_idx_or_None, value)]。

    ``want_argmax`` 打开时额外回传**贪心动作**。BC 数据生成要用它：
    对局按温度采样（保证状态覆盖面），但标签记贪心动作 —— 否则 BC 会去拟合
    采样噪声（温度 0.8 下约四分之一的标签是纯随机），那部分是不可学的。

    ``value``（M2-A · A7）：采集时顺手把价值头输出回传。原先 ``update()`` 要用
    同样的权重在同样的 eval 模式下再跑一遍**全量前向**估计优势（实测 10.2 s/轮），
    而那时权重根本没变 —— 这份输出与重算**逐位相同**，属于纯浪费。
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
    acts = _sample_actions(logits_list, masks_list, temperature, rng, explore_eps)
    am = None
    if want_argmax:
        am = _pack_logits(logits_list, masks_list).argmax(dim=-1).numpy()
    vals = out["value"].detach().reshape(-1).to(torch.float32).numpy()
    return [(it[0], it[1], it[2], int(a),
             (None if am is None else int(am[i])), float(vals[i]))
            for i, (it, a) in enumerate(zip(items, acts))]


def teacher_logprobs(obs_list, teacher):
    """对同一批观测跑一次**教师**前向，返回每条的遮罩后 log-softmax。

    M2-A · A4（KLT 蒸馏）需要的只是教师分布这一侧的 ``log π_teacher``；
    把它在 **worker 侧**算好（那里才有 game 对象，能以教师自己的编码器口径
    重新编码同一局面），主进程就只做「取出来做 KL」—— 于是教师前向从
    **串行主进程**挪到**并行 worker**，单轮墙钟几乎不增。

    返回形状 ``(len(obs_list), ACT_WIDTH)``，非法位置填 ``-1e9``
    （``exp(-1e9) == 0``，做 KL 时自然不贡献）。
    """
    tiles = torch.from_numpy(
        np.stack([o.tiles for o in obs_list]).astype(np.float32))
    glob = torch.from_numpy(
        np.stack([o.glob for o in obs_list]).astype(np.float32))
    with torch.no_grad():
        out = teacher(tiles, glob)
    res = np.full((len(obs_list), ACT_WIDTH), -1e9, dtype=np.float32)
    for i, o in enumerate(obs_list):
        dim = o.mask.shape[0]
        lg = out[o.head][i][:dim]
        lg = lg.masked_fill(~torch.from_numpy(o.mask), -1e9)
        res[i, :dim] = F.log_softmax(lg, dim=-1).numpy()
    return res


def collect_vectorized(model, n_games: int, seed0: int, temperature: float,
                       reward_scale: float, teacher_prob: float,
                       threads: int = 4, snapshots=None, past_prob: float = 0.3,
                       game_cls=None, config_cls=None, want_traj: bool = True,
                       encoder_version: int = 1, encoder_packs=None,
                       label_mode: str = "sample", explore_eps: float = 0.0,
                       teacher=None, teacher_ver=None, teacher_packs=None,
                       master_model=None, master_prob: float = 0.0,
                       aux: bool = False, snap_temp: float = 0.0):
    """向量化采集 ``n_games`` 局。返回与 selfplay.collect_games 兼容的结果。

    ``want_traj=False`` 时只关心对局结果（评测用），不记录训练样本。
    ``encoder_packs`` 只对 v2 有意义（见 :func:`zhuoji.encoder_v2.parse_packs`）；
    主模型与各快照的版本/包掩码优先取**模型自带属性**，因此跨代对手池
    可以混用 v1/v2 而不会串味。
    ``label_mode``：记进轨迹的标签用哪个动作 —— ``"sample"``（历史行为，
    按温度采样）或 ``"argmax"``（贪心）。后者给 BC 数据生成用：对局仍按温度
    采样以覆盖状态空间，但标签是教师的确定策略，**不把采样噪声当学习目标**。

    M2-A 新增
    ---------
    * ``explore_eps`` —— A2 的 ε-均匀探索（见 :func:`_sample_actions`）；
    * ``teacher`` / ``teacher_ver`` —— A4 的 KLT 教师。给了就**在同一局面上**
      用教师自己的编码器口径重编一次并前向，把教师的 log 概率随轨迹带回
      （主进程省掉一次教师前向）。``teacher_ver=None`` = 关。
    * ``master_model`` / ``master_prob`` —— A5：master 作为一等对手，
      座位级独立出场率。
    * ``aux`` —— M2-AUX：``True`` 时在**每个 net 决策点**从 ``game`` 对象
      直接算完美信息辅助标签（:mod:`zhuoji.aux_labels`），作为轨迹元组的
      **第 10 元** ``(sh_bucket, opp_ting, wall_frac, danger|None)`` 附上；
      ``False`` 时轨迹保持 9 元组、行为与旧版逐位一致。标签只在
      ``want_traj and key=="net"`` 的样本上计算，推理/评测路径零开销。

    轨迹元组固定为 **9 元**：``(tiles, glob, mask, head, act, player, gi,
    value, teacher_logprob或None)``。
    """
    if game_cls is None:
        from zhuoji import RulesConfig, ZhuojiGame
        game_cls, config_cls = ZhuojiGame, RulesConfig
    torch.set_num_threads(threads)
    rng = np.random.default_rng(seed0)
    lab_fn = None
    if aux:      # 延迟导入：aux=False 的调用方（评测/BC）不付任何开销
        from .aux_labels import labels_for as lab_fn

    snapshots = list(snapshots or [])
    if master_model is None:
        master_prob = 0.0
    traj: list = []
    rewards = np.zeros((n_games, 4), dtype=np.float32)
    t0 = time.time()

    slots: list[_GameSlot] = []
    for gi in range(n_games):
        seats, bots, net_keys, models = _make_seats(
            rng, teacher_prob, snapshots, past_prob, master_prob,
            snap_temp=snap_temp)
        models.setdefault("net", model)   # 主模型也必须登记，否则 _advance 取不到
        if master_model is not None:
            models.setdefault("master", master_model)
        game = game_cls(config_cls(), seed=int(seed0 + gi * 6151))
        slots.append(_GameSlot(game, gi, seats, bots, net_keys, models,
                               encoder_version, encoder_packs))

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

        want_am = (label_mode == "argmax")
        use_teacher = (teacher is not None and teacher_ver is not None)
        for key, items in pend.items():
            decided = _batch_decide(items, temperature, rng, want_am, explore_eps)
            tlp_rows = None
            if want_traj and key == "net" and use_teacher:
                # 教师按**自己的**编码器口径看同一局面（master 是 v1，主模型是 v3，
                # 二者输入通道不同，绝不能共用同一份编码张量）。
                tobs = [encode(d[0].game, d[1], teacher_ver, teacher_packs)
                        for d in decided]
                tlp_rows = teacher_logprobs(tobs, teacher)
            for j, (slot, p, obs, idx, am, val) in enumerate(decided):
                if want_traj and key == "net":
                    m = np.zeros(ACT_WIDTH, dtype=np.float32)
                    m[:obs.mask.shape[0]] = obs.mask.astype(np.float32)
                    lab = None
                    if aux:
                        # 标签在 **step 之前**算：此刻 slot.game 正是 obs 编码时
                        # 的那个状态，逐位对应。delay 无所谓 —— step 在循环体内。
                        lab = lab_fn(slot.game, p, obs.head)
                    traj.append((obs.tiles, obs.glob, m,
                                 HEAD_NAMES.index(obs.head),
                                 idx if am is None else am, p, slot.gi, val,
                                 None if tlp_rows is None else tlp_rows[j], lab))
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
                  game_cls, config_cls, threads: int = 2,
                  encoder_version: int = 1, encoder_packs=None):
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
                slots.append(_GameSlot(game, len(slots), seats, bots, net_keys,
                                       models, encoder_version, encoder_packs))
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
                for slot, p, obs, idx, _am, _v in _batch_decide(items, 0.0, rng):
                    slot.game.step(index_to_action(obs.head, idx, slot.game))
                    slot.steps += 1
        del slots
    return [r for r in results if r is not None]


# ---------------------------------------------------------------------------
# 多进程包装（Windows 用 spawn，worker 必须是模块级函数、参数可 pickle）
# ---------------------------------------------------------------------------
def _build_net(nc, sd):
    """按检查点自带的 ``net`` 元数据建模型（v1/v2、有无空间头都能还原）。"""
    from zhuoji.net import build_model, net_config_from_ckpt
    m = build_model(net_config_from_ckpt({"net": nc}))
    m.load_state_dict(sd)
    return m.eval()


def _build_snap(entry, fallback_nc):
    """按快照**自带**的元数据重建一个对手模型。

    ``entry`` 可以是 :func:`zhuoji.net.ck_from_model` 产出的检查点风格字典
    （推荐：版本/包掩码/网络配置都自带），也可以退化为裸 ``state_dict``
    （历史写法，此时用主模型的 ``fallback_nc``，且不挂版本属性）。
    """
    from zhuoji.net import build_from_ckpt
    if isinstance(entry, dict) and "state_dict" in entry \
            and ("net" in entry or "encoder_version" in entry):
        m, _v = build_from_ckpt(entry)
        return m
    return _build_net(fallback_nc, entry)


# ---- worker 侧快照缓存 ------------------------------------------------
# 为什么要缓存：训练每轮都要把「主模型 + 常驻对手 + 训练内快照」共 ~12 个
# state_dict（0.77M 参数 ≈ 3.1MB/个 ≈ 37MB/任务）pickle 后经管道发给每个
# worker —— 12 worker 就是 **~450MB/轮** 的序列化 + IPC + 反序列化 + 重建网络。
# 实测（i7-14700K，--workers 12）这占了每轮 10.4s 里的 ~7s，且表现为
# 「主进程和 worker 同时闲着」（CPU 只有 ~11%）——时间都耗在管道传输上。
# 快照在 selfplay 里只 append / 裁剪、从不改写已存在的项，内容不可变，
# 所以 worker 按「快照代数」缓存即可，行为与每轮全量重发**完全一致**。
_SNAP_CACHE = {"key": None, "snaps": None}
_SNAP_SENT_KEY = None          # 主进程侧：上一轮发出去的快照代数
_SNAP_SENT_SD = None           # 主进程侧：上一轮**真正发出去**的那份快照载荷
# 为什么要把载荷也记住：兜底重发（缓存未命中）时不能拿 ``snapshots_sd`` 当依据 ——
# 调用方在"复用缓存"的那些轮次按约定传 ``None``，于是 ``(None or [])`` 会变成空列表，
# 兜底就把**空对手池**当成全量发了下去（实测：样本数 1163 → 1116，轨迹与"无对手池"完全一致）。
# 也就是说缺陷 1 修好之后，这个兜底还留着第二次机会把对手池清空。记住真正发过的载荷即可根治。


class _SnapCacheMiss(RuntimeError):
    """worker 侧没有对应代数的快照缓存（进程池刚重建等）→ 上层全量重发。"""


# ---- worker 侧 KLT 教师缓存 -------------------------------------------
# 教师（master，0.77M ≈ 3.1MB）在一次训练里**恒定不变**，但每轮发送 12 份就是
# 37MB/轮的纯 IPC。所以按「与快照同一趟」发送：只在全量重发（首轮 / 快照换代 /
# 进程池重建）时带上，其余轮次 worker 复用本地实例。
_TEACH_CACHE = {"model": None}

# ---- worker 侧「一等对手座位」独立缓存（M2-C5）---------------------------
# 与教师分开：--klt-teacher 把蒸馏教师和座位对手解耦后，两者可能不是同一个模型。
# 复用 _TEACH_CACHE 会让换教师把陪练也一起换掉 —— 那正是 m2c4 起无法归因的根源。
_MASTER_CACHE = {"model": None}


def _worker(args):
    (state_dict, nc, n_games, seed0, temperature, reward_scale, teacher_prob,
     past_prob, snapshots_sd, rules, want_traj, encoder_version,
     encoder_packs, snap_key, explore_eps, teacher_sd, teacher_ver,
     teacher_packs, master_prob, aux_flag, snap_temp,
     master_sd, master_ver, master_shared) = args
    global _SNAP_CACHE
    if snapshots_sd is not None:
        # 全量：重建缓存（首次 / 快照代数变了 / 缓存丢失后的兜底重发）
        snaps = [_build_snap(sd, nc) for sd in snapshots_sd]
        _SNAP_CACHE["key"] = snap_key
        _SNAP_CACHE["snaps"] = snaps
    elif snap_key is not None and _SNAP_CACHE["key"] == snap_key:
        snaps = _SNAP_CACHE["snaps"]          # 命中：直接复用上一轮的快照网络
    else:
        raise _SnapCacheMiss(f"快照缓存未命中（key={snap_key!r}），需全量重发")
    teacher_model = None
    if teacher_ver is not None:
        if teacher_sd is not None:
            _TEACH_CACHE["model"] = _build_snap(teacher_sd, nc)
        if _TEACH_CACHE["model"] is None:
            raise _SnapCacheMiss("KLT 教师缓存缺失，需全量重发")
        teacher_model = _TEACH_CACHE["model"]
    # 座位对手：master_shared=True（默认）时与教师同一实例（旧行为，省一份前向内存）；
    # 解耦后单独缓存，缺失则抛缓存未命中让主进程全量重发。
    seat_master = None
    if master_prob > 0:
        # shared 且教师本轮可用 → 一次前向两用（旧行为）；
        # 否则（已解耦，或 β 退火到 0 使教师分支关闭）走独立缓存。
        if master_shared and teacher_model is not None:
            seat_master = teacher_model
        else:
            if master_sd is not None:
                _MASTER_CACHE["model"] = _build_snap(master_sd, nc)
            if _MASTER_CACHE["model"] is None:
                raise _SnapCacheMiss("master 座位模型缓存缺失，需全量重发")
            seat_master = _MASTER_CACHE["model"]
    game_cls, config_cls = _resolve_game_classes(rules)
    torch.set_num_threads(2)
    # denormal 冲刷：与主进程 selfplay.py 保持一致（详见该处注释）。
    # worker 侧的前向同样是 Mish 骨干，遇到 denormal 会走 AVX2 微码慢路径。
    # 每个 worker 是独立进程（MXCSR 是线程私有的），所以必须各自设置。
    try:
        torch.set_flush_denormal(True)
    except Exception:
        pass
    model = _build_net(nc, state_dict)
    model.encoder_version = int(encoder_version)
    model.encoder_packs = encoder_packs
    # 注意：**不要**在这里按 snapshots_sd 重建 snaps。
    # 上面的分支已经决定了本轮该用哪一份：
    #   * 全量发送 → 刚建好的新列表（已写进缓存）
    #   * 缓存命中 → _SNAP_CACHE 里的那份
    # 早期版本这里多写了一行 ``snaps = [_build_snap(sd, nc) for sd in (snapshots_sd or [])]``，
    # 于是缓存命中的那些轮次（默认每 4 轮里有 3 轮）snapshots_sd 为 None，
    # snaps 被覆盖成空列表 → **对手池静默消失**，_make_seats 里
    # ``if snapshots and r < past_prob`` 永不成立，league 自博弈名存实亡。
    # 见 scripts/verify_snap_cache.py 的回归断言。
    with torch.no_grad():
        traj, rewards, dt = collect_vectorized(
            model, n_games, seed0, temperature, reward_scale, teacher_prob,
            encoder_version=encoder_version, encoder_packs=encoder_packs,
            threads=2, snapshots=snaps, past_prob=past_prob,
            game_cls=game_cls, config_cls=config_cls, want_traj=want_traj,
            explore_eps=explore_eps, teacher=teacher_model,
            teacher_ver=teacher_ver, teacher_packs=teacher_packs,
            master_model=seat_master,
            master_prob=master_prob, aux=aux_flag, snap_temp=snap_temp)
    aux_out = None
    if want_traj and aux_flag and traj:
        sh = np.array([t[9][0] for t in traj], dtype=np.int64)
        opp = np.stack([t[9][1] for t in traj]).astype(np.float32)
        wall = np.array([t[9][2] for t in traj], dtype=np.float32)
        # danger 只在 discard 决策点有效（aux_labels 约定）；非出牌行填 0，
        # update() 会按 head==discard 取子集，零行永远不进损失。
        width = next((t[9][3].shape[0] for t in traj if t[9][3] is not None), 27)
        danger = np.zeros((len(traj), width), dtype=np.float32)
        for i, t in enumerate(traj):
            if t[9][3] is not None:
                danger[i] = t[9][3]
        aux_out = {"sh": sh, "opp": opp, "wall": wall, "danger": danger}
    if want_traj and traj:
        tiles = np.stack([t[0] for t in traj])
        glob = np.stack([t[1] for t in traj])
        mask = np.stack([t[2] for t in traj])
        head = np.array([t[3] for t in traj], dtype=np.int64)
        act = np.array([t[4] for t in traj], dtype=np.int64)
        ply = np.array([t[5] for t in traj], dtype=np.int64)
        gi = np.array([t[6] for t in traj], dtype=np.int64)
        val = np.array([t[7] for t in traj], dtype=np.float32)
        tlp = (np.stack([t[8] for t in traj]) if traj[0][8] is not None else None)
        return (tiles, glob, mask, head, act, ply, gi, val, tlp, rewards, dt,
                aux_out)
    return (None, None, None, None, None, None, None, None, None, rewards, dt,
            None)


def collect_parallel(state_dict, nc, n_games: int, seed0: int, temperature: float,
                     reward_scale: float, teacher_prob: float, workers: int = 4,
                     past_prob: float = 0.3, snapshots_sd=None, rules: str = "zhuoji",
                     want_traj: bool = True, pool=None, encoder_version: int = 1,
                     encoder_packs=None, snap_key=None, explore_eps: float = 0.0,
                     teacher_ck=None, teacher_ver=None, teacher_packs=None,
                     master_prob: float = 0.0, aux: bool = False,
                     snap_temp: float = 0.0,
                     master_ck=None, master_ver=None, master_shared: bool = True):
    """多进程向量化采集：``n_games`` 均分给 ``workers`` 个进程。

    ``pool`` 传入已建好的 ``multiprocessing.Pool`` 可复用进程，
    避免训练循环里每轮重新 spawn（Windows 下每次约 1-2 秒开销）。

    ``snap_key``：快照代数。``None`` = 每轮全量重发快照（旧行为）；
    传一个可比较的值（如自增整数）时，只在代数变化的那一轮把快照塞进任务，
    之后各轮只发 key，worker 用本地缓存 —— 省掉 ~450MB/轮 的 IPC。
    快照不可变是前提（selfplay 满足）；worker 缓存丢失会抛
    :class:`_SnapCacheMiss`，这里自动全量重发一次兜底，行为不变。

    ``snapshots_sd`` 里每项最好是 :func:`zhuoji.net.ck_from_model` 产出的
    检查点风格字典（自带网络配置/版本/包掩码），裸 ``state_dict`` 仍可用。

    ``teacher_ck`` / ``teacher_ver``：M2-A · A4 的 KLT 教师（``teacher_ver=None``
    表示不做蒸馏）。教师**搭快照的车**发送 —— 与快照同一节奏做全量/缓存复用，
    因为它同样在一次训练里不可变。

    **调用约定**：只在快照代数变化的那一轮传 ``snapshots_sd``；其余轮传 ``None``
    表示"复用 worker 缓存"。函数内部会把上一次真正发出去的载荷记住，
    所以兜底重发（缓存未命中）发的是真快照，不会是空池。
    """
    global _SNAP_SENT_KEY, _SNAP_SENT_SD
    if snapshots_sd is not None:
        _SNAP_SENT_SD = list(snapshots_sd)      # 记住真正发过的载荷
    send_full = snap_key is None or snap_key != _SNAP_SENT_KEY
    # 兜底重发与单进程路径都用这一份：优先本轮实参，否则用上一轮真正发过的
    payload = list(snapshots_sd) if snapshots_sd is not None else (_SNAP_SENT_SD or [])
    if workers <= 1:
        (tiles, glob, mask, head, act, ply, gi, val, tlp,
         rewards, dt, aux_out) = _worker(
            (state_dict, nc, n_games, seed0, temperature, reward_scale,
             teacher_prob, past_prob, payload, rules, want_traj,
             encoder_version, encoder_packs, None, explore_eps, teacher_ck,
             teacher_ver, teacher_packs, master_prob, aux, snap_temp,
             master_ck, master_ver, master_shared))
        traj = _stack_traj(tiles, glob, mask, head, act, ply, gi, val, tlp)
        return traj, rewards, dt, aux_out
    per = [n_games // workers] * workers
    for i in range(n_games - sum(per)):
        per[i] += 1

    def build_tasks(include_full):
        # 注意用 payload 而不是 snapshots_sd：后者在复用缓存的轮次是 None
        sd_for_task = payload if include_full else None
        t_for_task = teacher_ck if include_full else None
        # 座位对手只在「解耦」或「教师本轮不参与」时才需要单独下发；
        # shared 且教师可用时 worker 直接复用教师实例（省一份 IPC + 一份前向内存）。
        need_own_master = (not master_shared) or (teacher_ver is None)
        m_for_task = master_ck if (include_full and need_own_master) else None
        return [(state_dict, nc, k, seed0 + w * 102871, temperature,
                 reward_scale, teacher_prob, past_prob, sd_for_task, rules,
                 want_traj, encoder_version, encoder_packs, snap_key,
                 explore_eps, t_for_task, teacher_ver, teacher_packs,
                 master_prob, aux, snap_temp,
                 m_for_task, master_ver, master_shared)
                for w, k in enumerate(per) if k > 0]

    def run_map(tasks):
        ctx = mp.get_context("spawn")
        if pool is not None:
            return pool.map(_worker, tasks)
        with ctx.Pool(processes=len(tasks)) as p:
            return p.map(_worker, tasks)

    # 首选：快照代数没变就只发 key；任何 worker 缓存丢失则全量重发一次兜底
    try:
        results = run_map(build_tasks(not send_full))
    except _SnapCacheMiss:
        results = run_map(build_tasks(True))
    _SNAP_SENT_KEY = snap_key

    rewards = np.zeros((n_games, 4), dtype=np.float32)
    tiles_all, glob_all, mask_all = [], [], []
    head_all, act_all, ply_all, gi_all, val_all, tlp_all = [], [], [], [], [], []
    aux_all = []                       # M2-AUX：各 worker 的标签数组
    off = 0
    dt = 0.0
    for tiles, glob, mask, head, act, ply, gi, val, tlp, rw, wdt, ax in results:
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
            val_all.append(val)
            if tlp is not None:
                tlp_all.append(tlp)
            if ax is not None:
                aux_all.append((k, ax))
        off += k
        dt = max(dt, wdt)   # 并行耗时取最慢 worker
    aux_out = None
    if aux_all:
        w_d = aux_all[0][1]["danger"].shape[1]
        # 防呆：danger 宽度不一致说明进程池里混了旧结构 worker → 整体弃用
        if all(ax["danger"].shape[1] == w_d for _, ax in aux_all):
            aux_out = {
                "sh": np.concatenate([ax["sh"] for _, ax in aux_all]),
                "opp": np.concatenate([ax["opp"] for _, ax in aux_all]),
                "wall": np.concatenate([ax["wall"] for _, ax in aux_all]),
                "danger": np.concatenate([ax["danger"] for _, ax in aux_all]),
            }
    if tiles_all:
        traj = _stack_traj(
            np.concatenate(tiles_all), np.concatenate(glob_all),
            np.concatenate(mask_all), np.concatenate(head_all),
            np.concatenate(act_all), np.concatenate(ply_all),
            np.concatenate(gi_all), np.concatenate(val_all),
            (np.concatenate(tlp_all) if tlp_all else None))
    else:
        traj = []
    return traj, rewards, dt, aux_out


def _stack_traj(tiles, glob, mask, head, act, ply, gi, val, tlp) -> list:
    """把打包好的 numpy 列还原成轨迹元组列表（9 元，与 collect_vectorized 同口径）。"""
    if tiles is None:
        return []
    n = tiles.shape[0]
    return [(tiles[i], glob[i], mask[i], int(head[i]), int(act[i]),
             int(ply[i]), int(gi[i]), float(val[i]),
             (None if tlp is None else tlp[i]))
            for i in range(n)]


def _table_worker(args):
    """循环赛 worker：跑若干张牌桌的复式赛，返回对局记录。"""
    (table_ids, seat_specs, sd_bank, nc, deals, seed0, rules) = args
    game_cls, config_cls = _resolve_game_classes(rules)
    # ``sd_bank`` 的每项允许是「检查点风格字典」（自带网络配置/版本/包掩码），
    # 于是同一场循环赛里可以混放 v1 与 v2 检查点而不会串味。
    bank = {k: _build_snap(sd, nc) for k, sd in sd_bank.items()}
    torch.set_num_threads(2)
    out = []
    with torch.no_grad():
        for tid in table_ids:
            recs = run_duplicate(bank, seat_specs[tid], deals, seed0,
                                 game_cls, config_cls, threads=2)
            for rec in recs:
                out.append((tid,) + rec)
    return out
