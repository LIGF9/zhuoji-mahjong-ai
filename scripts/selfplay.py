"""
自我对弈强化学习（REINFORCE + 价值基线，即 A2C 式的单步更新）。

为什么不用 PPO
--------------
麻将一局很长（60+ 决策点），但回报只在终局出现一次，优势估计拿价值头当基线
就够用；PPO 的 clip 需要 ratio，在"每轮重新采样、只更新 1~2 个 epoch"的设定下
收益有限，反而增加实现风险。所以这里用最朴素但稳的做法：

    A(s,a) = 终局得分 - V(s)
    L = -E[log π(a|s) · A.detach()] + λ_v · SmoothL1(V, R) - λ_e · H(π)

配套三个工程手段保证不退化：
1. **对手池**：模型 vs 不同风格的启发式教师，避免自我对弈陷入同类循环；
2. **KL 锚定**：对 BC 初始策略加一个轻量 KL 惩罚，防止跑偏成"乱打"；
3. **复式评测**：每 N 轮用同副牌轮换座位对打固定教师，只有评测分提升才存档。
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from zhuoji import RulesConfig, ZhuojiGame  # noqa: E402
from zhuoji.dushan import DushanGame, DushanConfig  # noqa: E402
from zhuoji.agent import NetAgent, play_duplicate  # noqa: E402
from zhuoji.bots import TEACHER_STYLES, HeuristicBot  # noqa: E402
from zhuoji.encoder import HEAD_DIMS, encode, index_to_action  # noqa: E402
from zhuoji.net import NetConfig, build_model  # noqa: E402

HEAD_NAMES = ["discard", "self_kong", "respond"]


# ---------------------------------------------------------------------------
# 采样
# ---------------------------------------------------------------------------
def collect_games(model, n_games, seed0, temperature, reward_scale,
                  teacher_prob, threads=8, snapshots=None, past_prob=0.3,
                  game_cls=ZhuojiGame, config_cls=RulesConfig):
    """跑 ``n_games`` 局，返回模型自己决策点的训练样本。

    座位配置有三种，随机混合：
    * 全模型自我对弈；
    * 模型 + 启发式教师（提供不同风格的多样性）；
    * 模型 + 历史快照（league 式对手池，避免自我对弈陷入同类循环）。
    """
    torch.set_num_threads(threads)
    rng = np.random.default_rng(seed0)
    styles = list(TEACHER_STYLES.keys())
    snapshots = snapshots or []

    traj = []          # (tiles, glob, mask28, head, act, player, game_idx)
    rewards = []       # 每局每个玩家的最终得分
    t0 = time.time()

    for gi in range(n_games):
        seat_nets = _seat_config(rng, teacher_prob, bool(snapshots), past_prob)
        teachers = {}
        past_models = {}
        for i, kind in enumerate(seat_nets):
            if kind == "teacher":
                st = styles[int(rng.integers(0, len(styles)))]
                teachers[i] = HeuristicBot(seed=int(rng.integers(1, 10 ** 9)),
                                           weights=TEACHER_STYLES[st])
            elif kind == "past":
                past_models[i] = snapshots[int(rng.integers(0, len(snapshots)))]

        game = game_cls(config_cls(), seed=int(seed0 + gi * 6151))
        steps = 0
        while game.phase.value != "over" and steps < 4000:
            p = game.actor()
            if seat_nets[p] == "net":
                obs = encode(game, p)
                with torch.no_grad():
                    out = model(torch.from_numpy(obs.tiles[None].astype(np.float32)),
                                torch.from_numpy(obs.glob[None].astype(np.float32)))
                    lg = out[obs.head][0][:obs.mask.shape[0]].clone()
                    lg[~torch.from_numpy(obs.mask)] = -1e9
                    if temperature > 1e-6:
                        probs = torch.softmax(lg / temperature, dim=-1).numpy()
                        probs = np.clip(probs, 0, None)
                        probs = probs / probs.sum()
                        idx = int(rng.choice(len(probs), p=probs))
                    else:
                        idx = int(torch.argmax(lg))
                a = index_to_action(obs.head, idx, game)
                m = np.zeros(28, dtype=np.float32)
                m[:obs.mask.shape[0]] = obs.mask.astype(np.float32)
                traj.append((obs.tiles, obs.glob, m,
                             HEAD_NAMES.index(obs.head), idx, p, gi))
            elif seat_nets[p] == "teacher":
                a = teachers[p].choose(game)
            else:
                with torch.no_grad():
                    obs = encode(game, p)
                    out = past_models[p](torch.from_numpy(obs.tiles[None].astype(np.float32)),
                                         torch.from_numpy(obs.glob[None].astype(np.float32)))
                    lg = out[obs.head][0][:obs.mask.shape[0]].clone()
                    lg[~torch.from_numpy(obs.mask)] = -1e9
                    a = index_to_action(obs.head, int(torch.argmax(lg)), game)
            game.step(a)
            steps += 1
        rewards.append([d / reward_scale for d in game.score_delta])

    rewards = np.array(rewards, dtype=np.float32)
    dt = time.time() - t0
    return traj, rewards, dt


def _seat_config(rng, teacher_prob, has_past, past_prob):
    """决定一局里四个座位分别由谁打。"""
    r = rng.random()
    if has_past and r < past_prob:
        k = int(rng.integers(1, 3))
        idxs = set(rng.choice(4, size=k, replace=False).tolist())
        return ["past" if i in idxs else "net" for i in range(4)]
    if r < past_prob + teacher_prob:
        k = int(rng.integers(1, 3))
        idxs = set(rng.choice(4, size=k, replace=False).tolist())
        return ["teacher" if i in idxs else "net" for i in range(4)]
    return ["net"] * 4


# ---------------------------------------------------------------------------
# 更新
# ---------------------------------------------------------------------------
def update(model, ref_model, opt, traj, rewards, args):
    head_arr = np.array([t[3] for t in traj], dtype=np.int64)
    act_arr = np.array([t[4] for t in traj], dtype=np.int64)
    rew_arr = np.array([rewards[t[6], t[5]] for t in traj], dtype=np.float32)
    tiles = torch.from_numpy(np.stack([t[0] for t in traj]).astype(np.float32))
    glob = torch.from_numpy(np.stack([t[1] for t in traj]).astype(np.float32))
    mask = torch.from_numpy(np.stack([t[2] for t in traj]).astype(np.float32))
    head = torch.from_numpy(head_arr)
    act = torch.from_numpy(act_arr)
    ret = torch.from_numpy(rew_arr)
    n = tiles.shape[0]
    stats = {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0,
             "kl": 0.0, "n": n, "reward": float(rew_arr.mean())}
    nb = 0

    # 先算一遍优势（用更新前的价值头）
    with torch.no_grad():
        adv = torch.zeros(n)
        for i in range(0, n, 4096):
            sl = slice(i, i + 4096)
            v = model(tiles[sl], glob[sl])["value"]
            adv[sl] = ret[sl] - v
        if adv.std() > 1e-6:
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        adv = adv.clamp(-4.0, 4.0)

    model.train()
    perm = torch.randperm(n)
    for ep in range(args.epochs_per_iter):
        for i in range(0, n, args.batch):
            sel = perm[i:i + args.batch]
            if sel.numel() < 16:
                continue
            out = model(tiles[sel], glob[sel])
            logp_all = torch.zeros(sel.numel())
            ent_all = torch.zeros(sel.numel())
            kl_all = torch.zeros(sel.numel())
            for hid, name in enumerate(HEAD_NAMES):
                sub = (head[sel] == hid).nonzero(as_tuple=True)[0]
                if sub.numel() == 0:
                    continue
                dim = HEAD_DIMS[name]
                lg = out[name][sub][:, :dim]
                m = mask[sel][sub][:, :dim].bool()
                lg = lg.masked_fill(~m, -1e9)
                lp = F.log_softmax(lg, dim=-1)
                p = lp.exp()
                ent_all[sub] = -(p * lp).nan_to_num(0).sum(-1)
                logp_all[sub] = lp.gather(1, act[sel][sub].unsqueeze(1)).squeeze(1)
                if ref_model is not None:
                    with torch.no_grad():
                        rlg = ref_model(tiles[sel][sub], glob[sel][sub])[name][:, :dim]
                        rlg = rlg.masked_fill(~m, -1e9)
                        rlp = F.log_softmax(rlg, dim=-1)
                    kl_all[sub] = (p * (lp - rlp)).nan_to_num(0).sum(-1)

            a = adv[sel]
            loss_pi = -(logp_all * a).mean()
            v_pred = out["value"]
            loss_v = F.smooth_l1_loss(v_pred, ret[sel])
            ent = ent_all.mean()
            kl = kl_all.mean()
            loss = loss_pi + args.value_coef * loss_v \
                - args.ent_coef * ent + args.kl_coef * kl
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            stats["policy_loss"] += float(loss_pi.detach())
            stats["value_loss"] += float(loss_v.detach())
            stats["entropy"] += float(ent)
            stats["kl"] += float(kl)
            nb += 1
    for k in ("policy_loss", "value_loss", "entropy", "kl"):
        stats[k] /= max(1, nb)
    return stats


def adapt_coefs(args, ent, kl):
    """按上一轮的熵与 KL 自适应调系数，防止熵坍缩 / 策略漂移。

    - KL 超过上限 → 加大 KL 锚定；低于下限 → 适当放松（下限 0.02）。
    - 熵低于下限 → 加大熵奖励；高于上限 → 逐步放松（下限 0.03）。
    """
    if kl > args.kl_target:
        args.kl_coef = min(args.kl_coef * 1.6, 3.0)
    elif kl < args.kl_target * 0.4:
        args.kl_coef = max(args.kl_coef / 1.4, 0.02)
    if ent < args.ent_floor:
        args.ent_coef = min(args.ent_coef * 1.8, 1.0)
    elif ent > args.ent_floor * 2.2:
        args.ent_coef = max(args.ent_coef / 1.5, 0.03)


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
def _dump_ckpt(path, model, nc, args, it_no, evalr) -> None:
    """把一个检查点写到 ``path``（``--out`` 与周期性检查点共用）。"""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "state_dict": {k: v.clone() for k, v in model.state_dict().items()},
        "config": vars(args),
        "net": nc,
        "iter": it_no,
        "eval": evalr,
    }, path)


def choose_final(eval_log: list[tuple[int, float, float]],
                 select: str) -> tuple[str, str]:
    """决定 ``--out`` 最终保存哪个检查点。

    返回 ``(which, note)``，``which`` ∈ ``{"best", "last"}``；
    ``{"best"}`` 表示保留训练过程中已经写好的历史最佳检查点，不再覆盖。

    为什么要有 ``last`` / ``guard``
    ------------------------------
    训练内评测（对固定的三位启发式教师、默认 40 副 = 160 局）的 ``avg_score_se``
    实测约 0.97，即 95% 置信区间 **±1.9 分**；而真实的代际提升只有 **~0.3 分**
    （S11 vs S7，2400 副对位赛 = +0.319 ± 0.288）。于是
    ``argmax`` 评测分 = 在噪声里抽签，并且会把恰好抽高那一刻的检查点
    当成"最好"（winner's curse）——历史 s7（8.60）评测分高于 s11（7.231）、
    实际却更弱，就是这个原因。

    ``guard`` 只做一件事：**不追高，只在出现超出噪声的实质退化时回退**。
    回退判据是「最近 k 次评测的均值 + 2×SE 仍低于全程中位数」。
    """
    if select == "max" or not eval_log:
        return "best", "select=max（历史最佳教师评测分；该指标噪声大于代际差异）"
    if select == "last":
        return "last", "select=last：采用最后一轮"
    k = min(3, len(eval_log))
    recent = [x[1] for x in eval_log[-k:]]
    se_r = [x[2] for x in eval_log[-k:]]
    m_r = sum(recent) / k
    # 各次评测相互独立：均值方差 = Σσᵢ²/k²，故 SE(均值) = √(Σσᵢ²)/k
    se_m = (sum(s * s for s in se_r) ** 0.5) / k
    vals = sorted(x[1] for x in eval_log)
    med = vals[len(vals) // 2] if len(vals) % 2 else \
        (vals[len(vals) // 2 - 1] + vals[len(vals) // 2]) / 2
    if m_r + 2.0 * se_m < med:
        return "best", (f"select=guard：最近 {k} 次评测均值 {m_r:.2f} 低于全程中位数 "
                        f"{med:.2f} 达 {med - m_r:.2f}（> 2×SE={2 * se_m:.2f}），"
                        f"判定为实质退化 → 保持历史最佳")
    return "last", (f"select=guard：最近 {k} 次评测均值 {m_r:.2f} 未显著低于全程中位数 "
                    f"{med:.2f}（2×SE={2 * se_m:.2f}）→ 采用最后一轮")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--init", default=str(ROOT / "models" / "bc.pt"))
    ap.add_argument("--out", default=str(ROOT / "models" / "rl.pt"))
    ap.add_argument("--history", default=str(ROOT / "reports" / "rl_history.json"))
    ap.add_argument("--iters", type=int, default=30)
    ap.add_argument("--games-per-iter", type=int, default=64)
    ap.add_argument("--epochs-per-iter", type=int, default=2)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--temperature", type=float, default=1.1)
    ap.add_argument("--teacher-prob", type=float, default=0.5)
    ap.add_argument("--reward-scale", type=float, default=8.0)
    ap.add_argument("--value-coef", type=float, default=0.5)
    ap.add_argument("--ent-coef", type=float, default=0.01)
    ap.add_argument("--kl-coef", type=float, default=0.02)
    ap.add_argument("--kl-target", type=float, default=0.25,
                    help="KL 高于此值加大锚定，低于其 40%% 放松")
    ap.add_argument("--ent-floor", type=float, default=0.45,
                    help="熵低于此值加大熵奖励（防坍缩）")
    ap.add_argument("--no-adaptive", action="store_true",
                    help="关闭自适应系数（默认开启）")
    ap.add_argument("--eval-rounds", type=int, default=25)
    ap.add_argument("--eval-every", type=int, default=5)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--save-every", type=int, default=0,
                    help="每 N 轮额外存一份周期性检查点（0=关闭），"
                         "存到 <out 同目录>/_ckpts/。供训练后做对位评分选点——"
                         "训练内的教师评测噪声太大，不该用来挑检查点。")
    ap.add_argument("--select", choices=["max", "last", "guard"], default="max",
                    help="最终 --out 保存哪个检查点。max=历史最佳教师评测分（旧行为；"
                         "该指标 SE≈0.97，95%%CI≈±1.9 分，而真实代际提升只有 ~0.3 分，"
                         "argmax 等于在噪声里抽签并被 winner's curse 高估）；"
                         "last=最后一轮；guard=最后一轮，但最近若干次评测出现"
                         "超出噪声的实质退化时回退到历史最佳。")
    ap.add_argument("--vectorized", action="store_true",
                    help="用向量化采集（多局合批推理 + 多进程，快约 3 倍）")
    ap.add_argument("--workers", type=int, default=6,
                    help="向量化采集的进程数（0/1=单进程）")
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--rules", choices=["zhuoji", "dushan"], default="zhuoji",
                    help="对局规则：zhuoji=贵阳捉鸡，dushan=独山麻将")
    args = ap.parse_args()

    if args.rules == "dushan":
        GameCls, CfgCls = DushanGame, DushanConfig
    else:
        GameCls, CfgCls = ZhuojiGame, RulesConfig

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    ck = torch.load(args.init, map_location="cpu", weights_only=False)
    nc = ck.get("net") or {"channels": 32, "blocks": 3, "hidden": 256}
    model = build_model(NetConfig(nc["channels"], nc["blocks"], nc["hidden"]))
    model.load_state_dict(ck["state_dict"])
    ref_model = copy.deepcopy(model)
    ref_model.eval()
    for p in ref_model.parameters():
        p.requires_grad_(False)

    # 优化器只建一次：AdamW 的动量要跨轮延续，每轮重建等于把动量清空。
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    print(f"从 {args.init} 初始化 (epoch={ck.get('epoch')}, val_loss={ck.get('val_loss')})")
    print(f"参数量 {model.n_params():,}；{args.iters} 轮 × {args.games_per_iter} 局/轮")

    def make_eval():
        t1 = HeuristicBot(seed=101, weights=TEACHER_STYLES["balanced"])
        t2 = HeuristicBot(seed=202, weights=TEACHER_STYLES["aggressive"])
        t3 = HeuristicBot(seed=303, weights=TEACHER_STYLES["defensive"])
        # HeuristicBot.name 是类属性，默认全是 "heuristic"；不复写的话三个教师
        # 在 play_duplicate 里会合并成同一行统计，看不出"打三个不同风格总和"。
        t1.name, t2.name, t3.name = "teacher1", "teacher2", "teacher3"
        return [
            NetAgent(model, seed=11, temperature=0.0, name="net"),
            t1, t2, t3,
        ]

    history = []
    best_score = -1e9
    eval_log: list[tuple[int, float, float]] = []   # (iter, 评测均分, 该次的标准误)
    snapshots = []
    best_ckpt = None
    watchdog = [0, None]   # [连续异常次数, 上次恢复的存档 iter]
    _pool = [None]
    if args.vectorized and args.workers > 1:
        import multiprocessing as _mp
        _pool[0] = _mp.get_context("spawn").Pool(processes=args.workers)
    t0 = time.time()

    for it in range(args.iters):
        if args.vectorized:
            from zhuoji.vcollect import collect_parallel
            import multiprocessing as _mp

            def _fresh_pool():
                return _mp.get_context("spawn").Pool(processes=args.workers)

            # Windows 长驻 spawn worker 会累积内存（实测 16GB 机器上跑到 3k 轮左右耗尽），
            # 定期重建进程池 + 采集遇 MemoryError 时回收重试。
            if _pool[0] is not None and it > 0 and it % 150 == 0:
                _pool[0].close()
                _pool[0].join()
                _pool[0] = _fresh_pool()
            snap_sd = [m.state_dict() for m in snapshots]
            traj = None
            for attempt in range(3):
                try:
                    traj_arr, rewards, dt = collect_parallel(
                        model.state_dict(), nc, args.games_per_iter,
                        args.seed + it * 99991, args.temperature, args.reward_scale,
                        args.teacher_prob, workers=args.workers or 1,
                        past_prob=0.3, snapshots_sd=snap_sd, rules=args.rules,
                        want_traj=True, pool=_pool[0])
                    traj = traj_arr
                    break
                except MemoryError:
                    print(f"[mem] iter {it+1} 采集 MemoryError，回收进程池后重试"
                          f"（第 {attempt + 1} 次）", flush=True)
                    if _pool[0] is None:
                        raise
                    _pool[0].terminate()
                    _pool[0].join()
                    _pool[0] = _fresh_pool()
            if traj is None:
                raise SystemExit(f"iter {it+1} 连续 3 次 MemoryError，终止训练")
        else:
            traj, rewards, dt = collect_games(
                model, args.games_per_iter, args.seed + it * 99991,
                args.temperature, args.reward_scale, args.teacher_prob, args.threads,
                snapshots=snapshots)
        upd = update(model, ref_model, opt, traj, rewards, args)
        if not args.no_adaptive:
            adapt_coefs(args, upd["entropy"], upd["kl"])

        # 熵坍缩看门狗：熵持续过低说明策略已经塌缩成确定性
        # —— 恢复最佳存档、重置优化器动量、减半学习率；同一存档再塌缩则终止。
        if upd["entropy"] < 0.12 or upd["kl"] > 2.5:
            watchdog[0] += 1
            if watchdog[0] >= 2:
                if best_ckpt is None:
                    print("[watchdog] 尚无最佳存档且已坍缩，终止训练", flush=True)
                    break
                if watchdog[1] == best_ckpt.get("iter"):
                    print(f"[watchdog] 最佳存档 iter={best_ckpt['iter']} 也已坍缩，终止训练",
                          flush=True)
                    break
                print(f"[watchdog] 熵={upd['entropy']:.3f} KL={upd['kl']:.2f} 坍缩 -> "
                      f"恢复 iter={best_ckpt['iter']} 存档，lr 减半", flush=True)
                model.load_state_dict(best_ckpt["state_dict"])
                ref_model = copy.deepcopy(model).eval()
                for p in ref_model.parameters():
                    p.requires_grad_(False)
                args.lr /= 2
                opt = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                        weight_decay=1e-4)
                args.ent_coef = max(args.ent_coef, 0.15)
                args.kl_coef = max(args.kl_coef, 0.1)
                watchdog[1] = best_ckpt.get("iter")
                watchdog[0] = 0
                continue
        else:
            watchdog[0] = 0
        rec = {
            "iter": it + 1,
            "samples": len(traj),
            "collect_s": dt,
            "elapsed": time.time() - t0,
            **{k: v for k, v in upd.items()},
        }
        # 对手池：每 4 轮把当前策略冻结一份进池，供后续轮次当"历史自己"的对手。
        # KL 锚点 ref_model 保持 BC 初始策略不动，防止策略整体漂移成乱打。
        # 长训练时只保留最近 8 份快照，避免池子无限膨胀拖慢采集/序列化。
        if (it + 1) % 4 == 0:
            snapshots.append(copy.deepcopy(model).eval())
            if len(snapshots) > 8:
                snapshots = snapshots[-8:]

        if (it + 1) % args.eval_every == 0 or it == args.iters - 1:
            res = play_duplicate(make_eval, args.eval_rounds, seed0=4242,
                                 game_factory=lambda s: GameCls(CfgCls(), seed=s))
            rec["eval"] = res["net"]
            printed = {k: (round(v, 3) if isinstance(v, float) else v)
                       for k, v in res["net"].items()}
            print(f"  [eval] {printed}", flush=True)
            if rec["eval"]["avg_score"] > best_score:
                best_score = rec["eval"]["avg_score"]
                best_ckpt = {
                    "state_dict": {k: v.clone() for k, v in model.state_dict().items()},
                    "config": vars(args),
                    "net": nc,
                    "iter": it + 1,
                    "eval": rec["eval"],
                }
                Path(args.out).parent.mkdir(parents=True, exist_ok=True)
                torch.save(best_ckpt, args.out)
            eval_log.append((it + 1, float(rec["eval"]["avg_score"]),
                             float(rec["eval"].get("avg_score_se") or 0.0)))

        # 周期性检查点：训练结束后用 scripts/score_checkpoints.py 在共用牌池上
        # 对它们做对位评分，才是可信的选点依据（训练内评测分噪声太大）。
        if args.save_every and (it + 1) % args.save_every == 0:
            snap = Path(args.out).parent / "_ckpts" / \
                f"{Path(args.out).stem}_it{it + 1:04d}.pt"
            _dump_ckpt(snap, model, nc, args, it + 1, rec.get("eval"))
            print(f"  [snap] {snap.name}", flush=True)

        history.append(rec)
        if (it + 1) % 50 == 0:   # 增量落盘，进程崩溃也不丢历史
            Path(args.history).parent.mkdir(parents=True, exist_ok=True)
            Path(args.history).write_text(json.dumps(history, indent=2),
                                          encoding="utf-8")
        print(f"iter {it+1:3d}  samples={rec['samples']:5d}  "
              f"R={rec['reward']:+.3f}  pi={rec['policy_loss']:+.4f}  "
              f"V={rec['value_loss']:.4f}  H={rec['entropy']:.3f}  "
              f"KL={rec['kl']:.4f}  ec={args.ent_coef:.3f}  kc={args.kl_coef:.3f}  "
              f"lr={args.lr:.1e}  collect={dt:.1f}s  total={rec['elapsed']:.0f}s",
              flush=True)

    Path(args.history).parent.mkdir(parents=True, exist_ok=True)
    Path(args.history).write_text(json.dumps(history, indent=2), encoding="utf-8")

    # ---- 最终选点 -----------------------------------------------------------
    which, select_note = choose_final(eval_log, args.select)
    if which == "last" and history:
        _dump_ckpt(args.out, model, nc, args, len(history),
                   history[-1].get("eval"))
    print(f"[select] {select_note}")

    if _pool[0] is not None:
        _pool[0].close()
        _pool[0].join()
    print(f"完成。最佳评测平均分 {best_score:+.3f}，模型 -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
