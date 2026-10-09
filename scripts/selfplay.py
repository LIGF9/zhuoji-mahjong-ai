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
from zhuoji.encoder import (  # noqa: E402
    HEAD_DIMS, dims_for_version, encode, index_to_action)
from zhuoji.encoder_v2 import ALL_PACKS as ALL_PACKS_V2  # noqa: E402
from zhuoji.encoder_v2 import PACK_DESC, parse_packs  # noqa: E402
from zhuoji.net import (  # noqa: E402
    build_from_ckpt, build_model, ck_from_model, encoder_packs_from_ckpt,
    encoder_version_from_ckpt, load_into, net_config_from_ckpt, net_meta,
    warm_start_from_v1)

HEAD_NAMES = ["discard", "self_kong", "respond"]

# ---------------------------------------------------------------------------
# 选点指标（方案 A）
# ---------------------------------------------------------------------------
# 每个指标 = (方向, 中文名)。方向 +1 = 越大越强。
#
# 为什么不再用 avg_score 的 argmax：``scripts/analyze_metric_noise.py`` 实测
# 训练内评测是**确定性**的（temperature=0 + 固定教师种子 + 固定 seed0=4242），
# 于是 ``avg_score_se≈0.97`` 不是重跑抖动，而是「这 40 副牌相对全体牌局的取样
# 标准误」；用一阶差分测单次抖动、用各代均值测代际差，得到 F = 代际差/抖动：
#
#   avg_score        F = 1.00   ⇐ 旧依据，噪声等于信号，argmax 等于抽签
#   deal_in_rate     F = 0.40   ⇐ 点炮率是 1~2.5% 的稀有事件，比旧指标还差
#   win_rate         F = 1.29
#   bigfan15_rate    F = 1.65
#   bigfan20_rate    F = 2.06   ⇐ 目前信噪比最高
#   vs_anchor        （新增，见下）
#
# ``vs_anchor`` 是最值得用的一项：把**锚点模型**（例如现任大师）放进同一桌
# 轮换，逐副取 (本模型分 − 锚点分) 的**配对**均值。配对消掉了牌运方差，
# 且方向天然明确（正 = 更强），口径与对外的 ``scripts/score_checkpoints.py``
# 完全一致——训练内选点与对外晋升判定第一次用上了同一把尺子。
SELECT_METRICS: dict[str, tuple[int, str]] = {
    "avg_score": (+1, "评测均分（旧依据；F≈1.0，等于抽签）"),
    "vs_anchor": (+1, "对锚点模型的同牌池配对净分差（P0 标尺口径）"),
    "composite": (+1, "复合代理 = win_rate + bigfan20_rate − deal_in_rate（等权定标）"),
    "bigfan20_rate": (+1, "大牌率20+（F≈2.1，方向待天梯确认）"),
    "bigfan15_rate": (+1, "大牌率15+（F≈1.7）"),
    "win_rate": (+1, "胡牌率（F≈1.3）"),
    "deal_in_rate": (-1, "点炮率（F≈0.4，稀有事件，比旧指标更差）"),
}

# 复合代理的定标（把量纲不同的 rate 拉到可比尺度；分母取各指标在该牌池下的
# 典型波动幅度，是**前置声明**的常数，不随训练过程拟合，避免用结果调参）
_COMPOSITE_SCALE = {
    "win_rate": (+1.0, 0.05),
    "bigfan20_rate": (+1.0, 0.03),
    "deal_in_rate": (-1.0, 0.02),
}


def metric_score(ev: dict | None, metric: str) -> float | None:
    """从一次评测结果里取出选点分数（方向已归一到「越大越好」）。"""
    if not ev:
        return None
    if metric == "composite":
        total = 0.0
        used = 0
        for key, (sign, scale) in _COMPOSITE_SCALE.items():
            v = ev.get(key)
            if v is None:
                continue
            total += sign * float(v) / scale
            used += 1
        return total / used if used else None
    v = ev.get(metric)
    if v is None:
        return None
    sign = SELECT_METRICS.get(metric, (+1, ""))[0]
    return sign * float(v)



# ---------------------------------------------------------------------------
# 采样
# ---------------------------------------------------------------------------
def collect_games(model, n_games, seed0, temperature, reward_scale,
                  teacher_prob, threads=8, snapshots=None, past_prob=0.3,
                  game_cls=ZhuojiGame, config_cls=RulesConfig,
                  encoder_version: int = 1, encoder_packs=None):
    """跑 ``n_games`` 局，返回模型自己决策点的训练样本。

    座位配置有三种，随机混合：
    * 全模型自我对弈；
    * 模型 + 启发式教师（提供不同风格的多样性）；
    * 模型 + 历史快照（league 式对手池，避免自我对弈陷入同类循环）。
    """
    torch.set_num_threads(threads)
    # update() 结束时会留下 model.train()，若采集前不切回 eval，第 2 轮起的自对弈
    # 会用 **batch-of-1 的 BatchNorm 统计**去决策，同时把 running_mean/var 用
    # 单样本统计污染掉（影响所有 eval 与导出模型）。并行采集路径
    # （zhuoji/vcollect.py::_build_net）本来就 .eval()，这里补齐单进程路径。
    model.eval()
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
                obs = encode(game, p, encoder_version, encoder_packs)
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
                    # 快照座位按**快照自己**的版本/包掩码编码（跨代池里可能混着
                    # v1 快照与 v2 主模型，用主模型的口径会把对手喂错）。
                    pm = past_models[p]
                    obs = encode(game, p,
                                 getattr(pm, "encoder_version", encoder_version),
                                 getattr(pm, "encoder_packs", encoder_packs))
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

            # 参考前向**整批只跑一次**。
            # 原实现写在下面的头循环里，每遇到一个头就调一次 ref_model(...) 并只喂
            # 该头的子集 —— 整条 backbone 被调用 3 次、每次都只有约 1/3 的样本，
            # 小模型上固定开销占比极高（实测 0.77M 网每轮慢 24%）。
            # 等价性：ref_model 恒为 .eval() 的冻结副本（BN 走 running stats，
            # 不做 running stat 更新），所以单样本输出与 batch 组成无关，
            # 整批一次前向与按头分次前向在数学上完全一致。
            ref_out = None
            if ref_model is not None:
                if ref_model.training:
                    ref_model.eval()   # 只读锚点，绝不该处于 train 模式
                with torch.no_grad():
                    ref_out = ref_model(tiles[sel], glob[sel])

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
                if ref_out is not None:
                    rlg = ref_out[name][sub][:, :dim]
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
def _dump_ckpt(path, model, nc, args, it_no, evalr,
               encoder_version: int = 1, encoder_packs=None) -> None:
    """把一个检查点写到 ``path``（``--out`` 与周期性检查点共用）。

    ``encoder_version`` 必须写进去：加载方靠它决定用哪一版编码器，
    老检查点没有这个字段 → 默认 v1（向后兼容）。

    ``encoder_packs`` 只在 v2 时写：消融实验的训练臂关掉了某个包，
    评测时必须关掉同样的包，靠这个字段自动复现，而不是靠人记得。
    """
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    ck_out = {
        "state_dict": {k: v.clone() for k, v in model.state_dict().items()},
        "config": vars(args),
        "net": nc,
        "encoder_version": int(encoder_version),
        "iter": it_no,
        "eval": evalr,
    }
    if int(encoder_version) >= 2 and encoder_packs is not None:
        ck_out["encoder_packs"] = list(encoder_packs)
    torch.save(ck_out, path)


def choose_final(eval_log: list[dict], select: str, metric: str = "avg_score"
                 ) -> tuple[str, str]:
    """决定 ``--out`` 最终保存哪个检查点。

    ``eval_log`` 是 ``[{"iter": i, "score": s, "se": e}, ...]``，``score`` 已经
    按 ``metric`` 归一成「越大越好」。返回 ``(which, note)``，
    ``which`` ∈ ``{"best", "last"}``；``{"best"}`` 表示保留训练过程中已经写好的
    历史最佳检查点（按同一 metric），不再覆盖。

    为什么要有 ``last`` / ``guard``
    ------------------------------
    训练内评测的 ``avg_score_se`` 实测约 0.97（= 这副固定 40 副牌相对全体牌局的
    取样标准误，不是重跑抖动；评测是确定性的，重复评同一模型得到同一分数）。
    代价是 95% 区间 **±1.9 分**，而真实代际提升只有 **~0.3 分**。
    于是 ``argmax`` = 在噪声里抽签，并把恰好最高的那一刻当成"最好"
    （winner's curse）——s7（8.60）评测分高于 s11（7.231）、实际却更弱。

    ``guard`` 只做一件事：**不追高，只在出现超出噪声的实质退化时回退**。
    回退判据是「最近 k 次评测的均值 + 2×SE 仍低于全程中位数」。
    """
    note_metric = SELECT_METRICS.get(metric, (+1, metric))[1]
    if select == "max" or not eval_log:
        return "best", f"select=max（历史最佳 {metric}：{note_metric}）"
    if select == "last":
        return "last", "select=last：采用最后一轮"
    k = min(3, len(eval_log))
    recent = [x["score"] for x in eval_log[-k:]]
    se_r = [x["se"] for x in eval_log[-k:]]
    m_r = sum(recent) / k
    # 各次评测相互独立：均值方差 = Σσᵢ²/k²，故 SE(均值) = √(Σσᵢ²)/k
    se_m = (sum(s * s for s in se_r) ** 0.5) / k
    vals = sorted(x["score"] for x in eval_log)
    med = vals[len(vals) // 2] if len(vals) % 2 else \
        (vals[len(vals) // 2 - 1] + vals[len(vals) // 2]) / 2
    label = f"指标 {metric}"
    if m_r + 2.0 * se_m < med:
        return "best", (f"select=guard（{label}）：最近 {k} 次评测均值 {m_r:.2f} 低于"
                        f"全程中位数 {med:.2f} 达 {med - m_r:.2f}"
                        f"（> 2×SE={2 * se_m:.2f}），判定为实质退化 → 保持历史最佳")
    return "last", (f"select=guard（{label}）：最近 {k} 次评测均值 {m_r:.2f} 未显著低于"
                    f"全程中位数 {med:.2f}（2×SE={2 * se_m:.2f}）→ 采用最后一轮")


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
    ap.add_argument("--eval-anchor", default="",
                    help="把某个模型（裸文件名，不带 .pt，如 dushan_big_s11）作为"
                         "**锚点**放进评测桌轮换，逐副记录 (本模型分 − 锚点分) 的配对"
                         "均值与 SE，作为选点指标 ``vs_anchor``。这一项零额外算力"
                         "（仍是同样的牌数与局数），但配对消掉了牌运方差，且口径与"
                         "对外的 scripts/score_checkpoints.py 完全一致。留空则关闭。")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--save-every", type=int, default=50,
                    help="每 N 轮额外存一份周期性检查点（0=关闭），"
                         "存到 <out 同目录>/_ckpts/。默认 50——训练内的指标噪声太大，"
                         "绝不该用来挑检查点；把周期检查点留下来，事后用共用牌池对位"
                         "评分才有得选。（历史 s1–s11 从未开这个开关，导致"
                         "「代内重选」永久不可能。）")
    ap.add_argument("--select", choices=["max", "last", "guard"], default="guard",
                    help="最终 --out 保存哪个检查点。max=历史最佳（旧行为，等同于在"
                         "噪声里抽签）；last=最后一轮；guard=最后一轮，但最近若干次"
                         "评测出现超出噪声的实质退化时回退到历史最佳（默认）。")
    ap.add_argument("--select-metric", choices=sorted(SELECT_METRICS), default="avg_score",
                    help="选点（含 guard 判定与历史最佳）用哪个指标。"
                         "avg_score=旧依据（F≈1.0，等于抽签）；"
                         "vs_anchor=对锚点模型的配对净分差（需 --eval-anchor，推荐）；"
                         "composite=win_rate+bigfan20_rate−deal_in_rate 等权定标；"
                         "bigfan20_rate=目前信噪比最高（F≈2.1）；"
                         "deal_in_rate=稀有事件，F≈0.4，比旧指标更差（保留以便复核）。")
    ap.add_argument("--snapshot-pool", default="",
                    help="跨代快照池：逗号分隔的模型裸名（不带 .pt），会被加载并**常驻**"
                         "对手池（不参与最近 8 份的自有快照裁剪），让自我对弈同时面对"
                         "「不同世代的自己」。零采集成本，是本机唯一能立刻做的"
                         "对手池增强。例：--snapshot-pool dushan_big_s3,dushan_big_s7。")
    ap.add_argument("--past-prob", type=float, default=0.3,
                    help="一局里抽到「历史快照当对手」的概率（其余按 teacher-prob 分给"
                         "教师，剩下的是纯自我对弈）。")

    ap.add_argument("--vectorized", action="store_true",
                    help="用向量化采集（多局合批推理 + 多进程，快约 3 倍）")
    ap.add_argument("--workers", type=int, default=6,
                    help="向量化采集的进程数（0/1=单进程）")
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--encoder", choices=["auto", "v1", "v2"], default="auto",
                    help="状态编码器版本。auto=跟随 --init 检查点记录（老文件→v1）；"
                         "v2 会触发**部分热启动**：stem 卷积前 15 个通道、"
                         "fc_in 前 70 列从 v1 权重逐位复制，新增特征对应的权重"
                         "从零学起（见 zhuoji/net.py::warm_start_from_v1）。")
    ap.add_argument("--spatial-head", choices=["auto", "on", "off"], default="auto",
                    help="逐牌空间头：零初始化的 1×1 卷积加到 discard/self_kong 的 "
                         "27 个位置 logit 上（只 +2×(C+1) 个参数）。auto=跟随检查点；"
                         "on=强制开启（同版本加载时输出与旧模型逐位相同，是严格热启动）。")
    ap.add_argument("--encoder-packs", default="auto",
                    help="P1 特征包掩码（只对 --encoder v2 有意义）：auto=跟随 --init "
                         "检查点记录（老文件→全开）；abc=全开；a=只要牌河时序；"
                         "b=只要危险度与对手状态；c=只要番型潜力；"
                         "none=全关（纯 v1 特征 + 零填充，作对照臂）。"
                         "未启用的包只是**清零**，架构与参数量完全不变 —— 这是单变量消融的前提。")
    ap.add_argument("--rules", choices=["zhuoji", "dushan"], default="zhuoji",
                    help="对局规则：zhuoji=贵阳捉鸡，dushan=独山麻将")
    args = ap.parse_args()

    if args.rules == "dushan":
        GameCls, CfgCls = DushanGame, DushanConfig
    else:
        GameCls, CfgCls = ZhuojiGame, RulesConfig

    ck = torch.load(args.init, map_location="cpu", weights_only=False)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # ---- 编码器 / 网络结构版本（P1） ----------------------------------------
    # ``auto`` = 跟随检查点自己记录的版本（老文件 → v1），这样"
    # 拿哪个模型继续训"不需要调用方记得它的编码器版本。
    if args.encoder == "auto":
        enc_version = encoder_version_from_ckpt(ck)
    else:
        enc_version = int(args.encoder[1:])
    if args.spatial_head == "auto":
        spatial = bool((ck.get("net") or {}).get("spatial_head", False))
    else:
        spatial = args.spatial_head == "on"
    cfg = net_config_from_ckpt(ck)
    cfg.in_channels, cfg.global_dim = dims_for_version(enc_version)
    cfg.spatial_head = spatial

    # ---- P1 特征包掩码（方案 B 阶段 2b 的单变量消融） ----------------------
    # 只对 v2 有意义。auto = 跟随检查点（继续训练不会莫名其妙把包打开）；
    # 显式给出则以此为准，并写进新检查点，评测时能自动复现同一口径。
    if args.encoder_packs == "auto":
        enc_packs = encoder_packs_from_ckpt(ck)
    else:
        enc_packs = parse_packs(args.encoder_packs)

    model = build_model(cfg)
    src_version = encoder_version_from_ckpt(ck)
    if src_version != enc_version and args.encoder != "auto":
        info = warm_start_from_v1(model, ck)
        print(f"[warm] {args.init} 编码器 v{src_version} → v{enc_version}："
              f"原样复制 {info['copied']} 个张量；部分复制 {info['partial']}；"
              f"全新参数 {len(info['new_only'])} 个（保持随机初始化）")
    else:
        miss, unexp = load_into(model, ck["state_dict"])
        if miss or unexp:
            print(f"[load] 缺失 {len(miss)} 键（新增结构，保持初始化）/ "
                  f"意外 {len(unexp)} 键")
    nc = net_meta(cfg)
    # 把编码口径挂在模型上：这样 deepcopy 出来的快照、以及任何
    # "拿模型造 NetAgent" 的地方都能自动取到正确的版本与包掩码。
    model.encoder_version = enc_version
    model.encoder_packs = enc_packs if enc_version >= 2 else None

    ref_model = copy.deepcopy(model)
    ref_model.eval()
    for p in ref_model.parameters():
        p.requires_grad_(False)

    # 优化器只建一次：AdamW 的动量要跨轮延续，每轮重建等于把动量清空。
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    print(f"从 {args.init} 初始化 (epoch={ck.get('epoch')}, val_loss={ck.get('val_loss')})")
    print(f"参数量 {model.n_params():,}；{args.iters} 轮 × {args.games_per_iter} 局/轮")
    print(f"[编码] v{enc_version} spatial_head={'on' if spatial else 'off'}"
          + (f" packs={''.join(enc_packs) or 'none'}"
             f"（{', '.join(PACK_DESC[p] for p in enc_packs) or '全关'}）"
             if enc_version >= 2 else ""))
    print(f"[选点] select={args.select} metric={args.select_metric}"
          f"（{SELECT_METRICS.get(args.select_metric, (+1, '?'))[1]}）"
          f" save-every={args.save_every} eval-every={args.eval_every}"
          f" eval-rounds={args.eval_rounds}")

    # ---- 跨代快照池（常驻对手，不参与自有快照裁剪） --------------------------
    # 用 build_from_ckpt：快照自带网络配置 / 编码器版本 / 包掩码，于是
    # 「主模型 v2 + 快照 v1」这种跨版本对手池不会因为形状不同直接报错。
    pool_extra: list = []          # 模型实例（单进程路径用）
    pool_extra_ck: list = []       # 检查点风格字典（多进程路径用）
    for stem in [s.strip() for s in args.snapshot_pool.split(",") if s.strip()]:
        p = Path(stem)
        if not p.is_absolute():
            p = ROOT / "models" / (stem if stem.endswith(".pt") else f"{stem}.pt")
        if not p.exists():
            print(f"[pool] 跳过（不存在）：{p}")
            continue
        em, e_ver = build_from_ckpt(
            torch.load(p, map_location="cpu", weights_only=False))
        for q in em.parameters():
            q.requires_grad_(False)
        # ck_from_model 里显式带上版本/包掩码，worker 才能按同一口径编码
        pool_extra_ck.append(ck_from_model(
            em, encoder_version=e_ver,
            encoder_packs=getattr(em, "encoder_packs", None)))
        pool_extra.append(em)
        print(f"[pool] 常驻对手 +{p.name}（参数量 {em.n_params():,}，"
              f"编码器 v{e_ver}"
              + (f"，packs={''.join(getattr(em, 'encoder_packs', ALL_PACKS_V2)) or 'none'}"
                 if e_ver >= 2 else "") + "）")
    if args.snapshot_pool and not pool_extra:
        print("[pool] 警告：--snapshot-pool 未加载到任何模型")

    # ---- 评测锚点 ------------------------------------------------------------
    anchor_model = None
    anchor_version = enc_version
    anchor_packs = None
    if args.eval_anchor:
        ap_path = Path(args.eval_anchor)
        if not ap_path.is_absolute():
            ap_path = ROOT / "models" / (args.eval_anchor
                                         if args.eval_anchor.endswith(".pt")
                                         else f"{args.eval_anchor}.pt")
        if not ap_path.exists():
            raise SystemExit(f"--eval-anchor 找不到模型：{ap_path}")
        anchor_model, anchor_version = build_from_ckpt(
            torch.load(ap_path, map_location="cpu", weights_only=False))
        anchor_packs = getattr(anchor_model, "encoder_packs", None)
        for q in anchor_model.parameters():
            q.requires_grad_(False)
        print(f"[eval] 锚点 = {ap_path.name}（参数量 {anchor_model.n_params():,}，"
              f"编码器 v{anchor_version}）")

    def make_eval():
        t1 = HeuristicBot(seed=101, weights=TEACHER_STYLES["balanced"])
        t2 = HeuristicBot(seed=202, weights=TEACHER_STYLES["aggressive"])
        t3 = HeuristicBot(seed=303, weights=TEACHER_STYLES["defensive"])
        # HeuristicBot.name 是类属性，默认全是 "heuristic"；不复写的话三个教师
        # 在 play_duplicate 里会合并成同一行统计，看不出"打三个不同风格总和"。
        t1.name, t2.name, t3.name = "teacher1", "teacher2", "teacher3"
        if anchor_model is not None:
            # 锚点占一个座位：只剩 2 个教师位。仍是 4 座轮换、同一副牌，
            # 所以 res["net"] 与 res["anchor"] 在第 r 副上是**配对**观测。
            return [
                NetAgent(model, seed=11, temperature=0.0, name="net",
                         encoder_version=enc_version, encoder_packs=enc_packs),
                NetAgent(anchor_model, seed=11, temperature=0.0, name="anchor",
                         encoder_version=anchor_version,
                         encoder_packs=anchor_packs),
                t1, t2,
            ]
        return [
            NetAgent(model, seed=11, temperature=0.0, name="net",
                     encoder_version=enc_version, encoder_packs=enc_packs),
            t1, t2, t3,
        ]

    def run_eval():
        """跑一次训练内评测，返回 ``(某座位的统计字典, 选点分数字典)``。"""
        res, per_deal = play_duplicate(
            make_eval, args.eval_rounds, seed0=4242,
            game_factory=lambda s: GameCls(CfgCls(), seed=s),
            return_per_deal=True)
        scores: dict[str, float] = {}
        if metric_score(res.get("net"), args.select_metric) is not None:
            scores["_select"] = float(metric_score(res["net"], args.select_metric))
        if anchor_model is not None and "anchor" in per_deal and "net" in per_deal:
            d = [x - y for x, y in zip(per_deal["net"], per_deal["anchor"])]
            n = len(d)
            mean = sum(d) / n
            sd = (sum((v - mean) ** 2 for v in d) / (n - 1)) ** 0.5 if n > 1 else 0.0
            res["net"]["vs_anchor"] = mean
            res["net"]["vs_anchor_se"] = sd / n ** 0.5 if n else 0.0
            res["net"]["vs_anchor_deals"] = n
            if args.select_metric == "vs_anchor":
                scores["_select"] = float(mean)
        return res

    history = []
    best_score = -1e9
    eval_log: list[dict] = []   # [{"iter":i,"score":s,"se":e,"eval":{...}}, ...]
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
            snap_sd = pool_extra_ck + [
                ck_from_model(m, encoder_version=enc_version,
                              encoder_packs=(enc_packs if enc_version >= 2 else None))
                for m in snapshots]
            traj = None
            for attempt in range(3):
                try:
                    traj_arr, rewards, dt = collect_parallel(
                        model.state_dict(), nc, args.games_per_iter,
                        args.seed + it * 99991, args.temperature, args.reward_scale,
                        args.teacher_prob, workers=args.workers or 1,
                        past_prob=args.past_prob, snapshots_sd=snap_sd, rules=args.rules,
                        want_traj=True, pool=_pool[0], encoder_version=enc_version,
                        encoder_packs=enc_packs)
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
                snapshots=pool_extra + snapshots, past_prob=args.past_prob,
                game_cls=GameCls, config_cls=CfgCls, encoder_version=enc_version,
                encoder_packs=enc_packs)
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
            res = run_eval()
            rec["eval"] = res["net"]
            printed = {k: (round(v, 3) if isinstance(v, float) else v)
                       for k, v in res["net"].items()}
            print(f"  [eval] {printed}", flush=True)
            cur = metric_score(rec["eval"], args.select_metric)
            if cur is None:
                # 指标在当前评测里不可用（例如选了 vs_anchor 却没给 --eval-anchor）
                # ——退化为 avg_score，并在日志里明说，绝不静默换成别的口径。
                print(f"  [eval] 指标 {args.select_metric} 不可用 → 本次回退 avg_score",
                      flush=True)
                cur = metric_score(rec["eval"], "avg_score")
            cur = float(cur if cur is not None else -1e9)
            rec["select_metric"] = args.select_metric
            rec["select_score"] = cur
            if cur > best_score:
                best_score = cur
                best_ckpt = {
                    "state_dict": {k: v.clone() for k, v in model.state_dict().items()},
                    "config": vars(args),
                    "net": nc,
                    "encoder_version": int(enc_version),
                    "iter": it + 1,
                    "select_metric": args.select_metric,
                    "select_score": cur,
                    "eval": rec["eval"],
                }
                if enc_version >= 2:
                    best_ckpt["encoder_packs"] = list(enc_packs)
                Path(args.out).parent.mkdir(parents=True, exist_ok=True)
                torch.save(best_ckpt, args.out)
            # 选点分自身的标准误：vs_anchor 用配对 SE；其余指标沿用其字段的 SE。
            if args.select_metric == "vs_anchor":
                se = float(rec["eval"].get("vs_anchor_se") or 0.0)
            elif args.select_metric == "avg_score":
                se = float(rec["eval"].get("avg_score_se") or 0.0)
            else:
                se = 0.0
            eval_log.append({"iter": it + 1, "score": cur, "se": se,
                             "eval": dict(rec["eval"])})

        # 周期性检查点：训练结束后用 scripts/score_checkpoints.py 在共用牌池上
        # 对它们做对位评分，才是可信的选点依据（训练内评测分噪声太大）。
        if args.save_every and (it + 1) % args.save_every == 0:
            snap = Path(args.out).parent / "_ckpts" / \
                f"{Path(args.out).stem}_it{it + 1:04d}.pt"
            _dump_ckpt(snap, model, nc, args, it + 1, rec.get("eval"),
                       enc_version, enc_packs)
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
    which, select_note = choose_final(eval_log, args.select, args.select_metric)
    if which == "last" and history:
        _dump_ckpt(args.out, model, nc, args, len(history),
                   history[-1].get("eval"), enc_version, enc_packs)
        if eval_log:
            ck_out = torch.load(args.out, map_location="cpu", weights_only=False)
            ck_out["select_metric"] = args.select_metric
            ck_out["select_score"] = eval_log[-1]["score"]
            torch.save(ck_out, args.out)
    print(f"[select] {select_note}")
    # 选点档案：无论选了谁，都把「如果按别的指标选会选谁」一并写盘，
    # 方便事后用天梯复核——这正是历史 s1–s11 缺的那份记录。
    sel_report = {
        "select": args.select,
        "select_metric": args.select_metric,
        "which": which,
        "note": select_note,
        "saved_iter": (len(history) if which == "last"
                       else (best_ckpt or {}).get("iter")),
        "eval_log": eval_log,
        "alternatives": {m: max(eval_log, key=lambda r: metric_score(r["eval"], m)
                                or -1e9)["iter"]
                         for m in SELECT_METRICS
                         if any(metric_score(r["eval"], m) is not None for r in eval_log)},
    }
    sel_path = Path(args.history).with_name(Path(args.history).stem + "_selection.json")
    sel_path.write_text(json.dumps(sel_report, ensure_ascii=False, indent=2),
                        encoding="utf-8")
    print(f"[select] 按各指标分别会选中的轮次 -> {sel_report['alternatives']}")
    print(f"[select] 选点档案 -> {sel_path}")

    if _pool[0] is not None:
        _pool[0].close()
        _pool[0].join()
    print(f"完成。选点指标 {args.select_metric} 的历史最佳 {best_score:+.3f}，"
          f"模型 -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
