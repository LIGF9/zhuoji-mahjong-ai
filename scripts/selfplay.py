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
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _tree_rss_gb(pid: int) -> float:
    """进程自身 + 全部子孙进程的 RSS 之和（GB）。

    只看**主进程**会严重低估：向量化路径用 spawn 起 worker，Windows 下它们的命令行是
    ``python.exe -c "from multiprocessing.spawn import ..."`` —— 既不叫 ``selfplay.py``，
    也不算进主进程的 RSS。2026-10-12 冒烟实测系统已用 30.13 GB 时主进程只有 3.32 GB，
    12 个 worker 全在盲区里。进程池重建的判据必须看这个数。

    psutil 缺失或取不到时返回 0.0（调用方据此退回纯轮数兜底）。
    """
    try:
        import psutil
    except ImportError:
        return 0.0
    try:
        me = psutil.Process(pid)
        total = me.memory_info().rss
        for kid in me.children(recursive=True):
            try:
                total += kid.memory_info().rss
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
    except Exception:                                   # noqa: BLE001
        return 0.0
    return total / 1024 ** 3


def _total_ram_gb() -> float:
    """机器物理内存（GB），拿不到返回 0.0。"""
    try:
        import psutil
    except ImportError:
        return 0.0
    try:
        return psutil.virtual_memory().total / 1024 ** 3
    except Exception:                                   # noqa: BLE001
        return 0.0


def _sys_mem_pct() -> float:
    """**全系统**内存使用率（%），拿不到返回 0.0。

    为什么判据用系统口径而不是"训练进程树 / 物理内存"
    --------------------------------------------------
    2026-10-12 预检实测：机器上非训练占用（浏览器、编辑器等）基线就有 **≈11.3 GB**
    （系统 22.23 GB - 训练树 10.96 GB）。若按"训练树 ≤ 物理内存 72%"判，训练树可以涨到
    22.9 GB，那时系统总量 = 11.3 + 22.9 = **34.2 GB > 31.8 GB → 直接 OOM**。
    换句话说，训练树占比这个口径**漏掉了基线**，阈值再小也只是在赌别人的占用不变。
    直接用系统使用率，语义与 OOM 一一对应。
    """
    try:
        import psutil
    except ImportError:
        return 0.0
    try:
        return psutil.virtual_memory().percent
    except Exception:                                   # noqa: BLE001
        return 0.0


from zhuoji import RulesConfig, ZhuojiGame  # noqa: E402
from zhuoji.dushan import DushanGame, DushanConfig  # noqa: E402
from zhuoji.agent import NetAgent, play_duplicate  # noqa: E402
from zhuoji.bots import TEACHER_STYLES, HeuristicBot  # noqa: E402
from zhuoji.encoder import (  # noqa: E402
    HEAD_DIMS, dims_for_version, encode, index_to_action,
    pack_desc_for, parse_packs_for)
from zhuoji.encoder_v2 import ALL_PACKS as ALL_PACKS_V2  # noqa: E402
from zhuoji.net import (  # noqa: E402
    build_from_ckpt, build_model, ck_from_model, encoder_packs_from_ckpt,
    encoder_version_from_ckpt, load_into, net_config_from_ckpt, net_meta,
    warm_start_from_v1)
from zhuoji.vcollect import teacher_logprobs  # noqa: E402

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
                  encoder_version: int = 1, encoder_packs=None,
                  explore_eps: float = 0.0, teacher=None, teacher_ver=None,
                  teacher_packs=None, master_prob: float = 0.0):
    """跑 ``n_games`` 局，返回模型自己决策点的训练样本。

    座位配置有三种，随机混合：
    * 全模型自我对弈；
    * 模型 + 启发式教师（提供不同风格的多样性）；
    * 模型 + 历史快照（league 式对手池，避免自我对弈陷入同类循环）。

    轨迹元组与 :func:`zhuoji.vcollect.collect_vectorized` 统一为 **9 元**
    （末尾两项 = 价值头输出 / 教师 log 概率）。单进程路径不支持 master 一等
    对手（``master_prob`` 仅向量化路径生效），但保留同名形参以免调用方分叉。
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
                    val = float(out["value"][0])
                    if temperature > 1e-6:
                        probs = torch.softmax(lg / temperature, dim=-1).numpy()
                        probs = np.clip(probs, 0, None)
                        probs = probs / probs.sum()
                        if explore_eps > 0:
                            msk = obs.mask.astype(np.float64)
                            tot = msk.sum()
                            if tot > 0:
                                probs = ((1.0 - explore_eps) * probs
                                         + explore_eps * (msk / tot))
                                probs = probs / probs.sum()
                        idx = int(rng.choice(len(probs), p=probs))
                    else:
                        idx = int(torch.argmax(lg))
                a = index_to_action(obs.head, idx, game)
                m = np.zeros(28, dtype=np.float32)
                m[:obs.mask.shape[0]] = obs.mask.astype(np.float32)
                tlp = None
                if teacher is not None and teacher_ver is not None:
                    tobs = encode(game, p, teacher_ver, teacher_packs)
                    tlp = teacher_logprobs([tobs], teacher)[0]
                traj.append((obs.tiles, obs.glob, m,
                             HEAD_NAMES.index(obs.head), idx, p, gi, val, tlp))
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
def _gae_advantages(traj, rewards, gamma: float = 1.0, lam: float = 0.95) -> np.ndarray:
    """按 (对局, 座位) 分组做 GAE(λ)（M2-A · A6）。

    为什么**必须分组**而不是整批按时间顺序：轨迹来自**并行推进**的 N 局，
    ``traj`` 的顺序是「第 k 轮决策点按对局顺序」交错排列，不是单一时间轴；
    同一个玩家的相邻两个决策点之间还夹着其他三家的动作。所以把每个
    (game, seat) 各自当成一条**半 MDP 轨迹**：奖励只在终局出现一次
    （该座位该局的最终得点），中间步的自举值取该玩家**下一次**决策的 V。

    γ=1（本游戏无折扣），λ=0.95 用来在「MC 的无偏高方差」与「TD 的低方差高偏差」
    之间取折中 —— 手内自举把信用分配链从 60+ 步压到 ~1/(1−γλ)=20 步量级。

    对照：原实现是纯 MC（``adv = 终局分 − V(s)``），方差大且不能利用局内结构。
    """
    n = len(traj)
    adv = np.zeros(n, dtype=np.float64)
    groups: dict = {}
    for i, t in enumerate(traj):
        groups.setdefault((t[6], t[5]), []).append(i)
    for (gi, ply), idxs in groups.items():
        R = float(rewards[gi, ply])
        vals = [float(traj[i][7]) for i in idxs]
        T = len(idxs)
        g = 0.0
        for k in range(T - 1, -1, -1):
            v = vals[k]
            nxt = vals[k + 1] if k + 1 < T else 0.0
            r = R if k == T - 1 else 0.0
            delta = r + gamma * nxt - v
            g = delta + gamma * lam * g
            adv[idxs[k]] = g
    return adv


def _normalize_by_seat(adv: np.ndarray, seats: np.ndarray) -> np.ndarray:
    """按座位（role）分别标准化优势（RAE / SPIRAL 的做法）。

    整批混合标准化会把「庄家与非庄」「各座位得点尺度不同」这类**系统性差异**
    平均掉，等价于给不同座位施加了不同的有效步长。按座位各自零均值单位方差后，
    优势只在**座位内**可比较，与 actor 无关的尺度被消掉。
    """
    out = np.array(adv, dtype=np.float64, copy=True)
    for s in np.unique(seats):
        m = (seats == s)
        if int(m.sum()) < 2:
            continue
        seg = out[m]
        sd = seg.std()
        if sd > 1e-6:
            out[m] = (seg - seg.mean()) / (sd + 1e-8)
    return out


def _opt_restore_partial(opt, saved_sd: dict) -> int:
    """M2-AUX：结构新增参数后的优化器状态**部分恢复**。

    背景（B/B2/B3 的教训）：整体恢复失败时若直接冷启动 AdamW，骨干的动量
    全部归零 → 首步 Δθ≈lr·sign(g) 全局踢一脚。辅助头参数**追加在参数表
    末尾**（M1Net 的 aux 头在 head_value 之后注册），所以旧检查点的
    第 i 份状态与当前第 i 个参数在 i < len(saved) 范围内一一对应。
    逐位校验形状，不匹配就跳过 —— 宁可个别参数冷启动，也不让噪声进骨干。
    返回成功恢复的参数个数。
    """
    saved = saved_sd.get("state") or {}
    if not saved:
        return 0
    params = [p for g in opt.param_groups for p in g["params"]]
    ok = 0
    for i, p in enumerate(params):
        st = saved.get(i)
        if st is None:
            continue
        ea = st.get("exp_avg")
        if ea is None or tuple(ea.shape) != tuple(p.shape):
            continue
        opt.state[p] = {k: (v.clone() if torch.is_tensor(v) else v)
                        for k, v in st.items()}
        ok += 1
    return ok


def update(model, ref_model, opt, traj, rewards, args, teacher_model=None,
           klt_beta: float = 0.0, aux=None):
    """一轮梯度更新。

    M2-A 的三处改动（都在本函数内）：
    * **A6**：优势从纯 MC 换成 GAE(λ)，并按座位标准化；
    * **A7①**：优势用**采集时回传的 value**，不再重跑一次全量前向（省 ~10 s/轮）；
    * **A4**：新增 ``β(t)·KL(π_teacher ‖ π_θ)``（reverse-KL 蒸馏）。教师分布
      由 worker 侧算好随轨迹带回（``traj[i][8]``），这里零额外前向。

    ``klt_beta = 0`` 或 ``traj`` 里没有教师列时该项自动跳过。

    M2-AUX：``aux`` 是采集端随轨迹带回的完美信息标签字典
    （``sh/opp/wall/danger`` 四组 numpy 数组，与 traj 行一一对应）。
    ``args.aux_coef=0``、``aux=None`` 或模型无辅助头时自动跳过。
    """
    head_arr = np.array([t[3] for t in traj], dtype=np.int64)
    act_arr = np.array([t[4] for t in traj], dtype=np.int64)
    seat_arr = np.array([t[5] for t in traj], dtype=np.int64)
    gi_arr = np.array([t[6] for t in traj], dtype=np.int64)
    rew_arr = np.asarray(rewards, dtype=np.float32)[gi_arr, seat_arr]
    tiles = torch.from_numpy(np.stack([t[0] for t in traj]).astype(np.float32))
    glob = torch.from_numpy(np.stack([t[1] for t in traj]).astype(np.float32))
    mask = torch.from_numpy(np.stack([t[2] for t in traj]).astype(np.float32))
    head = torch.from_numpy(head_arr)
    act = torch.from_numpy(act_arr)
    ret = torch.from_numpy(rew_arr)
    n = tiles.shape[0]
    use_klt = (klt_beta > 0.0 and teacher_model is not None
               and n > 0 and traj[0][8] is not None)
    stats = {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0,
             "kl": None, "klt": 0.0, "aux": 0.0, "n": n,
             "reward": float(rew_arr.mean())}
    nb = 0

    # ---- 优势（A6 / A7①） --------------------------------------------------
    adv_mode = getattr(args, "adv_mode", "gae")
    if adv_mode == "gae":
        adv_np = _gae_advantages(traj, rewards,
                                 float(getattr(args, "gamma", 1.0)),
                                 float(getattr(args, "gae_lambda", 0.95)))
    else:                                   # 对照臂：纯 MC
        adv_np = rew_arr.astype(np.float64) - np.asarray(
            [t[7] for t in traj], dtype=np.float64)
    if getattr(args, "adv_norm", "seat") == "seat":
        adv_np = _normalize_by_seat(adv_np, seat_arr)
    else:
        sd = adv_np.std()
        if sd > 1e-6:
            adv_np = (adv_np - adv_np.mean()) / (sd + 1e-8)
    adv = torch.from_numpy(np.clip(adv_np, -4.0, 4.0).astype(np.float32))

    # ---- 教师 log 概率（A4，worker 已算好，这里只做索引） -------------------
    tlg_all = None
    if use_klt:
        tlg_all = torch.from_numpy(
            np.stack([t[8] for t in traj]).astype(np.float32))

    # ---- M2-AUX：完美信息辅助标签张量 ---------------------------------------
    # 静止目标（环境事实，不随策略变）→ 对共享主干是稳定的监督信号（UNREAL 式）；
    # 对手听牌/危险度标签在输入里不可见 → 教的是「隐藏状态推断」本身（PTIE 式）。
    use_aux = (aux is not None and bool(getattr(model, "aux", False))
               and float(getattr(args, "aux_coef", 0.0)) > 0.0)
    if use_aux:
        aux_sh = torch.from_numpy(np.ascontiguousarray(aux["sh"])).long()
        aux_opp = torch.from_numpy(aux["opp"]).float()
        aux_wall = torch.from_numpy(aux["wall"]).float()
        aux_danger = torch.from_numpy(aux["danger"]).float()
        _disc_id = HEAD_NAMES.index("discard")

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
            klt_all = torch.zeros(sel.numel())

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
            # 教师 log 概率**不在这里前向** —— worker 已在采集时按教师自己的
            # 编码器口径算好（A4）。这里只是按 minibatch 取行，零算力。
            tout = None if tlg_all is None else tlg_all[sel]

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
                if tout is not None:
                    # reverse-KL：KL(π_teacher ‖ π_θ) = Σ π_t·(log π_t − log π_θ)。
                    # 方向是硬要求（A4）：质量覆盖方向，学生只需在教师认为可能的动作上
                    # 接近，仍保留自己的多峰探索；forward-KL 会把分布压进教师的众数、提前锁死。
                    tlg = tout[sub][:, :dim]
                    tp = tlg.exp()          # 非法位 -1e9 → 0，自然不贡献
                    klt_all[sub] = (tp * (tlg - lp)).nan_to_num(0).sum(-1)

            a = adv[sel]
            loss_pi = -(logp_all * a).mean()
            v_pred = out["value"]
            loss_v = F.smooth_l1_loss(v_pred, ret[sel])
            ent = ent_all.mean()
            kl = kl_all.mean()
            loss = loss_pi + args.value_coef * loss_v \
                - args.ent_coef * ent + args.kl_coef * kl
            if tout is not None:
                klt = klt_all.mean()
                loss = loss + klt_beta * klt
                stats["klt"] += float(klt.detach())
            if use_aux:
                # 四项辅助损失各自按自己的子批取均值再求和 —— 保证每项的
                # 梯度尺度独立于该头样本占比（discard 行数变化不影响对手头权重）。
                l_sh = F.cross_entropy(out["aux_shanten"], aux_sh[sel])
                l_op = F.binary_cross_entropy_with_logits(
                    out["aux_opp"], aux_opp[sel])
                l_wa = F.smooth_l1_loss(out["aux_wall"], aux_wall[sel])
                sub_d = (head[sel] == _disc_id).nonzero(as_tuple=True)[0]
                l_da = out["aux_danger"].new_zeros(())
                if sub_d.numel() > 0:
                    # danger 标签语义是「打这张牌会怎样」→ 只在出牌决策点成立；
                    # 非 discard 行的 danger 数组是填零占位，绝不能进损失。
                    l_da = F.binary_cross_entropy_with_logits(
                        out["aux_danger"][sub_d], aux_danger[sel][sub_d])
                aux_l = l_sh + l_op + l_wa + l_da
                loss = loss + args.aux_coef * aux_l
                stats["aux"] += float(aux_l.detach())
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            stats["policy_loss"] += float(loss_pi.detach())
            stats["value_loss"] += float(loss_v.detach())
            stats["entropy"] += float(ent)
            if ref_out is not None:
                stats["kl"] = float(kl.detach()) + (stats["kl"] or 0.0)
            nb += 1
    for k in ("policy_loss", "value_loss", "entropy", "klt", "aux"):
        stats[k] /= max(1, nb)
    if stats["kl"] is not None:
        stats["kl"] /= max(1, nb)
    return stats


class CoefController:
    """熵 / KL 系数的自适应控制器（M2-A · A1 + A3）。

    为什么换掉旧规则
    ----------------
    旧 ``adapt_coefs`` 是**阈值 + 乘性**的死规则：``ent < floor → ×1.8``。它的
    致命问题不是"会不会动"，而是**动得太晚、动得太小、还会反复过冲**：
    一旦熵越过 floor 就立刻反向放松 —— 于是熵在 floor 附近来回震荡而不是被
    *保持*在某个健康带内；而当策略真的先坍缩下去，乘性加大在几百轮尺度上
    追不上坍缩速度。实测（2026-10-14 审计 S1→S3 共 2450 轮）``ec`` 全程钉在
    0.03、``kc`` 全程钉在 0.02 —— **自适应控制器从未真正生效**，这就是
    "熵坍缩"的直接机制。

    新控制器（AER 式**比例控制**）
    ----------------------------
        H₀ = 段首 N 轮的熵均值（不拿启动期的偏高值当基线）
        H* = ratio · H₀
        |H − H*| ≤ deadzone → 不动（死区，抑制震荡）
        否则 ent_coef ← clip(ent_coef · exp(η·(H* − H)), 0.01, 0.5)

    指数形式的好处：对数空间里是**线性比例控制**，H 偏离多少就补偿多少，
    不会像乘 1.8 那样一步跳过头；而且 *双向**对称**（偏低就加、偏高就减）。

    KL 侧（A3）
    -----------
    沿用原来的相对规则（超线 ×1.6、低于带下沿 ÷1.4），但有**三处必须纠正**：

    ★ ① 必须对**锚点年龄**归一化。``update()`` 返回的 ``kl`` 是
      ``KL(π_θ ‖ π_anchor)`` —— 它是**距离**，不是速率：锚点每 ``--kl-anchor-rolling``
      轮才 EMA 刷新一次，所以同样健康的训练，测出来的 kl 会随"锚点有多旧"线性增长。
      实测（2026-10-14 冒烟，无控制）：``iter 1 → 0.0232``、``iter 5 → 0.1540``、
      ``iter 9 → 0.2564``、``iter 13 → 0.3284`` —— 看着像在爆，换成**每轮率**
      却是 0.023 / 0.031 / 0.028 / 0.025，**稳得像常数**。
      所以判据用 ``rate = kl / age``，``--kl-target`` 的语义是"每轮漂移预算"。
      这样它对 ``--kl-anchor-every``（A7② 的降频）与 ``--kl-anchor-rolling`` 都是
      **不变**的 —— 否则改任一个都会让目标量级失效（这一条正是把 0.25 改 0.05
      之后反而"每次测量都触发、25 轮撞上 clamp 上限 0.5"的原因）。

    ★ ② 熵基线窗口里**不能提前 return**（见 ``step()`` 里的注释）。

    ★ ③ 放松阈值必须随"目标语义"一起改：目标既然是**危险线**，正常率就应落在带内、
      系数完全静止；若放松阈值仍取 ``0.4×target``（=0.02），会与实测正常率
      （0.019–0.031）重合 → 每个测量点都在"放松"，kc 被静默衰减到下限 0.01。
      改取 ``--kl-relax-ratio``（默认 0.25）。不对称是有意的、风险导向的：
      过度锚定会把学习锁死（致命），锚项偏弱只是少一层温和正则（可接受）。

    结果：正常训练下 rate ≈ 0.019–0.031 ∈ 带 [0.0125, 0.05] → 系数静止不动
    （与 S1–S3 的 0.02 一致，**少变更一个变量**）；只有漂移**加速**到 >0.05/轮
    （坍缩前的特征）才逐次×1.6 刹车。它是一个安全网，不是一个恒定的拉力。

    ★ ④ 熵侧与 KL 侧**语义相反**，这一点最容易搞错：
      ``--kl-target`` 是**危险线** → 常态静默才是正确（上面那条）；
      而 ``H*`` 是**要维持的目标** → 它必须**可达**，否则整套控制静默。
      60 轮实测：熵的稳态只有段首的 **91.6%**，而 ratio 原本是 0.75
      → ``H*=0.769`` 永远达不到 → ``ec`` 从起点（下限 0.01）**无路可走**
      → **60/60 轮零变化**（缺陷 C5）。这与它要修的旧规则把 ``ec`` 钉在 0.03
      是**同一类失效**，只是位置不同。
      现默认 **0.95**（H*≈0.974，落在稳态上方 0.036 > 死区 0.03）→ 控制器接管，
      实测增益 ``dH/d(ec)≈1.18`` → ``ec`` 稳定在 ~0.05（离两端都远）。
      两道防线防止再犯：① 基线窗口一结束就打**可达性预判**（见 ``step()``）；
      ② 贴端点持续告警 —— 下限+H>H* 只是目标偏保守，**上限+H<H* 才是真故障**。
      标定口径：``scripts/ent_gain_probe.py``（两组只差 ``--ent-coef`` 的短跑量增益）。
    """

    ENT_LO, ENT_HI = 0.01, 0.5
    KL_LO, KL_HI = 0.01, 0.5
    # 熵比例控制的对数增益（单一事实源：argparse 默认值与
    # scripts/test_coef_controller.py 的 T8 扫描都读这里）。
    #
    # ★ 2026-10-14 从 4.0 下调到 0.5。标定过程见 T8（玩具一阶被控对象 + η 扫描）：
    #   η=4.0 时 ec 从 0.06 撞到下限 0.01 **只用 2 步**、从 0.01 撞到上限 0.5 只用 4 步
    #   —— 全程在 [0.01,0.5] 两端继电器式来回，探索强度每几轮翻一倍，等于给训练
    #   注入高频噪声；而 η=0.5 对应"持续偏 0.10 nat 时 20 轮推一个 e 折"，
    #   既慢于被控对象的一轮延迟（不来回过冲），又远快于 600 轮段的时间尺度。
    ETA_DEFAULT = 0.5
    # 熵目标比例（H* = RATIO_DEFAULT · H₀）的单一事实源：argparse 默认值、
    # scripts/test_coef_controller.py 的 mk()/T8、scripts/verify_segment_recipe.py
    # 的期望值都读它。
    #
    # ★ 2026-10-14 从 0.75 上调到 0.95，依据是**两组实测**而不是直觉：
    #   ① 熵的自然稳态只有段首的 91.6%（60 轮：H₀=1.0253 → 稳态 0.9389）。
    #      0.75 意味着 H*=0.769 —— 数学上不可达 → ec 只减不加、起点即下限
    #      → 60/60 轮零变化（缺陷 C5），与旧规则钉死 ec 是同一类失效。
    #   ② 增益探针实测 dH/d(ec) ≈ 1.18（ec 0.02→0.30 把 H 从 0.938 抬到 1.267）。
    #      反推：0.95 → H*=0.974（稳态上方 0.036 > 死区 0.03）→ 控制器接管，
    #      ec 稳定在 ~0.05（上限的 10%、下限的 5 倍）。环路增益 η·g·ec≈0.03 ≪ 1。
    RATIO_DEFAULT = 0.95
    # 贴死端点多少轮后告警。取 25：熵的响应有滞后（KLT/动量的时间尺度约 10 轮），
    # 25 轮足以区分"暂时贴住"与"根本推不动"；又小到能在 60 轮冒烟里就触发。
    PEG_WARN_ITERS = 25

    def __init__(self, args, log=None):
        self.args = args
        self.eta = float(getattr(args, "ent_eta", self.ETA_DEFAULT) or self.ETA_DEFAULT)
        # 熵目标比例也在这里固化一次（单一事实源）：`step()` 与播报都用 self.ratio，
        # 免得三处各写一遍 `a.ent_target_ratio` 而漏改某一处（C5 就是这么来的）。
        self.ratio = float(getattr(args, "ent_target_ratio", self.RATIO_DEFAULT)
                           or self.RATIO_DEFAULT)
        self.dead = float(getattr(args, "ent_deadzone", 0.03))
        # M2-C2（2026-10-18）：--no-ent-control 彻底关掉**熵侧**（KL 侧照常）。
        # 动机：m2c1 冷启动段 H0=2.057（前 25 轮策略≈随机）→ H*=1.954 ≈ 随机策略熵，
        # 策略一学习熵就低于 H*，控制器把 ent_coef 乘到 clamp 上限 0.500 并持续
        # 300 轮把熵从 1.72 反推回 1.92 —— 熵奖励在**对抗收敛**（R11 的又一实证）。
        # 实现方式：把死区放到无穷大 ⇒ 任何 |H−H*| 都判「无差异」⇒ ec 恒定不动。
        # 不用「跳过整段」的写法是为了零侵入：KL 侧与日志仍走同一条路径。
        if not getattr(args, "ent_control", True):
            self.dead = 1e9
        self.relax_ratio = float(getattr(args, "kl_relax_ratio", 0.25) or 0.25)
        self.n_base = int(getattr(args, "ent_baseline_iters", 25))
        self.ents: list[float] = []
        h0 = float(getattr(args, "ent_baseline", 0.0) or 0.0)
        self.h0 = h0 if h0 > 0 else None
        self.log = log or (lambda _m: None)
        self.frozen = self.h0 is None
        self._log_it = -10 ** 9      # 系数变化播报的限频游标
        # 锚点最后一次"对齐到当前策略"的**轮次**（0-based）：``anchor_it = k`` 表示
        # 「第 k 轮结束后的策略 == 锚点」。年龄 = ``it - anchor_it``，所以 -1 对应
        # 「锚点就是 --init、训练开始前就位」（即 --anchor-warmup 0 的情形）。
        # KL 判据要用它把"距离"换算成"每轮率"。
        self.anchor_it = -1
        self.last_rate = None
        # 贴死端点检测（见 step() 的注释）：连续贴同一端点且方向仍与 gap 一致的轮数。
        self._peg_streak = 0
        self._peg_warned = False

    def note_anchor(self, it: int) -> None:
        """锚点刚刚被（重新）对齐到当前策略 —— 主循环在建锚 / 滚动刷新 / 看门狗回退时调。"""
        self.anchor_it = int(it)

    def step(self, it: int, ent: float, kl, kl_rate: float | None = None):
        a = self.args
        ec0, kc0 = float(a.ent_coef), float(a.kl_coef)
        if self.h0 is None:
            # 基线窗口：只记录、**不动 ent_coef**。用启动期（熵天然偏高）的滑动均值
            # 当 H₀ 会把目标顶到虚高的位置，反而鼓励过度探索，所以宁可先冻结熵侧。
            #
            # ★ 这里**绝不能 return**：KL 侧（A3）与熵基线互相独立，提前返回会让
            #   `--anchor-warmup 0` 的段在前 n_base 轮里 KL 随便飙也没人管。
            #   2026-10-14 冒烟实测：iter 5 的 KL 已经 0.1540（target 0.05，3 倍），
            #   而 kc 全程钉在 0.020 —— 因为这一行把 KL 分支一起吞了。
            self.ents.append(ent)
            if len(self.ents) >= self.n_base:
                self.h0 = float(np.mean(self.ents))
                self.frozen = False
                hstar = self.ratio * self.h0
                self.log(f"[coef] 熵基线 H0={self.h0:.3f}（段首 {len(self.ents)} 轮）"
                         f" → H*={hstar:.3f}；ent_coef 由 {a.ent_coef:.3f} 起控")
                # ---- 可达性预判（2026-10-14 新增，C5 的直接产物）----
                # C5 的教训：比例控制只有在 H* **可达**时才有意义。60 轮冒烟里
                # H 的自然稳态是段首的 91.6%，而 H* 被设成 75% → H > H* 恒成立
                # → exp(η(H*−H)) < 1 恒成立 → ec 只会减小，而它一开始就在下限
                # → **ec 全程静止，A1 等于没上**（与它要修的旧缺陷同类）。
                # 光看「ec 没动」要到几十轮后才可疑，所以在基线一确定就打预判：
                # 拿基线窗口**尾部**的熵（≈稳态）跟 H* 比，差多少就报多少。
                k = max(3, len(self.ents) // 3)
                h_tail = float(np.mean(self.ents[-k:]))
                slack = h_tail - hstar
                if abs(slack) > 2.0 * self.dead:
                    self.log(f"[coef] ℹ️ 可达性预判：段首窗口尾部 H≈{h_tail:.3f}，"
                             f"与 H* 相差 {slack:+.3f}（> 2×死区 {2 * self.dead:.2f}）→ "
                             + (f"熵还要再降 {slack:.3f} 控制器才开始接管，"
                                f"期间 ec 会静止在下限（A1 事实空转）"
                                if slack > 0 else
                                f"H* 高于当前稳态，ec 会开始上调"))
                    if slack > 0:
                        # 反推「要立刻接管，ratio 至少该多大」
                        need = (h_tail / self.h0) if self.h0 else 0.0
                        self.log(f"[coef] ℹ️ 若要立刻接管，ratio 需 ≳ "
                                 f"{need:.3f}（当前 {self.ratio:.3f}）")
        else:
            hstar = a.ent_target_ratio * self.h0
            if abs(ent - hstar) > self.dead:
                a.ent_coef = float(np.clip(
                    a.ent_coef * math.exp(self.eta * (hstar - ent)),
                    self.ENT_LO, self.ENT_HI))
            # ---- 贴死端点检测（唯一的真故障形态）----
            # 贴下限 + H > H* 不是故障（目标已满足、无需动作），只是目标偏保守，
            # 上面那条预判已经报过。真正无解的是：**ec 顶在上限、熵却仍低于 H***
            # —— 熵奖励开到最大也推不上去，说明 H* 在这个配置下不可达（增益不足
            # 或被 KLT/KL 锚压住）。这时比例控制会永远贴在上限输出最大熵奖励，
            # 既费解又危险，必须显式告警而不是等段末看结果反推。
            gap = hstar - ent
            peg = ((a.ent_coef >= self.ENT_HI - 1e-9 and gap > 0)
                   or (a.ent_coef <= self.ENT_LO + 1e-9 and gap < 0))
            self._peg_streak = self._peg_streak + 1 if peg else 0
            if self._peg_streak == self.PEG_WARN_ITERS and not self._peg_warned:
                self._peg_warned = True
                self.log(f"[coef] ⚠️ ent_coef 已连续 {self._peg_streak} 轮贴死在"
                         f"{'上限 0.50' if a.ent_coef >= self.ENT_HI - 1e-9 else '下限 0.01'}"
                         f" 而 H 仍{'低于' if gap > 0 else '高于'} H*（H={ent:.3f} "
                         f"H*={hstar:.3f}）→ A1 在这个配置下失效："
                         + ("熵奖励开到最大也推不动熵，H* 不可达"
                            if gap > 0 else
                            "目标低于自然稳态，熵会自由下滑到 H* 才被接管")
                         + f"；建议调 --ent-target-ratio（当前 {self.ratio:.3f}）")
        # ---- KL 侧（A3）：降频那几轮 kl=None → 跳过；预热期同样不动 ----
        # ★ 判据用「每轮漂移率」而不是裸的 kl —— 理由见类 docstring 的 ★①。
        #
        # ★③ 容差带必须随"目标语义"一起改。``--kl-target`` 现在是**危险线**（不是
        #   期望值），所以正常率应当落在带内、系数完全静止；放松的阈值若仍取
        #   ``0.4×target``（=0.02），就会与实测正常率（0.019–0.031）**重合** →
        #   每个测量点都在"放松"，kc 被悄悄衰减到下限 0.01，静默改掉历史基线。
        #   取 ``--kl-relax-ratio``（默认 0.25 → 带下沿 0.0125，约为正常率的一半）：
        #   只有率低到"锚项实际上没在干活"才放松。
        #   不对称是有意的、也是风险导向的：过度锚定会把学习锁死（致命），
        #   而锚项偏弱只是少一层温和正则（可接受）→ 快刹车、慢松手。
        if kl is not None and it >= int(getattr(a, "anchor_warmup", 0)):
            # 主循环会把「每轮率」算好传进来（它才是锚点事件的单一事实源）；
            # 单独用控制器时（单测）退回自算，年龄取 note_anchor 记下的轮次。
            rate = float(kl_rate) if kl_rate is not None \
                else float(kl) / max(1, it - self.anchor_it)
            self.last_rate = rate
            if rate > a.kl_target:
                a.kl_coef = min(a.kl_coef * 1.6, self.KL_HI)
            elif rate < a.kl_target * self.relax_ratio:
                a.kl_coef = max(a.kl_coef / 1.4, self.KL_LO)
            a.kl_coef = float(np.clip(a.kl_coef, self.KL_LO, self.KL_HI))
        # 变化播报（限频，每 10 轮最多一行）：段末报告要能拿它说清"系数到底怎么动的"。
        # 注意它是**独立的 [coef] 行**，不是 iter 行的追加字段 —— 判读脚本的 _ITER_RE
        # 只认 iter 行，多出来的行会被自然忽略，且新字段只能追加在 iter 行行尾。
        if (it - self._log_it >= 10
                and (abs(a.ent_coef - ec0) > 1e-9 or abs(a.kl_coef - kc0) > 1e-9)):
            self._log_it = it
            _h = self.ratio * (self.h0 if self.h0 else 0.0)
            _kl = "—" if kl is None else f"{kl:.4f}"
            _rt = "" if self.last_rate is None else f" rate={self.last_rate:.4f}/轮"
            _chg = []
            if abs(a.ent_coef - ec0) > 1e-9:
                _chg.append(f"ec {ec0:.3f}→{a.ent_coef:.3f}")
            if abs(a.kl_coef - kc0) > 1e-9:
                _chg.append(f"kc {kc0:.3f}→{a.kl_coef:.3f}")
            self.log(f"[coef] iter {it + 1}  " + "  ".join(_chg)
                     + f"  （H={ent:.3f} H*={_h:.3f} kl={_kl}{_rt}）")


def _ema_anchor(ref_model, model, alpha: float = 0.3) -> None:
    """把 KL 锚点刷新成「旧锚点与新策略」的指数滑动平均（A3 的滚动锚）。

    为什么要滚动：长段内锚点会过期 —— 锚在 300 轮前的自己身上时，KL 项在惩罚
    "正常的进步"，于是 ``kl_coef`` 会被自适应一路压到下限、KL 项形同虚设
    （或反过来顶到上限把学习锁死）。EMA 锚只惩罚**相对近期自我的漂移**，
    这正是防坍缩想要的东西。

    只对浮点张量做 EMA；``num_batches_tracked`` 这类整数缓冲直接复制。
    """
    with torch.no_grad():
        msd = model.state_dict()
        rsd = ref_model.state_dict()
        for k, rv in rsd.items():
            mv = msd.get(k)
            if mv is None:
                continue
            if rv.dtype.is_floating_point:
                rv.mul_(1.0 - alpha).add_(mv.detach(), alpha=alpha)
            else:
                rv.copy_(mv)


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
def _dump_ckpt(path, model, nc, args, it_no, evalr,
               encoder_version: int = 1, encoder_packs=None, opt=None) -> None:
    """把一个检查点写到 ``path``（``--out`` 与周期性检查点共用）。

    ``encoder_version`` 必须写进去：加载方靠它决定用哪一版编码器，
    老检查点没有这个字段 → 默认 v1（向后兼容）。

    ``encoder_packs`` 只在 v2 时写：消融实验的训练臂关掉了某个包，
    评测时必须关掉同样的包，靠这个字段自动复现，而不是靠人记得。

    ``opt``（B4）：传优化器则把 AdamW 状态一并写盘（体积约 ×3），
    加载方 ``--init`` 时自动恢复 —— 这是跨段续训**不掉动量**的唯一途径。
    不传 = 历史行为（只存权重）。
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
    if opt is not None:
        # AdamW 的 exp_avg / exp_avg_sq 是 step 数量的字典；直接 state_dict() 即可。
        ck_out["opt"] = opt.state_dict()
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
    # ★ 日志重定向必须由**子进程自己**打开，不能靠父进程传下来的 stdout 句柄。
    #   2026-10-15 事故：`m1_launch_segment.py` 用 DETACHED 起训练后立刻退出
    #   （`with log.open(...)` 块结束 → 句柄关闭），训练随后**零输出即死** ——
    #   日志里连启动横幅都没有，只有 launcher 写的那两行。
    #   根因是训练进程的 stdout 继承自一个很快消失的父进程。这里改成自己 open：
    #   `--log-file X` → 把 stdout/stderr 都指到 X（行缓冲），并把它从 argv 里摘掉
    #   以免 argparse 报未知参数。放在 `main()` 里（而不是模块级）还有个好处：
    #   multiprocessing 的 spawn worker 只 import 本模块、不执行 main，
    #   所以它们不会来抢这个日志文件。
    if "--log-file" in sys.argv:
        _i = sys.argv.index("--log-file")
        if _i + 1 < len(sys.argv) and sys.argv[_i + 1]:
            _p = Path(sys.argv[_i + 1])
            _p.parent.mkdir(parents=True, exist_ok=True)
            _fh = open(_p, "a", encoding="utf-8", buffering=1)
            sys.stdout = _fh
            sys.stderr = _fh
        del sys.argv[_i:_i + 2]

    ap = argparse.ArgumentParser()
    ap.add_argument("--init", default=str(ROOT / "models" / "bc.pt"))
    ap.add_argument("--out", default=str(ROOT / "models" / "rl.pt"))
    ap.add_argument("--history", default=str(ROOT / "reports" / "rl_history.json"))
    ap.add_argument("--history-append", action="store_true",
                    help="续训场景：--history 已存在时先读入再追加，而不是整份覆盖。"
                         "默认关（历史行为）。配合 --iter-offset 可让两段拼成一条连续曲线。")
    ap.add_argument("--iters", type=int, default=30)
    ap.add_argument("--iter-offset", type=int, default=0,
                    help="**续训**用：本段第 1 轮在全局编号里的起点减 1。"
                         "它只影响「对外显示的轮次编号」——即 stdout 的 iter、"
                         "history 记录的 iter、以及 _ckpts 的 itNNNN 文件名——"
                         "不改变任何训练语义。用途：段末崩溃后续跑时，让"
                         "`_ckpts/{stem}_it0280.pt` 不会覆盖第一段已存的 it0280，"
                         "且判读脚本（A6 取 iter 17–24、A7 查预热期）看到的是连续序列。")
    ap.add_argument("--games-per-iter", type=int, default=64)
    ap.add_argument("--epochs-per-iter", type=int, default=2)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--lr-min", type=float, default=0.0,
                    help="A8：余弦退火的终点学习率（0=不做退火，保持 --lr 恒定，"
                         "即历史行为）。M2-A 用 1.2e-5 → 4e-6。")
    ap.add_argument("--lr-warmup", type=int, default=0,
                    help="B4：前 N 轮把 lr 从 lr0/N 线性升到 lr0（0=关闭，即历史行为）。"
                         "为什么续训段需要它：检查点里**不含 AdamW 的动量/二阶矩**"
                         "（见 _dump_ckpt），跨段重启时 Adam 处于冷启动 —— 首步更新"
                         "≈ lr·sign(g)，是**与梯度大小无关**的 lr 量级扰动，"
                         "打在一个已退火到 lr_min 的收敛模型上等于全局随机踢一脚。"
                         "warmup 让 v 先积累起来再放大步长。")
    ap.add_argument("--save-opt", action="store_true",
                    help="B4：把 AdamW 的优化器状态一并写进检查点（体积约 ×3），"
                         "并在 --init 时自动恢复 —— 今后所有续训段都能无缝接续动量。"
                         "注意：老检查点没有该字段，从它们起跑仍会冷启动，需配 --lr-warmup。")
    ap.add_argument("--temperature", type=float, default=1.1)
    ap.add_argument("--teacher-prob", type=float, default=0.5)
    ap.add_argument("--reward-scale", type=float, default=8.0)
    ap.add_argument("--value-coef", type=float, default=0.5)
    ap.add_argument("--ent-coef", type=float, default=0.01)
    ap.add_argument("--aux-heads", choices=["off", "on"], default="off",
                    help="M2-AUX：挂 4 个完美信息辅助头（自家向听 / 对手听牌 / "
                         "牌墙余量 / 27 牌危险度），PTIE 式 —— 训练时算标签、"
                         "执行时零开销。老检查点起跑时辅助头为全新初始化"
                         "（load_into 自动识别缺失键）。")
    ap.add_argument("--aux-coef", type=float, default=0.4,
                    help="M2-AUX：四项辅助损失之和的权重。量级参照：CE(向听)"
                         "≈2.0 起步、BCE(对手)≈0.69、BCE(危险)≈0.4、"
                         "smooth_l1(墙)≈0.02 → 合计 ≈3；×0.4 ≈ 1.2，与"
                         "value 项(≈0.35)同量级偏强，属有意设计（辅助信号是"
                         "本轮的主角）。0 = 完全关闭。")
    ap.add_argument("--aux-warmup", type=int, default=0,
                    help="M2-AUX：辅助系数线性预热轮数（前 N 轮 aux_coef 按 "
                         "it/N 线性升至满值，0=不预热）。m2a1 教训：全新辅助头"
                         "起步 aux 损失 ≈4.5，×0.4 后对共享主干的梯度达 RL "
                         "主干的 56%，叠加全局 clip(5.0) 严重干涉策略梯度 → "
                         "段末 −0.99 显著劣。探针 scripts/aux_grad_probe.py 实测。")
    ap.add_argument("--kl-coef", type=float, default=0.02)
    ap.add_argument("--kl-target", type=float, default=0.05,
                    help="A3：KL 的**每轮漂移率危险线**（rate = kl ÷ 锚点年龄）。"
                         "率 > target → kl_coef ×1.6；率 < target×--kl-relax-ratio → ÷1.4。"
                         "★ 单位是率不是裸 kl：锚点每 --kl-anchor-rolling 轮才刷新，"
                         "裸 kl 是「距锚点的距离」、会随锚点变旧单调增长，健康训练也能涨到 1+；"
                         "实测的率却是稳的（0.019–0.031/轮）。用距离判据会在 25 轮内把 "
                         "kl_coef 顶到 clamp 上限 0.5，把模型锁在一个过期锚上。"
                         "标定：scripts/kl_scale_probe.py（峰值率×1.5 ≈ 0.05）。")
    ap.add_argument("--kl-relax-ratio", type=float, default=0.25,
                    help="A3：放松阈值 = target×该比例。默认 0.25（带下沿 0.0125，约正常率的"
                         "一半）。**不要调回 0.4** —— 0.4×0.05=0.02 会与实测正常率"
                         "（0.019–0.031）重合，每个测量点都触发放松、kc 被静默衰减到下限。")
    ap.add_argument("--ent-floor", type=float, default=0.0,
                    help="[已弃用] 旧阈值规则的熵下限。自适应已改为 M2-A · A1 的"
                         "比例控制器（见 --ent-target-ratio），此参数仅为向后兼容保留。")
    ap.add_argument("--ent-target-ratio", type=float, default=CoefController.RATIO_DEFAULT,
                    help="A1：熵目标 = ratio × 段首熵基线 H₀。"
                         "★ 默认已按**实测**标定（2026-10-14）："
                         "原值 0.75（M2-0 之前的估计）在数学上**不可达** —— "
                         "熵的自然稳态只有段首的 91.6%%（H₀=1.0253 → 稳态 0.9389），"
                         "0.75·H₀=0.769 → H>H* 恒成立 → ec 只会往下调、而起点就在下限 0.01 "
                         "→ **60/60 轮零变化**（缺陷 C5）。"
                         "现取 0.95（H*≈0.974，落在稳态上方 0.036 > 死区 0.03）→ 控制器真正接管；"
                         "实测增益 dH/d(ec)≈1.18（scripts/ent_gain_probe.py）→ "
                         "预期 ec 稳定在 ~0.05（上限 0.5 的 10%%、下限 0.01 的 5 倍）。"
                         "环路增益 η·g·ec≈0.03 ≪ 1 → 无震荡。")
    ap.add_argument("--ent-baseline-iters", type=int, default=25,
                    help="A1：用段首多少轮的熵均值当基线 H₀。这段窗口内**冻结**"
                         "ent_coef（启动期熵天然偏高，用它当基线会把目标顶到虚高）。")
    ap.add_argument("--ent-baseline", type=float, default=0.0,
                    help="A1：直接指定 H₀（>0 时跳过基线窗口，立刻开始控制）。")
    ap.add_argument("--ent-eta", type=float, default=CoefController.ETA_DEFAULT,
                    help="A1：比例控制增益 η，ent_coef ← clip(ent_coef·exp(η(H*−H)))。"
                         f"默认 {CoefController.ETA_DEFAULT}（**不是**初版的 4.0：η=4 时"
                         "ec 在 clamp 两端继电器式来回，见 scripts/test_coef_controller.py"
                         " 的 T8 增益扫描）。")
    ap.add_argument("--no-ent-control", dest="ent_control", action="store_false",
                    help="M2-C2：**只关熵侧**自适应控制（KL 侧照常工作）。用于冷启动："
                         "段首 25 轮的熵≈随机策略熵，H*=ratio·H₀ 会把目标顶到随机水平，"
                         "策略一收敛 ec 就被顶到 clamp 上限反向拉熵（m2c1 实测贴 0.500 "
                         "整整 300 轮）。关掉后 ent_coef 恒定，熵自由下降。")
    ap.add_argument("--ent-deadzone", type=float, default=0.03,
                    help="A1：熵死区（|H−H*| ≤ deadzone 时不动系数，抑制震荡）。")
    ap.add_argument("--no-adaptive", action="store_true",
                    help="关闭自适应系数控制（默认开启）")
    ap.add_argument("--explore-eps", type=float, default=0.0,
                    help="A2：自博弈采样时以该概率替换为**均匀合法动作**（ε-混合）。"
                         "0=关。M2-A 用 0.03 → 退火到 --explore-eps-end。")
    ap.add_argument("--explore-eps-end", type=float, default=0.01,
                    help="A2：ε 的退火终点（默认 1%%）。")
    ap.add_argument("--explore-anneal-iters", type=int, default=300,
                    help="A2：ε 从 --explore-eps 线性退火到终点的轮数。")
    ap.add_argument("--adv-mode", choices=["gae", "mc"], default="gae",
                    help="A6：优势估计。gae=GAE(λ)（默认，手内自举）；mc=纯 MC"
                         "（历史行为，保留作对照臂）。")
    ap.add_argument("--gae-lambda", type=float, default=0.95)
    ap.add_argument("--gamma", type=float, default=1.0)
    ap.add_argument("--adv-norm", choices=["seat", "batch"], default="seat",
                    help="A6：优势标准化口径。seat=按座位（role）分别标准化（默认，"
                         "RAE/SPIRAL 做法）；batch=整批混合标准化（历史行为）。")
    ap.add_argument("--kl-anchor-every", type=int, default=1,
                    help="A7②：KL 锚前向每多少轮做一次（1=每轮）。锚只前向不反向，"
                         "占单轮 ~9 s；降到每 4 轮一次可省 ~7 s，代价是该项的正则"
                         "强度按比例下降（自适应控制器用最近一次有效 KL 值判定）。")
    ap.add_argument("--kl-anchor-rolling", type=int, default=0,
                    help="A3：每 N 轮把 KL 锚刷新为「旧锚与当前策略」的 EMA"
                         "（0=不刷新，锚固定在 --init）。")
    ap.add_argument("--kl-anchor-alpha", type=float, default=0.3,
                    help="A3：EMA 刷新时当前策略的权重（默认 0.3，即 0.7/0.3）。")
    ap.add_argument("--master-ckpt", default="",
                    help="A4/A5 的 master：既作 KLT 蒸馏的教师（--klt-beta>0 时），"
                         "也作一等对手（--master-seat-prob>0 时）。裸名或路径。")
    ap.add_argument("--master-seat-prob", type=float, default=0.0,
                    help="A5：master 作为**一等对手**的每局出场概率。命中该分支时"
                         "只替换 1~2 个座位（与 teacher/past 分支同口径）。"
                         "关键意义：master 被**移出自快照池**，出场率与「池里有几份自快照」"
                         "无关 —— 旧写法下 8 份自快照会把 master 稀释到 1/9。0=关。")
    ap.add_argument("--klt-beta", type=float, default=0.0,
                    help="A4：KLT 蒸馏系数上限 β₀（0=关）。损失加 β(t)·"
                         "KL(π_teacher ‖ π_θ)（reverse-KL）。")
    ap.add_argument("--klt-beta-end", type=float, default=0.0,
                    help="M2-C2：β 退火的**终点**（默认 0，即退净）。>0 时保留一条弱"
                         "蒸馏直到段末：m2c1 实测 β 撤到 0 后 KLT 从 0.39 反弹到 0.73，"
                         "学生脱离教师后再无有效学习信号。")
    ap.add_argument("--klt-anneal-iters", type=int, default=300,
                    help="A4：β 从 β₀ 线性退火到 --klt-beta-end 的轮数（退火到 0 是硬要求，"
                         "否则策略上限被钉在教师上）。0=不退火（固定 β₀）。")
    ap.add_argument("--klt-hold-iters", type=int, default=0,
                    help="M2-G1：退火开始前**先按 β₀ 保持**的轮数（默认 0 = 旧行为，"
                         "从第 0 轮就开始退火）。用途：末段才解绑 KL 拉力——"
                         "先让蒸馏把策略拉到教师附近，再把 β 退净让学生越过教师。")
    ap.add_argument("--klt-teacher", default="",
                    help="M2-C5 归因开关：**只**替换 KLT 蒸馏的教师，不动 --master-ckpt "
                         "那个一等对手座位。留空 = 教师与 master 同一个模型（旧行为）。"
                         "动机：m2c4 起 --master-ckpt 一物两用，+0.386 无法拆"
                         "「教师更强」与「陪练更强」。裸名或路径。")
    ap.add_argument("--eval-rounds", type=int, default=120)
    ap.add_argument("--eval-every", type=int, default=40)
    ap.add_argument("--eval-anchor", default="",
                    help="把某个模型（裸文件名，不带 .pt，如 dushan_big_s11）作为"
                         "**锚点**放进评测桌轮换，逐副记录 (本模型分 − 锚点分) 的配对"
                         "均值与 SE，作为选点指标 ``vs_anchor``。这一项零额外算力"
                         "（仍是同样的牌数与局数），但配对消掉了牌运方差，且口径与"
                         "对外的 scripts/score_checkpoints.py 完全一致。留空则关闭。")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--no-flush-denormal", action="store_true",
                    help="关闭 denormal 冲刷（默认开启）。开启时把非规格化浮点"
                         "（绝对值小于约 1e-38）当作零处理，避免 AVX2 上的微码慢路径。"
                         "实测 i7-14700K 上 update 段提速 1.14 倍（3 轮交叉验证全为正）。")
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
                    help="多进程向量化采集（比单进程快一个量级）")
    ap.add_argument("--no-snap-cache", action="store_true",
                    help="关闭 worker 侧快照缓存，每轮全量重发快照（A/B 对照用）")
    ap.add_argument("--workers", type=int, default=6,
                    help="向量化采集的进程数（0/1=单进程）")
    ap.add_argument("--pool-rebuild-every", type=int, default=250,
                    help="进程池**轮数兜底**重建间隔（0=不按轮数重建）。Windows 长驻 "
                         "spawn worker 会累积内存（2026-10-12 实测 ≈6–9 GB/小时），重建是"
                         "唯一能把它打回起点的机制。M2-A · A7 从 150 上调到 250"
                         "（有了更灵敏的 RSS 判据之后，轮数兜底只是最后一道保险）。")
    ap.add_argument("--pool-rss-gb", type=float, default=16.0,
                    help="A7：**训练进程树 RSS** ≥ 该值（GB）即重建进程池（0=关）。"
                         "这是主判据：worker 的累积内存才是重建要解决的问题，"
                         "而系统级占用会把浏览器/编辑器的基线（本机 ≈11 GB）算进来。"
                         "16 GB 树 + 11 GB 基线 = 27 GB < 31.8 GB，留 4.8 GB 余量。")
    ap.add_argument("--snapshot-keep", type=int, default=4,
                    help="A7：自快照池保留最近多少份。从 8 降到 4 —— 快照换代会让"
                         "worker 缓存整体失效、下一轮全量重发（每份 39 MB），份数减半"
                         "就把这次 IPC 代价近似减半。")
    ap.add_argument("--snapshot-temp", type=float, default=0.0,
                    help="P2 · m2b1：自快照出场权重的温度。0=旧线性新近度加权"
                         "（1..n，池大时老快照几乎不出场）；T>0=softmax(exp(新近度/T))，"
                         "新近度归一化到 [0,1]，T→∞ 均匀、T→0 只打最新。"
                         "池扩到 12 份时配 T=1.0，保证老快照仍占可观出场率。")
    ap.add_argument("--pool-mem-pct", type=float, default=88.0,
                    help="进程池**内存阈值**重建：**全系统**内存使用率 ≥ 该值（%%）即重建。"
                         "0=关闭（只按轮数兜底）。为什么不能只靠轮数：累积速率随配方与"
                         "机器负载变化（实测 ≈9 GB/小时），赌固定轮数迟早 OOM。"
                         "为什么用系统口径而不是训练树占比：非训练占用基线就有 ≈11 GB，"
                         "按树占比判会漏掉它（见 _sys_mem_pct 的注释）。")
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--encoder", choices=["auto", "v1", "v2", "v3"], default="auto",
                    help="状态编码器版本：auto=跟随 --init 检查点（老文件→v1）；"
                         "v3 = P3 的 63 通道 / 130 全局维")
    ap.add_argument("--arch", choices=["auto", "s", "m1"], default="auto",
                    help="骨干架构：auto=跟随 --init 检查点（老文件→s）；"
                         "m1 = SE-ResNet 160×20（≈9.84M，P3）")
    ap.add_argument("--force-cold", action="store_true",
                    help="P4：**强制**冷启动 —— 即使 --init 与目标同 arch，也跳过权重装载，"
                         "只沿用它的编码口径/超参。用于「同架构不同宽度」的蒸馏瘦身"
                         "（10M 教师 → 4M 学生，arch 都是 m1，但 channels 不兼容，"
                         "默认路径会尝试 load_state_dict 并因形状不匹配报错）。"
                         "打开后优化器状态同样不恢复，必须配 --lr-warmup。")
    ap.add_argument("--m1-preset", choices=["m1", "m0", "m1l"], default="m1",
                    help="--arch m1 且发生骨干换代（冷启动）时采用的宽度/深度预设。"
                         "**必须显式取预设**：旧检查点里记的是 s 系列的 64/8/320，"
                         "若照抄会造出一个「名字叫 M1、尺寸其实是 s11」的网络"
                         "（实测参数量会显示 0.82M 而不是 9.84M）。"
                         "m1=160×20 / m0=128×12（4.07M 试点）/ m1l=192×16（11.2M）")
    ap.add_argument("--channels", type=int, default=0, help="显式指定宽度（0=自动）")
    ap.add_argument("--blocks", type=int, default=0, help="显式指定深度（0=自动）")
    ap.add_argument("--hidden", type=int, default=0, help="显式指定 MLP 隐层（0=自动）")
    ap.add_argument("--anchor-warmup", type=int, default=0,
                    help="KL 锚点的预热轮数：前 N 轮不做 KL 锚定，第 N+1 轮把**当时**的"
                         "策略冻结成锚点。冷启动（M1）必须设 >0 —— 否则锚点是一份随机"
                         "初始化的策略，KL 惩罚会把模型往噪声上拽（自适应还会把 "
                         "kl_coef 顶到 3.0）。0 = 旧行为（锚在 --init 上）")
    ap.add_argument("--snapshot-every", type=int, default=4,
                    help="每多少轮把当前策略冻结一份进对手池（快照池只保留最近 8 份）。"
                         "为什么要做成可调：快照池一变，worker 缓存就失效、下一轮要全量"
                         "重发 —— 载荷 ≈ 份数 × 网络权重。s11（3MB/份）无所谓，"
                         "M1（39MB/份）则是 ~350MB 级，必须把间隔拉长（建议 16）来摊薄 IPC。")
    ap.add_argument("--spatial-head", choices=["auto", "on", "off"], default="auto",
                    help="逐牌空间头：零初始化的 1×1 卷积加到 discard/self_kong 的 "
                         "27 个位置 logit 上（只 +2×(C+1) 个参数）。auto=跟随检查点；"
                         "on=强制开启（同版本加载时输出与旧模型逐位相同，是严格热启动）。")
    ap.add_argument("--new-param-init", choices=["random", "zero"], default="random",
                    help="v1→v2 热启动时，新增的 16 个输入通道权重与 28 列 fc_in 权重的"
                         "初始化方式（见 zhuoji/net.py::warm_start_from_v1）。"
                         "random=保留随机初始化（历史行为；实测会让热启动瞬间的 discard "
                         "logits 最大偏离 16.7，等于往训练好的策略里注入随机扰动）；"
                         "zero=整段置零 → 热启动后输出与旧模型**逐位相同**（无损迁移），"
                         "新权重再靠梯度长出来。这是把「迁移扰动」与「新特征本身的作用」"
                         "分成两个变量的唯一开关（P1 阶段 2 的 C1/C3 对照）。")
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
    # 主进程线程数显式设定。
    # 为什么必须在这里设：向量化路径下 ``collect_parallel`` 把采集丢给 worker 子进程
    # （子进程自己 set 2 线程），而 ``update()`` 是**在主进程**跑的梯度计算 ——
    # 在此之前主进程的线程数从未被设过，用的是 torch 的默认值（本机 20），
    # 于是 ``--threads`` 对最耗时的 update() 完全不起作用。
    # 实测（M1 9.84M，batch 512 训练步）：8→2175ms / 16→1401 / 20→1351 / **24→1222** / 28→1375，
    # 所以 24 线程比默认 20 快约 10%。
    torch.set_num_threads(args.threads)

    # ---- denormal 冲刷（2026-10-16 实测加入）--------------------------------
    # 把非规格化浮点（|x| < ~1.2e-38）当作零处理。Mish 的 softplus/log 在极端
    # 激活值下会产生 denormal，而 AVX2 上这类值走**微码慢路径**（单条指令可慢
    # 10-100 倍）。实测（batch=512，threads=24，3 轮交叉验证取中位）：
    #   baseline 265 samples/s → flush 301 samples/s = **1.14x**，三轮全部为正。
    # 数值影响：只把 ~1e-38 量级当 0，对梯度/损失无实质影响（远小于 fp32 分辨率）。
    # 用 --no-flush-denormal 可关闭（A/B 对照用）。
    if not args.no_flush_denormal:
        try:
            torch.set_flush_denormal(True)
            print("[opt] denormal 冲刷已开启（update 段实测 1.14x）", flush=True)
        except Exception as _e:                 # 部分平台/后端不支持
            print(f"[opt] denormal 冲刷不可用（忽略）：{_e}", flush=True)

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
    if args.arch != "auto":
        cfg.arch = args.arch
    if args.channels:
        cfg.channels = args.channels
    if args.blocks:
        cfg.blocks = args.blocks
    if args.hidden:
        cfg.hidden = args.hidden
    src_arch = str((ck.get("net") or {}).get("arch", "s"))
    src_version = encoder_version_from_ckpt(ck)
    src_channels = int((ck.get("net") or {}).get("channels") or 0)
    src_blocks = int((ck.get("net") or {}).get("blocks") or 0)
    # P4：同 arch 但宽度不同（10M m1 → 4M m0）也必须走冷启动。
    # 只看 arch 相等会让 load_state_dict 撞上形状不匹配 —— 而那不是"新增结构"，
    # 是**整网重来**，必须由 --force-cold 显式声明或在下面自动判定。
    same_arch = (src_arch == cfg.arch)
    size_changed = False
    if same_arch and cfg.arch == "m1" and not (args.channels or args.blocks or args.hidden):
        from zhuoji.net_m1 import M1_PRESETS
        _pr = M1_PRESETS[args.m1_preset]
        if src_channels and (src_channels != _pr["channels"]
                             or (src_blocks and src_blocks != _pr["blocks"])):
            size_changed = True
    cold = (not same_arch) or args.force_cold or size_changed
    if size_changed and not args.force_cold:
        print(f"[cold] 同 arch 但尺寸不同 → 自动冷启动"
              f"（源 {src_channels}ch×{src_blocks} → 预设 {args.m1_preset} 目标）")

    # 骨干换代（冷启动）时，宽度/深度**不能**沿用旧检查点：
    # 旧文件里记的是 s 系列的 64/8/320，照抄会得到一个「叫 M1、实际是 s11 尺寸」
    # 的网络（实测参数量 0.82M 而非 9.84M —— 冒烟时当场抓到过）。
    if cold and cfg.arch == "m1" and not (args.channels or args.blocks or args.hidden):
        from zhuoji.net_m1 import M1_PRESETS
        pr = M1_PRESETS[args.m1_preset]
        cfg.channels, cfg.blocks, cfg.hidden = pr["channels"], pr["blocks"], pr["hidden"]
        print(f"[arch] 骨干换代 → 采用预设 {args.m1_preset}："
              f"{cfg.channels}ch × {cfg.blocks} blocks，hidden {cfg.hidden}")

    # ---- P1/P3 特征包掩码（单变量消融） ------------------------------------
    # 解析必须带版本（v3 七包，用 v2 的解析器会把 d/e/f/g 静默丢掉），
    # 且**换代时不能继承旧检查点的掩码** —— 从 s11(v1) 冷启动到 v3 时，
    # 旧文件没有 encoder_packs 字段，auto 会退回 "v2 的三包 abc"，
    # 于是 v3 的全部新特征被悄悄清零（冒烟时实测日志里印的就是 packs=abc）。
    same_gen = (enc_version == src_version) and not cold
    if args.encoder_packs == "auto":
        enc_packs = (encoder_packs_from_ckpt(ck) if same_gen
                     else parse_packs_for(enc_version, None))
    else:
        enc_packs = parse_packs_for(enc_version, args.encoder_packs)

    # M2-AUX：辅助头开关。off（默认）时沿用 ckpt 记录（旧 ckpt → 无辅助头，
    # 与旧版行为逐位一致）；on 时强制挂上 —— 从旧 ckpt 起跑则辅助头全新初始化。
    cfg.aux = 1 if args.aux_heads == "on" else int(getattr(cfg, "aux", 0) or 0)

    model = build_model(cfg)
    if cold:
        # M1 与 s 系列没有可共享的权重（通道数/深度都不同），任何部分复制都是
        # 形状错误，所以这里是**真冷启动**：不装载 --init 的权重。
        _desc = (f"骨干 {src_arch} → {cfg.arch}" if not same_arch
                 else f"同骨干 {cfg.arch} 变宽度 {src_channels}ch → {cfg.channels}ch")
        print(f"[cold] {_desc}：结构不可共享 → **冷启动**"
              f"（跳过 --init 权重装载，仅沿用其训练超参）")
        print(f"[cold] 冷启动务必配 --anchor-warmup >0，否则 KL 锚点是一份随机策略")
    elif src_version != enc_version and args.encoder != "auto":
        info = warm_start_from_v1(model, ck, new_init=args.new_param_init)
        print(f"[warm] {args.init} 编码器 v{src_version} → v{enc_version}："
              f"原样复制 {info['copied']} 个张量；部分复制 {info['partial']}；"
              f"全新参数 {len(info['new_only'])} 个"
              f"（新增特征权重初始化 = {info['new_init']}）")
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
    # B4：同一句话要跨**段**也成立 —— 检查点里带了优化器状态就原样恢复。
    # 冷启动路径（骨干结构不可共享、权重是随机的）**不能**恢复：那份动量属于旧模型，
    # 装到随机权重上只会把噪声当坡度。这种情况必须靠 --lr-warmup 缓冲。
    _opt_ck = ck.get("opt") if isinstance(ck, dict) else None
    if _opt_ck is not None and not cold:
        try:
            opt.load_state_dict(_opt_ck)
            print(f"[opt] 已从 {args.init} 恢复 AdamW 状态"
                  f"（step 组 {len(opt.state)}）——跨段续训无冷启动")
        except Exception as _e:
            # 参数组不匹配（最常见：M2-AUX 给结构追加了辅助头参数）→
            # 位置对齐部分恢复骨干状态，只有新参数冷启动。
            n_ok = _opt_restore_partial(opt, _opt_ck)
            if n_ok > 0:
                print(f"[opt] 整体恢复失败（{_e.__class__.__name__}）→ "
                      f"部分恢复 {n_ok}/{sum(len(g['params']) for g in opt.param_groups)} "
                      f"个参数的 AdamW 状态（新增辅助头冷启动，骨干无冷启动踢）")
            else:
                print(f"[opt] 恢复失败（{_e}）→ 回退冷启动；请配 --lr-warmup > 0")
    elif _opt_ck is None:
        print(f"[opt] {args.init} 不含优化器状态 → Adam 冷启动"
              + ("" if args.lr_warmup > 0 else "；建议 --lr-warmup > 0"))
    kl_init = float(args.kl_coef)   # 锚点预热结束后要恢复到这个值

    print(f"从 {args.init} 初始化 (epoch={ck.get('epoch')}, val_loss={ck.get('val_loss')})")
    print(f"参数量 {model.n_params():,}；骨干 arch={cfg.arch} "
          f"{cfg.channels}ch × {cfg.blocks} blocks，hidden {cfg.hidden}"
          f"（输入 {cfg.in_channels} 通道 / {cfg.global_dim} 全局维）；"
          f"{args.iters} 轮 × {args.games_per_iter} 局/轮")
    print(f"[编码] v{enc_version} spatial_head={'on' if spatial else 'off'}"
          f" arch={cfg.arch}"
          + (f" packs={''.join(enc_packs) or 'none'}"
             f"（{', '.join(pack_desc_for(enc_version)[p] for p in enc_packs) or '全关'}）"
             if enc_version >= 2 else ""))
    print(f"[选点] select={args.select} metric={args.select_metric}"
          f"（{SELECT_METRICS.get(args.select_metric, (+1, '?'))[1]}）"
          f" save-every={args.save_every} eval-every={args.eval_every}"
          f" eval-rounds={args.eval_rounds}")
    print(f"[M2A] A1 熵比例控制 "
          + (f"H*={args.ent_target_ratio}·H₀（基线 {args.ent_baseline_iters} 轮，"
             f"η={args.ent_eta}，死区 ±{args.ent_deadzone}，clamp [0.01,0.5]）"
             if args.ent_control else
             f"**已关闭**（--no-ent-control，ent_coef 恒定 {args.ent_coef}，熵自由下降）")
          + f" | A2 ε {args.explore_eps}→{args.explore_eps_end}/{args.explore_anneal_iters} 轮"
          f" | A3 kl_target={args.kl_target} 滚动锚 每 {args.kl_anchor_rolling} 轮"
          f" | A4 β {args.klt_beta}→{args.klt_beta_end}/{args.klt_anneal_iters} 轮"
          f"（保持 {getattr(args, 'klt_hold_iters', 0)} 轮）"
          f" | A5 master 座位权重 {args.master_seat_prob}，teacher_prob={args.teacher_prob}"
          f" | A6 {args.adv_mode.upper()}(λ={args.gae_lambda}) 优势按 {args.adv_norm} 标准化"
          f" | A7 池重建 树RSS≥{args.pool_rss_gb}GB 或 {args.pool_rebuild_every} 轮"
          f"、快照留 {args.snapshot_keep} 份、温度 {getattr(args, 'snapshot_temp', 0.0)}、KL 锚每 {args.kl_anchor_every} 轮"
          f" | A8 lr {args.lr}→{args.lr_min if args.lr_min > 0 else args.lr}")

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

    # ---- master：A4 的 KLT 教师 + A5 的一等对手 --------------------------------
    # 一个模型两用：既作蒸馏的教师（只前向不反向），也作席位对手。二者都要求它
    # **冻结**，所以这里统一加载一次、共享同一实例。
    # 注意 master 是 v1 编码器（0.77M / 64ch），主模型是 v3（63 通道）——
    # 输入张量不可共用，所以教师必须在 worker 里**按自己的口径重新编码**同一局面
    # （见 zhuoji/vcollect.py::collect_vectorized 的 teacher 分支）。
    def _ck_path(raw: str, flag: str) -> Path:
        p = Path(raw)
        if not p.is_absolute():
            p = ROOT / "models" / (raw if raw.endswith(".pt") else f"{raw}.pt")
        if not p.exists():
            raise SystemExit(f"{flag} 找不到模型：{p}")
        return p

    def _load_frozen(p: Path):
        m, v = build_from_ckpt(torch.load(p, map_location="cpu", weights_only=False))
        for q in m.parameters():
            q.requires_grad_(False)
        m.eval()
        ck = ck_from_model(m, encoder_version=v,
                           encoder_packs=getattr(m, "encoder_packs", None))
        return m, v, ck

    master_model = None
    master_ver = None
    master_ck = None
    if args.master_ckpt:
        mp_path = _ck_path(args.master_ckpt, "--master-ckpt")
        master_model, master_ver, master_ck = _load_frozen(mp_path)
        print(f"[master] {mp_path.name}（参数量 {master_model.n_params():,}，"
              f"编码器 v{master_ver}）"
              f" → KLT β₀={args.klt_beta}→{args.klt_beta_end}"
              f"（保持 {getattr(args, 'klt_hold_iters', 0)} 轮后，退火 {args.klt_anneal_iters} 轮）"
              f"；一等对手座位权重 {args.master_seat_prob}")

    # ---- M2-C5 归因：KLT 教师可与一等对手解耦 -------------------------------
    # 旧行为（也是 m2c2–m2c5 的配方）：教师 = 座位对手 = --master-ckpt。
    # 于是"换 master"同时改了两件事，+0.386 无法归因。给了 --klt-teacher 时：
    #   * KLT 教师（只前向、算 log π）→ klt_model
    #   * 一等对手座位（0.40 权重）  → 仍是 master_model
    # 两者都冻结；worker 侧各有一份独立缓存（见 vcollect._TEACH_CACHE/_MASTER_CACHE）。
    klt_model, klt_ver, klt_ck = master_model, master_ver, master_ck
    klt_is_master = True
    if args.klt_teacher:
        kp = _ck_path(args.klt_teacher, "--klt-teacher")
        klt_model, klt_ver, klt_ck = _load_frozen(kp)
        klt_is_master = False
        print(f"[klt-teacher] {kp.name}（参数量 {klt_model.n_params():,}，"
              f"编码器 v{klt_ver}）→ **仅**作 KL 蒸馏教师；"
              f"座位对手仍是 {getattr(master_model, 'name', '') or args.master_ckpt or '（无）'}")
    if args.klt_beta > 0 and klt_model is None:
        raise SystemExit("--klt-beta 需要 --klt-teacher 或 --master-ckpt 之一")
    if args.master_seat_prob > 0 and master_model is None:
        raise SystemExit("--master-seat-prob 需要 --master-ckpt")
    if args.klt_beta > 0 and not args.vectorized:
        raise SystemExit("KLT 蒸馏需要 --vectorized（教师前向在 worker 里完成）")

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
    # 续训：把上一段的逐轮记录读进来，再往后追加 —— 覆盖式写盘会让段末判读
    # 只看到后半截（A7 查预热期 kc=0、A6 取 iter 17–24 都会因此失准）。
    if args.history_append and Path(args.history).exists():
        try:
            _prev = json.loads(Path(args.history).read_text(encoding="utf-8"))
            if isinstance(_prev, list):
                history = _prev
                print(f"[resume] 已载入既有 history {len(history)} 轮（将在其后追加）",
                      flush=True)
        except Exception as _e:                              # noqa: BLE001
            print(f"[resume] ⚠️ 既有 history 读取失败（{_e}），按空历史继续", flush=True)
    best_score = -1e9
    eval_log: list[dict] = []   # [{"iter":i,"score":s,"se":e,"eval":{...}}, ...]
    snapshots = []
    best_ckpt = None
    watchdog = [0, None]   # [连续异常次数, 上次恢复的存档 iter]
    _pool = [None]
    snap_epoch = 0          # 快照代数：快照池每变动一次 +1，作为 worker 缓存键（见 vcollect.collect_parallel）
    pool_reset_flag = [False]   # 进程池被重建过 → worker 缓存已丢，需全量重发快照
    if args.vectorized and args.workers > 1:
        import multiprocessing as _mp
        _pool[0] = _mp.get_context("spawn").Pool(processes=args.workers)

    t0 = time.time()

    # ---- M2-A 的控制器 / 调度器（A1 / A2 / A8） ------------------------------
    coef_ctl = CoefController(
        args, log=lambda m: print(m, flush=True)) if not args.no_adaptive else None
    if coef_ctl is not None:
        # 就绪横幅：没有它的话，前 n_base 轮（熵基线窗口）**一条 [coef] 都不会打**
        # （因为两侧系数都还没动），于是"控制器到底有没有被调用"要等到第 25 轮才看得出来。
        # 长跑里这个信息来得太晚，所以显式打一行。
        print(f"[coef] 控制器就绪  ec₀={args.ent_coef:.3f}（"
              + (f"H*={args.ent_target_ratio}×H₀，"
                 f"η={coef_ctl.eta}，死区 ±{coef_ctl.dead}，基线 {coef_ctl.n_base} 轮"
                 if args.ent_control else "**熵侧已关闭**，ec 恒定不动")
              + f"）  "
              f"kc₀={args.kl_coef:.3f}（危险线 {args.kl_target}/轮，"
              f"放松 <{args.kl_target * coef_ctl.relax_ratio:.4f}/轮，clamp "
              f"[{CoefController.KL_LO}, {CoefController.KL_HI}]）  "
              f"→ KL 判据用**每轮率**（kl ÷ 锚点年龄）", flush=True)
    else:
        print("[coef] ⚠️ --no-adaptive：系数全程恒定（对照臂用）", flush=True)
    lr0 = float(args.lr)
    lr_mult = [1.0]         # 看门狗回退时乘性下调，不破坏余弦调度的形状
    last_kl = 0.0           # KL 锚降频后，history 里写"最近一次有效测量"
    # 锚点年龄的**单一事实源**：anchor_it = 第几轮结束后策略被冻成锚点。
    # 建锚（预热结束）/ 滚动 EMA 刷新 / 看门狗回退 三处都要更新它。
    # KL 是**距离**不是速率 —— 判断"漂移是否异常"必须除以年龄（见 CoefController 的
    # ★① 说明）。main 算好之后同时喂给控制器、看门狗和 history。
    anchor_it = -1
    last_kl_rate = 0.0

    def _cur_lr(step: int) -> float:
        """A8：余弦退火 lr0 → lr_min（lr_min=0 时保持恒定，即历史行为）。

        B4：``--lr-warmup N`` 时前 N 轮在 ``lr0/N → lr0`` 之间线性升温。
        冷启动的 Adam 首步更新 ≈ ``lr·sign(g)``（与梯度大小无关），
        所以开头几步用 ``lr0/N`` 这样的小步长先把二阶矩喂起来。
        """
        lo = float(getattr(args, "lr_min", 0.0) or 0.0)
        if lo > 0 and args.iters > 1:
            frac = step / float(args.iters - 1)
            base = lo + 0.5 * (lr0 - lo) * (1.0 + math.cos(math.pi * frac))
        else:
            base = lr0
        w = int(getattr(args, "lr_warmup", 0) or 0)
        if w > 0 and step < w:
            base *= (step + 1) / float(w)
        return base * lr_mult[0]

    def _cur_eps(step: int) -> float:
        """A2：ε 从 --explore-eps 线性退火到 --explore-eps-end。"""
        e0 = float(getattr(args, "explore_eps", 0.0) or 0.0)
        if e0 <= 0:
            return 0.0
        e1 = float(getattr(args, "explore_eps_end", 0.01) or 0.0)
        n = int(getattr(args, "explore_anneal_iters", 300) or 0)
        frac = 1.0 if n <= 0 else min(1.0, step / float(n))
        return e0 + (e1 - e0) * frac

    for it in range(args.iters):
        # A8：把调度后的学习率写进优化器（history / 日志 / 检查点都读 args.lr）
        args.lr = _cur_lr(it)
        for g in opt.param_groups:
            g["lr"] = args.lr
        # A4：β 线性退火到 0。退火到 0 是硬要求（Kickstarting 的 self-regulation）——
        # 否则策略上限被永久钉在教师上。β 归零后连 worker 侧的教师前向也一并省掉。
        klt_beta_now = 0.0
        klt_on = False
        if klt_model is not None and args.klt_beta > 0:
            n_an = int(args.klt_anneal_iters or 0)
            # M2-C2：β 可以退火到**非零**下限（--klt-beta-end）。
            # m2c1 的教训：β 撤到 0 之后 KLT 从 0.39 单调反弹到 0.73，学生从 it120
            # 起就不再像教师，后 480 轮等于裸跑自博弈（10M 自博弈在 M2 已证零增益）。
            # 留一条弱蒸馏（0.2）当「不忘记教师」的常驻约束，比彻底撤掉安全。
            _b_end = float(getattr(args, "klt_beta_end", 0.0) or 0.0)
            # M2-G1：可先保持 n_hold 轮再退火（默认 0 = 从第 0 轮就退，旧行为）。
            # ``it`` 是**本段内** 0-based 轮次，与 --iter-offset 无关，所以续训段
            # 的「保 300 再退 300」按本段长度计，不会被历史轮次污染。
            _n_hold = max(0, int(getattr(args, "klt_hold_iters", 0) or 0))
            _t = max(0, it - _n_hold)
            klt_beta_now = (args.klt_beta if n_an <= 0
                            else args.klt_beta
                            + (_b_end - args.klt_beta) * min(1.0, _t / float(n_an)))
            klt_on = klt_beta_now > 0
        eps_now = _cur_eps(it)
        if args.vectorized:
            from zhuoji.vcollect import collect_parallel
            import multiprocessing as _mp

            def _fresh_pool():
                return _mp.get_context("spawn").Pool(processes=args.workers)

            # 进程池重建 —— Windows 长驻 spawn worker 会持续累积内存（2026-10-12 实测
            # ≈6–9 GB/小时，约 2 小时就顶到系统 94%），**重建是唯一能把它打回起点的机制**。
            # M2-A · A7 把主判据换成**训练进程树 RSS**：worker 的累积内存才是重建要
            # 解决的问题；系统级占用会把浏览器/编辑器的基线（本机 ≈11 GB）算进来，
            # 于是"系统 75%"可能对应树只有 ~13 GB（还没到该重建的程度）。轮数兜底
            # 与系统百分比都保留为兜底。
            if _pool[0] is not None and it > 0:
                why = ""
                if args.pool_rebuild_every > 0 and it % args.pool_rebuild_every == 0:
                    why = f"轮数兜底（每 {args.pool_rebuild_every} 轮）"
                else:
                    tree = _tree_rss_gb(os.getpid())
                    if args.pool_rss_gb > 0 and tree >= args.pool_rss_gb:
                        why = (f"训练进程树 RSS {tree:.1f} GB ≥ "
                               f"{args.pool_rss_gb:.1f} GB")
                    elif args.pool_mem_pct > 0:
                        sys_pct = _sys_mem_pct()
                        if sys_pct >= args.pool_mem_pct:
                            why = (f"系统内存 {sys_pct:.0f}% ≥ {args.pool_mem_pct:.0f}%"
                                   f"（训练进程树 {tree:.2f} GB）")
                if why:
                    _pool[0].close()
                    _pool[0].join()
                    _pool[0] = _fresh_pool()
                    pool_reset_flag[0] = True   # 新 worker 没有快照缓存，下一轮强制全量发
                    print(f"[pool] iter {it} 重建进程池：{why}", flush=True)
            snap_sd = pool_extra_ck + [
                ck_from_model(m, encoder_version=enc_version,
                              encoder_packs=(enc_packs if enc_version >= 2 else None))
                for m in snapshots]
            # 快照代数：进程池重建后 worker 缓存已丢，递增一次强制全量重发。
            # --no-snap-cache 传 None = 恢复旧的「每轮全量重发」，用于 A/B 对照。
            # 传给 collect_parallel 后，worker 只在代数变化那轮收到 ~37MB 快照，
            # 之后各轮只发 key —— 省掉 ~450MB/轮 的 IPC（实测占每轮 10.4s 中的 ~7s）。
            if pool_reset_flag[0]:
                snap_epoch += 1
                pool_reset_flag[0] = False
            traj = None
            aux_out = None               # M2-AUX：本轮回传的完美信息标签
            for attempt in range(3):
                try:
                    traj_arr, rewards, dt, aux_out = collect_parallel(
                        model.state_dict(), nc, args.games_per_iter,
                        args.seed + it * 99991, args.temperature, args.reward_scale,
                        args.teacher_prob, workers=args.workers or 1,
                        past_prob=args.past_prob, snapshots_sd=snap_sd, rules=args.rules,
                        want_traj=True, pool=_pool[0], encoder_version=enc_version,
                        encoder_packs=enc_packs,
                        snap_key=None if args.no_snap_cache else snap_epoch,
                        explore_eps=eps_now,
                        snap_temp=float(getattr(args, "snapshot_temp", 0.0)),
                        # 教师只在前向需求真实存在时才随任务下发 ——
                        # 退火完成后（β→0）省掉 worker 侧那一次重编码 + 前向。
                        teacher_ck=klt_ck if klt_on else None,
                        teacher_ver=klt_ver if klt_on else None,
                        # M2-C5：座位对手与教师**分开**下发。klt_is_master=True 时
                        # worker 复用教师实例（旧行为，一次前向两用，省内存）。
                        master_ck=master_ck,
                        master_ver=master_ver,
                        master_shared=klt_is_master,
                        master_prob=args.master_seat_prob,
                        aux=bool(getattr(model, "aux", False)))
                    traj = traj_arr
                    break
                except MemoryError:
                    print(f"[mem] iter {it + 1 + args.iter_offset} 采集 MemoryError，回收进程池后重试"
                          f"（第 {attempt + 1} 次）", flush=True)
                    if _pool[0] is None:
                        raise
                    _pool[0].terminate()
                    _pool[0].join()
                    _pool[0] = _fresh_pool()
                    pool_reset_flag[0] = True
            if traj is None:
                raise SystemExit(f"iter {it + 1 + args.iter_offset} 连续 3 次 MemoryError，终止训练")
        else:
            aux_out = None           # 单进程路径暂不产辅助标签（M2-AUX 走向量化路径）
            traj, rewards, dt = collect_games(
                model, args.games_per_iter, args.seed + it * 99991,
                args.temperature, args.reward_scale, args.teacher_prob, args.threads,
                snapshots=pool_extra + snapshots, past_prob=args.past_prob,
                game_cls=GameCls, config_cls=CfgCls, encoder_version=enc_version,
                encoder_packs=enc_packs, explore_eps=_cur_eps(it),
                teacher=klt_model if klt_on else None,
                teacher_ver=klt_ver if klt_on else None)
        # KL 锚点预热：冷启动（M1）时 ref_model 是**随机初始化**的策略，
        # 直接锚上去等于把模型往噪声拽，而且 kl 大 → 自适应会把 kl_coef 顶上去
        # （放大这个错误）。所以前 N 轮完全关闭 KL 项，第 N 轮结束时把**当时**的
        # 策略冻结成锚点 —— 之后它起到的作用与旧行为一致（防漂移），
        # 但锚的东西是有意义的策略。
        ref_for_update = ref_model if (
            it >= args.anchor_warmup
            and (it - args.anchor_warmup) % max(1, args.kl_anchor_every) == 0) else None
        # M2-AUX 预热（m2a2）：前 N 轮 aux_coef 线性升温。与上面 kl_coef 预热
        # 同款做法（直接改 args 字段，下一轮重算覆盖）。base 先存快照，避免
        # 连乘把系数越改越小。
        aux_wu = int(getattr(args, "aux_warmup", 0) or 0)
        if aux_wu > 0:
            if not hasattr(args, "_aux_coef_base"):
                args._aux_coef_base = args.aux_coef
            args.aux_coef = args._aux_coef_base * min(1.0, it / aux_wu)
        upd = update(model, ref_for_update, opt, traj, rewards, args,
                     teacher_model=klt_model if klt_on else None,
                     klt_beta=klt_beta_now, aux=aux_out)
        kl_meas = upd["kl"]                 # 降频那几轮是 None（没做锚前向）
        if kl_meas is not None:
            last_kl = kl_meas
            last_kl_rate = float(kl_meas) / max(1, it - anchor_it)
        upd["kl"] = last_kl                 # history 里始终写"最近一次有效 KL"
        upd["kl_rate"] = last_kl_rate       # 每轮漂移率（距上次锚点对齐的平均值）
        if coef_ctl is not None:
            coef_ctl.step(it, upd["entropy"], kl_meas, kl_rate=last_kl_rate)
        if it < args.anchor_warmup:
            args.kl_coef = 0.0
            if args.anchor_warmup > 0 and it == args.anchor_warmup - 1:
                ref_model = copy.deepcopy(model).eval()
                for p in ref_model.parameters():
                    p.requires_grad_(False)
                anchor_it = it               # 锚点年龄归零
                if coef_ctl is not None:
                    coef_ctl.note_anchor(it)
                print(f"[anchor] iter {it + 1 + args.iter_offset} 结束 → 冻结当时策略为 KL 锚点"
                      f"（kl_coef 恢复到 {kl_init:.3f}）", flush=True)
        elif it == args.anchor_warmup and args.anchor_warmup > 0:
            args.kl_coef = kl_init

        # A3：滚动 KL 锚 —— 定期把锚点刷新成「旧锚点 × 0.7 + 当前策略 × 0.3」。
        # 固定在 --init 上的锚会随训练"过期"：KL 项从"防漂移"变成"惩罚正常进步"，
        # 于是自适应把 kl_coef 一路压到下限，该项形同虚设。
        if (args.kl_anchor_rolling > 0 and it >= args.anchor_warmup
                and (it + 1) % args.kl_anchor_rolling == 0):
            _ema_anchor(ref_model, model, args.kl_anchor_alpha)
            anchor_it = it                   # 锚点年龄归零
            if coef_ctl is not None:
                coef_ctl.note_anchor(it)
            print(f"[anchor] iter {it + 1 + args.iter_offset} → KL 锚滚动刷新为"
                  f" 当前策略 EMA(α={args.kl_anchor_alpha})", flush=True)

        # 熵坍缩看门狗：熵持续过低说明策略已经塌缩成确定性
        # —— 恢复最佳存档、重置优化器动量、减半学习率；同一存档再塌缩则终止。
        #
        # ★ KL 条件必须用**每轮率**、不能用裸的 kl。旧写法 ``kl > 2.5`` 成立于
        #   ``--kl-anchor-every 1`` 的时代（那时 kl 就是"一轮的漂移"）；A7② 降频之后
        #   同一个测量值会被**连续 4 轮**参与判定，而"2 轮连续"的计数器一碰就满 →
        #   看门狗会在锚点变旧的正常训练里被误触发（回退存档 + lr 减半）。
        #   换算成率之后，2.5 表示"平均每轮漂移 2.5"（自然水平约 0.03，即 80 倍），
        #   这是个**真·灾难线**：它不会误触发，主要是熵那条在兜底。
        if upd["entropy"] < 0.12 or last_kl_rate > 2.5:
            watchdog[0] += 1
            if watchdog[0] >= 2:
                if best_ckpt is None:
                    print("[watchdog] 尚无最佳存档且已坍缩，终止训练", flush=True)
                    break
                if watchdog[1] == best_ckpt.get("iter"):
                    print(f"[watchdog] 最佳存档 iter={best_ckpt['iter']} 也已坍缩，终止训练",
                          flush=True)
                    break
                print(f"[watchdog] 熵={upd['entropy']:.3f} KL={upd['kl']:.2f}"
                      f"（率 {last_kl_rate:.3f}/轮）坍缩 -> "
                      f"恢复 iter={best_ckpt['iter']} 存档，lr 减半", flush=True)
                model.load_state_dict(best_ckpt["state_dict"])
                ref_model = copy.deepcopy(model).eval()
                for p in ref_model.parameters():
                    p.requires_grad_(False)
                if coef_ctl is not None:
                    coef_ctl.note_anchor(it)     # 回退后锚点重对齐，年龄归零
                # 用乘子而不是直接改 args.lr：A8 打开余弦调度后，直接写 args.lr
                # 会在下一轮被调度覆盖回去，看门狗就静默失效了。
                lr_mult[0] *= 0.5
                args.lr = _cur_lr(it)
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
            "iter": it + 1 + args.iter_offset,
            "samples": len(traj),
            "collect_s": dt,
            "elapsed": time.time() - t0,
            # 2026-10-12：这三个系数也写进 history。判读脚本的 A7「KL 锚点预热」要查
            # kl_coef，而它原先只在 stdout 打印 → history.json 路径取不到值，判读直接崩。
            "ent_coef": float(args.ent_coef),
            "kl_coef": float(args.kl_coef),
            "lr": float(args.lr),
            # M2-A：ε 探索率与 β 退火值也进 history —— 判读要能确认
            # "ε 真的在按 3%→1% 退火""β 真的在 300 轮内归零"，
            # 否则"控制器有没有动"这条最基本的健康断言无从查起。
            "explore_eps": float(eps_now),
            "klt_beta": float(klt_beta_now),
            # M2-G1：β 的保持轮数也留痕 —— 判读要能区分「一开场就退火」与「先保后退」
            "klt_hold": int(getattr(args, "klt_hold_iters", 0) or 0),
            **{k: v for k, v in upd.items()},
        }
        # 对手池：每 --snapshot-every 轮把当前策略冻结一份进池，供后续轮次当"历史自己"的对手。
        # KL 锚点 ref_model 保持 BC 初始策略不动，防止策略整体漂移成乱打。
        # 长训练时只保留最近 8 份快照，避免池子无限膨胀拖慢采集/序列化。
        if args.snapshot_every > 0 and (it + 1) % args.snapshot_every == 0:
            snapshots.append(copy.deepcopy(model).eval())
            if len(snapshots) > args.snapshot_keep:
                snapshots = snapshots[-args.snapshot_keep:]
            snap_epoch += 1                 # 快照池变了 → worker 缓存失效，下一轮全量重发

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
                    "iter": it + 1 + args.iter_offset,
                    "select_metric": args.select_metric,
                    "select_score": cur,
                    "eval": rec["eval"],
                }
                if enc_version >= 2:
                    best_ckpt["encoder_packs"] = list(enc_packs)
                if args.save_opt:      # B4：选点存档也带上优化器状态，便于从它无缝续训
                    best_ckpt["opt"] = opt.state_dict()
                Path(args.out).parent.mkdir(parents=True, exist_ok=True)
                torch.save(best_ckpt, args.out)
            # 选点分自身的标准误：vs_anchor 用配对 SE；其余指标沿用其字段的 SE。
            if args.select_metric == "vs_anchor":
                se = float(rec["eval"].get("vs_anchor_se") or 0.0)
            elif args.select_metric == "avg_score":
                se = float(rec["eval"].get("avg_score_se") or 0.0)
            else:
                se = 0.0
            eval_log.append({"iter": it + 1 + args.iter_offset, "score": cur, "se": se,
                             "eval": dict(rec["eval"])})

        # 周期性检查点：训练结束后用 scripts/score_checkpoints.py 在共用牌池上
        # 对它们做对位评分，才是可信的选点依据（训练内评测分噪声太大）。
        if args.save_every and (it + 1) % args.save_every == 0:
            snap = Path(args.out).parent / "_ckpts" / \
                f"{Path(args.out).stem}_it{it + 1 + args.iter_offset:04d}.pt"
            _dump_ckpt(snap, model, nc, args, it + 1 + args.iter_offset, rec.get("eval"),
                       enc_version, enc_packs, opt=opt if args.save_opt else None)
            print(f"  [snap] {snap.name}", flush=True)

        history.append(rec)
        if (it + 1) % 50 == 0:   # 增量落盘，进程崩溃也不丢历史
            Path(args.history).parent.mkdir(parents=True, exist_ok=True)
            Path(args.history).write_text(json.dumps(history, indent=2),
                                          encoding="utf-8")
        # 注意字段顺序：``ec=`` ``kc=`` ``lr=`` 必须保持**相邻且同序** ——
        # scripts/m1_segment_judge.py 的逐轮解析用一条正则在它们之间精确匹配，
        # 往中间插字段会让段末判读整套失效（M2-A 的 β/ε 因此追加在行尾）。
        print(f"iter {it + 1 + args.iter_offset:3d}  samples={rec['samples']:5d}  "
              f"R={rec['reward']:+.3f}  pi={rec['policy_loss']:+.4f}  "
              f"V={rec['value_loss']:.4f}  H={rec['entropy']:.3f}  "
              f"KL={rec['kl']:.4f}  ec={args.ent_coef:.3f}  kc={args.kl_coef:.3f}  "
              f"lr={args.lr:.1e}  collect={dt:.1f}s  total={rec['elapsed']:.0f}s  "
              f"beta={klt_beta_now:.3f}  eps={rec['explore_eps']:.3f}  "
              f"klt={rec['klt']:.4f}  klr={last_kl_rate:.4f}  "
              f"aux={rec.get('aux', 0.0):.4f}  auxc={args.aux_coef:.4f}",
              flush=True)

    Path(args.history).parent.mkdir(parents=True, exist_ok=True)
    Path(args.history).write_text(json.dumps(history, indent=2), encoding="utf-8")

    # ---- 最终选点 -----------------------------------------------------------
    which, select_note = choose_final(eval_log, args.select, args.select_metric)
    if which == "last" and history:
        _dump_ckpt(args.out, model, nc, args, len(history),
                   history[-1].get("eval"), enc_version, enc_packs,
                   opt=opt if args.save_opt else None)
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
