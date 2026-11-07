#!/usr/bin/env python3
"""PPO 分支决策器（只读 / 零 CPU / 不启对局）。

为什么需要它：PPO 是否值得继续占预算，唯一判据是「同池逐 seed 配对」的
P{r} − R{r}（P{r}=PPO 轮学生闸门 A，R{r}=同期 DAgger 轮闸门 A，两者同池
seed0=710000+300、同口径 deltas[cand_seat]−mean(其余三家) 四座平均）。

单轮 n=300 的 SE≈0.42~0.54 分 > 单轮真实增量（0.1~0.5 分）⇒ 单轮永远判不出。
必须累积多轮。但相邻段**共用中间轮**（P11−R11 与 P12−R12 都含 R11/P11 的扰动），
naive 合并 SE=√ΣSE² 会高估独立信息量 ⇒ 本工具同时报：

  naive SE  = √(Σ SE_k²)                       （假设各段独立，偏乐观）
  cluster SE = 逐 seed 先把 Σ_k(P_k−R_k) 求和再算 SE
               （同池同 seed ⇒ 自动吸收跨段相关，是决策该用的那个）

预登记规则（docs/14天训练计划.md §6.33.4，2026-11-07 14:34 先锁死、禁止事后改）：

  主判据 Σ = Σ_{r=10..12} (P_r − R_r)
  辅判据 D池化 = mean(P11−P10, P12−P11)（PPO 链内推进，仅支持不得覆盖）

  H1   Σ ≥ +0.30 且 p<0.10   或  D池化 ≥ +0.25      → 开加速模式
  H0   |Σ| < 2σ（无差）且 P 最新闸门 A 未退化        → 同样开加速（DAgger 期望已 0）
  H−1  Σ ≤ −0.30 且 p<0.10   或  P 最新 A 掉出 2σ     → 不加速、PPO_EVERY=2

用法::

    .venv/bin/python tools/ppo_decide.py                  # 默认 r=10,11,12
    .venv/bin/python tools/ppo_decide.py --rounds 10 11   # 少一轮时先看趋势
    .venv/bin/python tools/ppo_decide.py --md
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))

from final_eval_report import fmt, per_seed, stats  # noqa: E402  唯一实现，禁止另写口径

GATES = ROOT / "runs/gates"

# 预登记阈值（§6.33.4）——改这里等于改预登记，需同步写进计划文档
SUM_STRONG = 0.30
D_STRONG = 0.25
P_EFFECT = 0.10
DEGRADE_SIGMA = 2.0


def load(path: Path) -> dict[int, float] | None:
    """完整才算数：缺失 / 半截 / 解析失败 ⇒ None（绝不用半截数据下结论）。"""
    if not path.exists():
        return None
    try:
        d = per_seed(path)
    except (ValueError, OSError):
        return None
    return d if len(d) >= 100 else None


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="PPO 分支决策（只读，冻结口径）")
    ap.add_argument("--rounds", nargs="+", type=int, default=[10, 11, 12],
                    help="参与 Σ 的轮次（默认 10 11 12）")
    ap.add_argument("--md", action="store_true")
    a = ap.parse_args(argv)

    print("=" * 92)
    print(f"PPO 决策判据：Σ_{{{a.rounds[0]}..{a.rounds[-1]}}}(P−R) + 链内 D 池化"
          f"（预登记 §6.33.4）")
    print("=" * 92)

    seg: dict[int, dict[int, float]] = {}   # r -> {seed: P_r - R_r}
    pA: dict[int, dict[int, float]] = {}
    for r in a.rounds:
        p = load(GATES / f"p{r:02d}_gateA.jsonl")
        d = load(GATES / f"r{r:02d}_gateA.jsonl")
        if p is None or d is None:
            why = "P 缺" if p is None else "R 缺"
            print(f"  P{r:02d}−R{r:02d}: [跳过] {why}（该段不完整，不参与合并）")
            continue
        pA[r] = p
        common = sorted(set(p) & set(d))
        if len(common) < 100:
            print(f"  P{r:02d}−R{r:02d}: [跳过] 共同 seed 仅 {len(common)}")
            continue
        s = {k: p[k] - d[k] for k in common}
        seg[r] = s
        st = stats(list(s.values()))
        print("  " + fmt(f"P{r:02d}−R{r:02d}", st) + f"   共同 seed={len(common)}")

    if not seg:
        print("\n[结论] 无可用段（P12 尚未落盘）。本轮不定分支，保持主循环零干预。")
        return 0

    # ---- 主判据 Σ：逐 seed 求和（同池 ⇒ 吸收跨段相关）----
    rounds = sorted(seg)
    common = sorted(set.intersection(*(set(seg[r]) for r in rounds)))
    sums = {k: sum(seg[r][k] for r in rounds) for k in common}
    cl = stats(list(sums.values()))
    naive_se = math.sqrt(sum(stats(list(seg[r].values()))["se"] ** 2 for r in rounds))
    print("-" * 92)
    print(f"  Σ ({'+'.join(f'P{r:02d}−R{r:02d}' for r in rounds)})"
          f" = {cl['mean']:+.3f}   cluster SE={cl['se']:.3f}（naive √ΣSE²={naive_se:.3f}）"
          f"  z={cl['z']:+.2f}  p≈{cl['p']:.4f}  符号 {cl['pos']}:{cl['neg']}"
          f"  共同 seed={cl['n']}")
    print(f"  各段均值：{'  '.join(f'{r}:{stats(list(seg[r].values()))["mean"]:+.3f}' for r in rounds)}")

    # ---- 辅判据 D池化：PPO 链内相邻代（只支持，不覆盖）----
    dvals: list[float] = []
    for r0, r1 in zip(rounds, rounds[1:]):
        if r0 not in pA or r1 not in pA:
            continue
        cm = sorted(set(pA[r0]) & set(pA[r1]))
        if len(cm) < 100:
            continue
        s = stats([pA[r1][k] - pA[r0][k] for k in cm])
        dvals.append(s["mean"])
        print("  " + fmt(f"P{r1:02d}−P{r0:02d}(链内)", s))
    # 预登记写的是 mean(P11−P10, P12−P11) —— 必须 ≥2 段才算「池化」；
    # 只有 1 段时它就是单段值，不能当池化证据（否则单段 +0.25 就误触发 H1）。
    pooled_d = sum(dvals) / len(dvals) if len(dvals) >= 2 else None
    if pooled_d is not None:
        print(f"  D池化 = {pooled_d:+.3f}（{len(dvals)} 段，仅作支持证据）")
    elif dvals:
        print(f"  D池化 = 暂缺（仅 {len(dvals)} 段：{dvals[0]:+.3f}；"
              f"预登记要求 ≥2 段，单段不作证据）")

    # ---- P 最新闸门 A 是否退化（相对上一 P）----
    lsorted = sorted(pA)
    degrade = False
    ptrend = ""
    if len(lsorted) >= 2:
        r0, r1 = lsorted[-2], lsorted[-1]
        cm = sorted(set(pA[r0]) & set(pA[r1]))
        if len(cm) >= 100:
            s = stats([pA[r1][k] - pA[r0][k] for k in cm])
            degrade = s["mean"] < -DEGRADE_SIGMA * s["se"]
            ptrend = f"P{r1:02d}−P{r0:02d} = {s['mean']:+.3f}±{s['se']:.3f} p≈{s['p']:.4f}"
            print(f"  P 最新代对比：{ptrend}{'  ⇒ 退化 >2σ' if degrade else ''}")

    # ---- 预登记三分支 ----
    sum_v, sum_se, sum_p = cl["mean"], cl["se"], cl["p"]
    h1 = (sum_v >= SUM_STRONG and sum_p < P_EFFECT) or (pooled_d is not None and pooled_d >= D_STRONG)
    hm1 = (sum_v <= -SUM_STRONG and sum_p < P_EFFECT) or degrade

    print("-" * 92)
    if hm1:
        verdict, act = "H−1 PPO 无净收益（或退化）", "不加速；PPO_EVERY=2，累积到 P16 再判"
    elif h1:
        verdict, act = "H1 PPO 有净收益", "开加速模式（RAPID_FROM=下一轮，MAX_ROUND 60→400）"
    else:
        verdict, act = "H0 无差别（|Σ| 在 2σ 内）", "同样开加速模式（DAgger 期望已 ≈0，PPO 是唯一非零方差项）"
    print(f"  [结论] {verdict}  →  {act}")
    print(f"  依据：Σ={sum_v:+.3f}±{sum_se:.3f} p={sum_p:.4f}"
          f"（阈值 ±{SUM_STRONG} / p<{P_EFFECT}）"
          + (f"；D池化={pooled_d:+.3f}（阈值 +{D_STRONG}）" if pooled_d is not None else "")
          + ("；P 退化>2σ" if degrade else ""))
    print("  [护栏] TAU / Q-MSE / n_det / lr / 闸门 D 判据一字不改；"
          "web_best 仍待终评一次性刷新；切换只在阶段边界手工执行（禁日志文本触发器）。")

    if a.md:
        print("\n```markdown")
        print("| 口径 | n | diff ± SE | z | p | 符号 |")
        print("|---|---|---|---|---|---|")
        for r in rounds:
            s = stats(list(seg[r].values()))
            print(f"| P{r:02d}−R{r:02d} | {s['n']} | {s['mean']:+.3f}±{s['se']:.3f} | "
                  f"{s['z']:+.2f} | {s['p']:.4f} | {s['pos']}:{s['neg']} |")
        print(f"| **Σ（cluster）** | {cl['n']} | **{sum_v:+.3f}±{sum_se:.3f}** | "
              f"{cl['z']:+.2f} | {sum_p:.4f} | {cl['pos']}:{cl['neg']} |")
        print(f"\n结论：**{verdict}** → {act}")
        print("```")
    return 0


if __name__ == "__main__":
    sys.exit(main())
