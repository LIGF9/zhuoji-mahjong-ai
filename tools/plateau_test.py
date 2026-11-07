#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""平台期漂移检验（只读 / 零 CPU / 不启对局）。

目的：把「最近几轮闸门 A 环比不显著」升级为**可判定的漂移检验**，
并顺带给出「跨轮 pool 后 SE 到底是多少」——这决定 P12 的 PPO 判读窗口该取几轮。

模型（冻结池、逐 seed 配对）：
  记 M[r][s] = 第 r 轮在 seed s 上的 (cand − 其余三家均值)，四座已平均。
  若各轮策略真值相同、只差一份 iid 扰动 δ：
      M[r] = c + δ_r,  δ iid, Var=σ²
      D_k = M[k+1] − M[k] = δ_{k+1} − δ_k
  ⇒ Var(D)=2σ²，Corr(D_k, D_{k+1}) = −σ²/2σ² = **−0.50**
  若有系统性漂移 μ（模型真的每轮更强），相邻差分共变 ⇒ lag-1 相关被推向 0 甚至为正。
  ⇒ **lag-1 相关 ≈ −0.50 = 「固定中心 + 纯扰动」= 平台期**；显著 > −0.35 提示真漂移。

pool 的 SE：相邻段共用中间轮的扰动，pool 相邻 3 段时噪声部分相消，
  故 cluster-robust SE（按 seed 聚簇）**小于** naive iid SE（实测膨胀 ×0.57~0.59）。

判据口径与 tools/gate_decision.py / final_eval_report.py 一致：
  v = deltas[cand_seat] − mean(其余三家)；四座按 seed 平均。禁止 deltas[0]。
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GATES = os.path.join(ROOT, "runs", "gates")


def load_gate_a(path: str) -> dict[int, float]:
    """seed -> 四座平均的 (cand − 其余三家)。"""
    per_seed: dict[int, list[float]] = {}
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            ds = rec.get("deltas")
            if not ds or len(ds) != 4:
                continue
            seat = rec["cand_seat"]
            v = ds[seat] - float(np.mean([ds[i] for i in range(4) if i != seat]))
            per_seed.setdefault(rec["seed"], []).append(v)
    return {s: float(np.mean(v)) for s, v in per_seed.items()}


def collect(min_seeds: int = 300) -> dict[int, dict[int, float]]:
    out: dict[int, dict[int, float]] = {}
    for path in sorted(glob.glob(os.path.join(GATES, "r*_gateA.jsonl"))):
        m = re.search(r"r(\d+)_gateA\.jsonl$", path)
        if not m:
            continue
        d = load_gate_a(path)
        if len(d) >= min_seeds:  # 只收完整文件，半截不参与
            out[int(m.group(1))] = d
    return out


def pool_stats(seg: np.ndarray, n_seeds: int) -> tuple[float, float, float, float]:
    """seg: (k 段, n_seeds)。返回 mean, naive_se, cluster_se, inflate。"""
    mu = float(seg.mean())
    naive = float(seg.std(ddof=1) / np.sqrt(seg.size)) if seg.size > 1 else float("nan")
    cl = seg.mean(axis=0)
    cluster = float(cl.std(ddof=1) / np.sqrt(n_seeds)) if n_seeds > 1 else float("nan")
    inflate = cluster / naive if naive and naive == naive and naive > 0 else float("nan")
    return mu, naive, cluster, inflate


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--window", type=int, default=3, help="pool 段数（默认 3）")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    rounds = collect()
    if len(rounds) < 3:
        print(f"[FAIL] 完整闸门 A 文件不足（{len(rounds)} 个）")
        return 1
    ks = sorted(rounds)
    common = set(rounds[ks[0]])
    for k in ks:
        common &= set(rounds[k])
    seeds = sorted(common)
    if len(seeds) < 300:
        print(f"[FAIL] 共同 seed 仅 {len(seeds)}")
        return 1

    M = np.array([[rounds[k][s] for s in seeds] for k in ks])  # (轮, seed)
    D = np.diff(M, axis=0)  # (轮-1, seed) 相邻段差分
    n = len(M)

    # lag-1 相关：相邻两段差分（理论 −0.50 = 无漂移）
    lag1 = [float(np.corrcoef(D[i], D[i + 1])[0, 1]) for i in range(len(D) - 1)]
    nonadj = [
        float(np.corrcoef(D[i], D[j])[0, 1])
        for i in range(len(D))
        for j in range(i + 2, len(D))
    ]
    lag1_mean = float(np.mean(lag1))

    seg_mean = D.mean(axis=1)
    seg_se = np.array([D[i].std(ddof=1) / np.sqrt(len(seeds)) for i in range(len(D))])
    # 零漂移零假设下：段均值 ~ N(0, σ_se)，lag1 = −0.5
    drift_verdict = "平台（无系统性漂移）" if lag1_mean < -0.35 else "存在漂移迹象"
    if len(nonadj):
        nonadj_mean = float(np.mean(nonadj))
    else:
        nonadj_mean = float("nan")

    w = max(2, min(args.window, len(D)))
    seg = D[len(D) - w:]
    mu, naive, cluster, inflate = pool_stats(seg, len(seeds))

    payload = {
        "rounds": [f"R{k:02d}" for k in ks],
        "n_seeds": len(seeds),
        "round_means": {f"R{k:02d}": round(float(M[i].mean()), 3) for i, k in enumerate(ks)},
        "seg_means": {
            f"R{ks[i+1]}-R{ks[i]}": [round(float(seg_mean[i]), 3), round(float(seg_se[i]), 3)]
            for i in range(len(D))
        },
        "lag1_corr": [round(x, 3) for x in lag1],
        "lag1_mean": round(lag1_mean, 3),
        "nonadj_mean": round(nonadj_mean, 3),
        "pool_window": w,
        "pool_mean": round(mu, 3),
        "pool_naive_se": round(naive, 3),
        "pool_cluster_se": round(cluster, 3),
        "pool_inflate": round(inflate, 3),
        "per_seed_sd": round(float(M.std(axis=1, ddof=1).mean()), 2),
        "drift_verdict": drift_verdict,
    }
    if args.json:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 0

    print("=" * 92)
    print(f"平台期漂移检验 ｜ 轮次 {ks[0]}~{ks[-1]}（{n} 轮）｜ 共同 seed {len(seeds)}（冻结池，逐 seed 配对）")
    print("=" * 92)
    print("逐轮均值:", {f"R{k}": round(float(M[i].mean()), 2) for i, k in enumerate(ks)})
    print(f"每 seed 单轮 sd ≈ {payload['per_seed_sd']} 分  ⇒ 单轮 300 seed 的均值 SE ≈ {payload['per_seed_sd']/np.sqrt(len(seeds)):.2f}")
    print()
    print("逐段环比（mean±SE）:")
    for i in range(len(D)):
        print(f"  R{ks[i+1]:02d}−R{ks[i]:02d}: {seg_mean[i]:+6.3f} ± {seg_se[i]:.3f}")
    print()
    print("【漂移检验】相邻差分 lag-1 相关（理论 −0.50 = 固定中心纯扰动）:")
    for i, r in enumerate(lag1):
        print(f"  D(R{ks[i+2]}-R{ks[i+1]}) vs D(R{ks[i+1]}-R{ks[i]}): r={r:+.3f}")
    print(f"  均值 lag-1 = {lag1_mean:+.3f} ｜ 非相邻均值 = {nonadj_mean:+.3f}（≈0 表示各段独立）")
    print(f"  ⇒ 裁决：{drift_verdict}")
    print()
    print(f"【pool {w} 段】{'-'*60}")
    print(f"  mean = {mu:+.3f}   naive SE = {naive:.3f}(n={seg.size})   cluster SE = {cluster:.3f}(n={len(seeds)} seeds)")
    print(f"  膨胀因子 ×{inflate:.2f}（<1 表示相邻段噪声相消，pool 比 iid 假设更准）")
    print(f"  可分辨下限（2σ）= ±{2*cluster:.3f} 分")
    print()
    print("[禁止] 用 deltas[0] 统计 / 自对弈胜率决策 / val loss 跨轮判强度 / 半截 jsonl 下结论")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
