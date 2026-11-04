"""配对赛显著性分析：重复牌墙两场（候选坐 0 / 坐 2）合成配对样本。

指标
----
* 逐 seed 配对差：diff_s = [(cand-base)_seat0 + (cand-base)_seat2] / 2
  其中 (cand-base)_seat = deltas[cand] - mean(deltas[others])。
* 检验：配对 t 检验（正态近似）+ 符号检验 + bootstrap 95% CI。
* 附加：候选胡牌率 / 被胡率 / 流局率对照（观感指标）。

用法： .venv/bin/python analyze.py runs/v1_n128.jsonl [--min-seed-pairs 30]
"""
from __future__ import annotations

import glob
import json
import math
import random
import sys

import numpy as np


def load(paths):
    recs = {}
    for pat in paths:
        for f in glob.glob(pat):
            for line in open(f, encoding="utf-8"):
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                except Exception:
                    continue
                if not r.get("deltas"):
                    continue
                recs[(r["seed"], r["cand_seat"])] = r
    return recs


def pair_diffs(recs):
    seeds = sorted({s for (s, _c) in recs})
    diffs = []
    for s in seeds:
        r0 = recs.get((s, 0))
        r2 = recs.get((s, 2))
        if not r0 or not r2:
            continue
        def margin(r):
            d = r["deltas"]
            c = r["cand_seat"]
            others = [d[i] for i in range(4) if i != c]
            return d[c] - sum(others) / 3.0
        diffs.append(0.5 * (margin(r0) + margin(r2)))
    return seeds, np.array(diffs, dtype=np.float64)


def t_test_1sample(x):
    n = len(x)
    m = x.mean()
    sd = x.std(ddof=1)
    if sd == 0 or n < 2:
        return m, sd, float("nan"), float("nan")
    t = m / (sd / math.sqrt(n))
    # 正态近似（n>100 时误差可忽略）；双侧 p
    from math import erfc, sqrt
    p = erfc(abs(t) / sqrt(2))
    return m, sd, t, p


def sign_test(x):
    n_pos = int((x > 0).sum())
    n_neg = int((x < 0).sum())
    n = n_pos + n_neg
    if n == 0:
        return n_pos, n_neg, float("nan")
    from math import comb
    # 双侧二项检验 p=0.5
    k = max(n_pos, n_neg)
    p = sum(comb(n, i) for i in range(k, n + 1)) / 2 ** n * 2
    return n_pos, n_neg, min(1.0, p)


def boot_ci(x, iters=20000, seed=7):
    rng = np.random.default_rng(seed)
    n = len(x)
    idx = rng.integers(0, n, size=(iters, n))
    means = x[idx].mean(axis=1)
    return np.percentile(means, 2.5), np.percentile(means, 97.5)


def rates(recs):
    n = len(recs)
    cand_win = sum(1 for r in recs.values()
                   if r.get("type") == "win" and r.get("winner") == r["cand_seat"])
    cand_deal_in = 0
    for r in recs.values():
        if r.get("type") == "win" and r.get("winner") != r["cand_seat"]:
            d = r["deltas"]
            if d[r["cand_seat"]] < 0:
                cand_deal_in += 1
    draw = sum(1 for r in recs.values() if r.get("type") != "win")
    return cand_win / n, cand_deal_in / n, draw / n, n


def main():
    paths = sys.argv[1:] or ["runs/v1_n128.jsonl"]
    recs = load(paths)
    seeds, x = pair_diffs(recs)
    print(f"完成 seed 配对数: {len(x)}  (原始场次 {len(recs)})")
    if len(x) < 5:
        print("样本不足")
        return
    m, sd, t, p = t_test_1sample(x)
    lo, hi = boot_ci(x)
    npos, nneg, sp = sign_test(x)
    se = sd / math.sqrt(len(x))
    print(f"配对差均值 (候选-基线均值): {m:+.3f} 分/场  SD={sd:.2f}  SE={se:.3f}")
    print(f"配对 t = {t:.3f}  双侧 p ≈ {p:.4f}")
    print(f"bootstrap 95% CI: [{lo:+.3f}, {hi:+.3f}]")
    print(f"符号检验: +{npos} / -{nneg}  p ≈ {sp:.4f}")
    w, dl, dr, n = rates(recs)
    print(f"候选胡牌率 {w:.3f} | 点炮率 {dl:.3f} | 流局率 {dr:.3f}  (n={n})")
    # 等效换算：对三 N32 场均净得分
    print(f"折算整桌分差: {m:+.3f} x4/3 ≈ {m*4/3:+.3f} 分/场（对桌均值口径）")


if __name__ == "__main__":
    main()
