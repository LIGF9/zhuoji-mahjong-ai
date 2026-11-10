#!/usr/bin/env python3
"""多臂闸门汇总：同池、同座位、同基线下的差距矩阵（含配对差）。

为什么需要它
------------
单个 `eval_pairs` 的 summary 只给「候选 vs 基线」一个数。想回答「学生离教师
还差多少」时，直接对打（slim vs p48）和间接比较（各自 vs 同一基线再相减）
给出的方差完全不同，而且 **1v3 结构下不同对手的差值不可横向相加**——
对手换成三家强敌后，"边际优势"会变。所以必须：

1. 每个臂都在**同一 seed 池、同一座位旋转、同一基线**上跑；
2. 用 `--pair` 把同一 seed 的两个臂相减（牌墙难度这一共同因素被消掉，方差更小）；
3. 报告里明确标注每个数字的对手，不做跨越对手的加减。

口径逐字复用 `tools/tost_gate.py::per_seed`（四座按 seed 平均，与冻结基线一致）。

用法::

    .venv/bin/python tools/gate_matrix.py \\
        --arm "学生 vs master+N32=runs/gates/slimB_vs_master32.jsonl" \\
        --arm "P48 vs master+N32=runs/gates/slimB_p48_vs_master32.jsonl" \\
        --arm "学生 vs P48贪心=runs/gates/slimB_vs_p48g.jsonl" \\
        --pair "学生 vs master+N32,P48 vs master+N32"
"""
from __future__ import annotations

import argparse
import json
import statistics as st
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))

from tost_gate import per_seed, t_ppf  # noqa: E402


def stats(vals: list[float]) -> dict:
    n = len(vals)
    if n < 2:
        return {"n": n}
    m = st.mean(vals)
    se = st.stdev(vals) / (n ** 0.5)
    q = t_ppf(0.975, n - 1)
    return {"n": n, "mean": m, "se": se, "lo": m - q * se, "hi": m + q * se,
            "pos": sum(1 for v in vals if v > 0),
            "neg": sum(1 for v in vals if v < 0),
            "median": st.median(vals), "sd": st.stdev(vals)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", action="append", default=[],
                    help="label=path，可多次")
    ap.add_argument("--pair", action="append", default=[],
                    help="A,B（两个 arm 的 label），输出同 seed 配对差")
    ap.add_argument("--json-out", default="")
    a = ap.parse_args()
    if not a.arm:
        ap.error("至少一个 --arm")

    arms: dict[str, dict[int, float]] = {}
    print(f"{'臂（候选 vs 对手）':<34}{'n':>6}{'均值':>9}{'±SE':>8}"
          f"{'95%CI':>22}{'符号 正:负':>14}{'中位':>8}")
    print("-" * 104)
    for spec in a.arm:
        if "=" not in spec:
            ap.error(f"--arm 需要 label=path：{spec}")
        label, path = spec.split("=", 1)
        p = Path(path)
        if not p.is_absolute():
            p = ROOT / p
        if not p.exists():
            print(f"{label:<34}  文件不存在 {p}")
            continue
        fold, raw_n, seat_n = per_seed(p)
        vals = [fold[s] for s in sorted(fold)]
        arms[label] = fold
        s = stats(vals)
        if s["n"] < 2:
            print(f"{label:<34}{s['n']:>6}  （样本不足）")
            continue
        ci = f"[{s['lo']:+.3f}, {s['hi']:+.3f}]"
        posneg = f"{s['pos']}:{s['neg']}"
        print(f"{label:<34}{s['n']:>6}{s['mean']:>+9.3f}{s['se']:>8.3f}"
              f"{ci:>22}{posneg:>14}{s['median']:>+8.3f}")
        if seat_n != 4 * len(fold):
            print(f"{'':<34}  ⚠ {raw_n} 条记录 / {len(fold)} 个 seed / {seat_n} 个座位"
                  f"（四座不齐：{seat_n} ≠ {4 * len(fold)}）")

    out = {"arms": {}, "pairs": []}
    for label, fold in arms.items():
        out["arms"][label] = {**stats([fold[s] for s in sorted(fold)]),
                              "seeds": len(fold)}

    for spec in a.pair:
        parts = [x.strip() for x in spec.split(",")]
        if len(parts) != 2 or any(p not in arms for p in parts):
            print(f"\n[配对] 跳过 {spec}（label 未定义）")
            continue
        A, B = arms[parts[0]], arms[parts[1]]
        common = sorted(set(A) & set(B))
        if len(common) < 2:
            print(f"\n[配对] {spec}：共同 seed 不足（{len(common)}）")
            continue
        d = [A[s] - B[s] for s in common]
        s = stats(d)
        ci = f"[{s['lo']:+.3f}, {s['hi']:+.3f}]"
        print(f"\n[配对差] {parts[0]}  −  {parts[1]}   共同 seed {s['n']}")
        print(f"  差值 {s['mean']:+.3f} ± {s['se']:.3f}   95%CI {ci}   "
              f"符号 {s['pos']}:{s['neg']}   中位 {s['median']:+.3f}")
        out["pairs"].append({"a": parts[0], "b": parts[1], **s,
                            "seeds": len(common)})

    if a.json_out:
        p = Path(a.json_out)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n",
                     encoding="utf-8")
        print(f"\n→ {p}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
