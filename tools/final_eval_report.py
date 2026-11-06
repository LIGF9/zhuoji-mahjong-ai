#!/usr/bin/env python3
"""终评 / 闸门统计（只读 · 冻结口径固化 · 不启动任何对局）。

唯一正确口径（自检 #7 踩坑后修正，详见 docs/14天训练计划.md §6.11.7）：

    对每条 (seed, seat) 记录取  d = deltas[cand_seat] - mean(其余三家)
    再把同一 seed 的四座旋转结果取平均  => 1 个独立样本
    最后在 seed 维度算 mean / SE / z / 双侧正态近似 p

  ❌ 取 ``deltas[0]``、或不做四座平均，结果全错（自检 #7 已踩坑）。
  ❌ 禁止用自对弈胜率作决策；禁止用 val loss 判断强度（跨轮分布漂移，不可比）。
  ❌ 闸门 A 口径是「新贪心 vs master 贪心」，不等于 §4 主目标「新+N32 vs master+N32」。

本脚本纯读 jsonl，不加载模型、不启对局、不占 CPU，可在 7×24 标注期安全 dry-run。
DEADLINE(2026-11-13 02:30) 之后用它产出终评报告数据。

用法::

    # dry-run：逐轮闸门，应与 night_run.log 的 [summary] 逐位一致
    ./.venv/bin/python tools/final_eval_report.py \
        R1A=runs/gates/r01_gateA.jsonl R2A=runs/gates/r02_gateA.jsonl \
        R3A=runs/gates/r03_gateA.jsonl R4A=runs/gates/r04_gateA.jsonl --md

    # 相邻轮配对环比（同 seed 池逐 seed 配对）
    ./.venv/bin/python tools/final_eval_report.py \
        R2A=runs/gates/r02_gateA.jsonl R3A=runs/gates/r03_gateA.jsonl \
        R4A=runs/gates/r04_gateA.jsonl --pair --md

    # 终评：500 seed 独立池 3_000_000+500
    ./.venv/bin/python tools/final_eval_report.py \
        primary=runs/gates/final_primary.jsonl \
        gateA=runs/gates/final_gateA.jsonl --md
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def per_seed(path: Path) -> dict[int, float]:
    """把 jsonl 折叠为 {seed: 四座平均后的分差}。"""
    bucket: dict[int, list[float]] = defaultdict(list)
    with path.open(encoding="utf-8") as f:
        for lineno, raw in enumerate(f, 1):
            raw = raw.strip()
            if not raw:
                continue
            try:
                d = json.loads(raw)
                cs = int(d["cand_seat"])
                dl = d["deltas"]
                others = [v for i, v in enumerate(dl) if i != cs]
                if not others:
                    raise ValueError("deltas 少于 2 家")
                bucket[int(d["seed"])].append(dl[cs] - sum(others) / len(others))
            except (KeyError, TypeError, ValueError) as e:
                raise ValueError(f"{path}:{lineno} 记录不合法: {e}") from e
    if not bucket:
        raise ValueError(f"{path}: 无有效记录")
    return {s: sum(v) / len(v) for s, v in bucket.items()}


def stats(vals: list[float]) -> dict:
    n = len(vals)
    if n < 2:
        raise ValueError(f"样本不足（n={n}），无法估计 SE")
    mean = sum(vals) / n
    se = math.sqrt(sum((x - mean) ** 2 for x in vals) / (n - 1) / n)
    z = mean / se if se > 0 else math.inf
    p = math.erfc(abs(z) / math.sqrt(2)) if math.isfinite(z) else 0.0
    return {
        "n": n,
        "mean": mean,
        "se": se,
        "z": z,
        "p": p,
        "pos": sum(1 for x in vals if x > 0),
        "neg": sum(1 for x in vals if x < 0),
    }


def fmt(name: str, s: dict) -> str:
    return (f"{name:<24} n={s['n']:<4} diff={s['mean']:+.3f}±{s['se']:.3f}  "
            f"z={s['z']:+.2f}  p≈{s['p']:.4f}  符号 {s['pos']}:{s['neg']}")


def md_table(rows: list[tuple[str, dict]]) -> str:
    out = ["| 口径 | n | diff ± SE | z | p | 符号 |", "|---|---|---|---|---|---|"]
    for name, s in rows:
        out.append(f"| {name} | {s['n']} | {s['mean']:+.3f}±{s['se']:.3f} | "
                   f"{s['z']:+.2f} | {s['p']:.4f} | {s['pos']}:{s['neg']} |")
    return "\n".join(out)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="终评/闸门统计（冻结口径，只读）")
    ap.add_argument("inputs", nargs="+", metavar="LABEL=PATH",
                    help="若干个 标签=jsonl路径，标签用于输出标识")
    ap.add_argument("--pair", action="store_true",
                    help="对相邻两个口径按 seed 做配对差分（环比检验）")
    ap.add_argument("--md", action="store_true",
                    help="额外输出 Markdown 表格")
    args = ap.parse_args(argv)

    series: list[tuple[str, dict[int, float]]] = []
    for spec in args.inputs:
        if "=" not in spec:
            ap.error(f"参数须为 LABEL=PATH，收到 {spec!r}")
        label, raw = spec.split("=", 1)
        path = Path(raw)
        if not path.is_absolute():
            path = ROOT / path
        if not path.exists():
            print(f"[ERROR] 文件不存在: {path}", file=sys.stderr)
            return 2
        series.append((label, per_seed(path)))

    print("=" * 88)
    print("冻结口径统计：deltas[cand_seat] − mean(其余三家)，四座按 seed 平均，双侧正态近似 p")
    print("=" * 88)
    rows: list[tuple[str, dict]] = []
    for label, d in series:
        s = stats(list(d.values()))
        rows.append((label, s))
        print("  " + fmt(label, s))

    if args.pair:
        print("\n-- 相邻口径逐 seed 配对差分 --")
        prows: list[tuple[str, dict]] = []
        for (l0, d0), (l1, d1) in zip(series, series[1:]):
            common = sorted(set(d0) & set(d1))
            if len(common) < 2:
                print(f"  [跳过] {l1}−{l0}: 共同 seed 仅 {len(common)} 个")
                continue
            s = stats([d1[k] - d0[k] for k in common])
            prows.append((f"{l1}−{l0}", s))
            print("  " + fmt(f"{l1}−{l0}", s) + f"   共同 seed={len(common)}")
        if args.md and prows:
            print("\n```markdown\n" + md_table(prows) + "\n```")

    if args.md:
        print("\n```markdown\n" + md_table(rows) + "\n```")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
