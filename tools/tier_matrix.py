#!/usr/bin/env python3
"""离线版四档（adept / expert / master / legend）同池配对强度矩阵。

为什么单独一个驱动
------------------
`gate_matrix.py` 只汇总**已有** jsonl；四档是新档位组合，需要先按统一协议把
6 个配对跑出来。协议逐字复用冻结口径（docs/评测协议-v2.md / §0）：

  * 固定种子池（默认 7_300_000 起，与 E-SLIM 闸门 A 同池，方便与历史数字对比）；
  * 每个 seed 四座轮转（cand 坐 0/1/2/3），消掉座位与牌墙的共线性；
  * 全部**贪心**（det=0）；adept 是规则速听，不走任何搜索；
  * 四档各自 1 人 vs 对手 3 人全场同档 —— 与 P0 三臂同构，可直接与
    `reports/P0-学生N32搜索红利.md` 的数字并排读。

判据口径
--------
`tost_gate.per_seed`：每个 seed 先算 (cand_delta − 其余三家均值)，再跨 seed
取均值与 95% CI（t 分布）。正值 = 该档比对手每副多拿这么多分。

用法::

    .venv/bin/python tools/tier_matrix.py --seeds 40            # 冒烟
    .venv/bin/python tools/tier_matrix.py --seeds 400 --workers 6
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))

from tost_gate import per_seed, t_ppf  # noqa: E402

# 档位 → (权重, 模式)。模型只有三张网：expert 用 slim_smoke（s7 权重已不存在，
# 见 docs/离线版-可行性与实施方案.md §9 的决策记录）。
TIERS: dict[str, tuple[str, str]] = {
    "adept":  ("models/dushan_master.pt", "ting"),      # 占位路径，规则档不读权重
    "expert": ("models/expert_s7.pt", "greedy"),
    "master": ("models/dushan_master.pt", "greedy"),
    "legend": ("models/champ_slim_s.pt", "greedy"),
}

ORDER = ["adept", "expert", "master", "legend"]

# 六个无序配对；同时给出「对 master」与「两两直接对打」两类证据。
PAIRS = [
    ("adept", "expert"),
    ("adept", "master"),
    ("adept", "legend"),
    ("expert", "master"),
    ("expert", "legend"),
    ("master", "legend"),
]


def stats(vals: list[float]) -> dict:
    n = len(vals)
    m = sum(vals) / n
    sd = (sum((x - m) ** 2 for x in vals) / (n - 1)) ** 0.5 if n > 1 else 0.0
    se = sd / (n ** 0.5) if n else 0.0
    q = t_ppf(0.975, max(n - 1, 1))
    return {
        "n": n,
        "mean": m,
        "se": se,
        "lo": m - q * se,
        "hi": m + q * se,
        "pos": sum(1 for v in vals if v > 0),
        "neg": sum(1 for v in vals if v < 0),
        "sd": sd,
    }


def run_pair(a: str, b: str, seeds: int, seed0: int, workers: int, out: Path) -> dict:
    pa, ma = TIERS[a]
    pb, mb = TIERS[b]
    cmd = [
        str(ROOT / ".venv/bin/python"), "eval_pairs.py",
        "--a", pa, "--b", pb,
        "--a-mode", ma, "--b-mode", mb,
        "--seeds", str(seeds), "--seed0", str(seed0),
        "--seats", "0,1,2,3", "--workers", str(workers),
        "--out", str(out),
    ]
    t0 = time.time()
    with open(ROOT / "runs/tiers/eval.log", "a", encoding="utf-8") as lf:
        lf.write(f"$ {' '.join(cmd)}\n")
        lf.flush()
        rc = subprocess.call(cmd, cwd=str(ROOT), stdout=lf, stderr=lf)
    dt = time.time() - t0
    if rc != 0:
        raise SystemExit(f"{a} vs {b} 评测失败 rc={rc}，日志见 runs/tiers/eval.log")
    per, raw, dedup = per_seed(out)
    st = stats(list(per.values()))
    rec = {"a": a, "b": b, "file": str(out.relative_to(ROOT)),
           "raw": raw, "dedup": dedup, "secs": round(dt, 1), **st}
    print(f"  {a:>6} vs 3×{b:<6} {st['mean']:+.3f}±{st['se']:.3f} "
          f"CI[{st['lo']:+.3f},{st['hi']:+.3f}]  {st['pos']}:{st['neg']}  ({dt:.0f}s)",
          flush=True)
    return rec


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=400)
    ap.add_argument("--seed0", type=int, default=7_300_000)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--out", default="runs/tiers/matrix.json")
    args = ap.parse_args()

    (ROOT / "runs/tiers").mkdir(parents=True, exist_ok=True)
    print(f"=== 四档矩阵：{args.seeds} seeds × 4 座 × 6 配对 "
          f"（seed0={args.seed0}，workers={args.workers}）===", flush=True)
    recs = []
    for a, b in PAIRS:
        out = ROOT / f"runs/tiers/{a}_vs_{b}.jsonl"
        recs.append(run_pair(a, b, args.seeds, args.seed0, args.workers, out))

    dump = {"seeds": args.seeds, "seed0": args.seed0, "pairs": recs,
            "tiers": {k: {"ckpt": v[0], "mode": v[1]} for k, v in TIERS.items()}}
    Path(ROOT / args.out).write_text(json.dumps(dump, ensure_ascii=False, indent=2) + "\n",
                                     encoding="utf-8")

    # 单尺度：把每档 vs master 的配对差当成该档的「实力坐标」（master=0）。
    # 注意：不同对手下的差值不可横向相加，这里只做**展示用**的一维投影，
    # 正式判据仍看每一对自身的 CI。
    print("\n=== 一维投影（对 master 的配对差；仅展示，判据看各对 CI）===")
    by_pair = {(r["a"], r["b"]): r for r in recs}
    for t in ORDER:
        if t == "master":
            print(f"  {t:<7} 0.000（基准）")
        else:
            r = by_pair.get((t, "master")) or by_pair.get(("master", t))
            sign = 1 if (t, "master") in by_pair else -1
            print(f"  {t:<7} {sign * r['mean']:+.3f} ± {r['se']:.3f}")

    print("\n=== 两两直接对打（正 = 前者更强）===")
    for a, b in PAIRS:
        r = by_pair[(a, b)]
        verdict = "前者强" if r["lo"] > 0 else ("后者强" if r["hi"] < 0 else "未分胜负")
        print(f"  {a:>6} vs {b:<6} {r['mean']:+.3f} CI[{r['lo']:+.3f},{r['hi']:+.3f}]  {verdict}")
    print(f"\n已写 {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
