#!/usr/bin/env python3
"""P0 收口报告：学生 champ_slim_s 开 N32 搜索的棋力红利。

判据唯一来源是 ``docs/14天训练计划.md`` §6.73.9 五（看结果前锁死）：

* 主判据：同池同 seed 配对差 Δ = A − B（n=N=120）。mean>0 且 95% CI 下界>0
  ⇒ 「本池下学生 N32 的搜索红利成立」；否则只写「本样本量下无足够证据」，
  **禁写**「无差异 / 持平 / 到顶 / 搜索无用」。必须同时报告可分辨效应量 2×SE。
* 次判据：A、B 各自 vs master+N32 的 mean±SE / 95%CI / 符号，**只报数不相减**。
* 探索性 C：对手 = 3× 学生贪心，同口径单独给结论，**不得与 A/B 相减或相加**。
* 硬时限 11:29:14 截断后只报告已完成臂；被截断臂的部分行只作探索性描述。

折叠口径与 ``tools/tost_gate.py::per_seed`` 完全一致：
每局取 ``deltas[候选座] - mean(其它三座)``，四座轮转后按 seed 取平均。
"""
from __future__ import annotations

import json
import math
import statistics as st
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
from tost_gate import t_ppf, t_cdf  # noqa: E402  （无 scipy，自实现 t 分布）

ARMS = {
    "A": ("学生 N32 vs master+N32", ROOT / "runs/gates/p0_a_search_vs_master32.jsonl"),
    "B": ("学生贪心 vs master+N32", ROOT / "runs/gates/p0_b_greedy_vs_master32.jsonl"),
    "C": ("学生 N32 vs 3× 学生贪心", ROOT / "runs/gates/p0_c_search_vs_greedy.jsonl"),
}
N_EXPECT = 120
OUT_JSON = ROOT / "runs/gates/p0_n32_report.json"
OUT_MD = ROOT / "reports/P0-学生N32搜索红利.md"


def fold(path: Path) -> dict[int, tuple[float, int]]:
    """{seed: (四座平均分差, 座位数)}；坏行/半行跳过（硬停时可能截在半行）。"""
    bucket: dict[int, dict[int, float]] = {}
    skipped = 0
    if not path.exists():
        return {}
    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
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
            val = float(dl[cs]) - sum(others) / len(others)
            seed = int(d["seed"])
        except (KeyError, TypeError, ValueError, IndexError, json.JSONDecodeError):
            skipped += 1
            continue
        bucket.setdefault(seed, {})[cs] = val
    res = {s: (sum(v.values()) / len(v), len(v)) for s, v in bucket.items()}
    if skipped:
        print(f"  ⚠ {path.name}: 跳过 {skipped} 条坏行/半行（硬停截断常见）")
    return res


def describe(vals: list[float]) -> dict:
    n = len(vals)
    if n == 0:
        return {"n": 0}
    mean = st.mean(vals)
    if n < 2:
        return {"n": n, "mean": mean, "se": None, "ci": None, "sd": None}
    sd = st.stdev(vals)
    se = sd / math.sqrt(n)
    t = t_ppf(0.975, n - 1)
    return {
        "n": n, "mean": mean, "sd": sd, "se": se,
        "ci": [mean - t * se, mean + t * se],
        "resolvable_2se": 2 * se,
        "pos_frac": sum(1 for v in vals if v > 0) / n,
        "p_two_sided": 2 * (1 - t_cdf(abs(mean / se), n - 1)) if se > 0 else None,
    }


def fmt(d: dict) -> str:
    if not d or d.get("n", 0) == 0:
        return "无数据"
    n, m = d["n"], d["mean"]
    if d.get("se") is None:
        return f"n={n}  mean={m:+.3f}（样本不足，无法估 SE）"
    lo, hi = d["ci"]
    return (f"n={n:3d}  mean={m:+.3f}  SE={d['se']:.3f}  "
            f"95%CI=[{lo:+.3f},{hi:+.3f}]  2×SE≈{d['resolvable_2se']:.3f}  "
            f"正值占比={d['pos_frac']:.0%}")


def main(argv: list[str] | None = None) -> int:
    write = "--no-write" not in (argv if argv is not None else sys.argv[1:])
    lines: list[str] = []
    def say(s: str = "") -> None:
        lines.append(s)
        print(s)

    folds, raw_rows, secs = {}, {}, {}
    say("=== P0 收口报告 · 学生 champ_slim_s 开 N32 的棋力红利 ===")
    say()
    say("臂 | 说明 | 行数 | seed | 四座齐全 | 均值 s/局")
    for k, (label, p) in ARMS.items():
        f = fold(p)
        folds[k] = f
        rows = sum(1 for ln in p.read_text(encoding="utf-8", errors="replace").splitlines()
                   if ln.strip()) if p.exists() else 0
        raw_rows[k] = rows
        full = sum(1 for _, (_, ns) in f.items() if ns == 4)
        ss = []
        if p.exists():
            for ln in p.read_text(encoding="utf-8", errors="replace").splitlines():
                try:
                    ss.append(float(json.loads(ln)["s"]))
                except (ValueError, KeyError, TypeError):
                    pass
        secs[k] = st.mean(ss) if ss else None
        say(f" {k} | {label} | {rows} | {len(f)} | {full} | "
            f"{f'{secs[k]:.2f}s' if secs[k] is not None else '—'}")

    say()
    say("--- 次判据：各臂自身 vs 对手（只报数，互不相减）---")
    desc = {}
    for k in "ABC":
        desc[k] = describe([v for v, ns in folds[k].values() if ns == 4])
        say(f"  {k}: {fmt(desc[k])}")

    say()
    say("--- 主判据：Δ = A − B（同池同 seed 同座配对）---")
    common = sorted(set(s for s, (_, ns) in folds["A"].items() if ns == 4)
                    & set(s for s, (_, ns) in folds["B"].items() if ns == 4))
    paired = [folds["A"][s][0] - folds["B"][s][0] for s in common]
    d = describe(paired)
    say(f"  配对 seed 数 = {len(common)} / 预登记 {N_EXPECT}")
    say(f"  Δ: {fmt(d)}")
    truncated = len(common) < N_EXPECT
    if d["n"] >= 2 and d["mean"] > 0 and d["ci"][0] > 0 and not truncated:
        verdict = "本池下学生 N32 的搜索红利成立"
    elif truncated:
        verdict = ("硬时限截断导致配对样本不足预登记 n=120，"
                   "本样本量下无足够证据（仅作探索性描述，不参与裁决）")
    else:
        verdict = "本样本量下无足够证据"
    say(f"  裁决：{verdict}")
    say("  禁写「无差异 / 持平 / 到顶 / 搜索无用」；p>.05 不等于效应为零。")

    say()
    say("--- 探索性 C（对手 = 3× 学生贪心；不与 A/B 相减或相加）---")
    say(f"  C: {fmt(desc['C'])}")
    say("  说明：C 回答的是不同对手下的不同问题，不能替代 A−B 主判据。")

    say()
    say("--- 合规声明 ---")
    say("  权重 / champions.json / 线上主力 / seed 池 / 样本量 / worker 数 / "
        "n_det / 设备 全程未改动；成本剖析见 §6.73.9 一、八。")

    payload = {
        "arms": {k: {"label": ARMS[k][0], "rows": raw_rows[k],
                     "seeds": len(folds[k]),
                     "full_seeds": sum(1 for _, (_, ns) in folds[k].items() if ns == 4),
                     "mean_s_per_deal": secs[k], "describe": desc[k]}
                 for k in "ABC"},
        "primary": {"paired_seeds": len(common), "expect": N_EXPECT,
                    "describe": d, "verdict": verdict, "truncated": truncated},
    }
    if write:
        OUT_JSON.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
                            encoding="utf-8")
        OUT_MD.parent.mkdir(parents=True, exist_ok=True)
        OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
        say()
        say(f"已写 {OUT_JSON.relative_to(ROOT)} 与 {OUT_MD.relative_to(ROOT)}")
    else:
        say()
        say("--no-write：仅预览，未写任何文件")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
