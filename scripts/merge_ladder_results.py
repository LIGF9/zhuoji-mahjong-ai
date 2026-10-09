"""把多个 ``ladder_test.py`` 结果按逆方差加权合并成单一结论。

为什么需要
----------
单次评测的牌池是有限的：即使跑 5000 副，CI 也只到 ±0.3。要进一步提高精度，
最容易的做法是**换一个牌池 seed 再跑一遍**（独立复现），然后把两次结果合并——
这同时带来两个好处：精度提升，以及"结论在独立样本上是否可复现"的直接证据。

合并用逆方差加权（固定效应）：
    w_i = 1 / se_i²            m = Σ w_i m_i / Σ w_i       se = 1 / sqrt(Σ w_i)
并给出异质性统计量 Cochran's Q 与 I²（Q 大说明各批次效应量不一致，
此时合并值本身不可信，应先解释差异来源）。

重叠保护
--------
若两个结果文件的 ``seed0`` 相同，说明它们用的是**同一个牌池**
（例如 200 副 pilot 是 5000 副主检的前缀），逐副观测重叠，**不能当独立证据合并**。
脚本会检测到并直接报错退出。

用法::

    python scripts/merge_ladder_results.py \
        reports/ladder_arena_s3_s7_s11.json reports/ladder_arena_seed2.json
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load(path: Path) -> dict:
    d = json.loads(path.read_text(encoding="utf-8"))
    if "pairs" not in d:
        raise SystemExit(f"{path} 里没有 pairs 字段，确认是 ladder_test.py 的输出")
    return d


def chi2_sf(x: float, df: int) -> float:
    """卡方右尾概率（Wilson–Hilferty 近似，df>=1 时够准）。"""
    if df <= 0:
        return 1.0
    if x <= 0:
        return 1.0
    z = ((x / df) ** (1.0 / 3.0) - (1 - 2.0 / (9 * df))) / math.sqrt(2.0 / (9 * df))
    # 正态右尾
    return 0.5 * math.erfc(z / math.sqrt(2.0))


def merge_effect(entries: list[tuple[str, float, float]]) -> dict:
    """entries = [(tag, mean, se)] → 逆方差加权合并。"""
    ws = [1.0 / (se ** 2) for _, _, se in entries]
    sw = sum(ws)
    m = sum(w * e[1] for w, e in zip(ws, entries)) / sw
    se = 1.0 / math.sqrt(sw)
    k = len(entries)
    q = sum(w * (e[1] - m) ** 2 for w, e in zip(ws, entries)) if k > 1 else 0.0
    df = k - 1
    p_q = chi2_sf(q, df) if df > 0 else 1.0
    i2 = max(0.0, (q - df) / q) if q > 0 else 0.0
    verdict = ("显著优" if m - 1.96 * se > 0 else
               "显著劣" if m + 1.96 * se < 0 else "无显著差异")
    return {
        "k": k, "mean": round(m, 4), "se": round(se, 4),
        "ci95": round(1.96 * se, 4), "verdict": verdict,
        "Q": round(q, 4), "df": df, "p_Q": round(p_q, 4), "I2": round(i2, 4),
        "parts": [{"tag": t, "mean": round(mm, 4), "se": round(ss, 4),
                   "weight_pct": round(w / sw * 100, 1)}
                  for (t, mm, ss), w in zip(entries, ws)],
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+", help="两个或更多 ladder_test.py 输出的 JSON")
    ap.add_argument("--out", default=None, help="合并结果写入的 JSON 路径")
    args = ap.parse_args()

    files = [Path(f) for f in args.files]
    runs = []
    for f in files:
        d = load(f)
        runs.append((f, d))

    # 牌池重叠检查：同 seed0 + 同 mode 视为同一批牌，不可合并
    seen: dict[tuple, Path] = {}
    for f, d in runs:
        key = (d.get("seed0"), d.get("mode"), d.get("games_total"))
        if key in seen:
            raise SystemExit(
                f"牌池重叠：{f} 与 {seen[key]} 都是 seed0={key[0]} / mode={key[1]}\n"
                f"它们是同一批牌（或前缀关系），不能当独立证据合并。"
                f"请用不同 --seed 重跑后再合并。")
        seen[key] = f

    # 检查可比性
    modes = {d.get("mode") for _, d in runs}
    if len(modes) > 1:
        print(f"⚠️ 各批 mode 不一致 {modes}，跨模式合并需谨慎（尺度可能不同）")
    ck_signatures = {tuple(d["ckpts"]) for _, d in runs}
    if len(ck_signatures) > 1:
        raise SystemExit(f"各批检查点列表不一致：{ck_signatures}")

    pair_keys = list(runs[0][1]["pairs"].keys())
    print(f"合并 {len(runs)} 批结果（固定效应逆方差加权）")
    print(f"检查点：{'>'.join(runs[0][1]['ckpts'])}   模式：{sorted(modes, key=str)}")
    for f, d in runs:
        print(f"  · {f.name}: {d['deals']} 副 × {d.get('games_per_deal', 4)} 局"
              f"，seed0={d['seed0']}，{d['elapsed_min']} 分")
    print()

    out: dict = {"ckpts": runs[0][1]["ckpts"], "merged": {}, "runs": []}
    for f, d in runs:
        out["runs"].append({"file": f.name, "deals": d["deals"],
                            "seed0": d["seed0"], "mode": d.get("mode")})

    print(f"{'比较':<40}{'合并净分差':>11}{'±95%CI':>10}{'Q':>8}{'p(Q)':>8}"
          f"{'I²':>7}   结论")
    for k in pair_keys:
        entries = []
        for f, d in runs:
            s = d["pairs"][k]
            entries.append((f.stem, s["mean"], s["se"]))
        mg = merge_effect(entries)
        out["merged"][k] = mg
        warn = "  ⚠️异质" if (mg["p_Q"] < 0.05 and mg["k"] > 1) else ""
        print(f"{k:<40}{mg['mean']:>+11.3f}{mg['ci95']:>10.3f}{mg['Q']:>8.2f}"
              f"{mg['p_Q']:>8.3f}{mg['I2']:>7.2f}   {mg['verdict']}{warn}")

    # 阶梯判据（相邻代增量 = 晚代 − 早代）
    ck = runs[0][1]["ckpts"]
    print("\n== 阶梯判据（合并后） ==")
    out["increments"] = {}
    ok = True
    for i in range(len(ck) - 1):
        new, old = ck[i + 1], ck[i]
        key = f"{new} - {old}"
        entries = []
        for f, d in runs:
            # 增量 = −(old − new)
            base = d["pairs"].get(f"{old} - {new}")
            if base is None:
                continue
            entries.append((f.stem, -base["mean"], base["se"]))
        if not entries:
            continue
        mg = merge_effect(entries)
        out["increments"][key] = mg
        print(f"  {key:<38}{mg['mean']:>+9.3f} ± {mg['ci95']:.3f}   "
              f"{mg['verdict']}（I²={mg['I2']:.0%}）")
        ok = ok and mg["mean"] > 0 and mg["ci95"] < mg["mean"]
    print(f"  → 逐代稳定变强：{'【成立】' if ok else '【不成立】'}")
    out["ladder_verdict"] = {"monotone_by_increment": bool(ok)}

    if args.out:
        p = Path(args.out)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n合并结果 -> {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
