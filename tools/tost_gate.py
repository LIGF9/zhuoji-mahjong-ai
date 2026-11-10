#!/usr/bin/env python3
"""E-SLIM 闸门 A 判定（TOST 非劣）—— 只读 · 零 CPU · 不起任何对局。

口径来源
--------
``docs/14天训练计划.md §6.73.4``（2026-11-10 13:45 CST 预登记锁死）：

* 统计量恒 ``deltas[cand_seat] − mean(其余三家)`` → 四座按 seed 平均；
* 主判据池 ``7_300_000..7_300_299``（300 seed × 四座 = 1200 局）；
* 非劣阈值 ``δ = −0.5`` 分：要求 **95% CI 下界 > −0.5** 且 **TOST p < .05**。

本脚本只做算术，不做任何"尽量通过"的处理：
* 记录按 ``(seed, seat)`` 去重（后写覆盖），并校验 seed 池与座位覆盖；
* ``n < 2``、座位不足 4、seed 越界 → 直接退出 2，不出结论。

与 ``tools/final_eval_report.py`` 的关系
--------------------------------------
口径（deltas 折叠方式）逐字复用它的 ``per_seed``；差异只在检验：
它给正态近似 z/p，本脚本给 **t 分布** 与 **TOST 双侧等价**（scipy 不在依赖里，
用正则化不完全 Beta 函数自实现，见 ``_betacf``/``_betai``）。

用法
----
    .venv/bin/python tools/tost_gate.py --jsonl runs/gates/slim_gateA.jsonl
    # 工程项一起判（§6.73.4 四象限）：
    .venv/bin/python tools/tost_gate.py --jsonl runs/gates/slim_gateA.jsonl --eng-pass
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

DELTA = -0.5          # 非劣阈值（预登记，事后不得改）
DELTA_UP = 0.5        # TOST 上界（等价区间 −0.5 ~ +0.5）
SEED_LO, SEED_HI = 7_300_000, 7_300_299
ALPHA = 0.05


# ---------------- t 分布（无 scipy） ----------------

def _betacf(a: float, b: float, x: float, itmax: int = 300, eps: float = 3e-16) -> float:
    """不完全 Beta 的连分式（Numerical Recipes 6.4 betacf）。"""
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c, d = 1.0, 1.0 - qab * x / qap
    if abs(d) < 1e-300:
        d = 1e-300
    d = 1.0 / d
    h = d
    for m in range(1, itmax + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < 1e-300:
            d = 1e-300
        c = 1.0 + aa / c
        if abs(c) < 1e-300:
            c = 1e-300
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < 1e-300:
            d = 1e-300
        c = 1.0 + aa / c
        if abs(c) < 1e-300:
            c = 1e-300
        d = 1.0 / d
        de = d * c
        h *= de
        if abs(de - 1.0) < eps:
            break
    return h


def _betai(a: float, b: float, x: float) -> float:
    """正则化不完全 Beta 函数 I_x(a,b)。"""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    lbeta = math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
    front = math.exp(lbeta + a * math.log(x) + b * math.log1p(-x))
    if x < (a + 1.0) / (a + b + 2.0):
        return front * _betacf(a, b, x) / a
    return 1.0 - front * _betacf(b, a, 1.0 - x) / b


def t_cdf(t: float, df: float) -> float:
    """P(T ≤ t)，df 自由度。"""
    x = df / (df + t * t)
    tail = 0.5 * _betai(df / 2.0, 0.5, x)
    return 1.0 - tail if t > 0 else tail


def t_ppf(q: float, df: float) -> float:
    """分位数（对 t_cdf 二分；df≥1 单调，足够精确）。"""
    lo, hi = -1e3, 1e3
    for _ in range(200):
        mid = (lo + hi) / 2.0
        if t_cdf(mid, df) < q:
            lo = mid
        else:
            hi = mid
        if hi - lo < 1e-10:
            break
    return (lo + hi) / 2.0


# ---------------- 数据 ----------------

def per_seed(path: Path) -> tuple[dict[int, float], int, int]:
    """折叠成 {seed: 四座平均分差}，返回 (mapping, 记录数, 去重后记录数)。"""
    bucket: dict[int, list[float]] = {}
    seats: dict[int, set[int]] = {}
    raw_n = 0
    for lineno, raw in enumerate(path.open(encoding="utf-8"), 1):
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
        except (KeyError, TypeError, ValueError) as e:
            raise ValueError(f"{path}:{lineno} 记录不合法: {e}") from e
        raw_n += 1
        seed = int(d["seed"])
        # 同 (seed, seat) 后写覆盖 —— 分块重跑不会把同一局算两次
        prev = bucket.setdefault(seed, [])
        seat_set = seats.setdefault(seed, set())
        if cs in seat_set:
            idx = sorted(seat_set).index(cs)
            prev[idx] = dl[cs] - sum(others) / len(others)
        else:
            seat_set.add(cs)
            prev.append(dl[cs] - sum(others) / len(others))
    if not bucket:
        raise ValueError(f"{path}: 无有效记录")
    return ({s: sum(v) / len(v) for s, v in bucket.items()},
            raw_n, sum(len(v) for v in seats.values()))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl", required=True)
    ap.add_argument("--delta", type=float, default=DELTA)
    ap.add_argument("--eng-pass", action="store_true",
                    help="工程判据（参数量/N32/贪心/ONNX/编码）是否全 PASS，用于四象限")
    ap.add_argument("--out", default="", help="把结论写成 JSON")
    args = ap.parse_args()

    path = Path(args.jsonl)
    fold, raw_n, uniq_n = per_seed(path)
    vals = [fold[s] for s in sorted(fold)]
    n = len(vals)
    print(f"=== 闸门 A · TOST 非劣（δ={args.delta:+.2f}）===")
    print(f"  文件 {path}  记录 {raw_n} 条（去重后 {uniq_n}）  seed 数 {n}")

    # 池与座位校验（预登记池，不因数据缺失而"降级通过"）
    lo, hi = min(fold), max(fold)
    bad = [s for s in fold if not (SEED_LO <= s <= SEED_HI)]
    if bad:
        print(f"  FAIL seed 越界 {len(bad)} 个（池应为 {SEED_LO}..{SEED_HI}）：{bad[:5]}")
        return 2
    if uniq_n < 4 * n:
        print(f"  FAIL 座位覆盖不全：{uniq_n} 条 / 应 {4 * n} 条（四座轮转）")
        return 2
    if n < 2:
        print(f"  FAIL 样本不足（n={n}），无法估 SE")
        return 2
    print(f"  seed 范围 {lo}..{hi}  座位覆盖 {uniq_n}/{4 * n} ✓")

    mean = sum(vals) / n
    var = sum((x - mean) ** 2 for x in vals) / (n - 1)
    sd = math.sqrt(var)
    se = sd / math.sqrt(n)
    df = n - 1
    tq = t_ppf(0.975, df)
    ci_lo, ci_hi = mean - tq * se, mean + tq * se

    # 单侧非劣：H0 μ ≤ δ  vs  H1 μ > δ
    t_noninf = (mean - args.delta) / se
    p_noninf = 1.0 - t_cdf(t_noninf, df)
    # TOST 上界：H0 μ ≥ +δ_up  vs  H1 μ < +δ_up
    t_up = (mean - DELTA_UP) / se
    p_up = t_cdf(t_up, df)
    p_tost = max(p_noninf, p_up)

    z = mean / se
    p_z = math.erfc(abs(z) / math.sqrt(2))

    print(f"  n={n}  diff={mean:+.4f}  sd={sd:.4f}  se={se:.4f}")
    print(f"  95% CI [{ci_lo:+.4f}, {ci_hi:+.4f}]  (t[{df}], t.975={tq:.3f})")
    print(f"  正态近似 z={z:+.3f}  p={p_z:.4g}   （参考，不作判据）")
    print(f"  非劣单侧 t={t_noninf:+.3f}  p={p_noninf:.4g}")
    print(f"  TOST 上界 t={t_up:+.3f}  p={p_up:.4g}")
    print(f"  p_TOST = {p_tost:.4g}  (阈值 < {ALPHA})")

    stat_ok = (ci_lo > args.delta) and (p_tost < ALPHA)
    print(f"  → 统计判据 {'PASS' if stat_ok else 'FAIL'}"
          f"（CI 下界 {ci_lo:+.4f} {'>' if ci_lo > args.delta else '≤'} {args.delta:+.2f}"
          f"{'，' if stat_ok else ' 或 '}p_TOST {p_tost:.4g}"
          f" {'<' if p_tost < ALPHA else '≥'} {ALPHA}）")

    if args.eng_pass is not False and args.eng_pass:
        verdict = "采纳为候选冠军（champions.json 只增不替，线上主力仍 dushan_master）" if stat_ok \
            else "记「无足够证据表明非劣」，不给线上"
    else:
        verdict = ("统计过但工程不过：先查 ONNX/后端，不改统计判据" if stat_ok
                   else "统计不过且工程不过：终止，登记负面结果")
    print(f"  → 四象限结论：{verdict}")

    if args.out:
        Path(args.out).write_text(json.dumps({
            "jsonl": str(path), "n": n, "records": raw_n, "unique": uniq_n,
            "mean": mean, "sd": sd, "se": se, "df": df,
            "ci95": [ci_lo, ci_hi], "delta": args.delta,
            "t_noninf": t_noninf, "p_noninf": p_noninf,
            "p_tost": p_tost, "p_normal_z": p_z, "z": z,
            "stat_pass": stat_ok, "eng_pass": bool(args.eng_pass),
            "verdict": verdict,
        }, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"  → {args.out}")

    return 0 if stat_ok else 1


if __name__ == "__main__":
    sys.exit(main())
