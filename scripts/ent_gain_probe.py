#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""熵-系数增益探针：从两个「只差 --ent-coef」的短跑里量出 dH/d(ec)。

为什么需要它
------------
M2-A 的 A1 是**比例控制**：``ec ← clip(ec·exp(η(H*−H)), 0.01, 0.5)``。
比例控制能稳定工作的前提是**被控对象对控制量有响应**，即 dH/d(ec) ≠ 0。
60 轮验证跑的实测暴露了一个致命前提错误：

    H₀（段首 25 轮均值）= 1.0253    H* = 0.75·H₀ = 0.7690
    H_steady（末 20 轮均值）= 0.9389   距 H* 缺口 +0.1699

即 **H 的自然稳态是段首的 91.6%，而目标设在了 75% —— 在数学上不可达**。
后果：``H > H*`` 恒成立 → ``exp(η(H*−H)) < 1`` 恒成立 → ec 只会减小，
而它一开始就在下限 0.01 → **ec 全程静止，A1 等于没上**。

这与它要修的旧缺陷（旧阈值规则把 ec 钉在 0.03 从不动作）是**同一类失效**，
只是位置从 0.03 换到了 0.01。留着它，M2-A 会白烧 6.6 小时，且段末会把
「熵没控住」错误归因给蒸馏。

本脚本做两件事
--------------
1. 量增益 ``g = ΔH / Δec``（拿两个只差 ec 的短跑比），据此判断 ec 有没有控制力；
2. 反推「要维持 H* = ratio·H₀ 需要 ec 稳定在多少」，给出可落地的 ``--ent-target-ratio``。

用法
----
    python scripts/ent_gain_probe.py --lo reports/_ec_gain_lo.json \
                                     --hi reports/_ec_gain_hi.json \
                                     --ec-lo 0.02 --ec-hi 0.30 --h0 1.0253
"""
import argparse
import json
import pathlib
import sys

import numpy as np


def load(path: pathlib.Path):
    d = json.loads(path.read_text(encoding="utf-8"))
    recs = d if isinstance(d, list) else (d.get("history") or d.get("recs") or [])
    if not recs:
        sys.exit(f"✗ {path} 里没有记录")
    return recs


def tail_mean(recs, key, n, skip=2):
    """后 n 轮均值（丢前 skip 轮：ec 改变后 H 有滞后，瞬时值不是稳态）。"""
    v = [r[key] for r in recs if r.get(key) is not None]
    v = v[skip:]
    if not v:
        return float("nan")
    return float(np.mean(v[-n:]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lo", type=pathlib.Path, required=True,
                    help="低 ec 组的 history json")
    ap.add_argument("--hi", type=pathlib.Path, required=True,
                    help="高 ec 组的 history json")
    ap.add_argument("--ec-lo", type=float, default=None, help="低组 ec（默认从记录里读）")
    ap.add_argument("--ec-hi", type=float, default=None, help="高组 ec")
    ap.add_argument("--h0", type=float, default=None,
                    help="段首基线 H₀（用于把 H* 换算成 ratio；不给则只报增益）")
    ap.add_argument("--tail", type=int, default=8, help="取尾部多少轮做稳态均值")
    ap.add_argument("--deadzone", type=float, default=0.03)
    args = ap.parse_args()

    lo, hi = load(args.lo), load(args.hi)
    ec_lo = args.ec_lo if args.ec_lo is not None else float(lo[0]["ent_coef"])
    ec_hi = args.ec_hi if args.ec_hi is not None else float(hi[0]["ent_coef"])

    print("=" * 78)
    print("熵-系数增益探针")
    print("=" * 78)
    print(f"  低组：{args.lo.name}  ec={ec_lo:.3f}  {len(lo)} 轮")
    print(f"  高组：{args.hi.name}  ec={ec_hi:.3f}  {len(hi)} 轮")
    print()

    # ---------- 1. 逐轮对照 ----------
    keys = ["entropy", "klt", "kl", "samples"]
    hdr = "  it  " + "  ".join(f"{k:>22}" for k in keys)
    print(hdr)
    print("      " + "  ".join(f"{'低':>10} {'高':>10}" for _ in keys))
    print("-" * 78)
    n = min(len(lo), len(hi))
    for i in range(n):
        cells = []
        for k in keys:
            a, b = lo[i].get(k), hi[i].get(k)
            fmt = "{:>10.4f}" if isinstance(a, float) else "{:>10}"
            cells.append(f"{fmt.format(a) if a is not None else '         —'}"
                         f" {fmt.format(b) if b is not None else '         —'}")
        print(f"  {lo[i].get('iter', i + 1):>3}  " + "  ".join(cells))

    # ---------- 2. 增益 ----------
    h_lo = tail_mean(lo, "entropy", args.tail)
    h_hi = tail_mean(hi, "entropy", args.tail)
    dec = ec_hi - ec_lo
    dh = h_hi - h_lo
    g = dh / dec if abs(dec) > 1e-12 else float("nan")

    print()
    print("-" * 78)
    print(f"  稳态熵（尾部 {args.tail} 轮均值）：")
    print(f"    ec={ec_lo:.3f} → H={h_lo:.4f}")
    print(f"    ec={ec_hi:.3f} → H={h_hi:.4f}")
    print(f"    Δec={dec:+.3f}   ΔH={dh:+.4f}")
    print(f"  ★ 增益 g = dH/d(ec) = {g:+.4f}  "
          f"（ec 每翻 10 倍，H 变 {g * ec_lo * 9:+.4f}）")

    if abs(dh) < 0.01:
        print("\n  ✗ 增益≈0 —— 在这个配置下 ec 几乎推不动熵。")
        print("    可能原因：KLT 蒸馏/KL 锚把分布钉死，熵奖励被淹没。")
        print("    → A1 需要换机制（如直接对 logits 加温度/熵下限约束），不能靠 ec。")
        return 1

    print("\n  ✓ 增益显著 —— ec 有控制力，A1 的比例控制可用，问题**只在 H\\* 定标**。")

    # ---------- 3. 反推 ratio ----------
    if args.h0:
        h0 = args.h0
        print()
        print("-" * 78)
        print(f"  H₀ = {h0:.4f}（段首基线）")
        print(f"  当前自然稳态（ec 很小的时候）≈ {h_lo:.4f}"
              f"  →  H_steady/H₀ = {h_lo / h0:.4f}")
        print()
        print("  候选 --ent-target-ratio 的行为（用实测增益线性外推）：")
        print(f"    {'ratio':>6} {'H*':>8} {'需要 ec':>10}   判读")
        print("    " + "-" * 62)
        best, best_score = None, 1e9
        for ratio in (0.75, 0.80, 0.85, 0.88, 0.90, 0.92, 0.94, 0.95, 0.96, 0.98, 1.00):
            hstar = ratio * h0
            # 线性模型：H(ec) ≈ h_lo + g·(ec − ec_lo)
            need = ec_lo + (hstar - h_lo) / g if abs(g) > 1e-12 else float("inf")
            if need < 0.01:
                verdict = "○ 不可达（H* 低于自然稳态）→ ec 恒在下限，控制器静止"
                score = abs(hstar - h_lo)
            elif need > 0.5:
                verdict = "✗ 需 ec > 上限 0.5 → 会撞 clamp，熵奖励过强"
                score = 1e6
            else:
                tag = "★" if need <= 0.30 else " "
                verdict = f"{tag} 可达，ec 停在 {need:.3f}（区间内 {need / 0.5:.0%} 深度）"
                score = abs(np.log10(num_safe(need)) - np.log10(0.06))
            if score < best_score:
                best_score, best = score, ratio
            print(f"    {ratio:>6.2f} {hstar:>8.4f} {need:>10.3f}   {verdict}")
        print()
        print(f"  建议 --ent-target-ratio ≈ {best:.2f}"
              f"（H* = {best * h0:.4f}，ec 稳定在可用区间中部）")
    else:
        print("\n  （未给 --h0，跳过 ratio 反推）")

    # ---------- 4. 副作用 ----------
    print()
    print("-" * 78)
    print("  副作用对照（ec 变大是否伤样本量 / 蒸馏）：")
    for k, fmt, note in (("samples", "{:>10.0f}", "samples/轮"),
                         ("klt", "{:>10.4f}", "KLT(π_master‖π_θ)"),
                         ("kl", "{:>10.4f}", "KL(vs 锚)")):
        a = tail_mean(lo, k, args.tail)
        b = tail_mean(hi, k, args.tail)
        rel = (b / a - 1) if abs(a) > 1e-12 else float("nan")
        print(f"    {note:<22} 低 {fmt.format(a)} → 高 {fmt.format(b)}   ({rel:+.1%})")
    return 0


def num_safe(x):
    return max(x, 1e-6)


if __name__ == "__main__":
    sys.exit(main())
