# -*- coding: utf-8 -*-
"""从训练 history 里量出 KL 的**自然量级**，并为 `--kl-target` 给出候选值。

要解决的问题
------------
``--kl-target`` 是 `CoefController` 的 **KL 危险线**（KL 超过它就把 ``kl_coef``
乘 1.6，低于其 40% 就除 1.4）。它是**纯经验量**，写死在参数里极容易错：

* 2026-10-14 之前 ``--kl-target 0.25``：实测 KL 只有 ~0.02 → **永远不触发**，
  锚项形同虚设（计划 §3 的 R8）。
* 照 R8 把目标降到 ``0.05``：又把目标放到了**正常漂移之下** → 每次测量都触发，
  ``kl_coef`` 会在 25–30 轮内撞上 clamp 上限 0.5，把 9.8 M 模型锁在一个过期锚上。

两者是同一个错：**目标的量级没有对着测量口径标定**。而测量口径是被这几个参数
共同决定的 —— ``--kl-anchor-every``（降频把"每轮漂移"变成"n 轮累积漂移"）、
``--kl-anchor-rolling``（EMA 锚的刷新周期）、``--lr``。改其中任何一个，KL 的量级
都会变，所以这个标定必须**在目标配方确定之后、按实测数据做一次**。

用法
----
    # 从一个「无 KL 控制」的对照跑（kl_coef 恒定、或控制器被故意关掉）里量：
    python scripts/kl_scale_probe.py --history reports/_m2a_smoke_history.json

    # 也可以直接从 .out 日志抓 iter 行（无需 history）：
    python scripts/kl_scale_probe.py --log reports/_m2a_smoke.out

它做三件事：
1. 打印 KL 的实测序列（去掉降频造成的重复值）、分位与每轮增幅；
2. 对一组候选 ``--kl-target`` 做**反事实重放**：控制器按实测 KL 序列走一遍，
   报告"触发次数 / kc 终值 / kc 峰值 / 峰值占 clamp 上限的比例"；
3. 给出建议值，判据写在 ``suggest()`` 里（安全网语义：常态不触发，坍缩必触发）。
"""

from __future__ import annotations

import argparse
import json
import pathlib
import re
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
_SRC = (ROOT / "scripts" / "selfplay.py").read_text(encoding="utf-8")
_m = re.search(r"^class CoefController:.*?(?=^def |^class )", _SRC, re.S | re.M)
if not _m:
    sys.exit("✗ 找不到 CoefController")
import math  # noqa: E402

_ns: dict = {"np": np, "math": math}
exec(_m.group(0), _ns)                                   # noqa: S102
CoefController = _ns["CoefController"]

KL_HI = CoefController.KL_HI          # clamp 上限（0.5）
KL_LO = CoefController.KL_LO
DEFAULT_KC = 0.02                     # --kl-coef 默认起点
RELAX_RATIO = 0.25                    # 与 CoefController 的 --kl-relax-ratio 默认一致
WATCHDOG_KL = 2.5                     # selfplay.py 的 KL 看门狗阈值（率口径）

_LINE = re.compile(r"^iter\s+\d+.*?\bKL=(?P<kl>[-\d.]+)", re.M)


def load_series(path: pathlib.Path) -> tuple[list[int], list[float]]:
    """返回 (iters, kls)。自动兼容 history.json 与 .out 日志；去掉降频重复值。"""
    if path.suffix.lower() == ".json":
        recs = json.loads(path.read_text(encoding="utf-8"))
        its, kls = [], []
        for i, r in enumerate(recs):
            v = r.get("kl")
            if v is None:
                continue
            its.append(int(r.get("iter", i + 1)))
            kls.append(float(v))
    else:
        txt = path.read_text(encoding="utf-8", errors="replace")
        its, kls = [], []
        for m in _LINE.finditer(txt):
            its.append(len(its) + 1)
            kls.append(float(m.group("kl")))
    # 降频那几轮 history 里写的是"最近一次有效 KL"（同一个值重复），去重
    its_u, kls_u = [], []
    for i, (a, b) in enumerate(zip(its, kls)):
        if i and abs(b - kls_u[-1]) < 1e-12:
            continue
        its_u.append(a)
        kls_u.append(b)
    return its_u, kls_u


def rates_from(its: list[int], kls: list[float], rolling: int, warmup: int) -> list[float]:
    """把"距锚点的距离"换算成"每轮漂移率"。

    距离是 ``kl``，年龄是"距锚点上次对齐（建锚 / EMA 刷新）的轮数"。锚点年龄不归零
    的话，同一个健康训练的距离会随锚点变旧线性增长 —— 这正是把 0.05 当裸 kl 用时
    "每次测量都触发"的原因。锚点事件：--anchor-warmup 轮末建锚，之后每 --kl-anchor-rolling
    轮 EMA 刷新一次。

    ★ 日志/history 里的 ``iter`` 是 **1-based**（``it + 1 + iter_offset``），而主循环里
    算年龄用的是 0-based ``it`` —— 这里必须减 1，否则年龄整体多 1、率整体偏低约
    1/(年龄) 倍（实测：不修的话首个测量点会算出 0.0116 而真值是 0.0232）。
    """
    anchor_it = warmup - 1
    out = []
    for it, kl in zip(its, kls):
        it0 = it - 1                       # 还原成 0-based，与主循环口径一致
        while rolling > 0 and (anchor_it + 1 + rolling) <= it0:
            anchor_it += rolling
        out.append(kl / max(1, it0 - anchor_it))
    return out


def replay(rates: list[float], target: float, kc0: float = DEFAULT_KC,
           relax_ratio: float = RELAX_RATIO) -> dict:
    """把实测**每轮率**序列喂给 A3 的规则走一遍（不动熵侧），看 kc 会怎样。"""
    kc = kc0
    fires = relax = 0
    traj = []
    for r in rates:
        if r > target:
            kc = min(kc * 1.6, KL_HI)
            fires += 1
        elif r < target * relax_ratio:
            kc = max(kc / 1.4, KL_LO)
            relax += 1
        kc = float(np.clip(kc, KL_LO, KL_HI))
        traj.append(kc)
    return {"target": target, "fires": fires, "relax": relax,
            "kc_end": kc, "kc_max": max(traj) if traj else kc,
            "peg_hi": (kc >= KL_HI - 1e-9)}


def suggest(r_max: float, r_med: float) -> tuple[float, str]:
    """按**每轮率**给候选值：常态不触发、坍缩必触发。

    目标必须**高于实测率的峰值**（否则"正常"就会持续触发），留 1.5 倍余量；
    再取向上最近的整档，便于写进配方。下限 0.02 防止速率趋零时目标退化。
    （本段是**无控制**对照 → 率偏高 → 建议值偏保守。）
    """
    need = max(0.02, 1.5 * r_max)
    for nice in (0.02, 0.03, 0.04, 0.05, 0.06, 0.075, 0.10, 0.15, 0.20, 0.30, 0.50):
        if nice >= need:
            return nice, (f"峰值率 {r_max:.5f}/轮 ×1.5 = {need:.5f}"
                          f"（中位率 {r_med:.5f}）→ 取向上最近的整档 {nice:.3f}")
    return 0.50, f"峰值率×1.5={need:.5f} 已超 0.5，取 0.5"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--history", type=pathlib.Path, help="训练 history.json")
    ap.add_argument("--log", type=pathlib.Path, help="训练 .out 日志（二选一）")
    ap.add_argument("--kc0", type=float, default=DEFAULT_KC)
    ap.add_argument("--anchor-warmup", type=int, default=0,
                    help="建锚所在的轮次（--anchor-warmup；0 = 锚就是 --init）")
    ap.add_argument("--anchor-rolling", type=int, default=150,
                    help="锚点 EMA 刷新周期（--kl-anchor-rolling；0 = 从不刷新）")
    ap.add_argument("--candidates", type=str,
                    default="0.02,0.03,0.05,0.075,0.10,0.15,0.20,0.30,0.50")
    args = ap.parse_args()
    src = args.history or args.log
    if src is None or not src.exists():
        sys.exit(f"✗ 请给出存在的 --history 或 --log（收到：{src}）")

    its, kls = load_series(src)
    if len(kls) < 3:
        sys.exit(f"✗ {src} 里只找到 {len(kls)} 次有效 KL 测量，样本太少")
    arr = np.asarray(kls, dtype=np.float64)
    print(f"数据源：{src}")
    print(f"有效 KL 测量 {len(arr)} 次（iter {its[0]} … {its[-1]}）\n")

    print("KL 实测序列（测量点 → 距离；Δ 为相邻测量间增量）")
    prev = None
    for it, v in zip(its, kls):
        d = "" if prev is None else f"      Δ {v - prev:+.4f}"
        print(f"   iter {it:>4}   {v:.4f}{d}")
        prev = v

    rts = rates_from(its, kls, args.anchor_rolling, args.anchor_warmup)
    ra = np.asarray(rts, dtype=np.float64)
    print(f"\n距离口径分位：min={arr.min():.4f}  中位={np.median(arr):.4f}  max={arr.max():.4f}")
    print("★ 换算成**每轮漂移率**（kl ÷ 锚点年龄；锚点每 "
          f"{args.anchor_rolling} 轮刷新，建锚于第 {args.anchor_warmup} 轮）")
    for it, r in zip(its, rts):
        print(f"   iter {it:>4}   年龄 {max(1, it - args.anchor_warmup):>4}   "
              f"率 {r:.5f}/轮")
    print(f"\n率分位：min={ra.min():.5f}  中位={np.median(ra):.5f}  "
          f"p90={np.percentile(ra, 90):.5f}  max={ra.max():.5f}")
    if len(ra) > 1 and (np.diff(ra) > 0).all():
        print("（率仍在单调上升 → 段末可能更高，建议值偏保守）")

    sug, why = suggest(float(ra.max()), float(np.median(ra)))
    print(f"\n反事实重放（起点 kc={args.kc0}，只跑 A3 的规则；判据 = 每轮率）")
    print("  target   触发  放松   kc终值   kc峰值  峰值/上限   判读")
    print("  " + "-" * 74)
    for c in [float(x) for x in args.candidates.split(",") if x.strip()]:
        r = replay(rts, c, args.kc0)
        if r["peg_hi"]:
            verdict = "✗ 撞上限（过度锚定，会把学习锁死）"
        elif r["fires"] > len(kls) * 0.5:
            verdict = "△ 触发过频（把常态当危险）"
        elif r["fires"] == 0 and r["relax"] == 0:
            verdict = "✓ 安全网语义：常态静默、kc 恒定（= 历史值，少变更一个变量）"
        elif r["fires"] == 0:
            verdict = "△ 常态偏松：kc 会被缓慢衰减到下限（静默改动基线）"
        else:
            verdict = "✓ 安全网语义：偶尔刹车"
        print(f"  {c:<7.2f}  {r['fires']:>3}   {r['relax']:>3}   "
              f"{r['kc_end']:.4f}  {r['kc_max']:.4f}  {r['kc_max']/KL_HI:>7.1%}   {verdict}")
    print(f"\n建议 --kl-target {sug:.3f}    （{why}）")
    print("注：本段是**无 KL 控制**的对照，率会偏高；有控制之后实测会更低，")
    print("    所以按「峰值率×1.5」取是保守方向 —— 宁可少触发、不可撞上限。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
