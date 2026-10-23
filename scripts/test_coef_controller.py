# -*- coding: utf-8 -*-
"""`CoefController` 的行为回归测试（M2-A · A1 熵比例控制 + A3 KL 相对规则）。

为什么要有这个文件
------------------
控制器是整个 M2 阶段里**唯一"没有梯度可看"的部件** —— 它坏了不会 NaN、不会报错，
只会安静地把 ``ent_coef`` / ``kl_coef`` 钉在某个端点上，然后在 600 轮之后才通过
"熵塌了"或"学不动"暴露出来。2026-10-14 就抓到两个这种静默缺陷：

1. 熵基线窗口里 ``return`` 把 KL 分支一起吞了 → ``--anchor-warmup 0`` 的段前 25 轮
   KL 飙到目标 3 倍而 ``kc`` 一动不动（见 ``step()`` 里的 ★ 注释）。
2. ``--kl-target`` 的量级与 ``--kl-anchor-every`` 强耦合（降频把"每轮漂移"变成
   "4 轮累积漂移"）→ 目标定低了 kc 会在 25 轮内撞上限 0.5。

所以这里用**纯逻辑单测**（不 import torch，只把类源码抠出来 exec）把行为钉住。
运行：``python scripts/test_coef_controller.py``
"""

from __future__ import annotations

import math
import pathlib
import re
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
SRC = (ROOT / "scripts" / "selfplay.py").read_text(encoding="utf-8")
_m = re.search(r"^class CoefController:.*?(?=^def |^class )", SRC, re.S | re.M)
if not _m:
    sys.exit("✗ 在 scripts/selfplay.py 里找不到 CoefController")
_ns: dict = {"np": np, "math": math}
exec(_m.group(0), _ns)                     # noqa: S102 —— 只执行本仓库自己的类定义
CoefController = _ns["CoefController"]


class _Args:
    pass


def mk(**kw) -> _Args:
    a = _Args()
    a.ent_coef = 0.010
    a.kl_coef = 0.020
    a.kl_target = 0.05                    # 与 selfplay.py 的默认值保持一致（率口径）
    a.ent_target_ratio = CoefController.RATIO_DEFAULT   # 单一事实源（0.95）
    a.ent_eta = 4.0
    a.ent_deadzone = 0.03
    a.ent_baseline_iters = 25
    a.ent_baseline = 0.0
    a.anchor_warmup = 0
    for k, v in kw.items():
        setattr(a, k, v)
    return a


LOGS: list[str] = []
FAILS: list[str] = []


def check(name: str, cond: bool, detail: str) -> None:
    print(f"  {'PASS' if cond else 'FAIL'}  {name}  —  {detail}")
    if not cond:
        FAILS.append(name)


print("CoefController 行为回归（M2-A）\n" + "=" * 72)

# ── T1 · 熵基线窗口只冻结熵侧，KL 侧必须照动 ──────────────────────────────
# 这是 2026-10-14 抓到的真缺陷：旧写法在 len(ents)<n_base 时 return，
# 于是 --anchor-warmup 0 的段里 KL 严重超标也无人干预。
a = mk(ent_baseline_iters=25)
c = CoefController(a, log=LOGS.append)
for it in range(5):
    c.step(it, ent=1.10, kl=1.50)          # 基线窗口内 + KL 远超危险线
check("T1a 基线窗口内 ent_coef 冻结",
      abs(a.ent_coef - 0.010) < 1e-12, f"ec={a.ent_coef:.4f}（应恒 0.0100）")
check("T1b 基线窗口内 kl_coef 照常上调",
      a.kl_coef > 0.020, f"kc=0.020 → {a.kl_coef:.4f}（应 >0.0200）")

# ── T2 · 熵控制器双向对称（高于 H* 减、低于 H* 加） ───────────────────────
a = mk(ent_coef=0.060)
c = CoefController(a, log=LOGS.append)
for it in range(25):
    c.step(it, ent=1.20, kl=0.10)          # H0=1.20 → H*=0.90
h0 = c.h0
ec0 = a.ent_coef
for it in range(25, 35):
    c.step(it, ent=1.20, kl=0.10)
check("T2a 熵远高于 H* → ent_coef 下降",
      a.ent_coef < ec0,
      f"H0={h0:.3f} H*={a.ent_target_ratio*h0:.3f}  ec {ec0:.4f} → {a.ent_coef:.4f}")

a = mk(ent_coef=0.010)
c = CoefController(a, log=LOGS.append)
for it in range(25):
    c.step(it, ent=1.20, kl=0.10)
lo = a.ent_coef
for it in range(25, 45):
    c.step(it, ent=0.60, kl=0.10)          # 熵塌到远低于 H*=0.90
check("T2b 熵远低于 H* → ent_coef 上升",
      a.ent_coef > lo, f"ec {lo:.4f} → {a.ent_coef:.4f}")

# ── T3 · 死区：熵落在 H*±deadzone 内时系数不动 ────────────────────────────
a = mk(ent_coef=0.060)
c = CoefController(a, log=LOGS.append)
for it in range(25):
    c.step(it, ent=1.20, kl=0.10)
frozen = a.ent_coef
for it in range(25, 60):
    # 距 H* 仅 +0.02 < 死区 0.03（H* 随 ratio 变，所以这里必须跟着算，不能写死 0.75）
    c.step(it, ent=a.ent_target_ratio * 1.20 + 0.02, kl=0.10)
check("T3 死区内 ent_coef 不动",
      abs(a.ent_coef - frozen) < 1e-12, f"ec={a.ent_coef:.4f}（应恒 {frozen:.4f}）")

# ── T4 · KL 危险线用「每轮率」判定：距离再大也可能完全正常 ──────────────
# 这是第三个被静默吞掉的缺陷（2026-10-14）：裸 kl 是**距锚点的距离**，锚点 150 轮
# 才刷新一次，所以"距离"必然单调增大；实测每轮率却是稳的（0.023/0.031/0.028/0.025）。
# 用距离判据 → 每次测量都触发、25 轮撞上 clamp 上限 → 模型被锁在过期锚上。
a = mk(ent_coef=0.05, kl_target=0.05)
c = CoefController(a, log=LOGS.append)
# 冒烟实测（无控制，kc 冻结在 0.02）；注意喂 **0-based** it（日志里的 iter 是 1-based）
_kl_seq = [(0, 0.0232), (4, 0.1540), (8, 0.2564), (12, 0.3284)]
for it, kl in _kl_seq:
    c.step(it, ent=1.20, kl=kl)
check("T4a 距离增长但率稳定（0.023–0.031/轮）→ kc 不动",
      abs(a.kl_coef - 0.020) < 1e-12, f"kc={a.kl_coef:.4f}（末次率 {c.last_rate:.4f}/轮）")

a = mk(ent_coef=0.05, kl_target=0.05)
c = CoefController(a, log=LOGS.append)
for it in range(6):
    c.step(it, ent=1.20, kl=0.60 * (it + 1))    # 率恒为 0.60/轮（= 12× 危险线）
check("T4b 率远超危险线 → kc 迅速上调",
      a.kl_coef > 0.10, f"kc=0.020 → {a.kl_coef:.4f}（率 {c.last_rate:.3f}/轮）")

# ── T4c · 降频不变性：**每次测量的决策**必须与 cadence 无关 ──────────────
# 这是 A3 与 A7② 的**耦合点**：--kl-anchor-every 4 把测量值放大 4 倍。
# 率判据必须对 cadence 免疫，否则改任一个参数都要重新标定目标。
# 注意不变式是「同样次数测量 → 同样 kc」而**不是**「同样轮数 → 同样 kc」：
# 降频后控制器能调整的**次数**本来就少（every=4 只有 1/4 次机会），这是设计如此。
def _feed(every: int, rate: float, n_meas: int = 8, target: float = 0.05) -> float:
    a = mk(ent_coef=0.05, kl_target=target)
    c = CoefController(a, log=lambda _m: None)
    it = got = 0
    while got < n_meas:
        kl = rate * (it + 1) if (it + 1) % every == 0 else None
        c.step(it, ent=1.20, kl=kl)
        got += kl is not None
        it += 1
    return a.kl_coef


_kcs = {e: _feed(e, 0.030) for e in (1, 4, 10)}          # 率 0.03 ∈ 死区 → 都不该动
check("T4c 率在死区内：every=1/4/10 的决策一致（都不动）",
      max(_kcs.values()) - min(_kcs.values()) < 1e-12 and abs(min(_kcs.values()) - 0.02) < 1e-12,
      "  ".join(f"every={e}→kc={v:.4f}" for e, v in _kcs.items()))
_kcs2 = {e: _feed(e, 0.200) for e in (1, 4, 10)}          # 率 0.20 > 0.05 → 都该升
check("T4d 率超标：every=1/4/10 的决策同样一致（都升、且幅度相同）",
      max(_kcs2.values()) - min(_kcs2.values()) < 1e-12 and min(_kcs2.values()) > 0.02,
      "  ".join(f"every={e}→kc={v:.4f}" for e, v in _kcs2.items()))

# ── T4e · 滚动锚刷新后年龄归零 → 率不被"旧距离"污染 ──────────────────────
a = mk(ent_coef=0.05, kl_target=0.05)
c = CoefController(a, log=LOGS.append)
for it in range(20):
    c.step(it, ent=1.20, kl=0.030 * (it + 1))       # 年龄 = it+1 → 率恒 0.030
c.note_anchor(19)                                    # 第 19 轮结束刷新锚点
for it in range(20, 60):
    c.step(it, ent=1.20, kl=0.030 * (it - 19))       # 年龄从 1 重新起算 → 率仍 0.030
check("T4e 锚点刷新后率仍被正确还原（年龄归零）",
      abs(c.last_rate - 0.030) < 1e-9, f"率={c.last_rate:.4f}/轮（应 0.0300）")

# ── T5 · kl=None（A7② 降频跳过）必须完全不动 kc ───────────────────────────
a = mk(ent_coef=0.05, kl_coef=0.030)
c = CoefController(a, log=LOGS.append)
for it in range(40):
    c.step(it, ent=1.20, kl=None)
check("T5 kl=None 时 kc 恒定（不会被误当 KL=0 衰减到下限）",
      abs(a.kl_coef - 0.030) < 1e-12, f"kc={a.kl_coef:.4f}")

# ── T6 · clamp 上下限 ─────────────────────────────────────────────────────
a = mk(ent_coef=0.05, kl_target=0.01)
c = CoefController(a, log=LOGS.append)
for it in range(400):
    c.step(it, ent=1.20, kl=5.0)
check("T6a kc 不越过上限 0.5", a.kl_coef <= 0.5 + 1e-12, f"kc={a.kl_coef:.4f}")
a = mk(ent_coef=0.05, kl_target=2.0, kl_coef=0.03)
c = CoefController(a, log=LOGS.append)
for it in range(400):
    c.step(it, ent=1.20, kl=0.0)
check("T6b kc 不越过下限 0.01", a.kl_coef >= 0.01 - 1e-12, f"kc={a.kl_coef:.4f}")

# ── T7 · 播报限频（每 10 轮最多一行，且只在真的变了才打） ─────────────────
a = mk(ent_coef=0.05, kl_target=99.0)      # 只让熵侧动，隔离 kc 的播报
c = CoefController(a, log=LOGS.append)
LOGS.clear()
for it in range(60):
    c.step(it, ent=1.20, kl=0.10)          # 基线期不动 → 之后每轮减 ec
n_lines = len(LOGS)
check("T7 [coef] 播报限频生效",
      n_lines <= 8, f"60 轮产生 {n_lines} 行（应 ≤8，即最多每 10 轮一行）")

# ── T8 · 增益 η 的定量标定（玩具被控对象 + 扫描） ─────────────────────────
# 为什么需要这一步：η 是**比例控制的对数增益**，它决定"熵偏 δ 时每轮把 ec 乘多少"。
# 单看公式看不出量级对不对 —— 因为 ec 的可活动范围只有 [0.01, 0.5]（50 倍、4 个 e 折），
# η 稍大就会在两个端点之间**继电器式来回**，于是探索强度每几轮翻一次，
# 训练信号被注入一个高频噪声。必须用"被控对象 + 延迟"才能看出稳定域。
#
# 玩具被控对象（一阶 + 一轮延迟，与真实情形同构）：
#     H(t+1) = H_free(t) + k · ec(t)
#     H_free(t) = 1.30 − 0.003·t      ← 自然熵衰减（坍缩压力）
#     k = 8                            ← ec 的"有效性"（每 0.1 的 ec 抬 0.8 nat）
# 设计需求：持续的 δ=0.10（≈9% of H₀，已出死区）应在 **15–25 轮**内把 ec 推一个
# e 折（2.72×）—— 慢于被控对象的一轮延迟以免过冲，快于 600 轮段的时间尺度。
#     exp(η·0.10) = e^(1/20)  →  η ≈ 0.5
def _sim(eta: float, niters: int = 900) -> dict:
    a = mk(ent_coef=0.05, ent_eta=eta, kl_target=99.0)   # kl 侧不参与，只看熵
    c = CoefController(a, log=lambda _m: None)
    hs, ecs = [], []
    h_free = 1.30
    h = h_free + 8.0 * a.ent_coef
    for it in range(niters):
        c.step(it, ent=h, kl=0.10)
        h_free = 1.30 - 0.003 * it
        h = h_free + 8.0 * a.ent_coef
        hs.append(h)
        ecs.append(a.ent_coef)
    tail_h, tail_e = np.asarray(hs[-150:]), np.asarray(ecs[-150:])
    hstar = c.ratio * (c.h0 or 0.0)      # 用控制器的实际目标，别硬编码 0.75
    return {
        "eta": eta, "h0": c.h0, "hstar": hstar,
        "h_tail": float(tail_h.mean()), "h_err": float(tail_h.mean() - hstar),
        "ec_tail": float(tail_e.mean()), "ec_cv": float(tail_e.std() / max(tail_e.mean(), 1e-9)),
        "ec_min": float(np.min(ecs)), "ec_max": float(np.max(ecs)),
        "slam": float(np.mean([e <= 0.0101 or e >= 0.4999 for e in ecs])),
    }


print(f"\nT8 · η 增益扫描（900 轮玩具被控对象；H*={CoefController.RATIO_DEFAULT}·H₀）")
print("  η       H₀    H*     H末150   H偏     ec末150  ec变异  ec区间          钉端比例")
print("  " + "-" * 86)
_sweep = [_sim(e) for e in (0.25, 0.5, 1.0, 2.0, 4.0)]
for s in _sweep:
    print(f"  {s['eta']:<6.2f}  {s['h0']:.3f}  {s['hstar']:.3f}  {s['h_tail']:.3f}"
          f"   {s['h_err']:+.3f}  {s['ec_tail']:.4f}   {s['ec_cv']:.3f}"
          f"   [{s['ec_min']:.4f},{s['ec_max']:.4f}]  {s['slam']*100:5.1f}%")

_eta_default = float(getattr(CoefController, "ETA_DEFAULT", 0.5))
_s = next((x for x in _sweep if abs(x["eta"] - _eta_default) < 1e-9), None)
if _s is None:                              # 保底：默认值不在扫描点里就现算一次
    _s = _sim(_eta_default)
check(f"T8a 默认 η={_eta_default} 能把熵拉回 H*±0.06（稳态无静差）",
      abs(_s["h_err"]) <= 0.06, f"H末={_s['h_tail']:.3f} vs H*={_s['hstar']:.3f}（偏 {_s['h_err']:+.3f}）")
check(f"T8b 默认 η={_eta_default} 无继电器式震荡（ec 变异系数 ≤0.25）",
      _s["ec_cv"] <= 0.25, f"ec 末 150 轮 {_s['ec_tail']:.4f}±{_s['ec_cv']*_s['ec_tail']:.4f}（CV={_s['ec_cv']:.3f}）")
check(f"T8c 默认 η={_eta_default} 不长期钉在 clamp 端点（≤10% 轮次）",
      _s["slam"] <= 0.10, f"钉端比例 {_s['slam']*100:.1f}%")

# ── T9 · 可达性预判 + 贴死端点告警（C5 的直接产物）────────────────────────
# C5（2026-10-14 60 轮实测）：H 的自然稳态是段首的 **91.6%**，而 H* = 0.75·H₀。
# 于是 H > H* 恒成立 → exp(η(H*−H)) < 1 恒成立 → ec 只会往下调，而它一开始
# 就在下限 0.01 → **ec 全程静止，A1 事实空转**（与它要修的旧缺陷是同一类失效，
# 只是钉的位置从 0.03 换到 0.01）。
#
# 这一类故障**没有任何梯度信号**：不 NaN、不报错、损失曲线也正常，
# 只是"某个自适应的部件从没动过"。唯一的防线是在基线一确定就打可达性预判，
# 并区分两种端点状态：
#   ① H 高于 H* 且 ec 在下限  → 目标偏保守（不是故障，但要报"还要降多少"）
#   ② ec 在上限且 H 仍低于 H* → 增益不足/H* 不可达（**真故障**，必须显式告警）
LOGS.clear()
# ★ 这里**故意**用导致 C5 的那个 ratio（0.75）来复现现象 —— 回归用例要钉住的是
#   缺陷本身，不是当前默认值。若哪天有人把默认值调回去，这里仍会如实报"不可达"。
a9 = mk(ent_coef=0.01, ent_target_ratio=0.75)
c9 = CoefController(a9, log=LOGS.append)
for it in range(40):
    c9.step(it, ent=1.15 - 0.004 * it, kl=0.02)        # 熵自然衰减到 ~0.99
check("T9a 基线窗口结束时打出「可达性预判」",
      any("可达性预判" in L for L in LOGS),
      next((L[:100] for L in LOGS if "可达性预判" in L), "（无）"))
check("T9b 预判给出可执行的建议（ratio 需 ≳ 多少）",
      any("ratio 需" in L for L in LOGS), f"{len(LOGS)} 行日志")
check("T9c 目标不可达时 ec 静默贴在下限 —— 与 C5 现象一致",
      abs(a9.ent_coef - 0.01) < 1e-12, f"ec={a9.ent_coef:.4f}（全程未离开起点）")

LOGS.clear()
# 反向证明：同一条熵轨迹，只要换回**当前默认 ratio**，控制器就必须真的接管
# —— 这是"C5 已修"的直接证据，而不只是"告警能响"。
a9c = mk(ent_coef=0.01)                                # 默认 ratio（RATIO_DEFAULT）
c9c = CoefController(a9c, log=LOGS.append)
for it in range(40):
    c9c.step(it, ent=1.15 - 0.004 * it, kl=0.02)
check("T9g 改用默认 ratio 后同一轨迹下 ec 真的接管（C5 已修）",
      a9c.ent_coef > 0.0105,
      f"ratio={c9c.ratio:.2f} H*={c9c.ratio*(c9c.h0 or 0):.3f} → ec 0.0100 → {a9c.ent_coef:.4f}")

LOGS.clear()
a9b = mk(ent_coef=0.01, ent_target_ratio=3.0)           # H* 高到不可能达到
c9b = CoefController(a9b, log=LOGS.append)
for it in range(80):
    c9b.step(it, ent=1.0, kl=0.02)
_w = [L for L in LOGS if "贴死" in L]
check("T9d 熵奖励开到最大仍低于 H* 时**显式告警**（不静默）",
      bool(_w), _w[0][:110] if _w else "（无告警 —— 这正是 C5 无法被及时发现的原因）")
check("T9e 告警文本含「不可达」并点明 A1 失效",
      bool(_w) and "不可达" in _w[0] and "A1" in _w[0], f"{len(LOGS)} 行日志")
check("T9f 告警只打一次（不刷屏）",
      len(_w) <= 1, f"共 {len(_w)} 条")

print("=" * 72)
if FAILS:
    print(f"✗ {len(FAILS)} 项失败：" + ", ".join(FAILS))
    sys.exit(1)
print("✓ 全部通过")
print("\n[coef] 播报样例：")
for L in LOGS[:3] or ["（无）"]:
    print("   ", L)
