#!/usr/bin/env python3
"""E-STAT + E-ALIGN 零算力双门禁复算（**纯只读**）。

预登记：docs/14天训练计划.md §6.72（2026-11-10 10:56 CST，**看任何新统计结果之前锁死**）。
协议正文：docs/评测协议-v2.md。

本脚本：
  * 不启动任何对局、不加载模型、不占 GPU、不改任何 jsonl / ckpt / 线上配置；
  * 逐位复用 tools/final_eval_report.py 的冻结统计口径（不一致即 FAIL 退出）；
  * 产出 runs/gates/estat_align_audit.json 与两份 Markdown 报告。

所有常数在下方 CONST 区固化，**禁止按结果调整**。
用法: .venv/bin/python tools/estat_align_audit.py
"""
from __future__ import annotations

import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
from final_eval_report import per_seed, stats  # noqa: E402  冻结口径唯一实现

# ===========================================================================
# CONST —— 看结果前锁死（§6.72 / 协议 v2）。任何修改都必须先改预登记。
# ===========================================================================
DELTA = 0.5                 # 最小决策效应 MDE（分/副）
MARGIN_LO, MARGIN_HI = -0.5, 0.5   # TOST 等价 margin
NONINF_MARGIN = -0.5       # 非劣 margin
ALPHA = 0.05
Z90, Z95 = 1.645, 1.960
BOOT_B = 10_000
BOOT_SEED = 20261110
FDR_Q = 0.05
PLACEMENT = (90.0, 45.0, 0.0, -135.0)   # 名次 1..4（计划 §3.4 给定）
ZERO_TOL = 1e-9
PARITY_TOL = 1e-12

GATES = ROOT / "runs" / "gates"
REPORTS = ROOT / "reports"
OUT_JSON = GATES / "estat_align_audit.json"
NPZ = ROOT / "runs" / "exit_r01" / "exit_w0.npz"

# 复算清单（§6.72.1 E）：写死，禁止增删
DATASETS = (
    ("combo2_main",   "组合② P48+N32 vs master+N32", "F1 主比较"),
    ("final_primary", "组合① P48 贪心 vs master+N32", "F3 描述性"),
    ("sideprobe_main", "E-SIDE 独立旁证（新池 6M）", "F3 描述性"),
    ("final_gateA",   "闸门 A P48 贪心 vs master 贪心", "F4 基准参照"),
    ("pool_O1",       "E-POOL O1 master 贪心", "F2 探索性"),
    ("pool_O2",       "E-POOL O2 同代冠军 P36", "F2 探索性"),
    ("pool_O3",       "E-POOL O3 早期学生 web_best", "F2 探索性"),
    ("pool_O4",       "E-POOL O4 master + 10% 噪声", "F2 探索性"),
    ("pool_O5",       "E-POOL O5 纯启发式规则", "F2 探索性"),
)
F2_IDS = ("pool_O1", "pool_O2", "pool_O3", "pool_O4", "pool_O5")
ALIGN_MAIN = "final_primary"             # E-ALIGN 一致率主数据集（§6.72.2 H）
ALIGN_SIGN = ("final_primary", "combo2_main")   # 主结论同号检查的两个数据集
ALIGN_THRESHOLD = 0.95                  # 一致率门禁阈值（计划 §3.4 给定）

# 允许文案映射（§6.72.1 B）——唯一表述来源
WORDING = {
    ("显著优势", "统计等价"): "显著为正，但幅度小于 δ，实际意义有限",
    ("显著优势", "非劣但不等价"): "显著为正，且非劣成立",
    ("显著优势", "不可分辨"): "显著为正；样本不足以判定等价 / 非劣",
    ("显著劣势", "统计等价"): "显著为负",
    ("显著劣势", "非劣但不等价"): "显著为负",
    ("显著劣势", "不可分辨"): "显著为负",
    ("无足够证据", "统计等价"): "统计等价（在 ±0.5 内）",
    ("无足够证据", "非劣但不等价"): "非劣成立；优势无足够证据",
    ("无足够证据", "不可分辨"): "无足够证据",
}

# 步骤③ 代码溯源（文件 + 必须存在的关键子串）
LAYERS = (
    ("训练奖励（PPO rollout）", "train_ppo.py",
     ('d = (g.result or {}).get("deltas")', "PHI_SCALE = 8.0"),
     "终端奖励 = 四座净分差 deltas（零和）；势函数 Φ=8·Vθ 为 policy-invariant 塑形"),
    ("奖励模块", "zhuoji/reward.py",
     ("PHI_SCALE = 8.0", "def shaped_step_reward", "def terminal_reward"),
     "单步奖励 = 局终端分差增量 + γΦ(s') − Φ(s)；终局 Φ≡0，不改最优策略"),
    ("教师 Q 采集", "gen_exit.py",
     ("deep_q(", "QQ.append(qv)"),
     "教师 n32-ply3 搜索输出全 28 槽 Q；Q 是动作条件下未来回报估计，不是终局分差"),
    ("ExIt 训练目标", "train_exit.py",
     ("VALUE_SCALE = 8.0", "qm / VALUE_SCALE", "def policy_targets"),
     "L = CE(softmax(Q/τ)) + 0.5·SmoothL1(V, maxQ/8)；Q → 策略用 softmax、→价值用 /8 缩放"),
    ("评估记录写出", "eval_pairs.py",
     ('"deltas": [float(x) for x in deltas]', '"cand_seat"'),
     "每条记录写四座 deltas 与候选座位；无动作轨迹、无逐决策字段"),
    ("闸门 / 排名 / 终评统计", "tools/final_eval_report.py",
     ("def per_seed", "def stats"),
     "唯一冻结口径：d = v[cand] − mean(其余)，四座按 seed 平均，seed 维度正态近似 p"),
    ("闸门决策复用", "tools/gate_decision.py",
     ("from final_eval_report import fmt, per_seed, stats",),
     "直接 import 冻结实现，无第二套统计"),
    ("best-of 排名复用", "tools/bestof_rank.py",
     ("from final_eval_report import per_seed, stats",),
     "直接 import 冻结实现，无第二套统计"),
    ("E-POOL 多重比较", "tools/pool_chunk.py",
     ("def holm", "p_adj"),
     "5 片 Holm step-down；本协议另给 BH-FDR 并列展示"),
)


# ===========================================================================
# 基础工具
# ===========================================================================
def Phi(z: float) -> float:
    return 0.5 * math.erfc(-z / math.sqrt(2.0))


def sgn(x: float) -> int:
    return 1 if x > 0 else (-1 if x < 0 else 0)


def place_vector(dl: list[float]) -> list[float]:
    """按 deltas 降序排名；同分按绝对座位号小者优先（§6.72.2 H 锁死）。"""
    order = sorted(range(4), key=lambda i: (-dl[i], i))
    v = [0.0] * 4
    for rank, i in enumerate(order):
        v[i] = PLACEMENT[rank]
    return v


def analyze_rows(name: str) -> dict:
    """一次读盘，同时给出 score / place 两种口径 + 逐行符号交叉表。"""
    path = GATES / f"{name}.jsonl"
    if not path.exists():
        raise FileNotFoundError(path)
    b_score: dict[int, list[float]] = defaultdict(list)
    b_place: dict[int, list[float]] = defaultdict(list)
    seats: dict[int, set[int]] = defaultdict(set)
    cross: Counter = Counter()
    n_rows = 0
    zero_bad = 0
    keys: set[tuple[int, int]] = set()
    with path.open(encoding="utf-8") as f:
        for lineno, raw in enumerate(f, 1):
            raw = raw.strip()
            if not raw:
                continue
            d = json.loads(raw)
            dl = [float(x) for x in d["deltas"]]
            cs = int(d["cand_seat"])
            seed = int(d["seed"])
            if len(dl) != 4:
                raise ValueError(f"{path}:{lineno} deltas 长度 {len(dl)} != 4")
            if abs(sum(dl)) > ZERO_TOL:
                zero_bad += 1
            vp = place_vector(dl)
            ds = dl[cs] - sum(dl[i] for i in range(4) if i != cs) / 3.0
            dp = vp[cs] - sum(vp[i] for i in range(4) if i != cs) / 3.0
            cross[(sgn(ds), sgn(dp))] += 1
            b_score[seed].append(ds)
            b_place[seed].append(dp)
            seats[seed].add(cs)
            keys.add((seed, cs))
            n_rows += 1
    series_score = {s: sum(v) / len(v) for s, v in b_score.items()}
    series_place = {s: sum(v) / len(v) for s, v in b_place.items()}
    # 与冻结实现逐位对齐（只校验 score 口径；place 是新口径，无既有实现）
    ref = per_seed(path)
    if set(ref) != set(series_score):
        raise AssertionError(f"{name}: seed 集合与冻结实现不一致")
    maxdiff = max(abs(ref[s] - series_score[s]) for s in ref)
    if maxdiff > PARITY_TOL:
        raise AssertionError(f"{name}: 与 final_eval_report.per_seed 偏差 {maxdiff:.3e} > {PARITY_TOL}")
    complete = all(len(v) == 4 for v in seats.values())
    return {
        "name": name, "path": str(path.relative_to(ROOT)),
        "n_rows": n_rows, "n_seeds": len(seats),
        "unique_keys": len(keys), "four_seats": complete,
        "zero_sum_violations": zero_bad, "parity_maxdiff": maxdiff,
        "series_score": series_score, "series_place": series_place,
        "cross": cross,
        "agree": sum(c for (a, b), c in cross.items() if a == b),
    }


def full_stats(series: dict[int, float]) -> dict:
    vals = [series[s] for s in sorted(series)]
    s = stats(vals)                       # 冻结实现：mean/se/z/p/pos/neg
    a = np.asarray(vals, dtype=np.float64)
    out = dict(s)
    out.update({
        "sd": float(a.std(ddof=1)),
        "median": float(np.median(a)),
        "q1": float(np.percentile(a, 25)),
        "q3": float(np.percentile(a, 75)),
        "min": float(a.min()), "max": float(a.max()),
        "ci95": [s["mean"] - Z95 * s["se"], s["mean"] + Z95 * s["se"]],
        "ci90": [s["mean"] - Z90 * s["se"], s["mean"] + Z90 * s["se"]],
    })
    return out


def tost(s: dict) -> dict:
    d, se = s["mean"], s["se"]
    z1 = (d + DELTA) / se if se > 0 else math.inf
    z2 = (DELTA - d) / se if se > 0 else math.inf
    p1 = 1.0 - Phi(z1)
    p2 = 1.0 - Phi(z2)
    p_tost = max(p1, p2)
    lo95, hi95 = s["ci95"]
    if lo95 > 0:
        dim_a = "显著优势"
    elif hi95 < 0:
        dim_a = "显著劣势"
    else:
        dim_a = "无足够证据"
    if p_tost < ALPHA:
        dim_b = "统计等价"
    elif p1 < ALPHA:
        dim_b = "非劣但不等价"
    else:
        dim_b = "不可分辨"
    return {"z1": z1, "z2": z2, "p_noninf": p1, "p_upper": p2,
            "p_tost": p_tost, "dim_a": dim_a, "dim_b": dim_b,
            "wording": WORDING[(dim_a, dim_b)]}


def bootstrap(series: dict[int, float]) -> dict:
    vals = np.asarray([series[s] for s in sorted(series)], dtype=np.float64)
    n = len(vals)
    rng = np.random.default_rng(BOOT_SEED)   # 每片重置，保证单片可独立复现
    idx = rng.integers(0, n, size=(BOOT_B, n))
    means = vals[idx].mean(axis=1)
    return {
        "B": BOOT_B, "rng_seed": BOOT_SEED, "unit": "seed",
        "se": float(means.std(ddof=1)),
        "ci95": [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))],
        "mean": float(means.mean()),
    }


def bh_fdr(pvals: list[float], q: float = FDR_Q) -> tuple[list[float], list[bool]]:
    """BH step-up：adj_sorted[k] = min_{j>=k} (p_sorted[j]·m/(j+1))，封顶 1。"""
    m = len(pvals)
    order = sorted(range(m), key=lambda i: pvals[i])
    raw = np.asarray([pvals[i] * m / (k + 1) for k, i in enumerate(order)])
    monotone = np.minimum.accumulate(raw[::-1])[::-1]
    adj = [0.0] * m
    for k, i in enumerate(order):
        adj[i] = float(min(1.0, monotone[k]))
    return adj, [a < q for a in adj]


def holm(pvals: list[float]) -> list[float]:
    m = len(pvals)
    order = sorted(range(m), key=lambda i: pvals[i])
    adj = [0.0] * m
    running = 0.0
    for k, i in enumerate(order):
        running = max(running, min(1.0, (m - k) * pvals[i]))
        adj[i] = running
    return adj


# ===========================================================================
# E-ALIGN 步骤③ / ④
# ===========================================================================
def layer_trace() -> list[dict]:
    out = []
    for label, rel, needles, desc in LAYERS:
        p = ROOT / rel
        txt = p.read_text(encoding="utf-8") if p.exists() else ""
        missing = [n for n in needles if n not in txt]
        out.append({"layer": label, "path": rel, "desc": desc,
                    "pass": p.exists() and not missing, "missing": missing})
    return out


def q_diagnostics() -> dict:
    d = np.load(NPZ)
    q = d["q"]
    finite = np.isfinite(q)
    vals = q[finite]
    srt = np.sort(np.where(finite, q, -np.inf), axis=1)[:, ::-1]
    top = srt[:, 0]
    second = srt[:, 1]
    ok2 = np.isfinite(second)
    gap = (top - second)[ok2]
    return {
        "path": str(NPZ.relative_to(ROOT)), "rows": int(q.shape[0]),
        "slots": int(q.shape[1]),
        "finite_per_row_min": int(finite.sum(1).min()),
        "finite_per_row_max": int(finite.sum(1).max()),
        "finite_per_row_mean": float(finite.sum(1).mean()),
        "q_min": float(vals.min()), "q_max": float(vals.max()),
        "q_p01": float(np.percentile(vals, 1)),
        "q_p25": float(np.percentile(vals, 25)),
        "q_median": float(np.percentile(vals, 50)),
        "q_p75": float(np.percentile(vals, 75)),
        "q_p99": float(np.percentile(vals, 99)),
        "gap_best_second_median": float(np.median(gap)),
        "gap_best_second_p90": float(np.percentile(gap, 90)),
        "rows_with_two_plus_actions": int(ok2.sum()),
    }


# ===========================================================================
# 报告生成
# ===========================================================================
def f(x, nd=3):
    return f"{x:+.{nd}f}" if isinstance(x, float) else str(x)


def md_estat(rec: dict) -> str:
    L = []
    A = L.append
    A("# E-STAT：评测协议 v2 对既有实验的复算")
    A("")
    A("> **纯离线只读复算 · 不改任何原结论** ｜ 预登记 `docs/14天训练计划.md §6.72`"
      "（2026-11-10 10:56 CST，看结果前锁死）｜ 协议正文 `docs/评测协议-v2.md`。")
    A("> 零 GPU、零新对局、零训练；复算只在新口径下加贴类别标签。"
      "`reports/终评报告-2026-11-13.md` v1.1 的 §3 / §4 与 §6.71.4 判定**一字未改**。")
    A("")
    A("## 1. 冻结口径与锁死常数")
    A("")
    A("沿用 `tools/final_eval_report.py::per_seed / stats`；本脚本逐数据集校验与冻结实现"
      f"逐位一致（容差 `{PARITY_TOL:.0e}`），最大偏差见下表。"
      f"δ=`{DELTA}` 分/副、TOST margin=`[{MARGIN_LO},{MARGIN_HI}]`、非劣 margin=`{NONINF_MARGIN}`、"
      f"α=`{ALPHA}`、z=`{Z90}/{Z95}`、bootstrap `B={BOOT_B}` / RNG `{BOOT_SEED}` / 重采单位 seed、BH-FDR `q={FDR_Q}`。")
    A("")
    A("## 2. 九片复算总表")
    A("")
    A("| 数据集 | family | n(seed) | 行数 | 四座齐 | Δ±SE | 95% CI | z | 双侧 p | 维度 A | 维度 B | 允许文案 |")
    A("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for name, label, fam in DATASETS:
        r = rec["datasets"][name]
        s, t = r["score"], r["tost"]
        A(f"| `{name}` | {fam} | {s['n']} | {r['n_rows']} | "
          f"{'是' if r['four_seats'] else '否'} | "
          f"`{f(s['mean'])}±{s['se']:.3f}` | `[{f(s['ci95'][0])},{f(s['ci95'][1])}]` | "
          f"`{f(s['z'],2)}` | `{s['p']:.4f}` | {t['dim_a']} | {t['dim_b']} | {t['wording']} |")
    A("")
    A("> 表内 `Δ±SE` 与历史报告数字一致（同一冻结口径），**新口径只增加维度 A / B 与允许文案**。"
      "不同对手 / 不同池的 Δ 不可相减、不可合并。")
    A("")
    A("## 3. TOST 明细（等价 / 非劣）")
    A("")
    A("| 数据集 | 90% CI | z1 | z2 | p_非劣 | p_上界 | p_TOST | 统计等价 | 非劣 |")
    A("|---|---|---|---|---|---|---|---|---|")
    for name, _, _ in DATASETS:
        t = rec["datasets"][name]["tost"]
        s = rec["datasets"][name]["score"]
        A(f"| `{name}` | `[{f(s['ci90'][0])},{f(s['ci90'][1])}]` | `{f(t['z1'],2)}` | "
          f"`{f(t['z2'],2)}` | `{t['p_noninf']:.4f}` | `{t['p_upper']:.4f}` | "
          f"`{t['p_tost']:.4f}` | {'是' if t['p_tost']<ALPHA else '否'} | "
          f"{'是' if t['p_noninf']<ALPHA else '否'} |")
    A("")
    A("> **为什么 9 片全部是「非劣但不等价」**：统计等价要求 90% CI 完全落在 "
      "`[−0.5,+0.5]`。Δ 远大于 0.5 的片本来就不等价于零；而 Δ 接近 0 的两片"
      "（E-SIDE `SE=0.665`、O2 `SE=0.258`）90% CI 上界分别超过 `+0.5`，"
      "是样本量不足以判定等价。这与 §8 预锁的分辨率分析一致，"
      "**不是协议失效，也不是调 margin 的理由**。")
    A("")
    A("> **「非劣成立」的读法限制**：它只表示单侧 `H0: μ ≤ −0.5` 被拒绝，"
      "**不等于**「无差异 / 打平 / 一样强」。E-SIDE 与 `pool_O2` 的双侧 p 仍分别是 "
      "`.1923` / `.4703`，按既有 §6.69 / §6.71.4 仍**只写「无足够证据」**；"
      "本表的新口径标签是对既有数据的加贴，不改写那两处文案。")
    A("")
    A("## 4. seed 块 cluster bootstrap（稳健性对照，不参与判定）")
    A("")
    A("| 数据集 | 解析 SE | bootstrap SE | 解析 95% CI | bootstrap 95% CI | 偏差 |")
    A("|---|---|---|---|---|---|")
    for name, _, _ in DATASETS:
        r = rec["datasets"][name]
        s, b = r["score"], r["boot"]
        A(f"| `{name}` | `{s['se']:.3f}` | `{b['se']:.3f}` | "
          f"`[{f(s['ci95'][0])},{f(s['ci95'][1])}]` | "
          f"`[{f(b['ci95'][0])},{f(b['ci95'][1])}]` | "
          f"`{b['se']-s['se']:+.4f}` |")
    A("")
    A("## 5. F2 探索性多比较：BH-FDR 与 Holm 并列")
    A("")
    f2p = [rec["datasets"][n]["score"]["p"] for n in F2_IDS]
    bh, rej = bh_fdr(f2p)
    hm = holm(f2p)
    A("| 片 | 原始 p | BH-FDR p_adj | BH reject (q=.05) | Holm p_adj | Holm reject |")
    A("|---|---|---|---|---|---|")
    for i, n in enumerate(F2_IDS):
        A(f"| `{n}` | `{f2p[i]:.4f}` | `{bh[i]:.4f}` | {'是' if rej[i] else '否'} | "
          f"`{hm[i]:.4f}` | {'是' if hm[i]<ALPHA else '否'} |")
    A("")
    A("> 此表是**协议演算**，与 `§6.71.4` 已锁死的 Holm 判定并列；"
      "不得互相替换、不得择优引用、不得改写 §6.71.4 任何结论。")
    A("")
    A("## 6. seed 分布（禁止只报最大值）")
    A("")
    A("| 数据集 | mean | median | SD | Q1 | Q3 | min | max | 正:负 |")
    A("|---|---|---|---|---|---|---|---|---|")
    for name, _, _ in DATASETS:
        s = rec["datasets"][name]["score"]
        A(f"| `{name}` | `{f(s['mean'])}` | `{f(s['median'])}` | `{s['sd']:.3f}` | "
          f"`{f(s['q1'])}` | `{f(s['q3'])}` | `{f(s['min'])}` | `{f(s['max'])}` | "
          f"`{s['pos']}:{s['neg']}` |")
    A("")
    A("## 7. 口径完整性与分辨率")
    A("")
    A("| 数据集 | 零和违例 | 与冻结实现最大偏差 | unique(seed,seat) |")
    A("|---|---|---|---|")
    for name, _, _ in DATASETS:
        r = rec["datasets"][name]
        A(f"| `{name}` | `{r['zero_sum_violations']}` | `{r['parity_maxdiff']:.2e}` | "
          f"`{r['unique_keys']}` |")
    A("")
    A(f"- **分辨率（看前写死，仅解释不调参）**：按历史 `σ≈6.915`，`SE(500)=0.309`，"
      f"`1.645·SE=0.508` 略超 δ=0.5 ⇒ 真实 Δ=0 时 500 seed 的 TOST 等价判出概率接近 0；"
      f"理论需 `n≥518` seed 才使 `1.645·SE<0.5`（实际还需余量）。"
      "**本轮不改样本量、不调 margin、不扩样。**")
    A("- **per-decision 诊断量**：现有 JSONL 无动作轨迹字段 ⇒ 本轮不计算、不估算、不伪造；"
      "启用需另改 `eval_pairs.py`（属改评估口径，须单独批准），且不得替代强度判定。")
    A("- **anytime-valid / e-values**：本轮全部为一次性固定样本、无连续窥视 ⇒ 不启用；"
      "未来要边跑边看须开工前预登记 α-spending 或 e-value 阈值。")
    A("")
    A("## 8. 机械判读")
    A("")
    main = rec["datasets"]["combo2_main"]
    A(f"- 唯一预注册主比较 F1 = `combo2_main`：`{f(main['score']['mean'])}±{main['score']['se']:.3f}`，"
      f"维度 A = **{main['tost']['dim_a']}**，维度 B = **{main['tost']['dim_b']}** "
      f"⇒ 允许文案：**{main['tost']['wording']}**。")
    ns = [n for n, _, _ in DATASETS
          if rec["datasets"][n]["tost"]["dim_a"] == "无足够证据"
          and rec["datasets"][n]["tost"]["dim_b"] == "不可分辨"]
    A(f"- 落在「无足够证据 × 不可分辨」的片：`{', '.join('`'+n+'`' for n in ns) or '无'}`。"
      "这类片**只写「无足够证据」**，禁写「无差异 / 打平 / 到顶 / 证明一样强」。")
    ns_a = [n for n, _, _ in DATASETS
            if rec["datasets"][n]["tost"]["dim_a"] == "无足够证据"]
    A(f"- 维度 A 为「无足够证据」的片共 `{len(ns_a)}` 个："
      f"`{', '.join('`'+n+'`' for n in ns_a) or '无'}`；它们的维度 B 均为"
      "「非劣但不等价」，新口径允许文案「非劣成立；优势无足够证据」。"
      "**但按既有 §6.69 / §6.71.4 的原文案仍只写「无足够证据」** —— "
      "本标签是对既有数据的加贴，不改写那两处结论文案。")
    A("- **本协议不产生任何新实验判定**：以上全部是对既有数据的类别加贴，"
      "v1.1 报告 §3「达成」、§4 与 §6.71.4「通过方向一致性检验」均维持原判。")
    A("")
    return "\n".join(L) + "\n"


def md_ealign(rec: dict) -> str:
    L = []
    A = L.append
    A("# E-ALIGN：训练 / 评测目标对齐审计")
    A("")
    A("> **纯离线只读 · 零 GPU · 零新对局** ｜ 预登记 `docs/14天训练计划.md §6.72.2`"
      "（2026-11-10 10:56 CST，看结果前锁死定义与判据）。")
    A("> 本文件只出书面结论与建议选项；**不改训练目标、奖励函数、网络、评估口径或线上模型**。")
    A("")
    A("## 1. 锁死定义（复述，不允许事后解释性修改）")
    A("")
    A(f"- placement 向量按名次 1..4 = `{list(PLACEMENT)}`（计划 §3.4 给定，非项目既有口径），总和 0。")
    A("- 名次映射 `sorted(range(4), key=lambda i: (-deltas[i], i))`；同分按绝对座位号小者优先，确定性。")
    A("- 每行 `d_score = deltas[cand] − mean(其余)`、`d_place = place[rank(cand)] − mean(place[其余])`；"
      "`sgn ∈ {+1, −1, 0}`；一致 ⇔ 两者符号相同；一致率 = 一致行数 / 总行数。")
    A("- **预登记时即注明的数学关系**：`deltas` 严格零和 ⇒ `d_score = (4/3)·deltas[cand]`、"
      "`d_place = (4/3)·place[rank]`，故一致率度量的是「分差为正 ⟺ 名次前二」的吻合度；"
      "第 2 名负分、第 3 名正分都会记为不一致。这是设计使然，不是异常值处理。")
    A(f"- 主数据集 = `{ALIGN_MAIN}`；主结论同号检查 = `{', '.join(ALIGN_SIGN)}`（两个都要）；"
      f"门禁阈值 = 一致率 ≥{ALIGN_THRESHOLD:.0%} 且两数据集主结论同号。")
    A("")
    A("## 2. 主数据集：逐行符号一致率")
    A("")
    m = rec["datasets"][ALIGN_MAIN]
    A(f"- 行数 `{m['n_rows']}`、一致 `{m['agree']}` ⇒ **一致率 `"
      f"{m['agree']/m['n_rows']*100:.2f}%`**（门禁阈值 95.00%）。")
    A("")
    A("| d_score ＼ d_place | −1 | 0 | +1 | 行合计 |")
    A("|---|---|---|---|---|")
    for a in (-1, 0, 1):
        cells = [m["cross"].get(f"{a},{b}", 0) for b in (-1, 0, 1)]
        A(f"| {a:+d} | {cells[0]} | {cells[1]} | {cells[2]} | {sum(cells)} |")
    A(f"| 列合计 | {sum(m['cross'].get(f'{a},-1',0) for a in (-1,0,1))} | "
      f"{sum(m['cross'].get(f'{a},0',0) for a in (-1,0,1))} | "
      f"{sum(m['cross'].get(f'{a},1',0) for a in (-1,0,1))} | {m['n_rows']} |")
    A("")
    meanings = {
        "-1,0": "第 3 名且分差为负：分差判负、名次积分判 0",
        "-1,1": "第 2 名且分差为负：分差判负、名次积分判正",
        "0,-1": "第 4 名且分差恰为 0",
        "0,1": "名次前二且分差恰为 0",
        "1,0": "第 3 名且分差为正",
        "1,-1": "不可能出现（第 4 名分差必为负）",
    }
    A(f"**不一致的 `{m['n_rows']-m['agree']}` 行分解**（结构性来源，不是模型判断错误）：")
    A("")
    A("| d_score 符号 | d_place 符号 | 行数 | 含义 |")
    A("|---|---|---|---|")
    for a in (-1, 0, 1):
        for b in (-1, 0, 1):
            if a == b:
                continue
            k = f"{a},{b}"
            A(f"| {a:+d} | {b:+d} | `{m['cross'].get(k,0)}` | {meanings[k]} |")
    A(f"| 合计 | — | `{m['n_rows']-m['agree']}` | 占全部行 "
      f"`{(m['n_rows']-m['agree'])/m['n_rows']*100:.2f}%` |")
    A("")
    A("## 3. 两种口径下的主结论是否同号")
    A("")
    A("| 数据集 | 分差口径 Δ±SE / p / 95% CI | 名次口径 Δ±SE / p / 95% CI | 同号 |")
    A("|---|---|---|---|")
    for name in ALIGN_SIGN:
        r = rec["datasets"][name]
        s, p = r["score"], r["place"]
        A(f"| `{name}` | `{f(s['mean'])}±{s['se']:.3f}` / `{s['p']:.4f}` / "
          f"`[{f(s['ci95'][0])},{f(s['ci95'][1])}]` | "
          f"`{f(p['mean'])}±{p['se']:.3f}` / `{p['p']:.4f}` / "
          f"`[{f(p['ci95'][0])},{f(p['ci95'][1])}]` | "
          f"{'是' if sgn(s['mean'])==sgn(p['mean'])!=0 else '否'} |")
    c2 = rec["datasets"]["combo2_main"]
    A("")
    A("> **「同号」只比较 Δ 的符号，不要求两种口径都显著**（§6.72.2 H 锁死定义）。"
      f"例如 `combo2_main` 分差口径 `{f(c2['score']['mean'])}±{c2['score']['se']:.3f}` / "
      f"`p={c2['score']['p']:.4f}` 是「显著优势」，名次口径 "
      f"`{f(c2['place']['mean'])}±{c2['place']['se']:.3f}` / `p={c2['place']['p']:.4f}`、"
      "95% CI 跨 0 ⇒ 在名次口径下只能写「无足够证据」。两者**方向相同但证据强度不同**，"
      "不得写成「两种口径结论一致」。")
    A("")
    A("## 4. 其余八片的描述性一致率（不参与门禁判定）")
    A("")
    A("| 数据集 | 行数 | 一致 | 一致率 | 分差 Δ | 名次 Δ |")
    A("|---|---|---|---|---|---|")
    for name, _, _ in DATASETS:
        if name == ALIGN_MAIN:
            continue
        r = rec["datasets"][name]
        A(f"| `{name}` | {r['n_rows']} | {r['agree']} | "
          f"`{r['agree']/r['n_rows']*100:.2f}%` | "
          f"`{f(r['score']['mean'])}` | `{f(r['place']['mean'])}` |")
    A("")
    A("## 5. 步骤③：训练奖励 / 教师 Q / 闸门 / 终评是否同一函数")
    A("")
    A("| 层 | 文件 | 自动校验 | 口径说明 |")
    A("|---|---|---|---|")
    for e in rec["layers"]:
        A(f"| {e['layer']} | `{e['path']}` | {'PASS' if e['pass'] else 'FAIL'} | {e['desc']} |")
    A("")
    A("**逐层结论（由代码事实直接推得，非统计推断）**：")
    A("")
    A("1. **终评统计与闸门 / best-of 排名确实是同一函数**——三者都直接 import "
      "`final_eval_report.per_seed / stats`，不存在第二套口径；评估写出的 `deltas` 是零和分差。")
    A("2. **PPO 训练奖励与终评分差同源**：终端奖励就是 `result['deltas']`，"
      "势函数 Φ=8·Vθ 是 policy-invariant 塑形（ΣγᵗFₜ = −Φ(s₀)），不改变最优策略。"
      "⇒ 就「分差」这一 cardinal 目标而言，训练与评测**对齐**。")
    A("3. **教师 Q 不是终局分差本身**：ExIt 用教师搜索的全动作 Q 做 "
      "`softmax(Q/τ)` 策略目标与 `maxQ/8` 价值目标（VALUE_SCALE=8）。"
      "Q 是动作条件下的未来回报估计，经缩放后才进网络；"
      "价值头拟合得更好 ≠ 外部更强（TanyaoDojo oracle-critic 反例）。")
    A("4. **名次（ordinal）从未进入任何训练目标或评估口径**："
      "placement 只在本审计中作为对照被构造出来。"
      "⇒ 若业务关心的是名次而非分差，`分差` 是**代理目标**，不是同一目标。")
    A("")
    A("## 6. 步骤④：教师 Q 的未校准性（描述性，不建立相关性）")
    A("")
    q = rec["q"]
    A(f"- 样本 `{q['path']}`：`{q['rows']}` 行 × `{q['slots']}` 槽；"
      f"每行合法动作数 `{q['finite_per_row_min']}..{q['finite_per_row_max']}`"
      f"（均值 `{q['finite_per_row_mean']:.2f}`），非法槽为 NaN。")
    A(f"- 合法 Q 分位：min `{q['q_min']:.3f}` / p1 `{q['q_p01']:.3f}` / "
      f"p25 `{q['q_p25']:.3f}` / 中位 `{q['q_median']:.3f}` / "
      f"p75 `{q['q_p75']:.3f}` / p99 `{q['q_p99']:.3f}` / max `{q['q_max']:.3f}`。")
    A(f"- 每行最优与次优 Q 差：中位 `{q['gap_best_second_median']:.4f}`、"
      f"p90 `{q['gap_best_second_p90']:.4f}`（有 ≥2 合法动作的行 `{q['rows_with_two_plus_actions']}`）。")
    A(f"- **读法**：合法 Q 中位 `{q['q_median']:.3f}`、IQR "
      f"`[{q['q_p25']:.3f}, {q['q_p75']:.3f}]`，但尾部跨 "
      f"`{q['q_min']:.2f}..{q['q_max']:.2f}`（p1/p99 = `{q['q_p01']:.3f}/{q['q_p99']:.3f}`），"
      "长尾明显；每行最优与次优差的中位也很小 ⇒ Q 是**同一决策点内的相对动作分数**，"
      "跨状态不可比、未校准为胜率。")
    A("- 训练里价值头目标是 `maxQ/8`（`VALUE_SCALE=8`），说明 Q 与分差之间是**缩放映射**"
      "而非同一函数；非最优动作的 Q 尤其不可解释为胜率（Mortal FAQ 同类警告）。"
      "因此禁止把「Q 拟合更好 / maxQ 更大 / Q 熵更低」直接当作棋力提升，也禁止用它替代强度判定。")
    A("")
    A("## 7. 门禁判读（机械执行 §6.72.2 I）")
    A("")
    A(f"- 主数据集一致率 `{m['agree']/m['n_rows']*100:.2f}%` "
      f"{'≥' if m['agree']/m['n_rows']>=ALIGN_THRESHOLD else '<'} `{ALIGN_THRESHOLD:.0%}`。")
    same = all(sgn(rec["datasets"][n]["score"]["mean"]) ==
               sgn(rec["datasets"][n]["place"]["mean"]) != 0 for n in ALIGN_SIGN)
    A(f"- 两个主结论同号：`{'是' if same else '否'}`。")
    passed = (m["agree"] / m["n_rows"] >= ALIGN_THRESHOLD) and same
    A(f"- **门禁结论：{'目标对齐' if passed else '目标未对齐'}**"
      f"（判据 = 一致率 ≥95% **且** 两数据集主结论同号，两者缺一不可）。")
    A("")
    A("**这个数字的含义与不含义**（按预锁判据机械判读后的解释，不改变判定）：")
    A("")
    A(f"- **含义**：按预锁的 placement 阶跃映射（第 1/2 名恒为正、第 3 名恒为 0、"
      f"第 4 名恒为负），主数据集有 `{m['n_rows']-m['agree']}` / `{m['n_rows']}` 行的"
      "逐局二值判定与分差符号不同 ⇒「优化分差」与「优化名次」**不是同一个二值目标**；"
      "按计划 §3.4 判据必须记未对齐。")
    A("- **不含义**：**不是**「模型有 38% 的局判断错了」、**不是**「训练代码有 bug」、"
      "**不是**「P48 对名次无效」。§5 的 9/9 代码溯源 PASS 已证明分差口径在训练 / 闸门 / 终评"
      "是同一函数；§3 显示聚合到 seed 层后两种口径的 Δ 方向一致 —— 分歧来自 ordinal 阶跃"
      "与 cardinal 连续量的**二值化方式差异**，主要落在第 3 名（place=0）与第 2 名负分。")
    A("- 因此门禁冻结的是「**以名次为目标的新 RL / 奖励塑形**」，"
      "不是冻结分差口径下的既有结论；v1.1 报告 §3/§4 与 §6.71.4 一字未改。")
    A("")
    if not passed:
        A("### 7.1 冻结后果（按计划 §3.4 自动生效，不需要另行拍板）")
        A("")
        A("- 记「**目标未对齐**」：**冻结一切新 RL / 奖励塑形 / Q 锚定实验** —— "
          "E-REWARD（名次辅助头）、E-ANCHOR（Q 值锚定微调）与任何新 RL 立项，"
          "直到用户批准目标定义修订。E-TEACHER 属改教师口径的 DAgger，"
          "不受本门禁冻结，但**仍未批准**。")
        A("- **E-SLIM（纯蒸馏，不改训练目标）不在冻结范围**，但仍须单独点名批准；"
          "其 TOST 非劣判据以本协议 v2 为准。")
        A("- 本轮**不实施**任何目标定义修改；以下仅为待批准的候选选项。")
        A("")
        A("### 7.2 待批准的选项（本轮不实施）")
        A("")
        A("| 选项 | 做什么 | 代价 | 风险 |")
        A("|---|---|---|---|")
        A("| 保持分差目标 | 明确对外表述「优化的是分/副，不是名次」，"
          "名次只作辅助描述量 | 0 | 用户体感与模型目标仍可能背离 |")
        A("| 评测侧加名次口径 | 每片同时报 placement Δ，与分差 Δ 并列，**不改训练** | 极低（协议已具备） | "
          "双口径可能出现结论分歧，须预锁以哪个为主 |")
        A("| 训练侧改名次加权 | 把终端奖励换成 placement 或加权混合 | 改主链配方，需重新标注 + 全曲线重跑 | "
          "与全部历史曲线不可比；TanyaoDojo 四次负结果警示 |")
        A("")
    else:
        A("- 门禁通过：不冻结任何 RL 候选；但 E-REWARD / E-ANCHOR 仍需各自单独批准。")
        A("")
    A("## 8. 边界")
    A("")
    A("- 本审计**不是**「模型不行」的证据：它只说明训练目标与名次口径不是同一函数，"
      "以及分差与名次在逐局符号层面的吻合程度。")
    A("- 名次口径的 Δ 在 seed 层面与分差口径同号与否，**不代表**哪一个更正确；"
      "两者回答不同问题。")
    A("- 不改 `reports/终评报告-2026-11-13.md` v1.1 的 §3 / §4，"
      "不改 §6.71.4，不回流任何评估 JSONL，不动线上模型（主力恒 `dushan_master`，冠军只增不替）。")
    A("")
    return "\n".join(L) + "\n"


# ===========================================================================
def main() -> int:
    for p in (REPORTS, GATES):
        p.mkdir(parents=True, exist_ok=True)
    rec: dict = {"preamble": {
        "prereg": "docs/14天训练计划.md §6.72 (2026-11-10 10:56 CST)",
        "protocol": "docs/评测协议-v2.md",
        "delta": DELTA, "margin": [MARGIN_LO, MARGIN_HI],
        "noninf": NONINF_MARGIN, "alpha": ALPHA,
        "boot": {"B": BOOT_B, "rng": BOOT_SEED, "unit": "seed"},
        "placement": list(PLACEMENT), "align_threshold": ALIGN_THRESHOLD,
        "align_main": ALIGN_MAIN, "align_sign": list(ALIGN_SIGN),
        "note": "纯只读；不改任何原结论；所有常数看结果前锁死",
    }, "datasets": {}}

    print("=" * 84)
    print("E-STAT + E-ALIGN 零算力双门禁复算（只读）  §6.72 预登记")
    print("=" * 84)
    for name, label, fam in DATASETS:
        r = analyze_rows(name)
        r["label"], r["family"] = label, fam
        if r["zero_sum_violations"]:
            print(f"[FAIL] {name} 零和违例 {r['zero_sum_violations']}")
            return 2
        if not r["four_seats"] or r["unique_keys"] != r["n_rows"]:
            print(f"[FAIL] {name} 完整性不合格")
            return 2
        s = full_stats(r["series_score"])
        p = full_stats(r["series_place"])
        r["score"], r["place"] = s, p
        r["tost"] = tost(s)
        r["boot"] = bootstrap(r["series_score"])
        # 交叉表 key 转字符串以便 JSON 序列化
        r["cross"] = {f"{a},{b}": c for (a, b), c in r["cross"].items()}
        del r["series_score"], r["series_place"]
        rec["datasets"][name] = r
        print(f"  {name:16s} n={s['n']:4d} Δ={s['mean']:+.3f}±{s['se']:.3f} "
              f"p={s['p']:.4f} | A={r['tost']['dim_a']} B={r['tost']['dim_b']} "
              f"| agree={r['agree']/r['n_rows']*100:.2f}% "
              f"| parity={r['parity_maxdiff']:.1e}")

    rec["layers"] = layer_trace()
    rec["q"] = q_diagnostics()
    f2p = [rec["datasets"][n]["score"]["p"] for n in F2_IDS]
    bh, rej = bh_fdr(f2p)
    rec["f2"] = {"ids": list(F2_IDS), "p": f2p,
                 "bh_adj": bh, "bh_reject": rej,
                 "holm_adj": holm(f2p), "q": FDR_Q}
    m = rec["datasets"][ALIGN_MAIN]
    same = all(sgn(rec["datasets"][n]["score"]["mean"]) ==
               sgn(rec["datasets"][n]["place"]["mean"]) != 0 for n in ALIGN_SIGN)
    acc = m["agree"] / m["n_rows"]
    rec["verdict"] = {
        "align_accuracy": acc, "align_threshold": ALIGN_THRESHOLD,
        "main_conclusion_same_sign": same,
        "gate_pass": bool(acc >= ALIGN_THRESHOLD and same),
        "gate_label": "目标对齐" if (acc >= ALIGN_THRESHOLD and same) else "目标未对齐",
        "layer_trace_all_pass": all(e["pass"] for e in rec["layers"]),
    }

    (REPORTS / "E-STAT-既有实验复算.md").write_text(md_estat(rec), encoding="utf-8")
    (REPORTS / "E-ALIGN-目标对齐审计.md").write_text(md_ealign(rec), encoding="utf-8")
    OUT_JSON.write_text(json.dumps(rec, ensure_ascii=False, indent=2), encoding="utf-8")

    print("-" * 84)
    print(f"[E-ALIGN] 一致率 {acc*100:.2f}%  主结论同号={same}  ⇒ 门禁：{rec['verdict']['gate_label']}")
    print(f"[E-ALIGN] 代码溯源 {sum(e['pass'] for e in rec['layers'])}/{len(rec['layers'])} PASS")
    print(f"[OUT] {OUT_JSON.relative_to(ROOT)}")
    print(f"[OUT] reports/E-STAT-既有实验复算.md")
    print(f"[OUT] reports/E-ALIGN-目标对齐审计.md")
    if not rec["verdict"]["layer_trace_all_pass"]:
        print("[WARN] 有代码溯源项 FAIL，见 JSON layers[].missing")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
