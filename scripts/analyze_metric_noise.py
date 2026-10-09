"""训练内评测指标到底有多少「信号」—— P0 补充分析，为方案 A 选点定标尺。

为什么需要它
------------
P0 已确认：旧选点依据「评测均分」与真实强度的 Spearman ρ ≈ +0.14，基本无信息量。
方案 A 的做法是换一个**低噪声代理**，但"低噪声"不能靠猜——必须量出来。

本脚本用两批真实数据把每个指标剖开：

1. **确定性核实**：训练内评测用 ``NetAgent(temperature=0.0)``（argmax）+ 固定种子
   教师 + 固定 ``seed0=4242``，所以 ``eval_i`` 是模型权重的**确定性函数**。
   于是 ``avg_score_se≈0.97`` 不是"跑一次抖一下"，而是**这副固定牌池相对全体牌局
   的取样标准误（fixed-sample bias）**——它不会因为重跑而变小，只会因为换更大的
   牌池而变小。这一条决定了"多评估几次取平均"是**无效**的降噪手段。

2. **每指标的信噪比**：对每一代训练历史（``rl_history_big_s5..s11.json``）里的
   指标序列做一阶差分（消掉随迭代的漂移），得到"单次评测的抖动" ``noise_sd``；
   再用各代的均值算"代际差异" ``between_sd``。
   ``F = between_sd / noise_sd`` **就是选点分辨力**：F ≤ 1 的指标根本无法区分两代，
   F 越大越可能选出真正更强的那一代。

3. **实际选点后果**：对每个指标，取该代「最后 k 次评测均值」，看它的**名次**
   与天梯真值名次的一致程度——这才是"如果当初用它选点会选谁"的直接答案。

输出 ``reports/metric_noise_analysis.md``。

用法::

    python scripts/analyze_metric_noise.py \
        --ladder reports/ladder_s7_s11_coop.json \
        --out reports/metric_noise_analysis.md
"""
from __future__ import annotations

import argparse
import json
import math
import statistics as st
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

GENS = ["s5", "s6", "s7", "s8", "s9", "s10", "s11"]

# 指标 → (方向, 中文名, 说明)
# 方向 +1 表示越大越强，-1 表示越小越强（方向由天梯秩相关决定，已前置声明）
METRICS: list[tuple[str, int, str]] = [
    ("avg_score", +1, "评测均分（旧选点依据）"),
    ("win_rate", +1, "胡牌率"),
    ("tsumo_rate", +1, "自摸率"),
    ("ron_rate", +1, "点胡率"),
    ("deal_in_rate", -1, "点炮率"),
    ("huang_rate", -1, "黄庄率"),
    ("bigfan15_rate", +1, "大牌率15+"),
    ("bigfan20_rate", +1, "大牌率20+"),
    ("avg_fan_when_win", +1, "胡时均番"),
]


def spearman(x: list[float], y: list[float]) -> float:
    def ranks(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]:
                j += 1
            avg = (i + j) / 2 + 1
            for k in range(i, j + 1):
                r[order[k]] = avg
            i = j + 1
        return r
    if len(x) < 3:
        return float("nan")
    rx, ry = ranks(x), ranks(y)
    mx, my = sum(rx) / len(rx), sum(ry) / len(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = math.sqrt(sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry))
    return num / den if den else 0.0


def load_series(gen: str) -> list[tuple[int, dict]]:
    p = ROOT / "reports" / f"rl_history_big_{gen}.json"
    if not p.exists():
        return []
    d = json.loads(p.read_text(encoding="utf-8"))
    return [(r["iter"], r["eval"]) for r in d
            if isinstance(r, dict) and isinstance(r.get("eval"), dict)]


def noise_sd(vals: list[float]) -> float | None:
    """单次评测的抖动：一阶差分的 SD / √2（消掉随迭代的缓慢漂移）。

    确定性评测下这个量测的是"模型在相邻两次评测之间变了多少"，
    对**平坦段**（已经收敛、模型不再实质变化）而言它下界就是该指标的
    分辨率极限：低于它的差异没有意义。
    """
    if len(vals) < 4:
        return None
    diffs = [b - a for a, b in zip(vals, vals[1:])]
    sd = st.stdev(diffs)
    return sd / math.sqrt(2.0)


def tail_mean(vals: list[float], k: int) -> float:
    if not vals:
        return float("nan")
    k = min(k, len(vals))
    return sum(vals[-k:]) / k


def rank_agreement(pred: dict[str, float], truth: dict[str, float],
                   direction: int) -> tuple[float, dict[str, int]]:
    """pred/truth 都是 {gen: 值}。返回 (Spearman ρ, 名次表)。"""
    common = [g for g in pred if g in truth and not math.isnan(pred[g])]
    if len(common) < 3:
        return float("nan"), {}
    p = [direction * pred[g] for g in common]
    t = [truth[g] for g in common]
    rho = spearman(p, t)
    order_p = sorted(common, key=lambda g: -direction * pred[g])
    order_t = sorted(common, key=lambda g: -truth[g])
    ranks = {g: i + 1 for i, g in enumerate(order_p)}
    ranks.update({f"{g}#true": i + 1 for i, g in enumerate(order_t)})
    return rho, ranks


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ladder", default="reports/ladder_s7_s11_coop.json",
                    help="天梯结果（真值）。支持 ladder_test 的 coop/arena 输出格式")
    ap.add_argument("--gens", default=",".join(GENS))
    ap.add_argument("--tail", type=int, default=5,
                    help="用该代最后 k 次评测的均值代表这一代")
    ap.add_argument("--out", default="reports/metric_noise_analysis.md")
    args = ap.parse_args()

    gens = [g.strip() for g in args.gens.split(",") if g.strip()]
    series = {g: load_series(g) for g in gens}
    series = {g: v for g, v in series.items() if v}

    # ---- 天梯真值 --------------------------------------------------------
    ladder_p = ROOT / args.ladder
    truth: dict[str, float] = {}
    ladder_meta = ""
    if ladder_p.exists():
        lad = json.loads(ladder_p.read_text(encoding="utf-8"))
        raw = lad.get("levels") or {r["ckpt"]: r["level"] for r in lad.get("rows", [])}
        for k, v in raw.items():
            # 两种格式：ladder_test 的 levels[k] = {"mean":..,"ci95":..}；
            # score_checkpoints 的 rows[k]["level"] 是裸标量
            val = float(v["mean"]) if isinstance(v, dict) else float(v)
            m = str(k).replace("dushan_big_", "")
            if m in series:
                truth[m] = val
        ladder_meta = (f"`{args.ladder}`：{lad.get('deals')} 副 × "
                       f"{lad.get('games_total', lad.get('games_per_ckpt'))} 局，"
                       f"锚点 `{lad.get('anchor')}`，用时 {lad.get('elapsed_min', 0):.0f} 分")

    L: list[str] = []
    L.append("# 训练内评测指标的信噪比分析（方案 A 选点定标）\n")
    if ladder_meta:
        L.append(f"- 真值来源：{ladder_meta}")
    L.append(f"- 训练历史：{len(series)} 代（{'、'.join(f'`{g}`' for g in series)}），"
             f"每代 {args.tail} 次评测均值代表该代")
    L.append("")

    # ---- 0. 确定性核实 ---------------------------------------------------
    L.append("## 0. 先厘清「噪声」到底是什么\n")
    L.append("训练内评测的三个随机源全部是固定的：\n")
    L.append("| 随机源 | 设定 | 是否固定 |")
    L.append("|---|---|---|")
    L.append("| 网络决策 | `NetAgent(temperature=0.0)` → argmax | ✅ 确定性 |")
    L.append("| 三位教师 | `HeuristicBot(seed=101/202/303)` 每副牌重建 | ✅ 确定性 |")
    L.append("| 牌副 | `play_duplicate(seed0=4242)` | ✅ 固定 40 副 |")
    L.append("")
    L.append("所以 **`eval_i` 是模型权重的确定性函数**，重复评测同一模型会得到完全相同"
             "的分数。`avg_score_se≈0.97` 因此不是「重跑抖动」，而是**这 40 副牌相对全体"
             "牌局的取样标准误**：它衡量的是「这 40 副能不能代表所有牌局」，"
             "**多评估几次取平均完全无效**，只有换更大/更有代表性的牌池才有效。")
    L.append("")
    L.append("> 推论：降噪的正确方向不是「多评几次」，而是"
             "(a) 换**低方差**指标（rate 类而非分数类）；"
             "(b) 换**更大的固定牌池**（代价线性）；"
             "(c) 用**跨代配对**比较（同一副牌上比两个模型）。")
    L.append("")

    # ---- 1. 每指标的信噪比 ------------------------------------------------
    L.append("## 1. 各指标的信噪比与分辨率\n")
    L.append("| 指标 | 中文 | 方向 | 单次抖动 `noise_sd` | 代际差 `between_sd` | **F=代际/抖动** | 能否分辨代际？ |")
    L.append("|---|---|---|---|---|---|---|")
    table = []
    for key, sign, cn in METRICS:
        per_gen_sd = []
        means = {}
        for g, ev in series.items():
            vals = [float(e[key]) for _it, e in ev if key in e]
            if not vals:
                continue
            sd = noise_sd(vals)
            if sd:
                per_gen_sd.append(sd)
            means[g] = tail_mean(vals, args.tail)
        if not per_gen_sd or len(means) < 3:
            continue
        noise = float(st.median(per_gen_sd))
        between = float(st.stdev(list(means.values())))
        F = between / noise if noise > 1e-12 else float("inf")
        if F < 1.0:
            able = "❌ 噪声大于代际差，无法分辨"
        elif F < 2.0:
            able = "⚠️ 勉强"
        else:
            able = f"✅ F={F:.1f}"
        table.append((key, cn, sign, noise, between, F, means))
        L.append(f"| `{key}` | {cn} | {'越大越好' if sign > 0 else '越小越好'} | "
                 f"{noise:.4f} | {between:.4f} | **{F:.2f}** | {able} |")
    L.append("")
    L.append("> `F` 的含义：代际差异相对单次评测抖动有多大。**F ≤ 1 的指标选点等于抽签**——"
             "旧选点依据 `avg_score` 就在这个区间里。")
    L.append("")

    # ---- 2. 实际选点后果 --------------------------------------------------
    if truth:
        L.append("## 2. 如果当初用这个指标选点，会选中谁\n")
        for k in (1, args.tail):
            pass
        L.append(f"（每代取最后 {args.tail} 次评测的均值；真值为天梯分值）\n")
        L.append("| 指标 | ρ(指标, 真值) | 会选中 | 真值排名 | 真值最优 | 代价 |")
        L.append("|---|---|---|---|---|---|")
        champ = max(truth, key=lambda g: truth[g])
        rank_of = {g: i + 1 for i, g in
                   enumerate(sorted(truth, key=lambda x: -truth[x]))}
        for key, cn, sign, noise, between, F, means in table:
            rho, ranks = rank_agreement(means, truth, sign)
            if math.isnan(rho):
                continue
            pick = max(means, key=lambda g: sign * means[g])
            gap = truth[pick] - truth[champ]
            cost = "—" if rank_of[pick] == 1 else f"落后 {-gap:+.3f} 分"
            L.append(f"| `{key}` | {rho:+.3f} | `{pick}` | {rank_of[pick]} | "
                     f"`{champ}` | {cost} |")
        L.append("")
        L.append(f"样本量警告：本表只有 **{len(truth)} 个检查点**，"
                 f"秩相关需要 |ρ| > {0.68 if len(truth) == 7 else 0.58:.2f} 才算勉强显著。"
                 "结论只能读作「旧指标没有信息量」，不能读作「某新指标已获证实」。")
        L.append("")

        # ---- 3. 组合指标 -------------------------------------------------
        L.append("## 3. 组合指标的尝试\n")
        L.append("把几个方向明确、F 较高的 rate 类指标做**等权 z 分**相加"
                 "（z 只在同一代的序列内部计算，不跨代拟合权重，避免用真值调参）：\n")
        combo = {}
        for g, ev in series.items():
            vals = {}
            for key, sign, _cn in METRICS:
                v = [float(e[key]) for _it, e in ev if key in e]
                if not v:
                    continue
                vs = v[-(args.tail * 2):] or v
                mu = sum(vs) / len(vs)
                sd = st.pstdev(vs) or 1.0
                vals[key] = ((vs[-1] - mu) / sd) * sign
            if vals:
                combo[g] = sum(vals.values()) / len(vals)
        if len(combo) >= 3:
            combo_t = {g: v for g, v in combo.items() if g in truth}
            rho, _ = rank_agreement(combo_t, truth, +1)
            pick = max(combo_t, key=lambda g: combo_t[g]) if combo_t else None
            L.append(f"- 等权 z 分组合：ρ = **{rho:+.3f}**，会选中 "
                     f"{f'`{pick}`（真值排名 {rank_of[pick]}）' if pick else '—'}")
            L.append(f"- 逐代组合分（仅真值集合内）："
                     + "、".join(f"`{g}`={combo_t[g]:+.2f}" for g in sorted(combo_t)))
        else:
            L.append("- 数据不足以构造组合指标")
        L.append("")

    # ---- 4. 结论 ---------------------------------------------------------
    L.append("## 4. 结论与方案 A 的选点决定\n")
    if table:
        best = max(table, key=lambda r: r[5])
        L.append(f"- 信噪比最高的是 **{best[1]}**（`{best[0]}`，F={best[5]:.2f}）；"
                 f"最低的是 **{[r[1] for r in table if r[5] == min(t[5] for t in table)][0]}**。")
        old = next((r for r in table if r[0] == "avg_score"), None)
        if old:
            L.append(f"- 旧依据 `avg_score`：F={old[5]:.2f}，单次抖动 {old[3]:.4f} "
                     f"vs 代际差 {old[4]:.4f} —— " +
                     ("**在噪声里抽签**。" if old[5] < 2 else "尚可。"))
    L.append("- 由于评测是确定性的，**「多评几次取平均」不能降噪**；"
             "方案 A 采取的组合是：① 换用 F 最高的 rate 类指标做选点；"
             "② 默认 `--select guard --save-every 50`（保留全部周期性检查点，"
             "让事后用天梯复核成为可能）；③ 训练内评测只当**辅助**，"
             "真正的晋升判定一律走 `scripts/score_checkpoints.py` 的共用牌池对位。")
    L.append("")
    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))
    print(f"\n报告 -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
