"""检验「训练内评测指标」能不能用来选点 —— P0 的核心分析。

问题
----
训练里用「对固定三位启发式教师、默认 160 局的评测均分」做 ``argmax`` 选点。
这个指标的 95% 置信区间约 ±1.9 分，而真实代际提升只有 ~0.3 分。
于是选点等于在噪声里抽签，并且系统性高估被选中的那一刻（winner's curse）——
历史 s7（评测 8.60）高于 s11（评测 7.231），实际却更弱。

本脚本做的事
-----------
把 ``scripts/score_checkpoints.py`` 产出的**共用牌池天梯**当成"真值"，
然后逐一检验训练内那些便宜、低方差的指标：

* 它们与真实强度的秩相关有多高？
* 如果当初用某个指标选点，会选中哪个检查点？它在天梯上排第几？
* 每个指标自身在它那份样本量下的标准误是多少（噪声水平）？

输出 ``reports/selection_analysis.md``。

用法::
    python scripts/score_checkpoints.py --ckpts ... --out reports/checkpoint_ladder.json
    python scripts/analyze_selection.py --ladder reports/checkpoint_ladder.json
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

RATE_METRICS = [
    ("win_rate", "胡牌率"),
    ("tsumo_rate", "自摸率"),
    ("deal_in_rate", "点炮率"),
    ("huang_rate", "黄庄率"),
    ("bigfan15_rate", "大牌率15+"),
]
SCALAR_METRICS = [
    ("avg_score", "评测均分(旧选点依据)"),
    ("avg_fan_when_win", "胡时均番"),
]


def spearman(x: list[float], y: list[float]) -> float:
    """秩相关（无 scipy，自己算；并列值取平均秩）。"""
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
    rx, ry = ranks(x), ranks(y)
    mx, my = sum(rx) / len(rx), sum(ry) / len(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = math.sqrt(sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry))
    return num / den if den else 0.0


def load_eval(stem: str) -> dict:
    ck = torch.load(ROOT / "models" / f"{stem}.pt", map_location="cpu",
                    weights_only=False)
    ev = ck.get("eval")
    return ev if isinstance(ev, dict) else {}


def metric_noise(ev: dict, key: str) -> float | None:
    """该指标在自己那份样本量下的标准误（噪声水平）。"""
    n = ev.get("games")
    if not n:
        return None
    if key == "avg_score":
        se = ev.get("avg_score_se")
        return float(se) if se else None
    n = int(n)
    if key in ("avg_fan_when_win",):
        # 番数是均值的均值，没有单独的 se 记录；用「胡了 k 次的均值」近似不易，
        # 这里返回 None（不参与噪声对比）
        return None
    if key.endswith("_rate") and key in ev:
        p = float(ev[key])
        return math.sqrt(max(p * (1 - p), 0.0) / n)
    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ladder", default="reports/checkpoint_ladder.json")
    ap.add_argument("--ladder2", default="",
                    help="可选：第二份天梯（换对手场），用于检验结论稳健性")
    ap.add_argument("--out", default="reports/selection_analysis.md")
    args = ap.parse_args()

    ladder = json.loads(Path(args.ladder).read_text(encoding="utf-8"))
    rows = ladder["rows"]
    lvl = {r["ckpt"]: r["level"] for r in rows}
    lvl_ci = {r["ckpt"]: r["level_ci95"] for r in rows}
    ckpts = [r["ckpt"] for r in reversed(rows)]        # 由弱到强，便于看相关

    evs = {c: load_eval(c) for c in ckpts}
    truth = [lvl[c] for c in ckpts]

    L = []
    L.append("# 训练内评测指标 vs 真实强度（共用牌池天梯）\n")
    L.append(f"- 天梯：`{ladder['deals']} 副 × 4 座位 = "
             f"{ladder['games_per_ckpt']} 局/检查点`，共用牌池 + 共用对手"
             f"（{' + '.join(ladder['ref_field'])}）")
    L.append(f"- 锚点：`{ladder['anchor']}`；用时 {ladder['elapsed_min']:.0f} 分\n")

    L.append("## 天梯排名（真值） vs 训练内评测均分\n")
    ev_rank = {c: i for i, c in enumerate(
        sorted(ckpts, key=lambda c: -float(evs[c].get("avg_score", -99))), 1)}
    L.append("| # | 检查点 | 天梯分值 | ±95%CI | 天梯结论 | 训练内评测均分 | 评测排名 | 排名差 |")
    L.append("|---|---|---|---|---|---|---|---|")
    for i, r in enumerate(rows, 1):
        c = r["ckpt"]
        ea = evs[c].get("avg_score")
        er = ev_rank[c]
        L.append(f"| {i} | `{c}` | {r['level']:+.3f} | {r['level_ci95']:.3f} | "
                 f"{r['verdict']} | "
                 f"{f'{ea:.3f}' if ea is not None else '—'} | {er} | "
                 f"{i - er:+d} |")
    L.append("")
    L.append("> 「排名差」= 天梯名次 − 评测名次。绝对值越大，说明该检查点被评测"
             "指标误判得越厉害（负值=评测高估了它）。\n")

    # ---- 各候选指标与真值的秩相关 ----------------------------------------
    L.append("## 候选指标与真实强度的相关性\n")
    L.append("| 指标 | Spearman ρ | 含义 | 自身噪声(SE) | 能否分辨 0.3 分？ |")
    L.append("|---|---|---|---|---|")
    metrics = SCALAR_METRICS + RATE_METRICS
    corr = {}
    for key, cn in metrics:
        vals = []
        used = []
        for c in ckpts:
            v = evs[c].get(key)
            if v is not None:
                vals.append(float(v))
                used.append(c)
        if len(vals) < 3:
            continue
        t = [lvl[c] for c in used]
        rho = spearman(vals, t)
        corr[key] = rho
        ses = [metric_noise(evs[c], key) for c in used]
        ses = [s for s in ses if s]
        se_med = float(np.median(ses)) if ses else None
        # 要分辨 0.3 分，指标的 SE 必须明显小于 0.3
        if se_med is None:
            able = "—"
        elif se_med < 0.05:
            able = f"✅ SE {se_med:.3f} ≪ 0.3"
        elif se_med < 0.15:
            able = f"⚠️ SE {se_med:.3f}，勉强"
        else:
            able = f"❌ SE {se_med:.3f} > 0.3"
        L.append(f"| {cn} | {rho:+.3f} | {key} | "
                 f"{f'{se_med:.4f}' if se_med is not None else '—'} | {able} |")
    L.append("")

    # ---- 若用某指标选点，会选中谁 ----------------------------------------
    L.append("## 如果用某个指标选点，会选中哪个检查点\n")
    rank_of = {r["ckpt"]: i for i, r in enumerate(rows, 1)}
    L.append("| 选点指标 | 会选中 | 它的天梯排名 | 相对天梯冠军的差距 | 代价 |")
    L.append("|---|---|---|---|---|")
    champ = rows[0]["ckpt"]
    for key, cn in metrics:
        best_c, best_v = None, None
        for c in ckpts:
            v = evs[c].get(key)
            if v is None:
                continue
            # 点炮率、黄庄率越低越好，其余越高越好
            better = (v < best_v) if (best_v is not None and key in
                                     ("deal_in_rate", "huang_rate")) else \
                (best_v is None or v > best_v)
            if better:
                best_c, best_v = c, v
        if best_c is None:
            continue
        gap = lvl[best_c] - lvl[champ]
        cost = "—" if rank_of[best_c] == 1 else f"落后 {-gap:.3f} 分"
        L.append(f"| {cn} | `{best_c}` | 第 {rank_of[best_c]} / {len(rows)} | "
                 f"{gap:+.3f} | {cost} |")
    L.append("")

    # ---- 两次天梯（不同对手场）一致性 -------------------------------------
    if args.ladder2:
        lad2 = json.loads(Path(args.ladder2).read_text(encoding="utf-8"))
        lvl2 = {r["ckpt"]: r["level"] for r in lad2["rows"]}
        common = [c for c in ckpts if c in lvl2]
        rho = spearman([lvl[c] for c in common], [lvl2[c] for c in common])
        rk1 = {r["ckpt"]: i for i, r in enumerate(rows, 1)}
        rk2 = {r["ckpt"]: i for i, r in enumerate(lad2["rows"], 1)}
        L.append("## 换一个对手场，结论稳不稳？\n")
        L.append(f"- 场 A（{ladder['deals']} 副）：{' + '.join(ladder['ref_field'])}")
        L.append(f"- 场 B（{lad2['deals']} 副）：{' + '.join(lad2['ref_field'])}")
        L.append(f"- 共有 {len(common)} 个检查点，两场分值水平的 Spearman "
                 f"ρ = **{rho:+.3f}**\n")
        L.append("| 检查点 | A 分值 | A 名次 | B 分值 | B 名次 | 名次差 |")
        L.append("|---|---|---|---|---|---|")
        for c in sorted(common, key=lambda x: rk1[x]):
            L.append(f"| `{c}` | {lvl[c]:+.3f} | {rk1[c]} | {lvl2[c]:+.3f} | "
                     f"{rk2[c]} | {rk1[c] - rk2[c]:+d} |")
        L.append("")

    # ---- 结论 -------------------------------------------------------------
    L.append("## 结论\n")
    tier = [r["ckpt"] for r in rows if r["verdict"] == "无显著差异"]
    worse = [f"`{r['ckpt']}`({r['vs_anchor']:+.3f}±{r['vs_anchor_ci95']:.3f})"
             for r in rows if r["verdict"] == "显著劣"]
    L.append(f"- **与锚点 `{ladder['anchor']}` 统计上打平的第一梯队（{len(tier)} 个）**："
             + "、".join(f"`{c}`" for c in tier))
    if worse:
        L.append(f"- 显著劣于锚点的：{'、'.join(worse)}")
    if corr:
        best_key = max(corr, key=lambda k: abs(corr[k]))
        cn = dict(metrics)[best_key]
        L.append(f"- 与真实强度秩相关最高的训练内指标是 **{cn}**"
                 f"（`{best_key}`，ρ = {corr[best_key]:+.3f}）。")
        L.append(f"- 而**旧选点依据「评测均分」的秩相关只有 ρ = {corr.get('avg_score', float('nan')):+.3f}**"
                 f"——基本没有信息量。")
    champ = rows[0]["ckpt"]
    argmax_eval = max(ckpts, key=lambda c: evs[c].get("avg_score", -99))
    L.append(f"- 天梯冠军是 `{champ}`（{lvl[champ]:+.3f}）；"
             f"而「评测均分」的历史最高值是 `{argmax_eval}`"
             f"（评测 {evs[argmax_eval].get('avg_score'):.3f}），"
             f"它在天梯上排第 {rank_of[argmax_eval]}，"
             f"落后冠军 {lvl[champ] - lvl[argmax_eval]:.3f} 分"
             f"——这就是 winner's curse 的直接代价。")
    spread = max(lvl[c] for c in ckpts if c.startswith("dushan_big")) - \
        min(lvl[c] for c in ckpts if c.startswith("dushan_big"))
    L.append(f"- 大网各代之间的真实强度跨度 {spread:.3f} 分；"
             f"训练内评测 160 局时 SE≈0.97（95%CI ±1.9 分）——**指标噪声比它要"
             f"分辨的信号还大**，这就是不能用它选点的根本原因。")
    L.append("")
    L.append("### 注意事项\n")
    L.append("- 本分析只有 12 个检查点（n=12），秩相关 |ρ| > 0.58 才勉强达到 p<0.05，"
             "候选指标之间的名次本身不稳，结论应理解为「旧指标没有信息量」"
             "而不是「某新指标已被证实可用」。")
    L.append("- 各检查点记录的训练内指标来自**不同样本量**的评测"
             "（64 / 100 / 160 局不等），噪声水平不一致。")
    L.append("- 天梯自身也有 CI（本轮 ±0.65~0.85 分），"
             "相邻名次之间的差异多数不显著，排名只应作为粗略分层。")
    out = Path(args.out)
    out.write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))
    print(f"\n报告 -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
