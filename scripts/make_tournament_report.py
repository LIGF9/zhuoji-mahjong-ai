"""从 tournament_results.json 生成 HTML 可视化报告（内联 SVG，无外部依赖）。"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

DISPLAY = {
    "rl": "rl（大师·热启动原版）",
    "dushan_s1": "dushan_s1（独山专训一段）",
    "dushan_s2": "dushan_s2（独山专训二段）",
    "bc": "bc（模仿学习基线）",
    "teacher:balanced": "教师·均衡",
    "teacher:aggressive": "教师·激进",
    "teacher:defensive": "教师·保守",
    "random": "随机",
}
BAR_COLORS = ["#534AB7", "#1D9E75", "#378ADD", "#BA7517", "#888780", "#D85A30", "#993556", "#5F5E5A"]


def esc(s):
    return str(s).replace("&", "&amp;").replace("<", "&lt;")


def bar_chart(items, width=600, fmt="{:+.3f}"):
    """水平条形图（内联 SVG）。items: [(label, value)] 已按绝对值排序。"""
    vmax = max(abs(v) for _, v in items) or 1.0
    rows = []
    y = 8
    for i, (label, v) in enumerate(items):
        w = abs(v) / vmax * 260
        neg = v < 0
        x0 = 230 if not neg else 230 - w
        color = BAR_COLORS[i % len(BAR_COLORS)]
        rows.append(
            f'<text x="222" y="{y + 12}" text-anchor="end" font-size="12" fill="#2C2C2A">{esc(label)}</text>'
            f'<rect x="{x0:.1f}" y="{y + 2}" width="{max(w, 1):.1f}" height="14" rx="3" fill="{color}"/>'
            f'<text x="{(x0 + w + 6 if not neg else x0 - 6):.1f}" y="{y + 13}" '
            f'text-anchor="{"start" if not neg else "end"}" font-size="12" '
            f'fill="#444441">{fmt.format(v)}</text>')
        y += 24
    h = y + 8
    return (f'<svg viewBox="0 0 {width} {h}" width="100%" role="img">'
            f'<line x1="230" y1="4" x2="230" y2="{h - 4}" stroke="#B4B2A9" stroke-width="0.5"/>'
            + "".join(rows) + "</svg>")


def main():
    src = ROOT / "reports" / "tournament_results.json"
    data = json.loads(src.read_text(encoding="utf-8"))
    res = data["results"]
    order = sorted(res.items(), key=lambda kv: -kv[1]["avg_score"])

    total_games = sum(s["games"] for s in res.values()) // 4
    rows = []
    for i, (name, s) in enumerate(order):
        color = BAR_COLORS[i % len(BAR_COLORS)]
        rows.append(f"""
      <tr>
        <td><span class="dot" style="background:{color}"></span>{esc(DISPLAY.get(name, name))}</td>
        <td>{s['games']:,}</td>
        <td class="num"><b>{s['avg_score']:+.3f}</b> ± {s['avg_score_se']:.3f}</td>
        <td class="num">{s['win_rate']*100:.1f}%</td>
        <td class="num">{s['tsumo_rate']*100:.1f}%</td>
        <td class="num">{s['ron_rate']*100:.1f}%</td>
        <td class="num">{s['deal_in_rate']*100:.1f}%</td>
        <td class="num">{s['huang_rate']*100:.1f}%</td>
        <td class="num">{s['avg_fan_when_win']:.2f}</td>
      </tr>""")

    chart = bar_chart([(DISPLAY.get(n, n), s["avg_score"]) for n, s in order])
    wr_chart = bar_chart([(DISPLAY.get(n, n), s["win_rate"] * 100) for n, s in order], fmt="{:.1f}%")

    html = f"""<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="utf-8">
<title>独山麻将策略循环赛报告</title>
<style>
  body {{ font-family: "Segoe UI", "Microsoft YaHei", sans-serif; margin: 32px auto;
         max-width: 960px; color: #2C2C2A; line-height: 1.6; }}
  h1 {{ font-size: 22px; }} h2 {{ font-size: 17px; margin-top: 36px; }}
  .meta {{ color: #5F5E5A; font-size: 13px; }}
  table {{ border-collapse: collapse; width: 100%; font-size: 13px; margin-top: 12px; }}
  th, td {{ border-bottom: 1px solid #D3D1C7; padding: 7px 8px; text-align: left; }}
  th {{ color: #5F5E5A; font-weight: 500; }}
  td.num, th.num {{ text-align: right; font-variant-numeric: tabular-nums; }}
  .dot {{ display: inline-block; width: 10px; height: 10px; border-radius: 3px;
         margin-right: 8px; vertical-align: -1px; }}
  .card {{ background: #F1EFE8; border: 0.5px solid #B4B2A9; border-radius: 12px;
          padding: 16px 20px; margin-top: 12px; }}
</style>
</head>
<body>
<h1>独山麻将策略循环赛报告</h1>
<p class="meta">规则：独山麻将 · {data['tables']} 张桌（8 策略全组合）× 每桌 {data['deals_per_table']} 副 ×
4 座位轮换复式 · 共 {total_games:,} 副 / {total_games*4:,} 局 · 用时 {data['elapsed_s']:.0f} 秒（向量化调度）</p>

<h2>综合排名（按复式平均得分）</h2>
<table>
  <tr><th>策略</th><th class="num">局数</th><th class="num">平均得分 ± SE</th>
      <th class="num">胜率</th><th class="num">自摸率</th><th class="num">放铳胡率</th>
      <th class="num">点炮率</th><th class="num">黄庄率</th><th class="num">胡牌均番</th></tr>
  {''.join(rows)}
</table>
<p class="meta">平均得分 = 终局得分差按局平均（零和）；SE 按副聚合（同副 4 次轮换压成 1 个观测），可作显著性参考——
差距小于 2×SE 的相邻名次在统计上难分高下。</p>

<h2>平均得分对比</h2>
<div class="card">{chart}</div>

<h2>胜率对比</h2>
<div class="card">{wr_chart}</div>

<h2>解读要点</h2>
<div class="card">
  <ul style="margin:0; padding-left: 18px;">
    <li><b>复式赛制</b>：每副牌四个策略轮换座位各打一次，牌运被抵消，名次反映真实棋力。</li>
    <li>所有网络策略显著强于启发式教师与随机；前三名之间的差距需结合 ±SE 判断。</li>
    <li>点炮率 / 黄庄率 / 胡牌均番可用来诊断风格：保守型点炮低但黄庄多，激进型反之。</li>
  </ul>
</div>
</body>
</html>"""
    out = ROOT / "reports" / "tournament_report.html"
    out.write_text(html, encoding="utf-8")
    print(f"report -> {out}")


if __name__ == "__main__":
    main()
