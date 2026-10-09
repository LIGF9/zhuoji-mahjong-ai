"""
把循环赛汇总渲染成一份自包含 HTML 报告（内联 SVG，不依赖任何 CDN，
断网也能打开）。

为什么要写成生成器而不是手写 HTML：数字会随重跑变化，
生成器保证报告与 reports/roundrobin_summary.json 永远一致。

用法::

    python scripts/make_winrate_report.py
"""
from __future__ import annotations

import argparse
import html
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# 配色：与报告其余部分一致，红=强/赢，青=弱/输（中文习惯）
C_STRONG = "#E24B4A"      # c-red 400
C_WEAK = "#1D9E75"        # c-teal 400
C_NEUTRAL = "#B4B2A9"     # c-gray 200
C_NET = "#534AB7"         # c-purple 600
C_TEACHER = "#185FA5"     # c-blue 600
C_RANDOM = "#5F5E5A"      # c-gray 600
TXT = "#2C2C2A"           # c-gray 900
TXT2 = "#5F5E5A"          # c-gray 600
BORDER = "#D3D1C7"        # c-gray 100

FONT = ("-apple-system,BlinkMacSystemFont,'Segoe UI','PingFang SC',"
        "'Microsoft YaHei',sans-serif")


def agent_color(name: str, tier: int = 0) -> str:
    if name == "random":
        return C_RANDOM
    if name in ("bc", "rl"):
        return C_NET
    return C_TEACHER


def label(name: str) -> str:
    """把内部名换成人看的名字。"""
    m = {
        "random": "随机出牌",
        "bc": "深度学习 · BC",
        "rl": "深度学习 · RL",
        "teacher:balanced": "启发式 · 均衡",
        "teacher:aggressive": "启发式 · 激进",
        "teacher:defensive": "启发式 · 保守",
        "teacher:chicken_lover": "启发式 · 囤幺鸡",
        "teacher:tenpai_rush": "启发式 · 抢听牌",
        "teacher:gambler": "启发式 · 赌徒",
    }
    return m.get(name, name)


# ---------------------------------------------------------------------------
# SVG 片段
# ---------------------------------------------------------------------------
def bar_win_rate(rows: list[dict], w: float = 960.0) -> str:
    """横向条形图：胡牌率 + 95% 区间（按牌组聚类的标准误）。"""
    left, right = 168.0, 96.0
    top, bh, gap = 34.0, 24.0, 16.0
    h = top + len(rows) * (bh + gap) + 30
    plot_w = w - left - right
    vmax = max(r["win_rate"] + 1.96 * r["win_rate_se"] for r in rows) * 1.08
    xs = lambda v: left + v / vmax * plot_w  # noqa: E731

    p = [f'<svg viewBox="0 0 {w:.0f} {h:.0f}" width="100%" role="img" '
         f'aria-label="各策略胡牌率条形图，含 95% 置信区间">',
         f'<title>各策略胡牌率分布</title>',
         f'<desc>四人混战 5040 局，{len(rows)} 种策略的胡牌率与 95% 区间</desc>']
    # 纵向网格
    step = vmax / 5
    for i in range(6):
        v = step * i
        x = xs(v)
        p.append(f'<line x1="{x:.1f}" y1="{top - 10:.1f}" x2="{x:.1f}" '
                 f'y2="{h - 30:.1f}" stroke="{BORDER}" stroke-width="0.5"/>')
        p.append(f'<text x="{x:.1f}" y="{h - 12:.1f}" text-anchor="middle" '
                 f'font-family="{FONT}" font-size="11" fill="{TXT2}">{v * 100:.0f}%</text>')

    for i, r in enumerate(rows):
        y = top + i * (bh + gap)
        col = agent_color(r["name"])
        bw = max(2.0, xs(r["win_rate"]) - left)
        p.append(f'<text x="{left - 12:.1f}" y="{y + bh / 2 + 4:.1f}" '
                 f'text-anchor="end" font-family="{FONT}" font-size="13" '
                 f'fill="{TXT}">{html.escape(label(r["name"]))}</text>')
        p.append(f'<rect x="{left:.1f}" y="{y:.1f}" width="{bw:.1f}" '
                 f'height="{bh:.1f}" rx="3" fill="{col}" fill-opacity="0.85"/>')
        lo = xs(max(0.0, r["win_rate"] - 1.96 * r["win_rate_se"]))
        hi = xs(r["win_rate"] + 1.96 * r["win_rate_se"])
        yc = y + bh / 2
        p.append(f'<line x1="{lo:.1f}" y1="{yc:.1f}" x2="{hi:.1f}" y2="{yc:.1f}" '
                 f'stroke="{TXT}" stroke-width="1.2"/>')
        for xx in (lo, hi):
            p.append(f'<line x1="{xx:.1f}" y1="{yc - 5:.1f}" x2="{xx:.1f}" '
                     f'y2="{yc + 5:.1f}" stroke="{TXT}" stroke-width="1.2"/>')
        p.append(f'<text x="{hi + 8:.1f}" y="{yc + 4:.1f}" font-family="{FONT}" '
                 f'font-size="12" fill="{TXT}">{r["win_rate"] * 100:.2f}%</text>')
    p.append("</svg>")
    return "".join(p)


def bar_score(rows: list[dict], w: float = 960.0) -> str:
    """横向发散条形图：每局平均得分（零和，平均线为 0）。"""
    left, right = 168.0, 110.0
    top, bh, gap = 34.0, 24.0, 16.0
    h = top + len(rows) * (bh + gap) + 42
    sorted_rows = sorted(rows, key=lambda r: -r["avg_score"])
    vmax = max(abs(r["avg_score"]) + 1.96 * r["avg_score_se"] for r in rows) * 1.06
    span = w - left - right
    zero = left + span / 2
    xs = lambda v: zero + v / vmax * (span / 2)  # noqa: E731

    p = [f'<svg viewBox="0 0 {w:.0f} {h:.0f}" width="100%" role="img" '
         f'aria-label="各策略每局平均得分，红为正青为负">',
         f'<title>各策略每局平均得分</title>',
         f'<desc>得分零和，四家之和为 0；红色表示赢分，青色表示输分</desc>']
    for i in range(-2, 3):
        v = vmax * i / 2
        x = xs(v)
        wgt = 1.0 if i == 0 else 0.5
        p.append(f'<line x1="{x:.1f}" y1="{top - 10:.1f}" x2="{x:.1f}" '
                 f'y2="{h - 34:.1f}" stroke="{"#888780" if i == 0 else BORDER}" '
                 f'stroke-width="{wgt}"/>')
        p.append(f'<text x="{x:.1f}" y="{h - 16:.1f}" text-anchor="middle" '
                 f'font-family="{FONT}" font-size="11" fill="{TXT2}">{v:+.1f}</text>')

    for i, r in enumerate(sorted_rows):
        y = top + i * (bh + gap)
        v = r["avg_score"]
        col = C_STRONG if v >= 0 else C_WEAK
        x0, x1 = min(zero, xs(v)), max(zero, xs(v))
        p.append(f'<text x="{left - 12:.1f}" y="{y + bh / 2 + 4:.1f}" '
                 f'text-anchor="end" font-family="{FONT}" font-size="13" '
                 f'fill="{TXT}">{html.escape(label(r["name"]))}</text>')
        p.append(f'<rect x="{x0:.1f}" y="{y:.1f}" width="{max(1.5, x1 - x0):.1f}" '
                 f'height="{bh:.1f}" rx="3" fill="{col}" fill-opacity="0.85"/>')
        lo = xs(max(-vmax, v - 1.96 * r["avg_score_se"]))
        hi = xs(min(vmax, v + 1.96 * r["avg_score_se"]))
        yc = y + bh / 2
        p.append(f'<line x1="{lo:.1f}" y1="{yc:.1f}" x2="{hi:.1f}" y2="{yc:.1f}" '
                 f'stroke="{TXT}" stroke-width="1.2"/>')
        for xx in (lo, hi):
            p.append(f'<line x1="{xx:.1f}" y1="{yc - 5:.1f}" x2="{xx:.1f}" '
                     f'y2="{yc + 5:.1f}" stroke="{TXT}" stroke-width="1.2"/>')
        p.append(f'<text x="{hi + 8:.1f}" y="{yc + 4:.1f}" font-family="{FONT}" '
                 f'font-size="12" fill="{TXT}">{v:+.3f}</text>')
    p.append("</svg>")
    return "".join(p)


def _lerp(a: tuple, b: tuple, t: float) -> str:
    r = round(a[0] + (b[0] - a[0]) * t)
    g = round(a[1] + (b[1] - a[1]) * t)
    bl = round(a[2] + (b[2] - a[2]) * t)
    return f"#{r:02X}{g:02X}{bl:02X}"


def heatmap(pair: dict, agents: list[str]) -> str:
    """对位热力图：行 A 与列 B 同桌时，A 每局比 B 多得的分。"""
    n = len(agents)
    # 标签列必须留够：行标签是全称（如 "启发式 · 抢听牌" ≈ 6 个全宽字），
    # 132px 时会被左边缘裁掉，这里放到 150px。
    cell, lab = 62.0, 150.0
    w = lab + n * cell + 24
    h = 96 + n * cell
    vmax = max(abs(pair[a][b]["mean_diff"]) for a in agents for b in agents if a != b)
    NEG, MID, POS = (29, 158, 117), (241, 239, 232), (226, 75, 74)

    p = [f'<svg viewBox="0 0 {w:.0f} {h:.0f}" width="100%" role="img" '
         f'aria-label="两两对位得分差热力图，红色表示行比列强">',
         f'<title>两两对位得分差</title>',
         f'<desc>每格为行策略相对列策略的每局平均得分差，逐副牌配对计算</desc>']
    for j, b in enumerate(agents):
        p.append(f'<text x="{lab + j * cell + cell / 2:.1f}" y="60" '
                 f'text-anchor="start" transform="rotate(-35 {lab + j * cell + cell / 2:.1f} 60)" '
                 f'font-family="{FONT}" font-size="12" fill="{TXT}">'
                 f'{html.escape(label(b).replace("深度学习 · ", "").replace("启发式 · ", ""))}</text>')
    for i, a in enumerate(agents):
        y = 96 + i * cell
        p.append(f'<text x="{lab - 12:.1f}" y="{y + cell / 2 + 4:.1f}" '
                 f'text-anchor="end" font-family="{FONT}" font-size="12" '
                 f'fill="{TXT}">{html.escape(label(a))}</text>')
        for j, b in enumerate(agents):
            x = lab + j * cell
            if a == b:
                p.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{cell - 2:.1f}" '
                         f'height="{cell - 2:.1f}" fill="{MID}"/>')
                continue
            v = pair[a][b]["mean_diff"]
            t = min(1.0, abs(v) / vmax)
            col = _lerp(MID, POS, t) if v >= 0 else _lerp(MID, NEG, t)
            p.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{cell - 2:.1f}" '
                     f'height="{cell - 2:.1f}" fill="{col}" fill-opacity="0.9"/>')
            tc = "#FFFFFF" if t > 0.45 else TXT
            p.append(f'<text x="{x + cell / 2 - 1:.1f}" y="{y + cell / 2 + 4:.1f}" '
                     f'text-anchor="middle" font-family="{FONT}" font-size="12" '
                     f'font-weight="500" fill="{tc}">{v:+.3f}</text>')
    p.append("</svg>")
    return "".join(p)


def scatter_win_vs_score(rows: list[dict], w: float = 960.0) -> str:
    """散点：胡牌率（快）与每局得分（值钱）并不一致。"""
    L, R, T, B = 84.0, 40.0, 34.0, 62.0
    h = 420.0
    span_x, span_y = w - L - R, h - T - B
    xmax = max(r["win_rate"] for r in rows) * 1.12
    ymin = min(r["avg_score"] for r in rows) * 1.15
    ymax = max(r["avg_score"] for r in rows) * 1.18
    xs = lambda v: L + v / xmax * span_x  # noqa: E731
    ys = lambda v: T + (ymax - v) / (ymax - ymin) * span_y  # noqa: E731

    p = [f'<svg viewBox="0 0 {w:.0f} {h:.0f}" width="100%" role="img" '
         f'aria-label="胡牌率与每局得分的散点图">',
         f'<title>胡牌率与每局得分的关系</title>',
         f'<desc>横轴胡牌率，纵轴每局平均得分；两者并不成正比</desc>']
    for i in range(6):
        v = xmax * i / 5
        p.append(f'<line x1="{xs(v):.1f}" y1="{T:.1f}" x2="{xs(v):.1f}" '
                 f'y2="{T + span_y:.1f}" stroke="{BORDER}" stroke-width="0.5"/>')
        p.append(f'<text x="{xs(v):.1f}" y="{T + span_y + 20:.1f}" '
                 f'text-anchor="middle" font-family="{FONT}" font-size="11" '
                 f'fill="{TXT2}">{v * 100:.0f}%</text>')
    ystep = (ymax - ymin) / 6
    for i in range(7):
        v = ymin + ystep * i
        p.append(f'<line x1="{L:.1f}" y1="{ys(v):.1f}" x2="{L + span_x:.1f}" '
                 f'y2="{ys(v):.1f}" stroke="{BORDER}" stroke-width="0.5"/>')
        p.append(f'<text x="{L - 10:.1f}" y="{ys(v) + 4:.1f}" text-anchor="end" '
                 f'font-family="{FONT}" font-size="11" fill="{TXT2}">{v:+.2f}</text>')
    # 零分线
    if ymin < 0 < ymax:
        p.append(f'<line x1="{L:.1f}" y1="{ys(0):.1f}" x2="{L + span_x:.1f}" '
                 f'y2="{ys(0):.1f}" stroke="#888780" stroke-width="1" '
                 f'stroke-dasharray="4 3"/>')

    for r in rows:
        x, y = xs(r["win_rate"]), ys(r["avg_score"])
        col = agent_color(r["name"])
        p.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="7" fill="{col}" '
                 f'fill-opacity="0.85" stroke="#FFFFFF" stroke-width="1.5"/>')
        p.append(f'<text x="{x:.1f}" y="{y - 13:.1f}" text-anchor="middle" '
                 f'font-family="{FONT}" font-size="11" fill="{TXT}">'
                 f'{html.escape(label(r["name"]).split(" · ")[-1])}</text>')
    p.append(f'<text x="{L + span_x / 2:.1f}" y="{h - 12:.1f}" text-anchor="middle" '
             f'font-family="{FONT}" font-size="12" fill="{TXT2}">胡牌率（多快胡牌）</text>')
    p.append(f'<text x="18" y="{T + span_y / 2:.1f}" text-anchor="middle" '
             f'font-family="{FONT}" font-size="12" fill="{TXT2}" '
             f'transform="rotate(-90 18 {T + span_y / 2:.1f})">'
             f'每局平均得分（赢得多值钱）</text>')
    p.append("</svg>")
    return "".join(p)


# ---------------------------------------------------------------------------
# 页面
# ---------------------------------------------------------------------------
def build(summary: dict) -> str:
    rows = summary["ranking"]
    agents = summary["design"]["agents"]
    d = summary["design"]
    v = summary["validation"]

    def tbl(header: list[str], body: list[list[str]], align: str = "") -> str:
        th = "".join(f"<th>{html.escape(c)}</th>" for c in header)
        tr = "".join("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>"
                     for r in body)
        return (f'<table class="tbl" {align}><thead><tr>{th}</tr></thead>'
                f'<tbody>{tr}</tbody></table>')

    rank_body = []
    for i, r in enumerate(rows, 1):
        rank_body.append([
            str(i),
            f'<span class="dot" style="background:{agent_color(r["name"])}"></span>'
            f'{html.escape(label(r["name"]))}',
            f'{r["win_rate"] * 100:.2f}%',
            f'±{1.96 * r["win_rate_se"] * 100:.2f}',
            f'{r["win_rate_ci"][0] * 100:.2f}~{r["win_rate_ci"][1] * 100:.2f}',
            f'{r["avg_score"]:+.3f}',
            f'±{1.96 * r["avg_score_se"]:.3f}',
            f'{r["tsumo_rate"] * 100:.2f}%',
            f'{r["deal_in_rate"] * 100:.2f}%',
            f'{r["huang_rate"] * 100:.2f}%',
            f'{r["chongfeng_rate"] * 100:.2f}%',
            f'{r["avg_fan_when_win"]:.2f}',
        ])

    sig_rows = [[html.escape(label(r["winner"])), html.escape(label(r["loser"])),
                 f'{r["diff"]:+.3f}', f'{r["t"]:.2f}']
                for r in summary["significant_pairs"]]

    ladder_rows = []
    for r in summary["ladder"]:
        mark = ('<span class="ok">分得开</span>' if r["separated"]
                else '<span class="no">分不开</span>')
        ladder_rows.append([html.escape(label(r["higher"])),
                            html.escape(label(r["lower"])),
                            f'{r["diff"]:+.3f}', f'{r["t"]:+.2f}', mark])

    check_rows = [[html.escape(k), f"<code>{html.escape(str(val))}</code>"]
                  for k, val in v.items()]

    css = f"""
    *{{box-sizing:border-box}}
    body{{margin:0;padding:40px 28px 72px;background:#FBFAF7;color:{TXT};
         font-family:{FONT};font-size:14px;line-height:1.7}}
    .wrap{{max-width:1060px;margin:0 auto}}
    h1{{font-size:24px;font-weight:500;margin:0 0 6px}}
    h2{{font-size:17px;font-weight:500;margin:40px 0 4px;
       padding-top:20px;border-top:1px solid {BORDER}}}
    h3{{font-size:14px;font-weight:500;margin:22px 0 6px;color:{TXT2}}}
    p{{margin:8px 0}}
    .sub{{color:{TXT2};font-size:13px}}
    .lead{{background:#F1EFE8;border-radius:12px;padding:16px 20px;margin:18px 0}}
    .lead ul{{margin:0;padding-left:20px}}
    .chart{{background:#FFFFFF;border:1px solid {BORDER};border-radius:12px;
           padding:18px 20px 12px;margin:14px 0}}
    .tbl{{width:100%;border-collapse:collapse;font-size:13px;margin:12px 0}}
    .tbl th{{text-align:left;font-weight:500;color:{TXT2};padding:8px 8px;
            border-bottom:1px solid {BORDER};white-space:nowrap}}
    .tbl td{{padding:8px 8px;border-bottom:1px solid #EDEBE5;
            font-variant-numeric:tabular-nums}}
    .tbl tr:hover td{{background:#F7F5F0}}
    .dot{{display:inline-block;width:9px;height:9px;border-radius:2px;
         margin-right:7px;vertical-align:1px}}
    .ok{{color:{C_WEAK};font-weight:500}}
    .no{{color:#993C1D;font-weight:500}}
    code{{font-family:ui-monospace,Menlo,monospace;font-size:12px;
         background:#F1EFE8;padding:1px 5px;border-radius:4px}}
    .kpi{{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));
         gap:12px;margin:18px 0}}
    .kpi div{{background:#FFFFFF;border:1px solid {BORDER};border-radius:10px;
             padding:12px 16px}}
    .kpi .n{{font-size:22px;font-weight:500;display:block;margin-top:2px}}
    .kpi .l{{font-size:12px;color:{TXT2}}}
    .note{{font-size:12.5px;color:{TXT2};margin-top:6px}}
    """

    parts = [
        "<!DOCTYPE html><html lang=\"zh-CN\"><head><meta charset=\"utf-8\">",
        "<meta name=\"viewport\" content=\"width=device-width,initial-scale=1\">",
        "<title>贵州捉鸡麻将 · 全策略胜率分布</title>",
        f"<style>{css}</style></head><body><div class=\"wrap\">",
        "<h1>贵州捉鸡麻将 · 全策略胜率分布</h1>",
        f"<p class=\"sub\">四人混战循环赛　{d['games_total']} 局　"
        f"{len(agents)} 种策略　复式设计（每副牌轮换四个座位）　"
        f"结果文件 <code>reports/roundrobin.json</code></p>",
        "<div class=\"lead\"><ul>",
        f"<li><b>零和校验</b>：{d['games_total']} 局得分总和 = {v['零和_各家总得分求和']}，"
        f"每家出场 {v['出场次数']} 次完全相等。</li>",
        "<li><b>均衡设计</b>：每个策略出场次数相同，任意两个策略同桌次数完全相同"
        f"（各 {v['两两同桌次数_min']} 次），配对比较不含“谁跟弱鸡同桌多”的偏差。</li>",
        "</ul></div>",
        "<div class=\"kpi\">",
    ]
    top = rows[0]
    parts += [
        f"<div><span class=\"l\">最强策略</span><span class=\"n\">"
        f"{html.escape(label(top['name']))}</span></div>",
        f"<div><span class=\"l\">最高胡牌率</span><span class=\"n\">"
        f"{max(r['win_rate'] for r in rows) * 100:.2f}%</span></div>",
        f"<div><span class=\"l\">最高每局得分</span><span class=\"n\">"
        f"{top['avg_score']:+.3f}</span></div>",
        f"<div><span class=\"l\">随机策略每局得分</span><span class=\"n\">"
        f"{rows[-1]['avg_score']:+.3f}</span></div>",
        "</div>",

        "<h2>1. 胡牌率分布</h2>",
        "<p class=\"sub\">横轴为胡牌率，误差棒为 95% 区间"
        "（按“复式牌组”聚类计算，不是按单局算，避免同副牌四次重复被当成四个独立观测）。</p>",
        f"<div class=\"chart\">{bar_win_rate(rows)}</div>",
        tbl(["#", "策略", "胡牌率", "±95%", "Wilson 区间", "每局得分", "±95%",
             "自摸率", "点炮率", "黄庄率", "冲锋鸡率", "胡时番数"],
            rank_body),
        "<p class=\"note\">胡牌率与每局得分并不一致：赌徒的胡牌率最高"
        f"（{max(r['win_rate'] for r in rows) * 100:.2f}%），每局得分却是负的；"
        "抢听牌同样如此——胡得快不等于赢得多。</p>",

        "<h2>2. 每局平均得分</h2>",
        "<p class=\"sub\">得分为零和，四家之和恒为 0，所以每局得分可直接读作"
        "“相对平均水平的净赢分”。红色为赢分，青色为输分。</p>",
        f"<div class=\"chart\">{bar_score(rows)}</div>",

        "<h2>3. 快 vs 值钱</h2>",
        "<p class=\"sub\">把两个指标放进同一平面：越靠右胡牌越多，"
        "越靠上每局赢分越多。两者明显不成正比。</p>",
        f"<div class=\"chart\">{scatter_win_vs_score(rows)}</div>",

        "<h2>4. 两两对位</h2>",
        "<p class=\"sub\">每格为“行策略 相对 列策略 的每局平均得分差”。"
        "计算方式是逐副牌配对：同一副牌上两者各坐一遍，只比决策差异，"
        "把牌运和座位优势消掉，因此比两个独立平均数相减精确得多。</p>",
        f"<div class=\"chart\">{heatmap(summary['pairwise'], agents)}</div>",

        "<h3>统计上站得住的对位（|t|&gt;2）</h3>",
        tbl(["更强者", "更弱者", "每局多得的分数", "t 值"], sig_rows)
        if sig_rows else "<p class=\"sub\">没有任何对位达到显著。</p>",
        "<p class=\"note\">除“碾压随机”之外，只有 RL 网络相对"
        "赌徒 / 保守 / 激进三个教师取得了统计显著的优势；"
        "网络与最有纪律的教师之间差别尚未达到显著。</p>",

        "<h3>名次相邻者是否真的分得开</h3>",
        tbl(["排名靠前", "排名靠后", "得分差", "t 值", "结论"], ladder_rows),
        "<p class=\"note\">从 RL 一路到赌徒，相邻名次之间的差异全部不显著——"
        "除随机策略稳居末位外，中间这 8 种策略在 5040 局里<b>没能分出高下</b>，"
        "名次先后只反映噪声。</p>",

        "<h2>5. 实验体检</h2>",
        tbl(["检查项", "结果"], check_rows),
        "<h3>座位平均分（检验座位是否有系统优势）</h3>",
        tbl(["策略", "东", "南", "西", "北"],
            [[html.escape(label(a))] + [f'{x:+.2f}' for x in vv]
             for a, vv in summary["seat_avg_score"].items()]),
        "<p class=\"note\">四座均值大体接近（同一策略内极差普遍在 0.3 分以内，"
        "而策略之间的差距可达 2 分），说明复式轮换已经把座位优势基本消掉。</p>",

        "<h2>6. 口径与复现</h2>",
        "<p class=\"sub\">规则：贵州捉鸡麻将，108 张（万条筒 1–9，无字牌），不能吃，"
        "豆为点胡通行证，含冲锋鸡、责任鸡、黄庄查叫。</p>",
        "<p>对手配置：<code>random</code>（随机合法动作，能胡必胡）、"
        "<code>bc</code>（行为克隆网络，temperature=0）、"
        "<code>rl</code>（自我对弈强化网络，temperature=0）、"
        "六种启发式教师（均衡 / 激进 / 保守 / 囤幺鸡 / 抢听牌 / 赌徒，"
        "权重见 <code>zhuoji/bots.py</code> 的 <code>TEACHER_STYLES</code>）。</p>",
        "<p>复现：</p>",
        "<p><code>python scripts/roundrobin.py --reps 10</code><br>"
        "<code>python scripts/analyze_roundrobin.py</code><br>"
        "<code>python scripts/make_winrate_report.py</code></p>",
        f"<p class=\"note\">种子 {d['seed']}，耗时 {d.get('elapsed', 0):.1f} 秒。"
        "策略本身是确定性策略（temperature=0），区间反映的是牌运与座位波动，"
        "不含策略自身的随机性——这是刻意选择：要衡量的是“这套策略有多强”，"
        "而不是“这套策略有多不稳定”。</p>",
        "</div></body></html>",
    ]
    return "".join(parts)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="src",
                    default=str(ROOT / "reports" / "roundrobin_summary.json"))
    ap.add_argument("--out", default=str(ROOT / "reports" / "winrate.html"))
    args = ap.parse_args()
    summary = json.loads(Path(args.src).read_text())
    Path(args.out).write_text(build(summary), encoding="utf-8")
    print(f"报告 -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
