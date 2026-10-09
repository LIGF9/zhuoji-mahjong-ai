"""
生成自包含的 HTML 训练报告：规则档、数据统计、训练曲线、复式评测对比。

图表全部用**内联 SVG** 手绘，不依赖任何 CDN 或前端库，离线也能看。
"""
from __future__ import annotations

import argparse
import html
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


# ---------------------------------------------------------------------------
# SVG 图表
# ---------------------------------------------------------------------------
PALETTE = ["#2563eb", "#dc2626", "#059669", "#d97706", "#7c3aed", "#0891b2"]


def _nice_ticks(lo: float, hi: float, n: int = 5):
    if hi - lo < 1e-12:
        hi = lo + 1.0
    raw = (hi - lo) / n
    mag = 10 ** int(__import__("math").floor(__import__("math").log10(raw)))
    for m in (1, 2, 2.5, 5, 10):
        step = m * mag
        if step >= raw:
            break
    start = __import__("math").floor(lo / step) * step
    ticks = []
    v = start
    while v <= hi + step * 0.5:
        ticks.append(v)
        v += step
    return ticks


def line_chart(series, xs, title: str, ylabel: str = "", width: int = 680,
               height: int = 260, y0: float | None = None):
    """``series``: list of (label, values)。返回 SVG 字符串。"""
    pad_l, pad_r, pad_t, pad_b = 58, 108, 34, 40
    pw, ph = width - pad_l - pad_r, height - pad_t - pad_b
    vals = [v for _, vs in series for v in vs if v is not None]
    if not vals:
        return f'<svg viewBox="0 0 {width} {height}"></svg>'
    lo = min(vals + ([y0] if y0 is not None else []))
    hi = max(vals + ([y0] if y0 is not None else []))
    if hi - lo < 1e-9:
        hi = lo + 1.0
    margin = (hi - lo) * 0.08
    lo -= margin
    hi += margin
    ticks = _nice_ticks(lo, hi, 5)
    lo, hi = min(ticks), max(ticks)

    def X(i):
        return pad_l + (pw * i / max(1, len(xs) - 1))

    def Y(v):
        return pad_t + ph * (1 - (v - lo) / (hi - lo))

    out = [f'<svg viewBox="0 0 {width} {height}" width="100%" '
           f'style="max-width:{width}px;font-family:inherit">']
    out.append(f'<rect x="0" y="0" width="{width}" height="{height}" fill="none"/>')
    for t in ticks:
        y = Y(t)
        out.append(f'<line x1="{pad_l}" y1="{y:.1f}" x2="{pad_l+pw}" y2="{y:.1f}" '
                   f'stroke="#e5e7eb" stroke-width="1"/>')
        out.append(f'<text x="{pad_l-8}" y="{y+4:.1f}" text-anchor="end" '
                   f'font-size="11" fill="#6b7280">{t:.3g}</text>')
    step = max(1, len(xs) // 10)
    for i in range(0, len(xs), step):
        out.append(f'<text x="{X(i):.1f}" y="{pad_t+ph+18}" text-anchor="middle" '
                   f'font-size="11" fill="#6b7280">{xs[i]}</text>')
    out.append(f'<line x1="{pad_l}" y1="{pad_t+ph}" x2="{pad_l+pw}" y2="{pad_t+ph}" '
               f'stroke="#9ca3af" stroke-width="1"/>')
    out.append(f'<text x="{pad_l}" y="{pad_t-12}" font-size="12.5" fill="#111827" '
               f'font-weight="600">{html.escape(title)}</text>')
    if ylabel:
        out.append(f'<text x="{pad_l-46}" y="{pad_t-12}" font-size="11" '
                   f'fill="#9ca3af">{html.escape(ylabel)}</text>')

    for si, (label, vs) in enumerate(series):
        color = PALETTE[si % len(PALETTE)]
        d = []
        for i, v in enumerate(vs):
            if v is None:
                continue
            d.append(("M" if not d else "L") + f"{X(i):.1f},{Y(v):.1f}")
        out.append(f'<path d="{" ".join(d)}" fill="none" stroke="{color}" '
                   f'stroke-width="2" stroke-linejoin="round"/>')
        for i, v in enumerate(vs):
            if v is None:
                continue
            out.append(f'<circle cx="{X(i):.1f}" cy="{Y(v):.1f}" r="2.6" fill="{color}"/>')
        ly = pad_t + 14 + si * 18
        out.append(f'<line x1="{pad_l+pw+12}" y1="{ly-4}" x2="{pad_l+pw+30}" y2="{ly-4}" '
                   f'stroke="{color}" stroke-width="2.5"/>')
        out.append(f'<text x="{pad_l+pw+36}" y="{ly}" font-size="11.5" '
                   f'fill="#374151">{html.escape(label)}</text>')
    out.append("</svg>")
    return "".join(out)


def bar_chart(items, title: str, width: int = 680, height: int = 220):
    """``items``: list of (label, value)。正负分色（涨红跌绿 → 这里用蓝/橙表正负）。"""
    if not items:
        return ""
    pad_l, pad_r, pad_t, pad_b = 58, 24, 34, 46
    pw, ph = width - pad_l - pad_r, height - pad_t - pad_b
    vs = [v for _, v in items]
    lo, hi = min(vs + [0.0]), max(vs + [0.0])
    if hi - lo < 1e-9:
        hi = lo + 1.0
    lo -= (hi - lo) * 0.12
    hi += (hi - lo) * 0.12
    Y = lambda v: pad_t + ph * (1 - (v - lo) / (hi - lo))
    out = [f'<svg viewBox="0 0 {width} {height}" width="100%" '
           f'style="max-width:{width}px;font-family:inherit">']
    for t in _nice_ticks(lo, hi, 4):
        y = Y(t)
        out.append(f'<line x1="{pad_l}" y1="{y:.1f}" x2="{pad_l+pw}" y2="{y:.1f}" '
                   f'stroke="#e5e7eb"/>')
        out.append(f'<text x="{pad_l-8}" y="{y+4:.1f}" text-anchor="end" font-size="11" '
                   f'fill="#6b7280">{t:.3g}</text>')
    bw = pw / max(1, len(items)) * 0.62
    zero = Y(0.0)
    for i, (label, v) in enumerate(items):
        cx = pad_l + pw * (i + 0.5) / len(items)
        y = Y(v)
        top, h = (y, zero - y) if v >= 0 else (zero, y - zero)
        color = "#dc2626" if v >= 0 else "#059669"   # 中国习惯：正=红，负=绿
        out.append(f'<rect x="{cx-bw/2:.1f}" y="{top:.1f}" width="{bw:.1f}" '
                   f'height="{max(1.0,h):.1f}" fill="{color}" rx="3" opacity="0.85"/>')
        out.append(f'<text x="{cx:.1f}" y="{(top-5) if v>=0 else (top+h+14):.1f}" '
                   f'text-anchor="middle" font-size="11" fill="#374151">{v:+.2f}</text>')
        out.append(f'<text x="{cx:.1f}" y="{pad_t+ph+18:.1f}" text-anchor="middle" '
                   f'font-size="11" fill="#6b7280">{html.escape(label)}</text>')
    out.append(f'<line x1="{pad_l}" y1="{zero:.1f}" x2="{pad_l+pw}" y2="{zero:.1f}" '
               f'stroke="#9ca3af"/>')
    out.append(f'<text x="{pad_l}" y="{pad_t-12}" font-size="12.5" fill="#111827" '
               f'font-weight="600">{html.escape(title)}</text>')
    out.append("</svg>")
    return "".join(out)


# ---------------------------------------------------------------------------
# 报告
# ---------------------------------------------------------------------------
def load_json(p: Path, default=None):
    if p.exists():
        return json.loads(p.read_text(encoding="utf-8"))
    return default


def compute_data_stats(path: Path) -> dict:
    """直接从 BC 数据集算出统计量。"""
    if not path.exists():
        return {}
    d = np.load(path)
    head = d["head"]
    names = ["discard（出牌/自摸胡）", "self_kong（闷豆/爬坡豆）", "respond（碰/点豆/胡/过）"]
    out = {"数据文件大小": f"{path.stat().st_size / 1024 / 1024:.1f} MB"}
    out["总样本数（决策点）"] = int(len(head))
    for i, nm in enumerate(names):
        out[f"动作头 {nm}"] = int((head == i).sum())
    resp = head == 2
    if resp.any():
        for idx, nm in enumerate(["过", "碰", "点豆", "胡"]):
            out[f"  响应动作「{nm}」"] = int(((head == 2) & (d["act"] == idx)).sum())
    val = d["value"]
    out["终局得分均值 / 标准差"] = f"{val.mean():.3f} / {val.std():.3f}"
    out["终局得分范围"] = f"{val.min():.1f} ~ {val.max():.1f}"
    return out


def arena_block(arena: dict, header: str) -> list[str]:
    """把一份 arena.json 渲染成小节（标题 + 每个对手一张表）。"""
    out = [header]
    out.append(f'<p class="note">模型：<code>{html.escape(str(arena.get("model")))}</code>，'
               f'{arena.get("rounds")} 副牌 × 4 座位轮换（共 '
               f'{4 * int(arena.get("rounds", 0))} 局/对手，同一副牌四家轮换座位，'
               f'消掉牌运方差）。</p>')
    cols = ["agent", "avg_score", "win_rate", "tsumo_rate", "ron_rate",
            "deal_in_rate", "huang_rate", "avg_fan_when_win"]
    cn = {"agent": "Agent", "avg_score": "平均得分", "win_rate": "胡牌率",
          "tsumo_rate": "自摸率", "ron_rate": "点胡率", "deal_in_rate": "点炮率",
          "huang_rate": "黄庄率", "avg_fan_when_win": "胡牌平均番"}
    head = "".join(f"<th>{cn[c]}</th>" for c in cols)
    disp = {"bc:bc": "上一代 BC（bc.pt）", "prev:bc": "上一代 BC（bc.pt）",
            "prev:rl": "上一代 RL（rl.pt）"}
    for opp, res in arena["results"].items():
        rows = ""
        for name, s in sorted(res.items(), key=lambda kv: -kv[1]["avg_score"]):
            tds = ""
            for c in cols:
                if c == "agent":
                    cls = ' class="mebot"' if name.startswith("net") else ""
                    tds += f"<td{cls}>{html.escape(disp.get(name, name))}</td>"
                elif c == "avg_score":
                    se = s.get("avg_score_se") or 0.0
                    if name.startswith("net") and se > 0:
                        # 被测模型带上标准误：没有误差棒的均值在 80 副牌的量级上
                        # 几乎无法区分真信号和牌运噪声
                        tds += (f'<td>{s[c]:+.3f} '
                                f'<span style="color:#9ca3af">±{se:.2f}</span></td>')
                    else:
                        tds += f"<td>{s[c]:+.3f}</td>"
                elif isinstance(s[c], float):
                    tds += f"<td>{s[c]:.3f}</td>"
                else:
                    tds += f"<td>{s[c]}</td>"
            rows += f"<tr>{tds}</tr>"
        out.append(f"<h3>vs 3×{html.escape(opp)}</h3>"
                   f'<table class="tbl"><tr>{head}</tr>{rows}</table>')
        net = next((s for n, s in res.items() if n.startswith("net")), None)
        if net and net.get("avg_score_se"):
            t = net["avg_score"] / net["avg_score_se"]
            verdict = ("<b>与 0 无显著差异</b>（|t| &lt; 2），"
                       "这批对局的数量还不足以分辨" if abs(t) < 2
                       else "<b>显著偏离 0</b>（|t| ≥ 2）")
            out.append(f'<p class="note">被测模型平均分 '
                       f'{net["avg_score"]:+.3f} ± {net["avg_score_se"]:.3f}'
                       f'（{net.get("deals")} 副牌观测，标准误），t = {t:+.2f} → {verdict}。</p>')
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reports", default=str(ROOT / "reports"))
    ap.add_argument("--out", default=str(ROOT / "reports" / "report.html"))
    args = ap.parse_args()
    R = Path(args.reports)

    bc = load_json(R / "bc_history.json", [])
    rl = load_json(R / "rl_history.json", [])
    rl_alt = load_json(R / "rl_history_unanchored.json", [])
    arena = load_json(R / "arena.json", None)
    arena_bc = load_json(R / "arena_bc.json", None)
    cmp = load_json(R / "compare.json", None)
    stats = load_json(R / "data_stats.json", {})
    replay = load_json(R / "sample_game.json", None)

    if not stats:
        stats = compute_data_stats(ROOT / "data" / "bc_data.npz")

    blocks = []

    # ---- 数据来源核实 ----
    blocks.append("""
<h2>1. 数据来源核实</h2>
<table class="tbl">
<tr><th>核实项</th><th>结果</th></tr>
<tr><td>任务给出的 Botzone 页面</td><td>第六届国际麻将人工智能比赛（IJCAI 2026），规则是 <b>国标麻将 MCR</b>，<b>不是捉鸡</b></td></tr>
<tr><td>国标 vs 捉鸡差异</td><td>国标 144 张含字牌花牌、可吃、8 番起胡、81 种番型；捉鸡 108 张无字牌、<b>不能吃</b>、豆为点胡通行证、捉鸡/冲锋鸡/责任鸡算分</td></tr>
<tr><td>官方"强 AI 对局数据"（北大网盘）</td><td>AnyShare 单页应用，需交互登录才能取直链，脚本抓不到</td></tr>
<tr><td>Botzone 全站数据下载站</td><td>下载页可访问，但打包文件所在域 <code>extra.botzone.org.cn</code> <b>当前 502</b></td></tr>
<tr><td>Botzone 是否有"捉鸡麻将"游戏</td><td><b>没有</b>（全站游戏列表 865KB 中不存在该条目），平台内也无捉鸡对局可爬</td></tr>
<tr><td>可用替代数据</td><td><code>hf-mirror.com</code> 上有国标数据（9.8 万局）+ 决赛强 AI 数据 + 预训练权重</td></tr>
</table>
<p class="note">结论：<b>捉鸡没有公开对局数据，必须自产</b>。本流程 = 自研规则引擎 → 启发式教师生成对局 → 行为克隆 → 自我对弈强化。</p>
""")

    # ---- 数据统计 ----
    if stats:
        rows = "".join(
            f"<tr><td>{html.escape(str(k))}</td><td>{html.escape(str(v))}</td></tr>"
            for k, v in stats.items())
        blocks.append(f'<h2>2. 自产数据统计</h2><table class="tbl">'
                      f'<tr><th>项</th><th>值</th></tr>{rows}</table>')

    # ---- BC 曲线 ----
    if bc:
        eps = [r["epoch"] for r in bc]
        blocks.append("<h2>3. 行为克隆（监督学习）</h2>")
        blocks.append('<div class="card">' + line_chart(
            [("train loss", [r["train_loss"] for r in bc]),
             ("val loss", [r["val_loss"] for r in bc])],
            eps, "行为克隆损失", "loss") + "</div>")
        blocks.append('<div class="card">' + line_chart(
            [("train acc", [r["train_acc"] for r in bc]),
             ("val acc", [r["val_acc"] for r in bc])],
            eps, "动作预测准确率", "acc") + "</div>")
        last = bc[-1]
        blocks.append(f'<p class="note">最终：val_loss={last["val_loss"]:.4f}，'
                      f'val_acc={last["val_acc"]:.3f}，'
                      f'用时 {last["elapsed"]:.0f}s（CPU）。</p>')

    # ---- RL 曲线 ----
    if rl:
        its = [r["iter"] for r in rl]
        blocks.append("<h2>4. 自我对弈强化</h2>")
        blocks.append('<div class="card">' + line_chart(
            [("终局得分/局", [r["reward"] for r in rl])],
            its, "自我对弈平均终局得分（归一化）", "reward") + "</div>")
        ev = [(r["iter"], r["eval"]["avg_score"]) for r in rl if "eval" in r]
        if ev:
            blocks.append('<div class="card">' + line_chart(
                [("评测平均分", [v for _, v in ev])],
                [i for i, _ in ev],
                "复式评测：对 3 个启发式教师的平均得分") + "</div>")
        blocks.append('<div class="card">' + line_chart(
            [("策略熵", [r["entropy"] for r in rl]),
             ("KL(π‖π_BC)", [r["kl"] for r in rl])],
            its, "策略熵 / 与 BC 的 KL 距离", "") + "</div>")
        last = rl[-1]
        blocks.append(f'<p class="note">共 {len(rl)} 轮，每轮 '
                      f'{rl[0]["samples"]} 上下样本；最终策略熵 '
                      f'{last["entropy"]:.3f}、KL {last["kl"]:.3f}。</p>')

        # ---- 消融：KL 锚定系数 ----
        if rl_alt:
            n = max(len(rl), len(rl_alt))

            def pad(seq):
                seq = list(seq)
                return seq + [None] * (n - len(seq))

            blocks.append("<h3>4.1 消融实验：KL 锚定系数决定成败</h3>")
            blocks.append(
                '<p class="note">第一版配置 <code>kl_coef=0.02, lr=3e-4</code>：'
                '策略熵从 1.44 一路崩到 0.07，与 BC 的 KL 涨到 2.4 nats，'
                '策略彻底脱离教师分布。它在自我对弈的评测里看着"在涨"，'
                '但放到复式对局里<b>被 BC 反杀</b>——这正是"只看训练指标会被骗"的典型。'
                '把 KL 锚定提到 0.8、学习率降到 5e-5 重跑后，曲线明显受控。</p>')
            xs = list(range(1, n + 1))
            blocks.append('<div class="card">' + line_chart(
                [("KL · 未锚定", pad(r["kl"] for r in rl_alt)),
                 ("KL · 已锚定", pad(r["kl"] for r in rl))],
                xs, "与 BC 策略的 KL 距离（越小越贴近教师）") + "</div>")
            blocks.append('<div class="card">' + line_chart(
                [("熵 · 未锚定", pad(r["entropy"] for r in rl_alt)),
                 ("熵 · 已锚定", pad(r["entropy"] for r in rl))],
                xs, "策略熵（熵崩 = 策略塌成近乎确定性）") + "</div>")

    # ---- 评测 ----
    if arena or arena_bc:
        blocks.append("<h2>5. 复式对战评测</h2>")
        blocks.append('<p class="note">复式对局：同一副起手牌让 4 个 Agent 轮换坐四个座位'
                      '各打一次，四家打的是同一副牌，因此平均得分只反映策略差异。'
                      '表格里 <code>net:rl</code> / <code>net:bc</code> 是被测模型，'
                      '其余行是对手（3 个同名对手合并成一行）。</p>')
        if arena:
            blocks += arena_block(arena, "<h3>5.1 强化后模型（rl.pt）</h3>")
        if arena_bc:
            blocks += arena_block(arena_bc, "<h3>5.2 行为克隆模型（bc.pt）</h3>")

    # ---- 配对比较 ----
    if cmp:
        blocks.append("<h3>5.3 逐副牌配对比较：RL vs BC</h3>")
        blocks.append(
            '<p class="note">上面 5.1 / 5.2 是各自独立跑的，标准误都在 0.15 上下，'
            '<b>没法直接相减下结论</b>。但这两份评测吃的是<b>同一批牌</b>'
            '（同 <code>--seed</code>、同 <code>--rounds</code>），'
            '对手也由确定性种子生成，于是每副牌上的牌运在两次运行里是同一个值 —— '
            '逐副牌做差就把它消掉了。这张表是本报告里统计功效最高的一组数字：'
            '差值 &gt; 0 表示 RL 更强。</p>')
        rows = "".join(
            f"<tr><td>{html.escape(r['opponent'])}</td><td>{r['n_deals']}</td>"
            f"<td>{r['mean_diff']:+.3f}</td><td>{r['se']:.3f}</td>"
            f"<td>{r['t']:+.2f}</td>"
            f"<td>{'是' if abs(r['t']) >= 2 else '—'}</td></tr>"
            for r in cmp["rows"])
        blocks.append(
            f'<table class="tbl"><tr><th>对手</th><th>副牌数</th>'
            f'<th>差值（{html.escape(str(cmp["label_a"]))} − '
            f'{html.escape(str(cmp["label_b"]))}）</th><th>标准误</th>'
            f'<th>t</th><th>|t| ≥ 2</th></tr>{rows}</table>')
        c = cmp["combined"]
        vr = (sum(r["var_reduction"] for r in cmp["rows"]) / len(cmp["rows"])
              if cmp["rows"] else 0.0)
        blocks.append(
            f'<p class="note">逆方差加权合并：'
            f'<b>{c["mean_diff"]:+.3f} ± {c["se"]:.3f}（t = {c["t"]:+.2f}）</b>。'
            f'配对把差值方差相对「各跑一份再比均值」压缩了约 {vr:.0%} —— '
            f'这就是配对比较值得做的原因。</p>')

    # ---- 样例对局 ----
    if replay:
        blocks.append("<h2>6. 一局样例对局（模型坐在 0 号位）</h2>")
        blocks.append(f'<p class="note">{html.escape(replay.get("summary_text", ""))}</p>')
        s = replay.get("summary", {})
        if s.get("type") == "win":
            rows = "".join(
                f"<tr><td>{html.escape(str(x[1]))}</td><td>{html.escape(str(x[2]))}</td>"
                f"<td>{html.escape(str(x[3]))}</td></tr>"
                for x in (s.get("dou") or []) + (s.get("chicken") or []))
            if rows:
                blocks.append('<h3>豆 / 鸡 结算明细</h3><table class="tbl">'
                              '<tr><th>玩家</th><th>项</th><th>番值</th></tr>'
                              + rows + "</table>")
        hands = replay.get("hands") or []
        melds = replay.get("melds") or []
        if hands:
            rows = "".join(
                f"<tr><td>座位 {i}{'（模型）' if i == 0 else ''}</td>"
                f"<td>{html.escape(str(hands[i]))}</td>"
                f"<td>{html.escape('、'.join(melds[i]) if melds and melds[i] else '—')}</td></tr>"
                for i in range(len(hands)))
            blocks.append('<h3>终局手牌</h3><table class="tbl">'
                          '<tr><th>玩家</th><th>手牌</th><th>副露</th></tr>'
                          + rows + "</table>")

        dec = replay.get("decisions") or []
        if dec:
            blocks.append("<h3>模型在关键决策点上的动作概率</h3>")
            for d in dec:
                top = "".join(
                    f'<span class="pill">{html.escape(t["action"])} '
                    f'<b>{t["p"]:.0%}</b></span>' for t in d["top"])
                blocks.append(
                    f'<div class="dec"><div class="dechd">第 {d["step"]} 手 · '
                    f'牌墙剩 {d["wall_left"]} 张 · 价值估计 {d["value"]:+.2f}</div>'
                    f'<div class="dechand">手牌：{html.escape(d["hand"])}</div>'
                    f'<div class="dechand">副露：'
                    f'{html.escape("、".join(d["melds"]) or "—")}</div>'
                    f'<div>{top}</div></div>')

        rows = "".join(
            f'<tr><td>{i+1}</td><td>座位 {r[0]}</td><td>{html.escape(str(r[1]))}</td></tr>'
            for i, r in enumerate(replay.get("log", [])[:200]))
        blocks.append('<h3>动作流水</h3><table class="tbl">'
                      '<tr><th>#</th><th>座位</th><th>动作</th></tr>' + rows + "</table>")

    # ---- 结论与局限 ----
    def net_row(arena_d: dict | None, opp: str):
        if not arena_d:
            return None
        for n, s in ((arena_d.get("results") or {}).get(opp) or {}).items():
            if n.startswith("net"):
                return s
        return None

    blocks.append("<h2>7. 结论与局限</h2>")
    if arena:
        nrand = net_row(arena, "random")
        teach = [net_row(arena, k) for k in (arena.get("results") or {})
                 if k.startswith("teacher")]
        teach = [s for s in teach if s]
        nprev = net_row(arena, "prev:bc") or net_row(arena, "bc:bc")
        bc_rand = net_row(arena_bc, "random")
        bc_teach = [net_row(arena_bc, k) for k in (arena_bc.get("results") or {})
                    if k.startswith("teacher")] if arena_bc else []
        bc_teach = [s for s in bc_teach if s]

        li = []
        if nrand and bc_rand:
            li.append(
                f"<li><b>对随机基线：两代模型都是碾压级。</b>"
                f"RL {nrand['avg_score']:+.2f}（t = "
                f"{nrand['avg_score'] / max(1e-9, nrand['avg_score_se']):+.1f}），"
                f"BC {bc_rand['avg_score']:+.2f}。这是整份报告里唯一"
                f"误差棒远小于效应量的对比，可以放心下结论。</li>")
        if cmp:
            cc = cmp["combined"]
            nsig = sum(1 for r in cmp["rows"] if abs(r["t"]) >= 2)
            pos = sum(1 for r in cmp["rows"] if r["mean_diff"] > 0)
            li.append(
                f"<li><b>RL 相对 BC 的提升，只有配对比较才测得出来："
                f"{cc['mean_diff']:+.3f} ± {cc['se']:.3f}（t = {cc['t']:+.2f}）。</b>"
                f" {len(cmp['rows'])} 组对手里 {pos} 组差值为正、"
                f"{nsig} 组达到 |t| ≥ 2；方向上是一致的，"
                f"但单组的量级仍在噪声边缘。"
                f"<b>把同一批牌喂给两个模型再逐副做差</b>，"
                f"是这套里性价比最高的一步——牌运这个最大的噪声源直接消失了。</li>")
        if teach and bc_teach:
            rl_m = sum(s["avg_score"] for s in teach) / len(teach)
            bc_m = sum(s["avg_score"] for s in bc_teach) / len(bc_teach)
            li.append(
                f"<li><b>对启发式教师：两代模型都只是「略优」。</b>"
                f"RL 在 {len(teach)} 组教师上的平均得分 {rl_m:+.3f}，"
                f"BC {bc_m:+.3f}；单看任何一组，|t| 基本都小于 2。"
                f"所以<b>「RL 已经明显强于教师」这个结论目前站不住</b>——"
                f"按当前方差估算，要把它单独测出来需要上千副牌。</li>")
        if nprev:
            t_h = nprev["avg_score"] / max(1e-9, nprev["avg_score_se"])
            verdict = ("未分胜负" if abs(t_h) < 2
                       else ("RL 显著更强" if t_h > 0 else "RL 显著更弱"))
            li.append(
                f"<li><b>同桌正面对撞：{verdict}。</b>在「vs 3×BC」这张桌上 RL 拿 "
                f"{nprev['avg_score']:+.3f} ± {nprev['avg_score_se']:.3f}"
                f"（t = {t_h:+.2f}），BC 一侧按零和折算为 "
                f"{-nprev['avg_score'] / 3:+.3f}。"
                f"注意这张桌是 1 个 RL 打 3 个 BC，与 5.3 那张"
                f"「双方各带 3 个教师」的配对表不是同一个量，别混着读。</li>")
        li.append(
            "<li><b>唯一鲁棒的强化结论来自消融实验，而不是胜率。</b>"
            "未锚定那版 RL 的训练日志里评测分看着在涨（-0.80 → -0.24），"
            "但策略熵从 1.44 崩到 0.07、与教师的 KL 涨到 2.4 nats —— "
            "这两条曲线本身就把「策略已经跑飞」写清楚了，不需要等胜负统计。"
            "加上 KL 锚定后，熵稳在 1.27–1.34、KL 稳在 0.03，"
            "训练过程才真正受控。</li>")
        blocks.append("<ul>" + "".join(li) + "</ul>")
        blocks.append(
            '<p class="note"><b>需要说清楚的方法学局限：</b>捉鸡单局得分方差很大'
            '（含冲锋鸡、责任鸡、黄庄查叫这些高波动结算项），'
            '而本套件跑在纯 CPU 上，本文的评测规模（每张桌 数百副牌）'
            '只够分辨"碾压随机基线与否"这种量级很大的差异。'
            '所有 |t| &lt; 2 的对比都应读作"暂无结论"，而不是"两者相当"。'
            '想拿到可信的强弱排序，需要把 <code>--rounds</code> 提高一个数量级，'
            '或改用配对设计（同一副牌、同一座位下逐副牌比差）。</p>')

    blocks.append(
        "<ul>"
        "<li>自产数据的天花板是启发式教师：BC 只能逼近教师，"
        "真正的超越要靠强化学习，而 RL 在本算力预算下还没跑出稳定优势。</li>"
        "<li>未实现报听 / 杀报；未做对手手牌推断、也未接 MCTS 搜索，纯策略网络直出。</li>"
        "<li>参数规模 20 万、纯 CPU 训练；换 GPU 后可放大网络并大幅增加对局数，"
        "这两件事都会直接改善上面的统计功效。</li>"
        "</ul>")

    doc = f"""<!DOCTYPE html>
<html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>幺鸡 · 贵州捉鸡麻将深度学习智能体 — 训练报告</title>
<style>
:root {{ color-scheme: light; }}
body {{ margin:0; background:#f6f7f9; color:#111827;
  font-family:-apple-system,"Segoe UI","PingFang SC","Microsoft YaHei",sans-serif;
  line-height:1.65; }}
.wrap {{ max-width:880px; margin:0 auto; padding:36px 22px 80px; }}
h1 {{ font-size:26px; margin:0 0 6px; letter-spacing:-.4px; }}
.sub {{ color:#6b7280; font-size:13.5px; margin-bottom:26px; }}
h2 {{ font-size:18px; margin:34px 0 12px; padding-bottom:7px;
  border-bottom:2px solid #e5e7eb; }}
h3 {{ font-size:14.5px; margin:20px 0 8px; color:#374151; }}
.card {{ background:#fff; border:1px solid #e5e7eb; border-radius:10px;
  padding:12px 10px 4px; margin:10px 0 16px; box-shadow:0 1px 2px rgba(0,0,0,.03); }}
.tbl {{ width:100%; border-collapse:collapse; background:#fff; font-size:13px;
  border:1px solid #e5e7eb; border-radius:8px; overflow:hidden; margin:8px 0 16px; }}
.tbl th {{ background:#f3f4f6; text-align:left; padding:8px 10px; font-weight:600;
  border-bottom:1px solid #e5e7eb; white-space:nowrap; }}
.tbl td {{ padding:7px 10px; border-bottom:1px solid #f3f4f6; }}
.tbl tr:last-child td {{ border-bottom:none; }}
.tbl td.mebot {{ font-weight:700; color:#2563eb; }}
.note {{ color:#4b5563; font-size:13px; background:#fff; border-left:3px solid #93c5fd;
  padding:10px 14px; border-radius:0 8px 8px 0; }}
code {{ background:#f3f4f6; padding:1px 5px; border-radius:4px; font-size:12.5px; }}
.dec {{ background:#fff; border:1px solid #e5e7eb; border-left:3px solid #2563eb;
  border-radius:0 8px 8px 0; padding:10px 14px; margin:8px 0; }}
.dechd {{ font-size:12.5px; color:#6b7280; margin-bottom:4px; }}
.dechand {{ font-size:13px; color:#374151; }}
.pill {{ display:inline-block; background:#eff6ff; color:#1d4ed8;
  border:1px solid #bfdbfe; border-radius:14px; padding:2px 10px; margin:5px 6px 0 0;
  font-size:12.5px; }}
.pill b {{ color:#1e3a8a; }}
</style></head><body><div class="wrap">
<h1>幺鸡 · 贵州捉鸡麻将深度学习智能体</h1>
<div class="sub">自研规则引擎 → 启发式教师生成数据 → 行为克隆 → 自我对弈强化 → 复式评测<br>
训练环境：Intel i5-11320H / 8 线程 / 16GB / 纯 CPU（无独显）</div>
{''.join(blocks)}
</div></body></html>"""

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(doc, encoding="utf-8")
    print("报告已生成:", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
