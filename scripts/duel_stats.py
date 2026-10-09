"""详细指标对位统计：主视角模型 vs 一/三种对手，逐项统计胜率/点炮率/大牌率/冲鸡率(成功率)/包鸡率等。

对手规格（``--opp``）用小节描述，``|`` 分隔三个对手座位：
  ``net:dushan_big_s7``                                   三家同一对手
  ``net:dushan_big_s7|net:dushan_big_s3|teacher:tenpai_rush``  三家各不同

主视角固定占一席、三家为对手；每副牌主视角轮换坐遍 4 个座位（消除座位偏差），
所以「局数 = 副数 × 4」。

用法::
    python scripts/duel_stats.py --me dushan_big_s11 --deals 500 \
        --opp net:dushan_big_s7 --tag s11_vs_s7
"""
from __future__ import annotations

import argparse
import json
import statistics as st
import sys
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from zhuoji.agent import NetAgent  # noqa: E402
from zhuoji.bots import TEACHER_STYLES, HeuristicBot  # noqa: E402
from zhuoji.dushan import DushanConfig, DushanGame  # noqa: E402
from zhuoji.net import NetConfig, build_model  # noqa: E402

TEACHER_CN = {
    "balanced": "均衡", "aggressive": "激进", "defensive": "保守",
    "chicken_lover": "囤幺鸡", "tenpai_rush": "先听（抢听牌）", "gambler": "赌徒",
}


def load(stem: str):
    ck = torch.load(ROOT / "models" / f"{stem}.pt", map_location="cpu", weights_only=False)
    nc = ck.get("net") or {"channels": 48, "blocks": 8, "hidden": 256}
    m = build_model(NetConfig(nc["channels"], nc["blocks"], nc["hidden"]))
    m.load_state_dict(ck["state_dict"])
    return m.eval()


class Opp:
    """对手工厂：kind=net（权重）或 kind=teacher（启发式风格）。"""

    def __init__(self, kind: str, key: str):
        self.kind, self.key = kind, key
        if kind == "net":
            self.model = load(key)
            self.label = key
        elif kind == "teacher":
            if key not in TEACHER_STYLES:
                raise SystemExit(f"未知教师风格 {key}，可选 {list(TEACHER_STYLES)}")
            self.label = f"教师·{TEACHER_CN.get(key, key)}"
        else:
            raise SystemExit(f"未知对手类型 {kind}（用 net: 或 teacher:）")

    def make(self, seed: int) -> object:
        if self.kind == "net":
            return NetAgent(self.model, seed=seed, temperature=0.0, name=self.label)
        b = HeuristicBot(seed=seed, weights=TEACHER_STYLES[self.key])
        b.name = self.label
        return b


def parse_opp(spec: str) -> list[Opp]:
    parts = [p.strip() for p in spec.split("|") if p.strip()]
    if len(parts) not in (1, 3):
        raise SystemExit("--opp 需 1 段（三家同款）或 3 段（各不同），用 | 分隔")
    outs = []
    for p in parts:
        kind, _, key = p.partition(":")
        outs.append(Opp(kind, key))
    if len(outs) == 1:
        outs = outs * 3
    return outs


def blank() -> dict:
    return {
        "games": 0, "score": 0.0,
        "wins": 0, "tsumo": 0, "ron": 0, "deal_in": 0, "huang": 0,
        "fan": 0.0, "bigfan15": 0, "bigfan20": 0,
        "tenpai": 0,
        "cf_try": 0, "cf_ok": 0, "cf_games": 0,   # 冲锋鸡：尝试 / 成功（未被碰杠走）/ 有冲鸡的局
        "hj_try": 0, "hj_ok": 0,          # 横鸡：尝试 / 成功
        "dt_plain": 0,                    # 打出的普通鸡（幺鸡）
        "bao_ji_games": 0, "bao_ji_n": 0, "bao_ji_val": 0.0,
        "bao_gang_games": 0, "bao_gang_n": 0,
    }


def feed(s: dict, seat: int, res: dict, game) -> None:
    """把一局结果并入 seat 的统计。"""
    s["games"] += 1
    s["score"] += res["deltas"][seat]
    ty = res.get("type")
    if ty == "win":
        w = res["winner"]
        if w == seat:
            s["wins"] += 1
            s["fan"] += res.get("fan", 0)
            if res.get("fan", 0) >= 15:
                s["bigfan15"] += 1
            if res.get("fan", 0) >= 20:
                s["bigfan20"] += 1
            if res.get("is_tsumo"):
                s["tsumo"] += 1
            elif res.get("loser") is not None:
                s["ron"] += 1
        if res.get("loser") == seat and not res.get("is_tsumo"):
            s["deal_in"] += 1
    else:
        s["huang"] += 1
    if seat in (res.get("tenpai") or []):
        s["tenpai"] += 1

    # 打出的鸡：冲锋鸡 / 横鸡 记账（ji_events = (打出者, 基础分, 类型名)）
    # 被碰/杠走的（claimed_ji = (打出者, 碰杠者, 类型名, 是否明杠)）不算成功
    claimed = defaultdict(int)
    for ev in getattr(game, "claimed_ji", []) or []:
        if int(ev[0]) == seat:
            claimed[str(ev[2])] += 1
    had_cf = False
    for ev in getattr(game, "ji_events", []) or []:
        if int(ev[0]) != seat:
            continue
        tag = str(ev[2])
        if tag == "冲锋鸡":
            s["cf_try"] += 1
            had_cf = True
            if claimed.get(tag, 0) > 0:
                claimed[tag] -= 1
            else:
                s["cf_ok"] += 1
        elif tag == "横鸡":
            s["hj_try"] += 1
            if claimed.get(tag, 0) > 0:
                claimed[tag] -= 1
            else:
                s["hj_ok"] += 1
        else:
            s["dt_plain"] += 1
    if had_cf:
        s["cf_games"] += 1

    # 包鸡 / 包杠：结算明细里挂在该座位名下的项（责任方口径）
    # detail 元素 = (座位, 项目名, 说明, 分值)
    for d in (res.get("detail") or []):
        if len(d) < 4 or int(d[0]) != seat:
            continue
        if d[1] == "包鸡":
            s["bao_ji_n"] += 1
            s["bao_ji_val"] += float(d[3])
        elif d[1] == "包杠":
            s["bao_gang_n"] += 1
    # 「包鸡/包杠局数」按局去重（同一局可能有多个分项）
    if any(len(d) >= 4 and int(d[0]) == seat and d[1] == "包鸡"
           for d in (res.get("detail") or [])):
        s["bao_ji_games"] += 1
    if any(len(d) >= 4 and int(d[0]) == seat and d[1] == "包杠"
           for d in (res.get("detail") or [])):
        s["bao_gang_games"] += 1


def rate(a: float, b: float) -> float:
    return a / b if b else 0.0


def summarize(s: dict, deals: int) -> dict:
    g = max(1, s["games"])
    return {
        "games": s["games"],
        "deals": deals,
        "avg_score": s["score"] / g,
        "win_rate": rate(s["wins"], g),
        "tsumo_rate": rate(s["tsumo"], g),
        "ron_rate": rate(s["ron"], g),
        "deal_in_rate": rate(s["deal_in"], g),
        "huang_rate": rate(s["huang"], g),
        "tenpai_rate": rate(s["tenpai"], g),
        "bigfan15_rate": rate(s["bigfan15"], g),
        "bigfan20_rate": rate(s["bigfan20"], g),
        "avg_fan_when_win": rate(s["fan"], max(1, s["wins"])),
        "cf_try_per_game": rate(s["cf_try"], g),
        "cf_games_rate": rate(s["cf_games"], g),
        "cf_success_rate": rate(s["cf_ok"], s["cf_try"]),
        "cf_try_n": s["cf_try"], "cf_ok_n": s["cf_ok"],      # 原始计数（小样本时看 n）
        "hj_try_n": s["hj_try"], "hj_ok_n": s["hj_ok"],
        "bao_ji_n": s["bao_ji_n"], "bao_ji_games_n": s["bao_ji_games"],
        "hj_try_per_game": rate(s["hj_try"], g),
        "hj_success_rate": rate(s["hj_ok"], s["hj_try"]),
        "plain_ji_discard_per_game": rate(s["dt_plain"], g),
        "bao_ji_games_rate": rate(s["bao_ji_games"], g),
        "bao_ji_per_game": rate(s["bao_ji_n"], g),
        "bao_ji_val_per_game": rate(s["bao_ji_val"], g),
        "bao_ji_val_per_case": rate(s["bao_ji_val"], max(1, s["bao_ji_n"])),
        "bao_gang_games_rate": rate(s["bao_gang_games"], g),
    }


def run_matchup(me_stem, me_model, opps: list[Opp], deals: int, seed0: int,
                threads: int, tag: str) -> dict:
    me_label = me_stem
    stats_me, stats_op = blank(), blank()
    per_deal_me, per_deal_op = [], []
    t0 = time.time()
    for r in range(deals):
        d_me, d_op = [], []
        for shift in range(4):
            # 角色顺序固定为 [主视角, 对手×3]，整体旋转 shift 个座位 →
            # 每副牌主视角坐遍 4 个座位，座位偏差被抵消
            agents = [NetAgent(me_model, seed=1000 + shift, temperature=0.0, name=me_label)]
            agents += [o.make(seed=2000 + 7 * i + shift) for i, o in enumerate(opps)]
            seat_agents = [None] * 4
            for slot, ai in enumerate([(i + shift) % 4 for i in range(4)]):
                seat_agents[slot] = agents[ai]
            game = DushanGame(DushanConfig(), seed=seed0 + r * 1013 + shift)
            res = game.play(seat_agents)
            me_seat = seat_agents.index(agents[0])
            feed(stats_me, me_seat, res, game)
            for slot in range(4):
                if slot != me_seat:
                    feed(stats_op, slot, res, game)
            d_me.append(res["deltas"][me_seat])
            d_op.append(st.mean([res["deltas"][x] for x in range(4) if x != me_seat]))
        per_deal_me.append(st.mean(d_me))
        per_deal_op.append(st.mean(d_op))
        if (r + 1) % 50 == 0 or r + 1 == deals:
            el = time.time() - t0
            print(f"  [{tag}] {r + 1}/{deals} 副  已用 {el / 60:.1f} 分"
                  f"  预计剩余 {el / (r + 1) * (deals - r - 1) / 60:.0f} 分", flush=True)

    sm = summarize(stats_me, deals)
    so = summarize(stats_op, deals)
    paired = [a - b for a, b in zip(per_deal_me, per_deal_op)]
    mean = st.mean(paired)
    se = st.stdev(paired) / len(paired) ** 0.5 if len(paired) > 1 else 0.0
    ci = 1.96 * se
    return {
        "tag": tag,
        "me": me_label,
        "opp": [o.label for o in opps],
        "deals": deals,
        "me_stats": sm,
        "opp_stats": so,
        "paired_mean": mean,
        "paired_ci95": ci,
        "verdict": ("显著优于" if mean - ci > 0 else
                    "显著劣于" if mean + ci < 0 else "无显著差异"),
        "elapsed_min": (time.time() - t0) / 60,
    }


def fmt_pct(x: float) -> str:
    return f"{x * 100:.2f}%"


def report(res: dict) -> str:
    m, o = res["me_stats"], res["opp_stats"]
    lines = []
    lines.append(f"### {res['me']} vs {' + '.join(res['opp'])}"
                 f"（{res['deals']} 副 × 4 座位：主视角 {m['games']} 局，"
                 f"三家对手合计 {o['games']} 局）\n")
    rows = [
        ("平均分", lambda s: f"{s['avg_score']:+.3f}", lambda s: f"{s['avg_score']:+.3f}"),
        ("胜率", lambda s: fmt_pct(s["win_rate"]), lambda s: fmt_pct(s["win_rate"])),
        ("自摸率", lambda s: fmt_pct(s["tsumo_rate"]), lambda s: fmt_pct(s["tsumo_rate"])),
        ("点胡率", lambda s: fmt_pct(s["ron_rate"]), lambda s: fmt_pct(s["ron_rate"])),
        ("点炮率", lambda s: fmt_pct(s["deal_in_rate"]), lambda s: fmt_pct(s["deal_in_rate"])),
        ("黄庄率", lambda s: fmt_pct(s["huang_rate"]), lambda s: fmt_pct(s["huang_rate"])),
        ("终局听牌率", lambda s: fmt_pct(s["tenpai_rate"]), lambda s: fmt_pct(s["tenpai_rate"])),
        ("胡时均番", lambda s: f"{s['avg_fan_when_win']:.2f}", lambda s: f"{s['avg_fan_when_win']:.2f}"),
        ("大牌率(15+)", lambda s: fmt_pct(s["bigfan15_rate"]), lambda s: fmt_pct(s["bigfan15_rate"])),
        ("大牌率(20+)", lambda s: fmt_pct(s["bigfan20_rate"]), lambda s: fmt_pct(s["bigfan20_rate"])),
        ("冲鸡/局", lambda s: f"{s['cf_try_per_game']:.3f}", lambda s: f"{s['cf_try_per_game']:.3f}"),
        ("有冲鸡的局", lambda s: fmt_pct(s["cf_games_rate"]), lambda s: fmt_pct(s["cf_games_rate"])),
        ("冲鸡成功率", lambda s: (f"{fmt_pct(s['cf_success_rate'])}（n={s['cf_ok_n']}/{s['cf_try_n']}）"
                              if s["cf_try_n"] else "—"),
         lambda s: (f"{fmt_pct(s['cf_success_rate'])}（n={s['cf_ok_n']}/{s['cf_try_n']}）"
                    if s["cf_try_n"] else "—")),
        ("横鸡/局", lambda s: f"{s['hj_try_per_game']:.3f}", lambda s: f"{s['hj_try_per_game']:.3f}"),
        ("横鸡成功率", lambda s: (f"{fmt_pct(s['hj_success_rate'])}（n={s['hj_ok_n']}/{s['hj_try_n']}）"
                              if s["hj_try_n"] else "—"),
         lambda s: (f"{fmt_pct(s['hj_success_rate'])}（n={s['hj_ok_n']}/{s['hj_try_n']}）"
                    if s["hj_try_n"] else "—")),
        ("普通鸡弃牌/局", lambda s: f"{s['plain_ji_discard_per_game']:.3f}",
         lambda s: f"{s['plain_ji_discard_per_game']:.3f}"),
        ("包鸡局占比", lambda s: fmt_pct(s["bao_ji_games_rate"]), lambda s: fmt_pct(s["bao_ji_games_rate"])),
        ("包鸡次数/局", lambda s: f"{s['bao_ji_per_game']:.3f}", lambda s: f"{s['bao_ji_per_game']:.3f}"),
        ("包鸡失分/局", lambda s: f"{s['bao_ji_val_per_game']:.3f}", lambda s: f"{s['bao_ji_val_per_game']:.3f}"),
        ("包杠局占比", lambda s: fmt_pct(s["bao_gang_games_rate"]), lambda s: fmt_pct(s["bao_gang_games_rate"])),
    ]
    lines.append("| 指标 | 主视角 | 三家对手均值 |")
    lines.append("|---|---|---|")
    for name, fme, fop in rows:
        lines.append(f"| {name} | {fme(m)} | {fop(o)} |")
    lines.append("")
    lines.append(f"逐副配对净分差（主视角 − 对手三家均值）= "
                 f"{res['paired_mean']:+.3f} ± {res['paired_ci95']:.3f}（95% CI，"
                 f"{res['deals']} 副）→ **{res['verdict']}**")
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--me", default="dushan_big_s11")
    ap.add_argument("--deals", type=int, default=500)
    ap.add_argument("--seed", type=int, default=20261009)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--opp", action="append", required=True,
                    help="对手规格，可重复；net:xxx 或 teacher:tenpai_rush，| 分隔三家")
    ap.add_argument("--out", default="reports/duel_stats.json")
    ap.add_argument("--md", default="reports/duel_stats.md")
    args = ap.parse_args()

    torch.set_num_threads(args.threads)
    me_model = load(args.me)
    results = []
    for i, spec in enumerate(args.opp):
        opps = parse_opp(spec)
        tag = f"{args.me} vs " + " + ".join(o.label for o in opps)
        print(f"\n=== {tag}（{args.deals} 副）===", flush=True)
        r = run_matchup(args.me, me_model, opps, args.deals, args.seed + i * 7777,
                        args.threads, tag)
        results.append(r)
        print(report(r), flush=True)

    # 落盘（per_deal 序列另存，便于后续复核）
    slim = []
    for r in results:
        d = {k: v for k, v in r.items() if not k.startswith("per_deal")}
        slim.append(d)
    (ROOT / args.out).write_text(json.dumps(slim, ensure_ascii=False, indent=2),
                                encoding="utf-8")
    md = "# S11 详细指标对位统计\n\n"
    md += (f"- 主视角模型：`{args.me}`\n- 每场 {args.deals} 副 × 4 座位轮换 = "
           f"{args.deals * 4} 局/方\n- 生成时间：{time.strftime('%Y-%m-%d %H:%M')}\n\n")
    for r in results:
        md += report(r) + "\n"
    (ROOT / args.md).write_text(md, encoding="utf-8")
    print(f"\n报告已写入 {args.md} 与 {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
