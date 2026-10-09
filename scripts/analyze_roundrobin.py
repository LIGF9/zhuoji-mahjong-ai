"""
循环赛结果分析：把 roundrobin.py 的原始输出变成胜率分布 + 两两对位强弱。

关键一点：**逐副牌配对**。
roundrobin.py 里，一个 (轮, 牌组) 下 4 个策略用同一副牌轮换座位各打一遍。
同一个牌组内、A 与 B 的得分差，是"同牌同运、只差决策"的比较，
方差比拿两个独立的平均数相减小得多（之前的对比里方差能压掉 35%~59%）。
所以这里的标准误按"牌组"这一单位算，而不是按局。

而牌组的枚举顺序是确定的：C(k,4) 按组合顺序生成、每轮重复一遍，
因此每个策略的单元序列可以精确还原成 (轮, 牌组) 的键，两条序列就能对齐做配对 t 检验。

用法::

    python scripts/analyze_roundrobin.py
    python scripts/analyze_roundrobin.py --in reports/roundrobin.json
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402


def _se(vals) -> float:
    a = np.asarray(vals, dtype=float)
    if a.size < 2:
        return 0.0
    return float(a.std(ddof=1) / math.sqrt(a.size))


def _unit_index(agents: list[str], reps: int) -> dict[str, list[tuple[int, int]]]:
    """还原每个策略的复式单元顺序：[(轮, 牌组序号), ...]。

    必须与 roundrobin.py 的写入顺序严格一致：
    外层轮、内层枚举组合、某个策略只在包含它的组合里出现。
    """
    k = len(agents)
    blocks = list(itertools.combinations(range(k), 4))
    out: dict[str, list[tuple[int, int]]] = {a: [] for a in agents}
    for rep in range(reps):
        for bi, blk in enumerate(blocks):
            for i in blk:
                out[agents[i]].append((rep, bi))
    return out


def analyze(path: Path) -> dict:
    d = json.loads(Path(path).read_text())
    agents = d["agents"]
    reps = d["reps"]
    agg, units = d["agg"], d["units"]
    G = d["games_total"]
    k = len(agents)

    # --- 1. 每个策略的胜率分布 -------------------------------------------
    rows = []
    for a in agents:
        g = max(1.0, agg[a]["games"])
        u = units[a]
        wins = agg[a].get("wins", 0.0)
        row = {
            "name": a,
            "games": int(g),
            "units": len(u["score"]),
            "win_rate": wins / g,
            "win_rate_se": _se(u["wins"]),
            "avg_score": agg[a]["score"] / g,
            "avg_score_se": _se(u["score"]),
            "tsumo_rate": agg[a].get("tsumo", 0.0) / g,
            "tsumo_rate_se": _se(u["tsumo"]),
            "ron_rate": agg[a].get("ron", 0.0) / g,
            "deal_in_rate": agg[a].get("deal_in", 0.0) / g,
            "deal_in_rate_se": _se(u["deal_in"]),
            "huang_rate": agg[a].get("huang", 0.0) / g,
            "huang_rate_se": _se(u["huang"]),
            "chongfeng_rate": agg[a].get("chongfeng", 0.0) / g,
            "zeren_rate": agg[a].get("zeren", 0.0) / g,
            "avg_fan_when_win": (agg[a].get("fan", 0.0) / max(1.0, wins))
                                if wins else 0.0,
        }
        # 胡牌率的三点区间（Wilson，避免小比率时的正态近似失真）
        n, p = g, row["win_rate"]
        z = 1.96
        den = 1 + z * z / n
        ctr = (p + z * z / (2 * n)) / den
        half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
        row["win_rate_ci"] = [max(0.0, ctr - half), min(1.0, ctr + half)]
        rows.append(row)
    rows.sort(key=lambda r: -r["avg_score"])

    # --- 2. 逐副牌配对：两两得分差 ---------------------------------------
    order = _unit_index(agents, reps)
    seq = {}
    for a in agents:
        vals = units[a]["score"]
        assert len(vals) == len(order[a]), f"{a} 单元数与设计不符"
        seq[a] = dict(zip(order[a], vals))

    pair = {}
    for a in agents:
        pair[a] = {}
        for b in agents:
            if a == b:
                continue
            common = sorted(set(seq[a]) & set(seq[b]))
            diffs = [seq[a][key] - seq[b][key] for key in common]
            arr = np.asarray(diffs, dtype=float)
            # 逐副牌配对，可以去掉"这一轮牌整体偏谁"的影响
            se = float(arr.std(ddof=1) / math.sqrt(arr.size)) if arr.size > 1 else 0.0
            mean = float(arr.mean()) if arr.size else 0.0
            t = mean / se if se > 0 else 0.0
            wa = d["pair_wins"][a].get(b, 0)
            wb = d["pair_wins"][b].get(a, 0)
            pair[a][b] = {
                "mean_diff": mean,
                "se": se,
                "t": t,
                "n_blocks": len(common),
                "n_games": d["n_cooc"][a].get(b, 0),
                "h2h_win_share": (wa / (wa + wb)) if (wa + wb) else 0.5,
                "wins_a": wa,
                "wins_b": wb,
            }

    # --- 3. 体检 -----------------------------------------------------------
    total = sum(agg[a]["score"] for a in agents)
    coocs = [d["n_cooc"][a][b] for a in agents for b in agents if a != b]
    games_each = int(agg[agents[0]]["games"])
    checks = {
        "零和_各家总得分求和": round(total, 6),
        "出场次数各策略是否相等": len({int(agg[a]["games"]) for a in agents}) == 1,
        "出场次数": games_each,
        "每家应出场": G * 4 / len(agents),
        "两两同桌次数_min": int(min(coocs)),
        "两两同桌次数_max": int(max(coocs)),
        "两两同桌次数_理论值": d["reps"] * math.comb(len(agents) - 2, 2) * 4,
        "总胡牌次数": int(sum(agg[a].get("wins", 0.0) for a in agents)),
        "总牌局": G,
        "备注_黄庄次数": "各家黄庄次数天然不等：黄庄取决于同桌构成，"
                        "含 random 的牌桌更容易黄庄，非异常",
    }

    # --- 4. 确定优于的对手（|t|>2） ---------------------------------------
    beats = []
    for a in agents:
        for b in agents:
            if a >= b:
                continue
            rec = pair[a][b]
            if rec["t"] >= 2.0:
                beats.append({"winner": a, "loser": b, "diff": rec["mean_diff"],
                              "t": rec["t"]})
            elif rec["t"] <= -2.0:
                beats.append({"winner": b, "loser": a, "diff": -rec["mean_diff"],
                              "t": -rec["t"]})
    beats.sort(key=lambda r: -r["t"])
    # 排名相邻的两个策略之间是否"分得开"
    ladder = []
    for i in range(len(rows) - 1):
        a, b = rows[i]["name"], rows[i + 1]["name"]
        rec = pair[a][b]
        ladder.append({"higher": a, "lower": b,
                       "diff": rec["mean_diff"], "t": rec["t"],
                       "separated": abs(rec["t"]) >= 2.0})

    # 座位偏差体检：各家在各座位的平均分应当接近
    seat = d["seat_score"]
    seat_check = {a: [round(v / (checks["出场次数"] / 4), 2) for v in seat[a]]
                  for a in agents}

    return {
        "design": {"agents": agents, "reps": reps, "games_total": G,
                   "seed": d["seed"], "tag": d.get("tag", ""),
                   "elapsed": d.get("elapsed")},
        "ranking": rows,
        "pairwise": pair,
        "validation": checks,
        "seat_avg_score": seat_check,
        "significant_pairs": beats,
        "ladder": ladder,
    }


def fmt_table(rows: list[dict]) -> str:
    out = []
    out.append(f"{'排名':<4}{'策略':<24}{'局数':>6}{'胡牌率':>9}{'±':>7}"
               f"{'每局得分':>10}{'±':>7}{'自摸':>8}{'点炮':>8}{'黄庄':>8}{'冲锋鸡':>8}")
    for i, r in enumerate(rows, 1):
        out.append(f"{i:<4}{r['name']:<24}{r['games']:>6}"
                   f"{r['win_rate'] * 100:>8.2f}%{r['win_rate_se'] * 100:>6.2f}"
                   f"{r['avg_score']:>10.3f}{r['avg_score_se']:>7.3f}"
                   f"{r['tsumo_rate'] * 100:>7.2f}%{r['deal_in_rate'] * 100:>7.2f}%"
                   f"{r['huang_rate'] * 100:>7.2f}%{r['chongfeng_rate'] * 100:>7.2f}%")
    return "\n".join(out)


def fmt_matrix(pair: dict, key: str, agents: list[str], fmt: str) -> str:
    abbr = [a.replace("teacher:", "t:")[:9] for a in agents]
    out = [f"{'':<11}" + "".join(f"{x:>10}" for x in abbr)]
    for a in agents:
        cells = []
        for b in agents:
            if a == b:
                cells.append(f"{'—':>10}")
            else:
                cells.append(f"{format(pair[a][b][key], fmt):>10}")
        out.append(f"{a.replace('teacher:', 't:')[:10]:<11}" + "".join(cells))
    return "\n".join(out)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="src",
                    default=str(ROOT / "reports" / "roundrobin.json"))
    ap.add_argument("--out",
                    default=str(ROOT / "reports" / "roundrobin_summary.json"))
    args = ap.parse_args()

    res = analyze(Path(args.src))
    agents = res["design"]["agents"]
    rows = res["ranking"]

    print("=" * 100)
    print("全策略循环赛 · 胜率分布（四人混战，BIBD 全覆盖：每个策略出场次数相同、"
          "每对策略同桌次数相同）")
    print("=" * 100)
    print(f"总牌局 {res['design']['games_total']}  轮数 {res['design']['reps']}  "
          f"标签 {res['design']['tag']}")
    print()
    print(fmt_table(rows))
    print()

    print("=" * 100)
    print("两两对位 · 每局得分差（同一副牌配对，正=左边更强）")
    print("=" * 100)
    print(fmt_matrix(res["pairwise"], "mean_diff", agents, "+7.3f"))
    print()
    print("=" * 100)
    print("两两对位 · 同桌时相对胜率（胡牌次数占比，0.5=五五开）")
    print("=" * 100)
    print(fmt_matrix(res["pairwise"], "h2h_win_share", agents, "7.3f"))
    print()
    print("=" * 100)
    print("两两对位 · t 值（|t|>2 才算证据充分）")
    print("=" * 100)
    print(fmt_matrix(res["pairwise"], "t", agents, "+7.2f"))
    print()
    print("=" * 100)
    print("确定优于的对位（逐副牌配对 t 检验 |t|>2）")
    print("=" * 100)
    for r in res["significant_pairs"]:
        print(f"  {r['winner']:<24} 胜 {r['loser']:<24} "
              f"每局多 {r['diff']:+.3f} 分  t={r['t']:.2f}")
    print()
    print("排名相邻的两者是否真的分得开：")
    for r in res["ladder"]:
        mark = "分得开" if r["separated"] else "**分不开**（差异不显著，名次只是噪声）"
        print(f"  {r['higher']:<24} vs {r['lower']:<24} "
              f"{r['diff']:+.3f} 分  t={r['t']:+.2f}  {mark}")
    print()
    print("体检：")
    for k, v in res["validation"].items():
        print(f"  {k}: {v}")
    print(" 座位平均分（应四座接近，若差异大说明座位有系统优势）:")
    for a, v in res["seat_avg_score"].items():
        print(f"   {a:<24}{v}")

    Path(args.out).write_text(json.dumps(res, ensure_ascii=False, indent=1))
    print(f"\n汇总 -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
