"""模型循环赛：用向量化调度跑全组合复式赛，输出各策略对战胜率与对局数据。

牌桌构成：从策略池中取所有 4 策略组合各开一张桌；
每桌打 ``--deals`` 副牌 × 4 座位轮换（复式，消牌运）。
所有桌使用同一批副牌种子——各策略面对完全相同的牌墙，横向可比。

用法：
    python scripts/tournament.py --deals 100 --workers 6
"""
from __future__ import annotations

import argparse
import itertools
import json
import multiprocessing as mp
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from zhuoji.bots import RandomBot, TEACHER_STYLES, HeuristicBot  # noqa: E402
from zhuoji.vcollect import _build_net, _resolve_game_classes, _table_worker  # noqa: E402

MODELS_DIR = ROOT / "models"
SEED0 = 90000          # 所有桌共用同一批副牌


def load_model_bank(rules: str):
    """加载参赛网络。"""
    wanted = [("rl", "rl.pt"), ("dushan_s1", "dushan_s1.pt"),
              ("dushan_s2", "dushan_s2.pt"), ("bc", "bc.pt")]
    bank_sd, nc = {}, None
    for key, fn in wanted:
        p = MODELS_DIR / fn
        if not p.exists():
            continue
        ck = torch.load(p, map_location="cpu", weights_only=False)
        nc = ck.get("net") or {"channels": 32, "blocks": 3, "hidden": 256}
        bank_sd[key] = ck["state_dict"]
    return bank_sd, nc


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--deals", type=int, default=100, help="每桌副数")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--rules", choices=["zhuoji", "dushan"], default="dushan")
    ap.add_argument("--out", default=str(ROOT / "reports" / "tournament_results.json"))
    args = ap.parse_args()

    bank_sd, nc = load_model_bank(args.rules)
    nets = list(bank_sd.keys())
    bots = ["teacher:balanced", "teacher:aggressive", "teacher:defensive", "random"]
    roster = nets + bots
    n = len(roster)
    tables = list(itertools.combinations(range(n), 4))
    print(f"参赛策略 {n} 个: {roster}")
    print(f"共 {len(tables)} 张桌 × {args.deals} 副 × 4 座位轮换 = "
          f"{len(tables) * args.deals * 4} 局")

    # 每桌座次规格（局部座位 s 的策略 = 组合[(s+shift)%4]，轮换在 run_duplicate 内做）
    def spec(idx):
        name = roster[idx]
        if name in bank_sd:
            return ("net", name)
        if name.startswith("teacher:"):
            return ("bot", name.split(":", 1)[1])
        return ("bot", name)

    seat_specs = {tid: [spec(i) for i in combo] for tid, combo in enumerate(tables)}

    game_cls, config_cls = _resolve_game_classes(args.rules)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    partial_path = out_path.with_suffix(".partial.jsonl")
    ctx = mp.get_context("spawn")
    t0 = time.time()
    n_done = 0
    with ctx.Pool(processes=args.workers) as pool:
        chunks = [list(range(len(tables)))[i::args.workers] for i in range(args.workers)]
        tasks = [(tid_list, seat_specs, bank_sd, nc, args.deals, SEED0, args.rules)
                 for tid_list in chunks if tid_list]
        all_records = []
        with open(partial_path, "w", encoding="utf-8") as pf:
            for recs in pool.imap_unordered(_table_worker, tasks):
                all_records.extend(recs)
                n_done += len(recs)
                for rec in recs:   # 逐条落盘，崩溃不丢已完成的局
                    pf.write(json.dumps(rec) + "\n")
                pf.flush()
                print(f"  进度 {n_done}/{len(tables) * args.deals * 4} 局 "
                      f"({time.time() - t0:.0f}s)", flush=True)

    # ---- 聚合（与 play_duplicate 同口径）----
    stats = defaultdict(lambda: {"games": 0, "score": 0.0, "wins": 0, "ron": 0,
                                 "tsumo": 0, "deal_in": 0, "huang": 0, "fan": 0.0})
    per_deal = defaultdict(dict)   # name -> (tid, r) -> 均值
    for tid, r, shift, deltas, rtype, winner, loser, is_tsumo, fan in all_records:
        combo = tables[tid]
        deal_acc = defaultdict(float)
        for s in range(4):
            name = roster[combo[(s + shift) % 4]]
            st = stats[name]
            st["games"] += 1
            st["score"] += deltas[s]
            deal_acc[name] += deltas[s]
            if rtype == "win" and winner == s:
                st["wins"] += 1
                st["fan"] += fan
            elif rtype == "win" and loser == s:
                st["deal_in"] += 1
            elif rtype == "huangzhuang":
                st["huang"] += 1
        if rtype == "win" and winner >= 0:
            wname = roster[combo[(winner + shift) % 4]]
            if is_tsumo:
                stats[wname]["tsumo"] += 1
            else:
                stats[wname]["ron"] += 1
        for name, acc in deal_acc.items():
            per_deal[name][(tid, r)] = acc / 4.0   # 4 次轮换均值

    out = {}
    for name, st in stats.items():
        g = max(1, st["games"])
        ds = list(per_deal[name].values())
        se = float(np.std(ds, ddof=1) / np.sqrt(len(ds))) if len(ds) > 1 else 0.0
        out[name] = {
            "games": st["games"],
            "deals": len(ds),
            "avg_score": st["score"] / g,
            "avg_score_se": se,
            "win_rate": st["wins"] / g,
            "tsumo_rate": st["tsumo"] / g,
            "ron_rate": st["ron"] / g,
            "deal_in_rate": st["deal_in"] / g,
            "huang_rate": st["huang"] / g,
            "avg_fan_when_win": st["fan"] / max(1, st["wins"]),
        }

    order = sorted(out.items(), key=lambda kv: -kv[1]["avg_score"])
    cols = ["games", "avg_score", "avg_score_se", "win_rate", "tsumo_rate",
            "ron_rate", "deal_in_rate", "huang_rate", "avg_fan_when_win"]
    print(f"\n{'策略':<20}" + "".join(f"{c:>16}" for c in cols))
    for name, s in order:
        print(f"{name:<20}" + "".join(f"{s[c]:>16.3f}" for c in cols))

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps({
        "rules": args.rules, "deals_per_table": args.deals,
        "tables": len(tables), "seed0": SEED0, "roster": roster,
        "results": out, "elapsed_s": time.time() - t0,
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n完成，用时 {time.time() - t0:.0f}s -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
