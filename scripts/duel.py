"""两代模型高精度双人对位赛：同副牌、座位轮换、双方各占两个座位。

用法::
    python scripts/duel.py --a dushan_s6 --b dushan_s7 --deals 600
"""
from __future__ import annotations

import argparse
import statistics as st
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from zhuoji.agent import NetAgent, play_duplicate  # noqa: E402
from zhuoji.dushan import DushanConfig, DushanGame  # noqa: E402
from zhuoji.net import NetConfig, build_model  # noqa: E402


def load(stem: str):
    ck = torch.load(ROOT / "models" / f"{stem}.pt", map_location="cpu", weights_only=False)
    nc = ck.get("net") or {"channels": 48, "blocks": 8, "hidden": 256}
    m = build_model(NetConfig(nc["channels"], nc["blocks"], nc["hidden"]))
    m.load_state_dict(ck["state_dict"])
    return m.eval()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True)
    ap.add_argument("--b", required=True)
    ap.add_argument("--deals", type=int, default=600)
    ap.add_argument("--seed", type=int, default=31337)
    ap.add_argument("--threads", type=int, default=8)
    args = ap.parse_args()

    torch.set_num_threads(args.threads)
    ma, mb = load(args.a), load(args.b)

    def make():
        # A 坐 0/2，B 坐 1/3；复式座位轮换由 play_duplicate 负责，
        # 每副牌 4 个座次轮换后两方各坐遍每个座位，座位偏差被抵消。
        return [NetAgent(ma, seed=1, temperature=0.0, name=args.a),
                NetAgent(mb, seed=2, temperature=0.0, name=args.b),
                NetAgent(ma, seed=3, temperature=0.0, name=args.a),
                NetAgent(mb, seed=4, temperature=0.0, name=args.b)]

    # 分块跑并输出进度：每 50 副打印一次累计配对净分差，避免长赛程黑箱
    import time as _time

    chunk = 50
    all_da: list[float] = []
    all_db: list[float] = []
    chunk_res: list[tuple[dict, dict]] = []
    t0 = _time.time()
    done = 0

    while done < args.deals:
        n = min(chunk, args.deals - done)
        res, per_deal = play_duplicate(
            make, n, seed0=args.seed + done, return_per_deal=True,
            game_factory=lambda s: DushanGame(DushanConfig(), seed=s))
        chunk_res.append((res[args.a], res[args.b]))
        all_da.extend(per_deal[args.a])
        all_db.extend(per_deal[args.b])
        done += n
        paired_now = [x - y for x, y in zip(all_da, all_db)]
        m_now = st.mean(paired_now)
        se_now = st.stdev(paired_now) / len(paired_now) ** 0.5
        el = _time.time() - t0
        eta = el / done * (args.deals - done) / 60
        print(f"[进度 {done}/{args.deals}] 净分差 {m_now:+.3f} ± {1.96 * se_now:.3f}"
              f"  已用 {el / 60:.1f} 分，预计剩余 {eta:.0f} 分", flush=True)

    # 跨分块聚合全样本速率（games 加权；wins 由 rate*games 还原，均为精确整数比）
    def agg(pairs):
        g = sum(x["games"] for x in pairs)
        wins = sum(x["win_rate"] * x["games"] for x in pairs)
        fan_sum = sum(x["avg_fan_when_win"] * x["win_rate"] * x["games"] for x in pairs)
        return {
            "games": g,
            "avg_score": sum(x["avg_score"] * x["games"] for x in pairs) / g,
            "win_rate": wins / g,
            "tsumo_rate": sum(x["tsumo_rate"] * x["games"] for x in pairs) / g,
            "deal_in_rate": sum(x["deal_in_rate"] * x["games"] for x in pairs) / g,
            "avg_fan_when_win": fan_sum / max(1.0, wins),
            "bigfan15_rate": sum(x["bigfan15_rate"] * x["games"] for x in pairs) / g,
            "bigfan20_rate": sum(x["bigfan20_rate"] * x["games"] for x in pairs) / g,
        }

    ra, rb = agg([p[0] for p in chunk_res]), agg([p[1] for p in chunk_res])

    print(f"{'':<14}{'局数':>6}{'均分':>9}{'胡牌率':>8}{'自摸率':>8}{'放铳率':>8}"
          f"{'胡时均番':>9}{'大牌15+':>9}{'大牌20+':>9}")
    for name, r in ((args.a, ra), (args.b, rb)):
        print(f"{name:<14}{r['games']:>6}{r['avg_score']:>+9.3f}"
              f"{r['win_rate']:>8.3f}{r['tsumo_rate']:>8.3f}{r['deal_in_rate']:>8.3f}"
              f"{r['avg_fan_when_win']:>9.1f}{r['bigfan15_rate']:>9.3f}{r['bigfan20_rate']:>9.3f}")
    paired = [x - y for x, y in zip(all_da, all_db)]
    mean = st.mean(paired)
    se = st.stdev(paired) / len(paired) ** 0.5
    print(f"\n配对净分差 {args.a} − {args.b} = {mean:+.3f} ± {1.96 * se:.3f}"
          f"（95% CI，{len(paired)} 副牌配对）")
    verdict = "显著优于" if mean - 1.96 * se > 0 else (
        "显著劣于" if mean + 1.96 * se < 0 else "与...无显著差异")
    print(f"结论：{args.a} {verdict} {args.b}（α=0.05）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
