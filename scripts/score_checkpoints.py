"""多检查点「共用牌池」对位评分：给模型选点提供可信指标。

为什么需要这个脚本
------------------
训练内置的评测是对**固定的三位启发式教师**打 ``--eval-rounds`` 副（默认 40 副
× 4 座位 = 160 局），实测 ``avg_score_se ≈ 0.97``，即 95% 置信区间 **±1.9 分**；
而真实的代际提升只有 **~0.3 分**（S11 vs S7，2400 副对位赛 = +0.319 ± 0.288）。
拿这个指标做 ``argmax`` 选点等于在噪声里抽签，还会因 winner's curse 系统性高估
被选中的那个检查点——这正是 ``s7`` 评测分 8.60 高于 ``s11`` 7.231、实际却更弱的原因。

本脚本的做法
------------
1. **共用牌池（common random numbers）**：所有检查点打**完全相同**的牌——第 r 副
   牌的 seed 只由 ``seed0 + r * 1013`` 决定，与检查点无关。
2. **共用对手**：三个对手也完全固定，其随机种子只由「第几副牌 + 第几个座位 +
   对手名」推出，与检查点无关。于是同一副牌上的两个检查点面对的是同一局面。
3. **复式轮换**：每副牌让检查点把 4 个座位各坐一遍再取均值。**四个座位共用同一个
   牌 seed**（牌不变、只换座位），这才是真正的复式；``duel.py`` 里把 shift 混进了
   seed，四轮实际吃的是四副不同的牌，只是加了噪声、不产生偏倚。
4. 于是「同一副牌上两个检查点的分差」是一个**配对观测**，方差远低于各自的分值水平。

输出
----
* 各检查点的分值水平与 95% CI（这只是参考，因为它含对手/座位固定带来的偏倚）；
* **相对锚点检查点的配对净分差**排名与 CI —— 选点应该看这个。

用法::
    python scripts/score_checkpoints.py \
        --ckpts dushan_big_s11,dushan_big_s12_it050,dushan_big_s12_it100 \
        --ref "net:dushan_big_s7|net:dushan_big_s3|teacher:tenpai_rush" \
        --deals 600

    # 用小样本快速自检（例如复现 S11 vs S7 的方向）
    python scripts/score_checkpoints.py --ckpts dushan_big_s11 --ref "dushan_big_s7|dushan_big_s7|dushan_big_s7" \
        --deals 60 --out reports/_selfcheck.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import statistics as st
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from zhuoji.agent import NetAgent  # noqa: E402
from zhuoji.bots import TEACHER_STYLES, HeuristicBot  # noqa: E402
from zhuoji.dushan import DushanConfig, DushanGame  # noqa: E402
from zhuoji.net import NetConfig, build_model  # noqa: E402

_MODELS: dict[str, object] = {}


def load(stem: str):
    """按裸文件名（不带 .pt）加载模型；同一 stem 只读一次。"""
    if stem in _MODELS:
        return _MODELS[stem]
    path = ROOT / "models" / f"{stem}.pt"
    if not path.exists():
        raise SystemExit(f"找不到模型 {path}")
    ck = torch.load(path, map_location="cpu", weights_only=False)
    nc = ck.get("net") or {"channels": 48, "blocks": 8, "hidden": 256}
    m = build_model(NetConfig(nc["channels"], nc["blocks"], nc["hidden"]))
    m.load_state_dict(ck["state_dict"])
    m = m.eval()
    _MODELS[stem] = m
    return m


def _seed(*parts) -> int:
    """由名字推出确定性种子：对手行为跨检查点完全一致（共用随机数的前提）。"""
    h = hashlib.sha256("#".join(str(p) for p in parts).encode("utf-8")).hexdigest()
    return int(h[:8], 16) % (10 ** 9)


def make_opponent(spec: str, seat_seed: int):
    if spec.startswith("teacher:"):
        style = spec.split(":", 1)[1]
        if style not in TEACHER_STYLES:
            raise SystemExit(f"未知教师风格 {style!r}，可选：{list(TEACHER_STYLES)}")
        b = HeuristicBot(seed=seat_seed, weights=TEACHER_STYLES[style])
        b.name = spec
        return b
    stem = spec.split(":", 1)[1] if spec.startswith("net:") else spec
    return NetAgent(load(stem), seed=seat_seed, temperature=0.0, name=spec)


def _wrap(bot):
    def f(game):
        return bot.choose(game)
    return f


def summarize(per_deal: dict[str, list[float]], anchor: str) -> list[dict]:
    """把逐副分值汇总成「水平 + 相对锚点的配对净分差」，并按后者降序排名。

    配对是关键：``per_deal`` 里两个检查点在**第 r 副牌**上的观测是成对的
    （同一副牌、同一批对手），所以 ``d_i = level_a[r] − level_anchor[r]``
    的方差远小于各自水平值的方差，比较才有分辨力。
    """
    rows = []
    base = per_deal[anchor]
    for c, d in per_deal.items():
        lvl = st.mean(d)
        lvl_se = st.stdev(d) / len(d) ** 0.5 if len(d) > 1 else 0.0
        diff = [x - y for x, y in zip(d, base)]
        dm = st.mean(diff)
        dse = st.stdev(diff) / len(diff) ** 0.5 if len(diff) > 1 else 0.0
        rows.append({
            "ckpt": c,
            "deals": len(d),
            "level": round(lvl, 4),
            "level_ci95": round(1.96 * lvl_se, 4),
            "vs_anchor": round(dm, 4),
            "vs_anchor_ci95": round(1.96 * dse, 4),
            "verdict": ("显著优" if dm - 1.96 * dse > 0 else
                        "显著劣" if dm + 1.96 * dse < 0 else "无显著差异"),
        })
    rows.sort(key=lambda x: -x["vs_anchor"])
    return rows


def score(ckpts: list[str], refs: list[str], deals: int, seed0: int,
          threads: int, anchor: str, progress_every: int = 50) -> dict:
    if len(refs) != 3:
        raise SystemExit(f"--ref 需要 3 个对手（用 | 分隔），当前 {len(refs)} 个")
    torch.set_num_threads(threads)
    models = {c: load(c) for c in ckpts}

    per_deal: dict[str, list[float]] = {c: [] for c in ckpts}
    t0 = time.time()

    for r in range(deals):
        deal_seed = seed0 + r * 1013
        # 每副牌换一次对手的座位排布，避免某个对手长期坐在同一侧
        opp_order = [refs[(i + r) % 3] for i in range(3)]
        rot_scores: dict[str, list[float]] = {c: [] for c in ckpts}
        for shift in range(4):
            free = [s for s in range(4) if s != shift]
            for c in ckpts:
                # 每个检查点都重建一次对手：有状态的教师（带 noise 的 rng）必须拿到
                # 同一个初始种子，否则它会因为上一个检查点的走法不同而漂到别的随机流上，
                # 「共用随机数」的前提就破了。
                seat_agents = [None] * 4
                for k, slot in enumerate(free):
                    spec = opp_order[k]
                    seat_agents[slot] = make_opponent(spec, _seed(r, shift, slot, spec))
                seat_agents[shift] = NetAgent(models[c], seed=0, temperature=0.0, name=c)
                game = DushanGame(DushanConfig(), seed=deal_seed)
                res = game.play([_wrap(a) for a in seat_agents])
                rot_scores[c].append(res["deltas"][shift])
        for c in ckpts:
            per_deal[c].append(st.mean(rot_scores[c]))
        if progress_every and (r + 1) % progress_every == 0:
            el = time.time() - t0
            eta = el / (r + 1) * (deals - r - 1) / 60
            lead = max(ckpts, key=lambda c: st.mean(per_deal[c]))
            print(f"[进度 {r + 1}/{deals}] 已用 {el / 60:.1f} 分，"
                  f"预计剩余 {eta:.0f} 分；暂列第一 {lead}", flush=True)

    rows = summarize(per_deal, anchor)
    return {
        "anchor": anchor,
        "ref_field": refs,
        "seed0": seed0,
        "deals": deals,
        "games_per_ckpt": deals * 4,
        "elapsed_min": round((time.time() - t0) / 60, 2),
        "rows": rows,
        "per_deal": {c: per_deal[c] for c in ckpts},
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpts", required=True,
                    help="待评检查点，逗号分隔（裸文件名，不带 .pt）")
    ap.add_argument("--ref", required=True,
                    help="三个对手，用 | 分隔，支持 net:xxx 或 teacher:风格")
    ap.add_argument("--deals", type=int, default=600)
    ap.add_argument("--seed", type=int, default=31337)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--anchor", default="",
                    help="作为比较基准的检查点（默认取 --ckpts 的第一个）")
    ap.add_argument("--out", default="reports/checkpoint_scores.json")
    ap.add_argument("--no-per-deal", action="store_true",
                    help="JSON 里不写逐副明细（体积小一些）")
    args = ap.parse_args()

    ckpts = [c.strip() for c in args.ckpts.split(",") if c.strip()]
    refs = [r.strip() for r in args.ref.split("|") if r.strip()]
    anchor = args.anchor or ckpts[0]
    if anchor not in ckpts:
        raise SystemExit(f"--anchor {anchor} 不在 --ckpts 里")

    res = score(ckpts, refs, args.deals, args.seed, args.threads, anchor)

    print(f"\n对手（三个座位）：{' + '.join(res['ref_field'])}")
    print(f"牌池 seed={res['seed0']}，{res['deals']} 副 × 4 座位 = "
          f"{res['games_per_ckpt']} 局/检查点；共用牌池 + 共用对手")
    print(f"\n{'检查点':<26}{'分值水平':>12}{'±95%CI':>10}"
          f"{'vs 锚点':>11}{'±95%CI':>10}   结论")
    for r in res["rows"]:
        tag = "（锚点）" if r["ckpt"] == anchor else ""
        print(f"{r['ckpt']:<26}{r['level']:>+12.3f}{r['level_ci95']:>10.3f}"
              f"{r['vs_anchor']:>+11.3f}{r['vs_anchor_ci95']:>10.3f}   "
              f"{r['verdict']}{tag}")
    print(f"\n排名（按相对锚点的配对净分差）：")
    for i, r in enumerate(res["rows"], 1):
        print(f"  {i}. {r['ckpt']}  {r['vs_anchor']:+.3f} ± {r['vs_anchor_ci95']:.3f}")
    print(f"\n用时 {res['elapsed_min']:.1f} 分")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(res)
    if args.no_per_deal:
        payload.pop("per_deal", None)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"结果 -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
