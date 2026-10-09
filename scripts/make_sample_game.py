"""
挑一局"模型视角"的样例对局，导出成 JSON 供报告回放。

除了终局结果和动作流水，还会挑若干个决策点，把网络在该状态下的
**动作概率分布**打出来 —— 这比单纯看胜率更能说明模型学到了什么。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from zhuoji import Phase, RulesConfig, ZhuojiGame  # noqa: E402
from zhuoji.agent import NetAgent  # noqa: E402
from zhuoji.bots import TEACHER_STYLES, HeuristicBot  # noqa: E402
from zhuoji.encoder import encode, index_to_action  # noqa: E402
from zhuoji.fan import FAN_CN, MELD_CN  # noqa: E402
from zhuoji.net import NetConfig, build_model  # noqa: E402
from zhuoji.tiles import tile_cn  # noqa: E402


def load_model(path: str):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    nc = ck.get("net") or {"channels": 32, "blocks": 3, "hidden": 256}
    m = build_model(NetConfig(nc["channels"], nc["blocks"], nc["hidden"]))
    m.load_state_dict(ck["state_dict"])
    m.eval()
    return m


@torch.no_grad()
def topk(model, game, player, k=3):
    obs = encode(game, player)
    out = model(torch.from_numpy(obs.tiles[None].astype(np.float32)),
                torch.from_numpy(obs.glob[None].astype(np.float32)))
    lg = out[obs.head][0][:obs.mask.shape[0]].clone()
    lg[~torch.from_numpy(obs.mask)] = -1e9
    probs = torch.softmax(lg, dim=-1).numpy()
    order = np.argsort(-probs)[:k]
    res = []
    for i in order:
        if probs[i] < 1e-6:
            continue
        res.append({
            "action": str(index_to_action(obs.head, int(i), game).label()),
            "p": round(float(probs[i]), 3),
        })
    return obs.head, res, float(out["value"][0])


def tiles_str(c):
    return " ".join(tile_cn(t) for t, n in enumerate(c) for _ in range(n))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=str(ROOT / "models" / "rl.pt"))
    ap.add_argument("--out", default=str(ROOT / "reports" / "sample_game.json"))
    ap.add_argument("--seed", type=int, default=20260930)
    ap.add_argument("--max-hops", type=int, default=900)
    args = ap.parse_args()

    path = args.model
    if not Path(path).exists():
        path = str(ROOT / "models" / "bc.pt")
    model = load_model(path)
    me = NetAgent(model, seed=1, temperature=0.0, name="net")

    cfg = RulesConfig()
    rng = np.random.default_rng(args.seed)
    # 找一个模型真的赢了的对局，回放才有看头
    chosen = None
    for k in range(args.max_hops):
        game = ZhuojiGame(cfg, seed=args.seed + k)
        bots = [me,
                HeuristicBot(seed=k * 31 + 1, weights=TEACHER_STYLES["balanced"]),
                HeuristicBot(seed=k * 31 + 2, weights=TEACHER_STYLES["aggressive"]),
                HeuristicBot(seed=k * 31 + 3, weights=TEACHER_STYLES["defensive"])]
        res = game.play([b.choose for b in bots])
        if res["type"] == "win" and res["winner"] == 0 and args.seed + k < args.seed + 400:
            chosen = (game, res, k)
            break
    if chosen is None:
        k = 0
        game = ZhuojiGame(cfg, seed=args.seed)
        bots = [me,
                HeuristicBot(seed=1, weights=TEACHER_STYLES["balanced"]),
                HeuristicBot(seed=2, weights=TEACHER_STYLES["aggressive"]),
                HeuristicBot(seed=3, weights=TEACHER_STYLES["defensive"])]
        res = game.play([b.choose for b in bots])
        chosen = (game, res, 0)
    game, res, k = chosen

    # 重放一遍收集决策点
    game2 = ZhuojiGame(cfg, seed=args.seed + k)
    bots2 = [me,
             HeuristicBot(seed=k * 31 + 1, weights=TEACHER_STYLES["balanced"]),
             HeuristicBot(seed=k * 31 + 2, weights=TEACHER_STYLES["aggressive"]),
             HeuristicBot(seed=k * 31 + 3, weights=TEACHER_STYLES["defensive"])]
    decisions = []
    log = []
    from zhuoji.agent import _wrap  # noqa
    steps = 0
    while game2.phase != Phase.OVER and steps < 4000:
        p = game2.actor()
        if p == 0:
            head, top, val = topk(model, game2, p)
            decisions.append({
                "step": steps,
                "hand": tiles_str(game2.hands[0]),
                "melds": [f"{MELD_CN[mt]}{tile_cn(t)}" for mt, t, _ in game2.melds[0]],
                "head": head,
                "top": top,
                "value": round(val, 3),
                "wall_left": game2.wall_left,
            })
        a = bots2[p].choose(game2)
        log.append((p, a.label()))
        game2.step(a)
        steps += 1

    # 只保留有信息量的决策点（有多个候选的）
    picked = [d for d in decisions if len(d["top"]) >= 2]
    if len(picked) > 8:
        idxs = np.linspace(0, len(picked) - 1, 8).astype(int)
        picked = [picked[i] for i in idxs]

    summary = {
        "type": res["type"],
        "winner": res.get("winner"),
        "fan_cn": res.get("fan_cn"),
        "fan": res.get("fan"),
        "is_tsumo": res.get("is_tsumo"),
        "deltas": res["deltas"],
        "dou": res.get("dou", []),
        "chicken": res.get("chicken", []),
        "wall_left": res.get("wall_left"),
    }
    out = {
        "model": path,
        "seed": args.seed + k,
        "summary_text": (
            f"本局{'自摸' if summary['is_tsumo'] else '点炮'}胡，牌型 {summary['fan_cn']}"
            f"（{summary['fan']} 番），四家得分 "
            f"{[round(x, 1) for x in summary['deltas']]}，牌墙余 {summary['wall_left']} 张。"
            if res["type"] == "win" else
            f"本局黄庄，四家得分 {[round(x, 1) for x in summary['deltas']]}。"),
        "summary": summary,
        "hands": [tiles_str(game.hands[p]) if p == 0 else tiles_str(game.hands[p])
                  for p in range(4)],
        "melds": [[f"{MELD_CN[mt]}{tile_cn(t)}" for mt, t, _ in game.melds[p]]
                  for p in range(4)],
        "decisions": picked,
        "log": log,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, ensure_ascii=False, indent=2),
                              encoding="utf-8")
    print("样例对局 ->", args.out)
    print(out["summary_text"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
