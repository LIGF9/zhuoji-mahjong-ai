# -*- coding: utf-8 -*-
"""一炮多响合成场景基准（Python 侧生成，JS 侧同状态重放对拍）。"""
from __future__ import annotations

import dataclasses
import json
import os
import random
import sys

ROOT = r"C:\Users\LIGF\WorkBuddy\2026-09-30-12-49-43\zhuoji-mahjong"
sys.path.insert(0, ROOT)

from zhuoji.dushan import DushanConfig, DushanGame  # noqa: E402

spec_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         "export_stress_corpus.py")
spec = {"__name__": "scen"}
exec(compile(open(spec_path, encoding="utf-8").read(), spec_path, "exec"), spec)
dump_state = spec["dump_state"]


def counts(*pairs):
    h = [0] * 27
    for t, c in pairs:
        h[t] += c
    return h


def make_scenario(name, dealer_seed, hands, p2_discard):
    """hands: 4×27 计数向量；玩家 2 打 p2_discard 后 0/1 双胡。"""
    rng = random.Random(dealer_seed)
    cfg = DushanConfig()
    g = DushanGame(cfg, seed=rng.randrange(1 << 30))
    wall = list(g.wall)
    dealer = g.dealer
    # 覆盖状态：玩家 2 出牌阶段，准备打出 p2_discard
    g.wall = wall
    g.wall_pos = 60
    g.hands = [list(h) for h in hands]
    g.melds = [[], [], [], []]
    g.discards = [[], [], [], []]
    g.ever_tenpai = [False] * 4
    g.pass_lock = [set(), set(), set(), set()]
    g.gang_discard = [False] * 4
    g.drawn = True
    g.current = 2
    g.phase = g.phase.DISCARD
    g.last_discard = None
    g.last_discard_was_gang = False
    g.baojiao = [False] * 4
    g.first_discard_done = [False] * 4

    rec = {"name": name,
           "config": dataclasses.asdict(cfg),
           "wall": wall, "dealer": dealer,
           "setup": {
               "hands": [list(h) for h in hands],
               "wall_pos": 60, "current": 2, "drawn": True,
               "phase": "discard",
           },
           "steps": [], "result": None}
    si = 0
    while g.phase is not g.phase.OVER and si < 50:
        legal = [[a.kind, int(a.tile)] for a in g.legal_actions()]
        acts = g.legal_actions()
        if not acts:
            break
        # 决策：玩家 2 打指定牌；胡牌者都胡
        if g.actor() == 2 and g.phase is g.phase.DISCARD:
            act = [a for a in acts if a.kind == "discard"
                   and a.tile == p2_discard][0]
        elif any(a.kind == "hu" for a in acts):
            act = [a for a in acts if a.kind == "hu"][0]
        else:
            act = acts[0]
        rec["steps"].append({
            "legal": legal,
            "act": [act.kind, int(act.tile)],
            "state": dump_state(g),
        })
        g.step(act)
        si += 1
    rec["result"] = spec["_jsonable"](getattr(g, "result", None))
    rec["steps_count"] = si
    return rec


def main():
    out = {}
    # 场景 1：0、1 双听 W5 七对，玩家 2 打 W5 → 一炮多响
    h = [
        counts((0, 2), (1, 2), (2, 2), (3, 2), (5, 2), (6, 2), (4, 1)),   # p0: W1-4,6,7 对 + W5 单
        counts((18, 2), (19, 2), (20, 2), (21, 2), (23, 2), (24, 2), (4, 1)),  # p1: 筒子对 + W5 单
        counts((8, 3), (7, 3), (26, 3), (17, 3), (4, 1)),                  # p2: 打 W5
        counts((9, 3), (10, 3), (11, 3), (12, 3), (13, 1)),                # p3
    ]
    assert all(sum(x) == 13 for x in h), [sum(x) for x in h]
    out["multi_hu"] = make_scenario("multi_hu", 4242, h, 4)

    # 场景 2：0 过、1 胡（胡票中有人 pass 的混合表态路径）
    # p0 听 W5 但过水后……直接让 p0 胡后本局结束不行，需 p0 pass：
    # 把 p0 改成「能胡但脚本让它 pass」→ 决策里 hu 优先会胡。
    # 为覆盖 pass 分支：p0 的 legal 含 hu，但决策顺序改为 p0 pass、p1 hu。
    out["hu_then_pass"] = make_scenario_pass(31337, h, 4)
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "scenarios_multi_hu.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, separators=(",", ":"))
    print("scenarios ->", path,
          {k: v["steps_count"] for k, v in out.items()},
          {k: (v["result"] or {}).get("type") for k, v in out.items()})


def make_scenario_pass(dealer_seed, hands, p2_discard):
    """p0 pass、p1 胡：覆盖 _advance_respond 无票推进分支。"""
    import zhuoji.rules as R
    rng = random.Random(dealer_seed)
    cfg = DushanConfig()
    g = DushanGame(cfg, seed=rng.randrange(1 << 30))
    wall = list(g.wall)
    dealer = g.dealer
    g.wall = wall
    g.wall_pos = 60
    g.hands = [list(h) for h in hands]
    g.melds = [[], [], [], []]
    g.discards = [[], [], [], []]
    g.ever_tenpai = [False] * 4
    g.pass_lock = [set(), set(), set(), set()]
    g.gang_discard = [False] * 4
    g.drawn = True
    g.current = 2
    g.phase = R.Phase.DISCARD
    g.last_discard = None
    g.last_discard_was_gang = False
    g.baojiao = [False] * 4
    g.first_discard_done = [False] * 4

    rec = {"name": "hu_then_pass",
           "config": dataclasses.asdict(cfg),
           "wall": wall, "dealer": dealer,
           "setup": {
               "hands": [list(h) for h in hands],
               "wall_pos": 60, "current": 2, "drawn": True,
               "phase": "discard",
           },
           "steps": [], "result": None}
    si = 0
    while g.phase is not R.Phase.OVER and si < 50:
        legal = [[a.kind, int(a.tile)] for a in g.legal_actions()]
        acts = g.legal_actions()
        if not acts:
            break
        if g.actor() == 2 and g.phase is R.Phase.DISCARD:
            act = [a for a in acts if a.kind == "discard"
                   and a.tile == p2_discard][0]
        elif g.actor() == 0 and any(a.kind == "hu" for a in acts):
            act = R.Action(R.PASS)          # p0 过
        elif any(a.kind == "hu" for a in acts):
            act = [a for a in acts if a.kind == "hu"][0]
        else:
            act = acts[0]
        rec["steps"].append({
            "legal": legal,
            "act": [act.kind, int(act.tile)],
            "state": dump_state(g),
        })
        g.step(act)
        si += 1
    rec["result"] = spec["_jsonable"](getattr(g, "result", None))
    rec["steps_count"] = si
    return rec


if __name__ == "__main__":
    main()
