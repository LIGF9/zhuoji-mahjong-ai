# -*- coding: utf-8 -*-
"""临时加强验证语料导出（不进仓库，用完即删）。

与 scripts/export_standalone_corpus.py 同构，但：
- 不做 obs/logits（纯规则引擎对拍）
- 动作选择偏向胡牌/碰杠，保证覆盖胡牌、杠、包牌、鸡结算路径
- 只保留 result.type == "win" 的对局
"""
from __future__ import annotations

import dataclasses
import json
import os
import random
import sys

ROOT = r"C:\Users\LIGF\WorkBuddy\2026-09-30-12-49-43\zhuoji-mahjong"
sys.path.insert(0, ROOT)

from zhuoji.dushan import DushanConfig, DushanGame  # noqa: E402
from zhuoji.fan import is_tenpai, winning_tiles  # noqa: E402


def _potential(h):
    """粗略手牌势能：越集中、越成搭越高。"""
    pot = 0.0
    for t in range(27):
        c = h[t]
        if c == 0:
            continue
        pot += c * c * 0.5
        s = t // 9
        for d in (-1, 1):
            nt = t + d
            if 0 <= nt < 27 and nt // 9 == s and h[nt] > 0:
                pot += 0.4 * min(c, h[nt])
        if c >= 2:
            pot += 1.0
    return pot


def pick_discard(g, p, acts):
    discards = [a for a in acts if a.kind == "discard"]
    other = [a for a in acts if a.kind != "discard"]
    best, best_key = None, None
    for a in discards:
        g.hands[p][a.tile] -= 1
        if is_tenpai(g.hands[p], g.melds[p]):
            key = (1000.0 + len(winning_tiles(g.hands[p], g.melds[p]))
                   - a.tile * 1e-6)
        else:
            key = _potential(g.hands[p]) - a.tile * 1e-6
        g.hands[p][a.tile] += 1
        if best is None or key > best_key:
            best, best_key = a, key
    return other + [best]


def pick(acts, g, rng):
    kinds = [a.kind for a in acts]
    if "hu" in kinds:
        return acts[kinds.index("hu")]
    # SELF_KONG：能杠就杠
    for kind in ("angang", "bugang"):
        if kind in kinds:
            return acts[kinds.index(kind)]
    # RESPOND：明杠必杠，碰 70%
    if "minggang" in kinds:
        return acts[kinds.index("minggang")]
    if "peng" in kinds and rng.random() < 0.7:
        return acts[kinds.index("peng")]
    if "discard" in kinds:
        cands = pick_discard(g, g.actor(), acts)
        return cands[rng.randrange(len(cands))]
    return acts[rng.randrange(len(acts))]


def dump_state(g):
    out = {}
    for k, v in sorted(g.__dict__.items()):
        if k in ("rng", "cfg"):
            continue
        out[k] = _jsonable(v)
    return out


def _jsonable(v):
    if isinstance(v, (int, float, str, bool)) or v is None:
        return v
    if isinstance(v, (list, tuple)):
        return [_jsonable(x) for x in v]
    if isinstance(v, set):
        return {"__set__": sorted(_jsonable(x) for x in v)}
    if isinstance(v, dict):
        return {"__dict__": [[_jsonable(a), _jsonable(b)] for a, b in v.items()]}
    if dataclasses.is_dataclass(v):
        return {"__dc__": dataclasses.asdict(v)}
    if hasattr(v, "value"):
        return str(v.value)
    return f"<{type(v).__name__}>"



def main():
    seed0 = int(os.environ.get("SEED0", "20261023"))
    target = int(os.environ.get("TARGET", "12"))
    keep_draw = os.environ.get("KEEP_DRAW", "0") == "1"
    rng = random.Random(seed0)
    games = []
    tried = 0
    stats = {"huangzhuang": 0, "win": 0}
    how_stat = {}
    while len(games) < target and tried < 4000:
        tried += 1
        cfg = DushanConfig()
        seed = rng.randrange(1 << 30)
        g = DushanGame(cfg, seed=seed)
        rec = {
            "seed": seed,
            "config": dataclasses.asdict(cfg),
            "wall": list(g.wall),
            "dealer": g.dealer,
            "steps": [],
            "obs": [],
        }
        si = 0
        while g.phase.value != "over" and si < 2000:
            p = g.actor()
            legal = [[a.kind, int(a.tile)] for a in g.legal_actions()]
            acts = g.legal_actions()
            if not acts:
                break
            act = pick(acts, g, rng)
            rec["steps"].append({
                "legal": legal,
                "act": [act.kind, int(act.tile)],
                "state": dump_state(g),
            })
            g.step(act)
            si += 1
        res = _jsonable(getattr(g, "result", None))
        # _jsonable 对 dict 返回 {"__dict__": [...]} 包装形式，先解包再判类型
        plain = dict(res["__dict__"]) if isinstance(res, dict) and "__dict__" in res else res
        rtype = plain.get("type") if isinstance(plain, dict) else None
        if rtype == "win":
            stats["win"] += 1
            how_stat[plain.get("how")] = how_stat.get(plain.get("how"), 0) + 1
            if plain.get("rob_kong"):
                how_stat["rob_kong"] = how_stat.get("rob_kong", 0) + 1
            if plain.get("void_player") is not None:
                how_stat["void"] = how_stat.get("void", 0) + 1
            if len(plain.get("winners") or []) > 1:
                how_stat["multi_hu"] = how_stat.get("multi_hu", 0) + 1
            if plain.get("fanji_gold"):
                how_stat["gold"] = how_stat.get("gold", 0) + 1
            if plain.get("tenpai") and len(plain.get("tenpai")) < 4:
                how_stat["baoji_on_win"] = how_stat.get("baoji_on_win", 0) + 1
            rec["result"] = res
            games.append(rec)
        elif keep_draw and rtype == "huangzhuang" and plain.get("cha_jiao"):
            stats["huangzhuang"] += 1
            how_stat["cha_jiao"] = how_stat.get("cha_jiao", 0) + 1
            rec["result"] = res
            games.append(rec)
        else:
            stats["huangzhuang"] += 1
    print("tried:", tried, stats, how_stat)
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       os.environ.get("OUT", "stress_corpus.json"))
    corpus = {"seed": 20261023, "obs_every": 0, "games": games}
    with open(out, "w", encoding="utf-8") as f:
        json.dump(corpus, f, ensure_ascii=False, separators=(",", ":"))
    print(f"corpus -> {out} ({os.path.getsize(out) / 2**20:.1f} MiB)")


if __name__ == "__main__":
    main()
