"""
全策略循环赛（round-robin）：把随机、深度学习模型、各种启发式教师放进**同一张牌桌**混战，
测出每种策略的胜率分布与两两对位强弱。

与 ``arena.py`` 的分工
---------------------
``arena.py`` 回答的是"**我**打某个对手胜率多少"（1 vs 3×同一对手），
用来做单个模型的对抗评测。

本脚本回答的是另一个问题："**这些策略混在一桌**的时候，谁赢得更多" ——
也就是胜率在整个策略池上的分布。做法是让 4 个**互不相同**的策略凑一桌。

实验设计（为什么不是随便抽 4 个）
---------------------------------
1. **BIBD 全覆盖**：9 个策略里取 4 个上桌，组合数是 C(9,4)=126。把 126 种四人组合
   **各打一遍**作为一个"轮"，于是每个策略出场的次数完全相同（各 56 次），
   每一对策略同桌的次数也完全相同（各 21 次）。这叫均衡不完全区组设计——
   配对比较不会被"谁跟弱鸡同桌多"污染。
2. **真·复式**：同一个四人组合下，用**同一副牌**（同一 seed，牌墙与庄家都相同）
   把座位轮换 4 次。每个策略因此在这副牌上把东南西北各坐一遍，座位优势与牌运
   在组内被抵消。注意 ``arena.py`` 里把 shift 混进了 seed，四轮实际吃的是四副不同的牌，
   这里修正为 shift 只影响座位、不影响牌。
3. **多轮重复**：把 126 组 × 4 座位 重复 ``--reps`` 轮，每轮换一批牌。

用法::

    python scripts/roundrobin.py --reps 6
    python scripts/roundrobin.py --reps 6 --agents random,rl --out reports/rr.json
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import sys
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from zhuoji.agent import NetAgent  # noqa: E402
from zhuoji.bots import TEACHER_STYLES, HeuristicBot, RandomBot  # noqa: E402
from zhuoji.net import NetConfig, build_model  # noqa: E402
from zhuoji.rules import RulesConfig, ZhuojiGame  # noqa: E402

# 参赛名单：随机基线 + 两代神经网络 + 六种性格的启发式教师
DEFAULT_AGENTS = [
    "random",
    "bc",
    "rl",
    "teacher:balanced",
    "teacher:aggressive",
    "teacher:defensive",
    "teacher:chicken_lover",
    "teacher:tenpai_rush",
    "teacher:gambler",
]

_MODELS: dict[str, object] = {}


def _load_net(stem: str):
    if stem in _MODELS:
        return _MODELS[stem]
    path = ROOT / "models" / f"{stem}.pt"
    ck = torch.load(path, map_location="cpu", weights_only=False)
    nc = ck.get("net") or {"channels": 48, "blocks": 8, "hidden": 256}
    m = build_model(NetConfig(nc["channels"], nc["blocks"], nc["hidden"]))
    m.load_state_dict(ck["state_dict"])
    m.eval()
    _MODELS[stem] = (m, ck)
    return m, ck


def det_seed(*parts) -> int:
    """由名字推出确定性种子：同一份结果可复现，跨运行/跨机器一致。"""
    h = hashlib.sha256("#".join(str(p) for p in parts).encode("utf-8")).hexdigest()
    return int(h[:8], 16) % (10 ** 6)


def make_agent(name: str, seed: int):
    """按名字建一个 Agent 实例。"""
    if name == "random":
        return RandomBot(seed=seed)
    if name == "random":
        pass  # 上面已处理
    if name in ("bc", "rl") or (ROOT / "models" / f"{name}.pt").exists():
        model, _ = _load_net(name)
        ag = NetAgent(model, seed=seed, temperature=0.0, name=name)
        return ag
    if name.startswith("teacher:"):
        style = name.split(":", 1)[1]
        if style not in TEACHER_STYLES:
            raise SystemExit(f"未知教师风格 {style!r}，可选：{list(TEACHER_STYLES)}")
        return HeuristicBot(seed=seed, weights=TEACHER_STYLES[style])
    raise SystemExit(f"未知策略 {name!r}")


def _wrap(bot):
    def f(game):
        return bot.choose(game)
    return f


# ---------------------------------------------------------------------------
# 主循环
# ---------------------------------------------------------------------------
# 需要单独算标准误的比率指标（分子计数 / 局数）
COUNTERS = ("wins", "tsumo", "ron", "deal_in", "huang", "chongfeng", "zeren")


def _record(t: dict, res: dict, s: int, winners, deltas, game):
    """把一个座位在一局里的全部结果记进 ``t``。

    累计表与"复式单元"共用同一套记录逻辑，避免两处口径不一致
    （之前手写过一次两套，字段名对不上，静默错了一版）。
    """
    t["games"] += 1
    t["score"] += deltas[s]
    if res["type"] == "win":
        if s == winners:
            t["wins"] += 1
            t["fan"] += res.get("fan", 0)
            t["total_fan"] += res.get("total_fan", res.get("fan", 0))
            if res.get("is_tsumo"):
                t["tsumo"] += 1
            elif res.get("loser") is not None:
                t["ron"] += 1
        if res.get("loser") == s:
            t["deal_in"] += 1
    else:
        t["huang"] += 1
    if game.chongfeng_player == s:
        t["chongfeng"] += 1
    if game.zeren_player == s:
        t["zeren"] += 1


def run_tournament(agents: list[str], reps: int, seed0: int,
                   progress: bool = True) -> dict:
    k = len(agents)
    blocks = list(itertools.combinations(range(k), 4))

    # 每个策略的累计量
    agg = {a: defaultdict(float) for a in agents}
    # 复式单元：一个 (rep, block) 下某策略坐遍 4 个座位后的均值/比率。
    # 同一副牌打 4 次不是 4 个独立观测，标准误必须按"组"来算，
    # 否则区间会被系统性低估（这是 arena.py 里将 shift 混进 seed 时踩过的坑）。
    units = {a: defaultdict(list) for a in agents}
    # 对位：A 与 B 同桌时的得分差 / 各自胡牌次数
    pair_diff = {a: defaultdict(list) for a in agents}
    pair_wins = {a: defaultdict(int) for a in agents}
    pair_cooc = {a: defaultdict(int) for a in agents}
    seat_score = {a: [0.0] * 4 for a in agents}   # 座位偏差体检用

    cfg = RulesConfig()
    t0 = time.time()
    total_games = len(blocks) * 4 * reps
    done = 0

    for rep in range(reps):
        for bi, block in enumerate(blocks):
            names = [agents[i] for i in block]
            # 同一副牌：seed 只依赖 (rep, block)，不含 shift ⇒ 牌墙与庄家完全一致
            deal_seed = (seed0 + rep * 1000003 + bi * 7919) % (2 ** 31)
            blk = {nm: defaultdict(float) for nm in names}

            for shift in range(4):
                # 座位轮换：第 s 号座位坐 block 里第 (s+shift) 个人
                seat_agents = []
                for s in range(4):
                    nm = names[(s + shift) % 4]
                    ag = make_agent(nm, det_seed(nm, rep, bi, s))
                    ag.name = nm
                    seat_agents.append(ag)
                game = ZhuojiGame(cfg, seed=deal_seed)
                res = game.play([_wrap(a) for a in seat_agents])

                deltas = res["deltas"]
                winners = res.get("winner")
                for s in range(4):
                    nm = seat_agents[s].name
                    _record(agg[nm], res, s, winners, deltas, game)
                    _record(blk[nm], res, s, winners, deltas, game)
                    seat_score[nm][s] += deltas[s]

                # 对位统计：桌面上每一对
                for x in range(4):
                    for y in range(4):
                        if x == y:
                            continue
                        a, b = seat_agents[x].name, seat_agents[y].name
                        pair_diff[a][b].append(deltas[x] - deltas[y])
                        if res["type"] == "win":
                            if winners == x:
                                pair_wins[a][b] += 1
                            elif winners == y:
                                pair_wins[b][a] += 1
                        pair_cooc[a][b] += 1

                done += 1

            # 复式单元结算：本 block 内每个策略都坐遍了 4 个座位
            for nm in names:
                b = blk[nm]
                g = max(1.0, b["games"])
                units[nm]["score"].append(b["score"] / g)
                for c in COUNTERS:
                    units[nm][c].append(b[c] / g)

        if progress:
            el = time.time() - t0
            print(f"[rep {rep + 1}/{reps}] {done}/{total_games} 局  "
                  f"{el:6.1f}s  预计总耗时 {el / (rep + 1) * reps:6.1f}s", flush=True)

    return {
        "agg": {a: dict(v) for a, v in agg.items()},
        "units": {a: {k: [round(x, 4) for x in v] for k, v in d.items()}
                  for a, d in units.items()},
        "pair_diff": {a: {b: [round(x, 3) for x in v] for b, v in d.items()}
                      for a, d in pair_diff.items()},
        "pair_wins": {a: dict(d) for a, d in pair_wins.items()},
        "pair_cooc": {a: dict(d) for a, d in pair_cooc.items()},
        "seat_score": seat_score,
        "games_total": total_games,
        "elapsed": time.time() - t0,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=6)
    ap.add_argument("--seed", type=int, default=20260930)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--agents", default="", help="逗号分隔，默认全部 9 个")
    ap.add_argument("--out", default=str(ROOT / "reports" / "roundrobin.json"))
    ap.add_argument("--tag", default="", help="写进结果的标签，便于区分不同配置")
    args = ap.parse_args()

    torch.set_num_threads(args.threads)
    agents = ([a.strip() for a in args.agents.split(",") if a.strip()]
              if args.agents else list(DEFAULT_AGENTS))
    if len(agents) < 4:
        raise SystemExit("至少需要 4 个策略才能凑一桌")

    print(f"参赛：{', '.join(agents)}")
    print(f"设计：C({len(agents)},4)={math.comb(len(agents), 4)} 组 × 4 座位 × {args.reps} 轮"
          f" = {math.comb(len(agents), 4) * 4 * args.reps} 局\n")
    raw = run_tournament(agents, args.reps, args.seed)
    print(f"\n全部完成：{raw['games_total']} 局，耗时 {raw['elapsed']:.1f}s\n")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps({
        "agents": agents,
        "reps": args.reps,
        "seed": args.seed,
        "tag": args.tag,
        "games_total": raw["games_total"],
        "elapsed": raw["elapsed"],
        "agg": raw["agg"],
        "units": raw["units"],
        "n_cooc": raw["pair_cooc"],
        "pair_diff": raw["pair_diff"],
        "pair_wins": raw["pair_wins"],
        "seat_score": raw["seat_score"],
    }, ensure_ascii=False, indent=1))
    print(f"原始结果 -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
