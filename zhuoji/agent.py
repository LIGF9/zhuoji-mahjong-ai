"""
把训练好的网络包装成可对局的 Agent，并提供对战评测。

评测用**复式思路**：同一副起手牌让每个 Agent 轮换坐四个座位各打一次，
消掉牌运带来的方差，得到的平均得分才有可比性。
"""
from __future__ import annotations

import copy
from collections import defaultdict

import numpy as np
import torch

from .bots import Bot
from .encoder import encode, index_to_action
from .rules import Phase, RulesConfig, ZhuojiGame


class NetAgent(Bot):
    """单步推理的神经网络 Agent。"""

    name = "net"

    def __init__(self, model, seed: int = 0, temperature: float = 0.0,
                 name: str | None = None, threads: int | None = None):
        self.model = model
        self.model.eval()
        self.rng = np.random.default_rng(seed)
        self.temperature = temperature
        if name:
            self.name = name
        if threads is not None:
            torch.set_num_threads(threads)

    def reset(self, seed: int | None = None) -> None:
        if seed is not None:
            self.rng = np.random.default_rng(seed)

    @torch.no_grad()
    def choose(self, game):
        p = game.actor()
        obs = encode(game, p)
        out = self.model(
            torch.from_numpy(obs.tiles[None].astype(np.float32)),
            torch.from_numpy(obs.glob[None].astype(np.float32)),
        )
        logits = out[obs.head][0][:obs.mask.shape[0]].clone()
        logits[~torch.from_numpy(obs.mask)] = -1e9
        if self.temperature <= 1e-6:
            idx = int(torch.argmax(logits))
        else:
            probs = torch.softmax(logits / self.temperature, dim=-1).numpy()
            probs = probs / probs.sum()
            idx = int(self.rng.choice(len(probs), p=probs))
        return index_to_action(obs.head, idx, game)

    def distribution(self, game):
        """返回 (head, mask, probs, value)，供 RL 采样与训练使用。"""
        p = game.actor()
        obs = encode(game, p)
        out = self.model(
            torch.from_numpy(obs.tiles[None].astype(np.float32)),
            torch.from_numpy(obs.glob[None].astype(np.float32)),
        )
        logits = out[obs.head][0][:obs.mask.shape[0]].clone()
        logits[~torch.from_numpy(obs.mask)] = -1e9
        return obs.head, obs.mask, torch.softmax(logits, dim=-1), float(out["value"][0]), obs


# ---------------------------------------------------------------------------
# 评测
# ---------------------------------------------------------------------------
def play_duplicate(make_agents, n_rounds: int, seed0: int = 0,
                   config: RulesConfig | None = None,
                   rotate_seats: bool = True,
                   return_per_deal: bool = False,
                   game_factory=None):
    """复式对局：每一副牌让同样的 4 个 Agent 轮换座位打完整一圈。

    ``make_agents`` 是 ``() -> list[Bot]``，每次调用返回 4 个全新的 Agent 实例。

    ``return_per_deal=True`` 时额外返回 ``per_deal``：``{agent 名: [每副牌的均值]}``。
    长度恰好等于 ``n_rounds``——**同一副牌的四次轮换被压成一个观测**，
    这样算出来的标准误才不会被"同副牌内的相关性"作假。
    """
    cfg = copy.deepcopy(config or RulesConfig())
    stats = defaultdict(lambda: {
        "score": 0.0, "wins": 0, "ron": 0, "tsumo": 0, "deal_in": 0,
        "huang": 0, "games": 0, "fan": 0.0, "bigfan15": 0, "bigfan20": 0,
    })
    deal_scores = defaultdict(lambda: defaultdict(list))
    seats = [0, 1, 2, 3] if rotate_seats else [0]

    for r in range(n_rounds):
        for shift in seats:
            agents = make_agents()
            order = [(i + shift) % 4 for i in range(4)]
            seat_agents = [None] * 4
            for slot, agent_idx in enumerate(order):
                seat_agents[slot] = agents[agent_idx]
            if game_factory is not None:
                game = game_factory(seed0 + r * 1013 + shift)
            else:
                game = ZhuojiGame(cfg, seed=seed0 + r * 1013 + shift)
            res = game.play([_wrap(a) for a in seat_agents])
            for slot in range(4):
                a = seat_agents[slot]
                s = stats[a.name]
                s["games"] += 1
                s["score"] += res["deltas"][slot]
                deal_scores[r][a.name].append(res["deltas"][slot])
            if res["type"] == "win":
                w = res["winner"]
                stats[seat_agents[w].name]["wins"] += 1
                stats[seat_agents[w].name]["fan"] += res.get("fan", 0)
                if res.get("fan", 0) >= 15:
                    stats[seat_agents[w].name]["bigfan15"] += 1
                if res.get("fan", 0) >= 20:
                    stats[seat_agents[w].name]["bigfan20"] += 1
                if res.get("is_tsumo"):
                    stats[seat_agents[w].name]["tsumo"] += 1
                elif res.get("loser") is not None:
                    stats[seat_agents[w].name]["ron"] += 1
                    stats[seat_agents[res["loser"]].name]["deal_in"] += 1
            else:
                for slot in range(4):
                    stats[seat_agents[slot].name]["huang"] += 1

    per_deal = {name: [] for name in stats}
    for r in range(n_rounds):
        for name, vs in deal_scores[r].items():
            per_deal[name].append(sum(vs) / len(vs))

    out = {}
    for name, s in stats.items():
        g = max(1, s["games"])
        ds = per_deal.get(name) or []
        se = float(np.std(ds, ddof=1) / np.sqrt(len(ds))) if len(ds) > 1 else 0.0
        out[name] = {
            "games": s["games"],
            "avg_score": s["score"] / g,
            "avg_score_se": se,
            "deals": len(ds),
            "win_rate": s["wins"] / g,
            "tsumo_rate": s["tsumo"] / g,
            "ron_rate": s["ron"] / g,
            "deal_in_rate": s["deal_in"] / g,
            "huang_rate": s["huang"] / g,
            "bigfan15_rate": s["bigfan15"] / g,
            "bigfan20_rate": s["bigfan20"] / g,
            "avg_fan_when_win": s["fan"] / max(1, s["wins"]),
        }
    if return_per_deal:
        return out, per_deal
    return out


def _wrap(bot: Bot):
    def f(game):
        return bot.choose(game)
    return f


def print_table(result: dict) -> None:
    cols = ["games", "deals", "avg_score", "avg_score_se", "win_rate", "tsumo_rate",
            "ron_rate", "deal_in_rate", "huang_rate", "avg_fan_when_win"]
    print(f"{'agent':<24}" + "".join(f"{c:>16}" for c in cols))
    for name, s in sorted(result.items(), key=lambda kv: -kv[1]["avg_score"]):
        print(f"{name:<24}" + "".join(
            f"{s[c]:>16.3f}" if isinstance(s[c], float) else f"{s[c]:>16d}" for c in cols))
