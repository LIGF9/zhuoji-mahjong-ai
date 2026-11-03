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
                 name: str | None = None, threads: int | None = None,
                 encoder_version: int | None = None,
                 encoder_packs=None):
        self.model = model
        self.model.eval()
        self.rng = np.random.default_rng(seed)
        self.temperature = temperature
        # 编码器版本：显式传参优先，否则读模型自带属性（``build_from_ckpt`` 会设），
        # 再否则按 v1。这样所有"加载检查点 → 造 NetAgent"的旧代码不用改就能支持 v2。
        self.encoder_version = int(
            encoder_version if encoder_version is not None
            else getattr(model, "encoder_version", 1))
        # 启用的 P1 特征包（只对 v2 有意义）：同上的"显式 > 模型属性 > 默认"优先级。
        # 训练时用包掩码关掉了某个包，评测时必须关掉同样的包，否则输入口径不一致。
        self.encoder_packs = (
            encoder_packs if encoder_packs is not None
            else getattr(model, "encoder_packs", None))
        if name:
            self.name = name
        if threads is not None:
            torch.set_num_threads(threads)

    def reset(self, seed: int | None = None) -> None:
        if seed is not None:
            self.rng = np.random.default_rng(seed)

    def _obs(self, game):
        p = game.actor()
        return encode(game, p, self.encoder_version, self.encoder_packs)

    @torch.no_grad()
    def choose(self, game):
        obs = self._obs(game)
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
        obs = self._obs(game)
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
                   game_factory=None,
                   detail: dict | None = None):
    """复式对局：每一副牌让同样的 4 个 Agent 轮换座位打完整一圈。

    ``make_agents`` 是 ``() -> list[Bot]``，每次调用返回 4 个全新的 Agent 实例。

    ``return_per_deal=True`` 时额外返回 ``per_deal``：``{agent 名: [每副牌的均值]}``。
    长度恰好等于 ``n_rounds``——**同一副牌的四次轮换被压成一个观测**，
    这样算出来的标准误才不会被"同副牌内的相关性"作假。

    ``detail``：传入一个 dict 就地把**逐副明细**填进去（2026-11-02 加）::

        detail[agent 名] = [
            {"score": 该副均分, "wins": 0~4, "fan": 番数和, "tsumo": 0~4,
             "deal_in": 0~4, "huang": 0~4, "bigfan15": 0~4, "bigfan20": 0~4,
             "n": 4},  # 每副牌打了 4 遍（座位轮换）
            ...  # 长度 = n_rounds，与 per_deal 逐项对齐
        ]

    为什么要这个东西：只存每副**得分**时，胡牌率/放铳率/大牌率只能靠最后一块的
    汇总表，全程读数永久丢失（2026-11-02 千局对比踩过）。明细把"每副发生了
    几次胡、几番、谁放铳"留下来，事后可以算任意阈值的率 + 配对检验，
    不必重跑。
    """
    cfg = copy.deepcopy(config or RulesConfig())
    stats = defaultdict(lambda: {
        "score": 0.0, "wins": 0, "ron": 0, "tsumo": 0, "deal_in": 0,
        "huang": 0, "games": 0, "fan": 0.0, "bigfan15": 0, "bigfan20": 0,
        "r1": 0, "r2": 0, "r3": 0, "r4": 0,
    })
    deal_scores = defaultdict(lambda: defaultdict(list))
    # 逐副逐 agent 的事件计数（4 次座位轮换累加，n=4）
    deal_rec = defaultdict(lambda: defaultdict(lambda: {
        "score": 0.0, "wins": 0, "fan": 0.0, "tsumo": 0, "deal_in": 0,
        "huang": 0, "bigfan15": 0, "bigfan20": 0, "n": 0,
        "r1": 0, "r2": 0, "r3": 0, "r4": 0,
    }))
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
                rr = deal_rec[r][a.name]
                rr["score"] += res["deltas"][slot]
                rr["n"] += 1
            # —— 名次（2026-11-03 加）——
            # 「分/副」衡量的是得分期望，但四人麻将的真实目标是名次：领先时该防守、
            # 落后时该冒险，这两个行为在均分里可能互相抵消（Suphx 为此专门做了 GRP）。
            # place = 1 + 严格比我分高的人数 → 并列时共享同一名次（流局四家 0 分
            # 全算并列 1st，因此 4 个名次的占比之和可能 > 100%，均值略偏乐观）。
            deltas = res["deltas"]
            for slot in range(4):
                place = 1 + sum(1 for t in range(4) if deltas[t] > deltas[slot])
                stats[seat_agents[slot].name][f"r{place}"] += 1
                deal_rec[r][seat_agents[slot].name][f"r{place}"] += 1
            if res["type"] == "win":
                w = res["winner"]
                wname = seat_agents[w].name
                stats[wname]["wins"] += 1
                stats[wname]["fan"] += res.get("fan", 0)
                wr = deal_rec[r][wname]
                wr["wins"] += 1
                wr["fan"] += res.get("fan", 0)
                if res.get("fan", 0) >= 15:
                    stats[wname]["bigfan15"] += 1
                    wr["bigfan15"] += 1
                if res.get("fan", 0) >= 20:
                    stats[wname]["bigfan20"] += 1
                    wr["bigfan20"] += 1
                if res.get("is_tsumo"):
                    stats[wname]["tsumo"] += 1
                    wr["tsumo"] += 1
                elif res.get("loser") is not None:
                    stats[wname]["ron"] += 1
                    lname = seat_agents[res["loser"]].name
                    stats[lname]["deal_in"] += 1
                    deal_rec[r][lname]["deal_in"] += 1
            else:
                for slot in range(4):
                    stats[seat_agents[slot].name]["huang"] += 1
                    deal_rec[r][seat_agents[slot].name]["huang"] += 1

    per_deal = {name: [] for name in stats}
    for r in range(n_rounds):
        for name, vs in deal_scores[r].items():
            per_deal[name].append(sum(vs) / len(vs))

    if detail is not None:
        # 逐副明细：率 = 事件数 / n（n = 该副的轮换次数，通常 4）
        for name in stats:
            detail.setdefault(name, [])
        for r in range(n_rounds):
            for name in stats:
                d = deal_rec[r][name]
                n = max(1, d["n"])
                detail[name].append({
                    "score": d["score"] / n,
                    "wins": d["wins"], "fan": d["fan"], "tsumo": d["tsumo"],
                    "deal_in": d["deal_in"], "huang": d["huang"],
                    "bigfan15": d["bigfan15"], "bigfan20": d["bigfan20"],
                    "r1": d["r1"], "r2": d["r2"], "r3": d["r3"], "r4": d["r4"],
                    "n": d["n"],
                })

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
            "rank1_rate": s["r1"] / g,
            "rank2_rate": s["r2"] / g,
            "rank3_rate": s["r3"] / g,
            "rank4_rate": s["r4"] / g,
            "avg_rank": sum(p * s[f"r{p}"] for p in (1, 2, 3, 4)) / g,
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
