"""
基线 Bot：随机 / 启发式教师。

启发式教师是整个自产数据流程的"老师"：先用它跑出大规模对局做行为克隆，
再用自我对弈强化去超越它。所以它必须满足两点——
1. 明显强于随机（BC 才有意义）；
2. 决策快（否则生成数据太慢）。

它的核心指标是**向听数**，再加若干贵阳捉鸡特有的偏好项：
冲锋鸡收益、幺鸡的持有价值、豆（杠）与通行证、点胡安全性。
"""
from __future__ import annotations

import random
from dataclasses import dataclass

from .fan import MELD_KONG_ADDED, MELD_KONG_CONCEALED, MELD_KONG_EXPOSED, MELD_PONG
from .rules import (
    ANGANG, BUGANG, DISCARD, HU, MINGGANG, PASS, PENG, Action, Phase,
)
from .shanten import shanten, useful_tile_score
from .tiles import NUM_TILE_TYPES, YAOJI


class Bot:
    """Bot 只需要实现 ``choose(game) -> Action``。"""

    name = "bot"

    def __call__(self, game) -> Action:  # pragma: no cover - 由子类实现
        return self.choose(game)

    def choose(self, game) -> Action:  # pragma: no cover
        raise NotImplementedError

    def reset(self) -> None:
        pass


# ---------------------------------------------------------------------------
# 随机基线
# ---------------------------------------------------------------------------
class RandomBot(Bot):
    name = "random"

    def __init__(self, seed: int = 0, hu_prob: float = 1.0):
        self.rng = random.Random(seed)
        self.hu_prob = hu_prob

    def reset(self, seed: int | None = None) -> None:
        if seed is not None:
            self.rng = random.Random(seed)

    def choose(self, game) -> Action:
        acts = game.legal_actions()
        if not acts:
            return Action(PASS)
        if self.hu_prob >= 1.0:
            for a in acts:
                if a.kind == HU:
                    return a
        elif any(a.kind == HU for a in acts) and self.rng.random() > self.hu_prob:
            acts = [a for a in acts if a.kind != HU]
        return self.rng.choice(acts)


# ---------------------------------------------------------------------------
# 启发式教师
# ---------------------------------------------------------------------------
@dataclass
class HeuristicWeights:
    """教师风格的权重。换一组权重就是一个不同"性格"的教师。"""

    chongfeng: float = 9.0       # 抢先打出第一张幺鸡拿冲锋鸡的激励
    keep_yaoji: float = 2.0      # 非首张时保留幺鸡的激励
    safety: float = 0.6          # 打熟张（已现张多）的安全性权重
    tenpai_bonus: float = 40.0   # 进入听牌的额外奖励
    noise: float = 0.0           # 决策噪声，用于制造教师多样性
    gang_aggression: float = 0.0 # >0 更爱杠，<0 更保守
    hu_greed: float = 1.0        # 1.0 = 能胡就胡


DEFAULT_WEIGHTS = HeuristicWeights()


class HeuristicBot(Bot):
    """向听数驱动的启发式教师。"""

    name = "heuristic"

    def __init__(self, seed: int = 0, weights: HeuristicWeights | None = None):
        self.rng = random.Random(seed)
        self.w = weights or DEFAULT_WEIGHTS

    def reset(self, seed: int | None = None) -> None:
        if seed is not None:
            self.rng = random.Random(seed)

    # -- 入口 --------------------------------------------------------------
    def choose(self, game) -> Action:
        ph = game.phase
        if ph == Phase.SELF_KONG:
            # 抢杠决策点的动作形态是 [过, 胡]，语义属于"响应"，不是"要不要杠"
            if getattr(game, "_rob_kong", None) is not None:
                return self._respond(game)
            return self._self_kong(game)
        if ph == Phase.DISCARD:
            return self._discard(game)
        if ph == Phase.RESPOND:
            return self._respond(game)
        return Action(PASS)

    # -- 出牌 --------------------------------------------------------------
    def _discard(self, game) -> Action:
        p = game.current
        acts = game.legal_actions()
        hu = [a for a in acts if a.kind == HU]
        if hu and self.w.hu_greed >= 1.0:
            return hu[0]

        tiles = [a.tile for a in acts if a.kind == DISCARD]
        if not tiles:
            return Action(DISCARD, game.hand_tiles(p)[0]) if game.hand_tiles(p) else Action(PASS)

        n_melds = len(game.melds[p])
        hand = game.hands[p]
        visible = self._visible_counts(game, p)
        first_yaoji = (game.chongfeng_player is None
                       and not game._first_yaoji_discarded)

        best_a, best_s = None, -1e18
        for t in tiles:
            h = list(hand)
            h[t] -= 1
            s = shanten(h, n_melds)
            score = -100.0 * s + useful_tile_score(h, n_melds)

            if s == 0:
                score += self.w.tenpai_bonus

            # 冲锋鸡：抢第一张幺鸡是明显的正收益
            if t == YAOJI:
                if first_yaoji:
                    score += self.w.chongfeng
                else:
                    score -= self.w.keep_yaoji * 2
            elif h[YAOJI]:
                score += self.w.keep_yaoji

            # 安全性：打过/现过越多的牌越安全
            score += self.w.safety * visible[t]

            # 豆与鸡的持有价值：成对/成刻的幺鸡别乱拆（由 useful 覆盖一部分）
            if self.w.noise > 0:
                score += self.rng.gauss(0.0, self.w.noise)
            if score > best_s:
                best_s, best_a = score, Action(DISCARD, t)
        return best_a

    def _visible_counts(self, game, p: int) -> list[int]:
        v = [0] * NUM_TILE_TYPES
        for q in range(4):
            for t in game.discards[q]:
                v[t] += 1
            for _, t, _ in game.melds[q]:
                v[t] += 3
        for t, c in enumerate(game.hands[p]):
            v[t] += c
        return v

    # -- 摸牌后的闷豆 / 爬坡豆 --------------------------------------------
    def _self_kong(self, game) -> Action:
        p = game.actor()          # 抢杠时 actor 是抢杠者，不是 current
        acts = game.legal_actions()
        if len(acts) == 1:
            return acts[0]
        hand = game.hands[p]
        n_melds = len(game.melds[p])
        cur = shanten(hand, n_melds)

        best = Action(PASS)
        best_s = cur - self.w.gang_aggression
        for a in acts:
            # 注意：抢杠判断时 acts 只含 [过, 胡]，必须显式过滤，
            # 否则会把 tile=-1 的 HU 当成补杠去算 h[-1]，污染手牌计数。
            if a.kind not in (ANGANG, BUGANG):
                continue
            t = a.tile
            if a.kind == ANGANG:
                h = list(hand)
                h[t] -= 4
                after = shanten(h, n_melds + 1)
            else:  # BUGANG
                h = list(hand)
                h[t] -= 1
                after = shanten(h, n_melds)
            if after <= best_s:
                best_s = after
                best = a
        return best

    # -- 响应 --------------------------------------------------------------
    def _respond(self, game) -> Action:
        p = game.actor()
        acts = game.legal_actions()
        hu = [a for a in acts if a.kind == HU]
        if hu and self.w.hu_greed >= 1.0:
            return hu[0]

        hand = game.hands[p]
        n_melds = len(game.melds[p])
        cur = shanten(hand, n_melds)
        tile = game.last_discard[1] if game.last_discard else -1

        mg = [a for a in acts if a.kind == MINGGANG]
        if mg:
            h = list(hand)
            h[tile] -= 3
            after = shanten(h, n_melds + 1)
            # 点豆送一个通行证，向听不变也值得做
            if after <= cur + 0.0:
                return mg[0]

        pg = [a for a in acts if a.kind == PENG]
        if pg:
            h = list(hand)
            h[tile] -= 2
            after = shanten(h, n_melds + 1)
            if after < cur:
                return pg[0]
            if after == cur and self.rng.random() < 0.25:
                return pg[0]

        return Action(PASS)

    # -- 供自对弈使用：动作打分（未使用，保留接口） -----------------------
    def action_values(self, game):  # pragma: no cover
        return None


# ---------------------------------------------------------------------------
# 教师池
# ---------------------------------------------------------------------------
TEACHER_STYLES = {
    "balanced": HeuristicWeights(),
    "aggressive": HeuristicWeights(chongfeng=14.0, safety=0.2, noise=3.0,
                                   gang_aggression=0.5),
    "defensive": HeuristicWeights(chongfeng=3.0, safety=1.4, noise=2.0,
                                  gang_aggression=-0.5),
    "chicken_lover": HeuristicWeights(chongfeng=4.0, keep_yaoji=6.0, noise=2.5),
    "tenpai_rush": HeuristicWeights(tenpai_bonus=90.0, safety=0.3, noise=4.0,
                                    keep_yaoji=0.5),
    "gambler": HeuristicWeights(noise=9.0, gang_aggression=1.5, safety=0.0),
}


def make_teacher_pool(seed: int = 0) -> list[HeuristicBot]:
    """构造一组风格不同的教师，用于混合策略的行为克隆。"""
    bots = []
    for i, (name, w) in enumerate(TEACHER_STYLES.items()):
        b = HeuristicBot(seed=seed * 131 + i, weights=w)
        b.name = f"heuristic:{name}"
        bots.append(b)
    return bots
