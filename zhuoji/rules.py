"""
贵州捉鸡麻将（贵阳捉鸡）规则引擎。

设计
----
* 纯 Python、无第三方依赖；单局毫秒级，便于批量自对弈生成数据。
* 规则分歧点全部收进 :class:`RulesConfig`，默认取"通用贵阳捉鸡"口径，
  文档里标注了各地差异，方便按自己牌桌的规矩改。
* 动作是显式的 :class:`Action`，对局分成三个决策点：
  ``SELF_KONG``（摸牌后是否闷豆/爬坡豆）→ ``DISCARD``（出牌 或 自摸胡）
  → ``RESPOND``（对他家出牌的 碰/点豆/胡/过）。

核心规则
--------
* 108 张：万/条/筒 各 1-9，每种 4 张。**不能吃**。
* 「豆」（杠）是点胡的通行证：无豆时平胡只能自摸；平胡以上牌型可无豆点胡。
* 「鸡」：幺鸡(T1) 恒为鸡；胡牌后翻牌墙第一张未摸牌，其"点数+1 同花色"也是鸡；
  翻到 9 条时幺鸡升为「金鸡」（每张 2 番）。
* 黄庄（牌摸完无人胡）→ 查叫，未叫者赔叫牌者番值；黄庄时豆、鸡一律不算。
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from enum import Enum

from .fan import (
    FAN_CN,
    MELD_KONG_ADDED,
    MELD_KONG_CONCEALED,
    MELD_KONG_EXPOSED,
    MELD_PONG,
    classify_fan,
    is_high_value_hand,
    is_tenpai,
    is_winning_hand,
    meld_is_kong,
    winning_tiles,
)
from .tiles import (
    NUM_TILE_TYPES,
    YAOJI,
    full_deck,
    rank_of,
    suit_of,
    tile_cn,
    tile_name,
)


# ---------------------------------------------------------------------------
# 规则配置
# ---------------------------------------------------------------------------
@dataclass
class RulesConfig:
    """贵阳捉鸡规则档 —— 所有地区分歧点都在这里。"""

    #: 底分。最终得分 = 底分 × (牌型番 + 豆番 + 鸡番)
    base_score: float = 1.0

    # ---- 豆（杠）----
    men_dou: float = 2.0            # 闷豆（暗杠）：其他三家各付
    pa_po_dou: float = 3.0          # 爬坡豆（补杠）：其他三家各付
    dian_dou: float = 1.0           # 点豆（明杠）数额
    #: 点豆番由谁承担。"konger"=杠的人付给被点者（百科口径，默认）；"discarder"=反过来
    dian_dou_payer: str = "konger"
    #: 豆必须听牌才生效（"得豆必须听牌"）
    dou_requires_tenpai: bool = True

    # ---- 鸡 ----
    #: 鸡的统计范围："hand_meld"=手牌+副露；"include_river"=再加上自己打出的
    chicken_scope: str = "hand_meld"
    chongfeng_multiplier: float = 2.0   # 冲锋鸡倍率
    gold_chicken: float = 2.0           # 金鸡每张番数
    zeren_extra: float = 1.0            # 责任鸡：打出者比其他家多付的番
    man_tang_chicken: bool = False      # 满堂鸡（部分玩法，默认关）

    # ---- 胡牌通行证 ----
    ping_hu_needs_dou_for_ron: bool = True

    # ---- 特殊规则 ----
    enable_qiang_gang: bool = True
    enable_gang_shang_pao: bool = True
    enable_baoting: bool = False        # 报听/杀报（复杂，默认关）
    enable_pass_lock: bool = True       # 过水：过掉胡牌后摸牌前不能再胡同张

    # ---- 庄家 ----
    dealer_extra_fan: float = 1.0
    dealer_streak_fan: float = 1.0
    first_dealer: int | None = None


# ---------------------------------------------------------------------------
# 阶段与动作
# ---------------------------------------------------------------------------
class Phase(str, Enum):
    SELF_KONG = "self_kong"
    DISCARD = "discard"
    RESPOND = "respond"
    OVER = "over"


DISCARD = "discard"
HU = "hu"
ANGANG = "angang"
BUGANG = "bugang"
PENG = "peng"
MINGGANG = "minggang"
PASS = "pass"

#: 网络输出用的固定动作编号
RESPOND_ACTIONS = (PASS, PENG, MINGGANG, HU)   # 0..3
SELF_KONG_ACTIONS = (PASS, ANGANG, BUGANG)     # 0..2


@dataclass(frozen=True)
class Action:
    kind: str
    tile: int = -1

    def __repr__(self) -> str:  # pragma: no cover
        return self.kind if self.tile < 0 else f"{self.kind}:{tile_name(self.tile)}"

    def label(self) -> str:
        cn = {
            DISCARD: "打", HU: "胡", ANGANG: "闷豆", BUGANG: "爬坡豆",
            PENG: "碰", MINGGANG: "点豆", PASS: "过",
        }[self.kind]
        return cn if self.tile < 0 else f"{cn}{tile_cn(self.tile)}"


# ---------------------------------------------------------------------------
# 主引擎
# ---------------------------------------------------------------------------
class ZhuojiGame:
    """一局贵阳捉鸡。"""

    def __init__(self, config: RulesConfig | None = None, seed: int | None = None,
                 dealer: int | None = None, dealer_streak: int = 0):
        self.cfg = config or RulesConfig()
        self.rng = random.Random(seed)
        self.dealer_streak = dealer_streak
        self._start(dealer)

    # -- 初始化 ------------------------------------------------------------
    def _start(self, dealer: int | None) -> None:
        cfg = self.cfg
        wall = full_deck()
        self.rng.shuffle(wall)
        self.wall: list[int] = wall
        self.wall_pos = 0

        self.hands: list[list[int]] = [[0] * NUM_TILE_TYPES for _ in range(4)]
        for _ in range(13):
            for p in range(4):
                self.hands[p][self._draw_from_wall()] += 1

        self.melds: list[list[tuple[int, int, int]]] = [[] for _ in range(4)]
        self.discards: list[list[int]] = [[] for _ in range(4)]
        self.ever_tenpai = [False] * 4
        self.pass_lock: list[set[int]] = [set() for _ in range(4)]
        self.gang_discard = [False] * 4
        self.drawn = False          # 当前玩家是否刚摸过牌（碰/杠进牌后为 False）

        if dealer is not None:
            self.dealer = dealer
        elif cfg.first_dealer is not None:
            self.dealer = cfg.first_dealer
        else:
            self.dealer = self.rng.randrange(4)

        self.current = self.dealer
        self.hands[self.dealer][self._draw_from_wall()] += 1
        self.drawn = True

        self.phase = Phase.OVER
        self.last_discard: tuple[int, int] | None = None
        self.last_discard_was_gang = False
        self._respond_order: list[int] = []
        self._respond_idx = 0
        self._rob_kong: tuple[int, int] | None = None

        self.chongfeng_player: int | None = None
        self.zeren_player: int | None = None
        self.zeren_ponger: int | None = None
        self._first_yaoji_discarded = False

        self.result: dict | None = None
        self.score_delta: list[float] = [0.0] * 4
        self.next_dealer: int = self.dealer
        self.log: list[tuple[int, str]] = []

        self._enter_self_phase(self.dealer)

    # -- 基础设施 ----------------------------------------------------------
    def _draw_from_wall(self) -> int:
        t = self.wall[self.wall_pos]
        self.last_draw = t          # 最近一次从牌墙摸进的牌（供流水/展示）
        self.wall_pos += 1
        return t

    @property
    def wall_left(self) -> int:
        return len(self.wall) - self.wall_pos

    def hand(self, p: int) -> list[int]:
        return self.hands[p]

    def has_dou(self, p: int) -> bool:
        return any(meld_is_kong(mt) for mt, _, _ in self.melds[p])

    def melded_count(self, p: int, tile: int) -> int:
        c = 0
        for mt, t, _ in self.melds[p]:
            if t == tile:
                c += 4 if meld_is_kong(mt) else 3
        return c

    def hand_tiles(self, p: int) -> list[int]:
        out = []
        for t, c in enumerate(self.hands[p]):
            out.extend([t] * c)
        return out

    def _opponents(self, p: int) -> list[int]:
        return [(p + i) % 4 for i in range(1, 4)]

    def actor(self) -> int:
        """当前该行动的人。

        注意抢杠：补杠者发起补杠后，决策权立刻转给抢杠者，
        此时 ``current`` 仍是补杠者，必须返回 ``_rob_kong[0]``。
        """
        if self.phase == Phase.RESPOND:
            return self._respond_order[self._respond_idx]
        if self.phase == Phase.SELF_KONG and self._rob_kong is not None:
            return self._rob_kong[0]
        return self.current

    # -- 胡牌判定 ----------------------------------------------------------
    def can_tsumo(self, p: int) -> bool:
        return is_winning_hand(self.hands[p], self.melds[p])

    def can_ron(self, p: int, tile: int) -> bool:
        if self.cfg.enable_pass_lock and tile in self.pass_lock[p]:
            return False
        h = list(self.hands[p])
        h[tile] += 1
        if not is_winning_hand(h, self.melds[p]):
            return False
        if self.cfg.ping_hu_needs_dou_for_ron:
            if not self.has_dou(p) and not is_high_value_hand(h, self.melds[p]):
                return False
        return True

    def _can_peng(self, p: int, tile: int) -> bool:
        return len(self.melds[p]) < 4 and self.hands[p][tile] >= 2

    def _can_minggang(self, p: int, tile: int) -> bool:
        return len(self.melds[p]) < 4 and self.hands[p][tile] >= 3

    # -- 阶段切换 ----------------------------------------------------------
    def _self_kong_options(self, p: int) -> list[tuple[str, int]]:
        opts: list[tuple[str, int]] = []
        h = self.hands[p]
        for t in range(NUM_TILE_TYPES):
            if h[t] == 4:
                opts.append((ANGANG, t))
        for mt, t, _ in self.melds[p]:
            if mt == MELD_PONG and h[t] >= 1:
                opts.append((BUGANG, t))
        return opts

    def _enter_self_phase(self, p: int) -> None:
        self._update_tenpai(p)
        if self._rob_kong is not None or self._self_kong_options(p):
            self.phase = Phase.SELF_KONG
        else:
            self.phase = Phase.DISCARD

    def _update_tenpai(self, p: int) -> None:
        if not self.ever_tenpai[p] and is_tenpai(self.hands[p], self.melds[p]):
            self.ever_tenpai[p] = True

    # -- 合法动作 ----------------------------------------------------------
    def legal_actions(self) -> list[Action]:
        if self.phase == Phase.SELF_KONG:
            if self._rob_kong is not None:
                p, t = self._rob_kong
                acts = [Action(PASS)]
                if self.can_ron(p, t):
                    acts.append(Action(HU))
                return acts
            p = self.current
            acts = [Action(PASS)]
            for kind, t in self._self_kong_options(p):
                acts.append(Action(kind, t))
            return acts

        if self.phase == Phase.DISCARD:
            p = self.current
            acts = [Action(DISCARD, t) for t in range(NUM_TILE_TYPES) if self.hands[p][t] > 0]
            if self.drawn and self.can_tsumo(p):
                acts.append(Action(HU))
            return acts

        if self.phase == Phase.RESPOND:
            p = self._current_responder()
            acts = [Action(PASS)]
            if self._rob_kong is not None:
                if self.can_ron(p, self._rob_kong[1]):
                    acts.append(Action(HU))
                return acts
            tile = self.last_discard[1]
            if self.can_ron(p, tile):
                acts.append(Action(HU))
            if self._can_minggang(p, tile):
                acts.append(Action(MINGGANG, tile))
            if self._can_peng(p, tile):
                acts.append(Action(PENG, tile))
            return acts

        return []

    def _current_responder(self) -> int:
        return self._respond_order[self._respond_idx]

    # -- 主循环 ------------------------------------------------------------
    def step(self, action: Action) -> bool:
        if self.phase == Phase.OVER:
            return True
        self.log.append((self.actor(), action.label()))
        if self.phase == Phase.SELF_KONG:
            self._step_self_kong(action)
        elif self.phase == Phase.DISCARD:
            self._step_discard(action)
        else:
            self._step_respond(action)
        return self.phase == Phase.OVER

    # 阶段一：摸牌后的闷豆 / 爬坡豆
    def _step_self_kong(self, action: Action) -> None:
        if self._rob_kong is not None:
            p, t = self._rob_kong
            konger = self._rob_kong_konger
            self._rob_kong = None
            if action.kind == HU:
                self._settle_win(p, t, is_tsumo=False, loser=konger, rob_kong=True)
            else:
                self._do_bugang(konger, t)
            return

        p = self.current
        if action.kind == PASS:
            self.phase = Phase.DISCARD
            return
        if action.kind == ANGANG:
            t = action.tile
            self.hands[p][t] -= 4
            self.melds[p].append((MELD_KONG_CONCEALED, t, p))
            self._gang_draw(p)
            return
        if action.kind == BUGANG:
            t = action.tile
            if self.cfg.enable_qiang_gang:
                robbers = [q for q in self._opponents(p) if self.can_ron(q, t)]
                if robbers:
                    self._rob_kong = (robbers[0], t)
                    self._rob_kong_konger = p
                    self.phase = Phase.SELF_KONG
                    return
            self._do_bugang(p, t)

    def _do_bugang(self, p: int, t: int) -> None:
        self.hands[p][t] -= 1
        for i, (mt, tt, src) in enumerate(self.melds[p]):
            if mt == MELD_PONG and tt == t:
                self.melds[p][i] = (MELD_KONG_ADDED, t, src)
                break
        self._gang_draw(p)

    def _gang_draw(self, p: int) -> None:
        if self.wall_left <= 0:
            self._settle_huangzhuang()
            return
        self.hands[p][self._draw_from_wall()] += 1
        self.current = p
        self.drawn = True
        self.gang_discard[p] = True
        self.pass_lock[p] = set()
        self._enter_self_phase(p)

    # 阶段二：出牌 / 自摸胡
    def _step_discard(self, action: Action) -> None:
        p = self.current
        if action.kind == HU:
            self._settle_win(p, -1, is_tsumo=True)
            return
        t = action.tile
        self.hands[p][t] -= 1
        self.discards[p].append(t)

        if t == YAOJI and not self._first_yaoji_discarded:
            self._first_yaoji_discarded = True
            self.chongfeng_player = p

        self._update_tenpai(p)
        self.last_discard_was_gang = self.gang_discard[p]
        self.gang_discard[p] = False
        self.last_discard = (p, t)
        self._begin_respond(p, t)

    def _begin_respond(self, discarder: int, tile: int) -> None:
        others = self._opponents(discarder)
        hu_list = [q for q in others if self.can_ron(q, tile)]
        claim_list = [q for q in others if q not in hu_list
                      and (self._can_peng(q, tile) or self._can_minggang(q, tile))]
        order = hu_list + claim_list
        if order:
            self._respond_order = order
            self._respond_idx = 0
            self.phase = Phase.RESPOND
        else:
            self._next_player(discarder)

    def _next_player(self, from_p: int) -> None:
        nxt = (from_p + 1) % 4
        if self.wall_left <= 0:
            self._settle_huangzhuang()
            return
        self.hands[nxt][self._draw_from_wall()] += 1
        self.current = nxt
        self.drawn = True
        self.gang_discard[nxt] = False
        self.pass_lock[nxt] = set()
        self._enter_self_phase(nxt)

    # 阶段三：响应
    def _step_respond(self, action: Action) -> None:
        p = self._current_responder()
        discarder, tile = self.last_discard

        if action.kind == PASS:
            self.pass_lock[p].add(tile)
            self._respond_idx += 1
            if self._respond_idx >= len(self._respond_order):
                self._respond_order = []
                self._respond_idx = 0
                self._next_player(discarder)
            return

        if action.kind == HU:
            self._respond_order = []
            self._settle_win(p, tile, is_tsumo=False, loser=discarder)
            return

        if self.discards[discarder] and self.discards[discarder][-1] == tile:
            self.discards[discarder].pop()

        if action.kind == PENG:
            self.hands[p][tile] -= 2
            self.melds[p].append((MELD_PONG, tile, discarder))
            if tile == YAOJI and self.zeren_player is None \
                    and self.chongfeng_player is not None:
                self.zeren_player = discarder
                self.zeren_ponger = p
                self.chongfeng_player = None
            self.current = p
            self.drawn = False        # 碰后未摸牌，不可自摸胡（须先出牌）
            self.pass_lock[p] = set()
            self._respond_order = []
            self._respond_idx = 0
            self._enter_self_phase(p)

        elif action.kind == MINGGANG:
            self.hands[p][tile] -= 3
            self.melds[p].append((MELD_KONG_EXPOSED, tile, discarder))
            self.current = p
            self.pass_lock[p] = set()
            self._respond_order = []
            self._respond_idx = 0
            self._gang_draw(p)

    # -- 结算 --------------------------------------------------------------
    def _chicken_tiles(self) -> tuple[set[int], bool]:
        if self.wall_left <= 0:
            return set(), False
        flip = self.wall[self.wall_pos]
        s, r = suit_of(flip), rank_of(flip)
        nxt = s * 9 + (r % 9)          # 点数 +1；9 回到 1
        return {YAOJI, nxt}, (nxt == YAOJI)

    def _chicken_count_and_fan(self, p: int, chicken: set[int], gold: bool):
        n = 0
        fan = 0.0
        for t in chicken:
            c = self.hands[p][t] + self.melded_count(p, t)
            if self.cfg.chicken_scope == "include_river":
                c += self.discards[p].count(t)
            if c:
                n += c
                fan += c * (self.cfg.gold_chicken if (gold and t == YAOJI) else 1.0)
        return n, fan

    def _settle_dou(self, void_player: int | None):
        cfg = self.cfg
        delta = [0.0] * 4
        detail = []
        for p in range(4):
            if void_player is not None and p == void_player:
                continue
            if cfg.dou_requires_tenpai and not self.ever_tenpai[p]:
                continue
            for mt, t, src in self.melds[p]:
                if mt == MELD_KONG_CONCEALED:
                    for q in range(4):
                        if q != p:
                            delta[q] -= cfg.men_dou
                            delta[p] += cfg.men_dou
                    detail.append((p, "闷豆", tile_name(t), cfg.men_dou))
                elif mt == MELD_KONG_ADDED:
                    for q in range(4):
                        if q != p:
                            delta[q] -= cfg.pa_po_dou
                            delta[p] += cfg.pa_po_dou
                    detail.append((p, "爬坡豆", tile_name(t), cfg.pa_po_dou))
                elif mt == MELD_KONG_EXPOSED:
                    if cfg.dian_dou_payer == "konger":
                        delta[p] -= cfg.dian_dou
                        delta[src] += cfg.dian_dou
                    else:
                        delta[src] -= cfg.dian_dou
                        delta[p] += cfg.dian_dou
                    detail.append((p, "点豆", tile_name(t), cfg.dian_dou))
        return detail, delta

    def _settle_chicken(self, void_player: int | None):
        cfg = self.cfg
        delta = [0.0] * 4
        detail = []
        chicken, gold = self._chicken_tiles()
        if not chicken:
            return detail, delta

        counts = [0] * 4
        fans = [0.0] * 4
        for p in range(4):
            if void_player is not None and p == void_player:
                continue
            n, f = self._chicken_count_and_fan(p, chicken, gold)
            counts[p], fans[p] = n, f
            if n:
                detail.append((p, "鸡", n, f))

        if cfg.man_tang_chicken and all(c > 0 for c in counts):
            fans = [f * 2 for f in fans]
            detail.append((-1, "满堂鸡", 0, 2.0))

        for p in range(4):
            if fans[p] <= 0:
                continue
            for q in range(4):
                if q != p:
                    delta[q] -= fans[p]
                    delta[p] += fans[p]

        if self.chongfeng_player is not None and self.chongfeng_player != void_player:
            cf = self.chongfeng_player
            val = (cfg.gold_chicken if gold else 1.0) * cfg.chongfeng_multiplier
            for q in range(4):
                if q != cf:
                    delta[q] -= val
                    delta[cf] += val
            detail.append((cf, "冲锋鸡", 1, val))

        if self.zeren_player is not None and self.zeren_ponger is not None:
            z, k = self.zeren_player, self.zeren_ponger
            if z != void_player:
                delta[z] -= cfg.zeren_extra
                delta[k] += cfg.zeren_extra
                detail.append((z, "责任鸡", 1, cfg.zeren_extra))

        return detail, delta

    def _settle_win(self, winner: int, win_tile: int, is_tsumo: bool,
                    loser: int | None = None, rob_kong: bool = False) -> None:
        cfg = self.cfg
        h = list(self.hands[winner])
        if win_tile >= 0:
            h[win_tile] += 1
        fan_type, fan_val = classify_fan(h, self.melds[winner])

        bonus = (cfg.dealer_extra_fan + cfg.dealer_streak_fan * self.dealer_streak) \
            if winner == self.dealer else 0.0
        total_fan = fan_val + bonus

        delta = [0.0] * 4
        if rob_kong:
            delta[winner] += 3 * total_fan
            delta[loser] -= 3 * total_fan
        elif is_tsumo:
            for q in range(4):
                if q != winner:
                    delta[q] -= total_fan
                    delta[winner] += total_fan
        else:
            delta[loser] -= total_fan
            delta[winner] += total_fan

        # 全烧：点炮者若是在自己杠后打出的第一张牌上放炮，其豆、鸡作废
        void_player = None
        if cfg.enable_gang_shang_pao and (not is_tsumo) and (not rob_kong) \
                and loser is not None and self.last_discard_was_gang:
            void_player = loser

        dou_detail, dou_delta = self._settle_dou(void_player)
        chicken_detail, chicken_delta = self._settle_chicken(void_player)
        for i in range(4):
            delta[i] += dou_delta[i] + chicken_delta[i]

        base = cfg.base_score
        self.result = {
            "type": "win",
            "winner": winner,
            "loser": loser,
            "is_tsumo": is_tsumo,
            "rob_kong": rob_kong,
            "fan_type": fan_type,
            "fan_cn": FAN_CN.get(fan_type, fan_type),
            "fan": fan_val,
            "dealer_bonus": bonus,
            "total_fan": total_fan,
            "dou": dou_detail,
            "chicken": chicken_detail,
            "void_player": void_player,
            "wall_left": self.wall_left,
        }
        self._finalize(delta, next_dealer=winner)

    def _settle_huangzhuang(self) -> None:
        cfg = self.cfg
        tenpai = [p for p in range(4) if is_tenpai(self.hands[p], self.melds[p])]
        delta = [0.0] * 4
        detail = []
        if 0 < len(tenpai) < 4:
            not_tenpai = [p for p in range(4) if p not in tenpai]
            for p in tenpai:
                best, best_type = 0, ""
                for t in winning_tiles(self.hands[p], self.melds[p]):
                    h = list(self.hands[p])
                    h[t] += 1
                    ft, fv = classify_fan(h, self.melds[p])
                    if fv > best:
                        best, best_type = fv, ft
                if best <= 0:
                    continue
                for q in not_tenpai:
                    delta[q] -= best
                    delta[p] += best
                detail.append((p, FAN_CN.get(best_type, best_type), best))
        self.result = {
            "type": "huangzhuang",
            "tenpai": tenpai,
            "cha_jiao": detail,
            "wall_left": 0,
        }
        self._finalize(delta, next_dealer=self.dealer)

    def _finalize(self, delta: list[float], next_dealer: int) -> None:
        base = self.cfg.base_score
        self.score_delta = [d * base for d in delta]
        self.result["deltas"] = list(self.score_delta)
        self.next_dealer = next_dealer
        self.phase = Phase.OVER

    # -- 便捷接口 ----------------------------------------------------------
    def play(self, agents, max_steps: int = 4000) -> dict:
        """用 4 个 ``agent(game) -> Action`` 跑完整局。"""
        steps = 0
        while self.phase != Phase.OVER and steps < max_steps:
            self.step(agents[self.actor()](self))
            steps += 1
        if self.phase != Phase.OVER:
            self.result = {"type": "timeout", "deltas": [0.0] * 4, "wall_left": self.wall_left}
            self.score_delta = [0.0] * 4
            self.next_dealer = self.dealer
            self.phase = Phase.OVER
        return self.result
