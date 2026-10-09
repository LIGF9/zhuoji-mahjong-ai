"""
状态编码 —— 把一名玩家视角下的对局状态转成定长张量。

设计原则
--------
1. **相对视角**：永远把"当前决策者"放在座位 0，下家 1、对家 2、上家 3。
   这样一次训练就能覆盖四个座位，网络不需要额外学座位不变性。
2. **多通道牌面特征**：15 个通道 × 27 种牌。通道区分"自己的手牌 / 四家牌河 /
   四家副露 / 全局可见度"，让卷积核能看出"这张牌在谁那儿出现过"。
3. **全局标量**：牌墙剩余、庄家、豆数、听牌状态、冲锋鸡/责任鸡归属等。
4. **动作头统一为三个**（和规则引擎的三个决策点一一对应）::

       head_discard    : 28  (0..26 打某张牌, 27 = 自摸胡)
       head_self_kong  : 28  (0..26 闷豆/爬坡豆某张牌, 27 = 过)
       head_respond    : 4   (过 / 碰 / 点豆 / 胡)

   每个头都配一个 mask，非法动作在 softmax 前被置为 -inf。
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .fan import MELD_KONG_ADDED, MELD_KONG_CONCEALED, MELD_KONG_EXPOSED, meld_is_kong
from .rules import ANGANG, BUGANG, DISCARD, HU, MINGGANG, PASS, PENG, Phase
from .tiles import NUM_TILE_TYPES, YAOJI

NUM_CHANNELS = 15
CTX_DIM = 27 + 4          # 上一张被打出的牌 one-hot + 出牌者相对座位
GLOBAL_DIM = 39 + CTX_DIM

HEAD_DISCARD = "discard"
HEAD_SELF_KONG = "self_kong"
HEAD_RESPOND = "respond"
HEAD_DIMS = {HEAD_DISCARD: 28, HEAD_SELF_KONG: 28, HEAD_RESPOND: 4}


@dataclass
class Observation:
    tiles: np.ndarray      # (NUM_CHANNELS, 27) float32
    glob: np.ndarray       # (GLOBAL_DIM,) float32
    head: str              # 当前阶段对应的动作头
    mask: np.ndarray       # (HEAD_DIMS[head],) bool


def _rel(p: int, me: int) -> int:
    return (p - me) % 4


def encode(game, player: int) -> Observation:
    """把 ``game`` 编码成 ``player`` 视角的观测。"""
    me = player
    cfg = game.cfg

    tiles = np.zeros((NUM_CHANNELS, NUM_TILE_TYPES), dtype=np.float32)
    hand = game.hands[me]

    # -- 自己的手牌（多层阈值特征） ----------------------------------------
    for t in range(NUM_TILE_TYPES):
        c = hand[t]
        if c:
            tiles[0, t] = 1.0
            tiles[4, t] = c * 0.25
            if c >= 2:
                tiles[1, t] = 1.0
            if c >= 3:
                tiles[2, t] = 1.0
            if c >= 4:
                tiles[3, t] = 1.0

    # -- 四家牌河（相对座位） ----------------------------------------------
    for p in range(4):
        r = _rel(p, me)
        for t in game.discards[p]:
            tiles[5 + r, t] += 0.25

    # -- 四家副露 ----------------------------------------------------------
    for p in range(4):
        r = _rel(p, me)
        for mt, t, _ in game.melds[p]:
            tiles[9 + r, t] += 1.0 if meld_is_kong(mt) else 0.75

    # -- 可见度 / 剩余 ------------------------------------------------------
    visible = np.zeros(NUM_TILE_TYPES, dtype=np.float32)
    visible += tiles[4] * 4.0
    for r in range(4):
        visible += tiles[5 + r] * 4.0
        visible += tiles[9 + r] * 4.0
    tiles[13] = np.clip(visible, 0.0, 4.0) / 4.0
    tiles[14] = np.clip(4.0 - visible, 0.0, 4.0) / 4.0

    # -- 全局标量 ----------------------------------------------------------
    g = np.zeros(GLOBAL_DIM, dtype=np.float32)
    wall_left = game.wall_left
    g[0] = wall_left / 108.0
    g[1] = (108.0 - wall_left) / 108.0
    g[2] = _rel(me, game.dealer) / 3.0
    g[3] = 1.0 if me == game.dealer else 0.0
    g[4] = game.dealer_streak / 5.0
    g[5 + _rel(game.dealer, me)] = 1.0

    for p in range(4):
        r = _rel(p, me)
        kongs = sum(1 for mt, _, _ in game.melds[p] if meld_is_kong(mt))
        g[9 + r] = kongs / 4.0
        g[13 + r] = len(game.melds[p]) / 4.0
        g[17 + r] = min(len(game.discards[p]), 24) / 24.0
        g[21 + r] = 1.0 if game.ever_tenpai[p] else 0.0

    g[25] = sum(hand) / 14.0
    g[26] = 1.0 if game.has_dou(me) else 0.0
    wait = _waits(game, me)
    g[27] = 1.0 if wait else 0.0
    g[28] = len(wait) / 13.0
    g[29] = (4 - visible[YAOJI]) / 4.0
    g[30] = (hand[YAOJI] + game.melded_count(me, YAOJI)) / 4.0
    g[31] = 1.0 if game._first_yaoji_discarded else 0.0
    g[32] = 1.0 if game.chongfeng_player == me else 0.0
    g[33] = 1.0 if game.chongfeng_player is not None else 0.0
    g[34] = 1.0 if game.zeren_player == me else 0.0
    g[35] = 1.0 if game.phase == Phase.SELF_KONG else 0.0
    g[36] = 1.0 if game.phase == Phase.DISCARD else 0.0
    g[37] = 1.0 if game.phase == Phase.RESPOND else 0.0
    g[38] = 1.0 if game.actor() == me else 0.0

    base = 39
    if game.last_discard is not None:
        d_p, d_t = game.last_discard
        g[base + d_t] = 1.0
        g[base + 27 + _rel(d_p, me)] = 1.0

    head, mask = action_head_and_mask(game, me)
    return Observation(tiles=tiles, glob=g, head=head, mask=mask)


def _waits(game, p: int) -> list[int]:
    from .fan import winning_tiles
    h = game.hands[p]
    if sum(h) != (4 - len(game.melds[p])) * 3 + 1:
        return []
    return winning_tiles(h, game.melds[p])


def action_head_and_mask(game, player: int):
    """返回当前阶段应以哪个头决策，以及该头的合法动作掩码。"""
    if game.phase == Phase.DISCARD:
        p = game.current
        mask = np.zeros(HEAD_DIMS[HEAD_DISCARD], dtype=bool)
        for t in range(NUM_TILE_TYPES):
            if game.hands[p][t] > 0:
                mask[t] = True
        # 必须与 legal_actions 一致：碰牌后 drawn=False（未摸牌）不可自摸胡，
        # 否则模型会把「非法自摸」当成合法动作推荐（网页端表现为无胡按钮却提交 hu）
        if game.drawn and game.can_tsumo(p):
            mask[27] = True
        return HEAD_DISCARD, mask

    if game.phase == Phase.SELF_KONG:
        mask = np.zeros(HEAD_DIMS[HEAD_SELF_KONG], dtype=bool)
        mask[27] = True                      # 过
        if game._rob_kong is not None:
            p, t = game._rob_kong
            if game.can_ron(p, t):
                return HEAD_RESPOND, _respond_mask(game, p)
            return HEAD_SELF_KONG, mask
        for kind, t in game._self_kong_options(game.current):
            mask[t] = True
        return HEAD_SELF_KONG, mask

    if game.phase == Phase.RESPOND:
        p = game.actor()
        return HEAD_RESPOND, _respond_mask(game, p)

    return HEAD_DISCARD, np.zeros(HEAD_DIMS[HEAD_DISCARD], dtype=bool)


def _respond_mask(game, p: int) -> np.ndarray:
    mask = np.zeros(4, dtype=bool)
    mask[0] = True                                              # 过
    if game._rob_kong is not None:
        if game.can_ron(p, game._rob_kong[1]):
            mask[3] = True                                      # 抢杠胡
        return mask
    tile = game.last_discard[1]
    if game.can_ron(p, tile):
        mask[3] = True
    if game._can_minggang(p, tile):
        mask[2] = True
    if game._can_peng(p, tile):
        mask[1] = True
    return mask


def action_to_index(action) -> int:
    """把 :class:`Action` 映射到它所在头的下标（需已知头时请用 :func:`head_index`）。"""
    if action.kind == DISCARD:
        return action.tile
    if action.kind == HU:
        return 27
    if action.kind in (ANGANG, BUGANG):
        return action.tile
    if action.kind == PASS:
        return 27
    if action.kind == PENG:
        return 1
    if action.kind == MINGGANG:
        return 2
    raise ValueError(f"未知动作 {action}")


def head_index(head: str, action) -> int:
    """给定动作头，返回动作下标。"""
    if head == HEAD_DISCARD:
        if action.kind == HU:
            return 27
        if action.kind == DISCARD:
            return action.tile
        raise ValueError(f"{action} 不属于 {head}")
    if head == HEAD_SELF_KONG:
        if action.kind == PASS:
            return 27
        if action.kind in (ANGANG, BUGANG):
            return action.tile
        raise ValueError(f"{action} 不属于 {head}")
    # respond
    if action.kind == PASS:
        return 0
    if action.kind == PENG:
        return 1
    if action.kind == MINGGANG:
        return 2
    if action.kind == HU:
        return 3
    raise ValueError(f"{action} 不属于 {head}")


def index_to_action(head: str, idx: int, game=None):
    """把头部下标还原成 :class:`Action`。"""
    from .rules import Action
    if head == HEAD_RESPOND:
        kind = (PASS, PENG, MINGGANG, HU)[idx]
        tile = -1
        if kind in (PENG, MINGGANG) and game is not None and game.last_discard:
            tile = game.last_discard[1]
        return Action(kind, tile)
    if head == HEAD_DISCARD:
        return Action(HU if idx == 27 else DISCARD, -1 if idx == 27 else idx)
    # self_kong
    if idx == 27:
        return Action(PASS)
    if game is not None:
        for kind, t in game._self_kong_options(game.current):
            if t == idx:
                return Action(kind, t)
    return Action(PASS)
