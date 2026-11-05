# -*- coding: utf-8 -*-
"""公开信息视图（``PublicView``）：推荐理由的**物理隔离层**。

2026-11-04 用户口径：「优化搜索 agent 的推荐出牌理由，确保没有使用全局信息
来计算推荐和编写推荐理由」。这里采用**物理隔离**而不是「自律不读」——理由层
拿到的对象里根本没有别家手牌、未翻开的牌墙、未公开的听牌状态等字段，从
数据结构上就无从访问全局信息。

公开信息清单（= p 座位玩家在实战中真正看得见的东西）：

  * 自己的暗手 27 维计数；
  * 四家弃牌、四家副露（含类型 / 牌种 / 来源座位）；
  * 本局鸡牌种、开局已知金鸡牌种（开局翻鸡已公开）；
  * 各家报叫标志（报叫是公开动作）、刚打出的那张牌；
  * 规则配置里的公开计分常量；
  * 牌墙剩余**张数**（只是计数，不含牌序）。

明确**不**包含：

  * 别家暗手 ``hands[q]``（q != p）；
  * 牌墙内容 ``wall[]`` / ``wall_pos`` 指向的真实牌序；
  * 任何终局诊断 / determinization 内部统计 / 价值头输出。

构造只通过 ``PublicView.from_game(game, p)``：这一步会短暂接触完整 game，把
公开字段拷成只读结构后立刻丢掉引用；之后所有派生计算（向听、进张、已现
张数、鸡分、点炮敞口）都只依赖 view 自身。审计脚本
``tools/verify_reason_public.py`` 会验证：任意打乱别家手牌与牌墙的真实排列，
Q 值与生成的理由逐字不变。
"""
from __future__ import annotations

from dataclasses import dataclass

from .fan import meld_is_kong, winning_tiles
from .shanten import shanten
from .tiles import NUM_TILE_TYPES, tile_cn

# 只抽取「公开且理由层用得到」的计分常量，view 不持有整个 cfg 对象。
_PUB_CFG = {
    "jin_mult": ("jin_ji_multiplier", 2.0),
    "ji_chongfeng": ("ji_chongfeng", 3.0),
    "ji_heng": ("ji_heng", 2.0),
    "ji_discard_plain": ("ji_discard_plain", 1.0),
    "ji_hold": ("ji_hold", 1.0),
    "dian_pao": ("dian_pao", 3.0),
    "kong_chickens": ("kong_chickens", 3.0),
}


@dataclass(frozen=True)
class PublicView:
    """p 座位的公开信息快照；**不含任何隐藏字段**。"""

    me: int
    hand: tuple[int, ...]                                   # 自己暗手 27 维
    discards: tuple[tuple[int, ...], ...]                   # 四家弃牌
    melds: tuple[tuple[tuple[int, int, int], ...], ...]    # 四家副露 (mt, tile, src)
    ji_tiles: frozenset[int]
    gold_tiles: frozenset[int]                              # 开局已知金鸡（公开）
    baojiao: tuple[bool, ...]
    wall_left: int                                          # 仅计数，不含牌序
    last_discard: tuple[int, int] | None                    # (seat, tile)
    kaiju_flip: int | None
    score: dict                                             # 公开计分常量
    ji_discard_names: dict                                  # {tile: 冲锋鸡/横鸡/幺鸡/None}
    n_melds: int

    # ---- 唯一构造入口 ---------------------------------------------------
    @classmethod
    def from_game(cls, game, p: int) -> "PublicView":
        cfg = game.cfg
        score = {k: float(getattr(cfg, attr, dflt))
                 for k, (attr, dflt) in _PUB_CFG.items()}
        ji_tiles = frozenset(int(t) for t in game.ji_tiles)
        try:
            gold = frozenset(int(t) for t in game.kaiju_gold_tiles())
        except Exception:
            gold = frozenset()
        # 打某张鸡会成为什么类型（横鸡轮是公开动作序列，构造时一次算好）
        names = {}
        for t in ji_tiles:
            try:
                names[int(t)] = game.ji_name_if_discarded(p, int(t)) or None
            except Exception:
                names[int(t)] = None
        melds = tuple(tuple((int(mt), int(tt), int(src))
                            for mt, tt, src in game.melds[q])
                      for q in range(4))
        disc = tuple(tuple(int(x) for x in game.discards[q]) for q in range(4))
        bao = tuple(bool(game.baojiao[q]) for q in range(4))
        ld = game.last_discard
        return cls(
            me=int(p),
            hand=tuple(int(x) for x in game.hands[p]),
            discards=disc,
            melds=melds,
            ji_tiles=ji_tiles,
            gold_tiles=gold,
            baojiao=bao,
            wall_left=int(getattr(game, "wall_left", 0)),
            last_discard=(int(ld[0]), int(ld[1])) if ld else None,
            kaiju_flip=(int(game.kaiju_flip) if game.kaiju_flip is not None else None),
            score=score,
            ji_discard_names=names,
            n_melds=len(game.melds[p]),
        )

    # ---- 公开派生指标（全部只读 view，不碰 game） ------------------------
    def visible(self, t: int) -> int:
        """牌 t 在公开信息里已现几张：四家弃牌 + 四家副露 + 自己手牌。"""
        n = sum(d.count(t) for d in self.discards)
        for ms in self.melds:
            for mt, tt, _src in ms:
                if tt == t:
                    n += 4 if meld_is_kong(mt) else 3
        n += self.hand[t]
        return min(4, n)

    def live(self, t: int) -> int:
        """公开信息之外牌 t 还剩几张（可供别人摸/胡，也供自己进张）。"""
        return max(0, 4 - self.visible(t))

    def after_discard(self, t: int) -> tuple[int, ...]:
        h = list(self.hand)
        if 0 <= t < NUM_TILE_TYPES and h[t] > 0:
            h[t] -= 1
        return tuple(h)

    def shanten_after(self, t: int) -> int:
        """打出 t 后的向听数（-1=已成胡，0=听牌）。"""
        return shanten(list(self.after_discard(t)), self.n_melds)

    def ting_after(self, t: int) -> list[int]:
        """打出 t 后听哪些牌（未听牌返回空表）。"""
        return list(winning_tiles(list(self.after_discard(t)),
                                 list(self.melds[self.me])))

    def ukeire_after(self, t: int) -> tuple[int, int]:
        """打出 t 后的 (有效进张种类, 总剩余张数)。"""
        h = list(self.after_discard(t))
        base = shanten(h, self.n_melds)
        if base <= 0:
            calls = list(winning_tiles(h, list(self.melds[self.me])))
            return len(calls), sum(self.live(x) for x in calls)
        kinds = total = 0
        for x in range(NUM_TILE_TYPES):
            if h[x] >= 4:
                continue
            h[x] += 1
            better = shanten(h, self.n_melds) < base
            h[x] -= 1
            if better:
                lv = self.live(x)
                if lv > 0:
                    kinds += 1
                    total += lv
        return kinds, total

    def hand_ji_count(self) -> int:
        """自己手中（暗手）持有的鸡牌张数。"""
        return sum(self.hand[t] for t in self.ji_tiles)

    def ji_name(self, t: int) -> str | None:
        return self.ji_discard_names.get(int(t))

    def ji_mult(self, t: int) -> float:
        """这张鸡牌的金鸡倍率（非鸡牌 0；普通鸡 1；金鸡 jin_mult）。"""
        if t not in self.ji_tiles:
            return 0.0
        return self.score["jin_mult"] if t in self.gold_tiles else 1.0

    def danger(self, t: int) -> str:
        """生熟张 / 点炮敞口：只用已现张数，不臆测别家听牌。"""
        vis = self.visible(t)
        if vis >= 3:
            return f"已现 {vis}/4 张，几乎安全"
        if vis >= 1:
            return f"已现 {vis}/4 张，点炮风险中等"
        return "生张（0/4 张已现），点炮敞口最大"

    def calls_cn(self, t: int, limit: int = 4) -> str:
        """打出 t 后的叫牌描述（含各牌剩余张数）。"""
        calls = self.ting_after(t)
        if not calls:
            return ""
        parts = [f"{tile_cn(x)}(剩{self.live(x)}张)" for x in calls[:limit]]
        if len(calls) > limit:
            parts.append("…")
        return "、".join(parts)
