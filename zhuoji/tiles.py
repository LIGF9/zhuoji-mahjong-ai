"""
贵州捉鸡麻将（贵阳捉鸡）—— 牌张基础定义。

牌面共 108 张：万(W)/条(T)/筒(B) 各 1-9，每种 4 张。无字牌、无花牌。

内部统一用 0..26 的整数表示一种"牌"（tile type），数组下标即编码：
    0.. 8  ->  W1..W9   万
    9..17  ->  T1..T9   条
   18..26  ->  B1..B9   筒
"""
from __future__ import annotations

NUM_TILE_TYPES = 27
TILES_PER_TYPE = 4
DECK_SIZE = NUM_TILE_TYPES * TILES_PER_TYPE  # 108

SUITS = ("W", "T", "B")          # 万、条、筒
SUIT_CN = {"W": "万", "T": "条", "B": "筒"}

W, T, B = 0, 1, 2                # 花色下标

#: 幺鸡 —— 一条，捉鸡规则里的"常鸡"，固定为鸡牌
T1 = YAOJI = 9


def suit_of(t: int) -> int:
    """返回牌的花色下标 0/1/2。"""
    return t // 9


def rank_of(t: int) -> int:
    """返回牌的点数 1..9。"""
    return t % 9 + 1


def tile_from(suit: int, rank: int) -> int:
    """由花色下标与点数构造牌编码。"""
    return suit * 9 + (rank - 1)


def tile_name(t: int) -> str:
    """人类可读牌名，如 ``W5`` / ``T1``。"""
    return f"{SUITS[suit_of(t)]}{rank_of(t)}"


def tile_cn(t: int) -> str:
    """中文牌名，如 ``五万`` / ``一条``。"""
    cn = "一二三四五六七八九"[rank_of(t) - 1]
    return f"{cn}{SUIT_CN[SUITS[suit_of(t)]]}"


def parse_tile(s: str) -> int:
    """解析 ``W5`` / ``T1`` 形式的牌名。"""
    s = s.strip().upper()
    suit = SUITS.index(s[0])
    return tile_from(suit, int(s[1:]))


def full_deck() -> list[int]:
    """返回一副完整的 108 张牌（未洗）。"""
    deck: list[int] = []
    for t in range(NUM_TILE_TYPES):
        deck.extend([t] * TILES_PER_TYPE)
    return deck


def tiles_to_counts(tiles) -> list[int]:
    """把牌列表压成 27 维计数向量。"""
    counts = [0] * NUM_TILE_TYPES
    for t in tiles:
        counts[t] += 1
    return counts


def counts_to_tiles(counts) -> list[int]:
    """把 27 维计数向量展回牌列表。"""
    out: list[int] = []
    for t, c in enumerate(counts):
        out.extend([t] * c)
    return out


def all_tile_names() -> list[str]:
    return [tile_name(t) for t in range(NUM_TILE_TYPES)]
