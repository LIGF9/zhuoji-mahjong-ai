"""
向听数（shanten）与进张（ukeire）计算。

这是启发式教师 Bot 的核心。公式：

    shanten = 8 - 2 * 面子数 - 搭子数 - 将牌标志

其中「面子数 = 副露面子 + 暗手拆出的面子」，且 面子数 + 搭子数 + 将牌标志 ≤ 5。

实现方式是经典的「从最小非零牌开始枚举块」DFS，配合按 27 维计数向量的
全局 memo，让同一副手牌的重复评估退化成字典查询。
"""
from __future__ import annotations

from functools import lru_cache

from .fan import MELD_SET, _POW5
from .tiles import NUM_TILE_TYPES


@lru_cache(maxsize=1 << 21)
def _opts(counts: tuple) -> frozenset:
    """枚举从 ``counts`` 中能抽出的所有 ``(面子, 搭子, 将牌)`` 组合。"""
    # 防御：调用方偶尔会传入被扣成负数的计数（例如误把非出牌动作当成减牌）。
    # 负数会让"弃孤张"分支永不收敛，直接把递归打爆，所以这里夹到 0。
    if min(counts) < 0:
        counts = tuple(c if c > 0 else 0 for c in counts)
    start = -1
    for i, c in enumerate(counts):
        if c:
            start = i
            break
    if start < 0:
        return frozenset({(0, 0, 0)})

    res = set()
    lst = list(counts)
    rank = start % 9

    # (1) 当作孤张弃掉
    lst[start] -= 1
    res.update(_opts(tuple(lst)))
    lst[start] += 1

    # (2) 刻子
    if lst[start] >= 3:
        lst[start] -= 3
        for m, p, k in _opts(tuple(lst)):
            res.add((m + 1, p, k))
        lst[start] += 3

    # (3) 顺子
    if rank <= 6 and lst[start + 1] and lst[start + 2]:
        lst[start] -= 1
        lst[start + 1] -= 1
        lst[start + 2] -= 1
        for m, p, k in _opts(tuple(lst)):
            res.add((m + 1, p, k))
        lst[start] += 1
        lst[start + 1] += 1
        lst[start + 2] += 1

    # (4) 对子：既可能当将牌，也可能当搭子
    if lst[start] >= 2:
        lst[start] -= 2
        sub = _opts(tuple(lst))
        for m, p, k in sub:
            if k == 0:
                res.add((m, p, 1))      # 作将牌
            res.add((m, p + 1, k))      # 作搭子
        lst[start] += 2

    # (5) 两面 / 嵌张搭子
    for d in (1, 2):
        j = start + d
        if rank + d <= 8 and lst[j]:
            lst[start] -= 1
            lst[j] -= 1
            for m, p, k in _opts(tuple(lst)):
                res.add((m, p + 1, k))
            lst[start] += 1
            lst[j] += 1

    return frozenset(res)


def shanten(hand_counts, n_exposed_melds: int = 0) -> int:
    """返回向听数；``-1`` 表示已胡，``0`` 表示听牌。"""
    c = tuple(hand_counts)
    best = -1
    for m, p, k in _opts(c):
        total_blocks = n_exposed_melds + m + p + k
        if total_blocks > 5:
            # 超出 5 个块，需要丢弃质量最差的块来裁剪
            over = total_blocks - 5
            # k 是必需的将牌，优先裁搭子
            cut = min(over, p + m)
            mm, pp, kk = m, p, k
            d = cut
            use_p = min(d, pp)
            pp -= use_p
            d -= use_p
            mm -= d
            if mm < 0:
                continue
        else:
            mm, pp, kk = m, p, k
        val = 2 * (n_exposed_melds + mm) + pp + kk
        if val > best:
            best = val
    return 8 - best


def is_tenpai_like(hand_counts, n_exposed_melds: int = 0) -> bool:
    return shanten(hand_counts, n_exposed_melds) <= 0


@lru_cache(maxsize=1 << 20)
def ukeire_key(counts: tuple) -> tuple:
    """返回 (向听数, 有效进张种类数, 有效进张总张数)——假定手牌为 13 张。"""
    base = shanten(counts, 0)
    kinds = 0
    total = 0
    lst = list(counts)
    for t in range(NUM_TILE_TYPES):
        if lst[t] >= 4:
            continue
        lst[t] += 1
        if shanten(tuple(lst), 0) < base:
            kinds += 1
            total += 4 - counts[t]
        lst[t] -= 1
    return base, kinds, total


def useful_tile_score(counts, n_exposed_melds: int = 0) -> int:
    """一个便宜的「牌型潜力」打分：搭子与对子加权，孤张惩罚。

    用于向听数相同时的取舍，比完整 ukeire 快得多。
    """
    score = 0
    for s in range(3):
        v = counts[s * 9:(s + 1) * 9]
        for i in range(9):
            c = v[i]
            if c == 0:
                continue
            score += min(c, 2) * 2                    # 对子价值
            if c >= 3:
                score += 4
            if i > 0 and v[i - 1]:
                score += 2
            if i > 1 and v[i - 2]:
                score += 1
    return score
