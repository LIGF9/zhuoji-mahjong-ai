"""
贵州捉鸡麻将 —— 胡牌判定与番型计算。

性能设计
--------
训练时要判几百万次胡牌，纯递归太慢。这里用**预计算表 + 花色分解**：

* 单花色只有 9 个点数，把「能被完全拆成面子」的所有计数向量（每项 ≤4）
  预先枚举出来，用 5 进制整数编码放进 ``MELD_SET``（含空集）。
* 于是「14 张能否成胡」= 枚举将牌（≤13 种）→ 去掉 2 张 → 三个花色的
  9 维向量是否都在 ``MELD_SET`` 里。全是 set 查询，避免了递归。

由于面子不可能跨花色，这个分解是完备且无重复的。

牌型表（通用规则，可在 ``RulesConfig`` 覆盖）
-------------------------------------------
    平胡     1
    大对子   5     4 刻子/杠 + 1 对
    七对     7
    龙七对  10     七对且含 4 张相同
    清一色  10
    清大对  15
    清七对  17
    青龙背  20
"""
from __future__ import annotations

from .tiles import NUM_TILE_TYPES, suit_of

# ---------------------------------------------------------------------------
# 番型常量
# ---------------------------------------------------------------------------
PING_HU = "ping_hu"
DA_DUI_ZI = "da_dui_zi"
QI_DUI = "qi_dui"
LONG_QI_DUI = "long_qi_dui"
QING_YI_SE = "qing_yi_se"
QING_DA_DUI = "qing_da_dui"
QING_QI_DUI = "qing_qi_dui"
QING_LONG_BEI = "qing_long_bei"

FAN_TABLE: dict[str, int] = {
    PING_HU: 1,
    DA_DUI_ZI: 5,
    QI_DUI: 7,
    LONG_QI_DUI: 10,
    QING_YI_SE: 10,
    QING_DA_DUI: 15,
    QING_QI_DUI: 17,
    QING_LONG_BEI: 20,
}

FAN_CN: dict[str, str] = {
    PING_HU: "平胡",
    DA_DUI_ZI: "大对子",
    QI_DUI: "七对",
    LONG_QI_DUI: "龙七对",
    QING_YI_SE: "清一色",
    QING_DA_DUI: "清大对",
    QING_QI_DUI: "清七对",
    QING_LONG_BEI: "青龙背",
}

#: 副露类型
MELD_CHOW = 0             # 顺子（捉鸡不能吃，保留枚举以兼容）
MELD_PONG = 1             # 碰
MELD_KONG_EXPOSED = 2     # 点豆（明杠）
MELD_KONG_ADDED = 3       # 爬坡豆（补杠）
MELD_KONG_CONCEALED = 4   # 闷豆（暗杠）

MELD_CN = {
    MELD_CHOW: "吃",
    MELD_PONG: "碰",
    MELD_KONG_EXPOSED: "点豆",
    MELD_KONG_ADDED: "爬坡豆",
    MELD_KONG_CONCEALED: "闷豆",
}

KONG_TYPES = (MELD_KONG_EXPOSED, MELD_KONG_ADDED, MELD_KONG_CONCEALED)


def meld_is_kong(mt: int) -> bool:
    return mt in KONG_TYPES


# ---------------------------------------------------------------------------
# 预计算：单花色「可完全拆成面子」的向量集合
# ---------------------------------------------------------------------------
_POW5 = [5 ** i for i in range(9)]


def _encode9(vec) -> int:
    e = 0
    for i, v in enumerate(vec):
        e += v * _POW5[i]
    return e


def _build_meld_set() -> frozenset[int]:
    base_melds = []
    for r in range(9):                      # 刻子
        v = [0] * 9
        v[r] = 3
        base_melds.append(tuple(v))
    for r in range(7):                      # 顺子
        v = [0] * 9
        v[r] = v[r + 1] = v[r + 2] = 1
        base_melds.append(tuple(v))

    out = set()

    def rec(start: int, chosen: list) -> None:
        if len(chosen) <= 4:
            acc = [0] * 9
            for m in chosen:
                for i in range(9):
                    acc[i] += m[i]
            if max(acc) <= 4:
                out.add(_encode9(acc))
        if len(chosen) == 4:
            return
        for i in range(start, len(base_melds)):
            rec(i, chosen + [base_melds[i]])

    rec(0, [])
    return frozenset(out)


MELD_SET: frozenset[int] = _build_meld_set()
MELD_SET_LIST: tuple = tuple(sorted(MELD_SET))

# 胡牌/听牌判定缓存（见 is_winning_hand / winning_tiles）
_WIN_CACHE: dict = {}
_TENPAI_CACHE: dict = {}
_CACHE_CAP = 300_000   # 每 worker 约 60-80MB，防止多进程并发时内存爆掉


def _melds_ok(counts27) -> bool:
    """三花色是否都能被完全拆成面子。"""
    for s in range(3):
        v = counts27[s * 9:(s + 1) * 9]
        e = 0
        ok = True
        for i in range(9):
            c = v[i]
            if c > 4:
                return False
            e += c * _POW5[i]
        if e not in MELD_SET:
            return False
    return True


# ---------------------------------------------------------------------------
# 胡牌判定
# ---------------------------------------------------------------------------
def is_seven_pairs(counts) -> bool:
    """14 张是否七对结构（可含 4 张相同，即龙七对）。"""
    if sum(counts) != 14:
        return False
    pairs = 0
    for c in counts:
        if c == 0:
            continue
        if c == 2:
            pairs += 1
        elif c == 4:
            pairs += 2
        else:
            return False
    return pairs == 7


def is_winning_hand(hand_counts, melds) -> bool:
    """暗手 ``hand_counts`` 配合副露 ``melds`` 是否成胡（带缓存）。

    自对弈/评测中同一手牌状态会被反复判定（听牌检查每次要试 34 张），
    按状态缓存是纯加速，无行为差异。
    """
    key = (tuple(hand_counts), tuple(tuple(m) for m in melds))
    cached = _WIN_CACHE.get(key)
    if cached is None:
        cached = _is_winning_hand_impl(hand_counts, melds)
        if len(_WIN_CACHE) < _CACHE_CAP:
            _WIN_CACHE[key] = cached
    return cached


def _is_winning_hand_impl(hand_counts, melds) -> bool:
    n_melds = len(melds)
    need = 4 - n_melds
    if need < 0:
        return False
    total = sum(hand_counts)
    if total != need * 3 + 2:
        return False

    if n_melds == 0 and is_seven_pairs(hand_counts):
        return True

    c = list(hand_counts)
    for t in range(NUM_TILE_TYPES):
        if c[t] >= 2:
            c[t] -= 2
            if _melds_ok(c):
                c[t] += 2
                return True
            c[t] += 2
    return False


def winning_tiles(hand_counts, melds) -> list[int]:
    """13 张（或等价）暗手的听牌张列表（带缓存）。"""
    key = (tuple(hand_counts), tuple(tuple(m) for m in melds))
    cached = _TENPAI_CACHE.get(key)
    if cached is None:
        cached = _winning_tiles_impl(hand_counts, melds)
        if len(_TENPAI_CACHE) < _CACHE_CAP:
            _TENPAI_CACHE[key] = cached
    return cached


def _winning_tiles_impl(hand_counts, melds) -> list[int]:
    n_melds = len(melds)
    if n_melds > 4:
        return []
    expect = (4 - n_melds) * 3 + 1
    if sum(hand_counts) != expect:
        return []
    c = list(hand_counts)
    res = []
    for t in range(NUM_TILE_TYPES):
        if c[t] >= 4:
            continue
        c[t] += 1
        if is_winning_hand(c, melds):
            res.append(t)
        c[t] -= 1
    return res


def is_tenpai(hand_counts, melds) -> bool:
    """是否听牌（叫牌）。"""
    return bool(winning_tiles(hand_counts, melds))


# ---------------------------------------------------------------------------
# 番型判定
# ---------------------------------------------------------------------------
def _is_qing_yi_se(hand_counts, melds) -> bool:
    suits = set()
    for t, c in enumerate(hand_counts):
        if c:
            suits.add(suit_of(t))
            if len(suits) > 1:
                return False
    for m in melds:
        suits.add(suit_of(m[1]))
        if len(suits) > 1:
            return False
    return len(suits) == 1


def _all_triplets(hand_counts, melds) -> bool:
    for m in melds:
        if m[0] == MELD_CHOW:
            return False
    c = list(hand_counts)
    n_melds = len(melds)
    for t in range(NUM_TILE_TYPES):
        if c[t] >= 2:
            c[t] -= 2
            if _melds_ok(c) and all(x % 3 == 0 for x in c) and sum(c) == (4 - n_melds) * 3:
                c[t] += 2
                return True
            c[t] += 2
    return False


def classify_fan(hand_counts, melds, win_tile: int | None = None) -> tuple[str, int]:
    """返回 ``(牌型名, 番值)``；不成胡时返回 ``("", 0)``。

    ``win_tile`` 给定时按「这张胡牌张」判龙七对（必须是它补上第四张同牌）；
    为 None 时保留宽松口径（手里有四张同牌即算龙七对）。
    """
    c = tuple(hand_counts)
    n_melds = len(melds)

    if n_melds == 0 and is_seven_pairs(c):
        if win_tile is None:
            dragon = any(x == 4 for x in c)
        else:
            dragon = c[int(win_tile)] == 4
        base = LONG_QI_DUI if dragon else QI_DUI
        if _is_qing_yi_se(c, melds):
            name = QING_LONG_BEI if base == LONG_QI_DUI else QING_QI_DUI
            return name, FAN_TABLE[name]
        return base, FAN_TABLE[base]

    if not is_winning_hand(c, melds):
        return "", 0

    qing = _is_qing_yi_se(c, melds)
    if _all_triplets(c, melds):
        name = QING_DA_DUI if qing else DA_DUI_ZI
        return name, FAN_TABLE[name]
    if qing:
        return QING_YI_SE, FAN_TABLE[QING_YI_SE]
    return PING_HU, FAN_TABLE[PING_HU]


def is_high_value_hand(hand_counts, melds) -> bool:
    """是否"平胡以上"牌型（点胡通行证判定用）。"""
    _, fan = classify_fan(hand_counts, melds)
    return fan > FAN_TABLE[PING_HU]
