"""状态编码 v2 —— P1 表征升级（31 通道 / 98 全局维）。

为什么要有 v2
-------------
v1（``zhuoji/encoder.py``）有三个已被文档记录的信息缺口，最严重的是**牌河无时序**：
``discards[p]`` 本来是有序列表，编码时被压成「累加 0.25/张」，于是
「谁在第几巡打了什么」被彻底抹掉 —— 而读牌的核心恰恰是「上家最近打了什么」、
「这张牌是不是他刚打过」。时序一丢，危险度判断基本无从谈起。

v2 在 **完全保留 v1 全部 15 个通道 / 70 个标量** 的基础上追加：

``tiles`` 15 → 31
    15–18  四家牌河**最近第 1 张**（one-hot，相对座位）
    19–22  四家牌河**最近第 2 张**
    23–26  四家牌河**最近第 3 张**
    27     现物安全度 = 已打出过该牌的玩家数 / 3
    28     筋牌安全度（简化）= 同花色 ±3 内可见张数 / 4
    29     生牌危险度 = 1 − 可见张数/4
    30     对手潜在需求 = 副露花色与牌河密集花色的加权指示

``glob`` 70 → 98
    70–73  各家估计向听数下界 / 6
    74–77  各家已亮明鸡数 / 4
    78–81  各家当下听牌置信（启发式）
    82     巡目 / 12
    83–85  阶段 one-hot（序盘 ≤4 巡 / 中盘 5–9 / 终盘 ≥10）
    86–93  八个番型可行性（连续值，不是布尔）
    94     自己可达最大番型值 / 30
    95     黄庄紧迫度
    96     包鸡风险敞口
    97     自己手牌鸡价值 / 4

红线（不可违反）
----------------
**所有新特征只能来自公开信息**：自己的手牌、四家的副露、四家的牌河、牌墙剩余。
**绝不读取任何对手的手牌**。对手状态一律用「从公开信息推断的启发式」表示，
公式写死在下面并在文档里说明，免得把作弊当成特征工程。
``tests/test_encoder_v2.py`` 有一条「篡改对手手牌不得改变编码」的断言守着这条线。
"""
from __future__ import annotations

import numpy as np

from .fan import FAN_TABLE, meld_is_kong, winning_tiles
from .tiles import DECK_SIZE, NUM_TILE_TYPES, YAOJI, rank_of, suit_of, tile_from

NUM_CHANNELS_V2 = 31
GLOBAL_DIM_V2 = 98

#: v1 的维度（v2 是 v1 的**超集**：前 15 通道与前 70 维逐位相同，这条性质有测试断言）
_V1_CHANNELS = 15
_V1_GLOBAL_DIM = 70

# ---------------------------------------------------------------------------
# P1 三包与「包掩码」
# ---------------------------------------------------------------------------
#: P1 的三个互不依赖的特征包（见 ``docs/P1-表征升级设计.md`` §2）。
ALL_PACKS: tuple[str, ...] = ("a", "b", "c")

#: 每个包占用的 **tiles 通道**区间（半开）与 **glob 维度**区间（半开）。
#: 注意 P1b 同时占通道与全局标量，所以两处都要写。
PACK_TILE_SLICES: dict[str, tuple[int, int]] = {"a": (15, 27), "b": (27, 31)}
PACK_GLOB_SLICES: dict[str, tuple[int, int]] = {"b": (70, 82), "c": (82, 98)}

PACK_DESC: dict[str, str] = {
    "a": "牌河时序（ch 15–26）",
    "b": "危险度与对手状态（ch 27–30 + glob 70–81）",
    "c": "番型潜力与巡目情境（glob 82–97）",
}


def parse_packs(spec) -> tuple[str, ...]:
    """把 ``--encoder-packs`` 的写法规范成元组。

    * ``None``        → 全开 ``("a","b","c")``（历史默认，向后兼容）
    * ``"abc"``/``"a"``/``"ac"``/``["a","b"]`` → 只留其中的包（按 a,b,c 固定顺序）
    * ``""``/``"none"``/``"-"`` → 一个都不开（纯 v1 特征 + 零填充，作对照臂）
    """
    if spec is None:
        return ALL_PACKS
    if isinstance(spec, str):
        chars = [c for c in spec.lower() if c in "abc"]
    else:
        chars = [str(c).lower() for c in spec]
        chars = [c for c in chars if c in "abc"]
    keep = set(chars)
    return tuple(p for p in ALL_PACKS if p in keep)


def apply_pack_mask(tiles: np.ndarray, g: np.ndarray,
                    packs: tuple[str, ...]) -> tuple[np.ndarray, np.ndarray]:
    """把**未启用**包的通道/维度整体清零。

    这是单变量消融的关键：**架构与参数量完全不变**，只是把某个包的信息
    置零。因此"开包 / 关包"两个臂之间唯一变化的就是信息本身，
    而不是容量或初始化——否则根本分不清涨分是来自特征还是来自参数。
    """
    for name, (lo, hi) in PACK_TILE_SLICES.items():
        if name not in packs:
            tiles[lo:hi] = 0.0
    for name, (lo, hi) in PACK_GLOB_SLICES.items():
        if name not in packs:
            g[lo:hi] = 0.0
    return tiles, g

#: 「开局每家手上已发的牌」：13×4 + 庄家多摸 1 = 53，取 52 作保守下界
_OPENING_TILES = 52

_FAN_ORDER = ("dan_diao", "ping_hu", "da_dui_zi", "qi_dui", "long_qi_dui",
              "qing_yi_se", "qing_da_dui", "qing_qi_dui", "qing_long_bei")
#: 番值表直接取自 ``fan.FAN_TABLE``，不另造一套（单吊按平胡口径取 1）
_FAN_VALUE = dict(FAN_TABLE)
_FAN_VALUE["dan_diao"] = 1


def _rel(p: int, me: int) -> int:
    return (p - me) % 4


def _visible_counts(game, tiles) -> np.ndarray:
    """每张牌的可见张数（手牌 + 四家牌河 + 四家副露），与 v1 同口径。"""
    visible = np.zeros(NUM_TILE_TYPES, dtype=np.float32)
    visible += tiles[4] * 4.0
    for r in range(4):
        visible += tiles[5 + r] * 4.0
        visible += tiles[9 + r] * 4.0
    return visible


def encode_tiles_v2(game, me: int, tiles: np.ndarray) -> np.ndarray:
    """在 v1 的 ``tiles``（已填好 0–14 通道）上补齐 15–30 通道。"""
    # -- 15–26：四家牌河最近 1/2/3 张（时间倒序，第 1 张 = 最近） -------------
    for p in range(4):
        r = _rel(p, me)
        river = game.discards[p]
        for k in range(3):
            if len(river) >= k + 1:
                tiles[15 + k * 4 + r, river[-1 - k]] = 1.0

    visible = _visible_counts(game, tiles)
    remain = np.clip(4.0 - visible, 0.0, 4.0)

    # -- 27：现物安全度 = 已打出过该牌的玩家数 / 3 ----------------------------
    genbutsu = np.zeros(NUM_TILE_TYPES, dtype=np.float32)
    for r in range(4):
        genbutsu += (tiles[5 + r] > 0).astype(np.float32)
    tiles[27] = genbutsu / 3.0

    # -- 28：筋牌安全度（简化）= 同花色 ±3 点数范围内的可见张数 / 4 -----------
    # 说明：真正的筋牌规则要考虑「鸣牌后放铳」等细节，这里只做一阶近似——
    # 同花色相邻三个点数（±3）越"见过"，该牌越接近筋牌。
    # 向量化写法（逐牌 Python 循环在这里是热点）：把 27 张牌 reshape 成 (3,9)，
    # 同花色内整体平移 3 位即可，等价于逐牌查 ±3 两个邻居。
    vis3 = visible.reshape(3, 9)
    acc = np.zeros_like(vis3)
    acc[:, :-3] += vis3[:, 3:]      # 点数 +3 的邻居
    acc[:, 3:] += vis3[:, :-3]      # 点数 -3 的邻居
    tiles[28] = np.clip(acc, 0.0, 4.0).ravel() / 4.0

    # -- 29：生牌危险度 = 1 − 可见张数/4（完全不可见 = 1） -------------------
    tiles[29] = 1.0 - visible / 4.0

    # -- 30：对手潜在需求 = 副露花色 + 牌河密集花色的加权指示 ----------------
    # 权重前置声明：副露花色 1.0（明牌最可信）、牌河密集花色 0.5。
    # 先各自归一化再加权，避免很长的牌河把只有 0–4 组的副露淹没。
    meld_suit = np.zeros(3, dtype=np.float32)
    river_suit = np.zeros(3, dtype=np.float32)
    for p in range(4):
        if p == me:
            continue
        for _mt, t, _from in game.melds[p]:
            meld_suit[suit_of(t)] += 1.0
        for t in game.discards[p]:
            river_suit[suit_of(t)] += 1.0
    w = meld_suit / max(1.0, float(meld_suit.sum()))
    w = w + 0.5 * (river_suit / max(1.0, float(river_suit.sum())))
    w = w / max(1e-6, float(w.max()))
    for t in range(NUM_TILE_TYPES):
        tiles[30, t] = float(w[suit_of(t)])
    return tiles


def encode_glob_v2(game, me: int, g: np.ndarray) -> np.ndarray:
    """在 v1 的 ``g``（已填好 0–69）上补齐 70–97 维。"""
    hand = game.hands[me]
    n_melds_me = len(game.melds[me])
    wall_left = float(game.wall_left)
    turn = float(np.clip((DECK_SIZE - wall_left - _OPENING_TILES) / 4.0, 0.0, 12.0))

    # -- 70–73：各家估计向听数下界 / 6 ---------------------------------------
    # 公式（前置声明，仅用公开信息）：
    #   lb = clamp(4 − 副露组数 − 牌河中成对/连续组估计, 0, 6)
    # 「牌河中成对/连续组估计」= 该家牌河里同种牌出现 ≥2 次的种类数
    #   ＋ 同花色相邻点数同时出现的对数，上限 3。
    for p in range(4):
        r = _rel(p, me)
        river = list(game.discards[p])
        counts: dict[int, int] = {}
        for t in river:
            counts[t] = counts.get(t, 0) + 1
        pair_kinds = sum(1 for v in counts.values() if v >= 2)
        adj = 0
        for a in counts:
            for dr in (1, -1):
                rr = rank_of(a) + dr
                if 0 <= rr < 9 and tile_from(suit_of(a), rr) in counts:
                    adj += 1
        adj //= 2
        est = min(3, pair_kinds + adj)
        lb = float(np.clip(4 - len(game.melds[p]) - est, 0.0, 6.0))
        g[70 + r] = lb / 6.0

    # -- 74–77：各家已亮明鸡数 / 4 -------------------------------------------
    # 口径：该家**打出**过的鸡（ji_events）+ 该家**碰/杠走**的鸡（claimed_ji）。
    per_player = np.zeros(4, dtype=np.float32)
    for p, _val, _tag in getattr(game, "ji_events", []):
        per_player[int(p) % 4] += 1.0
    # 注意：claimed_ji 元素为 4 元组(旧)或 5 元组(2026-10-12 起附加牌面)，
    # 故此处处用下标访问第 2 位（碰/杠者），不做元组解包，避免契约扩展后再崩。
    for ev in getattr(game, "claimed_ji", []):
        per_player[int(ev[1]) % 4] += 1.0
    for p in range(4):
        g[74 + _rel(p, me)] = min(per_player[p], 4.0) / 4.0

    # -- 78–81：各家当下听牌置信（启发式，非真值） ---------------------------
    # conf = clamp(0.2×副露组数 + 0.05×巡目, 0, 1)
    # 明确声明：这是从公开信息推出的有噪声估计，不引入任何未来信息。
    for p in range(4):
        conf = 0.2 * len(game.melds[p]) + 0.05 * turn
        g[78 + _rel(p, me)] = float(np.clip(conf, 0.0, 1.0))

    # -- 82：巡目 / 12 --------------------------------------------------------
    g[82] = turn / 12.0
    # -- 83–85：阶段 one-hot（序盘 ≤4 / 中盘 5–9 / 终盘 ≥10） ----------------
    if turn <= 4:
        g[83] = 1.0
    elif turn <= 9:
        g[84] = 1.0
    else:
        g[85] = 1.0

    # -- 86–93：八个番型可行性（连续值） --------------------------------------
    all_counts = np.array(hand, dtype=np.float32).copy()
    for _mt, t, _from in game.melds[me]:
        all_counts[t] += 3.0
    total = float(all_counts.sum()) or 1.0

    n_kongs = sum(1 for mt, _t, _f in game.melds[me] if meld_is_kong(mt))
    n_triplets_hand = int((np.array(hand) >= 3).sum())
    suit_tot = [float(all_counts[s * 9:(s + 1) * 9].sum()) for s in range(3)]
    g[86] = min(1.0, (n_kongs + n_triplets_hand) / 2.0)                  # 豆
    g[87] = max(suit_tot) / total                                        # 清一色
    g[88] = (int((np.array(hand) == 2).sum()) / 7.0
             if n_melds_me == 0 else 0.0)                                # 七对
    g[89] = min(1.0, (n_triplets_hand + n_melds_me) / 4.0)               # 大对子
    g[90] = 1.0 / (1.0 + n_triplets_hand + n_kongs)                      # 平胡
    # 将对：手牌+副露里 2/5/8 点数占的比重（独山将对口径的一阶近似）
    jiang = 0.0
    for t in range(NUM_TILE_TYPES):
        if rank_of(t) in (1, 4, 7):        # rank 从 0 起：点数 2/5/8 → 1/4/7
            jiang += float(all_counts[t])
    g[91] = jiang / total
    g[92] = 1.0 if n_melds_me == 0 else 0.0                              # 门前清
    best_run = 0.0
    for s in range(3):
        seg = all_counts[s * 9:(s + 1) * 9]
        best_run = max(best_run, float((seg > 0).sum()) / 9.0)
    g[93] = best_run                                                     # 青龙（覆盖度近似）

    # -- 94：自己可达最大番型值 / 30 -----------------------------------------
    g[94] = _max_reachable_fan(game, me) / 30.0

    # -- 95：黄庄紧迫度 = (1 − wall_left/108) × 自己未听牌 -------------------
    tenpai = bool(g[27] > 0.5)
    g[95] = (1.0 - wall_left / float(DECK_SIZE)) * (0.0 if tenpai else 1.0)
    # -- 96：包鸡风险敞口 = 自己未听牌 × 他家已亮明鸡数 > 0 -------------------
    others_ji = float(per_player.sum() - per_player[me])
    g[96] = (0.0 if tenpai else 1.0) * (1.0 if others_ji > 0 else 0.0)
    # -- 97：自己手牌鸡价值 = (幺鸡数 + 已成刻的鸡牌) / 4 --------------------
    ji_tiles = getattr(game, "ji_tiles", {YAOJI})
    ji_val = 0.0
    for t in ji_tiles:
        ji_val += float(hand[t])
        if hand[t] >= 3 or game.melded_count(me, t) >= 3:
            ji_val += float(hand[t])
    g[97] = min(1.0, ji_val / 8.0)
    return g


def _max_reachable_fan(game, me: int) -> float:
    """「自己若立刻胡牌能达到的最大番值」——直接复用引擎的牌型判定。

    优先用 ``DushanGame._win_type_set``（独山叠加制口径），拿不到时退回
    ``fan.classify_fan``（贵阳捉鸡口径）。两者都是**已有实现**，不另造一套
    番值表，避免特征里混进与规则不一致的假设。
    """
    hand = [int(x) for x in game.hands[me]]
    melds = game.melds[me]
    if sum(hand) != (4 - len(melds)) * 3 + 1:
        return 0.0
    waits = winning_tiles(hand, melds)
    if not waits:
        return 0.0
    win_type_set = getattr(game, "_win_type_set", None)
    best = 0.0
    for t in waits:
        h = list(hand)
        h[int(t)] += 1
        if win_type_set is not None:
            types = win_type_set(h, melds, int(t))
            val = sum(_FAN_VALUE.get(tp, 0) for tp in types if tp != "ping_hu")
            if not val:
                val = _FAN_VALUE["ping_hu"]
        else:
            from .fan import classify_fan
            _name, val = classify_fan(h, melds, int(t))
        best = max(best, float(val))
    return best


def encode_v2(game, player: int, base_encode, packs=None):
    """v2 编码。

    ``base_encode`` 为 v1 的 ``encode``（由 :mod:`zhuoji.encoder` 注入，
    避免模块间循环导入）；v1 的 0–14 通道与 0–69 标量**逐位原样保留**，
    只在后面追加新特征，于是「v1 是 v2 的子集」这个性质可以被测试断言。

    ``packs`` 为要**启用**的包（见 :func:`parse_packs`）；``None`` = 全开。
    未启用的包在最后统一清零（:func:`apply_pack_mask`），维度与参数量不变。
    """
    obs = base_encode(game, player)
    tiles = np.zeros((NUM_CHANNELS_V2, NUM_TILE_TYPES), dtype=np.float32)
    tiles[:15] = obs.tiles
    obs.tiles = encode_tiles_v2(game, player, tiles)
    g = np.zeros(GLOBAL_DIM_V2, dtype=np.float32)
    g[:_V1_GLOBAL_DIM] = obs.glob
    obs.glob = encode_glob_v2(game, player, g)
    if packs is not None:
        obs.tiles, obs.glob = apply_pack_mask(
            obs.tiles, obs.glob, parse_packs(packs))
    return obs
