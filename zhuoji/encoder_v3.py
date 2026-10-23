"""状态编码 v3 —— P3 表征二次升级（63 通道 / 130 全局维）。

定位
----
v2 补齐了「牌河时序」与「危险度/番型潜力」，但仍在三处有明显缺口，v3 全部补上：

1. **横鸡轮状态完全缺失**。v2 只有「各家已亮明鸡数」，没有「谁开了横鸡轮、
   轮是否还开着、我现在打哪张牌会算横鸡」。而独山里横鸡分值最高（默认 2 分）
   且**同一轮内跟打也算**，这是一个纯公开、却直接决定单张弃牌收益的状态量。
2. **牌河时序只到最近 3 张**。终盘判断「这张牌最后一次出现是第几巡」需要更长窗口。
3. **手牌结构没有被显式编码**。向听/搭子/孤张全靠卷积从 5 个手牌通道里反推，
   在 3×9 的小网格上这部分信息密度很高、但反推成本不低；显式给出可提速收敛。

红线（与 v2 相同，绝不放宽）
----------------------------
**所有新特征只来自公开信息**：自己的手牌、四家的副露、四家的牌河、牌墙剩余，
以及引擎已经对外公告过的事件（``ji_events`` / ``claimed_ji`` / ``hengji_events`` /
``baojiao``）。**绝不读取任何对手手牌**。对手状态一律用「从公开信息推断的启发式」
表示，公式写死在下面，免得把作弊当特征工程。

严格超集性质
------------
``tiles[:31] == v2.tiles`` 且 ``glob[:98] == v2.glob``，**逐位相同**，
由 ``tests/test_encoder_v3.py`` 断言。因此：

* 旧（v2）检查点的输入口径不受影响；
* 想做「纯容量 vs 容量+新特征」的单变量消融时，可以把 d/e/f/g 四个包单独关掉，
  此时输入与 v2 完全一致，架构与参数量却不变。

包（消融用）
------------
``a``/``b``/``c`` 沿用 v2 的三包（牌河时序 / 危险度与对手状态 / 番型情境）；
``d`` = tiles 31–42（长时序牌河）
``e`` = tiles 43–52（横鸡轮与鸡牌状态）
``f`` = tiles 53–62（放铳风险与手牌结构）
``g`` = glob 98–129（全局扩展）
"""
from __future__ import annotations

import numpy as np

from .fan import meld_is_kong, winning_tiles
from .tiles import DECK_SIZE, NUM_TILE_TYPES, YAOJI, rank_of, suit_of, tile_from

NUM_CHANNELS_V3 = 63
GLOBAL_DIM_V3 = 130

#: v2 的维度（v3 是 v2 的**超集**，这条性质有测试断言）
_V2_CHANNELS = 31
_V2_GLOBAL_DIM = 98

# ---------------------------------------------------------------------------
# 特征包（单变量消融）
# ---------------------------------------------------------------------------
ALL_PACKS_V3: tuple[str, ...] = ("a", "b", "c", "d", "e", "f", "g")

PACK_TILE_SLICES_V3: dict[str, tuple[int, int]] = {
    "d": (31, 43),
    "e": (43, 53),
    "f": (53, 63),
}
PACK_GLOB_SLICES_V3: dict[str, tuple[int, int]] = {
    "g": (98, 130),
}

PACK_DESC_V3: dict[str, str] = {
    "a": "牌河时序（ch 15–26，v2）",
    "b": "危险度与对手状态（ch 27–30 + glob 70–81，v2）",
    "c": "番型潜力与巡目情境（glob 82–97，v2）",
    "d": "长时序牌河 4/5/6 张（ch 31–42）",
    "e": "横鸡轮与鸡牌状态（ch 43–52）",
    "f": "放铳风险与手牌结构（ch 53–62）",
    "g": "全局扩展（glob 98–129）",
}

#: 「开局每家手上已发的牌」：13×4 + 庄家多摸 1 = 53，取 52 作保守下界（与 v2 同口径）
_OPENING_TILES = 52

Q1_DANGER_CONF = "0.2×副露组数 + 0.05×巡目（与 v2 glob 78–81 同一个公式，保证口径一致）"


def parse_packs_v3(spec) -> tuple[str, ...]:
    """把 ``--encoder-packs`` 规范成七包元组（顺序固定 a,b,c,d,e,f,g）。

    ``None`` → 全开；``"de"`` → 只开 d/e；``""``/``"none"``/``"-"`` → 全关（纯 v1 输入 + 零填充）。
    """
    if spec is None:
        return ALL_PACKS_V3
    if isinstance(spec, str):
        chars = [c for c in spec.lower() if c in "abcdefg"]
    else:
        chars = [str(c).lower() for c in spec]
        chars = [c for c in chars if c in "abcdefg"]
    keep = set(chars)
    return tuple(p for p in ALL_PACKS_V3 if p in keep)


def apply_pack_mask_v3(tiles: np.ndarray, g: np.ndarray,
                       packs: tuple[str, ...]) -> tuple[np.ndarray, np.ndarray]:
    """把**未启用**的 v3 包的通道/维度整体清零（架构与参数量不变）。"""
    for name, (lo, hi) in PACK_TILE_SLICES_V3.items():
        if name not in packs:
            tiles[lo:hi] = 0.0
    for name, (lo, hi) in PACK_GLOB_SLICES_V3.items():
        if name not in packs:
            g[lo:hi] = 0.0
    return tiles, g


def _rel(p: int, me: int) -> int:
    return (p - me) % 4


def _seat_of(r: int, me: int) -> int:
    """相对座位 ``r`` 对应的绝对座位。"""
    return (me + r) % 4


def _turn(game) -> float:
    """巡目估计（与 v2 完全同口径）。"""
    return float(np.clip((DECK_SIZE - game.wall_left - _OPENING_TILES) / 4.0, 0.0, 12.0))


def _visible(game, me) -> np.ndarray:
    """每张牌的可见张数 = 自己手牌 + 四家牌河 + 四家副露（副露按碰 3 / 杠 4 折算）。"""
    vis = np.array(game.hands[me], dtype=np.float32)
    for p in range(4):
        for t in game.discards[p]:
            vis[int(t)] += 1.0
        for mt, t, _f in game.melds[p]:
            vis[int(t)] += 4.0 if meld_is_kong(mt) else 3.0
    return vis


def _conf(game, p: int, turn: float) -> float:
    """对手 ``p`` 的听牌置信（启发式，公开信息）：见 :data:`Q1_DANGER_CONF`。"""
    return float(np.clip(0.2 * len(game.melds[p]) + 0.05 * turn, 0.0, 1.0))


def _hand_shape(hand) -> tuple[float, float, float, float]:
    """自己手牌的 (刻子数, 对子数, 搭子数, 孤张数)。搭子 = 相邻/隔张对（不重复计数）。"""
    h = np.asarray(hand, dtype=np.float32)
    triplets = float((h >= 3).sum())
    pairs = float((h == 2).sum())
    partials = 0.0
    lone = 0.0
    for s in range(3):
        base = s * 9
        for rk in range(9):
            t = base + rk
            if h[t] <= 0:
                continue
            nb1 = [base + rk + d for d in (-1, 1) if 0 <= rk + d < 9]
            nb2 = [base + rk + d for d in (-2, 2) if 0 <= rk + d < 9]
            near = any(h[u] > 0 for u in nb1)
            far = any(h[u] > 0 for u in nb2)
            if near or far:
                # 只从「较小下标」那头计一次，避免一对搭子被数两遍
                if far and not near:
                    partials += 0.5
                elif near and not any(h[u] > 0 for u in nb1 if u < t):
                    partials += 1.0
            elif h[t] == 1:
                lone += 1.0
    return triplets, pairs, partials, lone


def encode_tiles_v3(game, me: int, tiles: np.ndarray) -> np.ndarray:
    """在 v2 的 ``tiles``（已填好 0–30 通道）上补齐 31–62 通道。"""
    hand = game.hands[me]

    # -- 31–42：四家牌河第 4/5/6 近张（v2 只到第 3 张，终盘读牌需要更长窗口） ----
    for p in range(4):
        r = _rel(p, me)
        river = game.discards[p]
        for k in (3, 4, 5):
            if len(river) >= k + 1:
                tiles[31 + (k - 3) * 4 + r, river[-1 - k]] = 1.0

    ji_tiles = getattr(game, "ji_tiles", {YAOJI})

    # -- 43–46：四家本局**开出横鸡轮**的次数 / 2 -------------------------------
    opened = np.zeros(4, dtype=np.float32)
    for p, _t, tag in getattr(game, "hengji_events", []):
        if str(tag) == "open":
            opened[int(p) % 4] += 1.0
    for p in range(4):
        tiles[43 + _rel(p, me)] = min(opened[p] / 2.0, 1.0)

    # -- 47：当前是否还有横鸡轮开放（广播到整张牌面） ---------------------------
    if bool(getattr(game, "hengji_active", False)):
        tiles[47] = 1.0

    # -- 48/49：我**现在**打这张牌会记成什么鸡（引擎同源函数，不改状态） --------
    name_if = getattr(game, "ji_name_if_discarded", None)
    if callable(name_if):
        for t in range(NUM_TILE_TYPES):
            nm = name_if(me, t)
            if nm == "横鸡":
                tiles[48, t] = 1.0
            elif nm == "冲锋鸡":
                tiles[49, t] = 1.0

    # -- 50：本局鸡牌种 ---------------------------------------------------------
    for t in ji_tiles:
        tiles[50, int(t)] = 1.0

    # -- 51：我自己手牌里的鸡牌张数 / 4 ----------------------------------------
    for t in ji_tiles:
        tiles[51, int(t)] = min(float(hand[int(t)]) / 4.0, 1.0)

    # -- 52：已被别家碰/杠走的鸡牌张数 / 4 -------------------------------------
    # 引擎把「(碰/杠者, 牌) → 鸡类型名」存在 _ji_val_by_meld，这里只取键（= 牌）。
    for _key in list(getattr(game, "_ji_val_by_meld", {}).keys()):
        t = int(_key[1])
        tiles[52, t] = min(tiles[52, t] + 0.25, 1.0)

    visible = _visible(game, me)
    remain = np.clip(4.0 - visible, 0.0, 4.0)
    turn = _turn(game)

    # -- 53：放铳风险 = max_r(听牌置信_r) × (1 − 可见张数/4)，r 只取对手 ---------
    conf = max((_conf(game, _seat_of(r, me), turn) for r in (1, 2, 3)), default=0.0)
    tiles[53] = conf * (1.0 - visible / 4.0)

    # -- 54–57：自己手牌结构（显式给出，省掉卷积从 5 个手牌通道反推的成本） -----
    for t in range(NUM_TILE_TYPES):
        c = float(hand[t])
        if c <= 0:
            continue
        s, rk = suit_of(t), rank_of(t)
        nb1 = [tile_from(s, rk + d) for d in (-1, 1) if 0 <= rk + d < 9]
        nb2 = [tile_from(s, rk + d) for d in (-2, 2) if 0 <= rk + d < 9]
        if any(hand[u] > 0 for u in nb2):
            tiles[54, t] = 1.0                       # 隔张搭子（含嵌张）
        if any(hand[u] > 0 for u in nb1):
            tiles[55, t] = 1.0                       # 相邻搭子
        run = 1
        rr = rk - 1
        while rr >= 0 and hand[tile_from(s, rr)] > 0:
            run += 1
            rr -= 1
        rr = rk + 1
        while rr < 9 and hand[tile_from(s, rr)] > 0:
            run += 1
            rr += 1
        tiles[56, t] = min(run / 3.0, 1.0)           # 所在连续段长度 / 3
        if c == 1 and not any(hand[u] > 0 for u in nb1 + nb2):
            tiles[57, t] = 1.0                       # 孤张（无邻无重复）

    # -- 58/59：时效性 —— 最近 3 手内被谁打过（线性衰减 1, 2/3, 1/3） ----------
    # 58 = 任意家（含自己）；59 = **上家**（相对座位 3）→ 独山/贵阳的「跟打见现」安全牌。
    for r in range(4):
        river = game.discards[_seat_of(r, me)]
        for k in range(3):
            if len(river) >= k + 1:
                w = 1.0 - k / 3.0
                t = int(river[-1 - k])
                if w > tiles[58, t]:
                    tiles[58, t] = w
                if r == 3 and w > tiles[59, t]:
                    tiles[59, t] = w

    # -- 60：自己是否已报叫（报叫锁定碰杠，是公开状态） -------------------------
    bj = getattr(game, "baojiao", None)
    if bj is not None and bj[me]:
        tiles[60] = 1.0

    # -- 61：严格现物 —— 对其他三家**都**是现物（比 v2 的 genbutsu/3 更强） ----
    strict = np.ones(NUM_TILE_TYPES, dtype=np.float32)
    for r in (1, 2, 3):
        strict *= (tiles[5 + r] > 0).astype(np.float32)
    tiles[61] = strict

    # -- 62：「放铳风险 × 还摸得到」—— 危险与可摸性的联合（能摸到的危险才是真危险）
    tiles[62] = tiles[53] * (remain / 4.0)
    return tiles


def _max_fan(game, me: int) -> float:
    """自己立刻胡牌可达的最大番值（复用 v2 的实现，避免第二套番值表）。"""
    from .encoder_v2 import _max_reachable_fan
    return _max_reachable_fan(game, me)


def encode_glob_v3(game, me: int, g: np.ndarray) -> np.ndarray:
    """在 v2 的 ``g``（已填好 0–97 维）上补齐 98–129 维。"""
    hand = game.hands[me]
    turn = _turn(game)
    wall_left = float(game.wall_left)
    melds_me = game.melds[me]
    n_melds_me = len(melds_me)

    # -- 98–101：四家本局「因鸡被碰/杠」次数 / 4 -------------------------------
    claim = np.zeros(4, dtype=np.float32)
    # 同 encoder_v2：claimed_ji 可能为 4/5 元组，用下标访问第 2 位（碰/杠者）。
    for ev in getattr(game, "claimed_ji", []):
        claim[int(ev[1]) % 4] += 1.0
    for p in range(4):
        g[98 + _rel(p, me)] = min(claim[p] / 4.0, 1.0)

    # -- 102–105：四家本局「打出鸡牌」次数 / 4 -------------------------------
    ev = np.zeros(4, dtype=np.float32)
    for p, _val, _tag in getattr(game, "ji_events", []):
        ev[int(p) % 4] += 1.0
    for p in range(4):
        g[102 + _rel(p, me)] = min(ev[p] / 4.0, 1.0)

    # -- 106–108：横鸡轮总体状态 ---------------------------------------------
    species = list(getattr(game, "hengji_species", []) or [])
    g[106] = min(len(species) / 2.0, 1.0)
    g[107] = 1.0 if getattr(game, "hengji_opened", False) else 0.0
    starter = getattr(game, "hengji_starter", None)
    g[108] = _rel(int(starter), me) / 3.0 if starter is not None else 0.0

    # -- 109–112：四家是否已报叫 ---------------------------------------------
    bj = getattr(game, "baojiao", None)
    if bj is not None:
        for p in range(4):
            g[109 + _rel(p, me)] = 1.0 if bj[p] else 0.0

    # -- 113–117：自己手牌结构（刻/对/搭/孤 + 综合结构分） -------------------
    triplets, pairs, partials, lone = _hand_shape(hand)
    g[113] = float(np.clip((3 * triplets + 2 * pairs + partials) / 8.0, 0.0, 1.0))
    g[114] = min(partials / 6.0, 1.0)
    g[115] = min(pairs / 6.0, 1.0)
    g[116] = min(triplets / 4.0, 1.0)
    g[117] = min(lone / 6.0, 1.0)

    # -- 118–121：四家副露速度 = 副露组数 / max(1, 巡目)（早副露 = 快牌信号） --
    for p in range(4):
        g[118 + _rel(p, me)] = min(1.0, len(game.melds[p]) / max(1.0, turn))

    # -- 122–123：自己的听牌质量（v1 只有「等待牌种数」，没有「还剩几张」） ---
    waits: list[int] = []
    if sum(int(x) for x in hand) == (4 - n_melds_me) * 3 + 1:
        waits = list(winning_tiles([int(x) for x in hand], melds_me))
    visible = _visible(game, me)
    remain = np.clip(4.0 - visible, 0.0, 4.0)
    if waits:
        mass = float(sum(remain[int(t)] for t in waits))
        g[122] = min(mass / 13.0, 1.0)
        g[123] = min((mass / max(1, len(waits))) / 4.0, 1.0)

    # -- 124：全场已亮明张数（弃牌 + 副露折算）/ 64 --------------------------
    shown = float(sum(len(game.discards[p]) for p in range(4)))
    for p in range(4):
        for mt, _t, _f in game.melds[p]:
            shown += 4.0 if meld_is_kong(mt) else 3.0
    g[124] = min(shown / 64.0, 1.0)

    # -- 125：场上最高听牌置信（v2 有逐家值，这里给「最危险的那家」便于直取） -
    g[125] = max((_conf(game, _seat_of(r, me), turn) for r in (1, 2, 3)), default=0.0)

    # -- 126：自己可达最大番值 × 是否已听（有听牌才有兑现价值） ---------------
    tenpai = bool(g[27] > 0.5)
    g[126] = min(_max_fan(game, me) / 30.0, 1.0) * (1.0 if tenpai else 0.0)

    # -- 127：距黄庄的剩余空间 × 未听牌（v2 的 95 是「已流逝」视角，这里给「还剩多少」）
    g[127] = (wall_left / float(DECK_SIZE)) * (0.0 if tenpai else 1.0)

    # -- 128：本局鸡相关事件总数 / 8（整体鸡局势的激烈程度） -----------------
    n_ji = (len(getattr(game, "ji_events", []))
            + len(getattr(game, "claimed_ji", []))
            + len(getattr(game, "hengji_events", [])))
    g[128] = min(n_ji / 8.0, 1.0)

    # -- 129：自己手牌 + 副露的总张数 / 14 -----------------------------------
    g[129] = min((float(sum(int(x) for x in hand)) + 3.0 * n_melds_me) / 14.0, 1.0)
    return g


def encode_v3(game, player: int, base_v1, packs=None):
    """v3 编码。

    ``base_v1`` 为 v1 的 ``encode``（由 :mod:`zhuoji.encoder` 注入，避免循环导入）。
    ``packs`` 为要**启用**的包（``None`` = 全开）；其中 ``a``/``b``/``c`` 透传给 v2，
    ``d``/``e``/``f``/``g`` 在本文件内做掩码。未启用的部分整体置零，维度与参数量不变。
    """
    from .encoder_v2 import encode_v2
    spec = parse_packs_v3(packs)
    v2_packs = tuple(p for p in ("a", "b", "c") if p in spec)
    obs = encode_v2(game, player, base_v1, v2_packs)

    tiles = np.zeros((NUM_CHANNELS_V3, NUM_TILE_TYPES), dtype=np.float32)
    tiles[:_V2_CHANNELS] = obs.tiles
    obs.tiles = encode_tiles_v3(game, player, tiles)

    g = np.zeros(GLOBAL_DIM_V3, dtype=np.float32)
    g[:_V2_GLOBAL_DIM] = obs.glob
    obs.glob = encode_glob_v3(game, player, g)

    obs.tiles, obs.glob = apply_pack_mask_v3(obs.tiles, obs.glob, spec)
    return obs
