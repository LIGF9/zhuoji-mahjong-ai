"""M2-AUX：完美信息辅助标签（PTIE 式，2026-10-18）。

动机（证据链见 ``docs/训练决策铁律_20261017.md`` §5.2 / R11）
----------------------------------------------------------------
M2 的四个优化器旋钮（轮次 / epoch / 数据量 / 蒸馏强度）全部受控排除后，
剩下的机制性假设只有一个：**稀疏终局奖励是唯一的梯度来源**。本模块把
「训练时可见、执行时不可见」的完美信息变成四个稠密、**静止**的监督目标，
挂在共享主干上（UNREAL 式辅助头）：

1. ``shanten``   —— 自家手牌向听数（8 桶分类）。强迫主干表示手牌结构。
2. ``opp_ting``  —— 三个对手是否听牌（3 维二值）。输入里没有对手手牌，
   要预测它就必须从牌河/副露推断隐藏状态 —— 这正是对 defensive 对手
   落后 43.4% 的那条短板（``reports/M1瓶颈诊断-20261014.md`` T0.2）。
3. ``wall``      —— 牌墙余量（0..1 回归）。荒牌/尾盘的节奏判断。
4. ``danger``    —— 27 种牌此刻打出去是否会被任一对手吃碰杠胡
   （多标签二值）。**只在自家出牌决策点**计算；这是直接的放铳风险地图。

标签由采集端在决策时刻从 ``game`` 对象直接算出（自对弈采集器天然持有
完整状态），**推理路径完全不用** —— 与 PerfectDou 的 PTIE 框架同构
（perfect-training, imperfect-execution）。

所有判定函数（``is_tenpai`` / ``winning_tiles`` / ``shanten``）内部带缓存，
同一手牌状态的重复判定接近零成本；实测开销见 ``scripts/bench_aux_labels.py``。
"""
from __future__ import annotations

import numpy as np

from .fan import is_tenpai, winning_tiles
from .shanten import shanten
from .tiles import NUM_TILE_TYPES

#: 向听桶数：-1(已胡)..5+ 合并为 8 类
SHANTEN_BUCKETS = 8
#: 独山麻将开局牌墙 = 108 − 13×4 − 庄家 1 = 55；标签做 0..1 归一化
WALL_INIT = 55.0


def shanten_bucket(hand_counts, melds) -> int:
    """自家向听数 → 0..7 的桶号（-1 已胡并入 0 号桶）。"""
    s = shanten(list(hand_counts), len(melds))
    return int(min(max(s + 1, 0), SHANTEN_BUCKETS - 1))


def opp_ting_flags(game, me: int) -> np.ndarray:
    """三个对手（按相对座位顺序）是否听牌，shape (3,) float32。"""
    out = np.zeros(3, dtype=np.float32)
    for k in range(3):
        q = (me + 1 + k) % 4
        out[k] = 1.0 if is_tenpai(list(game.hands[q]), game.melds[q]) else 0.0
    return out


def wall_frac(game) -> float:
    """牌墙余量，归一化到 [0, 1]。"""
    return float(min(max(game.wall_left / WALL_INIT, 0.0), 1.0))


def danger_map(game, me: int) -> np.ndarray:
    """27 种牌此刻打出会被任一对手吃/碰/杠/胡的多标签地图，shape (27,)。

    判定口径（当前决策瞬间、忽略 pass_lock 的近似）：
    * 胡：``t ∈ winning_tiles(hand_q, melds_q)``（对手处于 13 张听口形态时）；
    * 碰：``hand_q[t] >= 2``；
    * 杠：``hand_q[t] >= 3``（直杠）。

    输入编码里没有对手暗手，这个标签**只能**通过牌河/副露/自家手牌推断 ——
    它教的不是"背答案"，是"隐藏状态推断"本身。
    """
    danger = np.zeros(NUM_TILE_TYPES, dtype=np.float32)
    for q in range(4):
        if q == me:
            continue
        hand = game.hands[q]
        melds = game.melds[q]
        # 碰/杠：纯计数，零判定成本
        for t in range(NUM_TILE_TYPES):
            if hand[t] >= 2:
                danger[t] = 1.0
        # 胡：只有听牌形态才可能（winning_tiles 内部对非听口手牌返回 []）
        for t in winning_tiles(list(hand), melds):
            danger[t] = 1.0
    return danger


def labels_for(game, me: int, head: str):
    """一次决策点的全部辅助标签。

    返回 ``(shanten_bucket:int, opp_ting:(3,)f32, wall:float, danger:(27,)f32|None)``；
    非出牌头（self_kong/respond）的 ``danger`` 为 ``None``（该头的标签语义
    是"打这张牌会怎样"，只在出牌决策点成立）。
    """
    danger = danger_map(game, me) if head == "discard" else None
    return (shanten_bucket(game.hands[me], game.melds[me]),
            opp_ting_flags(game, me), wall_frac(game), danger)
