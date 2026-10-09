"""独山麻将引擎机制测试。"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from zhuoji.dushan import (  # noqa: E402
    DushanConfig,
    DushanGame,
    JI_CHONGFENG,
    JI_HENG,
    JI_YAO,
    ZI_MO_JI,
    fanji_tiles_from,
)
from zhuoji.fan import is_seven_pairs  # noqa: E402
from zhuoji.rules import Action, DISCARD, HU, MELD_KONG_CONCEALED, MELD_KONG_EXPOSED, MELD_PONG, PASS, Phase, meld_is_kong  # noqa: E402
from zhuoji.tiles import B, T, W, YAOJI, tile_from  # noqa: E402


def counts(*tiles: int) -> list[int]:
    c = [0] * 27
    for t in tiles:
        c[t] += 1
    return c


def new_game(seed: int = 7) -> DushanGame:
    return DushanGame(DushanConfig(), seed=seed)


# ---------------------------------------------------------------------------
# 牌型判定
# ---------------------------------------------------------------------------
def test_win_types():
    g = new_game()
    # 平胡：123万 456万 789万 123条 55筒（无清一色、无刻子结构）
    h = counts(*[tile_from(W, r) for r in (1, 2, 3, 4, 5, 6, 7, 8, 9)],
               *[tile_from(T, r) for r in (1, 2, 3)], tile_from(T, 5), tile_from(T, 5))
    assert g._win_type_set(h, []) == {"ping_hu"}

    # 清一色：全万顺子
    h = counts(*[tile_from(W, r) for r in (1, 2, 3, 4, 5, 6, 7, 8, 9, 1, 2, 3, 5, 5)])
    assert g._win_type_set(h, []) == {"qing_yi_se"}

    # 小七对：10 分（全万 → 叠加清一色）
    h = counts(*[tile_from(W, r) for r in (1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7)])
    assert g._win_type_set(h, []) == {"qi_dui", "qing_yi_se"}

    # 混色小七对：仅 10 分
    h = counts(*[tile_from(W, r) for r in (1, 1, 2, 2, 3, 3)],
               *[tile_from(T, r) for r in (4, 4, 5, 5)],
               *[tile_from(B, r) for r in (6, 6, 7, 7)])
    assert g._win_type_set(h, []) == {"qi_dui"}

    # 龙七对：23 分（含 4 张，混色）
    h = counts(*[tile_from(W, r) for r in (1, 1, 1, 1, 2, 2)],
               *[tile_from(T, r) for r in (3, 3, 4, 4)],
               *[tile_from(B, r) for r in (5, 5, 6, 6)])
    assert g._win_type_set(h, []) == {"long_qi_dui"}

    # 大对子（4 刻子 + 1 对，无副露）
    h = counts(*([tile_from(W, 1)] * 3 + [tile_from(W, 2)] * 3 +
                 [tile_from(W, 3)] * 3 + [tile_from(T, 5)] * 3 + [tile_from(T, 7)] * 2))
    types = g._win_type_set(h, [])
    assert "da_dui_zi" in types

    # 清一色 + 大对子 叠加
    h = counts(*([tile_from(W, 1)] * 3 + [tile_from(W, 2)] * 3 +
                 [tile_from(W, 3)] * 3 + [tile_from(W, 5)] * 3 + [tile_from(W, 7)] * 2))
    types = g._win_type_set(h, [])
    assert types == {"da_dui_zi", "qing_yi_se"}

    # 单钓将：4 个副露刻子 + 手里单张（全条 → 叠加清一色）
    melds = [(MELD_PONG, tile_from(T, 1), 1), (MELD_PONG, tile_from(T, 2), 2),
             (MELD_PONG, tile_from(T, 3), 3), (MELD_KONG_CONCEALED, tile_from(T, 4), 0)]
    h = counts(tile_from(T, 5), tile_from(T, 5))
    assert g._win_type_set(h, melds) == {"dan_diao", "qing_yi_se"}

    # 混色单钓将：仅 10 分
    melds = [(MELD_PONG, tile_from(W, 1), 1), (MELD_PONG, tile_from(T, 2), 2),
             (MELD_PONG, tile_from(B, 3), 3), (MELD_KONG_CONCEALED, tile_from(W, 4), 0)]
    h = counts(tile_from(T, 5), tile_from(T, 5))
    assert g._win_type_set(h, melds) == {"dan_diao"}
    print("ok 牌型判定")


# ---------------------------------------------------------------------------
# 通行证
# ---------------------------------------------------------------------------
def test_passport():
    g = new_game()
    # 只听平胡、无杠、无报叫 → 无通行证
    g.hands[0] = counts(*[tile_from(W, r) for r in (1, 2, 3, 4, 5, 6, 7, 8, 9)],
                        *[tile_from(T, r) for r in (1, 2, 3)], tile_from(T, 5))
    g.melds[0] = []
    g.current = 0
    g.phase = Phase.DISCARD
    assert not g._has_passport(0)
    assert not g.can_ron(0, tile_from(T, 5))       # 平胡点炮被拦

    # 有杠 → 通行证（1 副露时听牌手 10 张）
    g.melds[0] = [(MELD_KONG_CONCEALED, tile_from(W, 1), 0)]
    g.hands[0] = counts(*[tile_from(W, r) for r in (2, 3, 4, 5, 6, 7)],
                        *[tile_from(T, r) for r in (1, 2, 3)], tile_from(T, 5))
    assert g._has_passport(0)
    assert g.can_ron(0, tile_from(T, 5))

    # 听非平胡（七对）→ 通行证
    g.melds[0] = []
    g.hands[0] = counts(*[tile_from(W, r) for r in (1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7)])
    assert g._has_passport(0)
    print("ok 通行证")


def test_baojiao():
    g = new_game()
    # 报叫后不可碰
    g.baojiao[1] = True
    g.hands[1] = counts(*([tile_from(W, 3)] * 2 + [tile_from(W, 5)]))
    assert not g._can_peng(1, tile_from(W, 3))
    g.baojiao[1] = False
    assert g._can_peng(1, tile_from(W, 3))

    # 报叫玩家的弃牌，别人无通行证也能胡（平胡点炮）
    g.hands[0] = counts(*[tile_from(W, r) for r in (1, 2, 3, 4, 5, 6, 7, 8, 9)],
                        *[tile_from(T, r) for r in (1, 2, 3)], tile_from(T, 5))
    g.melds[0] = []
    g.last_discard = (2, tile_from(T, 5))
    g.baojiao[2] = True
    g.repao_discard = False
    assert g.can_ron(0, tile_from(T, 5))
    g.baojiao[2] = False
    assert not g.can_ron(0, tile_from(T, 5))
    print("ok 报叫")


def test_repao():
    g = new_game()
    # 热炮：杠后第一张弃牌，无通行证也能胡
    g.hands[0] = counts(*[tile_from(W, r) for r in (1, 2, 3, 4, 5, 6, 7, 8, 9)],
                        *[tile_from(T, r) for r in (1, 2, 3)], tile_from(T, 5))
    g.melds[0] = []
    g.last_discard = (1, tile_from(T, 5))
    g.repao_discard = True
    g.baojiao[1] = False
    assert g.can_ron(0, tile_from(T, 5))
    g.repao_discard = False
    assert not g.can_ron(0, tile_from(T, 5))
    print("ok 热炮")


# ---------------------------------------------------------------------------
# 鸡事件
# ---------------------------------------------------------------------------
def test_chicken_discard_events():
    g = new_game()
    # 清空各家手牌，换成无碰/杠可能的散牌，避免响应干扰
    from zhuoji.tiles import B
    for p in range(4):
        g.hands[p] = [0] * 27
        for i, r in enumerate((2, 3, 4, 5, 6, 7)):
            g.hands[p][tile_from((p + i) % 3, r)] += 1
        g.melds[p] = []
    g.hands[0][YAOJI] += 3   # 玩家 0 打 1条 没人能碰（其余人手里没有 1条 对）
    for p in range(4):
        g.hands[p][YAOJI] += 3  # 每家发几张 1条 备用

    def discard(p, tile):
        g.current = p
        g.phase = Phase.DISCARD
        g.step(Action("discard", tile))

    # 玩家 0 首张打 1条 → 冲锋鸡
    discard(0, YAOJI)
    assert g.ji_events[-1] == (0, JI_CHONGFENG, "冲锋鸡")
    assert not g.hengji_opened

    # 玩家 1 先打非鸡牌（首张用掉），再打 1条 → 全场第一张鸡且非冲锋鸡 → 横鸡开轮
    discard(1, tile_from(W, 2))
    discard(1, YAOJI)
    assert g.ji_events[-1] == (1, JI_HENG, "横鸡")
    assert g.hengji_opened and g.hengji_active          # 横鸡轮已开启（可同轮跟打）
    assert g.hengji_species == [YAOJI]

    # 玩家 2 首张打 1条 → 冲锋鸡优先于横鸡（该手也算本轮参与）
    discard(2, YAOJI)
    assert g.ji_events[-1] == (2, JI_CHONGFENG, "冲锋鸡")
    assert g.hengji_active

    # 玩家 2 再次出牌 → 已在本轮内 → 该轮收口，此手按普通鸡计
    discard(2, YAOJI)
    assert g.ji_events[-1] == (2, JI_YAO, "幺鸡")
    assert not g.hengji_active

    # 玩家 1（开轮者）再打 1条 → 本种横鸡已开且轮已收口 → 普通鸡
    discard(1, YAOJI)
    assert g.ji_events[-1] == (1, JI_YAO, "幺鸡")
    print("ok 鸡事件")


def test_baojiao_auto():
    cfg = DushanConfig(baojiao_auto=True)
    g = DushanGame(cfg, seed=123)
    # 玩家 0 首张打出后听牌 → 自动报叫
    g.hands[0] = counts(*[tile_from(W, r) for r in (1, 2, 3, 4, 5, 6, 7, 8, 9)],
                        *[tile_from(T, r) for r in (1, 2, 3)], tile_from(T, 4),
                        tile_from(T, 9))
    g.hands[1] = counts(*[tile_from(B, r) for r in (1, 2, 3, 4, 5, 6, 7, 8, 9)],
                        *[tile_from(B, r) for r in (1, 2, 3)], tile_from(B, 5))
    g.current = 0
    g.phase = Phase.DISCARD
    g.step(Action("discard", tile_from(T, 9)))
    assert g.baojiao[0], "听牌玩家首张打出后应自动报叫"
    # 不可再碰/杠
    g.hands[0][tile_from(T, 9)] += 2
    assert not g._can_peng(0, tile_from(T, 9))
    print("ok 自动报叫")


# ---------------------------------------------------------------------------
# 自摸分：每家付 3 + 牌型
# ---------------------------------------------------------------------------
def test_tsumo_payment():
    g = new_game()
    g.hands[0] = counts(*[tile_from(W, r) for r in (1, 2, 3, 4, 5, 6, 7, 8, 9)],
                        *[tile_from(T, r) for r in (2, 3, 4, 5, 5)])
    g.melds[0] = []
    # 其余各家换成不沾鸡、不沾翻鸡的散牌
    for p in (1, 2, 3):
        g.hands[p] = [0] * 27
        for r in (1, 3, 5):
            g.hands[p][tile_from(B, r)] += 1
        g.melds[p] = []
    # 翻鸡牌设为 T7（上下鸡 → T6/T8），无人持有
    g.wall[g.wall_pos] = tile_from(T, 7)
    g.current = 0
    g.phase = Phase.DISCARD
    g._any_discard = True          # 非天胡
    g._settle_win(g.current, -1, is_tsumo=True)   # 直接触发自摸结算（不走动作合法性路径）
    res = g.result
    assert res["type"] == "win"
    d = res["deltas"]
    # 自摸 +3 鸡 → 两两互减：每家付 3（平胡无牌型鸡、无人持鸡/翻鸡牌）
    assert abs(d[0] - 3 * ZI_MO_JI) < 1e-9, d
    assert all(abs(d[q] + ZI_MO_JI) < 1e-9 for q in (1, 2, 3))
    assert abs(sum(d)) < 1e-9
    print("ok 自摸计分")


# ---------------------------------------------------------------------------
# 黄庄包鸡
# ---------------------------------------------------------------------------
def _huang_scenario(cfg: DushanConfig) -> DushanGame:
    """构造「p0 打出最后一张牌 → 黄庄、p1 未听且有冲锋鸡」的盘子。"""
    g = DushanGame(cfg, seed=7)
    # 玩家 0：杂牌不听（14 张），打 T8（无人能碰/胡）
    g.hands[0] = counts(*[tile_from(W, r) for r in (1, 1, 2, 3, 4, 5, 6, 7, 8, 9)],
                        *[tile_from(T, r) for r in (2, 4, 6, 8)])
    # 未听手：2 副露刻子 + 7 张杂牌（三家同构不同花色）
    for p, s in ((1, B), (2, W), (3, T)):
        g.hands[p] = [0] * 27
        for r in (1, 1, 3, 6, 7, 8, 9):
            g.hands[p][tile_from(s, r)] += 1
        g.melds[p] = [(MELD_PONG, tile_from(s, 2), 0), (MELD_PONG, tile_from(s, 5), 0)]
    g.melds[0] = []
    # 玩家 1 打过一张冲锋鸡（+3）
    g.ji_events.append((1, JI_CHONGFENG, "冲锋鸡"))
    g.wall_pos = len(g.wall)
    g.phase = Phase.DISCARD
    g.current = 0
    g.step(Action("discard", tile_from(T, 8)))
    return g


def test_huangzhuang_baoji():
    # 旧口径：流局未听包鸡开、包杠开
    g = _huang_scenario(DushanConfig(huang_baoji=True, huang_baogang=True,
                                     huang_baodapai=False))
    res = g.result
    assert res["type"] == "huangzhuang", res
    d = res["deltas"]
    # 玩家 1 包鸡 3 → 各赔 3 给其余三家
    assert abs(d[1] + 9.0) < 1e-9, d
    assert abs(d[0] - 3.0) < 1e-9 and abs(d[2] - 3.0) < 1e-9 and abs(d[3] - 3.0) < 1e-9
    assert abs(sum(d)) < 1e-9
    print("ok 黄庄包鸡（旧口径显式开启）")


def test_huang_defaults_no_bao():
    """新默认：流局不包鸡、不包杠（且无人听牌 → 包大牌面也不触发）→ 全零。"""
    g = _huang_scenario(DushanConfig())
    d = g.result["deltas"]
    assert all(abs(x) < 1e-9 for x in d), d
    print("ok 流局默认不包鸡/不包杠")


def test_end_baoji_baogang_switches():
    """胡牌局：包鸡默认开、包杠默认关；开关分别生效。"""
    def run(baoji: bool, baogang: bool) -> tuple[list[float], dict]:
        g = DushanGame(DushanConfig(end_baoji=baoji, end_baogang=baogang), seed=7)
        _setup_plain_hands(g)
        g.melds[1] = [(MELD_KONG_CONCEALED, tile_from(B, 1), 1)]   # p1 未听带 1 个杠
        g.ji_events.append((1, JI_CHONGFENG, "冲锋鸡"))             # p1 未听、打出过冲锋鸡
        g._settle_win(g.current, -1, is_tsumo=True)   # 直接触发自摸结算（不走动作合法性路径）
        assert g.result["type"] == "win", g.result
        return g.result["deltas"], g.result

    d_on, res_on = run(True, False)
    assert abs(d_on[1] + 12.0) < 1e-9, d_on         # 自摸 3 + 包鸡 3×3 家
    d_kong, res_kong = run(True, True)
    assert abs(d_kong[1] + 21.0) < 1e-9, d_kong     # 再加包杠 3×3 家
    d_off, _ = run(False, False)
    assert abs(d_off[1] + 3.0) < 1e-9, d_off        # 都关 → 只剩自摸 3
    # 分项归属：包鸡/包杠记在**付款方**（未听牌者 p1）名下 —— 责任方口径
    for want, res in (("包鸡", res_on), ("包杠", res_kong)):
        rows = [it for pd in res["pair_detail"] for it in pd["items"]
                if it["label"] == want]
        assert rows, want
        assert all(int(it["who"]) == 1 for it in rows), (want, rows)
        assert all(abs(float(it["value"]) - 3.0) < 1e-9 for it in rows), (want, rows)
    print("ok 胡牌局 包鸡/包杠开关（分项归付款方）")


def test_huang_baodapai():
    """流局包大牌面（新口径）：未听者按「自摸分 + 听牌者可达最大牌型分」赔付。"""
    g = DushanGame(DushanConfig(huang_baodapai=True, huang_baoji=False,
                                huang_baogang=False), seed=11)
    # p1 听小七对（B1..B6 各两张 + 单张 T9，等 T9）——非平胡牌面
    g.hands[1] = counts(*[tile_from(B, r) for r in (1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6)],
                        tile_from(T, 9))
    g.melds[1] = []
    # p0/p2/p3 未听杂牌
    g.hands[0] = counts(*[tile_from(W, r) for r in (1, 1, 2, 3, 4, 5, 6, 7, 8, 9)],
                        *[tile_from(T, r) for r in (2, 4, 6, 8)])
    for p, s in ((2, W), (3, T)):
        g.hands[p] = [0] * 27
        for r in (1, 1, 3, 6, 7, 8, 9):
            g.hands[p][tile_from(s, r)] += 1
    g.wall_pos = len(g.wall)
    g.phase = Phase.DISCARD
    g.current = 0
    g.step(Action("discard", tile_from(T, 8)))
    res = g.result
    assert res["type"] == "huangzhuang", res
    val, cn = g._max_hu_type(1)
    assert val > 0 and cn, (val, cn)
    pay = 3.0 + val                     # 自摸分 3 + 大牌面分
    labels = [row for row in res["detail"] if row[1] == "包大牌面"]
    assert labels and abs(labels[0][3] - pay) < 1e-9, res["detail"]
    assert res["baodapai"] == [{"seat": 1, "type": cn, "type_value": float(val),
                                "pay": pay}], res["baodapai"]
    d = res["deltas"]
    # 三家未听各赔 pay 给 p1
    assert abs(d[1] - 3 * pay) < 1e-9, d
    assert abs(d[0] + pay) < 1e-9 and abs(d[2] + pay) < 1e-9 and abs(d[3] + pay) < 1e-9
    # 关掉开关则无此赔付
    g2 = DushanGame(DushanConfig(huang_baodapai=False, huang_baoji=False,
                                 huang_baogang=False), seed=11)
    import copy
    g2.hands = [list(h) for h in g.hands]
    g2.melds = [[m for m in m_] for m_ in g.melds]
    g2.wall_pos = len(g2.wall)
    g2.phase = Phase.DISCARD
    g2.current = 0
    g2.step(Action("discard", tile_from(T, 8)))
    assert all(abs(x) < 1e-9 for x in g2.result["deltas"]), g2.result["deltas"]
    print(f"ok 流局包大牌面（{cn} 自摸3+牌型{val:g} = 每家 {pay:g} 分）")


def test_huang_new_rules():
    """流局新口径：不结算鸡与杠；平胡听牌者只收一个自摸分；听牌者之间互不结算。"""
    g = DushanGame(DushanConfig(), seed=23)
    # p0 平胡听：123万 456万 789万 + 2条3条 + 9条9条（等 1条/4条）→ 平胡（大牌面 0）
    g.hands[0] = counts(*[tile_from(W, r) for r in (1, 2, 3, 4, 5, 6, 7, 8, 9)],
                        tile_from(T, 2), tile_from(T, 3), tile_from(T, 9), tile_from(T, 9))
    g.melds[0] = []
    # p1 听小七对：B1..B6 各两张 + 单张 T9
    g.hands[1] = counts(*[tile_from(B, r) for r in (1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6)],
                        tile_from(T, 9))
    g.melds[1] = []
    # p2/p3 未听杂牌 + 各 2 个碰（其中 p2 有一张打出的冲锋鸡，验证流局不再结算鸡）
    g.hands[2] = [0] * 27
    for r in (1, 1, 3, 6, 7, 8, 9):
        g.hands[2][tile_from(W, r)] += 1
    g.melds[2] = [(MELD_PONG, tile_from(W, 1), 0), (MELD_PONG, tile_from(W, 4), 0)]
    g.hands[3] = [0] * 27
    for r in (1, 1, 3, 6, 7, 8, 9, 9):     # 多一张 9筒：打出后回到「未听」杂牌形
        g.hands[3][tile_from(B, r)] += 1
    g.melds[3] = [(MELD_PONG, tile_from(B, 1), 0), (MELD_PONG, tile_from(B, 4), 0)]
    g.ji_events.append((2, JI_CHONGFENG, "冲锋鸡"))
    g.ji_events.append((3, JI_HENG, "横鸡"))
    assert g._tenpai_flag(0) and g._tenpai_flag(1), "p0/p1 应听牌"
    v0, cn0 = g._max_hu_type(0)
    v1, cn1 = g._max_hu_type(1)
    assert v0 == 0, (v0, cn0)              # 只能平胡 → 无大牌面
    assert v1 > 0, (v1, cn1)

    g.wall_pos = len(g.wall)
    g.phase = Phase.DISCARD
    g.current = 3
    g.step(Action("discard", tile_from(B, 9)))   # p3 打出，无人能要 → 黄庄
    assert not g._tenpai_flag(2) and not g._tenpai_flag(3), "p2/p3 应未听"
    res = g.result
    assert res["type"] == "huangzhuang", res
    pay0, pay1 = 3.0 + v0, 3.0 + v1
    d = res["deltas"]
    assert abs(d[0] - 2 * pay0) < 1e-9, d          # 两个未听者各赔一份
    assert abs(d[1] - 2 * pay1) < 1e-9, d
    assert abs(d[2] + pay0 + pay1) < 1e-9, d
    assert abs(d[3] + pay0 + pay1) < 1e-9, d
    assert abs(sum(d)) < 1e-9
    # 听牌者之间不互算（pair[0][1] / pair[1][0] 均为 0）
    assert res["pair"][0][1] == 0 and res["pair"][1][0] == 0, res["pair"]
    assert res["pair"][2][3] == 0 and res["pair"][3][2] == 0, res["pair"]
    # 鸡与杠分：流局一律不结算
    assert res["chickens"] == [0.0, 0.0, 0.0, 0.0], res["chickens"]
    bad = [r for r in res["detail"]
           if r[1] in ("包鸡", "包杠", "杠分", "责任鸡", "手中鸡", "翻鸡")]
    assert not bad, bad
    print(f"ok 流局新口径（平胡听 每家 {pay0:g}；{cn1} 每家 {pay1:g}；鸡/杠不结算）")


# ---------------------------------------------------------------------------
# 教程口径：责任鸡 / 牌型奖励鸡数 / 鸡数两两互减
# ---------------------------------------------------------------------------
def _setup_plain_hands(g):
    """清一色自摸平胡牌面 + 三家杂牌（避免鸡/翻鸡干扰）。"""
    g.hands[0] = counts(*[tile_from(W, r) for r in (1, 2, 3, 4, 5, 6, 7, 8, 9)],
                        *[tile_from(T, r) for r in (2, 3, 4, 5, 5)])
    for p in (1, 2, 3):
        g.hands[p] = [0] * 27
        for r in (1, 3, 5):
            g.hands[p][tile_from(B, r)] += 1
        g.melds[p] = []
    g.melds[0] = []
    g.wall[g.wall_pos] = tile_from(T, 7)   # 翻鸡牌 = T8，无人持有
    g.current = 0
    g.phase = Phase.DISCARD
    g._any_discard = True


def test_zeeren_ji():
    g = new_game()
    _setup_plain_hands(g)
    # 玩家 1 打出的冲锋鸡被玩家 0 碰走：打出者不再计打出冲锋鸡，
    # 改为赔溢价 2（冲锋 3 − 幺鸡 1）给碰者；牌面 3 鸡计入碰者手中鸡
    g.ji_events.append((1, JI_CHONGFENG, "冲锋鸡"))
    g.melds[0].append((MELD_PONG, YAOJI, 1))     # 碰走的鸡牌（3 张牌面）
    g.claimed_ji.append((1, 0, "冲锋鸡", False))   # claimed_ji 第 3 位是「类型名」
    g._settle_win(g.current, -1, is_tsumo=True)   # 直接触发自摸结算（不走动作合法性路径）
    res = g.result
    assert res["type"] == "win"
    c = res["chickens"]
    assert abs(c[0] - 3.0) < 1e-9, c               # 牌面鸡 3（对所有人有效）
    d = res["deltas"]
    # 自摸 3×3 家 = 9；鸡数互减：p0 牌面 3 → 各家再付 3 = 9；溢价：p1 赔 p0 2
    # d0 = 9 + 9 + 2 = 20, d1 = -3 - 3 - 2 = -8, d2 = d3 = -6
    assert abs(d[0] - 20.0) < 1e-9, d
    assert abs(d[1] + 8.0) < 1e-9, d
    assert abs(d[2] + 6.0) < 1e-9 and abs(d[3] + 6.0) < 1e-9
    assert abs(sum(d)) < 1e-9
    print("ok 责任鸡（溢价口径）")


def test_type_bonus_pairwise():
    g = new_game()
    _setup_plain_hands(g)
    # 玩家 0 手牌全万 → 清一色（10 鸡）+ 自摸 3 = 每家付 13
    g.hands[0] = counts(*[tile_from(W, r) for r in (1, 2, 3, 4, 5, 6, 7, 8, 9)],
                        *[tile_from(W, r) for r in (1, 2, 3, 5, 5)])
    # 玩家 1 听牌（听 3条）且持 2 只幺鸡 → c1 = 2
    g.hands[1] = counts(*[tile_from(W, r) for r in (1, 2, 3, 4, 5, 6, 7, 8, 9)],
                        *[tile_from(T, r) for r in (1, 1, 2, 2)])
    g._settle_win(g.current, -1, is_tsumo=True)   # 直接触发自摸结算（不走动作合法性路径）
    res = g.result
    c = res["chickens"]
    assert abs(c[0] - 0.0) < 1e-9, c               # 鸡数池不含胡牌方式/牌型
    assert abs(c[1] - 2.0) < 1e-9, c
    assert "qing_yi_se" in res["fan_type"]
    d = res["deltas"]
    # 自摸3+清一色10 → 每家付 p0 13；鸡数互减：p1 的 2 鸡对 p0/p2/p3 各生效
    # d0 = 13×3 − 2 = 37；d1 = 2×3 − 13 = −7；d2 = d3 = −13 − 2 = −15
    assert abs(d[0] - 37.0) < 1e-9, d
    assert abs(d[1] + 7.0) < 1e-9, d
    assert abs(d[2] + 15.0) < 1e-9 and abs(d[3] + 15.0) < 1e-9
    assert abs(sum(d)) < 1e-9
    # pair 矩阵与 delta 一致
    pair = res["pair"]
    for p in range(4):
        row = sum(pair[p]) - sum(pair[q][p] for q in range(4))
        assert abs(row - d[p]) < 1e-9
    print("ok 牌型奖励/两两互减")


# ---------------------------------------------------------------------------
# 杠分归属（明杠点杠者独付 / 暗杠全场付）与点炮独付（方式+牌型）
# ---------------------------------------------------------------------------
def test_kong_and_ron_settlement():
    # A. 明杠：杠分 3 只由点杠者付
    g = new_game()
    _setup_plain_hands(g)
    g.hands[0] = counts(*[tile_from(W, r) for r in (1, 2, 3, 4, 5, 6)],
                        *[tile_from(T, r) for r in (2, 3, 4)], tile_from(B, 9))
    g.melds[0] = [(MELD_KONG_EXPOSED, tile_from(B, 1), 1)]   # p1 点杠
    for p in (1, 2, 3):
        g.hands[p] = [0] * 27
        for r in (1, 3, 5):
            g.hands[p][tile_from(B, r)] += 1
        g.melds[p] = []
    g.repao_discard = False
    g._settle_win(0, tile_from(B, 9), is_tsumo=False, loser=1)
    d = g.result["deltas"]
    # p1 付：点炮 3 + 明杠分 3 = 6；p2/p3 不付（明杠分与其无关）
    assert abs(d[0] - 6.0) < 1e-9, d
    assert abs(d[1] + 6.0) < 1e-9, d
    assert abs(d[2]) < 1e-9 and abs(d[3]) < 1e-9, d
    assert abs(sum(d)) < 1e-9
    print("ok 明杠杠分只对点杠者")

    # B. 暗杠 + 自摸：杠分与自摸全场每家付
    g = new_game()
    _setup_plain_hands(g)
    g.hands[0] = counts(*[tile_from(W, r) for r in (1, 2, 3, 4, 5, 6)],
                        *[tile_from(T, r) for r in (2, 3, 4)], tile_from(B, 9), tile_from(B, 9))
    g.melds[0] = [(MELD_KONG_CONCEALED, tile_from(B, 1), 0)]
    for p in (1, 2, 3):
        g.hands[p] = [0] * 27
        for r in (1, 3, 5):
            g.hands[p][tile_from(B, r)] += 1
        g.melds[p] = []
    g._settle_win(0, -1, is_tsumo=True)
    d = g.result["deltas"]
    # 每家付：自摸 3 + 暗杠分 3 = 6
    assert abs(d[0] - 18.0) < 1e-9, d
    assert all(abs(d[q] + 6.0) < 1e-9 for q in (1, 2, 3)), d
    assert abs(sum(d)) < 1e-9
    print("ok 暗杠全场付+自摸全场收")

    # C. 点炮：牌型分只由点炮者付
    g = new_game()
    _setup_plain_hands(g)
    # 清一色听牌：W1W1 + W234 W456 W789 + W5，点胡 W5 成 W555 刻
    g.hands[0] = counts(*[tile_from(W, r) for r in (1, 1, 2, 3, 4, 4, 5, 6, 7, 8, 9, 5, 5)])
    for p in (1, 2, 3):
        g.hands[p] = [0] * 27
        for r in (1, 3, 5):
            g.hands[p][tile_from(B, r)] += 1
        g.melds[p] = []
    g.repao_discard = False
    g._settle_win(0, tile_from(W, 5), is_tsumo=False, loser=2)   # p2 点炮
    res = g.result
    assert res["type"] == "win" and "qing_yi_se" in res["fan_type"]
    d = res["deltas"]
    # p2 付：点炮 3 + 清一色 10 = 13；其余不付
    assert abs(d[0] - 13.0) < 1e-9, d
    assert abs(d[2] + 13.0) < 1e-9, d
    assert abs(d[1]) < 1e-9 and abs(d[3]) < 1e-9, d
    assert abs(sum(d)) < 1e-9
    print("ok 点炮方式+牌型独付")


# ---------------------------------------------------------------------------
# 教师兼容 + 批量零和
# ---------------------------------------------------------------------------
def test_teachers_and_zero_sum():
    from zhuoji.bots import TEACHER_STYLES, HeuristicBot

    def play_one(seed):
        bots = [HeuristicBot(seed=seed * 4 + i, weights=TEACHER_STYLES["balanced"])
                for i in range(4)]
        g = DushanGame(seed=seed)
        res = g.play(bots)
        assert res["type"] != "timeout", f"seed={seed} 超时"
        s = sum(res["deltas"])
        assert abs(s) < 1e-6, f"seed={seed} 非零和: {s}"
        return res

    huang = 0
    wins = 0
    for i in range(120):
        r = play_one(3000 + i)
        if r["type"] == "huangzhuang":
            huang += 1
        else:
            wins += 1
    print(f"ok 教师 120 局零和（胡 {wins} / 黄 {huang}）")




# ---------------------------------------------------------------------------
# 开关特性：上下鸡 / 满堂鸡 / 开局翻鸡
# ---------------------------------------------------------------------------
def _fanji_of(mode: str, flip: int) -> tuple[bool, set]:
    g = DushanGame(DushanConfig(jiesuan_fanji=mode), seed=11)
    g.wall = g.wall[:40]                   # 截断墙保证非黄庄（wall_left=len-pos）
    g.wall[30] = flip                      # wall_pos 所指 = 结算翻的牌
    g.wall_pos = 30
    return g._dushan_chicken_state()


def test_jiesuan_fanji():
    # 结算翻鸡默认 = 上下鸡（+1 与 -1 同花色都算）
    g = new_game()
    assert g.cfg.jiesuan_fanji == "both"
    # 翻出 5万 → 上鸡 6万 + 下鸡 4万
    gold, fanji = _fanji_of("both", tile_from(W, 5))
    assert fanji == {tile_from(W, 6), tile_from(W, 4)}, fanji
    assert not gold
    # 仅上鸡
    _gold, f_up = _fanji_of("up", tile_from(W, 5))
    assert f_up == {tile_from(W, 6)}, f_up
    # 仅下鸡
    _gold, f_down = _fanji_of("down", tile_from(W, 5))
    assert f_down == {tile_from(W, 4)}, f_down

    # 金鸡：上鸡时翻九条（9+1 回一条）/ 下鸡时翻二条（2-1 回一条）
    assert _fanji_of("up", tile_from(T, 9))[0] is True
    assert _fanji_of("up", tile_from(T, 2))[0] is False
    assert _fanji_of("down", tile_from(T, 2))[0] is True
    assert _fanji_of("down", tile_from(T, 9))[0] is False
    assert _fanji_of("both", tile_from(T, 9))[0] is True
    assert _fanji_of("both", tile_from(T, 2))[0] is True
    # 兼容旧写法：布尔 True/False（结算翻鸡不允许关闭 → False 回落上下鸡）
    assert DushanConfig(jiesuan_fanji=True).jiesuan_fanji == "both"
    assert DushanConfig(jiesuan_fanji=False).jiesuan_fanji == "both"
    print("ok 结算翻鸡（上下鸡/上鸡/下鸡）")


def test_mantiangji():
    # 满堂鸡：翻鸡牌即使已在弃牌区也计入持有鸡数
    cfg = DushanConfig(mantiangji=True, jiesuan_fanji="up")
    g = DushanGame(cfg, seed=13)
    for p in range(4):
        g.hands[p] = counts(*[tile_from(W, r) for r in (1, 2, 3, 4, 5, 6, 7, 8, 9)],
                            *[tile_from(B, r) for r in (1, 2, 3, 5)])
        g.ever_tenpai[p] = True
    flip = tile_from(W, 5)                 # 翻出 5万 → 上鸡 6万
    g.wall = g.wall[:40]
    g.wall[39] = flip
    g.wall_pos = 39
    _gold, fanji = g._dushan_chicken_state()
    assert fanji == {tile_from(W, 6)}
    g.hands[0][tile_from(W, 6)] = 1        # 玩家 0 手持 1 张 6万
    g.hands[0][tile_from(W, 5)] -= 1       # 保持 14 张不严格要求，鸡数统计只看鸡
    g.discards[0].append(tile_from(W, 6))  # 弃牌区还有 1 张 6万
    c, items, _v, _t = g._chicken_counts([True, True, True, True], fanji, 1.0)
    fanji_items = [it for it in items[0] if it[0] == "翻鸡"]
    assert fanji_items and fanji_items[0][2] == 2.0, items[0]   # 手持1 + 弃牌1
    assert "含弃牌" in fanji_items[0][1]

    # 关闭满堂鸡：弃牌区不计
    g2 = DushanGame(DushanConfig(jiesuan_fanji="up"), seed=13)
    for p in range(4):
        g2.hands[p] = counts(*[tile_from(W, r) for r in (1, 2, 3, 4, 5, 6, 7, 8, 9)],
                             *[tile_from(B, r) for r in (1, 2, 3, 5)])
    g2.hands[0][tile_from(W, 6)] = 1
    g2.discards[0].append(tile_from(W, 6))
    c2, items2, _v, _t = g2._chicken_counts([True, True, True, True], fanji, 1.0)
    f2 = [it for it in items2[0] if it[0] == "翻鸡"]
    assert f2 and f2[0][2] == 1.0, items2[0]
    print("ok 满堂鸡")


def test_kaiju_fanji():
    # 开局翻鸡默认关闭：不翻牌、不新增鸡牌种
    g0 = DushanGame(DushanConfig(), seed=16)
    assert g0.cfg.kaiju_fanji == "off"
    assert g0.kaiju_flip is None
    assert g0.ji_tiles == {YAOJI}

    # 关闭 → 上鸡：尾部翻指示牌，本局新增「上鸡」一种
    g = DushanGame(DushanConfig(kaiju_fanji="up"), seed=17)
    assert g.kaiju_flip is not None
    wall_left_now = len(g.wall) - g.wall_pos
    s = g.kaiju_flip // 9
    r = g.kaiju_flip % 9 + 1
    expect = {YAOJI, s * 9 + (r % 9)}      # 幺鸡 + 上鸡
    assert g.ji_tiles == expect, (g.ji_tiles, expect)
    # 2026-10-08 口径修正：开局翻的指示牌**只翻开查看，不移出牌墙**
    # （108 张发出 53，剩 55；翻牌后仍是 55，这张牌本局仍可能被摸到）
    assert wall_left_now == 55, wall_left_now
    assert g.wall[-1] == g.kaiju_flip      # 仍在牌墙尾部

    # 下鸡：只新增「下鸡」一种
    gd = DushanGame(DushanConfig(kaiju_fanji="down"), seed=17)
    sd = gd.kaiju_flip // 9
    rd = gd.kaiju_flip % 9 + 1
    assert gd.ji_tiles == {YAOJI, sd * 9 + ((rd - 2) % 9)}, gd.ji_tiles

    # 上下鸡：新增两张
    g2 = DushanGame(DushanConfig(kaiju_fanji="both"), seed=18)
    s2 = g2.kaiju_flip // 9
    r2 = g2.kaiju_flip % 9 + 1
    expect2 = {YAOJI, s2 * 9 + (r2 % 9), s2 * 9 + ((r2 - 2) % 9)}
    assert g2.ji_tiles == expect2, (g2.ji_tiles, expect2)
    # 兼容旧写法：True → 上下鸡
    assert DushanConfig(kaiju_fanji=True).kaiju_fanji == "both"

    # 新增鸡牌打出产生鸡事件（冲锋鸡/横鸡/幺鸡），手中鸡计数包含新增种
    from zhuoji.tiles import suit_of as _so, rank_of as _ro, tile_cn as _tc
    up = (g.kaiju_flip // 9) * 9 + ((g.kaiju_flip % 9 + 1) % 9)
    # 让玩家 0 先打一张非鸡牌，再打新增鸡牌 → 横鸡（若横鸡未开）
    g.first_discard_done[0] = False
    g.hands[0][tile_from(W, 5)] += 1
    g.hands[0][up] += 1
    g.current = 0
    g.phase = Phase.DISCARD
    g.drawn = True
    g._step_discard(Action(DISCARD, tile_from(W, 5)))
    g._begin_respond(0, tile_from(W, 5))
    # 强制进入下一轮出牌（简化：直接再次出牌）
    g.current = 0
    g.phase = Phase.DISCARD
    g._step_discard(Action(DISCARD, up))
    assert g.ji_events[-1][:2] == (0, JI_HENG), g.ji_events[-1]
    # 手中鸡计数：玩家 1 手里塞两张新增鸡牌
    g.hands[1][up] += 2
    c, items, _v, _t = g._chicken_counts([True, True, False, False], set(), 1.0)
    shou = [it for it in items[1] if it[0] == "手中鸡"]
    assert shou and shou[0][2] >= 2.0, items[1]
    print("ok 开局翻鸡")


# ---------------------------------------------------------------------------
# 终局听牌检测（tenpai 座位列表 + 非标准牌数容错）
# ---------------------------------------------------------------------------
def test_tenpai_result_seat_list():
    """胡牌/黄庄结算的 result['tenpai'] 必须是座位列表（int），不能是布尔列表。"""
    import random
    seen_types = set()
    for seed in range(400):
        if "win" in seen_types and "huangzhuang" in seen_types:
            break
        g = new_game(seed)
        rng = random.Random(seed)
        steps = 0
        while g.phase.name != "OVER" and steps < 4000:
            steps += 1
            acts = g.legal_actions()
            if not acts:
                break
            g.step(rng.choice(acts))
        if g.phase.name != "OVER" or not g.result:
            continue
        seen_types.add(g.result["type"])
        tp = g.result["tenpai"]
        assert all(isinstance(x, int) and not isinstance(x, bool) for x in tp), tp
        assert len(tp) == len(set(tp)), tp
        # 与逐座位鲁棒判听一致
        for p in range(4):
            assert (p in tp) == g._tenpai_flag(p), (seed, p, tp)
    assert "win" in seen_types and "huangzhuang" in seen_types
    print("ok 结算听牌为座位列表")


def test_ting_info_odd_shapes():
    """非标准牌数听牌：14 张胡牌形、杠后未补牌（9 张）按打张口径判听。"""
    g = new_game()
    # 14 张已胡形：123万 456万 789万 123条 55筒（已成胡，多摸状态）→ 打张后仍应判听
    base = [tile_from(W, r) for r in (1, 2, 3, 4, 5, 6, 7, 8, 9)] + \
           [tile_from(T, 1), tile_from(T, 2), tile_from(T, 3),
            tile_from(T, 5), tile_from(T, 5)]
    assert len(base) == 14
    g.hands[0] = counts(*base)
    waits, _types = g.ting_info(0)
    assert waits, "14 张胡牌形应判听"
    # 杠后未补牌（9 张 + 1 杠）：打张后仅 8 张 ≠ 标准 10 张 → 正确判未听
    g2 = new_game()
    h9 = [tile_from(W, r) for r in (1, 2, 3, 4, 5, 6, 7, 8, 9)]
    g2.hands[0] = counts(*h9)
    g2.melds[0] = [(MELD_KONG_CONCEALED, tile_from(B, 1), 1)]
    waits2, _t2 = g2.ting_info(0)
    assert waits2 == [], waits2
    print("ok 非标准牌数听牌容错")


def test_mask_matches_legal_actions():
    """模型动作掩码必须与引擎合法动作完全一致（否则网页端会提交非法动作卡死）。

    重点回归：碰牌后 ``drawn=False``，即便手牌已成胡也不得给出「自摸胡」
    （历史上 encoder 漏判 drawn，导致模型推荐 hu → “非法动作 hu:-1” 死循环）。
    """
    from zhuoji.encoder import action_head_and_mask, index_to_action
    from zhuoji.bots import RandomBot

    def mask_set(game, me):
        head, mask = action_head_and_mask(game, me)
        return {(index_to_action(head, i, game).kind,
                 int(index_to_action(head, i, game).tile))
                for i in range(len(mask)) if mask[i]}

    # 1) 定点复现：我 13 张听 2万/9条 且无通行证 → 只能碰不能胡
    g = new_game(seed=3)
    hand0 = ([tile_from(W, 2)] * 2 + [tile_from(W, 3), tile_from(W, 4), tile_from(W, 5)]
             + [tile_from(T, 6), tile_from(T, 7), tile_from(T, 8)]
             + [tile_from(B, 1), tile_from(B, 2), tile_from(B, 3)]
             + [tile_from(B, 9)] * 2)
    g.hands[0] = counts(*hand0)
    assert not g._has_passport(0)
    g.hands[3] = counts(tile_from(W, 2))
    g.phase = Phase.DISCARD
    g.current = 3
    g.step(Action(DISCARD, tile_from(W, 2)))
    assert g.actor() == 0
    assert [a.kind for a in g.legal_actions()] == [PASS, "peng"], g.legal_actions()
    assert mask_set(g, 0) == {(PASS, -1), ("peng", tile_from(W, 2))}
    g.step(Action("peng", tile_from(W, 2)))
    assert g.can_tsumo(0), "碰后已成胡形（程序上）"
    assert not g.drawn, "碰后未摸牌"
    assert "hu" not in {a.kind for a in g.legal_actions()}, "碰后不可自摸胡"
    assert (DISCARD, tile_from(B, 9)) in mask_set(g, 0)      # 成胡形的对子仍可打
    try:
        g.step(Action(HU))
        raise AssertionError("碰后自摸胡应被引擎拒绝")
    except ValueError:
        pass

    # 2) 随机对局全流程一致性（含响应/闷豆/出牌三种决策点）
    checked = 0
    for seed in range(60):
        g2 = new_game(seed=1000 + seed)
        bot = RandomBot(seed=seed)
        for _ in range(400):
            if g2.phase == Phase.OVER:
                break
            me = g2.actor()
            legal = {(a.kind, int(a.tile)) for a in g2.legal_actions()}
            assert mask_set(g2, me) == legal, (seed, g2.phase, legal, mask_set(g2, me))
            checked += 1
            g2.step(bot.choose(g2))
    print(f"ok 动作掩码与合法动作一致（含碰后成胡回归，{checked} 个决策点）")


# ---------------------------------------------------------------------------
# 2026-10-08 回归：通行证逐张判定 / 分鸡种横鸡轮 / 赢家听张还原
# ---------------------------------------------------------------------------
def test_passport_per_tile():
    """混合听牌（4万=平胡 / 5万=大对子）无豆只能胡非平胡张（截图对局回归）。

    局面：副露 3 刻子 + 暗手 333万+5万，实际听 4万(成平胡)/5万(成大对子)。
    旧口径「任一等待张成非平胡 → 全手有通行证」导致无豆平胡点炮，已修。
    """
    g = new_game()
    base_hand = counts(*([tile_from(W, 3)] * 3 + [tile_from(W, 5)]))
    g.hands[0] = base_hand
    g.melds[0] = [(MELD_PONG, tile_from(B, 7), 1),
                  (MELD_PONG, tile_from(B, 8), 2),
                  (MELD_PONG, tile_from(B, 9), 3)]
    g.current = 0
    g.phase = Phase.DISCARD
    # 4万 → 平胡：无杠无报叫 → 不能点炮胡
    assert not g.can_ron(0, tile_from(W, 4))
    # 5万 → 大对子（非平胡）：无豆可点胡
    assert g.can_ron(0, tile_from(W, 5))
    # 有杠 → 通行证 → 平胡张也可胡
    g.melds[0][0] = (MELD_KONG_CONCEALED, tile_from(B, 7), 0)
    assert g.can_ron(0, tile_from(W, 4))
    # 热炮 → 平胡张也可胡
    g.melds[0] = [(MELD_PONG, tile_from(B, 7), 1),
                  (MELD_PONG, tile_from(B, 8), 2),
                  (MELD_PONG, tile_from(B, 9), 3)]
    g.repao_discard = True
    assert g.can_ron(0, tile_from(W, 4))
    g.repao_discard = False
    # 报叫玩家的弃牌 → 平胡张也可胡
    g.last_discard = (2, tile_from(W, 4))
    g.baojiao[2] = True
    assert g.can_ron(0, tile_from(W, 4))
    g.baojiao[2] = False
    g.last_discard = None
    print("ok 通行证逐张判定（平胡张需杠/报叫，非平胡张无需）")


def test_chicken_species_hengji():
    """分鸡种横鸡轮：幺鸡种与翻鸡种的横鸡各自独立，互不影响（轮内跟打算横鸡）。"""
    g = new_game()
    g.ji_tiles = {YAOJI, tile_from(B, 5)}     # 模拟开局翻鸡新增鸡种 5筒
    for p in range(4):
        g.hands[p] = [0] * 27
        for i, r in enumerate((2, 3, 4, 5, 6, 7)):
            g.hands[p][tile_from((p + i) % 3, r)] += 1
        g.hands[p][YAOJI] += 1
    g.hands[1][tile_from(B, 5)] += 1
    g.hands[3][tile_from(B, 5)] += 1

    def discard(p, tile):
        g.current = p
        g.phase = Phase.DISCARD
        g.step(Action("discard", tile))

    # p0 首张打幺鸡 → 冲锋鸡（幺鸡种）
    discard(0, YAOJI)
    assert g.ji_events[-1] == (0, JI_CHONGFENG, "冲锋鸡")
    # p1 首张打非鸡，再打 5筒（翻鸡种）→ 5筒种的横鸡（开 5筒轮）
    discard(1, tile_from(T, 2))
    discard(1, tile_from(B, 5))
    assert g.ji_events[-1] == (1, JI_HENG, "横鸡")
    assert g.hengji_species == [tile_from(B, 5)]
    # p2 首张打非鸡，再打幺鸡 → 幺鸡种的横鸡（分种轮：5筒开过横鸡不影响幺鸡种）
    discard(2, tile_from(B, 2))
    discard(2, YAOJI)
    assert g.ji_events[-1] == (2, JI_HENG, "横鸡")
    assert g.hengji_species == sorted([tile_from(B, 5), YAOJI])
    # p3 首张打非鸡，再打 5筒 → 仍在 5筒轮内跟打 → 横鸡
    discard(3, tile_from(W, 2))
    discard(3, tile_from(B, 5))
    assert g.ji_events[-1] == (3, JI_HENG, "横鸡")
    # p0 打幺鸡 → 仍在幺鸡轮内（p0 此前打的是冲锋鸡，未开轮）→ 横鸡
    discard(0, YAOJI)
    assert g.ji_events[-1] == (0, JI_HENG, "横鸡")
    # p2（幺鸡轮开轮者）再出牌 → 两轮分别收口（p2 也曾在 5筒轮…未参与，只收幺鸡轮）
    discard(2, tile_from(W, 2))
    assert g.hengji_species == [tile_from(B, 5)] and g.hengji_active
    # 轮外再打幺鸡 → 普通鸡
    discard(2, YAOJI)
    assert g.ji_events[-1] == (2, JI_YAO, "幺鸡")
    # p3（5筒轮参与者）再出牌 → 5筒轮收口，此手打 5筒 按普通鸡计
    discard(3, tile_from(B, 5))
    assert g.ji_events[-1] == (3, JI_YAO, "幺鸡")
    assert not g.hengji_active and g.hengji_opened
    print("ok 分鸡种横鸡轮（每张鸡牌独立成轮，轮内跟打算横鸡）")


def test_hengji_follow_discard():
    """横鸡轮跟打（2026-10-09 用户口径）：轮内其他家跟打同种鸡同样记横鸡。"""
    g = new_game()
    g.ji_tiles = {YAOJI}
    for p in range(4):
        g.hands[p] = [0] * 27
        for i, r in enumerate((2, 3, 4, 5, 6, 7)):
            g.hands[p][tile_from((p + i) % 3, r)] += 1
        g.hands[p][YAOJI] += 2

    def discard(p, tile):
        g.current = p
        g.phase = Phase.DISCARD
        g.step(Action("discard", tile))

    def last(p):
        assert g.ji_events and g.ji_events[-1][0] == p, g.ji_events[-3:]
        return g.ji_events[-1][2]

    # p0 首张非鸡（用掉首张），再打幺鸡 → 开轮，横鸡
    discard(0, tile_from(T, 2))
    discard(0, YAOJI)
    assert last(0) == "横鸡" and g.hengji_active
    # p1 首张非鸡 → 跟打幺鸡 → 横鸡（旧口径会误记普通鸡）
    discard(1, tile_from(T, 3))
    discard(1, YAOJI)
    assert last(1) == "横鸡"
    # p2 首张非鸡 → 跟打幺鸡 → 横鸡
    discard(2, tile_from(T, 4))
    discard(2, YAOJI)
    assert last(2) == "横鸡"
    # p1 已在本轮吃过横鸡，再次出牌 → 收口；此手打幺鸡按普通鸡计
    discard(1, YAOJI)
    assert last(1) == "幺鸡" and not g.hengji_active
    # 轮外：p3 先打非鸡（用掉首张），再打幺鸡 → 普通鸡
    discard(3, tile_from(T, 5))
    discard(3, YAOJI)
    assert last(3) == "幺鸡"
    # 同一牌种本局不再开新轮：p0 再打幺鸡 → 普通鸡
    discard(0, YAOJI)
    assert last(0) == "幺鸡"
    print("ok 横鸡轮同轮跟打（跟打算横鸡、收口后算普通鸡）")


def test_ting_info_drop_winner():
    """赢家 14 张去掉胡牌张后还原真实等待张。

    回归：333万+5万 胡 4万 后，14 张手牌按「打张后兜底」会误报 听3万/6万；
    drop=胡牌张 后应还原为 听4万/5万。
    """
    g = new_game()
    g.hands[0] = counts(*([tile_from(W, 3)] * 3
                          + [tile_from(W, 4), tile_from(W, 5)]))
    g.melds[0] = [(MELD_PONG, tile_from(B, 7), 1),
                  (MELD_PONG, tile_from(B, 8), 2),
                  (MELD_PONG, tile_from(B, 9), 3)]
    waits, _types = g.ting_info(0, drop=tile_from(W, 4))
    assert waits == [tile_from(W, 4), tile_from(W, 5)], waits
    # 未带 drop（旧调用方）行为保持不变：非标准牌数走打张后兜底
    waits_old, _ = g.ting_info(0)
    assert waits_old != [tile_from(W, 4), tile_from(W, 5)]
    print("ok 赢家听张还原（ting_info drop）")


# ---------------------------------------------------------------------------
# 点炮牌归属 / 一炮多响 / 抢杠 / 热炮全烧
# ---------------------------------------------------------------------------
def _clear(g: DushanGame) -> None:
    for p in range(4):
        g.hands[p] = [0] * 27
        g.melds[p] = []


def _wait_T1_hand(other_suit: int) -> list[int]:
    """一副听 1条（T1）的 13 张手牌：T2..T9 + 其他花色 1×3、5×2。"""
    return counts(*[tile_from(T, r) for r in (2, 3, 4, 5, 6, 7, 8, 9)],
                  *([tile_from(other_suit, 1)] * 3 + [tile_from(other_suit, 5)] * 2))


def test_dianpao_tile_transfer():
    """点炮张归胡牌者：从打出者弃牌区移出、并入赢家手牌；鸡分转给赢家。"""
    g = new_game()
    _clear(g)
    # p0：手里有一张 1条（首张打出 → 冲锋鸡）+ 一副听 1条 的牌
    g.hands[0] = counts(*[tile_from(T, r) for r in (2, 3, 4, 5, 6, 7, 8, 9)],
                        *([tile_from(W, 1)] * 3 + [tile_from(W, 5)] * 2), YAOJI)
    g.baojiao[0] = True            # 报叫家弃牌 → 他人无通行证也可胡
    # p1：听 1条（T2..T9 + 条 111/55）
    g.hands[1] = _wait_T1_hand(B)
    for p in (2, 3):
        g.hands[p] = counts(tile_from(W, 2), tile_from(W, 4), tile_from(W, 6))
    g.wall[g.wall_pos] = tile_from(B, 7)      # 翻鸡牌无人持有，排除翻鸡干扰
    g.current = 0
    g.phase = Phase.DISCARD
    g.drawn = True
    g.step(Action(DISCARD, YAOJI))
    assert g.ji_events[-1] == (0, JI_CHONGFENG, "冲锋鸡")
    assert g.can_ron(1, YAOJI)
    g.step(Action(HU))
    res = g.result
    assert res["type"] == "win" and res["winner"] == 1, res
    # 1) 点炮张从打出者弃牌区移出
    assert YAOJI not in g.discards[0], g.discards[0]
    # 2) 点炮张并入胡牌者手牌
    assert g.hands[1][YAOJI] == 1
    # 3) 捉炮鸡：冲锋鸡 3 分记到赢家
    cap = [d for d in res["detail"] if d[1] == "捉炮鸡"]
    assert cap == [(1, "捉炮鸡", "冲锋鸡·一条", 3.0)], cap
    # 4) 打出者不再计这张冲锋鸡
    assert not any(d[0] == 0 and d[1] == "冲锋鸡" for d in res["detail"])
    # 5) 鸡数池：打出者 0、赢家 3（原先归打出者的冲锋鸡转给了赢家）
    assert res["chickens"] == [0.0, 3.0, 0.0, 0.0], res["chickens"]
    # 6) p0 向赢家付：点炮 3 + 捉炮鸡 3 = 6
    assert abs(res["pair"][1][0] - 6.0) < 1e-9, res["pair"]
    assert abs(sum(res["deltas"])) < 1e-9
    print("ok 点炮牌归属（弃牌区移出 + 并入赢家手牌 + 捉炮鸡转移）")


def test_multi_ron():
    """一炮多响：一张牌同时点炮两名听牌者，各赢家均得牌、各得赔付与捉炮鸡。"""
    g = new_game()
    _clear(g)
    g.hands[0] = counts(*[tile_from(T, r) for r in (2, 3, 4, 5, 6, 7, 8, 9)],
                        *([tile_from(W, 1)] * 3 + [tile_from(W, 5)] * 2), YAOJI)
    g.baojiao[0] = True
    g.hands[1] = _wait_T1_hand(B)          # 听 1条
    g.hands[2] = _wait_T1_hand(W)          # 也听 1条
    g.hands[3] = counts(tile_from(B, 2), tile_from(B, 4), tile_from(B, 6))
    g.wall[g.wall_pos] = tile_from(B, 7)
    g.current = 0
    g.phase = Phase.DISCARD
    g.drawn = True
    g.step(Action(DISCARD, YAOJI))
    # 两名可胡者依次表态
    assert g.phase == Phase.RESPOND
    g.step(Action(HU))
    assert g.phase == Phase.RESPOND, "第二名可胡者尚未表态就结算了"
    g.step(Action(HU))
    res = g.result
    assert res["type"] == "win", res
    assert res["winners"] == [1, 2] and res["winner"] == 1, res
    # 每位赢家都拿到这张牌
    assert g.hands[1][YAOJI] == 1 and g.hands[2][YAOJI] == 1
    assert YAOJI not in g.discards[0]
    # 两名赢家各转移一份捉炮鸡
    caps = sorted((d[0], d[3]) for d in res["detail"] if d[1] == "捉炮鸡")
    assert caps == [(1, 3.0), (2, 3.0)], caps
    # 鸡数池：打出者 0，两名赢家各 3
    assert res["chickens"] == [0.0, 3.0, 3.0, 0.0], res["chickens"]
    # 点炮者向两家各付：点炮 3 + 捉炮鸡 3 = 6 → 共 12
    assert abs(res["pair"][1][0] - 6.0) < 1e-9, res["pair"]
    assert abs(res["pair"][2][0] - 6.0) < 1e-9, res["pair"]
    assert abs(res["deltas"][0] + 12.0) < 1e-9, res["deltas"]
    assert abs(sum(res["deltas"])) < 1e-9
    print("ok 一炮多响（多赢家各得牌 + 各自赔付 + 捉炮鸡）")


def _wait_T5_hand() -> list[int]:
    """一副卡 5条（T4_T6）听牌的 13 张手牌（不含幺鸡，排除手中鸡干扰）。"""
    h = [0] * 27
    for r in (4, 6, 7, 8, 9):
        h[tile_from(T, r)] = 1
    h[tile_from(W, 1)] = 3
    h[tile_from(W, 5)] = 2
    h[tile_from(B, 1)] = 3
    return h


def test_qiang_gang():
    """抢杠胡：补杠被听该张者抢胡（无视通行证）；被抢者鸡分全烧。"""
    g = new_game()
    _clear(g)
    # p0：碰 5条 + 手牌 11 张（含补杠用的第 4 张 5条；抢杠后剩 10 张=听 T9）
    g.melds[0] = [(MELD_PONG, tile_from(T, 5), 2)]
    g.hands[0] = counts(*[tile_from(W, r) for r in (1, 2, 3, 4, 5, 6, 7, 8, 9)],
                        tile_from(T, 9), tile_from(T, 5))
    g.ji_events.append((0, JI_CHONGFENG, "冲锋鸡"))
    g.hands[1] = _wait_T5_hand()          # 卡 5条 听牌
    for p in (2, 3):
        g.hands[p] = counts(tile_from(B, 2), tile_from(B, 4), tile_from(B, 6))
    g.wall[g.wall_pos] = tile_from(B, 7)
    g.current = 0
    g.phase = Phase.SELF_KONG
    g.drawn = True
    g.step(Action("bugang", tile_from(T, 5)))
    assert g._rob_kong is not None and g.phase == Phase.SELF_KONG, g._rob_kong
    assert g.actor() == 1
    assert g.can_ron(1, tile_from(T, 5)), "抢杠应无视通行证"
    g.step(Action(HU))
    res = g.result
    assert res["type"] == "win" and res["winner"] == 1, res
    assert res["rob_kong"] is True and res["how"] == "抢杠", res.get("how")
    assert res["void_player"] == 0
    # 被抢的那张从被抢者手中移出、并入抢杠者手牌
    assert g.hands[0][tile_from(T, 5)] == 0
    assert g.hands[1][tile_from(T, 5)] == 1
    # 被抢者碰牌保持为碰（补杠未生效）
    assert g.melds[0] == [(MELD_PONG, tile_from(T, 5), 2)]
    # 全烧：被抢杠者打出的冲锋鸡作废（鸡数池为 0），抢杠者不再倒付给他
    assert [d for d in res["detail"] if d[1] == "全烧"]
    assert res["chickens"] == [0.0, 0.0, 0.0, 0.0], res["chickens"]
    assert abs(res["deltas"][1] - 3.0) < 1e-9, res["deltas"]   # 仅得抢杠 3
    assert abs(res["deltas"][0] + 3.0) < 1e-9, res["deltas"]
    print("ok 抢杠胡（无视通行证 + 被抢者鸡分全烧）")


def test_repao_burn():
    """热炮：杠后第一张被胡 → 无通行证可胡；放热炮者鸡分全烧。"""
    g = new_game()
    _clear(g)
    # p0：有一副杠（有通行证）；杠后打出的第一张为 5条
    g.melds[0] = [(MELD_KONG_CONCEALED, tile_from(B, 9), 0)]
    g.hands[0] = counts(*[tile_from(W, r) for r in (1, 2, 3, 4, 5, 6, 7, 8, 9)],
                        tile_from(T, 9), tile_from(T, 5))
    g.ji_events.append((0, JI_CHONGFENG, "冲锋鸡"))
    g.hands[1] = _wait_T5_hand()          # 卡 5条 听牌
    for p in (2, 3):
        g.hands[p] = counts(tile_from(B, 2), tile_from(B, 4), tile_from(B, 6))
    g.wall[g.wall_pos] = tile_from(B, 7)
    g.current = 0
    g.phase = Phase.DISCARD
    g.drawn = True
    g.gang_discard[0] = True          # 本次弃牌来自杠后 → 热炮
    g.step(Action(DISCARD, tile_from(T, 5)))
    assert g.repao_discard is True
    assert g.can_ron(1, tile_from(T, 5)), "热炮应无视通行证"
    g.step(Action(HU))
    res = g.result
    assert res["type"] == "win" and res["how"] == "热炮", res.get("how")
    assert res["void_player"] == 0
    # 全烧：放热炮者的冲锋鸡与杠分（暗杠）全部作废
    assert [d for d in res["detail"] if d[1] == "全烧"]
    assert res["chickens"] == [0.0, 0.0, 0.0, 0.0], res["chickens"]
    # 全烧者的暗杠杠分也作废
    assert not any(d[0] == 0 and d[1] == "杠分" for d in res["detail"])
    assert abs(res["deltas"][1] - 5.0) < 1e-9, res["deltas"]   # 仅得热炮 5
    assert abs(res["deltas"][0] + 5.0) < 1e-9, res["deltas"]
    assert abs(sum(res["deltas"])) < 1e-9
    print("ok 热炮（无视通行证 + 放热炮者鸡分全烧）")


def test_random_hands_invariants():
    """随机对局不变量：零和、赢家必在听牌集合、牌张守恒。

    牌张守恒：各家手牌 + 副露 + 弃牌 + 未摸牌墙 = 108；一炮多响每多一名赢家 +1
    （现实只有一张，游戏中该张同时并入各赢家手牌）。
    """
    from zhuoji.bots import RandomBot
    n_multi = 0
    for seed in range(4000, 4120):
        bots = [RandomBot(seed=seed * 5 + i) for i in range(4)]
        g = DushanGame(seed=seed)
        res = g.play(bots)
        assert res["type"] != "timeout", f"seed={seed} 超时"
        assert abs(sum(res["deltas"])) < 1e-6, f"seed={seed} 非零和"
        n_extra = 0
        if res["type"] == "win":
            ws = res.get("winners") or [res["winner"]]
            assert res["winner"] == ws[0]
            n_extra = len(ws) - 1
            if len(ws) > 1:
                n_multi += 1
            for w in ws:
                assert w in res["tenpai"], f"seed={seed} 赢家 {w} 不在听牌集合"
        total = sum(sum(g.hands[p]) for p in range(4))
        for p in range(4):
            for mt, _t, _s in g.melds[p]:
                total += 4 if meld_is_kong(mt) else 3
        total += sum(len(g.discards[p]) for p in range(4))
        total += g.wall_left
        assert total == 108 + n_extra, f"seed={seed} 牌张守恒失败: {total}"
    print(f"ok 随机 120 局不变量（零和/听牌/牌张守恒；含一炮多响 {n_multi} 局）")


# ---------------------------------------------------------------------------
# 2026-10-08 用户口径修正（教程 vs 实现对齐）
# ---------------------------------------------------------------------------
def _fixed_tenpai(g, flags: dict[int, bool]) -> None:
    """把听牌状态钉死（避免依赖随机牌型）：flags[seat] = 是否听牌。"""
    g._tenpai_flag = lambda p, _f=dict(flags): bool(_f[int(p)])  # type: ignore[assignment]


def test_plain_ji_counts():
    """打出的普通鸡也算 1 只（原来记 0）。"""
    g = new_game()
    _setup_plain_hands(g)
    _fixed_tenpai(g, {0: True, 1: True, 2: False, 3: False})
    g.ji_events.append((1, g.ji_value("幺鸡"), "幺鸡"))   # p1 打出过一张普通鸡
    g._settle_win(0, -1, is_tsumo=True)
    res = g.result
    assert abs(res["chickens"][1] - 1.0) < 1e-9, res["chickens"]
    # p1（1 只）比 p0（0 只）多 → p0 付 p1 1
    assert abs(res["pair"][1][0] - 1.0) < 1e-9, res["pair"]
    print("ok 打出的普通鸡计 1（2026-10-08 修正）")


def test_claimed_ji_pair_totals():
    """碰走打出的鸡：AB 之间合计 = 面 + 溢价；未听牌包鸡时方向反转但合计不变。

    用户例：A 打出横鸡、B 碰走（3 张）；B 听牌 → A 付 B 4（面 3 + 溢价 1），
    其他家付 3；B 未听牌 → B 包鸡，其他家得 3、A 得 3+1=4。
    """
    HE = tile_from(W, 3)

    def build(b_tenpai: bool):
        g = new_game()
        _setup_plain_hands(g)
        g.ji_tiles = {YAOJI, HE}
        _fixed_tenpai(g, {0: b_tenpai, 1: True, 2: True, 3: True})
        # 翻鸡牌种与幺鸡、HE 一起避开，手牌里不放，保证鸡数只来自碰来的 3 张
        _gold, fj = g._dushan_chicken_state()
        banned = set(fj) | {int(YAOJI), int(HE)}
        pool = [t for t in range(27) if t not in banned]
        # A = p1 打出横鸡 → B = p0 碰走（3 张牌面）
        g.ji_events.append((1, g.ji_value("横鸡"), "横鸡"))
        g.claimed_ji.append((1, 0, "横鸡", False))
        g.melds[0] = [(MELD_PONG, HE, 1)]
        g.melds[1] = g.melds[2] = g.melds[3] = []

        def fill(seat: int, n: int) -> None:
            h = [0] * 27
            i = 0
            while sum(h) < n:
                h[pool[i % len(pool)]] += 1
                i += 1
            g.hands[seat] = h

        fill(0, 11)          # B 手里 11 张 + 碰 3 张
        fill(1, 12)
        fill(2, 13)          # 赢家（牌面随意，本用例只验 AB 两两合计）
        fill(3, 12)
        return g

    # ① B（p0）听牌、p2 收尾 → p0 从 p1 收 4（面 3 + 溢价 1），从 p2/p3 各收 3
    g = build(True)
    g._settle_win(2, -1, is_tsumo=True)
    pair = g.result["pair"]
    assert abs(g.result["chickens"][0] - 3.0) < 1e-9, g.result["chickens"]
    assert abs(pair[0][1] - 4.0) < 1e-9, pair
    assert abs(pair[0][2] - 3.0) < 1e-9 and abs(pair[0][3] - 3.0) < 1e-9, pair
    # ② B（p0）未听牌 → 包鸡：p0 付 p1 4（面 3 + 溢价 1）、付 p2/p3 各 3
    g2 = build(False)
    g2._settle_win(2, -1, is_tsumo=True)
    p2 = g2.result["pair"]
    assert abs(p2[1][0] - 4.0) < 1e-9, p2          # A 得 4
    assert abs(p2[3][0] - 3.0) < 1e-9, p2          # 其他家得 3
    assert abs(p2[0][1] - 0.0) < 1e-9, p2          # 反向无残留
    print("ok 碰走鸡的 AB 合计（听牌 4 / 包鸡仍 4，其他家 3）")


def test_long_qi_dui_needs_winning_tile():
    """龙七对必须由胡牌张补上第四张；胡别的对子只能算小七对。"""
    g = new_game()
    quad = tile_from(W, 1)
    h = counts(*([quad] * 4), *([tile_from(T, 2)] * 2), *([tile_from(T, 4)] * 2),
               *([tile_from(B, 6)] * 2), *([tile_from(B, 9)] * 2),
               *([tile_from(T, 7)] * 2))
    assert sum(h) == 14 and is_seven_pairs(h)
    assert g._win_type_set(h, [], quad) == {"long_qi_dui"}, g._win_type_set(h, [], quad)
    other = tile_from(T, 7)
    assert g._win_type_set(h, [], other) == {"qi_dui"}, g._win_type_set(h, [], other)
    # 只有牌面（不知道胡哪张）时保留宽松口径，供听牌分析
    assert "long_qi_dui" in g._win_type_set(h, [])
    print("ok 龙七对必须由胡牌张凑成第四张")


def test_ji_values_configurable():
    """鸡的分值全部走配置（网页设置可调）。"""
    cfg = DushanConfig(ji_hold=3.0, ji_discard_plain=2.0, ji_chongfeng=7.0,
                       ji_heng=5.0, ji_fanji=4.0)
    g = DushanGame(cfg, seed=19)
    assert g.ji_value("冲锋鸡") == 7.0 and g.ji_value("横鸡") == 5.0
    assert g.ji_value("幺鸡") == 2.0
    # 溢价 = 打出的值 − 手中鸡单值
    g.ji_events.append((1, 5.0, "横鸡"))
    g.claimed_ji.append((1, 0, "横鸡", False))
    rows = g._claimed_ji_rows()
    assert rows and rows[0][0] == 0 and abs(rows[0][2] - (5.0 - 3.0)) < 1e-9, rows
    # 打出的普通鸡改用配置值
    g2 = new_game()
    _setup_plain_hands(g2)
    _fixed_tenpai(g2, {0: True, 1: True, 2: False, 3: False})
    g2.cfg.ji_discard_plain = 4.0
    g2.ji_events.append((1, g2.ji_value("幺鸡"), "幺鸡"))
    g2._settle_win(0, -1, is_tsumo=True)
    assert abs(g2.result["chickens"][1] - 4.0) < 1e-9, g2.result["chickens"]
    print("ok 鸡分值可配置（含溢价随配置变化）")


if __name__ == "__main__":
    test_win_types()
    test_passport()
    test_passport_per_tile()
    test_baojiao()
    test_repao()
    test_chicken_discard_events()
    test_chicken_species_hengji()
    test_hengji_follow_discard()
    test_ting_info_drop_winner()
    test_baojiao_auto()
    test_tsumo_payment()
    test_zeeren_ji()
    test_type_bonus_pairwise()
    test_kong_and_ron_settlement()
    test_huangzhuang_baoji()
    test_huang_defaults_no_bao()
    test_end_baoji_baogang_switches()
    test_huang_baodapai()
    test_huang_new_rules()
    test_teachers_and_zero_sum()
    test_jiesuan_fanji()
    test_mantiangji()
    test_kaiju_fanji()
    test_tenpai_result_seat_list()
    test_ting_info_odd_shapes()
    test_mask_matches_legal_actions()
    test_dianpao_tile_transfer()
    test_multi_ron()
    test_qiang_gang()
    test_plain_ji_counts()
    test_claimed_ji_pair_totals()
    test_long_qi_dui_needs_winning_tile()
    test_ji_values_configurable()
    test_repao_burn()
    test_random_hands_invariants()
    print("\n全部独山规则测试通过 ✅")
