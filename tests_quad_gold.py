# -*- coding: utf-8 -*-
"""四倍金鸡（2026-11-04 用户口径）快速验证：
开局翻鸡翻到九条 + 结算翻鸡再翻出幺鸡（九条/二条，按结算翻鸡模式算）
→ 幺鸡升四倍金鸡（相关分 = 金鸡倍率 ×2，默认档 4 倍），result.quad_gold_tiles=[9]。
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from zhuoji.dushan import DushanConfig, DushanGame, fanji_tiles_from
from zhuoji.tiles import YAOJI, tile_cn

JIU_TIAO = 17   # 九条
ER_TIAO = 10    # 二条
BA_TIAO = 16    # 八条


def make_game(kaiju_flip):
    g = DushanGame(DushanConfig(kaiju_fanji="both", jiesuan_fanji="both"), seed=7)
    # 强开开局翻鸡：九条 → 开局翻鸡种 = {幺鸡, 八条}
    g.kaiju_flip = kaiju_flip
    g.ji_tiles = {YAOJI} | set(fanji_tiles_from(kaiju_flip, "both"))
    # 强开结算翻鸡指示牌（不动 wall_left，只改 wall_pos 那一张）
    return g


def settle_with_flip(g, flip):
    assert g.wall_left > 0
    g.wall[g.wall_pos] = flip
    # 给庄家手里塞两张幺鸡便于核对倍率（手中鸡 1 分/张 × 倍率）
    g.hands[0][YAOJI] += 2
    g._settle_win_multi([0], -1, True)
    return g.result


def check(name, cond):
    print(("PASS  " if cond else "FAIL  ") + name)
    return cond


ok = True

# ① 开局九条 + 结算九条 → 幺鸡四倍金鸡
g = make_game(JIU_TIAO)
res = settle_with_flip(g, JIU_TIAO)
ok &= check("开局九条+结算九条: quad=[幺鸡]",
            res["quad_gold_tiles"] == [YAOJI])
hold = [d for d in res["detail"] if d[1] == "手中鸡" and d[0] == 0]
# 手中鸡 2 张 ×(2×2)=8（ji_hold=1）
ok &= check("开局九条+结算九条: 手中幺鸡×4倍 (2张=8分)",
            hold and abs(hold[0][3] - 8.0) < 1e-6)
ok &= check("开局九条+结算九条: gold 含幺鸡与八条",
            set(res["gold_tiles"]) == {YAOJI, BA_TIAO})
ok &= check("翻鸡 detail 标注四倍金鸡",
            any("四倍金鸡" in str(d[2]) for d in res["detail"] if d[1] == "翻鸡"))

# ② 开局九条 + 结算二条 → 翻鸡种 {三条, 幺鸡} ∩ 开局鸡牌 {幺鸡,八条} = {幺鸡} → 四倍
g = make_game(JIU_TIAO)
res = settle_with_flip(g, ER_TIAO)
ok &= check("开局九条+结算二条: quad=[幺鸡]",
            res["quad_gold_tiles"] == [YAOJI])
hold = [d for d in res["detail"] if d[1] == "手中鸡" and d[0] == 0]
ok &= check("开局九条+结算二条: 手中幺鸡×4倍 (2张=8分)",
            hold and abs(hold[0][3] - 8.0) < 1e-6)
ok &= check("开局九条+结算二条: gold 只有幺鸡",
            res["gold_tiles"] == [YAOJI])

# ③ 对照（2026-11-04 用户口径修订）：开局九条 + 结算七条（翻鸡种 {八条,六条}，
#    无幺鸡）→ **开局已知金鸡仍按金鸡 ×2 结算**。旧口径只在结算翻鸡再次翻出
#    幺鸡时才认金鸡，正是用户实测「开局翻到金鸡、结算却没按金鸡算」的 bug；
#    但结算没有「再次翻出幺鸡」，所以不升级四倍金鸡。
g = make_game(JIU_TIAO)
res = settle_with_flip(g, 15)
ok &= check("开局九条+结算七条: quad=[]（金鸡不升级四倍）",
            res["quad_gold_tiles"] == [])
ok &= check("开局九条+结算七条: gold 含开局金鸡幺鸡 + 结算翻鸡八条",
            set(res["gold_tiles"]) == {YAOJI, BA_TIAO})
hold = [d for d in res["detail"] if d[1] == "手中鸡" and d[0] == 0]
ok &= check("开局九条+结算七条: 手中幺鸡×2倍 (2张=4分)",
            hold and abs(hold[0][3] - 4.0) < 1e-6)

# ④ 对照：开局八条（开局翻鸡种 {七条,九条}，无幺鸡）+ 结算九条 → 金鸡但不四倍
g = make_game(BA_TIAO)
res = settle_with_flip(g, JIU_TIAO)
ok &= check("开局八条+结算九条: quad=[]（金鸡不升级）",
            res["quad_gold_tiles"] == [] and YAOJI in res["gold_tiles"])
hold = [d for d in res["detail"] if d[1] == "手中鸡" and d[0] == 0]
ok &= check("开局八条+结算九条: 手中幺鸡×2倍 (2张=4分)",
            hold and abs(hold[0][3] - 4.0) < 1e-6)

# ⑤ 对照：关闭开局翻鸡 + 结算九条 → 金鸡 ×2，无四倍
g = DushanGame(DushanConfig(kaiju_fanji="off", jiesuan_fanji="both"), seed=7)
res = settle_with_flip(g, JIU_TIAO)
ok &= check("无开局翻鸡+结算九条: quad=[]",
            res["quad_gold_tiles"] == [] and YAOJI in res["gold_tiles"])

# ⑥ 打出的幺鸡（冲锋鸡 3 分）四倍 → 12 分
g = make_game(JIU_TIAO)
g.ji_events.append((0, 3.0, "冲锋鸡"))
g.ji_event_tiles.append(YAOJI)
res = settle_with_flip(g, JIU_TIAO)
disc = [d for d in res["detail"] if d[1] == "冲锋鸡" and d[0] == 0]
ok &= check("打出冲锋幺鸡四倍 (3×4=12分)",
            disc and abs(disc[0][3] - 12.0) < 1e-6)

# ⑦ 局中 gold_known（服务端 state.gold_known）与规则引擎结算口径**同源**：
#    开局九条 → 开局已知金鸡 = {幺鸡}，局内标记与结算金鸡不会各说各话。
g = make_game(JIU_TIAO)
kaiji = fanji_tiles_from(g.kaiju_flip, "both")
ok &= check("开局九条: 开局翻鸡种含幺鸡（gold_known 依据）", YAOJI in kaiji)
ok &= check("开局九条: kaiju_gold_tiles()=[幺鸡]（与 gold_known 同源）",
            set(g.kaiju_gold_tiles()) == {YAOJI})
g8 = make_game(BA_TIAO)
ok &= check("开局八条: kaiju_gold_tiles()=[]（非金鸡）",
            set(g8.kaiju_gold_tiles()) == set())

# ⑧ 结算翻鸡关闭时，开局已知金鸡依然必须按金鸡结算
g9 = DushanGame(DushanConfig(kaiju_fanji="both", jiesuan_fanji="off"), seed=7)
g9.kaiju_flip = JIU_TIAO
g9.ji_tiles = {YAOJI} | set(fanji_tiles_from(JIU_TIAO, "both"))
g9.hands[0][YAOJI] += 2
g9._settle_win_multi([0], -1, True)
res9 = g9.result
ok &= check("结算翻鸡 off + 开局九条: gold=[幺鸡]（开局金鸡不受结算翻鸡开关影响）",
            res9["gold_tiles"] == [YAOJI])
hold9 = [d for d in res9["detail"] if d[1] == "手中鸡" and d[0] == 0]
ok &= check("结算翻鸡 off + 开局九条: 手中幺鸡×2倍 (2张=4分)",
            hold9 and abs(hold9[0][3] - 4.0) < 1e-6)

print()
print("ALL PASS" if ok else "SOME FAILED")
sys.exit(0 if ok else 1)
