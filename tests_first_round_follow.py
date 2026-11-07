# -*- coding: utf-8 -*-
"""「第一巡同轮跟打」规则回归测试（2026-11-07 新规则）。

用户口径：第一巡庄家打的第一张牌被另外三名玩家**同轮跟打**（四家第一张
弃牌牌种相同）时：
  ① 庄家赔付每名玩家 3 只鸡；
  ② 这张被跟打的牌（4 张）视为庄家的通行证（可吃胡）；
  ③ 结算翻鸡计鸡时，这 4 张算作庄家手中的鸡。
开关 ``DushanConfig.first_round_follow`` 默认打开。

跑法：cd /Users/ligf9/zhuoji-mahjong-ai && .venv/bin/python tests_first_round_follow.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from zhuoji.dushan import DushanConfig, DushanGame
from zhuoji.rules import Action, DISCARD, Phase
from zhuoji.tiles import tile_cn

FOLLOW_TILE = 18          # 一筒（非鸡牌种，避免和幺鸡/翻鸡口径纠缠）
WIN_TILE = 22             # 五筒：下面摆的平胡单钓将

N_PASS = 0
N_FAIL = 0


def check(name: str, cond: bool, extra: str = "") -> bool:
    global N_PASS, N_FAIL
    ok = bool(cond)
    print(("PASS  " if ok else "FAIL  ") + name + (("  | " + extra) if extra else ""))
    if ok:
        N_PASS += 1
    else:
        N_FAIL += 1
    return ok


def new_game(follow: bool = True, seed: int = 7, dealer: int = 0) -> DushanGame:
    return DushanGame(DushanConfig(first_round_follow=follow), seed=seed, dealer=dealer)


def force_first_discards(g: DushanGame, tiles: list[int]) -> None:
    """按「庄家→下家→对家→上家」模拟四家各自的第一张弃牌。

    白盒：直接调 ``_step_discard`` 并临时屏蔽 ``_begin_respond``（响应阶段
    与本规则无关），这样能精确控制「全场前 4 次弃牌」而不被碰/杠/胡打断。
    每家手牌张数保持不变（从别的牌种挪一张过来）。
    """
    saved = g._begin_respond
    g._begin_respond = lambda p, t: None  # type: ignore[method-assign]
    try:
        for i, t in enumerate(tiles):
            p = (int(g.dealer) + i) % 4
            g.current = p
            g.phase = Phase.DISCARD
            g.drawn = True
            if g.hands[p][t] < 1:
                for x in range(27):
                    if x != t and g.hands[p][x] > 0:
                        g.hands[p][x] -= 1
                        g.hands[p][t] += 1
                        break
            g._step_discard(Action(DISCARD, t))
    finally:
        g._begin_respond = saved  # type: ignore[method-assign]


def flat_pinghu_hand() -> list[int]:
    """13 张平胡听牌手：万 123/456/789 + 条 123 + 单张五筒（听五筒做将）。

    无杠、无报叫、非平胡牌型为零 ⇒ 按既有口径**没有通行证**，正好用来
    单独验证「同轮跟打」带来的通行证。
    """
    h = [0] * 27
    for t in (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, WIN_TILE):
        h[t] += 1
    return h


def pair_rows(res: dict, label: str) -> list[dict]:
    out = []
    for row in res.get("pair_detail") or []:
        for it in row.get("items") or []:
            if it.get("label") == label:
                out.append({"a": row["a"], "b": row["b"], "value": it["value"]})
    return out


def main() -> int:
    # ------------------------------------------------------------------
    print("== 1. 触发判定 ==")
    g = new_game()
    force_first_discards(g, [FOLLOW_TILE] * 4)
    check("四家首张同牌 → 触发", g.follow_dealer is True,
          f"order={g.first_discard_order} tiles={g.first_discard_tiles}")
    check("记录被跟打牌种", g.follow_tile == FOLLOW_TILE,
          f"follow_tile={g.follow_tile} ({tile_cn(g.follow_tile)})")
    check("流水只记一次", sum(1 for _, s in g.log if s == "第一巡同轮跟打") == 1)
    check("总弃牌数=4（同轮）", g.total_discards == 4, f"{g.total_discards}")

    g2 = new_game()
    force_first_discards(g2, [FOLLOW_TILE, FOLLOW_TILE, FOLLOW_TILE, 19])
    check("有一家首张不同 → 不触发", g2.follow_dealer is False)

    g3 = new_game()
    g3.total_discards = 1          # 模拟首巡前已插入过一次非首张弃牌
    force_first_discards(g3, [FOLLOW_TILE] * 4)
    check("非连续第一巡（弃牌>4）→ 不触发", g3.follow_dealer is False,
          f"total_discards={g3.total_discards}")

    g4 = new_game(follow=False)
    force_first_discards(g4, [FOLLOW_TILE] * 4)
    check("开关关闭 → 不触发、无流水", g4.follow_dealer is False
          and not any(s == "第一巡同轮跟打" for _, s in g4.log))
    check("开关关闭 → 结算不带触发信息", not new_game(follow=False)._follow_result()["follow_dealer"])

    # 庄家不是第一个打：直接摆判定函数的前置状态
    g5 = new_game()
    g5.total_discards = 4
    g5.first_discard_order = [1, 2, 3, 0]
    g5.first_discard_tiles = [FOLLOW_TILE] * 4
    check("首张非庄家打出 → 不触发", g5._check_first_round_follow() is False)

    # ------------------------------------------------------------------
    print("== 2. 通行证（可吃胡）==")
    g6 = new_game()
    force_first_discards(g6, [FOLLOW_TILE] * 4)
    g6.baojiao = [False] * 4
    g6.melds = [[] for _ in range(4)]
    g6.hands[int(g6.dealer)] = flat_pinghu_hand()
    check("触发后庄家有通行证", g6._has_passport(int(g6.dealer)) is True)
    check("触发后庄家可吃平胡点炮", g6.can_ron(int(g6.dealer), WIN_TILE) is True)
    check("胡牌理由含「同轮跟打·通行证」",
          any("同轮跟打" in x for x in g6.ron_reasons(int(g6.dealer), WIN_TILE)))
    check("非庄家不因该规则获得通行证", g6._has_passport((int(g6.dealer) + 1) % 4) is False)

    g7 = new_game(follow=False)
    force_first_discards(g7, [FOLLOW_TILE] * 4)
    g7.baojiao = [False] * 4
    g7.melds = [[] for _ in range(4)]
    g7.hands[int(g7.dealer)] = flat_pinghu_hand()
    check("对照组：未触发时同一手牌无通行证、不能吃平胡",
          g7._has_passport(int(g7.dealer)) is False
          and g7.can_ron(int(g7.dealer), WIN_TILE) is False)

    # ------------------------------------------------------------------
    print("== 3. 结算：赔付 + 4 张手中鸡 + 零和 ==")
    cfg = DushanConfig(first_round_follow=True)
    check("配置默认打开", DushanConfig().first_round_follow is True)
    check("默认赔付 3 只鸡 / 4 张手中鸡",
          cfg.follow_pay == 3.0 and cfg.follow_hold_chickens == 4)

    gw = new_game()
    force_first_discards(gw, [FOLLOW_TILE] * 4)
    # 庄家作为赢家 → 必为听牌者，手中的 4 张跟打鸡应计入
    gw.wall[gw.wall_pos] = 25       # 固定结算翻鸡指示牌，避免随机金鸡干扰
    gw._settle_win_multi([int(gw.dealer)], -1, True)
    res = gw.result
    check("胡牌结算带触发信息", res.get("follow_dealer") is True
          and res.get("follow_tile") == FOLLOW_TILE)
    rows = pair_rows(res, "第一巡同轮跟打")
    check("庄家赔付另外三家各 3 鸡", len(rows) == 3
          and all(r["b"] == int(gw.dealer) and r["value"] == 3.0 for r in rows),
          f"rows={rows}")
    hold_labels = [d[1] for d in res.get("detail", []) if "同轮跟打鸡" in str(d[1])]
    check("结算明细含「同轮跟打鸡」4 张", "同轮跟打鸡" in hold_labels,
          f"labels={hold_labels[:8]}")
    hold_bd = (res.get("ji_hold_detail") or {}).get(str(int(gw.dealer)) or "0") or []
    check("手中鸡逐张明细含 4 张该牌",
          any(int(t) == FOLLOW_TILE and int(n) == 4 for t, n in hold_bd),
          f"hold_bd={hold_bd}")
    check("胡牌结算零和", abs(sum(res["deltas"])) < 1e-9,
          f"deltas={res['deltas']}")

    # 开关关闭：不应有赔付、也不该有跟打鸡
    gn = new_game(follow=False)
    force_first_discards(gn, [FOLLOW_TILE] * 4)
    gn.wall[gn.wall_pos] = 25
    gn._settle_win_multi([int(gn.dealer)], -1, True)
    check("开关关闭 → 无赔付明细", pair_rows(gn.result, "第一巡同轮跟打") == [])
    check("开关关闭 → 结算零和", abs(sum(gn.result["deltas"])) < 1e-9)

    # 黄庄：即时约定照赔，且 deltas 仍零和
    gh = new_game()
    force_first_discards(gh, [FOLLOW_TILE] * 4)
    gh._settle_huangzhuang()
    rh = gh.result
    check("黄庄也赔付三家各 3 鸡", len(pair_rows(rh, "第一巡同轮跟打")) == 3)
    check("黄庄结算带触发信息", rh.get("follow_dealer") is True)
    check("黄庄结算零和", abs(sum(rh["deltas"])) < 1e-9,
          f"deltas={rh['deltas']}")

    # ------------------------------------------------------------------
    print("== 4. 服务端开关传递 ==")
    try:
        from web.dushan_server import config_from_rules, RULE_DEFAULTS
        check("RULE_DEFAULTS 默认打开", RULE_DEFAULTS.get("first_round_follow") is True)
        check("config_from_rules 默认打开",
              config_from_rules({}).first_round_follow is True)
        check("config_from_rules 可关闭",
              config_from_rules({"first_round_follow": False}).first_round_follow is False)
    except Exception as e:  # pragma: no cover
        check("服务端开关传递", False, f"import/config 失败: {e!r}")

    print(f"\n{N_PASS} passed, {N_FAIL} failed")
    return 1 if N_FAIL else 0


if __name__ == "__main__":
    raise SystemExit(main())
