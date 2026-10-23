# -*- coding: utf-8 -*-
"""报叫碰杠锁压力测试（2026-10-12 移动端 bug 复现用）。

用户报告：报叫之后还能碰牌，且捉炮成功。规则口径：
  报叫后严禁碰/杠（暗杠/补杠/明杠/碰全禁），只能摸牌、胡牌、打牌；
  报叫是通行证——可吃胡别人打出的牌；普通玩家可无视通行证吃胡报叫玩家的牌。

本脚本跑 N 局随机/启发式混合对局，断言：
  A1) baojiao[p] 为 True 期间，p 从不打出 PENG/MINGGANG/ANGANG/BUGANG 动作；
  A2) baojiao[p] 为 True 期间，p 的副露数不增加（碰杠必增副露）；
  A3) 报叫玩家可以吃胡别人打出的牌（出现即记数，不强制出现）；
  A4) 普通玩家可以吃胡报叫玩家的弃牌（出现即记数）。
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from zhuoji.dushan import DushanGame, DushanConfig  # noqa: E402
from zhuoji.rules import (  # noqa: E402
    Action, PENG, MINGGANG, ANGANG, BUGANG, HU, DISCARD, PASS, Phase,
)
from zhuoji.bots import HeuristicBot, RandomBot, TEACHER_STYLES  # noqa: E402

KONG_KINDS = {PENG, MINGGANG, ANGANG, BUGANG}
STYLES = tuple(TEACHER_STYLES.keys())


def run_game(gi: int) -> dict:
    cfg = DushanConfig()
    game = DushanGame(cfg, seed=gi * 977 + 13)
    rng = __import__("random").Random(gi)
    bots = {p: (HeuristicBot(seed=gi * 31 + p, weights=list(TEACHER_STYLES.values())[p % 6])
                if gi % 2 == 0 else RandomBot(seed=gi * 17 + p))
            for p in range(1, 4)}
    stats = {"bao_peng": 0, "bao_meld_grow": 0, "bao_ron": 0, "ron_on_bao_discard": 0}
    prev_melds = [0, 0, 0, 0]

    while game.result is None:
        cur = game.actor()
        if game.baojiao[cur]:
            # A0（强判据）：报叫玩家的合法动作表里出现碰/杠即违规
            bad = [a.label() for a in game.legal_actions() if a.kind in KONG_KINDS]
            if bad:
                stats["bao_peng"] += len(bad)
                print(f"  !! game{gi} 报叫玩家 {cur} 的合法动作含碰杠: {bad}")
        before = list(game.baojiao)
        if cur == 0:
            acts = game.legal_actions()
            act = rng.choice(acts) if acts else Action(PASS)
        else:
            act = bots[cur].choose(game)

        # A1: 报叫玩家打出碰杠动作
        if before[cur] and act.kind in KONG_KINDS:
            stats["bao_peng"] += 1
            print(f"  !! game{gi} 报叫玩家 {cur} 打出 {act.label()}")

        # A3: 报叫玩家吃胡别人打出的牌（合法，计数）
        if act.kind == HU and before[cur] and not getattr(game, "drawn", False):
            stats["bao_ron"] += 1
        # A4: 普通玩家吃胡报叫玩家刚打出的牌（合法，计数）
        if act.kind == HU and not before[cur] and game.last_discard \
                and isinstance(game.last_discard, tuple) and before[game.last_discard[0]]:
            stats["ron_on_bao_discard"] += 1

        game.step(act)

        # A2: 报叫玩家副露增长（碰/杠必增副露；暗杠也增）
        for p in range(4):
            if before[p] and len(game.melds[p]) > prev_melds[p]:
                stats["bao_meld_grow"] += 1
                print(f"  !! game{gi} 报叫玩家 {p} 副露增长 "
                      f"{prev_melds[p]}->{len(game.melds[p])}")
            prev_melds[p] = len(game.melds[p])

        # A4 计数已移到 step 前（A3 旁）；此处仅清理旧逻辑占位
        if False:
            pass
    return stats


def guard_test() -> bool:
    """纵深防御验证：直接构造「报叫玩家为当前响应者」的 RESPOND 状态，
    把 PENG 动作喂给 step 链路，必须抛 ValueError。"""
    cfg = DushanConfig()
    g = DushanGame(cfg, seed=7)
    g.baojiao[0] = True
    tile = next(t for t in range(27) if g.hands[0][t] >= 2)
    g.phase = Phase.RESPOND
    g._respond_order = [0]
    g._respond_idx = 0
    g.last_discard = (1, tile)
    g._n_hu = 0
    try:
        g.step(Action(PENG, tile))   # 走完整 step 链路（含兜底硬闸）
    except ValueError as exc:
        print(f"  兜底硬闸生效：{exc}")
        return True
    print("  !! 兜底硬闸未生效：报叫玩家被静默碰牌")
    return False


def main() -> int:
    print("== 兜底硬闸验证 ==")
    if not guard_test():
        return 1
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 300
    agg = {"bao_peng": 0, "bao_meld_grow": 0, "bao_ron": 0, "ron_on_bao_discard": 0}
    done = 0
    for gi in range(n):
        try:
            s = run_game(gi)
        except Exception as exc:  # noqa: BLE001
            print(f"[FAIL] game{gi} 引擎抛异常：{type(exc).__name__}: {exc}")
            return 1
        for k in agg:
            agg[k] += s[k]
        done += 1
    print(f"\n跑完 {done} 局：")
    print(f"  A1 报叫玩家碰杠动作次数        = {agg['bao_peng']}   （必须 0）")
    print(f"  A2 报叫玩家副露增长次数        = {agg['bao_meld_grow']}   （必须 0）")
    print(f"  A3 报叫玩家吃胡（合法）出现    = {agg['bao_ron']}")
    print(f"  A4 普通玩家胡报叫者弃牌（合法）= {agg['ron_on_bao_discard']}")
    ok = agg["bao_peng"] == 0 and agg["bao_meld_grow"] == 0
    print(f"\n{'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
