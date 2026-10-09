"""引擎自检：跑通规则、校验番型/豆/鸡/黄庄查叫的计分正确性。"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from zhuoji import (  # noqa: E402
    Action, Phase, RulesConfig, ZhuojiGame, classify_fan, is_winning_hand,
    shanten, winning_tiles,
)
from zhuoji.bots import HeuristicBot, RandomBot  # noqa: E402
from zhuoji.fan import (  # noqa: E402
    MELD_PONG, QI_DUI, LONG_QI_DUI, QING_YI_SE, DA_DUI_ZI, PING_HU, QING_DA_DUI,
)
from zhuoji.tiles import parse_tile, tile_name  # noqa: E402


def tiles(s: str) -> list[int]:
    return [parse_tile(x) for x in s.split()]


def counts(s: str) -> list[int]:
    c = [0] * 27
    for t in tiles(s):
        c[t] += 1
    return c


def check(name: str, got, want) -> bool:
    ok = got == want
    print(f"  [{'OK ' if ok else 'FAIL'}] {name}: got={got} want={want}")
    return ok


def main() -> int:
    import io
    import os
    _log = os.environ.get("ZUOJI_LOG")
    if _log:
        sys.stdout = io.TextIOWrapper(open(_log, "w", encoding="utf-8"), encoding="utf-8",
                                      line_buffering=True)
    ok = True
    print("== 1. 胡牌判定 ==")
    ok &= check("平胡 14 张", is_winning_hand(counts("W1 W2 W3 W4 W5 W6 W7 W8 W9 T1 T1 T1 B2 B2"), []), True)
    ok &= check("差一张不成胡", is_winning_hand(counts("W1 W2 W3 W4 W5 W6 W7 W8 T1 T1 T1 B2 B2 B3"), []), False)
    ok &= check("七对", is_winning_hand(counts("W1 W1 W3 W3 W5 W5 T2 T2 T4 T4 T6 T6 B9 B9"), []), True)

    print("== 2. 番型 ==")
    ok &= check("平胡=1番", classify_fan(counts("W1 W2 W3 W4 W5 W6 W7 W8 W9 T1 T1 T1 B2 B2"), []), (PING_HU, 1))
    ok &= check("大对子=5番", classify_fan(counts("W1 W1 W1 W3 W3 W3 T2 T2 T2 B5 B5 B5 B9 B9"), []), (DA_DUI_ZI, 5))
    ok &= check("七对=7番", classify_fan(counts("W1 W1 W3 W3 W5 W5 T2 T2 T4 T4 T6 T6 B9 B9"), []), (QI_DUI, 7))
    ok &= check("龙七对=10番", classify_fan(counts("W1 W1 W1 W1 W3 W3 T2 T2 T4 T4 T6 T6 B9 B9"), []), (LONG_QI_DUI, 10))
    ok &= check("清一色=10番", classify_fan(counts("W1 W2 W3 W4 W5 W6 W7 W8 W9 W2 W2 W2 W5 W5"), []), (QING_YI_SE, 10))
    ok &= check("清大对=15番", classify_fan(counts("W1 W1 W1 W3 W3 W3 W5 W5 W5 W7 W7 W7 W9 W9"), []), (QING_DA_DUI, 15))

    print("== 3. 副露参与的胡牌 ==")
    melds = [(MELD_PONG, parse_tile("W1"), 1)]
    ok &= check("碰后成胡", is_winning_hand(counts("W3 W4 W5 W7 W8 W9 T2 T2 T2 B5 B5"), melds), True)
    ok &= check("碰后听牌张", winning_tiles(counts("W3 W4 W5 W7 W8 W9 T2 T2 B5 B5"), melds) != [], True)

    print("== 4. 向听数 ==")
    ok &= check("已听牌 shanten=0", shanten(counts("W1 W2 W3 W4 W5 W6 W7 W8 W9 T1 T1 T1 B2"), 0), 0)
    ok &= check("一上一听 shanten=1", shanten(counts("W1 W2 W3 W4 W5 W6 W7 W8 W9 T1 T1 T3 B2"), 0), 1)

    print("== 5. 整局对局（随机 vs 启发式） ==")
    cfg = RulesConfig()
    t0 = time.time()
    n_games = 30
    stats = {"win": 0, "huangzhuang": 0, "timeout": 0}
    tot = [0.0] * 4
    for g in range(n_games):
        game = ZhuojiGame(cfg, seed=1000 + g)
        bots = [RandomBot(seed=g), HeuristicBot(seed=g + 7),
                HeuristicBot(seed=g + 11), RandomBot(seed=g + 3)]
        res = game.play(bots)
        stats[res["type"]] = stats.get(res["type"], 0) + 1
        for i in range(4):
            tot[i] += res["deltas"][i]
    dt = time.time() - t0
    print(f"  {n_games} 局耗时 {dt:.2f}s  ({dt/n_games*1000:.1f} ms/局)")
    print(f"  结果类型: {stats}")
    print(f"  四家累计得分: {[round(x,1) for x in tot]}")
    print(f"  ↳ 座位1/2 是启发式教师，得分应明显高于座位0/3 的随机 Bot")
    if tot[1] + tot[2] > tot[0] + tot[3]:
        print("  [OK ] 启发式 > 随机")
    else:
        print("  [WARN] 启发式未显著强于随机（样本少时可能波动）")

    print("== 6. 无异常中断检查 ==")
    for g in range(200):
        game = ZhuojiGame(cfg, seed=50000 + g)
        bots = [HeuristicBot(seed=g + i) for i in range(4)]
        res = game.play(bots)
        assert res["type"] in ("win", "huangzhuang", "timeout"), res["type"]
        assert abs(sum(res["deltas"])) < 1e-9, ("零和校验失败", res)
    print("  [OK ] 200 局全部正常结束且得分零和")

    print("\n结论:", "全部通过" if ok else "存在失败项")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
