"""
回归测试：抢杠路径。

这是曾经打爆向听计算的一个真 bug：
教师 Bot 在抢杠决策点拿到 ``acts = [过, 胡]``，把"胡"（tile=-1）
当成补杠去算 ``hand[-1] -= 1``，把九筒的计数扣成负数，导致向听数的
"弃孤张"分支永不收敛 → RecursionError。

这里手工摆出抢杠局面，验证：
1. ``actor()`` 在抢杠时返回抢杠者；
2. 教师 Bot 在该决策点不会污染手牌计数；
3. 抢杠胡的结算与"包三家"分值正确。
"""
from __future__ import annotations

import copy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from zhuoji import Action, Phase, RulesConfig, ZhuojiGame  # noqa: E402
from zhuoji.bots import HeuristicBot  # noqa: E402
from zhuoji.fan import MELD_KONG_CONCEALED, MELD_PONG  # noqa: E402
from zhuoji.tiles import parse_tile  # noqa: E402


def counts(s: str) -> list[int]:
    c = [0] * 27
    for x in s.split():
        c[parse_tile(x)] += 1
    return c


def build_rob_kong_game():
    """0 号补杠 T5，1 号听 T5 且手握闷豆 —— 必成抢杠。"""
    g = ZhuojiGame(RulesConfig(), seed=42)
    T5 = parse_tile("T5")
    g.dealer = 0
    g.dealer_streak = 0
    g.current = 0
    g.ever_tenpai = [True, True, True, True]

    # 0 号：碰过 T5，手里还有 1 张 T5 → 可补杠
    g.melds[0] = [(MELD_PONG, T5, 2)]
    g.hands[0] = counts("T5 W1 W2 W3 W4 W5 W6 B1 B2 B3 B7")

    # 1 号：单一归（T5 单张钓将），且已有闷豆 → 点胡通行证成立
    g.melds[1] = [(MELD_KONG_CONCEALED, parse_tile("B9"), 1)]
    g.hands[1] = counts("W1 W2 W3 W4 W5 W6 B1 B2 B3 T5")

    # 2/3 号：随便给点牌，不参与
    g.hands[2] = counts("W1 W2 W3 W4 W5 W6 B1 B2 B3 T1 T2 T3 T4")
    g.hands[3] = counts("W1 W2 W3 W4 W5 W6 B1 B2 B3 T6 T7 T8 T9")

    g.phase = Phase.SELF_KONG
    g._rob_kong = None
    return g, T5


def main() -> int:
    ok = True
    g, T5 = build_rob_kong_game()

    acts = [str(a) for a in g.legal_actions()]
    print("补杠前合法动作:", acts)
    ok &= "bugang:T5" in acts

    # 发起补杠 → 应当转入抢杠判定
    g.step(Action("bugang", T5))
    print("发起补杠后 phase =", g.phase, " _rob_kong =", g._rob_kong,
          " actor =", g.actor())
    ok &= g._rob_kong is not None
    if g.actor() != 1:
        ok = False
        print("  [FAIL] 抢杠时 actor 必须是抢杠者 1 号")
    else:
        print("  [OK ] actor() 正确指向抢杠者")

    acts = [str(a) for a in g.legal_actions()]
    print("抢杠决策点合法动作:", acts)
    if acts != ["pass", "hu"]:
        ok = False
        print("  [FAIL] 抢杠决策点动作应为 [pass, hu]")

    before = copy.deepcopy(g.hands)
    bot = HeuristicBot(seed=0)
    a = bot.choose(g)
    print("教师选择:", a)
    if g.hands != before or min(min(h) for h in g.hands) < 0:
        ok = False
        print("  [FAIL] 教师决策污染了手牌计数")
    else:
        print("  [OK ] 手牌计数未被污染（负数 bug 已修）")
    if a.kind != "hu":
        ok = False
        print("  [FAIL] 能抢杠胡就必须胡")

    # 执行抢杠胡，检查结算
    g.step(Action("hu"))
    res = g.result
    print("结算:", res["type"], res["fan_cn"], res["fan"], "rob_kong=", res["rob_kong"])
    print("得分:", res["deltas"])
    ok &= res["rob_kong"] is True
    tot = res["total_fan"]
    d = res["deltas"]
    print(f"  牌型总番={tot}（含坐庄加成）")
    # 「被抢杠者包三家」指的是**胡牌番值**由被抢杠者一家承担；
    # 豆、鸡属于独立结算项，仍按规则正常赔付。
    if abs(sum(d)) > 1e-9:
        ok = False
        print("  [FAIL] 得分不零和", d)
    if not (d[0] <= -3 * tot + 1e-9):
        ok = False
        print(f"  [FAIL] 被抢杠者应至少承担 3×{tot} 的手牌番，实为 {d[0]}")
    if not (d[1] >= 3 * tot - 1e-9):
        ok = False
        print(f"  [FAIL] 抢杠者应至少拿到 3×{tot} 的手牌番，实为 {d[1]}")
    if abs(d[2] + d[3]) >= 3 * tot - 1e-9:
        ok = False
        print("  [FAIL] 另两家被分摊了手牌番，包三家未生效")
    else:
        print("  [OK ] 包三家生效：被抢杠者 %+.1f，抢杠者 %+.1f，"
              "另两家合计 %+.1f（仅豆/鸡项）" % (d[0], d[1], d[2] + d[3]))

    # 压力回归：大量对局不得再爆栈
    from zhuoji.bots import TEACHER_STYLES
    import time
    t0 = time.time()
    for i in range(1500):
        gg = ZhuojiGame(RulesConfig(), seed=600000 + i)
        bots = [HeuristicBot(seed=i * 5 + k,
                             weights=TEACHER_STYLES[list(TEACHER_STYLES)[k]])
                for k in range(4)]
        gg.play([b.choose for b in bots])
    print(f"  [OK ] 1500 局压力回归通过，用时 {time.time()-t0:.1f}s")

    print("\n结论:", "通过" if ok else "失败")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
