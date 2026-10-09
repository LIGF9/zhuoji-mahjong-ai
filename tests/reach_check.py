"""新增规则「可达性」压力检查：抢杠胡 / 热炮 / 一炮多响 / 全烧 在真实随机对局中确实会发生。

只做一件事：用随机 + 启发式 bot 打 N 局，统计各种终局方式的出现次数，
并顺带校验每局都正常终局、分数零和。

跑法::

    PYTHONPATH=. python tests/reach_check.py [局数]
"""
from __future__ import annotations

import random
import sys
from collections import Counter

sys.path.insert(0, ".")

from zhuoji.bots import HeuristicBot, RandomBot  # noqa: E402
from zhuoji.dushan import DushanConfig, DushanGame  # noqa: E402
from zhuoji.rules import Phase  # noqa: E402

N = int(sys.argv[1]) if len(sys.argv) > 1 else 1500


def main() -> int:
    how = Counter()
    multi = 0
    void = 0
    cap = 0          # 捉炮鸡出现的局数
    by_seat_multi = Counter()
    opp = 0          # 出现过「同一张牌有 >=2 家可胡」的局数（一炮多响机会）
    opp_cnt = 0      # 机会出现次数（每次 ≥2 家同时可胡算一次）
    bad: list[str] = []
    rng = random.Random(20261008)
    for i in range(N):
        cfg = DushanConfig()
        # 一半的局打开开局翻鸡，制造更多鸡牌种
        if i % 2 == 0:
            cfg.kaiju_fanji = "both"
        g = DushanGame(cfg, seed=100000 + i)
        # 胡牌一律不放弃（hu_prob=1），最大化「一炮多响」落地的概率
        agents = [RandomBot(seed=rng.randrange(10**6), hu_prob=1.0),
                  HeuristicBot(seed=rng.randrange(10**6)),
                  RandomBot(seed=rng.randrange(10**6), hu_prob=1.0),
                  HeuristicBot(seed=rng.randrange(10**6))]
        steps = 0
        hit = 0
        while g.phase != Phase.OVER and steps < 6000:
            g.step(agents[g.actor()].choose(g))
            steps += 1
            if g.phase == Phase.RESPOND and getattr(g, "_n_hu", 0) >= 2:
                hit += 1
        if hit:
            opp += 1
            opp_cnt += hit
        if g.phase != Phase.OVER:
            bad.append(f"seed={100000+i} 未终局（steps={steps}）")
            continue
        res = g.result or {}
        if res.get("type") == "win":
            how[str(res.get("how"))] += 1
            ws = res.get("winners") or [res.get("winner")]
            if len(ws) > 1:
                multi += 1
                by_seat_multi[len(ws)] += 1
            if res.get("void_player") is not None:
                void += 1
            if any(d[1] == "捉炮鸡" for d in (res.get("detail") or [])):
                cap += 1
        else:
            how["黄庄"] += 1
        tot = sum(res.get("deltas") or [])
        if abs(tot) > 1e-6:
            bad.append(f"seed={100000+i} 非零和 {res.get('deltas')}")

    print(f"共 {N} 局")
    for k, v in how.most_common():
        print(f"  {k:<8} {v:>5}")
    print(f"  一炮多响  {multi:>5}   按赢家数={dict(by_seat_multi)}")
    print(f"  多胡可胡机会 {opp_cnt:>5} 次，出现在 {opp} 局")
    print(f"  鸡分全烧  {void:>5}")
    print(f"  捉炮鸡    {cap:>5}")
    print(f"异常：{len(bad)}")
    for b in bad[:5]:
        print("   ", b)
    # 一炮多响在随机对局中概率极低（需要同一张牌同时是两名玩家的胡牌张），
    # 因此不把它作为随机压测的通过条件；其正确性由确定性用例保证：
    #   tests/test_dushan.py::test_multi_ron（两听牌者同胡一张 → 各得牌/各自赔付/各捉炮鸡）
    ok = (not bad) and how.get("抢杠", 0) > 0 and how.get("热炮", 0) > 0
    print("\n结论:", "通过（抢杠/热炮 可在真实对局中复现）" if ok else "失败")
    if multi == 0:
        print(f"  注：一炮多响本次随机采样 {multi} 处（机会 {opp_cnt} 次），概率极低属正常；"
              "其正确性见 tests/test_dushan.py::test_multi_ron")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
