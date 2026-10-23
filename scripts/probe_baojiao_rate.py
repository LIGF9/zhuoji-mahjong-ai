"""实测「报叫」触发率：首张打出后即听牌的概率。

报叫的引擎条件是 `not first_discard_done[p] and is_tenpai(hands[p], melds[p])`，
即在玩家**本局第一次出牌之后**就已经听牌。这个事件在真实对局里到底有多常见？
本脚本直接包一层 _step_discard 统计，给出分子/分母（每手每位玩家各 1 次机会）。

用法：
    python scripts/probe_baojiao_rate.py --hands 60 --opp model
"""
from __future__ import annotations

import argparse
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import web.dushan_server as srv  # noqa: E402
import zhuoji.dushan as D  # noqa: E402

STATS = Counter()
_HOOK_READY = False


def install_hook() -> None:
    global _HOOK_READY
    if _HOOK_READY:
        return
    orig = D.DushanGame._step_discard

    def patched(self, action):  # noqa: ANN001
        p = int(self.current)
        is_first = not self.first_discard_done[p]
        orig(self, action)                      # 内部会置 first_discard_done / baojiao
        if is_first:
            STATS["first_discards"] += 1
            if self.baojiao[p]:
                STATS["baojiao"] += 1
            # 顺带记录：首张打出后是否已听牌（即使因 baojiao_auto=False 没报叫）
            try:
                from zhuoji.dushan import is_tenpai
                if is_tenpai(self.hands[p], self.melds[p]):
                    STATS["tenpai_after_first_discard"] += 1
            except Exception:
                pass

    D.DushanGame._step_discard = patched
    _HOOK_READY = True


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hands", type=int, default=60)
    ap.add_argument("--seed", type=int, default=20261011)
    ap.add_argument("--opp", choices=("random", "model"), default="model")
    ap.add_argument("--model", default="dushan_master")
    args = ap.parse_args()

    install_hook()

    done = [0]

    def on_end(sess, game):
        done[0] += 1

    spec = ({"type": "model", "model": args.model, "temperature": 0}
            if args.opp == "model" else {"type": "random"})
    sess = srv.new_session([dict(spec) for _ in range(3)], my_seat=0,
                           seed=args.seed, speed=0.0, auto=True)
    sess.on_hand_end = on_end

    t0 = time.time()
    while done[0] < args.hands and time.time() - t0 < 600:
        time.sleep(0.3)
    sess.stop = True
    time.sleep(0.4)

    fd = STATS["first_discards"] or 0
    bj = STATS["baojiao"] or 0
    tp = STATS["tenpai_after_first_discard"] or 0
    print("=" * 66)
    print(f"对手={args.opp}  完成手数={done[0]}")
    print(f"  首张打出次数（分母，每手每人 1 次）: {fd}")
    print(f"  其中打出后已听牌                 : {tp}"
          + (f"  ({tp / fd:.2%})" if fd else ""))
    print(f"  实际触发报叫                     : {bj}"
          + (f"  ({bj / fd:.2%})" if fd else ""))
    print("=" * 66)


if __name__ == "__main__":
    main()
