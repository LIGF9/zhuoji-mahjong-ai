"""探针：确认 碰/杠/听牌(报叫) 在事件流 & 引擎日志中的真实产出。

用法：
    python scripts/probe_log_events.py --hands 4

结论用于 UI「流水」面板改造前的事实核对（不修改任何生产文件）。
"""
from __future__ import annotations

import argparse
import collections
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import web.dushan_server as srv  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hands", type=int, default=4)
    ap.add_argument("--seed", type=int, default=20261011)
    ap.add_argument("--opp", choices=("random", "model"), default="random",
                    help="对手类型：random=随机（快，几乎不会报叫）；model=大师模型（真实分布）")
    ap.add_argument("--model", default="dushan_master")
    args = ap.parse_args()

    kind_count: collections.Counter = collections.Counter()
    kind_sample: dict[str, str] = {}
    log_count: collections.Counter = collections.Counter()
    log_samples: list[str] = []
    hands_done = 0

    orig_push = srv.Session._push

    def push(self, kind, rel, text, tile=None, ctag=""):
        kind_count[kind] += 1
        kind_sample.setdefault(kind, text)
        return orig_push(self, kind, rel, text, tile, ctag)

    srv.Session._push = push

    def on_end(sess, game):
        nonlocal hands_done
        hands_done += 1
        for seat, txt in game.log:
            log_count[txt] += 1
            for key in ("报叫", "碰", "暗杠", "补杠", "点杠", "胡"):
                if txt.startswith(key):
                    log_count["__" + key] += 1
        if hands_done <= 2:
            log_samples.append(" | ".join(t for _s, t in game.log))

    if args.opp == "model":
        spec = {"type": "model", "model": args.model, "temperature": 0}
    else:
        spec = {"type": "random"}
    specs = [dict(spec) for _ in range(3)]
    sess = srv.new_session(specs, my_seat=0, seed=args.seed, speed=0.0,
                          auto=True, rules=None, names=None)
    sess.on_hand_end = on_end

    t0 = time.time()
    while hands_done < args.hands and time.time() - t0 < 180:
        time.sleep(0.3)
    sess.stop = True
    time.sleep(0.4)

    print("=" * 70)
    print(f"完成手数: {hands_done}")
    print("-" * 70)
    print("事件流 kind 统计（_push 实际调用）:")
    for k, v in kind_count.most_common():
        print(f"  {k:10s} {v:7d}   例: {kind_sample.get(k, '')}")
    print("-" * 70)
    print("引擎 game.log 关键条目统计:")
    for k in ("__报叫", "__碰", "__暗杠", "__补杠", "__点杠", "__胡"):
        print(f"  {k:10s} {log_count.get(k, 0):7d}")
    print(f"  (log 总条目 {sum(v for k, v in log_count.items() if not k.startswith('__'))})")
    print("-" * 70)
    for i, s in enumerate(log_samples):
        print(f"第{i + 1}手 log 前 600 字:\n{s[:600]}\n")
    print("=" * 70)


if __name__ == "__main__":
    main()
