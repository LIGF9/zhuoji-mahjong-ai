#!/usr/bin/env python
"""为「手动接力的当前段」补挂判读 + 止损守望（沙箱巡检专用）。

为什么需要
----------
总控（m2_night_autopilot.py）在起训练时才会调用 attach_watchdogs() 挂守望。
巡检里训练是由 relay_auto6rrrr.py 手动接力的，守望于是缺位 —— 段末判读点
（auto6rrrr 的 it120）到点后没人跑判读，接力会卡在「等判决」上。

这个脚本按总控同一套参数把守望挂回去，并用「父进程常驻」的方式保证子进程
活得过本工具调用（沙箱会清理整棵进程树，所以本进程必须挂在平台后台任务里）。

用法::

    python scripts/relay_watch.py --stem dushan_m2_auto6rrr_r1 --at 120 \
        --gate AUTO6RRRR@120 --threshold -1.620 --log reports/_m1_auto6rrrr.out \
        --pid-file reports/_m1_auto6rrrr.pid
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stem", required=True)
    ap.add_argument("--at", type=int, required=True)
    ap.add_argument("--gate", required=True)
    ap.add_argument("--threshold", type=float, required=True)
    ap.add_argument("--baseline", type=float, required=True)
    ap.add_argument("--log", required=True)
    ap.add_argument("--pid-file", required=True)
    ap.add_argument("--deals", type=int, default=2400)
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()

    svc = ROOT / "scripts"
    out = ROOT / "reports" / f"{args.gate.split('@')[0]}_it{args.at:04d}.json"
    env = dict(os.environ, PYTHONIOENCODING="utf-8")

    gw = [PY, str(svc / "gate_watch.py"), "--stem", args.stem,
          "--at", str(args.at), "--gate", args.gate,
          "--threshold", str(args.threshold),
          "--baseline-name", "本段起点（实测）",
          "--baseline", f"{args.baseline:.3f}", "--baseline-ci", "0.414",
          "--deals", str(args.deals), "--seed", "31337",
          "--workers", str(args.workers), "--threads", "2", "--worker-threads", "1",
          "--log", args.log, "--stall-min", "30", "--max-wait-h", "14",
          "--out", str(out.relative_to(ROOT))]

    sl_log = ROOT / "reports" / f"_stoploss_{args.gate.replace('@', '_')}.log"
    sl = [PY, str(svc / "m2_stoploss_watch.py"),
          "--verdict", str(out.with_suffix(".verdict.json").relative_to(ROOT)),
          "--pid-file", args.pid_file, "--threshold", str(args.threshold),
          "--gate", args.gate, "--stem", args.stem,
          "--reports", "reports", "--log", str(sl_log.relative_to(ROOT)),
          "--poll-s", "60", "--max-wait-h", "14"]

    p_sl = subprocess.Popen(sl, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                            stderr=subprocess.STDOUT, cwd=str(ROOT), env=env)
    print(f"[watch] 止损守望 pid {p_sl.pid}（≤ {args.threshold:+.3f} 则停跑）", flush=True)
    p_gw = subprocess.Popen(gw, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                            stderr=subprocess.STDOUT, cwd=str(ROOT), env=env)
    print(f"[watch] 判读守望 pid {p_gw.pid}（{args.gate}，{args.deals} 副）", flush=True)

    # 父进程常驻：子进程是它的孩子，工具调用不结束，整棵树就不会被清
    while True:
        rc = p_gw.poll()
        if rc is not None:
            print(f"[watch] 判读守望退出（exit={rc}），一并结束止损守望", flush=True)
            try:
                p_sl.terminate()
            except Exception:                                    # noqa: BLE001
                pass
            return rc
        time.sleep(60)


if __name__ == "__main__":
    raise SystemExit(main())
