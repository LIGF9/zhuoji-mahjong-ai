#!/usr/bin/env python3
"""§6.36.7 预登记一次性测量：P20 vs R12，全新池 730000+600（600 seed, greedy 四座）。

约束（计划书 §6.36.7，结果看之前已锁死）：
  * 必须跑在 CPU 空闲窗口 = DAgger 轮的训练段（annotate 已退出、train_exit 在跑）
  * 严禁与 annotate 并发（会把标注吞吐压到 <8k，逼近 3.5h 超时线）
  * 本脚本只等 + 只跑一次，零删除、固定路径覆盖写

用法：
  .venv/bin/python tools/run_fresh_pool_measure.py --wait-pid 85665 --wait-log train_r21.log
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOGS = ROOT / "runs/night_logs"


def ts() -> str:
    return time.strftime("%H:%M:%S")


def say(msg: str) -> None:
    line = f"[{ts()}] {msg}"
    print(line, flush=True)
    with (LOGS / "p20_vs_r12_fresh600.log").open("a") as fh:
        fh.write(line + "\n")


def alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--wait-pid", type=int, default=0, help="等这个 PID 退出（annotate）")
    ap.add_argument("--wait-log", default="", help="等这个年轮训练日志出现（train_rNN.log）")
    ap.add_argument("--wait-max-min", type=int, default=90)
    ap.add_argument("--a", default="models/exit_p20.pt")
    ap.add_argument("--b", default="models/exit_r12.pt")
    ap.add_argument("--seeds", type=int, default=600)
    ap.add_argument("--seed0", type=int, default=730_000)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--out", default="runs/gates/p20_vs_r12_fresh600.jsonl")
    args = ap.parse_args()

    deadline = time.time() + args.wait_max_min * 60
    if args.wait_pid:
        say(f"等待 annotate PID {args.wait_pid} 退出（上限 {args.wait_max_min}min）")
        while alive(args.wait_pid):
            if time.time() > deadline:
                say("[abort] 等待 annotate 超时，放弃本次测量（不并发）")
                return 2
            time.sleep(20)
        say("annotate 已退出")

    if args.wait_log:
        p = LOGS / args.wait_log
        say(f"等待训练日志 {args.wait_log} 出现")
        while not p.exists():
            if time.time() > deadline:
                say("[abort] 等待训练开始超时，放弃（宁可不测也不并发）")
                return 2
            time.sleep(10)
        say(f"训练已开始（{args.wait_log} 存在），进入 CPU 空闲窗口")

    cmd = [sys.executable, "-u", "eval_pairs.py",
           "--a", args.a, "--b", args.b, "--mode", "greedy",
           "--seeds", str(args.seeds), "--seed0", str(args.seed0),
           "--workers", str(args.workers), "--out", args.out]
    say("RUN " + " ".join(cmd))
    t0 = time.time()
    rc = subprocess.call(cmd, cwd=str(ROOT),
                         stdout=open(LOGS / "p20_vs_r12_fresh600_eval.log", "a"),
                         stderr=subprocess.STDOUT)
    say(f"完成 rc={rc} 耗时 {time.time() - t0:.0f}s → {args.out}")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
