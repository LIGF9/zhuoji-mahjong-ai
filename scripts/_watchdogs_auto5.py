"""为 AUTO5 段补挂守望（gate_watch ×2 + stoploss ×2），并常驻看护。

背景（2026-10-16 事故）：旧总控被终止时，平台连带回收了它挂的全部守望，
导致 AUTO5 段处于"无判读 / 无止损"的裸奔状态。

本脚本作为**长驻启动器**（外层 run_in_background）运行；它 Popen 出的 4 个守望
会随它存活。**不要 kill 本脚本**；若它意外退出，重跑一次即可恢复全部守望。

每 60s 巡检：按 `--gate AUTO5@<at>` 匹配在位性，缺则补挂（幂等）。
"""
import subprocess
import time
from pathlib import Path

import psutil

ROOT = Path(__file__).resolve().parent.parent
PY = r"C:\Users\LIGF\.workbuddy\binaries\python\envs\zhuoji\Scripts\python.exe"
STEM = "dushan_m2_auto5"
THRESHOLD = "-1.42"
GATES = (400, 800)

LOG = ROOT / "reports" / "_watchdogs_auto5.out"
FLAGS = (subprocess.CREATE_NEW_PROCESS_GROUP
         | subprocess.CREATE_BREAKAWAY_FROM_JOB
         | subprocess.DETACHED_PROCESS)


def _log(msg: str) -> None:
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with LOG.open("a", encoding="utf-8") as fh:
        fh.write(line + "\n")


def gate_cmd(at: int) -> list[str]:
    return [
        PY, str(ROOT / "scripts" / "gate_watch.py"),
        "--stem", STEM, "--at", str(at), "--gate", f"AUTO5@{at}",
        "--threshold", THRESHOLD,
        "--baseline-name", "本段起点（动态档假设）",
        "--baseline", "-1.000", "--baseline-ci", "0.414",
        "--deals", "2400", "--seed", "31337",
        "--workers", "4", "--threads", "2", "--worker-threads", "1",
        "--log", str(ROOT / "reports" / "_m1_auto5.out"),
        "--stall-min", "30", "--max-wait-h", "14",
        "--out", str(ROOT / "reports" / f"AUTO5_it{at:04d}.json"),
    ]


def stop_cmd(at: int) -> list[str]:
    return [
        PY, str(ROOT / "scripts" / "m2_stoploss_watch.py"),
        "--verdict", str(ROOT / "reports" / f"AUTO5_it{at:04d}.verdict.json"),
        "--pid-file", str(ROOT / "reports" / "_m1_auto5.pid"),
        "--threshold", THRESHOLD, "--gate", f"AUTO5@{at}", "--stem", STEM,
        "--reports", "reports",
        "--log", str(ROOT / "reports" / f"_stoploss_AUTO5_{at}.log"),
        "--poll-s", "60", "--max-wait-h", "14",
    ]


def alive_matching(gate: str, script: str) -> bool:
    for p in psutil.process_iter(["cmdline"]):
        try:
            cl = " ".join(p.info.get("cmdline") or [])
            if script in cl and f"--gate {gate}" in cl:
                return True
        except psutil.Error:
            pass
    return False


_log("AUTO5 守望看护启动")
while True:
    for at in GATES:
        g = f"AUTO5@{at}"
        if not alive_matching(g, "gate_watch.py"):
            pr = subprocess.Popen(gate_cmd(at), stdin=subprocess.DEVNULL,
                                  stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT,
                                  cwd=str(ROOT), creationflags=FLAGS)
            _log(f"补挂判读守望 {g}（pid {pr.pid}）")
        if not alive_matching(g, "m2_stoploss_watch.py"):
            pr = subprocess.Popen(stop_cmd(at), stdin=subprocess.DEVNULL,
                                  stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT,
                                  cwd=str(ROOT), creationflags=FLAGS)
            _log(f"补挂止损守望 {g}（pid {pr.pid}）")
    time.sleep(60)
