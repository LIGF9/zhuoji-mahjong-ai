"""训练值班进程：防休眠 + 等 big_s6 收尾 + 自动接续 big_s7。

设计要点
- 全程持有 SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED)，阻止系统空闲休眠；
- 只以日志收尾标志（"完成。"）作为 big_s6 结束判据，避免误判导致并发启动两个训练；
- big_s7 以子进程前台方式运行，本进程存活期间一直持有防休眠状态。
"""
from __future__ import annotations

import ctypes
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PY = rr"C:\Users\hyzor\.workbuddy\binaries\python\envs\zhuoji\Scripts\python.exe"
LOG6 = ROOT / "reports" / "dushan_big_s6.log"
LOG7 = ROOT / "reports" / "dushan_big_s7.log"
SHIFT_LOG = ROOT / "reports" / "run_shift.log"

ES_CONTINUOUS = 0x80000000
ES_SYSTEM_REQUIRED = 0x00000001

S7_ITERS = 850


def awake(on: bool = True) -> bool:
    try:
        ctypes.windll.kernel32.SetThreadExecutionState(
            (ES_CONTINUOUS | ES_SYSTEM_REQUIRED) if on else ES_CONTINUOUS)
        return True
    except Exception:
        return False


def main() -> int:
    log = open(SHIFT_LOG, "w", encoding="utf-8", buffering=1)

    def say(msg: str) -> None:
        print(f"[{time.strftime('%H:%M:%S')}] {msg}", file=log)

    say(f"值班启动，防休眠={'OK' if awake(True) else '失败'}")

    while True:
        try:
            text = LOG6.read_text(encoding="utf-8", errors="ignore")
        except FileNotFoundError:
            text = ""
        if "完成。" in text:
            say("检测到 big_s6 收尾标志")
            break
        time.sleep(30)

    time.sleep(20)
    say("启动 big_s7（init=models/dushan_big_s6.pt）")

    with open(LOG7, "w", encoding="utf-8") as out:
        rc = subprocess.run(
            [PY, "scripts/selfplay.py",
             "--init", "models/dushan_big_s6.pt",
             "--out", "models/dushan_big_s7.pt",
             "--history", "reports/rl_history_big_s7.json",
             "--iters", str(S7_ITERS), "--games-per-iter", "96", "--batch", "512",
             "--lr", "2.0e-5", "--temperature", "1.0", "--teacher-prob", "0.25",
             "--threads", "8", "--vectorized", "--workers", "4",
             "--seed", "20263", "--rules", "dushan"],
            cwd=str(ROOT), stdout=out, stderr=subprocess.STDOUT,
        ).returncode

    say(f"big_s7 结束，退出码 {rc}")
    try:
        tail = LOG7.read_text(encoding="utf-8", errors="ignore").strip().splitlines()[-3:]
        for line in tail:
            say("  " + line)
    except Exception:
        pass
    awake(False)
    say("已释放防休眠限制，值班结束")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
