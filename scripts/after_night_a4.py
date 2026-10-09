"""等方案 B 阶段 2 夜跑队列退出后，自动接续方案 A4（s9 → s12 重跑一代 + 代内重选）。

为什么需要这个脚本
------------------
A4 的第一版用 ``nohup ... &`` 脱离式启动，父 shell 一退出就被连带杀死：它在
21:42:14 写完第一行日志后就死了，而它要等的天梯 22:40 才交付 —— 整条流水线因此
断了 2 小时（夜跑队列白等了 129 分钟）。教训是"脱离式后台"在这台机器上不可靠。

改为：本脚本用 ``subprocess.run`` **串行**启动 A4，脚本自身走工具托管后台。
等待期间每 10 分钟写一条心跳，便于事后核对。

判据（只看进程，不看文件）
--------------------------
夜跑队列每跑完一臂就会更新 ``night_plan_b_stage2_summary.json``，所以**文件存在
不能当作"夜跑结束"**；唯一可靠的判据是进程表里不再有跑
``night_plan_b_stage2`` 的 python 进程。

用法::

    python scripts/after_night_a4.py          # 等待并接续
    python scripts/after_night_a4.py --now    # 立即接续（调试用）
"""
from __future__ import annotations

import argparse
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = r"C:\Users\hyzor\.workbuddy\binaries\python\envs\zhuoji\Scripts\python.exe"
NIGHT_SUMMARY = ROOT / "reports" / "night_plan_b_stage2_summary.json"
LOG = ROOT / "reports" / "after_night_a4.log"
A4 = ROOT / "scripts" / "shift_plan_a_s12.py"

#: 认人的关键词：命令行里出现它就说明夜跑队列（或其子进程）还活着
NEEDLE = "night_plan_b_stage2"


def say(msg: str) -> None:
    with open(LOG, "a", encoding="utf-8", buffering=1) as f:
        f.write(f"[{time.strftime('%m-%d %H:%M:%S')}] {msg}\n")


def night_alive() -> bool:
    """夜跑队列（含其子进程）是否还在跑。psutil 缺失时保守返回 False。"""
    try:
        import psutil
    except ImportError:
        say("警告：psutil 不可用，无法检测夜跑进程；按已退出处理")
        return False
    for p in psutil.process_iter(["name", "cmdline"]):
        name = (p.info.get("name") or "").lower()
        if "python" not in name:
            continue
        if any(NEEDLE in c for c in (p.info.get("cmdline") or [])):
            return True
    return False


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--now", action="store_true", help="不等夜跑，立即接续 A4")
    ap.add_argument("--max-hours", type=float, default=20.0,
                    help="最长等待小时数，超时放弃（默认 20h）")
    args = ap.parse_args()

    say("=" * 62)
    say("A4 接续监控启动（夜跑队列退出后自动跑 shift_plan_a_s12.py）")

    if not args.now:
        # 起跑缓冲：避免夜跑刚启动/进程表尚未稳定时误判为"已退出"
        time.sleep(120)
        t0 = time.time()
        while night_alive():
            waited_h = (time.time() - t0) / 3600
            if waited_h > args.max_hours:
                say(f"等待夜跑超过 {args.max_hours:.0f}h，放弃（请人工检查夜跑队列）")
                return 1
            if int(time.time() - t0) % 600 < 30:
                hint = ("summary 已落盘" if NIGHT_SUMMARY.exists()
                        else "summary 尚未落盘")
                say(f"夜跑队列仍在运行… 已等 {(time.time() - t0) / 60:.0f} 分钟（{hint}）")
            time.sleep(60)
        say(f"夜跑队列已退出（等待 {(time.time() - t0) / 60:.0f} 分钟）")

    if NIGHT_SUMMARY.exists():
        say("夜跑 summary 已落盘，可核对各臂结果：" + str(NIGHT_SUMMARY.name))
    else:
        say("警告：夜跑 summary 不存在（可能异常退出）；仍继续 A4")

    say("启动 A4：shift_plan_a_s12.py --no-wait")
    t0 = time.time()
    rc = subprocess.run([PY, str(A4), "--no-wait"], cwd=str(ROOT)).returncode
    say(f"A4 退出 rc={rc}，用时 {(time.time() - t0) / 3600:.2f}h")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
