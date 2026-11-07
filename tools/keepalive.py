#!/usr/bin/env python3
"""独山麻将 7×24 看门狗（2026-11-07）。

单进程循环，每 30s 自检两件事：
  1. Web 对局服务（127.0.0.1:8770）不可达 → 拉起 web/dushan_server.py；
  2. 训练驱动心跳 runs/night_run.lock 超过 120s 未更新 → 拉起 night_run.py
     （驱动自带 driver_alive() 防重复：已有活驱动时第二个实例立即退出，
      所以这里重试是安全的）。
     **例外（§6.43 提前终评）**：runs/early_stop 存在时不再拉起驱动，
     但 Web 守护照常继续 —— 用户在线打牌不受影响。

背景：Web/训练进程曾随会话边界被回收（nohup 也保不住），launchd 在本机
不可用（bootstrap EIO）、crontab 被 TCC 拒绝 → 用受管长驻脚本 + 每小时
自动化兜底。绝不 kill 正在服务的进程；只在确认不可达时拉新。

日志：runs/night_logs/keepalive.log（只记状态变化与拉起事件，防刷屏）。
"""
from __future__ import annotations

import datetime as dt
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path("/Users/ligf9/zhuoji-mahjong-ai")
PY = str(ROOT / ".venv/bin/python")
PY3 = str(ROOT / ".venv/bin/python3")
LOCK = ROOT / "runs/night_run.lock"
PIDFILE = ROOT / "runs/keepalive.pid"
LOG = ROOT / "runs/night_logs/keepalive.log"
DEADLINE = dt.datetime(2026, 11, 13, 2, 30, 0)
EARLY_STOP = ROOT / "runs" / "early_stop"   # §6.43：存在即停止拉起训练驱动


def already_running() -> bool:
    """pidfile + os.kill(pid,0) 防重复实例（不依赖 ps/TCC）。"""
    try:
        pid = int(PIDFILE.read_text().strip())
        os.kill(pid, 0)
        return True
    except (OSError, ValueError):
        return False

WEB_OK = True
DRV_OK = True


def log(msg: str) -> None:
    line = f"[{dt.datetime.now():%F %H:%M:%S}] {msg}"
    print(line, flush=True)
    try:
        LOG.parent.mkdir(parents=True, exist_ok=True)
        with open(LOG, "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except OSError:
        pass


def web_alive() -> bool:
    """TCP 连通即认为在服务（HTTP 细节留给自动化复核，这里只求快而稳）。"""
    try:
        with socket.create_connection(("127.0.0.1", 8770), timeout=2):
            return True
    except OSError:
        return False


def driver_alive() -> bool:
    if not LOCK.exists():
        return False
    return time.time() - LOCK.stat().st_mtime < 120


def start_web() -> None:
    out = open(ROOT / "runs/night_logs/web_server.log", "a", encoding="utf-8")
    subprocess.Popen(
        [PY3, str(ROOT / "web/dushan_server.py"), "--host", "0.0.0.0", "--port", "8770"],
        cwd=str(ROOT), stdout=out, stderr=subprocess.STDOUT,
        start_new_session=True,           # 脱离本进程会话，避免被连带回收
    )


def start_driver() -> None:
    out = open(ROOT / "runs/night_logs/keepalive_night_run.log", "a", encoding="utf-8")
    subprocess.Popen(
        [PY, str(ROOT / "night_run.py")],
        cwd=str(ROOT), stdout=out, stderr=subprocess.STDOUT,
        start_new_session=True,
    )


def main() -> int:
    global WEB_OK, DRV_OK
    if already_running():
        print(f"keepalive 已在运行（pid {PIDFILE.read_text().strip()}），退出", flush=True)
        return 0
    PIDFILE.write_text(str(os.getpid()), encoding="utf-8")
    log(f"keepalive 启动（pid {os.getpid()}；30s 一轮；Web 8770 + 训练驱动锁心跳）")
    while True:
        # Web 服务
        if web_alive():
            if not WEB_OK:
                log("Web 服务恢复")
            WEB_OK = True
        else:
            if WEB_OK:
                log("Web 服务不可达 → 拉起")
            else:
                log("Web 仍不可达 → 再次拉起")
            start_web()
            WEB_OK = False
            time.sleep(3)

        # 训练驱动（早停标记或截止时间后不再拉起；驱动正常跑完自动退出）
        if EARLY_STOP.exists():
            if DRV_OK:
                log("检测到 runs/early_stop（§6.43 提前终评）→ 不再拉起训练驱动，Web 守护继续")
                DRV_OK = False
        elif dt.datetime.now() < DEADLINE:
            if driver_alive():
                if not DRV_OK:
                    log("训练驱动恢复")
                DRV_OK = True
            else:
                if DRV_OK:
                    log("训练驱动心跳停（>120s）→ 拉起（自动断点续跑）")
                else:
                    log("训练驱动仍无心跳 → 再次拉起")
                start_driver()
                DRV_OK = False
                time.sleep(3)

        time.sleep(30)


if __name__ == "__main__":
    sys.exit(main())
