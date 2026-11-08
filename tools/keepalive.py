#!/usr/bin/env python3
"""独山麻将 7×24 看门狗（2026-11-07）。

单进程循环，每 30s 自检三件事：
  1. Web 对局服务（127.0.0.1:8770）不可达 → 拉起 web/dushan_server.py；
  2. 训练驱动心跳 runs/night_run.lock 超过 120s 未更新 → 拉起 night_run.py
     （驱动自带 driver_alive() 防重复：已有活驱动时第二个实例立即退出，
      所以这里重试是安全的）。
     **例外（§6.43 提前终评）**：runs/early_stop 存在时不再拉起驱动，
     但 Web 守护照常继续 —— 用户在线打牌不受影响。
  3. 组合② 终评评估（§6.64.4）停滞 → 幂等拉起 runs/gates/combo2_chunk.py。
     与训练驱动完全解耦：early_stop 存在时**仍然生效**（这是离线评估，不是训练）。
     组合② 父进程曾三次随宿主会话边界被回收，故必须由本常驻进程以
     start_new_session=True 拉起，并严格防并发两片评估。

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

# ---- 组合② 终评评估兜底（§6.64.4；用户 2026-11-08 22:3x 指令「挂上每小时兜底」）----
# 本看门狗每 30s 检查，比每小时更密；自动化另有每小时外层保险（只查 keepalive 存活）。
COMBO2_CHUNK = ROOT / "runs/gates/combo2_chunk.py"
COMBO2_MAIN = ROOT / "runs/gates/combo2_main.jsonl"        # 2000 行 = 500 seed × 四座
COMBO2_LOG = ROOT / "runs/gates/combo2.log"
COMBO2_WLOG = ROOT / "runs/gates/combo2_watchdog.log"
COMBO2_PID = ROOT / "runs/gates/combo2_chunk.pid"          # chunk 驱动自写
COMBO2_REPORT = ROOT / "runs/gates/combo2_report_section.md"
COMBO2_NEED_ROWS = 2000


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
        cwd=str(ROOT), stdin=subprocess.DEVNULL,
        stdout=out, stderr=subprocess.STDOUT,
        start_new_session=True,           # 脱离本进程会话，避免被连带回收
    )


def start_driver() -> None:
    out = open(ROOT / "runs/night_logs/keepalive_night_run.log", "a", encoding="utf-8")
    # stdin 显式 DEVNULL：本进程若继承到已失效的 fd 0，子进程会在
    # init_sys_streams 阶段直接 EBADF 崩溃且瞬间退出（2026-11-08 00:44 事故）。
    subprocess.Popen(
        [PY, str(ROOT / "night_run.py")],
        cwd=str(ROOT), stdin=subprocess.DEVNULL,
        stdout=out, stderr=subprocess.STDOUT,
        start_new_session=True,
    )


# ------------------------------------------------------------------ 组合② --
def _pid_alive(pidfile: Path) -> bool:
    try:
        os.kill(int(pidfile.read_text().strip()), 0)
        return True
    except (OSError, ValueError):
        return False


def eval_pairs_alive():
    """True=确认有评估 worker；False=确认没有；None=探测不可用（TCC 等）。

    pgrep -f 用 [e] 正则技巧，避免匹配到 pgrep 自身命令行。
    """
    try:
        r = subprocess.run(["pgrep", "-f", "[e]val_pairs.py"],
                           capture_output=True, timeout=5)
    except (OSError, subprocess.TimeoutExpired):
        return None
    return r.returncode == 0


def combo2_log_stale(minutes: float) -> bool:
    try:
        return time.time() - COMBO2_LOG.stat().st_mtime > minutes * 60
    except OSError:
        return True


def combo2_done() -> bool:
    """完整判据：主汇总 2000 行 + 报告段已生成（§6.65.2 第 1 条）。"""
    if not (COMBO2_MAIN.exists() and COMBO2_REPORT.exists()):
        return False
    try:
        with open(COMBO2_MAIN, encoding="utf-8") as f:
            return sum(1 for line in f if line.strip()) >= COMBO2_NEED_ROWS
    except OSError:
        return False


def start_combo2() -> None:
    with open(COMBO2_WLOG, "a", encoding="utf-8") as out:
        subprocess.Popen(
            [PY, str(COMBO2_CHUNK)],
            cwd=str(ROOT), stdin=subprocess.DEVNULL,
            stdout=out, stderr=subprocess.STDOUT,
            start_new_session=True,           # 关键：脱离会话，避免随宿主被回收
        )


def combo2_guard() -> None:
    """组合② 幂等续跑兜底：只在「未完成 + 无评估 worker + 驱动不在跑」时拉起。

    * chunk 驱动自带 PIDFILE 防重、5-seed 分块、按 (seed,cand_seat) 去重，
      重跑零污染、零删除（§6.65.1）。
    * 严禁并发两片评估（§6.64.5）：确认有 eval_pairs worker 时一律不动。
    * 不碰 runs/early_stop、不拉训练驱动、不改任何评估口径。
    """
    if combo2_done():
        return
    ep = eval_pairs_alive()
    if ep is True:
        return                       # 孤儿 worker 仍在算，等它落盘
    if ep is None and not combo2_log_stale(20):
        return                       # 探测不可用且日志新鲜 ⇒ 保守不动
    if _pid_alive(COMBO2_PID) and not combo2_log_stale(45):
        return                       # 父进程活着且有进展
    if _pid_alive(COMBO2_PID):
        try:
            os.kill(int(COMBO2_PID.read_text().strip()), 15)
            log("组合② chunk 父进程疑似卡死（>45min 无日志）→ 终止后重拉")
            time.sleep(2)
        except (OSError, ValueError):
            pass
    log("组合② 停滞 → 拉起 combo2_chunk.py（幂等续跑）")
    start_combo2()


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

        # 组合② 终评评估兜底（与训练驱动解耦；early_stop 存在时仍生效）
        combo2_guard()

        time.sleep(30)


if __name__ == "__main__":
    sys.exit(main())
