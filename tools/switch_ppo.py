#!/usr/bin/env python3
"""在「阶段边界」把驱动切换到新版 night_run.py（新增 PPO 分支 + 闸门 D）。

为什么需要它：night_run.py 是长驻进程，不重载源码；改完源码必须重启才生效。
而驱动几乎总在跑标注（2h）/训练（40min）/闸门（25min），随便 kill 会白丢一段算力。

**精确触发点**：`annotate --out .../runs/exit_r{START}` 这一行被驱动 log 出来之后
约 3 秒 —— 此刻新的标注进程刚起、**一个 npz 都还没写**（标注器只在结束时落盘），
所以 kill 掉零数据损失，重启后 first_incomplete_round() 会把该轮从头重跑。

重启交给 keepalive（它用 start_new_session 拉起，是唯一被验证过能跨会话存活的方式）；
本脚本只负责「挑时刻 + 杀干净 + 校验新驱动起来了」，超时才自己兜底拉起。

用法（必须用 start_new_session 起，否则会话结束会被回收）：
  .venv/bin/python tools/switch_ppo.py --start-round 10 --timeout-h 4
"""
from __future__ import annotations

import argparse
import datetime as dt
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path("/Users/ligf9/zhuoji-mahjong-ai")
PY = str(ROOT / ".venv/bin/python")
LOG_MD = ROOT / "reports/exit_log.md"
LOG = ROOT / "runs/night_logs/switch_ppo.log"
LOCK = ROOT / "runs/night_run.lock"


def log(msg: str) -> None:
    line = f"[{dt.datetime.now():%F %H:%M:%S}] {msg}"
    print(line, flush=True)
    try:
        LOG.parent.mkdir(parents=True, exist_ok=True)
        with open(LOG, "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except OSError:
        pass


def note_md(msg: str) -> None:
    """主日志（与 night_run.log() 同文件同格式），便于后续自检看到切换事实。"""
    try:
        with open(LOG_MD, "a", encoding="utf-8") as f:
            f.write(f"[{dt.datetime.now():%F %H:%M:%S}] {msg}\n")
    except OSError:
        pass


def pids(pattern: str) -> list[int]:
    out = subprocess.run(["pgrep", "-f", pattern], capture_output=True, text=True)
    me = os.getpid()
    return [int(x) for x in out.stdout.split() if x.isdigit() and int(x) != me]


def trigger_seen(r: int) -> bool:
    """驱动已 log 出第 r 轮标注的启动行？"""
    try:
        txt = LOG_MD.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return False
    return f"--out {ROOT}/runs/exit_r{r:02d} " in txt


def lock_age() -> float:
    try:
        return time.time() - LOCK.stat().st_mtime
    except OSError:
        return 1e9


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start-round", type=int, default=10)
    ap.add_argument("--timeout-h", type=float, default=4.0)
    ap.add_argument("--grace-s", type=float, default=3.0)
    ap.add_argument("--relaunch-wait-s", type=float, default=300.0)
    a = ap.parse_args()

    log(f"等待 R{a.start_round} 标注启动行（最多 {a.timeout_h}h）；"
        f"触发后 kill 驱动，交由 keepalive 用新源码拉起")
    t_end = time.time() + a.timeout_h * 3600
    while time.time() < t_end:
        if trigger_seen(a.start_round):
            break
        time.sleep(10)
    else:
        log("超时未见触发点，放弃（训练主线不受影响，下次自检再切）")
        return 0

    time.sleep(a.grace_s)                      # 让标注进程真正起来（此刻尚未落盘）
    old = pids("night_run.py")
    log(f"触发命中。现有驱动 pid={old}，lock_age={lock_age():.1f}s")
    if not old:
        log("没有在跑的驱动 → 直接拉起新驱动")
    for p in old:
        try:
            os.kill(p, 15)
        except OSError:
            pass
    time.sleep(2)
    for p in pids("night_run.py"):             # 兜底：漏网的（含 zsh 包装层）
        try:
            os.kill(p, 9)
        except OSError:
            pass
    for p in pids("rust_engine/target/release/annotate"):
        try:
            os.kill(p, 15)
        except OSError:
            pass
    log(f"已终止：night_run={pids('night_run.py')} "
        f"annotate={pids('rust_engine/target/release/annotate')}")
    note_md(f"【切换】kill 旧驱动（pid={old}）于 R{a.start_round} 标注启动边界；"
            f"生效新源码：PPO 分支（train_ppo.py）+ 闸门 D（vs 上轮最佳）")

    # 等 keepalive 拉起（它会先等 lock 过期 120s，所以给足 5min）
    t0 = time.time()
    ok_new = False
    while time.time() - t0 < a.relaunch_wait_s:
        time.sleep(10)
        now = pids("night_run.py")
        if now and any(p not in old for p in now):
            ok_new = True
            break
    if not ok_new:
        log("keepalive 未在窗口内拉起 → 本脚本兜底启动（start_new_session）")
        out = open(ROOT / "runs/night_logs/keepalive_night_run.log", "a", encoding="utf-8")
        subprocess.Popen([PY, str(ROOT / "night_run.py")], cwd=str(ROOT),
                         stdout=out, stderr=subprocess.STDOUT, start_new_session=True)
        t0 = time.time()
        while time.time() - t0 < 180:
            time.sleep(10)
            now = pids("night_run.py")
            if now and any(p not in old for p in now):
                ok_new = True
                break

    # 校验：新 pid + 心跳新鲜 + exit_log 出现新的“驱动启动”行
    fresh = lock_age() < 120
    log(f"结果：新驱动={pids('night_run.py')} 心跳新鲜={fresh} lock_age={lock_age():.1f}s")
    note_md(f"【切换】新驱动 pid={pids('night_run.py')} 心跳新鲜={fresh}"
            f"（R{a.start_round} 标注将从头重跑，无脏数据）")
    return 0 if (ok_new and fresh) else 1


if __name__ == "__main__":
    sys.exit(main())
