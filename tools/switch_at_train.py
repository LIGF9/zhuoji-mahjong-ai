#!/usr/bin/env python3
"""在「训练启动」这一刻把驱动切到新版 night_run.py（PPO 链式初始化）。

为什么需要它：night_run.py 是长驻进程，**不重载源码**；`ppo_init()` 这条修正只有在
重新执行 night_run.py 后才生效。而随便 kill 会白丢一段算力：

  - 标注期 kill → 丢最多 2h（annotate 只在结束时落盘）
  - 训练期 kill → 丢最多 34min（train_exit 无断点，.done 未写则从头重跑）
  - 闸门期 kill → 丢最多 25min（jsonl 半截则整段重跑）

**最优点 = 训练刚启动的那 2 秒**：此刻上一阶段标注已经 rc=0 且 4×npz 全部落盘，
train_exit 刚起还没写任何 ckpt/.done ⇒ kill 掉后 `first_incomplete_round()` 只会重跑
训练，净损失 ≈2 秒、零数据损失。

触发信号：驱动 log 出 `→ train_r{R}.log` 这一行（在 spawn 训练进程之前写入
reports/exit_log.md）。

重启交给 keepalive（start_new_session，唯一被验证能跨会话存活的方式）；本脚本只做
「挑时刻 + 杀干净 + 校验新驱动起来 + 兜底拉起」。

用法（必须 start_new_session 起，否则随会话被回收）：
  .venv/bin/python tools/switch_at_train.py --round 10 --timeout-h 4
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
LOG = ROOT / "runs/night_logs/switch_at_train.log"
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
    """写主日志（与 night_run.log() 同文件同格式），便于后续自检看到切换事实。"""
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
    """第 r 轮训练是否已真正开跑？

    **绝不能用「日志里出现某字符串」来判定**（2026-11-07 06:20 事故教训）：
    自检笔记里引用 `→ train_r10.log` 这几个字，就让本脚本误判触发、杀掉健康驱动。
    改为**看训练日志文件本身的落盘事实**：`runs/night_logs/train_r{R}.log` 存在且 60s 内
    有更新 ⇒ 训练进程确实在写 ⇒ 才是阶段边界。
    """
    p = ROOT / f"runs/night_logs/train_r{r:02d}.log"
    try:
        return p.exists() and (time.time() - p.stat().st_mtime) < 60
    except OSError:
        return False


def lock_age() -> float:
    try:
        return time.time() - LOCK.stat().st_mtime
    except OSError:
        return 1e9


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--round", type=int, default=10)
    ap.add_argument("--timeout-h", type=float, default=4.0)
    ap.add_argument("--grace-s", type=float, default=2.0)
    ap.add_argument("--relaunch-wait-s", type=float, default=300.0)
    a = ap.parse_args()

    log(f"等待 R{a.round} 训练启动行（最多 {a.timeout_h}h）；"
        f"触发后 kill 驱动，交由 keepalive 用新源码拉起")
    t_end = time.time() + a.timeout_h * 3600
    while time.time() < t_end:
        if trigger_seen(a.round):
            break
        time.sleep(5)
    else:
        log("超时未见触发点，放弃（训练主线不受影响，下次自检再切）")
        return 0

    time.sleep(a.grace_s)                       # 训练进程刚 spawn，尚未写任何 ckpt/.done
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
    for p in pids("night_run.py"):              # 兜底：漏网的（含 zsh 包装层）
        try:
            os.kill(p, 9)
        except OSError:
            pass
    # 关键：训练子进程必须一起收掉，否则孤儿 train_exit 会继续占 MPS/CPU 并与新驱动抢卡
    for p in pids("train_exit.py"):
        try:
            os.kill(p, 15)
        except OSError:
            pass
    log(f"已终止：night_run={pids('night_run.py')} "
        f"train_exit={pids('train_exit.py')}")
    note_md(f"【切换】kill 旧驱动（pid={old}）于 R{a.round} 训练启动边界（净损失≈2s）；"
            f"生效新源码：PPO 链式初始化 ppo_init()（exit_p{a.round-1:02d} 优先）")

    # 等 keepalive 拉起（它先等 lock 过期 120s，给足 5min）
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

    fresh = lock_age() < 120
    log(f"结果：新驱动={pids('night_run.py')} 心跳新鲜={fresh} lock_age={lock_age():.1f}s")
    note_md(f"【切换】新驱动 pid={pids('night_run.py')} 心跳新鲜={fresh}"
            f"（R{a.round} 训练从头重跑，标注数据无损）")
    return 0 if (ok_new and fresh) else 1


if __name__ == "__main__":
    sys.exit(main())
