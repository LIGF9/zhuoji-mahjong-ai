#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""训练/评测进度健康度巡检（**只读**，零干扰）。

为什么另写一个
--------------
`tools/health_watch.py` 是 **PPO 轮专用**：只读 `runs/night_logs/ppo_p*.log` 里的
`[metrics]` 行，判据围绕 kl/clipfrac/ent/gnorm。ExIt 与蒸馏链路（`train_exit.py`、
`tools/distill_slim.py`）以及闸门评测（`eval_pairs.py`）根本不打 `[metrics]`，
在那张表里是空的。本工具补的是**任务级**健康度：还在不在跑、有没有在推进、
val 是不是在退化、有没有静默挂掉。

三类任务各自认什么
------------------
* ``epoch``（蒸馏/ExIt 训练）：``[ep NN] train … val … total …s`` → 轮数、val 曲线；
* ``eval``（配对评测）：``[wN] k/T`` → 分片进度；``[summary]``/``[done]`` → 收口；
* ``generic``：只看存活 + 日志新鲜度。

判定（**巡检阈值，不是预登记判据**）
------------------------------------
这些阈值只用于"要不要叫人看一眼"，**绝不参与任何闸门判定**，也不改任何 `δ`/池/seed：

* **RED**：进程活着但连续 2 次巡检进度不动；日志出现 Traceback/Killed/OOM；
  磁盘可用 < 5 GiB；val 连续 3 轮上升（价值头退化）。
* **YELLOW**：日志静默超过该任务的 `stall_sec`；负载 > 核数×1.5；
  进程已退出但没有收口标记（`[done]`/`[early-stop]`/`[w*] done`）。
* **GREEN**：其余。
* **IDLE**：没有任何存活任务，也没有 2 小时内的日志（无事可做 ≠ 异常）。

为什么不用 ps / psutil
----------------------
本机沙箱拒绝 ``ps``（operation not permitted），也没装 psutil。所以存活用 ``pgrep -f``、
推进用"解析计数 + 与上一条历史记录差分"。这反而更稳：日志读数不受调度抖动影响。

用法::

    .venv/bin/python tools/patrol_health.py            # 人读表 + 结论
    .venv/bin/python tools/patrol_health.py --json     # 机器读（自动化用）
    .venv/bin/python tools/patrol_health.py --no-write # 只看，不写历史

历史落在 ``runs/patrol/health.jsonl``，每行一次巡检；差分只取上一条，不回溯。
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
HIST = ROOT / "runs" / "patrol" / "health.jsonl"

# ---- 巡检阈值（非预登记判据；改这里不影响任何闸门） ----
DISK_MIN_GB = 5.0
LOAD_FACTOR = 1.5
STALL_STREAK = 2          # 连续几次巡检进度不动 → RED
VAL_RISE_STREAK = 3       # val 连续几轮上升 → RED
RECENT_SEC = 2 * 3600     # 多久内有日志算"近期任务"

EP_RE = re.compile(
    r"\[ep\s+(\d+)\]\s+train\s+([\d.]+)\s+val\s+([\d.]+).*?total\s+(\d+)s")
W_RE = re.compile(r"\[w(\d+)\]\s+(\d+)/(\d+)")
BEST_RE = re.compile(r"best val\s+([\d.]+)")
DONE_MARKS = ("[done]", "[early-stop]", "[w0] done", "ALL PASS", "HAS FAIL")


@dataclass
class Job:
    name: str
    kind: str                       # epoch | eval | generic
    proc_pat: str
    log: Path | None
    expect: int = 0                 # epoch 任务=预期轮数；eval=每 worker 总片数
    stall_sec: int = 900
    alive: bool = False
    # 解析结果
    epoch: int = 0
    val: float = float("nan")
    best: float = float("nan")
    vals: list[float] = field(default_factory=list)
    prog: int = 0                   # eval: 各 worker 已完成的**最大**片数之和
    prog_total: int = 0
    workers: set = field(default_factory=set)
    wk: dict = field(default_factory=dict)    # worker -> 该 worker 已达的最大 k
    last_line: str = ""
    age: float = float("inf")       # 日志静默秒数
    finished: bool = False
    errors: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


def pgrep_alive(pat: str) -> bool:
    try:
        r = subprocess.run(["pgrep", "-f", pat], capture_output=True, text=True)
    except (OSError, subprocess.SubprocessError):
        return False
    # 排除把自己也算进去的情况：pgrep -f 会匹配到本进程的 argv
    pids = [p for p in r.stdout.split() if p.isdigit() and int(p) != os.getpid()]
    return bool(pids)


def read_tail(path: Path, limit: int = 400) -> list[str]:
    try:
        with path.open("r", encoding="utf-8", errors="replace") as f:
            return f.read().splitlines()[-limit:]
    except OSError:
        return []


def fill(job: Job) -> Job:
    job.alive = pgrep_alive(job.proc_pat)
    if job.log is None:
        return job          # generic 任务（如线上服务）只判存活，不看日志
    if not job.log.exists():
        if job.alive:
            job.errors.append("进程存活但日志文件不存在")
        return job
    job.age = time.time() - job.log.stat().st_mtime
    lines = read_tail(job.log)
    job.last_line = next((ln for ln in reversed(lines) if ln.strip()), "")
    job.finished = any(m in job.last_line for m in DONE_MARKS)

    for ln in lines:
        low = ln.lower()
        if any(k in low for k in ("traceback", "killed", "out of memory", "oom")):
            job.errors.append(ln.strip()[:160])
        m = EP_RE.search(ln)
        if m:
            job.epoch = int(m.group(1))
            job.vals.append(float(m.group(3)))
            job.val = job.vals[-1]
        m = BEST_RE.search(ln)
        if m:
            job.best = float(m.group(1))
        m = W_RE.search(ln)
        if m:
            w, k = int(m.group(1)), int(m.group(2))
            job.workers.add(w)
            # 每个 worker 取**最大** k 再求和：同一 worker 的 10/60、20/60 是
            # 累计进度，直接相加会重复计数；read_tail 只读尾部 400 行，
            # 窗口滑动会漏掉早期行，也只有"取最大"才稳定。
            job.wk[w] = max(job.wk.get(w, 0), k)
            job.prog = sum(job.wk.values())
            job.prog_total = max(job.prog_total, int(m.group(3)))
    return job


def job_specs() -> list[Job]:
    """当前链路的任务清单（新增训练脚本时在这里登记即可）。"""
    runs = ROOT / "runs"
    specs = [
        Job(name="E-SLIM 蒸馏", kind="epoch", proc_pat="distill_slim.py",
            log=runs / "slim_s.log", expect=16, stall_sec=900),
        Job(name="闸门 A 评测", kind="eval", proc_pat="eval_pairs.py",
            log=runs / "gates" / "slim_gateA.log", expect=60, stall_sec=3600),
        Job(name="ExIt 训练", kind="epoch", proc_pat="train_exit.py",
            log=runs / "night_logs" / "exit_latest.log", expect=0, stall_sec=1800),
        Job(name="PPO 加速轮", kind="epoch", proc_pat="train_ppo.py",
            log=runs / "night_logs" / "ppo_latest.log", expect=0, stall_sec=1800),
        Job(name="线上服务", kind="generic", proc_pat="dushan_server",
            log=None, stall_sec=0),
    ]
    return specs


def load_hist() -> dict | None:
    if not HIST.exists():
        return None
    lines = read_tail(HIST, 5)
    for ln in reversed(lines):
        try:
            return json.loads(ln)
        except json.JSONDecodeError:
            continue
    return None


def judge(jobs: list[Job], prev: dict | None) -> tuple[str, list[str], list[str]]:
    red: list[str] = []
    yel: list[str] = []
    notes: list[str] = []
    n_cpu = os.cpu_count() or 1
    try:
        load1 = os.getloadavg()[0]
    except OSError:
        load1 = 0.0

    if load1 > n_cpu * LOAD_FACTOR:
        yel.append(f"1 分钟负载 {load1:.1f} > {n_cpu}×{LOAD_FACTOR} ⇒ 机器被压满，"
                   "任何耗时判据此刻都不可信（先空载再测）")

    try:
        free_gb = shutil.disk_usage(ROOT).free / 1024**3
        if free_gb < DISK_MIN_GB:
            red.append(f"磁盘可用 {free_gb:.1f} GiB < {DISK_MIN_GB} GiB")
        else:
            notes.append(f"磁盘可用 {free_gb:.0f} GiB")
    except OSError:
        pass

    for j in jobs:
        if j.errors:
            red.append(f"{j.name}：日志出现异常行 → {j.errors[0]}")
        if not (j.alive or j.age <= RECENT_SEC):
            continue
        if j.alive and not j.finished:
            # 与上一次巡检比进度：连续不动 = 疑似卡死
            p_prev = None
            if prev:
                p_prev = (prev.get("jobs", {}).get(j.name, {}) or {}).get("progress")
            cur = j.epoch if j.kind == "epoch" else j.prog
            streak = 0
            if prev:
                streak = int((prev.get("streak", {}) or {}).get(j.name, 0))
            if p_prev is not None and cur == p_prev:
                streak += 1
            else:
                streak = 0
            j.notes.append(f"streak={streak}")
            if streak >= STALL_STREAK:
                red.append(f"{j.name}：进程存活但进度连续 {streak} 次巡检未变"
                           f"（当前 {cur}）⇒ 疑似卡死")
            elif j.log and j.age > j.stall_sec:
                yel.append(f"{j.name}：日志静默 {j.age/60:.0f} min > 阈值 "
                           f"{j.stall_sec/60:.0f} min")
            # val 退化
            if len(j.vals) >= VAL_RISE_STREAK:
                tail = j.vals[-VAL_RISE_STREAK:]
                if all(b > a for a, b in zip(tail, tail[1:])):
                    red.append(f"{j.name}：val 连续 {VAL_RISE_STREAK} 轮上升 {tail}"
                               " ⇒ 价值头退化")
        elif j.alive and j.finished:
            notes.append(f"{j.name}：进程仍在（收口后未退出），日志已收口")
        elif not j.alive and j.age <= RECENT_SEC and not j.finished:
            yel.append(f"{j.name}：进程已退出但日志无收口标记"
                       f"（静默 {j.age/60:.0f} min）⇒ 可能被中断")
        elif not j.alive and j.finished:
            notes.append(f"{j.name}：已正常收口")

    verdict = "RED" if red else ("YELLOW" if yel else "GREEN")
    return verdict, red, yel


def main() -> int:
    ap = argparse.ArgumentParser(description="训练/评测进度健康度巡检（只读）")
    ap.add_argument("--json", action="store_true", help="输出 JSON（自动化用）")
    ap.add_argument("--no-write", action="store_true", help="不追加历史")
    a = ap.parse_args()

    jobs = [fill(j) for j in job_specs()]
    active = [j for j in jobs if j.alive and not j.finished]
    recent = [j for j in jobs if j.age <= RECENT_SEC]
    prev = load_hist()

    verdict, red, yel = judge(jobs, prev)
    if not active and not recent:
        verdict = "IDLE"
        red, yel = [], []

    rec = {
        "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "verdict": verdict,
        "load1": round(os.getloadavg()[0], 2) if hasattr(os, "getloadavg") else None,
        "jobs": {
            j.name: {
                "alive": j.alive, "kind": j.kind, "finished": j.finished,
                "progress": j.epoch if j.kind == "epoch" else j.prog,
                "progress_total": (j.expect if j.kind == "epoch"
                                   else j.prog_total * max(1, len(j.workers))),
                "val": None if j.val != j.val else round(j.val, 4),
                "age_min": None if j.age == float("inf") else round(j.age / 60, 1),
            } for j in jobs if j.alive or j.age <= RECENT_SEC
        },
        "streak": {j.name: int(j.notes[0].split("=")[1]) for j in jobs
                   if j.notes and j.notes[0].startswith("streak=")},
        "red": red, "yellow": yel,
    }

    if a.json:
        print(json.dumps(rec, ensure_ascii=False, indent=2))
    else:
        icon = {"GREEN": "🟢", "YELLOW": "🟡", "RED": "🔴", "IDLE": "⚪"}[verdict]
        print("=" * 92)
        print(f"{icon} 巡检 {rec['ts']}  结论 {verdict}  "
              f"负载 {rec['load1']}  ({os.cpu_count()} 核)")
        print("=" * 92)
        for j in jobs:
            if not (j.alive or j.age <= RECENT_SEC):
                continue
            if j.kind == "epoch" and j.epoch:
                prog = f"ep {j.epoch}" + (f"/{j.expect}" if j.expect else "")
                detail = (f"val {j.val:.4f}"
                          + (f"（best {j.best:.4f}）" if j.best == j.best else ""))
            elif j.kind == "eval" and j.prog_total:
                nw = max(1, len(j.workers))
                full = j.prog_total * nw
                prog = f"{j.prog}/{full} 片（{nw} worker）"
                detail = f"≈{j.prog/full:.0%}" if full else ""
            else:
                prog, detail = "—", ""
            state = "运行中" if j.alive and not j.finished else (
                "已收口" if j.finished else "已退出")
            age = "—" if j.age == float("inf") else f"{j.age/60:.0f} min 前"
            print(f"  {j.name:<12} {state:<5} {prog:<26} {detail:<12} {age}")
        for r in red:
            print(f"  🔴 {r}")
        for y in yel:
            print(f"  🟡 {y}")
        if verdict == "GREEN":
            print("  🟢 全部在推进，无异常")
        if verdict == "IDLE":
            print("  ⚪ 无运行中任务，也无 2 小时内日志")

    if not a.no_write:
        HIST.parent.mkdir(parents=True, exist_ok=True)
        with HIST.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    return {"GREEN": 0, "IDLE": 0, "YELLOW": 1, "RED": 2}[verdict]


if __name__ == "__main__":
    sys.exit(main())
