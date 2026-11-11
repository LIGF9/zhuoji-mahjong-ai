#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""训练/评测进度健康度巡检（**只读**，零干扰；任务级自动起止）。

为什么另写一个
--------------
`tools/health_watch.py` 是 **PPO 轮专用**：只读 `runs/night_logs/ppo_p*.log` 里的
`[metrics]` 行，判据围绕 kl/clipfrac/ent/gnorm。ExIt 与蒸馏链路（`train_exit.py`、
`tools/distill_slim.py`）以及闸门评测（`eval_pairs.py`）根本不打 `[metrics]`，
在那张表里是空的。本工具补的是**任务级**健康度：还在不在跑、有没有在推进、
val 是不是在退化、有没有静默挂掉、**是不是已经跑完了**。

任务的自动起止（2026-11-10 晚）
------------------------------
此前任务清单是硬编码的——新开一个评测要改代码，跑完的旧任务永远挂在表里。
现在改成两级：

1. **登记表** `runs/patrol/jobs.json`：`--register` 一条命令登记（也可手工编辑）；
2. **自动发现**：巡检时 `pgrep -fl` 扫已知脚本，从进程命令行解析
   `--out` / `--log` / `--seeds` / `--epochs`，自动补登记（已有登记则跳过）。

**自动结束判定**（不再依赖人去认"跑完没有"）三判据取或：
  日志收口标记（`[summary]`/`[early-stop]`/`[wN] done`）· 进度跑满 · 产物文件存在
进程已退出 + 任一判据 → `ended`（写 `ended_at`，此后不再参与 RED/YELLOW 判定）；
进程已退出 + 无判据 → 第一次记 `ending`（YELLOW，可能刚好在写盘），
连续第二次仍无 → `failed`（RED）。

**全部任务结束即收工**：所有非常驻任务都 `ended`/`failed` 时，输出
`session_done: true`，退出码 **3**，自动化据此暂停自己——不再每小时空转。

**预计结束时间（ETA）**
------------------------
登记表可带 ``expect_sec``（预登记时用冒烟实测锁定的预期墙钟秒）：有它时
ETA = ``started + expect_sec``，因为冷启动阶段只有几个粗粒度的进度点，
拿它外推会把 1.5 h 估成 13 h（2026-11-11 实测踩到）。没有 ``expect_sec`` 时
退化为「最近一次进度差分 → 启动以来平均速率」。两种口径都**只用于汇报**，
不影响任何闸门判定；进度本身始终用日志里真实的打点数做卡死判定。

三类任务各自认什么
------------------
* ``epoch``（蒸馏/ExIt 训练）：``[ep NN] train … val … total …s`` → 轮数、val 曲线；
* ``eval``（配对评测）：``[wN] k/T`` → 分片进度；``[summary]``/``[done]`` → 收口；
* ``generic``：只看存活 + 日志新鲜度。

判定（**巡检阈值，不是预登记判据**）
------------------------------------
这些阈值只用于"要不要叫人看一眼"，**绝不参与任何闸门判定**，也不改任何 `δ`/池/seed：

* **RED**：进程活着但连续 2 次巡检进度不动；日志出现 Traceback/Killed/OOM；
  磁盘可用 < 5 GiB；val 连续 3 轮上升（价值头退化）；任务 `failed`。
* **YELLOW**：日志静默超过该任务的 `stall_sec`；负载 > 核数×1.5；
  进程已退出但还没有收口证据（`ending`，等下一次确认）。
* **GREEN**：其余。
* **IDLE**：没有任何存活任务，也没有 2 小时内的日志（无事可做 ≠ 异常）。

为什么不用 ps / psutil
----------------------
本机沙箱拒绝 ``ps``（operation not permitted），也没装 psutil。所以存活用 ``pgrep -f``、
推进用"解析计数 + 与上一条历史记录差分"。这反而更稳：日志读数不受调度抖动影响。

用法::

    .venv/bin/python tools/patrol_health.py                 # 人读表 + 结论
    .venv/bin/python tools/patrol_health.py --json          # 机器读（自动化用）
    .venv/bin/python tools/patrol_health.py --no-write      # 只看，不写历史
    .venv/bin/python tools/patrol_health.py --list          # 看登记表
    .venv/bin/python tools/patrol_health.py --register \\
        --name "M1 评测" --kind eval --log runs/gates/x.log \\
        --proc eval_pairs.py --expect 60 --artifact runs/gates/x.jsonl

历史落在 ``runs/patrol/health.jsonl``，每行一次巡检；差分只取上一条，不回溯。
登记表落在 ``runs/patrol/jobs.json``。
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
REGISTRY = ROOT / "runs" / "patrol" / "jobs.json"

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
# 收口标记：注意 eval_pairs 正常收口时最后一行是 `[summary] 判定: …`，
# 只有中间的 `[wN] done N seeds` 带 done —— 所以必须扫**全部**行，
# 不能只看最后一行（否则每轮跑完都误报"可能被中断"）。
DONE_MARKS = ("[done]", "[early-stop]", "[summary]", "] done ", "[w0] done",
              "ALL PASS", "HAS FAIL")

# 自动发现：脚本名 → (kind, 默认日志, 是否常驻)。命令行里有 --out/--log 时以命令行为准。
AUTO_DISCOVER = {
    "distill_slim.py": ("epoch", "runs/slim_s.log", False),
    "train_exit.py": ("epoch", "runs/night_logs/exit_latest.log", False),
    "train_ppo.py": ("epoch", "runs/night_logs/ppo_latest.log", False),
    "eval_pairs.py": ("eval", "", False),
    "export_onnx_cache.py": ("generic", "", False),
    "dushan_server.py": ("generic", "", True),      # 常驻服务，不阻断收工
}


@dataclass
class Job:
    name: str
    kind: str                       # epoch | eval | generic
    proc_pat: str
    log: Path | None
    expect: int = 0                 # epoch 任务=预期轮数；eval=每 worker 总片数
    expect_sec: int = 0             # 预登记的预期墙钟秒；>0 时 ETA 走日程口径
    stall_sec: int = 900
    artifact: Path | None = None    # 产物（.pt / .jsonl）；存在即强收口证据
    persistent: bool = False        # 常驻任务（服务），不参与"全部结束"判定
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
    artifact_ok: bool = False
    errors: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    # 生命周期（由巡检写入登记表）
    status: str = "running"         # running | ending | ended | failed | persistent
    started: str = ""
    ended: str = ""
    # ETA（只读推算，不参与任何闸门判定）
    eta: str = ""                   # 预计结束的本地时间 "YYYY-MM-DD HH:MM"
    eta_min: float | None = None    # 剩余分钟

    def label(self) -> str:
        return self.name or self.proc_pat


def pgrep_lines(pat: str) -> list[str]:
    """`pgrep -fl` 的行（pid + cmdline）。沙箱没有 ps，这是唯一能拿到命令行的路子。"""
    try:
        r = subprocess.run(["pgrep", "-fl", pat], capture_output=True, text=True)
    except (OSError, subprocess.SubprocessError):
        return []
    out = []
    for ln in r.stdout.splitlines():
        head = ln.split(None, 1)
        if not head or not head[0].isdigit():
            continue
        if int(head[0]) == os.getpid():     # 别把自己算进去
            continue
        out.append(ln)
    return out


def pgrep_alive(pat: str) -> bool:
    return bool(pgrep_lines(pat))


def read_tail(path: Path, limit: int = 400) -> list[str]:
    try:
        with path.open("r", encoding="utf-8", errors="replace") as f:
            return f.read().splitlines()[-limit:]
    except OSError:
        return []


def _arg(cmd: str, flag: str) -> str:
    m = re.search(rf"{re.escape(flag)}[=\s]+(\S+)", cmd)
    return m.group(1).strip("'\"") if m else ""


def discover() -> list[Job]:
    """扫已知脚本的存活进程，从命令行补出日志 / 产物 / 规模。"""
    out: list[Job] = []
    seen: set[str] = set()
    for script, (kind, dlog, persistent) in AUTO_DISCOVER.items():
        for ln in pgrep_lines(script):
            cmd = ln.split(None, 1)[1] if " " in ln else ""
            out_p = _arg(cmd, "--out")
            log_p = _arg(cmd, "--log")
            if not log_p:
                log_p = (out_p.rsplit(".", 1)[0] + ".log") if out_p else dlog
            if not log_p:
                if not persistent:
                    continue    # eval_pairs 没 --out 的进程无法定位日志，跳过
                out.append(Job(name=script.replace(".py", ""), kind=kind,
                               proc_pat=script, log=None, persistent=True))
                continue
            log = Path(log_p) if Path(log_p).is_absolute() else ROOT / log_p
            if not log.exists():
                continue
            name = log.stem
            if name in seen:
                continue
            seen.add(name)
            expect = 0
            if kind == "epoch":
                expect = int(_arg(cmd, "--epochs") or 0)
            elif kind == "eval":
                expect = int(_arg(cmd, "--seeds") or 60)
            art = None
            if out_p:
                ap = Path(out_p) if Path(out_p).is_absolute() else ROOT / out_p
                art = ap if kind != "eval" else ap
            out.append(Job(name=name, kind=kind,
                           # proc_pat 必须**唯一到任务**：只写 "eval_pairs.py" 会让
                           # 任意一个评测进程都让别的评测显示"运行中"（实测踩到：
                           # 闸门跑完后仍报 running）。--out 文件名在当前仓库唯一。
                           proc_pat=(Path(out_p).name if out_p
                                     else (log.name if log else script)),
                           log=log, expect=expect, artifact=art,
                           stall_sec=900 if kind == "epoch" else 3600,
                           persistent=persistent))
    return out


# ---------------------------------------------------------------------------
# 登记表
# ---------------------------------------------------------------------------
def load_registry() -> list[dict]:
    try:
        data = json.loads(REGISTRY.read_text(encoding="utf-8"))
        rows = data.get("jobs", []) if isinstance(data, dict) else []
        return [r for r in rows if isinstance(r, dict) and r.get("name")]
    except (OSError, ValueError):
        return []


def save_registry(rows: list[dict]) -> None:
    REGISTRY.parent.mkdir(parents=True, exist_ok=True)
    REGISTRY.write_text(
        json.dumps({"_note": "巡检任务登记表（由 tools/patrol_health.py 维护）",
                    "jobs": rows}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")


def _p(v) -> Path | None:
    if not v:
        return None
    p = Path(str(v))
    return p if p.is_absolute() else ROOT / p


def from_row(row: dict) -> Job:
    return Job(name=str(row["name"]), kind=str(row.get("kind") or "generic"),
               proc_pat=str(row.get("proc_pat") or ""),
               log=_p(row.get("log")),
               expect=int(row.get("expect") or 0),
               expect_sec=int(row.get("expect_sec") or 0),
               stall_sec=int(row.get("stall_sec") or 900),
               artifact=_p(row.get("artifact")),
               persistent=bool(row.get("persistent")),
               status=str(row.get("status") or "running"),
               started=str(row.get("started") or ""),
               ended=str(row.get("ended") or ""))


def merge_registry(dry: bool = False) -> list[Job]:
    """登记表 + 自动发现（同名以登记表为准：人写的 expect/artifact 更可信）。"""
    rows = load_registry()
    jobs = [from_row(r) for r in rows]
    have = {j.name for j in jobs}
    added = 0
    for j in discover():
        if j.name in have:
            continue
        # 自动发现的任务补进登记表，保证下次巡检仍认它（进程退出后日志还在）
        rows.append({"name": j.name, "kind": j.kind, "proc_pat": j.proc_pat,
                     "log": str(j.log.relative_to(ROOT)) if j.log else "",
                     "expect": j.expect,
                     "artifact": (str(j.artifact.relative_to(ROOT))
                                  if j.artifact else ""),
                     "stall_sec": j.stall_sec, "persistent": j.persistent,
                     "started": time.strftime("%Y-%m-%dT%H:%M:%S"),
                     "status": "running", "ended": ""})
        jobs.append(j)
        have.add(j.name)
        added += 1
    if added and not dry:
        save_registry(rows)
        print(f"[patrol] 自动登记 {added} 个新任务", file=sys.stderr)
    return jobs


def sync_registry(jobs: list[Job]) -> None:
    """把巡检算出的 status/ended 写回登记表（只改生命周期字段）。"""
    rows = load_registry()
    by = {str(r["name"]): r for r in rows}
    for j in jobs:
        r = by.get(j.name)
        if r is None:
            rows.append({"name": j.name, "kind": j.kind, "proc_pat": j.proc_pat,
                         "log": str(j.log.relative_to(ROOT)) if j.log else "",
                         "expect": j.expect, "expect_sec": j.expect_sec,
                         "artifact": (str(j.artifact.relative_to(ROOT))
                                      if j.artifact else ""),
                         "stall_sec": j.stall_sec, "persistent": j.persistent,
                         "started": j.started or time.strftime("%Y-%m-%dT%H:%M:%S")})
            r = rows[-1]
        r["status"] = j.status
        r["ended"] = j.ended
        if j.started and not r.get("started"):
            r["started"] = j.started
    save_registry(rows)


# ---------------------------------------------------------------------------
# 采集
# ---------------------------------------------------------------------------
def fill(job: Job) -> Job:
    job.alive = pgrep_alive(job.proc_pat) if job.proc_pat else False
    if job.artifact is not None and job.artifact.exists():
        job.artifact_ok = True
        # 骨架式收口文件不能当完成证据：有些驱动脚本启动时先写一个空的
        # done.json，再逐臂回填 arms_done（实测 p0_n32.sh 的 mark "" 就是这样）。
        # 只认「arms_done 覆盖全部臂」，否则进程一退出就会被误判 ended，
        # 进而误触发 session_done / 自动化提前暂停。
        if job.artifact.suffix == ".json":
            try:
                d = json.loads(job.artifact.read_text(encoding="utf-8"))
                arms = d.get("arms_done")
                if arms is not None:
                    job.artifact_ok = {"A", "B", "C"} <= {
                        str(x).upper() for x in arms if x}
            except (OSError, ValueError):
                pass
    if job.log is None:
        return job          # generic 任务（如线上服务）只判存活，不看日志
    if not job.log.exists():
        if job.alive:
            job.errors.append("进程存活但日志文件不存在")
        return job
    job.age = time.time() - job.log.stat().st_mtime
    lines = read_tail(job.log)
    job.last_line = next((ln for ln in reversed(lines) if ln.strip()), "")
    job.finished = any(m in ln for ln in lines for m in DONE_MARKS)

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
    # 进度跑满也算收口（不依赖日志措辞）
    if job.kind == "eval" and job.prog_total:
        if job.prog >= job.prog_total * max(1, len(job.workers)):
            job.finished = True
    if job.kind == "epoch" and job.expect and job.epoch >= job.expect:
        job.finished = True
    return job


def lifecycle(job: Job) -> str:
    """自动判定任务状态（running / ending / ended / failed / persistent）。"""
    if job.persistent:
        return "persistent"
    if job.alive:
        return "running"
    # 进程已退出：三判据取或
    if job.finished or job.artifact_ok:
        return "ended"
    if job.status == "ending":      # 上一次已经"无证据退出"过一次 → 判定失败
        return "failed"
    return "ending"


def scan_orphans(jobs: list[Job], window: float = 3600.0) -> list[str]:
    """近期被写入、未登记、且无收口标记的日志 —— 覆盖「短于巡检间隔」的任务。

    实测踩到：M4（P48g vs masterg）只跑 2 min 39 s，每小时巡检永远抓不到它。
    正常结束无所谓（产物已落盘），但**失败**时它会静默消失、无人知晓。
    这里只提示不阻断：近 1 小时写过 + 不在登记表 + 全文无收口标记 + 无 [summary]。
    """
    known = {str(j.log.resolve()) for j in jobs if j.log}
    out: list[str] = []
    now = time.time()
    for d in (ROOT / "runs" / "gates", ROOT / "runs"):
        if not d.is_dir():
            continue
        for p in sorted(d.glob("*.log")):
            if str(p.resolve()) in known:
                continue
            try:
                age = now - p.stat().st_mtime
            except OSError:
                continue
            if age > window:
                continue
            lines = read_tail(p, 120)
            if any(m in ln for ln in lines for m in DONE_MARKS):
                continue        # 有收口标记 ⇒ 正常结束的短任务，不必提示
            try:
                rel = p.relative_to(ROOT)
            except ValueError:
                rel = p
            out.append(f"{rel}：{age/60:.0f} min 前仍在写但无收口标记，且未登记 ⇒ "
                       f"疑似短任务中断（起止都在巡检间隔内的任务会漏）")
    return out


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


def _ts_epoch(s: str) -> float:
    try:
        return time.mktime(time.strptime(s, "%Y-%m-%dT%H:%M:%S"))
    except (ValueError, TypeError):
        return 0.0


def estimate_eta(jobs: list[Job], now: float | None = None) -> None:
    """预计结束时间：进度差分（优先）→ 启动以来平均速率（兜底）。

    只读：用 ``runs/patrol/health.jsonl`` 最近 48 条里**最早一条进度更小**的记录
    作差分，避免"同一小时重复巡检 / 进度点稀疏"时速率为 0。预估只用于汇报，
    **不参与任何闸门判定，也不改预登记的样本量或判据**。
    """
    now = now or time.time()
    hist: list[dict] = []
    if HIST.exists():
        for ln in read_tail(HIST, 48):
            try:
                hist.append(json.loads(ln))
            except json.JSONDecodeError:
                continue
    for j in jobs:
        if j.status != "running" or j.kind == "generic":
            continue
        # 日程口径优先：预登记时用冒烟实测锁过预期墙钟，比"冷启动的几个粗粒度
        # 进度点"可靠得多（实测首点速率曾把 1.5 h 估成 13 h）。进度仍用真实
        # seed 下界做卡死判定，两者互不干扰。
        if j.expect_sec > 0 and j.started:
            t0 = _ts_epoch(j.started)
            if t0 > 0:
                finish = t0 + j.expect_sec
                j.eta = time.strftime("%Y-%m-%d %H:%M", time.localtime(finish))
                j.eta_min = max(0.0, (finish - now) / 60.0)
                continue
        total = (j.expect if j.kind == "epoch"
                 else (j.expect or j.prog_total * max(1, len(j.workers))))
        cur = j.epoch if j.kind == "epoch" else j.prog
        if not total or cur <= 0 or cur >= total:
            continue
        rate = None
        for h in reversed(hist):
            hj = (h.get("jobs", {}) or {}).get(j.name) or {}
            hp, ht = hj.get("progress"), _ts_epoch(str(h.get("ts", "")))
            if hp is None or ht <= 0 or now - ht <= 60:
                continue
            if hp < cur:
                r = (cur - hp) / (now - ht)
                if r > 0:
                    rate = r
                    break
        if rate is None and j.started:
            t0 = _ts_epoch(j.started)
            if t0 > 0 and now - t0 > 60:
                rate = cur / (now - t0)
        if rate and rate > 0:
            rem = (total - cur) / rate          # 秒
            j.eta_min = rem / 60.0
            j.eta = time.strftime("%Y-%m-%d %H:%M", time.localtime(now + rem))


def judge(jobs: list[Job], prev: dict | None) -> tuple[str, list[str], list[str], list[str]]:
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
        if j.status in ("ended", "failed"):
            # 已定案的任务不再重复告警；failed 只在"首次变 failed"那次报，
            # 之后降为备注（否则一条中断会让巡检永远 RED，掩盖新问题）。
            if j.status == "failed":
                was = ((prev or {}).get("jobs", {}).get(j.name, {}) or {}).get("status")
                if was != "failed":
                    red.append(f"{j.name}：进程退出且无任何收口证据（日志无收口标记、"
                               "产物不存在）⇒ 判定中断，需人工确认")
                else:
                    notes.append(f"{j.name}：已判中断（沿用上次结论）")
            continue
        if j.errors:
            red.append(f"{j.name}：日志出现异常行 → {j.errors[0]}")
        if j.status == "ending":
            yel.append(f"{j.name}：进程已退出但还没有收口证据（日志静默 "
                       f"{j.age/60:.0f} min）⇒ 可能刚好在写盘；下次巡检仍无证据则判中断")
            continue
        if not j.alive and j.age > RECENT_SEC:
            continue
        if j.alive and not j.finished:
            # 与上一次巡检比进度：连续不动 = 疑似卡死。
            # 但"进度还是 0"有两种完全正常的情形——① 刚起、第一个分片块还没跑完
            # （eval_pairs 每 10 个 seed 才打一行）；② 常驻服务（generic）根本没有
            # 进度概念。这两种只在日志确实静默超阈值时才告警，否则每晚会误报。
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
            if streak >= STALL_STREAK and (cur > 0 or j.kind != "generic"):
                sil = j.age if j.age != float("inf") else 0.0
                if cur > 0 or sil > j.stall_sec:
                    red.append(f"{j.name}：进程存活但进度连续 {streak} 次巡检未变"
                               f"（当前 {cur}，日志静默 {sil/60:.0f} min）⇒ 疑似卡死")
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

    verdict = "RED" if red else ("YELLOW" if yel else "GREEN")
    return verdict, red, yel, notes


def main() -> int:
    ap = argparse.ArgumentParser(description="训练/评测进度健康度巡检（只读）")
    ap.add_argument("--json", action="store_true", help="输出 JSON（自动化用）")
    ap.add_argument("--no-write", action="store_true", help="不追加历史、不改登记表")
    ap.add_argument("--list", action="store_true", help="只打印登记表")
    ap.add_argument("--register", action="store_true", help="登记一个任务")
    ap.add_argument("--unregister", default="", help="按名字移除登记")
    ap.add_argument("--name", default="")
    ap.add_argument("--kind", default="eval", choices=("epoch", "eval", "generic"))
    ap.add_argument("--log", default="")
    ap.add_argument("--proc", default="", help="进程匹配串（pgrep -f）")
    ap.add_argument("--expect", type=int, default=0)
    ap.add_argument("--expect-sec", type=int, default=0,
                    help="预登记的预期墙钟秒；>0 时 ETA = 启动时间 + 该值")
    ap.add_argument("--artifact", default="")
    ap.add_argument("--stall-sec", type=int, default=0)
    ap.add_argument("--persistent", action="store_true", help="常驻任务，不阻断收工")
    a = ap.parse_args()

    if a.list:
        rows = load_registry()
        if not rows:
            print("登记表为空（runs/patrol/jobs.json）")
        for r in rows:
            print(f"  {r.get('status','?'):<9} {str(r.get('name')):<24} "
                  f"{str(r.get('kind','')):<8} {str(r.get('log','')):<38} "
                  f"ended={r.get('ended') or '—'}")
        return 0

    if a.register:
        if not a.name:
            ap.error("--register 需要 --name")
        rows = load_registry()
        rows = [r for r in rows if str(r.get("name")) != a.name]
        rows.append({"name": a.name, "kind": a.kind, "proc_pat": a.proc,
                     "log": a.log, "expect": a.expect,
                     "expect_sec": a.expect_sec, "artifact": a.artifact,
                     "stall_sec": a.stall_sec or (900 if a.kind == "epoch" else 3600),
                     "persistent": a.persistent,
                     "started": time.strftime("%Y-%m-%dT%H:%M:%S"),
                     "status": "running", "ended": ""})
        save_registry(rows)
        print(f"已登记 {a.name}（{a.kind}）→ {a.log or '—'}")
        return 0

    if a.unregister:
        rows = [r for r in load_registry() if str(r.get("name")) != a.unregister]
        save_registry(rows)
        print(f"已移除 {a.unregister}")
        return 0

    jobs = [fill(j) for j in merge_registry(dry=a.no_write)]
    # 自动起止：先判 alive/finished，再定 status
    for j in jobs:
        new = lifecycle(j)
        if new in ("ended", "failed") and j.status not in ("ended", "failed"):
            j.ended = time.strftime("%Y-%m-%dT%H:%M:%S")
        j.status = new
        if not j.started and new == "running":
            j.started = time.strftime("%Y-%m-%dT%H:%M:%S")

    act = [j for j in jobs if j.status in ("running", "ending")]
    recent = [j for j in jobs if j.age <= RECENT_SEC and j.status == "running"]
    prev = load_hist()
    estimate_eta(jobs)
    session_eta = max((j for j in jobs if j.eta and j.status == "running"),
                      key=lambda x: x.eta_min or 0, default=None)

    verdict, red, yel, notes = judge(jobs, prev)
    orphans = scan_orphans(jobs)
    if orphans:
        yel = yel + orphans
        if verdict == "GREEN":
            verdict = "YELLOW"
    if not act and not recent:
        verdict = "YELLOW" if orphans else "IDLE"
        red = []
        if not orphans:
            yel = []

    work = [j for j in jobs if not j.persistent]
    session_done = bool(work) and all(j.status in ("ended", "failed") for j in work)
    if not work and not act:
        session_done = True        # 只有常驻服务 → 本批任务已全部结束

    rec = {
        "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "verdict": verdict,
        "session_done": session_done,
        "session_eta": (session_eta.eta if session_eta else None),
        "session_eta_min": (round(session_eta.eta_min, 1)
                            if session_eta and session_eta.eta_min is not None else None),
        "load1": round(os.getloadavg()[0], 2) if hasattr(os, "getloadavg") else None,
        "jobs": {
            j.name: {
                "alive": j.alive, "kind": j.kind, "finished": j.finished,
                "status": j.status, "persistent": j.persistent,
                "progress": j.epoch if j.kind == "epoch" else j.prog,
                "progress_total": (j.expect if j.kind == "epoch"
                                   else j.prog_total * max(1, len(j.workers))),
                "val": None if j.val != j.val else round(j.val, 4),
                "age_min": None if j.age == float("inf") else round(j.age / 60, 1),
                "eta": j.eta or None,
                "eta_min": (round(j.eta_min, 1) if j.eta_min is not None else None),
                "ended": j.ended or None,
            } for j in jobs
        },
        "streak": {j.name: int(j.notes[0].split("=")[1]) for j in jobs
                   if j.notes and j.notes[0].startswith("streak=")},
        "red": red, "yellow": yel, "orphans": orphans,
    }

    if a.json:
        print(json.dumps(rec, ensure_ascii=False, indent=2))
    else:
        icon = {"GREEN": "🟢", "YELLOW": "🟡", "RED": "🔴", "IDLE": "⚪"}[verdict]
        print("=" * 96)
        print(f"{icon} 巡检 {rec['ts']}  结论 {verdict}  "
              f"负载 {rec['load1']}  ({os.cpu_count()} 核)"
              + ("  · 本批任务已全部结束" if session_done else "")
              + (f"  · 预计结束 {session_eta.eta}（剩 {session_eta.eta_min:.0f} min）"
                 if session_eta else ""))
        print("=" * 96)
        for j in jobs:
            if not (j.alive or j.status in ("running", "ending", "failed")
                    or j.age <= RECENT_SEC):
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
            state = {"running": "运行中", "ending": "待确认", "ended": "已结束",
                     "failed": "已中断", "persistent": "常驻"}[j.status]
            age = "—" if j.age == float("inf") else f"{j.age/60:.0f} min 前"
            if j.persistent or j.kind == "generic":
                eta = "—（常驻）"
            elif j.eta:
                eta = f"ETA {j.eta}（剩 {j.eta_min:.0f} min）"
            else:
                eta = "ETA 待测（首个进度点未到）"
            print(f"  {j.name:<14} {state:<5} {prog:<26} {detail:<12} "
                  f"{age:<10} {eta}")
        for r in red:
            print(f"  🔴 {r}")
        for y in yel:
            print(f"  🟡 {y}")
        if verdict == "GREEN":
            print("  🟢 全部在推进，无异常")
        if verdict == "IDLE":
            print("  ⚪ 无运行中任务，也无 2 小时内日志")
        if session_done:
            print("  ✅ 本批任务已全部结束 → 巡检可以暂停（退出码 3）")

    if not a.no_write:
        HIST.parent.mkdir(parents=True, exist_ok=True)
        with HIST.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        sync_registry(jobs)

    if session_done:
        return 3
    return {"GREEN": 0, "IDLE": 0, "YELLOW": 1, "RED": 2}[verdict]


if __name__ == "__main__":
    sys.exit(main())
