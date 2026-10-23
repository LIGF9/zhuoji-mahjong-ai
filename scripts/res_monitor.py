"""长跑期间的资源监视（CPU / 内存 / 目标进程 RSS），写 CSV 便于事后画图。

为什么需要
----------
2026-10-10 的 C3 被系统空闲睡眠冻结 22.2 小时而日志里 ``iter`` 编号连续 —— 事后
只能靠 ``total`` 字段的数量级跳变才发现。除了 ``keep_awake.py`` 主动防休眠，
还需要一个**独立于训练进程**的旁路记录：训练进程本身被挂起时，它写不出任何东西，
但本脚本仍会按真实 wall clock 打点。两者对照即可定位"被冻结"区间。

输出 CSV 列：
``ts, iso, cpu_pct, mem_used_gb, mem_pct, targets, max_rss_gb, tree_rss_gb, procs``

``max_rss_gb`` 只看主进程；``tree_rss_gb`` 是**目标进程树**（含 spawn 出来的 worker）
的 RSS 之和 —— 判「训练到底吃了多少内存」必须看后者。

用法::

    python scripts/res_monitor.py                       # 每 60s 一次，写入默认路径
    python scripts/res_monitor.py --interval 30
    python scripts/res_monitor.py --out reports/_res_monitor.csv
    python scripts/res_monitor.py --idle-limit 5        # 连续 5 min 无目标进程则退出
"""
from __future__ import annotations

import argparse
import csv
import time
from datetime import datetime
from pathlib import Path

import psutil

ROOT = Path(__file__).resolve().parents[1]

#: 与 keep_awake.py 保持一致：这些脚本被认为"是长跑本体"
TARGETS = ("after_a4_plan_c.py", "after_night_a4.py", "shift_plan_a_s12.py",
           "selfplay.py", "score_checkpoints.py", "night_plan_b_stage2.py",
           "plan_b_drop_p1b.py")

OUT = ROOT / "reports" / "_res_monitor.csv"


def targets() -> list:
    """返回目标进程列表 [{"pid","name","rss_gb","tree_gb"}]。

    ``tree_gb`` = 该进程自身 + **全部子孙进程**的 RSS 之和。

    为什么必须算子孙（2026-10-12 实测教训）
    --------------------------------------
    向量化路径用 ``multiprocessing`` 的 spawn 上下文起 worker。Windows 下子进程的
    cmdline 是 ``python.exe -c "from multiprocessing.spawn import spawn_main; ..."``，
    ``cl[1]`` 变成 ``-c`` —— **一个都匹配不到**。于是旧版监控在系统已用 30.13 GB 时
    只认到主进程 3.32 GB，A5 内存判据实际上失去了分辨力（既算进了别人的占用，
    又漏掉了自己的 12 个 worker）。
    """
    found = []
    for p in psutil.process_iter(["cmdline", "memory_info"]):
        cl = list(p.info.get("cmdline") or [])
        # 脚本路径不一定在 cl[1]：`python -u scripts/selfplay.py` 会让 cl[1]="-u"，
        # 旧版只看 cl[1] 于是**连主进程都漏掉**（2026-10-12 预检实测 targets=0）。
        # 改为扫描 cl[1:] 里第一个以目标脚本名结尾的段。
        # 注意 spawn 出来的 worker 命令行是 `-c "from multiprocessing.spawn import ..."`，
        # 不含脚本名 —— 它们不靠这条匹配，而是由下面的 children(recursive=True) 计入。
        hit = next((a for a in cl[1:] if any(a.endswith(t) for t in TARGETS)), None)
        if hit is None:
            continue
        rss = getattr(p.info.get("memory_info"), "rss", 0) or 0
        tree = rss
        try:
            for kid in p.children(recursive=True):
                try:
                    tree += kid.memory_info().rss or 0
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
        found.append({"pid": p.pid, "name": Path(hit).name,
                      "rss_gb": rss / 1024 ** 3,
                      "tree_gb": tree / 1024 ** 3})

    # 去重：Windows 上 `python -u x.py` 会出现**两级同 cmdline** 的进程
    # （外层 shim RSS≈0 + 真身），两者的树互相包含 —— 直接 sum(tree_gb) 会虚高近一倍。
    # 只保留"树根"（祖先不在 found 集合里的那些）。
    if len(found) > 1:
        pids = {p["pid"] for p in found}
        roots = []
        for p in found:
            try:
                has_found_ancestor = any(a.pid in pids
                                         for a in psutil.Process(p["pid"]).parents())
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                has_found_ancestor = False
            if not has_found_ancestor:
                roots.append(p)
        found = roots
    return found


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--interval", type=float, default=60.0, help="打点间隔（秒）")
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--idle-limit", type=float, default=5.0,
                    help="连续多少分钟看不到目标进程就退出")
    ap.add_argument("--max-minutes", type=float, default=24 * 60,
                    help="最长运行分钟数（兜底，防孤儿进程）")
    args = ap.parse_args()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    new = not out.exists()
    fh = open(out, "a", encoding="utf-8", newline="")
    w = csv.writer(fh)
    if new:
        w.writerow(["ts", "iso", "cpu_pct", "mem_used_gb", "mem_pct",
                    "targets", "max_rss_gb", "tree_rss_gb", "procs"])
        fh.flush()

    psutil.cpu_percent(interval=None)      # 丢弃首次采样（无基准）
    t0 = time.time()
    idle = 0.0
    print(f"[res_monitor] 启动 -> {out}（间隔 {args.interval:.0f}s）", flush=True)

    while True:
        procs = targets()
        vm = psutil.virtual_memory()
        cpu = psutil.cpu_percent(interval=None)
        max_rss = max((p["rss_gb"] for p in procs), default=0.0)
        tree_rss = sum(p["tree_gb"] for p in procs)
        now = time.time()
        w.writerow([
            f"{now:.0f}",
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            f"{cpu:.1f}", f"{vm.used / 1024 ** 3:.2f}", f"{vm.percent:.1f}",
            len(procs), f"{max_rss:.2f}", f"{tree_rss:.2f}",
            "|".join(f"{p['name']}#{p['pid']}:{p['rss_gb']:.2f}G"
                     f"(tree {p['tree_gb']:.2f}G)" for p in procs),
        ])
        fh.flush()

        if procs:
            idle = 0.0
        else:
            idle += args.interval
            if idle >= args.idle_limit * 60:
                print("[res_monitor] 目标进程已全部退出，收工", flush=True)
                break
        if (now - t0) / 60 > args.max_minutes:
            print("[res_monitor] 达到最长运行时长，收工", flush=True)
            break
        time.sleep(args.interval)

    fh.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
