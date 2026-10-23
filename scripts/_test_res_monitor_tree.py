"""验证 res_monitor.targets() 能统计 spawn 出来的 worker 内存。

背景：向量化路径用 multiprocessing spawn 起 worker，Windows 下子进程 cmdline
是 `python.exe -c "from multiprocessing.spawn import spawn_main; ..."`，
按 `cmdline[1]` 匹配脚本名的旧写法**一个 worker 都抓不到**。
本测试：父进程故意匹配（把 TARGETS 换成自己的脚本名），起 4 个 spawn worker
各占 200 MB，断言 tree_gb 明显大于 max_rss_gb（即 worker 被计入）。

    python scripts/_test_res_monitor_tree.py
"""
from __future__ import annotations

import multiprocessing as mp
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import res_monitor  # noqa: E402

MB = 1024 * 1024
BALLAST_MB = 200
WORKERS = 4
_ballast = []          # worker 内持有，撑起 RSS


def _hold(_):
    _ballast.append(bytearray(BALLAST_MB * MB))      # 触碰页面 → RSS 真实增长
    time.sleep(20)
    return 1


def main() -> int:
    res_monitor.TARGETS = (Path(__file__).name,)      # 让父进程能被 targets() 认出来
    ctx = mp.get_context("spawn")
    kids = 0
    with ctx.Pool(processes=WORKERS) as pool:
        pool.map_async(_hold, range(WORKERS))
        time.sleep(6)                                  # 等 worker 分配完内存
        procs = res_monitor.targets()
        if procs:                                      # 必须在池还活着时数子孙
            try:
                kids = len(res_monitor.psutil.Process(procs[0]["pid"])
                           .children(recursive=True))
            except Exception:                          # noqa: BLE001
                pass

    if not procs:
        print("FAIL：targets() 没认出父进程")
        return 1
    me = procs[0]

    print(f"父进程 RSS      = {me['rss_gb']:.2f} GB")
    print(f"进程树 RSS      = {me['tree_gb']:.2f} GB   （子孙 {kids} 个）")
    expect = WORKERS * BALLAST_MB / 1024
    ok_kids = kids >= WORKERS
    ok_tree = me["tree_gb"] > me["rss_gb"] + expect * 0.6
    print(f"期望 worker 占用 ≈ {expect:.2f} GB（{WORKERS} × {BALLAST_MB} MB）")
    print(f"  子孙数 ≥ {WORKERS}          : {'PASS' if ok_kids else 'FAIL'}")
    print(f"  tree >> main（worker 被计入）: {'PASS' if ok_tree else 'FAIL'}")
    return 0 if (ok_kids and ok_tree) else 1


if __name__ == "__main__":
    raise SystemExit(main())
