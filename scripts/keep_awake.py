"""训练期间阻止系统空闲休眠。

用法: python scripts/keep_awake.py [最长小时数]

原理: Windows SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED)，
只要本进程存活，系统就不会因空闲进入睡眠/待机（屏幕仍可正常关闭）。
到时间后自动调用 SetThreadExecutionState(ES_CONTINUOUS) 释放。
"""
from __future__ import annotations

import ctypes
import sys
import time

ES_CONTINUOUS = 0x80000000
ES_SYSTEM_REQUIRED = 0x00000001


def _set(flags: int) -> bool:
    try:
        ctypes.windll.kernel32.SetThreadExecutionState(flags)
        return True
    except Exception:  # pragma: no cover - 非 Windows
        return False


def main() -> int:
    hours = float(sys.argv[1]) if len(sys.argv) > 1 else 7.0
    if not _set(ES_CONTINUOUS | ES_SYSTEM_REQUIRED):
        print("[awake] 无法设置执行状态（非 Windows？），退出")
        return 1
    print(f"[awake] 防休眠守护已启用，最长 {hours:g} 小时（{time.strftime('%H:%M:%S')}）", flush=True)
    deadline = time.time() + hours * 3600
    while time.time() < deadline:
        time.sleep(60)
        _set(ES_CONTINUOUS | ES_SYSTEM_REQUIRED)   # 周期性续期，防被其他程序覆盖
    _set(ES_CONTINUOUS)
    print(f"[awake] 守护退出，已释放休眠限制（{time.strftime('%H:%M:%S')}）", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
