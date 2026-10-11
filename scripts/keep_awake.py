"""训练期间的防休眠守护（Windows）。

**为什么需要这个**：2026-10-10 的 C3 训练在 iter 24 之后被系统睡眠**冻结了 22.2 小时**
（日志里 ``total`` 从 328s 直接跳到 80196s，而 iter 编号是连续的 —— 说明计算被挂起、
wall clock 继续走）。``after_a4_plan_c.py`` 缺了 A4 脚本里那段防休眠代码，于是整条
「A4 → C3」链路在无人值守时静默停摆。

**做法**：``SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED)`` 阻止空闲睡眠；
守护进程每 60 s 检查目标进程是否还活着，**它们全部退出后自动清除标记**并退出 ——
不会永久改变机器的电源行为。

用法：``python scripts/keep_awake.py``
"""
import ctypes
import sys
import time

import psutil

ES_CONTINUOUS = 0x80000000
ES_SYSTEM_REQUIRED = 0x00000001

#: 需要保护的进程（按命令行第 2 段的尾名匹配，避免匹配到本脚本自己）
TARGETS = ("after_a4_plan_c.py", "after_night_a4.py", "shift_plan_a_s12.py",
           "selfplay.py", "score_checkpoints.py", "night_plan_b_stage2.py")

#: 连续多少分钟看不到任何目标进程就收工
IDLE_LIMIT = 5


def set_state(flags: int) -> int:
    """设置执行状态。返回 0 表示失败。"""
    return int(ctypes.windll.kernel32.SetThreadExecutionState(
        ctypes.c_uint(flags)))


def targets_alive() -> list:
    alive = []
    for p in psutil.process_iter(["cmdline"]):
        cl = list(p.info.get("cmdline") or [])
        # cl[1] 是脚本路径；只看第 2 段，绝不匹配本守护进程自己
        if len(cl) > 1 and any(cl[1].endswith(t) for t in TARGETS):
            alive.append(p.pid)
    return alive


def main() -> int:
    print("[keep_awake] 启动：阻止系统空闲睡眠（ES_CONTINUOUS | ES_SYSTEM_REQUIRED）",
          flush=True)
    if set_state(ES_CONTINUOUS | ES_SYSTEM_REQUIRED) == 0:
        print("[keep_awake] 警告：SetThreadExecutionState 失败，防休眠未生效", flush=True)
    idle = 0
    try:
        while True:
            alive = targets_alive()
            if alive:
                idle = 0
            else:
                idle += 1
                if idle >= IDLE_LIMIT:
                    print("[keep_awake] 目标进程已全部退出，恢复电源行为", flush=True)
                    break
            time.sleep(60)
    except KeyboardInterrupt:
        print("[keep_awake] 收到中断", flush=True)
    finally:
        set_state(ES_CONTINUOUS)
        print("[keep_awake] 已退出（电源标记已清除）", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
