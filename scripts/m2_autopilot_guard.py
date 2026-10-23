"""M2 总控守护（宿主侧长驻）——解决"总控/训练被连带清理后无人拉起"的问题。

用法（在真实宿主环境、非沙箱内运行）：
    <venv-python> -u scripts/m2_autopilot_guard.py --workers 12 --max-hours 20 --poll-s 60

行为：
  - 每 --check-s 秒检查 reports/_m1_<current>.pid 对应的训练进程是否存活；
  - 同时检查总控 m2_night_autopilot.py 是否存活；
  - 任一缺失 → 打印诊断并**重新拉起总控**（总控自身会按 night_autopilot_state.json
    的 current / dyn_segs 完成接力，不会重头再来）；
  - 绝不再挂已收官段（auto4/auto5）的旧守望，绝不杀训练/判读守望。
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REPORTS = ROOT / "reports"
PY = sys.executable


def ts() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def say(msg: str) -> None:
    line = f"[{ts()}] {msg}"
    print(line, flush=True)
    try:
        with open(REPORTS / "_autopilot_guard.log", "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except Exception:
        pass


def load_state() -> dict:
    p = REPORTS / "night_autopilot_state.json"
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {}


def pid_alive(pid: int | None) -> bool:
    if not pid:
        return False
    try:
        import psutil
        return psutil.pid_exists(pid) and psutil.Process(pid).is_running()
    except Exception:
        try:
            os.kill(pid, 0)
            return True
        except Exception:
            return False


def scan() -> tuple[list[int], list[int]]:
    """返回 (总控pids, 训练pids)。"""
    try:
        import psutil
    except Exception:
        return [], []
    auto, self = [], []
    for q in psutil.process_iter(["pid", "cmdline"]):
        try:
            cl = " ".join(q.info["cmdline"] or [])
        except Exception:
            continue
        if "process_iter" in cl or "m2_autopilot_guard" in cl:
            continue
        if "m2_night_autopilot.py" in cl:
            auto.append(q.info["pid"])
        elif "selfplay.py" in cl and "--log-file" in cl:
            self.append(q.info["pid"])
    return auto, self


def read_pid_file(name: str) -> int | None:
    p = REPORTS / f"_m1_{name}.pid"
    try:
        return int(p.read_text(encoding="utf-8").strip().split()[0])
    except Exception:
        return None


def spawn_autopilot(workers: int, max_hours: float, poll_s: int) -> int | None:
    log = open(REPORTS / "_night_autopilot.out", "ab")
    DETACHED = 0x00000008
    NEWGRP = 0x00000200
    NOWIN = 0x08000000
    try:
        p = subprocess.Popen(
            [PY, "-u", "scripts/m2_night_autopilot.py",
             "--workers", str(workers), "--max-hours", str(max_hours),
             "--poll-s", str(poll_s)],
            cwd=str(ROOT), creationflags=DETACHED | NEWGRP | NOWIN,
            close_fds=True, stdin=subprocess.DEVNULL,
            stdout=log, stderr=subprocess.STDOUT,
        )
        return p.pid
    except Exception as e:  # pragma: no cover
        say(f"❌ 拉起总控失败：{e}")
        return None


def main() -> int:
    ap = argparse.ArgumentParser(description="M2 总控守护")
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--max-hours", type=float, default=20.0)
    ap.add_argument("--poll-s", type=int, default=60)
    ap.add_argument("--check-s", type=int, default=120)
    args = ap.parse_args()

    say("=" * 70)
    say(f"M2 总控守护启动（每 {args.check_s}s 巡检；workers={args.workers}）")
    say("=" * 70)

    t0 = time.time()
    while (time.time() - t0) / 3600.0 < args.max_hours:
        st = load_state()
        seg = st.get("current", "?")
        auto_pids, self_pids = scan()

        if not auto_pids:
            say(f"⚠️ 总控缺失（current={seg}，训练={self_pids}）→ 拉起总控")
            pid = spawn_autopilot(args.workers, args.max_hours, args.poll_s)
            say(f"   已拉起总控 pid={pid}")
            time.sleep(45)
            continue

        if not self_pids:
            say(f"⚠️ 训练缺失（current={seg}，总控={auto_pids}）→ 等总控 900s 保底自愈")
        else:
            pid_file = read_pid_file(seg)
            say(f"✅ 正常 current={seg} 总控={auto_pids} 训练={self_pids} "
                f"pid_file={pid_file} alive={pid_alive(pid_file)}")

        time.sleep(args.check_s)

    say("守护到时退出")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
