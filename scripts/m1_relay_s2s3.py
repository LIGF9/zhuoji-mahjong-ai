#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""M1 训练接力守护：S2 判读完成 → 自动启动 S3。

背景
----
``m1_autopilot.py --from s2 --to s2`` 在 S2 段判读完成后即退出（无论判读是否
通过），原计划是「停在 S3 之前等人工确认」。2026-10-14 用户改口径：

    **判读完成后把结果交给用户，同时立即续跑 S3，不再等人工确认。**

本脚本只做机械动作，绝不触碰正在跑的训练：

1. 每 30 s 轮询，等 S2 的托管进程族（``m1_autopilot.py`` / ``selfplay.py``）
   全部退出 —— 判读（finalize + 基线比较）由 autopilot 自己跑完才会退；
2. 确认 ``models/dushan_m1_s2.pt`` 存在（S3 的 ``--init`` 依赖它，缺了不启动）；
3. 以**完全脱离会话**的方式启动 ``m1_autopilot.py --from s3 --to s3``；
4. 写 ``reports/_relay_s2s3.json`` 作为交接凭证（幂等标记）。

安全约束
--------
- 任何一步失败都**不强杀**任何进程，只记录；
- 最长等 12 h，超时只写警告、不擅自启动；
- 已有 ``_relay_s2s3.json``（本脚本自己或人工写过）→ 直接退出，不重复动作。

用法::

    python scripts/m1_relay_s2s3.py            # 前台（调试用）
    # 生产：由外层以 DETACHED_PROCESS 起，见 --spawn
    python scripts/m1_relay_s2s3.py --spawn    # 自己把守护进程脱离会话拉起后返回
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable

LOG_P = ROOT / "reports" / "_relay_s2s3.log"
DONE_P = ROOT / "reports" / "_relay_s2s3.json"
LOCK_P = ROOT / "reports" / "_relay_s2s3.lock"
STATE_P = ROOT / "reports" / "autopilot_state.json"
JUDGE_P = ROOT / "reports" / "m1_s2_judge.json"
DUEL_P = ROOT / "reports" / "m1_s2_vs_master.json"
MODEL_P = ROOT / "models" / "dushan_m1_s2.pt"
S3_PID_P = ROOT / "reports" / "_m1_s3.pid"

POLL_S = 30
MAX_WAIT_S = 12 * 3600
NEXT_SEG = "s3"
WORKERS = 12

# Windows 进程创建标志（非 Windows 上为 0）
DETACHED_PROCESS = 0x00000008
CREATE_NEW_PROCESS_GROUP = 0x00000200
_FLAGS = (DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP) if sys.platform == "win32" else 0


def log(msg: str) -> None:
    line = f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {msg}"
    print(line, flush=True)
    LOG_P.parent.mkdir(parents=True, exist_ok=True)
    with LOG_P.open("a", encoding="utf-8") as fh:
        fh.write(line + "\n")


def acquire_singleton() -> bool:
    """单例锁：同一时刻只允许一个接力守护。

    没有锁的话，两个守护会在 S2 收工那一刻**同时**通过放行条件，各自起一个
    autopilot → 两个 selfplay 抢 CPU/内存，整段指标全废。

    实现：先探活已有锁（进程已死 / 不是本脚本 → 视为残留，清掉），再用
    ``O_CREAT|O_EXCL`` **原子创建**（并发启动时只有一个能拿到）。
    """
    import psutil
    if LOCK_P.exists():
        try:
            pid = int(LOCK_P.read_text(encoding="utf-8").strip())
            if psutil.pid_exists(pid):
                cl = " ".join(psutil.Process(pid).cmdline() or [])
                if "m1_relay_s2s3.py" in cl:
                    log(f"已有守护在跑（pid={pid}）→ 本实例退出，不重复动作")
                    return False
            log(f"发现残留锁（pid={pid} 已不在）→ 清理后接管")
        except Exception:                        # noqa: BLE001
            log("锁文件不可解析 → 清理后接管")
        try:
            LOCK_P.unlink()
        except Exception:                        # noqa: BLE001
            pass
    try:
        fd = os.open(LOCK_P, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        log("锁文件刚被别的实例创建 → 本实例退出")
        return False
    os.write(fd, str(os.getpid()).encode("utf-8"))
    os.close(fd)
    return True


def procs_matching(needles: list[str]) -> list[tuple[int, str]]:
    try:
        import psutil
    except ImportError:
        log("⚠️ psutil 不可用，进程检测退化为『只看产物』")
        return []
    out: list[tuple[int, str]] = []
    for p in psutil.process_iter(["pid", "cmdline"]):
        try:
            cl = " ".join(p.info.get("cmdline") or [])
        except Exception:                        # noqa: BLE001
            continue
        if any(n in cl for n in needles):
            out.append((p.info["pid"], cl))
    return out


def s2_verdict_state() -> dict:
    """读 autopilot 状态里 s2 段的决策记录（空 dict = 还没判读）。"""
    if not STATE_P.exists():
        return {}
    try:
        st = json.loads(STATE_P.read_text(encoding="utf-8"))
    except Exception:                            # noqa: BLE001
        return {}
    return ((st.get("segments") or {}).get("s2") or {})


def write_done(payload: dict) -> None:
    payload = dict(payload)
    payload["when"] = datetime.now().isoformat(timespec="seconds")
    DONE_P.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def launch_next() -> int | None:
    """启动下一段托管。返回 pid（失败 None）。"""
    cmd = [PY, "-u", str(ROOT / "scripts" / "m1_autopilot.py"),
           "--from", NEXT_SEG, "--to", NEXT_SEG, "--workers", str(WORKERS)]
    out_p = ROOT / "reports" / f"_autopilot_{NEXT_SEG}.out"
    fh = out_p.open("a", encoding="utf-8")
    fh.write(f"\n===== relay 启动 {NEXT_SEG} @ {datetime.now():%Y-%m-%d %H:%M:%S} =====\n")
    fh.write(" ".join(cmd) + "\n")
    fh.flush()
    p = subprocess.Popen(cmd, cwd=str(ROOT), stdout=fh, stderr=subprocess.STDOUT,
                         creationflags=_FLAGS, close_fds=True)
    log(f"{NEXT_SEG} 托管已起 pid={p.pid}（日志 {out_p.name}）")
    return p.pid


#: 托管进程族消失后，再等这么久才放行（避免恰好在 autopilot 退出的瞬间抢先）
IDLE_CONFIRM_S = 90


def wait_for_s2() -> dict:
    """轮询直到 S2 收工。返回交接信息。

    放行条件（**不依赖判读产物**，因为 autopilot 退出即代表它把
    finalize + 基线比较 + 决策三步都跑完了）：

    1. ``m1_autopilot.py`` / ``selfplay.py`` 全部消失，且已稳定 ``IDLE_CONFIRM_S`` 秒；
    2. ``models/dushan_m1_s2.pt`` 存在（S3 的 ``--init`` 依赖它）。

    判读产物只作为**证据**记录进凭证，不作为门槛 —— 万一 finalize 崩了，
    也不该把用户卡在原地干等（会在凭证里标 ``judge_missing`` 留痕）。
    """
    t0 = time.time()
    last = 0.0
    idle_since: float | None = None
    while time.time() - t0 < MAX_WAIT_S:
        running = procs_matching(["m1_autopilot.py", "selfplay.py"])
        st = s2_verdict_state()
        has_model = MODEL_P.exists()
        el = time.time() - t0

        idle_since = None if running else (idle_since or time.time())
        idle_for = 0.0 if idle_since is None else time.time() - idle_since

        if idle_since is not None and idle_for >= IDLE_CONFIRM_S:
            if not has_model:
                if el - last >= 600:
                    last = el
                    log(f"⚠️ 托管已退出 {idle_for/60:.1f} min，但 models/dushan_m1_s2.pt "
                        f"仍不存在 —— 不启动 {NEXT_SEG}，继续等（最长 {MAX_WAIT_S/3600:.0f} h）")
                time.sleep(POLL_S)
                continue
            judged = bool(st.get("checks"))
            log(f"S2 收工确认：无托管进程已 {idle_for:.0f}s；判读记录={'有' if judged else '无'}"
                f"（judge.json={'有' if JUDGE_P.exists() else '无'}）；"
                f"模型 dushan_m1_s2.pt=有 → 放行 {NEXT_SEG}")
            return dict(ok=True, waited_s=round(el), idle_confirm_s=round(idle_for),
                        s2_status=st.get("status"), s2_verdict=(st.get("verdict") or {}).get("ok"),
                        judge_missing=not judged, duel=st.get("duel_vs_master"))

        if el - last >= 600:                     # 每 10 分钟留一条心跳
            last = el
            log(f"等待中 {el/3600:.2f} h：托管进程 {len(running)} 个；"
                f"判读记录={'有' if st.get('checks') else '无'}；"
                f"judge.json={'有' if JUDGE_P.exists() else '无'}；"
                f"模型={'有' if has_model else '无'}")
        time.sleep(POLL_S)

    log(f"⚠️ 等待超过 {MAX_WAIT_S/3600:.0f} h，未满足放行条件 —— 不擅自启动 {NEXT_SEG}")
    return dict(ok=False, reason="timeout")


def main() -> int:
    ap = argparse.ArgumentParser(description="S2→S3 接力守护")
    ap.add_argument("--spawn", action="store_true",
                    help="把本守护以脱离会话的方式拉起后立即返回（生产用法）")
    ap.add_argument("--probe", action="store_true",
                    help="只打印当前判定状态后退出（不等待、不启动任何东西）")
    args = ap.parse_args()

    if args.probe:
        running = procs_matching(["m1_autopilot.py", "selfplay.py"])
        st = s2_verdict_state()
        print(json.dumps({
            "托管进程": [r[0] for r in running],
            "s2_状态": st.get("status"),
            "s2_判读记录": bool(st.get("checks")),
            "s2_四条判据": [(c.get("id"), c.get("ok")) for c in (st.get("checks") or [])],
            "judge.json": JUDGE_P.exists(),
            "vs_master.json": DUEL_P.exists(),
            "models/dushan_m1_s2.pt": MODEL_P.exists(),
            "交接凭证": DONE_P.exists(),
        }, ensure_ascii=False, indent=2))
        return 0

    if args.spawn:
        fh = (ROOT / "reports" / "_relay_s2s3_boot.out").open("a", encoding="utf-8")
        fh.write(f"\n===== relay spawn @ {datetime.now():%Y-%m-%d %H:%M:%S} =====\n")
        fh.flush()
        p = subprocess.Popen([PY, "-u", str(Path(__file__).resolve())],
                             cwd=str(ROOT), stdout=fh, stderr=subprocess.STDOUT,
                             creationflags=_FLAGS, close_fds=True)
        print(f"relay 守护已脱离会话启动 pid={p.pid}")
        return 0

    log("=" * 60)
    log("接力守护启动：等 S2 判读完成 → 自动启动 S3")

    if not acquire_singleton():
        return 0

    try:
        return _serve()
    finally:
        try:                                     # 正常收工才释放；被强杀时靠探活接管
            LOCK_P.unlink()
        except Exception:                        # noqa: BLE001
            pass


def _serve() -> int:
    if DONE_P.exists():
        log(f"交接凭证已存在（{DONE_P}），不重复动作。内容："
            + DONE_P.read_text(encoding="utf-8")[:300].replace("\n", " "))
        return 0

    # 幂等：已有「同一段」的托管在跑就别再起一个。
    # ⚠️ 不能拿 selfplay 进程当信号 —— S2 正在跑时它必然存在（命令行含 --init），
    #    那样会让本守护一启动就误判为「已经在跑」而直接跳过。
    busy = procs_matching([f"m1_autopilot.py --from {NEXT_SEG}"])
    if busy:
        log(f"检测到 {NEXT_SEG} 相关进程已在跑 {[b[0] for b in busy]} → 只登记，不重复启动")
        write_done(dict(action="skip", reason="already_running",
                        pids=[b[0] for b in busy]))
        return 0

    info = wait_for_s2()
    if not info.get("ok"):
        write_done(dict(action="none", reason=info.get("reason"), **info))
        return 1

    pid = launch_next()
    if pid is None:
        write_done(dict(action="failed", reason="spawn_error", **info))
        return 1

    write_done(dict(action="launched_s3", s3_pid=pid, **info))
    log(f"交接完成：{NEXT_SEG} pid={pid}；凭证 {DONE_P.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
