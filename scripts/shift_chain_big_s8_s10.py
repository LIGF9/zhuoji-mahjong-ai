"""12 小时训练值班进程：防休眠 + 三代自动接力 + 时间预算自适应。

设计要点
1. 全程持有 SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED)，阻止系统空闲休眠
   （上次 big_s6 曾因系统休眠空转 7.65 小时，这是最大的算力杀手）。
2. 依次训练 big_s8 → big_s9 → big_s10，每代以前一代的最佳 checkpoint 初始化（真正的接力）。
3. 时间预算自适应：每代开跑前按上一代的实测速率（秒/轮）估算剩余时间，
   动态裁剪本轮 iters，保证全链在训练截止时间（start + 12h - RESERVE）前收尾；
   若剩余时间不足以跑满最少轮数，则跳过后续代直接结束（不硬撑）。
4. 只以前一代的 --out 文件存在且日志含「完成。」作为成功判据，失败即中止链条，
   绝不在上一代未结束时并发启动（并发会让训练慢 ~20 倍）。
"""
from __future__ import annotations

import ctypes
import json
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PY = rr"C:\Users\hyzor\.workbuddy\binaries\python\envs\zhuoji\Scripts\python.exe"
SHIFT_LOG = ROOT / "reports" / "shift_chain_big_s8_s10.log"
SUMMARY = ROOT / "reports" / "shift_chain_big_s8_s10_summary.json"

SHIFT_HOURS = 12.0
RESERVE_S = 45 * 60          # 留给收尾对位赛（600 副初筛 + 2400 副终裁）与缓冲
MIN_ITERS = 300              # 单代最少轮数，低于此值不再启动新代
DEFAULT_RATE = 11.0          # 秒/轮（保守初值；首代结束后用实测值替换）

ES_CONTINUOUS = 0x80000000
ES_SYSTEM_REQUIRED = 0x00000001

# 三代接力配置：教师座位占比逐代下调（数据越"纯网"），学习率逐代衰减
GENS = [
    dict(tag="big_s8", init="models/dushan_big_s7.pt", iters=1200,
         lr="1.6e-5", temperature="1.0", teacher_prob="0.18", seed=20264),
    dict(tag="big_s9", init="models/dushan_big_s8.pt", iters=1200,
         lr="1.3e-5", temperature="1.0", teacher_prob="0.14", seed=20265),
    dict(tag="big_s10", init="models/dushan_big_s9.pt", iters=1000,
         lr="1.0e-5", temperature="0.97", teacher_prob="0.10", seed=20266),
]


def awake(on: bool = True) -> bool:
    try:
        ctypes.windll.kernel32.SetThreadExecutionState(
            (ES_CONTINUOUS | ES_SYSTEM_REQUIRED) if on else ES_CONTINUOUS)
        return True
    except Exception:
        return False


def main() -> int:
    log = open(SHIFT_LOG, "a", encoding="utf-8", buffering=1)
    t_start = time.time()
    train_deadline = t_start + SHIFT_HOURS * 3600 - RESERVE_S

    def say(msg: str) -> None:
        print(f"[{time.strftime('%H:%M:%S')}] {msg}", file=log)

    say("—" * 46)
    say(f"值班启动；防休眠={'OK' if awake(True) else '失败'}；"
        f"训练截止={time.strftime('%H:%M:%S', time.localtime(train_deadline))}"
        f"（总窗口 {SHIFT_HOURS:.0f}h，预留 {RESERVE_S // 60} 分钟做对位赛）")

    rate = DEFAULT_RATE
    done = []

    for gi, gen in enumerate(GENS):
        tag = gen["tag"]
        init = ROOT / gen["init"]
        out = ROOT / "models" / f"dushan_{tag}.pt"     # 与历代 dushan_big_s*.pt 同名法
        hist = ROOT / "reports" / f"rl_history_{tag}.json"
        glog = ROOT / "reports" / f"dushan_{tag}.log"

        # 断点续跑：上一轮值班已把这一代完整练完（out 存在且日志含「完成。」）→ 直接跳过
        prev_text = glog.read_text(encoding="utf-8", errors="ignore") if glog.exists() else ""
        if out.exists() and "完成。" in prev_text:
            say(f"{tag} 已完成（检测到 {out.name} 且日志含「完成。」），跳过")
            rate = 14.0
            continue

        if not init.exists():
            say(f"{tag} 跳过：初始化模型不存在 {init}")
            break

        remain = train_deadline - time.time()
        # 预留 60s 启动开销 + 首代更保守
        budget_iters = int((remain - 60) / rate)
        iters = min(gen["iters"], max(0, budget_iters))
        if iters < MIN_ITERS:
            say(f"{tag} 跳过：剩余 {remain / 60:.0f} 分钟仅够 ~{max(0, budget_iters)} 轮"
                f"（< {MIN_ITERS}）")
            break

        say(f"{tag} 启动：init={gen['init']} 计划 {gen['iters']} 轮 → 实跑 {iters} 轮"
            f"（速率估计 {rate:.1f}s/轮，剩余 {remain / 60:.0f} 分钟）"
            f" lr={gen['lr']} temp={gen['temperature']} teacher-prob={gen['teacher_prob']}")
        t0 = time.time()
        with open(glog, "w", encoding="utf-8") as out_f:
            rc = subprocess.run(
                [PY, "scripts/selfplay.py",
                 "--init", gen["init"], "--out", f"models/dushan_{tag}.pt",
                 "--history", f"reports/rl_history_{tag}.json",
                 "--iters", str(iters), "--games-per-iter", "96", "--batch", "512",
                 "--lr", gen["lr"], "--temperature", gen["temperature"],
                 "--teacher-prob", gen["teacher_prob"],
                 "--threads", "8", "--vectorized", "--workers", "4",
                 "--eval-every", "10", "--eval-rounds", "40",
                 "--seed", str(gen["seed"]), "--rules", "dushan"],
                cwd=str(ROOT), stdout=out_f, stderr=subprocess.STDOUT,
            ).returncode
        dt = time.time() - t0

        text = glog.read_text(encoding="utf-8", errors="ignore") if glog.exists() else ""
        finished = "完成。" in text
        tail = [ln for ln in text.strip().splitlines()[-2:]]
        for ln in tail:
            say(f"  {ln}")

        # 用实际耗时更新速率估计（含启动开销，偏保守）
        completed_iters = max(1, iters)
        rate = max(6.0, dt / completed_iters)
        say(f"{tag} 结束：rc={rc} 完成={finished} 用时 {dt / 3600:.2f}h"
            f"（{dt / completed_iters:.1f}s/轮）；新速率估计 {rate:.1f}s/轮")
        done.append(dict(tag=tag, iters=iters, hours=round(dt / 3600, 2),
                         rc=rc, finished=finished,
                         rate=round(dt / completed_iters, 2)))

        if rc != 0 or not finished or not out.exists():
            say(f"{tag} 未正常完成，中止接力（避免在坏模型上继续训练）")
            break

    SUMMARY.write_text(
        json.dumps(dict(start=time.strftime("%Y-%m-%d %H:%M:%S",
                                            time.localtime(t_start)),
                        hours=round((time.time() - t_start) / 3600, 2),
                        gens=done), ensure_ascii=False, indent=2),
        encoding="utf-8")
    say(f"训练接力结束，共 {len(done)} 代；总计 {((time.time() - t_start) / 3600):.2f}h")
    say("提示：请在 models/ 下核对各代 .pt 与 reports/rl_history_*.json，再跑对位赛决定换届")
    awake(False)
    say("已释放防休眠限制，值班结束")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
