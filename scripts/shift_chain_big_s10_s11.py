"""7 小时训练值班进程：防休眠 + 接力训练（big_s10 → big_s11）+ 收尾对位赛。

背景
----
上一条 12h 值班链（shift_chain_big_s8_s10）在 big_s9 跑到 iter 1171/1200 时被终止
（任务回收连带杀掉了整棵进程树）；models/dushan_big_s9.pt 里保存的是训练中途的
最佳评测检查点，可直接作为下一代的初始化。

本脚本从启动时刻起占用 7 小时：
  1. big_s10：init=dushan_big_s9.pt，计划 1000 轮（lr 1.0e-5 / temp 0.97 / teacher 0.10）；
  2. big_s11：init=dushan_big_s10.pt，计划 600 轮（lr 8e-6 / temp 0.95 / teacher 0.08）；
  3. 剩余时间做对位赛：big_s11 vs big_s7（现任大师）2400 副，
     以及 big_s10 vs big_s7 600 副初筛（时间不够就跳过）。

时间预算自适应：速率初值取 s9 实测 17.1s/轮，每代结束后用实测值更新；
剩余时间不够 MIN_ITERS(300) 轮就不再开新代。
"""
from __future__ import annotations

import ctypes
import json
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PY = rr"C:\Users\hyzor\.workbuddy\binaries\python\envs\zhuoji\Scripts\python.exe"
SHIFT_LOG = ROOT / "reports" / "shift_chain_big_s10_s11.log"
SUMMARY = ROOT / "reports" / "shift_chain_big_s10_s11_summary.json"

SHIFT_HOURS = 7.0
RESERVE_S = 45 * 60          # 留给收尾对位赛与缓冲
MIN_ITERS = 300              # 单代最少轮数，低于此值不再启动新代
DEFAULT_RATE = 17.1          # 秒/轮（s9 实测值）

ES_CONTINUOUS = 0x80000000
ES_SYSTEM_REQUIRED = 0x00000001

# 两代接力：延续 s8→s9 的衰减节奏（学习率/教师占比逐代下调，数据越"纯网"）
GENS = [
    dict(tag="big_s10", init="models/dushan_big_s9.pt", iters=1000,
         lr="1.0e-5", temperature="0.97", teacher_prob="0.10", seed=20266),
    dict(tag="big_s11", init="models/dushan_big_s10.pt", iters=600,
         lr="8.0e-6", temperature="0.95", teacher_prob="0.08", seed=20267),
]

# 收尾对位赛：候选 vs 现任大师 big_s7（副数越多越可信；时间不够按列表顺序取舍）
DUELS = [
    dict(a="dushan_big_s11", b="dushan_big_s7", deals=2400,
         log="reports/duel_big_s11_vs_big_s7.log"),
    dict(a="dushan_big_s10", b="dushan_big_s7", deals=600,
         log="reports/duel_big_s10_vs_big_s7_screen.log"),
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
    done: list[dict] = []

    for gen in GENS:
        tag = gen["tag"]
        init = ROOT / gen["init"]
        out = ROOT / "models" / f"dushan_{tag}.pt"
        hist = ROOT / "reports" / f"rl_history_{tag}.json"
        glog = ROOT / "reports" / f"dushan_{tag}.log"

        prev_text = glog.read_text(encoding="utf-8", errors="ignore") if glog.exists() else ""
        if out.exists() and "完成。" in prev_text:
            say(f"{tag} 已完成（检测到 {out.name} 且日志含「完成。」），跳过")
            continue

        if not init.exists():
            say(f"{tag} 跳过：初始化模型不存在 {init}")
            break

        remain = train_deadline - time.time()
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

    # ---- 收尾对位赛（用剩余时间，逐条取舍） ----
    duels = []
    for d in DUELS:
        if not (ROOT / d["a"]).exists() or not (ROOT / d["b"]).exists():
            say(f"对位赛跳过（缺模型）：{d['a']} vs {d['b']}")
            continue
        remain = t_start + SHIFT_HOURS * 3600 - time.time()
        # 2400 副约 16 分钟、600 副约 4 分钟（threads=8）；留 3 分钟缓冲
        need = d["deals"] * 0.4 / 60 + 3
        if remain / 60 < need:
            say(f"对位赛跳过（时间不足）：剩 {remain / 60:.0f} 分钟 < 需 ~{need:.0f} 分钟")
            continue
        say(f"对位赛：{Path(d['a']).name} vs {Path(d['b']).name} × {d['deals']} 副")
        with open(ROOT / d["log"], "w", encoding="utf-8") as out_f:
            rc = subprocess.run(
                [PY, "scripts/duel.py", "--a", d["a"], "--b", d["b"],
                 "--deals", str(d["deals"]), "--threads", "8"],
                cwd=str(ROOT), stdout=out_f, stderr=subprocess.STDOUT,
            ).returncode
        tail = (ROOT / d["log"]).read_text(encoding="utf-8", errors="ignore") \
            .strip().splitlines()[-3:]
        for ln in tail:
            say(f"  {ln}")
        duels.append(dict(**d, rc=rc, tail=tail))

    awake(False)
    SUMMARY.write_text(json.dumps(dict(
        start=time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(t_start)),
        hours=round((time.time() - t_start) / 3600, 2),
        gens=done, duels=duels), ensure_ascii=False, indent=2), encoding="utf-8")
    say(f"值班结束，共 {len(done)} 代、{len(duels)} 场对位赛；"
        f"总计 {(time.time() - t_start) / 3600:.2f}h；已释放防休眠限制")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
