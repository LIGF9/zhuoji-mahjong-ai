"""方案 A 的第二阶段：从 s9 重跑「一代」，把被选点规则丢掉的算力捡回来。

背景（为什么值得跑这一条）
--------------------------
``scripts/analyze_metric_noise.py`` 查出各代存档里的 ``iter`` 字段 = eval 最高那一刻：

    s4 475/2400 · s7 280/850 · s10 20/1000 · s11 120/600

计划合计 17150 轮，实际只保留 9095 轮 —— **47% 的算力被 argmax 直接扔掉**；
s7→s11 那「四代」更是丢了 70%。而 ``dushan_big_s10`` 其实就是 ``dushan_big_s9``
再加 20 轮，``s11`` = ``s9`` + 140 轮。于是"四代"实际上只训了 140 轮有用功。

本脚本做的事
------------
1. **等天梯跑完**（``reports/ladder_s7_s11_coop.json`` 出现），避免抢 CPU 影响它；
2. 从 ``dushan_big_s9`` 重跑 **600 轮**（超参与 s11 完全一致，保证可比），但：

   * ``--select guard --select-metric vs_anchor --eval-anchor dushan_big_s11``
     （选点指标换成"对现任大师的同牌池配对净分差"，与对外晋升判据同一把尺子）；
   * ``--save-every 50``（**历史 s1–s11 从未开过**，导致"代内重选"永久不可能）；
   * ``--snapshot-pool dushan_big_s3,dushan_big_s7,dushan_big_s11``
     （跨代快照池，零采集成本，让自我对弈同时面对不同世代的自己）。

3. 训练结束**用共用牌池给所有周期性检查点打分**（两步筛选：先 300 副粗筛，
   再对前 3 名与 s9/s11 做 1200 副复筛），得到可信的"这一代里最好的那个点"。
   这一步就是过去做不到的"代内重选"。

期望值要说清（前置声明，不许事后调整）
--------------------------------------
计划算力（1000+600 轮）如果真被用上，增益上界约 = 从 s9 再训 600 轮。
P0 已证明 **s7→s11 的真实增益统计上不显著（−0.243±0.290）**，所以本条的
**主要价值是记录一个可复核的负结果/正结果**，而不是"白捡一代"。
--- 不许看到结果再改门槛 ---

用法::

    python scripts/shift_plan_a_s12.py            # 全自动（含等待天梯）
    python scripts/shift_plan_a_s12.py --no-wait  # 天梯已跑完时直接开始
"""
from __future__ import annotations

import argparse
import ctypes
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = r"C:\Users\hyzor\.workbuddy\binaries\python\envs\zhuoji\Scripts\python.exe"
LADDER_JSON = ROOT / "reports" / "ladder_s7_s11_coop.json"
LOG = ROOT / "reports" / "shift_plan_a_s12.log"
SUMMARY = ROOT / "reports" / "shift_plan_a_s12_summary.json"

TAG = "big_s12"
INIT = "models/dushan_big_s9.pt"
ITERS = 600
ANCHOR = "dushan_big_s11"
POOL = "dushan_big_s3,dushan_big_s7,dushan_big_s11"
SAVE_EVERY = 50

ES_CONTINUOUS = 0x80000000
ES_SYSTEM_REQUIRED = 0x00000001

# 复筛口径（前置声明）：粗筛 300 副、复筛 1200 副，都只看「相对 s11 的配对净分差」
SCREEN_DEALS = 300
CONFIRM_DEALS = 1200
# 与 docs/P1-表征升级设计.md §7「通用评测口径」一致：现任大师 + 先听教师 + 均衡教师
REF = "net:dushan_big_s11|teacher:balanced|teacher:tenpai_rush"


def awake(on: bool = True) -> bool:
    try:
        ctypes.windll.kernel32.SetThreadExecutionState(
            (ES_CONTINUOUS | ES_SYSTEM_REQUIRED) if on else ES_CONTINUOUS)
        return True
    except Exception:
        return False


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-wait", action="store_true", help="不等天梯（它已跑完）")
    ap.add_argument("--iters", type=int, default=ITERS)
    ap.add_argument("--skip-train", action="store_true", help="只做检查点复评")
    ap.add_argument("--skip-rescore", action="store_true", help="只做训练")
    args = ap.parse_args()

    log = open(LOG, "a", encoding="utf-8", buffering=1)

    def say(msg: str) -> None:
        print(f"[{time.strftime('%H:%M:%S')}] {msg}", file=log)

    say("—" * 46)
    say("方案 A 第二阶段：s9 → s12 重跑一代 + 代内重选")

    # ---- 1. 等天梯 --------------------------------------------------------
    if not args.no_wait:
        t_wait = time.time()
        while not LADDER_JSON.exists():
            if time.time() - t_wait > 4 * 3600:
                say("等待天梯超过 4 小时，放弃等待（请检查 ladder 任务）")
                return 1
            if int(time.time() - t_wait) % 300 < 20:
                say(f"等待天梯产出 {LADDER_JSON.name} …（已等 "
                    f"{(time.time() - t_wait) / 60:.0f} 分钟）")
            time.sleep(20)
        say(f"天梯已就位（等待 {(time.time() - t_wait) / 60:.0f} 分钟）")

    say(f"防休眠={'OK' if awake(True) else '失败'}")

    # ---- 2. 训练 ----------------------------------------------------------
    out = ROOT / "models" / f"dushan_{TAG}.pt"
    glog = ROOT / "reports" / f"dushan_{TAG}.log"
    hist = ROOT / "reports" / f"rl_history_{TAG}.json"
    train_info: dict = {}
    if not args.skip_train:
        cmd = [PY, "scripts/selfplay.py",
               "--init", INIT, "--out", f"models/dushan_{TAG}.pt",
               "--history", f"reports/rl_history_{TAG}.json",
               "--iters", str(args.iters), "--games-per-iter", "96", "--batch", "512",
               "--lr", "8.0e-6", "--temperature", "0.95", "--teacher-prob", "0.08",
               "--threads", "8", "--vectorized", "--workers", "4",
               "--eval-every", "10", "--eval-rounds", "40",
               # ↓↓↓ 方案 A 的三处改动，其余超参与 s11 完全一致
               "--select", "guard", "--select-metric", "vs_anchor",
               "--eval-anchor", ANCHOR,
               "--save-every", str(SAVE_EVERY),
               "--snapshot-pool", POOL,
               "--seed", "20268", "--rules", "dushan"]
        say("训练启动：" + " ".join(cmd[2:]))
        t0 = time.time()
        with open(glog, "w", encoding="utf-8") as f:
            rc = subprocess.run(cmd, cwd=str(ROOT), stdout=f,
                                stderr=subprocess.STDOUT).returncode
        dt = time.time() - t0
        text = glog.read_text(encoding="utf-8", errors="ignore")
        train_info = {
            "rc": rc, "hours": round(dt / 3600, 2),
            "finished": "完成。" in text,
            "rate_s_per_iter": round(dt / max(1, args.iters), 2),
            "select_lines": [ln for ln in text.splitlines() if ln.startswith("[select]")],
        }
        for ln in train_info["select_lines"]:
            say("  " + ln)
        say(f"训练结束：rc={rc} 完成={train_info['finished']} "
            f"用时 {dt / 3600:.2f}h（{train_info['rate_s_per_iter']}s/轮）")
        if rc != 0 or not out.exists():
            say("训练未正常完成，跳过复评")
            SUMMARY.write_text(json.dumps(
                {"train": train_info, "rescore": None}, ensure_ascii=False,
                indent=2), encoding="utf-8")
            awake(False)
            return 1

    # ---- 3. 代内重选：共用牌池给所有周期检查点打分 -------------------------
    rescore: dict = {}
    if not args.skip_rescore:
        ck_dir = ROOT / "models" / "_ckpts"
        snaps = sorted(ck_dir.glob(f"dushan_{TAG}_it*.pt"))
        cands = ["dushan_big_s9", "dushan_big_s11"] + \
            [f"_ckpts/{p.stem}" for p in snaps]
        say(f"粗筛：{len(cands)} 个检查点 × {SCREEN_DEALS} 副（共用牌池 + 共用对手）")
        screen_out = ROOT / "reports" / "s12_rescore_screen.json"
        _run_score(cands, SCREEN_DEALS, screen_out, anchor="dushan_big_s11")
        scr = json.loads(screen_out.read_text(encoding="utf-8"))
        fixed = ("dushan_big_s9", "dushan_big_s11")
        top = [r["ckpt"] for r in scr["rows"] if r["ckpt"] not in fixed][:3]
        side = [r["ckpt"] for r in scr["rows"] if r["ckpt"] in fixed]
        final_cands = sorted(set(top + side))
        say(f"粗筛第一梯队：{top}；复筛 {final_cands} × {CONFIRM_DEALS} 副")
        confirm_out = ROOT / "reports" / "s12_rescore_confirm.json"
        _run_score(final_cands, CONFIRM_DEALS, confirm_out, anchor="dushan_big_s11")
        conf = json.loads(confirm_out.read_text(encoding="utf-8"))
        rescore = {"screen": scr["rows"], "confirm": conf["rows"],
                   "screen_deals": SCREEN_DEALS, "confirm_deals": CONFIRM_DEALS}
        for r in conf["rows"]:
            say(f"  {r['ckpt']:<40} vs s11 {r['vs_anchor']:+.3f} "
                f"± {r['vs_anchor_ci95']:.3f}  {r['verdict']}")

    awake(False)
    SUMMARY.write_text(json.dumps(
        {"train": train_info, "rescore": rescore,
         "anchor": ANCHOR, "pool": POOL, "select": "guard/vs_anchor"},
        ensure_ascii=False, indent=2), encoding="utf-8")
    say(f"全部结束，汇总 -> {SUMMARY}")
    return 0


def _run_score(ckpts: list[str], deals: int, out: Path, anchor: str) -> None:
    cmd = [PY, "scripts/score_checkpoints.py",
           "--ckpts", ",".join(ckpts), "--ref", REF,
           "--deals", str(deals), "--threads", "8",
           "--anchor", anchor, "--out", str(out.relative_to(ROOT)),
           "--no-per-deal"]
    subprocess.run(cmd, cwd=str(ROOT), stdout=subprocess.DEVNULL,
                   stderr=subprocess.STDOUT)


if __name__ == "__main__":
    raise SystemExit(main())
