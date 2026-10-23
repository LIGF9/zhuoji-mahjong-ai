#!/usr/bin/env python
"""门禁守望：等周期检查点落盘 → 自动跑 1200 副配对判读 → 打可判读的结论。

为什么单独做一件事
------------------
M2-A 600 轮 ≈ 6.6 h、M2-B 1200 轮 ≈ 12–16 h，而门禁（G1/G2×3/G3）落在
第 300/400/800/1200 轮 —— **盯着倒计时去手动敲命令是纯浪费注意力**，而且容易
在"检查点还没写完"时就开跑（`torch.save` 期间文件已存在但不可读）。

本脚本把三步合成一条命令：
1. **等**：轮询 `models/_ckpts/{stem}_it{at:04d}.pt`，要求文件存在 + 大小连续两次
   采样不变（跨过 `torch.save` 的写入窗口）；
2. **跑**：按历史段逐位可比的口径调 `score_checkpoints.py`（共用牌池 + 三个教师
   refs + `--anchor dushan_master`，对齐 S1/S2/S3 的读数）；
3. **判**：从 JSON 里取 `vs_anchor` 与 CI，按 `--threshold` 给通过/不通过，
   并算出与历史基线的改善量（改善量的 CI 按两段独立观测的方差平方和开方，偏保守）。

等待期间**顺手体检训练**：日志里出现 `Traceback` / `nan` 立刻收工报错；
日志超过 `--stall-min` 分钟不增长则告警（长跑冻结过一次，22.2 h 没被发现）。

用法::

    # G1（M2-A 第 300 轮）
    python scripts/gate_watch.py --stem dushan_m2_a --at 300 \
        --also dushan_m1_s3 --threshold -1.6 \
        --baseline-name M1-S3 --baseline -3.105 --baseline-ci 0.437 \
        --log reports/_m1_s4a.out --out reports/G1_s4a_it300.json

    # G2（M2-B，判「单调不减」：与上一次门禁读数比，跌幅不超过合并 CI）
    python scripts/gate_watch.py --stem dushan_m2_b --at 400 --gate G2-400 \
        --threshold -99 --monotonic-prev -1.2 --prev-ci 0.4 ...

    # G3（终验收：追平 = va ≥ +0.5 且 CI 下界 > 0；超越 = va ≥ +1.5 且 CI 下界 > 0）
    python scripts/gate_watch.py --stem dushan_m2_b --at 1200 --gate G3 \
        --threshold 0.5 --threshold2 1.5 --ci-lower-min 0 ...

    # 目标检查点已存在时：`--now` 跳过等待，直接判读（用来补跑）
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable

try:                                    # 后台任务里 stdout 可能不是 UTF-8
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass


def say(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def ckpt_path(stem: str, at: int) -> Path:
    return ROOT / "models" / "_ckpts" / f"{stem}_it{at:04d}.pt"


def wait_ckpt(stem: str, at: int, log: Path | None, stall_min: int,
              max_wait_h: float) -> Path | None:
    """等检查点落盘（含写入完成判定），顺带体检训练日志。"""
    p = ckpt_path(stem, at)
    t0 = time.time()
    prev_size = -1
    stable = 0
    last_log = (log.stat().st_size if log and log.exists() else 0)
    last_progress = time.time()
    warned = False
    while time.time() - t0 < max_wait_h * 3600:
        if p.exists():
            sz = p.stat().st_size
            if sz == prev_size and sz > 0:
                stable += 1
                if stable >= 2:                     # 连续两次采样不变
                    say(f"检查点就绪：{p}（{sz/1048576:.1f} MB）")
                    return p
            else:
                stable = 0
                say(f"检测到 {p.name}（{sz/1048576:.1f} MB）写入中…")
            prev_size = sz

        if log and log.exists():
            txt_size = log.stat().st_size
            if txt_size != last_log:
                last_log = txt_size
                last_progress = time.time()
                warned = False
            body = log.read_text(encoding="utf-8", errors="replace")
            if "Traceback" in body:
                say("❌ 训练日志出现 Traceback —— 立刻收工，请人工排查：")
                for line in body.splitlines()[-20:]:
                    print("    " + line, flush=True)
                return None
            if re.search(r"=\s*nan\b", body, re.I):
                say("❌ 训练日志出现 NaN —— 立刻收工")
                return None
            idle = (time.time() - last_progress) / 60.0
            if idle > stall_min and not warned:
                say(f"⚠️ 日志已 {idle:.0f} 分钟无增长（超过 --stall-min {stall_min}）"
                    f" —— 可能被冻结，检查 reports/_res_monitor_*.csv")
                warned = True

        n = 0
        if log and log.exists():
            n = len(re.findall(r"^iter", log.read_text(encoding="utf-8",
                                                       errors="replace"), re.M))
        say(f"等待 {p.name} …（训练已 {n} 轮 / 目标 {at}）")
        time.sleep(60)
    say(f"❌ 等待超时（{max_wait_h:g} h）仍未见到 {p}")
    return None


def judge(args) -> int:
    ckpt = args.now_ckpt or ckpt_path(args.stem, args.at)
    stem_rel = ckpt.relative_to(ROOT / "models").with_suffix("").as_posix()

    ckpts = [args.anchor] + [c for c in (args.also or []) if c] + [stem_rel]
    cmd = [PY, str(ROOT / "scripts" / "score_checkpoints.py"),
           "--ckpts", ",".join(ckpts),
           "--anchor", args.anchor,
           "--ref", args.ref,
           "--deals", str(args.deals), "--seed", str(args.seed),
           "--workers", str(args.workers), "--threads", str(args.threads),
           "--worker-threads", str(args.worker_threads),
           "--out", str(Path(args.out).resolve())]
    say("跑判读：" + " ".join(cmd))
    t0 = time.time()
    r = subprocess.run(cmd, cwd=str(ROOT))
    if r.returncode != 0:
        say(f"❌ 判读进程 exit={r.returncode}")
        return r.returncode
    say(f"判读完成，用时 {(time.time()-t0)/60:.1f} 分")

    res = json.loads(Path(args.out).read_text(encoding="utf-8"))
    rows = {row["ckpt"]: row for row in res["rows"]}
    row = rows.get(stem_rel)
    if row is None:
        say(f"❌ 结果里找不到 {stem_rel}，实际行：{list(rows)}")
        return 2

    print("\n" + "=" * 74)
    print(f"门禁 {args.gate} —— {stem_rel}（{args.deals} 副配对，anchor={args.anchor}）")
    print("=" * 74)
    print(f"{'检查点':<32}{'分值水平':>12}{'±95%CI':>10}{'vs 锚点':>12}{'±95%CI':>10}  结论")
    for r_ in res["rows"]:
        print(f"{r_['ckpt']:<32}{r_['level']:>+12.3f}{r_['level_ci95']:>10.3f}"
              f"{r_['vs_anchor']:>+12.3f}{r_['vs_anchor_ci95']:>10.3f}  {r_['verdict']}")

    va, ci = row["vs_anchor"], row["vs_anchor_ci95"]
    lo = va - ci
    print(f"\n目标行：vs_anchor = {va:+.3f} ± {ci:.3f}（下界 {lo:+.3f}）")
    if args.baseline is not None:
        gain = va - args.baseline
        # 改善量的 CI：两段独立配对观测，方差按平方和开方（保守）
        gci = (ci ** 2 + (args.baseline_ci or 0.0) ** 2) ** 0.5
        print(f"历史基线 {args.baseline_name}：{args.baseline:+.3f} ± "
              f"{args.baseline_ci:.3f}")
        print(f"改善量 Δ = {gain:+.3f} ± {gci:.3f}"
              f"（{'显著改善' if gain - gci > 0 else '未达显著'}）")

    # ---- 判据组装（G1 / G2 / G3 各取所需，全部满足才算通过）----
    conds: list[tuple[bool, str]] = []
    conds.append((va >= args.threshold,
                  f"点估计 va ≥ {args.threshold:+.3f}（实测 {va:+.3f}）"))
    if args.ci_lower_min is not None:
        conds.append((lo > args.ci_lower_min,
                      f"CI 下界 > {args.ci_lower_min:+.3f}（实测 {lo:+.3f}）"
                      f" —— 「追平/超越」必须是统计显著的，不是点估计擦线"))
    mono_ok = None
    if args.monotonic_prev is not None:
        drop_tol = ci + (args.prev_ci or 0.0)
        mono_ok = va >= args.monotonic_prev - drop_tol
        strict = va >= args.monotonic_prev
        conds.append((mono_ok,
                      f"单调不减（G2）：va ≥ 上次 {args.monotonic_prev:+.3f} − "
                      f"合并CI {drop_tol:.3f} = {args.monotonic_prev - drop_tol:+.3f}"
                      f"（实测 {va:+.3f}；{'点估计也未跌' if strict else '点估计有跌但未达显著'}）"))
    ok = all(c for c, _ in conds)
    for c, desc in conds:
        print(f"  {'✅' if c else '❌'}  {desc}")
    print(f"\n判据（{args.gate}）→ {'✅ 通过' if ok else '❌ 未通过'}")
    if args.threshold2 is not None:
        ok2 = va >= args.threshold2 and lo > (args.ci_lower_min
                                              if args.ci_lower_min is not None
                                              else float("-inf"))
        print(f"第二档（{'超越' if args.gate == 'G3' else '门槛2'}）：va ≥ "
              f"{args.threshold2:+.3f} 且 CI 下界 > 0 → "
              f"{'✅ 达成' if ok2 else '—— 未达成'}（第一档=追平）")
        verdict2 = bool(ok2)
    else:
        verdict2 = None
    print("=" * 74)

    verdict = {"gate": args.gate, "ckpt": stem_rel, "vs_anchor": va,
               "vs_anchor_ci95": ci, "lower": lo, "threshold": args.threshold,
               "ci_lower_min": args.ci_lower_min, "threshold2": args.threshold2,
               "passed2": verdict2, "monotonic_prev": args.monotonic_prev,
               "monotonic_ok": mono_ok,
               "passed": bool(ok), "deals": args.deals, "seed": args.seed,
               "baseline": args.baseline, "baseline_ci": args.baseline_ci,
               "out": str(args.out)}
    vp = Path(args.out).with_suffix(".verdict.json")
    vp.write_text(json.dumps(verdict, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"判决 -> {vp}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="等检查点 → 跑门禁判读 → 打结论")
    ap.add_argument("--stem", required=True, help="检查点 stem（= models/{out} 去后缀）")
    ap.add_argument("--at", type=int, required=True, help="目标轮次（对齐 _itNNNN）")
    ap.add_argument("--gate", default="G1", help="门禁名（只用于日志与判决文件）")
    ap.add_argument("--anchor", default="dushan_master")
    ap.add_argument("--also", action="append", default=None,
                    help="额外并列的检查点（可重复）；默认 dushan_m1_s3")
    ap.add_argument("--ref", default="teacher:balanced|teacher:tenpai_rush|teacher:defensive")
    ap.add_argument("--deals", type=int, default=1200)
    ap.add_argument("--seed", type=int, default=31337)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--worker-threads", type=int, default=1)
    ap.add_argument("--threshold", type=float, default=-1.6)
    ap.add_argument("--threshold2", type=float, default=None,
                    help="第二档门槛（G3 的「超越」= va ≥ 1.5 且 CI 下界 > 0）；"
                         "不设则只报一档")
    ap.add_argument("--ci-lower-min", type=float, default=None,
                    help="要求 CI 下界严格大于此值才算通过（G3 用 0）；"
                         "不设则只看点估计")
    ap.add_argument("--monotonic-prev", type=float, default=None,
                    help="G2 单调不减：与上一次门禁的 vs_anchor 比，"
                         "允许跌幅不超过合并 CI（点估计跌但不显著仍算过）")
    ap.add_argument("--prev-ci", type=float, default=0.0,
                    help="配 --monotonic-prev：上一次读数的 ±95%CI")
    ap.add_argument("--baseline-name", default="M1-S3")
    ap.add_argument("--baseline", type=float, default=None)
    ap.add_argument("--baseline-ci", type=float, default=0.0)
    ap.add_argument("--log", default="", help="训练日志（体检用）；空则不体检")
    ap.add_argument("--stall-min", type=int, default=20)
    ap.add_argument("--max-wait-h", type=float, default=8.0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--now", action="store_true",
                    help="跳过等待：检查点已存在时直接判读")
    ap.add_argument("--now-ckpt", default="", help="直接用指定 ckpt（配 --now）")
    args = ap.parse_args()

    if args.also is None:
        args.also = ["dushan_m1_s3"]

    if args.now:
        return judge(args)

    log = (ROOT / args.log) if args.log else None
    p = wait_ckpt(args.stem, args.at, log, args.stall_min, args.max_wait_h)
    if p is None:
        return 1
    time.sleep(20)                          # 让训练进程把权重完全落盘再读
    return judge(args)


if __name__ == "__main__":
    raise SystemExit(main())
