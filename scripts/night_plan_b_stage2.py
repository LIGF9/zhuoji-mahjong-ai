"""方案 B 阶段 2：空间头对照（2a）+ P1 三包单变量消融（2b）—— 夜跑串行排队。

为什么要串行排队而不是同时跑
----------------------------
本机 4 核 8 线程、无独显，一次只能把 4 个采集 worker 喂饱。同时开两条训练会让
**两条都变慢**，而且互相扰动 CPU 频率，评测耗时统计也会失真。所以这里做成
「一个接一个」的队列，并带 ``--max-hours`` 预算闸门：预算用完就停止启动新臂，
已跑完的臂结果照常落盘（每臂一份 summary，不留"要么全有要么全无"）。

臂的设计（**前置声明，不许看到结果再改**）
------------------------------------------
全部从 ``dushan_big_s11``（现任大师）出发，**同一超参、同一 seed**，
只改一个变量；400 轮；判读一律走对外的共用牌池复评（``score_checkpoints.py``）。

| 臂 | 变量 | 想回答的问题 |
|---|---|---|
| ``sp_off`` | 空间头 off（**对照组**） | 「从 s11 再多训 400 轮」本身值多少分？ |
| ``sp_on``  | 空间头 on | 零成本架构修正（+2×(C+1) 参数）值不值分？ |
| ``p_none`` | v2 架构 + 三包全关 | 只换"外壳"、不加任何新特征，会不会白掉分？ |
| ``p_abc``  | v2 三包全开 | P1 表征整包到底有没有收益？ |
| ``p_a``    | 只开牌河时序 | 哪个包在起作用（时序读牌）？ |
| ``p_b``    | 只开危险度/对手状态 | 哪个包在起作用（防守放铳）？ |
| ``p_c``    | 只开番型潜力/巡目 | 哪个包在起作用（大牌选择）？ |

``sp_*`` 臂用 v1 编码器（``--encoder auto`` 即 v1），``p_*`` 臂用 ``--encoder v2``
（自动走 v1→v2 部分热启动）。**空间头在 ``p_*`` 臂里一律 off**，否则包的效果
会和架构修正混在一起，两个变量分不开。

门槛（与 ``docs/P1-表征升级设计.md`` §7 一致）
----------------------------------------------
* 单包消融（阶段 2）：净分差 **≥ +0.40** 且 95% CI 下界 > 0 → 该包采纳；
* 空间头（阶段 2a）：同一门槛；
* 合并长训练（阶段 3）才用 **≥ +0.50**。这里**不预先宣布任何臂通过**，
  只把数报出来。

用法::

    python scripts/night_plan_b_stage2.py                 # 全自动（等方案 A4 结束）
    python scripts/night_plan_b_stage2.py --skip-wait     # 立刻开始
    python scripts/night_plan_b_stage2.py --arms sp_off,sp_on --iters 50   # 试跑
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

#: 方案 A4 的收尾标志（它跑完我们才上机，避免两条训练互相拖慢）
A_SUMMARY = ROOT / "reports" / "shift_plan_a_s12_summary.json"
LOG = ROOT / "reports" / "night_plan_b_stage2.log"
SUMMARY = ROOT / "reports" / "night_plan_b_stage2_summary.json"

INIT = "models/dushan_big_s11.pt"
ANCHOR = "dushan_big_s11"
POOL = "dushan_big_s3,dushan_big_s7,dushan_big_s11"
SAVE_EVERY = 50
REF = "net:dushan_big_s11|teacher:balanced|teacher:tenpai_rush"

#: 臂顺序 = 执行顺序 = 优先级（前面的是"没有它后面就不好解释"的）
ARM_ORDER = ["sp_off", "sp_on", "p_none", "p_abc", "p_a", "p_b", "p_c"]

#: 每臂的**唯一变量**。spatial=None 表示"用 auto"（跟随检查点，即 s11 的 off）。
#: packs=None 表示不传（v1 时无意义）。
ARMS: dict[str, dict] = {
    "sp_off": {"encoder": "auto", "spatial": "off", "packs": None,
               "desc": "对照组：v1 + 空间头 off（= 纯多训 400 轮）"},
    "sp_on": {"encoder": "auto", "spatial": "on", "packs": None,
              "desc": "v1 + 空间头 on（零成本架构修正）"},
    "p_none": {"encoder": "v2", "spatial": "off", "packs": "none",
               "desc": "v2 外壳 + 三包全关（不含任何新特征）"},
    "p_abc": {"encoder": "v2", "spatial": "off", "packs": "abc",
              "desc": "v2 三包全开（P1 整包）"},
    "p_a": {"encoder": "v2", "spatial": "off", "packs": "a",
            "desc": "只开 P1a 牌河时序"},
    "p_b": {"encoder": "v2", "spatial": "off", "packs": "b",
            "desc": "只开 P1b 危险度与对手状态"},
    "p_c": {"encoder": "v2", "spatial": "off", "packs": "c",
            "desc": "只开 P1c 番型潜力与巡目情境"},
}

CONFIRM_DEALS = 1200
GATE = 0.40          # 阶段 2 门槛（净分差与 CI 下界）

ES_CONTINUOUS = 0x80000000
ES_SYSTEM_REQUIRED = 0x00000001


def awake(on: bool = True) -> bool:
    try:
        ctypes.windll.kernel32.SetThreadExecutionState(
            (ES_CONTINUOUS | ES_SYSTEM_REQUIRED) if on else ES_CONTINUOUS)
        return True
    except Exception:
        return False


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-wait", action="store_true", help="不等方案 A4")
    ap.add_argument("--arms", default=",".join(ARM_ORDER),
                    help=f"只跑这些臂，逗号分隔（可选：{','.join(ARM_ORDER)}）")
    ap.add_argument("--iters", type=int, default=400)
    ap.add_argument("--seed", type=int, default=20270)
    ap.add_argument("--max-hours", type=float, default=8.0,
                    help="预算闸门：累计训练超过这个小时数就不再启动新臂")
    ap.add_argument("--deals", type=int, default=CONFIRM_DEALS)
    ap.add_argument("--skip-score", action="store_true")
    args = ap.parse_args()

    log = open(LOG, "a", encoding="utf-8", buffering=1)

    def say(msg: str) -> None:
        print(f"[{time.strftime('%H:%M:%S')}] {msg}", file=log)

    want = [a.strip() for a in args.arms.split(",") if a.strip()]
    bad = [a for a in want if a not in ARMS]
    if bad:
        raise SystemExit(f"未知的臂：{bad}；可选 {list(ARMS)}")

    say("=" * 66)
    say(f"方案 B 阶段 2 夜跑队列：{len(want)} 臂 × {args.iters} 轮"
        f"（预算 {args.max_hours} h）")
    for a in want:
        say(f"  · {a:<7} {ARMS[a]['desc']}")

    # ---- 0. 等方案 A4 -------------------------------------------------------
    if not args.skip_wait:
        t_wait = time.time()
        while not A_SUMMARY.exists():
            if time.time() - t_wait > 10 * 3600:
                say("等待方案 A4 超过 10 小时，放弃等待（请检查 shift_plan_a_s12 任务）")
                return 1
            if int(time.time() - t_wait) % 300 < 20:
                say(f"等待方案 A4 产出 {A_SUMMARY.name} …（已等 "
                    f"{(time.time() - t_wait) / 60:.0f} 分钟）")
            time.sleep(30)
        say(f"方案 A4 已就位（等待 {(time.time() - t_wait) / 60:.0f} 分钟）")

    say(f"防休眠={'OK' if awake(True) else '失败'}")

    # ---- 1. 逐臂训练 --------------------------------------------------------
    state_path = ROOT / "reports" / "night_plan_b_stage2_summary.json"
    state: dict = {"arms": {}, "iters": args.iters, "gate": GATE,
                   "anchor": ANCHOR, "ref": REF}
    if state_path.exists():
        try:
            old = json.loads(state_path.read_text(encoding="utf-8"))
            if old.get("arms"):
                state["arms"] = old["arms"]
                say(f"已有 {len(old['arms'])} 臂历史结果，将跳过已完成的臂")
        except Exception:
            pass

    t_start = time.time()
    spent = sum(float(r.get("hours") or 0.0) for r in state["arms"].values())
    for arm in want:
        if arm in state["arms"] and state["arms"][arm].get("finished"):
            say(f"[skip] {arm} 已完成（净分差 "
                f"{(state['arms'][arm].get('score') or {}).get('vs_anchor')}）")
            continue
        if spent >= args.max_hours:
            say(f"[budget] 已用 {spent:.2f} h ≥ 预算 {args.max_hours} h，"
                f"停止启动新臂（{arm} 及之后跳过）")
            break
        spec = ARMS[arm]
        if not (ROOT / INIT).exists():
            raise SystemExit(f"找不到起点模型 {INIT}")

        out = ROOT / "models" / f"dushan_b2_{arm}.pt"
        glog = ROOT / "reports" / f"dushan_b2_{arm}.log"
        hist = ROOT / "reports" / f"rl_history_b2_{arm}.json"

        cmd = [PY, "scripts/selfplay.py",
               "--init", INIT, "--out", f"models/dushan_b2_{arm}.pt",
               "--history", f"reports/rl_history_b2_{arm}.json",
               "--iters", str(args.iters), "--games-per-iter", "96",
               "--batch", "512", "--lr", "8.0e-6", "--temperature", "0.95",
               "--teacher-prob", "0.08", "--threads", "8",
               "--vectorized", "--workers", "4",
               "--eval-every", "10", "--eval-rounds", "40",
               # ↓ 方案 A 定下的选点口径，所有臂一致
               "--select", "guard", "--select-metric", "vs_anchor",
               "--eval-anchor", ANCHOR,
               "--save-every", str(SAVE_EVERY), "--snapshot-pool", POOL,
               "--encoder", spec["encoder"],
               "--encoder-packs", spec["packs"] if spec["packs"] else "auto",
               "--spatial-head", spec["spatial"] or "auto",
               "--seed", str(args.seed), "--rules", "dushan"]
        say(f"—— {arm}：{spec['desc']} ——")
        # 把「唯一变量」原样打进日志，方便事后核对臂之间确实只差一个开关
        say("  cmd: " + " ".join(cmd[2:]))
        t0 = time.time()
        with open(glog, "w", encoding="utf-8") as f:
            rc = subprocess.run(cmd, cwd=str(ROOT), stdout=f,
                                stderr=subprocess.STDOUT).returncode
        dt = time.time() - t0
        spent += dt / 3600
        text = glog.read_text(encoding="utf-8", errors="ignore")
        rec = {
            "desc": spec["desc"], "cmd": cmd[2:], "rc": rc,
            "hours": round(dt / 3600, 3),
            "finished": "完成。" in text,
            "s_per_iter": round(dt / max(1, args.iters), 2),
            "select_lines": [ln for ln in text.splitlines()
                             if ln.startswith("[select]")],
            "head": [ln for ln in text.splitlines()
                     if ln.startswith(("[编码]", "[pool]", "[eval] 锚点"))],
        }
        state["arms"][arm] = rec
        state_path.write_text(json.dumps(state, ensure_ascii=False, indent=2),
                              encoding="utf-8")
        for ln in rec["head"]:
            say("  " + ln)
        say(f"  {arm} 训练结束 rc={rc} 完成={rec['finished']} "
            f"{dt / 3600:.2f} h（{rec['s_per_iter']} s/轮）；累计 {spent:.2f} h")
        if rc != 0 or not out.exists():
            say(f"  {arm} 未正常产出模型，跳过后续复评")
            continue

    # ---- 2. 共用牌池复评（所有臂 + s11 锚点，一次算完） ---------------------
    done = [a for a in want
            if (ROOT / "models" / f"dushan_b2_{a}.pt").exists()]
    if done and not args.skip_score:
        ckpts = [ANCHOR] + [f"dushan_b2_{a}" for a in done]
        say(f"复评：{len(ckpts)} 个检查点 × {args.deals} 副（共用牌池 + 共用对手）")
        out_json = ROOT / "reports" / "b2_rescore.json"
        subprocess.run(
            [PY, "scripts/score_checkpoints.py",
             "--ckpts", ",".join(ckpts), "--ref", REF,
             "--deals", str(args.deals), "--threads", "8",
             "--anchor", ANCHOR, "--out", str(out_json.relative_to(ROOT)),
             "--no-per-deal"],
            cwd=str(ROOT), stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
        if out_json.exists():
            sc = json.loads(out_json.read_text(encoding="utf-8"))
            rows = {r["ckpt"]: r for r in sc.get("rows", [])}
            for arm in done:
                r = rows.get(f"dushan_b2_{arm}") or {}
                state["arms"].setdefault(arm, {})["score"] = r
            state["rescore_deals"] = args.deals
            say("—— 复评结果（相对 s11 的配对净分差）——")
            for arm in done:
                r = state["arms"][arm].get("score") or {}
                lo = (r.get("vs_anchor") or 0) - (r.get("vs_anchor_ci95") or 0)
                ok = (r.get("vs_anchor") or -9) >= GATE and lo > 0
                say(f"  {arm:<7} {r.get('vs_anchor', float('nan')):+.3f} "
                    f"± {r.get('vs_anchor_ci95', float('nan')):.3f}  "
                    f"CI下界 {lo:+.3f}  → "
                    f"{'过门槛 ✅' if ok else '未过门槛'}   {ARMS[arm]['desc']}")
            state_path.write_text(
                json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")

    awake(False)
    state["elapsed_hours"] = round((time.time() - t_start) / 3600, 2)
    state_path.write_text(json.dumps(state, ensure_ascii=False, indent=2),
                          encoding="utf-8")
    say(f"队列结束，累计 {state['elapsed_hours']} h，汇总 -> {state_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
