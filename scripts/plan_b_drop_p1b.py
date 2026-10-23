"""方案 B（重做版）：**去掉 P1b（危险度/对手状态包）后重做 P1 消融**。

为什么重做
----------
2026-10-10 的第一批七臂消融给出全负结果，元凶指向 P1b（−0.635）与整包
`p_abc`（−0.670）；而 P1a（−0.138）、P1c（−0.036）都在噪声内
（见 `docs/P1-阶段2-消融结论.md`）。但那一批**全部使用随机初始化**热启动
v1→v2，新增的 16 通道 / 28 列 `fc_in` 权重是随机的，实测会让热启动瞬间的
discard logits 偏离 16.7 —— 足以改变 argmax。所以"P1b 有害"这个判读**可能被
迁移方式污染**。

判据来自 `scripts/after_a4_plan_c.py` 的决定性对照 C3（零初始化 + 400 轮）：
* C3 显著优于 `p_abc` → **H1（迁移扰动）**：特征未被证伪，应在零初始化下重测；
* C3 与 `p_abc` 无显著差异且都显著劣于 s11 → **H2（特征本身有害）**：需返工。

**本脚本无条件采用零初始化**（这正是"修 P1 后重做消融"里的"修"），
并把 C3 的判读记进汇总，便于事后对照。

臂的设计（前置声明，不许看到结果再改）
--------------------------------------
全部从 `dushan_big_s11` 出发，同 seed（20270）、同超参、400 轮，
`--encoder v2 --spatial-head off --new-param-init zero`，**只改 `--encoder-packs`**：

| 臂 | packs | 作用 |
|---|---|---|
| `p_none` | none | v2 外壳对照组 —— **不重训，直接复用第一批的 `dushan_b2_p_none.pt`**（见下） |
| `p_a_z` | a | 只开 P1a 牌河时序 |
| `p_c_z` | c | 只开 P1c 番型潜力/巡目 |
| `p_ac_z` | ac | **a+c 合并**（去掉 b 之后的候选组合） |

为什么 `p_none` 可以复用而不重训
--------------------------------
`--encoder-packs none` 会把新通道输入与新增 28 维全局量**整体清零**，于是新增
权重无论取什么值都乘零 —— 前向输出与 v1 逐位相同，反向梯度也是 0（新权重永远
不更新）。也就是说 **`p_none` 对 `--new-param-init` 完全不敏感**，第一批那个
`dushan_b2_p_none.pt` 与"零初始化重训 400 轮"在数学上等价。省下约 35 min。

判读口径（与第一批、与 `docs/P1-表征升级设计.md` §7 完全一致）
-------------------------------------------------------------
共用牌池 + 共用对手的 **1200 副**配对净分差（`score_checkpoints.py`），
门槛 **净分差 ≥ +0.40 且 95% CI 下界 > 0**。复评时把第一批的随机初始化臂
（`p_a` / `p_c` / `p_abc`）与 C3（`dushan_b2c_p_abc_zero`）以及纯扰动对照
（`_diag_c1_v2warm`）**放进同一批牌池**，于是所有臂是配对观测，差值可直接比。

用法::

    python scripts/plan_b_drop_p1b.py                 # 等 Phase A（C3）出结果再跑
    python scripts/plan_b_drop_p1b.py --skip-wait      # 不等，立即跑
    python scripts/plan_b_drop_p1b.py --arms p_a_z --iters 20   # 试跑
"""
from __future__ import annotations

import argparse
import ctypes
import json
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = r"C:\Users\LIGF\.workbuddy\binaries\python\envs\zhuoji\Scripts\python.exe"

LOG = ROOT / "reports" / "plan_b_drop_p1b.log"
SUMMARY = ROOT / "reports" / "plan_b_drop_p1b_summary.json"
#: Phase A（C3）的产出 —— 它的 H1/H2 判读会在本报告里被引用
C3_SUMMARY = ROOT / "reports" / "plan_c_summary.json"

INIT = "models/dushan_big_s11.pt"
ANCHOR = "dushan_big_s11"
POOL = "dushan_big_s3,dushan_big_s7,dushan_big_s11"
REF = "net:dushan_big_s11|teacher:balanced|teacher:tenpai_rush"
SEED = 20270
GATE = 0.40
DEALS = 1200

#: 本批要**训练**的臂（按执行顺序）
ARM_ORDER = ["p_a_z", "p_c_z", "p_ac_z"]

ARMS: dict[str, dict] = {
    "p_a_z": {"packs": "a", "desc": "零初始化 + 只开 P1a 牌河时序"},
    "p_c_z": {"packs": "c", "desc": "零初始化 + 只开 P1c 番型潜力/巡目"},
    "p_ac_z": {"packs": "ac", "desc": "零初始化 + a+c 合并（去掉 b 的候选组合）"},
}

#: 复评清单。分三组，全部丢进同一批牌池 → 配对可比。
RESCORE_NEW = ["dushan_b2_p_a_z", "dushan_b2_p_c_z", "dushan_b2_p_ac_z"]
RESCORE_REUSE = [
    # 第一批（随机初始化）：同一批牌池下与"零初始化"版直接对照，是 H1/H2 的旁证
    "dushan_b2_p_none", "dushan_b2_p_a", "dushan_b2_p_c", "dushan_b2_p_abc",
    "dushan_b2_p_b",
    # Phase A 的决定性对照 + 纯扰动对照（只热启动、零训练）
    "dushan_b2c_p_abc_zero", "_diag_c1_v2warm",
]
#: 名字 → 实际 .pt 文件名（复评脚本内部按 models/{stem}.pt 拼路径）
RESCORE_STEM = {"_diag_c1_v2warm": "_diag_c1_v2warm"}

ES_CONTINUOUS = 0x80000000
ES_SYSTEM_REQUIRED = 0x00000001


def awake(on: bool = True) -> bool:
    try:
        ctypes.windll.kernel32.SetThreadExecutionState(
            (ES_CONTINUOUS | ES_SYSTEM_REQUIRED) if on else ES_CONTINUOUS)
        return True
    except Exception:
        return False


def log_open():
    return open(LOG, "a", encoding="utf-8", buffering=1)


def c3_verdict() -> dict:
    """把 Phase A 的 H1/H2 判读抽出来（拿不到就返回空）。"""
    if not C3_SUMMARY.exists():
        return {}
    try:
        d = json.loads(C3_SUMMARY.read_text(encoding="utf-8"))
    except Exception:
        return {}
    out = {"verdict_lines": [], "rows": {}}
    for ln in (d.get("train", {}).get("select_lines") or []):
        out["verdict_lines"].append(ln)
    for k, r in (d.get("rescore") or {}).items():
        if isinstance(r, dict):
            out["rows"][k] = {"vs_anchor": r.get("vs_anchor"),
                              "ci95": r.get("vs_anchor_ci95")}
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-wait", action="store_true", help="不等 Phase A（C3）")
    ap.add_argument("--arms", default=",".join(ARM_ORDER))
    ap.add_argument("--iters", type=int, default=400)
    ap.add_argument("--threads", type=int, default=24)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--deals", type=int, default=DEALS)
    ap.add_argument("--max-hours", type=float, default=10.0)
    ap.add_argument("--skip-score", action="store_true")
    ap.add_argument("--force-retrain", action="store_true",
                    help="即使该臂已有模型也重训（默认跳过已完成的臂）")
    args = ap.parse_args()

    log = log_open()

    def say(msg: str) -> None:
        print(f"[{time.strftime('%m-%d %H:%M:%S')}] {msg}", file=log)

    want = [a.strip() for a in args.arms.split(",") if a.strip()]
    bad = [a for a in want if a not in ARMS]
    if bad:
        raise SystemExit(f"未知的臂：{bad}；可选 {list(ARMS)}")

    say("=" * 68)
    say(f"方案 B（去 P1b 重做版）：训练 {len(want)} 臂 × {args.iters} 轮"
        f"（threads={args.threads} workers={args.workers}，预算 {args.max_hours} h）")
    for a in want:
        say(f"  · {a:<8} packs={ARMS[a]['packs']:<4} {ARMS[a]['desc']}")

    # ---- 0. 等 Phase A（C3）出结果 -----------------------------------------
    if not args.skip_wait:
        t_wait = time.time()
        while not C3_SUMMARY.exists():
            if time.time() - t_wait > 4 * 3600:
                say("等待 Phase A（C3）超过 4 小时，放弃等待（请人工检查）")
                return 1
            if int(time.time() - t_wait) % 300 < 20:
                say(f"等待 C3 汇总 …（已等 {(time.time() - t_wait) / 60:.0f} 分钟）")
            time.sleep(30)
        say(f"Phase A（C3）已就位（等待 {(time.time() - t_wait) / 60:.0f} 分钟）")
    else:
        say("--skip-wait：不等待 Phase A")

    v = c3_verdict()
    if v.get("rows"):
        say("—— Phase A（C3）判读（引用，供本批对照）——")
        for k, r in v["rows"].items():
            say(f"    {k:<24} {r['vs_anchor']:+.3f}"
                if isinstance(r.get("vs_anchor"), (int, float))
                else f"    {k}: {r}")

    say(f"防休眠={'OK' if awake(True) else '失败'}")

    # ---- 1. 逐臂训练 --------------------------------------------------------
    state: dict = {"arms": {}, "iters": args.iters, "gate": GATE, "anchor": ANCHOR,
                   "ref": REF, "deals": args.deals, "seed": SEED,
                   "new_param_init": "zero",
                   "note": "去掉 P1b，保留 a/c；全部零初始化热启动",
                   "phase_a_c3": v}
    if SUMMARY.exists():
        try:
            old = json.loads(SUMMARY.read_text(encoding="utf-8"))
            if old.get("arms"):
                state["arms"] = old["arms"]
                say(f"已有 {len(old['arms'])} 臂历史结果（完成臂将跳过）")
        except Exception:
            pass

    t_start = time.time()
    spent = sum(float(r.get("hours") or 0.0) for r in state["arms"].values())

    for arm in want:
        out = ROOT / "models" / f"dushan_b2_{arm}.pt"
        rec0 = state["arms"].get(arm) or {}
        if out.exists() and rec0.get("finished") and not args.force_retrain:
            say(f"[skip] {arm} 已有训练成果，跳过")
            continue
        if spent >= args.max_hours:
            say(f"[budget] 已用 {spent:.2f} h ≥ 预算 {args.max_hours} h，停止启动新臂")
            break

        spec = ARMS[arm]
        glog = ROOT / "reports" / f"dushan_b2_{arm}.log"
        cmd = [PY, "scripts/selfplay.py",
               "--init", INIT, "--out", f"models/dushan_b2_{arm}.pt",
               "--history", f"reports/rl_history_b2_{arm}.json",
               "--iters", str(args.iters), "--games-per-iter", "96",
               "--batch", "512", "--lr", "8.0e-6", "--temperature", "0.95",
               "--teacher-prob", "0.08",
               "--threads", str(args.threads), "--vectorized",
               "--workers", str(args.workers),
               "--eval-every", "10", "--eval-rounds", "40",
               "--select", "guard", "--select-metric", "vs_anchor",
               "--eval-anchor", ANCHOR,
               "--save-every", "50", "--snapshot-pool", POOL,
               "--encoder", "v2", "--encoder-packs", spec["packs"],
               "--spatial-head", "off",
               "--new-param-init", "zero",     # ← 本批的"修"
               "--seed", str(SEED), "--rules", "dushan"]
        say(f"—— {arm}：{spec['desc']} ——")
        say("  cmd: " + " ".join(cmd[2:]))
        t0 = time.time()
        with open(glog, "w", encoding="utf-8") as f:
            rc = subprocess.run(cmd, cwd=str(ROOT), stdout=f,
                                stderr=subprocess.STDOUT).returncode
        dt = time.time() - t0
        spent += dt / 3600
        text = glog.read_text(encoding="utf-8", errors="ignore")
        state["arms"][arm] = {
            "desc": spec["desc"], "packs": spec["packs"], "cmd": cmd[2:], "rc": rc,
            "hours": round(dt / 3600, 3),
            "finished": "完成。" in text,
            "s_per_iter": round(dt / max(1, args.iters), 2),
            "head": [ln for ln in text.splitlines()
                     if ln.startswith(("[warm]", "[编码]", "[pool]"))],
            "select_lines": [ln for ln in text.splitlines()
                             if ln.startswith("[select]")],
        }
        SUMMARY.write_text(json.dumps(state, ensure_ascii=False, indent=2),
                           encoding="utf-8")
        for ln in state["arms"][arm]["head"]:
            say("  " + ln)
        say(f"  {arm} 训练结束 rc={rc} 完成={state['arms'][arm]['finished']} "
            f"{dt / 3600:.2f} h（{state['arms'][arm]['s_per_iter']} s/轮）；"
            f"累计 {spent:.2f} h")
        if rc != 0 or not out.exists():
            say(f"  {arm} 未正常产出模型，跳过复评")

    # ---- 2. 同一批牌池复评（新臂 + 复用对照 + C3 + 纯扰动） -----------------
    trained = [a for a in want if (ROOT / "models" / f"dushan_b2_{a}.pt").exists()]
    if not args.skip_score:
        ckpts = [ANCHOR] + RESCORE_NEW + RESCORE_REUSE
        have = [c for c in ckpts
                if (ROOT / "models" / f"{RESCORE_STEM.get(c, c)}.pt").exists()]
        missing = [c for c in ckpts if c not in have]
        if missing:
            say(f"复评跳过缺失的检查点：{', '.join(missing)}")
        say(f"复评：{len(have)} 个检查点 × {args.deals} 副"
            f"（共用牌池 + 共用对手，配对可比）")
        out_json = ROOT / "reports" / "plan_b_drop_p1b_rescore.json"
        subprocess.run(
            [PY, "scripts/score_checkpoints.py",
             "--ckpts", ",".join(have), "--ref", REF,
             "--deals", str(args.deals), "--threads", str(args.threads),
             "--anchor", ANCHOR, "--out", str(out_json.relative_to(ROOT)),
             "--no-per-deal"],
            cwd=str(ROOT), stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
        if out_json.exists():
            sc = json.loads(out_json.read_text(encoding="utf-8"))
            rows = {r["ckpt"]: r for r in sc.get("rows", [])}
            state["rescore"] = rows
            state["rescore_ckpts"] = have
            say("—— 复评结果（相对 s11 的 1200 副配对净分差；门槛 ≥ +0.40 且 CI 下界 > 0）——")
            for c in have:
                r = rows.get(c) or {}
                val = r.get("vs_anchor")
                ci = r.get("vs_anchor_ci95")
                if not isinstance(val, (int, float)):
                    say(f"  {c:<24} （无数据）")
                    continue
                lo = val - (ci or 0.0)
                ok = val >= GATE and lo > 0
                say(f"  {c:<24} {val:+.3f} ± {(ci or 0):.3f}  CI下界 {lo:+.3f}  → "
                    f"{'过门槛 ✅' if ok else '未过门槛'}")
            SUMMARY.write_text(json.dumps(state, ensure_ascii=False, indent=2),
                               encoding="utf-8")
        else:
            say("复评未产出 json")

    awake(False)
    state["elapsed_hours"] = round((time.time() - t_start) / 3600, 2)
    SUMMARY.write_text(json.dumps(state, ensure_ascii=False, indent=2),
                       encoding="utf-8")
    say(f"队列结束，累计 {state['elapsed_hours']} h，汇总 -> {SUMMARY}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
