#!/usr/bin/env python3
"""训练饱和早停判读守卫（docs/14天训练计划.md §6.43，纯只读 + 一次性写标记）。

用户 2026-11-07 22:1x 口径：「不要求强行用完所有时间，一旦发现训练的循环
无法提升模型性能了，即可做终评」。但「发现无法提升」必须由看结果前锁死的
判据定义，不能由每小时自检临场判断（事后挑显著）。本工具就是那把尺子。

设计约束
--------
* **零对局**：本工具不跑 eval_pairs，只读已落盘的独立池 jsonl；测量由自检在
  DAgger train 段（CPU 空闲窗口）单独执行，严禁与 annotate 并发。
* **零删除**：只覆盖写 runs/shadow/early_stop_check.json；`--arm` 最多写一次
  runs/early_stop（已存在则不再写）。不 unlink / 不 rmtree（safe-delete 守卫）。
* **不改配方**：不碰任何超参、闸门判据、ckpt、exit_best、线上模型。

判据（§6.43.2，2026-11-07 22:1x 锁死，看到任何早停数据之前）
------------------------------------------------------------
检查点：仅 S 周期边界 r = 36/48/60/…（DAgger 轮 `(r-13)%12==11`），R24 不检查。
最早 R36；若 runs/gates/plan_b_enabled 存在且边界 < R48 ⇒ 拒绝早停。

主判据 M（唯一裁决量，全新独立池 600 seed，seed0 = 750000 + 600k）：
  A = 本周期闸门 A 均值最高的 ckpt，B = 上一周期同规则 ckpt，逐 seed 配对。
  Δ <= +0.50 且 p >= 0.10  ⇒ 无提升（M 成立）
  Δ >= +1.00 且 p <  0.05  ⇒ 明确仍在提升 ⇒ 继续训练
  其余（中间带）            ⇒ 证据不足 ⇒ 继续训练
辅证（四条全成立才可停）：
  S1 闸门 A 趋势：|slope| < 0.10 分/轮 或 p > 0.05（天花板压缩，只作辅证）
  S2 探索枯竭：ent_rollout < 0.80，或最近 6 轮相对变化 < 5%
  S3 重锚无新信息：本周期 DAgger 轮 A 均值 <= 上一周期 DAgger 轮 + 1×SE
  S4 链健康：本周期 exit_p*.pt 无缺号、无 rapid_abort / ppo_disabled、无 panic

用法::
  # 只读判读（默认，恒不写 early_stop）
  .venv/bin/python tools/early_stop_guard.py --boundary 36 --measure runs/gates/early_stop_r36.jsonl \
      --a models/exit_pXX.pt --b models/exit_pYY.pt
  # 判据全通过才写标记
  .venv/bin/python tools/early_stop_guard.py --boundary 36 ... --arm
  # 用户手工通道（owner 覆盖，跳过统计判据）
  .venv/bin/python tools/early_stop_guard.py --manual --reason "用户指令：提前终评"

退出码：0 正常（含 continue）、2 边界/门槛不合格或 arm 被拒、3 缺独立池测量。
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
GATES = ROOT / "runs" / "gates"
LOGS = ROOT / "runs" / "night_logs"
MODELS = ROOT / "models"
SHADOW = ROOT / "runs" / "shadow"
MARKER = ROOT / "runs" / "early_stop"
REPORT = SHADOW / "early_stop_check.json"
PLAN_B = GATES / "plan_b_enabled"

# ---- 预登记常量（改这里 = 改预登记，必须同步 docs/14天训练计划.md §6.43.2）----
M_NO_GAIN_DELTA, M_NO_GAIN_P = 0.50, 0.10
M_GAIN_DELTA, M_GAIN_P = 1.00, 0.05
S1_SLOPE, S1_P = 0.10, 0.05
S2_ENT_FLOOR, S2_ENT_FLAT = 0.80, 0.05
S3_SE_MULT = 1.0
EARLIEST_BOUNDARY = 36
POOL_BASE, POOL_STEP, POOL_SIZE = 750000, 600, 600


def p_two_sided(z: float) -> float:
    return math.erfc(abs(z) / math.sqrt(2.0))


def mean_se(x) -> tuple[float, float]:
    a = np.asarray(x, dtype=np.float64)
    n = len(a)
    return (float(a.mean()), float(a.std(ddof=1) / math.sqrt(n))) if n > 1 else (0.0, float("nan"))


def excess(j: dict) -> float:
    """与 final_eval_report.py 一致：候选超额分。"""
    ds = j["deltas"]
    c = int(j["cand_seat"])
    return float(ds[c]) - (sum(ds) - ds[c]) / 3.0


def gate_path(r: int, gate: str = "A") -> Path | None:
    for prefix in ("p", "r"):
        p = GATES / f"{prefix}{r}_gate{gate}.jsonl"
        if p.exists():
            return p
    return None


def gate_stats(r: int, gate: str = "A") -> dict | None:
    """该轮 300-seed 四座旋转的逐 seed 均值的 mean / SE。"""
    p = gate_path(r, gate)
    if p is None:
        return None
    by_seed: dict[int, list[float]] = {}
    for line in p.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        j = json.loads(line)
        by_seed.setdefault(int(j["seed"]), []).append(excess(j))
    per_seed = [float(np.mean(v)) for v in by_seed.values()]
    m, se = mean_se(per_seed)
    return {"round": r, "path": str(p.relative_to(ROOT)), "n": len(per_seed),
            "mean": round(m, 4), "se": round(se, 4)}


def ckpt_for(r: int) -> str:
    for stem in (f"exit_p{r}.pt", f"exit_r{r}.pt"):
        if (MODELS / stem).exists():
            return f"models/{stem}"
    return f"models/exit_p{r}.pt (缺失)"


def pick_strongest(lo: int, hi: int) -> dict | None:
    """周期 [lo, hi] 内闸门 A 均值最高的轮（只比数值，不做显著性比较）。"""
    stats = [gate_stats(r) for r in range(lo, hi + 1)]
    stats = [s for s in stats if s is not None]
    if not stats:
        return None
    best = max(stats, key=lambda s: s["mean"])
    return {**best, "ckpt": ckpt_for(best["round"])}


def used_seeds(exclude: set[Path] | None = None) -> set[int]:
    """已使用过的 seed 池；必须排除本次测量文件自身，否则自己的 seed 会把自己判脏。"""
    skip = {p.resolve() for p in (exclude or set())}
    out: set[int] = set()
    for p in GATES.glob("*.jsonl"):
        if p.resolve() in skip:
            continue
        for line in p.read_text(encoding="utf-8", errors="replace").splitlines():
            if line.strip().startswith('{"seed"'):
                try:
                    out.add(int(json.loads(line)["seed"]))
                except (ValueError, KeyError, json.JSONDecodeError):
                    pass
    return out


def next_pool_seed0() -> int:
    """按 §6.43.2「seed0 = 750000 + 600k，k 依次递增、永不复用」取下一个干净池。

    修正（§6.57.1，2026-11-08 10:5x，R48 边界前、看结果前）：此前 measure_cmd 恒用
    POOL_BASE(=750000)，而该池已在 R36 边界被 early_stop_r36.jsonl 消耗 ⇒ 按原命令
    跑出来的测量必然 overlap>0、被 measure_stats 判 fresh_pool=False ⇒
    verdict=continue_invalid_pool（白跑一次 600 seed 且不出结论）。
    本函数只落实预登记里的「k 递增」，**不改任何阈值、不改任何判定分支**。
    """
    used = used_seeds()
    for k in range(0, 40):
        s0 = POOL_BASE + k * POOL_STEP
        if not (used & set(range(s0, s0 + POOL_SIZE))):
            return s0
    return POOL_BASE + 40 * POOL_STEP


def measure_stats(path: Path) -> dict:
    by_seed: dict[int, list[float]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        j = json.loads(line)
        by_seed.setdefault(int(j["seed"]), []).append(excess(j))
    per_seed = [float(np.mean(v)) for v in by_seed.values()]
    m, se = mean_se(per_seed)
    p = p_two_sided(m / se) if se and se > 0 else 1.0
    seeds = set(by_seed)
    in_pool = any(POOL_BASE + k * POOL_STEP <= min(seeds) <
                  POOL_BASE + k * POOL_STEP + POOL_SIZE and
                  max(seeds) < POOL_BASE + k * POOL_STEP + POOL_SIZE
                  for k in range(0, 40))
    overlap = len(seeds & used_seeds(exclude={path.resolve()}))
    return {"path": str(path.relative_to(ROOT)), "n_seed": len(seeds),
            "delta": round(m, 4), "se": round(se, 4), "p": round(p, 6),
            "fresh_pool": bool(in_pool and overlap == 0),
            "overlap_with_used": overlap,
            "rows": sum(len(v) for v in by_seed.values())}


def ols_slope(xs, ys) -> tuple[float, float, float]:
    x = np.asarray(xs, dtype=np.float64)
    y = np.asarray(ys, dtype=np.float64)
    if len(x) < 3:
        return 0.0, float("nan"), 1.0
    slope, intercept = np.polyfit(x, y, 1)
    resid = y - (slope * x + intercept)
    s2 = float(resid @ resid) / (len(x) - 2)
    se = math.sqrt(s2 / float(((x - x.mean()) ** 2).sum())) if s2 > 0 else 0.0
    p = p_two_sided(slope / se) if se > 0 else 1.0
    return float(slope), float(se), float(p)


def s1_trend(boundary: int) -> dict:
    stats = [gate_stats(r) for r in range(boundary - 11, boundary + 1)]
    stats = [s for s in stats if s is not None]
    if len(stats) < 3:
        return {"ok": False, "reason": "本周期闸门 A 数据不足 3 轮"}
    slope, se, p = ols_slope([s["round"] for s in stats], [s["mean"] for s in stats])
    ok = abs(slope) < S1_SLOPE or p > S1_P
    return {"ok": bool(ok), "slope": round(slope, 5), "se": round(se, 5),
            "p": round(p, 5), "n": len(stats)}


def s2_entropy() -> dict:
    vals: dict[int, float] = {}
    num = re.compile(r"ppo_p(\d+)\.log$")
    for p in sorted(LOGS.glob("ppo_p*.log")):
        m = num.search(p.name)
        if not m:
            continue
        last = None
        for line in p.read_text(encoding="utf-8", errors="replace").splitlines():
            if line.startswith("[metrics] "):
                try:
                    last = json.loads(line[len("[metrics] "):])
                except json.JSONDecodeError:
                    pass
        if last and "ent_rollout" in last:
            try:
                vals[int(m.group(1))] = float(last["ent_rollout"])
            except (TypeError, ValueError):
                pass
    recent = [v for _, v in sorted(vals.items())[-6:]]
    if len(recent) < 6:
        return {"ok": False, "reason": "ent_rollout 不足 6 轮", "recent": recent}
    flat = (max(recent) - min(recent)) / max(recent) < S2_ENT_FLAT if max(recent) > 0 else False
    ok = min(recent) < S2_ENT_FLOOR or flat
    return {"ok": bool(ok), "recent": [round(v, 4) for v in recent],
            "min": round(min(recent), 4), "flat": bool(flat)}


def s3_reanchor(boundary: int) -> dict:
    cur, prev = gate_stats(boundary), gate_stats(boundary - 12)
    if cur is None or prev is None:
        return {"ok": False, "reason": "本/上周期 DAgger 轮闸门 A 缺失"}
    se = math.sqrt(cur["se"] ** 2 + prev["se"] ** 2)
    delta = cur["mean"] - prev["mean"]
    ok = delta <= S3_SE_MULT * se
    return {"ok": bool(ok), "cur": cur, "prev": prev,
            "delta": round(delta, 4), "se_combined": round(se, 4)}


def s4_health(boundary: int) -> dict:
    missing = [r for r in range(boundary - 11, boundary)
               if not (MODELS / f"exit_p{r}.pt").exists()]
    blocks = [str(p.name) for p in (ROOT / "runs" / "rapid_abort",
                                    ROOT / "runs" / "ppo_disabled") if p.exists()]
    panic = False
    for p in LOGS.glob(f"*_p*.log"):
        tail = p.read_text(encoding="utf-8", errors="replace")[-4000:]
        if "panic" in tail.lower() or "nan loss" in tail.lower():
            panic = True
            break
    ok = not missing and not blocks and not panic
    return {"ok": bool(ok), "missing_ppo_ckpt": missing,
            "block_markers": blocks, "panic_in_log": panic}


def boundary_ok(r: int) -> tuple[bool, str]:
    if (r - 13) % 12 != 11:
        return False, f"R{r} 不是 S 周期边界（DAgger 轮需 (r-13)%12==11）"
    if r < EARLIEST_BOUNDARY:
        return False, f"早于 R{EARLIEST_BOUNDARY}：R24 边界不检查（第三周期决策点未到）"
    if not (LOGS / f"gateD_r{r}.log").exists():
        return False, f"R{r} 完整 DAgger 轮尚未完成（缺 gateD_r{r}.log），不得提前判读"
    if PLAN_B.exists() and r < 48:
        return False, "方案 B 已启用但尚未到 R48 生效检验轮，不得提前停"
    return True, "合格"


def build_report(boundary: int, measure: Path | None, a: str, b: str) -> dict:
    ok, why = boundary_ok(boundary)
    rep: dict = {"ts": dt.datetime.now().isoformat(timespec="seconds"),
                 "boundary": boundary, "boundary_ok": ok, "boundary_reason": why,
                 "period": {"cur": [boundary - 11, boundary],
                            "prev": [boundary - 23, boundary - 12]},
                 "constants": {"M_NO_GAIN_DELTA": M_NO_GAIN_DELTA, "M_NO_GAIN_P": M_NO_GAIN_P,
                               "M_GAIN_DELTA": M_GAIN_DELTA, "M_GAIN_P": M_GAIN_P,
                               "S1_SLOPE": S1_SLOPE, "S2_ENT_FLOOR": S2_ENT_FLOOR,
                               "S2_ENT_FLAT": S2_ENT_FLAT, "EARLIEST": EARLIEST_BOUNDARY},
                 "candidate_A": a, "candidate_B": b}
    rep["strongest_cur"] = pick_strongest(boundary - 11, boundary)
    rep["strongest_prev"] = pick_strongest(boundary - 23, boundary - 12)
    if not ok or rep["strongest_cur"] is None or rep["strongest_prev"] is None:
        # 未到边界 / 数据不齐 ⇒ 绝不生成残缺的测量命令，也绝不 arm。
        rep["verdict"] = "not_ready"
        rep["measure_cmd"] = None
        return rep
    rep["S1"] = s1_trend(boundary)
    rep["S2"] = s2_entropy()
    rep["S3"] = s3_reanchor(boundary)
    rep["S4"] = s4_health(boundary)
    if measure is None or not measure.exists():
        rep["M"] = None
        rep["measure_cmd"] = (
            f".venv/bin/python eval_pairs.py --a {a} --b {b} --mode greedy "
            f"--seeds 600 --seed0 {next_pool_seed0()} --workers 6 "
            f"--out runs/gates/early_stop_r{boundary}.jsonl")
        rep["verdict"] = "need_measure"
        return rep
    m = measure_stats(measure)
    rep["M"] = m
    if not m["fresh_pool"]:
        rep["verdict"] = "continue_invalid_pool"
        return rep
    if m["delta"] >= M_GAIN_DELTA and m["p"] < M_GAIN_P:
        rep["verdict"] = "continue_gain"
        return rep
    no_gain = m["delta"] <= M_NO_GAIN_DELTA and m["p"] >= M_NO_GAIN_P
    aux = all(bool(rep[k]["ok"]) for k in ("S1", "S2", "S3", "S4"))
    rep["verdict"] = ("early_stop" if ok and no_gain and aux
                      else "continue_insufficient")
    return rep


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--boundary", type=int, default=36)
    ap.add_argument("--measure", type=str, default="")
    ap.add_argument("--a", type=str, default="")
    ap.add_argument("--b", type=str, default="")
    ap.add_argument("--arm", action="store_true", help="仅判据全通过才写 runs/early_stop")
    ap.add_argument("--manual", action="store_true", help="用户手工早停通道（跳过统计判据）")
    ap.add_argument("--reason", type=str, default="")
    args = ap.parse_args()

    if MARKER.exists():
        print(f"[early-stop] 标记已存在：{MARKER}，不重复写。当前状态：训练应已停止。")
        return 0

    if args.manual:
        payload = {"mode": "manual", "reason": args.reason or "用户明确指令：提前终评",
                   "ts": dt.datetime.now().isoformat(timespec="seconds"),
                   "note": "owner 覆盖通道（§6.43.3），未使用统计判据"}
        MARKER.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[early-stop] 手工通道已写 {MARKER}")
        return 0

    a = args.a or (pick_strongest(args.boundary - 11, args.boundary) or {}).get("ckpt", "")
    b = args.b or (pick_strongest(args.boundary - 23, args.boundary - 12) or {}).get("ckpt", "")
    measure = Path(args.measure) if args.measure else None
    if measure is not None and not measure.is_absolute():
        measure = ROOT / measure
    rep = build_report(args.boundary, measure, a, b)
    SHADOW.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(rep, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"=== 早停判读 R{args.boundary}（§6.43.2 锁死判据）===")
    print(f"  边界合法性 : {rep['boundary_ok']}  {rep['boundary_reason']}")
    print(f"  本周期最强 : {rep['strongest_cur']}")
    print(f"  上周期最强 : {rep['strongest_prev']}")
    if rep.get("M") is None:
        print("  主判据 M   : 缺独立池测量，请先执行：")
        print("    " + (rep.get("measure_cmd") or "（未到边界或候选未齐，暂不生成命令）"))
    else:
        m = rep["M"]
        print(f"  主判据 M   : Δ={m['delta']:+.4f}±{m['se']:.4f} p={m['p']:.4f} "
              f"fresh={m['fresh_pool']} (n={m['n_seed']})")
    if "S1" in rep:
        for k in ("S1", "S2", "S3", "S4"):
            print(f"  {k}          : {rep[k]['ok']}  {rep[k]}")
    print(f"  verdict    : {rep['verdict']}   报告 {REPORT}")

    if rep["verdict"] == "need_measure":
        return 3
    if rep["verdict"] == "not_ready":
        return 2
    if args.arm:
        if rep["verdict"] != "early_stop":
            print(f"[early-stop] --arm 被拒：verdict={rep['verdict']}（不写标记）")
            return 2
        payload = {"mode": "statistical", "boundary": args.boundary,
                   "ts": dt.datetime.now().isoformat(timespec="seconds"),
                   "delta": rep["M"]["delta"], "se": rep["M"]["se"], "p": rep["M"]["p"],
                   "candidate_A": a, "candidate_B": b,
                   "aux": {k: rep[k]["ok"] for k in ("S1", "S2", "S3", "S4")},
                   "note": "§6.43.2 判据全通过；ns 不等于已证明到顶，见手册报告口径"}
        MARKER.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[early-stop] 已写 {MARKER} ⇒ 优雅停机后按终评手册执行")
    return 0


if __name__ == "__main__":
    sys.exit(main())
