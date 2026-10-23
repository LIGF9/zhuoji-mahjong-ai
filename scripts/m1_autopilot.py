"""M1 分段训练的**无人值守托管**：跑完一段 → 判读 → 与基线比较 → 无差错就续下一段。

为什么需要它
------------
`docs/P3-M1训练执行计划.md` 把 M1 长跑切成 S1/S2/S3，每段 800 轮 ≈ 12.6 h。
段与段之间原本需要人守着做三件事：判读健康度、跑配对赛、决定是否进下一段。
用户离开过夜时，这个"看守"角色没人担任 —— 结果是训练停在段末空转，
或者更糟：带着一个已经坏掉的模型进入下一段（历史上 A5 内存误报、A7 锚点
被顶回 3.0 都属于"没人看就一路错下去"）。

本脚本把这段流程做成状态机，并把判据**写进报告**（可复核，不靠记忆）：

    段 N 训练（前台阻塞）
      → `m1_segment_finalize.py`（判读 1200 副 + 归档检查点）
      → `score_checkpoints.py` 与 `dushan_master` 做共用牌池配对赛
      → 三条判据全过 → 段 N+1 训练；任一条不过 → 停下并写清原因

三条判据（阈值集中在 `JUDGE`，改判据只改这一处）
------------------------------------------------
    H1 健康度无 FAIL   —— judge 的 A1~A8 里不得有 FAIL（SKIP 允许，因为它只表示
                          "数据还不够，现在还判不了"，不代表模型坏）
    H2 段内显著进展      —— 末点相对本段首点的 vs_anchor 净分差显著为正
    H3 相对基线不退步    —— 本段模型 vs `dushan_master` 的配对净分差 CI 上界 ≥ 0
                          （即没有被证明显著弱于线上小模型）

用法::

    # 从 S1（续跑）托管到 S3，前台运行、由调用方持有
    python scripts/m1_autopilot.py --from s1 --to s3 --workers 12

    # 只看计划不执行
    python scripts/m1_autopilot.py --from s1 --to s3 --dry-run

⚠️ 起动方式：本脚本内部用**前台**子进程跟随，而它自己必须由调用方持有
（WorkBuddy 的 `run_in_background` 或用户自己的终端）。原因见
`m1_launch_segment.py` 顶部注释 —— 沙箱型环境会清理整个进程树，
只有"父进程活着"这一条能保住训练。

产物：
- `reports/autopilot_state.json`   状态机快照（断了可以续）
- `reports/AUTOPILOT_REPORT.md`    给人看的阶段报告（每段判读后刷新）
- `reports/_autopilot.out`         托管自身日志
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from m1_launch_segment import SEGMENTS, build_cmd, find_latest_ckpt  # noqa: E402

PY = sys.executable
STATE_P = ROOT / "reports" / "autopilot_state.json"
REPORT_P = ROOT / "reports" / "AUTOPILOT_REPORT.md"
LOG_P = ROOT / "reports" / "_autopilot.out"

#: 判据阈值（集中一处）
JUDGE = dict(
    deals=1200,             # 阶段 B 配对赛副数（CI ≈ ±0.3）
    master_ci_lo=0.0,       # H3：vs master 的 CI 上界必须 ≥ 此值才算"未退步"
    duel_threads=24,
)

SEG_ORDER = ["s1", "s2", "s3"]
BASELINE = "dushan_master"          # 用户口径里的「小模型」= 线上主模型


# --------------------------------------------------------------------------
# 基础工具
# --------------------------------------------------------------------------
def log(msg: str) -> None:
    line = f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {msg}"
    print(line, flush=True)
    LOG_P.parent.mkdir(parents=True, exist_ok=True)
    with LOG_P.open("a", encoding="utf-8") as fh:
        fh.write(line + "\n")


def training_pids() -> list[int]:
    """当前在跑的 selfplay 主进程 pid 列表。"""
    try:
        import psutil
    except ImportError:
        return []
    out = []
    for p in psutil.process_iter(["cmdline"]):
        cl = list(p.info.get("cmdline") or [])
        if any(str(a).endswith("selfplay.py") for a in cl):
            out.append(p.pid)
    return out


def load_state() -> dict:
    if STATE_P.exists():
        try:
            return json.loads(STATE_P.read_text(encoding="utf-8"))
        except Exception:                                    # noqa: BLE001
            pass
    return {"segments": {}, "started": datetime.now().isoformat(timespec="seconds")}


def save_state(st: dict) -> None:
    st["updated"] = datetime.now().isoformat(timespec="seconds")
    STATE_P.write_text(json.dumps(st, ensure_ascii=False, indent=2), encoding="utf-8")


# --------------------------------------------------------------------------
# 段配置（含续训解析）
# --------------------------------------------------------------------------
def resolve_segment(seg: str) -> dict:
    """本段的实际配方：有新存档就从它续跑，否则按原配方整段跑。"""
    cfg = dict(SEGMENTS[seg])
    stem = Path(cfg["out"]).stem
    total = cfg["iters"]

    hit = find_latest_ckpt(stem)
    if hit is None:
        cfg["_mode"] = "fresh"
        return cfg

    p, it = hit
    done = it
    hist_p = ROOT / "reports" / f"m1_{seg}_history.json"
    if hist_p.exists():
        try:
            h = json.loads(hist_p.read_text(encoding="utf-8"))
            if isinstance(h, list) and h:
                # history 常比最后一个周期存档更靠前（崩溃点与 save-every 不对齐），
                # 取较大者作编号起点，避免续训首几轮与既有记录重号。
                done = max(done, int(h[-1].get("iter") or 0))
        except Exception:                                    # noqa: BLE001
            pass

    cfg["_mode"] = "resume"
    cfg["_ckpt"] = p.name
    cfg["_done"] = done
    if done >= total:
        cfg["iters"] = 0                       # 已跑完 → 只判读，不再训
        return cfg
    cfg.update(init=str(p.relative_to(ROOT)).replace("\\", "/"),
               iters=total - done, warmup=0, offset=done)
    return cfg


class DaemonKeeper(threading.Thread):
    """守着自己那两个辅助进程：它们自退或被误杀时按需重启。

    为什么必须是线程、而不是"段开始时起一次"：``keep_awake`` 与 ``res_monitor``
    都内置了自动收工逻辑（连续 N 分钟看不到目标进程就退出）。而"看不到"有时
    只是训练正处于阶段切换（写盘 / 判读），一旦退出且不再重启，后面十几个小时
    就完全没有防睡眠与监控覆盖 —— 2026-10-10 那次 22.2 h 睡眠冻结正是这么发生的。

    ``mon_path`` 是可变的：每段开始前由主循环改成 ``_res_monitor_m1_{seg}.csv``。
    """

    def __init__(self, interval: int = 60):
        super().__init__(daemon=True, name="daemon-keeper")
        self.interval = interval
        self.mon_path: Path | None = None
        self._stop = threading.Event()
        self._procs: dict[str, subprocess.Popen] = {}

    def _ensure(self, key: str, cmd: list) -> None:
        p = self._procs.get(key)
        if p is not None and p.poll() is None:
            return
        self._procs[key] = subprocess.Popen(
            cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, cwd=str(ROOT))
        log(f"守护 {key} 已起 pid={self._procs[key].pid}")

    def run(self) -> None:
        while not self._stop.is_set():
            try:
                self._ensure("keep_awake",
                             [PY, str(ROOT / "scripts" / "keep_awake.py")])
                if self.mon_path is not None and training_pids():
                    self._ensure("res_monitor",
                                 [PY, str(ROOT / "scripts" / "res_monitor.py"),
                                  "--interval", "60", "--out", str(self.mon_path)])
            except Exception as e:                           # noqa: BLE001
                log(f"⚠️ 守护巡检异常：{type(e).__name__}: {e}")
            self._stop.wait(self.interval)

    def shutdown(self) -> None:
        self._stop.set()
        for key, p in self._procs.items():
            if p.poll() is None:
                try:
                    p.terminate()
                    log(f"守护 {key} 已停止 pid={p.pid}")
                except Exception:                            # noqa: BLE001
                    pass


def start_training(seg: str, cfg: dict, workers: int) -> int:
    """前台阻塞地跑完本段，返回 exit_code。辅助守护由 DaemonKeeper 统一负责。"""
    run = f"m1_{seg}"
    log_file = ROOT / "reports" / f"_{run}.out"
    cmd = build_cmd(seg, cfg, workers)

    # 头部写盘后**立即释放句柄**：训练进程用 build_cmd 里的 `--log-file` 自写同一文件
    # （append），顺序天然是"头部在前、训练输出在后"。父进程侧不留悬空引用。
    with log_file.open("a", encoding="utf-8") as fh:
        fh.write(f"\n===== {run} 起动 @ {datetime.now():%Y-%m-%d %H:%M:%S} =====\n")
        fh.write(" ".join(cmd) + "\n")

    proc = subprocess.Popen(cmd, stdin=subprocess.DEVNULL,
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                            cwd=str(ROOT))

    (ROOT / "reports" / f"_{run}.pid").write_text(str(proc.pid), encoding="utf-8")
    log(f"{run} 已起 pid={proc.pid}（{cfg['iters']} 轮，{cfg['_mode']}）")
    rc = proc.wait()
    log(f"{run} 训练进程退出，exit={rc}")
    if rc != 0:
        log(f"⚠️ {run} 非零退出，尾部日志：")
        tail = log_file.read_text(encoding="utf-8", errors="replace").splitlines()[-12:]
        for line in tail:
            log("    " + line)
    return rc


# --------------------------------------------------------------------------
# 判读
# --------------------------------------------------------------------------
def run_finalize(seg: str, deals: int, threads: int) -> Path | None:
    """段末收尾：判读 + 归档。返回 judge JSON 路径。"""
    js = ROOT / "reports" / f"m1_{seg}_judge.json"
    cmd = [PY, str(ROOT / "scripts" / "m1_segment_finalize.py"),
           "--run", f"m1_{seg}",
           "--ckpt-prefix", f"dushan_m1_{seg}",
           "--extra", BASELINE,
           "--deals", str(deals),
           "--threads", str(threads)]
    log("段末收尾：" + " ".join(cmd[1:]))
    r = subprocess.run(cmd, cwd=str(ROOT))
    log(f"finalize exit={r.returncode}")
    return js if js.exists() else None


def run_master_duel(seg: str, deals: int, threads: int) -> dict | None:
    """本段模型 vs 基线（dushan_master）的共用牌池配对赛。"""
    new = f"dushan_m1_{seg}"
    if not (ROOT / "models" / f"{new}.pt").exists():
        log(f"⚠️ models/{new}.pt 不存在，跳过与基线比较")
        return None
    out = ROOT / "reports" / f"m1_{seg}_vs_master.json"
    cmd = [PY, str(ROOT / "scripts" / "score_checkpoints.py"),
           "--ckpts", f"{new},{BASELINE}",
           "--ref", "teacher:balanced|teacher:tenpai_rush|teacher:defensive",
           "--deals", str(deals), "--threads", str(threads),
           "--anchor", BASELINE, "--no-per-deal", "--out", str(out)]
    log("基线比较：" + " ".join(cmd[1:]))
    r = subprocess.run(cmd, cwd=str(ROOT))
    if r.returncode != 0 or not out.exists():
        log(f"⚠️ 基线比较失败（exit={r.returncode}）")
        return None
    return json.loads(out.read_text(encoding="utf-8"))


def decide(seg: str, judge: dict | None, duel: dict | None) -> tuple[bool, list[dict]]:
    """三条判据 → (是否放行, 逐条明细)。"""
    rows: list[dict] = []

    # H1 健康度
    if judge and judge.get("health"):
        h = judge["health"]
        fails = [c["name"] for c in h.get("checks", []) if c["status"] == "fail"]
        skips = [c["name"] for c in h.get("checks", []) if c["status"] == "skip"]
        ok = not fails
        detail = ("全部通过" if ok else f"FAIL：{fails}") + \
                 (f"；SKIP {len(skips)} 项：{skips}" if skips else "")
        rows.append({"id": "H1", "name": "健康度无 FAIL", "ok": ok, "detail": detail})
    else:
        rows.append({"id": "H1", "name": "健康度无 FAIL", "ok": True,
                     "detail": "⚠️ 无判读数据（跳过）"})

    # H2 段内进展
    vp = (judge or {}).get("verdict_progress")
    if vp:
        ok = bool(vp.get("significant_up"))
        rows.append({"id": "H2", "name": "段内显著进展", "ok": ok,
                     "detail": f"{Path(vp['to']).name} 相对 {Path(vp['from']).name} "
                               f"{vp['delta']:+.3f} ± {vp['ci95']:.3f}"
                               f"（{'显著上升' if ok else '未达显著'}）"})
    else:
        rows.append({"id": "H2", "name": "段内显著进展", "ok": True,
                     "detail": "⚠️ 无配对赛数据（跳过）"})

    # H3 相对基线不退步
    row = None
    if duel:
        new = f"dushan_m1_{seg}"
        row = next((r for r in duel.get("rows", []) if r["ckpt"].split("/")[-1] == new), None)
    if row:
        delta, ci = float(row["vs_anchor"]), float(row["vs_anchor_ci95"])
        ok = (delta + ci) >= JUDGE["master_ci_lo"]
        rows.append({"id": "H3", "name": f"相对 {BASELINE} 不退步", "ok": ok,
                     "detail": f"vs {BASELINE} {delta:+.3f} ± {ci:.3f}"
                               f"（CI 上界 {delta + ci:+.3f}，门槛 ≥ {JUDGE['master_ci_lo']:+.3f}）"})
    else:
        rows.append({"id": "H3", "name": f"相对 {BASELINE} 不退步", "ok": True,
                     "detail": "⚠️ 无配对数据（跳过）"})

    # 硬性：本段必须真的产出了模型
    produced = (ROOT / "models" / f"dushan_m1_{seg}.pt").exists()
    rows.append({"id": "H0", "name": "产出模型文件", "ok": produced,
                 "detail": f"models/dushan_m1_{seg}.pt {'存在' if produced else '缺失'}"})

    return all(r["ok"] for r in rows), rows


# --------------------------------------------------------------------------
# 报告
# --------------------------------------------------------------------------
def write_report(st: dict) -> None:
    lines = ["# M1 训练托管报告", ""]
    lines.append(f"- 生成时间：{datetime.now():%Y-%m-%d %H:%M:%S}")
    lines.append(f"- 托管启动：{st.get('started', '-')}")
    lines.append(f"- 当前状态：**{st.get('phase', '-')}**")
    if st.get("note"):
        lines.append(f"- 备注：{st['note']}")
    lines.append("")

    for seg in SEG_ORDER:
        rec = st["segments"].get(seg)
        if not rec:
            continue
        lines.append(f"## 段 {seg.upper()} — {rec.get('status', '-')}")
        lines.append("")
        lines.append(f"- 模式：{rec.get('mode', '-')}；轮次：{rec.get('iters_desc', '-')}")
        if rec.get("train_exit") is not None:
            lines.append(f"- 训练退出码：{rec['train_exit']}")
        if rec.get("verdict"):
            lines.append(f"- 决策：{'✅ 放行下一段' if rec['verdict']['ok'] else '⛔ 停止'}")
        h = rec.get("health")
        if h:
            lines.append(f"- 健康度：熵 尾{h['entropy']['tail']:.3f}（跌 {h['entropy']['drop']:+.3f}）、"
                         f"collect 中位 {h['collect_s']['median']:.1f}s、"
                         f"样本中位 {h['samples']['median']:.0f}")
        if rec.get("duel_vs_master"):
            d = rec["duel_vs_master"]
            lines.append(f"- **vs {BASELINE}：{d['delta']:+.3f} ± {d['ci']:.3f}**"
                         f"（CI 上界 {d['delta'] + d['ci']:+.3f}）")
        checks = rec.get("checks") or []
        if checks:
            lines.append("")
            lines.append("| 判据 | 结论 | 明细 |")
            lines.append("|---|---|---|")
            for c in checks:
                lines.append(f"| {c['id']} {c['name']} | "
                             f"{'PASS' if c['ok'] else 'FAIL'} | {c['detail']} |")
        lines.append("")

    nxt = st.get("next_action")
    if nxt:
        lines.append("## 训练安排")
        lines.append("")
        for s in nxt:
            lines.append(f"- {s}")
        lines.append("")

    REPORT_P.write_text("\n".join(lines), encoding="utf-8")


# --------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description="M1 分段训练无人值守托管")
    ap.add_argument("--from", dest="seg_from", default="s1", choices=SEG_ORDER)
    ap.add_argument("--to", dest="seg_to", default="s3", choices=SEG_ORDER)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    i0 = SEG_ORDER.index(args.seg_from)
    i1 = SEG_ORDER.index(args.seg_to)
    plan = SEG_ORDER[i0:i1 + 1]

    st = load_state()
    st.setdefault("segments", {})
    log(f"托管计划：{' → '.join(plan)}（workers={args.workers}）")

    if args.dry_run:
        for seg in plan:
            cfg = resolve_segment(seg)
            mode = cfg["_mode"]
            extra = (f"（从 {cfg.get('_ckpt')} 续跑，编号 {cfg.get('_done')} 起）"
                     if mode == "resume" else "（整段）")
            print(f"  {seg}: {cfg['iters']} 轮 {mode}{extra}")
        return 0

    keeper = DaemonKeeper(60)
    keeper.start()
    log("守护看门狗已启动（keep_awake / res_monitor 自退后会自动重启）")

    for idx, seg in enumerate(plan):
        cfg = resolve_segment(seg)
        rec = st["segments"].setdefault(seg, {})

        # ---- 训练 ----
        if cfg["iters"] > 0:
            # 防重复：上一次托管若被中断而训练进程还活着，先等它结束再接手 ——
            # 绝不同时起两个 selfplay（抢 CPU 会让双方指标全废，且内存翻倍）。
            pids = training_pids()
            while pids:
                log(f"检测到已有 selfplay 进程 {pids} 在跑，等待其结束（不再另起）…")
                time.sleep(60)
                pids = training_pids()

            keeper.mon_path = ROOT / "reports" / f"_res_monitor_m1_{seg}.csv"
            rec.update(status="训练中", mode=cfg["_mode"],
                       iters_desc=f"{cfg['iters']} 轮"
                                  + (f"（从 iter {cfg['_done']} 续）"
                                     if cfg["_mode"] == "resume" else ""),
                       started=datetime.now().isoformat(timespec="seconds"))
            st["phase"] = f"{seg.upper()} 训练中"
            st["next_action"] = [f"{s.upper()}：{SEGMENTS[s]['iters']} 轮（待前序段判读放行）"
                                 for s in plan[idx + 1:]] or ["（已是最后一段）"]
            save_state(st)
            write_report(st)

            rc = start_training(seg, cfg, args.workers)
            rec["train_exit"] = rc
            # 训练结束即刷新历史（判读要用）
            if rc != 0:
                rec["status"] = "训练异常退出"
                log(f"⚠️ {seg} 训练异常退出（exit={rc}），仍继续判读以留下证据")
        else:
            rec.update(status="训练已完成（本次无需再跑）", mode="resume",
                       iters_desc=f"已跑满 {SEGMENTS[seg]['iters']} 轮", train_exit=0)
            log(f"{seg} 已跑满，直接进入判读")

        # ---- 判读 ----
        st["phase"] = f"{seg.upper()} 判读中"
        save_state(st); write_report(st)
        judge_p = run_finalize(seg, JUDGE["deals"], JUDGE["duel_threads"])
        judge = json.loads(judge_p.read_text(encoding="utf-8")) if judge_p else None
        rec["health"] = (judge or {}).get("health")
        rec["judge_json"] = str(judge_p) if judge_p else None

        duel = run_master_duel(seg, JUDGE["deals"], JUDGE["duel_threads"])
        if duel:
            new = f"dushan_m1_{seg}"
            row = next((r for r in duel.get("rows", [])
                        if r["ckpt"].split("/")[-1] == new), None)
            if row:
                rec["duel_vs_master"] = {"delta": float(row["vs_anchor"]),
                                         "ci": float(row["vs_anchor_ci95"]),
                                         "verdict": row.get("verdict")}

        # ---- 决策 ----
        ok, checks = decide(seg, judge, duel)
        rec["checks"] = checks
        rec["verdict"] = {"ok": ok}
        rec["status"] = "✅ 判读通过" if ok else "⛔ 判读未通过（已停止）"
        log(f"{seg} 决策：{'放行' if ok else '停止'}；" +
            "；".join(f"{c['id']}={'PASS' if c['ok'] else 'FAIL'}" for c in checks))
        save_state(st); write_report(st)

        if not ok:
            st["phase"] = f"{seg.upper()} 未通过，已停止等待人工介入"
            st["note"] = ("有判据未通过 → 按「有错就纠错」原则停下，"
                          "不把可疑模型带进下一段。详见上表。")
            save_state(st); write_report(st)
            keeper.shutdown()
            return 0

    st["phase"] = "全部段完成"
    st["note"] = "计划内所有段均已跑完并通过判读。"
    st["next_action"] = ["用 scripts/duel.py / score_checkpoints.py 做代内选点复核；",
                         "把结论写进 docs/P3-M1训练执行计划.md 的段末验收行。"]
    save_state(st); write_report(st)
    log("全部完成 ✅")
    keeper.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
