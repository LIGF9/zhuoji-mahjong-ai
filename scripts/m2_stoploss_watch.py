"""G2@300 自动止损守望。

等 <verdict>.verdict.json 落盘 → 读 vs_anchor：
  - vs_anchor <= --threshold （默认 -1.05）→ 立即终止训练进程树（保留全部存档）
  - 否则记录「继续」，自身退出

判据依据：B2 段是从 A 段末（models/dushan_m2_a.pt）续训的，
连续 300 轮训练后理应不劣于起点（A 段末 it0600 判读 = -1.046 ± 0.448）。
若反而跌破 -1.05，说明配方仍不成立 → 止损重查，不烧后 300 轮。

只用 psutil 终止进程树，不用 shell。
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

try:
    import psutil
except ImportError:  # pragma: no cover
    sys.exit("需要 psutil：用 zhuoji venv 的 python 运行本脚本")

ROOT = Path(__file__).resolve().parent.parent


def ts() -> str:
    return datetime.now().strftime("%H:%M:%S")


class Log:
    def __init__(self, path: Path):
        self.path = path

    def __call__(self, msg: str) -> None:
        line = f"[{ts()}] {msg}"
        print(line, flush=True)
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(line + "\n")


def kill_tree(pid: int, log: Log) -> dict:
    """终止 pid 及其所有子孙；先 terminate 后 kill。"""
    try:
        root = psutil.Process(pid)
    except psutil.NoSuchProcess:
        log(f"训练进程 {pid} 已不存在，无需终止")
        return {"killed": False, "reason": "already_gone", "n": 0}
    try:
        procs = root.children(recursive=True)
    except psutil.Error:
        procs = []
    procs.append(root)
    names = [f"{p.pid}:{p.name()}" for p in procs]
    log(f"准备终止 {len(procs)} 个进程：{names}")
    for p in procs:
        try:
            p.terminate()
        except psutil.Error:
            pass
    gone, alive = psutil.wait_procs(procs, timeout=20)
    for p in alive:
        try:
            p.kill()
        except psutil.Error:
            pass
    gone2, alive2 = psutil.wait_procs(alive, timeout=10)
    survivors = [f"{p.pid}:{p.name()}" for p in alive2]
    if survivors:
        log(f"⚠️ 仍有存活：{survivors}")
    log(f"✅ 终止完成：{len(gone) + len(gone2)} 个已退出")
    return {"killed": True, "reason": "stoploss",
            "n": len(gone) + len(gone2), "survivors": survivors}


def main() -> int:
    ap = argparse.ArgumentParser(description="G2 判负则自动停训练")
    ap.add_argument("--verdict", required=True, help="gate_watch 的 *verdict.json 路径")
    ap.add_argument("--pid-file", required=True, help="训练 launcher 的 pid 文件")
    ap.add_argument("--threshold", type=float, default=-1.05,
                    help="vs_anchor 低于（含）此值即止损")
    ap.add_argument("--gate", default="G2@300")
    ap.add_argument("--stem", default="dushan_m2_b2")
    ap.add_argument("--reports", default="reports",
                    help="报告目录（写诊断 md 与决策 json）")
    ap.add_argument("--log", default="reports/_stoploss_b2.log")
    ap.add_argument("--poll-s", type=int, default=60)
    ap.add_argument("--max-wait-h", type=float, default=8.0)
    ap.add_argument("--dry-run", action="store_true", help="只判决不杀进程")
    args = ap.parse_args()

    verdict_path = (ROOT / args.verdict) if not Path(args.verdict).is_absolute() else Path(args.verdict)
    pid_path = (ROOT / args.pid_file) if not Path(args.pid_file).is_absolute() else Path(args.pid_file)
    reports = (ROOT / args.reports) if not Path(args.reports).is_absolute() else Path(args.reports)
    log = Log((ROOT / args.log) if not Path(args.log).is_absolute() else Path(args.log))

    started = time.time()
    started_wall = datetime.now()
    log("=" * 66)
    log(f"止损守望启动 —— 门禁 {args.gate}，判据 vs_anchor <= {args.threshold:+.3f}")
    log(f"观望文件：{verdict_path}")
    log(f"训练 pid 文件：{pid_path}")
    log("=" * 66)

    # 目标 pid
    pid: int | None = None
    if pid_path.exists():
        try:
            pid = int(pid_path.read_text(encoding="utf-8").strip())
        except ValueError:
            log(f"⚠️ pid 文件内容无法解析：{pid_path.read_text(encoding='utf-8')!r}")

    last_note = 0.0
    while True:
        elapsed_h = (time.time() - started) / 3600.0
        if elapsed_h > args.max_wait_h:
            log(f"⏱ 超过 {args.max_wait_h} h 未见到判决文件，退出（不动作）")
            return 3

        # 判决文件必须比本脚本新，避免读到陈旧结果
        if verdict_path.exists() and verdict_path.stat().st_mtime >= started:
            try:
                v = json.loads(verdict_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                time.sleep(args.poll_s)
                continue
            va = v.get("vs_anchor")
            ci = v.get("vs_anchor_ci95")
            lo = v.get("lower")
            gate_passed = v.get("passed")
            mono_ok = v.get("monotonic_ok")
            base = v.get("baseline")
            log("-" * 66)
            log(f"收到判决：{verdict_path.name}")
            log(f"  vs_anchor = {va:+.3f}  95%CI ± {ci:.3f}  下界 {lo:+.3f}")
            log(f"  基线（{v.get('gate')} 的 baseline）= {base}")
            log(f"  门禁自身判据 passed={gate_passed}（monotonic_ok={mono_ok}）")
            triggered = va is not None and va <= args.threshold
            log(f"  止损判据 va <= {args.threshold:+.3f} → {'触发 ❌' if triggered else '不触发 ✅'}")

            decision = {
                "gate": args.gate, "stem": args.stem,
                "decided_at": datetime.now().isoformat(timespec="seconds"),
                "vs_anchor": va, "vs_anchor_ci95": ci, "lower": lo,
                "threshold": args.threshold, "triggered": bool(triggered),
                "gate_passed": gate_passed, "monotonic_ok": mono_ok,
                "baseline": base, "dry_run": bool(args.dry_run),
                "action": None,
            }

            if triggered and not args.dry_run:
                if pid is None:
                    log("⚠️ 触发止损但拿不到 pid，需人工处理")
                    decision["action"] = "triggered_no_pid"
                else:
                    res = kill_tree(pid, log)
                    decision["action"] = res
            elif triggered:
                log("（dry-run：不终止进程）")
                decision["action"] = "dry_run"
            else:
                log("训练继续，无需干预")
                decision["action"] = "continue"

            # 决策留档（文件名按 stem+gate 分档 —— 原来写死 "G2_stoploss_decision.json"，
            # 多道守望会互相覆盖：B2 的触发记录就是这样被 B3 的台账盖掉的）
            gate_slug = "".join(ch if ch.isalnum() else "_" for ch in str(args.gate)).strip("_")
            out = reports / f"{args.stem}_{gate_slug}_stoploss_decision.json"
            out.write_text(json.dumps(decision, ensure_ascii=False, indent=2), encoding="utf-8")

            # 触发时写一份人读的诊断 md
            if triggered:
                md = reports / f"{args.stem}_止损诊断.md"
                _drop = (va - base) if (va is not None and base is not None) else None
                _drop_txt = f"{_drop:+.3f}" if _drop is not None else "n/a"
                _rel = "低于" if (_drop is not None and _drop < 0) else "高于"
                _both = ("门禁自身判据（passed=False）也未通过 —— 两条独立判据一致判负。"
                         if gate_passed is False else
                         "门禁自身判据通过，仅更严的止损线被击穿。")
                md.write_text(
                    "\n".join([
                        f"# {args.stem} 止损诊断（{args.gate}）",
                        "",
                        f"- 判决时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                        f"- 判决文件：`{verdict_path.relative_to(ROOT)}`",
                        f"- **vs_anchor = {va:+.3f} ± {ci:.3f}**（下界 {lo:+.3f}）",
                        f"- 止损线：{args.threshold:+.3f}（基线 {base}）",
                        f"- 门禁自身判据：passed={gate_passed}，monotonic_ok={mono_ok}",
                        f"- 动作：{'已终止训练进程树' if not args.dry_run else 'dry-run，未终止'}",
                        "",
                        "## 解读",
                        "",
                        f"{args.stem} 从基线（{base}）续训到 {args.gate}，实测 vs_anchor = {va:+.3f}，"
                        f"{_rel}基线 {_drop_txt}。",
                        _both,
                        "按预注册方案：停跑重查，不烧后续轮次。",
                        "",
                        "## 建议的复查顺序",
                        "",
                        "1. 用 1200 副共用牌池核对本段多个检查点（如 it0100 / it0200 / it0300），确认是单调下滑还是单点噪声。",
                        "2. 若多点齐跌 → 优先怀疑**断点续训机制本身**（优化器动量未随检查点保存 = 冷 Adam 起步），而非单纯 lr 数值。",
                        "3. 若仅末点塌 → 检查该点附近日志的 `collect` 时长与 RSS 尖峰，排除资源抖动导致的坏批次。",
                        "",
                    ]),
                    encoding="utf-8",
                )
                log(f"诊断报告 → {md.relative_to(ROOT)}")
            log(f"决策留档 → {out.relative_to(ROOT)}")
            return 0 if not triggered else 10

        # 训练是否还活着
        if pid is not None and not psutil.pid_exists(pid):
            log(f"⚠️ 训练进程 {pid} 已不在，且未见到判决文件 → 守望退出")
            _gs = "".join(ch if ch.isalnum() else "_" for ch in str(args.gate)).strip("_")
            (reports / f"{args.stem}_{_gs}_stoploss_decision.json").write_text(
                json.dumps({"triggered": False, "action": "training_gone",
                            "note": f"pid {pid} 消失，未收到判决",
                            "decided_at": datetime.now().isoformat(timespec="seconds")},
                           ensure_ascii=False, indent=2), encoding="utf-8")
            return 4

        now = time.time()
        if now - last_note > 900:
            last_note = now
            eta = started_wall + timedelta(hours=args.max_wait_h)
            log(f"等待中…（已等 {(now - started) / 60:.0f} 分；上限 {eta:%H:%M}）")
        time.sleep(args.poll_s)


if __name__ == "__main__":
    raise SystemExit(main())
