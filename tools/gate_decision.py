#!/usr/bin/env python3
"""ExIt 逐轮闸门决策辅助（只读 · 零 CPU · 不启动任何对局/训练/评估）。

把 docs/14天训练计划.md §6.4 / §6.7.5 / §6.7.6 / §6.10.6 / §6.15.4 的硬决策判据
代码化，供每次自检（尤其 R6 末硬决策点 ≈19:1x~19:3x）一键得到一致结论，
避免每次手写脚本导致口径漂移（自检 #7 曾因取 deltas[0] 全错）。

冻结口径唯一实现复用 tools/final_eval_report.py：
    d = deltas[cand_seat] − mean(其余三家)，四座按 seed 平均，seed 维度算 SE/z/p

只读：runs/gates/*.jsonl、r*.promoted、runs/night_run.lock、
      runs/night_logs/annotate_r*.log、runs/night_run.log、train_exit.py(TAU 常量)。
不加载 torch/onnx、不启评估、不写任何文件、不与标注并发；可在 7×24 标注期安全运行。

用法::

    ./.venv/bin/python tools/gate_decision.py             # 自动发现最新完整轮
    ./.venv/bin/python tools/gate_decision.py --md        # 附 Markdown 决策表
    ./.venv/bin/python tools/gate_decision.py --round 6   # 以 R6 视角（未落盘则回退）

注意：脚本只给「建议」，最终动作仍由自检按计划文档判定；
唯一允许的超参改动是 §6.4 触发时编辑 train_exit.py:38 的 TAU，且禁止同时改两项。

闸门 C 尚未落盘时（每轮 A 通过后约 22min 的窗口，§6.17.4）脚本自动进入
「巡检」模式：正常输出 A 的冻结口径统计，标注 C 待落盘并暂缓依赖 C 的
② 天花板 / ③ P2a 判据，不崩溃、不误报、不提前触发不可逆的 TAU 改动
（自检 #14 修复：旧版仅检查 exists()，0 字节 jsonl 会让 per_seed 抛异常）。
"""
from __future__ import annotations

import argparse
import glob
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))
from final_eval_report import fmt, per_seed, stats  # noqa: E402

HEALTH_MIN_THR = 8000          # <8k 标注/h·worker 判异常（健康判据）
WEAK_P = 0.2                   # §6.4「p>0.2 或均值<=0」
PLATEAU_ABS = 1.0              # §6.10.6「|Δ|<1.0 且 p>0.2」


GATE_A_SEEDS = 300          # 闸门 A 冻结池 seed 数
GATE_C_SEEDS = 50           # 闸门 C 冻结池 seed 数


def _complete(path: Path, n_min: int) -> bool:
    """闸门 jsonl 是否已完整落盘（按唯一 seed 数判定，而非 exists()）。

    评估进程边跑边追加写 jsonl：闸门 A 通过后立即可用，闸门 C 还要再跑 ≈22min。
    自检 #14 时 r06_gateC.jsonl 已存在但为 0 字节，旧版 discover() 仅检查
    exists() 就纳入配对，per_seed() 抛 ValueError 导致整个决策脚本崩溃。
    未完成的闸门 C 应视为「该轮 C 待落盘」，只展示 A 并暂缓依赖 C 的判据。
    """
    try:
        return path.stat().st_size > 0 and len(per_seed(path)) >= n_min
    except (OSError, ValueError):
        return False


def discover() -> list[tuple[int, Path, Path | None]]:
    """返回 (轮次, gateA 路径, gateC 路径或 None)。

    gateC 允许为 None：闸门 A 先落盘并晋升、闸门 C 随后才跑，此窗口内脚本须
    正常输出 A 的冻结口径统计并暂缓 C 相关判据，而不是崩溃或丢弃整轮。
    """
    out = []
    for p in sorted(glob.glob(str(ROOT / "runs/gates/r*_gateA.jsonl"))):
        m = re.search(r"r(\d+)_gateA\.jsonl$", p)
        if not m:
            continue
        r = int(m.group(1))
        fa = Path(p)
        if not _complete(fa, GATE_A_SEEDS):
            continue
        fc = ROOT / f"runs/gates/r{r:02d}_gateC.jsonl"
        out.append((r, fa, fc if _complete(fc, GATE_C_SEEDS) else None))
    return out


def health() -> dict:
    h: dict = {}
    lock = ROOT / "runs/night_run.lock"
    h["lock_age_s"] = round(time.time() - lock.stat().st_mtime, 1) if lock.exists() else None
    logs = sorted(glob.glob(str(ROOT / "runs/night_logs/annotate_r*.log")),
                  key=lambda p: Path(p).stat().st_mtime)
    thr = None
    if logs:
        tail = Path(logs[-1]).read_text(encoding="utf-8", errors="ignore").splitlines()[-16:]
        vals = [int(m.group(1)) for ln in tail
                if (m := re.search(r"\((\d+)\s*标注/h\)", ln))]
        if vals:
            thr = min(vals)
    h["thr_min"] = thr
    h["annotate_log"] = Path(logs[-1]).name if logs else None
    nlog = ROOT / "runs/night_run.log"
    if nlog.exists():
        lines = nlog.read_text(encoding="utf-8", errors="ignore").splitlines()
        start = next((i for i, l in enumerate(lines) if "02:33:57" in l), 0)
        tail = lines[start:]
        h["rc0"] = sum(1 for l in tail if "rc=0" in l)
        h["rc_bad"] = sum(1 for l in tail if re.search(r"rc=[1-9]", l))
        h["panic"] = sum(1 for l in tail
                         if "panic" in l.lower() or "traceback" in l.lower())
    t = (ROOT / "train_exit.py").read_text(encoding="utf-8")
    m = re.search(r"^TAU\s*=\s*([0-9.]+)", t, re.M)
    h["TAU"] = m.group(1) if m else "?"
    return h


def paired(d0: dict, d1: dict):
    ks = sorted(set(d0) & set(d1))
    return stats([d1[k] - d0[k] for k in ks]) if len(ks) >= 2 else None


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="ExIt 闸门决策辅助（只读）")
    ap.add_argument("--round", type=int, default=None, help="以指定轮为最新轮视角")
    ap.add_argument("--md", action="store_true", help="附 Markdown 决策表")
    args = ap.parse_args(argv)

    rows = discover()
    if not rows:
        print("[ERROR] 未发现任何已落盘的 rNN_gateA", file=sys.stderr)
        return 2
    latest = rows[-1][0]
    r = args.round or latest
    if r > latest:
        print(f"[提示] R{r} 闸门尚未落盘，回退到最新完整轮 R{latest}")
        r = latest
    rows = [x for x in rows if x[0] <= r]
    A, C = {}, {}
    for rr, fa, fc in rows:
        A[rr] = per_seed(fa)
        C[rr] = per_seed(fc) if fc is not None else None
    rounds = sorted(A)
    c_pending = C.get(r) is None

    h = health()
    print("=" * 92)
    print(f"ExIt 闸门决策复核 · 最新闸门 A 轮 R{r:02d} · DEADLINE 2026-11-13 02:30")
    print("=" * 92)
    if c_pending:
        print(f"  ⚠ R{r:02d} 闸门 C 尚未完整落盘（评估进行中 / jsonl 为空或不全）")
        print("    → 本轮属「巡检」：只记录闸门 A + 核落盘/训练；依赖 C 的判据"
              "（② 天花板 / ③ P2a）暂缓，按 §6.17.4 不执行任何硬决策。")
    print(f"健康：lock={h['lock_age_s']}s  标注吞吐(min)={h['thr_min']}  "
          f"rc=0×{h.get('rc0')}  rc!=0×{h.get('rc_bad')}  panic×{h.get('panic')}  "
          f"TAU={h['TAU']}  日志={h['annotate_log']}")
    if h["thr_min"] is not None and h["thr_min"] < HEALTH_MIN_THR:
        print(f"  ⚠ 吞吐 {h['thr_min']} < {HEALTH_MIN_THR} 健康线 → 按 §6.4 先定位修复")
    if h["lock_age_s"] is not None and h["lock_age_s"] > 120:
        print("  ⚠ lock >120s → 结合日志判驱动是否已停，仅确停才后台重启一次")
    print("\n-- 逐轮闸门（冻结口径，与 night_run.log [summary] 逐位一致）--")
    for rr in rounds:
        sa = stats(list(A[rr].values()))
        print(f"  R{rr:02d} A {fmt('', sa)}")
        if C[rr] is not None:
            print(f"  R{rr:02d} C {fmt('', stats(list(C[rr].values())))}")
        else:
            print(f"  R{rr:02d} C 未完整落盘（进行中，跳过）")

    # 环比增量（同冻结池逐 seed 配对）
    marg: dict[int, dict] = {}
    cmarg: dict[int, dict] = {}
    print("\n-- 相邻轮环比（新轮 − 旧轮，同 seed 配对）--")
    for rr in rounds:
        if rr - 1 in A:
            s = paired(A[rr - 1], A[rr])
            if s:
                marg[rr] = s
                print(f"  A R{rr}−R{rr-1}: {s['mean']:+.3f}±{s['se']:.3f} p≈{s['p']:.4f}")
            if C.get(rr) is not None and C.get(rr - 1) is not None:
                t = paired(C[rr - 1], C[rr])
                if t:
                    cmarg[rr] = t
                    print(f"  C R{rr}−R{rr-1}: {t['mean']:+.3f}±{t['se']:.3f} p≈{t['p']:.4f}")
            else:
                print(f"  C R{rr}−R{rr-1}: 跳过（闸门 C 未完整落盘）")

    # 判据 1：增速三选一（§6.4）—— A 本身连续两轮弱
    def weak(s):
        return s["p"] > WEAK_P or s["mean"] <= 0
    sa_last = stats(list(A[r].values()))
    sa_prev = stats(list(A[r - 1].values())) if r - 1 in A else None
    boost = bool(sa_prev and weak(sa_last) and weak(sa_prev))
    # 判据 2：平台期/天花板确认（§6.10.6，R4~R6 三轮）
    need = [rr for rr in (r - 2, r - 1, r) if rr in marg]
    ok_marg = all(abs(marg[rr]["mean"]) < PLATEAU_ABS and marg[rr]["p"] > 0.2 for rr in need)
    c_recent = [stats(list(C[rr].values())) for rr in (r - 2, r - 1, r)
                if C.get(rr) is not None]
    ok_c = len(c_recent) == 3 and all(abs(s["mean"]) < 1.0 and s["p"] > 0.05 for s in c_recent)
    gate_r6 = r >= 6
    plateau = (gate_r6 and ok_marg and ok_c and not c_pending)
    # 判据 3：P2a（§6.7.5）—— C 连续 >=2 段收敛 + A 显著为正
    conv = [rr for rr in (r - 1, r) if rr in cmarg and cmarg[rr]["mean"] > 0]
    p2a_ready = (not c_pending and len(conv) >= 2
                 and sa_last["p"] < 0.05 and sa_last["mean"] > 0)
    # 判据 4：PPO-lite —— 自 §6.24 起已从「是否启动」变成「已在跑，看净增量」
    ppo_live = False
    ppo_from = 10
    try:
        m = re.search(r"^PPO_FROM\s*=\s*(\d+)", (ROOT / "night_run.py").read_text(
            encoding="utf-8", errors="replace"), re.M)
        if m:
            ppo_from = int(m.group(1))
        ppo_live = r >= ppo_from and not (ROOT / "runs/ppo_disabled").exists()
    except OSError:
        pass
    ppo_rounds = sorted(int(p.name[6:8]) for p in (ROOT / "models").glob("exit_p[0-9][0-9].pt"))

    print("\n-- 决策建议（按 §6.4 / §6.7.6 / §6.10.6 / §6.15.4）--")
    d = []
    d.append(("① 增速三选一", "触发：仅改 train_exit.py:38 TAU=0.5（禁止同时改第二项）"
              if boost else "不触发：TAU / Q-MSE / n_det / lr 全维持",
              f"最近两轮 A 弱={bool(sa_prev and weak(sa_prev))}/{weak(sa_last)}；"
              f"当前 TAU={h['TAU']}"))
    if c_pending:
        plateau_act = "暂缓：闸门 C 待落盘，缺数据不判（§6.17.4 巡检窗口）"
    elif plateau:
        plateau_act = ("「确认」：DAgger 边际已连续为 0 ⇒ 不再指望主线增益，资源转 PPO 分支"
                       "（已在跑）+ 终评准备；仍不改任何超参"
                       if ppo_live else
                       "「确认」：DAgger 边际已连续为 0 ⇒ 不再指望主线增益，"
                       "资源转终评准备；仍不改任何超参")
    elif gate_r6:
        plateau_act = "不确认：条件未满足（C 平台 / 环比），顺延下一轮复核；仍不改超参"
    else:
        plateau_act = "不提前升级：按规则留 R6 末判定"
    d.append(("② 天花板确认", plateau_act,
              f"三轮环比满足={ok_marg}({len(need)}/3)，C 平台={ok_c}，"
              f"C 待落盘={c_pending}，R6 门槛={'已到' if gate_r6 else f'未到(r={r})'}"))
    d.append(("③ P2a Oracle", "双条件达成 → 规格冻结/离线准备态（不激活、不并发、不重启）"
              if p2a_ready else "维持观察态（暂停推进）：C 未连续 ≥2 段收敛 / C 待落盘",
              f"最近两段 C 正增量={len(conv)}/2，A 显著为正={sa_last['p'] < 0.05}，"
              f"C 待落盘={c_pending}"))
    if ppo_live:
        ppo_act = (f"已上线（PPO_FROM={ppo_from}，每轮 ≈8min）：R{r} 末自动产出 exit_p{r:02d}.pt，"
                   f"判读走 tools/ppo_report.py 的 P{r:02d}−R{r:02d} 同池配对；"
                   f"单轮不晋升=分辨率限制，绝不放宽闸门 D")
        ppo_why = f"r={r}≥{ppo_from}；已有 PPO ckpt={ppo_rounds or '无（本轮首次）'}"
    else:
        ppo_act = "未到首发轮次（PPO_FROM={}）".format(ppo_from)
        ppo_why = f"r={r}（门槛 R{ppo_from}+）"
    d.append(("④ PPO-lite", ppo_act, ppo_why))
    d.append(("⑤ 下一步训练", "驱动自动接续：标注 → 训练 → 闸门 A/C → 下一轮；零干预、"
              "不启第二驱动、不打断、不并发评估", "DEADLINE 为唯一硬约束"))
    for name, act, why in d:
        print(f"  {name}: {act}\n      依据：{why}")

    if args.md:
        print("\n```markdown\n| 判据 | 建议动作 | 依据 |\n|---|---|---|")
        for name, act, why in d:
            print(f"| {name} | {act} | {why} |")
        print("```")

    print("\n[禁止] 自对弈胜率作决策 / val loss 跨轮判强度 / 同时改两项超参 / "
          "与标注并发跑评估 / 用 deltas[0] 统计")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
