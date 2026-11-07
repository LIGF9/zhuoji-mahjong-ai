#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""加速模式提前回退护栏（只读检查 + 写标志；不改变任何晋升/决策判据）。

预登记时间：2026-11-07 16:28（P12 判读之后、**任何加速轮数据产生之前**）。
出处：外部专家评审「加速模式前 3 轮若 PPO 候选在闸门 D 上持续为负且无收敛
迹象，提前回退，不要等到 12 轮重锚点」。本工具把这条建议变成**可判定、
在看到数据前锁死阈值**的规则，而不是事后看图说话。

判据（三条全满足才回退，缺一不判）：
  ① 完整性：FROM, FROM+1, FROM+2 三轮的 pNN_gateD.jsonl 全部存在且完整
     （每个 ≥100 个共同 seed；半截 jsonl 一律不参与，见 load()）。
  ② 持续为负：三轮 D 的 mean 全部 < 0，且 Σmean ≤ SUM_ABORT(−0.60)。
  ③ 无收敛迹象：|D_3| ≥ CONVERGE_RATIO(0.8) × |D_1|
     （即最后一轮偏离 0 的程度没有明显小于第一轮）。

满足 ⇒ 写 runs/rapid_abort（内容含依据），night_run.is_rapid() 下一轮起
恒 False，自动回落 DAgger。不满足 ⇒ 只打印趋势，不动任何文件。

口径唯一实现：tools/final_eval_report.py 的 per_seed() / stats()，
与 tools/ppo_decide.py、tools/plateau_test.py 完全一致；禁止另写统计。
闸门 D 定义不变（PPO 候选 vs 本轮现最佳 exit_best，300 seed 四座、
deltas[cand_seat] − mean(其余三家)）。

用法::
    .venv/bin/python tools/rapid_guard.py            # 判读 + 必要时写标志
    .venv/bin/python tools/rapid_guard.py --dry      # 只打印，绝不写
    .venv/bin/python tools/rapid_guard.py --md       # 附 markdown 表
"""
from __future__ import annotations

import argparse
import datetime as dt
import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))

from final_eval_report import fmt, per_seed, stats  # noqa: E402

GATES = ROOT / "runs" / "gates"
ABORT = ROOT / "runs" / "rapid_abort"

# ---- 预登记阈值（改这里 = 改预登记，必须同步写进 docs/14天训练计划.md）----
WINDOW = 3
SUM_ABORT = -0.60
CONVERGE_RATIO = 0.8
MIN_SEEDS = 100


def rapid_from() -> int:
    """从驱动读 RAPID_FROM（import 无副作用：main() 在 __main__ 保护下）。"""
    spec = importlib.util.spec_from_file_location("_nr_guard", ROOT / "night_run.py")
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    v = getattr(mod, "RAPID_FROM", None)
    if not isinstance(v, int):
        raise SystemExit("RAPID_FROM 未武装（None）⇒ 无需护栏")
    return v


def load(path: Path) -> dict[int, float] | None:
    """完整才算数：缺失 / 半截 / 解析失败 ⇒ None。"""
    if not path.exists():
        return None
    try:
        d = per_seed(path)
    except (ValueError, OSError, KeyError):
        return None
    return d if len(d) >= MIN_SEEDS else None


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="加速模式前 3 轮回退护栏（只读）")
    ap.add_argument("--from", dest="frm", type=int, default=None,
                    help="覆盖 RAPID_FROM（默认从 night_run.py 读）")
    ap.add_argument("--dry", action="store_true", help="只打印，不写 rapid_abort")
    ap.add_argument("--md", action="store_true")
    a = ap.parse_args(argv)

    frm = a.frm if a.frm is not None else rapid_from()
    rounds = list(range(frm, frm + WINDOW))
    print("=" * 92)
    print(f"加速模式回退护栏 ｜ 判读窗口 R{frm}~R{frm + WINDOW - 1} ｜ "
          f"预登记 2026-11-07 16:28（阈值在加速轮数据产生前锁定）")
    print(f"阈值：三轮 D 全负 且 Σ ≤ {SUM_ABORT:+.2f} 且 "
          f"|D3| ≥ {CONVERGE_RATIO}×|D1|（无收敛迹象）")
    print("=" * 92)

    if ABORT.exists():
        print(f"[已回退] {ABORT} 存在 ⇒ is_rapid 恒 False，本次不再判读。")
        return 0

    vals: list[float] = []
    rows: list[tuple[int, dict]] = []
    for r in rounds:
        path = GATES / f"p{r:02d}_gateD.jsonl"
        d = load(path)
        if d is None:
            print(f"  P{r:02d} 闸门 D: [未就绪] {path.name} 缺失或不完整")
        else:
            s = stats(list(d.values()))
            vals.append(s["mean"])
            rows.append((r, s))
            print("  " + fmt(f"P{r:02d} 闸门 D", s))

    if len(vals) < WINDOW:
        print(f"\n[结论] 数据未齐（{len(vals)}/{WINDOW}）⇒ 不判读、不写标志。"
              " 半截数据永不用于回退决策。")
        return 0

    all_neg = all(v < 0 for v in vals)
    total = sum(vals)
    no_converge = abs(vals[-1]) >= CONVERGE_RATIO * abs(vals[0])
    abort = all_neg and total <= SUM_ABORT and no_converge

    print("-" * 92)
    print(f"  ① 三轮全负: {'是' if all_neg else '否'}  ({', '.join(f'{v:+.3f}' for v in vals)})")
    print(f"  ② Σ = {total:+.3f}  (阈值 ≤ {SUM_ABORT:+.2f})：{'满足' if total <= SUM_ABORT else '不满足'}")
    print(f"  ③ |D3|={abs(vals[-1]):.3f} vs {CONVERGE_RATIO}×|D1|="
          f"{CONVERGE_RATIO * abs(vals[0]):.3f}：{'无收敛迹象' if no_converge else '在收敛'}")
    print(f"  [结论] {'触发提前回退 ⇒ 回落 DAgger' if abort else '保持加速模式'}")

    if abort and not a.dry:
        ABORT.write_text(
            f"[{dt.datetime.now():%F %H:%M:%S}] 加速模式提前回退（预登记 §6.34）\n"
            f"窗口 R{rounds[0]}~R{rounds[-1]} 闸门 D: "
            + ", ".join(f"P{r:02d}={v:+.3f}" for r, v in zip(rounds, vals))
            + f"\nΣ={total:+.3f} ≤ {SUM_ABORT:+.2f}；|D3|={abs(vals[-1]):.3f} ≥ "
              f"{CONVERGE_RATIO}×|D1|={CONVERGE_RATIO * abs(vals[0]):.3f}\n"
            f"阈值锁定于加速轮数据产生前（2026-11-07 16:28），非事后调参。\n",
            encoding="utf-8")
        print(f"  已写 {ABORT} ⇒ 驱动下一轮起 is_rapid() 恒 False。")
    elif abort:
        print("  [--dry] 未写标志。")

    if a.md:
        print("\n```markdown")
        print("| 轮 | n | D ± SE | p |")
        print("|---|---|---|---|")
        for r, s in rows:
            print(f"| P{r:02d} | {s['n']} | {s['mean']:+.3f}±{s['se']:.3f} | {s['p']:.4f} |")
        print("```")
    print("\n[护栏] 不改 P12 预登记三分支、不改闸门 D/A 晋升判据、不改 web_best 换代策略。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
