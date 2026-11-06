#!/usr/bin/env python3
"""PPO 轮次观察器（只读 / 零 CPU / 不启对局）。

为什么需要它：PPO 从 R10 起每轮产出 exit_pNN.pt，但「是否在学」不能看 val loss、
更不能看它有没有晋升（300 seed 的池分辨率 SE≈0.5 分，单轮真实增量 0.2~0.5 分，
多半够不到显著）。唯一可靠的看法是三条同池口径：

  1. P{r} 闸门 A（vs master 贪心，300 seed seed0=710000）
     —— 与 rNN 闸门 A **同池同口径**，可与同轮 DAgger 直接逐 seed 配对；
  2. P{r} 闸门 D（vs 现最佳，300 seed）—— 是否真的超过 DAgger 主线；
  3. P{r} − r{r} 的配对差 —— 「PPO 相对同期 DAgger」的净增量，这才是 PPO 是否
     值得继续占预算的判据（而不是它有没有晋升）。

用法：
  .venv/bin/python tools/ppo_report.py            # 人读
  .venv/bin/python tools/ppo_report.py --md       # 追加到报告的 markdown
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))

from final_eval_report import fmt, per_seed, stats   # noqa: E402  唯一实现，禁止另写口径

GATES = ROOT / "runs/gates"
LOGS = ROOT / "runs/night_logs"
METRIC_RE = re.compile(r"^\[metrics\]\s*(.*)$")


def ppo_rounds() -> list[int]:
    r = set()
    for p in (ROOT / "models").glob("exit_p*.pt"):
        m = re.fullmatch(r"exit_p(\d+)\.pt", p.name)
        if m:
            r.add(int(m.group(1)))
    for p in LOGS.glob("ppo_p*.log"):
        m = re.fullmatch(r"ppo_p(\d+)\.log", p.name)
        if m:
            r.add(int(m.group(1)))
    return sorted(r)


def last_metrics(r: int) -> str:
    p = LOGS / f"ppo_p{r:02d}.log"
    if not p.exists():
        return "(无日志)"
    last = ""
    try:
        for line in p.read_text(encoding="utf-8", errors="replace").splitlines():
            m = METRIC_RE.match(line.strip())
            if m:
                last = m.group(1)
    except OSError:
        return "(日志不可读)"
    return last or "(本次运行尚未写出 [metrics])"


def safe_stats(path: Path, min_seeds: int) -> dict | None:
    """jsonl 缺失 / 未跑完 / 记录不全一律返回 None（绝不用半截数据下结论）。"""
    if not path.exists():
        return None
    try:
        d = per_seed(path)
        if len(d) < min_seeds:
            return None
        return stats(list(d.values()))
    except (ValueError, OSError):
        return None


def paired(a: Path, b: Path) -> dict | None:
    """同池逐 seed 配对（a − b）；两文件 seed 取交集，要求 ≥100 个。"""
    try:
        da, db = per_seed(a), per_seed(b)
    except (ValueError, OSError):
        return None
    common = sorted(set(da) & set(db))
    if len(common) < 100:
        return None
    return stats([da[s] - db[s] for s in common])


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--md", action="store_true", help="输出 markdown 表格")
    a = ap.parse_args(argv)

    rounds = ppo_rounds()
    disabled = (ROOT / "runs/ppo_disabled").exists()
    print("=" * 92)
    print(f"PPO 分支状态：{'已停用（runs/ppo_disabled 存在）' if disabled else '启用中'}"
          f"   PPO 轮次={len(rounds)}"
          + (f"  {rounds}" if rounds else "（R10 起才有，符合预期）"))
    if not rounds:
        print("尚无 PPO 轮次。R10 末起每轮跑 ppo_round()，产出 models/exit_pNN.pt + "
              "runs/gates/pNN_gate{D,A}.jsonl")
        return 0

    rows: list[tuple[str, dict]] = []
    print("-" * 92)
    print(f"{'轮次':<6}{'A vs master（同池 seed0=710000）':<44}{'晋升':<6}")
    for r in rounds:
        sa = safe_stats(GATES / f"p{r:02d}_gateA.jsonl", 300)
        promoted = (GATES / f"p{r:02d}.promoted").exists()
        name = f"P{r:02d} A"
        if sa:
            print(fmt(name, sa))
            rows.append((name, sa))
        else:
            print(f"{name:<24} (闸门 A 未跑完或缺席)")
        sd = safe_stats(GATES / f"p{r:02d}_gateD.jsonl", 300)
        if sd:
            print(fmt(f"P{r:02d} D(vs 现最佳)", sd))
            rows.append((f"P{r:02d} D vs best", sd))
        else:
            print(f"{'P' + f'{r:02d}' + ' D(vs 现最佳)':<24} (闸门 D 未跑完或缺席)")
        # 相对同期 DAgger 的净增量 —— PPO 是否值得继续占预算的真实判据
        sp = paired(GATES / f"p{r:02d}_gateA.jsonl", GATES / f"r{r:02d}_gateA.jsonl")
        if sp:
            print(fmt(f"P{r:02d} − R{r:02d}(同期)", sp))
            rows.append((f"P{r:02d} − R{r:02d}", sp))
        print(f"      晋升={'是' if promoted else '否（正常，池分辨率不足≠失败）'}"
              f"   metrics: {last_metrics(r)}")
        print()

    print("-" * 92)
    print("判读纪律：")
    print("  ① 单轮不晋升是**分辨率限制**（SE≈0.5 分），不是失败；绝不因此加 lr 或放宽闸门。")
    print("  ② 是否在学只看：跨轮 P 闸门 A 绝对值趋势 + metrics 的 kl/ratio/clipfrac 是否健康")
    print("     （kl 应 ∈(0, 0.03]，ratio ≈1.00±0.05，clipfrac < 0.3）。")
    print("  ③ 决策看 `P{r} − R{r}`（同池配对）：连续 ≥3 轮 ≤0 ⇒ PPO 无净收益，"
          "降为 PPO_EVERY=2 或停用；连续 ≥2 轮 >0 且累积 >2 分 ⇒ 加预算（hands 1.5~2×）。")

    if a.md:
        print()
        print("```markdown")
        print("| 口径 | n | diff ± SE | z | p | 符号 |")
        print("|---|---|---|---|---|---|")
        for name, s in rows:
            print(f"| {name} | {s['n']} | {s['mean']:+.3f}±{s['se']:.3f} | "
                  f"{s['z']:+.2f} | {s['p']:.4f} | {s['pos']}:{s['neg']} |")
        print("```")
    return 0


if __name__ == "__main__":
    sys.exit(main())
