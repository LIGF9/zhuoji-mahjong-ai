#!/usr/bin/env python3
"""终评离线预检（只读 · 不执行评估 · 不与标注并发）。

在 DEADLINE(2026-11-13 02:30) 之后、正式跑终评前先执行本脚本，确认
「新模型 + N32 搜索」的编码兼容性，并打印冻结口径的命令清单。

背景（2026-11-06 自检 #7 发现，详见 docs/14天训练计划.md §6.11）：
  Rust 搜索内核 dushan_engine 的 take_batch() 固定输出 v1 编码
  (15 通道 / 70 全局维)；而 exit_best.pt 是 enc3 (63 通道 / 130 全局维)。
  eval_pairs.py 的 RustSearchAgent 用 zhuoji.encoder 的 v1 常量构造缓冲，
  因此 `--a <enc3模型> --a-mode search32` 必然形状 panic。
  历史闸门 A(双 greedy)、闸门 C(新 greedy vs master+N32) 均绕开了该组合，
  所以 7x24 主循环完全健康，缺陷只影响「闸门 B / 终评主目标」。

本脚本只做静态/构造级检查，不启动任何对局、不占 CPU、不写入模型或数据。
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

PY = "./.venv/bin/python"
POOL = "3000000"          # 终评独立 500 seed 池（与 710000/720000/标注池完全隔离）
SEEDS = "500"
A = "models/exit_best.pt"
B = "models/dushan_master.pt"


def ckpt_version(path: str) -> tuple[int, int | None]:
    import torch
    from zhuoji.net import build_from_ckpt

    ck = torch.load(ROOT / path, map_location="cpu", weights_only=False)
    net, _ = build_from_ckpt(ck, map_location="cpu")
    return int(getattr(net, "encoder_version", 1)), int(getattr(net, "in_channels", 15))


def main() -> int:
    import dushan_engine as de
    from zhuoji.encoder import GLOBAL_DIM, NUM_CHANNELS

    kern_ch = int(getattr(de, "NUM_CHANNELS", NUM_CHANNELS))
    a_ver, a_ch = ckpt_version(A)
    b_ver, b_ch = ckpt_version(B)

    print("=" * 72)
    print("终评编码兼容性预检（只读）")
    print("=" * 72)
    print(f"  Rust 搜索内核 take_batch : {kern_ch} 通道 / glob {GLOBAL_DIM}（v1 固定）")
    print(f"  A 侧 {A:<28} enc{a_ver} / in_channels={a_ch}")
    print(f"  B 侧 {B:<28} enc{b_ver} / in_channels={b_ch}")
    print()

    # 组合判定：某侧跑 search32 时，该侧模型通道数必须等于内核通道数
    ok_a_s32 = a_ch == kern_ch
    ok_b_s32 = b_ch == kern_ch
    print("  组合 1  新 greedy  vs master+N32   (闸门 C 放大) :",
          "可用" if ok_b_s32 else "不可用", "← B 侧 search32")
    print("  组合 2  新+N32     vs master+N32   (计划书主目标):",
          "可用" if (ok_a_s32 and ok_b_s32) else "不可用", "← A/B 双侧 search32")
    print("  组合 3  新 greedy  vs master greedy(闸门 A 放大) : 可用（不经搜索内核）")
    print()

    if not ok_a_s32:
        print("  [P0] A 侧 enc3 模型不能挂 search32："
              "`eval_pairs.py --a <exit_best> --a-mode search32` 会形状 panic。")
        print("       默认终评口径改用组合 1（对应 §4『超额：新贪心 ≥ 现 N32』）；")
        print("       组合 2 需先完成 enc3 搜索内核（§6.11.3），完成前禁止执行。")
    print()

    print("-" * 72)
    print("终评命令清单（DEADLINE 之后执行；三项共用同一独立池以便逐 seed 配对）")
    print("-" * 72)
    cmds = [
        ("① 主口径 新贪心 vs master+N32",
         f"{PY} eval_pairs.py --a {A} --b {B} --a-mode greedy --b-mode search32 "
         f"--seeds {SEEDS} --seed0 {POOL} --seats 0,1,2,3 --workers 6 "
         f"--out runs/gates/final_primary.jsonl"),
        ("② 辅口径 新贪心 vs master 贪心",
         f"{PY} eval_pairs.py --a {A} --b {B} --mode greedy "
         f"--seeds {SEEDS} --seed0 {POOL} --seats 0,1,2,3 --workers 6 "
         f"--out runs/gates/final_gateA.jsonl"),
        ("③ 趋势口径 新贪心 vs master+N32（训练期同池对照，可选）",
         f"{PY} eval_pairs.py --a {A} --b {B} --a-mode greedy --b-mode search32 "
         f"--seeds 50 --seed0 720000 --seats 0,1,2,3 --workers 6 "
         f"--out runs/gates/final_trend.jsonl"),
    ]
    for title, cmd in cmds:
        print(f"\n  {title}\n    {cmd}")

    print("\n  [禁用] 组合 2（enc3 搜索内核完成前）：")
    print(f"    {PY} eval_pairs.py --a {A} --b {B} --a-mode search32 "
          f"--b-mode search32 --seeds {SEEDS} --seed0 {POOL} ...")
    print("\n  统计口径：deltas[cand_seat] − mean(其余三家)，四座按 seed 平均，"
          "正态近似双侧 p；禁止自对弈胜率作决策。")
    print("  冻结基线池：master 贪心 / master+N32 / 上轮最佳；牌墙 seed 池固定。")
    print("  统计生成：./.venv/bin/python tools/final_eval_report.py "
          "primary=runs/gates/final_primary.jsonl gateA=runs/gates/final_gateA.jsonl --md")
    print("  导出/在线冒烟清单 + 终评报告模板：reports/终评执行手册-2026-11-13.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
