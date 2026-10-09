"""
一键跑通整条流水线：自检 → 生成数据 → 行为克隆 → 自我对弈强化 → 复式评测
→ 样例对局 → HTML 报告。

    python scripts/run_all.py --quick      # 几分钟的小规模全链路冒烟
    python scripts/run_all.py              # 默认的小规模正式跑

每一步都用 ``sys.executable`` 起子进程，所以只要当前解释器装好了依赖，
整条链路就能跑；中途任何一步非零退出都会立刻停下并保留已有产物。
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"


def run(label: str, args: list[str], cwd: Path = ROOT) -> None:
    print(f"\n{'=' * 72}\n>>> {label}\n    {' '.join(args)}\n{'=' * 72}", flush=True)
    t0 = time.time()
    proc = subprocess.run([sys.executable, *args], cwd=str(cwd))
    if proc.returncode != 0:
        raise SystemExit(f"[FAIL] {label} 退出码 {proc.returncode}")
    print(f"<<< {label} 完成，用时 {time.time() - t0:.1f}s", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true",
                    help="极小规模，只为验证链路能跑通")
    ap.add_argument("--regen-data", action="store_true",
                    help="即使 data/bc_data.npz 已存在也重新生成")
    ap.add_argument("--no-rl", action="store_true", help="跳过自我对弈强化")
    args = ap.parse_args()

    q = args.quick
    data = ROOT / "data" / "bc_data.npz"

    # ---- 0. 自检 ----------------------------------------------------------
    run("自检 · 规则/番型/零和", [str(SCRIPTS.parent / "tests" / "smoke_test.py")])
    run("自检 · 编码器一致性", [str(SCRIPTS.parent / "tests" / "test_encoder.py")])
    run("自检 · 抢杠回归", [str(SCRIPTS.parent / "tests" / "test_rob_kong.py")])

    # ---- 1. 数据 ----------------------------------------------------------
    if data.exists() and not args.regen_data:
        print(f"\n[skip] 复用已有数据集 {data}（加 --regen-data 可强制重建）")
    else:
        run("生成行为克隆数据（启发式教师池）",
            [str(SCRIPTS / "gen_bc_data.py"),
             "--games", "400" if q else "8000",
             "--workers", "4"])

    # ---- 2. 行为克隆 ------------------------------------------------------
    run("行为克隆（监督学习）",
        [str(SCRIPTS / "train_bc.py"),
         "--epochs", "2" if q else "6",
         "--batch", "1024" if q else "2048",
         "--channels", "32" if q else "48",
         "--blocks", "1" if q else "2",
         "--hidden", "256",
         "--lr", "2e-3",
         "--pass-keep", "0.35"])

    # ---- 3. 自我对弈强化 --------------------------------------------------
    if args.no_rl:
        print("\n[skip] 按要求跳过自我对弈强化")
    else:
        run("自我对弈强化（REINFORCE + 价值基线）",
            [str(SCRIPTS / "selfplay.py"),
             "--init", str(ROOT / "models" / "bc.pt"),
             "--iters", "4" if q else "12",
             "--games-per-iter", "24" if q else "48",
             "--eval-rounds", "8" if q else "20",
             "--eval-every", "2" if q else "4"])

    # ---- 4. 评测 ----------------------------------------------------------
    # 报告要并排对比"强化后"和"强化前"两代模型，所以两份 arena 都跑：
    #   reports/arena.json    <- rl.pt（若没跑 RL 则退化为 bc.pt）
    #   reports/arena_bc.json <- bc.pt
    rl = ROOT / "models" / "rl.pt"
    bc = ROOT / "models" / "bc.pt"
    model = rl if rl.exists() else bc
    rounds = "20" if q else "60"
    run(f"复式对战评测（{model.name}）",
        [str(SCRIPTS / "arena.py"),
         "--model", str(model),
         *(["--baseline", str(bc)] if model == rl else []),
         "--rounds", rounds])
    if model != bc:
        run(f"复式对战评测（{bc.name}，作为报告基线）",
            [str(SCRIPTS / "arena.py"),
             "--model", str(bc),
             "--out", str(ROOT / "reports" / "arena_bc.json"),
             "--rounds", rounds])
        # 两份评测吃同一批牌、对手种子也确定 → 可以逐副牌配对，把牌运消掉。
        # 单看两份独立评测的均值是分不出 RL 和 BC 强弱的。
        run("逐副牌配对比较（RL vs BC）",
            [str(SCRIPTS / "compare_arena.py"),
             "--a", str(ROOT / "reports" / "arena.json"),
             "--b", str(ROOT / "reports" / "arena_bc.json"),
             "--label-a", f"RL ({rl.name})",
             "--label-b", f"BC ({bc.name})"])

    # ---- 5. 样例对局 ------------------------------------------------------
    run("导出样例对局（含动作概率）",
        [str(SCRIPTS / "make_sample_game.py"), "--model", str(model)])

    # ---- 6. 报告 ----------------------------------------------------------
    run("生成 HTML 报告", [str(SCRIPTS / "make_report.py")])

    print(f"\n全部完成。报告：{ROOT / 'reports' / 'report.html'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
