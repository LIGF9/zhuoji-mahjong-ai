"""探针：更新段（`selfplay.update`）还能不能再提速？

背景
----
2026-10-12 的 M1 冒烟运行中，用 PowerShell 对训练**主进程**做 2 秒粒度 CPU 采样，
得到 60 秒内的剖面：

    77 78 54 0 0 0 0 0 159 728 713 698 684 62 94 61 78 62 97 57 76 149 80 70 55 68 16 0 0 0

即：大量窗口在 0%（主进程阻塞、等 worker 采集），而真正在算梯度的窗口只有
**684–728%（约 7 个核）** —— 但命令行给的是 `--threads 24`。这与"更新段 =
24 线程满负荷"的假设不符，也意味着**更新段可能还有 2–3× 的余量**。

本项目此前把 `update()` 认定为"唯一的、加机器也提不上去的瓶颈"（占单轮 2/3）。
如果 7 核→20 核能成立，单轮从 42s 降到 ~25s，四段计划的预算要整体重写。
所以这个探针值得花 10 分钟去证伪或证实。

本探针直接测三件事（用 `time.process_time()` / wall 得**有效并行度**）：
1. 线程扩展曲线：wall 随 threads 的下降有多快；
2. `channels_last`：CPU conv 走 oneDNN 的通道后置布局，AVX2 机器上常有收益；
3. 绑核：把进程限制在 8 个 P 核（逻辑 0–15）上，避免被 12 个 E 核拖累调度。

> **必须在机器空闲时跑**。带着训练进程跑会被线程争抢污染（实测 wall 时间
> 整体膨胀 3 倍、并行度读数不可信）。合适的窗口：上一段训练结束、下一段起动前。

用法::

    python scripts/probe_threads.py                          # 线程扩展曲线
    python scripts/probe_threads.py --variants default,channels_last,pin_p
    python scripts/probe_threads.py --json reports/probe_threads.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from zhuoji.net_m1 import M1Net  # noqa: E402


#: i7-14700K：逻辑 0–15 = 8 个 P 核（各 2 超线程），16–27 = 12 个 E 核。
#: 小 spatial（4×9）的 conv 在 E 核上拿不到好处，却会被调度器来回搬，绑 P 核常更稳。
PCORE_CPUS = list(range(16))


def _set_affinity(cpus: list[int] | None):
    if cpus is None:
        return "全部逻辑核"
    try:
        import psutil
        psutil.Process().cpu_affinity(cpus)
        return f"绑核 {cpus[0]}–{cpus[-1]}（{len(cpus)} 个逻辑核）"
    except Exception as e:                      # noqa: BLE001
        return f"绑核失败（{type(e).__name__}）：{e}"


def one_step(model, opt, tiles, glob, batch: int, threads: int,
             reps: int = 1) -> tuple[float, float]:
    """跑 reps 个 batch 的 forward+backward，返回 (wall_s, cpu_s) 均值。"""
    torch.set_num_threads(threads)
    sel = slice(0, batch)
    # 预热：第一个 step 含 kernel 选择/线程池建立的开销，不计入
    out = model(tiles[sel], glob[sel])
    loss = sum(o.float().pow(2).mean() for o in out.values())
    opt.zero_grad(set_to_none=True)
    loss.backward()
    opt.step()

    w0, c0 = time.perf_counter(), time.process_time()
    for _ in range(reps):
        out = model(tiles[sel], glob[sel])
        loss = sum(o.float().pow(2).mean() for o in out.values())
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    w1, c1 = time.perf_counter(), time.process_time()
    return (w1 - w0) / reps, (c1 - c0) / reps


def _warn_if_busy() -> bool:
    """检测是否有训练/评测在跑。有的话本探针的读数不可信（线程争抢）。"""
    try:
        import psutil
    except ImportError:
        return False
    busy = []
    for p in psutil.process_iter(["cmdline"]):
        cl = list(p.info.get("cmdline") or [])
        if len(cl) > 1 and any(cl[1].endswith(t) for t in
                                ("selfplay.py", "score_checkpoints.py", "duel.py")):
            busy.append(f"{Path(cl[1]).name}#{p.pid}")
    if busy:
        print("!! 检测到正在运行的任务：" + "、".join(busy))
        print("!! 线程争抢会让本探针的 wall 时间整体膨胀、并行度读数失真 ——")
        print("!! 请在训练间隙（上一段结束、下一段起动前）重跑，读数才有意义。\n")
        return True
    return False


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--channels", type=int, default=160)
    ap.add_argument("--blocks", type=int, default=20)
    ap.add_argument("--hidden", type=int, default=512)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--reps", type=int, default=1,
                    help="每个档位测几个 step（默认 1，省时间；要更稳给 3）")
    ap.add_argument("--threads-list", default="8,16,24,28")
    ap.add_argument("--variants", default="default",
                    help="逗号分隔：default / channels_last / pin_p / pin_p_cl "
                         "（cl = channels_last）")
    ap.add_argument("--json", default="")
    ap.add_argument("--force", action="store_true", help="机器忙时也照样跑")
    args = ap.parse_args()

    busy = _warn_if_busy()
    if busy and not args.force:
        print("（加 --force 可忽略此提示强行测量）")
        return 2

    torch.manual_seed(0)
    print(f"M1Net {args.channels}ch × {args.blocks} blocks，hidden {args.hidden}")
    print(f"torch {torch.__version__}，cpu_capability={torch.backends.cpu.get_cpu_capability()}，"
          f"逻辑核 {os.cpu_count()}，interop_threads {torch.get_num_interop_threads()}")
    print(f"OMP_NUM_THREADS={os.environ.get('OMP_NUM_THREADS', '(未设)')}")

    records = []
    for variant in [v.strip() for v in args.variants.split(",") if v.strip()]:
        cl = variant.endswith("_cl") or variant == "channels_last"
        pin = variant.startswith("pin_p")
        aff = _set_affinity(PCORE_CPUS if pin else None)
        print(f"\n--- 变体 {variant}：{aff}，memory_format="
              f"{'channels_last' if cl else 'contiguous'} ---")

        torch.manual_seed(0)
        model = M1Net(args.channels, args.blocks, args.hidden)
        if cl:
            model = model.to(memory_format=torch.channels_last)
        model.train()
        opt = torch.optim.Adam(model.parameters(), lr=1e-4)
        n = args.batch
        tiles = torch.rand(n, 63, 4, 9)
        glob = torch.rand(n, 130)
        if cl:
            tiles = tiles.to(memory_format=torch.channels_last)

        print(f"{'线程':>5}{'wall s/step':>14}{'cpu s/step':>13}{'有效并行度':>12}   备注")
        base = None
        for t in [int(x) for x in args.threads_list.split(",")]:
            wall, cpu = one_step(model, opt, tiles, glob, args.batch, t, args.reps)
            par = cpu / wall if wall > 0 else 0
            note = "基准" if base is None else f"加速 {base / wall:.2f}×"
            if base is None:
                base = wall
            print(f"{t:>5}{wall:>14.3f}{cpu:>13.3f}{par:>12.2f}   {note}")
            records.append({"variant": variant, "threads": t,
                            "wall_s": round(wall, 4), "cpu_s": round(cpu, 4),
                            "parallel": round(par, 3), "affinity": aff,
                            "channels_last": cl})

    print("\n判读：")
    print("  * wall 随线程下降得越慢 → 并行扩展越差，多给的线程只是在互相争抢；")
    print("  * 有效并行度（cpu/wall）远小于线程数 → 这段计算**本质上没并行起来**；")
    print("  * 比较变体之间的 wall：这才是「还能不能再提速」的直接证据 ——")
    print("    若 channels_last 或绑 P 核明显更快，就把它写进 selfplay 的更新段。")

    if args.json:
        p = Path(args.json)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n原始数据 -> {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

