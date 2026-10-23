"""采集段 worker 扩展性基准 —— 只读实验。

背景：auto4 实测 collect=12.6s / 192 局，但只有 12 个 worker 在跑，
而机器有 28 个逻辑核 ⇒ collect 阶段有 16 核结构性闲置。
本脚本量化 workers 增加后的实际收益（墙钟），并监控内存增量。

用法::
    python scripts/opt_bench_collect.py --list 12,16,20,24
"""
from __future__ import annotations

import argparse
import multiprocessing as mp
import sys
import time
from pathlib import Path

import psutil
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from zhuoji.net import (build_from_ckpt, ck_from_model,   # noqa: E402
                        net_config_from_ckpt, net_meta)
from zhuoji.vcollect import collect_parallel              # noqa: E402

CKPT = ROOT / "models/_ckpts/dushan_m2_auto4_it0260.pt"


def tree_rss_gb() -> float:
    me = psutil.Process()
    tot = me.memory_info().rss
    try:
        for ch in me.children(recursive=True):
            try:
                tot += ch.memory_info().rss
            except psutil.Error:
                pass
    except psutil.Error:
        pass
    return tot / 1e9


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", default="12,16,20,24")
    ap.add_argument("--games", type=int, default=192)
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--ckpt", default=str(CKPT))
    args = ap.parse_args()

    workers_list = [int(x) for x in args.list.split(",") if x.strip()]

    print(f"# 加载 {Path(args.ckpt).name}")
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    model, ver = build_from_ckpt(ck)
    nc = net_meta(net_config_from_ckpt(ck))
    sd = model.state_dict()
    snap = ck_from_model(model, encoder_version=ver,
                         encoder_packs=model.encoder_packs)
    print(f"# arch={nc['arch']} {nc['channels']}ch×{nc['blocks']} ver=v{ver} "
          f"packs={model.encoder_packs}")
    vm = psutil.virtual_memory()
    print(f"# 起始内存 {vm.percent:.0f}% 可用 {vm.available/1e9:.1f}G  "
          f"逻辑核 {mp.cpu_count()}")
    print()

    ctx = mp.get_context("spawn")
    out = []
    for w in workers_list:
        loop = ctx.Pool(processes=w)
        try:
            # 预热：建 worker 侧缓存（快照），不计入测量
            collect_parallel(
                sd, nc, args.games, 2026, 1.2, 1.0, 0.0, workers=w,
                past_prob=0.3, snapshots_sd=[snap], rules="dushan",
                want_traj=True, pool=loop, encoder_version=ver,
                encoder_packs=model.encoder_packs, snap_key="warm",
                explore_eps=0.01, master_prob=0.40)
            times = []
            for _ in range(args.reps):
                t0 = time.perf_counter()
                traj, rewards, dt, _aux = collect_parallel(
                    sd, nc, args.games, 2026, 1.2, 1.0, 0.0, workers=w,
                    past_prob=0.3, snapshots_sd=None, rules="dushan",
                    want_traj=True, pool=loop, encoder_version=ver,
                    encoder_packs=model.encoder_packs, snap_key="warm",
                    explore_eps=0.01, master_prob=0.40)
                times.append(time.perf_counter() - t0)
            best = min(times)
            gb = tree_rss_gb()
            vm = psutil.virtual_memory()
            print(f"workers={w:2d}  采集 {best:6.2f}s ({args.games} 局)  "
                  f"内部计时 {dt:6.2f}s  "
                  f"进程树 RSS {gb:5.2f}G  系统 {vm.percent:3.0f}% "
                  f"({', '.join(f'{t:.2f}' for t in times)})")
            out.append((w, best))
        finally:
            loop.close()
            loop.join()

    print()
    if out:
        base = out[0][1]
        print("=== 相对 workers=%d 的加速比 ===" % out[0][0])
        for w, t in out:
            print(f"  workers={w:2d}  {base / t:5.2f}x   {t:6.2f}s  "
                  f"(相对省 {base - t:+.2f}s/轮)")


if __name__ == "__main__":
    main()
