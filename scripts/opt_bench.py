"""训练性能基准 —— 只读实验，不改动任何生产代码。

目的：在**不影响正在运行的 auto4** 的前提下，量化 update 段各项优化的墙钟收益。

用法::

    python scripts/opt_bench.py --threads 24 --n 2048 --skip-compile

设计说明
--------
* 使用真实的 :class:`zhuoji.net_m1.M1Net`（160ch / 20blk / 512hidden）与真实的
  输入形状 (B, 63, 27) + (B, 130)，确保测得的吞吐可外推到真实训练。
* loss 结构与 ``selfplay.update`` 保持一致：4 个头各自 log_softmax → NLL，
  value 头用 smooth_l1。backward 计算图与生产代码同构。
* 所有结论只用**墙钟**。本项目已确认 CPU 时间计量不可信（见 2026-10-16 记录）。
"""
from __future__ import annotations

import argparse
import os
import statistics
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from zhuoji.net_m1 import M1Net                      # noqa: E402
from zhuoji.encoder import HEAD_DIMS                 # noqa: E402
from zhuoji.vcollect import HEAD_NAMES               # noqa: E402


def make_batch(n: int, channels_last: bool = False):
    tiles = torch.randn(n, 63, 27)
    glob = torch.randn(n, 130)
    mask = torch.ones(n, 28)
    head = torch.randint(0, len(HEAD_NAMES), (n,))
    act = torch.randint(0, 27, (n,))
    if channels_last:
        tiles = tiles.reshape(n, 63, 3, 9).contiguous(memory_format=torch.channels_last)
    return tiles, glob, mask, head, act


def build_loss(out, mask, head, act, value_coef=0.5):
    """与 selfplay.update 同构的 loss（去掉 KL/蒸馏项，它们不改计算图主干）。"""
    total = None
    for hid, name in enumerate(HEAD_NAMES):
        sub = (head == hid).nonzero(as_tuple=True)[0]
        if sub.numel() == 0:
            continue
        dim = HEAD_DIMS[name]
        lg = out[name][sub][:, :dim]
        m = mask[sub][:, :dim].bool()
        lg = lg.masked_fill(~m, -1e9)
        lp = torch.log_softmax(lg, dim=-1)
        a = act[sub].clamp(max=dim - 1)
        term = -lp.gather(1, a.unsqueeze(1)).squeeze(1).mean()
        total = term if total is None else total + term
    v = out["value"]
    ret = torch.randn_like(v) * 0.1
    total = total + value_coef * torch.nn.functional.smooth_l1_loss(v, ret)
    return total


def run_once(model, opt, data, batch, epochs, ref_model=None, clip=True):
    tiles, glob, mask, head, act = data
    n = tiles.shape[0]
    perm = torch.randperm(n)
    nb = 0
    for _ in range(epochs):
        for i in range(0, n, batch):
            sel = perm[i:i + batch]
            if sel.numel() < 16:
                continue
            out = model(tiles[sel], glob[sel])
            if ref_model is not None:
                with torch.no_grad():
                    ref_model(tiles[sel], glob[sel])
            loss = build_loss(out, mask[sel], head[sel], act[sel])
            opt.zero_grad()
            loss.backward()
            if clip:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            nb += 1
    return nb


def bench(tag, threads, n, batch, epochs=1, channels_last=False,
          ref=False, clip=True, reps=1):
    torch.set_num_threads(threads)
    model = M1Net(160, 20, 512)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-6, weight_decay=1e-4)
    ref_model = None
    if ref:
        import copy
        ref_model = copy.deepcopy(model).eval()
        for p in ref_model.parameters():
            p.requires_grad_(False)
    data = make_batch(n, channels_last)
    # warmup（分配缓冲、第一次 conv 算法选择）
    run_once(model, opt, data, batch, 1, ref_model, clip)
    times = []
    for _ in range(reps):
        t0 = time.perf_counter()
        nb = run_once(model, opt, data, batch, epochs, ref_model, clip)
        times.append(time.perf_counter() - t0)
    dt = min(times)
    sps = n * epochs / dt
    print(f"{tag:38s} threads={threads:2d} batch={batch:4d} "
          f"n={n} nb={nb:2d}  {dt:6.3f}s  {sps:8.0f} samples/s")
    return dt, sps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--threads", type=int, default=24)
    ap.add_argument("--n", type=int, default=2048)
    ap.add_argument("--skip-compile", action="store_true")
    ap.add_argument("--only", default="", help="只跑匹配关键字的配置")
    ap.add_argument("--affinity", default="",
                    help="CPU 亲和性，如 '0-15'（i7-14700K 的 P-core 逻辑核）")
    args = ap.parse_args()

    if args.affinity:
        import psutil
        lo, _, hi = args.affinity.partition("-")
        cpus = list(range(int(lo), int(hi or lo) + 1))
        psutil.Process().cpu_affinity(cpus)
        print(f"# 已绑定 CPU 亲和性 {cpus[0]}-{cpus[-1]}（{len(cpus)} 个逻辑核）")

    print(f"# torch {torch.__version__}  CPU cap={torch.backends.cpu.get_cpu_capability()}")
    print(f"# 逻辑核 {os.cpu_count()}   基准 n={args.n}（真实训练 n≈8700，17 minibatch）")
    print()
    results = {}

    def want(tag):
        return (not args.only) or (args.only in tag)

    # ---- 基线：与生产配置完全一致 ----
    if want("A_base"):
        results["A_base"] = bench("A_base(生产配置 contig)", args.threads,
                                  args.n, 512)

    # ---- 布局：channels_last ----
    if want("B_chlast"):
        results["B_chlast"] = bench("B_chlast(channels_last)", args.threads,
                                    args.n, 512, channels_last=True)

    # ---- flush denormal ----
    if want("C_denorm"):
        torch.set_flush_denormal(True)
        results["C_denorm"] = bench("C_denorm(flush_denormal)", args.threads,
                                    args.n, 512)
        torch.set_flush_denormal(False)

    # ---- threads 扫描 ----
    for th in (8, 12, 16, 20, 28):
        tag = f"D_threads{th}"
        if want(tag):
            results[tag] = bench(f"D_threads{th}", th, args.n, 512)

    # ---- batch 大小 ----
    for bs in (1024, 2048):
        tag = f"E_batch{bs}"
        if want(tag):
            results[tag] = bench(f"E_batch{bs}", args.threads, args.n, bs)

    # ---- ref 前向开销（模拟 kl-anchor 轮）----
    if want("F_ref"):
        results["F_ref"] = bench("F_ref(带锚点前向)", args.threads,
                                 args.n, 512, ref=True)

    # ---- 组合：flush_denormal + batch1024 ----
    if want("H_combo"):
        torch.set_flush_denormal(True)
        results["H_combo"] = bench("H_combo(denorm+batch1024)", args.threads,
                                   args.n, 1024)
        torch.set_flush_denormal(False)

    # ---- torch.compile ----
    if not args.skip_compile and want("G_compile"):
        torch.set_num_threads(args.threads)
        model = M1Net(160, 20, 512)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-6)
        data = make_batch(args.n)
        print("  [compile] 正在编译（默认 mode，首次可能需数分钟）...", flush=True)
        t0 = time.perf_counter()
        try:
            cmodel = torch.compile(model)
            run_once(cmodel, opt, data, 512, 1)
        except Exception as e:                       # noqa: BLE001
            print(f"  [compile] 失败：{type(e).__name__}: {str(e)[:120]}")
            return
        t_compile = time.perf_counter() - t0
        print(f"  [compile] 编译+首跑耗时 {t_compile:.1f}s", flush=True)
        times = []
        for _ in range(2):
            t0 = time.perf_counter()
            nb = run_once(cmodel, opt, data, 512, 1)
            times.append(time.perf_counter() - t0)
        dt = min(times)
        print(f"{'G_compile(torch.compile)':38s} threads={args.threads:2d} "
              f"batch= 512  n={args.n} nb={nb:2d}  {dt:6.3f}s  {args.n / dt:8.0f} samples/s")
        results["G_compile"] = (dt, args.n / dt)

    print()
    if "A_base" in results:
        base = results["A_base"][1]
        print("=== 相对基线加速比（越大越好）===")
        for k, (_, sps) in sorted(results.items()):
            print(f"  {k:16s} {sps / base:5.2f}x   ({sps:8.0f} samples/s)")


if __name__ == "__main__":
    main()
