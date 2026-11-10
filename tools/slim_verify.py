#!/usr/bin/env python3
"""E-SLIM 工程判据验收（§6.73.4）：参数量 / 贪心单步 / N32 单步 / ONNX 一致性。

只读（除导出 ONNX 缓存），不改任何模型与线上配置。逐项打印并与阈值比较：
  * 参数量 ≤ 1.2M
  * N32 单步（ply3/n_det32/topk3/ff_cap24/fair，MPS）≤ 0.35 s
  * 贪心单步（CPU 单前向）≤ 1.5 ms
  * ONNX vs torch 的 logits max|Δ| ≤ 1e-4
  * 编码口径 == v3（63/130）

用法：
    .venv/bin/python tools/slim_verify.py --stem slim_s
"""
from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from zhuoji.agent import NetAgent                      # noqa: E402
from zhuoji.dushan import DushanConfig, DushanGame     # noqa: E402
from zhuoji.forward import OrtForward, ensure_onnx     # noqa: E402
from zhuoji.net import build_from_ckpt                 # noqa: E402
from zhuoji.rules import Phase                         # noqa: E402
from zhuoji.rust_search import RustSearchAgent         # noqa: E402

LIMITS = {"params": 1.2e6, "n32_s": 0.35, "greedy_ms": 1.5, "onnx_abs": 1e-4}


def load(stem: str):
    ck = torch.load(str(ROOT / "models" / f"{stem}.pt"),
                    map_location="cpu", weights_only=False)
    net, enc = build_from_ckpt(ck)
    net.onnx_stem = stem
    return net, enc


def decision_point(seed: int = 909090, depth: int = 20):
    g = DushanGame(DushanConfig(), seed=seed)
    rng = np.random.default_rng(1)
    for _ in range(depth):
        if g.phase == Phase.OVER:
            break
        acts = g.legal_actions()
        if not acts:
            break
        g.step(acts[int(rng.integers(0, len(acts)))])
    return g


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stem", default="slim_s")
    ap.add_argument("--repeat", type=int, default=3)
    args = ap.parse_args()

    net, enc = load(args.stem)
    n_par = sum(p.numel() for p in net.parameters())
    ok = True

    def chk(name, val, limit, unit, cmp="le"):
        nonlocal ok
        good = (val <= limit) if cmp == "le" else (val >= limit)
        ok &= good
        print(f"  {'PASS' if good else 'FAIL'}  {name:14s} {val:10.4f}{unit}"
              f"  (阈值 {'≤' if cmp=='le' else '≥'} {limit:g}{unit})", flush=True)

    print(f"=== {args.stem} ===", flush=True)
    print(f"  encoder_version={enc}  in_channels={net.in_channels} "
          f"global_dim={net.global_dim}  arch={getattr(net,'arch','?')}", flush=True)
    chk("参数量", n_par / 1e6, LIMITS["params"] / 1e6, "M")
    if enc != 3 or net.in_channels != 63 or net.global_dim != 130:
        print("  FAIL  编码口径非 v3(63/130)", flush=True)
        ok = False

    g = decision_point()
    legal = g.legal_actions()
    print(f"  [game] phase={g.phase} actor={g.actor()} legal={len(legal)}", flush=True)

    # 贪心单步（CPU 单前向，与 master 0.66ms / P48 4.35ms 同机同法）
    ag = NetAgent(net, seed=0, temperature=0.0, name="greedy")
    ag.choose(g)
    ts = []
    for _ in range(20):
        t0 = time.perf_counter()
        ag.choose(g)
        ts.append((time.perf_counter() - t0) * 1000)
    chk("贪心单步", statistics.median(ts), LIMITS["greedy_ms"], "ms")

    # N32 单步（线上搜索档同口径，MPS）
    rs = RustSearchAgent(net, seed=7, device="mps", name="n32",
                         ply=3, n_det=32, topk=3, ff_cap=24, fair=True)
    rs.rank_detail(g)                       # 冷启动（建会话/编译）
    ss = []
    for _ in range(args.repeat):
        t0 = time.perf_counter()
        rs.rank_detail(g)
        ss.append(time.perf_counter() - t0)
    n32 = statistics.median(ss)
    chk("N32 单步", n32, LIMITS["n32_s"], "s")
    print(f"        （各次 {[round(x,3) for x in ss]} s）", flush=True)

    # ONNX 一致性
    path = Path(ensure_onnx(net, args.stem))
    print(f"  onnx → {path.name}  ({path.stat().st_size/1024**2:.1f} MB)", flush=True)
    d = np.load(ROOT / "runs/exit_r48/exit_w3.npz")
    xt = d["tiles"][:8]
    xg = d["glob"][:8]
    net.eval()
    with torch.no_grad():
        out = net(torch.from_numpy(xt), torch.from_numpy(xg))
    fwd = OrtForward(path, "cpu", warm=False, ch=int(net.in_channels), gdim=int(net.global_dim))
    ro = fwd.raw(xt, xg)
    worst = 0.0
    for k in ("discard", "self_kong", "respond", "value"):
        a = out[k].cpu().numpy().reshape(len(xt), -1)
        b = np.asarray(ro[k]).reshape(len(xt), -1)
        worst = max(worst, float(np.max(np.abs(a - b))))
    chk("ONNX |Δ|", worst, LIMITS["onnx_abs"], "")

    print(f"=== {'ALL PASS' if ok else 'HAS FAIL'} ===", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
