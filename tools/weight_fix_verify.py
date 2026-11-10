#!/usr/bin/env python3
"""E-WEIGHT 验收：修根因后「冠军搜索档」是否真的用冠军权重 + 正确编码。

只读，不改任何模型/标志。检查：
  1. get_model 语义（onnx_stem=stem）⇒ ensure_onnx 指向 stem 自己的 ONNX；
  2. RustSearchAgent.rank_detail 对 v3 权重传 enc=3（Rust 内核 shape=63/130）；
  3. 同一决策点，冠军搜索 Q 与主力搜索 Q **不同**（假档时逐位相同）；
  4. 记录 search128(ply1×128) / search32(ply3×32) 的真实单步耗时；
  5. v1 主力仍走 enc=1，行为不回归。

用法：
    .venv/bin/python tools/weight_fix_verify.py [--heavy]
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "web"))

from zhuoji.dushan import DushanConfig, DushanGame  # noqa: E402
from zhuoji.forward import ensure_onnx  # noqa: E402
from zhuoji.net import build_from_ckpt  # noqa: E402
from zhuoji.rules import Phase  # noqa: E402
from zhuoji.rust_search import RustSearchAgent  # noqa: E402


def load(stem: str):
    ck = torch.load(str(ROOT / "models" / f"{stem}.pt"),
                    map_location="cpu", weights_only=False)
    net, enc = build_from_ckpt(ck)
    net.onnx_stem = stem            # 与修后的 get_model() 一致
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
    ap.add_argument("--heavy", action="store_true",
                    help="额外跑 ply3×n_det32（慢，用于估线上 search32 真实成本）")
    ap.add_argument("--device", default="ort", help="前向后端：ort / ort-cpu / mps / cpu")
    ap.add_argument("--n-det", type=int, default=128)
    ap.add_argument("--ply", type=int, default=1)
    ap.add_argument("--only", default="", help="只测某个 stem（后端横评比测用）")
    ap.add_argument("--repeat", type=int, default=1, help="同一 agent 连跑几次，分离冷/热启动")
    args = ap.parse_args()
    g = decision_point()
    legal = g.legal_actions()
    print(f"[game] phase={g.phase} actor={g.actor()} legal={len(legal)}", flush=True)
    ok = True
    q_by_stem: dict[str, np.ndarray] = {}

    for stem, want_enc in (("dushan_master", 1), ("champ_exit_p48", 3)):
        if args.only and stem != args.only:
            q_by_stem[stem] = np.zeros(len(legal))
            continue
        net, enc = load(stem)
        path = Path(ensure_onnx(net, stem))
        print(f"[{stem}] encoder_version={enc} onnx={path.name} "
              f"in_channels={net.in_channels} global_dim={net.global_dim}", flush=True)
        if path.stem != stem:
            print(f"  !! 假档：ONNX 落到 {path.stem}", flush=True)
            ok = False
        if enc != want_enc:
            print(f"  !! 编码版本异常：期望 {want_enc}，实得 {enc}", flush=True)
            ok = False

        ag = RustSearchAgent(net, seed=7, device=args.device, name="verify",
                             ply=args.ply, n_det=args.n_det, topk=3,
                             ff_cap=24, fair=True)
        if ag.enc != want_enc:
            print(f"  !! RustSearchAgent.enc={ag.enc}，期望 {want_enc}", flush=True)
            ok = False
        q, qs, diag = np.zeros(len(legal)), [], []
        for rep in range(args.repeat):
            t0 = time.perf_counter()
            q, qs, diag = ag.rank_detail(g)
            dt = time.perf_counter() - t0
            print(f"  [{rep+1}/{args.repeat}] device={args.device} "
                  f"ply{args.ply}×n_det{args.n_det} rank_detail: {dt:.2f}s  "
                  f"len(q)={len(q)}  Q[:4]={np.round(q[:4], 3).tolist()}",
                  flush=True)
        q_by_stem[stem] = np.asarray(q, dtype=np.float64)
        if len(q) != len(legal):
            print("  !! Q 长度与合法动作数不符", flush=True)
            ok = False
        if args.heavy:
            ag32 = RustSearchAgent(net, seed=7, device="ort", name="verify32",
                                   ply=3, n_det=32, topk=3, ff_cap=24, fair=True)
            t0 = time.perf_counter()
            q32, _, _ = ag32.rank_detail(g)
            print(f"  search32(ply3×32)  rank_detail: "
                  f"{time.perf_counter() - t0:.2f}s  len(q)={len(q32)}", flush=True)

    if args.only:
        print("[区分度] --only 模式跳过跨模型比对", flush=True)
    else:
        a, b = q_by_stem["dushan_master"], q_by_stem["champ_exit_p48"]
        if a.shape == b.shape:
            maxd = float(np.max(np.abs(a - b))) if a.size else 0.0
            print(f"[区分度] max|Q_master − Q_champ| = {maxd:.4f}", flush=True)
            if maxd < 1e-6:
                print("  !! 两个权重给出逐位相同的 Q ⇒ 仍是假档", flush=True)
                ok = False
    print("VERDICT:", "PASS" if ok else "FAIL", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
