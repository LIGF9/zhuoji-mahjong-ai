# -*- coding: utf-8 -*-
"""A1：把 M1 系检查点导出为 ONNX（推理态），并做 torch/ORT logits 对拍。

用法：
  python scripts/export_onnx.py --ckpt models/_ckpts/dushan_m2_c4_it2280 \
      --out models/standalone/dushan_m2_c4_it2280.onnx

对拍内容：
  1. 随机张量（覆盖数值路径）
  2. 真实编码观测（随机独山对局若干步，走 encode_v3 真实分布）
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from zhuoji.net_m1 import M1Net  # noqa: E402


def build_net(ck: dict) -> M1Net:
    net_cfg = ck["net"]
    net = M1Net(
        channels=net_cfg["channels"], blocks=net_cfg["blocks"],
        hidden=net_cfg["hidden"], global_dim=net_cfg["global_dim"],
        in_channels=net_cfg["in_channels"],
        spatial_head=bool(net_cfg.get("spatial_head")),
        se=bool(net_cfg.get("se", True)),
        bn_eps=float(net_cfg.get("bn_eps", 1e-3)),
        aux=bool(net_cfg.get("aux", False)),
    )
    net.load_state_dict(ck["state_dict"], strict=True)
    net.eval()
    return net


def real_observations(n: int = 64, seed: int = 7):
    """随机独山对局的真实编码观测（head 也一并返回，供分组对拍）。"""
    from zhuoji.dushan import DushanConfig, DushanGame
    from zhuoji.encoder import encode
    from zhuoji.rules import Phase

    games, rng = [], np.random.default_rng(seed)
    obs_list = []
    while len(obs_list) < n:
        g = DushanGame(DushanConfig(), seed=int(rng.integers(1 << 30)))
        steps = 0
        while g.phase is not Phase.OVER and steps < 400:
            p = g.actor()
            obs = encode(g, p, 3)
            obs_list.append((obs.tiles.astype(np.float32),
                             obs.glob.astype(np.float32)))
            acts = g.legal_actions()
            if not acts:
                break
            g.step(acts[int(rng.integers(len(acts)))])
            steps += 1
    return obs_list[:n]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--opset", type=int, default=17)
    args = ap.parse_args()

    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    net = build_net(ck)
    print(f"[net] channels={net.channels} blocks={net.blocks} "
          f"spatial={net.spatial_head} aux={net.aux}")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    tiles = torch.zeros(1, net.in_channels, 3, 9)
    glob = torch.zeros(1, net.global_dim)
    torch.onnx.export(
        net, (tiles, glob), args.out, opset_version=args.opset,
        input_names=["tiles", "glob"],
        output_names=["discard", "self_kong", "respond", "value"],
        dynamic_axes={"tiles": {0: "B"}, "glob": {0: "B"}},
        do_constant_folding=True,
    )
    print(f"[onnx] 导出 -> {args.out} "
          f"({os.path.getsize(args.out) / 2**20:.1f} MiB)")

    # ---- 对拍 ----
    import onnxruntime as ort
    so = ort.SessionOptions()
    so.intra_op_num_threads = 4
    sess = ort.InferenceSession(args.out, so, providers=["CPUExecutionProvider"])

    def torch_out(t, g):
        with torch.no_grad():
            o = net(torch.from_numpy(t), torch.from_numpy(g))
        return {k: v.numpy() for k, v in o.items()}

    cases = []
    rng = np.random.default_rng(0)
    for i in range(8):
        cases.append(("rand", rng.standard_normal((3, net.in_channels, 3, 9)).astype(np.float32),
                      rng.standard_normal((3, net.global_dim)).astype(np.float32)))
    for t, g in real_observations(32):
        cases.append(("real", t.reshape(1, net.in_channels, 3, 9),
                      g.reshape(1, -1)))

    worst = {"rand": 0.0, "real": 0.0}
    for kind, t, g in cases:
        to = torch_out(t, g)
        oo = sess.run(None, {"tiles": t, "glob": g})
        for name, ov in zip(["discard", "self_kong", "respond", "value"], oo):
            d = float(np.abs(to[name] - ov).max())
            worst[kind] = max(worst[kind], d)
            if d > 1e-3 and kind == "real":
                print(f"  MISMATCH {kind} {name} maxdiff={d:.2e}")
    print(f"[parity] torch vs ORT worst maxdiff: real={worst['real']:.3e} "
          f"rand={worst['rand']:.3e} "
          f"({'PASS' if worst['real'] < 1e-3 else 'FAIL'}——判据只看 real）")


if __name__ == "__main__":
    main()
