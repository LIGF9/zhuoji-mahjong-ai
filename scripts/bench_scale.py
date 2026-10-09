"""网络规模 / 机器的训练吞吐基准。

用途：换机器前后各跑一次，直接对比「每轮更新耗时」得到真实加速比，
避免凭感觉估算。

口径与 selfplay.py 的实际训练循环一致：
    每轮更新 = epochs_per_iter(2) 个 epoch
               x ceil(samples / batch) 个 mini-batch
    每个 mini-batch = 1 次策略前向 + 1 次 KL 参考前向(no_grad) + 反向 + Adam

用法：
    python scripts/bench_scale.py                     # 默认档位
    python scripts/bench_scale.py --threads 20
    python scripts/bench_scale.py --configs 64,8,320 128,16,512
    python scripts/bench_scale.py --samples 4300 --batch 512 --epochs 2
    python scripts/bench_scale.py --json out.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from zhuoji.encoder import GLOBAL_DIM  # noqa: E402
from zhuoji.net import NetConfig, build_model  # noqa: E402

# (名称, channels, blocks, hidden)。参数量对照：
#   64/8/320   -> 0.77M  当前 big 网
#   96/12/448  -> 2.3M
#   128/16/512 -> 5.1M
#   160/20/512 -> 9.7M   目标「10M 级」
#   192/24/576 -> 16.5M
DEFAULT_CONFIGS = [
    ("big-0.77M", 64, 8, 320),
    ("mid-2.3M", 96, 12, 448),
    ("large-5.1M", 128, 16, 512),
    ("x10M-9.7M", 160, 20, 512),
    ("x15M-16.5M", 192, 24, 576),
]


def resolve_device(name: str) -> torch.device:
    """把 --device 解析成可用的 torch.device，不可用时给出明确提示。"""
    name = name.lower()
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if name == "mps":
        if not (getattr(torch.backends, "mps", None)
                and torch.backends.mps.is_available()):
            raise SystemExit(
                "MPS 不可用：需 Apple Silicon + 支持 MPS 的 PyTorch。\n"
                "检查 `python -c \"import torch;print(torch.backends.mps.is_available())\"`")
        return torch.device("mps")
    if name == "cuda":
        if not torch.cuda.is_available():
            raise SystemExit("CUDA 不可用：未检测到可用的 NVIDIA GPU。")
        return torch.device("cuda")
    return torch.device("cpu")


def bench_one(ch: int, bl: int, hd: int, samples: int, batch: int,
              epochs: int, reps: int, global_dim: int,
              device: torch.device) -> dict:
    model = build_model(NetConfig(ch, bl, hd)).to(device)
    n_params = model.n_params()
    opt = torch.optim.Adam(model.parameters(), lr=8e-6)

    tiles = torch.randn(batch, 15, 27, device=device)
    glob = torch.randn(batch, global_dim, device=device)
    ret = torch.randn(batch, device=device)
    mask28 = torch.ones(batch, 28, dtype=torch.bool, device=device)

    def one_step() -> None:
        out = model(tiles, glob)
        with torch.no_grad():
            ref = model(tiles, glob)
        lp = F.log_softmax(out["discard"].masked_fill(~mask28, -1e9), -1)
        rlp = F.log_softmax(ref["discard"].masked_fill(~mask28, -1e9), -1)
        loss = -lp.mean() + F.smooth_l1_loss(out["value"], ret) \
            + ((lp.exp() * (lp - rlp)).sum(-1)).mean()
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()

    for _ in range(3):          # warmup（MPS 首次编译 kernel 较慢，多跑几次）
        one_step()
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize()

    best = float("inf")
    for _ in range(reps):
        if device.type == "mps":
            torch.mps.synchronize()
        elif device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(3):
            one_step()
        if device.type == "mps":
            torch.mps.synchronize()
        elif device.type == "cuda":
            torch.cuda.synchronize()
        best = min(best, (time.time() - t0) / 3)

    n_batches = -(-samples // batch) * epochs
    return dict(params=n_params, ms_per_batch=best * 1000,
                update_s=best * n_batches, n_batches=n_batches)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", nargs="*", default=None,
                    help="自定义档位，如 64,8,320 160,20,512")
    ap.add_argument("--samples", type=int, default=4300,
                    help="每轮样本数（big 网实测约 4100~4500）")
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--device", default="cpu",
                    help="cpu / mps / cuda / auto（CPU 上同时决定线程数；"
                         "mps 用于 Apple Silicon）")
    ap.add_argument("--collect-s", type=float, default=2.0,
                    help="每轮自对弈采集耗时（big 网 96 局实测约 2s），仅用于合计")
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    device = resolve_device(args.device)
    if device.type == "cpu":
        torch.set_num_threads(args.threads)
    else:
        # CPU 侧的采集仍受线程数影响，这里一并设好
        torch.set_num_threads(args.threads)

    if args.configs:
        parsed = []
        for i, spec in enumerate(args.configs):
            ch, bl, hd = (int(v) for v in spec.split(","))
            parsed.append((f"cfg{i}:{ch}/{bl}/{hd}", ch, bl, hd))
        configs = parsed
    else:
        configs = DEFAULT_CONFIGS

    print(f"torch {torch.__version__}  device={device}  threads={args.threads}  "
          f"global_dim={GLOBAL_DIM}  samples={args.samples} batch={args.batch} "
          f"epochs={args.epochs}")
    print(f"{'配置':<16}{'参数量':>10}{'ms/batch':>11}{'更新/轮(s)':>12}"
          f"{'合计/轮(s)':>12}")
    rows = []
    for name, ch, bl, hd in configs:
        r = bench_one(ch, bl, hd, args.samples, args.batch, args.epochs,
                      args.reps, GLOBAL_DIM, device)
        total = r["update_s"] + args.collect_s
        print(f"{name:<16}{r['params'] / 1e6:>9.2f}M{r['ms_per_batch']:>11.0f}"
              f"{r['update_s']:>12.1f}{total:>12.1f}")
        rows.append(dict(name=name, channels=ch, blocks=bl, hidden=hd,
                         params=r["params"], ms_per_batch=r["ms_per_batch"],
                         update_s=r["update_s"], total_s=total))
    if args.json:
        Path(args.json).write_text(
            json.dumps(dict(device=str(device), threads=args.threads,
                            samples=args.samples, batch=args.batch,
                            epochs=args.epochs, rows=rows),
                       ensure_ascii=False, indent=2),
            encoding="utf-8")
        print(f"-> {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
