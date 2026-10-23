"""网络吞吐微基准：为 10M 模型方案提供真实算力预算。

用法：
  python scripts/bench_net_throughput.py                 # 默认对比 s11 与候选 10M
  python scripts/bench_net_throughput.py --threads 24

测量三件事：
  1) train step（fwd+bwd，batch 512）→ 决定 selfplay update() 耗时
  2) eval forward batch 64        → 决定向量化采集耗时
  3) eval forward batch 1         → 决定 Web 端单步延迟上限
"""
from __future__ import annotations

import argparse
import time

import torch
import torch.nn as nn


class SE(nn.Module):
    def __init__(self, c: int, r: int = 16):
        super().__init__()
        self.fc1 = nn.Conv2d(c, c // r, 1)
        self.fc2 = nn.Conv2d(c // r, c, 1)

    def forward(self, x):
        s = x.mean(dim=(2, 3), keepdim=True)
        s = torch.relu(self.fc1(s))
        s = torch.sigmoid(self.fc2(s))
        return x * s


class Block(nn.Module):
    def __init__(self, c: int, se: bool):
        super().__init__()
        self.c1 = nn.Conv2d(c, c, 3, padding=1)
        self.b1 = nn.BatchNorm2d(c, eps=1e-3)
        self.c2 = nn.Conv2d(c, c, 3, padding=1)
        self.b2 = nn.BatchNorm2d(c, eps=1e-3)
        self.se = SE(c) if se else None
        self.act = nn.Mish()

    def forward(self, x):
        h = self.act(self.b1(self.c1(x)))
        h = self.b2(self.c2(h))
        if self.se is not None:
            h = self.se(h)
        return self.act(x + h)


class BenchNet(nn.Module):
    """与 zhuoji/net.py 同构的候选网：stem + N×Block(SE) + 池化 MLP + 三头。"""

    def __init__(self, cin: int, gd: int, c: int, blocks: int, hid: int, se: bool = True):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(cin, c, 3, padding=1), nn.BatchNorm2d(c, eps=1e-3), nn.Mish())
        self.blocks = nn.ModuleList([Block(c, se) for _ in range(blocks)])
        self.fc_in = nn.Linear(c + gd, hid)
        self.fc_mid = nn.Linear(hid, hid)
        self.act = nn.Mish()
        self.head_discard = nn.Linear(hid, 28)
        self.head_self_kong = nn.Linear(hid, 28)
        self.head_respond = nn.Linear(hid, 4)
        self.head_value = nn.Linear(hid, 1)

    def forward(self, tiles, glob):
        h = self.stem(tiles)
        for b in self.blocks:
            h = b(h)
        h = h.mean(dim=(2, 3))
        h = self.act(self.fc_in(torch.cat([h, glob], dim=1)))
        h = self.act(self.fc_mid(h))
        return self.head_discard(h), self.head_self_kong(h), self.head_respond(h), self.head_value(h)


def n_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())


def bench(model, batches, mode: str, threads: int, warmup: int = 3, rep: int = 8, amp: bool = False):
    torch.set_num_threads(threads)
    model.train(mode == "train")
    opt = torch.optim.AdamW(model.parameters(), lr=1e-6) if mode == "train" else None
    B, cin, gd = batches
    x = torch.randn(B, cin, 3, 9)
    g = torch.randn(B, gd)
    y = torch.randn(B, 1)

    def step():
        if opt is not None:
            opt.zero_grad(set_to_none=True)
        if amp:
            with torch.autocast("cpu", dtype=torch.bfloat16):
                ph, pk, pr, v = model(x, g)
                loss = (ph.sum() + pk.sum() + pr.sum() + v.mean() + (v - y).pow(2).mean()) * 0.0 + v.square().mean()
        else:
            ph, pk, pr, v = model(x, g)
            loss = v.square().mean()
        if mode == "train":
            loss.backward()
            opt.step()

    for _ in range(warmup):
        step()
    t0 = time.perf_counter()
    for _ in range(rep):
        step()
    dt = (time.perf_counter() - t0) / rep
    return dt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--threads", type=int, default=24)
    ap.add_argument("--batch", type=int, default=512)
    args = ap.parse_args()

    # (名称, cin, gd, C, blocks, hidden)
    cfgs = [
        ("s11  64/8/320  (0.79M, 31ch/98g)", 31, 98, 64, 8, 320),
        ("M1   160/20/512 (9.8M, 63ch/130g)", 63, 130, 160, 20, 512),
        ("M0   128/12/512 (3.9M, 63ch/130g)", 63, 130, 128, 12, 512),
    ]
    print(f"threads={args.threads} batch={args.batch}")
    for name, cin, gd, c, b, hid in cfgs:
        m = BenchNet(cin, gd, c, b, hid)
        n = n_params(m)
        t_train = bench(m, (args.batch, cin, gd), "train", args.threads)
        t_eval64 = bench(m, (64, cin, gd), "eval", args.threads)
        t_eval1 = bench(m, (1, cin, gd), "eval", args.threads)
        smp_s = args.batch / t_train
        print(
            f"{name:34s} params={n/1e6:5.2f}M | train b{args.batch}: {t_train*1000:7.1f}ms "
            f"({smp_s:7.1f} sample/s, {1000/smp_s:5.2f}ms/sample) | eval b64: {t_eval64*1000:6.1f}ms "
            f"| eval b1: {t_eval1*1000:5.2f}ms"
        )


if __name__ == "__main__":
    main()
