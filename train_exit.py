"""ExIt 联合蒸馏训练（D1 管线）：教师 Q → M1(v3) 策略头 CE + 价值头 Huber。

损失（14 天计划 §0）：
    L = CE(softmax(Q/τ)) + 0.5 · SmoothL1(V, maxQ/8)
策略与价值联合训练 —— V9 实验证明只蒸价值头无效。

用法：
  .venv/bin/python train_exit.py --data 'runs/exit_r0*/exit_w*.npz' \
      --out models/exit_r01.pt --epochs 30 --window 400000
续训：
  ... --init models/exit_best.pt
"""
from __future__ import annotations

import argparse
import glob as _glob
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import torch.nn as nn

from zhuoji.net import NetConfig, net_meta
from zhuoji.net_m1 import M1Net, assert_no_amp

# 计划常量（改动须同步 docs/14天训练计划.md §0）
STUDENT_CFG = NetConfig(
    channels=160, blocks=20, hidden=512,
    global_dim=130, in_channels=63,
    arch="m1", se=True, bn_eps=1e-3,
    aux=False, dueling=True, se_dual=True, bn_momentum=0.01,
)
TAU = 1.0
VALUE_SCALE = 8.0          # maxQ 量纲（分差）→ 归一到 ~1
VALUE_W = 0.5


def load_window(patterns, window):
    tiles, glob_, head, mask, q = [], [], [], [], []
    files = []
    for pat in patterns:
        files.extend(_glob.glob(pat))
    files = sorted(set(files))
    assert files, f"没有匹配的数据文件: {patterns}"
    for f in files:
        d = np.load(f)
        tiles.append(d["tiles"]); glob_.append(d["glob"])
        head.append(d["head"]); mask.append(d["mask"]); q.append(d["q"])
    tiles = np.concatenate(tiles); glob_ = np.concatenate(glob_)
    head = np.concatenate(head); mask = np.concatenate(mask)
    q = np.concatenate(q)
    n = len(head)
    if n > window:                       # 滑动窗口：保留最近样本（文件按轮次排序）
        tiles, glob_, head, mask, q = (x[-window:] for x in (tiles, glob_, head, mask, q))
        n = window
    print(f"[data] {len(files)} files, {n} samples（窗口 {window}）")
    return tiles, glob_, head, mask, q


def policy_targets(q, mask, tau):
    """softmax(Q/τ) 作用在合法动作上；非法动作概率 0。q 非法槽位为 NaN。"""
    qm = np.where(mask, q, -np.inf)
    qm = np.nan_to_num(qm, nan=-np.inf)
    qm = qm - np.max(np.where(np.isfinite(qm), qm, -np.inf), axis=1, keepdims=True)
    ex = np.exp(np.where(np.isfinite(qm), qm / tau, -np.inf))
    ex = np.nan_to_num(ex, nan=0.0)
    s = ex.sum(axis=1, keepdims=True)
    s[s == 0] = 1.0
    return (ex / s).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", nargs="+", default=["runs/exit_r*/exit_w*.npz"])
    ap.add_argument("--out", default=str(ROOT / "models/exit_best.pt"))
    ap.add_argument("--init", default="", help="续训起点检查点（空 = 冷启动）")
    ap.add_argument("--window", type=int, default=400_000)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=0.01)
    ap.add_argument("--val-frac", type=float, default=0.05)
    ap.add_argument("--patience", type=int, default=3)
    ap.add_argument("--tau", type=float, default=TAU)
    ap.add_argument("--seed", type=int, default=20261106)
    ap.add_argument("--device", default="mps")
    args = ap.parse_args()

    assert_no_amp()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    tiles, glob_, head, mask, q = load_window(args.data, args.window)
    tgt = policy_targets(q, mask, args.tau)
    qmax = np.nanmax(np.where(mask, q, np.nan), axis=1).astype(np.float32)

    n = len(head)
    idx = np.random.permutation(n)
    n_val = max(1, int(n * args.val_frac))
    va, tr = idx[:n_val], idx[n_val:]

    def batch_t(ids):
        ts = (torch.from_numpy(tiles[ids]),
              torch.from_numpy(glob_[ids]),
              torch.from_numpy(head[ids]),
              torch.from_numpy(mask[ids]),
              torch.from_numpy(tgt[ids]),
              torch.from_numpy(qmax[ids]))
        return tuple(t.to(args.device, non_blocking=True) for t in ts)

    net = M1Net(STUDENT_CFG.channels, STUDENT_CFG.blocks, STUDENT_CFG.hidden,
                global_dim=STUDENT_CFG.global_dim,
                in_channels=STUDENT_CFG.in_channels,
                spatial_head=STUDENT_CFG.spatial_head,
                se=STUDENT_CFG.se, bn_eps=STUDENT_CFG.bn_eps,
                aux=STUDENT_CFG.aux, dueling=STUDENT_CFG.dueling,
                se_dual=STUDENT_CFG.se_dual, bn_momentum=STUDENT_CFG.bn_momentum)
    if args.init:
        ck0 = torch.load(str(args.init), map_location="cpu", weights_only=False)
        net.load_state_dict(ck0["state_dict"])
        print(f"[init] 续训自 {args.init}")
    net = net.to(args.device)
    print(f"[net] M1 {net.n_params()/1e6:.2f}M params, device={args.device}")

    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=args.wd)
    huber = nn.SmoothL1Loss(reduction="none")

    def _pad28(t):
        if t.shape[-1] == 28:
            return t
        out = torch.full((t.shape[0], 28), -1e9, device=t.device, dtype=t.dtype)
        out[:, :t.shape[-1]] = t
        return out

    def run_eval(ids):
        net.eval()
        tot, cnt = 0.0, 0
        with torch.no_grad():
            for i in range(0, len(ids), 1024):
                b = batch_t(ids[i:i + 1024])
                loss = _loss(net, b)
                tot += float(loss) * len(b[0]); cnt += len(b[0])
        return tot / max(cnt, 1)

    def _loss(net, b):
        x_t, x_g, hd, mk, tp, qm = b
        out = net(x_t, x_g)
        # 逐样本按 head 取对应头的 logits（respond 只有 4 维 → 补 -1e9 到 28）
        lg = torch.stack((_pad28(out["discard"]), _pad28(out["self_kong"]),
                          _pad28(out["respond"])), dim=1)          # (B,3,28)
        idx = hd.long()
        sel = lg[torch.arange(len(idx)), idx]              # (B, 28)
        logp = torch.log_softmax(sel, dim=-1)
        pol = -(tp * logp).sum(dim=-1)                     # 软标签 CE
        val = huber(out["value"], qm / VALUE_SCALE) * VALUE_W
        return (pol + val).mean()

    best_val = float("inf")
    best_path = Path(args.out)
    best_path.parent.mkdir(parents=True, exist_ok=True)
    bad = 0
    t0 = time.perf_counter()
    for ep in range(1, args.epochs + 1):
        net.train()
        order = np.random.permutation(tr)
        tot, cnt = 0.0, 0
        for i in range(0, len(order), args.batch):
            b = batch_t(order[i:i + args.batch])
            opt.zero_grad(set_to_none=True)
            loss = _loss(net, b)
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            tot += float(loss) * len(b[0]); cnt += len(b[0])
        vl = run_eval(va)
        print(f"[ep {ep:02d}] train {tot/max(cnt,1):.4f}  val {vl:.4f}  "
              f"{time.perf_counter()-t0:.0f}s", flush=True)
        if vl < best_val - 1e-4:
            best_val = vl
            bad = 0
            torch.save({
                "net": net_meta(STUDENT_CFG),
                "state_dict": net.state_dict(),
                "encoder_version": 3,
                "encoder_packs": ["a", "b", "c", "d", "e", "f", "g"],
                "iter": ep,
                "eval": {"val_loss": vl, "n_train": int(len(tr)), "tau": args.tau},
                "distill": {"kind": "exit", "teacher": "n32-ply3"},
            }, str(best_path))
        else:
            bad += 1
            if bad >= args.patience:
                print(f"[early-stop] val 无改进 {bad} 轮，停在 ep {ep}", flush=True)
                break
    print(f"[done] best val {best_val:.4f} → {best_path}", flush=True)


if __name__ == "__main__":
    main()
