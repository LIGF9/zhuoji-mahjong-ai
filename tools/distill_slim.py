"""E-SLIM 学生瘦身：全 28 槽教师 Q 的分级蒸馏（方案 E 落地）。

预登记见 ``docs/14天训练计划.md §6.73``（2026-11-10 13:45 看任何结果前锁死）：
  * 数据：``runs/exit_r*/exit_w*.npz`` 共 1.6M 行教师 Q，全量、不重采；
  * 损失：``CE(softmax(Q/τ)) · w`` + ``0.5 · SmoothL1(V, maxQ/8)``，
    ``w = 1 + β·H(p_T)/log|A_legal|``（contested 加权，β=1.0）；
  * 两级：TA(m1 96×10) 直蒸教师 → S1(s 64×8) 蒸「教师+TA」混合软标签。

为什么按分片流式（而不是一次 concat）
--------------------------------------
1.6M 行的 ``tiles(63,27)`` float32 约 **10.1 GB**，本机 32 GB 内存 + MPS
统一内存，一次载入会和推理/系统抢内存。分片（25k 行 / 183 MB）流式训练
把峰值压在 ~1 GB，且每 epoch 仍能看到全量数据（分片顺序 + 片内顺序都重排）。

为什么不用 AMP/bf16
-------------------
``zhuoji/net_m1.py`` 已记录：本机 CPU 只有 AVX2，bf16 训练比 fp32 慢 47×。
``assert_no_amp()`` 在启动时做一次断言，防止有人顺手打开 autocast。

用法
----
    # 阶段 1：教师助教
    .venv/bin/python tools/distill_slim.py --arch m1 --channels 96 --blocks 10 \
        --hidden 384 --epochs 12 --out models/slim_ta.pt
    # 阶段 2：目标学生（蒸「教师 + TA」混合软标签）
    .venv/bin/python tools/distill_slim.py --arch s --channels 64 --blocks 8 \
        --hidden 320 --epochs 16 --assist models/slim_ta.pt --out models/slim_s.pt
"""
from __future__ import annotations

import argparse
import glob as _glob
import math
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import torch.nn as nn

from zhuoji.net import NetConfig, build_model, net_meta
from zhuoji.net_m1 import assert_no_amp

TAU = 1.0            # 教师温度（与历史 ExIt 一致，§6.73.3 锁死）
VALUE_SCALE = 8.0    # maxQ 量纲（分差）→ ~1
VALUE_W = 0.5
ENT_BETA = 1.0       # contested 加权系数（§6.73.3 锁死）
ENC_PACKS = ["a", "b", "c", "d", "e", "f", "g"]
VAL_FILE = "runs/exit_r48/exit_w3.npz"   # 监控用留出（不参与判定）


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def shard_files(patterns, exclude=VAL_FILE):
    files = []
    for pat in patterns:
        files.extend(_glob.glob(str(ROOT / pat)))
    skip = str((ROOT / exclude).resolve())
    files = [f for f in sorted(set(files)) if str(Path(f).resolve()) != skip]
    assert files, f"没有匹配的数据文件: {patterns}"
    return files


def rows_of(path: str) -> int:
    with np.load(path) as z:
        return int(len(z["head"]))


def teacher_targets(q: np.ndarray, mask: np.ndarray, tau: float):
    """返回 (策略目标 p_T, 熵权重 w, 价值目标 maxQ/8)。

    只在 ``mask`` 合法槽上归一；``q`` 的非法槽可能是 NaN，一律先置 −inf。
    ``w = 1 + β·H/log|A_legal|``，``|A_legal|=1`` 时 ``w=1``。
    """
    qm = np.where(mask, q, -np.inf)
    qm = np.nan_to_num(qm, nan=-np.inf, posinf=1e9, neginf=-np.inf)
    mx = np.max(np.where(np.isfinite(qm), qm, -np.inf), axis=1, keepdims=True)
    ex = np.exp(np.where(np.isfinite(qm), qm / tau, -np.inf) - mx)
    ex = np.nan_to_num(ex, nan=0.0, posinf=0.0)
    s = ex.sum(axis=1, keepdims=True)
    s[s == 0] = 1.0
    p = (ex / s).astype(np.float32)
    with np.errstate(divide="ignore", invalid="ignore"):
        h = -np.where(p > 0, p * np.log(np.where(p > 0, p, 1.0)), 0.0).sum(axis=1)
    n_legal = mask.sum(axis=1).astype(np.float64)
    hmax = np.log(np.maximum(n_legal, 2.0))
    w = 1.0 + ENT_BETA * (h / hmax)
    w[n_legal <= 1] = 1.0
    v = np.nanmax(np.where(mask, q, np.nan), axis=1).astype(np.float32)
    return p, w.astype(np.float32), (v / VALUE_SCALE).astype(np.float32)


def pad28(t: torch.Tensor) -> torch.Tensor:
    if t.shape[-1] == 28:
        return t
    out = torch.full((t.shape[0], 28), -1e9, device=t.device, dtype=t.dtype)
    out[:, :t.shape[-1]] = t
    return out


def head_logits(out: dict, hd: torch.Tensor) -> torch.Tensor:
    lg = torch.stack((pad28(out["discard"]), pad28(out["self_kong"]),
                      pad28(out["respond"])), dim=1)      # (B,3,28)
    idx = hd.long()
    return lg[torch.arange(len(idx)), idx]               # (B,28)


def assist_probs(net, tiles, glob_, head, mask, device, bs=1024):
    """助教在**同一批状态**上的软标签（温度 1.0，只对合法槽归一）。"""
    net.eval()
    outs = []
    with torch.no_grad():
        for i in range(0, len(head), bs):
            xt = torch.from_numpy(tiles[i:i + bs]).to(device)
            xg = torch.from_numpy(glob_[i:i + bs]).to(device)
            hd = torch.from_numpy(head[i:i + bs]).to(device)
            lg = head_logits(net(xt, xg), hd)
            mk = torch.from_numpy(mask[i:i + bs]).to(device)
            lg = lg.masked_fill(~mk, -1e9)
            outs.append(torch.softmax(lg, dim=-1).float().cpu().numpy())
    return np.concatenate(outs).astype(np.float32)


def build(cfg: NetConfig):
    net = build_model(cfg)
    return net


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", nargs="+", default=["runs/exit_r*/exit_w*.npz"])
    ap.add_argument("--out", default=str(ROOT / "models/slim_s.pt"))
    ap.add_argument("--arch", choices=["s", "m1"], default="s")
    ap.add_argument("--channels", type=int, default=64)
    ap.add_argument("--blocks", type=int, default=8)
    ap.add_argument("--hidden", type=int, default=320)
    ap.add_argument("--epochs", type=int, default=16)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=0.01)
    ap.add_argument("--patience", type=int, default=3)
    ap.add_argument("--tau", type=float, default=TAU)
    ap.add_argument("--assist", default="", help="助教 .pt（阶段 2 用；混合软标签 0.5/0.5）")
    ap.add_argument("--assist-w", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=20261110)
    ap.add_argument("--device", default="mps")
    ap.add_argument("--limit-shards", type=int, default=0, help="冒烟用：只跑前 N 片")
    args = ap.parse_args()

    assert_no_amp()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    cfg = NetConfig(
        channels=args.channels, blocks=args.blocks, hidden=args.hidden,
        global_dim=130, in_channels=63, spatial_head=False,
        arch=args.arch,
        se=True if args.arch == "m1" else False,
        bn_eps=1e-3,
        aux=0,
        dueling=(args.arch == "m1"),
        se_dual=(args.arch == "m1"),
        bn_momentum=0.01 if args.arch == "m1" else 0.1,
    )
    net = build(cfg).to(args.device)
    n_par = sum(p.numel() for p in net.parameters())
    log(f"net arch={args.arch} {args.channels}x{args.blocks} h{args.hidden} "
        f"→ {n_par/1e6:.3f}M params, device={args.device}")

    assist = None
    if args.assist:
        from zhuoji.net import build_from_ckpt
        ck = torch.load(str(ROOT / args.assist), map_location="cpu", weights_only=False)
        assist, _v = build_from_ckpt(ck)
        assist = assist.to(args.device).eval()
        log(f"assist 载入 {args.assist}（混合权重 {args.assist_w:.2f}）")

    files = shard_files(args.data)
    if args.limit_shards:
        files = files[:args.limit_shards]
    log(f"[data] {len(files)} 个分片（留出 {VAL_FILE}）")

    # 留出片常驻内存（25k 行 ≈ 190 MB），用于早停
    dv = np.load(ROOT / VAL_FILE)
    v_tiles, v_glob = dv["tiles"], dv["glob"]
    v_head, v_mask = dv["head"], dv["mask"]
    v_p, v_w, v_v = teacher_targets(dv["q"], v_mask, args.tau)
    if assist is not None:
        va = assist_probs(assist, v_tiles, v_glob, v_head, v_mask, args.device)
        v_p = (1 - args.assist_w) * v_p + args.assist_w * va
        v_p /= v_p.sum(axis=1, keepdims=True)
    log(f"[val] {len(v_head)} 行；目标熵均值 {-np.nansum(v_p*np.log(np.maximum(v_p,1e-12)),axis=1).mean():.3f}")

    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=args.wd)
    # 步数按分片实际行数估（只读一页 npz 头，代价可忽略）；cosine 到 lr*0.05
    n_rows = rows_of(files[0]) * len(files)
    steps_per_epoch = max(1, n_rows // args.batch)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=args.epochs * steps_per_epoch, eta_min=args.lr * 0.05)
    log(f"[plan] 约 {n_rows} 行/epoch，{steps_per_epoch} step/epoch")
    huber = nn.SmoothL1Loss(reduction="none")

    def loss_fn(b):
        x_t, x_g, hd, mk, tp, tw, tv = b
        out = net(x_t, x_g)
        sel = head_logits(out, hd)
        logp = torch.log_softmax(sel, dim=-1)
        pol = -(tp * logp).sum(dim=-1)
        pol = (pol * tw).sum() / tw.sum().clamp_min(1e-6)
        val = huber(out["value"], tv) * VALUE_W
        return pol + val.mean()

    def eval_val():
        net.eval()
        tot, cnt = 0.0, 0
        with torch.no_grad():
            for i in range(0, len(v_head), 512):
                b = (torch.from_numpy(v_tiles[i:i + 512]).to(args.device),
                     torch.from_numpy(v_glob[i:i + 512]).to(args.device),
                     torch.from_numpy(v_head[i:i + 512]).to(args.device),
                     torch.from_numpy(v_mask[i:i + 512]).to(args.device),
                     torch.from_numpy(v_p[i:i + 512]).to(args.device),
                     torch.from_numpy(v_w[i:i + 512]).to(args.device),
                     torch.from_numpy(v_v[i:i + 512]).to(args.device))
                tot += float(loss_fn(b)) * len(v_head[i:i + 512])
                cnt += len(v_head[i:i + 512])
        net.train()
        return tot / max(cnt, 1)

    best = float("inf")
    best_path = Path(args.out)
    best_path.parent.mkdir(parents=True, exist_ok=True)
    bad = 0
    t0 = time.perf_counter()
    for ep in range(1, args.epochs + 1):
        net.train()
        order = np.random.permutation(len(files))
        tot = cnt = 0
        t_ep = time.perf_counter()
        for fi in order:
            d = np.load(files[fi])
            tiles, glob_, head, mask = d["tiles"], d["glob"], d["head"], d["mask"]
            p, w, v = teacher_targets(d["q"], mask, args.tau)
            if assist is not None:
                pa = assist_probs(assist, tiles, glob_, head, mask, args.device)
                p = (1 - args.assist_w) * p + args.assist_w * pa
                p /= p.sum(axis=1, keepdims=True)
            idx = np.random.permutation(len(head))
            for i in range(0, len(idx), args.batch):
                bi = idx[i:i + args.batch]
                b = (torch.from_numpy(tiles[bi]).to(args.device),
                     torch.from_numpy(glob_[bi]).to(args.device),
                     torch.from_numpy(head[bi]).to(args.device),
                     torch.from_numpy(mask[bi]).to(args.device),
                     torch.from_numpy(p[bi]).to(args.device),
                     torch.from_numpy(w[bi]).to(args.device),
                     torch.from_numpy(v[bi]).to(args.device))
                opt.zero_grad(set_to_none=True)
                loss = loss_fn(b)
                loss.backward()
                nn.utils.clip_grad_norm_(net.parameters(), 1.0)
                opt.step()
                sched.step()
                tot += loss.detach().item() * len(bi)
                cnt += len(bi)
            del d, tiles, glob_, head, mask, p, w, v
        vl = eval_val()
        log(f"[ep {ep:02d}] train {tot/max(cnt,1):.4f} val {vl:.4f} "
            f"ep {time.perf_counter()-t_ep:.0f}s total {time.perf_counter()-t0:.0f}s")
        if vl < best - 1e-4:
            best, bad = vl, 0
            torch.save({
                "net": net_meta(cfg),
                "state_dict": net.state_dict(),
                "encoder_version": 3,
                "encoder_packs": list(ENC_PACKS),
                "iter": ep,
                "eval": {"val_loss": vl, "n_shards": len(files), "tau": args.tau,
                         "ent_beta": ENT_BETA, "assist_w": args.assist_w},
                "distill": {"kind": "exit-slim", "teacher": "champ_exit_p48",
                            "assist": args.assist or None,
                            "params": n_par, "epochs": args.epochs},
            }, str(best_path))
            log(f"  ↑ 新最优 {vl:.4f} → {best_path.name}")
        else:
            bad += 1
            if bad >= args.patience:
                log(f"[early-stop] val 无改进 {bad} 轮，停在 ep {ep}")
                break
    log(f"[done] best val {best:.4f} → {best_path}  "
        f"{n_par/1e6:.3f}M params  {(time.perf_counter()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
