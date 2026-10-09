"""
监督学习阶段：用启发式教师的对局做行为克隆（BC）。

损失 = 三个动作头的掩码交叉熵（按头加权，缓解 discard/respond 样本量悬殊）
     + λ_v · 终局得分的价值回归（给后面的自对弈 RL 一个好起点）
     - λ_e · 策略熵正则（避免过早收敛成确定性策略）

数据按批惰性从 float16 转 float32，避免把 40 万样本一次性展开成 float32 吃掉几个 G。

输出：``models/bc.pt``（含网络配置与训练历史）、``reports/bc_history.json``。
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from zhuoji.encoder import HEAD_DIMS, dims_for_version  # noqa: E402
from zhuoji.net import (  # noqa: E402
    NetConfig, build_model, encoder_version_from_ckpt, load_into, net_meta,
    warm_start_from_v1)

HEAD_NAMES = ["discard", "self_kong", "respond"]


def tb(arr, idx):
    return torch.from_numpy(np.ascontiguousarray(arr[idx]).astype(np.float32))


def batch_loss(model, tiles, glob, mask, head, act, value, head_weights,
               value_coef, ent_coef):
    out = model(tiles, glob)
    total = tiles.new_zeros(())
    correct = 0
    counted = 0
    ent_total = 0.0

    for hid, name in enumerate(HEAD_NAMES):
        sel = head == hid
        if not bool(sel.any()):
            continue
        dim = HEAD_DIMS[name]
        lg = out[name][sel][:, :dim]
        m = mask[sel][:, :dim].bool()
        lg = lg.masked_fill(~m, -1e9)
        tgt = act[sel]
        total = total + F.cross_entropy(lg, tgt, reduction="sum") * head_weights[hid]
        with torch.no_grad():
            correct += int((lg.argmax(dim=1) == tgt).sum())
            counted += int(sel.sum())
        if ent_coef > 0:
            ent_total += float(_entropy(lg).sum()) * head_weights[hid]

    total = (total - ent_coef * ent_total) / max(1, counted)
    v_pred = out["value"]
    v_loss = F.smooth_l1_loss(v_pred, value)
    total = total + value_coef * v_loss
    return total, {"acc": correct / max(1, counted), "v_loss": float(v_loss.detach())}


def _entropy(logits):
    logp = F.log_softmax(logits, dim=-1)
    p = logp.exp()
    return -(p * logp).nan_to_num(0.0).sum(dim=-1)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(ROOT / "data" / "bc_data.npz"))
    ap.add_argument("--out", default=str(ROOT / "models" / "bc.pt"))
    ap.add_argument("--history", default=str(ROOT / "reports" / "bc_history.json"))
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=1.5e-3)
    ap.add_argument("--channels", type=int, default=32)
    ap.add_argument("--blocks", type=int, default=3)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--value-coef", type=float, default=0.2)
    ap.add_argument("--ent-coef", type=float, default=0.003)
    ap.add_argument("--pass-keep", type=float, default=0.35,
                    help="响应头里'过'动作的保留比例，缓解类别不平衡")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--bench", type=int, default=0, help=">0 时只跑 N 个 batch 测速")
    ap.add_argument("--encoder", choices=["auto", "v1", "v2"], default="auto",
                    help="编码器版本。auto=跟随数据文件里记录的版本（P1 阶段 1）")
    ap.add_argument("--spatial-head", choices=["on", "off"], default="off",
                    help="逐牌空间头（零初始化加法修正，只多 2×(C+1) 个参数）")
    ap.add_argument("--init", default="",
                    help="可选：从某个检查点热启动（P1 用 s11 起，省掉重学共享部分）")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.set_num_threads(args.threads)

    d = np.load(args.data)
    n = len(d["head"])
    print(f"数据集 {args.data}  样本数 {n:,}")
    for hid, h in enumerate(HEAD_NAMES):
        print(f"  头 {h:9s}: {int((d['head'] == hid).sum()):>7d} 条")

    keep = np.ones(n, dtype=bool)
    is_pass = (d["head"] == 2) & (d["act"] == 0)
    rng = np.random.default_rng(args.seed)
    keep[is_pass] = rng.random(int(is_pass.sum())) < args.pass_keep
    all_idx = np.nonzero(keep)[0]
    rng.shuffle(all_idx)
    n_val = max(3000, int(len(all_idx) * 0.05))
    val_idx, tr_idx = all_idx[:n_val], all_idx[n_val:]
    print(f"PASS 下采样后 {len(all_idx):,} 条（训练 {len(tr_idx):,} / 验证 {len(val_idx):,}）")

    counts = np.array([max(1, int((d["head"][tr_idx] == i).sum())) for i in range(3)],
                      dtype=np.float64)
    head_weights = (counts.sum() / (3 * counts)).tolist()
    print("头权重:", [round(w, 3) for w in head_weights])

    # 数据文件自带 encoder_version（旧文件没有 → v1）
    data_enc = 1
    try:
        if "encoder_version" in d.files:
            data_enc = int(np.asarray(d["encoder_version"]).ravel()[0])
    except Exception:
        data_enc = 1
    if args.encoder == "auto":
        args.encoder = f"v{data_enc}"
    enc_version = int(args.encoder[1:])
    n_ch, g_dim = dims_for_version(enc_version)
    t_axis = int(d["tiles"].shape[1])
    assert t_axis == n_ch, (f"数据 tiles 通道数 {t_axis} 与编码器 v{enc_version} "
                            f"的 {n_ch} 不符——请检查 --encoder 或重新生成数据")
    cfg = NetConfig(args.channels, args.blocks, args.hidden,
                    global_dim=g_dim, in_channels=n_ch,
                    spatial_head=(args.spatial_head == "on"))
    model = build_model(cfg)
    if args.init:
        ck0 = torch.load(args.init, map_location="cpu", weights_only=False)
        src_v = encoder_version_from_ckpt(ck0)
        if src_v != enc_version:
            info = warm_start_from_v1(model, ck0)
            print(f"[warm] {args.init}（v{src_v}）→ v{enc_version}："
                  f"原样复制 {info['copied']} 个张量，部分复制 {info['partial']}")
        else:
            miss, unexp = load_into(model, ck0["state_dict"])
            print(f"[warm] 同版本热启动：缺失 {len(miss)} / 意外 {len(unexp)} 键")
    print(f"模型参数量 {model.n_params():,}；编码器 v{enc_version}"
          f"（{n_ch} 通道 / {g_dim} 全局维）")

    def get(idx, sl):
        s = idx[sl]
        return (tb(d["tiles"], s), tb(d["glob"], s), tb(d["mask"], s),
                torch.from_numpy(d["head"][s].astype(np.int64)),
                torch.from_numpy(d["act"][s].astype(np.int64)),
                torch.from_numpy(d["value"][s].astype(np.float32)))

    # 测速
    if args.bench:
        model.train()
        t0 = time.time()
        for i in range(args.bench):
            b = get(tr_idx, slice(i * args.batch, (i + 1) * args.batch))
            loss, st = batch_loss(model, *b, head_weights, args.value_coef, args.ent_coef)
            loss.backward()
            model.zero_grad(set_to_none=True)
        dt = time.time() - t0
        per = dt / args.bench
        print(f"测速: {args.bench} 批 / {dt:.1f}s → {per*1000:.0f} ms/批，"
              f"预计每轮 {per * (len(tr_idx)//args.batch):.0f}s，"
              f"{args.epochs} 轮约 {per * (len(tr_idx)//args.batch) * args.epochs / 60:.1f} 分钟")
        return 0

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    ntr = len(tr_idx)
    nbatch = ntr // args.batch
    history = []
    best_val = 1e18
    t0 = time.time()

    for ep in range(args.epochs):
        model.train()
        perm = rng.permutation(ntr)
        ep_loss = 0.0
        for i in range(nbatch):
            s = tr_idx[perm[i * args.batch:(i + 1) * args.batch]]
            b = (tb(d["tiles"], s), tb(d["glob"], s), tb(d["mask"], s),
                 torch.from_numpy(d["head"][s].astype(np.int64)),
                 torch.from_numpy(d["act"][s].astype(np.int64)),
                 torch.from_numpy(d["value"][s].astype(np.float32)))
            loss, _ = batch_loss(model, *b, head_weights, args.value_coef, args.ent_coef)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            ep_loss += float(loss.detach())
        sched.step()

        model.eval()
        with torch.no_grad():
            va_loss, va_stats = batch_loss(
                model, *get(val_idx, slice(0, min(len(val_idx), 30000))),
                head_weights, args.value_coef, 0.0)
            tr_loss, tr_stats = batch_loss(
                model, *get(tr_idx, slice(0, 30000)),
                head_weights, args.value_coef, 0.0)
        rec = {
            "epoch": ep + 1,
            "train_loss": ep_loss / max(1, nbatch),
            "val_loss": float(va_loss),
            "train_acc": tr_stats["acc"],
            "val_acc": va_stats["acc"],
            "val_value_loss": va_stats["v_loss"],
            "lr": sched.get_last_lr()[0],
            "elapsed": time.time() - t0,
        }
        history.append(rec)
        print(f"  epoch {ep+1:2d}  train_loss={rec['train_loss']:.4f}  "
              f"val_loss={rec['val_loss']:.4f}  train_acc={rec['train_acc']:.3f}  "
              f"val_acc={rec['val_acc']:.3f}  ({rec['elapsed']:.0f}s)", flush=True)

        if float(va_loss) < best_val:
            best_val = float(va_loss)
            Path(args.out).parent.mkdir(parents=True, exist_ok=True)
            torch.save({
                "state_dict": model.state_dict(),
                "config": vars(args),
                "net": net_meta(cfg),
                "encoder_version": enc_version,
                "epoch": ep + 1,
                "val_loss": best_val,
                "head_weights": head_weights,
            }, args.out)

    Path(args.history).parent.mkdir(parents=True, exist_ok=True)
    Path(args.history).write_text(json.dumps(history, indent=2), encoding="utf-8")
    print(f"最优 val_loss={best_val:.4f} -> {args.out}")
    print(f"总耗时 {time.time()-t0:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
