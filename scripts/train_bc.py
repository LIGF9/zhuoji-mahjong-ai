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
from zhuoji.encoder_v2 import (  # noqa: E402
    PACK_GLOB_SLICES, PACK_TILE_SLICES, parse_packs)
from zhuoji.net import (  # noqa: E402
    NetConfig, build_model, encoder_packs_from_ckpt, encoder_version_from_ckpt,
    load_into, net_config_from_ckpt, net_meta, warm_start_from_v1)

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
    ap.add_argument("--channels", type=int, default=0,
                    help="0 = 自动：有 --init 时跟随检查点架构，否则 32")
    ap.add_argument("--blocks", type=int, default=0,
                    help="0 = 自动：有 --init 时跟随检查点架构，否则 3")
    ap.add_argument("--hidden", type=int, default=0,
                    help="0 = 自动：有 --init 时跟随检查点架构，否则 256")
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
    ap.add_argument("--encoder-packs", default="auto",
                    help="v2 特征包掩码（a/b/c 的任意组合，''/'none' 表全关）。"
                         "auto=跟随 --init 检查点里记录的掩码（无 --init 则全开）。"
                         "未启用的包**在这里就要把数据整段清零**，"
                         "因为包掩码是编码器行为（apply_pack_mask），不是网络行为")
    ap.add_argument("--new-param-init", choices=["random", "zero"], default="random",
                    help="v1→v2 热启动时，新增的 16 个输入通道与 28 列 fc_in 权重"
                         "怎么初始化。zero = 严格无损（热启动瞬间输出与 v1 逐位相同）")
    ap.add_argument("--freeze-old", action="store_true",
                    help="**只训练新增参数**，共享部分逐位冻结（在 backward 之后把"
                         "冻结切片的梯度清零，并把 weight_decay 关掉）。"
                         "BC 预热的正确形态：目标是'让新特征学会按教师的分布说话'，"
                         "而不是把已经调好的共享策略重训一遍——实测 1 个 epoch "
                         "lr=3e-4 就会把 discard 一致率拉低 1.1pp")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.set_num_threads(args.threads)

    # ---- 架构：有 --init 时**必须**跟随检查点 ---------------------------------
    # 否则会静默地建出另一个尺寸的网络，热启动只装上一部分张量，
    # BC 就在一个"半空的网络"上训练 —— 且日志看起来一切正常。
    # 历史默认值 32/3/256 是启发式 BC 的口径，与主力模型 64/8/320 不同，这个坑真实存在。
    ck_arch = None
    if args.init:
        ck_arch = net_config_from_ckpt(
            torch.load(args.init, map_location="cpu", weights_only=False))
    args.channels = args.channels or (ck_arch.channels if ck_arch else 32)
    args.blocks = args.blocks or (ck_arch.blocks if ck_arch else 3)
    args.hidden = args.hidden or (ck_arch.hidden if ck_arch else 256)
    if ck_arch and (args.channels != ck_arch.channels
                    or args.blocks != ck_arch.blocks
                    or args.hidden != ck_arch.hidden):
        raise SystemExit(
            f"架构与 --init 不符：给定 {args.channels}/{args.blocks}/{args.hidden}，"
            f"检查点是 {ck_arch.channels}/{ck_arch.blocks}/{ck_arch.hidden}。"
            f"要么去掉这三个参数（自动跟随），要么改对。")

    d = np.load(args.data)
    d_files = set(d.files)
    # ---- NPZ 是懒加载且**不缓存**的：每访问一次 d["tiles"] 就重新解压整个成员。
    # 训练循环里每个 batch 都要取一次 → 522k 样本的数据集会让一个 epoch 从
    # 3 分钟变成 20 分钟以上（实测：10 分钟跑不完一个 epoch）。
    # 因此这里一次性搬进内存（tiles f16 约 0.9 GB，合计约 1 GB）。BC 是独立阶段，
    # 此刻没有别的训练在跑，峰值内存可接受。
    data_enc = 1
    try:
        if "encoder_version" in d_files:
            data_enc = int(np.asarray(d["encoder_version"]).ravel()[0])
    except Exception:
        data_enc = 1
    _KEEP_KEYS = ("tiles", "glob", "mask", "head", "act", "value")
    d = {k: np.ascontiguousarray(d[k]) for k in _KEEP_KEYS if k in d_files}
    n = len(d["head"])
    print(f"数据集 {args.data}  样本数 {n:,}（已全部读入内存）")
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

    # 数据文件自带 encoder_version（旧文件没有 → v1）；已在读盘阶段取好
    if args.encoder == "auto":
        args.encoder = f"v{data_enc}"
    enc_version = int(args.encoder[1:])
    n_ch, g_dim = dims_for_version(enc_version)
    t_axis = int(d["tiles"].shape[1])
    assert t_axis == n_ch, (f"数据 tiles 通道数 {t_axis} 与编码器 v{enc_version} "
                            f"的 {n_ch} 不符——请检查 --encoder 或重新生成数据")

    # ---- 特征包掩码 --------------------------------------------------------
    # 包掩码在**编码器**里执行（apply_pack_mask），数据文件里存的是全开版本。
    # 所以要在这里、按同样的切片把未启用的包整段清零，BC 才与 RL 臂看到同一种输入。
    init_packs: tuple[str, ...] | None = None
    if args.init:
        try:
            init_packs = encoder_packs_from_ckpt(
                torch.load(args.init, map_location="cpu", weights_only=False))
        except Exception:
            init_packs = None
    if enc_version < 2:
        packs = None
    elif args.encoder_packs == "auto":
        packs = init_packs if init_packs is not None else parse_packs(None)
    else:
        packs = parse_packs(args.encoder_packs)

    def pack_mask_inplace(tiles, glob):
        """把未启用包的通道/维度清零（张量原地位改，省内存）。"""
        if not packs:
            return tiles, glob
        for name, (lo, hi) in PACK_TILE_SLICES.items():
            if name not in packs:
                tiles[:, lo:hi] = 0.0
        for name, (lo, hi) in PACK_GLOB_SLICES.items():
            if name not in packs:
                glob[:, lo:hi] = 0.0
        return tiles, glob

    cfg = NetConfig(args.channels, args.blocks, args.hidden,
                    global_dim=g_dim, in_channels=n_ch,
                    spatial_head=(args.spatial_head == "on"))
    model = build_model(cfg)
    if enc_version >= 2:
        model.encoder_packs = packs
    if args.init:
        ck0 = torch.load(args.init, map_location="cpu", weights_only=False)
        src_v = encoder_version_from_ckpt(ck0)
        if src_v != enc_version:
            info = warm_start_from_v1(model, ck0, new_init=args.new_param_init)
            print(f"[warm] {args.init}（v{src_v}）→ v{enc_version}："
                  f"原样复制 {info['copied']} 个张量，部分复制 {info['partial']}；"
                  f"新增参数初始化={args.new_param_init}")
        else:
            miss, unexp = load_into(model, ck0["state_dict"])
            print(f"[warm] 同版本热启动：缺失 {len(miss)} / 意外 {len(unexp)} 键")
    print(f"模型参数量 {model.n_params():,}；编码器 v{enc_version}"
          f"（{n_ch} 通道 / {g_dim} 全局维）"
          + (f"；packs={''.join(packs) or 'none'}" if enc_version >= 2 else ""))

    def get(idx, sl):
        s = idx[sl]
        tt = tb(d["tiles"], s)
        gg = tb(d["glob"], s)
        return (*pack_mask_inplace(tt, gg), tb(d["mask"], s),
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

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr,
                            weight_decay=(0.0 if args.freeze_old else 1e-4))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    # ---- 冻结共享部分：只让 v2 新增的切片拿梯度 ----------------------------
    # 张量的哪一段是新增的，由编码器版本决定（v1 的 15 通道 / 70 全局维在前）：
    #   stem.0.weight[:, 15:31]      → v2 新增的 16 个输入通道
    #   fc_in.weight[:, C+70:]       → v2 新增的 28 个全局维
    # 未启用包对应的切片输入恒为 0 ⇒ 梯度本来也是 0，因此这里不需要按包分情况。
    NEW_CH_LO, NEW_CH_HI = 15, 31
    NEW_GLOB_OFF = 70

    def mask_frozen_grads():
        for name, p in model.named_parameters():
            if p.grad is None:
                continue
            if name == "stem.0.weight":
                p.grad[:, :NEW_CH_LO] = 0.0
            elif name == "fc_in.weight":
                # fc_in 的输入 = [通道展平(C 个 × 牌型) | 全局维]，
                # 所以全局维从第 C 列开始（C = 输入宽 − 该版本的全局维数）
                C = p.shape[1] - g_dim
                p.grad[:, :C + NEW_GLOB_OFF] = 0.0
            else:
                p.grad = None       # 优化器会跳过 grad=None 的参数

    if args.freeze_old:
        print(f"[freeze] 只训练新增参数：stem[:,{NEW_CH_LO}:{NEW_CH_HI}] 与 "
              f"fc_in[:, C+{NEW_GLOB_OFF}:{NEW_GLOB_OFF + 28}]，其余冻结"
              f"（weight_decay=0）")

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
            b = get(s, slice(None))     # 走 get()，保证包掩码在训练集上也生效
            loss, _ = batch_loss(model, *b, head_weights, args.value_coef, args.ent_coef)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            if args.freeze_old:
                mask_frozen_grads()     # ← 必须在 backward 之后、step 之前
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
                # 掩码必须落盘：下游 RL 用 --encoder-packs auto 时要能读到
                "encoder_packs": (list(packs) if enc_version >= 2 else None),
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
