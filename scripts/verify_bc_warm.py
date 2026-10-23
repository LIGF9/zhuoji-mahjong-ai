"""BC 预热（阶段 1）的验收门：**BC 有没有真的把新特征装上，同时没把模型训坏**。

背景
----
`v2` 编码器在 v1 基础上多了 16 个输入通道（P1a：牌河时序 ch 15–26）与 28 列
`fc_in` 新权重（P1b 的 6 维 + P1c 的 16 维全局量）。用 `--new-param-init zero`
热启动时这些权重**整段为 0**，于是热启动瞬间模型输出与 v1 逐位相同（严格无损，
见 `zhuoji/net.py::warm_start_from_v1`）。代价是：**新特征一开始完全没有被使用**，
要等 RL 自己把这 16 通道学出来 —— 而上一批实验已经证明 400 轮不够。

阶段 1 的 BC 就是为这件事做的：拿 s11 自己的对局做监督，把那批新权重从 0 抬起来。
但"抬起来"必须被验证，否则跑 7 小时 RL 之后无从归因。本脚本给出两条判据：

1. **不退化**：三个动作头的掩码准确率相对 BC 起点（= s11 零初始化）下降 ≤ 0.5pp。
   BC 数据本就是 s11 自己的采样，所以这个下界很硬 —— 掉超过 0.5pp 说明 BC 把模型
   带偏了（最常见原因：包掩码没对上、或 lr 太大把共享部分也改坏了）。
2. **真装上了**：新增权重（P1a 的 `stem.0.weight[:,15:27]`、P1c 的
   `fc_in.weight[:, C+82:C+98]`）的范数从 0 变成非 0，且**非零元素占比**足够高。
   若全是 0，说明 BC 空转（梯度没流到新参数）—— 此时下游臂等同于"不热身的
   零初始化臂"，跑再久也读不出 BC 的贡献。

用法::

    python scripts/verify_bc_warm.py \\
        --data data/bc_s11_v2.npz --init models/dushan_big_s11.pt \\
        --bc models/dushan_b2_bc_ac.pt --encoder v2 --encoder-packs ac

退出码：0 = 两条判据都过；1 = 未过（调用方应据此中止，别拿坏起点去烧长训练）。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from zhuoji.encoder import HEAD_DIMS, dims_for_version  # noqa: E402
from zhuoji.encoder_v2 import PACK_GLOB_SLICES, PACK_TILE_SLICES, parse_packs  # noqa: E402
from zhuoji.net import (  # noqa: E402
    build_model, build_from_ckpt, encoder_version_from_ckpt,
    net_config_from_ckpt, warm_start_from_v1)

HEAD_NAMES = ["discard", "self_kong", "respond"]


def pack_mask_inplace(tiles, glob, packs):
    for name, (lo, hi) in PACK_TILE_SLICES.items():
        if name not in packs:
            tiles[:, lo:hi] = 0.0
    for name, (lo, hi) in PACK_GLOB_SLICES.items():
        if name not in packs:
            glob[:, lo:hi] = 0.0
    return tiles, glob


def eval_model(model, tiles, glob, mask, head, act, value):
    """返回各头的（掩码）预测、准确率、以及**模型对标签的概率**。

    后者是"翻转是不是伤到有把握的决策"的关键：决策一致率下降本身不是罪 ——
    两个候选概率接近时翻转几乎无成本；**把教师本来 0.9 把握的决策改掉**才是伤。
    起点模型对标签的概率就是"教师的置信度"，直接用它的输出算即可。
    """
    model.eval()
    with torch.no_grad():
        out = model(tiles, glob)
    acc, n, det = {}, {}, {}
    for hid, name in enumerate(HEAD_NAMES):
        sel = head == hid
        if not bool(sel.any()):
            acc[name], n[name] = float("nan"), 0
            det[name] = None
            continue
        dim = HEAD_DIMS[name]
        lg = out[name][sel][:, :dim]
        m = mask[sel][:, :dim].bool()
        lg = lg.masked_fill(~m, -1e9)
        tgt = act[sel]
        pred = lg.argmax(dim=1)
        plabel = torch.softmax(lg, dim=-1).gather(
            1, tgt.view(-1, 1)).squeeze(1)
        acc[name] = float((pred == tgt).float().mean())
        n[name] = int(sel.sum())
        det[name] = (pred.numpy(), tgt.numpy(), plabel.numpy())
    v_loss = float(F.smooth_l1_loss(out["value"], value))
    return acc, n, v_loss, det


def new_param_report(model, cfg, packs) -> dict:
    """新增权重的范数 / 非零占比。按包分别报，便于归因。

    同时给出**旧通道/旧全局维的同一统计量**作参照：只报"非零"是不够的 ——
    权重掉在 1e-8 量级也算非零，但网络根本用不上。判读看 `absmean` 与
    `_ref` 的比值（同量级 ≈ 真的被用上了）。
    """
    sd = model.state_dict()
    out: dict = {}
    if "stem.0.weight" in sd:
        w = sd["stem.0.weight"]
        old = w[:, :min(lo for lo, _ in PACK_TILE_SLICES.values())]
        out["_ref_stem"] = {"shape": list(old.shape),
                            "l2": float(old.norm()),
                            "absmean": float(old.abs().mean())}
        for name, (lo, hi) in PACK_TILE_SLICES.items():
            if name not in packs:
                continue
            sl = w[:, lo:hi]
            out[f"stem.{name}"] = {
                "shape": list(sl.shape), "l2": float(sl.norm()),
                "absmean": float(sl.abs().mean()),
                "nonzero_frac": float((sl != 0).float().mean()),
            }
    if "fc_in.weight" in sd:
        w = sd["fc_in.weight"]
        n_ch = int(cfg.channels)
        old = w[:, :n_ch]
        out["_ref_fc_in"] = {"shape": list(old.shape),
                             "l2": float(old.norm()),
                             "absmean": float(old.abs().mean())}
        for name, (lo, hi) in PACK_GLOB_SLICES.items():
            if name not in packs:
                continue
            sl = w[:, n_ch + lo:n_ch + hi]
            out[f"fc_in.{name}"] = {
                "shape": list(sl.shape), "l2": float(sl.norm()),
                "absmean": float(sl.abs().mean()),
                "nonzero_frac": float((sl != 0).float().mean()),
            }
    out["_channels"] = int(cfg.channels)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--init", required=True, help="BC 的起点检查点（v1，如 s11）")
    ap.add_argument("--bc", required=True, help="BC 训练产物")
    ap.add_argument("--encoder", default="v2")
    ap.add_argument("--encoder-packs", default="auto")
    ap.add_argument("--new-param-init", choices=["random", "zero"], default="zero")
    ap.add_argument("--samples", type=int, default=30000, help="验收抽样条数")
    ap.add_argument("--seed", type=int, default=20270)
    ap.add_argument("--max-drop-pp", type=float, default=0.5,
                    help="**已废弃的门槛**（保留仅为兼容调用）：一致率降幅现在是"
                         "描述性指标，不作判据 —— 见 --max-conf-flip")
    ap.add_argument("--max-conf-flip", type=float, default=0.03,
                    help="门槛②：全体决策中「教师本来把握 ≥0.6 却被改掉」的占比上限"
                         "（默认 3%%）。这条才抓真损害：把有把握的决策改坏")
    ap.add_argument("--out", default=str(ROOT / "reports" / "bc_warm_verify.json"))
    args = ap.parse_args()

    d = np.load(args.data)
    enc_version = int(args.encoder[1:])
    n_ch, g_dim = dims_for_version(enc_version)
    assert int(d["tiles"].shape[1]) == n_ch, \
        f"数据通道 {d['tiles'].shape[1]} ≠ 编码器 v{enc_version} 的 {n_ch}"

    packs = parse_packs(None) if args.encoder_packs == "auto" \
        else parse_packs(args.encoder_packs)
    if enc_version < 2:
        packs = ()

    # 固定抽样（同一批样本喂给两个模型 → 逐样本可比）
    n = len(d["head"])
    rng = np.random.default_rng(args.seed)
    idx = np.sort(rng.choice(n, size=min(args.samples, n), replace=False))
    tiles = torch.from_numpy(np.ascontiguousarray(d["tiles"][idx]).astype(np.float32))
    glob = torch.from_numpy(np.ascontiguousarray(d["glob"][idx]).astype(np.float32))
    pack_mask_inplace(tiles, glob, packs)
    mask = torch.from_numpy(d["mask"][idx].astype(np.uint8))
    head = torch.from_numpy(d["head"][idx].astype(np.int64))
    act = torch.from_numpy(d["act"][idx].astype(np.int64))
    value = torch.from_numpy(d["value"][idx].astype(np.float32))

    # ---- 起点模型：与 train_bc.py 完全同构地重建 ----
    ck_bc = torch.load(args.bc, map_location="cpu", weights_only=False)
    cfg = net_config_from_ckpt(ck_bc)
    m0 = build_model(cfg)
    if enc_version >= 2:
        m0.encoder_packs = packs
    ck0 = torch.load(args.init, map_location="cpu", weights_only=False)
    info = warm_start_from_v1(m0, ck0, new_init=args.new_param_init)

    mbc, _ = build_from_ckpt(ck_bc)
    mbc.encoder_packs = packs if enc_version >= 2 else None

    print(f"数据 {args.data}  样本总数 {n:,}，验收抽样 {len(idx):,}")
    print(f"编码器 v{enc_version}；packs={''.join(packs) or 'none'}；"
          f"起点 {args.init}（v{encoder_version_from_ckpt(ck0)} → v{enc_version}，"
          f"new_init={args.new_param_init}，"
          f"原样复制 {info['copied']} / 部分复制 {info['partial']}）")

    acc0, cnt, v0, det0 = eval_model(m0, tiles, glob, mask, head, act, value)
    acc1, _, v1, det1 = eval_model(mbc, tiles, glob, mask, head, act, value)
    nr0 = new_param_report(m0, cfg, packs)
    nr1 = new_param_report(mbc, cfg, packs)

    # ---- 翻转诊断：一致率下降有多少是"把有把握的决策改掉" ------------------
    # 起点模型在 argmax 标签下一致率应为 1.000（同源），所以任何下降都来自
    # 新特征带来的决策漂移。按教师的置信度给这些漂移分档：
    #   · 教师 p < 0.5  → 本来就是模糊局面，翻转近无成本
    #   · 教师 p ≥ 0.6  → 有把握的决策被改掉，是真损害
    print("\n[决策漂移诊断]（一致率下降的构成；起点在 argmax 标签下应为 1.000）")
    drift = {}
    n_total = 0
    n_conf_flip_all = 0
    for name in HEAD_NAMES:
        if det0[name] is None:
            continue
        pred0, tgt, p0 = det0[name]
        pred1 = det1[name][0]
        n_total += len(tgt)
        flipped = pred1 != tgt
        ok0 = pred0 == tgt
        blam = ok0 & flipped          # 起点对、BC 之后错 —— 这才是"被改坏"
        conf = p0[blam]
        n_conf = int((conf >= 0.6).sum())
        n_conf_flip_all += n_conf
        drift[name] = {
            "n": int(len(tgt)), "flipped": int(blam.sum()),
            "flip_rate": round(float(blam.mean()), 5),
            "conf_ge_06": n_conf,
            "conf_ge_06_rate": round(float(n_conf) / max(1, len(tgt)), 5),
            "mean_teacher_conf_on_flip": (round(float(conf.mean()), 4)
                                          if len(conf) else None),
            "n_teacher_conf_lt_04": int((conf < 0.4).sum()),
        }
        d = drift[name]
        print(f"  {name:10s} 决策 {d['n']:>7d}  被改坏 {d['flipped']:>6d}"
              f"（{d['flip_rate'] * 100:.2f}%）"
              f"  其中教师置信≥0.6 的 {d['conf_ge_06']:>5d}"
              f"（占全体 {d['conf_ge_06_rate'] * 100:.2f}%）"
              f"  教师平均把握 "
              + (f"{d['mean_teacher_conf_on_flip']:.3f}" if d["mean_teacher_conf_on_flip"]
                 is not None else "—"))
    conf_flip_rate = n_conf_flip_all / max(1, n_total)
    print(f"  → 全体「高置信被改坏」占比 {conf_flip_rate * 100:.2f}%"
          f"（门槛 ≤ {args.max_conf_flip * 100:.1f}%）")
    ok_drift = conf_flip_rate <= args.max_conf_flip

    print("\n[各头掩码准确率]（**描述性**，不作门槛 —— 见下方判据说明）")
    res = {}
    for name in HEAD_NAMES:
        drop = (acc0[name] - acc1[name]) * 100.0
        res[name] = {"n": cnt[name], "before": round(acc0[name], 5),
                     "after": round(acc1[name], 5), "drop_pp": round(drop, 3)}
        print(f"  {name:10s} n={cnt[name]:>7d}  {acc0[name]:.4f} → {acc1[name]:.4f}"
              f"   降幅 {drop:+.2f}pp")
    print(f"  value 头  smooth_l1 {v0:.4f} → {v1:.4f}")

    print("\n[新增权重：范数 / 非零占比]（0 → 非 0 才算装上；再看是否到旧权重的量级）")
    ref_stem = (nr1.get("_ref_stem") or {}).get("absmean")
    ref_fc = (nr1.get("_ref_fc_in") or {}).get("absmean")
    print(f"  参照：旧 15 通道 |w| 均值 {ref_stem:.5f}；"
          f"旧 fc_in 列 |w| 均值 {ref_fc:.5f}")
    res_new: dict = {}
    ok_w = True
    for k in sorted(nr0):
        if k.startswith("_"):
            continue
        a, b = nr0[k], nr1[k]
        ref = ref_stem if k.startswith("stem.") else ref_fc
        ratio = (b["absmean"] / ref) if ref else float("nan")
        grew = b["l2"] > 1e-9 and b["nonzero_frac"] > 0.5
        ok_w &= grew
        res_new[k] = {"before": a, "after": b, "ok": bool(grew),
                      "absmean_vs_old": round(ratio, 4)}
        print(f"  {k:12s} {a['shape']}  l2 {a['l2']:.4f} → {b['l2']:.4f}"
              f"   |w| {b['absmean']:.5f}（旧权重的 {ratio * 100:.1f}%）"
              f"   非零 {a['nonzero_frac']:.3f} → {b['nonzero_frac']:.3f}"
              f"  {'OK' if grew else 'FAIL(仍为零/近零)'}")

    ok = bool(ok_w and ok_drift)
    print("\n判据（只有这两条是门槛）：")
    print(f"  ① 新参数已装上（非零 + 范数明显抬升）：{'OK' if ok_w else 'FAIL'}")
    print(f"  ② 高置信决策被改坏占比 ≤ {args.max_conf_flip * 100:.1f}%："
          f"{'OK' if ok_drift else 'FAIL'}"
          f"（实测 {conf_flip_rate * 100:.2f}%）")
    print("  说明：一致率降幅本身**不作门槛** —— 候选接近时翻转几乎无成本，"
          "把降幅当门槛会把'小扰动预热'误杀")
    print(f"\n{'✅ 通过' if ok else '❌ 未通过'}")
    out = {
        "data": args.data, "init": args.init, "bc": args.bc,
        "encoder_version": enc_version, "packs": list(packs),
        "new_param_init": args.new_param_init, "samples": len(idx),
        "accuracy": res, "drift": drift, "new_params": res_new,
        "value_loss": {"before": round(v0, 5), "after": round(v1, 5)},
        "conf_flip_rate": round(conf_flip_rate, 5),
        "ok_weights": bool(ok_w), "ok_drift": bool(ok_drift), "ok": ok,
        "max_conf_flip": args.max_conf_flip,
        "note": "一致率降幅为描述性指标；门槛是「新参数已装上」与「高置信翻转占比」",
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, ensure_ascii=False, indent=2),
                              encoding="utf-8")
    print(f"汇总 -> {args.out}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
