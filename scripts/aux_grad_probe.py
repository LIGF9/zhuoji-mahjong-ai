#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""M2-AUX 梯度占比探针（2026-10-18，m2a1 判负后的预登记归因步骤）。

问题：m2a1（aux_coef=0.4）段末 vs 锚 −0.991±0.262，显著劣。
预登记决策树规定「<0 ⇒ 读 aux/pi 梯度占比」。

方法：在段首权重（_ckpts/dushan_m2_a1_it0020，aux 头全新初始化态，
aux 损失 8.95 与训练 iter1 完全同条件）上采集**真实轨迹**（aux=True），
分别对 RL 损失（pi+v−ent+kl）与 aux 损失做一次 backward，
比较两者在**共享主干**（非 head_aux 参数）上的梯度范数。

解读：
* 比值 aux/RL ≫ 1 ⇒ aux 梯度淹没策略梯度（梯度干涉，coef 过大/无预热）；
* 比值 ≪ 1 ⇒ aux 影响可忽略，判负另有原因。
"""
import argparse
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from zhuoji.net import build_model, load_into, net_config_from_ckpt, net_meta  # noqa: E402
from zhuoji.encoder import HEAD_DIMS  # noqa: E402
from zhuoji.vcollect import collect_parallel  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from selfplay import HEAD_NAMES, update  # noqa: E402  (复用 update 的批组装+损失口径)


def grad_norms(model, loss, groups):
    model.zero_grad(set_to_none=True)
    loss.backward(retain_graph=True)
    out = {}
    for name, pat in groups.items():
        s = 0.0
        for n, p in model.named_parameters():
            if p.grad is not None and pat(n):
                s += float(p.grad.norm() ** 2)
        out[name] = s ** 0.5
    return out


class _Args:
    """最小 update() 参数集（与 m2a1 配方一致，aux_coef 由 CLI 传）。"""
    epochs_per_iter = 1
    batch = 512
    adv_mode = "gae"
    adv_norm = "seat"
    gamma = 1.0
    gae_lambda = 0.95
    value_coef = 0.5
    ent_coef = 0.01
    kl_coef = 0.10
    aux_coef = 0.4


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="models/_ckpts/dushan_m2_a1_it0020.pt")
    ap.add_argument("--ckpt2", default="models/_ckpts/dushan_m2_a1_it0240.pt",
                    help="对照：段末权重（aux 已收敛到 ~1.56）")
    ap.add_argument("--games", type=int, default=16)
    ap.add_argument("--aux-coef", type=float, default=0.4)
    ap.add_argument("--workers", type=int, default=4)
    args_cli = ap.parse_args()

    torch.set_num_threads(min(8, os.cpu_count() or 4))

    groups = {
        "trunk": lambda n: not n.startswith("head_aux"),
        "aux_heads": lambda n: n.startswith("head_aux"),
    }

    for tag, path in (("段首(init)", args_cli.ckpt), ("段末(对照)", args_cli.ckpt2)):
        if not os.path.exists(path):
            print(f"[skip] {path} 不存在")
            continue
        ck = torch.load(path, map_location="cpu", weights_only=False)
        cfg = net_config_from_ckpt(ck)
        print(f"\n===== {tag}: {path}（cfg.aux={getattr(cfg, 'aux', 0)}）=====")
        model = build_model(cfg)
        miss, unexp = load_into(model, ck["state_dict"])
        if miss or unexp:
            print(f"[load] 缺失 {len(miss)} / 意外 {len(unexp)}")
        model.train()
        nc = net_meta(cfg)
        enc_ver = int(ck.get("encoder_version", 3) or 3)

        print(f"采集 {args_cli.games} 局真实轨迹（aux=True, workers={args_cli.workers}）…")
        trajs, rewards, dt, aux_out = collect_parallel(
            model.state_dict(), nc, args_cli.games, 2026018, 1.0, 1.0,
            teacher_prob=0.0, workers=args_cli.workers, past_prob=0.0,
            snapshots_sd=[], rules="dushan", want_traj=True,
            encoder_version=enc_ver, aux=True)
        traj = trajs          # collect_parallel 已打平：9 元组列表，与 aux 标签逐行对应
        rew = rewards
        print(f"轨迹 {len(traj)} 行；aux 标签 sh={aux_out['sh'].shape} "
              f"opp={aux_out['opp'].shape} wall={aux_out['wall'].shape} "
              f"danger={aux_out['danger'].shape}")

        a = _Args()
        a.aux_coef = args_cli.aux_coef

        # ---- 组装张量（与 update() 同口径，但拆开两项损失分别 backward） ----
        head_arr = np.array([t[3] for t in traj], dtype=np.int64)
        act_arr = np.array([t[4] for t in traj], dtype=np.int64)
        seat_arr = np.array([t[5] for t in traj], dtype=np.int64)
        gi_arr = np.array([t[6] for t in traj], dtype=np.int64)
        rew_arr = np.asarray(rew, dtype=np.float32)[gi_arr, seat_arr]
        tiles = torch.from_numpy(np.stack([t[0] for t in traj]).astype(np.float32))
        glob = torch.from_numpy(np.stack([t[1] for t in traj]).astype(np.float32))
        mask = torch.from_numpy(np.stack([t[2] for t in traj]).astype(np.float32))
        head = torch.from_numpy(head_arr)
        act = torch.from_numpy(act_arr)
        ret = torch.from_numpy(rew_arr)
        n = tiles.shape[0]

        # 优势：直接复用 update 内部同款 GAE
        from selfplay import _gae_advantages, _normalize_by_seat
        adv_np = _gae_advantages(traj, rew, 1.0, 0.95)
        adv_np = _normalize_by_seat(adv_np, seat_arr)
        adv = torch.from_numpy(np.clip(adv_np, -4.0, 4.0).astype(np.float32))

        sel = torch.arange(n)
        out = model(tiles, glob)
        logp_all = torch.zeros(n)
        ent_all = torch.zeros(n)
        for hid, name in enumerate(HEAD_NAMES):
            sub = (head == hid).nonzero(as_tuple=True)[0]
            if sub.numel() == 0:
                continue
            dim = HEAD_DIMS[name]
            lg = out[name][sub][:, :dim]
            m = mask[sub][:, :dim].bool()
            lg = lg.masked_fill(~m, -1e9)
            lp = F.log_softmax(lg, dim=-1)
            p = lp.exp()
            ent_all[sub] = -(p * lp).nan_to_num(0).sum(-1)
            logp_all[sub] = lp.gather(1, act[sub].unsqueeze(1)).squeeze(1)
        loss_pi = -(logp_all * adv).mean()
        loss_v = F.smooth_l1_loss(out["value"], ret)
        ent = ent_all.mean()
        loss_rl = loss_pi + a.value_coef * loss_v - a.ent_coef * ent
        # klt：无教师（teacher_prob=0，traj[8] 为 None）→ 与训练的 β·klt 项无关

        aux_sh = torch.from_numpy(np.ascontiguousarray(aux_out["sh"])).long()
        aux_opp = torch.from_numpy(aux_out["opp"]).float()
        aux_wall = torch.from_numpy(aux_out["wall"]).float()
        aux_danger = torch.from_numpy(aux_out["danger"]).float()
        _disc = HEAD_NAMES.index("discard")
        l_sh = F.cross_entropy(out["aux_shanten"], aux_sh)
        l_op = F.binary_cross_entropy_with_logits(out["aux_opp"], aux_opp)
        l_wa = F.smooth_l1_loss(out["aux_wall"], aux_wall)
        sub_d = (head == _disc).nonzero(as_tuple=True)[0]
        l_da = F.binary_cross_entropy_with_logits(
            out["aux_danger"][sub_d], aux_danger[sub_d]) if sub_d.numel() else out["aux_danger"].new_zeros(())
        aux_l = l_sh + l_op + l_wa + l_da
        loss_aux = a.aux_coef * aux_l

        print(f"损失值：pi={float(loss_pi):+.4f} v={float(loss_v):.4f} "
              f"ent={float(ent):.4f} | aux 各项 sh={float(l_sh):.3f} "
              f"opp={float(l_op):.3f} wall={float(l_wa):.3f} danger={float(l_da):.3f} "
              f"（Σ={float(aux_l):.3f} × coef{a.aux_coef} = {float(loss_aux):.3f}）")

        g_rl = grad_norms(model, loss_rl, groups)
        g_aux = grad_norms(model, loss_aux, groups)
        print(f"梯度范数  RL(pi+v-ent) : trunk={g_rl['trunk']:.4f}  aux_heads={g_rl['aux_heads']:.6f}")
        print(f"梯度范数  AUX(×{a.aux_coef}) : trunk={g_aux['trunk']:.4f}  aux_heads={g_aux['aux_heads']:.4f}")
        r_tr = g_aux["trunk"] / max(g_rl["trunk"], 1e-12)
        print(f"★ trunk 占比 aux/RL = {r_tr:.2f}  "
              f"（>1 ⇒ aux 主干梯度淹没策略梯度；<0.3 ⇒ 可忽略）")
        # 合成总范数 vs 裁剪阈值 5.0
        g_tot = grad_norms(model, loss_rl + loss_aux, groups)
        tot = (g_tot["trunk"] ** 2 + g_tot["aux_heads"] ** 2) ** 0.5
        print(f"合成总范数 = {tot:.4f}（clip 阈值 5.0 → {'触发裁剪' if tot > 5 else '未裁剪'}）")

        del out, loss_rl, loss_aux
        model.zero_grad(set_to_none=True)

    print("\n完成。")


if __name__ == "__main__":
    main()
