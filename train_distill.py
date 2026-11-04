"""搜索蒸馏训练：把 n32-ply3 搜索的 max-Q 判断蒸馏进价值头（只训 head_value）。

原理：搜索叶子处 V(s) 由原网络给出，其噪声是 ply3 搜索质量的瓶颈；
用教师（n32-ply3 搜索）在自博弈状态上的 max_a Q(a) 回归微调 head_value，
策略头保持不动（快进/ rollout 行为不变，严格隔离「估值改进」变量）。
"""
from __future__ import annotations

import glob as _glob
import sys

sys.path.insert(0, "/Users/ligf9/WorkBuddy/2026-11-03-18-40-11/zhuoji-mahjong-ai")

import numpy as np
import torch
import torch.nn as nn


def main():
    # ---- 汇总各 shard ----
    files = sorted(_glob.glob("/Users/ligf9/WorkBuddy/2026-11-03-18-40-11/zhuoji-mahjong-ai/runs/distill_w*.npz"))
    assert files, "no distill shards"
    T, G, H, M, QX, QM = [], [], [], [], [], []
    for f in files:
        d = np.load(f)
        T.append(d["tiles"]); G.append(d["glob"])
        QX.append(d["qmax"]); QM.append(d["qmean"])
    tiles = np.concatenate(T); glob = np.concatenate(G)
    qmax = np.concatenate(QX); qmean = np.concatenate(QM)
    n = len(qmax)
    print(f"targets: {n}")
    assert n >= 3000, "too few targets"

    # ---- 构建模型、冻结 trunk ----
    from zhuoji.net import ZhuojiNet
    ck = torch.load("/Users/ligf9/WorkBuddy/2026-11-03-18-40-11/zhuoji-mahjong-ai/models/dushan_master.pt",
                    map_location="cpu", weights_only=False)
    net = ZhuojiNet(channels=64, blocks=8, hidden=320)
    net.load_state_dict(ck["state_dict"])
    net.eval()
    for p_ in net.parameters():
        p_.requires_grad_(False)

    dev = "mps"
    net = net.to(dev)
    x = torch.from_numpy(tiles).view(n, 15, 3, 9).to(dev)
    g_ = torch.from_numpy(glob).to(dev)
    # 目标：max-Q 为主（与搜索叶子「按最优续走」的用法一致），掺 25% mean 防纯乐观
    y = torch.from_numpy(0.75 * qmax + 0.25 * qmean).float().to(dev)

    # ---- 预计算 trunk 特征 h（冻结，只算一次）----
    hs = []
    with torch.no_grad():
        for i in range(0, n, 1024):
            feat = net.res(net.stem(x[i:i + 1024]))
            pooled = feat.mean(dim=(2, 3))
            hh = torch.cat([pooled, g_[i:i + 1024]], dim=1)
            hh = torch.relu(net.fc_in(hh))
            hh = torch.relu(net.fc_mid(hh))
            hs.append(hh)
    H_all = torch.cat(hs)
    print("h features:", tuple(H_all.shape))

    # ---- 只训 head_value ----
    head = net.head_value
    head.weight.requires_grad_(True)
    head.bias.requires_grad_(True)
    opt = torch.optim.Adam(head.parameters(), lr=2e-3, weight_decay=1e-4)
    lossf = nn.MSELoss()

    idx_all = torch.randperm(n)
    n_val = max(500, n // 10)
    val_idx = idx_all[:n_val]
    tr_idx = idx_all[n_val:]

    # 基线：原 head 在 val 上的 MSE（对照改进幅度）
    with torch.no_grad():
        base_mse = float(((head(H_all[val_idx]).squeeze(-1) - y[val_idx]) ** 2).mean())
    print(f"val MSE before: {base_mse:.4f}")

    for ep in range(30):
        perm = tr_idx[torch.randperm(len(tr_idx))]
        tot = 0.0
        for i in range(0, len(perm), 512):
            b = perm[i:i + 512]
            opt.zero_grad()
            pred = head(H_all[b]).squeeze(-1)
            loss = lossf(pred, y[b])
            loss.backward()
            opt.step()
            tot += float(loss) * len(b)
        with torch.no_grad():
            vm = float(((head(H_all[val_idx]).squeeze(-1) - y[val_idx]) ** 2).mean())
        if ep % 5 == 0 or ep == 29:
            print(f"ep{ep}: train {tot/len(perm):.4f}  val {vm:.4f}")

    # ---- 保存完整可加载 checkpoint（master 权重 + 新价值头）----
    sd = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
    out = dict(ck)
    out["state_dict"] = sd
    out["config"] = dict(ck["config"], init="models/dushan_master.pt",
                         out="models/dushan_distill_v1.pt")
    out["iter"] = -1
    out["distill"] = {"targets": int(n), "target": "0.75*qmax+0.25*qmean (n32-ply3 teacher)",
                      "val_mse_before": base_mse,
                      "val_mse_after": vm, "trained": "head_value only, 30 epochs"}
    torch.save(out, "/Users/ligf9/WorkBuddy/2026-11-03-18-40-11/zhuoji-mahjong-ai/models/dushan_distill_v1.pt")
    print("saved models/dushan_distill_v1.pt")


if __name__ == "__main__":
    main()
