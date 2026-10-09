"""
策略/价值网络。

结构
----
牌面特征 ``(15, 27)`` 本来就有结构：27 = 3 花色 × 9 点数，所以 reshape 成
``(15, 3, 9)`` 走 2D 卷积；点数在花色内是循环的（1↔9 相邻），环形卷积收益有限，
这里用普通 3×3 加 padding 即可。

    输入 (B,15,3,9)
      └─ stem conv 15→C
      └─ N × ResBlock(C)
      └─ global average pool → (B,C)
      └─ concat 全局标量 (B,G) → MLP(256→256)
      └─ 四个头：discard(28) / self_kong(28) / respond(4) / value(1)

参数量按 ``C=48, N=8`` 约 60 万，CPU 上也能训。
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoder import GLOBAL_DIM, HEAD_DIMS, NUM_CHANNELS


class ResBlock(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.conv1 = nn.Conv2d(ch, ch, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(ch)
        self.conv2 = nn.Conv2d(ch, ch, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(ch)

    def forward(self, x):
        y = F.relu(self.bn1(self.conv1(x)))
        y = self.bn2(self.conv2(y))
        return F.relu(x + y)


class ZhuojiNet(nn.Module):
    def __init__(self, channels: int = 48, blocks: int = 8,
                 hidden: int = 256, global_dim: int = GLOBAL_DIM):
        super().__init__()
        self.channels = channels
        self.blocks = blocks
        self.global_dim = global_dim

        self.stem = nn.Sequential(
            nn.Conv2d(NUM_CHANNELS, channels, 3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )
        self.res = nn.Sequential(*[ResBlock(channels) for _ in range(blocks)])

        self.fc_in = nn.Linear(channels + global_dim, hidden)
        self.fc_mid = nn.Linear(hidden, hidden)

        self.head_discard = nn.Linear(hidden, HEAD_DIMS["discard"])
        self.head_self_kong = nn.Linear(hidden, HEAD_DIMS["self_kong"])
        self.head_respond = nn.Linear(hidden, HEAD_DIMS["respond"])
        self.head_value = nn.Linear(hidden, 1)

    def forward(self, tiles: torch.Tensor, glob: torch.Tensor):
        """
        tiles: (B, 15, 27) 或 (B, 15, 3, 9)
        glob : (B, G)
        """
        if tiles.dim() == 3:
            b = tiles.shape[0]
            x = tiles.view(b, NUM_CHANNELS, 3, 9)
        else:
            x = tiles
        x = self.res(self.stem(x))
        x = x.mean(dim=(2, 3))                      # 全局平均池化
        h = torch.cat([x, glob], dim=1)
        h = F.relu(self.fc_in(h))
        h = F.relu(self.fc_mid(h))
        return {
            "discard": self.head_discard(h),
            "self_kong": self.head_self_kong(h),
            "respond": self.head_respond(h),
            "value": self.head_value(h).squeeze(-1),
        }

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


@dataclass
class NetConfig:
    channels: int = 48
    blocks: int = 8
    hidden: int = 256


def build_model(cfg: NetConfig | None = None) -> ZhuojiNet:
    cfg = cfg or NetConfig()
    return ZhuojiNet(cfg.channels, cfg.blocks, cfg.hidden)
