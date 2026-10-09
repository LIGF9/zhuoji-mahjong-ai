"""
策略/价值网络。

结构
----
牌面特征 ``(C, 27)`` 本来就有结构：27 = 3 花色 × 9 点数，所以 reshape 成
``(C, 3, 9)`` 走 2D 卷积；点数在花色内是循环的（1↔9 相邻），环形卷积收益有限，
这里用普通 3×3 加 padding 即可。

    输入 (B,C,3,9)
      └─ stem conv C→channels
      └─ N × ResBlock(channels)
      ├─ 全局平均池化 → (B,channels) ┐
      │                              ├─ concat 全局标量 (B,G) → MLP
      └─ (可选) 空间头 1×1 卷积 → 27 个逐牌 logit
      └─ 头：discard(28) / self_kong(28) / respond(4) / value(1)

两处与"旧版"的关键差异
----------------------
1. **输入维度由配置决定**（``NetConfig.in_channels`` / ``global_dim``）。
   旧检查点（v1：15 通道 / 70 全局维）不带这两个字段，``net_config_from_ckpt``
   会补默认值，所以 s1–s11 仍能原样加载。
2. **空间头（``spatial_head``）是"零初始化的加法修正"**，不是替换：
   ``discard_logits[0:27] = head_discard(h)[0:27] + conv1x1(x).flatten()``。
   这样

   * 参数只多 ``2×(channels+1)``（64 通道时 130 个）；
   * 权重零初始化 ⇒ **加载旧权重后输出与旧版逐位相同**，是严格的热启动，
     不存在"改了头就得重训"的问题；
   * 它修的是旧结构的一个真实缺陷：``x.mean(dim=(2,3))`` 把 3×9 的牌面空间
     整个池化掉，之后三个头都是 ``Linear(hidden → 牌型数)``，28 个弃牌 logit
     **共享同一个输入向量 h**（``logit_t = w_t·h + b_t``），"这张是孤张所以该打
     它"只能靠"用池化通道近似重建位置"。空间头让逐牌定位走直路。
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
                 hidden: int = 256, global_dim: int = GLOBAL_DIM,
                 in_channels: int = NUM_CHANNELS, spatial_head: bool = False):
        super().__init__()
        self.channels = channels
        self.blocks = blocks
        self.global_dim = global_dim
        self.in_channels = in_channels
        self.spatial_head = spatial_head

        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, channels, 3, padding=1),
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

        if spatial_head:
            # 逐牌 logit：1×1 卷积把 (B,C,3,9) 直接投成 (B,1,3,9) → 27 个位置各一个
            # logit。零初始化 ⇒ 起步时与"无空间头"完全等价（严格热启动）。
            self.sp_discard = nn.Conv2d(channels, 1, 1)
            self.sp_self_kong = nn.Conv2d(channels, 1, 1)
            for conv in (self.sp_discard, self.sp_self_kong):
                nn.init.zeros_(conv.weight)
                nn.init.zeros_(conv.bias)

    def _spatial_add(self, conv, x, pooled_logits):
        """把空间头加到最后一张牌（下标 27 = 自摸胡 / 过）之前的 27 个 logit 上。"""
        add = conv(x).flatten(1)                       # (B, 27)
        out = pooled_logits.clone()
        out[:, :27] = out[:, :27] + add
        return out

    def forward(self, tiles: torch.Tensor, glob: torch.Tensor):
        """
        tiles: (B, C, 27) 或 (B, C, 3, 9)
        glob : (B, G)
        """
        if tiles.dim() == 3:
            b = tiles.shape[0]
            x = tiles.view(b, self.in_channels, 3, 9)
        else:
            x = tiles
        feat = self.res(self.stem(x))                   # (B,C,3,9)
        pooled = feat.mean(dim=(2, 3))                  # 全局平均池化
        h = torch.cat([pooled, glob], dim=1)
        h = F.relu(self.fc_in(h))
        h = F.relu(self.fc_mid(h))

        discard = self.head_discard(h)
        self_kong = self.head_self_kong(h)
        if self.spatial_head:
            discard = self._spatial_add(self.sp_discard, feat, discard)
            self_kong = self._spatial_add(self.sp_self_kong, feat, self_kong)
        return {
            "discard": discard,
            "self_kong": self_kong,
            "respond": self.head_respond(h),
            "value": self.head_value(h).squeeze(-1),
        }

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


@dataclass
class NetConfig:
    """网络配置。

    默认值 = **v1 的历史配置**，因此 ``NetConfig()`` 与旧代码逐位等价，
    旧检查点（``net`` 字典里只有 channels/blocks/hidden）也能原样加载。
    """
    channels: int = 48
    blocks: int = 8
    hidden: int = 256
    global_dim: int = GLOBAL_DIM
    in_channels: int = NUM_CHANNELS
    spatial_head: bool = False


def net_config_from_ckpt(ck: dict) -> NetConfig:
    """从检查点的 ``net`` 字段还原 :class:`NetConfig`（旧文件自动补默认值）。"""
    d = ck.get("net") or {}
    if not isinstance(d, dict):
        d = {}
    return NetConfig(
        channels=int(d.get("channels", 48)),
        blocks=int(d.get("blocks", 8)),
        hidden=int(d.get("hidden", 256)),
        global_dim=int(d.get("global_dim", GLOBAL_DIM)),
        in_channels=int(d.get("in_channels", NUM_CHANNELS)),
        spatial_head=bool(d.get("spatial_head", False)),
    )


def net_meta(cfg: NetConfig) -> dict:
    """写进检查点 ``net`` 字段的元数据（加载时用 :func:`net_config_from_ckpt` 还原）。"""
    return {
        "channels": cfg.channels,
        "blocks": cfg.blocks,
        "hidden": cfg.hidden,
        "global_dim": cfg.global_dim,
        "in_channels": cfg.in_channels,
        "spatial_head": cfg.spatial_head,
    }


def encoder_version_from_ckpt(ck: dict) -> int:
    """检查点用的编码器版本（老文件没有这个字段 → 一律 v1）。"""
    return int(ck.get("encoder_version", 1))


def build_from_ckpt(ck: dict, map_location="cpu"):
    """按检查点自带的元数据构建模型并装入权重，返回 ``(model, encoder_version)``。

    这是加载旧模型的**唯一推荐入口**：v1/v2 的输入维度、有无空间头、
    编码器版本全部由检查点自己声明，调用方不需要知道这些细节。
    返回的模型上会挂一个 ``encoder_version`` 属性，于是
    ``NetAgent(model, ...)`` 不用额外传参就能选对编码器。
    """
    cfg = net_config_from_ckpt(ck)
    model = build_model(cfg)
    model.load_state_dict(ck["state_dict"])
    model.eval()
    version = encoder_version_from_ckpt(ck)
    model.encoder_version = version
    return model, version


def load_model_file(path, map_location="cpu"):
    """从 ``.pt`` 路径加载，返回 ``(model, ckpt_dict)``。"""
    import torch
    ck = torch.load(path, map_location=map_location, weights_only=False)
    model, _v = build_from_ckpt(ck, map_location=map_location)
    return model, ck


def build_model(cfg: NetConfig | None = None) -> ZhuojiNet:
    cfg = cfg or NetConfig()
    return ZhuojiNet(cfg.channels, cfg.blocks, cfg.hidden,
                     global_dim=cfg.global_dim, in_channels=cfg.in_channels,
                     spatial_head=cfg.spatial_head)


def load_into(model: ZhuojiNet, state_dict: dict) -> tuple[list[str], list[str]]:
    """把（可能少几个新参数的）旧权重装进新结构，返回 (缺失键, 意外键)。

    空间头是零初始化加法项，所以"缺失键 = 两个 1×1 卷积"时，
    装载后的前向输出与旧模型**逐位相同**——这正是它可以直接热启动的原因。

    只忽略「缺失/多余」的键；**尺寸对不上的键会显式报错**，因为那说明输入
    结构真的变了（15 通道 → 31 通道），应当走 :func:`warm_start_from_v1`。
    """
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    return list(missing), list(unexpected)


def warm_start_from_v1(model: ZhuojiNet, ck: dict) -> dict:
    """把 v1 检查点的权重**部分**搬进 v2 结构（其余部分保持随机初始化）。

    只做两件形状不匹配的搬运，别的一律原样加载：

    * ``stem.weight``：``(C,15,3,3) → (C,31,3,3)``，前 15 个输入通道**逐位复制**
      （v1 的 15 个特征通道在 v2 里就是前 15 个，顺序完全一致），
      新增的 16 个通道保留初始化值；
    * ``fc_in.weight``：``(H, C+70) → (H, C+98)``，前 70 列逐位复制。

    效果：新架构不再必须「从零训练 BC」——共享部分直接继承 s11 的知识，
    新特征对应的权重从零开始学。返回一份说明装载情况的字典，便于写进日志。
    """
    src = ck["state_dict"]
    dst = model.state_dict()
    copied: list[str] = []
    skipped: list[str] = []
    new_only: list[str] = []

    for k, tgt in dst.items():
        s = src.get(k)
        if s is None:
            new_only.append(k)
            continue
        if tuple(s.shape) == tuple(tgt.shape):
            tgt.copy_(s)
            copied.append(k)
        elif k == "stem.0.weight" and s.shape[0] == tgt.shape[0] and tgt.dim() == 4:
            n = min(s.shape[1], tgt.shape[1])
            tgt[:, :n] = s[:, :n]
            copied.append(f"{k}(前{n}通道)")
            skipped.append(f"{k}(新增 {tgt.shape[1] - n} 通道随机初始化)")
        elif k == "fc_in.weight" and s.shape[0] == tgt.shape[0] and tgt.dim() == 2:
            n = min(s.shape[1], tgt.shape[1])
            tgt[:, :n] = s[:, :n]
            copied.append(f"{k}(前{n}列)")
            skipped.append(f"{k}(新增 {tgt.shape[1] - n} 列随机初始化)")
        else:
            skipped.append(f"{k}（形状 {tuple(s.shape)} → {tuple(tgt.shape)}，丢弃）")

    return {"copied": len(copied), "partial": skipped, "new_only": new_only,
            "detail": copied}
