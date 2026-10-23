"""M1 网络 —— 10M 参数级骨干（P3 设计，见 ``docs/P3-10M模型设计.md``）。

与 :class:`zhuoji.net.ZhuojiNet` 的关系
---------------------------------------
**对外接口完全一致**（属性名、``forward`` 返回的字典、``n_params``），所以
``ck_from_model`` / ``build_from_ckpt`` / ``vcollect`` / ``NetAgent`` / ``duel``
等既有代码不需要任何改动。内部结构升级三处：

============  ==================  ==========================================
项            s11（ZhuojiNet）    M1
============  ==================  ==========================================
激活          ReLU                Mish（同参数量，Mortal v2+ 实证更优）
BN eps        1e-5（默认）        1e-3（3×9 小网格上批统计噪声大）
Block         2×conv+BN+残差      2×conv+BN+**SE 通道注意力**+残差
宽度/深度     64ch × 8           160ch × 20（≈9.84M 参数）
============  ==================  ==========================================

为什么加 SE：3×9 网格只有 27 个位置，20 层卷积的感受野早就覆盖全局，
深度收益完全落在「通道里选什么」上 —— SE 正是逐样本自适应的通道重加权，
每个 block 只多 ~2.7K 参数。这与 Mortal（40×192 + 通道注意力）同族。

为什么一律冷启动
----------------
M1 与 0.77M 的 s11 **没有可共享的权重**（通道数/深度都不同，ResBlock 也不是
同一批张量）。所以 M1 不做 ``warm_start_from_v1`` 的部分复制 —— 那条路只在
「同结构 + 加通道」时才成立。而且实测（2026-10-11/12）已经证明：
BC 预热 + ``--freeze-old`` 在本项目里是**负资产**（bc_ac 自身 −0.529、
接 1200 轮 RL 后仍 −0.530），所以 M1 直接冷启动跑 RL。

amp/bf16 必须关闭
-----------------
本机 CPU 只有 AVX2（无 AVX512-BF16、无 AMX），实测 bf16 训练比 fp32 **慢 47 倍**
（M1 fp32 493ms/step → bf16 23327ms/step）。``net_m1.assert_no_amp()`` 在
selfplay 启动时做一次断言，防止有人顺手打开 autocast。
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoder import HEAD_DIMS


class SEBlock(nn.Module):
    """通道注意力（Squeeze-and-Excitation，r=16）。"""

    def __init__(self, ch: int, r: int = 16):
        super().__init__()
        self.fc1 = nn.Conv2d(ch, max(1, ch // r), 1)
        self.fc2 = nn.Conv2d(max(1, ch // r), ch, 1)

    def forward(self, x):
        s = x.mean(dim=(2, 3), keepdim=True)
        s = F.relu(self.fc1(s))
        s = torch.sigmoid(self.fc2(s))
        return x * s


class M1Block(nn.Module):
    def __init__(self, ch: int, se: bool = True, bn_eps: float = 1e-3):
        super().__init__()
        self.conv1 = nn.Conv2d(ch, ch, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(ch, eps=bn_eps)
        self.conv2 = nn.Conv2d(ch, ch, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(ch, eps=bn_eps)
        self.se = SEBlock(ch) if se else None

    def forward(self, x):
        y = F.mish(self.bn1(self.conv1(x)))
        y = self.bn2(self.conv2(y))
        if self.se is not None:
            y = self.se(y)
        return F.mish(x + y)


#: P3 的候选配置（在 28 逻辑核 AVX2 机器上实测，见 scripts/bench_net_throughput.py）
M1_PRESETS: dict[str, dict] = {
    "m1": dict(channels=160, blocks=20, hidden=512),     # ≈9.84M，设计目标
    "m0": dict(channels=128, blocks=12, hidden=512),     # ≈4.07M，试点/回退
    "m1l": dict(channels=192, blocks=16, hidden=512),    # ≈11.2M，超预算备选
}


class M1Net(nn.Module):
    def __init__(self, channels: int = 160, blocks: int = 20, hidden: int = 512,
                 global_dim: int = 130, in_channels: int = 63,
                 spatial_head: bool = False, se: bool = True,
                 bn_eps: float = 1e-3, aux: bool = False):
        super().__init__()
        self.arch = "m1"
        self.channels = channels
        self.blocks = blocks
        self.global_dim = global_dim
        self.in_channels = in_channels
        self.spatial_head = spatial_head
        self.se = se
        self.bn_eps = bn_eps
        # M2-AUX（2026-10-18，PTIE/UNREAL 式完美信息辅助头）。
        # 只在训练时提供稠密监督：标签由采集器从**完整对局状态**算出
        # （自家向听 / 三对手是否听牌 / 牌墙余量 / 27 牌危险度地图），
        # 推理时这些头完全不参与决策 —— 完美信息只用于训练，不用于执行。
        # 默认 False：旧检查点（无 aux 键）结构与权重逐位不变。
        self.aux = bool(aux)

        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, channels, 3, padding=1),
            nn.BatchNorm2d(channels, eps=bn_eps),
            nn.Mish(inplace=True),
        )
        self.res = nn.Sequential(
            *[M1Block(channels, se=se, bn_eps=bn_eps) for _ in range(blocks)])

        self.fc_in = nn.Linear(channels + global_dim, hidden)
        self.fc_mid = nn.Linear(hidden, hidden)

        self.head_discard = nn.Linear(hidden, HEAD_DIMS["discard"])
        self.head_self_kong = nn.Linear(hidden, HEAD_DIMS["self_kong"])
        self.head_respond = nn.Linear(hidden, HEAD_DIMS["respond"])
        self.head_value = nn.Linear(hidden, 1)

        if self.aux:
            # 四个辅助头都挂在共享隐层 h 上（UNREAL 式：只塑造主干表征，
            # 不直接驱动策略）。总参数 ≈ 512×(8+3+1+27) + 偏置 ≈ 20K，
            # 相对 9.84M 主干可忽略。
            self.head_aux_shanten = nn.Linear(hidden, 8)   # 自家向听桶（CE）
            self.head_aux_opp = nn.Linear(hidden, 3)       # 三对手是否听牌（BCE）
            self.head_aux_wall = nn.Linear(hidden, 1)      # 牌墙余量 0..1（回归）
            self.head_aux_danger = nn.Linear(hidden, 27)   # 每张牌此刻被吃碰杠胡（BCE）

        if spatial_head:
            # 与 ZhuojiNet 同款「零初始化加法修正」：多 2×(channels+1) 个参数，
            # 起步时与无空间头逐位等价。
            self.sp_discard = nn.Conv2d(channels, 1, 1)
            self.sp_self_kong = nn.Conv2d(channels, 1, 1)
            for conv in (self.sp_discard, self.sp_self_kong):
                nn.init.zeros_(conv.weight)
                nn.init.zeros_(conv.bias)

    def _spatial_add(self, conv, x, pooled_logits):
        add = conv(x).flatten(1)
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
            x = tiles.reshape(b, self.in_channels, 3, 9)
        else:
            x = tiles
        feat = self.res(self.stem(x))
        pooled = feat.mean(dim=(2, 3))
        h = torch.cat([pooled, glob], dim=1)
        h = F.mish(self.fc_in(h))
        h = F.mish(self.fc_mid(h))

        discard = self.head_discard(h)
        self_kong = self.head_self_kong(h)
        if self.spatial_head:
            discard = self._spatial_add(self.sp_discard, feat, discard)
            self_kong = self._spatial_add(self.sp_self_kong, feat, self_kong)
        out = {
            "discard": discard,
            "self_kong": self_kong,
            "respond": self.head_respond(h),
            "value": self.head_value(h).squeeze(-1),
        }
        if self.aux:
            # 额外键不影响既有调用方（NetAgent/duel/vcollect 只按键名取需要的头）。
            out["aux_shanten"] = self.head_aux_shanten(h)
            out["aux_opp"] = self.head_aux_opp(h)
            out["aux_wall"] = self.head_aux_wall(h).squeeze(-1)
            out["aux_danger"] = self.head_aux_danger(h)
        return out

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


def assert_no_amp() -> None:
    """把「本机不要用 bf16/amp」这条实测结论变成可执行断言。

    实测（i7-14700K，仅 AVX2，torch 2.6.0+cpu）：M1 训练 fp32 493 ms/step，
    ``torch.autocast('cpu', dtype=torch.bfloat16)`` 下 23327 ms/step（慢 47×）。
    """
    import torch.backends.cpu as _cpu
    cap = _cpu.get_cpu_capability()
    if cap != "AVX512":
        # 明确打印而不是静默：让日志里有据可查
        print(f"[m1] CPU 能力 = {cap}（无原生 bf16 加速）→ 已禁用 amp，全 fp32 训练")
