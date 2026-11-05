"""RL 奖励口径（docs/14天训练计划.md §0 / D10-12，外部建议 #3，RVR=CoG'22）。

三层设计，按需叠加：

1. **多局分差终端奖励**：episode = K 局一圈（轮庄），终端奖励 = Σ delta_k
   （本座位逐局净分差之和）。降单局随机性、与评测「分/场」口径对齐。
2. **势函数塑形**（Ng et al. 1999，policy-invariant）：
   ``F(s,a,s') = γ·Φ(s') − Φ(s)``，``Φ(s) = PHI_SCALE·V_θ(s)``（V_θ 预测
   maxQ/8，乘回 8 与分差同量纲），**终局 Φ≡0**。
   性质：``Σ γ^t·F_t = −Φ(s_0)``（与轨迹无关的常数偏移，不改最优策略）。
3. **RVR 升级位**（Reward Variance Reduction，Li et al., CoG 2022）：
   期望奖励网络 ``b(s)≈E[R|s]`` 做基线减除 ``R − b(s)``，显著降回报方差。
   `ExpectedRewardBaseline` 只定接口与减除；网络训练见 D10-12 实现。

用法（trainer 侧，逐状态）：
    phi_t   = 8.0 * v_pred(s_t)      # 势能（分差单位）
    phi_tp1 = 8.0 * v_pred(s_{t+1})
    r = shaped_step_reward(phi_t, phi_tp1,
                           deal_delta=(delta if deal_over else 0.0),
                           terminal=episode_over)
"""
from __future__ import annotations

from dataclasses import dataclass, field

# 与 train_exit.VALUE_SCALE 同量纲（maxQ 归一用 /8，势能乘回 8）
PHI_SCALE = 8.0
GAMMA = 1.0
DEALS_PER_EPISODE = 4


def shaped_step_reward(phi_t: float, phi_tp1: float, deal_delta: float = 0.0,
                       *, gamma: float = GAMMA, terminal: bool = False) -> float:
    """单步塑形奖励 = 局终端分差增量 + γΦ(s') − Φ(s)。

    ``terminal=True``（episode 终局）时强制 Φ(s')=0，保证 Σγ^t F_t = −Φ(s_0)。
    ``deal_delta`` 为本步结束的那一局净分差（非终局步为 0）。
    """
    phi_next = 0.0 if terminal else float(phi_tp1)
    return float(deal_delta) + gamma * phi_next - float(phi_t)


@dataclass
class MultiDealEpisode:
    """一圈 K 局的分差聚合（多局分差终端奖励）。"""

    deals: int = DEALS_PER_EPISODE
    deltas: list[float] = field(default_factory=list)

    def add_deal(self, delta: float) -> None:
        if self.done():
            raise RuntimeError(f"episode 已满 {self.deals} 局，先 reset()")
        self.deltas.append(float(delta))

    def done(self) -> bool:
        return len(self.deltas) >= self.deals

    def terminal_reward(self) -> float:
        """episode 终端奖励 = Σ 逐局净分差。"""
        return float(sum(self.deltas))

    def reset(self) -> None:
        self.deltas.clear()


@dataclass
class ExpectedRewardBaseline:
    """RVR 期望奖励基线：``advantage = R − b(s)``。

    ``predict``：callable(obs_batch) -> array([B])，返回 b(s)≈E[R|s]。
    训练器负责拟合（D10-12）；此处只做减除，保证接口稳定。
    """

    predict: callable

    def reduce(self, returns, obs):
        """按位减除基线；returns 与 predict(obs) 广播对齐。"""
        base = self.predict(obs)
        return returns - base
