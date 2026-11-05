"""zhuoji.reward 单测：势函数塑形恒等式、多局聚合、RVR 基线减除。

跑法： .venv/bin/python tests_reward.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from zhuoji.reward import (  # noqa: E402
    DEALS_PER_EPISODE,
    ExpectedRewardBaseline,
    MultiDealEpisode,
    shaped_step_reward,
)


def test_telescoping_identity():
    """Σ γ^t F_t = −Φ(s_0)（终局 Φ=0），任意 γ、任意 Φ 轨迹。"""
    rng = np.random.default_rng(0)
    for gamma in (1.0, 0.9, 0.99):
        T = 24
        phis = rng.normal(size=T + 1)
        total = 0.0
        for t in range(T):
            terminal = t == T - 1
            total += (gamma ** t) * shaped_step_reward(
                phis[t], phis[t + 1], 0.0, gamma=gamma, terminal=terminal)
        expect = -phis[0]                       # γ^T·Φ(T)=0
        assert abs(total - expect) < 1e-9, (gamma, total, expect)
    print("telescoping identity OK")


def test_multi_deal_aggregation():
    """K 局分差求和 = 终端奖励；满局拒绝、reset 复位。"""
    ep = MultiDealEpisode(deals=DEALS_PER_EPISODE)
    deltas = [3.0, -5.0, 12.0, -2.5]
    for i, d in enumerate(deltas):
        assert not ep.done()
        ep.add_deal(d)
    assert ep.done()
    assert abs(ep.terminal_reward() - sum(deltas)) < 1e-9
    try:
        ep.add_deal(1.0)
        raise AssertionError("应拒绝第 5 局")
    except RuntimeError:
        pass
    ep.reset()
    assert not ep.done() and ep.terminal_reward() == 0.0
    print("multi-deal aggregation OK")


def test_episode_return_composition():
    """整圈回报 = Σ 局分差 + 塑形和(=−Φ(s_0))，逐局终端给 delta、终局 Φ=0。"""
    rng = np.random.default_rng(1)
    phis = rng.normal(size=9)                   # 8 步（4 局 × 2 步）
    deltas = [3.0, -5.0, 12.0, -2.5]
    total = 0.0
    step = 0
    for deal, d in enumerate(deltas):
        for k in range(2):
            terminal = deal == len(deltas) - 1 and k == 1
            deal_over = k == 1
            total += shaped_step_reward(
                phis[step], phis[step + 1], d if deal_over else 0.0,
                gamma=1.0, terminal=terminal)
            step += 1
    expect = sum(deltas) - phis[0]
    assert abs(total - expect) < 1e-9, (total, expect)
    print("episode return composition OK")


def test_constant_potential_is_noop():
    """Φ≡c 且终局置 0 时，塑形只在终局步给 −c，其余步为 0。"""
    c = 2.5
    r_mid = shaped_step_reward(c, c, 0.0, gamma=1.0, terminal=False)
    r_end = shaped_step_reward(c, c, 5.0, gamma=1.0, terminal=True)
    assert abs(r_mid - 0.0) < 1e-9
    assert abs(r_end - (5.0 - c)) < 1e-9
    print("constant potential OK")


def test_rvr_baseline():
    """完美基线 → 残差=纯噪声（std≈σ_noise）；弱基线 → 方差显著低于原始回报。"""
    rng = np.random.default_rng(2)
    n = 4000
    obs = rng.normal(size=(n, 3))
    noise = rng.normal(scale=2.0, size=n)
    true_e = obs[:, 0] * 3.0                    # E[R|s] = 3·o0
    returns = true_e + noise

    perfect = ExpectedRewardBaseline(predict=lambda o: o[:, 0] * 3.0)
    resid = perfect.reduce(returns, obs)
    # 完美基线残差 = 噪声（std≈2.0），远小于回报本身（std≈sqrt(9+4)≈3.6）
    assert abs(resid.std() - 2.0) < 0.3, resid.std()
    assert abs(resid.mean()) < 0.1

    # 弱基线（只捕获一半信号）也应显著降方差
    weak = ExpectedRewardBaseline(predict=lambda o: o[:, 0] * 1.5)
    rw = weak.reduce(returns, obs)
    assert rw.std() < returns.std() * 0.9, (rw.std(), returns.std())
    print("RVR baseline OK")


if __name__ == "__main__":
    test_telescoping_identity()
    test_multi_deal_aggregation()
    test_episode_return_composition()
    test_constant_potential_is_noop()
    test_rvr_baseline()
    print("ALL REWARD TESTS PASSED")
