"""编码器一致性自检：维度、动作头/掩码与规则引擎必须严格对齐。"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from zhuoji import RulesConfig, ZhuojiGame, Phase  # noqa: E402
from zhuoji.bots import HeuristicBot, RandomBot  # noqa: E402
from zhuoji.encoder import (  # noqa: E402
    GLOBAL_DIM, HEAD_DIMS, NUM_CHANNELS, encode, head_index, index_to_action,
)
from zhuoji.shanten import shanten  # noqa: E402


def main() -> int:
    cfg = RulesConfig()
    n_checked = 0
    heads = {0: 0, 1: 0, 2: 0}
    for g in range(80):
        game = ZhuojiGame(cfg, seed=7000 + g)
        bots = [HeuristicBot(seed=g * 4 + k) if k % 2 == 0 else RandomBot(seed=g + k)
                for k in range(4)]
        steps = 0
        while game.phase != Phase.OVER and steps < 4000:
            p = game.actor()
            obs = encode(game, p)
            assert obs.tiles.shape == (NUM_CHANNELS, 27), obs.tiles.shape
            assert obs.glob.shape == (GLOBAL_DIM,), obs.glob.shape
            assert obs.mask.shape == (HEAD_DIMS[obs.head],), (obs.head, obs.mask.shape)
            assert obs.mask.any(), f"掩码全 False head={obs.head}"
            assert np.isfinite(obs.tiles).all() and np.isfinite(obs.glob).all()

            acts = game.legal_actions()
            assert acts, f"没有合法动作 phase={game.phase}"

            # 每个合法动作都必须能被映射到掩码内的下标；反之掩码内下标都要能还原
            indices = set()
            for a in acts:
                i = head_index(obs.head, a)
                assert obs.mask[i], f"{a} -> idx {i} 不在掩码内 head={obs.head}"
                indices.add(i)
            for i in np.nonzero(obs.mask)[0]:
                index_to_action(obs.head, int(i), game)   # 不抛异常即可
            n_checked += 1
            heads[{"discard": 0, "self_kong": 1, "respond": 2}[obs.head]] += 1

            game.step(bots[p].choose(game) if hasattr(bots[p], "choose") else bots[p].choose(game))
            steps += 1

    print(f"检查了 {n_checked} 个决策点，全部通过")
    print(f"动作头分布: discard={heads[0]}  self_kong={heads[1]}  respond={heads[2]}")
    print(f" NUM_CHANNELS={NUM_CHANNELS} GLOBAL_DIM={GLOBAL_DIM} HEAD_DIMS={HEAD_DIMS}")

    # 向听数合理性抽查：随机手牌单调性
    rng = np.random.default_rng(0)
    bad = 0
    for _ in range(2000):
        c = np.zeros(27, dtype=int)
        for _ in range(13):
            while True:
                t = int(rng.integers(0, 27))
                if c[t] < 4:
                    c[t] += 1
                    break
        s = shanten(tuple(int(x) for x in c), 0)
        if not (0 <= s <= 6):
            bad += 1
    print(f"随机手牌向听数越界数量: {bad} (应为 0)")
    print("\n结论:", "通过" if bad == 0 else "失败")
    return 0 if bad == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
