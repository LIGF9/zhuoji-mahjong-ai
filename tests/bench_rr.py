"""循环赛前的分诊脚本：量清楚每种策略组合的单局耗时，再决定跑多少局。

教训复用：上次"卡死"其实是引擎在等外部信号，不是慢。任何要跑几千局的
实验，先在这里把 ms/局 量出来，用实测数字反推预算，而不是拍脑袋。
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from zhuoji.agent import NetAgent  # noqa: E402
from zhuoji.bots import TEACHER_STYLES, HeuristicBot, RandomBot  # noqa: E402
from zhuoji.net import NetConfig, build_model  # noqa: E402
from zhuoji.rules import RulesConfig, ZhuojiGame  # noqa: E402

torch.set_num_threads(4)


def load(path: str):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    nc = ck.get("net") or {"channels": 48, "blocks": 8, "hidden": 256}
    m = build_model(NetConfig(nc["channels"], nc["blocks"], nc["hidden"]))
    m.load_state_dict(ck["state_dict"])
    m.eval()
    return m


bc = load(str(ROOT / "models" / "bc.pt"))
rl = load(str(ROOT / "models" / "rl.pt"))


def mk(name: str):
    if name == "random":
        return RandomBot(seed=1)
    if name == "bc":
        return NetAgent(bc, seed=0, name="bc")
    if name == "rl":
        return NetAgent(rl, seed=0, name="rl")
    return HeuristicBot(seed=1, weights=TEACHER_STYLES[name.split(":")[1]])


def bench(names, n=30, cfg=None):
    cfg = cfg or RulesConfig()
    t0 = time.time()
    for i in range(n):
        bots = [mk(x) for x in names]
        g = ZhuojiGame(cfg, seed=5000 + i)
        g.play(bots)
    dt = time.time() - t0
    print(f"{'/'.join(names):<52} {n:>4}局 {dt:7.2f}s  {dt / n * 1000:7.1f} ms/局")
    return dt / n


if __name__ == "__main__":
    print("=== 单策略速度 ===")
    bench(["random"] * 4, 60)
    bench(["teacher:balanced"] * 4, 60)
    bench(["bc"] * 4, 30)
    bench(["rl"] * 4, 30)
    print("=== 混合速度（每局 2 个网络）===")
    bench(["random", "teacher:aggressive", "rl", "bc"], 30)
    bench(["teacher:balanced", "rl", "bc", "teacher:gambler"], 30)
