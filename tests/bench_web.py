"""
网页对局服务端的性能分诊：把"规则引擎本身"和"模型推理"的耗时分开测。

跑法::

    python tests/bench_web.py

输出直接写文件（不靠 stdout，避免被管道缓冲/独占锁干扰）。
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from zhuoji.agent import NetAgent  # noqa: E402
from zhuoji.bots import HeuristicBot, RandomBot, TEACHER_STYLES  # noqa: E402
from zhuoji.net import NetConfig, build_model  # noqa: E402
from zhuoji.rules import Phase, RulesConfig, ZhuojiGame  # noqa: E402

OUT = ROOT / "reports" / "bench_web.txt"
LINES: list[str] = []


def say(s: str) -> None:
    LINES.append(s)
    OUT.write_text("\n".join(LINES) + "\n", encoding="utf-8")


def run_hand(bots: list, seed: int, cfg: RulesConfig) -> tuple[float, int, str]:
    game = ZhuojiGame(cfg, seed=seed)
    n = 0
    t0 = time.perf_counter()
    while game.phase != Phase.OVER and n < 3000:
        game.step(bots[game.actor()].choose(game))
        n += 1
    dt = time.perf_counter() - t0
    return dt, n, (game.result or {}).get("type", "?")


def main() -> int:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    LINES.clear()
    cfg = RulesConfig()
    torch.set_num_threads(4)
    say(f"=== bench {time.strftime('%Y-%m-%d %H:%M:%S')} torch={torch.__version__} ===")

    # 1) 纯规则引擎：4 个随机 Bot
    t0 = time.perf_counter()
    hs = [run_hand([RandomBot(seed=100 + i) for _ in range(4)], 5000 + i, cfg)
          for i in range(3)]
    say(f"[engine/random] 3 局共 {time.perf_counter() - t0:.2f}s  "
        + "  ".join(f"{d:.2f}s/{n}步/{t}" for d, n, t in hs))

    # 2) 纯规则引擎：4 个启发式教师（含向听数计算）
    t0 = time.perf_counter()
    hs = [run_hand([HeuristicBot(seed=200 + i, weights=TEACHER_STYLES["balanced"])
                    for _ in range(4)], 6000 + i, cfg) for i in range(3)]
    say(f"[engine/teacher] 3 局共 {time.perf_counter() - t0:.2f}s  "
        + "  ".join(f"{d:.2f}s/{n}步/{t}" for d, n, t in hs))

    # 3) 载入模型
    t0 = time.perf_counter()
    ck = torch.load(ROOT / "models" / "rl.pt", map_location="cpu", weights_only=False)
    say(f"[load] torch.load {time.perf_counter() - t0:.2f}s  keys={sorted(ck.keys())}")
    say(f"[load] ck.net={ck.get('net')}")

    nc = ck.get("net") or {}
    net = build_model(NetConfig(int(nc.get("channels", 48)), int(nc.get("blocks", 2)),
                                int(nc.get("hidden", 256))))
    t0 = time.perf_counter()
    net.load_state_dict(ck["state_dict"])
    net.eval()
    say(f"[load] build+load_state_dict {time.perf_counter() - t0:.2f}s  "
        f"params={sum(p.numel() for p in net.parameters())}")

    # 4) 单次推理耗时
    agent = NetAgent(net, seed=0, temperature=0.0, name="bench")
    g = ZhuojiGame(cfg, seed=777)
    t0 = time.perf_counter()
    for _ in range(50):
        agent.choose(g)
    say(f"[model] 50 次决策 {time.perf_counter() - t0:.2f}s "
        f"({(time.perf_counter() - t0) / 50 * 1000:.1f} ms/次，同一局面)")

    # 5) 含模型的整局
    t0 = time.perf_counter()
    hs = [run_hand([agent, RandomBot(seed=300 + i), RandomBot(seed=400 + i),
                    HeuristicBot(seed=500 + i, weights=TEACHER_STYLES["balanced"])],
                   7000 + i, cfg) for i in range(3)]
    say(f"[1model+3bot] 3 局共 {time.perf_counter() - t0:.2f}s  "
        + "  ".join(f"{d:.2f}s/{n}步/{t}" for d, n, t in hs))

    say("=== done ===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
