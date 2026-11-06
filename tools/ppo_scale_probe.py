"""PPO-lite 规模测算探针（只读、不落盘、不训练、不评估闸门）。

目的：给 docs/14天训练计划.md §6.22.7⑤「切换前置条件 = PPO 冒烟」提供**成本实测**，
回答一个问题：2k 局纯策略自对弈在本机要多久？>1.5h 则判不可行。

做法：
  * 用**当前 exit_best.pt**（M1Net + enc v3）当 rollout 策略，temperature=1.0 采样；
  * 只做**单步前向**（无搜索、无 determinization），与 PPO rollout 的真实成本同构；
  * 逐决策 `NetAgent.distribution()` 取 (head, mask, probs, value) —— 这正是 PPO 需要的量，
    顺带验证这条既有管线**能否直接喂 PPO**（不必新写 env）；
  * 报告 s/hand、决策数/hand、以及并行 k 进程的折算。

用法：
  .venv/bin/python tools/ppo_scale_probe.py --hands 12 --device cpu --workers 4

注意：训练期运行请用 --device cpu（MPS 被 train_exit.py 占用，抢会拖慢主循环）。
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=str(ROOT / "models/exit_best.pt"))
    ap.add_argument("--hands", type=int, default=12)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--workers", type=int, default=4,
                    help="折算用的并行进程数（仅做线性外推，不真的开进程）")
    ap.add_argument("--budget", type=float, default=1.5, help="可接受上限（小时）")
    a = ap.parse_args()

    import numpy as np
    import torch

    torch.set_num_threads(2)
    from zhuoji.net import build_from_ckpt
    from zhuoji.dushan import DushanConfig, DushanGame
    from zhuoji.agent import NetAgent

    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    net, enc_ver = build_from_ckpt(ck)
    net.eval().to(a.device)
    print(f"[ckpt] {Path(a.ckpt).name}  params={net.n_params():,}  encoder=v{enc_ver}"
          f"  device={a.device}")

    agent = NetAgent(net, seed=11, temperature=1.0,
                     encoder_version=enc_ver,
                     encoder_packs=getattr(net, "encoder_packs", None))

    n_dec = 0
    n_ent, n_v, n_legal = [], [], []
    t0 = time.perf_counter()
    for h in range(a.hands):
        g = DushanGame(DushanConfig(), seed=770000 + h)
        agent.reset(770000 + h)
        steps = 0
        while not g.result and steps < 600:
            with torch.no_grad():
                head, mask, probs, value, _obs = agent.distribution(g)
            p = probs.detach().numpy().astype(np.float64)
            p = p / p.sum()
            idx = int(np.random.default_rng(770000 + h + steps).choice(len(p), p=p))
            from zhuoji.encoder import index_to_action
            act = index_to_action(head, idx, g)
            n_dec += 1
            n_legal.append(int(mask.sum()))
            n_ent.append(float(-(p * np.log(np.clip(p, 1e-12, None))).sum()))
            n_v.append(value)
            g.step(act)
            steps += 1
    el = time.perf_counter() - t0
    per_hand = el / a.hands
    dec_per_hand = n_dec / a.hands

    print(f"[rollout] {a.hands} 局 / {n_dec} 决策 / {el:.1f}s"
          f"  → {per_hand:.3f} s/局, {dec_per_hand:.1f} 决策/局")
    print(f"[合法性] 合法动作数 中位={np.median(n_legal):.0f} "
          f"min={min(n_legal)} max={max(n_legal)}（>0 即掩码正确）")
    print(f"[策略] τ=1 熵 中位={np.median(n_ent):.3f} nats（PPO 需非退化，>0.3 可学）")
    print(f"[价值] V 中位={np.median(n_v):.2f} 范围=[{min(n_v):.2f}, {max(n_v):.2f}]"
          f"（PPO 需 value 有方差，否则优势≈0）")

    for n in (2_000, 10_000):
        single = per_hand * n / 3600
        par = single / max(1, a.workers)
        print(f"[外推] {n} 局：单进程 {single:.2f}h  /  {a.workers} 进程 {par:.2f}h"
              f"  → {'可行' if par <= a.budget else '超预算，判不可行'}")
    print(f"[判据] 上限 {a.budget}h × {a.workers} 进程 ⇒ 最多 "
          f"{int(a.budget * 3600 / per_hand * a.workers):,} 局/轮")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
