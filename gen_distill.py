"""搜索蒸馏数据生成：自博弈采状态 + n32-ply3 搜索 Q 作为价值回归目标。

两阶段（并行）：
  A) n8-ply1 快速自博弈（~0.05s/决策），每个决策点（合法动作 ≥2）记录
     编码 obs + 合法动作标签；随机抽 `--targets` 条进入 B 阶段。
  B) 对被抽中的状态跑 n32-ply3 完整搜索，得逐动作 Q；目标 = max_a Q(a)
     （与搜索叶子「回到本座位决策点取后续最优」的用法一致）。

输出：runs/distill_data.npz（tiles/glob/heads/masks/q_max/q_mean/act_labels）
"""
from __future__ import annotations

import argparse
import random
import sys
import time

sys.path.insert(0, "/Users/ligf9/WorkBuddy/2026-11-03-18-40-11/zhuoji-mahjong-ai")

import numpy as np


def load_net(dev):
    import torch
    from zhuoji.net import ZhuojiNet
    ck = torch.load("/Users/ligf9/WorkBuddy/2026-11-03-18-40-11/zhuoji-mahjong-ai/models/dushan_master.pt",
                    map_location="cpu", weights_only=False)
    net = ZhuojiNet(channels=64, blocks=8, hidden=320)
    net.load_state_dict(ck["state_dict"])
    net.eval()
    return net.to(dev)


def worker(wid, dev, n_hands, n_targets, seed0, out):
    import torch
    torch.set_num_threads(2)
    from zhuoji.dushan import DushanConfig, DushanGame
    from zhuoji.encoder import encode, index_to_action
    from zhuoji.rust_search import RustSearchAgent, deep_q

    net = load_net(dev)
    fast = RustSearchAgent(net, seed=7, device=dev, name="fast", ply=1, n_det=8, fair=True)
    deep = RustSearchAgent(net, seed=8, device=dev, name="teacher", ply=3, n_det=32, fair=True)

    obs_tiles, obs_glob = [], []
    qmax, qmean = [], []
    t0 = time.perf_counter()
    n_states = 0
    for h in range(n_hands):
        g = DushanGame(DushanConfig(), seed=seed0 + h)
        fast.seed = seed0 + h; fast._call = 0
        steps = 0
        while not g.result and steps < 600:
            acts = g.legal_actions()
            if not acts:
                break
            if len(acts) >= 2:
                p = g.actor()
                o = encode(g, p, 1)
                # B 阶段抽样：按概率就地计算教师 Q（n32-ply3）
                if random.random() < 0.22:
                    deep.seed = 90000 + seed0 + h; deep._call = 0
                    legal = g.legal_actions()
                    q, _st = deep_q(net, g, p, legal, n_det=32, ff_cap=24,
                                    topk=3, ply=3, seed=deep._next_seed(),
                                    device=dev)
                    obs_tiles.append(np.asarray(o.tiles, dtype=np.float32))
                    obs_glob.append(np.asarray(o.glob, dtype=np.float32))
                    qmax.append(float(np.max(q)))
                    qmean.append(float(np.mean(q)))
                a = fast.choose(g)
                g.step(a)
            else:
                g.step(acts[0])
            steps += 1
        n_states += steps
        if (h + 1) % 20 == 0:
            el = time.perf_counter() - t0
            print(f"[w{wid}] hand {h+1}/{n_hands}  targets={len(qmax)}  {el:.0f}s",
                  flush=True)
    np.savez_compressed(out,
                        tiles=np.array(obs_tiles), glob=np.array(obs_glob),
                        qmax=np.array(qmax, dtype=np.float32),
                        qmean=np.array(qmean, dtype=np.float32))
    print(f"[w{wid}] saved {out}: {len(qmax)} targets", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--hands", type=int, default=40, help="每 worker 自博弈手数")
    ap.add_argument("--seed0", type=int, default=900000)
    ap.add_argument("--out", default="runs/distill_w{}.npz")
    args = ap.parse_args()

    import multiprocessing as mp
    ps = []
    for i in range(args.workers):
        p = mp.Process(target=worker, args=(i, "mps", args.hands, 0,
                                            args.seed0 + i * 10000,
                                            args.out.format(i)))
        p.start()
        ps.append(p)
    for p in ps:
        p.join()
    print("[gen] all done", flush=True)


if __name__ == "__main__":
    main()
