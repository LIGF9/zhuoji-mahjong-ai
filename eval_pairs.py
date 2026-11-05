"""配对闸门评测（D1 管线）：候选 A vs 基线 B，重复牌墙两场（坐 0 / 坐 2）。

输出 JSONL 与 analyze.py / duel.py 口径一致：
    {"seed", "cand_seat", "deltas", "steps", "type", "winner", "s"}
用法：
  # 闸门 A：贪心对贪心，300 seed（快，~分钟级）
  .venv/bin/python eval_pairs.py --a models/exit_best.pt --b models/dushan_master.pt \
      --mode greedy --seeds 300 --workers 6 --out runs/gateA_r01.jsonl
  # 闸门 B：N32 搜索对 N32 搜索，100 seed（慢，~1.5h）
  ... --mode search32 --seeds 100
分析：.venv/bin/python analyze.py runs/gateA_r01.jsonl
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import numpy as np

from zhuoji.dushan import DushanConfig, DushanGame
from zhuoji.rules import Phase


def make_agent(model_path, mode, seat_seed, dev):
    import torch
    from zhuoji.agent import NetAgent
    from zhuoji.net import build_from_ckpt

    ck = torch.load(str(model_path), map_location="cpu", weights_only=False)
    net, _ = build_from_ckpt(ck, map_location="cpu")
    if mode == "greedy":
        return NetAgent(net, seed=seat_seed, temperature=0.0, name="greedy")
    from zhuoji.rust_search import RustSearchAgent
    return RustSearchAgent(net, seed=seat_seed, device=dev, name="s32",
                           ply=3, n_det=32, topk=3, ff_cap=24, fair=True)


def play_deal(a_path, b_path, mode, seed, cand_seat, dev):
    """一局：cand_seat 坐 A，其余三家坐 B。返回 record dict。"""
    agents = []
    for s in range(4):
        p = a_path if s == cand_seat else b_path
        agents.append(make_agent(p, mode, seed * 131 + s, dev))
    g = DushanGame(DushanConfig(), seed=seed)
    steps = 0
    t0 = time.perf_counter()
    while g.phase != Phase.OVER and steps < 600:
        acts = g.legal_actions()
        if not acts:
            break
        a = agents[g.actor()].choose(g) if len(acts) >= 2 else acts[0]
        if a is None:
            a = acts[0]
        g.step(a)
        steps += 1
    res = g.result or {}
    deltas = res.get("deltas") or [0.0, 0.0, 0.0, 0.0]
    return {
        "seed": int(seed), "cand_seat": int(cand_seat),
        "deltas": [float(x) for x in deltas], "steps": int(steps),
        "type": res.get("type", "over"),
        "winner": res.get("winner"),
        "s": round(time.perf_counter() - t0, 2),
    }


def worker(wid, a_path, b_path, mode, seeds, dev, out_path):
    import torch
    torch.set_num_threads(2)
    lines = []
    for k, seed in enumerate(seeds):
        for cand_seat in (0, 2):
            rec = play_deal(a_path, b_path, mode, seed, cand_seat, dev)
            lines.append(json.dumps(rec, ensure_ascii=False))
        if (k + 1) % 10 == 0:
            print(f"[w{wid}] {k+1}/{len(seeds)}", flush=True)
    with open(out_path, "a", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"[w{wid}] done {len(seeds)} seeds → {out_path}", flush=True)


def summary(out_path):
    recs = {}
    for line in open(out_path, encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        recs[(r["seed"], r["cand_seat"])] = r
    diffs = []
    for s in sorted({k[0] for k in recs}):
        r0, r2 = recs.get((s, 0)), recs.get((s, 2))
        if not r0 or not r2:
            continue
        def margin(r):
            d, c = r["deltas"], r["cand_seat"]
            return d[c] - sum(d[i] for i in range(4) if i != c) / 3.0
        diffs.append(0.5 * (margin(r0) + margin(r2)))
    x = np.array(diffs, dtype=np.float64)
    n = len(x)
    if n < 2:
        print("[summary] 样本不足")
        return
    m, se = float(x.mean()), float(x.std(ddof=1) / np.sqrt(n))
    z = m / se if se > 0 else 0.0
    from math import erfc, sqrt
    p = erfc(abs(z) / sqrt(2))                      # 正态近似双侧 p
    wins = int((x > 0).sum()); losses = int((x < 0).sum())
    print(f"[summary] n={n}  diff={m:+.3f}±{se:.3f}  z={z:+.2f}  p≈{p:.4f}  "
          f"符号 {wins}:{losses}")
    print(f"[summary] 判定: {'通过（p<0.05 且均值>0）' if (p < 0.05 and m > 0) else '不通过'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True, help="候选模型 .pt")
    ap.add_argument("--b", default=str(ROOT / "models/dushan_master.pt"))
    ap.add_argument("--mode", choices=["greedy", "search32"], default="greedy")
    ap.add_argument("--seeds", type=int, default=300)
    ap.add_argument("--seed0", type=int, default=710_000)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--device", default="mps")
    ap.add_argument("--out", default=str(ROOT / "runs/gate.jsonl"))
    args = ap.parse_args()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("")                       # 清空旧结果
    per = (args.seeds + args.workers - 1) // args.workers
    import multiprocessing as mp
    ps = []
    for i in range(args.workers):
        lo = args.seed0 + i * per
        seeds = list(range(lo, min(lo + per, args.seed0 + args.seeds)))
        if not seeds:
            break
        p = mp.Process(target=worker, args=(
            i, args.a, args.b, args.mode, seeds, args.device, str(out)))
        p.start()
        ps.append(p)
    for p in ps:
        p.join()
        assert p.exitcode == 0, f"worker exit {p.exitcode}"
    summary(str(out))


if __name__ == "__main__":
    main()
