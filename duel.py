"""独山麻将配对赛驱动：候选算法 vs N32 基线，固定牌墙 + 四座旋转。

用法：
    .venv/bin/python duel.py --hands 300 --workers 6 --out runs/v1.jsonl \
        [--cand-ndet 128] [--cand-ply 3] [--cand-agg 0] [--cand-lam 1.0]

设计（冻结基线池协议，docs/14天训练计划.md §0）：
  * 每个牌墙 seed 打四场：候选轮流坐 0/1/2/3 号位，其余三家用 N32。
  * 基线严格复刻线上「深算 n32」：ply=3 / n_det=32 / topk=3 / fair=True（Rust 内核）。
  * 每条记录流式追加 JSONL：{"seed","cand_seat","deltas","steps","type","winner"}。
  * 中断可续：已完成的 seed 跳过（按输出文件扫描）。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

OUT = ""
CAND = {}
SEATS = (0, 1, 2, 3)


def worker_main(wid, jobs, dev, out, cand, cand_model=""):
    import torch
    torch.set_num_threads(2)
    from zhuoji.dushan import DushanConfig, DushanGame
    from zhuoji.net import ZhuojiNet
    from zhuoji.rust_search import RustSearchAgent

    ck = torch.load(ROOT / "models/dushan_master.pt", map_location="cpu",
                    weights_only=False)
    net = ZhuojiNet(channels=64, blocks=8, hidden=320)
    net.load_state_dict(ck["state_dict"])
    net.eval()
    if cand_model:
        cck = torch.load(ROOT / cand_model if not cand_model.startswith("/") else cand_model,
                         map_location="cpu", weights_only=False)
        cand_net = ZhuojiNet(channels=64, blocks=8, hidden=320)
        cand_net.load_state_dict(cck["state_dict"])
        cand_net.eval()
    else:
        cand_net = net

    cand_ag = RustSearchAgent(cand_net, seed=1, device=dev, name="cand",
                              ply=int(cand.get("ply", 3)), topk=int(cand.get("topk", 3)),
                              ff_cap=int(cand.get("ff", 24)),
                              n_det=int(cand.get("ndet", 128)), fair=True,
                              agg=int(cand.get("agg", 0)), lam=float(cand.get("lam", 1.0)),
                              prior_weight=float(cand.get("pw", 0.0)),
                              decide=str(cand.get("decide", "q")))
    base = RustSearchAgent(net, seed=2, device=dev, name="n32",
                           ply=3, topk=3, n_det=32, fair=True)

    fout = open(f"{out}.w{wid}", "a", encoding="utf-8")
    ferr = open(f"{out}.w{wid}.err", "a", encoding="utf-8")
    try:
        for (seed, cand_seat) in jobs:
            g = DushanGame(DushanConfig(), seed=seed)
            cand_ag.seed = 1000 + seed; cand_ag._call = 0
            base.seed = 2000 + seed; base._call = 0

            def agent_for(p):
                return cand_ag if p == cand_seat else base

            t0 = time.perf_counter()
            steps = 0
            while not g.result and steps < 600:
                acts = g.legal_actions()
                if not acts:
                    break
                ag = agent_for(g.actor())
                if len(acts) <= 1:
                    a = acts[0]
                else:
                    a = ag.choose(g)
                g.step(a)
                steps += 1
            dt = time.perf_counter() - t0
            res = g.result or {}
            rec = {"seed": seed, "cand_seat": cand_seat, "deltas": res.get("deltas"),
                   "steps": steps, "type": res.get("type"), "winner": res.get("winner"),
                   "s": round(dt, 2)}
            fout.write(json.dumps(rec) + "\n")
            fout.flush()
    except Exception:
        import traceback
        traceback.print_exc(file=ferr)
        ferr.flush()
    finally:
        fout.close()
        ferr.close()


def main():
    global OUT, CAND
    ap = argparse.ArgumentParser()
    ap.add_argument("--hands", type=int, default=300, help="牌墙 seed 数（每 seed 打两场）")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--out", default="runs/v1.jsonl")
    ap.add_argument("--cand-ndet", type=int, default=128)
    ap.add_argument("--cand-ply", type=int, default=3)
    ap.add_argument("--cand-agg", type=int, default=0)
    ap.add_argument("--cand-lam", type=float, default=1.0)
    ap.add_argument("--cand-pw", type=float, default=0.0)
    ap.add_argument("--cand-decide", default="q")
    ap.add_argument("--cand-model", default="")
    ap.add_argument("--cand-topk", type=int, default=3)
    ap.add_argument("--cand-ff", type=int, default=24)
    ap.add_argument("--device", default="mps")
    ap.add_argument("--seed0", type=int, default=500000)
    args = ap.parse_args()

    OUT = args.out
    CAND = {"ndet": args.cand_ndet, "ply": args.cand_ply,
            "agg": args.cand_agg, "lam": args.cand_lam, "pw": args.cand_pw,
            "decide": args.cand_decide, "topk": args.cand_topk, "ff": args.cand_ff}
    os.makedirs(os.path.dirname(OUT) or ".", exist_ok=True)

    # 断点续跑：扫描已完成 (seed, seat)
    done = set()
    for suf in ([f"{OUT}.w{i}" for i in range(args.workers)] + [OUT]):
        if os.path.exists(suf):
            for line in open(suf, encoding="utf-8"):
                try:
                    r = json.loads(line)
                    done.add((r["seed"], r["cand_seat"]))
                except Exception:
                    pass

    seeds = [args.seed0 + i for i in range(args.hands)]
    jobs = []
    for s in seeds:
        for c in SEATS:
            if (s, c) in done:
                continue
            jobs.append((s, c))
    random.Random(0).shuffle(jobs)          # 打散：中断时各场不偏科
    chunks = [jobs[i::args.workers] for i in range(args.workers)]

    import multiprocessing as mp
    t0 = time.perf_counter()
    procs = []
    for i, ch in enumerate(chunks):
        if not ch:
            continue
        p = mp.Process(target=worker_main, args=(i, ch, args.device, OUT, CAND,
                                                 args.cand_model))
        p.start()
        procs.append(p)
    print(f"[duel] {len(jobs)} hands / {args.workers} workers -> {OUT}", flush=True)
    for p in procs:
        p.join()
    print(f"[duel] done in {(time.perf_counter()-t0)/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
