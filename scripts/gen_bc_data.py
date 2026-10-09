"""
用启发式教师池生成行为克隆（BC）数据集。

多进程：每个 worker 生成若干"分片"，写到 ``--out`` 目录，最后合并成单个 npz。

样本内容：决策者视角的观测 (tiles, glob)、动作头、动作下标、以及该玩家本局的
最终得分（既当 BC 的辅助回归目标，也是后续 RL 的价值基线初始化）。
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from zhuoji import RulesConfig, ZhuojiGame  # noqa: E402
from zhuoji.bots import TEACHER_STYLES, HeuristicBot, HeuristicWeights  # noqa: E402
from zhuoji.encoder import (  # noqa: E402
    GLOBAL_DIM, HEAD_DISCARD, HEAD_RESPOND, HEAD_SELF_KONG, NUM_CHANNELS,
    encode, head_index,
)

HEAD_ID = {HEAD_DISCARD: 0, HEAD_SELF_KONG: 1, HEAD_RESPOND: 2}


def make_teacher(style: str, seed: int) -> HeuristicBot:
    b = HeuristicBot(seed=seed, weights=TEACHER_STYLES[style])
    b.name = f"heuristic:{style}"
    return b


def gen_shard(worker: int, n_games: int, seed0: int, styles: list[str],
              out_path: str) -> tuple[str, int]:
    cfg = RulesConfig()
    tiles_buf: list[np.ndarray] = []
    glob_buf: list[np.ndarray] = []
    mask_buf: list[np.ndarray] = []
    head_buf: list[int] = []
    act_buf: list[int] = []
    val_buf: list[float] = []
    seat_buf: list[int] = []
    turn_buf: list[int] = []

    for gi in range(n_games):
        seed = seed0 + worker * 1_000_003 + gi * 7919
        game = ZhuojiGame(cfg, seed=seed)
        bots = [make_teacher(styles[(gi + k) % len(styles)], seed + 17 * k + k)
                for k in range(4)]

        recs = []
        steps = 0
        while game.phase.value != "over" and steps < 4000:
            p = game.actor()
            obs = encode(game, p)
            a = bots[p].choose(game)
            idx = head_index(obs.head, a)
            m = np.zeros(28, dtype=np.uint8)
            m[:obs.mask.shape[0]] = obs.mask.astype(np.uint8)
            recs.append((p, obs.tiles, obs.glob, m, HEAD_ID[obs.head], idx, steps))
            game.step(a)
            steps += 1

        for (p, tiles, glob, m, hid, idx, st) in recs:
            tiles_buf.append(tiles.astype(np.float16))
            glob_buf.append(glob.astype(np.float16))
            mask_buf.append(m)
            head_buf.append(hid)
            act_buf.append(idx)
            val_buf.append(game.score_delta[p])
            seat_buf.append(p)
            turn_buf.append(st)

    np.savez_compressed(
        out_path,
        tiles=np.stack(tiles_buf) if tiles_buf else np.zeros((0, NUM_CHANNELS, 27), np.float16),
        glob=np.stack(glob_buf) if glob_buf else np.zeros((0, GLOBAL_DIM), np.float16),
        mask=np.stack(mask_buf) if mask_buf else np.zeros((0, 28), np.uint8),
        head=np.array(head_buf, dtype=np.uint8),
        act=np.array(act_buf, dtype=np.uint8),
        value=np.array(val_buf, dtype=np.float32),
        seat=np.array(seat_buf, dtype=np.uint8),
        turn=np.array(turn_buf, dtype=np.uint16),
    )
    return out_path, len(head_buf)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=4000)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) - 1))
    ap.add_argument("--seed", type=int, default=20260930)
    ap.add_argument("--out", type=str, default=str(ROOT / "data" / "bc_data.npz"))
    ap.add_argument("--shards-dir", type=str, default=str(ROOT / "data" / "shards"))
    args = ap.parse_args()

    styles = list(TEACHER_STYLES.keys())
    shards_dir = Path(args.shards_dir)
    shards_dir.mkdir(parents=True, exist_ok=True)

    per = max(1, args.games // args.workers)
    jobs = [(w, per, args.seed, styles, str(shards_dir / f"shard_{w:02d}.npz"))
            for w in range(args.workers)]

    t0 = time.time()
    import multiprocessing as mp
    ctx = mp.get_context("spawn")
    total = 0
    with ctx.Pool(args.workers) as pool:
        for path, n in pool.starmap(gen_shard, jobs):
            total += n
            print(f"  shard done: {path}  samples={n}", flush=True)
    dt = time.time() - t0
    print(f"生成 {args.workers * per} 局 / {total} 样本，用时 {dt:.1f}s "
          f"({dt / max(1, args.workers * per) * 1000:.1f} ms/局)")

    # 合并
    parts = []
    for w in range(args.workers):
        p = shards_dir / f"shard_{w:02d}.npz"
        if p.exists():
            parts.append(np.load(p))
    out = args.out
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    merged = {
        k: np.concatenate([p[k] for p in parts], axis=0)
        for k in ("tiles", "glob", "mask", "head", "act", "value", "seat", "turn")
    }
    np.savez_compressed(out, **merged)
    print("合并完成 ->", out, {k: v.shape for k, v in merged.items()})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
