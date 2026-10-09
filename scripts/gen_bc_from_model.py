"""从指定模型（默认 s7）的自对弈中生成独山麻将 BC 数据集。

与 ``gen_bc_data.py`` 的区别：那个用启发式教师池 + 捉鸡规则；本脚本用**强模型在
独山规则下自对弈**采样，目标是给更大网络做行为克隆预热（克隆 s7 的策略与价值）。

输出 npz 字段与 ``train_bc.py`` 兼容：
    tiles(f16) glob(f16) mask(u8,28) head(i64) act(i64) value(f32)

value = 该决策玩家本局最终得分（原始分，非缩放），给价值头当回归目标。

用法::
    python scripts/gen_bc_from_model.py --model models/dushan_s7.pt --games 10000
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from zhuoji.dushan import DushanConfig, DushanGame  # noqa: E402
from zhuoji.net import NetConfig, build_model  # noqa: E402
from zhuoji.vcollect import collect_vectorized  # noqa: E402


def load_model(path: str):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    nc = ck.get("net") or {"channels": 48, "blocks": 8, "hidden": 256}
    m = build_model(NetConfig(nc["channels"], nc["blocks"], nc["hidden"]))
    m.load_state_dict(ck["state_dict"])
    return m.eval(), nc


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=str(ROOT / "models" / "dushan_s7.pt"))
    ap.add_argument("--games", type=int, default=10000)
    ap.add_argument("--chunk", type=int, default=128, help="每批对局数")
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--teacher-prob", type=float, default=0.0,
                    help=">0 时混入启发式教师当对手，增加状态多样性")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--seed", type=int, default=20261005)
    ap.add_argument("--out", default=str(ROOT / "data" / "bc_dushan_s7.npz"))
    args = ap.parse_args()

    torch.set_num_threads(args.threads)
    model, nc = load_model(args.model)
    print(f"教师模型 {args.model}  网络 {nc}  参数量 {model.n_params():,}")
    print(f"目标 {args.games} 局，温度 {args.temperature}，教师概率 {args.teacher_prob}")

    T, G, M, H, A, V = [], [], [], [], [], []
    n_done = 0
    t0 = time.time()
    chunk_i = 0
    while n_done < args.games:
        k = min(args.chunk, args.games - n_done)
        traj, rewards, dt = collect_vectorized(
            model, k, args.seed + chunk_i * 7919, args.temperature,
            reward_scale=1.0, teacher_prob=args.teacher_prob,
            threads=max(2, args.threads // 2), snapshots=[], past_prob=0.0,
            game_cls=DushanGame, config_cls=DushanConfig, want_traj=True)
        for (tiles, glob, mask, head, act, seat, gi) in traj:
            T.append(tiles.astype(np.float16))
            G.append(glob.astype(np.float16))
            M.append(mask.astype(np.uint8))
            H.append(head)
            A.append(act)
            V.append(float(rewards[gi, seat]))
        n_done += k
        chunk_i += 1
        el = time.time() - t0
        print(f"  {n_done}/{args.games} 局  样本 {len(H):,}  {el:.0f}s "
              f"(预计 {el / n_done * args.games:.0f}s)", flush=True)

    data = {
        "tiles": np.stack(T), "glob": np.stack(G), "mask": np.stack(M),
        "head": np.array(H, dtype=np.int64), "act": np.array(A, dtype=np.int64),
        "value": np.array(V, dtype=np.float32),
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, **data)
    n = len(data["head"])
    print(f"完成：{n:,} 样本 -> {args.out}")
    for hid, name in enumerate(["discard", "self_kong", "respond"]):
        print(f"  头 {name:9s}: {int((data['head'] == hid).sum()):>8,} 条")
    print(f"  每局决策数均值 {n / max(1, args.games):.1f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
