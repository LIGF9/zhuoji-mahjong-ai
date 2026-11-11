"""ExIt 数据采集（D1 管线）：学生 rollout 出状态 + 教师（n32-ply3 搜索）标注全动作 Q。

与 gen_distill.py 的三点不同（对应 14 天计划）：
1. 记录**全 Q 向量**（28 槽位布局）而不只是 qmax/qmean —— 训练侧要同时
   蒸馏策略头（softmax(Q/τ)）与价值头（maxQ），这是 V9「只蒸价值无效」的教训。
2. rollout 策略 = 学生自身（温度采样），状态分布集中在学生弱区（ExIt 迭代）。
3. 观测走 **encoder v3**（63 通道 / 130 全局维），供 M1 学生训练。

用法（单 worker 调试）：
  .venv/bin/python gen_exit.py --targets 2000 --workers 1 --out runs/exit_smoke/ \
      --student models/dushan_master.pt --teacher models/dushan_master.pt
多 worker 正式跑：
  .venv/bin/python gen_exit.py --targets 100000 --workers 4 --out runs/exit_r01/
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import numpy as np

HEAD_ID = {"discard": 0, "self_kong": 1, "respond": 2}
N_SLOTS = 28


def load_net(path, dev):
    import torch
    from zhuoji.net import build_from_ckpt

    ck = torch.load(str(path), map_location="cpu", weights_only=False)
    net, ver = build_from_ckpt(ck, map_location="cpu")
    # ONNX 缓存名 = 检查点文件名（ensure_onnx 按 models/{stem}.pt 的 mtime
    # 决定是否重导出，训练出新检查点会自动刷新 ONNX）。
    net.onnx_stem = Path(path).stem
    if str(dev).lower().startswith("ort"):
        # ORT 后端不搬 torch 设备：_BatchRunner 直接喂 numpy 给 onnxruntime。
        return net.eval(), ver
    return net.to(dev).eval(), ver


def worker(wid, student_path, teacher_path, dev, n_targets, sample_rate,
           temperature, enc_version, seed0, out_dir):
    import random
    import torch

    torch.set_num_threads(2)
    from zhuoji.agent import NetAgent
    from zhuoji.dushan import DushanConfig, DushanGame
    from zhuoji.encoder import encode, head_index
    from zhuoji.rust_search import deep_q

    random.seed(seed0 + wid)
    np.random.seed(seed0 + wid)

    # NetAgent 直接喂 CPU 张量，rollout 模型必须留在 CPU（实测 1231 步/s 足够快）；
    # 教师搜索走 _BatchRunner 自己搬运设备，可以放 mps。
    stu_net, stu_ver = load_net(student_path, "cpu")
    tea_net, tea_ver = load_net(teacher_path, dev)
    # 教师搜索的观测编码必须跟着**教师权重**走：v1 教师（dushan_master）用 15/70，
    # v3 教师（champ_exit_p48 等）用 63/130。此前 deep_q 沿用默认 enc=1，
    # 换 v3 教师会把 15 通道喂进 63 通道的 ONNX → 直接 shape 报错。
    tea_enc = int(getattr(tea_net, "encoder_version", tea_ver) or 1)
    if str(dev).lower().startswith("ort"):
        # 预热 CoreML 会话（每个形状一次图编译 150~220ms；不预热会摊进首轮标注）。
        from zhuoji.forward import OrtForward, build_forward
        build_forward(tea_net, dev, stem=getattr(tea_net, "onnx_stem", None),
                      threads=1).warm(OrtForward.LADDER)
    # 记录给训练的观测**恒为 v3**（M1 学生口径）；rollout 策略自己按模型声明的
    # 版本编码决策（master 走 v1），两者互不影响。
    rec_ver = int(enc_version) if enc_version else 3

    roller = NetAgent(stu_net, seed=seed0 + wid, temperature=temperature,
                      name=f"rollout-w{wid}", encoder_version=int(stu_ver))

    T, G, HD, MK, QQ = [], [], [], [], []
    t0 = time.perf_counter()
    deals = 0
    while len(QQ) < n_targets:
        deals += 1
        g = DushanGame(DushanConfig(), seed=seed0 + wid * 100000 + deals)
        steps = 0
        while g.result is None and steps < 600:
            acts = g.legal_actions()
            if not acts:
                break
            if len(acts) >= 2 and random.random() < sample_rate and len(QQ) < n_targets:
                p = g.actor()
                o = encode(g, p, rec_ver)
                q, _st = deep_q(tea_net, g, p, list(acts), n_det=32, ff_cap=24,
                                topk=3, ply=3, seed=seed0 + wid * 7919 + deals,
                                device=dev, enc=tea_enc)
                qv = np.full(N_SLOTS, np.nan, dtype=np.float32)
                for a, qa in zip(acts, q):
                    qv[head_index(o.head, a)] = float(qa)
                m28 = np.zeros(N_SLOTS, dtype=bool)
                m28[: o.mask.shape[0]] = o.mask
                T.append(np.asarray(o.tiles, dtype=np.float32))
                G.append(np.asarray(o.glob, dtype=np.float32))
                HD.append(HEAD_ID[o.head])
                MK.append(m28)
                QQ.append(qv)
            if len(acts) == 1:
                g.step(acts[0])
            else:
                g.step(roller.choose(g))
            steps += 1
        if deals % 10 == 0:
            el = time.perf_counter() - t0
            rate = len(QQ) / max(el, 1e-9) * 3600
            print(f"[w{wid}] deals={deals} targets={len(QQ)}/{n_targets} "
                  f"{el:.0f}s ({rate:.0f} 标注/h)", flush=True)

    out = Path(out_dir) / f"exit_w{wid}.npz"
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out,
        tiles=np.array(T, dtype=np.float32),
        glob=np.array(G, dtype=np.float32),
        head=np.array(HD, dtype=np.uint8),
        mask=np.array(MK, dtype=bool),
        q=np.array(QQ, dtype=np.float32),
    )
    print(f"[w{wid}] saved {out}: {len(QQ)} targets", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--targets", type=int, default=100_000, help="总标注状态数")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--sample-rate", type=float, default=0.25,
                    help="每个决策点跑教师标注的概率")
    ap.add_argument("--temperature", type=float, default=0.4,
                    help="rollout 采样温度（0 = 贪心）")
    ap.add_argument("--student", default=str(ROOT / "models/dushan_master.pt"))
    ap.add_argument("--teacher", default=str(ROOT / "models/dushan_master.pt"))
    ap.add_argument("--enc-version", type=int, default=3,
                    help="记录观测的编码器版本（默认 3 = M1 口径）")
    ap.add_argument("--device", default="mps")
    ap.add_argument("--seed0", type=int, default=610_000)
    ap.add_argument("--out", default=str(ROOT / "runs/exit_r01"))
    args = ap.parse_args()

    import multiprocessing as mp

    per = (args.targets + args.workers - 1) // args.workers
    ps = []
    for i in range(args.workers):
        p = mp.Process(target=worker, args=(
            i, args.student, args.teacher, args.device, per,
            args.sample_rate, args.temperature, args.enc_version,
            args.seed0 + i * 10_000, args.out))
        p.start()
        ps.append(p)
    for p in ps:
        p.join()
        assert p.exitcode == 0, f"worker exit {p.exitcode}"
    print("[gen_exit] all done", flush=True)


if __name__ == "__main__":
    main()
