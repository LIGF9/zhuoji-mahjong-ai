"""Rust↔Python 管线对拍：encode_v3 / deep_q 逐位核对。

流程：
1. 随机 rollout 采样 N 个决策状态（固定种子，可复现）。
2. Python 参考：encode(g, p, 3) 编码 + deep_q(teacher, n32-ply3) 教师 Q。
3. Rust 对照：annotate --check-encode / --check-q 读同一 states.jsonl。
4. 逐位/容差比较并报告 max|Δ|。

用法：
  .venv/bin/python check_pairs.py --states 30
"""
from __future__ import annotations

import argparse
import json
import random
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import numpy as np

HEAD_ID = {"discard": 0, "self_kong": 1, "respond": 2}
KIND_CODE = {"discard": 0, "hu": 1, "angang": 2, "bugang": 3,
             "peng": 4, "minggang": 5, "pass": 6}


def act_code(a) -> int:
    t = 27 if a.tile < 0 else int(a.tile)
    return KIND_CODE[a.kind] * 28 + t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--states", type=int, default=30)
    ap.add_argument("--seed0", type=int, default=555_000)
    ap.add_argument("--teacher", default=str(ROOT / "models/dushan_master.pt"))
    ap.add_argument("--teacher-onnx", default=str(ROOT / "models/onnx/dushan_master.onnx"))
    ap.add_argument("--device", default="ort", help="Python 教师前向后端（ort=CPU EP 对拍最稳）")
    ap.add_argument("--q-tol", type=float, default=1e-4)
    ap.add_argument("--enc-tol", type=float, default=1e-6)
    args = ap.parse_args()

    from zhuoji.dushan import DushanConfig, DushanGame
    from zhuoji.encoder import encode
    from zhuoji.rust_search import deep_q, snapshot_of

    # ---- 教师（ORT-CPU，1 线程，与 Rust 侧同 dylib 同 EP）----
    from gen_exit import load_net
    tea_net, _ = load_net(args.teacher, args.device)
    if str(args.device).lower().startswith("ort"):
        from zhuoji.forward import OrtForward, build_forward
        build_forward(tea_net, args.device,
                      stem=getattr(tea_net, "onnx_stem", None),
                      threads=1).warm(OrtForward.LADDER)

    # ---- 1. 采样状态 ----
    rng = random.Random(args.seed0)
    states = []  # (game, p, acts, seed)
    deals = 0
    while len(states) < args.states and deals < 4000:
        deals += 1
        g = DushanGame(DushanConfig(), seed=args.seed0 + deals)
        steps = 0
        while g.result is None and steps < 600:
            acts = g.legal_actions()
            if not acts:
                break
            if len(acts) >= 2 and len(states) < args.states:
                states.append((g, g.actor(), list(acts), 10_000 + len(states)))
            g.step(acts[rng.randrange(len(acts))] if len(acts) > 1 else acts[0])
            steps += 1
    print(f"[pairs] 采样 {len(states)} 状态（{deals} 局）", flush=True)

    # ---- 2. Python 参考 ----
    jl = ROOT / "runs/pairs_states.jsonl"
    jl.parent.mkdir(parents=True, exist_ok=True)
    T, G, HD, MK, QQ, codes_all = [], [], [], [], [], []
    with jl.open("w") as f:
        for g, p, acts, seed in states:
            o = encode(g, p, 3)
            q, _st = deep_q(tea_net, g, p, list(acts), n_det=32, ff_cap=24,
                            topk=3, ply=3, seed=seed, device=args.device)
            codes = [act_code(a) for a in acts]
            f.write(json.dumps({"snap": snapshot_of(g, p), "p": p,
                                "legal": codes, "seed": seed}) + "\n")
            T.append(np.asarray(o.tiles, dtype=np.float32).reshape(-1))
            G.append(np.asarray(o.glob, dtype=np.float32).reshape(-1))
            HD.append(HEAD_ID[o.head])
            MK.append(np.asarray(o.mask, dtype=bool).reshape(-1))
            QQ.append(np.asarray(q, dtype=np.float64).reshape(-1))
            codes_all.append(codes)

    # ---- 3. Rust 对照 ----
    ann = ROOT / "rust_engine/target/release/annotate"
    env = dict(**__import__("os").environ)
    env.setdefault("ORT_DYLIB_PATH", str(
        ROOT / ".venv/lib/python3.13/site-packages/onnxruntime/capi/libonnxruntime.1.31.1.dylib"))
    enc_out = ROOT / "runs/pairs_enc_rust.jsonl"
    q_out = ROOT / "runs/pairs_q_rust.jsonl"
    subprocess.run([str(ann), "--check-encode", str(jl), str(enc_out)],
                   check=True, env=env)
    subprocess.run([str(ann), "--teacher", args.teacher_onnx, "--device", "cpu",
                    "--check-q", str(jl), str(q_out)], check=True, env=env)

    # ---- 4. 比较 ----
    enc_lines = [json.loads(s) for s in enc_out.read_text().splitlines() if s.strip()]
    q_lines = [json.loads(s) for s in q_out.read_text().splitlines() if s.strip()]
    assert len(enc_lines) == len(states) == len(q_lines)

    e_max_t = e_max_g = 0.0
    head_bad = mask_bad = 0
    for i, rec in enumerate(enc_lines):
        rt = np.asarray(rec["tiles"], dtype=np.float64).reshape(-1)
        rg = np.asarray(rec["glob"], dtype=np.float64).reshape(-1)
        e_max_t = max(e_max_t, float(np.abs(rt - T[i]).max()))
        e_max_g = max(e_max_g, float(np.abs(rg - G[i]).max()))
        if int(rec["head"]) != int(HD[i]):
            head_bad += 1
        rmask = np.asarray(rec["mask"], dtype=bool).reshape(-1)
        rlen = int(rec.get("mask_len", rmask.shape[0]))
        if rlen != MK[i].shape[0] or not np.array_equal(rmask[:rlen], MK[i]):
            mask_bad += 1

    q_max = 0.0
    q_bad = 0
    for i, rec in enumerate(q_lines):
        rq = np.asarray(rec["q"], dtype=np.float64).reshape(-1)
        if rq.shape != QQ[i].shape:
            q_bad += 1
            continue
        d = float(np.abs(rq - QQ[i]).max()) if rq.size else 0.0
        q_max = max(q_max, d)
        if d > args.q_tol:
            q_bad += 1

    print("==== 对拍结果 ====")
    print(f"encode tiles max|Δ| = {e_max_t:.3e}   glob max|Δ| = {e_max_g:.3e}")
    print(f"head mismatch = {head_bad}/{len(states)}   mask mismatch = {mask_bad}/{len(states)}")
    print(f"Q max|Δ| = {q_max:.3e}   Q 超容差({args.q_tol:g})状态 = {q_bad}/{len(states)}")
    ok = (e_max_t <= args.enc_tol and e_max_g <= args.enc_tol
          and head_bad == 0 and mask_bad == 0 and q_bad == 0)
    print("PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
