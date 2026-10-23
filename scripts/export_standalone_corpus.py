# -*- coding: utf-8 -*-
"""A2/A3 差分对拍基准：从 Python 引擎导出随机对局语料。

每局记录：
  config   — DushanConfig 全字段
  wall     — 初始牌墙（发牌顺序），JS 侧按墙重放，不移植 MT19937
  dealer   — 庄家
  steps[]  — 每步：legal_actions（JS 必须产出完全一致的合法动作表）、
             action（Python 随机选择的动作）、state（引擎全量实例字段 dump）
  obs[]    — 每 obs_every 步采样一次 encode_v3 观测：
             head / mask / tiles+glob 的 base64 float32 / torch 模型四头 logits
             （JS 侧编码器逐位对拍 + 编码→ORT 端到端对拍共用）

输出 JSON 到 reports/standalone_corpus.json。
"""
from __future__ import annotations

import argparse
import base64
import dataclasses
import json
import os
import random
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from zhuoji.dushan import DushanConfig, DushanGame  # noqa: E402
from zhuoji.encoder import encode  # noqa: E402
from zhuoji.rules import Phase  # noqa: E402


def dump_state(g: DushanGame) -> dict:
    """引擎全量实例字段的可 JSON 化 dump（排除 rng/cfg/派生）。"""
    out: dict = {}
    for k, v in sorted(g.__dict__.items()):
        if k in ("rng", "cfg"):
            continue
        out[k] = _jsonable(v)
    return out


def _jsonable(v):
    if isinstance(v, (int, float, str, bool)) or v is None:
        return v
    if isinstance(v, (list, tuple)):
        return [_jsonable(x) for x in v]
    if isinstance(v, set):
        return {"__set__": sorted(_jsonable(x) for x in v)}
    if isinstance(v, dict):
        return {"__dict__": [[_jsonable(a), _jsonable(b)] for a, b in v.items()]}
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return float(v)
    if dataclasses.is_dataclass(v):
        return {"__dc__": dataclasses.asdict(v)}
    # Phase 等 Enum
    if hasattr(v, "value"):
        return str(v.value)
    return f"<{type(v).__name__}>"


def b64_f32(arr: np.ndarray) -> str:
    return base64.b64encode(np.ascontiguousarray(arr, dtype="<f4").tobytes()).decode()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=6)
    ap.add_argument("--obs-every", type=int, default=4)
    ap.add_argument("--seed", type=int, default=31337)
    ap.add_argument("--model", default="models/standalone/dushan_m2_c4_it2280.onnx")
    ap.add_argument("--out", default="reports/standalone_corpus.json")
    args = ap.parse_args()

    import onnxruntime as ort
    sess = ort.InferenceSession(args.model, providers=["CPUExecutionProvider"])

    rng = random.Random(args.seed)
    corpus = {"seed": args.seed, "obs_every": args.obs_every, "games": []}

    for gi in range(args.games):
        cfg = DushanConfig()
        seed = rng.randrange(1 << 30)
        g = DushanGame(cfg, seed=seed)
        rec = {
            "seed": seed,
            "config": dataclasses.asdict(cfg),
            "wall": list(g.wall),
            "dealer": g.dealer,
            "steps": [],
            "obs": [],
        }
        si = 0
        while g.phase is not Phase.OVER:
            p = g.actor()
            obs = encode(g, p, 3)
            legal = [[a.kind, int(a.tile)] for a in g.legal_actions()]
            acts = g.legal_actions()
            if not acts:
                break
            act = acts[rng.randrange(len(acts))]
            step = {"legal": legal, "act": [act.kind, int(act.tile)],
                    "state": dump_state(g)}
            if si % args.obs_every == 0:
                out = sess.run(None, {
                    "tiles": obs.tiles.reshape(1, -1, 3, 9).astype(np.float32),
                    "glob": obs.glob.reshape(1, -1).astype(np.float32),
                })
                rec["obs"].append({
                    "step": si, "head": obs.head,
                    "mask": [int(m) for m in obs.mask],
                    "tiles": b64_f32(obs.tiles),
                    "glob": b64_f32(obs.glob),
                    "logits": [b64_f32(o) for o in out[:4]],
                })
            rec["steps"].append(step)
            g.step(act)
            si += 1
        rec["result"] = _jsonable(getattr(g, "result", None))
        corpus["games"].append(rec)
        print(f"game {gi}: seed={seed} steps={si} "
              f"result_type={(rec['result'] or {}).get('type')}")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(corpus, f, ensure_ascii=False, separators=(",", ":"))
    print(f"corpus -> {args.out} ({os.path.getsize(args.out) / 2**20:.1f} MiB)")


if __name__ == "__main__":
    main()
