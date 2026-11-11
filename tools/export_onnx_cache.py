#!/usr/bin/env python3
"""为线上可选模型预导出「按 stem 隔离」的 ONNX 缓存（E-WEIGHT 根因③）。

背景（2026-11-10，§6.68 / §6.71）
--------------------------------
Web 搜索档的 Q 值由 ORT 会话算出，会话的权重文件由
``zhuoji.forward.ensure_onnx(net, stem)`` 决定。修 ``get_model`` 设
``net.onnx_stem = stem`` 之后，每个 stem 都需要一份自己的 ``.onnx``；
否则首次请求会在**玩家等待的路径上**现场导出 39 MB 的 v3 图（数秒~数十秒）。

本脚本把这件事挪到部署前一次性做掉，并做三项校验：
  1. 每个 stem 的 ONNX 都由**该 stem 的 .pt** 导出（不复用 adhoc 别名）；
  2. ONNX 输入维度 == 权重自带的 ``in_channels / global_dim``；
  3. ORT 会话对 (B=8, ch, 27) / (B=8, gdim) 能跑通且返回四个头。

用法：
    .venv/bin/python tools/export_onnx_cache.py            # 导出 + 校验
    .venv/bin/python tools/export_onnx_cache.py --check    # 只校验，不导出

只读/幂等：已存在且 mtime ≥ .pt 的缓存直接复用；--check 不写任何文件。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from zhuoji.forward import ensure_onnx  # noqa: E402
from zhuoji.net import build_from_ckpt  # noqa: E402

MODEL_DIR = ROOT / "models"
ONNX_DIR = MODEL_DIR / "onnx"
CHAMPIONS = MODEL_DIR / "champions.json"


def stems() -> list[str]:
    # expert_s7（2026-11-11）：离线版「进阶」档的 0.12M 小网，不在 champions.json
    # 里（它不是冠军，只是离线四档之一），但它同样被 offline/src/assets.rs
    # include_bytes! 编进 APK —— 缺 ONNX 会直接让离线 crate 编译失败。
    out = ["dushan_master", "expert_s7"]
    try:
        reg = json.loads(CHAMPIONS.read_text(encoding="utf-8"))
        out += [str(c["stem"]) for c in reg.get("champions", [])]
    except (OSError, ValueError, KeyError, TypeError):
        pass
    return list(dict.fromkeys(s for s in out if (MODEL_DIR / f"{s}.pt").exists()))


def onnx_input_dims(path: Path) -> tuple[int | None, int | None]:
    """从 ONNX 图里读 tiles/glob 的静态输入维度（动态批为 None）。"""
    try:
        import onnx  # noqa: PLC0415

        g = onnx.load(str(path), load_external_data=False).graph
        got: dict[str, list] = {}
        for x in g.input:
            dims = [d.dim_value or None for d in x.type.tensor_type.shape.dim]
            got[x.name] = dims
        t, gl = got.get("tiles", []), got.get("glob", [])
        return (t[1] if len(t) > 1 else None, gl[1] if len(gl) > 1 else None)
    except Exception:  # noqa: BLE001 - 没装 onnx 包就跳过这项静态校验
        return (None, None)


def ort_smoke(path: Path, ch: int, gdim: int) -> str:
    """按真实维度跑一次 B=8 前向，确认会话能吃下这张图。"""
    import onnxruntime as ort  # noqa: PLC0415

    so = ort.SessionOptions()
    so.log_severity_level = 3
    sess = ort.InferenceSession(str(path), so,
                                providers=["CPUExecutionProvider"])
    outs = sess.run(None, {
        "tiles": np.zeros((8, ch, 27), np.float32),
        "glob": np.zeros((8, gdim), np.float32)})
    return ",".join(str(np.asarray(o).shape) for o in outs)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="只校验现有缓存，不导出")
    args = ap.parse_args()

    ok = True
    for stem in stems():
        pt = MODEL_DIR / f"{stem}.pt"
        ck = torch.load(str(pt), map_location="cpu", weights_only=False)
        net, enc_v = build_from_ckpt(ck)
        net.onnx_stem = stem
        ch, gdim = int(net.in_channels), int(net.global_dim)
        target = ONNX_DIR / f"{stem}.onnx"

        if args.check and not target.exists():
            print(f"[MISS] {stem:22s} 缺 {target.name}", flush=True)
            ok = False
            continue
        if target.exists() and target.stat().st_mtime >= pt.stat().st_mtime:
            path, how = target, "复用"
        elif args.check:
            print(f"[STALE] {stem:22s} 缓存早于 .pt，需要重导", flush=True)
            ok = False
            continue
        else:
            path, how = Path(ensure_onnx(net, stem)), "新导出"
        if path.stem != stem:
            print(f"[FAIL] {stem:22s} 竟回落到 {path.name}（别名串用）", flush=True)
            ok = False
            continue

        d_ch, d_gl = onnx_input_dims(path)
        if d_ch is not None and (d_ch != ch or d_gl != gdim):
            print(f"[FAIL] {stem:22s} 图输入 {d_ch}/{d_gl} ≠ 权重 {ch}/{gdim}",
                  flush=True)
            ok = False
            continue
        try:
            shapes = ort_smoke(path, ch, gdim)
        except Exception as exc:  # noqa: BLE001
            print(f"[FAIL] {stem:22s} ORT 冒烟失败 {type(exc).__name__}: {exc}",
                  flush=True)
            ok = False
            continue
        digest = hashlib.sha256(path.read_bytes()).hexdigest()[:12]
        print(f"[ OK ] {stem:22s} {how:4s} enc={enc_v} {ch}/{gdim} "
              f"{path.stat().st_size / 1e6:.1f}MB sha={digest} heads={shapes}",
              flush=True)

    # 历史假档别名：adhoc 与 master 逐位相同。修根因后不应再被任何路径用到。
    adhoc = ONNX_DIR / "adhoc.onnx"
    master = ONNX_DIR / "dushan_master.onnx"
    if adhoc.exists() and master.exists():
        same = (hashlib.sha256(adhoc.read_bytes()).hexdigest()
                == hashlib.sha256(master.read_bytes()).hexdigest())
        print(f"[note] models/onnx/adhoc.onnx 与 dushan_master.onnx "
              f"{'仍逐位相同（历史假档别名，已不再被引用）' if same else '已不同'}",
              flush=True)
    print("VERDICT:", "PASS" if ok else "FAIL", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
