#!/usr/bin/env python3
"""npz 标注分片合规校验（只读 · 轻量 · 可在 7×24 标注期安全运行）。

判据（docs/14天训练计划.md §6.1、自检 #6 / #10）：

  - 4 分片/轮，各 >= 24000 行（实产 25000）
  - tiles(N,63,27)、glob(N,130)、q/mask(N,28)
  - q 的 NaN 是「非法动作槽位」约定：**NaN <-> mask == 0**
    overlap = (NaN & mask!=0) 必须为 0，gap = (~NaN & mask==0) 必须为 0。
    ⚠️ 按 ``NaN == mask`` 校验会全部 False，属脚本口径错误，不是数据异常。

实现要点：只读 zip 成员里的 npy header（v1 前缀 10B / v2+ 12B）取 shape，
不解压 ~170MB 的 tiles；仅解压 q/mask 两个小数组做互补校验。
标注器四分片逐个落盘，写盘途中读会抛 BadZipFile（非数据损坏），
本脚本明确提示等 annotate 退出或分片大小稳定后再核（自检 #2 竞态）。

本脚本不加载模型、不启对局、不写任何文件，不与训练/标注抢 CPU。

用法::

    ./.venv/bin/python tools/check_npz.py                # 全部轮次
    ./.venv/bin/python tools/check_npz.py --round r05    # 只核某一轮
"""
from __future__ import annotations

import argparse
import ast
import io
import zipfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
MIN_ROWS = 24000


def npy_shape(fh) -> tuple[int, ...]:
    """从 npy 头部读 shape，不解压数组本体。"""
    head = fh.read(512)
    if head[:6] != b"\x93NUMPY":
        raise ValueError("不是 npy 格式")
    off = 10 if head[6] == 1 else 12  # v1: 2B hlen；v2/v3: 4B hlen
    hlen = int.from_bytes(head[8:off], "little")
    return tuple(ast.literal_eval(head[off:off + hlen].decode("latin1").strip())["shape"])


def check(path: Path) -> tuple[bool, str]:
    try:
        with zipfile.ZipFile(path) as z:
            shapes = {}
            for n in z.namelist():
                with z.open(n) as fh:
                    shapes[n.replace(".npy", "")] = npy_shape(fh)
            q = np.load(io.BytesIO(z.read("q.npy")))
            mask = np.load(io.BytesIO(z.read("mask.npy")))
    except (zipfile.BadZipFile, OSError, ValueError, KeyError) as e:
        return False, (f"读取失败({type(e).__name__})：分片可能仍在写入，"
                       "等 annotate 退出 / 分片大小稳定后再核（非数据损坏）")

    nan, zero = np.isnan(q), (mask == 0)
    overlap, gap = int((nan & ~zero).sum()), int((~nan & zero).sum())
    t, g = shapes.get("tiles", (0,)), shapes.get("glob", (0,))
    ok = (
        t[1:] == (63, 27) and g[1:] == (130,) and t[0] >= MIN_ROWS
        and shapes.get("q", (0,))[1:] == (28,)
        and shapes.get("mask", (0,))[1:] == (28,)
        and q.shape == mask.shape and overlap == 0 and gap == 0
    )
    detail = (f"rows={t[0]} tiles={t} glob={g} q={shapes.get('q')} "
              f"mask={shapes.get('mask')} NaN={int(nan.sum())} "
              f"overlap={overlap} gap={gap}")
    return ok, detail


def main() -> int:
    ap = argparse.ArgumentParser(description="npz 标注分片合规校验（只读）")
    ap.add_argument("--round", dest="rnd", default=None, metavar="rNN",
                    help="只核指定轮次，如 r05；缺省核全部")
    args = ap.parse_args()

    pattern = f"runs/exit_{args.rnd}/exit_w*.npz" if args.rnd else "runs/exit_r*/exit_w*.npz"
    files = sorted(ROOT.glob(pattern))
    if not files:
        print(f"[WARN] 未匹配到分片：{pattern}")
        return 2

    print(f"检查 {len(files)} 个分片（{pattern}）")
    bad = []
    for f in files:
        ok, detail = check(f)
        print(f"  {f.relative_to(ROOT)}: {detail} -> {'OK' if ok else 'BAD'}")
        if not ok:
            bad.append(str(f.relative_to(ROOT)))
    print(f"SUMMARY: {len(files) - len(bad)}/{len(files)} compliant; bad={bad}")
    return 0 if not bad else 1


if __name__ == "__main__":
    raise SystemExit(main())
