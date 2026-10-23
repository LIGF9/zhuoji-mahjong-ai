#!/usr/bin/env python
"""按 night_autopilot_state.json 的 dyn_segs 配置接力「训练已停」的当前段。

背景
----
沙箱型执行环境会在每次工具调用结束时清理整棵进程树，总控（m2_night_autopilot.py）
在巡检结束后常被连带回收。此时 state 的 current 段（如 auto6rrrr）已经写好启动头、
却没有跑出任何 iter —— dyn_segs 里只有 auto4 在 m1_launch_segment.SEGMENTS 表中，
动态段只能在这里按 state 里的配方重建命令行。

做法
----
1. 读 night_autopilot_state.json → current 段 → dyn_segs[current]；
2. 用段内最新周期存档（models/_ckpts/<stem>_itNNNN.pt）做 --init；
3. 复用 m1_launch_segment.launch()：它会一并拉起 keep_awake / res_monitor、
   写 _{run}.pid、并做 8 s 存活确认；
4. foreground + wait=True —— 本进程就是训练的"宿主"，必须挂在平台后台任务里跑。

用法::

    python scripts/relay_auto6rrrr.py            # 按 state 的 current 段接力
    python scripts/relay_auto6rrrr.py --dry-run  # 只打印命令行
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import m1_launch_segment as L  # noqa: E402

STATE = ROOT / "reports" / "night_autopilot_state.json"


def latest_ckpt(stem: str) -> tuple[Path, int] | None:
    d = ROOT / "models" / "_ckpts"
    best: tuple[Path, int] | None = None
    for p in d.glob(f"{stem}_it*.pt"):
        m = re.search(r"_it(\d+)\.pt$", p.name)
        if not m:
            continue
        it = int(m.group(1))
        if best is None or it > best[1]:
            best = (p, it)
    return best


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    st = json.loads(STATE.read_text(encoding="utf-8"))
    seg = st["current"]
    d = st["dyn_segs"][seg]
    stem = Path(d["out"]).stem

    # 优先用本段自己的周期存档（真正"续训"）；本段一轮都没跑出时（如刚起就死）
    # 回落到 dyn_segs 指定的 --init（= 上一段的段末存档），等于把这一段重跑一遍。
    hit = latest_ckpt(stem)
    if hit is None:
        init_p = ROOT / d["init"]
        if not init_p.exists():
            print(f"[relay] ❌ 本段无存档，且 dyn_segs.init 不存在：{init_p}")
            return 2
        ckpt, it = init_p, 0
        print(f"[relay] 本段 0 轮产出 → 回落到 dyn_segs.init：{init_p.name}")
    else:
        ckpt, it = hit
    cfg = dict(iters=d["iters"], init=str(ckpt.relative_to(ROOT)).replace("\\", "/"),
               out=d["out"], warmup=d.get("warmup", 0),
               teacher_prob=d.get("teacher_prob", 0.0),
               temperature=d.get("temperature", 1.2), lr=d["lr"],
               needs="", m2=bool(d.get("m2")), extra=list(d.get("extra") or []))
    print(f"[relay] 段 {seg}：{cfg['iters']} 轮，起点 {ckpt.name}（iter {it}），"
          f"lr={cfg['lr']}，out={cfg['out']}", flush=True)

    if it >= cfg["iters"]:
        print(f"[relay] 起点轮次 {it} 已达段长 {cfg['iters']}，本段其实已跑完，"
              f"应交给判读而非续训")
        return 2

    if args.dry_run:
        print("  " + " ".join(L.build_cmd(seg, cfg, args.workers)))
        return 0

    return L.launch(seg, cfg, args.workers, foreground=True, settle=10.0, wait=True)


if __name__ == "__main__":
    raise SystemExit(main())
