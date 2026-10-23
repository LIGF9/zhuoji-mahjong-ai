# -*- coding: utf-8 -*-
"""段配方自检：把 `m1_launch_segment.build_cmd()` 生成的命令行解析一遍，
断言每个**关键参数的实际生效值**。

为什么需要它
------------
`build_cmd()` 是 `COMMON + (M2_COMMON if m2) + extra` 三段拼接，靠 **argparse
"后者覆盖前者"** 生效。这让 S1–S3 的历史配方可以一个字不动，代价是：

* **同一个参数会出现 2–3 次**，读命令行原文根本看不出哪个生效；
* 想改一个 M2 的值，必须记得**在 M2_COMMON 里再写一遍** —— 漏了就静默沿用旧值。

2026-10-14 就踩到了：计划 R6 明确要求 M2 段把 `--pool-mem-pct` 放到 92
（只留硬保护），但 M2_COMMON 里漏写 → 段命令里生效的仍是 COMMON 的 **75**，
而本机桌面基线已占 63.7%，75% 只剩 ~4 GB 余量 —— 正是那次"重建池风暴"的成因。
这类**配方漂移**不会报错、只会让一段 8 小时训练的隐含配置与文档不符，
所以必须有一个能一眼看穿"最终生效值"的自检。

运行：``python scripts/verify_segment_recipe.py``（全部段）
      ``python scripts/verify_segment_recipe.py --segment s4a``
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _load():
    spec = importlib.util.spec_from_file_location(
        "_mls", ROOT / "scripts" / "m1_launch_segment.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


# 每个段必须生效的参数（含 M2 专属）。"--X": "期望值"
EXPECT = {
    "s1": {"--iters": "800", "--anchor-warmup": "200", "--teacher-prob": "0.15",
           "--lr": "1.2e-05", "--init": "models/dushan_master.pt"},
    "s2": {"--iters": "800", "--teacher-prob": "0.08", "--lr": "1.2e-05"},
    "s3": {"--iters": "850", "--teacher-prob": "0.05", "--lr": "6e-06"},
    "s4a": {
        "--iters": "600", "--anchor-warmup": "0", "--lr": "1.2e-05", "--lr-min": "4e-6",
        # A1 熵控制器（η 是 0.5 不是初版的 4；ratio 是 0.95 不是计划里的 0.75
        # —— 实测稳态只有段首的 91.6%，0.75 不可达，会让 ec 静默钉在下限，见 C5）
        "--ent-eta": "0.5", "--ent-target-ratio": "0.95", "--ent-baseline-iters": "25",
        "--ent-deadzone": "0.03",
        # A2 ε 退火
        "--explore-eps": "0.03", "--explore-eps-end": "0.01", "--explore-anneal-iters": "300",
        # A3 滚动锚 + 率判据
        "--kl-target": "0.05", "--kl-relax-ratio": "0.25",
        "--kl-anchor-rolling": "150", "--kl-anchor-alpha": "0.3", "--kl-anchor-every": "4",
        # A4 KLT
        "--master-ckpt": "dushan_master", "--klt-beta": "0.4", "--klt-anneal-iters": "300",
        # A5 对手池
        "--master-seat-prob": "0.30", "--teacher-prob": "0.0", "--snapshot-pool": "",
        # A6 优势
        "--adv-mode": "gae", "--gae-lambda": "0.95", "--gamma": "1.0", "--adv-norm": "seat",
        # A7 吞吐
        "--pool-rss-gb": "16", "--pool-mem-pct": "92", "--pool-rebuild-every": "250",
        "--snapshot-keep": "4", "--eval-every": "40", "--eval-rounds": "120",
        "--snapshot-every": "16",
    },
    "s4b": {
        "--iters": "1200", "--klt-beta": "0.0", "--master-seat-prob": "0.20",
        "--lr-min": "3e-6", "--snapshot-pool": "dushan_m1_s3",
        "--pool-mem-pct": "92", "--eval-every": "40", "--eval-rounds": "120",
    },
}


def effective(cmd: list[str]) -> dict[str, str]:
    """把 argv 列表折成 {参数: 最后生效的值}。带 `--flag` 形式的开关值记 "<flag>"。"""
    out: dict[str, str] = {}
    i = 0
    while i < len(cmd):
        t = cmd[i]
        if t.startswith("--"):
            nxt = cmd[i + 1] if i + 1 < len(cmd) else None
            if nxt is None or nxt.startswith("--"):
                out[t] = "<flag>"
                i += 1
            else:
                out[t] = nxt
                i += 2
        else:
            i += 1
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--segment", default="", help="只查某一段；空 = 全部有 EXPECT 的段")
    ap.add_argument("--workers", type=int, default=12)
    args = ap.parse_args()

    mls = _load()
    segs = [args.segment] if args.segment else list(EXPECT)
    bad: list[str] = []
    for seg in segs:
        cfg = mls.SEGMENTS.get(seg)
        if cfg is None:
            bad.append(f"{seg}: 配方不存在")
            continue
        cmd = mls.build_cmd(seg, cfg, args.workers)
        eff = effective(cmd)
        want = EXPECT.get(seg, {})
        print(f"\n=== {seg}（{len(cmd)} token，{len(eff)} 个参数生效）===")
        dups = {}
        for k in cmd:
            if k.startswith("--"):
                dups[k] = cmd.count(k)
        multi = {k: v for k, v in dups.items() if v > 1}
        if multi:
            print("  被覆盖的参数（COMMON → 后续段）：" +
                  ", ".join(f"{k}×{v}" for k, v in sorted(multi.items())))
        for k, v in want.items():
            got = eff.get(k, "<缺失>")
            # 路径类参数：build_cmd 会拼成绝对路径（Windows 用反斜杠）→ 归一后只比后缀
            gots = got.replace(chr(92), "/")
            ok = (gots.endswith(v) if ("/" in v and got != "<缺失>") else got == v)
            print(f"  {'PASS' if ok else 'FAIL'}  {k:<22} 生效 {got!r:<18} 期望 {v!r}")
            if not ok:
                bad.append(f"{seg} {k}: 生效 {got!r} ≠ 期望 {v!r}")

        # 额外结构断言
        # ★ 每个段都必须带 --log-file：训练进程自开日志，不依赖父进程传下来的 stdout
        #   句柄。2026-10-15 事故——launcher 用 `with log.open(...)` 起训练后立刻退出，
        #   日志只有 launcher 写的两行，训练输出**静默消失**。
        if not eff.get("--log-file"):
            bad.append(f"{seg}: build_cmd 必须给 --log-file <日志路径>（训练自开日志）")
        if cfg.get("m2"):
            a = cfg.get("extra") or []
            if "--klt-beta" not in a:
                bad.append(f"{seg}: m2 段必须在 extra 里显式给 --klt-beta")
            if "--lr-min" not in a:
                bad.append(f"{seg}: m2 段必须在 extra 里显式给 --lr-min")

    print("\n" + "=" * 72)
    if bad:
        print(f"✗ {len(bad)} 项失败：")
        for b in bad:
            print("   - " + b)
        return 1
    print("✓ 全部段配方核对通过")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
