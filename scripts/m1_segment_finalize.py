"""M1 分段训练的**段末一键收尾**：判读 + 归档 + 核对。

为什么需要它
------------
2026-10-12 的教训：`models/_ckpts/` 被 `.gitignore`（第 34 行）忽略、**没有任何备份**，
而阶段 B 的配对赛判读完全依赖它。那次 5 检查点 × 600 副的配对赛跑到 100/600 就断了
（结合内存峰值 30.13/31.8 GB，大概率被内存打崩），当时一度以为检查点丢了。

所以把"段末"固化成一步：

1. **判读**：跑 `m1_segment_judge.py`（阶段 A 健康度 + 阶段 B 配对赛）；
2. **归档**：把 `models/_ckpts/{prefix}_it*.pt`、判读 JSON、训练 stdout、history、
   资源打点 CSV 一并收进 `reports/archive/{run}_{时间戳}/`；
   大文件用**硬链接**（同盘零拷贝、删原件也不影响归档内容），失败才退回复制；
3. **核对**：打印归档清单与体积，并给出"这一段判读结论"的落点。

用法::

    # 段末标准流程（判读 1200 副 + 归档）
    python scripts/m1_segment_finalize.py --run m1_s1 --ckpt-prefix dushan_m1_s1

    # 只归档（判读已经跑过）
    python scripts/m1_segment_finalize.py --run m1_s1 --ckpt-prefix dushan_m1_s1 --skip-judge

    # 冒烟那种小判读
    python scripts/m1_segment_finalize.py --run m1_smoke --ckpt-prefix dushan_m1_smoke --deals 600
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CKPT_DIR = ROOT / "models" / "_ckpts"
ARCHIVE = ROOT / "reports" / "archive"
PY = sys.executable


def discover_ckpts(prefix: str) -> list[Path]:
    """在 models/_ckpts/ 里找该段的存档点。"""
    if not CKPT_DIR.is_dir():
        return []
    hits = sorted(CKPT_DIR.glob(f"{prefix}_it*.pt"))
    if not hits:
        # 前缀猜错时给点线索，而不是静默返回空
        near = sorted({p.name.rsplit("_it", 1)[0] for p in CKPT_DIR.glob("*_it*.pt")})
        raise SystemExit(f"在 {CKPT_DIR} 找不到 {prefix}_it*.pt\n"
                         f"  现有前缀：{near[:12]}{' …' if len(near) > 12 else ''}\n"
                         f"  用 --ckpt-prefix 显式指定（通常 = --out 的文件名去掉 .pt）")
    return hits


def link_or_copy(src: Path, dst: Path) -> str:
    """优先硬链接（零拷贝），失败退回复制。返回实际用的方式。

    2026-10-20 修复：``extra_files`` 里 ``judge_json`` 与 ``reports/{run}_judge.json``
    是同一路径，会被归档两次；旧实现在第二次走 ``dst.unlink()``，而沙箱把 unlink
    重定向到回收站，环境里的 SHFileOperationW 会失败（0x2），整个归档 process 崩掉。
    改为：同文件直接跳过；需要覆盖时先 unlink，失败则原地覆写内容。
    """
    if dst.exists():
        try:
            if os.path.samefile(src, dst):
                return "link"          # 已经是同一个 inode（硬链接或同一文件）
        except OSError:
            pass
        try:
            dst.unlink()
        except OSError:
            shutil.copyfile(src, dst)   # 退路：原地覆写，不依赖 unlink/回收站
            return "copy"
    try:
        os.link(src, dst)
        return "link"
    except OSError:
        shutil.copy2(src, dst)
        return "copy"


def run_judge(args) -> Path | None:
    """跑段末判读，返回 JSON 产物路径。"""
    out_json = ROOT / "reports" / f"{args.run}_judge.json"
    cmd = [PY, str(ROOT / "scripts" / "m1_segment_judge.py"),
           "--run", args.run,
           "--ckpt-prefix", args.ckpt_prefix,
           "--deals", str(args.deals),
           "--threads", str(args.threads)]
    if args.extra:
        cmd += ["--extra", args.extra]
    print(f"[finalize] 判读：{' '.join(cmd)}", flush=True)
    r = subprocess.run(cmd, cwd=str(ROOT))
    if r.returncode != 0:
        print(f"[finalize] ⚠️ 判读退出码 {r.returncode}，仍继续归档", flush=True)
    return out_json if out_json.exists() else None


def archive(run: str, prefix: str, ckpts: list[Path], extra_files: list[Path]) -> Path:
    """把检查点与本段产物收进 reports/archive/{run}_{ts}/。"""
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    dst_dir = ARCHIVE / f"{run}_{stamp}"
    (dst_dir / "ckpts").mkdir(parents=True, exist_ok=True)

    n_link = n_copy = 0
    total = 0
    for src in ckpts:
        how = link_or_copy(src, dst_dir / "ckpts" / src.name)
        n_link += how == "link"
        n_copy += how == "copy"
        total += src.stat().st_size
    for src in extra_files:
        if src.exists() and src.stat().st_size > 0:
            link_or_copy(src, dst_dir / src.name)

    manifest = {
        "run": run, "ckpt_prefix": prefix,
        "archived_at": datetime.now().isoformat(timespec="seconds"),
        "ckpts": [p.name for p in ckpts],
        "ckpt_count": len(ckpts),
        "ckpt_bytes": total,
        "extra": [p.name for p in extra_files if p.exists() and p.stat().st_size > 0],
        "note": "检查点用硬链接归档（同盘零拷贝）；删除 models/_ckpts 不影响本目录内容",
    }
    (dst_dir / "MANIFEST.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[finalize] 已归档 {len(ckpts)} 个检查点（{total / 1024 ** 2:.1f} MB，"
          f"硬链接 {n_link} / 复制 {n_copy}）→ {dst_dir}", flush=True)
    return dst_dir


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True,
                    help="段名，如 m1_s1 / m1_smoke（用于定位 reports/{run}_history.json 等）")
    ap.add_argument("--ckpt-prefix", required=True,
                    help="存档点前缀，通常 = --out 的文件名去掉 .pt（如 dushan_m1_s1）")
    ap.add_argument("--deals", type=int, default=1200, help="配对赛副数（阶段 B）")
    ap.add_argument("--threads", type=int, default=24)
    ap.add_argument("--extra", default="dushan_big_s11",
                    help="阶段 B 的额外参照模型（M1 主门槛是 vs s11）")
    ap.add_argument("--skip-judge", action="store_true", help="跳过判读，只归档")
    args = ap.parse_args()

    t0 = time.time()
    ckpts = discover_ckpts(args.ckpt_prefix)
    print(f"[finalize] 段 {args.run}：找到 {len(ckpts)} 个检查点"
          f"（{ckpts[0].name} … {ckpts[-1].name}）", flush=True)

    judge_json = None
    if not args.skip_judge:
        judge_json = run_judge(args)

    extra_files = [
        ROOT / "reports" / f"{args.run}_judge.json",
        ROOT / "reports" / f"{args.run}_judge_duel.json",
        ROOT / "reports" / f"{args.run}_history.json",
        ROOT / "reports" / f"_{args.run}.out",
        ROOT / "reports" / f"_res_monitor_{args.run}.csv",
        judge_json,
    ]
    dst = archive(args.run, args.ckpt_prefix, ckpts, [f for f in extra_files if f])

    print(f"\n[finalize] ✅ 段 {args.run} 收尾完成，用时 {time.time() - t0:.0f}s")
    print(f"  归档目录：{dst}")
    judge_path = ROOT / "reports" / f"{args.run}_judge.json"
    if judge_path.exists():
        print(f"  判读产物：{judge_path}（{judge_path.stat().st_size} 字节）")
    else:
        print(f"  判读产物：⚠️ 不存在（{judge_path}）—— 本段还没判读，"
              "去掉 --skip-judge 再跑一次")
    print("  下一步：把判读结论写进 docs/P3-M1训练执行计划.md 的段末验收行，"
          "再决定是否进下一段。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
