"""把项目打包成可带到新机器的迁移包。

背景
----
项目已经是一个 git 仓库（``git@github.com:LIGF9/zhuoji-mahjong-ai.git``），
但 git 只跟踪「跑得起来 + 训得出来」的复现必要文件（约 15 MB）；大量
**未入库的产物**（历代权重、周期性检查点、训练日志）只存在于本机磁盘，
跨机器必须靠这个脚本打包。

通道选择
--------
1. **代码 + 文档**：优先 ``git push``（走 SSH）→ 新机 ``git clone``，
   比打包快得多，还能带回 git 历史；
2. **权重 + 日志**（本脚本负责）：``models/`` 里 9 个基础权重之外的
   历代权重、``models/_ckpts/`` 周期检查点、``reports/`` 日志，全在
   本机，只能离线搬。

两件事都做完，新机器才等于「旧机器的完整副本」。

打包内容
--------
* 代码：``zhuoji/`` ``scripts/`` ``tests/`` ``web/`` ``docs/``
* 权重：``models/``（全部检查点，含 ``_ckpts/`` 周期快照）
* 数据：``data/``（若存在；2026-10-09 已删除，可再生成）
* 报告：``reports/``（结论 md/json/log）
* 根文件：``README.md`` ``requirements.txt``
* 额外注入：包根 ``MIGRATION.md``（迁移手册）+ ``MANIFEST.txt``（清单）

**正在训练的产物会被跳过**：训练进程持续写入 ``.pt`` / ``.log`` / ``.json``，
拷到一半的文件是损坏的（尤其 ``.pt``）。脚本默认跳过「最近 ``--skip-recent-min``
分钟内被修改的文件」，也可用 ``--exclude-glob`` 精确指定。

排除：``__pycache__`` / ``*.pyc`` / ``.tmp_*`` / ``reports/ui_snaps`` /
``migration``（输出目录自身）/ ``.git``（走 clone 通道）

用法::

    python scripts/make_migration_bundle.py                    # 完整包
    python scripts/make_migration_bundle.py --no-ckpts         # 省 218MB
    python scripts/make_migration_bundle.py --list             # 只列清单
    python scripts/make_migration_bundle.py --exclude-glob "models/dushan_b2c_*"
"""
from __future__ import annotations

import argparse
import fnmatch
import time
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

TOP_DIRS = ["zhuoji", "scripts", "tests", "web", "docs", "models", "data", "reports"]
TOP_FILES = ["README.md", "requirements.txt"]

EXCLUDE_DIR_NAMES = {"__pycache__", "migration", "ui_snaps", ".git", ".idea"}
EXCLUDE_SUFFIXES = {".pyc", ".pyo"}
EXCLUDE_PREFIXES = (".tmp_",)


def human(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.1f}{unit}"
        n /= 1024
    return f"{n:.1f}TB"


def _matches_any(rel: str, globs: list[str]) -> bool:
    """``rel`` 为相对 ROOT 的 posix 路径；用 fnmatch 支持 ``*`` / ``**``。"""
    for g in globs:
        if fnmatch.fnmatch(rel, g) or fnmatch.fnmatch(Path(rel).name, g):
            return True
    return False


def _keep(p: Path, *, exclude_globs: list[str], no_ckpts: bool,
          skip_recent_min: float, now: float) -> tuple[bool, str]:
    """返回 (是否保留, 跳过原因)。"""
    if any(part in EXCLUDE_DIR_NAMES for part in p.parts):
        return False, "excluded-dir"
    if p.suffix.lower() in EXCLUDE_SUFFIXES:
        return False, "bytecode"
    if p.name.startswith(EXCLUDE_PREFIXES):
        return False, "tmp-prefix"
    if no_ckpts and "_ckpts" in p.parts:
        return False, "no-ckpts"
    rel = p.relative_to(ROOT).as_posix()
    if exclude_globs and _matches_any(rel, exclude_globs):
        return False, "exclude-glob"
    # 「最近写入」检查只作用于 models/ 与 reports/ ——
    # 只有这两处会被训练进程持续写入；源码/文档目录不该被时间窗口误伤。
    if skip_recent_min > 0 and p.parts[0] in ("models", "reports"):
        try:
            age_min = (now - p.stat().st_mtime) / 60.0
        except OSError:
            return False, "stat-failed"
        if age_min < skip_recent_min:
            return False, "recent-write"
    return True, ""


def collect(include_data: bool = True, include_reports: bool = True,
            no_ckpts: bool = False, exclude_globs: list[str] | None = None,
            skip_recent_min: float = 10.0) -> tuple[list[Path], list[tuple[str, str]]]:
    """收集待打包文件；同时返回被跳过的文件与原因（便于核对）。"""
    exclude_globs = exclude_globs or []
    now = time.time()
    files: list[Path] = []
    skipped: list[tuple[str, str]] = []

    for name in TOP_DIRS:
        if name == "data" and not include_data:
            continue
        if name == "reports" and not include_reports:
            continue
        base = ROOT / name
        if not base.is_dir():
            continue
        for p in sorted(base.rglob("*")):
            if not p.is_file():
                continue
            ok, why = _keep(p, exclude_globs=exclude_globs, no_ckpts=no_ckpts,
                            skip_recent_min=skip_recent_min, now=now)
            if ok:
                files.append(p)
            elif why in ("recent-write", "exclude-glob", "stat-failed"):
                # 只有「本该打包但被动态跳过」的才值得汇报
                skipped.append((p.relative_to(ROOT).as_posix(), why))

    for name in TOP_FILES:
        f = ROOT / name
        if f.is_file():
            ok, why = _keep(f, exclude_globs=exclude_globs, no_ckpts=no_ckpts,
                            skip_recent_min=skip_recent_min, now=now)
            if ok:
                files.append(f)
            else:
                skipped.append((name, why))
    return files, skipped


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "migration"),
                    help="输出目录（默认 ./migration）")
    ap.add_argument("--name", default=None, help="zip 文件名")
    ap.add_argument("--no-data", action="store_true", help="不打 data/")
    ap.add_argument("--no-reports", action="store_true", help="不打 reports/")
    ap.add_argument("--no-ckpts", action="store_true",
                    help="不打 models/_ckpts/ 周期检查点（省约 218MB）")
    ap.add_argument("--exclude-glob", action="append", default=[],
                    help="排除匹配该 glob 的文件（相对项目根，可多次）")
    ap.add_argument("--skip-recent-min", type=float, default=10.0,
                    help="跳过最近 N 分钟内被写入的文件（默认 10，防拷到在途产物；0=关闭）")
    ap.add_argument("--list", action="store_true", help="只列清单")
    ap.add_argument("--compresslevel", type=int, default=6)
    args = ap.parse_args()

    files, skipped = collect(include_data=not args.no_data,
                             include_reports=not args.no_reports,
                             no_ckpts=args.no_ckpts,
                             exclude_globs=args.exclude_glob,
                             skip_recent_min=args.skip_recent_min)
    raw = sum(f.stat().st_size for f in files)
    print(f"待打包 {len(files)} 个文件，原始体积 {human(raw)}")
    if skipped:
        print(f"\n已跳过 {len(skipped)} 个「在途/排除」文件（训练写入中或命中 --exclude-glob）：")
        for rel, why in skipped[:30]:
            print(f"  [{why}] {rel}")
        if len(skipped) > 30:
            print(f"  ... 其余 {len(skipped) - 30} 个略")
        print()

    if args.list:
        by_top: dict[str, list[Path]] = {}
        for f in files:
            by_top.setdefault(f.relative_to(ROOT).parts[0], []).append(f)
        for top, fs in sorted(by_top.items()):
            size = sum(x.stat().st_size for x in fs)
            print(f"  {top:<12}{len(fs):>5} 个文件  {human(size):>10}")
        return 0

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    name = args.name or f"zhuoji-mahjong_{time.strftime('%Y%m%d')}.zip"
    zip_path = out_dir / name

    manual = ROOT / "docs" / "迁移到新机器.md"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED,
                         compresslevel=args.compresslevel) as zf:
        for f in files:
            zf.write(f, f.relative_to(ROOT).as_posix())
        if manual.is_file():
            zf.write(manual, "MIGRATION.md")
        zf.writestr("BOOTSTRAP_README.txt",
                    "新机器上解压后，在本目录执行：\n\n"
                    "    python scripts/migration_bootstrap.py --fix-paths --check\n\n"
                    "详见 MIGRATION.md。\n")
        zf.writestr("MANIFEST.txt",
                    f"zhuoji-mahjong 迁移包\n"
                    f"生成时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n"
                    f"文件数  : {len(files)}\n"
                    f"原始体积: {human(raw)}\n"
                    f"跳过    : {len(skipped)} 个在途/排除文件\n\n"
                    + "\n".join(sorted(f.relative_to(ROOT).as_posix() for f in files)))

    size = zip_path.stat().st_size
    print(f"-> {zip_path}  ({human(size)})")

    # ---- 完整性校验：zip CRC + 关键文件存在 ----
    print("\n校验中 ...")
    with zipfile.ZipFile(zip_path) as zf:
        bad = zf.testzip()
        names = set(zf.namelist())
    key = ["MIGRATION.md", "MANIFEST.txt", "requirements.txt", "README.md",
           "scripts/selfplay.py", "scripts/night_plan_b_stage2.py",
           "scripts/keep_awake.py", "scripts/migration_bootstrap.py",
           "scripts/make_migration_bundle.py",
           "zhuoji/encoder_v2.py", "zhuoji/net.py", "zhuoji/encoder.py",
           "models/dushan_big_s11.pt", "models/dushan_big_s12.pt",
           "models/dushan_b2_p_abc.pt", "docs/P1-阶段2-消融结论.md"]
    missing = [k for k in key if k not in names]
    if bad:
        print(f"  ✗ CRC 校验失败: {bad}")
        return 1
    if missing:
        print(f"  ✗ 关键文件缺失: {missing}")
        return 1
    print(f"  ✓ CRC 全通过；关键文件 {len(key)}/{len(key)} 齐备")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
