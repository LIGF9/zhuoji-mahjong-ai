"""把项目打包成可带到新机器的迁移包。

本项目不是 git 仓库，也没有 CI，跨机器只能手工搬。这个脚本把「跑起来所需的
全部文件」压成一个 zip，并保证解压后即可用 ``scripts/migration_bootstrap.py``
自检。

打包内容
--------
* 代码：``zhuoji/`` ``scripts/`` ``tests/`` ``web/`` ``docs/``
* 权重：``models/``（全部历史检查点，55MB）
* 数据：``data/``（BC 数据集 + 分片，99MB）
* 报告：``reports/``（结论 md/json/log，剔除 UI 截图快照 11MB）
* 根文件：``README.md`` ``requirements.txt``
* 额外注入：包根 ``MIGRATION.md``（迁移手册）

排除：``__pycache__`` / ``*.pyc`` / ``.tmp_*`` / ``reports/ui_snaps`` /
``migration``（输出目录自身）/ ``reports/_*_tmp``

用法::

    python scripts/make_migration_bundle.py
    python scripts/make_migration_bundle.py --out D:/bundle --no-data
    python scripts/make_migration_bundle.py --list      # 只列清单不打包
"""
from __future__ import annotations

import argparse
import time
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

TOP_DIRS = ["zhuoji", "scripts", "tests", "web", "docs", "models", "data", "reports"]
TOP_FILES = ["README.md", "requirements.txt"]

EXCLUDE_DIR_NAMES = {"__pycache__", "migration", "ui_snaps", ".git", ".idea"}
EXCLUDE_SUFFIXES = {".pyc", ".pyo"}
EXCLUDE_PREFIXES = (".tmp_",)


def _keep(p: Path) -> bool:
    if any(part in EXCLUDE_DIR_NAMES for part in p.parts):
        return False
    if p.suffix.lower() in EXCLUDE_SUFFIXES:
        return False
    if p.name.startswith(EXCLUDE_PREFIXES):
        return False
    return True


def collect(include_data: bool = True, include_reports: bool = True) -> list[Path]:
    files: list[Path] = []
    for name in TOP_DIRS:
        if name == "data" and not include_data:
            continue
        if name == "reports" and not include_reports:
            continue
        base = ROOT / name
        if not base.is_dir():
            continue
        files += [p for p in sorted(base.rglob("*")) if p.is_file() and _keep(p)]
    for name in TOP_FILES:
        f = ROOT / name
        if f.is_file():
            files.append(f)
    return files


def human(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.1f}{unit}"
        n /= 1024
    return f"{n:.1f}TB"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "migration"),
                    help="输出目录（默认 ./migration）")
    ap.add_argument("--name", default=None, help="zip 文件名")
    ap.add_argument("--no-data", action="store_true", help="不打 data/（省 99MB）")
    ap.add_argument("--no-reports", action="store_true", help="不打 reports/")
    ap.add_argument("--list", action="store_true", help="只列清单")
    ap.add_argument("--compresslevel", type=int, default=6)
    args = ap.parse_args()

    files = collect(include_data=not args.no_data,
                    include_reports=not args.no_reports)
    raw = sum(f.stat().st_size for f in files)
    print(f"待打包 {len(files)} 个文件，原始体积 {human(raw)}")

    if args.list:
        for f in files:
            print(f"  {f.relative_to(ROOT).as_posix():<60}{human(f.stat().st_size):>10}")
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

    print(f"-> {zip_path}  ({human(zip_path.stat().st_size)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
