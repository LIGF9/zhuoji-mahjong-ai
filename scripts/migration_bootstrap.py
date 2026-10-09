"""迁移引导：在新机器上修复硬编码路径 + 环境自检。

背景
----
本项目在旧机器上运行时留下了两处硬编码的绝对路径：
  1. ``scripts/shift_chain_*.py`` 里的 ``PY = r"...\\python.exe"``；
  2. ``scripts/ui_*_check.js`` 里的 ``const ROOT = 'C:/.../zhuoji-mahjong';``。
换机器后（用户名不同 / 盘符不同 / 目录不同）这两处必须改，否则值班链和
UI 检查脚本会直接失败。本项目不是 git 仓库，也没有安装脚本，所以用这个
脚本一次性修好。

用法（在项目根目录执行）::

    python scripts/migration_bootstrap.py --fix-paths
    python scripts/migration_bootstrap.py --check
    python scripts/migration_bootstrap.py --fix-paths --check
"""
from __future__ import annotations

import argparse
import importlib
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# 旧机器上残留的解释器绝对路径（任意 ``...python.exe`` 都视为待替换目标）
PY_RE = re.compile(r'^(PY\s*=\s*r?)"[^"]*python\.exe"', re.MULTILINE)
JS_ROOT_RE = re.compile(r"const ROOT\s*=\s*'[^']*';")

REQUIRED_FILES = [
    "README.md",
    "requirements.txt",
    "zhuoji/net.py",
    "zhuoji/encoder.py",
    "zhuoji/dushan.py",
    "scripts/selfplay.py",
    "scripts/duel.py",
    "scripts/score_checkpoints.py",
    "scripts/bench_scale.py",
    "web/dushan_server.py",
    "web/static/dushan.html",
    "models/dushan_master.pt",
]

OPTIONAL_FILES = [
    "data/bc_dushan_s7.npz",
    "data/bc_data.npz",
    "models/dushan_big_s11.pt",
]


def _rel(p: Path) -> str:
    try:
        return str(p.relative_to(ROOT)).replace("\\", "/")
    except ValueError:
        return str(p)


def _read(p: Path) -> str:
    """按原始换行读取，避免改写时把 LF 文件转成 CRLF。"""
    with open(p, "r", encoding="utf-8", newline="") as fh:
        return fh.read()


def _write(p: Path, text: str) -> None:
    with open(p, "w", encoding="utf-8", newline="") as fh:
        fh.write(text)


def fix_paths(apply: bool) -> int:
    """把硬编码路径改写成当前机器的实际路径。"""
    changed = 0
    py_exe = sys.executable
    root_posix = ROOT.as_posix()

    for f in sorted((ROOT / "scripts").glob("shift_chain_*.py")):
        text = _read(f)
        new, n = PY_RE.subn(lambda m: f'{m.group(1)}r"{py_exe}"', text)
        if n:
            print(f"[py]  {_rel(f)}  ({n} 处) -> {py_exe}")
            if apply:
                _write(f, new)
            changed += n

    for f in sorted((ROOT / "scripts").glob("ui_*_check.js")):
        text = _read(f)
        new, n = JS_ROOT_RE.subn(f"const ROOT = '{root_posix}';", text)
        if n:
            print(f"[js]  {_rel(f)}  ({n} 处) -> {root_posix}")
            if apply:
                _write(f, new)
            changed += n

    if changed == 0:
        print("未发现需要修复的硬编码路径（可能已修复或文件缺失）。")
    elif not apply:
        print(f"\n[dry-run] 共 {changed} 处待修复；加 --fix-paths 实际写入。")
    else:
        print(f"\n已修复 {changed} 处硬编码路径。")
    return 0


def check(verbose: bool = True) -> int:
    """环境自检：解释器 / 依赖 / 关键文件 / 轻量功能自测。"""
    problems: list[str] = []

    print(f"项目根目录: {ROOT}")
    print(f"解释器    : {sys.executable}")
    print(f"Python    : {sys.version.split()[0]}")

    if sys.version_info < (3, 10):
        problems.append("Python 版本过低（需 >= 3.10，项目在 3.13 上开发）")

    # ---- 依赖 ----
    versions = {}
    for mod in ("torch", "numpy"):
        try:
            m = importlib.import_module(mod)
            versions[mod] = getattr(m, "__version__", "?")
            print(f"{mod:<10}: {versions[mod]}")
        except Exception as exc:  # noqa: BLE001
            problems.append(f"缺少依赖 {mod}（{exc}）")

    if "torch" in versions:
        try:
            import torch
            print(f"torch 线程: {torch.get_num_threads()}  "
                  f"CUDA: {torch.cuda.is_available()}")
        except Exception:  # noqa: BLE001
            pass

    # ---- 文件 ----
    missing = [p for p in REQUIRED_FILES if not (ROOT / p).exists()]
    if missing:
        problems.append("缺少必需文件: " + ", ".join(missing))
    else:
        print(f"必需文件  : {len(REQUIRED_FILES)}/{len(REQUIRED_FILES)} 齐备")

    absent_opt = [p for p in OPTIONAL_FILES if not (ROOT / p).exists()]
    if absent_opt:
        print(f"可选文件缺失（不影响跑通）: {', '.join(absent_opt)}")

    # ---- 残留硬编码 ----
    stale = []
    for f in sorted((ROOT / "scripts").glob("shift_chain_*.py")):
        t = _read(f)
        m = PY_RE.search(t)
        if m and sys.executable not in m.group(0):
            stale.append(_rel(f))
    for f in sorted((ROOT / "scripts").glob("ui_*_check.js")):
        t = _read(f)
        m = JS_ROOT_RE.search(t)
        if m and ROOT.as_posix() not in m.group(0):
            stale.append(_rel(f))
    if stale:
        problems.append("仍有指向旧机器路径的文件: " + ", ".join(stale)
                        + "  → 运行 --fix-paths")
    else:
        print("路径      : 无旧机器残留")

    # ---- 功能自测：建模型 + 走一步前向 + 引擎落一局 ----
    if not problems:
        try:
            import torch
            sys.path.insert(0, str(ROOT))
            from zhuoji.net import NetConfig, build_model
            m = build_model(NetConfig(64, 8, 320))
            out = m(torch.randn(2, 15, 27), torch.randn(2, 70))
            assert out["discard"].shape[0] == 2
            print(f"网络前向  : OK（复现 big 网参数量 {m.n_params():,}）")

            from zhuoji.dushan import DushanConfig, DushanGame
            from zhuoji.bots import HeuristicBot
            bots = [HeuristicBot(seed=i + 1) for i in range(4)]
            g = DushanGame(DushanConfig(), seed=1)
            steps = 0
            while g.phase.value != "over" and steps < 4000:
                g.step(bots[g.actor()].choose(g))
                steps += 1
            assert g.phase.value == "over", "对局未能正常结束"
            delta = [round(d, 2) for d in g.score_delta]
            assert abs(sum(g.score_delta)) < 1e-6, f"分账不平: {delta}"
            print(f"引擎自测  : OK（一局 {steps} 步，分账 {delta}）")
        except Exception as exc:  # noqa: BLE001
            import traceback
            traceback.print_exc()
            problems.append(f"功能自测失败: {exc}")

    # ---- 训练吞吐基准（可选，慢） ----
    if verbose and not problems:
        print("\n提示: 跑 `python scripts/bench_scale.py` 可与旧机器对比吞吐。")

    print()
    if problems:
        print("自检未通过:")
        for p in problems:
            print(f"  - {p}")
        return 1
    print("自检通过：环境就绪。")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fix-paths", action="store_true",
                    help="改写 shift_chain_*.py / ui_*_check.js 中的绝对路径")
    ap.add_argument("--dry-run", action="store_true",
                    help="只显示将要修改的内容，不写入")
    ap.add_argument("--check", action="store_true", help="运行环境自检")
    ap.add_argument("--json", default=None, help="把自检结果写成 JSON")
    args = ap.parse_args()

    if not (args.fix_paths or args.check):
        ap.print_help()
        return 0

    if args.fix_paths:
        fix_paths(apply=not args.dry_run)
        print()

    rc = check() if (args.check or args.fix_paths) else 0
    if args.json:
        Path(args.json).write_text(json.dumps(
            dict(root=str(ROOT), python=sys.version.split()[0],
                 executable=sys.executable, ok=(rc == 0)),
            ensure_ascii=False, indent=2), encoding="utf-8")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
