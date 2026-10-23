"""迁移引导：在新机器上修复硬编码路径 + 环境自检。

背景
----
换机器后（用户名不同 / 盘符不同 / 目录不同）必须修两类硬编码绝对路径，
否则值班链、夜跑队列、接续监控和 UI 检查脚本都会直接失败：

  1. ``scripts/*.py`` 里的 ``PY = r"...\\python.exe"``
     （不只 ``shift_chain_*``——``shift_plan_a_s12.py``、``night_plan_b_stage2.py``、
     ``after_night_a4.py``、``after_a4_plan_c.py`` 同样含此变量）；
  2. ``scripts/ui_*_check.js`` 里的 ``const ROOT = 'C:/.../zhuoji-mahjong';``。

项目本身已是 git 仓库（``git@github.com:LIGF9/zhuoji-mahjong-ai.git``），但路径
是**本机绝对路径**，换机不会自动对上，所以仍需要这个脚本。

用法（在项目根目录执行；**用新机的 venv 解释器跑**，避免写入系统 Python 路径）::

    python scripts/migration_bootstrap.py --fix-paths
    python scripts/migration_bootstrap.py --check
    python scripts/migration_bootstrap.py --fix-paths --check
    python scripts/migration_bootstrap.py --fix-paths --py C:/path/to/venv/python.exe
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
#
# 注意 group(1) **只捕获 ``PY = ``**，前缀 r 用 ``r*`` 吞掉（不是 ``r?``）：
# 若把 ``r?`` 写进捕获组再在替换串里补一个 ``r``，源本是 ``PY = r"..."`` 的
# 文件会被改写成 ``PY = rr"..."``（2026-10-11 在 14700K 上实际踩到，
# 7 个脚本全部 SyntaxError）。用 ``r*`` 还有一个好处：它**能反向修复**
# 已经被写成 ``rr"..."`` 的文件，重跑一次 --fix-paths 即可自愈。
PY_RE = re.compile(r'^(PY\s*=\s*)r*"[^"]*python\.exe"', re.MULTILINE)
JS_ROOT_RE = re.compile(r"const ROOT\s*=\s*'[^']*';")

#: 修路径时扫描的脚本范围（曾经只扫 shift_chain_*，漏掉 4 个新脚本）
PY_GLOBS = ("*.py",)
JS_GLOBS = ("*.js",)

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


def _py_targets() -> list[Path]:
    files: list[Path] = []
    for g in PY_GLOBS:
        files += list((ROOT / "scripts").glob(g))
    return sorted(set(files))


def _js_targets() -> list[Path]:
    files: list[Path] = []
    for g in JS_GLOBS:
        files += list((ROOT / "scripts").glob(g))
    return sorted(set(files))


def fix_paths(apply: bool, py_exe: str | None = None) -> int:
    """把硬编码路径改写成当前机器的实际路径。"""
    changed = 0
    py_exe = py_exe or sys.executable
    root_posix = ROOT.as_posix()

    for f in _py_targets():
        text = _read(f)
        new, n = PY_RE.subn(lambda m: f'{m.group(1)}r"{py_exe}"', text)
        if n:
            print(f"[py]  {_rel(f)}  ({n} 处) -> {py_exe}")
            if apply:
                _write(f, new)
            changed += n

    for f in _js_targets():
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
    for f in _py_targets():
        t = _read(f)
        m = PY_RE.search(t)
        if m and sys.executable not in m.group(0):
            stale.append(_rel(f))
    for f in _js_targets():
        t = _read(f)
        m = JS_ROOT_RE.search(t)
        if m and ROOT.as_posix() not in m.group(0):
            stale.append(_rel(f))
    if stale:
        problems.append("仍有指向旧机器路径的文件: " + ", ".join(stale)
                        + "  → 运行 --fix-paths")
    else:
        print("路径      : 无旧机器残留")

    # ---- 脚本语法：被改写的 .py 必须仍能解析 ----
    # 这一项是补上 2026-10-11 的真实盲区：fix-paths 写出了 ``PY = rr"..."``，
    # 而旧版 check() 只 import 库模块（zhuoji/*）、从不解析 scripts/*.py，
    # 于是"自检全绿"却是 7 个脚本全 SyntaxError。改写过的文件必须过一遍 compile()。
    broken = []
    for f in _py_targets():
        try:
            compile(_read(f), str(f), "exec")
        except SyntaxError as exc:
            broken.append(f"{_rel(f)}:{exc.lineno}: {exc.msg}")
    if broken:
        problems.append("脚本语法错误: " + "; ".join(broken))
    else:
        print(f"脚本语法  : OK（{len(_py_targets())} 个 .py 全部可解析）")

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
                    help="改写 scripts/*.py 与 scripts/*.js 中的绝对路径")
    ap.add_argument("--dry-run", action="store_true",
                    help="只显示将要修改的内容，不写入")
    ap.add_argument("--py", default=None,
                    help="写入脚本的解释器路径（默认 sys.executable；"
                         "建议用新机的 venv python 运行本脚本）")
    ap.add_argument("--check", action="store_true", help="运行环境自检")
    ap.add_argument("--json", default=None, help="把自检结果写成 JSON")
    args = ap.parse_args()

    if not (args.fix_paths or args.check):
        ap.print_help()
        return 0

    if args.fix_paths:
        fix_paths(apply=not args.dry_run, py_exe=args.py)
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
