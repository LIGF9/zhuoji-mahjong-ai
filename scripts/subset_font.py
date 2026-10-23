"""手机端字体子集化（2026-10-12 #89）。

问题
----
``zhunyuan.ttf`` 全量 **9.9 MB**，手机端首访要整包下载，只有几十个字被真正用到。

做法
----
从游戏全部文案源收集实际用到的字符，生成 woff2 子集：

  · ``web/static/dushan.html`` 全文（含内联 JS 的全部 UI 文案）
  · 服务端 / 引擎 / 教练的 Python 源码里的**字符串字面量**（开局翻鸡、黄庄、
    一炮多响、包大牌面……这些动态流水文案不在 HTML 里）
  · ASCII 可打印字符 + 常用中文标点 + 全角符号

产出：``web/static/assets/font/zhunyuan_subset.woff2``（预计几百 KB）。

前端按设备分流（见 dushan.html 的 @font-face 注入脚本）：
手机/触屏 → 子集 woff2；桌面 → 原全量 ttf。子集外字符回退系统字体。

运行::

    python scripts/subset_font.py          # 生成子集
    python scripts/subset_font.py --check  # 只打印统计，不写文件
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC_HTML = ROOT / "web" / "static" / "dushan.html"
OUT_WOFF2 = ROOT / "web" / "static" / "assets" / "font" / "zhunyuan_subset.woff2"
SRC_TTF = ROOT / "web" / "static" / "assets" / "font" / "zhunyuan.ttf"

# 动态文案来源：服务端 / 引擎 / 教练（字符串字面量里的中文 = 游戏会推送的文字）
PY_SOURCES = [
    ROOT / "web" / "dushan_server.py",
    ROOT / "web" / "coach.py",
    ROOT / "zhuoji" / "dushan.py",
    ROOT / "zhuoji" / "fan.py",
    ROOT / "zhuoji" / "rules.py",
    ROOT / "zhuoji" / "bots.py",
    ROOT / "zhuoji" / "tiles.py",
    ROOT / "zhuoji" / "agent.py",
]
_STR_RE = re.compile(r"""('[^'\n]*'|"[^"\n]*"|'[^']*'|"[^"]*")""", re.S)

# ASCII 可打印 + 常用中文标点 / 全角符号（数字、字母、标点在 UI 里大量出现）
BASE_CHARS = (
    "".join(chr(c) for c in range(0x20, 0x7F))
    + "，。、；：？！“”‘’（）《》〈〉【】——…·～％＋－×÷＝℃"
    + "〇一二三四五六七八九十百千万亿"
    + "东南西北中发白"
    + "筒条万"
)


def collect_chars() -> str:
    chars: set[str] = set(BASE_CHARS)
    # 1) HTML 全文（含内联 JS 文案）
    chars.update(SRC_HTML.read_text(encoding="utf-8"))
    # 2) Python 源码字符串字面量里的字符（动态流水 / 牌名 / 结算文案）
    for p in PY_SOURCES:
        if not p.exists():
            print(f"  (跳过不存在的源文件 {p.name})")
            continue
        src = p.read_text(encoding="utf-8")
        for m in _STR_RE.finditer(src):
            chars.update(m.group(0))
    # 去掉控制字符（子集里没用且 pyftsubset 不喜欢）
    return "".join(sorted(c for c in chars if ord(c) >= 0x20))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="只打印统计，不写文件")
    args = ap.parse_args()

    if not SRC_TTF.exists():
        print(f"✗ 找不到源字体 {SRC_TTF}")
        return 1
    text = collect_chars()
    n_cjk = sum(1 for c in text if ord(c) > 0x2E7F)
    print(f"字符集：共 {len(text)} 个（其中 CJK ≈ {n_cjk}）")

    if args.check:
        return 0

    from fontTools import subset

    args_ = [
        str(SRC_TTF),
        f"--text={text}",
        "--flavor=woff2",
        f"--output-file={OUT_WOFF2}",
        "--layout-features=*",
        "--drop-tables+=DSIG",
        "--no-hinting",
        "--desubroutinize",
    ]
    subset.main(args_)
    sz_src = SRC_TTF.stat().st_size
    sz_out = OUT_WOFF2.stat().st_size
    print(f"✓ 生成 {OUT_WOFF2.name}: {sz_out / 1024:.0f} KB"
          f"（原 {sz_src / 1024 / 1024:.1f} MB → {sz_out / sz_src * 100:.1f}%）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
