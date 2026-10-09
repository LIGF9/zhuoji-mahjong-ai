"""把 docs/独山麻将规则教程.md 渲染成可双击打开的独立 HTML（深绿金色主题）。

用法（依赖 markdown 包）::

    python scripts/render_tutorial.py

输出：``docs/独山麻将规则教程.html``
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import markdown

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "docs" / "独山麻将规则教程.md"
DST = ROOT / "docs" / "独山麻将规则教程.html"

CSS = """
:root {
  --bg: #0d1f16; --panel: #123322; --panel2: #0f2a1c;
  --gold: #e8b84b; --gold-dim: #b8903a; --txt: #e7dfc6; --dim: #9fb39a;
  --green: #1f6b45; --line: rgba(232,184,75,.26);
}
* { box-sizing: border-box; }
body {
  margin: 0; background: radial-gradient(120% 90% at 50% 0%, #16422c 0%, var(--bg) 60%) fixed;
  color: var(--txt); font-family: "Microsoft YaHei","PingFang SC","Noto Sans SC",system-ui,sans-serif;
  font-size: 15.5px; line-height: 1.85;
}
.wrap { max-width: 980px; margin: 0 auto; padding: 40px 26px 90px; }
h1 {
  font-size: 30px; color: var(--gold); font-weight: 600; letter-spacing: 2px;
  text-align: center; margin: 0 0 6px;
}
h1 + blockquote { margin-top: 0; }
h2 {
  font-size: 22px; color: var(--gold); font-weight: 600; margin: 46px 0 14px;
  padding-bottom: 8px; border-bottom: 1px solid var(--line);
}
h3 { font-size: 18px; color: #f0d68b; font-weight: 600; margin: 30px 0 10px; }
h4 { font-size: 16px; color: #f0d68b; margin: 22px 0 8px; }
p { margin: 10px 0; }
a { color: #8fd4a0; }
strong { color: #ffe9a8; }
code {
  background: rgba(255,255,255,.08); border: 1px solid rgba(255,255,255,.10);
  border-radius: 4px; padding: 1px 5px; font-size: 13.5px;
  font-family: Consolas,"Cascadia Mono",monospace; color: #ffd98a;
}
pre {
  background: var(--panel2); border: 1px solid var(--line); border-radius: 10px;
  padding: 14px 16px; overflow-x: auto;
}
pre code { background: none; border: none; padding: 0; color: #cfe6d2; line-height: 1.7; }
blockquote {
  margin: 14px 0; padding: 10px 16px; border-left: 3px solid var(--gold-dim);
  background: rgba(232,184,75,.07); border-radius: 0 8px 8px 0; color: #d8e0cf;
}
blockquote p { margin: 6px 0; }
table {
  width: 100%; border-collapse: collapse; margin: 16px 0; font-size: 14.5px;
  background: var(--panel); border-radius: 10px; overflow: hidden;
}
th, td { padding: 8px 12px; text-align: left; border-bottom: 1px solid rgba(255,255,255,.08); }
th { background: rgba(232,184,75,.14); color: var(--gold); font-weight: 600; white-space: nowrap; }
td { color: #d9e2d2; }
tbody tr:last-child td { border-bottom: none; }
tbody tr:hover td { background: rgba(255,255,255,.03); }
ul, ol { padding-left: 24px; }
li { margin: 5px 0; }
hr { border: none; border-top: 1px solid var(--line); margin: 34px 0; }
.top {
  text-align: center; color: var(--dim); font-size: 13px; margin-bottom: 28px;
}
"""


def main() -> int:
    text = SRC.read_text(encoding="utf-8")
    # 去掉文档内的目录（HTML 里有锚点也没人点，保留反而冗长）——这里保留，方便跳转
    body = markdown.markdown(
        text, extensions=["tables", "fenced_code", "sane_lists", "attr_list", "md_in_html"])
    body = re.sub(r"<h1[^>]*>.*?</h1>", "", body, count=1, flags=re.S)  # h1 单独放到顶部

    title = "独山麻将（贵州捉鸡）规则教程"
    html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title>
<style>{CSS}</style>
</head>
<body>
<div class="wrap">
<h1>{title}</h1>
<div class="top">对照实际实现逐条整理 · 生成自 docs/独山麻将规则教程.md</div>
{body}
</div>
</body>
</html>
"""
    DST.write_text(html, encoding="utf-8")
    print(f"已生成 {DST}  ({len(html)} 字节)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
