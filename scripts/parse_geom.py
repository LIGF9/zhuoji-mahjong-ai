# -*- coding: utf-8 -*-
"""解析夹具写到 document.title 的 __GEOM__ 几何数据（供移动端布局验收）。"""
import json
import re
import sys

path = sys.argv[1] if len(sys.argv) > 1 else "reports/_m_dom.html"
h = open(path, encoding="utf-8", errors="replace").read()
m = re.search(r"__GEOM__(\{.*?\})</title>", h, re.S)
if not m:
    print("NO GEOM")
    e = re.search(r'id="injerr".*?>(.*?)</pre>', h, re.S)
    print("injerr:", (e.group(1)[:500] if e else "none"))
    raise SystemExit(1)
d = json.loads(m.group(1))
print("errs:", d.get("errs"))
mm = re.search(r'<html[^>]*class="([^"]*)"', h)
print("html class:", mm.group(1) if mm else "(无)")
im = re.search(r'<html[^>]*style="([^"]*)"', h)
print("html style:", im.group(1) if im else "(无)")
print("meldsBox:", d.get("meldsBox"), " handBox:", d.get("handBox"))
print("rowDelta(副露底-手牌底):", d.get("rowDelta"), " hand:", d.get("hand"))
for c in d.get("melds", []):
    print("  g%d.%d%s x=%-7s w=%-5s h=%-5s bottom=%-7s dGroup=%s"
          % (c["g"], c["i"], "H" if c["heng"] else " ", c["x"], c["w"], c["h"], c["bottom"], c["dGroup"]))
print("log:", json.dumps(d.get("log"), ensure_ascii=False)[:260])
