"""生成「两两结算」截图夹具：把真实终局快照注入页面，按指定视角渲染结算表。

用法::

    python scripts/make_settle_shot.py <fixture_idx> <view_seat> <out.html> [hash]

产出 web/static/assets/_shot_settle.html（可被无头 Edge 直接访问）。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
IDX = int(sys.argv[1]) if len(sys.argv) > 1 else 9
SEAT = int(sys.argv[2]) if len(sys.argv) > 2 else 0
OUT = ROOT / "web/static/assets" / (sys.argv[3] if len(sys.argv) > 3 else "_shot_settle.html")

src = (ROOT / "web/static/dushan.html").read_text(encoding="utf-8")
fixtures = json.loads((ROOT / "reports/ui_fixtures_over.json").read_text(encoding="utf-8"))
state = fixtures[IDX]["state"]
state["over"] = True

inject = """<script>
var __ST = %s;
setTimeout(function(){
  var b = document.getElementById('boot');
  if (b) b.style.display = 'none';
  var st = JSON.parse(JSON.stringify(__ST));
  var s = %d;
  var old = st.players.slice();
  st.my_seat = s;
  st.players = [0,1,2,3].map(function(rel){
    var p = JSON.parse(JSON.stringify(old[(rel + s) %% 4]));
    p.seat = (rel + s) %% 4;
    return p;
  });
  LAST_STATE = st; RES_VIEW = s; RES_DISMISSED = false;
  render(st);
}, 500);
</script>
""" % (json.dumps(state, ensure_ascii=False), SEAT)

assert "</body>" in src, "未找到 </body>"
OUT.write_text(src.replace("</body>", inject + "</body>", 1), encoding="utf-8")
print("wrote", OUT)
