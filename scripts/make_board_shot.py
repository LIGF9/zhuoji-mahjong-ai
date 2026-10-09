"""生成「对局面板」截图夹具：注入构造对局态，验证各座位副露横置牌指向。

用法::

    python scripts/make_board_shot.py [out.html]

产出 web/static/assets/_shot_board.html（可被无头 Edge 直接访问）。
构造场景：下家(座位1)碰对家(座位2)的幺鸡 —— 横置牌应位于其面板组内靠上端
（屏幕上方 = 其下家方向）；对家(座位2)碰上家(座位1)的幺鸡作对照。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "web/static/assets" / (sys.argv[1] if len(sys.argv) > 1 else "_shot_board.html")

src = (ROOT / "web/static/dushan.html").read_text(encoding="utf-8")
fixtures = json.loads((ROOT / "reports/ui_fixtures_over.json").read_text(encoding="utf-8"))
state = fixtures[0]["state"]
state["over"] = False

inject = """<script>
/* 屏蔽后续 /api/state 轮询与 /api/meta 重取：防止真实服务器状态覆盖注入态
   （meta 已在注入前发出原生请求，不受影响） */
(function(){
  var orig = window.fetch;
  window.fetch = function(url, opt){
    var u = String(url);
    if (u.indexOf('/api/state') >= 0 || u.indexOf('/api/meta') >= 0)
      return new Promise(function(){});
    return orig.apply(this, arguments);
  };
})();
var __ST = %s;
setTimeout(function(){
  var b = document.getElementById('boot');
  if (b) b.style.display = 'none';
  var st = JSON.parse(JSON.stringify(__ST));
  st.my_seat = 0;
  st.players = [0,1,2,3].map(function(i){
    var p = st.players[i] || {};
    p.seat = i;
    return p;
  });
  // 场景：下家(座位1)碰对家(座位2)的幺鸡；对家(座位2)碰上家(座位1)的幺鸡
  st.players[1].melds = [{ type: 1, label: "碰", tile: 9, src: 2, ji: "横鸡" }];
  st.players[1].hand_count = 10;
  st.players[2].melds = [{ type: 1, label: "碰", tile: 9, src: 1, ji: "横鸡" }];
  st.players[2].hand_count = 10;
  st.turn = 0;   // 轮到我（已摸牌待出牌）
  st.hand = [0, 1, 2, 3, 4, 5, 6, 7, 8, 19, 20, 24, 25, 25];
  st.last_draw = 25;   // 刚摸进的牌 → 不应有黄色描边框
  st.pending = { mode: "discard", discardables: st.hand };
  LAST_STATE = st;
  render(st);
}, 500);
</script>
""" % json.dumps(state, ensure_ascii=False)

assert "</body>" in src, "未找到 </body>"
OUT.write_text(src.replace("</body>", inject + "</body>", 1), encoding="utf-8")
print("wrote", OUT)
