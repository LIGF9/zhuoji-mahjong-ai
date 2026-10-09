"""生成「鸡牌元素」截图夹具：验证鸡牌红点标记 + 冲锋鸡/横鸡统一角标。

用法::

    python scripts/make_chicken_shot.py [out.html]

产出 web/static/assets/_shot_chicken.html（可被无头 Edge 直接访问）。
构造场景：
  - 我的手牌含幺鸡(1条)与开局翻鸡种鸡牌(3筒)，右上角均为小红点；
  - 下家碰了对家的横鸡（横置牌带「横」蓝角标）；
  - 对家打出了 3筒冲锋鸡（开局翻鸡种 → 牌河显示真实牌面 + 「冲」红角标）；
  - 上家打出了幺鸡冲锋鸡（牌河显示 1条牌面 + 「冲」红角标）。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "web/static/assets" / (sys.argv[1] if len(sys.argv) > 1 else "_shot_chicken.html")

src = (ROOT / "web/static/dushan.html").read_text(encoding="utf-8")
fixtures = json.loads((ROOT / "reports/ui_fixtures_over.json").read_text(encoding="utf-8"))
state = fixtures[0]["state"]
state["over"] = False

inject = """<script>
/* 屏蔽后续 /api/state 轮询与 /api/meta 重取：防止真实服务器状态覆盖注入态 */
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
  st.ji_tiles = [9, 20];   // 幺鸡 + 开局翻鸡种(3筒)
  // 我的手牌：含幺鸡(9) 与 翻鸡种(20)，最后一张为刚摸进的 20 → 红点、无黄框
  st.hand = [0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 12, 9, 20];
  st.last_draw = 20;
  st.pending = { mode: "discard", discardables: st.hand };
  st.turn = 0;
  // 下家(1)碰对家(2)的横鸡；对家(2)明杠上家(3)的幺鸡（横置指向）
  st.players[1].melds = [{ type: 1, label: "碰", tile: 9, src: 2, ji: "横鸡" }];
  st.players[1].hand_count = 10;
  st.players[2].melds = [{ type: 2, label: "明杠", tile: 9, src: 3, ji: "幺鸡" }];
  st.players[2].hand_count = 7;
  // 对家打出 3筒冲锋鸡（翻鸡种）+ 幺鸡冲锋鸡；下家打出普通幺鸡
  st.players[2].discards = [20, 9, 5];
  st.players[2].discard_tags = ["冲锋鸡", "冲锋鸡", null];
  st.players[1].discards = [9];
  st.players[1].discard_tags = [null];
  st.players[3].discards = [12];
  st.players[3].discard_tags = ["横鸡"];
  st.last = null;
  LAST_STATE = st;
  render(st);
}, 500);
</script>
""" % json.dumps(state, ensure_ascii=False)

assert "</body>" in src, "未找到 </body>"
OUT.write_text(src.replace("</body>", inject + "</body>", 1), encoding="utf-8")
print("wrote", OUT)
