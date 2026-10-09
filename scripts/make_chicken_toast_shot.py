"""生成「横鸡 toast」截图夹具：注入对局态 + 一条带 ctag 的弃牌事件。

用法::

    python scripts/make_chicken_toast_shot.py [out.html]

产出 web/static/assets/_shot_ctoast.html（可被无头 Edge 直接访问）。
验证：状态栏不再有「🐔横鸡轮」常驻芯片；改由统一 toast 播报「XX 打出 横鸡」。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "web/static/assets" / (sys.argv[1] if len(sys.argv) > 1 else "_shot_ctoast.html")

src = (ROOT / "web/static/dushan.html").read_text(encoding="utf-8")
fixtures = json.loads((ROOT / "reports/ui_fixtures_over.json").read_text(encoding="utf-8"))
state = fixtures[0]["state"]
state["over"] = False

inject = """<script>
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
try {
  var b = document.getElementById('boot');
  if (b) b.style.display = 'none';
  var st = JSON.parse(JSON.stringify(__ST));
  st.my_seat = 0;
  st.players = [0,1,2,3].map(function(i){
    var p = st.players[i] || {};
    p.seat = i;
    return p;
  });
  st.turn = 0;
  st.hand = [0, 1, 2, 3, 4, 5, 6, 7, 8, 19, 20, 24, 25, 25];
  st.pending = { mode: "discard", discardables: st.hand, buttons: [] };
  // 横鸡轮开着（旧版会在状态栏显示常驻芯片，现在应只靠 toast 提示）
  st.hengji_active = true;
  st.hengji_species = [9];
  LAST_STATE = st;
  render(st);
  // 模拟一条「下家打出横鸡」的新事件 → 触发统一 toast
  SID = "shot"; el('log').dataset.sid = "shot"; el('log').dataset.count = 0;
  renderLog([{t: 3.2, kind: 'discard', rel: 1, text: '大乔(B11) 打出 一条（鸡）',
              tile: 9, ctag: '横鸡'}]);
  // 截图用：无头浏览器的虚拟时间会走完 2.2 秒显示期，这里把同一条 toast 重新点亮并常驻
  toast('大乔(B11) 打出 横鸡', 999999);
} catch (err) {
  var d = document.createElement('div');
  d.style.cssText = 'position:fixed;top:0;left:0;z-index:99999;background:#900;color:#fff;' +
    'font-size:16px;padding:16px;white-space:pre-wrap;max-width:96vw';
  d.textContent = 'ERR: ' + ((err && err.stack) || err);
  document.body.appendChild(d);
}
}, 500);
</script>
""" % json.dumps(state, ensure_ascii=False)

assert "</body>" in src, "未找到 </body>"
OUT.write_text(src.replace("</body>", inject + "</body>", 1), encoding="utf-8")
print("wrote", OUT)
