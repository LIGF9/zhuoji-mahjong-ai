"""生成「设置面板」截图夹具：打开设置面板并切到指定分类页。

用法::

    python scripts/make_settings_shot.py [out.html] [pane]

pane 默认 score（分值表）；可传 play（出牌与推荐）等。
产出 web/static/assets/_shot_settings.html（可被无头 Edge 直接访问）。
验证：项目名与输入框的间距、纯数字输入框（无上下调节按钮）、
      「出牌与推荐」里的节奏/推荐理由等分组排布。
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "web/static/assets" / (sys.argv[1] if len(sys.argv) > 1 else "_shot_settings.html")
PANE = sys.argv[2] if len(sys.argv) > 2 else "score"

src = (ROOT / "web/static/dushan.html").read_text(encoding="utf-8")

inject = """<script>
/* 屏蔽 /api/state、/api/meta：避免真实服务器状态干扰 */
(function(){
  var orig = window.fetch;
  window.fetch = function(url, opt){
    var u = String(url);
    if (u.indexOf('/api/state') >= 0 || u.indexOf('/api/meta') >= 0)
      return new Promise(function(){});
    return orig.apply(this, arguments);
  };
})();
setTimeout(function(){
  var b = document.getElementById('boot');
  if (b) b.style.display = 'none';
  el('btn-settings').onclick();      // fillSetSelects + 打开面板
  showSetPane('%s');
}, 500);
</script>
""" % PANE

assert "</body>" in src, "未找到 </body>"
OUT.write_text(src.replace("</body>", inject + "</body>", 1), encoding="utf-8")
print("wrote", OUT)
