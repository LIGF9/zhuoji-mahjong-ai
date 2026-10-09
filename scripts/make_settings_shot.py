"""生成「设置-分值表」截图夹具：打开设置面板并切到分值表页。

用法::

    python scripts/make_settings_shot.py [out.html]

产出 web/static/assets/_shot_settings.html（可被无头 Edge 直接访问）。
验证：项目名与输入框的间距、纯数字输入框（无上下调节按钮）。
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "web/static/assets" / (sys.argv[1] if len(sys.argv) > 1 else "_shot_settings.html")

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
  showSetPane('score');              // 切到分值表
}, 500);
</script>
"""

assert "</body>" in src, "未找到 </body>"
OUT.write_text(src.replace("</body>", inject + "</body>", 1), encoding="utf-8")
print("wrote", OUT)
