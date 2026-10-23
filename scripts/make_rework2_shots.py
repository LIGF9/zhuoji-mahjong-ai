"""生成 2026-10-12 第二轮 UI 改造的截图夹具（注入脚本 → 无头 Edge 截图）。

覆盖本轮 8 项反馈里可视觉核对的部分::

    _shot_rw2_boot.html     开局页（确认 #boot-players 没被塞进模型下拉）
    _shot_rw2_who.html      设置 · 玩家模型（行内模型下拉 + 推荐策略/理由同行 + 推荐出牌形式）
    _shot_rw2_play.html     设置 · 出牌与显示（出牌节奏 + 界面显示已合并）
    _shot_rw2_rule.html     设置 · 规则开关（鸡牌入口）
    _shot_rw2_ji.html       鸡牌浮窗（全 27 张宫格直选）
    _shot_rw2_hist.html     对局记录（名字 / 回放按钮左对齐）
    _shot_rw2_replay.html   回放控制条（拆成左右两块，落在对家手牌两侧）

跑法::

    python scripts/make_rework2_shots.py           # 只生成夹具页面
    python scripts/make_rework2_shots.py --shoot   # 生成并用无头 Edge 截图到 reports/
"""
from __future__ import annotations

import json
import subprocess
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = (ROOT / "web/static/dushan.html").read_text(encoding="utf-8")
OUTDIR = ROOT / "web/static/assets"
REPORTS = ROOT / "reports"

EDGE = Path(r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe")
# 服务端把 /assets/** 映射到 web/static/assets/**（不是 /static/assets/）
BASE = "http://127.0.0.1:8770/assets/"
API = "http://127.0.0.1:8770"


def page(body_js: str) -> str:
    """把注入脚本挂到 dushan.html 的 </body> 前。

    只屏蔽 /api/state（实盘轮询会覆盖回放/夹具画面）；
    /api/meta 必须放行 —— TILES 由它填充，挡掉鸡牌宫格就空了。
    """
    block = """<script>
(function(){
  var orig = window.fetch;
  window.fetch = function(url, opt){
    var u = String(url);
    if (u.indexOf('/api/state') >= 0) return new Promise(function(){});
    return orig.apply(this, arguments);
  };
})();
</script>
"""
    inject = block + "<script>\n" + body_js + "\n</script>\n"
    assert "</body>" in SRC, "未找到 </body>"
    return SRC.replace("</body>", inject + "</body>", 1)


def write(name: str, html: str) -> None:
    (OUTDIR / name).write_text(html, encoding="utf-8")
    print("wrote", OUTDIR / name)


HIDE_BOOT = "var b=document.getElementById('boot'); if(b) b.style.display='none';"


def settings_page(pane: str, extra: str = "") -> str:
    """打开设置面板并切到 pane（who / play / rule）。"""
    return page(f"""
setTimeout(function(){{
  {HIDE_BOOT}
  fillSetSelects();                       // 填下拉 + 同步置灰状态
  showSetPane('{pane}');
  document.getElementById('set-mask').classList.add('on');
  {extra}
}}, 700);
""")


# ---- 1) 开局页（保留 #boot 面板：核对玩家行里没有混入模型下拉）----
write("_shot_rw2_boot.html", page(""))

# ---- 2) 设置 · 玩家模型 ----
write("_shot_rw2_who.html", settings_page("who"))

# ---- 3) 设置 · 出牌与显示 ----
write("_shot_rw2_play.html", settings_page("play"))

# ---- 4) 设置 · 规则开关 ----
write("_shot_rw2_rule.html", settings_page("rule"))

# ---- 5) 鸡牌浮窗（宫格）----
write("_shot_rw2_ji.html", settings_page("rule", """
  setTimeout(function(){ openJiPicker(); }, 350);
"""))

# ---- 6) 对局记录（真数据）----
write("_shot_rw2_hist.html", page(f"""
setTimeout(function(){{
  {HIDE_BOOT}
  openHistory();
}}, 700);
"""))


def fetch_rid() -> str:
    """取一条有回放的对局 rid（无需真实对局也能出图）。"""
    try:
        with urllib.request.urlopen(API + "/api/history?limit=30", timeout=10) as r:
            recs = json.loads(r.read().decode("utf-8")).get("records", [])
        for rec in recs:
            if rec.get("has_replay") and rec.get("rid"):
                return rec["rid"]
    except Exception as e:                     # noqa: BLE001
        print("warn: 取 rid 失败：", e)
    return ""


RID = fetch_rid()
print("replay rid =", RID or "(无)")

# ---- 7) 回放控制条 ----
#  openReplay 是异步的：轮询等 RP 就绪，再跳到第 N 步让控制条填上内容。
write("_shot_rw2_replay.html", page(f"""
setTimeout(function(){{
  {HIDE_BOOT}
  var rid = {json.dumps(RID)};
  if (!rid) return;
  openReplay(rid);
  var tries = 0;
  var t = setInterval(function(){{
    tries++;
    /* RP 是顶层 let 声明，不在 window 上 —— 用 typeof 探测 */
    if (typeof RP !== 'undefined' && RP && RP.ev && RP.ev.length) {{
      clearInterval(t);
      RP_IDX = Math.min(24, RP.ev.length - 1);
      renderReplayFrame();
    }} else if (tries > 60) {{ clearInterval(t); }}
  }}, 150);
}}, 700);
"""))

if "--shoot" not in sys.argv:
    print("\n（加 --shoot 用无头 Edge 截图到 reports/）")
    raise SystemExit(0)

SHOTS = [
    ("_shot_rw2_boot.html", "_shot_rw2_boot.png", 1600, 900, 4000),
    ("_shot_rw2_who.html", "_shot_rw2_who.png", 1600, 1000, 4000),
    ("_shot_rw2_play.html", "_shot_rw2_play.png", 1600, 1000, 4000),
    ("_shot_rw2_rule.html", "_shot_rw2_rule.png", 1600, 1000, 4000),
    ("_shot_rw2_ji.html", "_shot_rw2_ji.png", 1600, 1000, 4000),
    ("_shot_rw2_hist.html", "_shot_rw2_hist.png", 1600, 1000, 5000),
    ("_shot_rw2_replay.html", "_shot_rw2_replay.png", 1600, 900, 12000),
]
if not EDGE.exists():
    print("找不到 Edge：", EDGE)
    raise SystemExit(1)

for src, dst, w, h, budget in SHOTS:
    out = REPORTS / dst
    subprocess.run([
        str(EDGE), "--headless=new", "--disable-gpu", "--hide-scrollbars",
        "--force-device-scale-factor=1",
        f"--window-size={w},{h}", f"--screenshot={out}",
        f"--virtual-time-budget={budget}",
        BASE + src,
    ], check=True, capture_output=True)
    print("shot", out, out.stat().st_size)
