"""生成本批「辅助功能」改动的截图夹具（注入真实快照 → 由无头 Edge 截图）。

产出（写在 web/static/assets/ 下，靠运行中的 8770 服务提供，相对 /api、/assets 才可用）::

    _shot_aux_rule.html      设置 · 规则开关（未听牌包赔已并入）
    _shot_aux_ui.html        设置 · 界面显示（听牌提示开关 + 结算布局）
    _shot_aux_ting.html      打牌听牌 toast
    _shot_aux_pair.html      结算单人视角（新版布局）
    _shot_aux_over.html      结算总览（赢家标签 = 胡牌方式）

跑法::

    python scripts/make_aux_shots.py
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = (ROOT / "web/static/dushan.html").read_text(encoding="utf-8")
OUTDIR = ROOT / "web/static/assets"

OVER = json.loads((ROOT / "reports/ui_fixtures_over.json").read_text(encoding="utf-8"))
MID = json.loads((ROOT / "reports/ui_fixtures_mid.json").read_text(encoding="utf-8"))
WIN = next(o for o in OVER if o["state"]["result"]["type"] == "win")


def hi(st_raw: dict, seat: int) -> str:
    """把快照旋到「我 = seat」的前端口径（与 make_settle_shot.py 同款）。"""
    st = json.loads(json.dumps(st_raw))
    old = st["players"]
    st["my_seat"] = seat
    st["players"] = []
    for rel in range(4):
        p = json.loads(json.dumps(old[(rel + seat) % 4]))
        p["seat"] = (rel + seat) % 4
        st["players"].append(p)
    return json.dumps(st, ensure_ascii=False)


def page(body_js: str, extra_head: str = "") -> str:
    inject = f"<script>\n{body_js}\n</script>\n"
    return SRC.replace("</body>", inject + "</body>", 1)


def write(name: str, html: str) -> None:
    (OUTDIR / name).write_text(html, encoding="utf-8")
    print("wrote", OUTDIR / name)


HIDE_BOOT = "document.getElementById('boot').style.display='none';"

# ---- 1) 设置 · 规则开关（包赔并入）+ 2) 设置 · 界面显示 ----
for name, pane in [("_shot_aux_rule.html", "rule"), ("_shot_aux_ui.html", "ui")]:
    write(name, page(f"""
setTimeout(function(){{
  {HIDE_BOOT}
  fillSetSelects();
  showSetPane('{pane}');
  document.getElementById('set-mask').classList.add('on');
}}, 600);
"""))

# ---- 3) 打牌听牌 toast ----
mid = json.loads(json.dumps(MID[0]["state"]))
mid["turn"] = mid["my_seat"]
T = (mid.get("pending") or {}).get("discardables", [1])[0]
write("_shot_aux_ting.html", page(f"""
window.__TING = {{ ok: true, map: {{ "{T}": {{
  tiles: [{{t: 9, left: 3}}, {{t: 18, left: 0}}, {{t: 21, left: 2}}],
  types: ["大对子"] }} }} }};
var __realFetch = window.fetch;
window.fetch = function(u){{
  if (String(u).indexOf('/api/ting') >= 0)
    return Promise.resolve({{ ok: true, json: function(){{ return Promise.resolve(window.__TING); }} }});
  return __realFetch.apply(this, arguments);
}};
setTimeout(function(){{
  {HIDE_BOOT}
  var st = JSON.parse(JSON.stringify({json.dumps(mid, ensure_ascii=False)}));
  st.turn = st.my_seat;
  SID = 'shot-ting';
  LAST_STATE = st; RES_VIEW = null; RES_DISMISSED = false;
  render(st);
  setTimeout(function(){{
    var d = document.querySelector('#hand .tile-mine[data-t="{T}"]');
    if (d) d.click();
    // 无头 Edge 用 virtual-time-budget 快进计时器：取消 toast 的自动隐藏，否则截图时已淡出
    var t = document.getElementById('toast');
    if (t && t._tm) clearTimeout(t._tm);
  }}, 350);
}}, 600);
"""))

# ---- 4) 结算单人视角（新版布局）----
seat = 0
write("_shot_aux_pair.html", page(f"""
setTimeout(function(){{
  {HIDE_BOOT}
  SET.resLayout = 'new';
  var st = JSON.parse(JSON.stringify({hi(WIN["state"], seat)}));
  st.over = true;
  LAST_STATE = st; RES_VIEW = {seat}; RES_DISMISSED = false;
  render(st);
}}, 600);
"""))

# ---- 5) 结算总览（赢家标签）----
write("_shot_aux_over.html", page(f"""
setTimeout(function(){{
  {HIDE_BOOT}
  var st = JSON.parse(JSON.stringify({hi(WIN["state"], seat)}));
  st.over = true;
  LAST_STATE = st; RES_VIEW = "ov"; RES_DISMISSED = false;
  render(st);
}}, 600);
"""))
