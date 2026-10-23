"""生成本轮「界面打磨」六件事的截图夹具（注入真实快照 → 无头 Edge 截图）。

产出页面（写在 web/static/assets/ 下，由运行中的 8770 服务提供）::

    _shot_pol_nav.html       左上角：图标折叠为 ⋯
    _shot_pol_navopen.html   左上角：点开 ⋯ 后竖排展开
    _shot_pol_narrow.html    1024px 窄桌面：我的信息板块在手牌左上方
    _shot_pol_hint.html      设置 · 大师推荐（独立分类）
    _shot_pol_score.html     设置 · 分值表（底部提示文字已弱化）
    _shot_pol_pair.html      结算单人视角（首行总计 + 头像列 + 板块合并）

跑法::

    python scripts/make_polish_shots.py           # 只生成夹具页面
    python scripts/make_polish_shots.py --shoot   # 生成并用无头 Edge 截图到 reports/
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = (ROOT / "web/static/dushan.html").read_text(encoding="utf-8")
OUTDIR = ROOT / "web/static/assets"
REPORTS = ROOT / "reports"

OVER = json.loads((ROOT / "reports/ui_fixtures_over.json").read_text(encoding="utf-8"))
MID = json.loads((ROOT / "reports/ui_fixtures_mid.json").read_text(encoding="utf-8"))
WIN = next(o for o in OVER if o["state"]["result"]["type"] == "win")

EDGE = Path(r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe")
# 服务端把 /assets/** 映射到 web/static/assets/**（不是 /static/assets/）
BASE = "http://127.0.0.1:8770/assets/"


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


def page(body_js: str) -> str:
    inject = f"<script>\n{body_js}\n</script>\n"
    return SRC.replace("</body>", inject + "</body>", 1)


def write(name: str, html: str) -> None:
    (OUTDIR / name).write_text(html, encoding="utf-8")
    print("wrote", OUTDIR / name)


HIDE_BOOT = "document.getElementById('boot').style.display='none';"

# ---- 1) 左上角：默认折叠 / 展开 ----
write("_shot_pol_nav.html", page(f"""
setTimeout(function(){{ {HIDE_BOOT} }}, 600);
"""))
write("_shot_pol_navopen.html", page(f"""
setTimeout(function(){{
  {HIDE_BOOT}
  setNavMore(true);
  document.getElementById('btn-log').classList.add('toggle-on');   // 示意展开态的高亮
}}, 600);
"""))

# ---- 2) 1024px 窄桌面：信息板块抬到手牌左上方 ----
#   （窗口宽度由截图命令的 --window-size 决定，页面只负责喂一个残局局面）
mid = json.loads(json.dumps(MID[0]["state"]))
mid["turn"] = mid["my_seat"]
write("_shot_pol_narrow.html", page(f"""
setTimeout(function(){{
  {HIDE_BOOT}
  var st = JSON.parse(JSON.stringify({json.dumps(mid, ensure_ascii=False)}));
  st.turn = st.my_seat;
  st.last_draw = st.hand[st.hand.length - 1];
  LAST_STATE = st; RES_VIEW = null;
  render(st);
}}, 600);
"""))

# ---- 3) 设置 · 大师推荐 / 分值表 ----
for name, pane, scroll in [("_shot_pol_hint.html", "hint", 0),
                           ("_shot_pol_score.html", "score", 9999)]:
    write(name, page(f"""
setTimeout(function(){{
  {HIDE_BOOT}
  fillSetSelects();
  showSetPane('{pane}');
  document.getElementById('set-mask').classList.add('on');
  setTimeout(function(){{   // 等高度过渡收尾后再滚到底部（分值表：看提示文字）
    document.getElementById('set-panes').scrollTop = {scroll};
  }}, 400);
}}, 600);
"""))

# ---- 4) 结算单人视角（新表结构）----
seat = 0
write("_shot_pol_pair.html", page(f"""
setTimeout(function(){{
  {HIDE_BOOT}
  SET.resLayout = 'new';
  var st = JSON.parse(JSON.stringify({hi(WIN["state"], seat)}));
  st.over = true;
  LAST_STATE = st; RES_VIEW = {seat}; RES_DISMISSED = false;
  render(st);
}}, 600);
"""))

if "--shoot" not in sys.argv:
    print("\n（加 --shoot 用无头 Edge 截图到 reports/）")
    raise SystemExit(0)

SHOTS = [
    ("_shot_pol_nav.html", "_shot_pol_nav.png", 1600, 900),
    ("_shot_pol_navopen.html", "_shot_pol_navopen.png", 1600, 900),
    ("_shot_pol_narrow.html", "_shot_pol_narrow.png", 1024, 800),
    ("_shot_pol_hint.html", "_shot_pol_hint.png", 1600, 900),
    ("_shot_pol_score.html", "_shot_pol_score.png", 1600, 900),
    ("_shot_pol_pair.html", "_shot_pol_pair.png", 1600, 900),
]
for src, dst, w, h in SHOTS:
    out = REPORTS / dst
    subprocess.run([
        str(EDGE), "--headless=new", "--disable-gpu", "--hide-scrollbars",
        "--force-device-scale-factor=1",
        f"--window-size={w},{h}", f"--screenshot={out}",
        "--virtual-time-budget=4000", BASE + src,
    ], check=True, capture_output=True)
    print("shot", out, out.stat().st_size)
