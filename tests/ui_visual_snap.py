"""真实浏览器可视化快照（Playwright）：为「横置牌消缝 / 总览对齐 / 结算大表」
等纯视觉需求提供像素级证据。

产出（reports/ui_snaps/）::

    melds_*.png   四家副露（含横置牌）放大截图
    ov_meta.png   结算总览：打出的鸡牌与文字标签底部对齐
    settle_*.png  单人视角合并大表
    coach_default.png  大师建议面板默认位（上家手牌与弃牌区之间）

同时打印几何测量（横置牌图与格子的相对位置、鸡牌图与标签的底边差）。

跑法::

    python tests/ui_visual_snap.py [端口]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]
PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 8770
BASE = f"http://127.0.0.1:{PORT}"
OUT = ROOT / "reports" / "ui_snaps"
OUT.mkdir(parents=True, exist_ok=True)

SETUP_JS = """
async () => {
  const r = await fetch('/api/new', {
    method: 'POST', headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({
      opponents: [{type:'teacher',style:'balanced'},
                  {type:'teacher',style:'aggressive'},
                  {type:'random'}],
      my_seat: 0, speed: 0, seed: 20261008})});
  const j = await r.json();
  if (!j.sid) return {ok: false, err: String(j)};
  for (let i = 0; i < 300; i++) {
    const st = await (await fetch('/api/state?sid=' + j.sid)).json();
    if (st.ready && st.pending && st.turn === st.my_seat) { window.__MID = st; return {ok: true}; }
    await new Promise(t => setTimeout(t, 50));
  }
  return {ok: false, err: '未走到人类决策点'};
}
"""

# 页面自己的 poll() 会用服务器状态覆盖注入的快照：截图前冻结渲染并收起弹窗
FREEZE_JS = """
() => {
  window.__realRender = window.render;   // 保留真渲染，注入时手动调用
  window.render = () => {};
  el('boot').classList.remove('on2');    // 开局大厅
  ['set-mask', 'result-mask', 'stats-mask', 'hist-mask'].forEach(
    id => el(id) && el(id).classList.remove('on'));
  return true;
}
"""

# 四家副露：横置牌分别落在 最左/居中/最右，覆盖碰鸡、明杠、补杠
MELDS_JS = """
() => {
  const st = JSON.parse(JSON.stringify(window.__MID));
  st.players[0].melds = [
    {type:2, label:'明杠', tile:12, src:3, ji:null},   // 上家点杠 → 横置最左
    {type:1, label:'碰',   tile:9,  src:2, ji:'横鸡'}, // 对家碰鸡 → 横置居中
    {type:2, label:'明杠', tile:21, src:1, ji:null},   // 下家点杠 → 横置最右
  ];
  st.players[0].hand_count = 14 - 3 * 3;
  st.players[1].melds = [{type:2, label:'明杠', tile:5, src:2, ji:null}];
  st.players[2].melds = [{type:1, label:'碰',   tile:8, src:1, ji:'横鸡'}];
  st.players[3].melds = [{type:2, label:'明杠', tile:14, src:0, ji:null}];
  window.LAST_STATE = st; (window.__realRender || window.render)(st);
  window.render = () => {};
  const g = s => document.querySelector(s).getBoundingClientRect();
  const meHeng = [...document.querySelectorAll('#me-melds .mc.heng img')].map(i => i.getBoundingClientRect());
  const cell = document.querySelector('#me-melds .mc.heng').getBoundingClientRect();
  return {
    cell: {l: cell.left, r: cell.right},
    imgs: meHeng.map(r => ({l: r.left, r: r.right})),
    rects: {me: g('#me-melds'), p1: g('#p1 .melds'), p2: g('#p2 .melds'), p3: g('#p3 .melds')},
  };
}
"""

# 结算总览：打出的鸡牌与标签底部对齐测量
OV_JS = """
() => {
  const st = JSON.parse(JSON.stringify(window.__MID));
  st.over = true;
  const rel = s => s;   // my_seat=0
  const me = st.players[0];
  me.hand_tiles = [0, 1, 2, 10, 11, 12, 19, 20, 21, 25, 25, 9, 9];
  me.melds = [{type:2, label:'明杠', tile:12, src:3, ji:null}];
  me.discard_tags = ['冲锋鸡', null, '横鸡'];
  me.discards = [9, 5, 9];
  me.ting_tiles = [9]; me.ting_types = ['平胡'];
  const deltas = [12, -4, -4, -4];
  const pair = [[0,13,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0]];
  RES_DISMISSED = false; RES_VIEW = null; RES_VIEW_PREV = null;
  st.result = {
    type: 'win', winner: 0, winners: [0], winner_names: [me.name || '你'],
    winners_fan: [{seat: 0, rel: 0, name: me.name || '你', fan_cn: '平胡', fan: 12}],
    winner_hands: {'0': me.hand_tiles}, winner_melds_map: {'0': me.melds},
    loser: 1, loser_name: st.players[1].name,
    is_tsumo: false, rob_kong: false, how: '点炮', void_name: null,
    win_tile: 9, fan_cn: '平胡', fan: 12, deltas: deltas, pair: pair,
    pair_detail: [{a: 0, b: 1, items: [{label: '点炮', info: '', value: 3, who: 0},
                                       {label: '手中鸡', info: '2 只', value: 2, who: 0},
                                       {label: '冲锋鸡', info: '打出', value: 3, who: 0},
                                       {label: '横鸡', info: '打出', value: 2, who: 0},
                                       {label: '包鸡', info: '横鸡+横鸡溢价', value: 3, who: 1}]}],
    detail: [], chickens: [5, 0, 0, 0], wall_left: 4,
    fanji_flip: 3, fanji_tiles: [4], fanji_gold: false,
    jiesuan_fanji: 'both', kaiju_fanji: 'off',
    ji_tiles: [9], kaiju_flip: null,
  };
  window.LAST_STATE = st; (window.__realRender || window.render)(st);
  window.render = () => {};
  const card = document.querySelector('#res-pairs .ovcard');
  const tag = card.querySelector('.ovmeta .tag');
  const disc = card.querySelector('.ovmeta .disc');
  const tr = tag.getBoundingClientRect(), dr = disc.getBoundingClientRect();
  return {tagBottom: tr.bottom, discBottom: dr.bottom, gap: +(tr.bottom - dr.bottom).toFixed(2),
          cardTop: card.getBoundingClientRect().top};
}
"""


# 新摸进的牌：应单独摆在手牌最右侧、与手牌隔开一段（不插入排序位置）
DRAWN_JS = """
() => {
  const st = JSON.parse(JSON.stringify(window.__MID));
  st.turn = st.my_seat;
  st.pending = Object.assign({}, st.pending,
    {mode: 'discard', discardables: st.hand.slice()});
  const sorted = st.hand.slice().sort((a, b) => a - b);
  st.last_draw = sorted[sorted.length - 1];     // 假定最后摸到的是最大那张
  window.LAST_STATE = st; (window.__realRender || window.render)(st);
  window.render = () => {};
  const h = el('hand');
  const tiles = [...h.querySelectorAll('.tile-mine:not(.ghost)')];
  const drawn = h.querySelector('.tile-mine.newdraw');
  const before = tiles[tiles.length - 2] ? tiles[tiles.length - 2].getBoundingClientRect() : null;
  const dr = drawn ? drawn.getBoundingClientRect() : null;
  const all = tiles.map(t => t.getBoundingClientRect());
  const widths = {};
  tiles.forEach(t => { widths[+t.dataset.t] = t.getBoundingClientRect().width; });
  return {
    handLen: st.hand.length, modOk: st.hand.length % 3 === 2,
    nTiles: tiles.length, hasSep: !!h.querySelector('.drawsep'),
    drawnTile: drawn ? +drawn.dataset.t : null, expect: st.last_draw,
    gap: (before && dr) ? +(dr.left - before.right).toFixed(2) : null,
    handWidth: +getComputedStyle(h).width.replace('px', ''),
    tilesRight: all.length ? +all[all.length - 1].right.toFixed(2) : null,
    handRight: +h.getBoundingClientRect().right.toFixed(2),
  };
}
"""


def main() -> int:
    fails: list[str] = []

    def chk(name: str, cond: bool, extra: str = "") -> None:
        print(("  [ok  ] " if cond else "  [FAIL] ") + name + (("   " + extra) if extra else ""))
        if not cond:
            fails.append(name)

    with sync_playwright() as pw:
        try:
            browser = pw.chromium.launch(channel="msedge", headless=True)
        except Exception:
            browser = pw.chromium.launch(headless=True)
        ctx = browser.new_context(viewport={"width": 1700, "height": 950},
                                  device_scale_factor=3)
        page = ctx.new_page()
        page.goto(BASE, wait_until="load")
        page.wait_for_function("typeof META !== 'undefined' && META !== null", timeout=20000)
        r = page.evaluate(SETUP_JS)
        if not r.get("ok"):
            print("准备失败：", r)
            browser.close()
            return 1
        page.evaluate(FREEZE_JS)     # 冻结页面轮询渲染，避免覆盖注入的快照

        # --- 副露横置牌 ---
        m = page.evaluate(MELDS_JS)
        print("横置格 l=%.1f r=%.1f" % (m["cell"]["l"], m["cell"]["r"]))
        for i, im in enumerate(m["imgs"]):
            print("  横置图%d l=%.1f (格左%+.1f) r=%.1f (格右%+.1f)"
                  % (i, im["l"], im["l"] - m["cell"]["l"], im["r"], im["r"] - m["cell"]["r"]))
        for name in ["me", "p1", "p2", "p3"]:
            page.locator("#me-melds" if name == "me" else f"#{name} .melds") \
                .screenshot(path=str(OUT / f"melds_{name}.png"))
        page.screenshot(path=str(OUT / "table_full.png"))

        # --- 结算总览对齐 ---
        ov = page.evaluate(OV_JS)
        print("总览 标签底=%.1f 鸡牌底=%.1f 差=%.2f px" %
              (ov["tagBottom"], ov["discBottom"], ov["gap"]))
        page.locator("#res-pairs").screenshot(path=str(OUT / "ov_meta.png"))

        # --- 单人视角大表：5 列 + 两端突出 + 空行两倍高 ---
        page.evaluate("() => { RES_VIEW = 0; renderResultView(LAST_STATE); }")
        page.wait_for_timeout(600)
        page.locator("#res-pairs").screenshot(path=str(OUT / "settle_pairs.png"))
        geo = page.evaluate("""() => {
          const t = document.querySelector('.pairs-big');
          const th = [...t.querySelectorAll('tr:first-child th')].map(x => x.textContent.trim());
          const det = t.querySelector('tr.det');
          const sum = t.querySelector('tr.sum');
          const gap = t.querySelector('tr.gap');
          const r = x => x.getBoundingClientRect();
          const padL = x => parseFloat(getComputedStyle(x).paddingLeft) || 0;
          const padR = x => parseFloat(getComputedStyle(x).paddingRight) || 0;
          // 名字/数字的「实际可见边界」= 格子边界 ± 内边距
          const nameL = td => r(td).left + padL(td);
          const numR = td => r(td).right - padR(td);
          return {
            th: th, cols: det.children.length,
            det: [...det.children].map(x => x.textContent.trim()),
            sum: [...sum.children].map(x => x.textContent.trim()),
            gapH: +r(gap).height.toFixed(1), detH: +r(det).height.toFixed(1),
            nameDet: +nameL(det.children[0]).toFixed(1),
            nameSum: +nameL(sum.children[0]).toFixed(1),
            scoreDet: +numR(det.children[3]).toFixed(1),
            scoreSum: +numR(sum.children[4]).toFixed(1),
            col5W: +r(sum.children[4]).width.toFixed(1),
            plDet: getComputedStyle(det.children[0]).paddingLeft,
            plSum: getComputedStyle(sum.children[0]).paddingLeft,
            prDet: getComputedStyle(det.children[3]).paddingRight,
            prSum: getComputedStyle(sum.children[4]).paddingRight,
          };
        }""")
        print("大表 表头=%s 列数=%d" % (geo["th"], geo["cols"]))
        print("   明细行=%s" % (geo["det"],))
        print("   小计行=%s" % (geo["sum"],))
        print("   空行高=%.1f 明细行高=%.1f 末列宽=%.1f" % (geo["gapH"], geo["detH"], geo["col5W"]))
        print("   内边距 名字列 pl 明细=%s 小计=%s ｜ 积分 pr 明细=%s 汇总=%s"
              % (geo["plDet"], geo["plSum"], geo["prDet"], geo["prSum"]))
        print("   名字可见左界：明细 %.1f / 小计 %.1f（小计应更小=更靠左）" % (geo["nameDet"], geo["nameSum"]))
        print("   数字可见右界：明细 %.1f / 小计 %.1f（小计应更大=更靠右）" % (geo["scoreDet"], geo["scoreSum"]))
        chk("大表 5 列（名字｜项目｜说明｜积分｜汇总分）", geo["cols"] == 5, f"cols={geo['cols']}")
        chk("首末两列无表头", geo["th"][0] == "" and geo["th"][4] == "",
            "|".join(geo["th"]))
        chk("表头三列 = 项目/说明/积分", geo["th"][1:4] == ["项目", "说明", "积分"], "|".join(geo["th"]))
        chk("明细行首列=责任方 末列空", geo["det"][0] != "" and geo["det"][4] == "", "|".join(geo["det"]))
        chk("小计行末列=该玩家汇总分", geo["sum"][4] not in ("", "0"), "|".join(geo["sum"]))
        chk("汇总行名字相对明细行向左突出（>=12px）", geo["nameDet"] - geo["nameSum"] >= 12,
            f"差 {geo['nameDet'] - geo['nameSum']:.1f}px")
        chk("汇总分相对明细数字向右突出（>=12px）", geo["scoreSum"] - geo["scoreDet"] >= 12,
            f"差 {geo['scoreSum'] - geo['scoreDet']:.1f}px")
        chk("空行高度 = 20px（原 10px 的两倍）", 19.0 <= geo["gapH"] <= 21.0, f"gap={geo['gapH']}")

        # --- 新摸进的牌：单独摆右、与手牌隔开 ---
        dn = page.evaluate(DRAWN_JS)
        print("新摸牌 手牌=%d 张（3n+2=%s）格子=%d 间隔=%.1f px 新牌=%s 期望=%s"
              % (dn["handLen"], dn["modOk"], dn["nTiles"], dn["gap"] if dn["gap"] is not None else -1,
                 dn["drawnTile"], dn["expect"]))
        chk("新摸的牌带 newdraw 且在最右", dn["drawnTile"] == dn["expect"] and dn["nTiles"] == dn["handLen"],
            f"tile={dn['drawnTile']}")
        chk("新摸的牌与手牌隔开一段（>=18px）", (dn["gap"] or 0) >= 18, f"gap={dn['gap']}")
        chk("有间隔占位元素", dn["hasSep"])
        page.locator("#hand").screenshot(path=str(OUT / "hand_newdraw.png"))

        # --- 大师建议默认位 ---
        page.evaluate("""() => {
          localStorage.removeItem('dushan_coach_pos_v2');
          SET.coach = true; el('coach').classList.add('on'); tryPlaceCoach();
        }""")
        info = page.evaluate("""() => {
          const b = el('coach').getBoundingClientRect();
          const p3 = el('p3');
          const hand = p3.querySelector('.phand').getBoundingClientRect();
          const riv = el('river-left').getBoundingClientRect();
          const av = p3.querySelector('.avatar').getBoundingClientRect();
          return {coach: {l: b.left, t: b.top, w: b.width, h: b.height},
                  handRight: hand.right, riverLeft: riv.left, avatarTop: av.top};
        }""")
        print("大师建议面板 l=%.1f t=%.1f w=%.1f | 上家手牌右=%.1f 弃牌左=%.1f 头像顶=%.1f"
              % (info["coach"]["l"], info["coach"]["t"], info["coach"]["w"],
                 info["handRight"], info["riverLeft"], info["avatarTop"]))
        page.screenshot(path=str(OUT / "coach_default.png"))
        browser.close()
    print("快照目录：", OUT)
    if fails:
        print(f"几何校验失败 {len(fails)} 项：" + "；".join(fails))
        return 1
    print("几何校验全部通过 ✅")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
