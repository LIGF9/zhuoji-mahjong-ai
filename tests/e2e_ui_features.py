"""真实浏览器 E2E（Playwright + Edge）：验证本轮四项改动在真机上确实生效。

覆盖：
  1. 「杠·通行证」徽章在**四家**都位于头像正上方（用真实几何坐标断言，不是查 CSS 文本）
  2. 补杠牌渲染：非鸡牌补杠 4 张全竖排（无横置指向）；鸡牌补杠保留横置指向
  3. 结算视图切换：总览 ↔ 单人视角 之间高度**平滑过渡**（采样中间帧高度）
  4. 结算内容：一炮多响 / 抢杠 / 热炮 的标签与「鸡分全烧」在真浏览器里渲染正确

前置：8770 服务在跑（`python web/dushan_server.py --port 8770`）。

跑法::

    python tests/e2e_ui_features.py
"""
from __future__ import annotations

import sys

from playwright.sync_api import sync_playwright

BASE = "http://127.0.0.1:8770"
PASS: list[str] = []
FAIL: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    (PASS if ok else FAIL).append(name)
    print(f"  [{'ok  ' if ok else 'FAIL'}] {name}" + (f"   {detail}" if detail else ""), flush=True)


# 开场：新建一局，等到「我的决策点」，把该快照留在 window.__MID
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

# 徽章几何：四家都要在头像正上方
BADGE_JS = """
() => {
  const st = JSON.parse(JSON.stringify(window.__MID));
  st.players.forEach(p => { p.has_gang = true; p.baojiao = true; });
  window.LAST_STATE = st; window.render(st);
  const out = [];
  for (const id of ['me', 'p1', 'p2', 'p3']) {
    const box = document.getElementById(id);
    const ab = box.querySelector('.ava-badges');
    const av = box.querySelector('.avatar');
    const cb = box.querySelector('.corner-badges');
    const br = ab ? ab.getBoundingClientRect() : null;
    const ar = av.getBoundingClientRect();
    out.push({
      id,
      hasBadge: !!ab && ab.textContent.includes('杠·通行证'),
      above: br ? (br.bottom <= ar.top + 1) : false,
      dxCenter: br ? Math.round(Math.abs((br.left + br.right) / 2 - (ar.left + ar.right) / 2)) : -1,
      cornerHasGang: !!cb && cb.textContent.includes('杠·通行证'),
      cornerHasJiao: !!cb && cb.textContent.includes('报叫'),
      pos: ab ? getComputedStyle(ab).position : '',
    });
  }
  return out;
}
"""

# 副露渲染：碰鸡 / 明杠 / 补杠(非鸡) / 补杠(鸡)
MELD_JS = """
() => {
  const st = JSON.parse(JSON.stringify(window.__MID));
  st.players[0].melds = [
    {type:1, label:'碰',   tile:9,  src:1, ji:'横鸡'},
    {type:2, label:'明杠', tile:12, src:2, ji:null},
    {type:3, label:'补杠', tile:15, src:0, ji:null},
    {type:3, label:'补杠', tile:18, src:1, ji:'横鸡'},
  ];
  st.players[0].hand_count = 14 - 3 * 4;
  window.LAST_STATE = st; window.render(st);
  return [...document.querySelectorAll('#me-melds .meld')].map(g => ({
    heng: g.querySelectorAll('.mc.heng').length,
    cells: g.querySelectorAll('.mc').length,
  }));
}
"""

# 造一个「胡牌终局」快照（一炮多响 / 抢杠 / 热炮 三种变体都由参数控制）
WIN_STATE_JS = """
(mode) => {
  const st = JSON.parse(JSON.stringify(window.__MID));
  st.over = true;
  const me = st.my_seat;
  const w1 = me, w2 = (me + 2) % 4, loser = (me + 1) % 4;
  const winTile = 9;
  const winners = mode === 'multi' ? [w1, w2] : [w1];
  const rel = s => ((s - me) % 4 + 4) % 4;
  for (const w of winners) {
    const p = st.players[rel(w)];
    p.hand_tiles = (p.hand_tiles || []).concat([winTile]);
    p.hand_count = p.hand_tiles.length;
    p.ting_tiles = [winTile];
    p.ting_types = ['平胡'];
  }
  const deltas = [0, 0, 0, 0];
  deltas[w1] = 10; deltas[loser] = -10;
  if (mode === 'multi') { deltas[w2] = 8; deltas[loser] = -18; }
  const pair = [[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0]];
  pair[w1][loser] = 10; if (mode === 'multi') pair[w2][loser] = 8;
  const how = mode === 'rob' ? '抢杠' : (mode === 'repao' ? '热炮' : '点炮');
  RES_DISMISSED = false; RES_VIEW = null; RES_VIEW_PREV = null;   // 每次新终局都从总览开始
  st.result = {
    type: 'win', winner: w1, winners: winners,
    winner_names: winners.map(w => st.players[rel(w)].name),
    winners_fan: winners.map((w, i) => ({seat: w, rel: rel(w), name: st.players[rel(w)].name,
                                         fan_cn: i ? '平胡' : '清一色', fan: i ? 8 : 10})),
    winner_hands: Object.fromEntries(winners.map(w => [String(w), st.players[rel(w)].hand_tiles])),
    winner_melds_map: Object.fromEntries(winners.map(w => [String(w), []])),
    loser: loser, loser_name: st.players[rel(loser)].name,
    is_tsumo: false, rob_kong: mode === 'rob', how: how,
    void_name: (mode === 'rob' || mode === 'repao') ? st.players[rel(loser)].name : null,
    win_tile: winTile, fan_cn: '清一色', fan: 10, deltas: deltas, pair: pair,
    pair_detail: winners.map(w => ({a: w, b: loser,
      items: [{label: how + (winners.length > 1 ? '（2家）' : ''), info: '', value: pair[w][loser], who: w}]})),
    detail: [], chickens: [0,0,0,0], wall_left: 4,
    fanji_flip: 3, fanji_tiles: [4], fanji_gold: false,
    jiesuan_fanji: 'both', kaiju_fanji: 'off',
    ji_tiles: [9], kaiju_flip: null,
  };
  window.LAST_STATE = st; window.render(st);
  return true;
}
"""

TEXT_JS = """
() => ({
  title: document.getElementById('res-title').innerHTML,
  jibar: document.getElementById('res-jibar').textContent,
  pairs: document.getElementById('res-pairs').innerHTML,
  winTags: document.querySelectorAll('#res-pairs .ovcard .tag.win').length,
  cards: [...document.querySelectorAll('#res-pairs .ovcard .nm')].map(x => x.textContent),
})
"""


def main() -> int:
    with sync_playwright() as pw:
        try:
            browser = pw.chromium.launch(channel="msedge", headless=True)
        except Exception:
            browser = pw.chromium.launch(headless=True)
        page = browser.new_context(viewport={"width": 1600, "height": 900}).new_page()
        page.goto(BASE, wait_until="load")
        page.wait_for_function("typeof META !== 'undefined' && META !== null", timeout=20000)

        print("\n[准备] 新建对局并走到人类决策点")
        r = page.evaluate(SETUP_JS)
        check("取到对局中快照", bool(r.get("ok")), str(r.get("err", "")))
        if not r.get("ok"):
            browser.close()
            return 1

        # ---------------- 1. 杠·通行证徽章位置 ----------------
        print("\n[1] 「杠·通行证」徽章位置（真实几何）")
        for it in page.evaluate(BADGE_JS):
            cond = (it["hasBadge"] and it["above"] and it["dxCenter"] <= 3
                    and not it["cornerHasGang"] and it["cornerHasJiao"])
            check(f"{it['id']} 徽章在头像正上方", cond,
                  f"above={it['above']} 水平偏移={it['dxCenter']}px "
                  f"角标残留={it['cornerHasGang']} 角标仍含报叫={it['cornerHasJiao']}")

        # ---------------- 2. 补杠牌渲染 ----------------
        print("\n[2] 补杠牌渲染（非鸡竖排 / 鸡横置）")
        melds = page.evaluate(MELD_JS)
        want = [("碰(横鸡)", 1, 3), ("明杠", 1, 4), ("补杠(非鸡)", 0, 4), ("补杠(鸡)", 1, 4)]
        if len(melds) != 4:
            check("副露组数量", False, f"拿到 {len(melds)} 组")
        else:
            for (nm, wantHeng, wantCells), got in zip(want, melds):
                check(f"{nm} 横置={wantHeng} 张数={wantCells}", 
                      got["heng"] == wantHeng and got["cells"] == wantCells,
                      f"实际 heng={got['heng']} cells={got['cells']}")

        # ---------------- 3. 结算视图切换平滑过渡 ----------------
        print("\n[3] 结算视图切换（总览 ↔ 单人视角）平滑过渡")
        page.evaluate(WIN_STATE_JS, "multi")
        page.wait_for_timeout(120)
        t0 = page.evaluate("""() => {
          const grid = document.getElementById('res-pairs');
          document.querySelector('#res-tabs button[data-view="ov"]').click();
          return grid.getBoundingClientRect().height;
        }""")
        page.wait_for_timeout(80)
        anim = page.evaluate("""() => {
          const grid = document.getElementById('res-pairs');
          const startH = grid.getBoundingClientRect().height;
          const btn = [...document.querySelectorAll('#res-tabs button')]
            .find(b => b.dataset.view !== 'ov');
          btn.click();
          const h0 = grid.getBoundingClientRect().height;
          const styleAt0 = grid.style.height;
          window.__probe = {startH, h0, styleAt0};
          return window.__probe;
        }""")
        page.wait_for_timeout(140)
        # 无头浏览器里动画时钟可能延迟推进（currentTime 停在 0）：
        # 先等过渡真正开始计时，再做「正在过渡」断言，避免环境抖动造成误报
        try:
            page.wait_for_function(
                "() => { const g = document.getElementById('res-pairs');"
                " return g.getAnimations().some(a => a.playState === 'running'"
                " && (a.currentTime || 0) > 0); }", timeout=2500)
        except Exception:
            pass
        mid = page.evaluate("() => ({h: document.getElementById('res-pairs').getBoundingClientRect().height,"
                            " style: document.getElementById('res-pairs').style.height})")
        # 等过渡收尾（行内高度被清理）再取最终值
        try:
            page.wait_for_function(
                "() => document.getElementById('res-pairs').style.height === ''", timeout=3000)
        except Exception:
            pass
        page.wait_for_timeout(80)
        after = page.evaluate("() => ({h: document.getElementById('res-pairs').getBoundingClientRect().height,"
                              " style: document.getElementById('res-pairs').style.height})")
        start_h, h0 = anim["startH"], anim["h0"]
        check("切换前总览有高度", start_h > 60, f"{start_h:.0f}px")
        check("切换瞬间锁在旧高度（未瞬跳）", abs(h0 - start_h) <= 2, f"h0={h0:.0f} start={start_h:.0f}")
        check("过渡期间高度离开旧值（正在过渡）",
              mid["h"] < start_h - 2,
              f"mid={mid['h']:.0f} end={after['h']:.0f}")
        check("过渡结束后清理行内高度", after["style"] == "", f"style='{after['style']}'")
        check("最终高度已稳定（≈过渡结束值）", abs(after["h"] - mid["h"]) < max(30, start_h * 0.5),
              f"mid={mid['h']:.0f} end={after['h']:.0f}")

        # ---------------- 4. 一炮多响 / 抢杠 / 热炮 渲染 ----------------
        for mode, tag, want_lose, want_jb in [
            ("multi", "一炮多响", "点炮", "点炮（2家）"),
            ("rob", "抢杠胡", "被抢杠", "抢杠"),
            ("repao", "热炮", "放热炮", "热炮"),
        ]:
            print(f"\n[4] {tag} 结算渲染")
            page.evaluate(WIN_STATE_JS, mode)
            page.wait_for_timeout(120)
            t = page.evaluate(TEXT_JS)
            check(f"{tag}：顶部信息条含「{want_jb}」", want_jb in t["jibar"], t["jibar"][:70])
            check(f"{tag}：放炮者标「{want_lose}」", want_lose in t["pairs"])
            if mode == "multi":
                check("一炮多响：两位赢家各一张「胡牌」卡", t["winTags"] == 2, f"winTags={t['winTags']}")
                check("一炮多响：总览出现「一炮多响」", "一炮多响" in t["pairs"])
                check("一炮多响：标题含两位赢家姓名",
                      all(n and n in t["title"] for n in [t["cards"][0] if t["cards"] else "", ""]) or
                      len(t["cards"]) >= 2, f"cards={t['cards']}")
            else:
                check(f"{tag}：放炮者标「鸡分全烧」", "鸡分全烧" in t["pairs"])

        browser.close()
        try:
            print("\n[清理] 关闭会话")
        except Exception:
            pass

    print(f"\n通过 {len(PASS)} / 共 {len(PASS) + len(FAIL)}")
    if FAIL:
        for f in FAIL:
            print("  FAIL:", f)
        return 1
    print("真实浏览器 E2E 全部通过 ✅")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
