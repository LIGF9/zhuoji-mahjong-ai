// 用 jsdom 加载真实页面，注入真实结算态调用 render()，定位前端渲染异常
const fs = require('fs');
const { JSDOM } = require('jsdom');

const ROOT = 'C:/Users/hyzor/WorkBuddy/2026-09-30-12-49-43/zhuoji-mahjong';
const html = fs.readFileSync(ROOT + '/web/static/dushan.html', 'utf8');
const errs = [];

const meta = JSON.parse(fs.readFileSync(ROOT + '/reports/ui_fixtures_meta.json', 'utf8'));

const dom = new JSDOM(html, {
  runScripts: 'dangerously',
  pretendToBeVisual: true,
  url: 'http://127.0.0.1:8770/static/dushan.html',
  beforeParse(w) {
    // 只喂 /api/meta（填充 TILES），其余请求挂起，让 boot() 停住但不抛未捕获异常
    w.fetch = (url) => {
      if (String(url).includes('/api/meta')) {
        return Promise.resolve({ ok: true, json: () => Promise.resolve(meta) });
      }
      return new Promise(() => {});
    };
    w.addEventListener('error', e => errs.push('window.error: ' + ((e.error && e.error.stack) || e.message)));
  },
});
const w = dom.window;

// ---------------------------------------------------------------------------
// 总览断言（结算展示口径）
//   1) 手牌组 / 副露组张数与数据一致，组内牌距 0
//   2) 听牌精简：只显示「听 <牌面>（牌型）」，不再有重复的「未听 / 未听牌」两个标签
//   3) 副露说明：碰鸡 → 碰<鸡类>(来源)；杠 → 杠<牌面>(自杠/补杠/来源)；普通碰不标注
//   4) 打出的鸡牌 → 直接渲染带标签牌图（替代「冲锋鸡 ×1」文字）
// ---------------------------------------------------------------------------
function checkOverview(st, cards, htmlSrc) {
  const relOf = s => ((s - st.my_seat) % 4 + 4) % 4;
  const nameOf = s => (st.players[relOf(s)] || {}).name || "";
  const msgs = [];
  let ok = cards.length === 4;
  // 展示顺序与前端同口径：胡牌局 = 赢家、我、其余两家；流局 0..3
  const res0 = st.result || {};
  const order = res0.type === "win"
    ? [((res0.winner - st.my_seat) % 4 + 4) % 4, ...[0, 1, 2, 3].filter(r => r !== ((res0.winner - st.my_seat) % 4 + 4) % 4)]
    : [0, 1, 2, 3];
  // 顺序断言：胡牌局第一张卡必须是赢家
  if (res0.type === "win") {
    const firstNm = cards[0] ? cards[0].querySelector('.nm').textContent : "";
    const wantNm = st.players[order[0]].name;
    if (!firstNm.includes(wantNm)) { ok = false; msgs.push(`首卡=${firstNm} 应为赢家 ${wantNm}`); }
  }

  for (let i = 0; i < cards.length; i++) {
    const rel = order[i];
    const p = st.players[rel];
    const card = cards[i];
    const probs = [];
    // 1) 张数
    const hg = card.querySelectorAll('.ovhandgroup img').length;
    const groups = card.querySelectorAll('.ovmelds .meld').length;
    if (hg !== (p.hand_tiles || []).length) probs.push(`手牌 ${hg}/${(p.hand_tiles || []).length}`);
    if (groups !== (p.melds || []).length) probs.push(`副露组 ${groups}/${(p.melds || []).length}`);
    // 2) 听牌精简
    const tags = Array.from(card.querySelectorAll('.ovmeta .tag')).map(x => x.textContent);
    if (card.innerHTML.includes('未听牌')) probs.push('仍有「未听牌」重复标签');
    const wantTing = (p.ting_tiles || []).length;
    const gotTing = card.querySelectorAll('.ovmeta .tag img.tin').length;
    if (wantTing !== gotTing) probs.push(`听牌牌面 ${gotTing}/${wantTing}`);
    if (wantTing && !tags.some(t => t.startsWith('听 '))) probs.push('缺「听 …」标签');
    const offTags = tags.filter(t => t === '未听').length;
    if (!wantTing && !offTags) probs.push('未听者无「未听」标签');
    if (offTags > 1) probs.push('「未听」标签重复');
    // 3) 副露说明
    const caps = Array.from(card.querySelectorAll('.ovmeta .cap')).map(x => x.textContent);
    for (const m of (p.melds || [])) {
      const want = (m.type === 1 && m.ji) ? `碰${m.ji}`
        : (m.type === 2 || m.type === 3 || m.type === 4) ? '杠' : null;
      if (!want) continue;                       // 普通碰牌不标注
      if (!caps.some(c => c.startsWith(want))) { probs.push(`缺副露说明「${want}」`); continue; }
      const tail = m.type === 4 ? '自杠' : m.type === 3 ? '补杠'
        : (Number.isInteger(m.src) ? nameOf(m.src) : '明杠');
      if (!caps.some(c => c.startsWith(want) && c.includes(`(${tail})`))) {
        probs.push(`副露说明缺(${tail})`);
      }
    }
    // 4) 打出的鸡牌牌图（冲锋鸡/横鸡专用牌面；普通鸡=鸡牌种弃牌用普通牌面）
    const jiSetC = new Set(st.ji_tiles && st.ji_tiles.length ? st.ji_tiles : [9]);
    const wantDisc = (p.discards || []).filter((t, i) => {
      const tg = (p.discard_tags || [])[i];
      return tg === '冲锋鸡' || tg === '横鸡' || jiSetC.has(t);
    }).length;
    const gotDisc = card.querySelectorAll('.ovmeta .disc img').length;
    if (wantDisc !== gotDisc) probs.push(`打出鸡牌图 ${gotDisc}/${wantDisc}`);
    // 5) 手牌中的鸡牌带 🐔 标记
    const wantJiHand = (p.hand_tiles || []).filter(t => jiSetC.has(t)).length;
    const gotJiHand = card.querySelectorAll('.ovhand .ovtile .jimark').length;
    if (wantJiHand !== gotJiHand) probs.push(`手牌鸡标记 ${gotJiHand}/${wantJiHand}`);
    if (probs.length) { ok = false; msgs.push(`p${rel}:${probs.join('、')}`); }
  }
  const cssOk = htmlSrc.includes('.ovhand .ovhandgroup { display: flex; gap: 0; }')
    && htmlSrc.includes('.ovhand .meld { display: inline-flex; gap: 0;')
    && htmlSrc.includes('.ovmeta .disc img')
    && htmlSrc.includes('.ovmeta .cap img');
  if (!cssOk) { ok = false; msgs.push('零间距/标签样式缺失'); }
  return { ok, msgs };
}

// 构造态：覆盖「听牌 + 四种副露 + 打出的鸡牌」——保证上面每条断言都真的被跑到
function buildCraftedState() {
  const base = JSON.parse(fs.readFileSync(ROOT + '/reports/ui_fixtures_over.json', 'utf8'))[0].state;
  const st = JSON.parse(JSON.stringify(base));
  st.my_seat = 0;
  st.players = st.players.map((p, i) => Object.assign({}, p, { seat: i }));
  st.players[0].name = "你";
  st.players[1].name = "王昭君(S7)";
  st.players[2].name = "妲己(B3)";
  st.players[3].name = "大乔(抢听)";
  st.players[0].hand_tiles = [0, 1, 2, 3, 4, 5, 6, 7, 8, 19, 20, 25, 25];
  st.players[0].ting_tiles = [21, 24];
  st.players[0].ting_types = ["清一色"];
  st.players[0].melds = [{ type: 1, label: "碰", tile: 9, src: 3, ji: "横鸡" },
                         { type: 4, label: "暗杠", tile: 17, src: 0, ji: null }];
  st.players[0].discards = [9, 13];
  st.players[0].discard_tags = ["横鸡", null];
  st.players[0].hand_count = 7;
  st.players[1].hand_tiles = [0, 1, 2, 9, 10, 11, 18, 19, 20, 26, 26];
  st.players[1].ting_tiles = [3];
  st.players[1].ting_types = ["平胡"];
  st.players[1].melds = [{ type: 2, label: "明杠", tile: 13, src: 2, ji: "冲锋鸡" }];
  st.players[1].discards = [13, 22];
  st.players[1].discard_tags = [null, "冲锋鸡"];
  st.players[1].hand_count = 11;
  st.players[2].hand_tiles = [4, 5, 6, 15, 16, 17, 23, 24, 25, 26];
  st.players[2].ting_tiles = [];
  st.players[2].ting_types = [];
  st.players[2].melds = [{ type: 3, label: "补杠", tile: 25, src: 2, ji: null }];
  st.players[2].discards = [1];
  st.players[2].discard_tags = ["冲锋鸡"];
  st.players[2].hand_count = 10;
  st.players[3].hand_tiles = [7, 8, 12, 14, 20, 21, 22, 23];
  st.players[3].ting_tiles = [];
  st.players[3].ting_types = [];
  st.players[3].melds = [{ type: 1, label: "碰", tile: 11, src: 1, ji: "冲锋鸡" }];
  st.players[3].discards = [2];
  st.players[3].discard_tags = ["横鸡"];
  st.players[3].hand_count = 8;
  st.result = {
    type: "huangzhuang", tenpai: [0, 1], tenpai_names: ["你", "王昭君(S7)"],
    baodapai_on: true, zi_mo: 3, deltas: [26, 26, -26, -26],
    baodapai: [{ seat: 0, rel: 0, name: "你", type: "清一色", type_value: 10, pay: 13 },
               { seat: 1, rel: 1, name: "王昭君(S7)", type: "平胡", type_value: 0, pay: 3 }],
    pair: [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
    pair_detail: [], detail: [], chickens: [0, 0, 0, 0], wall_left: 0,
    jiesuan_fanji: "down", kaiju_fanji: "both",
  };
  st.kaiju_flip = 12;
  st.ji_tiles = [9, 12, 14];
  return st;
}

setTimeout(() => {
  const data = JSON.parse(fs.readFileSync(ROOT + '/reports/ui_fixtures_over.json', 'utf8'))
    .concat(JSON.parse(fs.readFileSync(ROOT + '/reports/ui_fixtures_mid.json', 'utf8')));
  let ok = 0, bad = 0;
  for (const item of data) {
    const st = item.state;
    const tag = item.seed != null ? 'seed ' + item.seed : item.tag;
    try {
      w.LAST_STATE = st;
      w.render(st);
      const mask = w.document.getElementById('result-mask');
      const noHandStrip = w.document.getElementById('res-hand') === null;
      const pairs = w.document.getElementById('res-pairs').innerHTML;
      const title = w.document.getElementById('res-title').innerHTML;
      if (!st.over) {
        // 对局进行中：结算面板必须关闭
        const closed = !mask.className.includes('on');
        console.log(`${tag} mid-game | mask="${mask.className}" closed=${closed} ${closed ? 'OK' : 'FAIL'}`);
        if (closed) ok++; else bad++;
        continue;
      }
      console.log(`${tag} ${st.result.type} | mask="${mask.className}" title=${title.length} 赢家牌条已删=${noHandStrip} pairs=${pairs.length} OK`);
      if (st.result.type === 'win' && !noHandStrip) { bad++; console.log('   FAIL: res-hand 应已删除'); }
      if (st.result.type !== 'win' || true) {
        const tabs = Array.from(w.document.querySelectorAll('#res-tabs button'));
        for (const b of tabs) {
          const before = errs.length;
          try { b.click(); } catch (e) { console.log(`   tab ${b.textContent} threw ${e}`); }
          const pl = w.document.getElementById('res-pairs').innerHTML.length;
          if (errs.length > before) console.log(`   tab ${b.textContent} -> window error: ${errs[errs.length - 1]}`);
          else console.log(`   tab ${b.textContent} ok pairs=${pl}`);
        }
      }
      ok++;

      // 切回总览视图再做断言（前面的 tab 点击会把视图切到某视角）
      const ovBtn = w.document.querySelector('#res-tabs button[data-view="ov"]');
      if (ovBtn) ovBtn.click();
      const ov = w.document.getElementById('res-pairs');
      const cards = ov.querySelectorAll('.ovcard');
      const r = checkOverview(st, cards, html);
      console.log(`   总览 ${r.ok ? 'OK' : 'FAIL'} | ${r.msgs.join(' | ') || '张数/听牌/副露说明/打出鸡牌 全部正确'}`);
      if (r.ok) ok++; else bad++;
    } catch (e) {
      bad++;
      console.log(`${tag} | RENDER THREW: ${String(e.stack).split('\n').slice(0, 7).join('\n    ')}`);
    }
  }

  // ---- 构造态：听牌 + 碰鸡/明杠/补杠/暗杠 + 打出的冲锋鸡/横鸡 ----
  try {
    const st = buildCraftedState();
    w.LAST_STATE = st;
    w.render(st);
    const cards = w.document.querySelectorAll('#res-pairs .ovcard');
    const r = checkOverview(st, cards, html);
    const jb = w.document.getElementById('res-jibar').textContent;
    const jbOk = jb.includes('本局鸡牌') && jb.includes('仅下鸡') && jb.includes('开局翻鸡');
    const noSub = !w.document.getElementById('res-sub');
    const meldCount = w.document.querySelectorAll('#res-pairs .ovcard .ovmelds .meld').length;
    // 对局中被碰鸡牌：不再显示「点碰者」小标（横置牌的方向+位置已能指示来源）
    const chips = Array.from(w.document.querySelectorAll('#me-melds .msrc, .seat-left .msrc, .seat-right .msrc, .seat-top .msrc'))
      .map(x => x.textContent);
    const chipOk = chips.length === 0;
    const ovChips = w.document.querySelectorAll('#res-pairs .msrc').length;
    // 左右家（rel3/rel1）组内被碰鸡牌应与组内垂直：组内牌带 rot/rotr，横置牌不带
    const perpOk =
      w.document.querySelectorAll('.seat-left .melds .meld .mc:not(.rot):not(.rotr) img').length >= 1 &&
      w.document.querySelectorAll('.seat-right .melds .meld .mc:not(.rot):not(.rotr) img').length >= 1;
    console.log(`构造态 流局+听牌 | 副露组=${meldCount}(应 5) 总览 ${r.ok ? 'OK' : 'FAIL'} | ${r.msgs.join(' | ') || '全部正确'}`);
    console.log(`   结算顶部信息条: "${jb}" ${jbOk ? 'OK' : 'FAIL'}｜旧小字已移除 ${noSub ? 'OK' : 'FAIL'}`);
    console.log(`   对局中碰鸡小标已移除 ${chipOk ? 'OK' : `FAIL ${JSON.stringify(chips)}`}｜总览无小标=${ovChips === 0} ${ovChips === 0 ? 'OK' : 'FAIL'}｜左右家横置垂直=${perpOk ? 'OK' : 'FAIL'}`);
    if (r.ok && meldCount === 5 && jbOk && noSub && chipOk && ovChips === 0 && perpOk) ok++; else bad++;
  } catch (e) {
    bad++;
    console.log(`构造态 | RENDER THREW: ${String(e.stack).split('\n').slice(0, 7).join('\n    ')}`);
  }

  console.log(`\nok=${ok} threw=${bad}`);
  if (errs.length) console.log('window errors:\n' + errs.slice(0, 6).join('\n'));
  process.exit(0);
}, 1800);
