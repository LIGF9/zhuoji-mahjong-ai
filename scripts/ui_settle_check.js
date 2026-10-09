// 结算新增能力的前端校验（jsdom 加载真实页面 + 真实快照改写）：
//   1) 一炮多响：标题含全部赢家姓名；每位赢家各一张「胡牌」卡；各赢家手牌含点炮张；
//      总览出现「一炮多响」；顶部信息条出现「点炮（2家）」
//   2) 抢杠胡：放炮者标「被抢杠」+「鸡分全烧」；顶部信息条「抢杠」
//   3) 热炮：放炮者标「放热炮」+「鸡分全烧」；顶部信息条「热炮」
//   4) 杠·通行证徽章：四家都必须落在「头像上方」容器 .ava-badges，且不再出现在 .corner-badges
//   5) 视角切换平滑过渡：.pairgrid 高度过渡 + 淡入动画样式就位
const fs = require('fs');
const { JSDOM } = require('jsdom');

const ROOT = 'C:/Users/hyzor/WorkBuddy/2026-09-30-12-49-43/zhuoji-mahjong';
const html = fs.readFileSync(ROOT + '/web/static/dushan.html', 'utf8');
const meta = JSON.parse(fs.readFileSync(ROOT + '/reports/ui_fixtures_meta.json', 'utf8'));
const errs = [];

const dom = new JSDOM(html, {
  runScripts: 'dangerously',
  pretendToBeVisual: true,
  url: 'http://127.0.0.1:8770/static/dushan.html',
  beforeParse(w) {
    w.fetch = (url) => String(url).includes('/api/meta')
      ? Promise.resolve({ ok: true, json: () => Promise.resolve(meta) })
      : new Promise(() => {});
    w.addEventListener('error', e => errs.push('window.error: ' + ((e.error && e.error.stack) || e.message)));
  },
});
const w = dom.window;

let ok = 0, bad = 0;
function check(name, cond, detail = '') {
  (cond ? ok++ : bad++);
  console.log(`  [${cond ? 'ok  ' : 'FAIL'}] ${name}${detail ? '   ' + detail : ''}`);
}

const OVER = JSON.parse(fs.readFileSync(ROOT + '/reports/ui_fixtures_over.json', 'utf8'));
const MID = JSON.parse(fs.readFileSync(ROOT + '/reports/ui_fixtures_mid.json', 'utf8'));

function clone(o) { return JSON.parse(JSON.stringify(o)); }
function relOf(st, seat) { return ((seat - st.my_seat) % 4 + 4) % 4; }
function cardOf(st, seat) {
  const rel = relOf(st, seat);
  const cards = Array.from(w.document.querySelectorAll('#res-pairs .ovcard'));
  const nm = st.players[rel].name;
  return cards.find(c => (c.querySelector('.nm') || {}).textContent &&
    c.querySelector('.nm').textContent.includes(nm)) || null;
}

setTimeout(() => {
  const winFixture = OVER.find(o => o.state.result.type === 'win');
  if (!winFixture) { console.log('没有胡牌夹具，先跑 scripts/make_ui_fixtures.py'); process.exit(2); }

  // ---------------------------------------------------------------- 一炮多响
  try {
    const base = clone(winFixture.state);
    base.my_seat = 0;
    base.players.forEach((p, i) => { p.seat = i; p.name = p.name || `P${i}`; });
    const res0 = base.result;
    const wt = Number.isInteger(res0.win_tile) ? res0.win_tile : 9;
    const w1 = res0.winner;
    const w2 = [0, 1, 2, 3].filter(s => s !== w1 && s !== res0.loser)[0];
    const loser = res0.loser != null ? res0.loser : [0, 1, 2, 3].filter(s => s !== w1 && s !== w2)[0];
    const r1 = relOf(base, w1), r2 = relOf(base, w2), rl = relOf(base, loser);
    // 两家赢家手牌 = 原手牌 + 点炮张（服务端结算后就是这样下发的）
    base.players[r1].hand_tiles = (base.players[r1].hand_tiles || []).concat([wt]);
    base.players[r2].hand_tiles = (base.players[r2].hand_tiles || []).concat([wt]);
    base.players[r2].hand_count = base.players[r2].hand_tiles.length;
    const deltas = [0, 0, 0, 0];
    deltas[w1] = 10; deltas[w2] = 8; deltas[loser] = -18;
    const pair = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]];
    pair[w1][loser] = 10; pair[w2][loser] = 8;
    base.result = Object.assign({}, res0, {
      type: 'win', is_tsumo: false, rob_kong: false, how: '点炮', void_name: null,
      winners: [w1, w2], winner: w1,
      loser, loser_name: base.players[rl].name,
      winner_names: [base.players[r1].name, base.players[r2].name],
      winners_fan: [
        { seat: w1, rel: relOf(base, w1), name: base.players[r1].name, fan_cn: '清一色', fan: 10 },
        { seat: w2, rel: relOf(base, w2), name: base.players[r2].name, fan_cn: '平胡', fan: 8 },
      ],
      winner_hands: { [w1]: base.players[r1].hand_tiles, [w2]: base.players[r2].hand_tiles },
      winner_melds_map: { [w1]: [], [w2]: [] },
      win_tile: wt, deltas, pair,
      pair_detail: [
        { a: w1, b: loser, items: [{ label: '点炮（2家）', info: '', value: 10, who: w1 }] },
        { a: w2, b: loser, items: [{ label: '点炮（2家）', info: '', value: 8, who: w2 }] },
      ],
    });
    w.LAST_STATE = base; w.render(base);
    const title = w.document.getElementById('res-title').innerHTML;
    const ovHtml = w.document.getElementById('res-pairs').innerHTML;
    const winTags = w.document.querySelectorAll('#res-pairs .ovcard .tag.win').length;
    const c1 = cardOf(base, w1), c2 = cardOf(base, w2);
    const n1 = c1 ? c1.querySelectorAll('.ovhandgroup img').length : -1;
    const n2 = c2 ? c2.querySelectorAll('.ovhandgroup img').length : -1;
    const firstTwo = Array.from(w.document.querySelectorAll('#res-pairs .ovcard .nm'))
      .slice(0, 2).map(x => x.textContent);
    const jb = w.document.getElementById('res-jibar').textContent;
    console.log('\n[一炮多响]');
    check('标题含两位赢家姓名', title.includes(base.players[r1].name) && title.includes(base.players[r2].name),
      `title=${title.slice(0, 90)}`);
    check('总览两张「胡牌」卡', winTags === 2, `winTags=${winTags}`);
    check('总览出现「一炮多响」', ovHtml.includes('一炮多响'));
    check('赢家卡排在最前两位',
      firstTwo.length === 2 && firstTwo[0].includes(base.players[r1].name) &&
      firstTwo[1].includes(base.players[r2].name), `first2=${JSON.stringify(firstTwo)}`);
    check('两位赢家手牌都含点炮张（各 14 张）',
      n1 === base.players[r1].hand_tiles.length && n2 === base.players[r2].hand_tiles.length,
      `w1=${n1}/${base.players[r1].hand_tiles.length} w2=${n2}/${base.players[r2].hand_tiles.length}`);
    check('顶部信息条标「点炮（2家）」', jb.includes('点炮（2家）'), `jibar="${jb}"`);
    const cl = cardOf(base, loser);
    const loseTag = cl ? (cl.querySelector('.tag.lose') || {}).textContent : '';
    check('放炮者卡标「点炮」', loseTag === '点炮', `loseTag="${loseTag}"`);
  } catch (e) {
    bad++; console.log('[一炮多响] THREW: ' + String(e.stack).split('\n').slice(0, 5).join('\n    '));
  }

  // ------------------------------------------------- 抢杠 / 热炮（全烧）
  for (const [tag, patch, wantLose, wantJb] of [
    ['抢杠胡', { rob_kong: true, how: '抢杠', is_tsumo: false }, '被抢杠', '抢杠'],
    ['热炮', { rob_kong: false, how: '热炮', is_tsumo: false }, '放热炮', '热炮'],
  ]) {
    try {
      const base = clone(winFixture.state);
      base.my_seat = 0;
      base.players.forEach((p, i) => { p.seat = i; });
      const res0 = base.result;
      const wid = res0.winner;
      const loser = res0.loser != null ? res0.loser : [0, 1, 2, 3].filter(s => s !== wid)[0];
      const rw = relOf(base, wid), rl = relOf(base, loser);
      const voidName = base.players[rl].name;
      const deltas = [0, 0, 0, 0];
      deltas[wid] = 8; deltas[loser] = -8;
      base.result = Object.assign({}, res0, patch, {
        type: 'win', winners: [wid], winner: wid,
        winner_names: [base.players[rw].name],
        winner_hands: { [wid]: base.players[rw].hand_tiles || [] },
        winner_melds_map: { [wid]: [] },
        winners_fan: [{ seat: wid, rel: relOf(base, wid), name: base.players[rw].name, fan_cn: '平胡', fan: 8 }],
        loser, loser_name: voidName, void_name: voidName, deltas,
        pair: [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
        pair_detail: [{ a: wid, b: loser, items: [{ label: patch.how, info: '', value: 8, who: wid }] }],
      });
      base.result.pair[wid][loser] = 8;
      w.LAST_STATE = base; w.render(base);
      const ovHtml = w.document.getElementById('res-pairs').innerHTML;
      const jb = w.document.getElementById('res-jibar').textContent;
      console.log(`\n[${tag}]`);
      check(`放炮者标「${wantLose}」`, ovHtml.includes(wantLose));
      check('放炮者标「鸡分全烧」', ovHtml.includes('鸡分全烧'));
      check(`顶部信息条出现「${wantJb}」`, jb.includes(wantJb), `jibar="${jb}"`);
      check('不误标「一炮多响」', !ovHtml.includes('一炮多响'));
    } catch (e) {
      bad++; console.log(`[${tag}] THREW: ` + String(e.stack).split('\n').slice(0, 5).join('\n    '));
    }
  }

  // --------------------------------------------------------- 杠·通行证位置
  try {
    const st = clone((MID[0] || OVER[0]).state);
    st.over = false;
    st.players.forEach((p, i) => { p.has_gang = true; p.baojiao = true; p.seat = i; });
    w.LAST_STATE = st; w.render(st);
    const meAb = w.document.getElementById('me-abadges');
    const meCb = w.document.getElementById('me-badges');
    const avaAll = Array.from(w.document.querySelectorAll('.player .ava-badges'));
    const avaWithBadge = avaAll.filter(x => x.textContent.includes('杠·通行证')).length;
    const cornerWithBadge = Array.from(w.document.querySelectorAll('.player .corner-badges'))
      .filter(x => x.textContent.includes('杠·通行证')).length;
    const cornerWithJiao = Array.from(w.document.querySelectorAll('.player .corner-badges'))
      .filter(x => x.textContent.includes('报叫')).length;
    console.log('\n[杠·通行证位置]');
    check('我的：徽章在头像上方容器', !!meAb && meAb.textContent.includes('杠·通行证'));
    check('我的：面板角标只留「报叫」',
      !!meCb && !meCb.textContent.includes('杠·通行证') && meCb.textContent.includes('报叫'));
    check('四家头像上方容器都出现徽章', avaWithBadge === 4, `avaWithBadge=${avaWithBadge} / ${avaAll.length}`);
    check('没有任何「杠·通行证」残留在面板角标', cornerWithBadge === 0, `cornerWithBadge=${cornerWithBadge}`);
    check('面板角标仍有「报叫」', cornerWithJiao === 4, `cornerWithJiao=${cornerWithJiao}`);
    check('CSS: .ava-badges 定位在头像上方', html.includes('.ava-badges { position: absolute; bottom: calc(100% + 8px)'));
  } catch (e) {
    bad++; console.log('[杠·通行证位置] THREW: ' + String(e.stack).split('\n').slice(0, 5).join('\n    '));
  }

  // ------------------------------------------------------- 视角切换平滑过渡
  console.log('\n[视角切换过渡]');
  check('CSS: .pairgrid 高度过渡', /\.pairgrid \{[^}]*transition: height/.test(html));
  check('CSS: 过渡期间 overflow hidden', html.includes('.pairgrid.animating { overflow: hidden; }'));
  check('CSS: 切换淡入动画', html.includes('@keyframes resViewFade'));
  check('CSS: 尊重 prefers-reduced-motion', html.includes('prefers-reduced-motion'));
  check('JS: 仅在真正切换视角时做过渡', html.includes('const changed = RES_VIEW_PREV !== RES_VIEW;'));

  console.log(`\nok=${ok} bad=${bad}`);
  if (errs.length) console.log('window errors:\n' + errs.slice(0, 5).join('\n'));
  process.exit(bad ? 1 : 0);
}, 1800);
