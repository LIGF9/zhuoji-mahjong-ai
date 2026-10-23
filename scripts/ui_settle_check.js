// 结算新增能力的前端校验（jsdom 加载真实页面 + 真实快照改写）：
//   1) 一炮多响：标题含全部赢家姓名；每位赢家各一张「胡牌」卡；各赢家手牌含点炮张；
//      总览出现「一炮多响」；顶部信息条出现「点炮（2家）」
//   2) 抢杠胡：放炮者标「被抢杠」+「鸡分全烧」；顶部信息条「抢杠」
//   3) 热炮：放炮者标「放热炮」+「鸡分全烧」；顶部信息条「热炮」
//   4) 杠·通行证徽章：四家都必须落在「头像上方」容器 .ava-badges，且不再出现在 .corner-badges
//   5) 视角切换平滑过渡：.pairgrid 高度过渡 + 淡入动画样式就位
const fs = require('fs');
const { JSDOM } = require('jsdom');

const ROOT = 'C:/Users/LIGF/WorkBuddy/2026-09-30-12-49-43/zhuoji-mahjong';
const html = fs.readFileSync(ROOT + '/web/static/dushan.html', 'utf8');
const meta = JSON.parse(fs.readFileSync(ROOT + '/reports/ui_fixtures_meta.json', 'utf8'));
const errs = [];

const dom = new JSDOM(html, {
  runScripts: 'dangerously',
  pretendToBeVisual: true,
  url: 'http://127.0.0.1:8770/static/dushan.html',
  beforeParse(w) {
    // jsdom 未实现 matchMedia（fitStage 用它判定触屏）→ 补一个「非触屏」桩
    if (!w.matchMedia) w.matchMedia = q => ({ matches: false, media: String(q),
      onchange: null, addEventListener() {}, removeEventListener() {},
      addListener() {}, removeListener() {}, dispatchEvent: () => false });
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
// 2026-10-11 起名字与头像绑定：卡片不再按 fixture 名字定位（显示名是派生名），
// 改用「头像文件名」定位——sessionAvatars()[rel] 在无随机头像时是确定值
function cardOf(st, seat) {
  const rel = relOf(st, seat);
  const ava = w.eval(`(sessionAvatars()[${rel}] || "")`);
  const cards = Array.from(w.document.querySelectorAll('#res-pairs .ovcard'));
  return cards.find(c => {
    const a = c.querySelector('.avatar');
    return a && ava && String(a.getAttribute('src')).includes(`/${ava}.jpg`);
  }) || null;
}
// 前端实际显示名（派生名）
const dispName = (st, seat) =>
  w.eval(`playerName(${relOf(st, seat)}, ${JSON.stringify(st.players[relOf(st, seat)].name || "")})`);

setTimeout(() => {
  // 单人视角默认已是「新版布局」（共有板块 / 各家独有项之上补一行 头像+手牌+标签）。
  // 本文件校验的是**表格本身**的口径（共享项置顶、块头、分割线、恒等式），与布局无关，
  // 因此显式切回旧版纯表格；新版布局另由 scripts/ui_aux_check.js 覆盖。
  w.eval('SET.resLayout = "old"');

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
    check('赢家卡排在最前两位（显示名 = 派生名）',
      firstTwo.length === 2 && firstTwo[0].includes(dispName(base, w1)) &&
      firstTwo[1].includes(dispName(base, w2)), `first2=${JSON.stringify(firstTwo)}`);
    check('两位赢家手牌都含点炮张（各 14 张）',
      n1 === base.players[r1].hand_tiles.length && n2 === base.players[r2].hand_tiles.length,
      `w1=${n1}/${base.players[r1].hand_tiles.length} w2=${n2}/${base.players[r2].hand_tiles.length}`);
    check('顶部信息条标「捉炮（2家）」（2026-10-13 责任口径）', jb.includes('捉炮（2家）'), `jibar="${jb}"`);
    const cl = cardOf(base, loser);
    const loseTag = cl ? (cl.querySelector('.tag.lose') || {}).textContent : '';
    check('放炮者卡标「放炮」', loseTag === '放炮', `loseTag="${loseTag}"`);
  } catch (e) {
    bad++; console.log('[一炮多响] THREW: ' + String(e.stack).split('\n').slice(0, 5).join('\n    '));
  }

  // ------------------------------------------------- 抢杠 / 热炮（全烧）
  for (const [tag, patch, wantLose, wantJb] of [
    ['抢杠胡', { rob_kong: true, how: '抢杠', is_tsumo: false }, '被抢杠', '抢杠'],
    ['热炮', { rob_kong: false, how: '热炮', is_tsumo: false }, '放热炮', '捉热炮'],
  ]) {
    try {
      const base = clone(winFixture.state);
      base.my_seat = 0;
      base.players.forEach((p, i) => { p.seat = i; });
      const res0 = base.result;
      const wid = res0.winner;
      const loser = res0.loser != null ? res0.loser : [0, 1, 2, 3].filter(s => s !== wid)[0];
      const rw = relOf(base, wid), rl = relOf(base, loser);
      const voidName = dispName(base, loser);   // 服务端下发的名字=派生名，与前端一致
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

  // ------------------------------------- 单人视角：鸡牌三分类 + 块区分 + 总计行满宽分割线
  // 2026-10-13 用户口径：结算里的鸡分只保留三类 —— 手中鸡牌 / 打出鸡牌 / 责任鸡，
  // 每一类都要给出「哪一张牌有几张」。分值一律取自 res.detail（与计分同源）：
  //   · 手中鸡牌 = 手中鸡 + 翻鸡 + 捉炮鸡（chkSum.hold）
  //   · 打出鸡牌 = 冲锋鸡 / 横鸡 / 普通鸡（chkSum.disc）
  //   · 责任鸡   = 被碰/杠走那张的溢价（res.ji_claim_detail，天生只发生在两家之间）
  // extra 用来塞新字段（detail / ji_*_detail / gold_tiles 等）。
  function pairState(pairDetail, pair, deltas, extra) {
    const base = clone(winFixture.state);
    base.my_seat = 0;
    base.players.forEach((p, i) => { p.seat = i; p.name = p.name || `P${i}`; });
    base.result = Object.assign({}, base.result, {
      type: 'win', is_tsumo: true, rob_kong: false, how: '自摸',
      winners: [2], winner: 2, loser: null, void_name: null,
      fanji_tiles: [9], deltas, pair, pair_detail: pairDetail,
      // 新字段默认清空，避免继承 winFixture 里那一局的鸡牌明细
      detail: [], ji_hold_detail: {}, fanji_detail: {}, ji_discard_detail: {},
      ji_capture_detail: {}, ji_claim_detail: [], gold_tiles: [], gold_mult: 2,
    }, extra || {});
    return base;
  }
  function renderView(st, seat) {
    w.__st = st;
    w.eval(`LAST_STATE = window.__st; RES_VIEW = ${seat}; RES_DISMISSED = false; render(window.__st);`);
    return Array.from(w.document.querySelectorAll('#res-pairs .pairs-big tr'));
  }
  const cellNum = tr => { const m = (tr.lastElementChild.textContent.match(/[+-]?\d+(?:\.\d+)?/) || ['0'])[0]; return parseFloat(m); };
  const detNum = tr => { const m = (tr.children[3].textContent.match(/[+-]?\d+(?:\.\d+)?/) || ['0'])[0]; return parseFloat(m); };
  const cnt = (s, sub) => s.split(sub).length - 1;
  // 把渲染出的表格切成「共享段」与「各对手块」：块头 = tr.opphead，块内明细 = 紧随其后的
  // tr.det（不带 sh 类）。共享段明细是 tr.det.sh。
  function blocksOf(rows) {
    const out = { shared: [], ops: [] };
    let cur = null;
    for (const r of rows) {
      const c = r.className;
      if (c === 'opphead') { cur = { head: r, dets: [] }; out.ops.push(cur); continue; }
      if (c.includes('gap') || c.includes('striprow')) { cur = null; continue; }
      if (c.startsWith('det')) {
        if (c.includes('sh')) out.shared.push(r);
        else if (cur) cur.dets.push(r);
        continue;
      }
      cur = null;
    }
    return out;
  }
  const sumDets = arr => Math.round(arr.reduce((s, r) => s + detNum(r), 0) * 100) / 100;
  const r2 = x => Math.round(x * 100) / 100;
  const labelOf = r => r.children[1].textContent.replace(/人均/g, '').trim();
  // 归属角注（2026-10-13 三次改版）：**纯色小字** <i class="own|opp">，在第 1 列（名字列）——
  // 最初放 label 后面会被挤换行；换成第 1 列 pill 后又嫌「很长」（用户截图反馈）→ 去 pill 改小字
  const tagOf = r => {
    const i = r.children[0].querySelector('i');
    return i ? i.className.trim() : '';
  };
  const cn = t => w.eval(`TILES[${t}].cn`);

  console.log('\n[鸡牌三分类·手中鸡牌 / 打出鸡牌 / 责任鸡]');
  try {
    // 用户 2026-10-13 举例：开局翻鸡 幺鸡/五条 → 五条 ∩ 翻鸡 = 金鸡（×2）；
    //   视角玩家(0) 手里 五条×3（金鸡）+ 七条×2（结算翻鸡），打出 冲锋幺鸡×3，
    //   并碰走了 P1 打出的 五条横鸡（金鸡）→ 责任鸡溢价 2（碰横鸡 2 − 手中鸡 1 = 1，金鸡 ×2）。
    // 期望：三类各一 —— 手中鸡牌 [五条]×3 X2 + [七条]×2 / 打出鸡牌 [冲锋幺鸡]×3 /
    //       责任鸡 [五条横鸡金鸡]×2（只出现在 P1 那一块）。
    const TR_5T = 22;   // 占位牌号：真实中文名由 TILES 决定，断言里动态取
    const st1 = pairState(
      [1, 2, 3].map(o => ({
        a: 0, b: o,
        items: [{ label: '手中鸡', info: '3 只（含金鸡 3 张 ×2）', value: 6, who: 0 },
                { label: '翻鸡', info: '共持有 2 张', value: 2, who: 0 },
                { label: '冲锋鸡', info: '打出', value: 3, who: 0 }]
          .concat(o === 1 ? [{ label: '责任鸡', info: '横鸡', value: 2, who: 0 }] : []),
      })),
      [[0, 13, 11, 11], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
      [35, -13, -11, -11],
      { detail: [
          { who: '全场', seat: -1, label: '翻鸡', info: '五条（金鸡）', value: 0 },
          { who: '我', seat: 0, label: '手中鸡', info: '3 只（含金鸡 3 张 ×2）', value: 6 },
          { who: '我', seat: 0, label: '翻鸡', info: '共持有 2 张', value: 2 },
          { who: '我', seat: 0, label: '冲锋鸡', info: '打出', value: 3 },
          { who: 'P1', seat: 1, label: '责任鸡', info: '横鸡', value: 2 }],
        ji_hold_detail: { '0': [[TR_5T, 3]] },
        fanji_detail: { '0': [[23, 2]] },
        ji_discard_detail: { '0': [{ tile: 9, tag: '冲锋鸡', val: 3 }] },
        ji_claim_detail: [{ claimer: 0, discarder: 1, tile: TR_5T, tag: '横鸡', premium: 2, gold: true }],
        gold_tiles: [TR_5T], gold_mult: 2 });
    const rows = renderView(st1, 0);
    const vName = dispName(st1, 0);
    const html2 = w.document.getElementById('res-pairs').innerHTML;
    const B = blocksOf(rows);
    const shsum = rows.find(r => r.className.includes('shsum'));
    const total = rows.find(r => r.className === 'grandrow');
    check('不再出现「对三家均生效」组头', !html2.includes('对三家均生效'));
    check('表头行（项目/说明/积分）已删除',
      w.document.querySelectorAll('#res-pairs .pairs-big th').length === 0);
    check('「各家独有项」提示行已删除', !html2.includes('各家独有项'));
    check('末行「本局总计」已删除', !html2.includes('本局总计'));
    check('首行居中显示「本局结算总计 +35」', (() => {
      const t = total && total.firstElementChild;
      return !!t && t.getAttribute('colspan') === '5' && t.textContent.includes('本局结算总计')
        && t.textContent.includes('+35');
    })(), total ? total.firstElementChild.textContent : '无');
    check('不再有独立「鸡分」段（无 tr.sum.chksum 行）',
      !rows.some(r => r.className.includes('chksum')));
    // ---- ① 共有段（2026-10-13 三次口径）：只有我方 2 行（人均）——对手鸡牌归各自专属块
    check('共有段 = 2 行我方鸡牌（手中鸡牌8 / 打出鸡牌3）',
      B.shared.length === 2 && B.shared.map(detNum).join(',') === '8,3',
      `n=${B.shared.length} vals=${B.shared.map(detNum).join(',')}`);
    check('共有段两行都是鸡分项（.chk 青绿底）+ 「人均」角注',
      B.shared.every(r => r.classList.contains('chk')) &&
      B.shared.map(labelOf).join(',') === '手中鸡牌,打出鸡牌' &&
      B.shared.every(r => tagOf(r) === ''),
      `labels=${B.shared.map(labelOf).join('/')} tags=${B.shared.map(tagOf).join('/')}`);
    check('共有块头 = 「共有分项」+「对所有玩家生效（我方行按每家一份 ×3 计入总计）」+ 人均合计 +11',
      shsum && shsum.children[0].textContent.trim() === '' &&
      shsum.children[1].textContent.trim() === '共有分项' &&
      shsum.children[2].textContent.includes('对所有玩家生效') &&
      shsum.children[2].textContent.includes('×3') && cellNum(shsum) === 11,
      shsum ? [1, 2].map(i => shsum.children[i].textContent).join('|') : '无');
    check('共有块头在最上面（ownstrip 之后、先于共有明细）',
      shsum && B.shared.length > 0 && rows.indexOf(shsum) < rows.indexOf(B.shared[0]) &&
      ['grandrow', 'pstriprow ownstrip'].includes(rows[rows.indexOf(shsum) - 1].className));
    check('我方共有明细行不带名字（第一列空）',
      B.shared.every(r => r.children[0].textContent.trim() === ''));
    // ---- ② 三类明细都给「哪一张牌有几张」 ----
    const rHold = B.shared.find(r => labelOf(r) === '手中鸡牌');
    const rDisc = B.shared.find(r => labelOf(r) === '打出鸡牌');
    check('手中鸡牌行 = 牌图×张数（金鸡带金角标 + X倍率）',
      !!rHold && /×3/.test(rHold.innerHTML) && /X2/.test(rHold.innerHTML) &&
      rHold.innerHTML.includes('jitag jin') && cnt(rHold.innerHTML, 'class="jcnt"') === 2,
      rHold ? rHold.innerHTML.replace(/<img[^>]*>/g, '[牌]').slice(0, 200) : '无');
    check('打出鸡牌行 = 牌图×分值（冲锋鸡带「冲」角标）',
      !!rDisc && /×3/.test(rDisc.innerHTML) && rDisc.innerHTML.includes('jitag chf') &&
      rDisc.innerHTML.includes('jcnts'),
      rDisc ? rDisc.innerHTML.replace(/<img[^>]*>/g, '[牌]').slice(0, 200) : '无');
    check('打出鸡牌的工具提示写明「打出 N 张 / 共 M 分」',
      !!rDisc && /打出 1 张/.test(rDisc.innerHTML) && /共 3 分/.test(rDisc.innerHTML));
    check('旧口径的 6 类散项标题已不再作为明细行出现',
      !rows.some(r => r.className.startsWith('det') &&
        ['冲锋鸡', '横鸡', '幺鸡', '捉炮鸡'].includes(labelOf(r))));
    // ---- ③ 责任鸡只落在那一对对手块里（2026-10-13 二次口径：块里只有两家专属分项）----
    check('恰好 1 行「责任鸡」明细', rows.filter(r => r.className.startsWith('det') &&
      labelOf(r) === '责任鸡').length === 1);
    check('只有 P1 有两家专属分项 → 只出 1 块（共有分项不再重复进块）',
      B.ops.length === 1 && B.ops[0].dets.length === 1,
      `n=${B.ops.length} dets=${B.ops.map(o => o.dets.length).join(',')}`);
    check('块头第 3 列 = 「两家专属往来」，末列 = 块内合计 +2',
      B.ops.every(o => o.head.children[2].textContent.trim() === '两家专属往来') &&
      B.ops.map(o => cellNum(o.head)).join(',') === '2',
      B.ops.map(o => cellNum(o.head)).join(','));
    check('责任鸡落在 P1 那一块，值为 +2（我碰走了他打出的横鸡）',
      (() => {
        const z = B.ops[0].dets.find(r => labelOf(r) === '责任鸡');
        return !!z && detNum(z) === 2 && z.classList.contains('chk') &&
          z.innerHTML.includes('jitag hj') && z.innerHTML.includes('jitag jin') &&
          z.innerHTML.includes('jcnts');
      })(), B.ops[0].dets.map(r => labelOf(r) + detNum(r)).join(' '));
    check('块内「我 / 对方」归属角注在第 1 列、纯色小字（我的行 = i.own）',
      B.ops[0].dets.every(r => ['own', 'opp'].includes(tagOf(r))) &&
      tagOf(B.ops[0].dets[0]) === 'own',
      B.ops[0].dets.map(r => labelOf(r) + '⟨' + tagOf(r) + '⟩' + detNum(r)).join(' '));
    // ---- ④ 块内自洽 + 全表恒等式（2026-10-13 二次口径版）----
    check('每块明细之和 == 该块块头值',
      B.ops.every(o => Math.abs(sumDets(o.dets) - cellNum(o.head)) < 1e-6),
      B.ops.map(o => sumDets(o.dets) + '/' + cellNum(o.head)).join(' '));
    check('恒等式：共有段人均 ×3 + Σ块内合计 == 首行「本局结算总计」（33 + 2 = 35）',
      Math.abs(r2(cellNum(shsum) * 3 +
        B.ops.reduce((s, o) => s + cellNum(o.head), 0)) - cellNum(total)) < 1e-6,
      `${cellNum(shsum) * 3}+${B.ops.reduce((s, o) => s + cellNum(o.head), 0)} vs ${cellNum(total)}`);
  } catch (e) { bad++; console.log('[鸡牌三分类] THREW: ' + String(e.stack).split('\n').slice(0, 4).join('\n    ')); }

  console.log('\n[鸡牌三分类·责任鸡的方向与符号]');
  try {
    // 反向：视角玩家(0) 是**打出者** —— 他打出的一张横鸡被 P1 碰走，
    // 责任鸡溢价（横鸡 2 − 手中鸡 1 = 1，非金鸡）由他赔给 P1。
    // 他自己没有手中鸡/打出鸡 → 共享段为空；P1 的「手中鸡牌 3 分」落进 P1 那一块。
    const rows = renderView(pairState(
      [{ a: 1, b: 0, items: [{ label: '手中鸡', info: '共持有 3 张', value: 3, who: 1 },
                              { label: '责任鸡', info: '横鸡', value: 1, who: 1 }] }],
      [[0, 0, 0, 0], [4, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
      [-4, 4, 0, 0],
      { detail: [{ who: 'P1', seat: 1, label: '手中鸡', info: '共持有 3 张', value: 3 },
                 { who: 'P1', seat: 1, label: '责任鸡', info: '横鸡', value: 1 }],
        ji_hold_detail: { '1': [[9, 3]] },
        ji_claim_detail: [{ claimer: 1, discarder: 0, tile: 9, tag: '横鸡', premium: 1, gold: false }] }), 0);
    const B = blocksOf(rows);
    const shsum = rows.find(r => r.className.includes('shsum'));
    check('只出 1 个对手块（P1）', B.ops.length === 1, `n=${B.ops.length}`);
    check('块头值 = -4（块内合计 = 他的手中鸡牌 -3 + 责任鸡 -1，都在这块）',
      B.ops[0] && cellNum(B.ops[0].head) === -4,
      B.ops[0] ? B.ops[0].head.lastElementChild.textContent : '无');
    const h = B.ops[0].dets.find(r => labelOf(r) === '手中鸡牌');
    const z = B.ops[0].dets.find(r => labelOf(r) === '责任鸡');
    check('对手的「手中鸡牌」在他自己的专属块（.chk + 〈对方〉小字角注），值 -3',
      !!h && detNum(h) === -3 && h.classList.contains('chk') && tagOf(h) === 'opp',
      h ? labelOf(h) + '⟨' + tagOf(h) + '⟩' + detNum(h) : '无');
    check('我没有鸡分、对手鸡牌已归块 → 不出共有段（无 shsum 行）',
      B.shared.length === 0 && !shsum,
      `shared=${B.shared.map(labelOf).join('/')}`);
    check('「责任鸡」带 〈我〉角注（我是打出者），值 -1',
      !!z && detNum(z) === -1 && tagOf(z) === 'own' && tagOf(z) !== 'opp',
      z ? labelOf(z) + '⟨' + tagOf(z) + '⟩' + detNum(z) : '无');
    check('本局结算总计 = -4', cellNum(rows.find(r => r.className === 'grandrow')) === -4);
    check('块内明细之和 == 块头值', Math.abs(sumDets(B.ops[0].dets) - cellNum(B.ops[0].head)) < 1e-6);
    check('恒等式：共有段 0×3 + 块内 -4 == 总计 -4',
      r2(0 * 3 + sumDets(B.shared) + cellNum(B.ops[0].head)) === -4,
      `0+${sumDets(B.shared)}+${cellNum(B.ops[0].head)}`);
  } catch (e) { bad++; console.log('[责任鸡方向] THREW: ' + String(e.stack).split('\n').slice(0, 4).join('\n    ')); }

  console.log('\n[单人视角·共享项与各家独有项]');
  try {
    // 非鸡分类的分项照旧：杠分对三家完全相同 → 共有项；点炮只出现在点炮者那一对 → 该家块。
    // 2026-10-13 二次口径：块里只留「仅这两家之间」的分项，共有项一律在共有段列一次；
    // 恒等式 = 共有段人均 ×3 + Σ块内合计 == 总计。
    const rows = renderView(pairState(
      [{ a: 2, b: 0, items: [{ label: '点炮', info: '', value: 3, who: 2 },
                             { label: '杠分', info: '补杠 九条', value: 3, who: 2 }] },
       { a: 2, b: 1, items: [{ label: '杠分', info: '补杠 九条', value: 3, who: 2 }] },
       { a: 2, b: 3, items: [{ label: '杠分', info: '补杠 九条', value: 3, who: 2 }] }],
      [[0, 0, -6, 0], [0, 0, -3, 0], [0, 0, 0, 0], [0, 0, -3, 0]],
      [-6, -3, 12, -3]), 2);
    const html2 = w.document.getElementById('res-pairs').innerHTML;
    const rows2 = Array.from(w.document.querySelectorAll('#res-pairs .pairs-big tr'));
    const B = blocksOf(rows2);
    const total = rows2.find(r => r.className === 'grandrow');
    check('共享段只有「杠分」1 行（人均 +3）',
      B.shared.length === 1 && labelOf(B.shared[0]) === '杠分' && detNum(B.shared[0]) === 3,
      `n=${B.shared.length} ${B.shared.map(r => labelOf(r)).join('/')}`);
    check('点炮 → 视角方口径「捉炮」，只出现在点炮者那一块',
      cnt(html2, '捉炮') === 1 && cnt(html2, '放炮') === 0, `捉炮=${cnt(html2, '捉炮')}`);
    check('只有点炮者有两家专属分项 → 只出 1 块（另两家无块）',
      B.ops.length === 1 && B.ops[0].dets.length === 1, `n=${B.ops.length}`);
    check('块头值 = 3（块内合计 = 捉炮 3；杠分是共有项不再进块）',
      B.ops.map(o => cellNum(o.head)).join(',') === '3', B.ops.map(o => cellNum(o.head)).join(','));
    check('不再有「独有小计」行', !html2.includes('独有小计'));
    check('块内行第一列是归属角注（我 / 对方），不再是空列',
      B.ops.every(o => o.dets.every(r => ['我', '对方'].includes(r.children[0].textContent.trim()))),
      B.ops.map(o => o.dets.map(r => r.children[0].textContent.trim()).join('/')).join(' '));
    check('每块明细之和 == 块头值',
      B.ops.every(o => Math.abs(sumDets(o.dets) - cellNum(o.head)) < 1e-6),
      B.ops.map(o => sumDets(o.dets) + '/' + cellNum(o.head)).join(' '));
    check('恒等式：共有段人均 3×3 + 块内 3 == 首行总计 +12',
      r2(cellNum(rows2.find(r => r.className.includes('shsum'))) * 3 +
        B.ops.reduce((s, o) => s + cellNum(o.head), 0)) === cellNum(total) && cellNum(total) === 12,
      `${B.ops.reduce((s, o) => s + cellNum(o.head), 0)} / ${cellNum(total)}`);
  } catch (e) { bad++; console.log('[共享/独有项] THREW: ' + String(e.stack).split('\n').slice(0, 4).join('\n    ')); }

  console.log('\n[单人视角·块样式与满宽分割线]');
  try {
    check('CSS: 首行总计居中且跨满 5 列',
      /\.paircard \.pairs-big tr\.grandrow td \{[^}]*text-align: center/.test(html));
    check('CSS: 旧的 tr.total 末行样式已移除', !/tr\.total td \{/.test(html));
    check('CSS: 小计行 border-top 覆盖全部列',
      /\.paircard \.pairs-big tr\.sum td \{[^}]*border-top: 1px solid/.test(html) &&
      !/tr\.sum td:nth-child\(-n\+3\) \{[^}]*border-top/.test(html));
    check('CSS: 对手块头有上边框 + 左竖条',
      /\.paircard \.pairs-big tr\.opphead td \{[^}]*border-top: 2px solid/.test(html) &&
      /\.paircard \.pairs-big tr\.opphead td:first-child \{ padding-left: var\(--pava-x\); border-left: 3px solid/.test(html));
    check('CSS: 块头末列（独有合计）加粗',
      /\.paircard \.pairs-big tr\.opphead td:last-child \{ font-weight: 700; \}/.test(html));
    check('CSS: 共享区用冷色块与对手块区分',
      /\.paircard \.pairs-big tr\.det\.sh td \{ background: rgba\(138,196,255/.test(html));
    /* 鸡分项强调（2026-10-13 修订）：不再单独成段，改为「谁的分项归谁那块」，
       鸡分项不论落共享段还是各家块，统一加 .chk（青绿冷色块 + 左竖条） */
    check('CSS: 鸡分项（tr.det.chk）用青绿冷色块（底色 + 左竖条）',
      /\.paircard \.pairs-big tr\.det\.chk td \{ background: rgba\(127,216,196,.06\)/.test(html) &&
      /\.paircard \.pairs-big tr\.det\.chk td:first-child \{[^}]*border-left: 3px solid rgba\(127,216,196/.test(html));
    check('CSS: 独立鸡分段的死代码（tr.sum.chksum）已删除', !/tr\.sum\.chksum/.test(html));
    check('CSS: 「人均」角注走 .cap 圆角描边；归属角注 = 纯色小字（i.own / i.opp 两色）',
      /\.paircard \.pairs-big tr\.det \.cap \{[^}]*border: 1px solid rgba\(127,216,196/.test(html) &&
      /\.paircard \.pairs-big tr\.det td:first-child i\.own \{ color: #8ac4ff; \}/.test(html) &&
      /\.paircard \.pairs-big tr\.det td:first-child i\.opp \{ color: #e8b84b; \}/.test(html));
    check('CSS: 块间有间隔行', /\.paircard \.pairs-big tr\.gap td \{ height: 14px/.test(html));
    check('CSS: 首列（表头列）左侧与头像右侧平齐（统一缩进 --pava-x）',
      /--pava-w: \d+px; --pava-x: \d+px;/.test(html) &&
      /\.paircard \.pairs-big td:first-child, \.paircard \.pairs-big th:first-child \{[^}]*padding-left: var\(--pava-x\)/.test(html));
  } catch (e) { bad++; console.log('[块样式] THREW: ' + String(e.stack).split('\n').slice(0, 4).join('\n    ')); }

  console.log('\n[单人视角·首行总计（原末行「本局总计」）]');
  try {
    const rows = renderView(pairState([], [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
      [0, 0, 0, 0]), 2);
    const total = rows.find(r => r.className === 'grandrow');
    const html2 = w.document.getElementById('res-pairs').innerHTML;
    check('总计=0 时无「两讫」字样', !html2.includes('两讫'));
    check('总计值显示 0 且带「本局结算总计」说明',
      total && cellNum(total) === 0 && total.textContent.includes('本局结算总计'),
      total ? total.textContent.trim() : '无');
    check('总计行是卡片第一行', rows[0] === total);
  } catch (e) { bad++; console.log('[零总计] THREW: ' + String(e.stack).split('\n').slice(0, 4).join('\n    ')); }

  console.log('\n[鸡牌元素：红点 + 冲/横角标]');
  try {
    check('CSS: 手牌鸡牌右上角为红点（圆形 .jimark）',
      /\.tile-mine \.jimark \{[^}]*border-radius: 50%/.test(html) &&
      /\.ovhand \.ovtile \.jimark \{[^}]*border-radius: 50%/.test(html));
    check('JS: 鸡牌标记不再使用 🐔 字符', !html.includes('>🐔<'));
    check('JS: 角标生成（冲锋鸡→冲/横鸡→横/无标签→空）',
      w.eval('jiTagHTML("冲锋鸡")').includes('data-t="冲"') &&
      w.eval('jiTagHTML("横鸡")').includes('data-t="横"') &&
      w.eval('jiTagHTML(null)') === '' && w.eval('jiTagHTML(undefined)') === '');
    check('CSS: 角标配色 冲=红 / 横=蓝',
      html.includes('.jitag.chf') && html.includes('.jitag.hj'));
    check('JS: 牌河/副露不再引用冲.png/横.png 图',
      !html.includes('assets/tiles/冲.png') && !html.includes('assets/tiles/横.png'));
  } catch (e) { bad++; console.log('[鸡牌元素] THREW: ' + String(e.stack).split('\n').slice(0, 4).join('\n    ')); }

  console.log(`\nok=${ok} bad=${bad}`);
  if (errs.length) console.log('window errors:\n' + errs.slice(0, 5).join('\n'));
  process.exit(bad ? 1 : 0);
}, 1800);
