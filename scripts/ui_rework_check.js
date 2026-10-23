// 2026-10-11 界面改造批量校验（jsdom 加载真实页面）：
//   N1 左上角单列导航 + 结束本局（两段确认）
//   N2 记牌器（icon + 浮窗 27 格，0 张置灰）
//   N3 听牌提示条（弃牌区下方 + 数据源 my_ting + 打出后消失）
//   N4 流水（行编号 局-序 + 筛选器 + 鸡标记/横鸡轮行样式）
//   N5 设置项说明改问号 icon（? 弹泡，不误触开关）
//   N6 数据统计（历史累计 10 项 + 本会话四家 7 列，名字列左对齐）
//   N7 结算（鸡牌三分类 手中鸡牌/打出鸡牌/责任鸡 → 牌图×张数；视角玩家不再显示「你」）
const fs = require('fs');
const { JSDOM } = require('jsdom');

const ROOT = 'C:/Users/LIGF/WorkBuddy/2026-09-30-12-49-43/zhuoji-mahjong';
const html = fs.readFileSync(ROOT + '/web/static/dushan.html', 'utf8');
const errs = [];
const meta = JSON.parse(fs.readFileSync(ROOT + '/reports/ui_fixtures_meta.json', 'utf8'));
const mid0 = JSON.parse(fs.readFileSync(ROOT + '/reports/ui_fixtures_mid.json', 'utf8'))[0].state;

const STATS = { ok: true, all_time: {
  games: 42, zimo: 9, dianhu: 7, chongfeng_games: 11, ji_discard_games: 30,
  baoji_games: 5, tenpai_bigfan_games: 8, win_bigfan_games: 13,
  huang_games: 12, huang_noten_games: 4 } };

const dom = new JSDOM(html, {
  runScripts: 'dangerously',
  pretendToBeVisual: true,
  url: 'http://127.0.0.1:8770/static/dushan.html',
  beforeParse(w) {
    if (!w.matchMedia) w.matchMedia = q => ({ matches: false, media: String(q),
      onchange: null, addEventListener() {}, removeEventListener() {},
      addListener() {}, removeListener() {}, dispatchEvent: () => false });
    w.fetch = (url) => {
      const u = String(url);
      if (u.includes('/api/meta')) {
        return Promise.resolve({ ok: true, json: () => Promise.resolve(meta) });
      }
      if (u.includes('/api/stats')) {
        return Promise.resolve({ ok: true, json: () => Promise.resolve(STATS) });
      }
      if (u.includes('/api/end_hand')) {
        w.__endCalls = (w.__endCalls || 0) + 1;
        return Promise.resolve({ ok: true, json: () => Promise.resolve({ ok: true }) });
      }
      return new Promise(() => {});
    };
    w.addEventListener('error', e => errs.push('window.error: ' + ((e.error && e.error.stack) || e.message)));
  },
});
const w = dom.window;
const results = [];
const check = (name, ok, extra = '') => {
  results.push(ok);
  console.log(`${name}: ${ok ? 'OK' : 'FAIL'}${extra ? ' | ' + extra : ''}`);
};

setTimeout(async () => {
  const el = id => w.document.getElementById(id);
  const fire = (elm, type, opts) =>
    elm.dispatchEvent(new w.MouseEvent(type, Object.assign({ bubbles: true, cancelable: true }, opts)));

  // ---- N1 左上角导航（2026-10-12 #86：结束本局已迁出导航；#83：第三列鸡牌图片）----
  const navBtns = [...el('nav').querySelectorAll('button')].map(b => b.id);
  check('N1 导航按钮顺序（无结束本局）', navBtns.join(',') ===
    'btn-stats,btn-history,btn-settings,btn-peek,btn-sound', navBtns.join(','));
  check('N1b 左列 = 统计/记录/局数、图标列 = 设置/透视/声音、第三列 = 鸡牌图片',
    (() => {
      const cols = [...el('nav').children];
      if (cols.length !== 3) return false;
      const L = [...cols[0].children].map(n => n.id);
      const R = [...cols[1].children].map(n => n.id);
      return L.join(',') === 'btn-stats,btn-history,round-chip'
        && R.join(',') === 'btn-settings,btn-peek,btn-sound'
        && cols[2].id === 'kaiju-info';
    })(), [...el('nav').querySelectorAll('button,div.chip')].map(n => n.id || n.className).join(','));
  check('N1c 无 ⋯ / 无流水图标按钮', !el('btn-navmore') && !el('btn-log'), '');
  // 结束本局（#86 迁入设置 → 界面显示）：点击弹窗二次确认
  // 2026-10-12：确认窗与设置面板互斥 —— 弹出时收起设置面板；取消（按钮/遮罩/Esc）放回来；
  // 确认则直接结束、不再显示设置面板。
  w.eval('SID = "rework-e2e"');
  el('btn-settings').click();
  check('N1c2 入口在设置面板里（先打开设置面板）', el('set-mask').classList.contains('on'));
  el('set-endhand').click();
  check('N1d 点结束本局 → 弹确认窗、设置面板收起、不发请求',
    (w.__endCalls || 0) === 0 && el('endhand-mask').classList.contains('on')
    && !el('set-mask').classList.contains('on'),
    `calls=${w.__endCalls || 0} mask=${el('endhand-mask').classList.contains('on')} ` +
    `set=${el('set-mask').classList.contains('on')}`);
  check('N1d2 「继续打这局」→ 弹窗关、设置面板放回来、仍不发请求', (() => {
    el('endhand-cancel').click();
    return !el('endhand-mask').classList.contains('on') && (w.__endCalls || 0) === 0
      && el('set-mask').classList.contains('on');
  })(), `calls=${w.__endCalls || 0} set=${el('set-mask').classList.contains('on')}`);
  check('N1d3 点确认窗遮罩外关闭也算取消（设置面板同样放回来）', (() => {
    el('set-endhand').click();
    el('endhand-mask').dispatchEvent(new w.MouseEvent('click', { bubbles: true }));
    return !el('endhand-mask').classList.contains('on') && el('set-mask').classList.contains('on');
  })(), `set=${el('set-mask').classList.contains('on')}`);
  check('N1d4 Esc 关闭也算取消（设置面板同样放回来）', (() => {
    el('set-endhand').click();
    w.document.dispatchEvent(new w.KeyboardEvent('keydown', { key: 'Escape', bubbles: true }));
    return !el('endhand-mask').classList.contains('on') && el('set-mask').classList.contains('on');
  })(), `set=${el('set-mask').classList.contains('on')}`);
  el('set-endhand').click();
  el('endhand-ok').click();
  await new Promise(r => setTimeout(r, 60));
  check('N1e 确认后 → 调 /api/end_hand 一次，且不再显示设置面板',
    w.__endCalls === 1 && !el('set-mask').classList.contains('on'),
    `calls=${w.__endCalls || 0} set=${el('set-mask').classList.contains('on')}`);

  // ---- N2 记牌器 ----
  // 2026-10-12 晚用户口径：记牌器在「设置 → 界面显示」里有开关，**默认关闭** →
  // 桌面不显示记牌器按钮。所以这里先断言默认态，再显式打开开关跑后续交互。
  check('N2 记牌器默认关闭（SET.counterOn=false，桌面按钮真的不显示）',
    w.eval('SET.counterOn') === false && el('counter-btn').classList.contains('hidden') &&
    w.getComputedStyle(el('counter-btn')).display === 'none',
    `display=${w.getComputedStyle(el('counter-btn')).display}`);
  w.eval('SET.counterOn = true; applyCounterShow()');
  check('N2a 打开开关后桌面按钮出现（去 hidden，display 恢复 flex）',
    !el('counter-btn').classList.contains('hidden') &&
    w.getComputedStyle(el('counter-btn')).display === 'flex',
    `display=${w.getComputedStyle(el('counter-btn')).display}`);
  const st = JSON.parse(JSON.stringify(mid0));
  st.turn = -1;
  st.counter = new Array(27).fill(4);
  st.counter[9] = 0;           // 1条（幺鸡）全见 → 置灰
  st.counter[22] = 1;
  w.__st = st;
  w.eval('LAST_STATE = window.__st; render(window.__st)');
  check('N2 记牌器浮窗默认收起', el('counter-pop').classList.contains('hidden'), '');
  el('counter-btn').click();
  check('N2b 点开浮窗', !el('counter-pop').classList.contains('hidden')
    && el('counter-btn').classList.contains('toggle-on'), '');
  const cells = [...el('counter-grid').querySelectorAll('.cc')];
  check('N2c 27 格（图 + 数字）', cells.length === 27
    && cells.every(c => c.querySelector('img') && /^\d+$/.test(c.querySelector('i').textContent)),
    `n=${cells.length}`);
  check('N2d 0 张置灰、>0 不置灰', cells[9].classList.contains('zero')
    && !cells[10].classList.contains('zero'), '');
  el('counter-btn').click();
  check('N2e 再点收起并记忆', el('counter-pop').classList.contains('hidden')
    && w.localStorage.getItem('dushan_counter') === '0', '');
  // 关掉开关：按钮重新藏起，且浮窗强制收起
  el('counter-btn').click();                       // 先展开
  w.eval('SET.counterOn = false; applyCounterShow()');
  check('N2f 关掉开关 → 按钮真的不显示、浮窗强制收起',
    el('counter-btn').classList.contains('hidden') &&
    w.getComputedStyle(el('counter-btn')).display === 'none' &&
    el('counter-pop').classList.contains('hidden') &&
    w.eval('COUNTER_OPEN') === false, '');
  w.eval('SET.counterOn = true; applyCounterShow()');   // 复原，不污染后续用例

  // ---- N3 听牌提示条 ----
  check('N3a 提示条贴弃牌区下方（CSS 锚点 50%+184px，预留两行弃牌）',
    /#tingbar \{[^}]*top: min\(calc\(50% \+ 184px\)/.test(html), '');
  const stT = JSON.parse(JSON.stringify(mid0));
  stT.turn = stT.my_seat;
  stT.pending = { mode: 'discard', buttons: [], discardables: stT.hand.slice(), last: null, can_hint: true };
  stT.my_ting = [18, 22];
  w.__st = stT;
  w.eval('LAST_STATE = window.__st; COACH_HINTS = []; render(window.__st)');
  const tb = el('tingbar');
  check('N3b 决策点常显（数据源 my_ting，不依赖大师推荐）', tb.classList.contains('on')
    && (tb.innerHTML.match(/<img/g) || []).length === 2, tb.textContent.trim());
  w.eval('window.__st = Object.assign({}, window.__st, { pending: null, my_ting: [18, 22] })');
  w.eval('LAST_STATE = window.__st; render(window.__st)');
  check('N3c 非决策点隐藏', !tb.classList.contains('on'), '');

  // ---- N4 流水 ----
  const evs = [
    { t: 1, kind: 'draw', rel: 0, hand: 3, seq: 1, text: '云天明 摸进 1万' },
    { t: 1, kind: 'discard', rel: 0, hand: 3, seq: 2, tile: 0, ctag: '幺鸡', text: '云天明 打出 1条（普通鸡）' },
    { t: 1, kind: 'hengji', rel: 1, hand: 3, seq: 3, text: '大乔(S11) 开出横鸡轮 · 1条', ctag: '横鸡轮开始' },
    { t: 1, kind: 'peng', rel: 2, hand: 3, seq: 4, text: '妲己(B3) 碰 5万' },
  ];
  w.eval(`window.__evs = ${JSON.stringify(evs)}; LAST_STATE = { events: window.__evs };` +
    `el("log-body")._h = ""; renderLog(window.__evs)`);
  const rows = [...el('log-body').querySelectorAll('.le')];
  check('N4a 行编号 = 局-序', rows.every(r => /^\d+-\d+$/.test(r.querySelector('.no').textContent)),
    rows.map(r => r.querySelector('.no').textContent).join(' '));
  check('N4b 横鸡轮行带 k-hengji 高亮', !!el('log-body').querySelector('.k-hengji'), '');
  const fbtns = [...el('log-filter').querySelectorAll('button')];
  check('N4c 筛选器含 摸牌/出牌/碰杠/鸡牌/横鸡轮',
    ['全部', '摸牌', '出牌', '碰杠', '报叫/听/胡', '鸡牌', '横鸡轮', '结算']
      .every(lab => fbtns.some(b => b.textContent === lab)), fbtns.map(b => b.textContent).join(','));
  const fbtn = lab => [...el('log-filter').querySelectorAll('button')]
    .find(b => b.textContent === lab);   // 点击后 innerHTML 会重建，必须每次重查
  fbtn('鸡牌').click();
  const jiRows = [...el('log-body').querySelectorAll('.le')];
  check('N4d 筛选「鸡牌」只剩打出鸡的行', jiRows.length === 1
    && jiRows[0].textContent.includes('1条'), `n=${jiRows.length}`);
  fbtn('横鸡轮').click();
  check('N4e 筛选「横鸡轮」只剩横鸡轮行',
    el('log-body').querySelectorAll('.le').length === 1
    && !!el('log-body').querySelector('.k-hengji'), '');
  fbtn('全部').click();
  check('N4f 筛选选择持久化', w.localStorage.getItem('dushan_logfilter') === 'all', '');
  w.eval('LAST_STATE = window.__st');   // 还原为完整局面（后面打开设置面板要用）

  // ---- N5 设置说明问号 ----
  el('btn-settings').click();
  const qs = [...el('set-panes').querySelectorAll('.q')];
  check('N5a 说明全部改为问号 icon（面板内不再有 .tip 段落）', qs.length >= 18
    && el('set-panes').querySelectorAll('.tip').length === 0, `n=${qs.length}`);
  const q0 = qs[0];
  const txt = q0.getAttribute('data-q');
  const swBefore = el('set-tipon').checked;
  q0.click();
  check('N5b 点 ? 弹出说明气泡', el('qpop').classList.contains('on')
    && el('qpop').textContent === txt, el('qpop').textContent.slice(0, 24));
  check('N5c 点 ? 不误触所在开关', el('set-tipon').checked === swBefore, '');
  q0.click();
  check('N5d 再点 ? 收起', !el('qpop').classList.contains('on'), '');
  w.document.body.click();
  check('N5e 点面板外也收起', !el('qpop').classList.contains('on'), '');
  el('btn-set-save').click(); el('btn-set-close').click();

  // ---- N6 数据统计 ----
  w.__st = JSON.parse(JSON.stringify(mid0));
  w.eval('LAST_STATE = window.__st');
  await el('btn-stats').onclick ? null : null;
  el('btn-stats').click();
  await new Promise(r => setTimeout(r, 60));
  const sb = el('stats-body');
  const statLbl = [...sb.querySelectorAll('.stat .lbl')].map(x => x.textContent);
  check('N6a 历史累计 12 项（前 10 项 + 本人胡牌/本人累计）',
    statLbl.join(',') === ['对局', '自摸', '送胡', '冲锋鸡', '出鸡', '包鸡',
      '非平胡·听牌', '非平胡·胡牌', '黄庄·局占率', '黄庄·未听牌',
      '本人胡牌', '本人累计'].join(','),
    statLbl.join(','));
  const nums = [...sb.querySelectorAll('.stat .num')].map(x => x.textContent);
  // 前 10 项：对局=绝对值，其余=百分比；「本人胡牌」是百分比（分母为 0 时显示 —），
  // 「本人累计」是带符号的分数。
  check('N6b 对局=数、其余=百分比、本人累计=带符号分数',
    nums[0] === '42' && nums.slice(1, 10).every(x => x.endsWith('%'))
      && (nums[10] === '—' || /%$/.test(nums[10])) && /^[+\-]\d/.test(nums[11]),
    nums.join(' '));
  const ths = [...sb.querySelectorAll('table.res th')].map(x => x.textContent);
  check('N6c 本会话四家 7 列（送胡/横鸡 入列）',
    ths.join(',') === '玩家,胡牌,自摸,送胡,冲锋鸡,横鸡,包鸡,得分', ths.join(','));
  check('N6d 名字列左对齐（CSS）',
    /#stats-body table\.res th:first-child[^{]*\{ text-align: left/.test(html), '');
  el('stats-mask').classList.remove('on');

  // ---- N7 结算：鸡牌三分类（手中鸡牌 / 打出鸡牌 / 责任鸡 → 牌图×张数）；视角玩家名 ----
  // 2026-10-13 用户口径：鸡分只保留三类，且每类都要显示「哪一张鸡牌有几张」。
  // 本夹具：视角玩家(0) 手中 9 万×2 + 10 万×1（手中鸡），结算翻鸡 10 万×2 → 归入「手中鸡牌」。
  const stR = JSON.parse(JSON.stringify(mid0));
  stR.over = true;
  stR.my_seat = 0;
  stR.players.forEach((p, i) => { p.seat = i; p.name = p.name || `P${i}`; });
  stR.result = Object.assign({}, stR.result, {
    type: 'win', how: '自摸', is_tsumo: true, rob_kong: false, void_name: null,
    winners: [0], winner: 0, loser: null,
    fanji_tiles: [10], deltas: [21, -7, -7, -7],
    detail: [
      { who: '全场', seat: -1, label: '翻鸡', info: '10 万', value: 0 },
      { who: '我', seat: 0, label: '手中鸡', info: '3 只', value: 3 },
      { who: '我', seat: 0, label: '翻鸡', info: '共持有 2 张', value: 2 },
    ],
    ji_hold_detail: { '0': [[9, 2], [10, 1]], '1': [] },
    fanji_detail: { '0': [[10, 2]], '1': [] },
    ji_discard_detail: {}, ji_capture_detail: {}, ji_claim_detail: [],
    gold_tiles: [], gold_mult: 2,
    pair: [[0, 7, 7, 7], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
    pair_detail: [1, 2, 3].map(b => ({ a: 0, b, items: [
      { label: '自摸', info: '', value: 2, who: 0 },
      { label: '手中鸡', info: '3 只', value: 3, who: 0 },
      { label: '翻鸡', info: '共持有 2 张', value: 2, who: 0 }] })),
  });
  w.__st = stR;
  w.eval('LAST_STATE = window.__st; RES_VIEW = 0; RES_DISMISSED = false; render(window.__st)');
  const prHtml = el('res-pairs').innerHTML;
  const rowOf = label => {
    const tr = [...w.document.querySelectorAll('#res-pairs .pairs-big tr.det')]
      .find(r => r.children[1].textContent.replace(/人均|我|对方/g, '').trim() === label);
    return tr ? tr.innerHTML : '';
  };
  const holdRow = rowOf('手中鸡牌');
  const t9 = w.eval('tileImg(9)'), t10 = w.eval('tileImg(10)');
  check('N7a 手中鸡牌 → 鸡牌图×张数（9万×2 与 10万×1 各一格）',
    holdRow.includes(t9) && holdRow.includes(t10) && holdRow.includes('×2') && holdRow.includes('×1')
    && (holdRow.match(/class="jcnt"/g) || []).length === 3,
    (holdRow.match(/class="jcnt"/g) || []).length + ' 格');
  check('N7b 翻鸡并入「手中鸡牌」（10万×2 单列一格）',
    holdRow.includes('>×2<') && (prHtml.match(/×2/g) || []).length >= 1,
    (prHtml.match(/class="jcnt"/g) || []).length + ' 格');
  check('N7c 共有块头 = 「共有分项」+「对所有玩家生效（…×3 计入总计）」',
    prHtml.includes('共有分项') && prHtml.includes('对所有玩家生效（我方行按每家一份 ×3 计入总计）')
    && !prHtml.includes('我方分项（每家各一份，已计入下方各块）'), '');
  check('N7d 视角玩家名不再硬编码「你」', !prHtml.includes('>你<'), '');

  const bad = results.filter(x => !x).length;
  console.log(`\nwindow errors: ${errs.length}${errs.length ? '\n' + errs.slice(0, 3).join('\n') : ''}`);
  console.log(`合计 ${results.length} 项，失败 ${bad}`);
  process.exit(bad === 0 && errs.length === 0 ? 0 : 1);
}, 1800);
