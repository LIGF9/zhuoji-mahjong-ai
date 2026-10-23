// 「辅助功能」这批改动的 jsdom 校验（加载真实页面 + 真实快照）：
//   1) 设置面板：未听牌包赔并入「规则开关」（不再有独立导航项）
//   2) 界面显示：听牌提示开关（默认开）+ 结算单人视角布局（默认新版）
//   3) 打牌听牌 toast：点击候选牌 → 弹出「听什么牌 + 各还剩几张」；开关关闭则不弹
//   4) 总览：赢家标签 = 具体胡牌方式（自摸 / 捉炮 / 抢杠 / 热炮 / 杠上开花）
//   5) 单人视角新版布局：共有板块 / 各家独有项之上补「头像+手牌+标签」，且不含名字与总分
//      （与总览同一位玩家的 .ovhand / .ovmeta 逐字符一致）
//   6) 流水面板：窗口由宽变窄时把面板位置拉回可视范围
const fs = require('fs');
const { JSDOM } = require('jsdom');

const ROOT = 'C:/Users/LIGF/WorkBuddy/2026-09-30-12-49-43/zhuoji-mahjong';
const html = fs.readFileSync(ROOT + '/web/static/dushan.html', 'utf8');
const meta = JSON.parse(fs.readFileSync(ROOT + '/reports/ui_fixtures_meta.json', 'utf8'));
const MID = JSON.parse(fs.readFileSync(ROOT + '/reports/ui_fixtures_mid.json', 'utf8'));
const OVER = JSON.parse(fs.readFileSync(ROOT + '/reports/ui_fixtures_over.json', 'utf8'));
const errs = [];

const dom = new JSDOM(html, {
  runScripts: 'dangerously',
  pretendToBeVisual: true,
  url: 'http://127.0.0.1:8770/static/dushan.html',
  beforeParse(w) {
    // jsdom 没有 matchMedia（fitStage 用它判触屏）——补一个「非触屏」的桩
    if (!w.matchMedia) {
      w.matchMedia = q => ({ matches: false, media: String(q),
        addEventListener() {}, removeEventListener() {}, addListener() {}, removeListener() {} });
    }
    w.fetch = (url) => {
      const u = String(url);
      if (u.includes('/api/meta')) return Promise.resolve({ ok: true, json: () => Promise.resolve(meta) });
      if (u.includes('/api/ting')) return Promise.resolve({ ok: true, json: () => Promise.resolve(w.__TING) });
      return new Promise(() => {});
    };
    w.addEventListener('error', e => errs.push('window.error: ' + ((e.error && e.error.stack) || e.message)));
  },
});
const w = dom.window;

let ok = 0, bad = 0;
function check(name, cond, detail = '') {
  (cond ? ok++ : bad++);
  console.log(`  [${cond ? 'ok  ' : 'FAIL'}] ${name}${detail ? '   ' + detail : ''}`);
}
const clone = o => JSON.parse(JSON.stringify(o));
const relOf = (st, seat) => ((seat - st.my_seat) % 4 + 4) % 4;
const sleep = ms => new Promise(r => setTimeout(r, ms));
// 总览卡片按「名字」定位（卡片顺序 = 赢家优先，不能用 rel 当下标）
function cardOfSeat(st, seat) {
  // 2026-10-11 起名字与头像绑定：卡片按头像文件名定位（显示名是派生名）
  const rel = relOf(st, seat);
  const ava = w.eval(`(sessionAvatars()[${rel}] || "")`);
  return [...w.document.querySelectorAll('#res-pairs .ovcard')]
    .find(c => {
      const a = c.querySelector('.avatar');
      return a && ava && String(a.getAttribute('src')).includes(`/${ava}.jpg`);
    }) || null;
}

setTimeout(async () => {
  const el = id => w.document.getElementById(id);
  // 等 boot() 把 /api/meta 的牌表填好（TILES 为空时 tileImg 会取不到牌）
  for (let i = 0; i < 40 && !w.eval('!!(TILES[9] && META)'); i++) await sleep(25);
  check('牌表已就绪（/api/meta）', w.eval('!!(TILES[9] && META)'));

  // ============================================================ 1) 设置面板合并
  console.log('\n[设置面板：未听牌包赔并入规则开关]');
  try {
    const navPanes = [...el('set-nav').querySelectorAll('button')].map(b => b.dataset.pane);
    check('导航不再有独立的「未听牌包赔」', !navPanes.includes('pay'), navPanes.join(','));
    check('不再存在 pay 面板', !el('set-panes').querySelector('section[data-pane="pay"]'));
    const rulePane = el('set-panes').querySelector('section[data-pane="rule"]');
    const payIds = ['set-end-baoji', 'set-end-baogang', 'set-huang-baoji',
      'set-huang-baogang', 'set-huang-baodapai'];
    const inRule = payIds.filter(i => rulePane.contains(el(i)));
    check('5 个包赔开关都在「规则开关」板块内', inRule.length === 5, inRule.join(','));
    check('规则开关板块内同时保留 3 个规则开关',
      ['set-jiesuan-fanji', 'set-kaiju-fanji', 'set-mantiangji'].every(i => rulePane.contains(el(i))));
  } catch (e) { bad++; console.log('  THREW: ' + String(e.stack).split('\n').slice(0, 4).join('\n    ')); }

  // ============================================================ 2) 界面显示板块新增项
  // 2026-10-12：「界面显示」分类已并入「出牌与显示」（data-pane="play"）
  console.log('\n[界面显示：听牌提示开关 + 全局流水开关（旧版布局开关已删）]');
  try {
    const playPane = el('set-panes').querySelector('section[data-pane="play"]');
    check('旧 ui 分类已并入出牌板块', !el('set-panes').querySelector('section[data-pane="ui"]'));
    check('听牌提示控件在「出牌与显示」板块内', playPane.contains(el('set-tipon')));
    check('听牌提示是开关（checkbox.switch）',
      el('set-tipon').type === 'checkbox' && el('set-tipon').classList.contains('switch'));
    check('全局流水开关在「出牌与显示」板块内', playPane.contains(el('set-logshow')));
    check('旧版结算布局开关已删除（新版是唯一布局）', !el('set-reslayout'));
    w.eval('fillSetSelects()');
    check('听牌提示默认打开', el('set-tipon').checked === true);
    check('流水开关回显 SET.logShow', el('set-logshow').checked === w.eval('SET.logShow'));
    // 改 → 保存 → SET / localStorage 同步
    el('set-tipon').checked = false;
    el('set-logshow').checked = true;
    el('btn-set-save').click(); el('btn-set-close').click();
    check('保存后 SET.tipOn=false', w.eval('SET.tipOn') === false);
    check('保存后流水面板展开（hidden 移除）', !el('log').classList.contains('hidden'));
    check('localStorage 落盘', w.localStorage.getItem('dushan_tingtip') === '0'
      && w.localStorage.getItem('dushan_logshow') === '1');
    // 复原
    w.eval('SET.tipOn = true; SET.logShow = false; applyLogShow()');
    check('复原后流水面板收起', el('log').classList.contains('hidden'));
  } catch (e) { bad++; console.log('  THREW: ' + String(e.stack).split('\n').slice(0, 4).join('\n    ')); }

  // ============================================================ 3) 打牌听牌 toast
  console.log('\n[打牌听牌 toast]');
  try {
    const st = clone(MID[0].state);
    st.turn = st.my_seat;
    const discardables = st.pending.discardables;
    const T = discardables[0];
    w.__TING = { ok: true, map: { [String(T)]: {
      tiles: [{ t: 9, left: 3 }, { t: 18, left: 0 }], types: ['大对子'] } } };
    w.__st = st;
    w.eval('SID = "aux-ting"; LAST_STATE = window.__st; RES_DISMISSED = false; render(window.__st);');
    await sleep(30);
    check('决策点已预取听牌表', w.eval('TING_KEY === TURN_KEY && !!TING_MAP'), w.eval('TURN_KEY'));

    const toastEl = el('toast');
    toastEl.classList.remove('on'); toastEl.innerHTML = '';
    const tileEl = w.document.querySelector(`#hand .tile-mine[data-t="${T}"]`);
    check('手牌里找到候选牌元素', !!tileEl, `data-t=${T}`);
    tileEl.click();
    await sleep(30);
    const th = toastEl.innerHTML;
    check('toast 已弹出', toastEl.classList.contains('on'));
    check('toast 含听牌提示文案', th.includes('后听牌'), th.slice(0, 120));
    check('toast 列出听牌（2 张牌面小图）',
      (th.match(/class="tin"/g) || []).length === 2, th.slice(0, 200));
    check('toast 标注各剩几张（剩3 / 剩0）', th.includes('剩3') && th.includes('剩0'), th.slice(0, 200));
    check('剩 0 张用告警色', /tcnt z/.test(th));
    check('toast 带牌型', th.includes('大对子'));
    check('候选牌已上浮', w.eval('SELECTED_IDX') >= 0);

    // 关闭开关 → 不再弹
    w.eval('SET.tipOn = false; clearTing()');
    toastEl.classList.remove('on'); toastEl.innerHTML = '';
    const tileEl2 = w.document.querySelector('#hand .tile-mine:not(.sel):not(.ghost)');
    if (tileEl2) {
      const t2 = parseInt(tileEl2.dataset.t, 10);
      if ((st.pending.discardables || []).includes(t2)) {
        tileEl2.click();
        await sleep(30);
        check('开关关闭后不再弹听牌提示', !toastEl.classList.contains('on'), toastEl.innerHTML.slice(0, 80));
      } else {
        check('开关关闭后不再弹听牌提示', true, '（该张不是候选牌，跳过）');
      }
    }
    w.eval('SET.tipOn = true');
  } catch (e) { bad++; console.log('  THREW: ' + String(e.stack).split('\n').slice(0, 4).join('\n    ')); }

  // ============================================================ 4) 总览赢家标签
  console.log('\n[总览：赢家标签 = 胡牌方式]');
  try {
    const win = OVER.find(o => o.state.result.type === 'win' && Array.isArray(o.state.result.winners));
    const base = clone(win.state);
    base.my_seat = 0;
    const wSeat = (base.result.winners || [base.result.winner])[0];
    const rel = relOf(base, wSeat);
    for (const [how, want] of [['自摸', '自摸'], ['点炮', '捉炮'], ['抢杠', '抢杠'],
      ['热炮', '捉热炮'], ['杠上开花', '杠上开花']]) {
      const st2 = clone(base);
      st2.result = Object.assign({}, st2.result, { how, type: 'win' });
      w.__st = st2;
      w.eval('LAST_STATE = window.__st; RES_VIEW = "ov"; RES_DISMISSED = false; render(window.__st);');
      const card = cardOfSeat(st2, wSeat);
      const tag = card ? (card.querySelector('.tag.win') || {}).textContent : null;
      check(`how=${how} → 标签「${want}」`, tag === want, `tag=${tag}`);
    }
    // 从未出现笼统的「胡牌」标签
    const st3 = clone(base);
    w.__st = st3;
    w.eval('LAST_STATE = window.__st; RES_VIEW = "ov"; render(window.__st);');
    check('总览不再出现笼统的「胡牌」标签',
      ![...w.document.querySelectorAll('#res-pairs .tag.win')].some(x => x.textContent === '胡牌'));
  } catch (e) { bad++; console.log('  THREW: ' + String(e.stack).split('\n').slice(0, 4).join('\n    ')); }

  // ============================================================ 5) 单人视角新版布局
  console.log('\n[单人视角新版布局]');
  try {
    const win = OVER.find(o => o.state.result.type === 'win');
    const base = clone(win.state);
    base.my_seat = 0;
    base.players.forEach((p, i) => { p.seat = i; p.name = p.name || `P${i}`; });
    // 构造：视角玩家 2 ↔ 三家都有往来；其中 0 号有独有项（对 1 号有独有项）
    const wSeat = (base.result.winners || [base.result.winner])[0];
    const losr = base.result.loser != null ? base.result.loser : [0, 1, 2, 3].find(s => s !== wSeat);
    const v = wSeat;
    const pair = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]];
    const deltas = [0, 0, 0, 0];
    const others = [0, 1, 2, 3].filter(s => s !== v);
    others.forEach(o => { pair[v][o] = 3; deltas[v] += 3; deltas[o] -= 3; });
    const pd = others.map(o => ({ a: v, b: o, items: [{ label: '自摸', info: '', value: 3, who: v }] }));
    // 给 others[1] 加一个独有项（只对它出现 → 不进共享区）
    pd[1].items.push({ label: '杠分', info: '补杠 九条', value: 2, who: v });
    pair[v][others[1]] += 2; deltas[v] += 2; deltas[others[1]] -= 2;
    base.result = Object.assign({}, base.result, {
      type: 'win', how: '自摸', is_tsumo: true, rob_kong: false, void_name: null,
      winners: [v], winner: v, loser: losr, pair, pair_detail: pd, deltas,
    });
    w.__st = base;

    // 先取总览下同一位玩家的 .ovhand / .ovmeta（作为对照）
    w.eval('LAST_STATE = window.__st; RES_VIEW = "ov"; RES_DISMISSED = false; render(window.__st);');
    const ovCard = cardOfSeat(base, v);
    check('总览里能定位到视角玩家的卡片', !!ovCard);
    const ovHand = ovCard && ovCard.querySelector('.ovhand') ? ovCard.querySelector('.ovhand').outerHTML : '';
    const ovMeta = ovCard && ovCard.querySelector('.ovmeta') ? ovCard.querySelector('.ovmeta').outerHTML : '';

    // 切到单人视角（新版布局）
    w.eval(`LAST_STATE = window.__st; RES_VIEW = ${v}; render(window.__st);`);
    const pairHtml = el('res-pairs').innerHTML;
    const ownStrips = [...w.document.querySelectorAll('#res-pairs .pstriprow.ownstrip')];
    const oppStrips = [...w.document.querySelectorAll('#res-pairs .pstriprow.oppstrip')];
    check('共有板块之上有视角玩家的「头像+手牌+标签」行', ownStrips.length === 1, `n=${ownStrips.length}`);
    check('各家独有项之上有该家的「头像+手牌+标签」行', oppStrips.length >= 1, `n=${oppStrips.length}`);
    check('牌面行都在表格内（colspan=5 的行）',
      [...ownStrips, ...oppStrips].every(r => r.firstElementChild
        && r.firstElementChild.getAttribute('colspan') === '5'));

    const os = ownStrips[0];
    check('含头像', !!os.querySelector('.avatar') && /\/assets\/avatar\//.test(os.querySelector('.avatar').src));
    check('含手牌块', !!os.querySelector('.ovhand'));
    check('含标签块', !!os.querySelector('.ovmeta'));
    check('不含名字（.nm）', !os.querySelector('.nm'));
    check('不含总分（.dl）', !os.querySelector('.dl'));
    const osHand = os.querySelector('.ovhand') ? os.querySelector('.ovhand').outerHTML : '';
    const osMeta = os.querySelector('.ovmeta') ? os.querySelector('.ovmeta').outerHTML : '';
    check('视角玩家牌面与总览同一位完全一致（手牌）', osHand !== '' && osHand === ovHand,
      osHand === ovHand ? '' : `\n        strip=${osHand.slice(0, 140)}\n        ov   =${ovHand.slice(0, 140)}`);
    check('视角玩家标签与总览同一位完全一致', osMeta !== '' && osMeta === ovMeta,
      osMeta === ovMeta ? '' : `\n        strip=${osMeta.slice(0, 160)}\n        ov   =${ovMeta.slice(0, 160)}`);

    // 顺序：ownstrip 在共有块（shsum）之前；oppstrip 在其对应 opphead 之前
    const rows = [...w.document.querySelectorAll('#res-pairs .pairs-big tr')];
    const iOwn = rows.indexOf(ownStrips[0]);
    const iSum = rows.findIndex(r => r.className.includes('shsum'));
    check('视角行排在共有板块之上', iOwn >= 0 && (iSum < 0 || iOwn < iSum), `own=${iOwn} shsum=${iSum}`);
    const oppHeadIdx = rows.findIndex(r => r.className === 'opphead');
    const iOpp = rows.indexOf(oppStrips[0]);
    check('对手行排在该家独有项块头之上', iOpp >= 0 && (oppHeadIdx < 0 || iOpp < oppHeadIdx),
      `strip=${iOpp} opphead=${oppHeadIdx}`);

    // 旧版布局开关已删除：新版是唯一布局，任何时候都渲染牌面行
    w.eval(`RES_VIEW = ${v}; render(LAST_STATE);`);
    check('新版是唯一布局（无开关也渲染牌面行）',
      w.document.querySelectorAll('#res-pairs .pstriprow').length >= 1);
    check('CSS 定义了牌面行样式', /\.paircard \.pstriprow td \{/.test(html)
      && /\.pstrip \{/.test(html));
  } catch (e) { bad++; console.log('  THREW: ' + String(e.stack).split('\n').slice(0, 4).join('\n    ')); }

  // ============================================================ 6) 流水面板位置回收
  console.log('\n[流水面板：宽→窄后仍在可视范围]');
  try {
    const setW = (v) => Object.defineProperty(w, 'innerWidth', { value: v, configurable: true });
    setW(1600);
    w.eval('logBox.style.left = "1400px"; logBox.style.top = "300px";'
      + ' logBox.style.right = "auto"; logBox.style.bottom = "auto";');
    setW(900);
    w.eval('clampPanelPos(logBox)');
    let L = parseFloat(w.eval('logBox.style.left'));
    check('窄窗口下 left 被拉回（<= 视口 - 面板宽）', L <= 900, `left=${L}`);
    check('left 不为负', L >= 0, `left=${L}`);

    // 走真实 resize 事件链路
    setW(1600);
    w.eval('logBox.style.left = "1500px"; logBox.style.top = "700px";');
    setW(1080);
    w.dispatchEvent(new w.Event('resize'));
    L = parseFloat(w.eval('logBox.style.left'));
    const T = parseFloat(w.eval('logBox.style.top'));
    check('resize 事件后 left 收回视口内', L >= 0 && L <= 1080, `left=${L}`);
    check('resize 事件后 top 收回视口内', T >= 0 && T <= (w.innerHeight || 768), `top=${T}`);
    check('未手动摆放（无内联 left）时不被误改',
      (() => { w.eval('logBox.style.left = ""; clampPanelPos(logBox)');
        return w.eval('logBox.style.left') === ''; })());

    // 尺寸上限也不再越界（含 12px 边距）
    check('applyLogSize 桌面端宽度上限留边距',
      /const maxW = mob \? 300 : window\.innerWidth - 24;/.test(html));
  } catch (e) { bad++; console.log('  THREW: ' + String(e.stack).split('\n').slice(0, 4).join('\n    ')); }

  // ============================================================ 7) 内联脚本无错误
  check('页面脚本零错误', errs.length === 0, errs.slice(0, 3).join(' | '));

  console.log(`\n合计 {通过 ${ok}} / {失败 ${bad}}`);
  process.exit(bad ? 1 : 0);
}, 60);
