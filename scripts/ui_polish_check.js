// 本轮「界面打磨」六件事的 e2e（jsdom 加载真实页面 + 真实局面夹具）：
//   F 大师建议面板：打出牌后隐藏，下一次轮到我决策时再显示
//   G 左上角按钮：文字按钮与局数竖排；四个图标折叠成「⋯」，点击竖排展开
//   H 窄桌面（1024~1200px）：我的信息板块抬到手牌左上方（头像/新摸牌不再出屏）
//   I 设置面板切换分类：高度过渡（CSS 过渡 + JS 量高赋值 + 收尾清理）
//   J 设置：「显示大师推荐」从「出牌与节奏」抽成独立的「大师推荐」分类
//   K 分值表底部提示文字：专用小字弱色样式（原 div 只挂 .sechint，继承正文字号与亮色）
//   L 单人视角结算表：删表头行/各家独有项提示行/末行总计；首行居中「本局结算总计 ±X」；
//                      头像突出、表格首列左侧与头像右侧平齐；手牌区与表格同底色成一个板块
const fs = require('fs');
const { JSDOM } = require('jsdom');

const ROOT = 'C:/Users/LIGF/WorkBuddy/2026-09-30-12-49-43/zhuoji-mahjong';
const html = fs.readFileSync(ROOT + '/web/static/dushan.html', 'utf8');
const meta = JSON.parse(fs.readFileSync(ROOT + '/reports/ui_fixtures_meta.json', 'utf8'));
const mid0 = JSON.parse(fs.readFileSync(ROOT + '/reports/ui_fixtures_mid.json', 'utf8'))[0].state;
const OVER = JSON.parse(fs.readFileSync(ROOT + '/reports/ui_fixtures_over.json', 'utf8'));
const errs = [];

const HINT_PAYLOAD = {
  ok: true, head: 'model', model: 'dushan_master', value: 0.3,
  items: [{ kind: 'discard', tile: mid0.hand[0], label: '打出 1万', prob: 0.5, tags: [], lines: ['打 1万 最稳'] }],
  tip: '保留中间搭子',
};

const dom = new JSDOM(html, {
  runScripts: 'dangerously',
  pretendToBeVisual: true,
  url: 'http://127.0.0.1:8770/static/dushan.html',
  beforeParse(w) {
    if (!w.matchMedia) w.matchMedia = q => ({ matches: false, media: String(q),
      onchange: null, addEventListener() {}, removeEventListener() {},
      addListener() {}, removeListener() {}, dispatchEvent: () => false });
    // jsdom 未实现 <dialog> 的 showModal/close（fitStage 在窄屏会挂门禁层）
    const dlg = w.HTMLDialogElement && w.HTMLDialogElement.prototype;
    if (dlg) {
      if (!dlg.showModal) dlg.showModal = function () { this.setAttribute('open', ''); };
      if (!dlg.close) dlg.close = function () { this.removeAttribute('open'); };
    }
    w.fetch = (url) => {
      const u = String(url);
      if (u.includes('/api/meta')) return Promise.resolve({ ok: true, json: () => Promise.resolve(meta) });
      if (u.includes('/api/hint')) return Promise.resolve({ ok: true, json: () => Promise.resolve(HINT_PAYLOAD) });
      if (u.includes('/api/ting')) return Promise.resolve({ ok: true, json: () => Promise.resolve({ ok: true, map: {} }) });
      return new Promise(() => {});
    };
    w.addEventListener('error', e => errs.push('window.error: ' + ((e.error && e.error.stack) || e.message)));
  },
});
const w = dom.window;
const el = id => w.document.getElementById(id);
const clone = o => JSON.parse(JSON.stringify(o));
const sleep = ms => new Promise(r => setTimeout(r, ms));
let ok = 0, bad = 0;
const check = (name, cond, detail = '') => {
  (cond ? ok++ : bad++);
  console.log(`  [${cond ? 'ok  ' : 'FAIL'}] ${name}${detail ? '   ' + detail : ''}`);
};

(async () => {
  await sleep(1500);                       // 等 boot 拿到 meta（TILES/META 就绪）
  w.eval('SID = "e2e"');

  // ==================================================== F 大师建议面板显隐时机
  console.log('\n[F 大师建议面板：打完牌隐藏 / 下次决策再显示]');
  try {
    w.eval('SET.hintOn = true; SET.reasonMode = "full"; SET.hintDelay = 0;');
    const coachOn = () => el('coach').classList.contains('on');

    const stA = clone(mid0); stA.turn = stA.my_seat;      // 轮到我出牌
    w.eval('TURN_KEY = ""; RECOMMEND = null;');
    w.render(stA);
    await sleep(250);
    check('轮到我决策 → 面板显示', coachOn() && w.eval('SET.coach') === true,
      `coachOn=${coachOn()} turnOn=${w.eval('coachTurnOn()')}`);

    // 打出牌：doAct 成功路径收尾就是 hideCoach()
    check('doAct 成功后调用 hideCoach()', w.eval('doAct.toString()').includes('hideCoach()')
      && !w.eval('doAct.toString()').includes('syncCoachPanel'), '');
    w.eval('hideCoach()');
    check('打出牌后面板立刻隐藏', !coachOn(), `coachOn=${coachOn()}`);

    // 别人的回合：即使开关开着也不显示
    const stB = clone(stA); stB.turn = (stA.my_seat + 1) % 4; stB.pending = null;
    w.render(stB);
    check('对手回合 → 面板保持隐藏', !coachOn() && w.eval('coachTurnOn()') === false,
      `coachOn=${coachOn()}`);

    // 下一个我的决策点（手牌/序号变化 → TURN_KEY 变化）→ 重新显示
    const stC = clone(stA);
    stC.hand = stA.hand.slice(0, 13);
    stC.hand_index = stA.hand_index + 1;
    stC.wall_left = stA.wall_left - 1;
    w.render(stC);
    await sleep(250);
    check('下一次轮到我 → 面板重新显示', coachOn(),
      `coachOn=${coachOn()} tk=${Boolean(w.eval('TURN_KEY'))}`);

    // 关掉推荐总开关：无论如何都不显示
    w.eval('SET.hintOn = false; syncCoachPanel();');
    check('推荐总开关关闭 → 面板隐藏', !coachOn(), '');
    w.eval('SET.hintOn = true;');
  } catch (e) { bad++; console.log('  THREW: ' + String(e.stack).split('\n').slice(0, 4).join('\n    ')); }

  // ==================================================== G 左上角：双列（文字列 + 纯图标列）
  // 2026-10-12 第三次改版（#83/#86）：导航 = 文字列 + 图标列 + 第三列（本局鸡牌图片）；
  // 结束本局从桌面导航迁入设置 → 界面显示（#set-endhand，红边弱化）。
  console.log('\n[G 左上角：左列 数据统计/对局记录/局数，右列 设置/透视/声音，第三列 鸡牌图片]');
  try {
    const nav = el('nav');
    const cols = [...nav.children];
    check('导航 = 文字列 + 图标列 + 鸡牌图片列',
      cols.length === 3 && cols[0].classList.contains('navcol') &&
      cols[1].classList.contains('navcol') && cols[1].classList.contains('icons') &&
      cols[2].id === 'kaiju-info',
      cols.map(c => c.className || c.id).join(' | '));
    check('左列 = 数据统计 / 对局记录 / 局数（结束本局已迁入设置面板）',
      [...cols[0].children].map(n => n.id).join(',') ===
      'btn-stats,btn-history,round-chip' && !el('btn-endhand') && !!el('set-endhand'),
      [...cols[0].children].map(n => n.id).join(','));
    check('右列（图标） = 设置 / 透视 / 声音',
      [...cols[1].children].map(n => n.id).join(',') ===
      'btn-settings,btn-peek,btn-sound',
      [...cols[1].children].map(n => n.id).join(','));
    check('设置/透视/声音只保留 icon（无 <i> 文字标签）',
      !nav.querySelector('#btn-settings i') && !nav.querySelector('#btn-peek i') &&
      !nav.querySelector('#btn-sound i'));
    check('结束本局在设置面板内（红边弱化 .setbtn-danger，桌面无 SVG 按钮）',
      !!el('set-endhand') && !nav.querySelector('svg') &&
      /button\.setbtn-danger \{[^}]*border: 1px solid rgba\(226, 75, 74/.test(html));
    check('CSS: 统一按钮高度变量（两列逐行对齐）',
      /--nav-btn-h: 32px/.test(html) && /#nav button\.icon \{ width: var\(--nav-btn-h\)/.test(html) &&
      /#nav #round-chip \{ height: var\(--nav-btn-h\)/.test(html));
    check('局数不再绝对定位（跟随左列竖排）', /#round-chip \{ position: static;/.test(html) ||
      /#nav #round-chip \{[^}]*height: var\(--nav-btn-h\)/.test(html));
    check('结束本局走弹窗确认（不再有 .arm 连点两段确认）',
      !!el('endhand-mask') && html.indexOf('openEndhandConfirm') !== -1 &&
      html.indexOf('endarm') === -1);
  } catch (e) { bad++; console.log('  THREW: ' + String(e.stack).split('\n').slice(0, 4).join('\n    ')); }

  // ==================================================== H 窄桌面：我的信息板块抬到左上方
  console.log('\n[H 窄桌面：信息板块移到手牌左上方]');
  try {
    const setW = v => Object.defineProperty(w, 'innerWidth', { value: v, configurable: true });
    const cls = () => w.document.documentElement.classList;
    check('窄屏阈值 > 桌面最小宽度', w.eval('NARROW_W') > w.eval('DESKTOP_MIN_W'),
      `NARROW_W=${w.eval('NARROW_W')}`);
    setW(1024); w.eval('fitStage()');
    check('1024px → html.narrow（信息板块抬起）', cls().contains('narrow') && !cls().contains('mob'),
      `narrow=${cls().contains('narrow')} mob=${cls().contains('mob')}`);
    setW(1400); w.eval('fitStage()');
    check('1400px → 恢复一行居中（无 narrow）', !cls().contains('narrow'));
    setW(760); w.eval('fitStage()');
    check('小屏走 mob 档、不叠加 narrow', cls().contains('mob') && !cls().contains('narrow'));
    setW(1024); w.eval('fitStage()');
    check('CSS: narrow 下 .seat-me 绝对定位到 me-row 左上',
      /html\.narrow #me-row \.seat-me \{ position: absolute; left: 0; bottom: calc\(100% \+ 4px\); \}/.test(html));
    setW(1400); w.eval('fitStage()');
  } catch (e) { bad++; console.log('  THREW: ' + String(e.stack).split('\n').slice(0, 4).join('\n    ')); }

  // ==================================================== I 设置分类切换高度过渡
  console.log('\n[I 设置面板切换分类的高度过渡]');
  try {
    check('CSS: .setpanes 有 height 过渡 + 过渡期 overflow:hidden',
      /\.setpanes \{[^}]*transition: height \.24s cubic-bezier/.test(html) &&
      /\.setpanes\.animating \{ overflow: hidden; \}/.test(html));
    const src = w.eval('showSetPane.toString()');
    check('JS: 量旧高→设内联高→过渡结束清理',
      src.includes('const from = panes.offsetHeight') &&
      src.includes('panes.style.height = from') &&
      src.includes('panes.style.height = to') &&
      src.includes('panes.style.height = ""'), '');
    check('JS: 多次点击会被打断（清 timer）', /clearTimeout\(PANE_TM\)/.test(src));
    check('JS: 尊重 prefers-reduced-motion',
      /@media \(prefers-reduced-motion: reduce\) \{ \.setpanes \{ transition: none; \} \}/.test(html));
  } catch (e) { bad++; console.log('  THREW: ' + String(e.stack).split('\n').slice(0, 4).join('\n    ')); }

  // ==================================================== J 设置分类（2026-10-12 三次合并）
  console.log('\n[J 设置分类：玩家与模型 / 出牌与显示 / 声音]');
  try {
    const panes = [...el('set-panes').querySelectorAll('section')].map(s => s.dataset.pane);
    check('分类顺序 = 玩家与模型/出牌与显示/声音/规则开关/分值表',
      panes.join(',') === 'who,play,sound,rule,score', panes.join(','));
    check('导航按钮文案同步', [...el('set-nav').querySelectorAll('button')]
      .map(b => b.textContent.trim()).join(',') === '玩家与模型,出牌与显示,音效与音量,规则开关,分值表');
    const whoPane = el('set-panes').querySelector('section[data-pane="who"]');
    const playPane = el('set-panes').querySelector('section[data-pane="play"]');
    check('旧「大师推荐」「界面显示」分类已删除',
      !el('set-panes').querySelector('section[data-pane="hint"]') &&
      !el('set-panes').querySelector('section[data-pane="ui"]'));
    check('grp-hint（含推荐出牌形式）并入「玩家模型」分类',
      !!whoPane.querySelector('#grp-hint') && !!whoPane.querySelector('#set-rec-form'));
    check('旧「推荐自动上浮」开关与上浮时机已删除',
      !el('set-floaton') && !el('set-floatat') && !whoPane.querySelector('#grp-float'));
    check('推荐策略与推荐理由同一行',
      (() => { const r = whoPane.querySelector('#set-strategy').closest('.setrow');
        return !!r && !!r.querySelector('#set-reasonmode'); })());
    check('推荐出牌指示器颜色在推荐块内（不再在界面显示里）',
      !!whoPane.querySelector('#set-rec-color') && !playPane.querySelector('#set-rec-color'));
    check('出牌板块 = 超时 + 节奏 + 界面显示三行开关',
      [...playPane.querySelectorAll('.setgroup')].map(g => g.id).join(',') === 'grp-timeout,grp-speed' &&
      playPane.contains(el('set-tipon')) && playPane.contains(el('set-logshow')) &&
      playPane.contains(el('set-ind-color')));
    check('「出牌与显示」导航文案',
      [...el('set-nav').querySelectorAll('button')][1].textContent.trim() === '出牌与显示');
  } catch (e) { bad++; console.log('  THREW: ' + String(e.stack).split('\n').slice(0, 4).join('\n    ')); }

  // ==================================================== K 分值表底部提示文字
  console.log('\n[K 分值表底部提示：小字弱色说明块]');
  try {
    const score = el('set-panes').querySelector('section[data-pane="score"]');
    const note = score.querySelector('.setnote');
    check('提示块改用 .setnote（不再是裸 .sechint）',
      !!note && note.textContent.includes('被碰/杠走的鸡') &&
      !score.querySelector('div.sechint'), note ? note.textContent.trim().slice(0, 18) : '无');
    check('CSS: 11.5px 弱色 + 左侧金色细条',
      /\.setnote \{[^}]*font-size: 11\.5px/.test(html) &&
      /\.setnote \{[^}]*color: #93a389/.test(html) &&
      /\.setnote \{[^}]*border-left: 2px solid rgba\(232,184,75,\.30\)/.test(html));
    const style = w.getComputedStyle(note);
    check('实际字号被压到说明级（< 正文 14px）',
      parseFloat(style.fontSize) <= 12.5, `font-size=${style.fontSize}`);
  } catch (e) { bad++; console.log('  THREW: ' + String(e.stack).split('\n').slice(0, 4).join('\n    ')); }

  // ==================================================== L 单人视角结算表
  console.log('\n[L 单人视角结算表：首行总计 + 头像列对齐 + 板块合并]');
  try {
    const win = OVER.find(o => o.state.result.type === 'win');
    const st = clone(win.state);
    st.my_seat = 0;
    st.players.forEach((p, i) => { p.seat = i; p.name = p.name || `P${i}`; });
    // 共享项（三家都有一份同键）→ 保证「共享区」存在；独有项 → 保证至少一个对手板块存在
    const others = [0, 1, 2, 3].filter(s => s !== 0);
    const other = others[0];
    const items = o => [{ label: '自摸', info: '', value: 3, who: 0 }]
      .concat(o === other ? [{ label: '杠分', info: '补杠 九条', value: 3, who: 0 }] : []);
    st.result = Object.assign({}, st.result, {
      pair: [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
      pair_detail: others.map(o => ({ a: 0, b: o, items: items(o) })),
      deltas: [9, 0, -9, 0],
      // 本节只测版式，鸡牌三分类另有 ui_settle_check 覆盖 → 清空鸡牌明细，
      // 免得「手中鸡牌」行（.chk 青绿底）成为共享段首行，与手牌行底色断言混淆。
      detail: [], ji_hold_detail: {}, fanji_detail: {}, ji_discard_detail: {},
      ji_capture_detail: {}, ji_claim_detail: [], gold_tiles: [], gold_mult: 2,
    });
    w.eval('SET.resLayout = "new"');
    w.__st = st;
    w.eval('LAST_STATE = window.__st; RES_VIEW = 0; RES_DISMISSED = false; render(window.__st);');

    const rows = [...w.document.querySelectorAll('#res-pairs .pairs-big tr')];
    const grand = rows.find(r => r.className === 'grandrow');
    const card = w.document.querySelector('#res-pairs .paircard');
    check('卡片首行 = 本局结算总计（居中、跨 5 列、带符号）',
      !!grand && rows[0] === grand && grand.firstElementChild.getAttribute('colspan') === '5' &&
      /本局结算总计[+-]?\d/.test(grand.textContent.replace(/\s/g, '')),
      grand ? grand.textContent.trim() : '无');
    check('表头行 / 各家独有项提示行 / 末行总计均已删除',
      w.document.querySelectorAll('#res-pairs .pairs-big th').length === 0 &&
      !el('res-pairs').textContent.includes('各家独有项') &&
      !el('res-pairs').textContent.includes('本局总计') &&
      !rows.some(r => r.className === 'total'));

    const own = w.document.querySelector('#res-pairs .pstriprow.ownstrip');
    const oppStrip = w.document.querySelector('#res-pairs .pstriprow.oppstrip');
    const ovhand = own && own.querySelector('.ovhand');
    check('头像放大突出（52px + 金边 + 绝对定位到首列缩进区）',
      /\.pstrip \.avatar \{\s*position: absolute; left: calc\(-1 \* var\(--pava-x\)\);/.test(html) &&
      /width: var\(--pava-w\); height: var\(--pava-w\)/.test(html) &&
      /--pava-w: 52px; --pava-x: 58px;/.test(html));
    check('表格首列（表头列）左侧与头像右侧平齐',
      /td:first-child, \.paircard \.pairs-big th:first-child \{\s*text-align: left; padding-left: var\(--pava-x\)/.test(html) &&
      /tr\.opphead td:first-child \{ padding-left: var\(--pava-x\);/.test(html) &&
      /tr\.det td:first-child \{ padding-left: var\(--pava-x\);/.test(html));

    const bgOf = node => (node ? w.getComputedStyle(node).backgroundColor : '');
    const ownBg = bgOf(own && own.firstElementChild);
    const sharedBg = bgOf(w.document.querySelector('#res-pairs tr.det.sh td'));
    const oppStripBg = bgOf(oppStrip && oppStrip.firstElementChild);
    const oppHeadBg = bgOf(w.document.querySelector('#res-pairs tr.opphead td'));
    check('手牌区底色 = 结算表格底色（视角玩家板块）',
      !!ownBg && ownBg === sharedBg && ownBg !== 'rgba(0, 0, 0, 0)', `${ownBg} vs ${sharedBg}`);
    check('手牌区底色 = 结算表格底色（对手板块）',
      !!oppStripBg && oppStripBg === oppHeadBg, `${oppStripBg} vs ${oppHeadBg}`);
    check('CSS: 手牌行与表格合并为一个板块（去掉板块内分割线）',
      /tr\.pstriprow\.ownstrip \+ tr\.shsum td \{ border-top: none; \}/.test(html) &&
      /tr\.pstriprow\.oppstrip \+ tr\.opphead td \{ border-top: none; \}/.test(html));
    check('CSS: 板块之间用间隔行分开，且每个对手板块有顶边',
      /tr\.gap td \{ height: 14px/.test(html) &&
      /\.pstriprow\.oppstrip td \{\s*padding-top: 12px; border-top: 2px solid/.test(html));
    check('手牌与标签仍在同一板块头里', !!ovhand && !!own.querySelector('.ovmeta'));
    w.eval('SET.resLayout = "new"');
  } catch (e) { bad++; console.log('  THREW: ' + String(e.stack).split('\n').slice(0, 4).join('\n    ')); }

  console.log(`\n合计 {通过 ${ok}} / {失败 ${bad}}`);
  if (errs.length) console.log('window errors:\n' + errs.slice(0, 5).join('\n'));
  process.exit(bad ? 1 : 0);
})();
