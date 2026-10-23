/* ui_perf_check.js —— 前端渲染性能与「每帧无条件写 DOM」回归闸门
 *
 * 背景（2026-10-12）
 * -----------------
 * 页面是 450ms 轮询 + 每次 `render(st)` 全量重绘的长驻 SPA。轮询期间状态**大多数时候
 * 根本没变**，所以渲染路径上任何「无条件写 DOM」都会变成每秒若干次的白白重排。
 * 实测元凶：`fitMyRow()` 每帧给 <html> 根元素 setProperty("--tile-w"/"--my-scale")
 * —— 写一次 = 整页样式失效 + 重排，表现就是「界面渲染滞后、手牌不跟手」。
 *
 * 本脚本做两件事：
 *   1) 结构断言：本轮 UI 改动（纯图标导航 / 头像浮窗 / 节奏可填 / 设置行紧凑 / 统计新口径）
 *   2) 性能闸门：用真实 DOM 连渲多次，断言「状态没变 → 根元素 CSS 变量写入 0 次」，
 *      同时断言「状态变了 → 必须写入」，防止缓存把必要更新一起吃掉（那会变成手牌不刷新）。
 *
 * 运行（jsdom 装在托管 node workspace 里）：
 *   NODE_PATH=C:/Users/LIGF/.workbuddy/binaries/node/workspace/node_modules \
 *     node scripts/ui_perf_check.js
 */
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
    // jsdom 未实现 matchMedia（页面用它判定触屏档位）→ 补非触屏桩，否则脚本会中断
    if (!w.matchMedia) w.matchMedia = q => ({ matches: false, media: String(q),
      onchange: null, addEventListener() {}, removeEventListener() {},
      addListener() {}, removeListener() {}, dispatchEvent: () => false });
    if (!w.ResizeObserver) w.ResizeObserver = class { observe() {} disconnect() {} };
    // 只喂 /api/meta（填充 TILES），其余请求挂起：boot() 会停住但不抛未捕获异常
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
const doc = w.document;
const $ = id => doc.getElementById(id);

let pass = 0, fail = 0;
function check(name, cond, extra = '') {
  if (cond) { pass++; console.log(`  PASS  ${name}${extra ? '  ' + extra : ''}`); }
  else { fail++; console.log(`  FAIL  ${name}${extra ? '  ' + extra : ''}`); }
}

/* 造一份最小牌局状态；hand 可替换，用来验证「内容变了必须放行」 */
function makeState(hand, extra = {}) {
  const st = {
    hand_index: 3, wall_left: 42, over: false, turn: 0, my_seat: 0,
    scores: [0, 0, 0, 0], last: null, last_draw: null, pending: null,
    ji_tiles: [9], kaiju_flip: null, my_ting: [], events: [],
    players: [0, 1, 2, 3].map(i => ({
      seat: i, name: 'P' + i, score: 0, hand_count: 13, hand_tiles: null,
      melds: [], discards: [], discard_tags: [],
      baojiao: false, has_gang: false, is_dealer: i === 0,
    })),
    hand: hand.slice(),
  };
  return Object.assign(st, extra);
}

/* 统计根元素上的 CSS 变量写入次数（--tile-w / --my-scale 就是重排的来源） */
function countRootVarWrites(fn) {
  const hits = [];
  const proto = w.CSSStyleDeclaration.prototype;
  const origSet = proto.setProperty;
  const origRm = proto.removeProperty;
  proto.setProperty = function (k, v) {
    if (String(k).indexOf('--') === 0) hits.push('set ' + k);
    return origSet.call(this, k, v);
  };
  proto.removeProperty = function (k) {
    if (String(k).indexOf('--') === 0) hits.push('rm ' + k);
    return origRm.call(this, k);
  };
  try { fn(); } finally { proto.setProperty = origSet; proto.removeProperty = origRm; }
  return hits.length;
}

setTimeout(() => {
  console.log('=== 1. 页面加载 ===');
  check('加载无 JS 运行时错误', errs.length === 0, errs.slice(0, 2).join(' | '));

  console.log('\n=== 2. 左上角导航：纯图标 + 两列 ===');
  check('#nav 分文字列与图标列', !!doc.querySelector('#nav .navcol') && !!doc.querySelector('#nav .navcol.icons'));
  /* 2026-10-12 #86：结束本局从桌面导航迁入设置 → 界面显示（#set-endhand） */
  check('结束本局迁入设置面板（桌面无 btn-endhand）',
    !doc.querySelector('#btn-endhand') && !!doc.querySelector('#set-endhand'));
  check('设置/透视/声音只留图标（无 <i> 文字）',
    !doc.querySelector('#btn-settings i') && !doc.querySelector('#btn-peek i') && !doc.querySelector('#btn-sound i'));
  check('文字项仍在左列（数据统计/对局记录/局数）',
    ['btn-stats', 'btn-history', 'round-chip'].every(id => $('nav').querySelector('.navcol') !== null && $('nav').querySelector('#' + id)));
  /* 2026-10-12 #83/#86 重排：导航 = 文字列 + 图标列 + 第三列（本局鸡牌图片），
     结束本局不再在导航里。 */
  check('导航第三列 = 鸡牌图片；左列末位是局数、图标列只剩设置/透视/声音',
    [...doc.querySelectorAll('#nav .navcol')[0].children].map(n => n.id).join(',') ===
      'btn-stats,btn-history,round-chip' &&
    [...doc.querySelectorAll('#nav .navcol')[1].children].map(n => n.id).join(',') ===
      'btn-settings,btn-peek,btn-sound' &&
    !!doc.querySelector('#nav > #kaiju-info'));
  check('两列逐行对齐：统一高度变量 --nav-btn-h 驱动 按钮/图标宽/局数高',
    /#nav \{[^}]*--nav-btn-h: 32px/.test(html) &&
    /#nav button\.icon \{ width: var\(--nav-btn-h\)/.test(html) &&
    /#nav #round-chip \{ height: var\(--nav-btn-h\)/.test(html));
  check('结束本局走弹窗确认（非 .arm 连点两段确认）',
    !!$('endhand-mask') && html.indexOf('openEndhandConfirm') !== -1 && html.indexOf('endarm') === -1);

  console.log('\n=== 3. 头像选择器：虚框 + 浮窗 ===');
  const picks = doc.querySelectorAll('#set-players .pickent');
  check('设置页渲染 4 个虚框入口', picks.length === 4, `实际 ${picks.length}`);
  check('入口含当前头像缩略图与头像名（「选择头像」按钮已删，点虚框直接开浮窗）',
    !!doc.querySelector('#set-players .pickthumb') && !!doc.querySelector('#set-players .pickname')
    && !doc.querySelector('#set-players .pickbtn'));
  check('不再平铺全部候选（无 .pickgrid）', doc.querySelectorAll('#set-players .pickgrid').length === 0);
  try {
    w.openAvaPicker(0);
    const n = doc.querySelectorAll('#ava-grid .pick').length;
    check('浮窗打开后列出全部候选头像', n >= 20, `${n} 个`);
    check('浮窗已显示', $('ava-mask').classList.contains('on'));
    w.closeAvaPicker();
    check('浮窗可关闭', !$('ava-mask').classList.contains('on'));
  } catch (e) { check('头像浮窗可用', false, e.message); }

  console.log('\n=== 4. 对手出牌节奏：可填秒数 ===');
  const sp = $('set-oppspeed');
  check('set-oppspeed 是数字输入框', !!sp && sp.tagName === 'INPUT' && sp.type === 'number');
  check('配 4 个常用档 chip（悠闲/普通/快速/瞬发）',
    doc.querySelectorAll('.chip[data-for="set-oppspeed"]').length === 4);
  check('开局页 speed 同为数字输入框', $('speed') && $('speed').tagName === 'INPUT');
  check('修掉旧的 " normal" 英文残留', html.indexOf('> normal<') === -1);

  console.log('\n=== 5. 设置行不再独占整行 ===');
  check('.setrow 已改为 flex-start + wrap',
    /\.setrow \{ display: flex; flex-wrap: wrap/.test(html));
  check('下拉不再固定 168px / 100% 宽',
    !/min-width: 168px/.test(html) && !/\.setgrid \.setrow select \{ width: 100%/.test(html));

  console.log('\n=== 6. 统计面板口径 ===');
  check('标签为「送胡」', html.indexOf('["送胡"') !== -1);
  check('「非平胡·胡牌」补回间隔点', html.indexOf('["非平胡·胡牌"') !== -1);
  check('旧标签已移除（点胡 / 非平胡胡牌）',
    html.indexOf('["点胡"') === -1 && html.indexOf('["非平胡胡牌"') === -1);
  check('统计项挂 data-q 悬浮解释', /data-q="\$\{tip\}/.test(html) || /class="stat"\$\{tip/.test(html));
  check('新增「本人胡牌」「本人累计」',
    html.indexOf('["本人胡牌"') !== -1 && html.indexOf('["本人累计"') !== -1);

  console.log('\n=== 7. 每帧渲染的闸门是否都在 ===');
  check('fitMyRow 输入指纹', html.indexOf('html._fitKey === key') !== -1);
  check('hand 宽度值比较', html.indexOf('handBox._w !== handW') !== -1);
  check('牌河指纹', html.indexOf('box._rk === rk') !== -1);
  check('流水用「最后一条事件 hand-seq」做指纹（滑动窗口下长度不可用）',
    html.indexOf('tail.hand + "-" + tail.seq') !== -1);
  check('wall-num / me-score 走 setText',
    html.indexOf('setText(el("wall-num")') !== -1 && html.indexOf('setText(el("me-score")') !== -1);
  check('庄标 display 走值比较（不再每帧写 style.display）', html.indexOf('mw._d !== d') !== -1);
  check('教练面板摆放闸门', html.indexOf('COACH_GEO === geo') !== -1);

  console.log('\n=== 8. 本轮排版：设置三页 + 统计合并 ===');
  /* 出牌节奏页：超时时间并入「超时自动操作」同一行（参数挂在 .ghead 里，不再是独立 .setrow） */
  check('超时：开关 + 时间同一行（参数挂在 #grp-timeout .ghead 内）', (() => {
    const g = $('grp-timeout');
    if (!g) return false;
    const head = g.querySelector('.ghead');
    if (!head || !g.querySelector('#set-timeouton')) return false;
    return !!head.querySelector('#set-timeouton') && !!head.querySelector('#set-timeout') &&
      head.querySelectorAll('.setrow').length === 0;
  })(), `ghead 内 setrow=${$('grp-timeout') ? $('grp-timeout').querySelectorAll('.setrow').length : 'n/a'}`);
  /* 界面显示行（2026-10-12 并入「出牌与显示」）：听牌提示 / 流水面板 /
     冲锋鸡横鸡动画（#84）/ 最新弃牌指示颜色 / 结束对局按钮（#86）/
     记牌器开关（2026-10-12 晚，默认关）/ 开局鸡牌提醒时长（2026-10-13 #3）共 7 项同属一个 .setgrid */
  check('界面显示：7 项同属一个 .setgrid（含 #84 动画时长、#86 结束对局、记牌器、开局鸡牌提醒）', (() => {
    const g = $('set-tipon') && $('set-tipon').closest('.setgrid');
    if (!g) return false;
    const ids = ['set-tipon', 'set-logshow', 'set-ind-color', 'set-fxdur',
                 'set-kaiju-fxdur', 'set-endhand', 'set-counteron'];
    return ids.every(id => g.querySelector('#' + id)) &&
      !g.querySelector('#set-rec-color') &&
      [...g.children].filter(c => c.classList.contains('setrow')).length === 7;
  })());
  /* 规则开关页：名字列等长 → 下拉左缘上下对齐（--setlbl-w 固定宽度 + 网格第二列） */
  check('规则开关页：名字列等长（--setlbl-w）→ 下拉左缘对齐',
    /--setlbl-w: 9\.8em/.test(html) &&
    /\.setgrid \.setrow \{ display: grid; grid-template-columns: var\(--setlbl-w\)/.test(html));
  /* 统计面板：本人两项并入主网格（不再有 h4.statsub 子标题块） */
  check('统计面板：12 项合成一个网格（已删 .statsub 子标题）',
    html.indexOf('statsub') === -1 &&
    html.indexOf('["本人胡牌"') !== -1 && html.indexOf('["本人累计"') !== -1);

  console.log('\n=== 9. 渲染路径实测 ===');
  if (!w.eval('TILES.length')) {
    // fixtures 没喂到时的兜底：补一张最小牌表，保证下面的渲染断言能跑
    w.eval('for (let i = 0; i < 27; i++) TILES[i] = ' +
      '{ id: i, name: "t" + i, cn: "牌" + i, suit: Math.floor(i / 9), rank: i % 9 };');
  }
  check('牌表就绪', w.eval('TILES.length') === 27);

  const st = makeState([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]);
  w.render(st);
  check('render 正常执行', $('hand').innerHTML.length > 0, $('hand').innerHTML.length + ' 字符');
  check('余牌数已上屏', $('wall-num').textContent === '42', $('wall-num').textContent);
  check('局数已上屏', ($('round-chip').textContent || '').indexOf('3') !== -1, $('round-chip').textContent);

  // ★ 核心闸门一：状态没变 → 一次根元素变量都不该写
  const idle = countRootVarWrites(() => { for (let i = 0; i < 8; i++) w.render(st); });
  check('同状态连渲 8 次：根元素 CSS 变量写入 0 次', idle === 0, `实际 ${idle} 次`);

  /* ★ 核心闸门三：非根元素的「每帧无条件写 style」也要盯住。
     典型漏网之鱼是 #me-wind（庄标）的 display —— 不在根元素上，上面的计数器看不见。
     做法：在定义 display 访问器的那层原型上换成带计数的 setter。
     注意 jsdom（>=24）把它放在 CSSStyleProperties.prototype 上，**不是**
     CSSStyleDeclaration.prototype —— 写死前者会挂不上钩子、造成假 PASS，
     所以这里沿原型链动态定位，并用一次「探测写」自证钩子确实生效。 */
  const mwEl = $('me-wind');
  const probeEl = doc.createElement('div');
  doc.body.appendChild(probeEl);
  const protoOf = (prop) => {
    let p = Object.getPrototypeOf(probeEl.style);
    while (p) { if (Object.getOwnPropertyDescriptor(p, prop)) return p; p = Object.getPrototypeOf(p); }
    return null;
  };
  const dProto = protoOf('display');
  const dDesc = dProto && Object.getOwnPropertyDescriptor(dProto, 'display');
  let hookOK = false, mwWrites = 0, probeWrites = 0, otherWrites = 0;
  if (dDesc && dDesc.set) {
    Object.defineProperty(dProto, 'display', {
      configurable: true, enumerable: dDesc.enumerable, get: dDesc.get,
      set: function (v) {
        if (this === mwEl.style) mwWrites++;
        else if (this === probeEl.style) probeWrites++;
        else otherWrites++;
        dDesc.set.call(this, v);
      },
    });
    try {
      probeEl.style.display = 'none';                       // 探测写：只碰临时元素
      hookOK = probeWrites === 1;
      mwWrites = 0; otherWrites = 0;
      for (let i = 0; i < 8; i++) w.render(st);             // 同状态连渲
    } finally { Object.defineProperty(dProto, 'display', dDesc); }
  }
  probeEl.remove();
  check('display 写入钩子有效（探测写被数到）', hookOK);
  check('同状态连渲 8 次：#me-wind display 写入 0 次', hookOK && mwWrites === 0,
    `实际 ${mwWrites} 次（其他元素 ${otherWrites} 次）`);

  // ★ 核心闸门二：状态变了 → 必须写（否则就是缓存把更新吃掉了）
  // 副露对象的真实结构（见 meldCells）：{ type, tile, src, ji }
  //   type 1=碰 2=明杠 3=补杠 4=暗杠；tile 是单张牌面索引（不是 tiles 数组）；
  //   src = 副露来源座位（0..3），ji 为鸡牌角标（可空）。
  const st2 = makeState([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 18], {
    players: st.players.map((p, i) => i === 0
      ? Object.assign({}, p, { melds: [{ type: 1, tile: 5, src: 1 }] })
      : i === 3
        ? Object.assign({}, p, { melds: [{ type: 2, tile: 9, src: 2 }] })
        : p),
  });
  const changed = countRootVarWrites(() => w.render(st2));
  check('手牌/副露变化后：根元素 CSS 变量重新写入', changed > 0, `实际 ${changed} 次`);
  check('手牌内容确实更新了（data-t=18 已上屏）',
    $('hand').innerHTML.indexOf('data-t="18"') !== -1);

  console.log('\n=== 10. 对局回放：UI 骨架与接线 ===');
  /* 2026-10-12 重构：回放不再自画「简化牌桌」（.replaymodal/#rp-board 已删除），
     改为复用实盘牌桌，只留一条 #replay-mask > .rp-bar 底部工具条。 */
  /* 2026-10-12 二次改版：控制条拆成左右两块（.rp-side），分放对家手牌两侧
     2026-10-12 五次改版：改为**底部弹出式抽屉**（.rp-drawer + .rp-hot），
     左上角只留 🔴回放角标 + 本局鸡牌（.rp-side.left）。 */
  check('回放控制区存在（底部 .rp-drawer 抽屉 + 左上角 .rp-side.left）、旧简化牌桌已删除',
    !!$('replay-mask') && !!$('rp-drawer') && !!$('rp-hot')
    && $('replay-mask').querySelectorAll('.rp-side').length === 1
    && !!$('replay-mask').querySelector('.rp-side.left')
    && !$('replay-mask').querySelector('.rp-side.right')
    && !$('replay-mask').querySelector('.replaymodal') && !$('rp-board'));
  check('回放控件齐全（标题/当前步/步数/进度条/上下步/播放/关闭；跳巡组件已删）',
    ['rp-title', 'rp-now', 'rp-step', 'rp-range',
     'rp-prev', 'rp-next', 'rp-play', 'rp-close'].every(id => !!$(id)) &&
    !$('rp-round') && !$('rp-go'));
  check('#rp-range 是 range 拖动条（按巡定位）', $('rp-range') && $('rp-range').type === 'range');
  check('控制区不遮我的手牌（挂在对家两侧，不再压底部）',
    /#replay-mask \{ background: none;[\s\S]{0,60}align-items: flex-start/.test(html) &&
    /\.rp-side \{ pointer-events: auto/.test(html));
  check('回放复用实盘牌桌：工具条不遮桌 + html.replaying 藏实时控件',
    /#replay-mask \{ background: none;[\s\S]{0,80}pointer-events: none/.test(html) &&
    /html\.replaying #nav/.test(html) && /html\.replaying #actions/.test(html));
  /* 历史行可点：委托在 #hist-body 上，靠 data-rid 认领 */
  check('历史行带 data-rid 且整行可点（CSS cursor:pointer）',
    html.indexOf('data-rid="${rec.rid}"') !== -1 &&
    /#hist-body table\.res\.clickable \{ cursor: pointer/.test(html));
  check('#hist-body 用事件委托（closest("table.res[data-rid]")）打开回放',
    /el\("hist-body"\)\.addEventListener\("click"/.test(html) &&
    html.indexOf('closest("table.res[data-rid]")') !== -1);
  check('回放状态机函数齐（openReplay/closeReplay/replayState/replayToState/renderReplayFrame）',
    ['function openReplay', 'function closeReplay', 'function replayState',
     'function replayToState', 'function renderReplayFrame',
     'function enterReplayMode', 'function exitReplayMode']
      .every(s => html.indexOf(s) !== -1) && html.indexOf('replayGotoRound') === -1);
  check('无回放的旧局显示「无回放」而非可点',
    html.indexOf('无回放') !== -1 && html.indexOf('has_replay') !== -1);

  check('全程无新异常', errs.length === 0, errs.slice(0, 2).join(' | '));

  console.log(`\n合计：${pass} PASS / ${fail} FAIL`);
  if (errs.length) { console.log('\n捕获到的错误：'); errs.slice(0, 5).forEach(e => console.log('  ' + e)); }
  dom.window.close();
  process.exit(fail || errs.length ? 1 : 0);
}, 900);
