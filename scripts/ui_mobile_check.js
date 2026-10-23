/* ui_mobile_check.js —— 移动端 UI 优化 + 追加鸡牌 + 鸡牌特效 回归闸门
 *
 * 背景（2026-10-12，用户小米 15 Pro 实机反馈）
 * ------------------------------------------------
 *  ① 开局面板：模型下拉与标题各占一行 → 合并同一行；头像选择器每人一行 → 点击直接换
 *  ② 移动端结算面板高度不够 → 按钮与面板重叠点不到
 *  ③ 流水面板拖不动（每次只挪一点点）→ 触摸被浏览器判成滚动立刻 pointercancel
 *  ④ 记牌器被遮挡 → 窄屏移到手牌右上；结束本局按钮与上家头像重叠 → 导航改单行
 *  ⑤ 新功能：开局/设置可手动追加本局鸡牌（如贵阳八筒）
 *  ⑥ 新功能：冲锋鸡/横鸡打出播放全屏特效（普通鸡不播）
 *
 * jsdom 无布局引擎（getBoundingClientRect 全 0），所以**几何位置**不做数值断言，
 * 改为断言「驱动布局的 CSS 规则存在且写法正确」+ 「DOM 结构与交互状态正确」。
 *
 * 运行：
 *   NODE_PATH=C:/Users/LIGF/.workbuddy/binaries/node/workspace/node_modules \
 *     node scripts/ui_mobile_check.js
 */
const fs = require('fs');
const { JSDOM } = require('jsdom');

const ROOT = 'C:/Users/LIGF/WorkBuddy/2026-09-30-12-49-43/zhuoji-mahjong';
const html = fs.readFileSync(ROOT + '/web/static/dushan.html', 'utf8');
const meta = JSON.parse(fs.readFileSync(ROOT + '/reports/ui_fixtures_meta.json', 'utf8'));

let pass = 0, fail = 0;
function check(name, cond, extra = '') {
  if (cond) { pass++; console.log(`  PASS  ${name}${extra ? '  ' + extra : ''}`); }
  else { fail++; console.log(`  FAIL  ${name}${extra ? '  ' + extra : ''}`); }
}
const css = (re) => re.test(html);   // 直接在源码里查 CSS/JS 写法

const errs = [];
const dom = new JSDOM(html, {
  runScripts: 'dangerously',
  pretendToBeVisual: true,
  url: 'http://127.0.0.1:8770/static/dushan.html',
  beforeParse(w) {
    if (!w.matchMedia) w.matchMedia = q => ({ matches: false, media: String(q),
      onchange: null, addEventListener() {}, removeEventListener() {},
      addListener() {}, removeListener() {}, dispatchEvent: () => false });
    if (!w.ResizeObserver) w.ResizeObserver = class { observe() {} disconnect() {} };
    // 只喂 /api/meta；其余挂起（boot 只跑到初始化 UI 为止）
    w.fetch = (url) => {
      const u = String(url);
      if (u.includes('/api/meta')) return Promise.resolve({ ok: true, json: () => Promise.resolve(meta) });
      return new Promise(() => {});
    };
    w.addEventListener('error', e => errs.push('window.error: ' + ((e.error && e.error.stack) || e.message)));
  },
});
const w = dom.window;
const doc = w.document;
const el = id => doc.getElementById(id);
const G = (expr) => { try { return w.eval(expr); } catch (e) { return '<<eval err: ' + e.message + '>>'; } };
const sleep = ms => new Promise(r => setTimeout(r, ms));

setTimeout(async () => {
  /* ============ 1. M1 开局面板压缩 ============ */
  console.log('\n=== 1. M1 开局面板压缩 ===');
  check('M1a 对手卡：标题与下拉同行（flex 横排、h4 不再占整行）',
    css(/\.opp\s*\{[^}]*display:\s*flex/) && css(/\.opp h4\s*\{[^}]*margin:\s*0/));
  check('M1b 下拉宽度自适应可收缩（刚好显示文字）',
    css(/\.opp select\s*\{[^}]*flex:\s*1 1 auto/) && css(/\.opp select\s*\{[^}]*min-width:\s*0/));
  check('M1c 三个对手仍是一行三列', css(/\.opp-grid\s*\{[^}]*repeat\(3,\s*1fr\)/));

  const picks = el('boot-players') ? el('boot-players').querySelectorAll('.pickent') : [];
  check('M1d 开局页 4 家各一个头像入口（虚框，不再平铺候选项）', picks.length === 4, `入口 ${picks.length}`);
  /* 注意：页面注释与浮窗标题里合法出现「选择头像」字样，所以不看字面量，
     直接断言 DOM —— 入口内只有缩略图 + 派生名，**没有** button 元素。 */
  check('M1e 入口 = 缩略图 + 派生名，且入口内不含按钮',
    picks.length === 4 && [...picks].every(p =>
      p.querySelector('img.pickthumb') && p.querySelector('.pickname') && !p.querySelector('button'))
      && html.indexOf('class="pickbtn"') === -1);
  check('M1f 旧 pickhead 头部行已删除（不再每人多占一行）', html.indexOf('class="pickhead"') === -1);

  // 点头像入口 → 直接弹浮窗（省掉中间按钮）
  let opened = false;
  if (picks.length) {
    const r = new w.MouseEvent('click', { bubbles: true });
    picks[0].querySelector('img.pickthumb').dispatchEvent(r);
    opened = el('ava-mask').classList.contains('on');
  }
  check('M1g 点击头像入口 → 直接弹出头像选择浮窗', opened,
    `ava-mask.on=${el('ava-mask') && el('ava-mask').classList.contains('on')}`);
  if (el('ava-grid')) check('M1h 浮窗列出全部候选头像', el('ava-grid').children.length > 0,
    `${el('ava-grid').children.length} 个`);
  if (el('ava-mask')) el('ava-mask').classList.remove('on');

  /* ============ 2. M2 移动端结算面板 ============ */
  console.log('\n=== 2. M2 移动端结算面板 ===');
  check('M2a 结算面板顶对齐（不再居中挤压）', css(/html\.mob #result-mask\s*\{[^}]*align-items:\s*flex-start/));
  check('M2b 结算面板限高（高度不够时内部滚动）',
    css(/html\.mob #result-mask \.modal\s*\{[^}]*max-height:\s*calc\(100dvh/));
  check('M2c 底部按钮贴底避让（不再与面板重叠）',
    css(/html\.mob #res-actions\s*\{[^}]*bottom:\s*8px/));
  check('M2d #res-actions 元素存在', !!el('res-actions'));

  /* ============ 3. M3 流水面板拖动 ============ */
  console.log('\n=== 3. M3 流水面板拖动 ===');
  check('M3a #log 声明 touch-action（修 pointercancel 根因）',
    css(/#log\s*\{[^}]*touch-action:\s*pan-y/));
  check('M3b 内容区可滚动、拖动守卫跳过内容区',
    css(/#log-body\s*\{[^}]*overflow-y:\s*auto/) && html.indexOf('log-body') !== -1);
  check('M3c 拖动用 pointer capture（跟手不丢事件）',
    html.indexOf('setPointerCapture') !== -1 && html.indexOf('pointermove') !== -1);

  /* ============ 4. M4 记牌器 + 导航单行 ============ */
  console.log('\n=== 4. M4 记牌器与导航 ===');
  check('M4a 窄屏记牌器悬到手牌上方（right:0 + bottom:100%+8px）',
    css(/html\.mob #counter-btn\s*\{[^}]*position:\s*absolute/)
      && css(/html\.mob #counter-btn\s*\{[^}]*bottom:\s*calc\(100% \+ 8px\)/));
  check('M4b 弹出层同步上移（不与按钮重叠）', css(/html\.mob #counter-pop\s*\{[^}]*bottom:\s*calc\(100% \+ 42px\)/));
  check('M4c 导航窄屏改单行（不换行挤压）',
    css(/html\.mob #nav\s*\{[^}]*flex-wrap:\s*wrap/)
      && css(/html\.mob \.navcol\s*\{[^}]*flex-direction:\s*row/));
  check('M4d 窄屏文字按钮图标化（省横向空间）',
    css(/html\.mob #nav button:not\(\.icon\)::before/) && css(/html\.mob #round-chip\s*\{[^}]*display:\s*none/));

  /* ============ 5. N1 追加鸡牌（2026-10-12 三次改造：全牌宫格直选） ============ */
  console.log('\n=== 5. N1 追加鸡牌 ===');
  check('N1a 开局页 + 设置页各有鸡牌入口',
    !!el('boot-extra-ji') && !!el('set-extra-ji'));
  const jiEntry = el('boot-extra-ji');
  check('N1b 入口 = 幺鸡（锁定）+ 已选缩略图 + 「+」框（浮窗里才铺全部牌）',
    !!jiEntry && !!jiEntry.querySelector('.jiadd')
      && !!jiEntry.querySelector('.jithumbs')
      && !!jiEntry.querySelector('.jithumb.lock')
      && !jiEntry.querySelector('button[data-t]')
      && el('ji-mask') && html.indexOf('class="jipick"') === -1);
  check('N1c 浮窗 = 全部牌宫格（#ji-grid，旧翻牌组件已删）',
    !!el('ji-grid') && !el('ji-left') && !el('ji-right') && !el('ji-cur')
      && !el('ji-prev') && !el('ji-next') && !el('ji-strip'));

  const prevJi = JSON.parse(JSON.stringify(G('SET.extraJi')));
  // 点「+」框 → 弹浮窗
  let jiOpened = false;
  if (jiEntry && jiEntry.querySelector('.jiadd')) {
    jiEntry.querySelector('.jiadd').dispatchEvent(new w.MouseEvent('click', { bubbles: true }));
    jiOpened = el('ji-mask').classList.contains('on');
  }
  check('N1d 点「+」框 → 弹出浮窗，27 张牌全部铺出',
    jiOpened && el('ji-grid').querySelectorAll('button[data-t]').length === 27,
    `${el('ji-grid').querySelectorAll('button[data-t]').length} 张`);
  if (jiOpened) {
    const g9 = el('ji-grid').querySelector('button[data-t="9"]');
    const g0 = el('ji-grid').querySelector('button[data-t="0"]');
    check('N1e 幺鸡锁定（永远是鸡）且浮窗副标列出已选牌种',
      g9.classList.contains('locked') && (el('ji-sub').textContent || '').indexOf('幺鸡永远是鸡') !== -1,
      g9.className);
    g0.dispatchEvent(new w.MouseEvent('click', { bubbles: true }));
    check('N1f 点一筒 → 加入本局鸡牌（SET.extraJi）',
      Array.isArray(G('SET.extraJi')) && G('SET.extraJi').indexOf(0) !== -1,
      `SET.extraJi=${JSON.stringify(G('SET.extraJi'))}`);
    check('N1g 选中态点亮（宫格 .on）并重绘',
      el('ji-grid').querySelector('button[data-t="0"]').classList.contains('on'));
    check('N1h 落盘 localStorage.dushan_extra_ji',
      (w.localStorage.getItem('dushan_extra_ji') || '').indexOf('0') !== -1,
      w.localStorage.getItem('dushan_extra_ji'));
    check('N1i 入口缩略图同步（已选牌种回显）',
      jiEntry && jiEntry.querySelectorAll('.jithumb').length > 0);
    // 再点一次取消
    el('ji-grid').querySelector('button[data-t="0"]').dispatchEvent(new w.MouseEvent('click', { bubbles: true }));
    check('N1j 再点一次取消选中',
      G('SET.extraJi').indexOf(0) === -1 &&
      !el('ji-grid').querySelector('button[data-t="0"]').classList.contains('on'));
    // 幺鸡不可取消
    const before9 = JSON.stringify(G('SET.extraJi'));
    g9.dispatchEvent(new w.MouseEvent('click', { bubbles: true }));
    check('N1k 幺鸡点不动（永远是鸡，不进 extraJi）',
      JSON.stringify(G('SET.extraJi')) === before9 && G('SET.extraJi').indexOf(9) === -1);
    // 重开浮窗：已选牌种全部点亮
    w.eval('SET.extraJi = [0, 19]; saveSet(); jiRender();');
    check('N1l 重开时已选牌种全部点亮（含幺鸡）',
      el('ji-grid').querySelectorAll('button.on').length === 3,
      `${el('ji-grid').querySelectorAll('button.on').length} 个`);
    el('ji-close').dispatchEvent(new w.MouseEvent('click', { bubbles: true }));
    check('N1m 「确定」关闭浮窗', !el('ji-mask').classList.contains('on'));
  }
  check('N1n 开局请求把 extra_ji 传给服务端（rules.extra_ji = SET.extraJi）',
    html.indexOf('extra_ji: SET.extraJi') !== -1 || html.indexOf('extra_ji: SET.extraJi.slice()') !== -1);
  check('N1o 设置项标了「下一局生效」', html.indexOf('下一局生效') !== -1);
  // 复位
  w.eval('SET.extraJi = ' + JSON.stringify(prevJi) + '; saveSet(); renderExtraJi("boot-extra-ji");');

  /* ============ 5b. N3 玩家模型（设置-玩家模型；模型下拉并入玩家行） ============ */
  console.log('\n=== 5b. N3 玩家模型（模型下拉并入玩家行） ===');
  const sp = el('set-players');
  const pmSel = () => [...sp.querySelectorAll('select.pickmodel')];
  check('N3a 玩家行内 4 个模型下拉（我 + 三家对手，不再单列「对手模型」板块）',
    pmSel().length === 4 && !el('set-models'),
    `${pmSel().length} 个`);
  check('N3b 下拉就在玩家行里（名字后面的空白区），同一 who 页面',
    pmSel().length === 4 && pmSel().every(s => !!s.closest('.pickrow')));
  const prevOpp = JSON.parse(JSON.stringify(G('SET.oppModels')));
  const prevStrat = G('SET.strategy');
  const s1 = sp.querySelector('select.pickmodel[data-pickrel="1"]');
  if (s1) {
    const opt = [...s1.options].find(o => o.value.startsWith('teacher:'));
    const v = opt ? opt.value : 'random';
    s1.value = v;
    s1.dispatchEvent(new w.Event('change', { bubbles: true }));
    check('N3c 改下家模型 → 写入 SET.oppModels 并落盘',
      G('SET.oppModels')[0] === v
        && (w.localStorage.getItem('dushan_opp_models') || '').indexOf(v) !== -1,
      `SET.oppModels=${JSON.stringify(G('SET.oppModels'))}`);
    check('N3d 开局页「玩家与模型」卡同步',
      (() => {
        const s = el('boot-players') &&
          el('boot-players').querySelector('select.pickmodel[data-pickrel="1"]');
        return !!s && s.value === v;
      })(),
      (el('boot-players') &&
        el('boot-players').querySelector('select.pickmodel[data-pickrel="1"]') || {}).value);
    check('N3e 名字里的模型版本缩写随模型变化（头像入口重绘）',
      html.indexOf('seatOppAbbr') !== -1 && html.indexOf('SET.oppModels[rel - 1]') !== -1);
    // 开局用 SET.oppModels 组 spec（不再各自读 DOM）
    check('N3f 开局用 SET.oppModels 生成 opponents', html.indexOf('SET.oppModels.map(v =>') !== -1);
  }
  const s0 = sp.querySelector('select.pickmodel[data-pickrel="0"]');
  check('N3g 「我」也有模型下拉，值 = 大师推荐策略（同一真源）',
    !!s0 && s0.value === G('SET.strategy'), `s0=${s0 && s0.value} strategy=${G('SET.strategy')}`);
  if (s0) {
    const opt = [...s0.options].find(o => o.value.startsWith('model:') && o.value !== prevStrat);
    const v = opt ? opt.value : 'random';
    s0.value = v;
    s0.dispatchEvent(new w.Event('change', { bubbles: true }));
    check('N3h 改「我」的模型 → SET.strategy 同步 + 推荐策略下拉跟随',
      G('SET.strategy') === v && el('set-strategy').value === v,
      `strategy=${G('SET.strategy')} set-strategy=${el('set-strategy').value}`);
  }
  w.eval('SET.oppModels = ' + JSON.stringify(prevOpp) + '; SET.strategy = ' +
    JSON.stringify(prevStrat) + '; saveSet(); fillOppSelects(); renderPickers("set-players");');

  /* ============ 6. N2 冲锋鸡/横鸡特效 ============ */
  console.log('\n=== 6. N2 鸡牌特效 ===');
  check('N2a 全屏特效容器 #ji-fx 存在', !!el('ji-fx'));
  check('N2b jiFx 是函数', typeof w.jiFx === 'function');
  // 牌表就绪后触发冲锋鸡
  if (typeof w.jiFx === 'function' && w.eval('TILES && TILES.length')) {
    // 2026-10-13：三种全屏特效（冲锋鸡/横鸡、杠、开局鸡牌）改用互斥队列 fxSchedule，
    // 同一时刻只播一个、其余排队 —— 所以每个用例前先 fxReset() 清锁与队列。
    w.eval('fxReset()');
    w.jiFx(8, '冲锋鸡');
    const on = el('ji-fx').classList.contains('on');
    const word = el('ji-fx').querySelector('.fx-word');
    const sparks = el('ji-fx').querySelectorAll('.spark').length;
    check('N2c 冲锋鸡：特效层开启 + 大字招牌 + 火花迸射',
      on && !!word && word.textContent.indexOf('冲锋鸡') !== -1 && sparks > 0,
      `on=${on} word=${word && word.textContent} sparks=${sparks}`);
    check('N2d 冲锋鸡不带横鸡色（未加 hj 类）', !el('ji-fx').classList.contains('hj'));
    w.eval('fxReset()');
    w.jiFx(8, '横鸡');
    check('N2e 横鸡：加 hj 类（蓝色调）', el('ji-fx').classList.contains('hj')
      && (el('ji-fx').querySelector('.fx-word') || {}).textContent.indexOf('横鸡') !== -1);
    w.eval('fxReset()');
    check('N2f 特效时长可设置（#84：--fx-dur 由 SET.fxDur 驱动，默认 3s）',
      html.indexOf('--fx-dur') !== -1 && html.indexOf('fx_dur') !== -1);
    // 2026-10-13 新增：杠牌特效 + 特效互斥队列（开局鸡牌特效等前面的播完再开始）
    check('N2g 有独立的杠牌特效层 #gang-fx（4 张牌面 + 「杠！」大字）',
      !!el('gang-fx') && html.indexOf('#gang-fx .gf-word') !== -1 &&
      html.indexOf('#gang-fx .gf-stage img:nth-child(4)') !== -1);
    w.jiFx(8, '冲锋鸡');
    check('N2h 特效互斥：前一个还在播时，后一个进队列不抢帧', (() => {
      const before = el('gang-fx').innerHTML;
      w.gangFx(8, '暗杠');
      return el('gang-fx').innerHTML === before && w.eval('FX_QUEUE.length') === 1;
    })(), `queue=${w.eval('FX_QUEUE.length')}`);
    w.gangFx(8, '暗杠');
    w.eval('FX_LOCK = null; fxDrain()');     // 手动放行队列
    check('N2i 队列放行后杠牌特效应渲染出 4 张牌',
      el('gang-fx').querySelectorAll('.gf-stage img').length === 4,
      `imgs=${el('gang-fx').querySelectorAll('.gf-stage img').length}`);
    w.eval('fxReset()');
  } else {
    check('N2c 牌表未就绪，跳过特效渲染断言', false, 'TILES 为空');
  }
  check('N2g 普通鸡/非鸡不触发特效（调用点带条件过滤）',
    html.indexOf('if (lastShown > 0) jiFx(e.tile, e.ctag)') !== -1
      && html.indexOf('e.ctag === "冲锋鸡"') !== -1 && html.indexOf('e.ctag === "横鸡"') !== -1);

  /* ============ 7. 流水结算行 + 对局记录精简 ============ */
  console.log('\n=== 7. 流水结算行 / 对局记录 ===');
  check('R1 结算行单独渲染（不加「你/下家」席位前缀、不吃掉首词）',
    html.indexOf('if (e.kind === "result")') !== -1
      && html.indexOf('结算</b>') !== -1
      && html.indexOf('e.text || ""}</div>') !== -1);
  check('R2 结算行不加席位前缀，普通行仍用相对称谓（你/下家）',
    html.indexOf('结算</b>') !== -1 && html.indexOf('${logSeatName(e.rel)}') !== -1);
  check('R3 对局记录行只留「时间 · 赢家 胡 类型」（不再拼放炮/听牌/自摸）',
    html.indexOf('胡 ${rec.fan_cn') !== -1
      && html.indexOf('function histLabel(rec)') !== -1
      && html.indexOf('tingTxt') === -1);

  /* ============ 8. 页面零脚本错误 ============ */
  console.log('\n=== 7. 健康检查 ===');
  check('全程无 window 脚本错误', errs.length === 0, errs.slice(0, 2).join(' | '));

  console.log(`\n合计：${pass} PASS / ${fail} FAIL`);
  process.exit(fail ? 1 : 0);
}, 320);
