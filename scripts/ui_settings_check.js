// 设置面板 / 玩家名字头像 / 结束印记几何 e2e（jsdom 加载真实页面）
const fs = require('fs');
const { JSDOM } = require('jsdom');

const ROOT = 'C:/Users/LIGF/WorkBuddy/2026-09-30-12-49-43/zhuoji-mahjong';
const html = fs.readFileSync(ROOT + '/web/static/dushan.html', 'utf8');
const errs = [];
const meta = JSON.parse(fs.readFileSync(ROOT + '/reports/ui_fixtures_meta.json', 'utf8'));
const mid0 = JSON.parse(fs.readFileSync(ROOT + '/reports/ui_fixtures_mid.json', 'utf8'))[0].state;

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
const results = [];
const check = (name, ok, extra = '') => {
  results.push(ok);
  console.log(`${name}: ${ok ? 'OK' : 'FAIL'}${extra ? ' | ' + extra : ''}`);
};

setTimeout(() => {
  const el = id => w.document.getElementById(id);

  // ---- S1：默认口径（与引擎 DushanConfig 一致） ----
  const r = w.eval('JSON.stringify(SET.rule)');
  const rule = JSON.parse(r);
  check('S1 默认包赔口径', rule.end_baoji === true && rule.end_baogang === false
    && rule.huang_baoji === false && rule.huang_baogang === false && rule.huang_baodapai === true, r);

  // ---- S2：设置面板控件齐备 ----
  const ids = ['set-end-baoji', 'set-end-baogang', 'set-huang-baoji', 'set-huang-baogang',
    'set-huang-baodapai', 'set-strategy', 'set-timeout', 'set-hintdelay', 'set-reasonmode',
    'set-oppspeed', 'set-hinton', 'set-timeouton', 'set-rec-form',
    'grp-hint', 'grp-timeout', 'grp-speed',
    'set-jiesuan-fanji', 'set-kaiju-fanji', 'set-mantiangji', 'set-nav', 'set-panes',
    'boot-extra-ji', 'set-extra-ji',   // N1 追加鸡牌选择条（2026-10-12）
    'set-players', 'boot-players', 'set-logshow', 'set-tipon',
    'set-endhand',                     // #86：结束本局迁入设置面板
    'set-fxdur',                       // #84：冲锋鸡/横鸡动画时长
    'set-counteron',                   // 2026-10-12 晚：记牌器功能开关（默认关）
    'counter-btn', 'sc-zimo'];
  const missing = ids.filter(i => !el(i));
  check('S2 设置控件齐备', missing.length === 0, missing.length ? '缺: ' + missing.join(',') : `${ids.length} 项`);

  // ---- S3：头像选择器（虚框入口 + 浮窗，不再平铺全部候选） ----
  // 2026-10-12 二次改版：入口 = 缩略图 + 头像名，点虚框任意处直接开浮窗（按钮删除）。
  const picks = el('set-players').querySelectorAll('.pickent');
  const poolN = (w.eval('AVATAR_POOL.length'));
  check('S3 4 家各一个虚框入口（不再平铺全部候选）', picks.length === 4, `入口 ${picks.length} 家`);
  check('S3a 入口 = 缩略图 + 头像名（「选择头像」按钮已删）',
    picks.length === 4 && [...picks].every(p => p.querySelector('img.pickthumb')
      && p.querySelector('.pickname') && !p.querySelector('.pickbtn')), '');
  check('S3b 未打开浮窗时不渲染候选网格', el('ava-grid').children.length === 0,
    `${el('ava-grid').children.length} 个`);
  check('S3c 不再有「随机」头像下拉', !el('sp-ava-0') && !el('pl-ava-0'), '');
  // 点虚框入口 → 浮窗列出全部候选（数量 = 头像池）
  picks[0].click();
  const openedN = el('ava-grid').querySelectorAll('.pick').length;
  check('S3d 浮窗打开并列出全部候选',
    el('ava-mask').classList.contains('on') && openedN === poolN,
    `mask=${el('ava-mask').classList.contains('on')} 候选=${openedN}/${poolN}`);
  w.eval('closeAvaPicker()');
  check('S3e 浮窗可关闭', !el('ava-mask').classList.contains('on'), '');

  // ---- S4：改设置并保存 → SET + localStorage 同步 ----
  w.eval('fillSetSelects()');
  el('set-end-baogang').value = '1';
  el('set-huang-baoji').value = '1';
  // 头像选择器：走浮窗（入口 → 候选网格）。点选下家的「金蝉」、我的「云天明」→ 即时保存
  const pickAva = (rel, nm) => {
    el('set-players').querySelector(`.pickent[data-rel="${rel}"]`).click();
    const b = [...el('ava-grid').querySelectorAll('.pick')].find(x => x.dataset.ava === nm);
    if (!b) throw new Error(`候选里没有「${nm}」`);
    b.click();   // setAvatar → 保存 + 双处重绘 + 自动关浮窗
  };
  pickAva(1, '金蝉');
  pickAva(0, '云天明');
  check('S4a 选完头像浮窗自动关闭', !el('ava-mask').classList.contains('on'), '');
  el('btn-set-save').click(); el('btn-set-close').click();
  const saved = JSON.parse(w.localStorage.getItem('dushan_rules') || '{}');
  const savedP = JSON.parse(w.localStorage.getItem('dushan_players') || '[]');
  check('S4 设置保存到本地', saved.end_baogang === true && saved.huang_baoji === true
    && saved.end_baoji === true && saved.huang_baodapai === true,
    JSON.stringify(saved));
  check('S4b 头像选择即时保存（名字随头像派生）',
    savedP[1] && savedP[1].avatar === '金蝉' && savedP[1].name === ''
    && savedP[0].avatar === '云天明' && savedP[0].name === '', JSON.stringify(savedP));

  // ---- S5：渲染使用头像派生名（我 = 头像名；对手 = 头像名(模型缩写)） ----
  w.eval('SID = "e2e-settings"');
  const st = JSON.parse(JSON.stringify(mid0));
  st.turn = -1;
  w.render(st);
  const abbr1 = w.eval('seatOppAbbr(1)');
  const seat1 = el('p1').innerHTML;
  check('S5 对手面板用派生名（头像名+模型缩写）',
    seat1.includes(`金蝉(${abbr1})`), `${seat1.slice(0, 90)} want 金蝉(${abbr1})`);
  check('S5b 对手面板用所选头像', seat1.includes('/assets/avatar/boy/金蝉.jpg'), '');
  check('S5c 我名字生效（不再是「你」）', el('me-name').textContent === '云天明',
    el('me-name').textContent);
  check('S5d 我的头像生效', el('me-avatar').getAttribute('src').includes('/云天明.jpg'),
    el('me-avatar').getAttribute('src'));

  // ---- S6：结束印记几何（不再遮对家弃牌区 / 不压计时盘） ----
  const stampW = +(html.match(/#end-stamp \{[^}]*?width:\s*(\d+)px/)[1]);
  const stampB = +(html.match(/#end-stamp \{[^}]*?bottom:\s*(\d+)px/)[1]);
  const png = fs.readFileSync(ROOT + '/web/static/assets/sprites/btn_end.png');
  const imgH = png.readUInt32BE(20), imgW = png.readUInt32BE(16);
  const stampH = stampW * imgH / imgW;
  const stampTop = 600 - stampB - stampH;      // #river 局部坐标（600×600）
  const stampBottom = 600 - stampB;
  const TILE_TOP = 150;                        // 对家弃牌区内缘（river-top: bottom 450 + 高 141）
  const TIMER_TOP = 234;                       // 计时盘上缘（300 居中区内的 132 圆盘）
  check('S6 结束印记不遮对家弃牌区', stampTop > TILE_TOP,
    `印记 y${stampTop.toFixed(0)}~${stampBottom.toFixed(0)} vs 弃牌区 ≤${TILE_TOP}`);
  check('S6b 结束印记不压计时盘', stampBottom < TIMER_TOP,
    `印记下缘 ${stampBottom.toFixed(0)} < 计时盘上缘 ${TIMER_TOP}`);

  // ---- S7：导航双列（2026-10-12 #86 起：左＝统计/记录/局数，右＝设置/透视/声音 纯图标；
  //           「结束本局」从桌面导航移进设置 → 界面显示（#set-endhand））----
  const navIds = [...el('nav').querySelectorAll('button')].map(b => b.id);
  check('S7 导航顺序：统计/记录 → 设置/透视/声音（结束本局已迁出导航）',
    navIds.join(',') === 'btn-stats,btn-history,btn-settings,btn-peek,btn-sound',
    navIds.join(','));
  check('S7a 「⋯」折叠层已取消（btn-navmore/nav-icons 不复存在）',
    !el('btn-navmore') && !el('nav-icons') && !el('nav-more'), '');
  check('S7b 流水图标开关已移入设置（牌桌上不再有 📄 按钮）', !el('btn-log'), '');
  // #86：桌面不再有结束本局按钮；设置面板「界面显示」里就位（红边按钮）
  // 2026-10-12 晚：图标由字符 ⏻ 换成**内联 SVG**（用户口径「只有框没有 icon」——
  // 实际是把字体字形换成矢量图标，避免字体缺字时显示成方块/空框）
  check('S7c 结束本局已迁入设置面板（#set-endhand，桌面无 btn-endhand；内联 SVG 图标）',
    !el('btn-endhand') && !!el('set-endhand') &&
    !!el('set-endhand').querySelector('svg') &&
    el('set-endhand').textContent.trim() === '',
    (el('set-endhand') || {}).outerHTML || 'missing');
  check('S7c′ 内联 SVG 图标尺寸有 CSS 兜底（button.icononly svg）',
    /button\.icononly svg \{[^}]*width: 16px; height: 16px/.test(html));

  // ---- S8：统计/对局记录无关闭按钮，点击面板外关闭 ----
  check('S8 已移除关闭按钮', !el('btn-stats-close') && !el('btn-hist-close'), '');
  el('btn-stats').click();
  const opened = el('stats-mask').classList.contains('on');
  el('stats-mask').dispatchEvent(new w.MouseEvent('click', { bubbles: true }));
  const closedOut = !el('stats-mask').classList.contains('on');
  el('btn-history').click();
  el('hist-mask').querySelector('.modal').dispatchEvent(new w.MouseEvent('click', { bubbles: true }));
  const keepOpen = el('hist-mask').classList.contains('on');
  el('hist-mask').dispatchEvent(new w.MouseEvent('click', { bubbles: true }));
  check('S8b 点面板外关闭、点面板内不关闭', opened && closedOut && keepOpen,
    `开=${opened} 外点关=${closedOut} 内点保留=${keepOpen}`);

  // ---- S9：大师建议面板可拖动并记忆位置 ----
  el('coach').classList.add('on');
  const hd = el('coach-hd');
  const md = (type, x, y) => hd.dispatchEvent(new w.MouseEvent(type, { bubbles: true, clientX: x, clientY: y }));
  md('pointerdown', 100, 100);
  md('pointermove', 140, 130);
  const movedLeft = el('coach').style.left, movedTop = el('coach').style.top;
  md('pointerup', 140, 130);
  const stored = JSON.parse(w.localStorage.getItem('dushan_coach_pos_v2') || 'null');
  check('S9 大师建议面板可拖动', movedLeft === '40px' && movedTop === '30px'
    && !el('coach').classList.contains('dragging'), `left=${movedLeft} top=${movedTop}`);
  check('S9b 拖动位置已记忆', stored && Number.isFinite(stored.x) && Number.isFinite(stored.y),
    JSON.stringify(stored));
  check('S9c 面板有拖动提示', hd.innerHTML.includes('拖动'), hd.textContent.trim());

  // ---- S10：碰鸡横置指示来源家（上家最左 / 对家居中 / 下家最右） ----
  const mj = (src, ji) => w.eval(`meldCells({type:1, tile:9, src:${src}, ji:${JSON.stringify(ji)}}, "", 0)`);
  const hengIdx = (h) => (h.match(/class="mc[^"]*heng/g) || []).length === 1
    ? [...h.matchAll(/<span class="mc[^"]*"/g)].findIndex(m => m[0].includes('heng')) : -1;
  const up = mj(3, '幺鸡'), opp = mj(2, '幺鸡'), dn = mj(1, '幺鸡');
  check('S10 碰鸡横置指示来源家',
    hengIdx(up) === 0 && hengIdx(opp) === 1 && hengIdx(dn) === 2,
    `上家=${hengIdx(up)} 对家=${hengIdx(opp)} 下家=${hengIdx(dn)}（应 0/1/2）`);
  check('S10b 幺鸡用原牌面、冲锋鸡/横鸡带统一角标（冲/横）',
    up.includes('1条.png') && mj(3, '冲锋鸡').includes('jitag chf') && mj(3, '冲锋鸡').includes('data-t="冲"')
    && mj(3, '横鸡').includes('jitag hj') && mj(3, '横鸡').includes('data-t="横"'), '');
  check('S10c 普通碰保持三张竖排', hengIdx(mj(3, null)) === -1, mj(3, null));

  // ---- S11：名字与头像绑定（我 = 头像名；对手 = 头像名(策略缩写)） ----
  w.eval('SET.players = AVA_DEFAULT.map(a => ({ name: "", avatar: a })); AVA_SESSION = null;');
  /* 2026-10-12：对手模型改由「玩家身份」页的模型选择器驱动，SET.oppModels 是唯一真源，
     #opp0..2 只是它的镜像（onchange → setOppModel）。
     所以这里必须走真实接线（改 value + change）；直接赋 .value 不再影响派生名。 */
  const oppBackup = w.eval('JSON.stringify(SET.oppModels)');
  /* 2026-10-12 五改版：开局页 #opp0..2 已删除（模型下拉并入「玩家与模型」卡）→
     这里直接写真源 SET.oppModels，再重绘两处选择器验证派生名。 */
  w.eval('SET.oppModels = ["teacher:tenpai_rush", "model:bc", "model:dushan_s7"];');
  w.eval('renderPickers("boot-players"); renderPickers("set-players")');
  const defs = JSON.parse(w.eval('JSON.stringify([0,1,2,3].map(defaultPName))'));
  // 2026-10-12 二次改版：pickhead 头部行删除，派生名在 .pickname 里（含 <i>座位</i> 前缀）
  const WHO = JSON.parse(w.eval('JSON.stringify(PICK_WHO)'));
  const curs = [0, 1, 2, 3].map(r => {
    const n = el('set-players').querySelector(`.pickrow[data-rel="${r}"] .pickname`);
    return n ? n.textContent.replace(WHO[r], "").trim() : "(缺)";
  });
  check('S11 名字=头像名(策略缩写)，我不再叫「你」',
    defs[0] === '云天明' && defs[1] === '大乔(抢听)' && defs[2] === '妲己(SBC)' && defs[3] === '王昭君(S7)',
    defs.join(' | '));
  check('S11b 选择器头部显示派生名', curs.join('|') === defs.join('|'), curs.join(' | '));
  // 大师缩写随冠军权重走（meta.master_info.abbr），无 meta 时回退 M
  const mAbbr = (meta.master_info && meta.master_info.abbr) || 'M';
  check('S11c 随机/大师/大网缩写', JSON.parse(w.eval(
    'JSON.stringify([oppAbbr("random"), oppAbbr("model:master"), oppAbbr("model:dushan_big_bc"), oppAbbr("teacher:gambler")])'
  )).join(',') === `随机,${mAbbr},BBC,赌徒`, `大师缩写=${mAbbr}`);

  // ---- S11d/S11e：大师选项文案 = 大师 + 版本号（规模），且不写「最强」 ----
  // 2026-10-12：开局页模型下拉已并入「玩家与模型」卡 → 改从那里取样
  const mLabel = "🤖 " + ((meta.master_info && meta.master_info.label) || '大师');
  const bootSel = el('boot-players') && el('boot-players').querySelector('select.pickmodel');
  check('S11d 大师项版本号与规模来自权重', !!bootSel && bootSel.options[0].textContent === mLabel
    && !/最强/.test(bootSel.options[0].textContent),
    bootSel ? bootSel.options[0].textContent : '(缺下拉)');
  w.eval('SET.oppModels = ' + oppBackup + ';');   // 还原真源（S11 改过它，且 saveSet 落过盘）
  w.eval('fillOppSelects(); renderPickers("boot-players");');  // 恢复默认值后再检查
  check('S11e 「玩家与模型」卡里三家对手默认都用最强模型',
    [...el('boot-players').querySelectorAll('select.pickmodel[data-pickrel]')]
      .filter(s => s.getAttribute('data-pickrel') !== '0')
      .every(s => s.value === 'model:master'),
    [...el('boot-players').querySelectorAll('select.pickmodel[data-pickrel]')]
      .map(s => s.getAttribute('data-pickrel') + '=' + s.value).join(','));

  // ---- S12：翻鸡设置项（结算翻鸡 4 选 1 含关闭；开局翻鸡 4 选 1 且默认关闭） ----
  const jOpts = [...el('set-jiesuan-fanji').options].map(o => o.value);
  const kOpts = [...el('set-kaiju-fanji').options].map(o => o.value);
  check('S12 结算翻鸡=上下鸡/上鸡/下鸡/关闭', jOpts.join(',') === 'both,up,down,off', jOpts.join(','));
  check('S12b 开局翻鸡=关闭/上鸡/下鸡/上下鸡', kOpts.join(',') === 'off,up,down,both', kOpts.join(','));
  check('S12c 默认口径 结算=both 开局=off',
    JSON.parse(w.eval('JSON.stringify([SET.jiesuan, SET.kaijuFanji])')).join(',') === 'both,off',
    w.eval('SET.jiesuan + "/" + SET.kaijuFanji'));
  // 旧「上下鸡」设置项已重命名，不再存在；板块改名「翻鸡规则」
  check('S12d 旧「上下鸡」开关已移除', !el('set-shangxia') && !el('set-kaiju'), '');
  check('S12e 规则板块 h4 改名「翻鸡规则」',
    el('set-panes').querySelector('section[data-pane="rule"] h4.setsec').textContent.includes('翻鸡规则'), '');

  // ---- S12f：音效与音量「逐列对齐」（2026-10-13 用户口径：多行开关对齐、滑块也对齐）----
  // 行 = 4 列网格（项目名固定宽 / 开关 / 音量组 / 曲目下拉），音量组内
  // 「音量」标签固定宽 + 数值框固定宽 → 各行开关与滑块的起止点落在同一竖线。
  check('S12f 音效行网格化：项目名列固定宽（各行开关左边界一致）',
    /\.sndrow \{ display: grid; grid-template-columns: var\(--sndlbl-w\) auto minmax\(0, 1fr\) auto;/.test(html) &&
    /\.sndrow > \.lbl \{ grid-column: 1;/.test(html) &&
    /\.sndrow > \.switch \{ grid-column: 2;/.test(html) &&
    /\.sndrow > \.volrow \{ grid-column: 3;/.test(html) &&
    /\.sndgrid \{[^}]*--sndlbl-w: 9\.8em;/.test(html), '');
  check('S12f2 音量组内对齐：音量标签固定宽 + 滑杆/数值框固定宽',
    /\.volrow \.vlb \{[^}]*min-width: 3\.4em;/.test(html) &&
    /\.volrow input\[type="range"\] \{ width: 138px;/.test(html) &&
    /\.volrow \.volval \{[^}]*min-width: 40px;/.test(html), '');
  check('S12f3 移动端降级两列（名字+开关一行、音量组另起一行，同样逐列对齐）',
    /html\.mob \.sndrow \{ grid-template-columns: var\(--sndlbl-w\) minmax\(0, 1fr\);/.test(html) &&
    /html\.mob \.sndrow > \.volrow, html\.mob \.sndrow > select \{ grid-column: 2; \}/.test(html), '');

  // ---- S13：设置页左右两栏（左侧分类 → 右侧面板） ----
  // 2026-10-12 三次合并：玩家身份 + 对手模型 + 大师推荐 →「玩家与模型」；
  // 出牌节奏 + 界面显示 →「出牌与显示」。
  // 2026-10-13 新增「声音」分类（音量滑块 / 背景音乐 / 读牌 / 操作音效）。
  // 5 类：玩家与模型/出牌与显示/声音/规则开关/分值表
  const navBtns = [...el('set-nav').querySelectorAll('button')];
  const panes = [...el('set-panes').querySelectorAll('section')];
  check('S13 左栏分类数 = 右侧面板数', navBtns.length === 5 && panes.length === 5,
    `${navBtns.length} / ${panes.length}`);
  check('S13b 分类与面板一一对应',
    navBtns.every((b, i) => b.dataset.pane === panes[i].dataset.pane), '');
  check('S13a 「大师推荐」并入「玩家模型」分类（含 grp-hint）',
    (() => {
      const wp = el('set-panes').querySelector('section[data-pane="who"]');
      return !!wp && !!wp.querySelector('#grp-hint') && !!wp.querySelector('#set-rec-form')
        && !el('set-panes').querySelector('section[data-pane="play"] #grp-hint');
    })(), '');
  const activeCount = () => panes.filter(s => s.classList.contains('on')).length;
  const ruleIdx = navBtns.findIndex(b => b.dataset.pane === 'rule');
  const before = activeCount();
  navBtns[ruleIdx].click();
  const onIdx = panes.findIndex(s => s.classList.contains('on'));
  check('S13c 点左侧栏切换右侧面板', before === 1 && onIdx === ruleIdx && activeCount() === 1,
    `切换前 ${before} 个显示 → 第 ${onIdx + 1} 个`);
  check('S13d 分类切换会记忆', w.localStorage.getItem('dushan_setpane') === 'rule',
    String(w.localStorage.getItem('dushan_setpane')));
  check('S13e 切换分类走高度过渡（CSS 过渡 + JS 量高赋值）',
    /\.setpanes \{[^}]*transition: height \.24s/.test(html) &&
    /\.setpanes\.animating \{ overflow: hidden; \}/.test(html) &&
    /panes\.style\.height = to \+ "px"/.test(html), 'jsdom 无排版，只能静态校验');
  navBtns[0].click();

  // ---- S14：大师推荐 / 大师建议 合并（推荐理由 = 完整推荐 / 一句话推荐） ----
  //  显隐时机本轮有变：面板不再是常驻，只在「轮到我的决策点」显示
  //  （打出牌后立刻收起）。这里先校验派生开关，时机另见 scripts/ui_aux_check.js。
  const coachOn = () => el('coach').classList.contains('on');
  w.eval('SET.hintOn = true; SET.reasonMode = "full"; syncCoachPanel();');
  check('S14 完整推荐 → 派生开关 SET.coach = true',
    w.eval('SET.coach') === true, `coach=${w.eval('SET.coach')}`);
  check('S14f 非我的决策点时面板保持收起（不再常驻）',
    !w.eval('coachTurnOn()') && !coachOn(), `turnOn=${w.eval('coachTurnOn()')}`);
  w.eval('SET.reasonMode = "brief"; syncCoachPanel();');
  check('S14b 一句话推荐 → 面板隐藏、SET.coach 派生为 false',
    !coachOn() && w.eval('SET.coach') === false, `coach=${w.eval('SET.coach')}`);
  w.eval('SET.hintOn = false; SET.reasonMode = "full"; syncCoachPanel();');
  check('S14c 推荐总开关关闭 → 面板也不显示',
    !coachOn() && w.eval('SET.coach') === false, `coach=${w.eval('SET.coach')}`);
  w.eval('SET.hintOn = true; SET.reasonMode = "full"; syncCoachPanel();');
  el('btn-settings').click();
  el('set-reasonmode').value = 'brief';
  el('btn-set-save').click(); el('btn-set-close').click();
  check('S14d 设置面板可选「一句话推荐」并持久化',
    w.eval('SET.reasonMode') === 'brief'
    && w.localStorage.getItem('dushan_reasonmode') === 'brief', String(w.eval('SET.reasonMode')));
  el('btn-settings').click();
  el('set-reasonmode').value = 'full';
  el('btn-set-save').click(); el('btn-set-close').click();
  check('S14e 改回「完整推荐」派生开关恢复 true', w.eval('SET.coach') === true,
    String(w.eval('SET.coach')));

  // ---- S15：流水开关移入设置（界面显示），双态联动 ----
  const logHidden = () => el('log').classList.contains('hidden');
  el('btn-settings').click();
  el('set-logshow').checked = true;
  el('btn-set-save').click(); el('btn-set-close').click();
  const openState = !logHidden();
  el('btn-settings').click();
  el('set-logshow').checked = false;
  el('btn-set-save').click(); el('btn-set-close').click();
  check('S15 流水开关（设置→界面显示）双态生效', openState && logHidden(),
    `开=${openState} 关=${!logHidden()}`);
  check('S15b 开关状态持久化', w.localStorage.getItem('dushan_logshow') === '0'
    && w.eval('SET.logShow') === false, String(w.eval('SET.logShow')));

  // ---- S16：结算页顶部（小字移除；本局鸡牌条上移到图片下方） ----
  check('S16 顶部图片下小字已移除', !el('res-sub'), '');
  const jibar = el('res-jibar');
  const tabsIdx = [...el('result-mask').querySelector('.modal').children].indexOf(el('res-tabs'));
  const jiIdx = [...el('result-mask').querySelector('.modal').children].indexOf(jibar);
  check('S16b 本局鸡牌条在视角切换之前', !!jibar && jiIdx >= 0 && jiIdx < tabsIdx,
    `jibar#${jiIdx} < tabs#${tabsIdx}`);
  // 内容：本局鸡牌 + 翻鸡策略（按模式显示）
  const ovRes = Object.assign({}, mid0.result, { jiesuan_fanji: 'down', kaiju_fanji: 'both' });
  const jh = w.eval(`resJibarHTML(${JSON.stringify({ ji_tiles: [9, 10], result: ovRes })})`);
  check('S16c 信息条含本局鸡牌与翻鸡策略',
    jh.includes('本局鸡牌') && jh.includes('仅下鸡') && jh.includes('开局翻鸡'), jh.slice(0, 160));
  check('S16d 总览不再重复渲染信息条',
    !w.eval(`(function(){const st=JSON.parse(JSON.stringify(${JSON.stringify(mid0)}));st.turn=-1;render(st);return el("res-pairs").innerHTML.includes("res-jibar")})()`), '');

  // ---- S17：指示器颜色支持「关闭」 ----
  const indOpts = [...el('set-ind-color').options].map(o => o.value);
  const recOpts = [...el('set-rec-color').options].map(o => o.value);
  check('S17 指示器颜色含关闭选项', indOpts.join(',') === 'red,yellow,green,off'
    && recOpts.join(',') === 'green,yellow,red,off', `${indOpts.join(',')} / ${recOpts.join(',')}`);
  check('S17b 关闭/空颜色不产出指示器',
    w.eval('indHTML("off") + "|" + indHTML("")') === '|'
    && w.eval('indHTML("red")').includes('tile_indicator_red.png'), '');
  // 构造：1 家刚打出 5（最新弃牌）+ 我有推荐 → 手牌与弃牌区各一枚指示器
  // 2026-10-12：推荐指示器只在「推荐出牌形式 = 指示器」时显示
  const indSt = JSON.parse(JSON.stringify(mid0));
  indSt.turn = -1; indSt.over = false;
  indSt.players[1].discards = [5]; indSt.last = [1, 5];
  indSt.pending = { mode: 'discard', buttons: [], discardables: indSt.hand.slice(), last: null, can_hint: true };
  w.eval(`RECOMMEND = {kind:"discard", tile:${indSt.hand[0]}, label:""}`);
  const indCount = () => (el('river-right').innerHTML.match(/class="ind"/g) || []).length
    + (el('hand').innerHTML.match(/class="ind"/g) || []).length;
  w.eval('SET.indColor = "red"; SET.recColor = "green"; SET.recForm = "float";');
  w.eval(`el("hand")._h=""; ["river-me","river-right","river-top","river-left"].forEach(id=>el(id)._h="")`);
  w.eval(`render(${JSON.stringify(indSt)})`);
  check('S17b2 推荐形式=上浮时手牌上没有指示器（弃牌区仍有）',
    (el('hand').innerHTML.match(/class="ind"/g) || []).length === 0 &&
    (el('river-right').innerHTML.match(/class="ind"/g) || []).length === 1,
    `hand=${(el('hand').innerHTML.match(/class="ind"/g) || []).length}`);
  w.eval('SET.recForm = "ind";');
  w.eval(`el("hand")._h=""`);
  w.eval(`render(${JSON.stringify(indSt)})`);
  const nColored = indCount();
  w.eval('fillSetSelects()');
  el('set-ind-color').value = 'off';
  el('set-rec-color').value = 'off';
  el('btn-set-save').click(); el('btn-set-close').click();
  const nOff = indCount();
  check('S17c 关闭后指示器立即消失（弃牌区+手牌）', nColored === 2 && nOff === 0,
    `开=${nColored} 关=${nOff}`);
  check('S17d 关闭状态已持久化', w.localStorage.getItem('dushan_ind_color') === 'off'
    && w.localStorage.getItem('dushan_rec_color') === 'off',
    `${w.localStorage.getItem('dushan_ind_color')}/${w.localStorage.getItem('dushan_rec_color')}`);
  w.eval('SET.indColor = "red"; SET.recColor = "green"; SET.recForm = "float"; saveSet();');

  // ---- S18：时间类设置改为「可直接填写的数字输入」+ 常用档 chip ----
  // 2026-10-12：上浮时机（set-floatat）已删除，出现时间统一用 set-hintdelay
  const numIn = id => el(id);
  const dlVals = id => [...((el(id) || {}).options || [])].map(o => o.value);
  const fire2 = id => el(id).dispatchEvent(new w.Event('change', { bubbles: true }));
  check('S18 时间设置为数字输入框（可自由填写）',
    ['set-timeout', 'set-hintdelay'].every(id =>
      numIn(id) && numIn(id).tagName === 'INPUT' && numIn(id).type === 'number'),
    ['set-timeout', 'set-hintdelay'].map(id => (numIn(id) || {}).tagName).join(','));
  check('S18b 时间设置带常用档 datalist',
    dlVals('dl-timeout').includes('0') && dlVals('dl-timeout').includes('3')
    && dlVals('dl-hintdelay').includes('0'), dlVals('dl-hintdelay').join(','));
  check('S18b2 旧「上浮时机」输入与档位已删除',
    !el('set-floatat') && !el('dl-floatat') && !el('set-floaton') && !el('grp-float'), '');
  // 自由填写（含小数）→ 保存并记忆
  w.eval('fillSetSelects()');
  el('set-timeout').value = '7.5';
  el('set-hintdelay').value = '2';
  el('btn-set-save').click(); el('btn-set-close').click();
  check('S18c 自由填写的秒数（含小数）可保存并记忆',
    w.eval('SET.timeout') === 7.5 && w.eval('SET.hintDelay') === 2
    && w.localStorage.getItem('dushan_timeout') === '7.5',
    `timeout=${w.eval('SET.timeout')} delay=${w.eval('SET.hintDelay')} store=${w.localStorage.getItem('dushan_timeout')}`);
  // 常用档 chip：超时「立即」(0.5)
  el('btn-settings').click();
  const chipTo = el('set-panes').querySelector('.chip[data-for="set-timeout"][data-v="0.5"]');
  check('S18e 超时自动操作有「立即」快捷档', !!chipTo && /立即/.test(chipTo.textContent));
  // 推荐出牌形式：选项 + 指示器颜色联动置灰
  const formOpts = [...el('set-rec-form').options].map(o => o.value);
  check('S18j 推荐出牌形式 = 上浮 / 指示器', formOpts.join(',') === 'float,ind',
    formOpts.join(','));
  el('set-rec-form').value = 'float'; fire2('set-rec-form');
  check('S18k 形式=上浮时指示器颜色置灰', el('set-rec-color').disabled === true, '');
  el('set-rec-form').value = 'ind'; fire2('set-rec-form');
  check('S18l 形式=指示器时颜色恢复可用', el('set-rec-color').disabled === false, '');
  el('btn-set-save').click(); el('btn-set-close').click();
  check('S18m 推荐形式可保存并记忆',
    w.eval('SET.recForm') === 'ind' && w.localStorage.getItem('dushan_rec_form') === 'ind',
    `SET.recForm=${w.eval('SET.recForm')}`);
  w.eval('SET.recForm = "float"; SET.timeout = 15; SET.hintDelay = 0; saveSet();');
  el('btn-settings').click();
  el('set-rec-form').value = 'float';
  el('set-timeout').value = '15';
  el('set-hintdelay').value = '0';
  el('btn-set-save').click(); el('btn-set-close').click();
  w.eval('SET.recForm = "float"; SET.timeout = 15; SET.hintDelay = 0; saveSet();');

  // ---- S19：功能开关与时间设置分离（关闭功能 → 其时间设置置灰禁用） ----
  el('btn-settings').click();
  const fire = id => el(id).dispatchEvent(new w.Event('change', { bubbles: true }));
  const grayed = id => el(id).disabled === true;
  el('set-hinton').checked = false; fire('set-hinton');
  check('S19 关「显示大师推荐」→ 策略/延迟/理由置灰',
    grayed('set-strategy') && grayed('set-hintdelay') && grayed('set-reasonmode')
    && el('grp-hint').classList.contains('disabled'),
    `strategy=${el('set-strategy').disabled} delay=${el('set-hintdelay').disabled} reason=${el('set-reasonmode').disabled}`);
  check('S19b 推荐关闭时「推荐出牌形式」与指示器颜色都置灰',
    grayed('set-rec-form') && grayed('set-rec-color'), '');
  el('set-hinton').checked = true; fire('set-hinton');
  check('S19c 重开推荐后参数恢复可用',
    !grayed('set-strategy') && !grayed('set-hintdelay') && !grayed('set-rec-form')
    && !el('grp-hint').classList.contains('disabled'), '');
  el('set-timeouton').checked = false; fire('set-timeouton');
  check('S19d 关「超时自动操作」→ 超时时间置灰',
    grayed('set-timeout') && el('grp-timeout').classList.contains('disabled'), '');
  check('S19e 置灰时时间值仍保留（不因关闭被清零）',
    el('set-timeout').value === '15', `值=${el('set-timeout').value}`);
  el('btn-set-save').click(); el('btn-set-close').click();
  check('S19g 开关与时间分开持久化',
    w.localStorage.getItem('dushan_hinton') === '1'
    && w.localStorage.getItem('dushan_timeouton') === '0'
    && w.localStorage.getItem('dushan_timeout') === '15',
    `on=${w.localStorage.getItem('dushan_hinton')}/${w.localStorage.getItem('dushan_timeouton')} t=${w.localStorage.getItem('dushan_timeout')}`);
  check('S19h 关闭后功能真的停用（SET 两态）',
    w.eval('SET.hintOn') === true && w.eval('SET.timeoutOn') === false,
    `hint=${w.eval('SET.hintOn')} timeout=${w.eval('SET.timeoutOn')}`);
  el('btn-settings').click();
  el('set-hinton').checked = true; fire('set-hinton');
  el('set-timeouton').checked = true; fire('set-timeouton');
  el('btn-set-save').click(); el('btn-set-close').click();
  check('S19i 恢复开关后参数重新可用',
    w.eval('SET.hintOn && SET.timeoutOn') === true
    && !el('set-timeout').disabled && !el('set-rec-form').disabled, '');
  // 旧配置迁移：timeout=0 且无独立开关键 → 关开关并把时间回落 15 秒
  check('S19j 旧配置(timeout=0)迁移为「关闭 + 15 秒」',
    /_timeoutLegacyOff/.test(html) && /dushan_timeouton/.test(html), '');

  // ---- S20：横鸡轮不再常驻状态芯片，改为「XX 打出 横鸡」toast ----
  const stHj = JSON.parse(JSON.stringify(mid0));
  stHj.turn = -1;
  stHj.hengji_active = true;
  stHj.hengji_species = [9];
  w.render(stHj);
  check('S20 状态栏不再显示「横鸡轮」',
    !el('timer-warn').textContent.includes('横鸡轮'), el('timer-warn').textContent);
  w.eval('toast("", 1); el("toast").classList.remove("on"); el("log").dataset.count = 0;');
  const evHj = [{ t: 1, kind: 'discard', rel: 1, text: '小李 打出 1条（鸡）', tile: 9, ctag: '横鸡' }];
  w.eval(`renderLog(${JSON.stringify(evHj)})`);
  check('S20b 打出横鸡 → toast 播报「X 打出 横鸡」',
    el('toast').classList.contains('on') && /打出\s*横鸡/.test(el('toast').textContent),
    el('toast').textContent);
  w.eval('el("toast").classList.remove("on")');
  const evPlain = [{ t: 1, kind: 'discard', rel: 1, text: '小李 打出 1条（鸡）', tile: 9, ctag: '幺鸡' }];
  w.eval(`renderLog(${JSON.stringify(evPlain)})`);
  check('S20c 普通鸡/非鸡不弹 toast', !el('toast').classList.contains('on'),
    el('toast').textContent);

  // ---- S21：对手出牌节奏并入设置面板（紧跟「超时自动操作」之后） ----
  // （「显示大师推荐」本轮已抽到独立的「大师推荐」分类，不再留在出牌分类里）
  const playPane = el('set-panes').querySelector('section[data-pane="play"]');
  const groups = [...playPane.querySelectorAll('.setgroup')].map(g => g.id);
  check('S21 节奏设置排在超时之后，且推荐块已移出该分类',
    groups.indexOf('grp-speed') === groups.indexOf('grp-timeout') + 1
    && groups.indexOf('grp-hint') < 0, groups.join(','));
  el('btn-settings').click();
  el('set-oppspeed').value = '0.2';
  el('btn-set-save').click(); el('btn-set-close').click();
  check('S21b 节奏改动写入 SET 与本地存储',
    Math.abs(w.eval('SET.speed') - 0.2) < 1e-9
    && w.localStorage.getItem('dushan_speed') === '0.2', String(w.localStorage.getItem('dushan_speed')));
  // 开局页与设置面板同源
  check('S21c 开局页节奏与设置同源',
    w.eval('document.getElementById("speed").value') === String(w.eval('SET.speed'))
    || w.eval('SET.speed') === 0.2, String(w.eval('SET.speed')));
  el('btn-settings').click();
  el('set-oppspeed').value = '0.45';
  el('btn-set-save').click(); el('btn-set-close').click();

  // ---- S22：出牌后中间倒计时清零重开（不残留上一轮余数） ----
  w.eval('FLOW_DEADLINE = Date.now() + 7000; TURN_DEADLINE = Date.now() + 7000;');
  w.eval('stopTurnTimer()');
  check('S22 出牌后倒计时清零（不在旧余数上继续）', w.eval('FLOW_DEADLINE') === 0,
    `FLOW_DEADLINE=${w.eval('FLOW_DEADLINE')}`);
  w.eval('updateClock()');
  const cdTxt = el('timer-count').textContent;
  check('S22b 清零后按完整超时重新开始倒数',
    /^\d+$/.test(cdTxt) && parseInt(cdTxt, 10) >= Math.min(3, w.eval('SET.timeout')), cdTxt);

  // ---- S23：底部两个按钮（2026-10-12 用户口径修订）----
  //   保存 = 落盘**并关闭面板**；关闭 = 不落盘、仅收起（放弃这次改动）
  check('S23a 底部是两个按钮（保存 + 关闭），旧「保存并关闭」已废',
    !!el('btn-set-save') && !!el('btn-set-close')
    && el('btn-set-save').textContent.trim() === '保存'
    && el('btn-set-close').textContent.trim() === '关闭'
    && !/<button[^>]*>\s*保存并关闭/.test(html),
    `${el('btn-set-save').textContent.trim()}/${el('btn-set-close').textContent.trim()}`);
  const rmBefore = w.eval('SET.reasonMode');
  el('btn-settings').click();
  el('set-reasonmode').value = rmBefore === 'brief' ? 'full' : 'brief';
  el('btn-set-close').click();
  check('S23b 只点「关闭」→ 不落盘、面板仅收起',
    w.eval('SET.reasonMode') === rmBefore && !el('set-mask').classList.contains('on'),
    `mode=${w.eval('SET.reasonMode')} open=${el('set-mask').classList.contains('on')}`);
  const rmWant = rmBefore === 'brief' ? 'full' : 'brief';
  el('btn-settings').click();
  el('set-reasonmode').value = rmWant;
  el('btn-set-save').click();
  check('S23c 点「保存」→ 落盘**并关闭**面板（2026-10-12 用户口径）',
    w.eval('SET.reasonMode') === rmWant && !el('set-mask').classList.contains('on'),
    `mode=${w.eval('SET.reasonMode')} open=${el('set-mask').classList.contains('on')}`);
  // 复原
  el('btn-settings').click();
  el('set-reasonmode').value = rmBefore;
  el('btn-set-save').click();

  const bad = results.filter(x => !x).length;
  console.log(`\nwindow errors: ${errs.length}${errs.length ? '\n' + errs.slice(0, 3).join('\n') : ''}`);
  console.log(`合计 ${results.length} 项，失败 ${bad}`);
  process.exit(bad === 0 && errs.length === 0 ? 0 : 1);
}, 1800);
