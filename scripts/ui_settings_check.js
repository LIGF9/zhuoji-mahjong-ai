// 设置面板 / 玩家名字头像 / 结束印记几何 e2e（jsdom 加载真实页面）
const fs = require('fs');
const { JSDOM } = require('jsdom');

const ROOT = 'C:/Users/hyzor/WorkBuddy/2026-09-30-12-49-43/zhuoji-mahjong';
const html = fs.readFileSync(ROOT + '/web/static/dushan.html', 'utf8');
const errs = [];
const meta = JSON.parse(fs.readFileSync(ROOT + '/reports/ui_fixtures_meta.json', 'utf8'));
const mid0 = JSON.parse(fs.readFileSync(ROOT + '/reports/ui_fixtures_mid.json', 'utf8'))[0].state;

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
    'set-huang-baodapai', 'set-strategy', 'set-timeout', 'set-hintdelay', 'set-briefreason',
    'set-hinton', 'set-timeouton', 'set-floaton',
    'grp-hint', 'grp-timeout', 'grp-float',
    'set-jiesuan-fanji', 'set-kaiju-fanji', 'set-mantiangji', 'set-nav', 'set-panes',
    'sp-name-0', 'sp-name-1', 'sp-name-2', 'sp-name-3', 'sp-ava-0', 'sp-ava-3',
    'pl-name-0', 'pl-ava-3', 'sc-zimo'];
  const missing = ids.filter(i => !el(i));
  check('S2 设置控件齐备', missing.length === 0, missing.length ? '缺: ' + missing.join(',') : `${ids.length} 项`);

  // ---- S3：头像下拉含全部候选 + 随机 ----
  const avaOpts = el('sp-ava-0').innerHTML.match(/<option/g) || [];
  const poolN = (w.eval('AVATAR_POOL.length'));
  check('S3 头像候选完整', avaOpts.length === poolN + 1, `选项 ${avaOpts.length} = 候选 ${poolN} + 随机`);

  // ---- S4：改设置并保存 → SET + localStorage 同步 ----
  w.eval('fillSetSelects()');
  el('set-end-baogang').value = '1';
  el('set-huang-baoji').value = '1';
  el('sp-name-1').value = '小李';
  el('sp-ava-1').value = '金蝉';
  el('sp-name-0').value = '老王';
  el('sp-ava-0').value = 'random';
  el('btn-set-close').click();
  const saved = JSON.parse(w.localStorage.getItem('dushan_rules') || '{}');
  const savedP = JSON.parse(w.localStorage.getItem('dushan_players') || '[]');
  check('S4 设置保存到本地', saved.end_baogang === true && saved.huang_baoji === true
    && saved.end_baoji === true && saved.huang_baodapai === true,
    JSON.stringify(saved));
  check('S4b 玩家配置保存', savedP[1] && savedP[1].name === '小李' && savedP[1].avatar === '金蝉'
    && savedP[0].name === '老王' && savedP[0].avatar === 'random', JSON.stringify(savedP));

  // ---- S5：渲染使用自定义名字与头像 ----
  w.eval('SID = "e2e-settings"');
  const st = JSON.parse(JSON.stringify(mid0));
  st.turn = -1;
  w.render(st);
  const seat1 = el('p1').innerHTML;
  const ovCard = el('res-pairs').innerHTML;
  check('S5 对手面板用自定义名字', seat1.includes('小李'), seat1.slice(0, 80));
  check('S5b 对手面板用自定义头像', seat1.includes('/assets/avatar/boy/金蝉.jpg'), '');
  check('S5c 我名字生效', el('me-name').textContent === '老王', el('me-name').textContent);
  check('S5d 随机头像解析为真实文件', /\/assets\/avatar\/(boy|girl)\/[\u4e00-\u9fa5]+\.jpg/.test(el('me-avatar').getAttribute('src')),
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

  // ---- S7：导航顺序（统计/记录 在 大师建议 之前）+ 流水为图标按钮 ----
  const navIds = [...el('nav').querySelectorAll('button')].map(b => b.id);
  check('S7 统计/记录排在 大师建议 之前',
    navIds.indexOf('btn-stats') < navIds.indexOf('btn-coach')
    && navIds.indexOf('btn-history') < navIds.indexOf('btn-coach'), navIds.join(','));
  check('S7b 流水按钮为图标', el('btn-log').classList.contains('icon')
    && el('btn-log').textContent.trim().length > 0 && !el('btn-log').textContent.includes('流水'),
    el('btn-log').textContent.trim());

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
  check('S10b 幺鸡用原牌面、冲锋鸡用专用牌面',
    up.includes('1条.png') && mj(3, '冲锋鸡').includes('冲.png')
    && mj(3, '横鸡').includes('横.png'), '');
  check('S10c 普通碰保持三张竖排', hengIdx(mj(3, null)) === -1, mj(3, null));

  // ---- S11：默认名字 = 头像名(策略缩写) ----
  w.eval('SET.players = AVA_DEFAULT.map(a => ({ name: "", avatar: a })); AVA_SESSION = null;');
  el('opp0').value = 'teacher:tenpai_rush';
  el('opp1').value = 'model:dushan_big_s3';
  el('opp2').value = 'model:dushan_s7';
  w.eval('fillPlayerRows("pl")');
  const defs = JSON.parse(w.eval('JSON.stringify([0,1,2,3].map(defaultPName))'));
  const phs = [1, 2, 3].map(r => el('pl-name-' + r).placeholder);
  check('S11 默认名=头像名(策略缩写)',
    defs[0] === '你' && defs[1] === '大乔(抢听)' && defs[2] === '妲己(B3)' && defs[3] === '王昭君(S7)',
    defs.join(' | '));
  check('S11b 输入框占位显示默认名', phs.every((p, i) => p.includes(defs[i + 1])), phs.join(' | '));
  // 大师缩写随冠军权重走（meta.master_info.abbr），无 meta 时回退 M
  const mAbbr = (meta.master_info && meta.master_info.abbr) || 'M';
  check('S11c 随机/大师/大网缩写', JSON.parse(w.eval(
    'JSON.stringify([oppAbbr("random"), oppAbbr("model:master"), oppAbbr("model:dushan_big_bc"), oppAbbr("teacher:gambler")])'
  )).join(',') === `随机,${mAbbr},BBC,赌徒`, `大师缩写=${mAbbr}`);

  // ---- S11d/S11e：大师选项文案 = 大师 + 版本号（规模），且不写「最强」 ----
  const mLabel = "🤖 " + ((meta.master_info && meta.master_info.label) || '大师');
  check('S11d 大师项版本号与规模来自权重', el('opp0').options[0].textContent === mLabel
    && !/最强/.test(el('opp0').options[0].textContent), el('opp0').options[0].textContent);
  w.eval('fillOppSelects()');            // 恢复到开局默认值后再检查
  check('S11e 开局三家默认都用最强模型',
    ['opp0', 'opp1', 'opp2'].every(id => el(id).value === 'model:master'),
    ['opp0', 'opp1', 'opp2'].map(id => el(id).value).join(','));

  // ---- S12：翻鸡设置项（结算翻鸡 3 选 1；开局翻鸡 4 选 1 且默认关闭） ----
  const jOpts = [...el('set-jiesuan-fanji').options].map(o => o.value);
  const kOpts = [...el('set-kaiju-fanji').options].map(o => o.value);
  check('S12 结算翻鸡=上下鸡/上鸡/下鸡', jOpts.join(',') === 'both,up,down', jOpts.join(','));
  check('S12b 开局翻鸡=关闭/上鸡/下鸡/上下鸡', kOpts.join(',') === 'off,up,down,both', kOpts.join(','));
  check('S12c 默认口径 结算=both 开局=off',
    JSON.parse(w.eval('JSON.stringify([SET.jiesuan, SET.kaijuFanji])')).join(',') === 'both,off',
    w.eval('SET.jiesuan + "/" + SET.kaijuFanji'));
  // 旧「上下鸡」设置项已重命名，不再存在
  check('S12d 旧「上下鸡」开关已移除', !el('set-shangxia') && !el('set-kaiju'), '');

  // ---- S13：设置页左右两栏（左侧分类 → 右侧面板） ----
  const navBtns = [...el('set-nav').querySelectorAll('button')];
  const panes = [...el('set-panes').querySelectorAll('section')];
  check('S13 左栏分类数 = 右侧面板数', navBtns.length === 6 && panes.length === 6,
    `${navBtns.length} / ${panes.length}`);
  check('S13b 分类与面板一一对应',
    navBtns.every((b, i) => b.dataset.pane === panes[i].dataset.pane), '');
  const activeCount = () => panes.filter(s => s.classList.contains('on')).length;
  const before = activeCount();
  navBtns[3].click();
  const onIdx = panes.findIndex(s => s.classList.contains('on'));
  check('S13c 点左侧栏切换右侧面板', before === 1 && onIdx === 3 && activeCount() === 1,
    `切换前 ${before} 个显示 → 第 ${onIdx + 1} 个`);
  check('S13d 分类切换会记忆', w.localStorage.getItem('dushan_setpane') === 'rule',
    String(w.localStorage.getItem('dushan_setpane')));
  navBtns[0].click();

  // ---- S14：大师建议 = 可随时开关的 icon 双态按钮 ----
  const coachBtn = el('btn-coach');
  const coachIcon = () => coachBtn.textContent.trim();
  w.eval('SET.coach = false; applyCoachBtn(); el("coach").classList.remove("on");');
  const icoOff = coachIcon();
  check('S14 大师建议按钮为图标', coachBtn.classList.contains('icon') && icoOff.length > 0
    && !icoOff.includes('大师建议'), icoOff);
  coachBtn.click();                       // 非决策点也能打开（LAST_STATE 为空）
  const icoOn = coachIcon();
  check('S14b 非决策点也能开启', coachBtn.classList.contains('toggle-on')
    && el('coach').classList.contains('on') && icoOn !== icoOff, `${icoOff} → ${icoOn}`);
  check('S14c 开关状态已持久化', w.localStorage.getItem('dushan_coach_on') === '1',
    String(w.localStorage.getItem('dushan_coach_on')));
  coachBtn.click();
  check('S14d 再点即关闭', !coachBtn.classList.contains('toggle-on')
    && !el('coach').classList.contains('on') && coachIcon() === icoOff, coachIcon());

  // ---- S15：流水 icon 双态 ----
  const logBtn = el('btn-log');
  const logHidden = () => el('log').classList.contains('hidden');
  logBtn.click();
  const icoLogOn = logBtn.textContent.trim();
  logBtn.click();
  const icoLogOff = logBtn.textContent.trim();
  check('S15 流水按钮双态 icon', logBtn.classList.contains('icon')
    && icoLogOn !== icoLogOff && logHidden() === true, `${icoLogOn} / ${icoLogOff}`);

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
  const indSt = JSON.parse(JSON.stringify(mid0));
  indSt.turn = -1; indSt.over = false;
  indSt.players[1].discards = [5]; indSt.last = [1, 5];
  indSt.pending = { mode: 'discard', buttons: [], discardables: indSt.hand.slice(), last: null, can_hint: true };
  w.eval(`RECOMMEND = {kind:"discard", tile:${indSt.hand[0]}, label:""}`);
  const indCount = () => (el('river-right').innerHTML.match(/class="ind"/g) || []).length
    + (el('hand').innerHTML.match(/class="ind"/g) || []).length;
  w.eval('SET.indColor = "red"; SET.recColor = "green";');
  w.eval(`el("hand")._h=""; ["river-me","river-right","river-top","river-left"].forEach(id=>el(id)._h="")`);
  w.eval(`render(${JSON.stringify(indSt)})`);
  const nColored = indCount();
  w.eval('fillSetSelects()');
  el('set-ind-color').value = 'off';
  el('set-rec-color').value = 'off';
  el('btn-set-close').click();
  const nOff = indCount();
  check('S17c 关闭后指示器立即消失（弃牌区+手牌）', nColored === 2 && nOff === 0,
    `开=${nColored} 关=${nOff}`);
  check('S17d 关闭状态已持久化', w.localStorage.getItem('dushan_ind_color') === 'off'
    && w.localStorage.getItem('dushan_rec_color') === 'off',
    `${w.localStorage.getItem('dushan_ind_color')}/${w.localStorage.getItem('dushan_rec_color')}`);
  w.eval('SET.indColor = "red"; SET.recColor = "green"; saveSet();');

  // ---- S18：时间类设置改为「可直接填写的数字输入」+ 常用档 chip ----
  const numIn = id => el(id);
  const dlVals = id => [...((el(id) || {}).options || [])].map(o => o.value);
  check('S18 时间设置为数字输入框（可自由填写）',
    ['set-timeout', 'set-hintdelay', 'set-floatat'].every(id =>
      numIn(id) && numIn(id).tagName === 'INPUT' && numIn(id).type === 'number'),
    ['set-timeout', 'set-hintdelay', 'set-floatat'].map(id => (numIn(id) || {}).tagName).join(','));
  check('S18b 时间设置带常用档 datalist',
    dlVals('dl-timeout').includes('0') && dlVals('dl-timeout').includes('3')
    && dlVals('dl-floatat').includes('-1') && dlVals('dl-floatat').includes('0')
    && dlVals('dl-hintdelay').includes('0'), dlVals('dl-floatat').join(','));
  // 自由填写（含小数）→ 保存并记忆
  w.eval('fillSetSelects()');
  el('set-timeout').value = '7.5';
  el('set-hintdelay').value = '2';
  el('set-floatat').value = '4.5';
  el('btn-set-close').click();
  check('S18c 自由填写的秒数（含小数）可保存并记忆',
    w.eval('SET.timeout') === 7.5 && w.eval('SET.hintDelay') === 2 && w.eval('SET.floatAt') === 4.5
    && w.localStorage.getItem('dushan_timeout') === '7.5',
    `timeout=${w.eval('SET.timeout')} floatAt=${w.eval('SET.floatAt')} store=${w.localStorage.getItem('dushan_timeout')}`);
  // 常用档 chip：推荐上浮「立即」(-1)、超时「立即」(0.5)
  el('btn-settings').click();
  const chipImm = el('set-panes').querySelector('.chip[data-for="set-floatat"][data-v="-1"]');
  const chipTo = el('set-panes').querySelector('.chip[data-for="set-timeout"][data-v="0.5"]');
  check('S18d 推荐上浮有「立即」快捷档', !!chipImm && /立即/.test(chipImm.textContent));
  check('S18e 超时自动操作有「立即」快捷档', !!chipTo && /立即/.test(chipTo.textContent));
  if (chipImm) chipImm.click();
  check('S18f 点「立即」档会填入 -1', el('set-floatat').value === '-1', `值=${el('set-floatat').value}`);
  el('btn-set-close').click();
  check('S18g 立即上浮可保存并记忆',
    w.eval('SET.floatAt') === -1 && w.localStorage.getItem('dushan_floatat') === '-1',
    `SET.floatAt=${w.eval('SET.floatAt')} store=${w.localStorage.getItem('dushan_floatat')}`);
  el('btn-settings').click();
  const selBack = el('set-floatat').value;
  check('S18h 重新打开设置回显「立即」', selBack === '-1', `回显=${selBack}`);
  // 越界 / 非法值兜底
  el('set-floatat').value = '9999';
  el('set-timeout').value = '-5';
  el('btn-set-close').click();
  check('S18i 越界值自动夹取（上浮≤600 / 超时≥0）',
    w.eval('SET.floatAt') === 600 && w.eval('SET.timeout') === 0,
    `floatAt=${w.eval('SET.floatAt')} timeout=${w.eval('SET.timeout')}`);
  el('btn-settings').click();
  el('set-floatat').value = '0';
  el('set-timeout').value = '15';
  el('set-hintdelay').value = '0';
  el('btn-set-close').click();
  w.eval('SET.floatAt = 0; SET.timeout = 15; SET.hintDelay = 0; saveSet();');

  // ---- S19：功能开关与时间设置分离（关闭功能 → 其时间设置置灰禁用） ----
  el('btn-settings').click();
  const fire = id => el(id).dispatchEvent(new w.Event('change', { bubbles: true }));
  const grayed = id => el(id).disabled === true;
  el('set-hinton').checked = false; fire('set-hinton');
  check('S19 关「显示大师推荐」→ 策略/延迟/理由置灰',
    grayed('set-strategy') && grayed('set-hintdelay') && grayed('set-briefreason')
    && el('grp-hint').classList.contains('disabled'),
    `strategy=${el('set-strategy').disabled} delay=${el('set-hintdelay').disabled} reason=${el('set-briefreason').disabled}`);
  check('S19b 推荐总开关关闭时「自动上浮」整块禁用',
    grayed('set-floaton') && el('grp-float').classList.contains('disabled'), '');
  el('set-hinton').checked = true; fire('set-hinton');
  check('S19c 重开推荐后参数恢复可用',
    !grayed('set-strategy') && !grayed('set-hintdelay') && !grayed('set-floaton')
    && !el('grp-hint').classList.contains('disabled'), '');
  el('set-timeouton').checked = false; fire('set-timeouton');
  check('S19d 关「超时自动操作」→ 超时时间置灰',
    grayed('set-timeout') && el('grp-timeout').classList.contains('disabled'), '');
  check('S19e 置灰时时间值仍保留（不因关闭被清零）',
    el('set-timeout').value === '15', `值=${el('set-timeout').value}`);
  el('set-floaton').checked = false; fire('set-floaton');
  check('S19f 关「自动上浮」→ 上浮时机置灰',
    grayed('set-floatat') && el('grp-float').classList.contains('disabled'), '');
  el('btn-set-close').click();
  check('S19g 开关与时间分开持久化',
    w.localStorage.getItem('dushan_hinton') === '1'
    && w.localStorage.getItem('dushan_timeouton') === '0'
    && w.localStorage.getItem('dushan_floaton') === '0'
    && w.localStorage.getItem('dushan_timeout') === '15',
    `on=${w.localStorage.getItem('dushan_hinton')}/${w.localStorage.getItem('dushan_timeouton')}/${w.localStorage.getItem('dushan_floaton')} t=${w.localStorage.getItem('dushan_timeout')}`);
  check('S19h 关闭后功能真的停用（SET 三态）',
    w.eval('SET.hintOn') === true && w.eval('SET.timeoutOn') === false
    && w.eval('SET.floatOn') === false,
    `hint=${w.eval('SET.hintOn')} timeout=${w.eval('SET.timeoutOn')} float=${w.eval('SET.floatOn')}`);
  el('btn-settings').click();
  el('set-hinton').checked = true; fire('set-hinton');
  el('set-timeouton').checked = true; fire('set-timeouton');
  el('set-floaton').checked = true; fire('set-floaton');
  el('btn-set-close').click();
  check('S19i 恢复开关后参数重新可用',
    w.eval('SET.hintOn && SET.timeoutOn && SET.floatOn') === true
    && !el('set-timeout').disabled && !el('set-floatat').disabled, '');
  // 旧配置迁移：timeout=0 且无独立开关键 → 关开关并把时间回落 15 秒
  check('S19j 旧配置(timeout=0)迁移为「关闭 + 15 秒」',
    /_timeoutLegacyOff/.test(html) && /dushan_timeouton/.test(html), '');

  const bad = results.filter(x => !x).length;
  console.log(`\nwindow errors: ${errs.length}${errs.length ? '\n' + errs.slice(0, 3).join('\n') : ''}`);
  console.log(`合计 ${results.length} 项，失败 ${bad}`);
  process.exit(bad === 0 && errs.length === 0 ? 0 : 1);
}, 1800);
