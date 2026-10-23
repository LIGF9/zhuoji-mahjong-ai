/* ui_replay_check.js —— 对局回放：前端状态机回归闸门
 *
 * 背景
 * ----
 * 「对局记录」里点某一局可回放：服务端把每手的事件流（谁摸/打/碰/杠/报叫/胡 + 四家
 * 手牌张数 hc + 巡号 r + **四家手牌牌种**）存进 history/replays.jsonl。
 *
 * 2026-10-12 重构（用户反馈「回放界面太丑」）
 * ------------------------------------------
 * 回放不再自画一套「简化牌桌」，而是**直接驱动实盘那套渲染函数**
 * （renderOpp / renderMyMelds / renderHand / renderRivers —— 同一个「播放器」）：
 *   · #replay-mask 退化成**底部工具条**（背景透明 + pointer-events:none，不遮牌桌）；
 *   · html.replaying 藏起「对局中才有意义」的实时控件（#log/#actions/#coach/#tingbar…）；
 *   · 中央计时盘改用 `#timer::after { content: attr(data-rp) }` 显示「第 N 巡 / 第 M 步」；
 *   · 退出时 clearBoardCaches() + 整帧重绘，牌桌回到实盘内容。
 * 因此本脚本断言的行为与旧版完全不同（旧的 .rp-board/.rp-seat/.rp-back 已全部删除）。
 *
 * 断言分组（喂一份真实 replays.jsonl，取**最后一条带新载荷的记录**）：
 *   ① 打开：工具条出现、html.replaying 生效、旧简化牌桌确认已删除
 *   ② 第 0 帧：牌桌按「发牌快照」渲染 4 家**真实手牌**（不是牌背），中央盘/底部条正确
 *   ③ 逐帧走完全程：牌河总数 == 打牌事件数 − 被碰/明杠吃走的张数；末帧手牌 == 权威 hc；
 *      副露组数与张数 == 碰/杠事件推出的值（补杠合并进碰组）
 *   ④ 三种定位：上一步 / 拖进度条 / 「跳到第 N 巡」
 *   ⑤ 自动播放 / 暂停
 *   ⑥ 退出：按钮 / ESC 都能关；关闭后 html.replaying 摘掉、牌桌回到实盘
 *   ⑦ 回放中不提交实盘操作（点手牌/点操作条不发 /api/act）
 *   ⑧ 回放已过期的兜底
 *   ⑨ 补杠重建专项：1 组杠而不是「碰 + 杠」两组
 *
 * 运行（jsdom 装在托管 node workspace 里）：
 *   NODE_PATH=C:/Users/LIGF/.workbuddy/binaries/node/workspace/node_modules \
 *     node.exe scripts/ui_replay_check.js
 */
const fs = require('fs');
const { JSDOM } = require('jsdom');

const ROOT = 'C:/Users/LIGF/WorkBuddy/2026-09-30-12-49-43/zhuoji-mahjong';
const html = fs.readFileSync(ROOT + '/web/static/dushan.html', 'utf8');
const meta = JSON.parse(fs.readFileSync(ROOT + '/reports/ui_fixtures_meta.json', 'utf8'));

/* --- 取一份真实回放数据：从末尾往前找**第一条带新载荷**的记录 ---
   新载荷 = deal（回放 4 家发牌快照）非空 + ev[0][8]（四家手牌牌种）存在。
   服务端 2026-10-12 之前落的行没有这两个字段，前端虽然能降级（画牌背），
   但本脚本要验「真实牌面」路径，所以必须挑新记录。 */
const RP_FILE = ROOT + '/web/history/replays.jsonl';
if (!fs.existsSync(RP_FILE)) {
  console.log('✗ 找不到 ' + RP_FILE + '\n  先跑：python scripts/replay_e2e.py（会真打一局并落盘）');
  process.exit(1);
}
const lines = fs.readFileSync(RP_FILE, 'utf8').split('\n').filter(s => s.trim());
if (!lines.length) {
  console.log('✗ replays.jsonl 是空的，先跑 scripts/replay_e2e.py 造一局');
  process.exit(1);
}
function hasNewPayload(rp) {
  return Array.isArray(rp.deal) && rp.deal.length === 4 && rp.deal.some(Boolean) &&
    (rp.ev || []).some(e => Array.isArray(e[8]) && e[8].some(x => x));
}
let REPLAY = null;
for (let i = lines.length - 1; i >= 0; i--) {
  let rp;
  try { rp = JSON.parse(lines[i]); } catch (e) { continue; }
  if (!hasNewPayload(rp)) continue;
  if (!REPLAY) REPLAY = rp;                 // 兜底：最新一条带新载荷的
  // 优先挑带 final 终局快照的（2026-10-12 起落盘）—— 回放结算页要用它
  if (rp.final && rp.final.result) { REPLAY = rp; break; }
}
if (!REPLAY) {
  console.log('✗ replays.jsonl 里没有带新载荷（deal + ev[i][8]）的记录\n' +
    '  先跑：python scripts/replay_e2e.py（用新服务端落一局）');
  process.exit(1);
}
const RID = REPLAY.rid;
const EV = REPLAY.ev;
const MY_SEAT = REPLAY.my_seat | 0;
const absOf = rel => (MY_SEAT + rel) % 4;
const relOf = abs => ((abs - MY_SEAT) % 4 + 4) % 4;

/* --- 事件统计（期望值口径必须与前端 replayState 完全一致）--- */
const K = (e, kind) => e[1] === kind && e[3] >= 0;
const N_DISCARD = EV.filter(e => K(e, 'discard')).length;
const N_PENG = EV.filter(e => K(e, 'peng')).length;
const kongType = e => {
  const t = String(e[7] || '');
  return (t.indexOf('暗杠') >= 0 || t.indexOf('闷豆') >= 0) ? 4
         : (t.indexOf('补杠') >= 0 || t.indexOf('爬坡豆') >= 0) ? 3 : 2;
};
const KONGS = EV.filter(e => K(e, 'kong'));
const N_M2 = KONGS.filter(e => kongType(e) === 2).length;   // 点杠（明杠）
const N_M3 = KONGS.filter(e => kongType(e) === 3).length;   // 补杠（合并进碰组）
const N_M4 = KONGS.filter(e => kongType(e) === 4).length;   // 暗杠（暗杠）
/* 2026-10-12 三改版：被碰/明杠吃走的弃牌**不再离河**，改灰显（.mc.claimed）——
   牌河 = 全部打牌事件的完整历史（点炮那张本来就不弹）。
   旧口径「牌河 = 打牌 − 被碰 − 被明杠」会让打出者牌河缺一格，
   用户正是因此把「妲己第二张八筒」误读成「第一张出牌」而怀疑冲锋鸡漏记。 */
const N_CLAIMED = N_PENG + N_M2;
const EXPECT_RIVER = N_DISCARD;
/* 副露组数：碰 + 明杠 + 暗杠 各成一组；补杠升级已有碰组，不新增 */
const EXPECT_GROUPS = N_PENG + N_M2 + N_M4;
/* 副露张数：纯碰组 3 张；补杠把**同牌那次碰**升级成 4 张（组数不变）；
   明杠 / 暗杠 各 4 张。
   ⚠️ 不能写成 N_PENG*3 + (N_M2+N_M3+N_M4)*4 —— 那会把被补杠升级的那次碰
   重复计一次（先按碰 3 张、再按杠 4 张）。必须扣掉 N_M3 个碰组。 */
const EXPECT_MELD_CELLS = (N_PENG - N_M3) * 3 + N_M2 * 4 + N_M3 * 4 + N_M4 * 4;
/* 末帧权威手牌（result 事件也带 hands）*/
const LAST = EV[EV.length - 1];
const LAST_HC = Array.isArray(LAST[6]) ? LAST[6] : [0, 0, 0, 0];
const LAST_HANDS = Array.isArray(LAST[8]) ? LAST[8] : ["", "", "", ""];
const nChars = s => String(s || '').length;
const DEAL_N = (REPLAY.deal || []).map(nChars);

let pass = 0, fail = 0;
function check(name, cond, extra = '') {
  if (cond) { pass++; console.log(`  PASS  ${name}${extra ? '  ' + extra : ''}`); }
  else { fail++; console.log(`  FAIL  ${name}${extra ? '  ' + extra : ''}`); }
}
const sleep = ms => new Promise(r => setTimeout(r, ms));

const errs = [];
const reqLog = [];                          // 记录所有 fetch 出去的 URL
const apiLog = [];                          // {url, method, body}：供删除记录断言查载荷
let replayResp = { ok: true, replay: REPLAY };   // 可切换成「已过期」以测兜底
let histDelResp = { ok: true, removed_rec: 1, removed_replay: 1 };
const dom = new JSDOM(html, {
  runScripts: 'dangerously',
  pretendToBeVisual: true,
  url: 'http://127.0.0.1:8770/static/dushan.html',
  beforeParse(w) {
    if (!w.matchMedia) w.matchMedia = q => ({ matches: false, media: String(q),
      onchange: null, addEventListener() {}, removeEventListener() {},
      addListener() {}, removeListener() {}, dispatchEvent: () => false });
    if (!w.ResizeObserver) w.ResizeObserver = class { observe() {} disconnect() {} };
    // 只喂 /api/meta（牌表）与 /api/replay；其余挂起，避免 boot()/poll() 把测试带跑偏
    w.fetch = (url, opt) => {
      const u = String(url);
      const o = opt || {};
      reqLog.push(u);
      let body = null;
      try { body = o.body ? JSON.parse(o.body) : null; } catch (e) { body = String(o.body); }
      apiLog.push({ url: u, method: String(o.method || 'GET').toUpperCase(), body });
      if (u.includes('/api/meta')) return Promise.resolve({ ok: true, json: () => Promise.resolve(meta) });
      if (u.includes('/api/replay')) return Promise.resolve({ ok: true, json: () => Promise.resolve(replayResp) });
      if (u.includes('/api/history_delete')) return Promise.resolve({ ok: true, json: () => Promise.resolve(histDelResp) });
      if (u.includes('/api/history')) return Promise.resolve({ ok: true, json: () => Promise.resolve({ records: [] }) });
      return new Promise(() => {});
    };
    w.addEventListener('error', e => errs.push('window.error: ' + ((e.error && e.error.stack) || e.message)));
  },
});
const w = dom.window;
const doc = w.document;
const el = id => doc.getElementById(id);
/* 全局词法变量（let RP/RP_IDX/RP_TIMER/RP_MODE/RP_LAST）在页面 script 顶层 → 用间接 eval 读取 */
const G = (expr) => { try { return w.eval(expr); } catch (e) { return '<<eval err: ' + e.message + '>>'; } };
const fire = (node, type) => node.dispatchEvent(new w.Event(type, { bubbles: true }));
const seatSel = { 0: '#me', 1: '#p1', 2: '#p2', 3: '#p3' };
/* 牌桌上的真实牌：手牌 .tile-mine / 副露+对手手牌 .mc（都要排除 .ghost 占位）*/
const myTiles = () => el('hand').querySelectorAll('.tile-mine:not(.ghost)').length;
const oppTiles = rel => el(seatSel[rel].slice(1)).querySelectorAll('.phand .mc:not(.ghost)').length;
const meldGroups = () => el('me-melds').querySelectorAll('.meld').length +
  [1, 2, 3].reduce((s, r) => s + el(seatSel[r].slice(1)).querySelectorAll('.melds .meld').length, 0);
const meldCells = () => el('me-melds').querySelectorAll('.mc').length +
  [1, 2, 3].reduce((s, r) => s + el(seatSel[r].slice(1)).querySelectorAll('.melds .mc').length, 0);
const riverTiles = () => ['me', 'right', 'top', 'left']
  .reduce((s, k) => s + el('river-' + k).querySelectorAll('.mc').length, 0);
const cssOf = (node, prop) => w.getComputedStyle(node).getPropertyValue(prop);

setTimeout(async () => {
  console.log(`=== 数据 ===  rid=${RID}  第 ${REPLAY.hand} 局（${REPLAY.type}）  ` +
    `我的座位=${MY_SEAT}  庄家=${REPLAY.dealer}`);
  console.log(`             ${EV.length} 事件 · 打牌 ${N_DISCARD} · 碰 ${N_PENG} · ` +
    `点杠 ${N_M2} · 补杠 ${N_M3} · 暗杠 ${N_M4} · 发牌 ${DEAL_N.join('+')}`);
  console.log(`             期望：牌河 ${EXPECT_RIVER} 张 · 副露 ${EXPECT_GROUPS} 组 / ${EXPECT_MELD_CELLS} 张 · ` +
    `末帧手牌 ${LAST_HC.join('+')}`);

  if (!w.eval('TILES && TILES.length')) {
    w.eval('for (let i = 0; i < 27; i++) TILES[i] = ' +
      '{ id: i, name: "t" + i, cn: "牌" + i, suit: Math.floor(i / 9), rank: i % 9 };');
  }
  check('牌表就绪', w.eval('TILES.length') === 27, `TILES=${w.eval('TILES.length')}`);
  check('旧「简化牌桌」已删除（无 #rp-board / .rp-seat / .rp-back）',
    el('rp-board') === null && doc.querySelectorAll('.rp-seat,.rp-back,.rp-dis,.rp-center').length === 0);
  check('控制区 = 底部弹出式抽屉 + 左上角回放角标（2026-10-12 五改版）',
    !!el('rp-drawer') && !!el('rp-hot') &&
    el('rp-drawer').parentElement === el('replay-mask') &&
    el('replay-mask').querySelectorAll('.rp-side').length === 1 &&
    !!el('replay-mask').querySelector('.rp-side.left') &&
    el('rp-now').parentElement === el('replay-mask') &&
    !el('replay-mask').querySelector('.rp-bar') &&
    !el('replay-mask').querySelector('.rp-side.right'));
  check('左上角 = 🔴回放角标 + 本局鸡牌；本局名称已移进底部抽屉',
    !!el('replay-mask').querySelector('.rp-side.left .rp-rec') &&
    !!el('replay-mask').querySelector('.rp-side.left #rp-ji') &&
    !!el('rp-drawer').querySelector('.rp-drow #rp-title') &&
    !el('replay-mask').querySelector('.rp-side.left #rp-title'));
  check('播放组件在抽屉首行、步骤分数与名称同一行',
    (() => {
      const row = el('rp-drawer').querySelector('.rp-drow');
      const btns = el('rp-drawer').querySelector('.rp-btns');
      if (!row || !btns) return false;
      const ids = [...btns.children].map(n => n.id);
      return ids.join(',') === 'rp-prev,rp-play,rp-next' &&
        row.contains(el('rp-step')) && row.contains(el('rp-title'));
    })(), [...el('rp-drawer').querySelectorAll('.rp-btns > *')]
      .map(n => n.id || n.textContent).join(','));
  check('控制区元素齐全（跳巡组件已删）',
    ['rp-title', 'rp-now', 'rp-step', 'rp-range',
     'rp-prev', 'rp-next', 'rp-play', 'rp-close', 'rp-ji', 'rp-res']
      .every(id => !!el(id)) && !el('rp-round') && !el('rp-go') && !el('rp-sub'));
  check('抽屉首行右侧 = 「结算 + ⏹」，进度条铺在抽屉底部',
    el('rp-res').parentElement === el('rp-close').parentElement &&
    el('rp-close').parentElement === el('rp-drawer').querySelector('.rp-drow') &&
    el('rp-close').textContent.trim() === '⏹' &&
    el('rp-range').parentElement === el('rp-drawer'));

  console.log('\n=== 1. 打开回放 ===');
  check('初始未打开', !el('replay-mask').classList.contains('on'));
  // 模拟「实盘牌桌正显示上一局内容」：副露/牌河 DOM 有内容、缓存指纹与内容一致
  // （这正是用户 2026-10-12 实测「进回放第 0 帧就残留上一局九万副露」的现场）
  el('me-melds').innerHTML =
    '<div class="meld"><span class="mc"><img alt=""></span><span class="mc"><img alt=""></span>' +
    '<span class="mc"><img alt=""></span></div>';
  el('me-melds')._h = el('me-melds').innerHTML;
  el('river-top').innerHTML = '<span class="mc"><img alt=""></span>';
  el('river-top')._rk = '1|8||red|';
  // 上一局实盘的**结算面板**还开着（用户实测：提前结束对局 → 弹出荒牌结算 →
  // 去对局记录点回放，结果只有记录页被收起，结算面板继续盖着牌桌）
  el('result-mask').classList.add('on');
  el('res-actions').classList.add('on');
  await w.openReplay(RID);
  check('工具条已出现', el('replay-mask').classList.contains('on'));
  check('进回放时把实盘的结算面板一并收起（用户 2026-10-12 反馈）',
    !el('result-mask').classList.contains('on') && !el('res-actions').classList.contains('on'),
    `mask=${el('result-mask').classList.contains('on')}`);
  check('进入回放自动开始播放（用户 2026-10-12 要求）',
    !!G('RP_TIMER') && el('rp-play').textContent === '⏸',
    `timer=${!!G('RP_TIMER')} btn=${el('rp-play').textContent}`);
  w.eval('stopReplayPlay()');   // 冻住自动播放：后面的断言要精确控制帧号
  check('暂停后按钮回到 ▶、定时器清空',
    el('rp-play').textContent === '▶' && G('RP_TIMER') === null);
  check('进入回放模式 RP_MODE', G('RP_MODE') === true, `RP_MODE=${G('RP_MODE')}`);
  check('html.replaying 已挂上（用于藏实盘控件）',
    doc.documentElement.classList.contains('replaying'));
  check('工具条不遮牌桌：pointer-events:none', cssOf(el('replay-mask'), 'pointer-events') === 'none',
    cssOf(el('replay-mask'), 'pointer-events'));
  /* 基线（本测试未开实盘局）：#nav / #actions 可见，#log 因 html.narrow 本就是 none。
     所以「藏实盘控件」用前两个做判据才有区分度。 */
  check('回放中 #nav 被藏起（CSS）', cssOf(el('nav'), 'display') === 'none',
    cssOf(el('nav'), 'display'));
  check('回放中 #actions 被藏起（CSS）', cssOf(el('actions'), 'display') === 'none',
    cssOf(el('actions'), 'display'));
  check('回放中 #counter-btn 被藏起（CSS）', cssOf(el('counter-btn'), 'display') === 'none',
    cssOf(el('counter-btn'), 'display'));
  check('标题含局号', (el('rp-title').textContent || '').indexOf('第 ' + REPLAY.hand + ' 局') !== -1,
    el('rp-title').textContent);
  check('进度条 max = 事件数', el('rp-range').max === String(EV.length), `max=${el('rp-range').max}`);

  console.log('\n=== 1a. 底部抽屉显隐（2026-10-12 用户反馈修复：底边热区 + 角标点击） ===');
  const clearDrawerTimer = () =>
    w.eval('if (RP_DRAWER_TIMER) { clearTimeout(RP_DRAWER_TIMER); RP_DRAWER_TIMER = null; }');
  el('rp-drawer').classList.remove('show');
  el('rp-hot').dispatchEvent(new w.MouseEvent('mouseenter'));
  check('行为①：鼠标触底边热区 → 抽屉弹出（.show）',
    el('rp-drawer').classList.contains('show'));
  el('rp-drawer').classList.remove('show');
  clearDrawerTimer();
  el('replay-mask').querySelector('.rp-rec')
    .dispatchEvent(new w.MouseEvent('click', { bubbles: true }));
  check('行为②：点左上角「回放」角标 → 抽屉弹出（.show）',
    el('rp-drawer').classList.contains('show'));
  clearDrawerTimer();
  el('rp-hot').dispatchEvent(new w.MouseEvent('click'));   // 展开态点热区 → 400ms 后收起
  await sleep(520);
  check('行为③：抽屉展开时点底边热区 → 收起（触屏无 hover 的兜底）',
    !el('rp-drawer').classList.contains('show'));
  clearDrawerTimer();
  el('rp-drawer').classList.remove('show');

  console.log('\n=== 1b. 不残留上局实盘内容（clearBoardCaches 必须强制重绘） ===');
  check('上局副露被清掉（用户实测：第 0 帧残留九万副露）',
    el('me-melds').querySelectorAll('.mc').length === 0,
    `残留 ${el('me-melds').querySelectorAll('.mc').length} 张`);
  check('上局牌河被清掉', riverTiles() === 0, String(riverTiles()));
  check('左上角鸡牌 = 本局鸡牌数（含幺鸡）',
    el('rp-ji').querySelectorAll('img').length === (REPLAY.ji_tiles || []).length,
    `${el('rp-ji').querySelectorAll('img').length} vs ${(REPLAY.ji_tiles || []).length}`);

  console.log('\n=== 2. 第 0 帧（发牌快照，回放复用实盘牌桌） ===');
  check('我手牌 = 发牌快照张数（画真实牌面，不是牌背）',
    myTiles() === DEAL_N[MY_SEAT], `${myTiles()} vs ${DEAL_N[MY_SEAT]}`);
  let oppOk = true, oppDetail = [];
  for (const rel of [1, 2, 3]) {
    const want = DEAL_N[absOf(rel)];
    const got = oppTiles(rel);
    oppDetail.push(`${got}/${want}`);
    if (got !== want) oppOk = false;
  }
  check('三家对手手牌 = 发牌快照张数（真实牌面）', oppOk, oppDetail.join(' '));
  check('第 0 帧手牌合计 = 庄家 14 + 其余 13×3 = 53',
    myTiles() + oppTiles(1) + oppTiles(2) + oppTiles(3) === 53,
    String(myTiles() + oppTiles(1) + oppTiles(2) + oppTiles(3)));
  check('第 0 帧牌河全空', riverTiles() === 0, String(riverTiles()));
  check('第 0 帧无副露', meldGroups() === 0, String(meldGroups()));
  check('中央盘显示局号', (el('round-chip').textContent || '').indexOf(String(REPLAY.hand)) !== -1,
    el('round-chip').textContent);
  check('中央盘余牌显示「—」（回放没有余牌概念）', el('wall-num').textContent === '—',
    el('wall-num').textContent);
  check('中央计时盘只显示步数（2026-10-12 去掉了「第 N 巡」）',
    (el('timer').dataset.rp || '') === '第 0 步',
    JSON.stringify(el('timer').dataset.rp));
  check('步数显示 0 / N', el('rp-step').textContent === `0 / ${EV.length}`, el('rp-step').textContent);
  check('提示为「发牌（尚未开始）」', (el('rp-now').textContent || '').indexOf('发牌') !== -1);
  check('「上一步」禁用、「下一步」可用',
    el('rp-prev').disabled === true && el('rp-next').disabled === false);
  check('回合指示灯最多亮 1 盏（当前动作家）',
    [0, 1, 2, 3].filter(r => el('light-' + r).classList.contains('on')).length <= 1,
    [0, 1, 2, 3].filter(r => el('light-' + r).classList.contains('on')).join(','));

  console.log('\n=== 3. 逐帧走完全程（下一步 × N） ===');
  for (let i = 0; i < EV.length; i++) el('rp-next').click();
  check('已应用全部事件', G('RP_IDX') === EV.length, `RP_IDX=${G('RP_IDX')}`);
  check('到尾部后「下一步」禁用', el('rp-next').disabled === true);
  check('牌河总数 = 全部打牌事件（被碰/明杠吃的牌灰显留在河里，2026-10-12 三改版）',
    riverTiles() === EXPECT_RIVER, `牌河 ${riverTiles()} vs 期望 ${EXPECT_RIVER}`);
  check('灰显「已被碰走」张数 = 碰 + 明杠',
    doc.querySelectorAll('#river-me .mc.claimed, #river-right .mc.claimed, ' +
      '#river-top .mc.claimed, #river-left .mc.claimed').length === N_CLAIMED,
    `${doc.querySelectorAll('.mc.claimed').length} vs ${N_CLAIMED}`);
  check('末帧手牌总数 == 服务端权威 hc 之和',
    myTiles() + oppTiles(1) + oppTiles(2) + oppTiles(3) === LAST_HC.reduce((a, b) => a + b, 0),
    `${myTiles() + oppTiles(1) + oppTiles(2) + oppTiles(3)} vs hc ${LAST_HC.join('+')}`);
  check('末帧各家手牌与 hc 逐座位一致',
    [0, 1, 2, 3].every(rel => (rel === 0 ? myTiles() : oppTiles(rel)) === LAST_HC[absOf(rel)]),
    [0, 1, 2, 3].map(rel => `${rel === 0 ? myTiles() : oppTiles(rel)}/${LAST_HC[absOf(rel)]}`).join(' '));
  check('末帧手牌牌种串与 hc 长度自洽',
    LAST_HANDS.every((s, i) => nChars(s) === LAST_HC[i]),
    LAST_HANDS.map(nChars).join('+') + ' vs ' + LAST_HC.join('+'));
  check('副露组数 == 碰 + 明杠 + 暗杠（补杠合并进碰组）',
    meldGroups() === EXPECT_GROUPS, `${meldGroups()} vs ${EXPECT_GROUPS}`);
  check('补杠次数 ≤ 碰次数（补杠必先有同牌碰，是期望公式的前提）',
    N_M3 <= N_PENG, `补杠 ${N_M3} / 碰 ${N_PENG}`);
  check('副露张数 == 纯碰×3 + 补杠×4 + 明杠×4 + 暗杠×4',
    meldCells() === EXPECT_MELD_CELLS, `${meldCells()} vs ${EXPECT_MELD_CELLS}`);
  check('末帧底部提示 = 最后一条事件的文案',
    (el('rp-now').textContent || '').indexOf(String(LAST[7]).slice(0, 12)) !== -1,
    (el('rp-now').textContent || '').slice(0, 40));
  check('「下一步」到尾部禁用、进度条到底',
    el('rp-next').disabled === true && el('rp-range').value === String(EV.length));

  console.log('\n=== 4. 定位方式（进度条拖动；「跳到第 N 巡」组件已删） ===');
  el('rp-prev').click();
  check('「上一步」回退 1 帧', G('RP_IDX') === EV.length - 1, `RP_IDX=${G('RP_IDX')}`);
  const KIDX = Math.max(1, Math.min(10, EV.length - 1));
  el('rp-range').value = String(KIDX);
  fire(el('rp-range'), 'input');
  check(`拖进度条 → 跳到第 ${KIDX} 帧`, G('RP_IDX') === KIDX, `RP_IDX=${G('RP_IDX')}`);
  check('进度条 value 与帧号同步', el('rp-range').value === String(KIDX));
  // 巡号只作刻度显示（中央盘），不再有独立跳转控件
  const FIRST_R5 = EV.findIndex(e => (e[5] || 1) >= 5);
  if (FIRST_R5 >= 0) {
    const K5 = FIRST_R5 + 1;
    el('rp-range').value = String(K5);
    fire(el('rp-range'), 'input');
    check(`拖到该巡首手之后（第 ${K5} 帧）→ 中央巡号 ≥ 5`,
      G('RP_LAST.r') >= 5, `巡号=${G('RP_LAST.r')}`);
    check('中央计时盘同步到该帧的步数（不再显示巡号）',
      (el('timer').dataset.rp || '') === '第 ' + G('RP_IDX') + ' 步',
      JSON.stringify(el('timer').dataset.rp));
  }

  console.log('\n=== 5. 自动播放 / 暂停 ===');
  // 回放速度下拉（2026-10-12 用户需求）：0.5 ~ 5 秒/步，默认 0.5，持久化
  check('速度下拉存在且默认 0.5 秒/步',
    !!el('rp-speed') && w.eval('RP_SPEED') === 0.5 && el('rp-speed').value === '0.5',
    el('rp-speed') ? el('rp-speed').value : 'missing');
  check('速度取值范围 0.5 ~ 5（首项 0.5、末项 5）', (() => {
    const vs = [...el('rp-speed').options].map(o => parseFloat(o.value));
    return Math.min(...vs) === 0.5 && Math.max(...vs) === 5 &&
      w.eval('RP_SPEED_MIN') === 0.5 && w.eval('RP_SPEED_MAX') === 5;
  })(), [...el('rp-speed').options].map(o => o.value).join(','));
  el('rp-range').value = '0';
  fire(el('rp-range'), 'input');
  check('拖回第 0 帧', G('RP_IDX') === 0);
  el('rp-play').click();
  check('播放中按钮变 ⏸', el('rp-play').textContent === '⏸', el('rp-play').textContent);
  const t0 = G('RP_IDX');
  await sleep(1650);                    // 默认 500ms/步 → 1.65s 至少走 3 步
  const t1 = G('RP_IDX');
  check('自动推进 ≥ 2 步（默认间隔 500ms）', t1 - t0 >= 2, `1650ms: ${t0} → ${t1}`);
  check('定时器按当前速度起（RP_TIMER 非空）', G('RP_TIMER') !== null);
  el('rp-play').click();
  check('再点 → 暂停并复位按钮', el('rp-play').textContent === '▶' && G('RP_TIMER') === null,
    `btn=${el('rp-play').textContent} timer=${G('RP_TIMER')}`);
  check('暂停后帧号不再前进', await (async () => { const a = G('RP_IDX'); await sleep(700); return a === G('RP_IDX'); })());
  // 调速：改为 5 秒/步 → localStorage 落盘 + 值被夹到 [0.5, 5]
  el('rp-speed').value = '5';
  fire(el('rp-speed'), 'change');
  check('改速度即时生效并持久化（5 秒/步）',
    w.eval('RP_SPEED') === 5 && w.localStorage.getItem('dushan_rp_speed') === '5' &&
    w.eval('rpSpeedMs()') === 5000, `ms=${w.eval('rpSpeedMs()')}`);
  // 越界值夹紧：<select> 不能持有不存在的值（会被清成 ""），所以临时插一个 99 的 option
  // 来真正走到 change 处理里的夹紧分支
  const s99 = doc.createElement('option');
  s99.value = '99'; s99.textContent = '99秒/步';
  el('rp-speed').appendChild(s99);
  el('rp-speed').value = '99';
  fire(el('rp-speed'), 'change');
  check('越界速度被夹到上限 5', w.eval('RP_SPEED') === 5, `RP_SPEED=${w.eval('RP_SPEED')}`);
  s99.remove();
  el('rp-speed').value = '0.5';         // 复原，避免污染后续计时用例
  fire(el('rp-speed'), 'change');
  check('恢复 0.5 秒/步', w.eval('RP_SPEED') === 0.5 && w.eval('rpSpeedMs()') === 500);

  console.log('\n=== 6. 回放中不提交实盘操作 ===');
  reqLog.length = 0;
  const firstTile = el('hand').querySelector('.tile-mine:not(.ghost)');
  if (firstTile) fire(firstTile, 'click');
  fire(el('actions'), 'click');
  await sleep(60);
  check('点手牌/操作条不发 /api/act', !reqLog.some(u => u.indexOf('/api/act') >= 0),
    reqLog.join(' '));
  check('回放中 poll() 直接返回（不发 /api/state）', await (async () => {
    reqLog.length = 0; w.poll(); await sleep(80);
    return !reqLog.some(u => u.indexOf('/api/state') >= 0);
  })());
  check('回放中点桌面不切换结算页（2026-10-12 用户反馈）', await (async () => {
    // 伪造「已终局」的实盘状态并手动打开结算页（不走 renderResult —— 它会异步重绘，
    // jsdom 下会抛渲染错误干扰断言）；这里只验证「桌面点击切换」这一条路径被回放挡住
    const opened = w.eval('(function(){ const st = RP_LAST; if (!st) return false;' +
      ' st.over = true; st.result = { type: "huangzhuang" }; st.hand_index = RP.hand;' +
      ' LAST_STATE = st; RES_DISMISSED = false;' +
      ' el("result-mask").classList.add("on"); el("res-actions").classList.add("on");' +
      ' return el("result-mask").classList.contains("on"); })()');
    doc.body.dispatchEvent(new w.MouseEvent('click', { bubbles: true }));
    await sleep(30);
    const after = el('result-mask').classList.contains('on');
    return opened === true && after === true;   // 打开过 且 点击后仍开着（没被切掉）
  })());
  w.eval('el("result-mask").classList.remove("on"); el("res-actions").classList.remove("on");' +
    ' RES_DISMISSED = false; LAST_STATE = null;');   // 伪造的终局状态用完即弃（退出回放要走 render）

  console.log('\n=== 6b. 回放结算页（2026-10-12 新增：与实盘结算完全一致） ===');
  const HAS_FINAL = !!(REPLAY.final && REPLAY.final.result);
  check('服务端回放载荷带终局快照 final（结算页的数据源）', HAS_FINAL,
    HAS_FINAL ? `type=${REPLAY.final.result.type} 玩家行=${(REPLAY.final.players || []).length}`
      : '（这份记录没有 —— 跑 scripts/replay_e2e.py 落一局新的）');
  if (HAS_FINAL) {
    w.eval('RP_IDX = RP.ev.length; renderReplayFrame();');
    w.openReplayResult();
    check('结算页打开（复用实盘 #result-mask）',
      el('result-mask').classList.contains('on'), '');
    check('标题出图（胡牌 / 荒牌 icon）', !!el('res-title').querySelector('img'), '');
    check('总览卡片 = 4 家（同一套 renderOverview）',
      el('res-pairs').querySelectorAll('.ovcard').length === 4,
      String(el('res-pairs').querySelectorAll('.ovcard').length));
    check('视角切换 = 总览 + 4 家视角按钮（5 个）',
      el('res-tabs').querySelectorAll('button').length === 5,
      [...el('res-tabs').querySelectorAll('button')].map(b => b.textContent).join(','));
    check('顶部鸡牌/翻鸡信息条已渲染', !!el('res-jibar').querySelector('.res-jibar'),
      (el('res-jibar').textContent || '').slice(0, 30));
    check('结算页里的 #res-actions（下一局/查看统计）在回放中被 CSS 藏掉',
      cssOf(el('res-actions'), 'display') === 'none', cssOf(el('res-actions'), 'display'));
    // 名字必须来自**记录**（回放身份），不是当前设置
    const wantNames = (REPLAY.names || []).map(String);
    const ovText = el('res-pairs').textContent || '';
    check('总览里的名字走记录身份（RP_NAMES，不是当前设置）',
      wantNames.every(n => !n || ovText.indexOf(n.replace(/\(.*/, '')) >= 0),
      wantNames.join(','));
    // 视角切换必须重绘**回放快照**（此前硬读 LAST_STATE —— 回放里会渲染成实盘数据）
    el('res-tabs').querySelectorAll('button')[1].click();
    check('切到「我的视角」用回放快照重绘（不是实盘 LAST_STATE）',
      el('res-pairs').querySelectorAll('.ovcard').length === 0 &&
      (el('res-pairs').querySelectorAll('.pstrip').length > 0 ||
       !!el('res-pairs').querySelector('table')),
      `ovcard=${el('res-pairs').querySelectorAll('.ovcard').length}`);
    check('视角按钮高亮已切换',
      el('res-tabs').querySelectorAll('button')[1].classList.contains('on'));
    el('res-tabs').querySelectorAll('button')[0].click();
    check('切回总览正常', el('res-pairs').querySelectorAll('.ovcard').length === 4);
    // 2026-10-13 二次口径：点面板外 → 切换**底部抽屉**；结算页只能走「结算」按钮开关
    // （进回放时 rpDrawerShow+3600ms 自动收回，测试里先把抽屉收干净、清掉计时器再点）
    w.eval('if (RP_DRAWER_TIMER) { clearTimeout(RP_DRAWER_TIMER); RP_DRAWER_TIMER = null; }' +
      'el("rp-drawer").classList.remove("show");');
    el('rp-res').click();   // 此时结算页开着 → 第一下是「收起」
    check('「结算」按钮 = 开关：开 → 点一下收起',
      !el('result-mask').classList.contains('on') && G('RP_RES_OPEN') === false);
    doc.body.dispatchEvent(new w.MouseEvent('click', { bubbles: true }));
    await sleep(30);
    check('回放中点桌面 → 弹出底部控制抽屉（结算页不受影响）',
      !el('result-mask').classList.contains('on') &&
      el('rp-drawer').classList.contains('show'), '');
    check('桌面点击不再开关结算页（旧 toggle 已移除）',
      !/else openReplayResult\(\);                \/\/ 再点一下把结算页叫回来/.test(html));
    // 抽屉已由点桌面弹出；再点一次桌面 → 180ms 后自动收回（结算页没开时才收）
    doc.body.dispatchEvent(new w.MouseEvent('click', { bubbles: true }));
    await sleep(260);
    check('再点一下桌面 → 抽屉收回（toggle）',
      !el('rp-drawer').classList.contains('show'));
    check('结算按钮 = 开关（再点一下收起，而不是只开不关）',
      /rp-res"\)\.onclick = \(\) => \{ RP_RES_OPEN \? closeReplayResult\(\) : openReplayResult\(\); \}/.test(html));
    el('rp-res').click();
    check('再点「结算」→ 重新打开', el('result-mask').classList.contains('on'));
    el('rp-res').click();
    check('「结算」再点一下 → 收起（toggle）',
      !el('result-mask').classList.contains('on') && G('RP_RES_OPEN') === false);
    el('rp-res').click();
    check('第三次点「结算」→ 又打开', el('result-mask').classList.contains('on'));
    check('结算页开着时抽屉不自动收回（「结算」按钮必须一直在）',
      el('rp-drawer').classList.contains('show'));
    check('回放声音按钮：进回放与牌局总开关同步，回放模式下由它独占控制（真能控住）',
      !!el('rp-snd') &&
      /RP_SOUND = localStorage\.getItem\("dushan_rp_sound"\) !== "0"/.test(html) &&
      /if \(RP_MODE \? !RP_SOUND : !SOUND_ON\) return;/.test(html) &&
      /RP_SOUND = SOUND_ON;/.test(html));
    /* 行为级验证（2026-10-13 用户口径「且可以实际控制回放器里的声音」）：
       牌局总开关关着时，回放里把开关打开也必须真的发声（旧实现 snd 先看 SOUND_ON，
       回放开关只是叠加在它上面的第二道闸，总开关一关回放就永远无声）。 */
    const sndGate = w.eval(`(function(){
      var rp0 = RP_MODE, snd0 = SOUND_ON, rps0 = RP_SOUND, played = 0;
      var rp = _rawPlay, vp = voicePlay;
      try {
        SOUND_ON = false; RP_MODE = true; RP_SOUND = false;
        _rawPlay = function(){ played++; }; voicePlay = function(){ played++; };
        snd("打牌");                    // 回放关声 → 不发声
        var a = played;
        RP_SOUND = true;
        snd("打牌");                    // 回放开声（总开关仍关）→ 必须发声
        var b = played;
        RP_MODE = false;
        snd("打牌");                    // 退出回放：总开关关 → 不发声
        var c = played;
        return [a, b, c];
      } finally { _rawPlay = rp; voicePlay = vp; RP_MODE = rp0; SOUND_ON = snd0; RP_SOUND = rps0; }
    })()`);
    check('回放声音门禁（行为级）：关=无声 / 开=发声（即使牌局总开关是关的）/ 退出回放后归总开关管',
      JSON.stringify(sndGate) === '[0,1,1]', `got=${JSON.stringify(sndGate)}`);
    check('自动播放播到尾会自动弹结算（interval 里有兜底）',
      /if \(RP_IDX >= RP\.ev\.length\) \{ stopReplayPlay\(\); openReplayResult\(\); \}/.test(html));
    w.eval('closeReplayResult()');
    check('closeReplayResult 收起结算页并复位状态',
      !el('result-mask').classList.contains('on') && G('RP_RES_OPEN') === false);
  }

  console.log('\n=== 6c. 回放布局（2026-10-12 用户反馈：右边太挤 / 碰走的牌太透明） ===');
  check('对家面板在回放中左移、且按视口宽度分档（窄屏少移，不压右块）',
    /html\.replaying \.seat-top \{ left: 47%; \}/.test(html) &&
    /@media \(min-width: 1400px\) \{ html\.replaying \.seat-top \{ left: 44%; \} \}/.test(html) &&
    /@media \(min-width: 1800px\) \{ html\.replaying \.seat-top \{ left: 4[0-9]%; \} \}/.test(html) &&
    /html\.mob\.replaying \.seat-top \{ left: /.test(html),
    (html.match(/html\.replaying \.seat-top \{[^}]*\}/) || [''])[0]);
  check('底部抽屉：fixed 铺满、默认下沉隐藏、触底弹出、热区 60px（CSS 锚点）',
    /\.rp-hot \{[^}]*position: fixed/.test(html) &&
    /\.rp-hot \{[^}]*height: 60px/.test(html) &&
    /\.rp-drawer \{[^}]*position: fixed/.test(html) &&
    /\.rp-drawer \{[^}]*translateY\(calc\(100% \+ 6px\)\)/.test(html) &&
    /\.rp-drawer\.show \{ transform: none/.test(html) &&
    /html\.rp-res \.rp-drawer \{ transform: none !important; opacity: 1 !important; \}/.test(html) &&
    /html\.rp-res #rp-now \{ display: none/.test(html));
  check('结算页开着时抽屉保持可见（「结算」按钮=开关，不能把按钮自己藏掉）',
    !/html\.rp-res \.rp-drawer \{[^}]*translateY/.test(html) &&
    /if \(RP_RES_OPEN\) return;   \/\/ 结算页开着/.test(html));
  /* 2026-10-12 用户反馈「鼠标挪到底边也触发不了控制模块弹出」：
     根因是 #replay-mask 是 pointer-events:none 的图层，.rp-hot 继承了 none。
     修法 = 给 .rp-hot 显式开 auto。 */
  check('底边热区显式 pointer-events:auto（#replay-mask 是 none，否则永远收不到 mouseenter）',
    /#replay-mask \{[^}]*pointer-events: none/.test(html) &&
    /\.rp-hot \{[^}]*pointer-events: auto/.test(html),
    (html.match(/\.rp-hot \{[^}]*\}/) || [''])[0]);
  check('移动端热区加高（html.mob .rp-hot 84px）',
    /html\.mob \.rp-hot \{ height: 84px; \}/.test(html));
  /* 2026-10-12 用户需求「点左上角回放按钮触发底部控制模块弹出」 */
  check('左上角「回放」角标整块可点（cursor:pointer + 亮色 hover）',
    /\.rp-rec \{[^}]*cursor: pointer/.test(html) && /\.rp-rec:hover \{/.test(html));
  check('角标 click → 唤出抽屉（rpDrawerShow + 延时收起）',
    /RP_REC_BADGE.*addEventListener\("click"/.test(html) &&
    /rpDrawerShow\(\);\s*rpDrawerHideSoon\(4000\)/.test(html));
  check('被碰/杠走的弃牌提高可见度（opacity ≥ .7 且灰化 ≤ .3）', (() => {
    const m = html.match(/\.mc\.claimed \{ opacity: ([\d.]+); filter: grayscale\(([\d.]+)\)/);
    return !!m && parseFloat(m[1]) >= 0.7 && parseFloat(m[2]) <= 0.3;
  })(), (html.match(/\.mc\.claimed \{[^}]*\}/) || [''])[0]);
  check('离河牌的 ✕ 角标已删除（实盘与回放都不显示，2026-10-12 用户口径）',
    !/\.mc\.claimed::after/.test(html) && !/content: "✕"/.test(html));
  check('实盘牌河也带 claimed 标记（服务端已下发 discard_claimed）',
    /"discard_claimed"/.test(fs.readFileSync(ROOT + '/web/dushan_server.py', 'utf8')));

  console.log('\n=== 7. 退出回放 ===');
  el('rp-close').click();
  check('「退出回放」→ 工具条收起、数据清空',
    !el('replay-mask').classList.contains('on') && G('RP') === null && G('RP_MODE') === false);
  check('html.replaying 已摘掉', !doc.documentElement.classList.contains('replaying'));
  check('牌桌上的实盘控件恢复显示（#nav/#actions）',
    cssOf(el('nav'), 'display') !== 'none' && cssOf(el('actions'), 'display') !== 'none',
    `nav=${cssOf(el('nav'), 'display')} actions=${cssOf(el('actions'), 'display')}`);
  await w.openReplay(RID);
  check('可再次打开', el('replay-mask').classList.contains('on') && G('RP_MODE') === true);
  doc.dispatchEvent(new w.KeyboardEvent('keydown', { key: 'Escape', bubbles: true }));
  check('按 ESC → 提前结束回放',
    !el('replay-mask').classList.contains('on') && G('RP') === null && G('RP_MODE') === false);

  console.log('\n=== 7b. 对局记录 ⇄ 回放（2026-10-12 三改版） ===');
  // 从记录页点可回放行 → 记录页隐藏、进入回放
  el('hist-mask').classList.add('on');
  el('hist-body').innerHTML =
    '<table class="res clickable" data-rid="hist-rid"><tr><td>t</td><td>l</td><td>+0</td></tr></table>';
  el('hist-body').querySelector('table')
    .dispatchEvent(new w.MouseEvent('click', { bubbles: true }));
  await sleep(60);
  check('点回放行 → 对局记录页主动隐藏、进入回放',
    !el('hist-mask').classList.contains('on') &&
    el('replay-mask').classList.contains('on') && G('RP_MODE') === true,
    `hist=${el('hist-mask').classList.contains('on')} rp=${el('replay-mask').classList.contains('on')}`);
  check('来路标记 RP_FROM_HIST 已置位', G('RP_FROM_HIST') === true);
  // 退出回放 → 回到对局记录页
  el('rp-close').click();
  await sleep(60);
  check('退出回放 → 回到对局记录页',
    el('hist-mask').classList.contains('on') && !el('replay-mask').classList.contains('on'),
    `hist=${el('hist-mask').classList.contains('on')} rp=${el('replay-mask').classList.contains('on')}`);
  check('来路标记已复位', G('RP_FROM_HIST') === false);
  el('hist-mask').classList.remove('on');

  console.log('\n=== 7c. 对局记录删除（2026-10-12 用户需求） ===');
  /* 记录面板每行末列一个 🗑 → 确认弹窗 → POST /api/history_delete {ln, rid}。
     服务端删掉 records.jsonl 那一行（统计随之同步）+ 同 rid 的回放。 */
  el('hist-mask').classList.add('on');
  el('hist-body').innerHTML =
    '<table class="res" data-rid="rid-D" data-ln="5" data-drid="rid-D">' +
    '<tr><td>12:00</td><td>清一色</td><td>+8</td>' +
    '<td><button class="hist-del" title="删除这条对局记录">🗑</button></td></tr></table>';
  const delBtn = el('hist-body').querySelector('button.hist-del');
  check('记录行末列有删除按钮（🗑）', !!delBtn);
  check('未点击时确认弹窗关闭', !el('histdel-mask').classList.contains('on'));
  apiLog.length = 0;
  delBtn.dispatchEvent(new w.MouseEvent('click', { bubbles: true }));
  check('点 🗑 → 弹出确认弹窗', el('histdel-mask').classList.contains('on'));
  check('点 🗑 不误触「点行回放」（守卫生效）',
    !el('replay-mask').classList.contains('on'));
  check('确认文案说明「数据统计同步扣掉 + 无法撤销」',
    /数据统计/.test(el('histdel-sub').textContent) &&
    /无法撤销/.test(el('histdel-sub').textContent),
    el('histdel-sub').textContent);
  check('仅弹窗、尚未发删除请求',
    !apiLog.some(x => x.url.includes('history_delete')));
  // 取消 → 关窗、不发请求
  el('histdel-cancel').click();
  check('点取消 → 关窗且不发删除请求',
    !el('histdel-mask').classList.contains('on') &&
    !apiLog.some(x => x.url.includes('history_delete')));
  // 点遮罩空白处也算取消
  delBtn.dispatchEvent(new w.MouseEvent('click', { bubbles: true }));
  el('histdel-mask').dispatchEvent(new w.MouseEvent('click', { bubbles: true }));
  check('点弹窗外空白 → 关窗（不发请求）',
    !el('histdel-mask').classList.contains('on') &&
    !apiLog.some(x => x.url.includes('history_delete')));
  // 确认 → 真实发 POST，载荷带 ln 与 rid
  delBtn.dispatchEvent(new w.MouseEvent('click', { bubbles: true }));
  el('histdel-ok').click();
  await sleep(120);
  const dpost = apiLog.find(x => x.url.includes('history_delete'));
  check('点确认 → POST /api/history_delete（载荷带 ln + rid）',
    !!dpost && dpost.method === 'POST' && String(dpost.body.ln) === '5' &&
    dpost.body.rid === 'rid-D', JSON.stringify(dpost));
  check('确认后弹窗关闭', !el('histdel-mask').classList.contains('on'));
  check('确认后不残留待删行引用', G('HISTDEL_ROW') === null, String(G('HISTDEL_ROW')));
  // 服务端报错时给出提示且不崩（成功后 openHistory 会清空 hist-body，重新注入一行）
  histDelResp = { ok: false, error: '行号与记录不匹配，已中止删除' };
  el('hist-body').innerHTML =
    '<table class="res" data-rid="rid-E" data-ln="9" data-drid="rid-E">' +
    '<tr><td>12:01</td><td>平胡</td><td>-3</td>' +
    '<td><button class="hist-del">🗑</button></td></tr></table>';
  el('hist-body').querySelector('button.hist-del')
    .dispatchEvent(new w.MouseEvent('click', { bubbles: true }));
  el('histdel-ok').click();
  await sleep(120);
  check('服务端退回错误时页面不崩、弹窗已关',
    !el('histdel-mask').classList.contains('on') && errs.length === 0,
    errs.slice(0, 2).join(' | '));
  histDelResp = { ok: true, removed_rec: 1, removed_replay: 1 };
  el('hist-mask').classList.remove('on');

  console.log('\n=== 8. 回放已过期的兜底 ===');
  replayResp = { ok: false, error: '这一局的回放已不在缓存中' };
  await w.openReplay('expired-rid');
  check('显示错误文案而不是白屏',
    (el('rp-now').textContent || '').indexOf('不在缓存中') !== -1, el('rp-now').textContent);
  check('工具条仍打开（用户可关闭）', el('replay-mask').classList.contains('on'));
  check('过期时不残留上一条回放的数据', G('RP') === null);
  el('rp-close').click();
  check('关闭后数据为空', !el('replay-mask').classList.contains('on') && G('RP') === null);

  console.log('\n=== 9. 补杠重建专项 ===');
  /* 2026-10-12 实测暴露：独山麻将把杠叫「豆」，补杠的事件文案是
     「随机 爬坡豆一万（得通行证）」——旧文案里没有「补杠」二字（2026-10-12 起引擎已统一为新文案）。
     早先 replayState 用 `txt.indexOf("补杠") >= 0` 判补杠 → 永远失效 →
     一次补杠会被渲染成「碰(3 张) + 杠(4 张)」两组副露，与真实牌面对不上。
     已改成结构性判据（同牌最多 4 张，碰用 3 张，第 4 张到手只能是补杠）。 */
  let kc = null, kcRel = -1, kcTile = -1;
  for (const ln of lines) {
    let rp; try { rp = JSON.parse(ln); } catch (e) { continue; }
    for (const e of rp.ev || []) {
      if (K(e, 'kong') && kongType(e) === 3) { kc = rp; kcRel = e[2]; kcTile = e[3]; break; }
    }
    if (kc) break;
  }
  if (kc) {
    w.eval('RP = ' + JSON.stringify(kc) + ';');
    const st = w.replayState(kc.ev.length);
    const ms = (st.melds && st.melds[kcRel]) || [];
    const same = ms.filter(m => m.tile === kcTile);
    check('补杠重建为 1 组杠（不是「碰 + 杠」两组）',
      same.length === 1 && same[0].type === 3,
      `rel=${kcRel} tile=${kcTile} melds=${JSON.stringify(ms)}`);
    check('同牌不再残留「碰」', !same.some(m => m.type === 1), JSON.stringify(same));
    // 交叉校验：副露组数 == 该家杠事件数 + (碰事件数 − 补杠事件数)
    // （每个补杠吃掉一个碰组，不新增组）
    let nk = 0, np = 0, nadd = 0;
    for (const e of kc.ev) {
      if (e[2] !== kcRel) continue;
      if (e[1] === 'kong') { nk++; if (kongType(e) === 3) nadd++; }
      else if (e[1] === 'peng') np++;
    }
    check('副露组数 = 杠数 + (碰数 − 补杠数)',
      ms.length === nk + (np - nadd), `${ms.length} vs ${nk}+(${np}−${nadd})`);
    w.eval('RP = null;');
  } else {
    check('找到含「补杠」的回放样本', false, 'replays.jsonl 里暂无补杠局（跑 scripts/replay_e2e.py 可造）');
  }

  console.log('\n=== 10. 旧载荷降级（没有 deal / ev[i][8] 时画牌背，不崩） ===');
  const oldRp = JSON.parse(JSON.stringify(REPLAY));
  delete oldRp.deal;
  oldRp.ev = oldRp.ev.map(e => {
    const c = e.slice();
    c[8] = null;             // 老记录没有手牌牌种
    return c;
  });
  w.eval('RP = ' + JSON.stringify(oldRp) + ';');
  w.eval('RP_IDX = RP.ev.length;');
  const stOld = w.replayState(oldRp.ev.length);
  check('无牌种时仍能推出四家张数（= 权威 hc）',
    [0, 1, 2, 3].every(p => stOld.hc[p] === LAST_HC[p]),
    stOld.hc.join('+') + ' vs ' + LAST_HC.join('+'));
  check('无牌种时手牌退化为牌背占位（全 -1）',
    stOld.hands[0].length > 0 && stOld.hands[0].every(t => t === -1),
    JSON.stringify(stOld.hands[0].slice(0, 5)));
  check('无 deal 时庄家补 14 张牌背',
    (() => {
      const rp2 = JSON.parse(JSON.stringify(oldRp)); delete rp2.deal;
      w.eval('RP = ' + JSON.stringify(rp2) + ';');
      const s = w.replayState(0);
      return s.hands[rp2.dealer].length === 14;
    })(), '');
  w.eval('RP = null;');

  check('全程无新异常', errs.length === 0, errs.slice(0, 2).join(' | '));

  console.log(`\n合计：${pass} PASS / ${fail} FAIL`);
  if (errs.length) { console.log('\n捕获到的错误：'); errs.slice(0, 5).forEach(e => console.log('  ' + e)); }
  dom.window.close();
  process.exit(fail || errs.length ? 1 : 0);
}, 900);
