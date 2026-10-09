// 功能 e2e：自动操作健壮性（过期/非法推荐不得提交；被拒不得死循环）
//  复现的历史故障：碰牌后成胡（drawn=False，不可自摸胡），模型掩码错误地推荐 hu，
//  前端无胡按钮却自动提交 → 服务端「非法动作 hu:-1」→ 一直 toast、对局卡死。
const fs = require('fs');
const { JSDOM } = require('jsdom');

const ROOT = 'C:/Users/hyzor/WorkBuddy/2026-09-30-12-49-43/zhuoji-mahjong';
const html = fs.readFileSync(ROOT + '/web/static/dushan.html', 'utf8');
const errs = [];
const meta = JSON.parse(fs.readFileSync(ROOT + '/reports/ui_fixtures_meta.json', 'utf8'));
const mid0 = JSON.parse(fs.readFileSync(ROOT + '/reports/ui_fixtures_mid.json', 'utf8'))[0].state;

let actCalls = [];      // 记录 /api/act 提交的动作
let hintCalls = 0;
let actReply = { ok: true };
let hintPayload = { ok: true, head: 'model', model: 'm', value: 0.1, items: [], tip: '' };

const dom = new JSDOM(html, {
  runScripts: 'dangerously',
  pretendToBeVisual: true,
  url: 'http://127.0.0.1:8770/static/dushan.html',
  beforeParse(w) {
    w.fetch = (url, opt) => {
      const u = String(url);
      if (u.includes('/api/meta')) return Promise.resolve({ ok: true, json: () => Promise.resolve(meta) });
      if (u.includes('/api/hint')) {
        hintCalls++;
        return Promise.resolve({ ok: true, json: () => Promise.resolve(hintPayload) });
      }
      if (u.includes('/api/act')) {
        actCalls.push(JSON.parse(opt.body));
        return Promise.resolve({ ok: true, json: () => Promise.resolve(actReply) });
      }
      if (u.includes('/api/state')) return Promise.resolve({ ok: true, json: () => Promise.resolve({ ready: false }) });
      return new Promise(() => {});
    };
    w.addEventListener('error', e => errs.push('window.error: ' + ((e.error && e.error.stack) || e.message)));
  },
});
const w = dom.window;
const sleep = ms => new Promise(r => setTimeout(r, ms));
const results = [];
const check = (name, ok, extra = '') => { results.push(ok); console.log(`${name}: ${ok ? 'OK' : 'FAIL'}${extra ? ' | ' + extra : ''}`); };

(async () => {
  await sleep(1500);                        // 等 boot
  w.eval('SID = "e2e-guard"; SET.timeoutOn = false; SET.hintDelay = 0');

  // 构造「轮到我在响应阶段（可碰/过）」的决策点
  const st = JSON.parse(JSON.stringify(mid0));
  st.turn = st.my_seat;
  st.over = false;
  st.pending = { mode: 'respond', buttons: [{ kind: 'pass', tile: -1, label: '过' },
    { kind: 'peng', tile: 4, label: '碰' }], discardables: [], can_hint: true };
  const stDiscard = JSON.parse(JSON.stringify(st));
  stDiscard.pending = { mode: 'discard', buttons: [], discardables: [mid0.hand[0], mid0.hand[1]], can_hint: true };

  // ---- G1：过期推荐 hu（当前决策点根本没有胡）→ 不得提交 hu ----
  w.eval('SET.strategy = "model:master"');
  hintPayload = { ok: true, head: 'model', model: 'm', value: 0.1,
    items: [{ kind: 'peng', tile: 4, label: '碰', prob: 0.6 }], tip: '' };
  actCalls = []; hintCalls = 0; actReply = { ok: true };
  w.render(st);
  w.eval('RECOMMEND = { kind: "hu", tile: -1, label: "胡" }');   // 故意塞入过期/非法推荐
  await w.eval('autoAct()');
  await sleep(60);
  const sentKinds = actCalls.map(c => c.kind);
  check('G1 过期 hu 推荐不会提交', !sentKinds.includes('hu') && sentKinds.length === 1,
    `提交=${JSON.stringify(actCalls)} hint=${hintCalls}`);
  check('G1b 改提交当前决策点的合法动作', actCalls[0] && actCalls[0].kind === 'peng' && actCalls[0].tile === 4,
    JSON.stringify(actCalls[0] || {}));

  // ---- G2：推荐非法且取不到推荐 → 安全兜底（respond→过） ----
  hintPayload = { ok: false, error: '现在不是你的决策点' };
  actCalls = []; hintCalls = 0;
  w.render(st);
  w.eval('RECOMMEND = { kind: "hu", tile: -1, label: "胡" }; AUTO_FAIL_KEY = ""');
  await w.eval('autoAct()');
  await sleep(60);
  check('G2 无推荐时 respond 兜底为「过」',
    actCalls.length === 1 && actCalls[0].kind === 'pass', JSON.stringify(actCalls));

  // ---- G2b：出牌阶段兜底打第一张可打牌 ----
  actCalls = []; hintCalls = 0;
  w.render(stDiscard);
  w.eval('RECOMMEND = null');
  await w.eval('autoAct()');
  await sleep(60);
  check('G2b 出牌阶段兜底打牌',
    actCalls.length === 1 && actCalls[0].kind === 'discard' && actCalls[0].tile === mid0.hand[0],
    JSON.stringify(actCalls));

  // ---- G3：推荐合法 → 正常提交，且不误触发兜底 ----
  hintPayload = { ok: true, head: 'model', model: 'm', value: 0.1,
    items: [{ kind: 'pass', tile: -1, label: '过', prob: 0.8 }], tip: '' };
  actCalls = []; hintCalls = 0;
  w.render(st);
  w.eval('RECOMMEND = null; AUTO_FAIL_KEY = ""');
  await w.eval('requestHint()');
  await w.eval('autoAct()');
  await sleep(60);
  check('G3 合法推荐照常提交', actCalls.length === 1 && actCalls[0].kind === 'pass', JSON.stringify(actCalls));

  // ---- G4：被服务端拒绝 → 只提示一次、同一决策点不再自动重试 ----
  actReply = { ok: false, error: '非法动作 hu:-1' };
  actCalls = []; hintCalls = 0;
  w.render(st);
  w.eval('AUTO_FAIL_KEY = ""; SENT_FAIL_KEY = ""; RECOMMEND = { kind: "peng", tile: 4, label: "碰" }');
  const toastEl = w.document.getElementById('toast');
  await w.eval('doAct("peng", 4)');
  await sleep(60);
  const failKey = w.eval('AUTO_FAIL_KEY');
  const toasts1 = (w.document.body.innerHTML.match(/非法动作/g) || []).length;
  w.render(st);
  await w.eval('autoAct()');          // 同一决策点再触发自动操作
  await sleep(60);
  const toasts2 = (w.document.body.innerHTML.match(/非法动作/g) || []).length;
  check('G4 拒绝后记录失败决策点', !!failKey, String(failKey).slice(0, 24));
  check('G4b 拒绝只提示一次且不再自动重试',
    toasts1 <= 1 && actCalls.length === 1 && toasts2 <= 1,
    `act 次数=${actCalls.length} toast=${toasts1}/${toasts2}`);

  // ---- G5：recommendLegalIn 语义 ----
  w.render(st);
  w.eval('RECOMMEND = { kind: "peng", tile: 4 }');
  const l1 = w.eval('recommendLegalIn(LAST_STATE)');
  w.eval('RECOMMEND = { kind: "pass", tile: -1 }');
  const l2 = w.eval('recommendLegalIn(LAST_STATE)');
  w.eval('RECOMMEND = { kind: "discard", tile: 4 }');
  const l3 = w.eval('recommendLegalIn(LAST_STATE)');
  w.render(stDiscard);
  w.eval('RECOMMEND = { kind: "discard", tile: ' + mid0.hand[1] + ' }');
  const l4 = w.eval('recommendLegalIn(LAST_STATE)');
  check('G5 合法性校验（碰/过/越界/打牌）', l1 && l2 && !l3 && l4, `peng=${l1} pass=${l2} badDiscard=${l3} discard=${l4}`);

  console.log(`\nwindow errors: ${errs.length}${errs.length ? '\n' + errs.slice(0, 3).join('\n') : ''}`);
  const bad = results.filter(x => !x).length;
  console.log(`合计 ${results.length} 项，失败 ${bad}`);
  process.exit(bad === 0 && errs.length === 0 ? 0 : 1);
})();
