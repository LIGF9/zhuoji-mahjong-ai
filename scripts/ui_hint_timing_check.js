// 功能 e2e：推荐延迟 + 简短大师建议（jsdom 加载真实页面）
const fs = require('fs');
const { JSDOM } = require('jsdom');

const ROOT = 'C:/Users/hyzor/WorkBuddy/2026-09-30-12-49-43/zhuoji-mahjong';
const html = fs.readFileSync(ROOT + '/web/static/dushan.html', 'utf8');
const errs = [];
const meta = JSON.parse(fs.readFileSync(ROOT + '/reports/ui_fixtures_meta.json', 'utf8'));
const mid0 = JSON.parse(fs.readFileSync(ROOT + '/reports/ui_fixtures_mid.json', 'utf8'))[0].state;

const HINT_PAYLOAD = {
  ok: true, head: 'model', model: 'dushan_master', value: 0.3,
  items: [
    { kind: 'discard', tile: mid0.hand[0], label: '打出 1万', prob: 0.55, tags: ['注意'], lines: ['打 1万 最稳'] },
    { kind: 'discard', tile: mid0.hand[1], label: '打出 2万', prob: 0.30, tags: [], lines: ['次选'] },
  ],
  tip: '保留中间搭子，1 万孤张价值最低',
};

let hintCalls = 0;
const dom = new JSDOM(html, {
  runScripts: 'dangerously',
  pretendToBeVisual: true,
  url: 'http://127.0.0.1:8770/static/dushan.html',
  beforeParse(w) {
    w.fetch = (url) => {
      const u = String(url);
      if (u.includes('/api/meta')) {
        return Promise.resolve({ ok: true, json: () => Promise.resolve(meta) });
      }
      if (u.includes('/api/hint')) {
        hintCalls++;
        return Promise.resolve({ ok: true, json: () => Promise.resolve(HINT_PAYLOAD) });
      }
      return new Promise(() => {});
    };
    w.addEventListener('error', e => errs.push('window.error: ' + ((e.error && e.error.stack) || e.message)));
  },
});
const w = dom.window;
const hasRec = () => w.eval('!!RECOMMEND');
const sleep = ms => new Promise(r => setTimeout(r, ms));

(async () => {
  await sleep(1500); // 等 boot 喂 meta
  w.eval('SID = "e2e"'); // 允许 requestHint 发请求

  // 构造"轮到我出牌"的决策点
  const st = JSON.parse(JSON.stringify(mid0));
  st.turn = st.my_seat;
  const r = w.document.getElementById('brief-tip');

  // --- 测试 1：hintDelay=0 立即推荐 ---
  w.eval('SET.hintDelay = 0; SET.reasonMode = "full"');
  hintCalls = 0;
  w.render(st);
  await sleep(400);
  const t1 = hintCalls >= 1 && hasRec();
  console.log(`T1 立即推荐: hintCalls=${hintCalls} RECOMMEND=${hasRec()} ${t1 ? 'OK' : 'FAIL'}`);

  // --- 测试 2：简短理由关闭 → brief-tip 隐藏 ---
  const t2 = r.style.display === 'none';
  console.log(`T2 简短理由关: display=${r.style.display} ${t2 ? 'OK' : 'FAIL'}`);

  // --- 测试 3：hintDelay=2 延迟推荐（新决策点触发调度） ---
  w.eval('SET.hintDelay = 2; RECOMMEND = null');
  hintCalls = 0;
  const st2 = JSON.parse(JSON.stringify(st));
  st2.hand_index = (st2.hand_index || 0) + 1;
  w.render(st2);
  await sleep(300);
  const early = hintCalls;
  const earlyRec = hasRec();
  await sleep(2200);
  const t3 = early === 0 && !earlyRec && hintCalls >= 1 && hasRec();
  console.log(`T3 延迟推荐: 0.3s时 calls=${early} rec=${earlyRec}；2.5s时 calls=${hintCalls} rec=${hasRec()} ${t3 ? 'OK' : 'FAIL'}`);

  // --- 测试 4：简短理由开启 → 只显示原因（无"大师建议："前缀、无牌名） ---
  w.eval('SET.reasonMode = "brief"; SET.hintDelay = 0');
  const st3 = JSON.parse(JSON.stringify(st2));
  st3.hand_index = (st3.hand_index || 0) + 1;
  hintCalls = 0;
  w.render(st3);
  await sleep(400);
  const txt = r.textContent;
  const t4 = r.style.display === 'block' && txt === '保留中间搭子，1 万孤张价值最低'
    && !txt.includes('大师建议') && !txt.includes('打出');
  console.log(`T4 简短理由开(仅原因): "${txt.slice(0, 42)}" ${t4 ? 'OK' : 'FAIL'}`);

  // --- 测试 4b：tip 为空 → 回退到首选动作 reason（仍不带前缀/牌名） ---
  const payloadBak = HINT_PAYLOAD.tip;
  HINT_PAYLOAD.tip = '';
  HINT_PAYLOAD.items[0].label = '打出 1万';
  HINT_PAYLOAD.items[0].reason = '打出 1万：保留中间搭子，孤张价值最低';
  const st4 = JSON.parse(JSON.stringify(st3));
  st4.hand_index = (st4.hand_index || 0) + 1;
  w.render(st4);
  await sleep(400);
  const txtB = r.textContent;
  const t4b = r.style.display === 'block' && txtB === '保留中间搭子，孤张价值最低' && !txtB.includes('打出');
  console.log(`T4b 无tip回退reason(剥牌名): "${txtB.slice(0, 42)}" ${t4b ? 'OK' : 'FAIL'}`);
  HINT_PAYLOAD.tip = payloadBak;
  HINT_PAYLOAD.items[0].reason = undefined;

  // --- 测试 5：回合结束 → brief-tip 隐藏 ---
  const over = JSON.parse(fs.readFileSync(ROOT + '/reports/ui_fixtures_over.json', 'utf8'))[0].state;
  w.render(over);
  const t5 = r.style.display === 'none';
  console.log(`T5 回合结束隐藏: display=${r.style.display} ${t5 ? 'OK' : 'FAIL'}`);

  console.log(`\nwindow errors: ${errs.length}${errs.length ? '\n' + errs.slice(0, 3).join('\n') : ''}`);
  process.exit(t1 && t2 && t3 && t4 && t4b && t5 && errs.length === 0 ? 0 : 1);
})();
