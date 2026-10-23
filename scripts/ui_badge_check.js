// 结算徽章方向的回归（2026-10-12 用户报告"点炮/捉炮反了"后固化）：
//   点炮局：赢家徽章 = 「捉炮」（win 样式·红橙框），放炮者徽章 = 「点炮」（lose 样式·蓝框）
//   自摸局：赢家徽章 = 「自摸」
// 背景：winHowTag 的"点炮→捉炮（赢家视角）"转换曾不存在于 git HEAD；用户在 16:22
// 截图看到反的徽章，与 HEAD 版及任何已知版本都不符 —— 判定为中间版本/缓存。
// 本文件锁死正确语义，谁改反了这里就红。
const fs = require('fs');
const { JSDOM } = require('jsdom');
const ROOT = 'C:/Users/LIGF/WorkBuddy/2026-09-30-12-49-43/zhuoji-mahjong';
const html = fs.readFileSync(ROOT + '/web/static/dushan.html', 'utf8');
const meta = JSON.parse(fs.readFileSync(ROOT + '/reports/ui_fixtures_meta.json', 'utf8'));
const OVER = JSON.parse(fs.readFileSync(ROOT + '/reports/ui_fixtures_over.json', 'utf8'));

const dom = new JSDOM(html, {
  runScripts: 'dangerously', pretendToBeVisual: true,
  url: 'http://127.0.0.1:8770/static/dushan.html',
  beforeParse(w) {
    if (!w.matchMedia) w.matchMedia = q => ({ matches: false, media: String(q),
      onchange: null, addEventListener() {}, removeEventListener() {},
      addListener() {}, removeListener() {}, dispatchEvent: () => false });
    w.fetch = (url) => String(url).includes('/api/meta')
      ? Promise.resolve({ ok: true, json: () => Promise.resolve(meta) })
      : new Promise(() => {});
  },
});
const w = dom.window;
const clone = o => JSON.parse(JSON.stringify(o));
let ok = 0, bad = 0;
const check = (name, cond, detail = '') => {
  (cond ? ok++ : bad++);
  console.log(`  [${cond ? 'ok  ' : 'FAIL'}] ${name}${detail ? '   ' + detail : ''}`);
};

setTimeout(() => {
  const winFixture = OVER.find(o => o.state.result.type === 'win');
  const base = clone(winFixture.state);
  const ME = 2;                                   // 非零视角：排掉 my_seat=0 恰好蒙对的情形
  base.my_seat = ME;
  base.players.forEach((p, i) => { p.seat = (ME + i) % 4; });
  const relOf = s => ((s - ME) % 4 + 4) % 4;

  function badges(r2) {
    w.__st = base;
    w.eval('window.__ctx = settlementCtx(window.__st)');
    const out = {};
    for (let rel = 0; rel < 4; rel++) {
      const seat = base.players[rel].seat;
      const m = String(w.eval(`playerHandBlock(window.__st, ${rel}, window.__ctx).metaHtml`));
      const mw = m.match(/class="tag win"[^>]*>([^<]+)</);
      const ml = m.match(/class="tag lose"[^>]*>([^<]+)</);
      out[seat] = { win: mw ? mw[1] : null, lose: ml ? ml[1] : null };
    }
    return out;
  }

  // ---- 用例 1：点炮局（seat1 胡 seat3 的炮）----
  let r = base.result;
  r.type = 'win'; r.how = '点炮'; r.is_tsumo = false;
  r.winners = [1]; r.winner = 1; r.loser = 3;
  r.loser_name = base.players[relOf(3)].name;
  let b = badges(r);
  check('点炮局 ctx.winTag=捉炮', w.eval('window.__ctx.winTag') === '捉炮');
  check('点炮局 ctx.loseTag=放炮（2026-10-13 责任口径）', w.eval('window.__ctx.loseTag') === '放炮');
  check('赢家(座1)徽章=捉炮(win样式)', b[1].win === '捉炮', `got ${JSON.stringify(b[1])}`);
  check('放炮者(座3)徽章=放炮(lose样式)', b[3].lose === '放炮', `got ${JSON.stringify(b[3])}`);

  // ---- 用例 2：自摸局（seat1 自摸）----
  r.how = '自摸'; r.is_tsumo = true; r.loser = null;
  b = badges(r);
  check('自摸局赢家徽章=自摸', b[1].win === '自摸', `got ${JSON.stringify(b[1])}`);

  // ---- 用例 3：热炮局（赢家=捉炮热炮者，放炮者=放热炮+全烧）----
  r.how = '热炮'; r.is_tsumo = false; r.loser = 3; r.void_name = base.players[relOf(3)].name;
  b = badges(r);
  check('热炮局赢家徽章=捉热炮', b[1].win === '捉热炮', `got ${JSON.stringify(b[1])}`);
  check('热炮局放炮者徽章=放热炮', b[3].lose === '放热炮', `got ${JSON.stringify(b[3])}`);

  console.log(`\n徽章方向回归：${ok} PASS / ${bad} FAIL`);
  process.exit(bad ? 1 : 0);
}, 400);
