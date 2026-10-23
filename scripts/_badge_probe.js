// 决定性探针 v2：不跑整页渲染，直接调 settlementCtx + playerHandBlock
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
setTimeout(() => {
  const base = clone(OVER.find(o => o.state.result.type === 'win').state);
  const ME = 2;
  base.my_seat = ME;
  base.players.forEach((p, i) => { p.seat = (ME + i) % 4; });
  const wSeat = 1, lSeat = 3;
  const relOf = s => ((s - ME) % 4 + 4) % 4;
  const r = base.result;
  r.type = 'win'; r.how = '点炮'; r.is_tsumo = false;
  r.winners = [wSeat]; r.winner = wSeat; r.loser = lSeat;
  r.loser_name = base.players[relOf(lSeat)].name;

  w.__st = base;
  w.eval('window.__ctx = settlementCtx(window.__st)');
  console.log(`视角 my_seat=${ME}; 赢家=座位${wSeat}(${base.players[relOf(wSeat)].name}); 放炮者=座位${lSeat}(${base.players[relOf(lSeat)].name})`);
  console.log(`ctx.winTag="${w.eval('window.__ctx.winTag')}"  ctx.loseTag="${w.eval('window.__ctx.loseTag')}"`);
  for (let rel = 0; rel < 4; rel++) {
    const seat = base.players[rel].seat;
    const html2 = w.eval(`playerHandBlock(window.__st, ${rel}, window.__ctx).metaHtml`);
    const m1 = html2.match(/class="tag win"[^>]*>([^<]+)</);
    const m2 = html2.match(/class="tag lose"[^>]*>([^<]+)</);
    if (rel <= 1) console.log(`  [dbg rel=${rel}] len=${html2.length} head=${html2.slice(0, 160).replace(/\n/g, ' ')}`);
    if (m1 || m2)
      console.log(`  rel=${rel} seat=${seat} ${seat === wSeat ? '(赢家)' : seat === lSeat ? '(放炮者)' : ''} → ` +
        (m1 ? `WIN="${m1[1]}" ` : '') + (m2 ? `LOSE="${m2[1]}"` : ''));
  }
  process.exit(0);
}, 400);
