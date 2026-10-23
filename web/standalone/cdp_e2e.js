/* CDP 端到端：开局 → 自动替人类出牌 → 等到终局/下一局，全程抓异常 */
const http = require("http");
const { execFile } = require("child_process");

const EDGE = "C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe";
const PORT = 9225;
const PAGE = "file:///C:/Users/LIGF/WorkBuddy/2026-09-30-12-49-43/zhuoji-mahjong/web/standalone/dushan_standalone_single.html";

function get(path) {
  return new Promise((res, rej) => {
    http.get({ host: "127.0.0.1", port: PORT, path }, r => {
      let d = ""; r.on("data", c => d += c); r.on("end", () => res(d));
    }).on("error", rej);
  });
}
const sleep = ms => new Promise(r => setTimeout(r, ms));

(async () => {
  const prof = "C:/Users/LIGF/AppData/Local/Temp/edge_cdp_e2e_" + Date.now();
  const child = execFile(EDGE, ["--headless=new", "--disable-gpu", "--no-first-run",
    `--remote-debugging-port=${PORT}`, `--user-data-dir=${prof}`, "about:blank"]);
  child.stderr.on("data", () => {});
  let targets = null;
  for (let i = 0; i < 40; i++) {
    try { const v = JSON.parse(await get("/json/list")); if (v.length) { targets = v; break; } } catch {}
    await sleep(250);
  }
  const page = targets.find(t => t.type === "page");
  const ws = new WebSocket(page.webSocketDebuggerUrl);
  await new Promise((res, rej) => { ws.onopen = res; ws.onerror = rej; });
  let mid = 0; const pend = new Map(); const excs = [];
  ws.onmessage = ev => {
    const m = JSON.parse(ev.data);
    if (m.id && pend.has(m.id)) { pend.get(m.id)(m); pend.delete(m.id); return; }
    if (m.method === "Runtime.exceptionThrown")
      excs.push(m.params.exceptionDetails.exception && m.params.exceptionDetails.exception.description || m.params.exceptionDetails.text);
  };
  const send = (method, params = {}) => new Promise(res => { const id = ++mid; pend.set(id, res); ws.send(JSON.stringify({ id, method, params })); });
  const evalp = async expr => {
    const r = await send("Runtime.evaluate", { expression: expr, returnByValue: true, awaitPromise: true });
    return r.result && r.result.result && r.result.result.value;
  };

  await send("Runtime.enable"); await send("Page.enable");
  await send("Page.navigate", { url: PAGE });
  await sleep(3000);

  await evalp(`window.__DushanReady.then(() => 'ok')`);
  console.log("[ready] model session OK");
  await evalp(`document.getElementById('startBtn').click()`);
  await sleep(2000);

  // 自动出牌循环：轮到人类就打第一张合法牌，直到本局结束
  const t0 = Date.now();
  let done = null;
  while (Date.now() - t0 < 180000) {
    const r = await evalp(`(async () => {
      const s = await window.__LocalAPI('/api/state?sid=' + SID);
      if (s.ok === false) return 'state_err:' + (s.error || '');
      if (s.over) return 'OVER:' + JSON.stringify((s.result && s.result.scores) || s.result || {}).slice(0, 200);
      if (s.turn !== s.my_seat || !s.you || !s.you.legal) return 'wait';
      const acts = s.you.legal;
      const pick = acts.find(a => a.kind === "hu") || acts.find(a => a.kind === "discard") || acts[0];
      const r2 = await window.__LocalAPI('/api/act', { sid: SID, kind: pick.kind, tile: pick.tile });
      return r2 && r2.ok === false ? 'act_err:' + (r2.error || '') : 'acted:' + pick.kind + ':' + pick.tile;
    })()`);
    if (String(r).startsWith("OVER")) { done = r; break; }
    if (String(r).startsWith("state_err") || String(r).startsWith("act_err")) { console.log(r); break; }
    await sleep(400);
  }
  console.log("[autoplay]", done || "180s 未终局（正常：独山一局可达数分钟）");
  const fin = await evalp(`JSON.stringify({ wall: (document.getElementById('wall-num')||{}).textContent,
    logCount: (document.getElementById('log')||{dataset:{}}).dataset.count })`);
  console.log("[final]", fin);
  console.log("[exceptions]", excs.length ? excs.slice(0, 5) : "(none)");
  ws.close(); child.kill(); process.exit(0);
})().catch(e => { console.error("fail:", e); process.exit(1); });
