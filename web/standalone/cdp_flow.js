/* CDP 复现：① 终局结算面板是否弹出 ② 下一局特效后棋盘是否铺开 ③ 全程异常 */
const http = require("http");
const { execFile } = require("child_process");

const EDGE = "C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe";
const PORT = 9226;
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
  const prof = "C:/Users/LIGF/AppData/Local/Temp/edge_cdp_flow_" + Date.now();
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
      excs.push((m.params.exceptionDetails.exception && m.params.exceptionDetails.exception.description || m.params.exceptionDetails.text).split("\n").slice(0, 3).join(" | "));
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
  await evalp(`document.getElementById('startBtn').click()`);
  await sleep(1500);

  // 自动打完第 1 局
  const t0 = Date.now();
  let over = false;
  while (Date.now() - t0 < 180000) {
    const r = await evalp(`(async () => {
      const s = await window.__LocalAPI('/api/state?sid=' + SID);
      if (!s || s.ok === false) return 'err';
      if (s.over) return 'over';
      if (s.turn !== s.my_seat || !s.you || !s.you.legal) return 'wait';
      const a = s.you.legal.find(a => a.kind === "hu") || s.you.legal.find(a => a.kind === "discard") || s.you.legal[0];
      await window.__LocalAPI('/api/act', { sid: SID, kind: a.kind, tile: a.tile });
      return 'acted';
    })()`);
    if (r === "over") { over = true; break; }
    if (r === "err") { console.log("state err"); break; }
    await sleep(350);
  }
  await sleep(1500); // 等一次轮询渲染
  const after1 = await evalp(`JSON.stringify({
    over: LAST_STATE && LAST_STATE.over,
    hasResult: !!(LAST_STATE && LAST_STATE.result),
    resMaskOn: document.getElementById('result-mask').className,
    resActionsOn: document.getElementById('res-actions').className,
    handIndex: LAST_STATE && LAST_STATE.hand_index })`);
  console.log("[hand1 end]", over, after1);

  if (!over) { console.log("[exceptions]", excs.slice(0, 6)); ws.close(); child.kill(); process.exit(0); }

  // 点下一局，观察特效扣牌与释放
  await evalp(`document.getElementById('btn-next').click()`);
  for (const ms of [1000, 3000, 6000, 10000]) {
    await sleep(ms === 1000 ? 1000 : ms - (ms === 3000 ? 1000 : ms === 6000 ? 3000 : 6000));
    const s = await evalp(`JSON.stringify({
      t: ${ms},
      handIndex: LAST_STATE && LAST_STATE.hand_index,
      wall: document.getElementById('wall-num').textContent,
      logCount: document.getElementById('log').dataset.count,
      kaijuOn: document.getElementById('kaiju-fx') ? document.getElementById('kaiju-fx').className : '-',
      held: !!KAIJU_HELD, flowHold: SERVER_HOLD })`);
    console.log("[after next]", s);
  }
  console.log("[exceptions]", excs.length ? excs.slice(0, 8) : "(none)");
  ws.close(); child.kill(); process.exit(0);
})().catch(e => { console.error("fail:", e); process.exit(1); });
