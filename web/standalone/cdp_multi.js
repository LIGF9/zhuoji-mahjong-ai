/* CDP 多局回归：连打 N 局，每局结束后核验结算面板，进入下一局后核验铺开；截图关键帧 */
const http = require("http");
const { execFile } = require("child_process");
const fs = require("fs");

const EDGE = "C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe";
const PORT = 9228;
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
  const prof = "C:/Users/LIGF/AppData/Local/Temp/edge_cdp_multi_" + Date.now();
  const child = execFile(EDGE, ["--headless=new", "--disable-gpu", "--no-first-run",
    `--remote-debugging-port=${PORT}`, `--user-data-dir=${prof}`, "about:blank", "--window-size=1400,900"]);
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
  const shot = async name => {
    const r = await send("Page.captureScreenshot", { format: "png" });
    if (r.result && r.result.data) fs.writeFileSync(`reports/_shot_${name}.png`, Buffer.from(r.result.data, "base64"));
  };

  await send("Runtime.enable"); await send("Page.enable");
  await send("Page.navigate", { url: PAGE });
  await sleep(3000);
  await evalp(`window.__DushanReady.then(() => 'ok')`);
  await evalp(`document.getElementById('startBtn').click()`);
  await sleep(1500);

  const HANDS = 6;
  for (let h = 1; h <= HANDS; h++) {
    // 打到终局
    const t0 = Date.now();
    let over = false;
    while (Date.now() - t0 < 240000) {
      const r = await evalp(`(async () => {
        const s = await window.__LocalAPI('/api/state?sid=' + SID);
        if (!s || s.ok === false) return 'err:' + (s && s.error);
        if (s.over) return 'over';
        if (s.turn !== s.my_seat || !s.you || !s.you.legal) return 'wait';
        // 人为不胡（除非只剩 hu）：增加流局概率，覆盖两种终局
        const a = (h % 2 === 0 && s.you.legal.length > 1)
          ? s.you.legal.find(a => a.kind === "discard") || s.you.legal[0]
          : (s.you.legal.find(a => a.kind === "hu") || s.you.legal.find(a => a.kind === "discard") || s.you.legal[0]);
        await window.__LocalAPI('/api/act', { sid: SID, kind: a.kind, tile: a.tile });
        return 'acted';
      })()`);
      if (String(r).startsWith("err")) { console.log(`hand${h}`, r); break; }
      if (r === "over") { over = true; break; }
      await sleep(300);
    }
    await sleep(1200);
    const endInfo = await evalp(`(async () => {
      const s = await window.__LocalAPI('/api/state?sid=' + SID);
      return JSON.stringify({ hand: s.hand_index, over: s.over, type: s.result && s.result.type,
        maskOn: document.getElementById('result-mask').classList.contains('on') });
    })()`);
    console.log(`[hand${h} end]`, over, endInfo);
    await shot(`hand${h}_settle`);

    if (h === HANDS) break;
    // 点下一局，等特效释放后核验铺开
    await evalp(`document.getElementById('btn-next').click()`);
    let spread = false;
    for (let i = 0; i < 30; i++) {
      await sleep(1000);
      const s = await evalp(`JSON.stringify({ held: !!KAIJU_HELD, handIndex: LAST_STATE && LAST_STATE.hand_index,
        wall: document.getElementById('wall-num').textContent })`);
      const o = JSON.parse(s);
      if (!o.held && o.handIndex === h + 1) { spread = true; console.log(`[hand${h + 1} spread]`, s); break; }
      if (i === 29) console.log(`[hand${h + 1} spread] TIMEOUT`, s);
    }
    if (!spread) await shot(`hand${h + 1}_stuck`);
  }
  console.log("[exceptions]", excs.length ? excs.slice(0, 8) : "(none)");
  ws.close(); child.kill(); process.exit(0);
})().catch(e => { console.error("fail:", e); process.exit(1); });
