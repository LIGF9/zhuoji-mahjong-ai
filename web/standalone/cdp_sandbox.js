/* 模拟沙箱环境：注入会抛 SecurityError 的 localStorage，验证 shim 救活页面 */
const http = require("http");
const { execFile } = require("child_process");

const EDGE = "C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe";
const PORT = 9229;
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
  const prof = "C:/Users/LIGF/AppData/Local/Temp/edge_cdp_sandbox_" + Date.now();
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
      excs.push((m.params.exceptionDetails.exception && m.params.exceptionDetails.exception.description || m.params.exceptionDetails.text).split("\n")[0]);
  };
  const send = (method, params = {}) => new Promise(res => { const id = ++mid; pend.set(id, res); ws.send(JSON.stringify({ id, method, params })); });
  const evalp = async expr => {
    const r = await send("Runtime.evaluate", { expression: expr, returnByValue: true, awaitPromise: true });
    return r.result && r.result.result && r.result.result.value;
  };

  await send("Runtime.enable"); await send("Page.enable");
  // 在任何页面脚本之前：把 localStorage 变成扔 SecurityError 的毒药（模拟沙箱 iframe）
  await send("Page.addScriptToEvaluateOnNewDocument", {
    source: `Object.defineProperty(window, "localStorage", {
      get() { throw new DOMException("Access denied for this document", "SecurityError"); }
    });`,
  });
  await send("Page.navigate", { url: PAGE });
  await sleep(3000);

  const boot1 = await evalp(`JSON.stringify({
    lsThrows: (() => { try { localStorage.getItem("x"); return false; } catch (e) { return true; } })(),
    shim: typeof window.__LS,
    shimWorks: (() => { try { window.__LS.setItem("t", "1"); return window.__LS.getItem("t") === "1"; } catch (e) { return "ERR"; } })(),
    tilesFilled: typeof META !== "undefined" && META && META.tiles && META.tiles.length || 0,
  })`);
  console.log("[sandbox env]", boot1);

  await evalp(`window.__DushanReady.then(() => 'ok')`);
  await evalp(`document.getElementById('startBtn').click()`);
  await sleep(8000);
  const st = await evalp(`JSON.stringify({
    SID: typeof SID !== "undefined" && SID, wall: document.getElementById("wall-num").textContent,
    logCount: document.getElementById("log").dataset.count })`);
  console.log("[game]", st);
  console.log("[exceptions]", excs.length ? excs.slice(0, 5) : "(none)");
  ws.close(); child.kill(); process.exit(0);
})().catch(e => { console.error("fail:", e); process.exit(1); });
