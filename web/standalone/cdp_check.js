/* 用 Node 22 内置 WebSocket 走 CDP：加载页面，抓 console/error，并模拟点击开局按钮 */
const http = require("http");
const { execFile } = require("child_process");

const EDGE = "C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe";
const PORT = 9223;
const URL_ = process.argv[2] || "file:///C:/Users/LIGF/WorkBuddy/2026-09-30-12-49-43/zhuoji-mahjong/web/standalone/dushan_standalone_single.html";

function get(path) {
  return new Promise((res, rej) => {
    http.get({ host: "127.0.0.1", port: PORT, path }, r => {
      let d = ""; r.on("data", c => d += c); r.on("end", () => res(d));
    }).on("error", rej);
  });
}
const sleep = ms => new Promise(r => setTimeout(r, ms));

(async () => {
  // 启动 Edge headless（复用用户 profile 会打架，用临时目录）
  const prof = "C:/Users/LIGF/AppData/Local/Temp/edge_cdp_" + Date.now();
  const child = execFile(EDGE, [
    "--headless=new", "--disable-gpu", "--no-first-run",
    `--remote-debugging-port=${PORT}`, `--user-data-dir=${prof}`,
    "about:blank",
  ]);
  child.stderr.on("data", () => {});
  // 等 devtools 端口就绪
  let targets = null;
  for (let i = 0; i < 40; i++) {
    try { const v = JSON.parse(await get("/json/list")); if (v.length) { targets = v; break; } } catch {}
    await sleep(250);
  }
  if (!targets) { console.error("CDP 端口未就绪"); process.exit(1); }
  const page = targets.find(t => t.type === "page");
  const ws = new WebSocket(page.webSocketDebuggerUrl);
  await new Promise((res, rej) => { ws.onopen = res; ws.onerror = rej; });

  let mid = 0; const pend = new Map();
  ws.onmessage = ev => {
    const m = JSON.parse(ev.data);
    if (m.id && pend.has(m.id)) { pend.get(m.id)(m); pend.delete(m.id); return; }
    if (m.method === "Runtime.consoleAPICalled") {
      const txt = m.params.args.map(a => a.value !== undefined ? String(a.value) : (a.description || a.type)).join(" ");
      console.log(`[console.${m.params.type}] ${txt}`);
    } else if (m.method === "Runtime.exceptionThrown") {
      const d = m.params.exceptionDetails;
      console.log(`[EXCEPTION] ${d.text} ${d.exception && d.exception.description || ""}`);
    }
  };
  const send = (method, params = {}) => new Promise(res => {
    const id = ++mid; pend.set(id, res);
    ws.send(JSON.stringify({ id, method, params }));
  });

  await send("Runtime.enable");
  await send("Page.enable");
  await send("Page.navigate", { url: URL_ });
  await sleep(2000); // 让脚本跑起来

  // 页面状态检查
  for (const expr of [
    "typeof ort", "typeof window.__DushanReady", "typeof window.__LocalAPI",
    "typeof el", "!!document.getElementById('startBtn')",
    "typeof el==='function' ? typeof el('startBtn').onclick : 'no el'",
  ]) {
    const r = await send("Runtime.evaluate", { expression: expr, returnByValue: true });
    console.log(`> ${expr} =`, JSON.stringify(r.result && r.result.result && r.result.result.value));
  }

  // 等 __DushanReady（最多 30s）
  const t0 = Date.now();
  while (Date.now() - t0 < 30000) {
    const r = await send("Runtime.evaluate", {
      expression: "window.__DushanReady ? (window.__DushanReady.then(a=>'ok').catch(e=>'ERR:'+e.message)) : 'pending'",
      returnByValue: true, awaitPromise: true,
    });
    const v = r.result && r.result.result && r.result.result.value;
    console.log("[ready?]", v);
    if (v === "ok" || String(v).startsWith("ERR")) break;
    await sleep(1000);
  }

  // 模拟点击开始对局
  const click = await send("Runtime.evaluate", {
    expression: `(async () => {
      try {
        const b = document.getElementById('startBtn');
        if (!b) return 'no startBtn';
        b.click();
        await new Promise(r => setTimeout(r, 3000));
        const boot = document.getElementById('boot');
        return JSON.stringify({ bootClass: boot && boot.className, toast: (document.getElementById('toast')||{}).textContent || '' });
      } catch (e) { return 'CLICK_ERR:' + (e && e.stack || e); }
    })()`,
    returnByValue: true, awaitPromise: true,
  });
  console.log("[click result]", JSON.stringify(click.result && click.result.result && click.result.result.value));

  // 观察 20 秒：模型座位应自动行动（墙数下降 / 流水增长），期间抓异常
  for (let k = 0; k < 5; k++) {
    await sleep(4000);
    const st = await send("Runtime.evaluate", {
      expression: `JSON.stringify({
        sid: typeof SID !== 'undefined' && SID, wall: (document.getElementById('wall-num')||{}).textContent,
        logCount: (document.getElementById('log')||{dataset:{}}).dataset.count,
        dcount: (document.getElementById('log-body')||{})._h ? (document.getElementById('log-body')._h.length) : 0
      })`, returnByValue: true, awaitPromise: true,
    });
    console.log(`[t+${(k + 1) * 4}s]`, JSON.stringify(st.result && st.result.result && st.result.result.value));
  }

  const st2 = await send("Runtime.evaluate", {
    expression: `typeof SID !== 'undefined' ? JSON.stringify({SID, MY_SEAT, poll: typeof POLL_TIMER !== 'undefined' && !!POLL_TIMER}) : 'SID undefined'`,
    returnByValue: true, awaitPromise: true,
  });
  console.log("[state]", JSON.stringify(st2.result && st2.result.result && st2.result.result.value));
  ws.close(); child.kill();
  process.exit(0);
})().catch(e => { console.error("harness fail:", e); process.exit(1); });
