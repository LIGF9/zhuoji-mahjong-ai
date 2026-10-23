// 单文件版 HTML 的 Node 全链路验证（非浏览器，仅验证内嵌运行时链路）：
//   解析 single HTML → 内嵌 ort.min.js + atob(wasm/model b64) + blob mjs →
//   ORT 会话 → 本地引擎 API → 真模型推理打完一局。
// 用法：node web/standalone/check_single_html.js
"use strict";
const fs = require("fs");
const path = require("path");

const html = fs.readFileSync(
  path.join(__dirname, "dushan_standalone_single.html"), "utf8");

function block(id) {
  const i = html.indexOf(`id="${id}"`);
  if (i < 0) throw new Error("missing block " + id);
  const s = html.indexOf(">", i) + 1;
  const e = html.indexOf("</script>", s);
  return html.slice(s, e);
}
function firstScriptContaining(needle) {
  const i = html.indexOf(needle);
  if (i < 0) throw new Error("needle not found: " + needle);
  const s = html.lastIndexOf("<script>", i) + 8;
  const e = html.indexOf("</script>", i);
  return html.slice(s, e);
}

(async () => {
  // 1) 内嵌 ort.min.js 进全局（strict 模式 eval 不泄漏 var → 显式取返回值）
  globalThis.ort = new Function(firstScriptContaining("ONNX Runtime Web")
    .replace(/^[^"]*?;\s*/, "").replace(/;\s*$/, "") + "; return ort;")();
  if (typeof ort === "undefined") throw new Error("ort 未定义");
  // 2) 沙箱 + 装配脚本（glue 用全局 document/ort/Blob/URL，Node 下补挂全局）
  globalThis.document = { getElementById: id => ({ textContent: block(id) }) };
  const w = { document: globalThis.document };
  const glue = firstScriptContaining("__engModule")
    // Node 的动态 import 不支持 blob: URL（浏览器支持）→ 检查器改用磁盘 mjs。
    .replace('URL.createObjectURL(new Blob([document.getElementById("__mjs_text")'
             + '.textContent], { type: "text/javascript" }))',
             'require("url").pathToFileURL(require("path").join(__dirname, "lib",'
             + ' "ort-wasm-simd-threaded.mjs")).href');
  new Function("window", "require", "__dirname", glue)(w, require, __dirname);
  const api = await w.__DushanReady;
  console.log("[single] 引擎+ORT 会话就绪（内嵌 wasm/mjs/model 链路 OK）");
  // 3) 真模型打一局（人类随机合法动作）
  const nr = await api.handle("/api/new", { my_seat: 0, speed: 0 });
  const t0 = Date.now();
  for (let g = 0; g < 50000; g++) {
    const st = await api.handle("/api/state?sid=" + nr.sid);
    if (st.over) {
      console.log("[single] 对局:", st.result.type,
        "deltas", JSON.stringify(st.result.deltas),
        "events", st.events.length,
        `${((Date.now() - t0) / 1000).toFixed(1)}s`);
      const dsum = st.result.deltas.reduce((a, b) => a + b, 0);
      if (Math.abs(dsum) > 1e-6) throw new Error("非零和 " + dsum);
      console.log("[single] == PASS ==");
      process.exit(0);
    }
    if (st.pending && st.turn === st.my_seat) {
      const pend = st.pending;
      const act = pend.mode === "discard" && pend.discardables.length
        ? { kind: "discard", tile: pend.discardables[Math.floor(Math.random() * pend.discardables.length)] }
        : (pend.buttons[0] ? { kind: pend.buttons[0].kind, tile: pend.buttons[0].tile } : null);
      if (!act) throw new Error("空决策点");
      await api.handle("/api/act", { sid: nr.sid, kind: act.kind, tile: act.tile });
    } else await new Promise(r => setTimeout(r, 1));
  }
  throw new Error("超步");
})().catch(e => { console.error("[single] ERR", e && e.message || e); process.exit(1); });
