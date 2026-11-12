/* 独山麻将 · 双模式启动页（2026-11-12）
   ① 首屏只给「在线模式 / 离线模式」两个入口，不自动进任一模式；
   ② 在线模式 = 联网客户端那套：填服务端地址 → 测试连接 → WebView 导航到移动网页版；
   ③ 离线模式 = 原来的流程：等本地 Rust 引擎就绪 → 导航到 127.0.0.1 的本地服务。
   浏览器预览（没有 Tauri IPC）时：离线只提示，在线直接跳转，便于视觉验收。 */

(function () {
  "use strict";

  var invoke = (window.__TAURI__ && window.__TAURI__.core && window.__TAURI__.core.invoke) || null;
  var IN_APP = !!invoke;
  var DEFAULT_URL = "http://172.26.68.203:8770";
  var URL_KEY = "zhuoji_online_server";

  var t0 = Date.now();
  var timer = null;
  var navigated = false;
  var wantOffline = false;      // 用户已选离线模式，引擎一就绪就进入
  var engineReady = false;
  var bootMs = 0;

  function $(id) { return document.getElementById(id); }
  /* 2026-11-12：选择页与两个面板是同一个 URL 的 SPA，安卓返回键原本无从判断
     当前在哪一屏（只能读到 `webView.url`）。解法是把面板状态写进 URL hash，
     原生同步可读，见下面 show() 的实现。 */
  var currentPane = "chooser";

  function show(id) {
    currentPane = id;
    ["chooser", "online-pane", "offline-pane"].forEach(function (p) {
      $(p).hidden = (p !== id);
    });
    /* 2026-11-12 定稿：面板状态**同步写进 URL hash**，不再靠异步 JS 回调。
       安卓返回键是原生同步事件，只能读 `webView.url`；原先用
       evaluateJavascript 问 currentPane，页面刚好在重载时回调会丢，
       表现为「首屏按一次返回没反应、再按一次才退出」（用户实测）。
       现在 URL 本身就是状态：
         选择页        http://tauri.localhost/
         在线面板      http://tauri.localhost/#online-pane
         离线面板      http://tauri.localhost/#offline-pane
       一律 replaceState（不堆历史条目），保证 loadUrl 回选择页后历史干净。 */
    var base = location.href.split("#")[0];
    var next = (id === "chooser") ? base : base + "#" + id;
    try { history.replaceState({ pane: id }, "", next); } catch (e) { /* 沙箱里可能禁用 */ }
  }
  window.addEventListener("popstate", function (e) {
    var pane = (e.state && e.state.pane) || "chooser";
    wantOffline = false;
    show(pane);
  });
  function setStatus(text, kind) {
    var el = $("status");
    el.textContent = text;
    el.className = "status" + (kind ? " " + kind : "");
  }
  function setBar(pct) {
    var el = $("bar");
    if (el) el.style.width = Math.max(2, Math.min(100, pct)) + "%";
  }
  function setOnlineStatus(text, kind) {
    var el = $("online-status");
    el.textContent = text;
    el.className = "status" + (kind ? " " + kind : "");
  }
  function showError(msg) {
    if (timer) { clearInterval(timer); timer = null; }
    setBar(100);
    $("bar").parentNode.classList.add("err");
    setStatus(msg, "err");
    $("btn-retry").hidden = false;
    $("hint").hidden = true;
    $("chooser-status").textContent = "本地引擎启动失败：" + msg;
  }

  /* ---------------- 离线模式 ---------------- */
  async function enterOffline() {
    if (navigated || !IN_APP || !engineReady) return;
    navigated = true;
    setStatus("正在进入牌局…", "busy");
    setBar(100);
    try {
      await invoke("enter_game");
    } catch (e) {
      navigated = false;
      showError("进入游戏失败：" + (e && e.message ? e.message : e));
    }
  }

  function pollEngine() {
    if (!IN_APP) return;
    invoke("engine_status").then(function (s) {
      if (s.error) { showError(s.error); return; }
      if (s.ready) {
        engineReady = true;
        bootMs = s.boot_ms || 0;
        if (timer) { clearInterval(timer); timer = null; }
        var txt = "本地引擎就绪（" + bootMs + " ms）";
        $("chooser-status").textContent = txt + " · 可随时进入离线模式";
        $("chooser-status").className = "status ok";
        setStatus(txt, "ok");
        setBar(100);
        if (wantOffline) setTimeout(enterOffline, 120);
        return;
      }
      var el = (Date.now() - t0) / 1000;
      var pct = el < 0.6 ? 8 + el * 40 : Math.min(92, 32 + Math.log2(el) * 24);
      setBar(pct);
      if (el > 3) {
        setStatus("正在加载本地引擎与模型… " + el.toFixed(0) + " s", "busy");
        $("chooser-status").textContent = "本地引擎加载中… " + el.toFixed(0) + " s";
      }
    }).catch(function (e) {
      showError("无法与本地引擎通信：" + (e && e.message ? e.message : e));
    });
  }

  function startEnginePoll() {
    if (!IN_APP) {
      $("chooser-status").textContent = "预览模式：离线引擎需在安卓 APP 内运行";
      return;
    }
    pollEngine();
    if (!timer) timer = setInterval(pollEngine, 400);
  }

  function chooseOffline() {
    wantOffline = true;
    show("offline-pane");
    $("chooser-status").className = "status";
    if (!IN_APP) {
      setBar(100);
      setStatus("预览模式：请在安卓 APP 内运行（浏览器里没有本地引擎）", "busy");
      $("hint").hidden = false;
      return;
    }
    if (engineReady) {
      setStatus("本地引擎就绪（" + bootMs + " ms），准备进入…", "ok");
      setBar(100);
      setTimeout(enterOffline, 120);
    } else {
      setStatus("正在唤醒本地引擎…", "busy");
    }
  }

  /* ---------------- 在线模式 ---------------- */
  function normalize(s) {
    s = (s || "").trim();
    if (!s) return "";
    if (!/^https?:\/\//i.test(s)) s = "http://" + s;
    return s.replace(/\/+$/, "");
  }
  function loadUrl() {
    try { return localStorage.getItem(URL_KEY) || DEFAULT_URL; } catch (e) { return DEFAULT_URL; }
  }
  function saveUrl(u) {
    try { localStorage.setItem(URL_KEY, u); } catch (e) {}
  }
  function setOnlineBusy(busy) {
    $("btn-test").disabled = busy;
    $("btn-enter").disabled = busy;
    $("btn-default").disabled = busy;
    $("url").disabled = busy;
  }
  async function testConnection(silent) {
    var raw = normalize($("url").value);
    if (!raw) { if (!silent) setOnlineStatus("请先填写服务器地址", "err"); return false; }
    if (!IN_APP) {
      setOnlineStatus("预览模式：请在安卓 APP 内测试连接（浏览器受跨域限制）", "busy");
      return false;
    }
    setOnlineBusy(true);
    setOnlineStatus("正在连接 " + raw + " …", "busy");
    try {
      var r = await invoke("check_online", { url: raw });
      if (r.ok) {
        setOnlineStatus("连接成功 · " + r.status + " · " + r.latencyMs + "ms", "ok");
        return true;
      }
      setOnlineStatus("连接失败：" + (r.error || r.status || "未知错误"), "err");
      return false;
    } catch (e) {
      setOnlineStatus("连接失败：" + (e && e.message ? e.message : e), "err");
      return false;
    } finally {
      setOnlineBusy(false);
    }
  }
  async function enterOnline() {
    var raw = normalize($("url").value);
    if (!raw) { setOnlineStatus("请先填写服务器地址", "err"); return; }
    saveUrl(raw);
    setOnlineBusy(true);
    setOnlineStatus("正在进入 " + raw + " …", "busy");
    try {
      if (IN_APP) {
        await invoke("enter_online", { url: raw });   // WebView 已导航，后面不会执行
      } else {
        window.location.href = raw;
      }
    } catch (e) {
      setOnlineStatus("进入失败：" + (e && e.message ? e.message : e), "err");
      setOnlineBusy(false);
    }
  }

  /* ---------------- 绑定 ---------------- */
  $("btn-online").addEventListener("click", function () {
    wantOffline = false;
    show("online-pane");
    $("url").value = loadUrl();
    if (IN_APP) testConnection(true);
    else {
      $("preview").hidden = false;
      setOnlineStatus("预览模式：填入服务端地址后点「进入网页版」", "");
    }
  });
  $("btn-offline").addEventListener("click", chooseOffline);
  // 面板上的「‹ 返回」按钮：与安卓返回键同一条路径 —— 直接 show("chooser")，
  // 面板状态由 URL hash 承载，不堆历史条目，两种返回方式状态永远一致。
  Array.prototype.forEach.call(document.querySelectorAll("[data-back]"), function (b) {
    b.addEventListener("click", function () {
      wantOffline = false;
      show("chooser");
    });
  });
  $("btn-test").addEventListener("click", function () { testConnection(false); });
  $("btn-enter").addEventListener("click", enterOnline);
  $("btn-default").addEventListener("click", function () {
    $("url").value = DEFAULT_URL;
    setOnlineStatus("已恢复默认地址，点「测试连接」确认", "");
  });
  $("url").addEventListener("keydown", function (e) {
    if (e.key === "Enter") { e.preventDefault(); enterOnline(); }
  });
  $("btn-retry").addEventListener("click", function () {
    navigated = false;
    t0 = Date.now();
    $("btn-retry").hidden = true;
    $("bar").parentNode.classList.remove("err");
    setBar(6);
    setStatus("正在唤醒本地引擎…", "busy");
    startEnginePoll();
  });

  if (IN_APP) {
    invoke("app_info").then(function (info) {
      if (info && info.version) {
        $("footer").textContent = "双模式 v" + info.version +
          " · 在线由服务端提供，离线由设备内引擎提供";
      }
    }).catch(function () {});
  }

  show("chooser");
  startEnginePoll();
})();
