/* 独山麻将 · 安卓客户端启动页逻辑
   - APP 内：通过 Tauri IPC 调用 Rust 侧命令（保存地址 / 探测 / 导航）
   - 浏览器预览：降级为 localStorage + 直接跳转，便于自动化视觉测试
   游戏页面本身不做任何注入或修改。 */

(function () {
  "use strict";

  var DEFAULT_URL = "http://172.26.99.125:8770";
  var invoke = (window.__TAURI__ && window.__TAURI__.core && window.__TAURI__.core.invoke) || null;
  var IN_APP = !!invoke;

  function $(id) { return document.getElementById(id); }
  function normalize(s) {
    s = (s || "").trim();
    if (!s) return "";
    if (!/^https?:\/\//i.test(s)) s = "http://" + s;
    return s.replace(/\/+$/, "");
  }
  function setStatus(text, kind) {
    var el = $("status");
    el.textContent = text;
    el.className = "status" + (kind ? " " + kind : "");
  }
  function setBusy(busy) {
    $("btn-test").disabled = busy;
    $("btn-enter").disabled = busy;
    $("btn-default").disabled = busy;
    $("url").disabled = busy;
  }

  async function loadSaved() {
    var u = "";
    if (IN_APP) {
      try { u = await invoke("get_server"); } catch (e) { u = ""; }
    } else {
      try { u = localStorage.getItem("zhuoji_server") || ""; } catch (e) { u = ""; }
      $("preview").hidden = false;
      try {
        var info = { version: "1.0.0" };
        $("footer").textContent = "预览模式 v" + info.version + " · 游戏界面、规则与 AI 均由服务端原样提供";
      } catch (e) {}
    }
    $("url").value = u || DEFAULT_URL;
  }

  async function testConnection(silent) {
    var raw = normalize($("url").value);
    if (!raw) { setStatus("请先填写服务器地址", "err"); return false; }

    if (!IN_APP) {
      setStatus("预览模式：请在安卓 APP 内测试连接（浏览器受跨域限制）", "busy");
      return false;
    }

    setBusy(true);
    setStatus("正在连接 " + raw + " …", "busy");
    try {
      var r = await invoke("check_server", { url: raw });
      if (r.ok) {
        setStatus("连接成功 · " + r.status + " · " + r.latencyMs + "ms", "ok");
        return true;
      }
      setStatus("连接失败：" + (r.error || r.status || "未知错误"), "err");
      return false;
    } catch (e) {
      setStatus("连接失败：" + (e && e.message ? e.message : e), "err");
      return false;
    } finally {
      setBusy(false);
    }
  }

  async function enterGame() {
    var raw = normalize($("url").value);
    if (!raw) { setStatus("请先填写服务器地址", "err"); return; }

    setBusy(true);
    setStatus("正在进入 " + raw + " …", "busy");
    try {
      if (IN_APP) {
        await invoke("enter_game", { url: raw });
        // WebView 已导航到游戏页，这里不会再执行
      } else {
        try { localStorage.setItem("zhuoji_server", raw); } catch (e) {}
        window.location.href = raw;
      }
    } catch (e) {
      setStatus("进入失败：" + (e && e.message ? e.message : e), "err");
      setBusy(false);
    }
  }

  function bind() {
    $("btn-test").addEventListener("click", function () { testConnection(false); });
    $("btn-enter").addEventListener("click", enterGame);
    $("btn-default").addEventListener("click", function () {
      $("url").value = DEFAULT_URL;
      setStatus("已恢复默认地址，点「测试连接」确认", "");
    });
    $("url").addEventListener("keydown", function (e) {
      if (e.key === "Enter") { e.preventDefault(); enterGame(); }
    });
    $("url").addEventListener("input", function () {
      if ($("status").className.indexOf("err") >= 0) setStatus("", "");
    });
  }

  bind();
  loadSaved().then(function () {
    // 启动即自动探测一次，让用户打开 APP 就知道服务端是否在线
    if (IN_APP) testConnection(true);
    else setStatus("预览模式：填入服务端地址后点「进入游戏」", "");
  });
})();
