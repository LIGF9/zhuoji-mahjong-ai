/* 独山麻将 · 单机离线版启动页
   - APP 内：Tauri IPC 轮询 engine_status → enter_game 导航到 127.0.0.1
   - 浏览器预览：没有 Tauri，降级为提示文案（便于视觉检查，不假装能玩）
   游戏页面本身不做任何注入或修改。 */

(function () {
  "use strict";

  var invoke = (window.__TAURI__ && window.__TAURI__.core && window.__TAURI__.core.invoke) || null;
  var IN_APP = !!invoke;

  var t0 = Date.now();
  var timer = null;
  var navigated = false;

  function $(id) { return document.getElementById(id); }

  function setStatus(text, kind) {
    var el = $("status");
    el.textContent = text;
    el.className = "status" + (kind ? " " + kind : "");
  }

  function setBar(pct) {
    var el = $("bar");
    if (el) el.style.width = Math.max(2, Math.min(100, pct)) + "%";
  }

  function showError(msg) {
    if (timer) { clearInterval(timer); timer = null; }
    setBar(100);
    $("bar").parentNode.classList.add("err");
    setStatus(msg, "err");
    $("btn-retry").hidden = false;
    $("hint").hidden = true;
  }

  async function enter() {
    if (navigated) return;
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

  function poll() {
    invoke("engine_status").then(function (s) {
      if (s.error) { showError(s.error); return; }
      if (s.ready) {
        if (timer) { clearInterval(timer); timer = null; }
        setStatus("本地引擎就绪（" + (s.boot_ms || 0) + " ms），准备进入…", "ok");
        setBar(100);
        setTimeout(enter, 120);
        return;
      }
      // 无进度可读，用时间做一个「不会到顶」的假进度：2s 后到 88% 慢爬。
      var el = (Date.now() - t0) / 1000;
      var pct = el < 0.6 ? 8 + el * 40 : Math.min(92, 32 + Math.log2(el) * 24);
      setBar(pct);
      if (el > 3) {
        setStatus("正在加载本地引擎与模型… " + el.toFixed(0) + " s", "busy");
      }
    }).catch(function (e) {
      showError("无法与本地引擎通信：" + (e && e.message ? e.message : e));
    });
  }

  function start() {
    $("btn-retry").hidden = true;
    $("bar").parentNode.classList.remove("err");
    setBar(6);
    if (!IN_APP) {
      setBar(100);
      setStatus("预览模式：请在安卓 APP 内运行（浏览器里没有本地引擎）", "busy");
      $("hint").hidden = false;
      return;
    }
    setStatus("正在唤醒本地引擎…", "busy");
    poll();
    timer = setInterval(poll, 400);
  }

  $("btn-retry").addEventListener("click", function () {
    navigated = false;
    t0 = Date.now();
    start();
  });

  if (IN_APP) {
    invoke("app_info").then(function (info) {
      if (info && info.version) {
        $("footer").textContent = "单机离线版 v" + info.version + " · 规则与 AI 全部在设备内，不联网";
      }
    }).catch(function () {});
  }

  start();
})();
