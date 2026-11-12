//! 独山麻将 · 单机离线版 —— Tauri 2 Android 宿主。
//!
//! 与联网版 `tauri-app/` 的**唯一区别**：这里不连局域网服务端，而是在进程内
//! 起一个只监听 `127.0.0.1` 的本地 HTTP 服务（`offline-core::http`），
//! 由它托管与 `web/static/` **逐字节相同**的 `dushan.html` 并实现 16 个 `/api/*`。
//!
//! 2026-11-12 起启动页是**双模式选择**：
//!   * 离线模式 → 本地引擎（原来的流程，断网可玩）；
//!   * 在线模式 → 把 WebView 导航到用户填的局域网服务端地址，进入移动网页版
//!     （与 `tauri-app/` 同一套地址归一化 / 连通性探测 / `?touch=1` 口径）。
//!
//! 设计红线（沿用 v1.1.0）：
//!   * 不复制、不修改任何游戏 HTML/CSS/JS，也不重写规则或 AI；
//!   * 规则/结算/搜索用 `dushan-engine`（与 Python 已对拍），推理用纯 Rust `tract`；
//!   * 模型与静态资源在**编译期**嵌入 `.so`，零 AssetManager、零 ORT `.so`；
//!   * Rust 侧只做两件事：启动本地引擎、把 WebView 导航到它并补 `?touch=1`。
//!
//! 启动顺序：先显示模式选择页（`src/index.html`）→ 后台线程加载两个 ONNX 并起服务
//! → 启动页轮询 `engine_status` 拿到端口 → 用户选「离线模式」才 `enter_game` 导航。
//! 模型加载在手机上约 2–8 s，必须有可见的等待界面，不能让用户看白屏。

use std::{
    io::{Read, Write},
    net::{TcpStream, ToSocketAddrs},
    sync::{Arc, Mutex},
    time::{Duration, Instant},
};

use offline_core::http::{serve, App};
use tauri::WebviewWindow;

/// 本地引擎监听的固定端口。见 `engine_status`/启动线程里的注释：
/// 端口是 localStorage 的 origin 一部分，必须固定，设置才不会每次启动清零。
const FIXED_PORT: u16 = 8791;

/// 「在线模式」首次启动的默认地址（当前机器的局域网地址，见 docs/手机访问.md）。
/// 用户可在启动页改，改动存在启动页自己的 localStorage 里。
const DEFAULT_SERVER: &str = "http://172.26.68.203:8770";
const CHECK_TIMEOUT_MS: u64 = 2500;

/// 归一化用户输入：补 http://、去尾部斜杠、只放行 http(s)。
fn normalize_url(raw: &str) -> Result<String, String> {
    let mut s = raw.trim().to_string();
    if s.is_empty() {
        return Err("请输入服务器地址".into());
    }
    if let Some(idx) = s.find("://") {
        let scheme = s[..idx].to_ascii_lowercase();
        if scheme != "http" && scheme != "https" {
            return Err(format!("不支持的协议 {scheme}，请用 http:// 或 https://"));
        }
    } else {
        s = format!("http://{s}");
    }
    let s = s.trim_end_matches('/').to_string();
    let u = tauri::Url::parse(&s).map_err(|e| format!("地址格式错误：{e}"))?;
    match u.scheme() {
        "http" | "https" => {}
        other => return Err(format!("不支持的协议 {other}，请用 http:// 或 https://")),
    }
    if u.host_str().unwrap_or("").is_empty() {
        return Err("地址缺少主机名".into());
    }
    Ok(s)
}

/// 引擎启动状态。`port` 有值即代表服务已在监听。
#[derive(Default)]
struct Engine {
    port: Mutex<Option<u16>>,
    error: Mutex<Option<String>>,
    boot_ms: Mutex<u128>,
}

impl Engine {
    fn ready(&self) -> Option<u16> {
        *self.port.lock().unwrap()
    }
    fn fail(&self, msg: impl Into<String>) {
        *self.error.lock().unwrap() = Some(msg.into());
    }
}

/// 给游戏地址补 `?touch=1`。
///
/// 游戏页用 `matchMedia("(hover: none) and (pointer: coarse)")` 判定触屏，
/// 但 Tauri 的 Android WebView 不命中该查询，会被当成"过窄的桌面窗口"弹
/// 「窗口宽度不足」门禁。`?touch=1` 是游戏自带的强制触屏口径，补上后与
/// 手机浏览器 / 无头测试完全一致。这是 v1.1.0 实测发现的集成 bug，离线版沿用。
fn with_touch_param(url_str: &str) -> Result<String, String> {
    let mut parsed = tauri::Url::parse(url_str).map_err(|e| format!("地址格式错误：{}", e))?;
    let has_touch = parsed
        .query_pairs()
        .any(|(k, v)| k == "touch" && v == "1");
    if !has_touch {
        parsed.query_pairs_mut().append_pair("touch", "1");
    }
    Ok(parsed.to_string())
}

/// 启动页轮询用：返回引擎是否就绪、端口、失败原因与加载耗时。
#[tauri::command]
fn engine_status(engine: tauri::State<'_, Arc<Engine>>) -> serde_json::Value {
    serde_json::json!({
        "ready": engine.ready().is_some(),
        "port": engine.ready(),
        "error": engine.error.lock().unwrap().clone(),
        "boot_ms": *engine.boot_ms.lock().unwrap(),
    })
}

/// 导航到本地引擎托管的游戏页。**不做任何前端资源替换**。
#[tauri::command]
fn enter_game(window: WebviewWindow, engine: tauri::State<'_, Arc<Engine>>) -> Result<String, String> {
    let port = engine
        .ready()
        .ok_or_else(|| "本地引擎尚未就绪".to_string())?;
    if let Some(e) = engine.error.lock().unwrap().clone() {
        return Err(e);
    }
    let game_url = with_touch_param(&format!("http://127.0.0.1:{port}/"))?;
    let parsed = tauri::Url::parse(&game_url).map_err(|e| format!("地址格式错误：{}", e))?;
    window
        .navigate(parsed)
        .map_err(|e| format!("无法打开游戏页面：{}", e))?;
    Ok(game_url)
}

/// 「在线模式」连通性探测：只用 std 明文 TCP 请求公开的 `GET /api/meta`，
/// 不读取任何对局数据。HTTPS 不做明文探测，直接放行（std 没有 TLS）。
#[tauri::command]
fn check_online(url: String) -> serde_json::Value {
    let normalized = match normalize_url(&url) {
        Ok(u) => u,
        Err(e) => return serde_json::json!({"ok": false, "status": "", "latencyMs": 0, "error": e}),
    };
    let parsed = match tauri::Url::parse(&normalized) {
        Ok(u) => u,
        Err(e) => return serde_json::json!({"ok": false, "status": "", "latencyMs": 0,
                                            "error": format!("地址格式错误：{e}")}),
    };
    if parsed.scheme() == "https" {
        return serde_json::json!({"ok": true, "status": "https 地址：跳过明文探测",
                                  "latencyMs": 0, "error": ""});
    }
    let host = parsed.host_str().unwrap_or("").to_string();
    let port = parsed.port_or_known_default().unwrap_or(8770);
    let addr = format!("{host}:{port}");
    let sock = match addr.to_socket_addrs() {
        Ok(mut it) => match it.next() {
            Some(a) => a,
            None => return serde_json::json!({"ok": false, "status": "", "latencyMs": 0,
                                             "error": "无法解析主机地址"}),
        },
        Err(e) => return serde_json::json!({"ok": false, "status": "", "latencyMs": 0,
                                            "error": format!("无法解析主机地址：{e}")}),
    };
    let t0 = Instant::now();
    let mut stream = match TcpStream::connect_timeout(&sock, Duration::from_millis(CHECK_TIMEOUT_MS)) {
        Ok(s) => s,
        Err(e) => return serde_json::json!({"ok": false, "status": "", "latencyMs": 0,
            "error": format!("连不上 {host}:{port}（{e}）。请确认电脑端服务已启动、手机与电脑在同一 Wi-Fi。")}),
    };
    let _ = stream.set_read_timeout(Some(Duration::from_millis(CHECK_TIMEOUT_MS)));
    let _ = stream.set_write_timeout(Some(Duration::from_millis(CHECK_TIMEOUT_MS)));
    let base = parsed.path().trim_end_matches('/');
    let req = format!(
        "GET {base}/api/meta HTTP/1.1\r\nHost: {addr}\r\nUser-Agent: ZhuojiOffline/1.2\r\nAccept: application/json\r\nConnection: close\r\n\r\n"
    );
    if let Err(e) = stream.write_all(req.as_bytes()) {
        return serde_json::json!({"ok": false, "status": "", "latencyMs": 0,
                                  "error": format!("发送请求失败：{e}")});
    }
    let mut buf = Vec::new();
    let _ = stream.read_to_end(&mut buf);  // 读超时是预期内的，用已收到的字节判断状态行
    let text = String::from_utf8_lossy(&buf);
    let status = text.lines().next().unwrap_or("").trim().to_string();
    let latency_ms = t0.elapsed().as_millis();
    let ok = status.contains(" 200 ");
    serde_json::json!({
        "ok": ok,
        "status": if status.is_empty() { "服务端无响应".to_string() } else { status },
        "latencyMs": latency_ms,
        "error": if ok { "" } else { "服务端未返回 HTTP 200，请检查地址与端口（默认 8770）" },
    })
}

/// 「在线模式」：把 WebView 导航到移动网页版。游戏资源一字不改，
/// 只补 `?touch=1`（Android WebView 不命中 pointer:coarse 媒体查询）。
#[tauri::command]
fn enter_online(window: WebviewWindow, url: String) -> Result<String, String> {
    let normalized = normalize_url(&url)?;
    let game_url = with_touch_param(&normalized)?;
    let parsed = tauri::Url::parse(&game_url).map_err(|e| format!("地址格式错误：{e}"))?;
    window
        .navigate(parsed)
        .map_err(|e| format!("无法打开网页版：{e}"))?;
    Ok(game_url)
}

/// 构建信息，供启动页展示与自动化验收（也用于确认「确实没联网」）。
///
/// 2026-11-12：版本号改取 `package_info()`（= `tauri.conf.json` 的 `version`）。
/// 原先用 `env!("CARGO_PKG_VERSION")` 是本 crate 的 `Cargo.toml` 版本，
/// 与 `tauri.conf.json`/APK 的 versionName 是两套 —— 结果 APK 明明是 1.2.0，
/// 启动页 footer 却一直显示 `双模式 v1.0.0`（main.js 用这个值覆盖页脚）。
#[tauri::command]
fn app_info(app: tauri::AppHandle, engine: tauri::State<'_, Arc<Engine>>) -> serde_json::Value {
    serde_json::json!({
        "name": "独山麻将 · 双模式",
        "version": app.package_info().version.to_string(),
        "identifier": "com.zhuoji.mahjong.offline",
        "mode": "offline-embedded + online-web",
        "default_online_server": DEFAULT_SERVER,
        "port": engine.ready(),
        "note": "离线模式：规则、结算与 AI 推理全部在设备内由 Rust 完成；在线模式：进入局域网服务端提供的移动网页版",
    })
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    let engine = Arc::new(Engine::default());
    let boot = engine.clone();

    tauri::Builder::default()
        .manage(engine)
        .setup(move |_app| {
            // 后台线程：加载两个 ONNX（tract 优化 + 编译）→ 绑定 127.0.0.1:0 → 起服务。
            // 用 0 让系统分配端口，避免与设备上其他进程抢固定端口。
            std::thread::Builder::new()
                .name("engine-boot".into())
                .spawn(move || {
                    let t0 = Instant::now();
                    let app = Arc::new(Mutex::new(App::new()));
                    // 固定端口：localStorage 按 **origin（含端口）** 隔离，
                    // 端口一变用户的全部设置就丢了（v1.1.0 漏掉的问题）。
                    // 8791 被占（如同一设备装了联网版、或上次进程残留）时退回
                    // 系统分配，宁可变端口也不能起不来。
                    let served = serve(FIXED_PORT, app.clone())
                        .or_else(|e| {
                            eprintln!("[offline] 固定端口 {FIXED_PORT} 不可用（{e}），改用临时端口");
                            serve(0, app.clone())
                        });
                    match served {
                        Ok(p) => {
                            *boot.boot_ms.lock().unwrap() = t0.elapsed().as_millis();
                            *boot.port.lock().unwrap() = Some(p);
                            println!("[offline] 引擎就绪 127.0.0.1:{p} in {} ms", t0.elapsed().as_millis());
                        }
                        Err(e) => {
                            *boot.boot_ms.lock().unwrap() = t0.elapsed().as_millis();
                            boot.fail(format!("本地服务启动失败：{e}"));
                            eprintln!("[offline] 本地服务启动失败: {e}");
                        }
                    }
                })
                .expect("无法创建引擎线程");
            Ok(())
        })
        .invoke_handler(tauri::generate_handler![
            engine_status,
            enter_game,
            check_online,
            enter_online,
            app_info
        ])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn touch_param_is_appended_once() {
        assert_eq!(
            with_touch_param("http://127.0.0.1:41234/").unwrap(),
            "http://127.0.0.1:41234/?touch=1"
        );
        assert_eq!(
            with_touch_param("http://127.0.0.1:41234/?touch=1").unwrap(),
            "http://127.0.0.1:41234/?touch=1"
        );
        let u = with_touch_param("http://127.0.0.1:41234/?a=b").unwrap();
        assert!(u.contains("a=b") && u.contains("touch=1"), "u={u}");
    }

    #[test]
    fn touch_param_rejects_garbage() {
        assert!(with_touch_param("not a url").is_err());
    }

    #[test]
    fn online_url_normalization() {
        assert_eq!(normalize_url("172.26.68.203:8770").unwrap(), DEFAULT_SERVER);
        assert_eq!(normalize_url("  http://192.168.1.5:8770/  ").unwrap(),
                   "http://192.168.1.5:8770");
        assert!(normalize_url("").is_err());
        assert!(normalize_url("ftp://x").is_err());
        assert!(normalize_url("javascript:alert(1)").is_err());
    }

    #[test]
    fn online_check_rejects_garbage() {
        let r = check_online("not a url".into());
        assert_eq!(r["ok"], false);
        assert!(!r["error"].as_str().unwrap_or("").is_empty());
    }
}
