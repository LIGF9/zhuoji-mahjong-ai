//! 独山麻将 · 单机离线版 —— Tauri 2 Android 宿主。
//!
//! 与联网版 `tauri-app/` 的**唯一区别**：这里不连局域网服务端，而是在进程内
//! 起一个只监听 `127.0.0.1` 的本地 HTTP 服务（`offline-core::http`），
//! 由它托管与 `web/static/` **逐字节相同**的 `dushan.html` 并实现 16 个 `/api/*`。
//!
//! 设计红线（沿用 v1.1.0）：
//!   * 不复制、不修改任何游戏 HTML/CSS/JS，也不重写规则或 AI；
//!   * 规则/结算/搜索用 `dushan-engine`（与 Python 已对拍），推理用纯 Rust `tract`；
//!   * 模型与静态资源在**编译期**嵌入 `.so`，零 AssetManager、零 ORT `.so`；
//!   * Rust 侧只做两件事：启动本地引擎、把 WebView 导航到它并补 `?touch=1`。
//!
//! 启动顺序：先显示启动页（`src/index.html`）→ 后台线程加载两个 ONNX 并起服务
//! → 启动页轮询 `engine_status` 拿到端口 → `enter_game` 导航。
//! 模型加载在手机上约 2–8 s，必须有可见的等待界面，不能让用户看白屏。

use std::{
    sync::{Arc, Mutex},
    time::Instant,
};

use offline_core::http::{serve, App};
use tauri::WebviewWindow;

/// 本地引擎监听的固定端口。见 `engine_status`/启动线程里的注释：
/// 端口是 localStorage 的 origin 一部分，必须固定，设置才不会每次启动清零。
const FIXED_PORT: u16 = 8791;

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

/// 构建信息，供启动页展示与自动化验收（也用于确认「确实没联网」）。
#[tauri::command]
fn app_info(engine: tauri::State<'_, Arc<Engine>>) -> serde_json::Value {
    serde_json::json!({
        "name": "独山麻将 · 单机",
        "version": env!("CARGO_PKG_VERSION"),
        "identifier": "com.zhuoji.mahjong.offline",
        "mode": "offline-embedded",
        "port": engine.ready(),
        "note": "规则、结算与 AI 推理全部在设备内由 Rust 完成，不访问网络",
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
}
