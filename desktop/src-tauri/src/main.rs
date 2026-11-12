//! 独山麻将 · 桌面版（Tauri 2）。
//!
//! 与手机版（`tauri-offline/`）共用同一个 Rust 内核 `offline-core`：
//! 规则、结算、AI 推理、前端页面、模型全部在**编译期**嵌入，运行时不需要
//! Python、不需要模型目录，也不需要网络。
//!
//! 与手机版的唯一区别在**监听地址**：
//!   * 手机版只绑 `127.0.0.1`（页面就在设备里，不对外）；
//!   * 桌面版绑 `0.0.0.0` —— 本机窗口玩宽屏版的同时，同一 Wi-Fi 下的手机
//!     浏览器打开 `http://<本机IP>:8771/?touch=1` 就是移动版后端。
//!
//! 启动顺序：先在当前线程起 HTTP 服务（模型按需惰性加载），再开窗口。

#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

use std::net::UdpSocket;
use std::sync::{Arc, Mutex};

use offline_core::http::{serve_on, App};
use tauri::Manager;

/// 固定端口：页面 origin 变了会清掉 localStorage（座位/模型/节奏等设置）。
const PORT: u16 = 8771;

/// 本机局域网 IPv4：UDP connect 只让内核选路，不发包；无网络时返回 None。
fn lan_ip() -> Option<String> {
    let sock = UdpSocket::bind("0.0.0.0:0").ok()?;
    sock.connect("8.8.8.8:80").ok()?;
    Some(sock.local_addr().ok()?.ip().to_string())
}

fn main() {
    let engine = Arc::new(Mutex::new(App::new()));
    match serve_on("0.0.0.0", PORT, engine) {
        Ok(p) => {
            println!("[desktop] 本地服务已启动 0.0.0.0:{p}");
            if let Some(ip) = lan_ip() {
                println!("[desktop] 手机访问 http://{ip}:{p}/?touch=1（同一 Wi-Fi）");
            }
        }
        Err(e) => {
            // 不直接退出：窗口仍会打开，显示 web/index.html 的兜底提示，
            // 比双击后无声闪退更容易排查。
            eprintln!("[desktop] 本地服务启动失败：{e}");
        }
    }

    tauri::Builder::default()
        .setup(|app| {
            if let Some(win) = app.get_webview_window("main") {
                let title = match lan_ip() {
                    Some(ip) => format!("独山麻将 · 手机访问 http://{ip}:{PORT}/?touch=1"),
                    None => "独山麻将".to_string(),
                };
                let _ = win.set_title(&title);
            }
            Ok(())
        })
        .run(tauri::generate_context!())
        .expect("独山麻将 桌面版启动失败");
}
