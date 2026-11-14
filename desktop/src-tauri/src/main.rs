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
//! 2026-11-14 起支持 Windows：同一份 Rust 内核 + `tauri.windows.conf.json`
//! （NSIS 安装包 / WebView2 引导）。Windows 首次启动绑 0.0.0.0 时防火墙
//! 可能询问，允许「专用网络」即可；只想本机玩可在防火墙里拒绝入站。
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
            register_share(app.handle().clone());
            Ok(())
        })
        .run(tauri::generate_context!())
        .expect("独山麻将 桌面版启动失败");
}

/* ---------------- 导出文件 → 系统分享面板（2026-11-12） ----------------
   用户口径：「先写文件再用系统分享面板打开」。

   游戏页在 WebView 里，`<a download>` 在 WKWebView 里不会给出任何可见反馈，
   所以「对局记录 → 导出」改成：内核先把文件写进 `<应用数据>/history/exports/`，
   再由这里弹 macOS 原生的 `NSSharingServicePicker`（隔空投送 / 邮件 / 信息 /
   备忘录…）。面板必须在**主线程**弹，所以经 `run_on_main_thread` 投递。 */

#[cfg(target_os = "macos")]
fn register_share(handle: tauri::AppHandle) {
    use std::sync::{Arc, Mutex};

    // 主窗口的 NSView* 只取一次：分享面板要相对它定位；窗口换了重开 App 即可。
    let view = handle
        .get_webview_window("main")
        .and_then(|w| w.ns_window().ok())
        .map(|ptr| unsafe { mac_share::content_view(ptr) })
        .unwrap_or(std::ptr::null_mut());
    let view_cell = Arc::new(Mutex::new(view as usize));

    offline_core::share::set_handler(Box::new(move |path| {
        let p = path.to_path_buf();
        let h = handle.clone();
        let v = *view_cell.lock().unwrap();
        h.run_on_main_thread(move || {
            if let Err(e) = unsafe { mac_share::show(&p, v as *mut std::ffi::c_void) } {
                // 面板弹不出来也不能让用户「以为导出失败」：文件已经写好了，
                // 退一步在 Finder 里定位它，用户自己拖去要用的地方。
                eprintln!("[desktop] 分享面板打开失败（{e}），改为在 Finder 中显示");
                let _ = std::process::Command::new("open").arg("-R").arg(&p).spawn();
            }
        })
        .map_err(|e| format!("投递主线程失败：{e}"))?;
        Ok(())
    }));
}

/// Windows 没有系统分享面板：导出完成后在资源管理器里**选中**那个文件，
/// 与 macOS 面板失败时的 `open -R` 回退同一口径（文件已经落盘，用户能
/// 直接拖走 / 另存）。不用 Tauri IPC，纯命令行即可，Windows 子系统模式下
/// 也不会弹出控制台窗口。
#[cfg(target_os = "windows")]
fn register_share(_handle: tauri::AppHandle) {
    offline_core::share::set_handler(Box::new(|path| {
        let target = path.to_path_buf();
        std::process::Command::new("explorer")
            .arg(format!("/select,{}", target.to_string_lossy()))
            .spawn()
            .map(|_| ())
            .map_err(|e| format!("打开资源管理器失败：{e}"))
    }));
}

#[cfg(not(any(target_os = "macos", target_os = "windows")))]
fn register_share(_handle: tauri::AppHandle) {}

#[cfg(target_os = "macos")]
mod mac_share {
    use std::ffi::c_void;
    use std::path::Path;

    use objc2::rc::autoreleasepool;
    use objc2::runtime::AnyObject;
    use objc2::{class, msg_send, AnyThread};
    use objc2_app_kit::{NSSharingServicePicker, NSView};
    use objc2_foundation::{NSArray, NSPoint, NSRect, NSRectEdge, NSSize, NSString, NSURL};

    /// `NSWindow*` → 它的 contentView（`NSView*`），后续用来定位分享面板。
    pub unsafe fn content_view(ns_window: *mut c_void) -> *mut c_void {
        if ns_window.is_null() {
            return std::ptr::null_mut();
        }
        let win = ns_window as *mut AnyObject;
        let view: *mut AnyObject = objc2::msg_send![win, contentView];
        view as *mut c_void
    }

    /// 相对 `view` 的中间偏下位置弹出分享面板（`NSMaxYEdge` = 向上展开）。
    pub unsafe fn show(path: &Path, view: *mut c_void) -> Result<(), String> {
        if view.is_null() {
            return Err("窗口视图还没就绪".to_string());
        }
        autoreleasepool(|_| {
            // 先把自己提到最前：面板是「相对某个视图弹出」的浮层，App 不在前台时
            // 可能弹不出来（导出是异步写盘完成才触发的，用户可能已经切走了）。
            let ns_app: *mut AnyObject = objc2::msg_send![class!(NSApplication), sharedApplication];
            let _: () = objc2::msg_send![ns_app, activateIgnoringOtherApps: true];

            let ns_path = NSString::from_str(&path.to_string_lossy());
            let url = NSURL::fileURLWithPath(&ns_path);
            let items: objc2::rc::Retained<NSArray<AnyObject>> =
                msg_send![class!(NSArray), arrayWithObject: &*url];
            let picker = NSSharingServicePicker::initWithItems(NSSharingServicePicker::alloc(), &items);
            let view = &*(view as *const NSView);
            let b = view.bounds();
            let rect = NSRect::new(
                NSPoint::new(b.size.width * 0.5, b.size.height * 0.32),
                NSSize::new(1.0, 1.0),
            );
            picker.showRelativeToRect_ofView_preferredEdge(rect, view, NSRectEdge::NSMaxYEdge);
            Ok(())
        })
    }
}
