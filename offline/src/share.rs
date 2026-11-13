//! 系统分享桥（2026-11-12 用户口径：「先写文件再用系统分享面板打开」）。
//!
//! 背景：离线版与桌面版的游戏页都跑在 **WebView** 里，`<a download>` 不会
//! 弹任何东西（Android WebView 直接忽略，WKWebView 也只能存到默认下载目录
//! 且没有任何反馈）。所以导出改成两步：
//!
//!   1. 服务端（`records::export_file`）把 CSV/JSON 落到应用私有目录；
//!   2. 由**外壳**（安卓 Kotlin / 桌面 macOS）把它交给系统分享面板。
//!
//! 内核 `offline-core` 不依赖任何平台 SDK：外壳启动时用 [`set_handler`]
//! 注册一个闭包，内核只在收到 `/api/history_export?...&share=1` 时回调它。
//! 没有外壳（`offline_server` 命令行 / 浏览器网页版）时 [`available`] 为
//! false，前端看到 `supported:false` 就退回浏览器原生下载。

use std::path::Path;
use std::sync::OnceLock;

type Handler = Box<dyn Fn(&Path) -> Result<(), String> + Send + Sync>;

static HANDLER: OnceLock<Handler> = OnceLock::new();

/// 外壳注册分享实现。重复注册被忽略（进程内只有一个平台外壳）。
pub fn set_handler(h: Handler) {
    let _ = HANDLER.set(h);
}

/// 当前进程是否有可用的系统分享面板。
pub fn available() -> bool {
    HANDLER.get().is_some()
}

/// 把已经落盘的文件交给系统分享面板。
///
/// 错误只用于提示（文件本身已经写好了），前端会把 `path` 一并显示出来，
/// 所以这里不必把失败当成致命错误。
pub fn share(path: &Path) -> Result<(), String> {
    match HANDLER.get() {
        Some(h) => h(path),
        None => Err("当前环境没有系统分享面板".to_string()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 没有外壳注册（浏览器网页版 / 命令行 offline_server）时不能报成功 ——
    /// 前端靠 `supported:false` 退回浏览器原生下载。
    #[test]
    fn without_handler_reports_unsupported() {
        assert!(!available());
        let e = share(Path::new("/tmp/dushan-none.csv")).unwrap_err();
        assert!(e.contains("分享面板"), "err={e}");
    }
}
