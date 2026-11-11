//! 离线版本机验收入口（O1/O2/O3）：在 macOS 上起 127.0.0.1:8771，
//! 用与 APK 内**完全相同**的 Rust 服务托管同一份前端，便于逐字段对拍与实机预演。
//!
//! 用法：cargo run --manifest-path offline/Cargo.toml --bin offline_server
//! 然后浏览器打开 http://127.0.0.1:8771/?touch=1

use std::sync::{Arc, Mutex};
use std::time::Duration;

use offline_core::http::{serve, App};

fn main() {
    let port: u16 = std::env::args()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(8771);
    let app = Arc::new(Mutex::new(App::new()));
    match serve(port, app) {
        Ok(p) => {
            println!("[offline] 服务已启动：http://127.0.0.1:{p}/?touch=1");
            println!("[offline] 模型与静态资源均编译期嵌入，Ctrl+C 退出");
        }
        Err(e) => {
            eprintln!("[offline] 启动失败: {e}");
            std::process::exit(1);
        }
    }
    loop {
        std::thread::sleep(Duration::from_secs(3600));
    }
}
