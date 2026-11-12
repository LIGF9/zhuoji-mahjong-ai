//! 单文件服务端（2026-11-12 起支持局域网）。
//!
//! 规则、结算、AI 推理、静态页面与模型全部编译进这一个可执行文件：
//! 拷到任何一台机器上都能直接跑，不需要 Python、不需要模型目录。
//!
//! 用法：
//!   offline_server                 # 127.0.0.1:8771，只给本机浏览器用
//!   offline_server 8771            # 同上，换端口
//!   offline_server 8771 0.0.0.0    # 同一 Wi-Fi 下的手机也能连（桌面服务端模式）
//!
//! 监听 `0.0.0.0` 时会打印本机局域网地址，手机浏览器直接打开即可。

use std::net::UdpSocket;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use offline_core::http::{serve_on, App};

/// 探测本机局域网 IPv4：向外网地址做一次 UDP connect（不发包），
/// 由内核选路后读回本地地址。没有网络时返回 None。
fn lan_ip() -> Option<String> {
    let sock = UdpSocket::bind("0.0.0.0:0").ok()?;
    sock.connect("8.8.8.8:80").ok()?;
    Some(sock.local_addr().ok()?.ip().to_string())
}

fn main() {
    let mut args = std::env::args().skip(1);
    let port: u16 = args.next().and_then(|s| s.parse().ok()).unwrap_or(8771);
    let host = args.next().unwrap_or_else(|| "127.0.0.1".to_string());
    let lan = host == "0.0.0.0";

    let app = Arc::new(Mutex::new(App::new()));
    match serve_on(&host, port, app) {
        Ok(p) => {
            println!("[offline] 服务已启动（{host}:{p}）");
            println!("[offline] 本机：http://127.0.0.1:{p}/?touch=1");
            if lan {
                match lan_ip() {
                    Some(ip) => println!("[offline] 手机：http://{ip}:{p}/?touch=1（同一 Wi-Fi）"),
                    None => println!("[offline] 手机：http://<本机局域网IP>:{p}/?touch=1"),
                }
            }
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
