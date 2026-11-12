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
    net::{IpAddr, Ipv4Addr, SocketAddr, TcpStream, ToSocketAddrs, UdpSocket},
    sync::{Arc, Mutex},
    time::{Duration, Instant},
};

use offline_core::http::{serve, App};
use tauri::WebviewWindow;

/// 本地引擎监听的固定端口。见 `engine_status`/启动线程里的注释：
/// 端口是 localStorage 的 origin 一部分，必须固定，设置才不会每次启动清零。
const FIXED_PORT: u16 = 8791;

/// 「在线模式」扫描端口清单（2026-11-12 用户口径：地址不再写死，改为局域网自动发现）。
///   * 8770 = `web/dushan_server.py` 默认端口（网页版与桌面版共用的服务端）；
///   * 8771 = 桌面版（`desktop/`）内嵌服务端端口。
/// 两者都是「手机开网页版」的合法入口，都扫。
const SCAN_PORTS: [u16; 2] = [8770, 8771];

/// 兜底地址：扫描全失败且用户从没存过地址时，输入框里给一个「历史默认」提示。
/// 只作为**预填/提示**，不再作为硬连接目标（用户要求「不定死地址」）。
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

/// 明文 TCP 探测 `GET {base}/api/meta`，返回 `(是否 200, 状态行, 耗时ms)`。
///
/// 抽出来是因为**连通性校验**与**局域网扫描后的候选校验**必须同一口径：
/// 扫描能「连上」不代表对端是独山麻将服务端（任何占用 8770 的进程都连得上），
/// 而 `/api/meta` 是本服务端独有的公开接口。
fn probe_meta(host: &str, port: u16, base: &str) -> (bool, String, u128) {
    let addr = format!("{host}:{port}");
    let sock = match addr.to_socket_addrs() {
        Ok(mut it) => match it.next() {
            Some(a) => a,
            None => return (false, String::new(), 0),
        },
        Err(_) => return (false, String::new(), 0),
    };
    let t0 = Instant::now();
    let mut stream = match TcpStream::connect_timeout(&sock, Duration::from_millis(CHECK_TIMEOUT_MS)) {
        Ok(s) => s,
        Err(_) => return (false, String::new(), t0.elapsed().as_millis()),
    };
    let _ = stream.set_read_timeout(Some(Duration::from_millis(CHECK_TIMEOUT_MS)));
    let _ = stream.set_write_timeout(Some(Duration::from_millis(CHECK_TIMEOUT_MS)));
    let base = base.trim_end_matches('/');
    let req = format!(
        "GET {base}/api/meta HTTP/1.1\r\nHost: {addr}\r\nUser-Agent: ZhuojiOffline/1.2\r\nAccept: application/json\r\nConnection: close\r\n\r\n"
    );
    if stream.write_all(req.as_bytes()).is_err() {
        return (false, String::new(), t0.elapsed().as_millis());
    }
    let mut buf = Vec::new();
    let _ = stream.read_to_end(&mut buf);  // 读超时是预期内的，用已收到的字节判断状态行
    let text = String::from_utf8_lossy(&buf);
    let status = text.lines().next().unwrap_or("").trim().to_string();
    let ok = status.contains(" 200 ");
    (ok, status, t0.elapsed().as_millis())
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
    // 先做一次 DNS/地址解析，失败时要给「连不上/无法解析」这两种不同口径的文案。
    if host.parse::<IpAddr>().is_err() && format!("{host}:{port}").to_socket_addrs().is_err() {
        return serde_json::json!({"ok": false, "status": "", "latencyMs": 0,
                                  "error": format!("无法解析主机地址 {host}")});
    }
    let (ok, status, latency_ms) = probe_meta(&host, port, parsed.path());
    let status_text = if status.is_empty() { "服务端无响应".to_string() } else { status.clone() };
    let error_text = if ok {
        String::new()
    } else if status.is_empty() {
        format!("连不上 {host}:{port}。请确认电脑端服务已启动、手机与电脑在同一 Wi-Fi。")
    } else {
        "服务端未返回 HTTP 200，请检查地址与端口（默认 8770）".to_string()
    };
    serde_json::json!({
        "ok": ok,
        "status": status_text,
        "latencyMs": latency_ms,
        "error": error_text,
    })
}

/* ---------------- 局域网自动发现（2026-11-12） ----------------
   用户口径：「app 的网页在线版本链接能不能不定死地址，而是在局域网内自动扫描？
   自动扫描到服务端然后连接上。」

   两级发现，都跑在 Rust 侧（WebView 里发 UDP 广播会被 CORS/CSP 与
   Android WebView 的混合内容策略拦掉，只能由原生做）：

   ① **UDP 广播**（首选，<1.2 s）：向 `255.255.255.255:{8770,8771}` 发
      `DUSHAN?`，服务端（`offline/src/http.rs::serve_on` 与
      `web/dushan_server.py::start_discovery`）回 `DUSHAN1 <port>`。
      应答包的**来源 IP** 就是服务端 IP（不用自己拼网段），端口以报文为准。
   ② **TCP 扫描**（兜底，仅在①全空时跑）：部分家用路由/手机厂商 ROM 会丢弃
      受限广播（255.255.255.255）。这时用「本机 IP 所在 /24」逐个连
      `8770/8771`，48 个 worker 并发、单连接 220 ms 超时，最坏 ~2 s。

   两个阶段的结果都要过 `probe_meta`（`/api/meta` 必须 200）才认定为服务端 ——
   任何占用 8770 的进程都能 TCP 连上，只有独山麻将服务端能答 `/api/meta`。 */

/// 本机局域网 IPv4：UDP connect 只让内核选路，不发包；无默认路由时返回 None。
fn local_ipv4() -> Option<Ipv4Addr> {
    let sock = UdpSocket::bind("0.0.0.0:0").ok()?;
    sock.connect("8.8.8.8:80").ok()?;
    match sock.local_addr().ok()?.ip() {
        IpAddr::V4(v) => Some(v),
        IpAddr::V6(_) => None,
    }
}

/// 解析 `DUSHAN1 <port>\n`。端口缺失/非法时退回 `peer_port`（应答包的来源端口，
/// 服务端的 UDP 与 HTTP 同端口，所以它本身也是对的）。
fn parse_discovery_reply(msg: &str, peer_port: u16) -> Option<u16> {
    let rest = msg.trim().strip_prefix("DUSHAN1")?;
    Some(rest.trim().parse::<u16>().unwrap_or(peer_port))
}

/// ① UDP 广播发现。`wait` 是总等待时间，期间每 200 ms 重发一次探测
/// （UDP 会丢包，单发一次在弱信号下命中率很低）。
fn udp_discover(ports: &[u16], wait: Duration) -> Vec<(String, u16)> {
    let mut out: Vec<(String, u16)> = Vec::new();
    let sock = match UdpSocket::bind("0.0.0.0:0") {
        Ok(s) => s,
        Err(_) => return out,
    };
    if sock.set_broadcast(true).is_err() {
        return out;
    }
    let _ = sock.set_read_timeout(Some(Duration::from_millis(120)));
    let deadline = Instant::now() + wait;
    let mut next_send = Instant::now();
    let mut buf = [0u8; 256];
    while Instant::now() < deadline {
        if Instant::now() >= next_send {
            for &p in ports {
                let _ = sock.send_to(b"DUSHAN?", ("255.255.255.255", p));
            }
            next_send = Instant::now() + Duration::from_millis(200);
        }
        match sock.recv_from(&mut buf) {
            Ok((n, peer)) => {
                let msg = String::from_utf8_lossy(&buf[..n]);
                if let Some(port) = parse_discovery_reply(&msg, peer.port()) {
                    let ip = peer.ip().to_string();
                    if !out.iter().any(|(i, p)| *i == ip && *p == port) {
                        out.push((ip, port));
                    }
                }
            }
            Err(_) => { /* 读超时/被拒：继续到 deadline，别把整轮扫描打断 */ }
        }
    }
    out
}

/// ② TCP 扫描兜底：本机 IP 所在 /24 × 端口清单，worker 池并发。
fn tcp_sweep(ports: &[u16], per_timeout: Duration, workers: usize) -> Vec<(String, u16)> {
    let own = match local_ipv4() {
        Some(ip) => ip,
        None => return Vec::new(),
    };
    let o = own.octets();
    let mut targets: Vec<SocketAddr> = Vec::new();
    for h in 1u8..=254 {
        let cand = Ipv4Addr::new(o[0], o[1], o[2], h);
        if cand == own {
            continue;
        }
        for &p in ports {
            targets.push(SocketAddr::from((cand, p)));
        }
    }
    let queue = Arc::new(Mutex::new(targets));
    let hits: Arc<Mutex<Vec<(String, u16)>>> = Arc::new(Mutex::new(Vec::new()));
    let mut handles = Vec::new();
    for _ in 0..workers.max(1) {
        let q = queue.clone();
        let h = hits.clone();
        handles.push(std::thread::spawn(move || loop {
            let next = { q.lock().unwrap().pop() };
            let addr = match next {
                Some(a) => a,
                None => break,
            };
            if TcpStream::connect_timeout(&addr, per_timeout).is_ok() {
                h.lock().unwrap().push((addr.ip().to_string(), addr.port()));
            }
        }));
    }
    for h in handles {
        let _ = h.join();
    }
    let out = hits.lock().unwrap().clone();
    out
}

/// 启动页「扫描局域网」：返回可用服务端列表（按延迟升序）。
///
/// 载荷：`{ok, found:[{url,ip,port,latencyMs,status}], mode:"udp"|"sweep",
///        udpHits, elapsedMs, ports:[...]}`
#[tauri::command]
fn scan_lan(timeout_ms: Option<u64>) -> serde_json::Value {
    let t0 = Instant::now();
    let wait = Duration::from_millis(timeout_ms.unwrap_or(1200).clamp(300, 4000));
    let mut cands = udp_discover(&SCAN_PORTS, wait);
    let udp_hits = cands.len();
    let mut mode = "udp";
    if cands.is_empty() {
        mode = "sweep";
        cands = tcp_sweep(&SCAN_PORTS, Duration::from_millis(220), 48);
    }
    let mut found: Vec<serde_json::Value> = Vec::new();
    for (ip, port) in cands {
        let (ok, status, latency_ms) = probe_meta(&ip, port, "");
        if !ok {
            continue;
        }
        found.push(serde_json::json!({
            "url": format!("http://{ip}:{port}"),
            "ip": ip, "port": port,
            "latencyMs": latency_ms,
            "status": status,
        }));
    }
    found.sort_by_key(|v| v["latencyMs"].as_u64().unwrap_or(u64::MAX));
    serde_json::json!({
        "ok": !found.is_empty(),
        "found": found,
        "mode": mode,
        "udpHits": udp_hits,
        "elapsedMs": t0.elapsed().as_millis(),
        "ports": SCAN_PORTS,
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
        .setup(move |app_handle| {
            // 对局记录 / 回放要落到**应用私有目录**（安卓没有 $HOME，内核自己
            // 猜不出可写路径）。必须在起服务之前设好，否则第一手结算就可能
            // 写到不可写目录、记录直接丢。
            {
                use tauri::Manager;
                match app_handle.path().app_data_dir() {
                    Ok(dir) => {
                        let hist = dir.join("history");
                        offline_core::records::set_dir(&hist);
                        println!("[offline] 对局记录目录 {}", hist.display());
                    }
                    Err(e) => eprintln!("[offline] 取应用数据目录失败（记录可能不可用）：{e}"),
                }
            }
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
            scan_lan,
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

    #[test]
    fn discovery_reply_is_parsed() {
        assert_eq!(parse_discovery_reply("DUSHAN1 8770\n", 9), Some(8770));
        assert_eq!(parse_discovery_reply("DUSHAN1 8771", 9), Some(8771));
        // 端口段缺失/非法 → 退回来源端口
        assert_eq!(parse_discovery_reply("DUSHAN1", 8770), Some(8770));
        assert_eq!(parse_discovery_reply("DUSHAN1 abc", 8770), Some(8770));
        // 不是本协议的报文一律忽略（网段里别的广播不能被当成服务端）
        assert_eq!(parse_discovery_reply("HELLO 1", 8770), None);
        assert_eq!(parse_discovery_reply("", 8770), None);
    }

    #[test]
    fn scan_reports_shapes_even_when_empty() {
        // 300 ms 内没应答也必须返回结构完整的 JSON（前端直接读 found/mode）。
        let r = scan_lan(Some(300));
        assert!(r["found"].is_array());
        assert!(r["ports"].is_array());
        assert_eq!(r["ok"], !r["found"].as_array().unwrap().is_empty());
    }
}
