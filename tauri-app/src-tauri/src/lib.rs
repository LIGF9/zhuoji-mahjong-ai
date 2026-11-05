//! 独山麻将 · 大师之路 —— Tauri 2 Android 客户端后端。
//!
//! 设计原则（对应「视觉和逻辑都没有 bug」的要求）：
//!   * 本 APP **不复制、不修改**任何游戏 HTML/CSS/JS，也不重写任何麻将规则或 AI。
//!   * WebView 直接进入用户电脑上已运行的 Python 对局服务端，
//!     因此界面、规则、结算、大师推荐与网页版**逐字节一致**。
//!   * Rust 侧只做三件事：记住服务器地址、探测连通性、把 WebView 导航到该地址。
//!   * 所有网络探测只用公开元数据接口 `GET /api/meta`，不读取任何对局数据，
//!     也不持有别家手牌 / 牌墙等隐藏信息。

use std::{
    fs,
    io::{Read, Write},
    net::{TcpStream, ToSocketAddrs},
    path::PathBuf,
    time::{Duration, Instant},
};

use tauri::{Manager, WebviewWindow};

/// 首次启动的默认地址。用户电脑的局域网 IP 会变化，因此可在启动页随时修改；
/// 修改后 APP 会记住（保存于应用数据目录）。
const DEFAULT_SERVER: &str = "http://172.19.32.162:8770";
const CONFIG_FILE: &str = "server.json";
const TIMEOUT_MS: u64 = 2500;

#[derive(serde::Serialize)]
#[serde(rename_all = "camelCase")]
struct CheckResult {
    ok: bool,
    status: String,
    latency_ms: u128,
    url: String,
    error: String,
}

impl CheckResult {
    fn fail(url: &str, error: impl Into<String>) -> Self {
        Self {
            ok: false,
            status: String::new(),
            latency_ms: 0,
            url: url.to_string(),
            error: error.into(),
        }
    }
}

/// 归一化用户输入：补 http://、去尾部斜杠、校验主机名。
/// 允许带路径（反向代理场景），但禁止非 http(s) 协议。
fn normalize_url(raw: &str) -> Result<String, String> {
    let mut s = raw.trim().to_string();
    if s.is_empty() {
        return Err("请输入服务器地址".into());
    }
    // 已带协议则只接受 http/https；否则补 http://（不能无脑前缀，
    // 否则 ftp://x 会被改写成 http://ftp://x 这样的畸形地址）
    if let Some(idx) = s.find("://") {
        let scheme = s[..idx].to_ascii_lowercase();
        match scheme.as_str() {
            "http" | "https" => {}
            other => {
                return Err(format!("不支持的协议 {other}，请用 http:// 或 https://"));
            }
        }
    } else {
        s = format!("http://{}", s);
    }
    let s = s.trim_end_matches('/').to_string();
    let u = tauri::Url::parse(&s).map_err(|e| format!("地址格式错误：{}", e))?;
    match u.scheme() {
        "http" | "https" => {}
        other => return Err(format!("不支持的协议 {}，请用 http:// 或 https://", other)),
    }
    if u.host_str().unwrap_or("").is_empty() {
        return Err("地址缺少主机名".into());
    }
    Ok(s)
}

fn config_path(app: &tauri::AppHandle) -> Option<PathBuf> {
    let dir = app.path().app_config_dir().ok()?;
    let _ = fs::create_dir_all(&dir);
    Some(dir.join(CONFIG_FILE))
}

/// 读取上次成功进入游戏时保存的地址（纯文本，无依赖）。
#[tauri::command]
fn get_server(app: tauri::AppHandle) -> String {
    let Some(path) = config_path(&app) else {
        return DEFAULT_SERVER.to_string();
    };
    match fs::read_to_string(path) {
        Ok(s) => {
            let s = s.trim().to_string();
            if s.is_empty() {
                DEFAULT_SERVER.to_string()
            } else {
                s
            }
        }
        Err(_) => DEFAULT_SERVER.to_string(),
    }
}

/// 连通性探测：只用 std，避免引入 TLS / 异步运行时（Android 交叉编译更稳）。
/// 向服务端请求公开的 `GET /api/meta`，仅检查 HTTP 状态行是否为 200。
#[tauri::command]
fn check_server(url: String) -> CheckResult {
    let normalized = match normalize_url(&url) {
        Ok(u) => u,
        Err(e) => return CheckResult::fail(url.trim(), e),
    };
    let parsed = match tauri::Url::parse(&normalized) {
        Ok(u) => u,
        Err(e) => return CheckResult::fail(&normalized, format!("地址格式错误：{}", e)),
    };

    // HTTPS 不做明文 TCP 探测（std 无 TLS），不阻碍用户直接进入。
    if parsed.scheme() == "https" {
        return CheckResult {
            ok: true,
            status: "https 地址：跳过本地明文探测，直接点「进入游戏」即可".into(),
            latency_ms: 0,
            url: normalized,
            error: String::new(),
        };
    }

    let host = parsed.host_str().unwrap_or("").to_string();
    let port = parsed.port_or_known_default().unwrap_or(8770);
    let addr = format!("{}:{}", host, port);
    let sock = match addr.to_socket_addrs() {
        Ok(mut it) => match it.next() {
            Some(a) => a,
            None => return CheckResult::fail(&normalized, "无法解析主机地址"),
        },
        Err(e) => return CheckResult::fail(&normalized, format!("无法解析主机地址：{}", e)),
    };

    let t0 = Instant::now();
    let mut stream = match TcpStream::connect_timeout(&sock, Duration::from_millis(TIMEOUT_MS)) {
        Ok(s) => s,
        Err(e) => {
            return CheckResult::fail(
                &normalized,
                format!("连不上 {}:{}（{}）。请确认电脑端服务已启动、手机与电脑在同一 Wi-Fi。",
                        host, port, e),
            )
        }
    };
    let _ = stream.set_read_timeout(Some(Duration::from_millis(TIMEOUT_MS)));
    let _ = stream.set_write_timeout(Some(Duration::from_millis(TIMEOUT_MS)));

    let base = parsed.path().trim_end_matches('/');
    let req = format!(
        "GET {}/api/meta HTTP/1.1\r\nHost: {}\r\nUser-Agent: ZhuojiTauri/1.0\r\nAccept: application/json\r\nConnection: close\r\n\r\n",
        base, addr
    );
    if let Err(e) = stream.write_all(req.as_bytes()) {
        return CheckResult::fail(&normalized, format!("发送请求失败：{}", e));
    }

    let mut buf = Vec::new();
    // 读超时是预期内的（服务端为 HTTP/1.0 短连接时也可能先返回 EOF），
    // 忽略 read 错误，使用已收到的字节判断状态行。
    let _ = stream.read_to_end(&mut buf);
    let text = String::from_utf8_lossy(&buf);
    let status = text.lines().next().unwrap_or("").trim().to_string();
    let latency_ms = t0.elapsed().as_millis();
    let ok = status.contains(" 200 ");

    CheckResult {
        ok,
        status: if status.is_empty() {
            "服务端无响应".into()
        } else {
            status
        },
        latency_ms,
        url: normalized,
        error: if ok {
            String::new()
        } else {
            "服务端未返回 HTTP 200，请检查地址与端口（默认 8770）".into()
        },
    }
}

/// 保存地址并让 WebView 载入真实的游戏页面。
/// 关键：**不做任何前端资源替换**，直接用服务端原样的 dushan.html。
#[tauri::command]
fn enter_game(app: tauri::AppHandle, window: WebviewWindow, url: String) -> Result<String, String> {
    let normalized = normalize_url(&url)?;
    let parsed = tauri::Url::parse(&normalized).map_err(|e| format!("地址格式错误：{}", e))?;
    if let Some(path) = config_path(&app) {
        let _ = fs::write(path, &normalized);
    }
    window
        .navigate(parsed)
        .map_err(|e| format!("无法打开游戏页面：{}", e))?;
    Ok(normalized)
}

/// 返回 APP 构建信息，供启动页展示（也用于自动化验证）。
#[tauri::command]
fn app_info() -> serde_json::Value {
    serde_json::json!({
        "name": "独山麻将 · 大师之路",
        "version": env!("CARGO_PKG_VERSION"),
        "identifier": "com.zhuoji.mahjong",
        "mode": "lan-direct",
        "note": "游戏界面与逻辑完全由局域网服务端提供，客户端未做任何改写",
    })
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .invoke_handler(tauri::generate_handler![
            get_server,
            check_server,
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
    fn normalize_adds_scheme_and_trims_slash() {
        assert_eq!(
            normalize_url("172.26.99.125:8770").unwrap(),
            "http://172.26.99.125:8770"
        );
        assert_eq!(
            normalize_url("  http://192.168.1.5:8770/  ").unwrap(),
            "http://192.168.1.5:8770"
        );
        assert_eq!(
            normalize_url("myhost.lan:8770/").unwrap(),
            "http://myhost.lan:8770"
        );
    }

    #[test]
    fn normalize_rejects_bad_input() {
        assert!(normalize_url("").is_err());
        assert!(normalize_url("   ").is_err());
        assert!(normalize_url("ftp://x").is_err());
        assert!(normalize_url("http://").is_err());
    }

    #[test]
    fn normalize_rejects_non_http_scheme() {
        assert!(normalize_url("javascript:alert(1)").is_err());
    }

    #[test]
    fn check_server_rejects_garbage() {
        let r = check_server("not a url".into());
        assert!(!r.ok);
        assert!(!r.error.is_empty());
    }

    #[test]
    fn check_server_unreachable_port_fails_fast() {
        // RFC5737 文档地址，必不通；验证失败路径与错误文案
        let r = check_server("http://203.0.113.1:9".into());
        assert!(!r.ok);
        assert!(r.error.contains("连不上") || r.error.contains("无法解析"));
    }

    /// 需要本机 8770 端口有真实的 Python 服务端在跑（E2E 前提）。
    #[test]
    fn check_server_hits_real_backend() {
        let r = check_server("http://127.0.0.1:8770".into());
        assert!(r.ok, "check_server 失败: {} {}", r.status, r.error);
        assert!(r.status.contains("200"), "status={}", r.status);
        assert!(r.latency_ms < TIMEOUT_MS as u128, "延迟异常: {}", r.latency_ms);
    }
}
