//! O3：纯 `std` 的 HTTP/1.1 服务 + 16 个 `/api/*` 路由。
//!
//! 按 §7.2 决策**不引入 axum/tokio/hyper**：16 个接口全是短 JSON，前端
//! 450 ms 轮询，单连接一线程足够；这也避开异步运行时在 Tauri Android
//! 进程里的生命周期问题。同源托管 `web/static`，前端 `dushan.html` 一字不改。

use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::sync::{Arc, Mutex};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use dushan_engine::fan::FanTable;
use serde_json::{json, Value};

use crate::assets;
use crate::policy::Net;
use crate::session::{Nets, Session};
use crate::view;

pub struct App {
    pub nets: Arc<Nets>,
    pub session: Option<Session>,
    counter: u64,
}

impl App {
    pub fn new() -> Self {
        // 三张网：master（v1）/ champ_slim_s（v3）/ expert_s7（v3，0.12M）。
        // 加载失败只打日志、不 panic —— 单张网坏掉时另外两张还能打。
        let mut map: Vec<(&'static str, Net)> = Vec::new();
        for (stem, bytes, enc) in [
            ("dushan_master", assets::MASTER_ONNX, 1u8),
            ("champ_slim_s", assets::SLIM_S_ONNX, 3),
            ("expert_s7", assets::EXPERT_S7_ONNX, 3),
        ] {
            match Net::from_bytes(bytes, enc) {
                Ok(n) => map.push((stem, n)),
                Err(e) => eprintln!("[offline] {stem} 加载失败: {e}"),
            }
        }
        println!("[offline] tract 模型: {}", map.iter().map(|(s, _)| *s).collect::<Vec<_>>().join(","));
        let nets = Arc::new(Nets { map });
        // 设备端耗时自检：后台线程跑一次，结果直接进 logcat（见 bench.rs）。
        {
            let n = nets.clone();
            let _ = std::thread::Builder::new()
                .name("offline-bench".into())
                .spawn(move || crate::bench::log_latency(&n));
        }
        Self { nets, session: None, counter: 0 }
    }

    fn next_seed(&mut self) -> u64 {
        self.counter += 1;
        let t = SystemTime::now().duration_since(UNIX_EPOCH)
            .map(|d| d.as_nanos() as u64).unwrap_or(1);
        t ^ self.counter.wrapping_mul(0x9E37_79B9_7F4A_7C15)
    }
}

enum Resp {
    Json(u16, Value),
    Static(&'static [u8], &'static str),
    Empty(u16),
}

fn unquote(s: &str) -> String {
    let b = s.as_bytes();
    let mut out = Vec::with_capacity(b.len());
    let mut i = 0;
    while i < b.len() {
        if b[i] == b'%' && i + 2 < b.len() {
            let h = (b[i + 1] as char).to_digit(16);
            let l = (b[i + 2] as char).to_digit(16);
            if let (Some(h), Some(l)) = (h, l) {
                out.push((h * 16 + l) as u8);
                i += 3;
                continue;
            }
        }
        if b[i] == b'+' { out.push(b' '); } else { out.push(b[i]); }
        i += 1;
    }
    String::from_utf8_lossy(&out).into_owned()
}

fn query_get<'a>(q: &'a str, key: &str) -> Option<String> {
    q.split('&').filter(|s| !s.is_empty()).find_map(|kv| {
        let mut it = kv.splitn(2, '=');
        let k = it.next().unwrap_or("");
        if k == key { Some(unquote(it.next().unwrap_or(""))) } else { None }
    })
}

fn split_target(target: &str) -> (&str, &str) {
    match target.split_once('?') {
        Some((p, q)) => (p, q),
        None => (target, ""),
    }
}

fn empty_stats() -> Value {
    json!({"ok": true, "all_time": {
        "games": 0, "zimo": 0, "songhu_games": 0, "score": 0,
        "chongfeng_games": 0, "ji_discard_games": 0, "baoji_games": 0,
        "tenpai_bigfan_games": 0, "win_bigfan_games": 0,
        "huang_games": 0, "huang_noten_games": 0,
        "solo_games": 0, "solo_win_games": 0, "solo_score": 0
    }})
}

fn route(app: &Arc<Mutex<App>>, method: &str, path: &str, q: &str, body: &Value) -> Resp {
    let mut guard = match app.lock() {
        Ok(g) => g,
        Err(_) => return Resp::Json(500, json!({"ok": false, "error": "服务内部状态锁失效"})),
    };
    let a = &mut *guard;

    if method == "GET" {
        return match path {
            "/api/meta" => Resp::Json(200, view::meta()),
            "/api/state" => {
                let nets = a.nets.clone();
                match a.session.as_mut() {
                    Some(s) => Resp::Json(200, s.snapshot(&nets)),
                    None => Resp::Json(404, json!({"ok": false, "ready": false, "error": "会话不存在"})),
                }
            }
            "/api/hint" => {
                let nets = a.nets.clone();
                let strategy = query_get(q, "strategy").unwrap_or_else(|| "model:champ_slim_s".into());
                match a.session.as_mut() {
                    Some(s) => Resp::Json(200, s.hint(&nets, &strategy)),
                    None => Resp::Json(404, json!({"ok": false, "error": "会话不存在"})),
                }
            }
            "/api/ting" => {
                let now = query_get(q, "mode").as_deref() == Some("now");
                match a.session.as_mut() {
                    Some(s) => Resp::Json(200, s.ting(now)),
                    None => Resp::Json(404, json!({"ok": false, "error": "会话不存在"})),
                }
            }
            "/api/history" => Resp::Json(200, json!({"ok": true, "records": []})),
            "/api/history_export" => Resp::Json(200, json!({"exported_at": "", "count": 0, "records": []})),
            "/api/replay" => Resp::Json(200, json!({"ok": false, "error": "离线单机版不保存回放"})),
            "/api/stats" => Resp::Json(200, empty_stats()),
            _ => static_resp(path),
        };
    }

    if method == "POST" {
        return match path {
            "/api/new" => {
                let seed = a.next_seed();
                // 页面 #my-seat 的「随机」是 value="-1"（不是缺字段），Python 侧按
                // 「不在 0..4 就 None → 随机」处理。这里以前用 as_u64 解析，-1 直接
                // 落到 unwrap_or(0) —— 选「随机」永远坐 0 号（东/先手），与线上不一致。
                let me = match body.get("my_seat").and_then(|x| x.as_i64()) {
                    Some(v) if (0..4).contains(&v) => v as u8,
                    // 用 seed 的高位派座位：Session 内部 dealer = seed % 4，
                    // 若这里也用 seed % 4，选「随机」就会**每次都坐庄**。
                    _ => ((seed / 4) % 4) as u8,
                };
                let mut names = ["我", "下家", "对家", "上家"].map(str::to_string);
                if let Some(arr) = body.get("names").and_then(|x| x.as_array()) {
                    for (i, v) in arr.iter().take(4).enumerate() {
                        if let Some(s) = v.as_str() { names[i] = s.to_string(); }
                    }
                }
                // 前端把档位与模型拼成一个串："ting:fast" / "model:dushan_master" /
                // "model:expert_s7" / "model:champ_slim_s"。原样透传给 Session，
                // 由它按前缀（规则/贪心/搜索）与 stem（哪张网）分派 —— 在这里
                // 只拆成 (kind, stem) 再拼回去就会丢掉 ting 的语义。
                let mut strats: [String; 3] = [
                    "model:champ_slim_s".into(),
                    "model:champ_slim_s".into(),
                    "model:champ_slim_s".into(),
                ];
                if let Some(arr) = body.get("opponents").and_then(|x| x.as_array()) {
                    for (i, v) in arr.iter().take(3).enumerate() {
                        if let Some(sel) = v.get("model").and_then(|x| x.as_str()) {
                            if !sel.is_empty() {
                                strats[i] = sel.to_string();
                            }
                        }
                    }
                }
                a.session = Some(Session::new(FanTable::build(), seed, me, &names, &strats));
                Resp::Json(200, json!({"ok": true, "sid": "offline", "my_seat": me}))
            }
            "/api/act" => {
                let kind = body.get("kind").and_then(|x| x.as_str()).unwrap_or("");
                let tile = body.get("tile").and_then(|x| x.as_i64()).unwrap_or(-1);
                match a.session.as_mut() {
                    Some(s) => {
                        let (ok, msg) = s.submit(kind, tile);
                        if ok { Resp::Json(200, json!({"ok": true, "action": msg})) }
                        else { Resp::Json(200, json!({"ok": false, "error": msg})) }
                    }
                    None => Resp::Json(404, json!({"ok": false, "error": "会话不存在"})),
                }
            }
            "/api/next" => match a.session.as_mut() {
                Some(s) => {
                    let ok = s.next_hand();
                    Resp::Json(200, json!({"ok": ok,
                        "error": if ok { Value::Null } else { json!("当前这局还没结束") }}))
                }
                None => Resp::Json(404, json!({"ok": false, "error": "会话不存在"})),
            },
            "/api/end_hand" => {
                if let Some(s) = a.session.as_mut() { s.end_hand_now(); }
                Resp::Json(200, json!({"ok": true}))
            }
            "/api/hold" => Resp::Json(200, json!({"ok": true, "hold": body.get("on").and_then(|x| x.as_bool()).unwrap_or(false)})),
            "/api/speed" => {
                let v = body.get("speed").and_then(|x| x.as_f64()).unwrap_or(0.45);
                if let Some(s) = a.session.as_mut() { s.speed = v.clamp(0.0, 5.0); }
                Resp::Json(200, json!({"ok": true, "speed": v}))
            }
            "/api/history_delete" => Resp::Json(200, json!({"ok": true, "removed_rec": 0, "skipped": []})),
            "/api/history_clear" => Resp::Json(200, json!({"ok": true, "removed_rec": 0})),
            "/api/quit" => Resp::Json(200, json!({"ok": true})),
            _ => Resp::Json(404, json!({"ok": false, "error": "not found"})),
        };
    }

    if method == "HEAD" { return Resp::Empty(200); }
    Resp::Json(405, json!({"ok": false, "error": "method not allowed"}))
}

fn static_resp(path: &str) -> Resp {
    if path == "/favicon.ico" {
        return Resp::Empty(204);
    }
    // 中文资源名必须解码：头像 `/assets/avatar/boy/云天明.jpg` 在请求行里是
    // `%E4%BA%91%E5%A4%A9%E6%98%8E.jpg`，直接拿去查 include_dir 会 404，
    // 实机上表现为「玩家头像全是一排碎图标」（v1.1.0 联网版由 Python 侧解码）。
    let decoded = unquote(path);
    match assets::static_file(&decoded) {
        Some((bytes, mime)) => Resp::Static(bytes, mime),
        None => Resp::Json(404, json!({"ok": false, "error": format!("not found: {decoded}")})),
    }
}

fn read_request(s: &mut TcpStream) -> Option<(String, String, Vec<u8>)> {
    let _ = s.set_read_timeout(Some(Duration::from_secs(10)));
    let mut buf: Vec<u8> = Vec::new();
    let mut tmp = [0u8; 8192];
    let head_end = loop {
        let n = s.read(&mut tmp).ok()?;
        if n == 0 { return None; }
        buf.extend_from_slice(&tmp[..n]);
        if let Some(i) = buf.windows(4).position(|w| w == b"\r\n\r\n") {
            break i + 4;
        }
        if buf.len() > 1 << 20 { return None; }
    };
    let head = String::from_utf8_lossy(&buf[..head_end]).into_owned();
    let mut lines = head.split("\r\n");
    let first = lines.next()?.to_string();
    let mut it = first.split_whitespace();
    let method = it.next()?.to_string();
    let target = it.next()?.to_string();
    let mut len = 0usize;
    for l in lines {
        if let Some(v) = l.to_ascii_lowercase().strip_prefix("content-length:") {
            len = v.trim().parse().unwrap_or(0);
        }
    }
    while buf.len() < head_end + len {
        let n = s.read(&mut tmp).ok()?;
        if n == 0 { break; }
        buf.extend_from_slice(&tmp[..n]);
    }
    let body = buf[head_end..(head_end + len).min(buf.len())].to_vec();
    Some((method, target, body))
}

/// 静态资源的缓存策略。
///
/// 2026-11-11 踩坑：原来**所有**静态资源（含 `dushan.html`）都发
/// `Cache-Control: max-age=86400`。换新 APK 重装后 WebView 仍按 24 小时缓存
/// 拿旧页面 —— 实机现象是「四档标签没变、界面还是旧 bug」，而从 adb 转发
/// 出来的 HTML 明明已经是新的（日志里三张网也都在），排查绕了很久。
/// 结论：**页面与「随时会变」的资源必须 no-store**，只有图片/音频这类
/// 按文件名指代的二进制才配长缓存。
fn cache_for(mime: &str) -> &'static str {
    let m = mime.split(';').next().unwrap_or("").trim();
    match m {
        "text/html" | "text/css" | "application/javascript" | "text/javascript"
        | "application/json" | "application/manifest+json" | "image/svg+xml" => "no-store",
        _ => "max-age=86400",
    }
}

fn write_resp(s: &mut TcpStream, code: u16, ctype: &str, cache: &str, body: &[u8]) {
    let reason = match code {
        200 => "OK", 204 => "No Content", 404 => "Not Found",
        405 => "Method Not Allowed", 500 => "Internal Server Error", _ => "OK",
    };
    let head = format!(
        "HTTP/1.1 {code} {reason}\r\nContent-Type: {ctype}\r\nContent-Length: {}\r\nCache-Control: {cache}\r\nConnection: close\r\n\r\n",
        body.len()
    );
    let _ = s.write_all(head.as_bytes());
    let _ = s.write_all(body);
    let _ = s.flush();
}

fn handle(mut s: TcpStream, app: &Arc<Mutex<App>>) {
    let Some((method, target, raw)) = read_request(&mut s) else { return };
    let (path, q) = split_target(&target);
    let body: Value = if raw.is_empty() {
        Value::Null
    } else {
        serde_json::from_slice(&raw).unwrap_or(Value::Null)
    };
    match route(app, &method, path, q, &body) {
        Resp::Json(code, v) => {
            let b = serde_json::to_vec(&v).unwrap_or_default();
            write_resp(&mut s, code, "application/json; charset=utf-8", "no-store", &b);
        }
        Resp::Static(bytes, mime) => {
            write_resp(&mut s, 200, mime, cache_for(mime), bytes);
        }
        Resp::Empty(code) => {
            write_resp(&mut s, code, "text/plain", "no-store", &[]);
        }
    }
}

/// 在 `127.0.0.1:port` 起服务；`port=0` 时由系统分配并返回实际端口。
/// 服务跑在独立线程，函数立即返回，供 Tauri 把 WebView 导航到该端口。
pub fn serve(port: u16, app: Arc<Mutex<App>>) -> std::io::Result<u16> {
    let listener = TcpListener::bind(("127.0.0.1", port))?;
    let actual = listener.local_addr()?.port();
    std::thread::Builder::new()
        .name("offline-http".into())
        .spawn(move || {
            for stream in listener.incoming() {
                match stream {
                    Ok(s) => {
                        let a = app.clone();
                        let _ = std::thread::Builder::new()
                            .name("offline-conn".into())
                            .spawn(move || handle(s, &a));
                    }
                    Err(e) => {
                        eprintln!("[offline] accept 失败: {e}");
                        break;
                    }
                }
            }
        })?;
    Ok(actual)
}
