//! O3：静态资源与 ONNX 模型**编译期嵌入**。
//!
//! 按 §7.2 决策：不用 JNI AssetManager、不捆 `libonnxruntime.so`。
//! `web/static`（10 MB / 242 文件）与两个 ONNX（6 MB）直接进 .so 的只读段，
//! 纯 Rust 零平台胶水；APK 里由压缩器再压一遍。

use include_dir::{include_dir, Dir};

pub static STATIC: Dir<'_> = include_dir!("$CARGO_MANIFEST_DIR/../web/static");

pub static SLIM_S_ONNX: &[u8] = include_bytes!("../../models/onnx/champ_slim_s.onnx");
pub static MASTER_ONNX: &[u8] = include_bytes!("../../models/onnx/dushan_master.onnx");

pub fn static_file(path: &str) -> Option<(&'static [u8], &'static str)> {
    let p = path.trim_start_matches('/');
    let p = if p.is_empty() || p == "index.html" { "dushan.html" } else { p };
    if p == "dushan.html" {
        return Some((page_html(), mime_of(p)));
    }
    let f = STATIC.get_file(p)?;
    Some((f.contents(), mime_of(p)))
}

/// 离线版默认推荐档：**贪心 · champ_slim_s**。
///
/// 依据（不是口味）：
///   * P0 三臂实测（`reports/P0-学生N32搜索红利.md`）：学生 N32 搜索对贪心的
///     配对差 −0.978±0.846，CI 跨零 —— 「换成搜索」没有可测的强度收益；
///   * 手机端实测（arm64 模拟器，4 线程）：贪心 1.9 ms / N32 1.25 s / N128 7.2 s，
///     而推荐是**每个决策点**都要算一次，搜索档在手机上不可用；
///   * E-SLIM 结论里 `champ_slim_s` 的强项就是贪心。
/// 通过页面里的 `window.__DUSHAN_DEFAULT_STRATEGY` 钩子生效（见 dushan.html），
/// 只改「出厂默认 / 恢复默认」，用户手动选过的档位仍以 localStorage 为准。
const DEFAULT_STRATEGY: &str = "model:champ_slim_s";

/// `dushan.html` 原样 + 一段「出厂默认」引导脚本（同源，故 localStorage 可用）。
///
/// 不改磁盘上的页面文件：联网版与离线版共用 `web/static/dushan.html`，
/// 只在离线服务出站时注入，避免两边互相污染。
fn page_html() -> &'static [u8] {
    static CACHE: std::sync::OnceLock<Vec<u8>> = std::sync::OnceLock::new();
    CACHE.get_or_init(|| {
        let raw = STATIC
            .get_file("dushan.html")
            .expect("web/static/dushan.html 缺失")
            .contents();
        let src = String::from_utf8_lossy(raw);
        // 只注入一个全局量，不碰 localStorage：页面自己的 migrateCfgVersion
        // 会用 DEFAULT_STRATEGY 覆盖存储值，靠写 localStorage 抢先是没有用的
        // （踩过：注入生效但界面仍显示「极算」）。
        let boot = format!(
            "<script>/*offline-boot*/window.__DUSHAN_DEFAULT_STRATEGY='{DEFAULT_STRATEGY}';</script>"
        );
        // 插在 <head> 之后：必须早于主脚本读 SET.strategy，也早于任何渲染。
        let out = match src.find("<head>") {
            Some(i) => {
                let at = i + "<head>".len();
                format!("{}{}{}", &src[..at], boot, &src[at..])
            }
            None => format!("{boot}{src}"),
        };
        out.into_bytes()
    })
}

pub fn mime_of(path: &str) -> &'static str {
    let ext = path.rsplit('.').next().unwrap_or("").to_ascii_lowercase();
    match ext.as_str() {
        "html" | "htm" => "text/html; charset=utf-8",
        "js" => "application/javascript; charset=utf-8",
        "css" => "text/css; charset=utf-8",
        "json" => "application/json; charset=utf-8",
        "png" => "image/png",
        "jpg" | "jpeg" => "image/jpeg",
        "webp" => "image/webp",
        "svg" => "image/svg+xml",
        "gif" => "image/gif",
        "mp3" => "audio/mpeg",
        "ogg" => "audio/ogg",
        "wav" => "audio/wav",
        "woff" => "font/woff",
        "woff2" => "font/woff2",
        "ttf" => "font/ttf",
        "ico" => "image/x-icon",
        _ => "application/octet-stream",
    }
}
