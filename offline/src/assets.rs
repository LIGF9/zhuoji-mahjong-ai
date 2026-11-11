//! O3：静态资源与 ONNX 模型**编译期嵌入**。
//!
//! 按 §7.2 决策：不用 JNI AssetManager、不捆 `libonnxruntime.so`。
//! `web/static`（10 MB / 242 文件）与两个 ONNX（6 MB）直接进 .so 的只读段，
//! 纯 Rust 零平台胶水；APK 里由压缩器再压一遍。

use include_dir::{include_dir, Dir};

pub static STATIC: Dir<'_> = include_dir!("$CARGO_MANIFEST_DIR/../web/static");

pub static SLIM_S_ONNX: &[u8] = include_bytes!("../../models/onnx/champ_slim_s.onnx");
pub static MASTER_ONNX: &[u8] = include_bytes!("../../models/onnx/dushan_master.onnx");
/// expert 档：0.12M 的 32×3 小网（v3 编码，470 KB）。
/// 原 s7 权重已随 2026-11-10 归档删除且远端无副本，这一份是按同定位重训的
/// 替代品 —— 见 docs/离线版-可行性与实施方案.md §9。
pub static EXPERT_S7_ONNX: &[u8] = include_bytes!("../../models/onnx/expert_s7.onnx");

pub fn static_file(path: &str) -> Option<(&'static [u8], &'static str)> {
    let p = path.trim_start_matches('/');
    let p = if p.is_empty() || p == "index.html" { "dushan.html" } else { p };
    if p == "dushan.html" {
        return Some((page_html(), mime_of(p)));
    }
    let f = STATIC.get_file(p)?;
    Some((f.contents(), mime_of(p)))
}

/// 离线版四档：**全部贪心**（不搜索）。
///
/// 档位与权重的对应（强度见 `reports/离线版-四档强度矩阵.md`）：
///   * `ting:fast`        入门 · 速听（纯规则，向听数驱动）
///   * `model:expert_s7`  进阶 · 小网（0.12M）
///   * `model:dushan_master` 大师 · 主力（0.77M）
///   * `model:champ_slim_s`  传奇 · 学生（0.82M，最强）
///
/// 为什么默认给最强的 legend：四档每步都是 0.9–3 ms 量级，选最弱也不会更快，
/// 却会明显更弱。难度对用户是「想被虐多狠」的选项，不是性能旋钮。
const DEFAULT_STRATEGY: &str = "model:champ_slim_s";

/// 注入到页面里的四档表。前端只在存在该全局量时才替换下拉选项，
/// 联网版不注入 → 仍是「大师 / 深算 / 极算」，两边行为不分叉。
const TIERS_JSON: &str = concat!(
    r#"[{"v":"ting:fast","label":"🐣 入门 · 速听"},"#,
    r#"{"v":"model:expert_s7","label":"🙂 进阶 · 小网"},"#,
    r#"{"v":"model:dushan_master","label":"🤖 大师 · 主力"},"#,
    r#"{"v":"model:champ_slim_s","label":"👑 传奇 · 学生"}]"#
);


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
            "<script>/*offline-boot*/window.__DUSHAN_DEFAULT_STRATEGY='{DEFAULT_STRATEGY}';\
             window.__DUSHAN_TIERS={TIERS_JSON};</script>"
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
