//! 独山麻将 · 单机离线版 Rust 核心。
//!
//! 分层（对应 `docs/离线版-可行性与实施方案.md` §7 的成本剖析）：
//!   * 规则/结算/编码/搜索：直接依赖 `dushan-engine`（rust_engine），一行不改；
//!   * `view`：Game → `/api/state` JSON 视图（O0）；
//!   * `session`：会话循环与三家 AI（O1）；
//!   * `policy`：tract 推理 / 启发式降级（O2）；
//!   * `http`：纯 std 的 HTTP/1.1 + 16 个接口路由（O3）；
//!   * `assets`：静态资源与 ONNX 编译期嵌入。
//!
//! 设计红线：不引入异步运行时、不依赖 JNI AssetManager、不捆 ORT .so。

// 模块按 O0→O3 逐个加入并各自通过对拍后才声明；spike 阶段保持空壳，
// 避免「声明了但没实现」污染 `cargo build`。
pub mod view;
pub mod policy;
pub mod ting;
pub mod search;
pub mod rules_cfg;
pub mod session;
pub mod assets;
pub mod http;
pub mod bench;
