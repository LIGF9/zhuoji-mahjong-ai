//! 独山麻将规则内核（Rust）—— 目标是替换 Python 侧的牌局模拟，而不是重写网络。
//!
//! 当前进度（2026-11-02 凌晨）
//! ---------------------------
//! * `tiles`：牌编码、牌墙、计数向量
//! * `fan`：胡牌判定（标准型 / 七对）+ 番型集合，已与 `zhuoji/fan.py` 算法一致
//! * 待做：`state`（摸打推进、碰杠胡响应、报叫/通行证/热炮）、`encode`（状态编码）、
//!   搜索树展开
//!
//! 对拍入口是 `bin/parity`：从 stdin 吃行式用例，输出与 Python 同样格式的结果，
//! 由 `scripts/rust_parity.py` 逐行 diff。

pub mod fan;
pub mod tiles;

pub use fan::{FanTable, Meld};
pub use tiles::Counts;

pub mod actions;
pub mod state;

pub use state::{Config, Game, Outcome};

pub mod encode;
pub mod rng;
pub mod search;
pub mod settle;
#[cfg(feature = "python")]
pub mod pymod;
pub mod deep;
