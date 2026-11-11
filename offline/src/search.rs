//! 设备端 PIMC 搜索（`search32` / `search128`）：Rust `DeepSearch` 推进 + tract 批量前向。
//!
//! 与联网版的关系：**同一份** `rust_engine::deep`，只是把「网络前向」从
//! 「攒批回传 Python / ORT-CoreML」换成「本进程 tract」。语义完全一致：
//!   * `fair=true`（公平 determinization：只重排别家手牌与牌墙，本家与牌河不动）
//!   * `agg=0`（max 聚合，与线上默认一致）
//!   * `lam=1.0`（均匀信念）、`reweight=false`
//!   * `ply=1`、`topk=3`、`ff_cap=24`（`/api/hint` 与对手出牌的线上默认）
//!
//! 为什么必须做：前端默认推荐档写死为 `search128:master`（`DEFAULT_STRATEGY`），
//! 若离线后端只做贪心，UI 会「说在搜索、实际贪心」——那是逻辑 bug，不是降级。

use dushan_engine::actions::Action;
use dushan_engine::deep::DeepSearch;
use dushan_engine::fan::FanTable;
use dushan_engine::state::Game;

use crate::policy::Net;

/// 搜索档参数（与线上 `/api/hint` 同口径）。
#[derive(Clone, Copy, Debug)]
pub struct Cfg {
    pub n_det: usize,
    pub ff_cap: usize,
    pub topk: usize,
    pub ply: u8,
}

impl Cfg {
    /// `search32` / `search128` → 参数；其余返回 None。
    pub fn from_kind(kind: &str) -> Option<Self> {
        let n_det = match kind {
            "search32" | "n32" | "dets32" => 32,
            "search128" | "n128" | "dets128" => 128,
            _ => return None,
        };
        Some(Self { n_det, ff_cap: 24, topk: 3, ply: 1 })
    }
}

/// 跑完整一次 PIMC，返回与 `legal` 同序的 Q 值。
pub fn deep_q(
    net: &Net,
    g: &Game,
    p: u8,
    legal: &[Action],
    cfg: &Cfg,
    seed: u64,
    ft: &FanTable,
) -> Result<Vec<f64>, String> {
    if legal.is_empty() {
        return Ok(Vec::new());
    }
    let mut ds = DeepSearch::new_fair_enc(
        g, p, legal, cfg.n_det, cfg.ff_cap, cfg.topk, cfg.ply, seed, ft,
        true, 0, 1.0, false, net.enc(),
    );

    let mut guard = 0usize;
    loop {
        if ds.finished() {
            break;
        }
        guard += 1;
        if guard > 200_000 {
            return Err("搜索未收敛（迭代护栏）".into());
        }

        if ds.current().needs_net() {
            let b = ds.current().take_batch();
            let n = b.head.len();
            if n == 0 {
                // 理论上不会发生（needs_net 为真必有 pending）；防死循环。
                return Err("批为空但仍在等网络".into());
            }
            let mut tiles = Vec::with_capacity(n * net.channels() * 27);
            let mut glob = Vec::with_capacity(n * net.global_dim());
            let mut masks: Vec<[u8; 28]> = Vec::with_capacity(n);
            for o in &b.obs {
                tiles.extend_from_slice(&o.tiles);
                glob.extend_from_slice(&o.glob);
                masks.push(o.mask);
            }
            let (acts, vals) = net.search_batch(n, &tiles, &glob, &b.head, &masks)?;
            ds.current_mut().feed(&acts, &vals);
            continue;
        }

        if ds.current().has_unsettled() {
            // 设备端直接用 Rust 结算（与 `Search::settle_internal` 同源）。
            let _ = ds.current_mut().settle_internal();
            continue;
        }

        if ds.stage_done() {
            if !ds.promote() {
                break;
            }
            continue;
        }

        break;
    }

    Ok(ds.result())
}
