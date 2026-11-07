//! PyO3 绑定：把 Rust 搜索内核暴露给 Python。
//!
//! Python 侧只做两件事：网络前向（批量）、终局精确结算（重放动作轨迹）。
//! 状态推进、determinization、克隆全留在 Rust。

use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::actions::*;
use crate::encode::{GLOBAL_DIM, NUM_CHANNELS};
use crate::fan::FanTable;
use crate::deep::DeepSearch;
use crate::search::Search;
use crate::state::{Config, Game};
use crate::tiles::NUM_TILES;

fn g_i64(d: &Bound<'_, PyDict>, k: &str, default: i64) -> i64 {
    d.get_item(k).ok().flatten()
        .and_then(|v| v.extract::<i64>().ok()).unwrap_or(default)
}
fn g_bool(d: &Bound<'_, PyDict>, k: &str, default: bool) -> bool {
    d.get_item(k).ok().flatten()
        .and_then(|v| v.extract::<bool>().ok()).unwrap_or(default)
}
fn g_vec(d: &Bound<'_, PyDict>, k: &str) -> Vec<i64> {
    d.get_item(k).ok().flatten()
        .and_then(|v| v.extract::<Vec<i64>>().ok()).unwrap_or_default()
}
fn g_vecf(d: &Bound<'_, PyDict>, k: &str) -> Vec<f64> {
    d.get_item(k).ok().flatten()
        .and_then(|v| v.extract::<Vec<f64>>().ok()).unwrap_or_default()
}
fn g_f64(d: &Bound<'_, PyDict>, k: &str, default: f64) -> f64 {
    d.get_item(k).ok().flatten()
        .and_then(|v| v.extract::<f64>().ok()).unwrap_or(default)
}

/// Python `DushanGame` 字段 → Rust `Game`。
///
/// 2026-11-05：字段解析统一走 `snap::game_from_json`（JSON 单一实现），
/// Python dict 先 `json.dumps` 再解析，保证与 Rust 标注器读同一套字段语义。
/// 旧逐字段版本保留在 `game_from_py_legacy`（不再调用），供对照。
fn game_from_py(d: &Bound<'_, PyDict>) -> PyResult<Game> {
    let py = d.py();
    let json_mod = py.import("json").map_err(|e| {
        pyo3::exceptions::PyRuntimeError::new_err(format!("json 模块不可用: {e}"))
    })?;
    let s: String = json_mod
        .call_method1("dumps", (d,))?
        .extract()?;
    let v: serde_json::Value = serde_json::from_str(&s).map_err(|e| {
        pyo3::exceptions::PyRuntimeError::new_err(format!("快照 JSON 解析失败: {e}"))
    })?;
    Ok(crate::snap::game_from_json(&v))
}

#[allow(dead_code)]
fn game_from_py_legacy(d: &Bound<'_, PyDict>) -> PyResult<Game> {
    let mut g = Game::blank();
    g.cfg = Config {
        enable_pass_lock: g_bool(d, "pass_lock", true),
        enable_qiang_gang: g_bool(d, "qiang_gang", true),
        baojiao_auto: g_bool(d, "baojiao_auto", true),
        ping_hu_needs_dou_for_ron: g_bool(d, "pinghu_needs_dou", true),
        ..Config::default()
    };
    g.wall = g_vec(d, "wall").iter().map(|&x| x as u8).collect();
    g.wall_pos = g_i64(d, "wall_pos", 0).max(0) as usize;
    g.dealer = g_i64(d, "dealer", 0) as u8;
    g.dealer_streak = g_i64(d, "dealer_streak", 0).max(0) as u32;
    g.current = g_i64(d, "current", 0) as u8;
    g.phase = g_i64(d, "phase", PHASE_OVER as i64) as u8;
    g.drawn = g_bool(d, "drawn", false);

    let hands = g_vec(d, "hands");
    for p in 0..4 {
        for t in 0..NUM_TILES {
            g.hands[p][t] = *hands.get(p * NUM_TILES + t).unwrap_or(&0) as u8;
        }
    }
    // melds 扁平化：[player, kind, tile, src] × n
    for p in 0..4 { g.melds[p].clear(); }
    let mv = g_vec(d, "melds");
    let mut i = 0usize;
    while i + 3 < mv.len() {
        let p = mv[i] as usize;
        if p < 4 {
            g.melds[p].push((mv[i + 1] as u8, mv[i + 2] as u8, mv[i + 3] as u8));
        }
        i += 4;
    }
    for p in 0..4 { g.discards[p].clear(); }
    let dv = g_vec(d, "discards");
    let mut i = 0usize;
    while i + 1 < dv.len() {
        let p = dv[i] as usize;
        if p < 4 {
            g.discards[p].push(dv[i + 1] as u8);
        }
        i += 2;
    }
    let pl = g_vec(d, "pass_lock");
    for p in 0..4 {
        g.pass_lock[p] = *pl.get(p).unwrap_or(&0) as u32;
    }
    let flags = |key: &str| -> [bool; 4] {
        let v = g_vec(d, key);
        let mut r = [false; 4];
        for p in 0..4 { r[p] = *v.get(p).unwrap_or(&0) != 0; }
        r
    };
    g.baojiao = flags("baojiao");
    g.ever_tenpai = flags("ever_tenpai");
    g.first_discard_done = flags("first_discard_done");
    g.gang_discard = flags("gang_discard");
    g.repao_discard = g_bool(d, "repao_discard", false);
    g.after_gang_draw = g_bool(d, "after_gang_draw", false);
    g.any_discard = g_bool(d, "any_discard", false);
    g.first_yaoji_discarded = g_bool(d, "first_yaoji_discarded", false);
    g.chongfeng_player = g_i64(d, "chongfeng_player", -1) as i8;
    g.zeren_player = g_i64(d, "zeren_player", -1) as i8;
    g.zeren_ponger = g_i64(d, "zeren_ponger", -1) as i8;

    let ld = g_vec(d, "last_discard");
    g.last_discard = if ld.len() == 2 && ld[0] >= 0 {
        Some((ld[0] as u8, ld[1] as u8))
    } else { None };
    let rk = g_vec(d, "rob_kong");
    g.rob_kong = if rk.len() >= 2 && rk[0] >= 0 {
        Some((rk[0] as u8, rk[1] as u8))
    } else { None };
    g.rob_kong_konger = *rk.get(2).unwrap_or(&0) as u8;
    g.respond_order = g_vec(d, "respond_order").iter().map(|&x| x as u8).collect();
    g.respond_idx = g_i64(d, "respond_idx", 0).max(0) as usize;
    g.respond_hu_votes = g_vec(d, "respond_hu_votes").iter().map(|&x| x as u8).collect();
    g.n_hu = g_i64(d, "n_hu", 0).max(0) as usize;
    g.ji_tiles = g_i64(d, "ji_tiles", 1 << 9) as u32;
    // 2026-11-04：Python 快照补齐了开局指示牌与两种翻鸡模式，Rust 搜索模拟的
    // 结算必须与真实 Python 结算同源（否则开局金鸡局会漏掉金鸡倍率）。
    g.cfg.jiesuan_fanji = g_i64(d, "jiesuan_fanji", 3) as u8;   // 默认 both
    g.cfg.kaiju_fanji = g_i64(d, "kaiju_fanji", 0) as u8;       // 默认 off
    let kf = g_i64(d, "kaiju_flip", -1);
    g.kaiju_flip = if kf >= 0 { Some(kf as u8) } else { None };

    // ---- 2026-11-05：鸡历史 + 结算配置全量过桥（Rust 内结算的正确性前提）----
    // 缺这些字段时，叶子终局直算会与 Python「从 determinization 起点重放」
    // 分叉（对拍 max|ΔQ|>4）。全部带默认值，旧快照仍可用（=旧行为）。
    g.last_draw = g_i64(d, "last_draw", 255).clamp(0, 255) as u8;
    g.last_discard_was_gang = g_bool(d, "last_discard_was_gang", false);
    let je = g_vec(d, "ji_events");                 // [p, tag] × n
    let jev = g_vecf(d, "ji_events_val");           // [f32] × n
    let jet = g_vec(d, "ji_event_tiles");
    let n_je = je.len() / 2;
    g.ji_events.clear();
    g.ji_event_tiles.clear();
    for i in 0..n_je {
        let val = jev.get(i).copied().unwrap_or(0.0) as f32;
        g.ji_events.push((je[i * 2] as u8, val, je[i * 2 + 1] as u8));
        let jt = jet.get(i).copied().unwrap_or(27);
        g.ji_event_tiles.push(if (0..27).contains(&jt) { jt as u8 } else { 27 });
    }
    let cj = g_vec(d, "claimed_ji");                // [打出者, 碰/杠者, tag, 明杠, 牌] × n
    g.claimed_ji.clear();
    let n_cj = cj.len() / 5;
    for i in 0..n_cj {
        g.claimed_ji.push((cj[i * 5] as u8, cj[i * 5 + 1] as u8, cj[i * 5 + 2] as u8,
                           cj[i * 5 + 3] != 0, cj[i * 5 + 4] as u8));
    }
    let cap = g_vec(d, "captured_ji");              // [胡牌者, tag, 牌] × n
    let capv = g_vecf(d, "captured_ji_val");
    g.captured_ji.clear();
    let n_cap = cap.len() / 3;
    for i in 0..n_cap {
        let val = capv.get(i).copied().unwrap_or(0.0) as f32;
        g.captured_ji.push((cap[i * 3] as u8, val, cap[i * 3 + 1] as u8,
                            cap[i * 3 + 2] as u8));
    }
    g.hengji_done = g_i64(d, "hengji_done", 0).max(0) as u32;
    g.hengji_open = g_i64(d, "hengji_open", 0).max(0) as u32;
    let hs = g_vec(d, "hengji_seats");
    for (i, v) in hs.iter().enumerate().take(27) {
        g.hengji_seats[i] = *v as u8;
    }
    let tl = g_vec(d, "ji_tag_of_last");
    for (i, v) in tl.iter().enumerate().take(4) {
        g.ji_tag_of_last[i] = *v as u8;
    }
    let ho = g_vec(d, "hengji_opens");
    for (i, v) in ho.iter().enumerate().take(4) {
        g.hengji_opens[i] = (*v).clamp(0, 255) as u8;
    }
    g.hengji_starter = g_i64(d, "hengji_starter", -1).clamp(-1, 3) as i8;
    g.hengji_event_n = g_i64(d, "hengji_event_n", 0).max(0) as u32;
    // 2026-11-07：第一巡同轮跟打（与 snap.rs 保持同源；本函数已停用，仅作对照）
    g.follow_dealer = g_bool(d, "follow_dealer", g.follow_dealer);
    g.follow_tile = g_i64(d, "follow_tile", 255).clamp(0, 255) as u8;
    let fdt = g_vec(d, "first_discard_tiles");
    for (i, v) in fdt.iter().enumerate().take(4) {
        g.first_discard_tiles[i] = if (0..27).contains(v) { *v as u8 } else { 255 };
    }
    let fdo = g_vec(d, "first_discard_order");
    g.first_discard_n = fdo.len().min(4) as u8;
    for (i, v) in fdo.iter().enumerate().take(4) {
        g.first_discard_order[i] = (*v).clamp(0, 3) as u8;
    }
    g.total_discards = g_i64(d, "total_discards", 0).clamp(0, u16::MAX as i64) as u16;
    // 结算配置（网页可调值表 + 开关）
    g.cfg.first_round_follow = g_bool(d, "cfg_first_round_follow", g.cfg.first_round_follow);
    g.cfg.follow_pay = g_f64(d, "cfg_follow_pay", g.cfg.follow_pay as f64) as f32;
    g.cfg.follow_hold = g_i64(d, "cfg_follow_hold_chickens", g.cfg.follow_hold as i64)
        .clamp(0, 255) as u8;
    g.cfg.mantiangji = g_bool(d, "cfg_mantiangji", g.cfg.mantiangji);
    g.cfg.jin_mult = g_f64(d, "cfg_jin_mult", g.cfg.jin_mult as f64) as f32;
    g.cfg.end_baoji = g_bool(d, "cfg_end_baoji", g.cfg.end_baoji);
    g.cfg.end_baogang = g_bool(d, "cfg_end_baogang", g.cfg.end_baogang);
    g.cfg.huang_baoji = g_bool(d, "cfg_huang_baoji", g.cfg.huang_baoji);
    g.cfg.huang_baogang = g_bool(d, "cfg_huang_baogang", g.cfg.huang_baogang);
    g.cfg.huang_baodapai = g_bool(d, "cfg_huang_baodapai", g.cfg.huang_baodapai);
    g.cfg.baojiao_bonus = g_f64(d, "cfg_baojiao_bonus", g.cfg.baojiao_bonus as f64) as f32;
    g.cfg.ji_hold = g_f64(d, "cfg_ji_hold", g.cfg.ji_hold as f64) as f32;
    g.cfg.ji_discard_plain = g_f64(d, "cfg_ji_discard_plain", g.cfg.ji_discard_plain as f64) as f32;
    g.cfg.ji_chongfeng = g_f64(d, "cfg_ji_chongfeng", g.cfg.ji_chongfeng as f64) as f32;
    g.cfg.ji_heng = g_f64(d, "cfg_ji_heng", g.cfg.ji_heng as f64) as f32;
    g.cfg.ji_fanji = g_f64(d, "cfg_ji_fanji", g.cfg.ji_fanji as f64) as f32;
    g.cfg.kong_chickens = g_f64(d, "cfg_kong_chickens", g.cfg.kong_chickens as f64) as f32;
    g.cfg.zi_mo = g_f64(d, "cfg_zi_mo", g.cfg.zi_mo as f64) as f32;
    g.cfg.dian_pao = g_f64(d, "cfg_dian_pao", g.cfg.dian_pao as f64) as f32;
    g.cfg.gang_kai = g_f64(d, "cfg_gang_kai", g.cfg.gang_kai as f64) as f32;
    g.cfg.gang_pao = g_f64(d, "cfg_gang_pao", g.cfg.gang_pao as f64) as f32;
    g.cfg.da_dui_zi = g_f64(d, "cfg_da_dui_zi", g.cfg.da_dui_zi as f64) as f32;
    g.cfg.qi_dui = g_f64(d, "cfg_qi_dui", g.cfg.qi_dui as f64) as f32;
    g.cfg.long_qi_dui = g_f64(d, "cfg_long_qi_dui", g.cfg.long_qi_dui as f64) as f32;
    g.cfg.dan_diao = g_f64(d, "cfg_dan_diao", g.cfg.dan_diao as f64) as f32;
    g.cfg.qing_yi_se = g_f64(d, "cfg_qing_yi_se", g.cfg.qing_yi_se as f64) as f32;
    Ok(g)
}

#[pyclass]
struct PySearch {
    inner: Search,
}

#[pymethods]
impl PySearch {
    #[new]
    #[pyo3(signature = (snapshot, p, legal, n_det=32, ff_cap=4, seed=0, lam=1.0, reweight=false))]
    fn new(snapshot: &Bound<'_, PyDict>, p: u8, legal: Vec<(u8, u8)>,
           n_det: usize, ff_cap: usize, seed: u64, lam: f64, reweight: bool) -> PyResult<Self> {
        let g = game_from_py(snapshot)?;
        let ft = FanTable::build();
        let acts: Vec<Action> = legal.iter().map(|&(k, t)| Action::new(k, t)).collect();
        Ok(PySearch { inner: Search::new_fair(&g, p, &acts, n_det, ff_cap, seed, &ft,
                                              true, lam, reweight) })
    }

    fn needs_net(&self) -> bool { self.inner.needs_net() }
    fn has_unsettled(&self) -> bool { self.inner.has_unsettled() }
    fn finished(&self) -> bool { self.inner.finished() }

    /// (tiles[n*405] f32, glob[n*70] f32, head[n] u8, mask[n*28] u8, kind[n] u8)
    fn take_batch<'py>(&self, py: Python<'py>) -> PyResult<(
        Bound<'py, PyArray1<f32>>,
        Bound<'py, PyArray1<f32>>,
        Bound<'py, PyArray1<u8>>,
        Bound<'py, PyArray1<u8>>,
        Bound<'py, PyArray1<u8>>,
    )> {
        let b = self.inner.take_batch();
        let n = b.obs.len();
        let mut tiles = vec![0f32; n * NUM_CHANNELS * NUM_TILES];
        let mut glob = vec![0f32; n * GLOBAL_DIM];
        let mut mask = vec![0u8; n * 28];
        let mut head = Vec::with_capacity(n);
        let mut kind = Vec::with_capacity(n);
        for (i, o) in b.obs.iter().enumerate() {
            tiles[i * NUM_CHANNELS * NUM_TILES..(i + 1) * NUM_CHANNELS * NUM_TILES]
                .copy_from_slice(&o.tiles);
            glob[i * GLOBAL_DIM..(i + 1) * GLOBAL_DIM].copy_from_slice(&o.glob);
            mask[i * 28..i * 28 + 28].copy_from_slice(&o.mask);
            head.push(b.head[i]);
            kind.push(b.kind[i]);
        }
        Ok((
            tiles.into_pyarray(py),
            glob.into_pyarray(py),
            head.into_pyarray(py),
            mask.into_pyarray(py),
            kind.into_pyarray(py),
        ))
    }

    fn feed(&mut self, actions: Vec<u8>, values: PyReadonlyArray1<f32>) {
        let v = values.as_slice().unwrap_or(&[]);
        self.inner.feed(&actions, v);
    }

    /// 待结算任务：[(task_idx, origin_wall, origin_pos, hands_flat, [(kind,tile)...])]
    fn unsettled(&self) -> Vec<(usize, Vec<u8>, usize, Vec<u8>, Vec<(u8, u8)>)> {
        self.inner.unsettled_tasks().iter().map(|&i| {
            let gi = self.inner.group_of(i);
            let o = self.inner.origin(gi);
            let mut hv = Vec::with_capacity(4 * NUM_TILES);
            for p in 0..4 {
                for t in 0..NUM_TILES {
                    hv.push(o.hands[p][t]);
                }
            }
            (i, o.wall.clone(), o.wall_pos, hv,
             self.inner.trace_of(i).iter().map(|a| (a.kind, a.tile)).collect())
        }).collect()
    }

    fn settle(&mut self, i: usize, delta: f64) { self.inner.settle(i, delta); }

    /// Rust 内结算：返回 [(task_idx, delta)]，见 `Search::settle_internal`。
    /// 调用方（rust_search.py）默认走此路径，不再 Python 重放动作轨迹。
    fn settle_internal(&mut self) -> Vec<(usize, f64)> { self.inner.settle_internal() }

    fn action_q(&self) -> Vec<f64> { self.inner.action_q() }

    fn n_legal(&self) -> usize { self.inner.n_origins() }
    fn n_tasks(&self) -> usize { self.inner.tasks.len() }

    /// 各任务的叶子游戏（仅调试：返回最后的 action 轨迹长度分布）
    fn stats(&self) -> Vec<usize> {
        self.inner.tasks.iter().map(|t| t.trace.len()).collect()
    }
}

fn batch_arrays<'py>(py: Python<'py>, b: &crate::search::Batch)
    -> PyResult<(Bound<'py, PyArray1<f32>>, Bound<'py, PyArray1<f32>>,
                 Bound<'py, PyArray1<u8>>, Bound<'py, PyArray1<u8>>,
                 Bound<'py, PyArray1<u8>>)> {
    let n = b.obs.len();
    let mut tiles = vec![0f32; n * NUM_CHANNELS * NUM_TILES];
    let mut glob = vec![0f32; n * GLOBAL_DIM];
    let mut mask = vec![0u8; n * 28];
    let mut head = Vec::with_capacity(n);
    let mut kind = Vec::with_capacity(n);
    for (i, o) in b.obs.iter().enumerate() {
        tiles[i * NUM_CHANNELS * NUM_TILES..(i + 1) * NUM_CHANNELS * NUM_TILES]
            .copy_from_slice(&o.tiles);
        glob[i * GLOBAL_DIM..(i + 1) * GLOBAL_DIM].copy_from_slice(&o.glob);
        mask[i * 28..i * 28 + 28].copy_from_slice(&o.mask);
        head.push(b.head[i]);
        kind.push(b.kind[i]);
    }
    Ok((tiles.into_pyarray(py), glob.into_pyarray(py), head.into_pyarray(py),
        mask.into_pyarray(py), kind.into_pyarray(py)))
}

/// 多层搜索（ply 1/2/3）
#[pyclass]
struct PyDeepSearch {
    inner: DeepSearch,
}

#[pymethods]
impl PyDeepSearch {
    #[new]
    #[pyo3(signature = (snapshot, p, legal, n_det=32, ff_cap=24, topk=3, ply=1, seed=0, fair=true, agg=0, lam=1.0, reweight=false))]
    fn new(snapshot: &Bound<'_, PyDict>, p: u8, legal: Vec<(u8, u8)>, n_det: usize,
           ff_cap: usize, topk: usize, ply: u8, seed: u64, fair: bool, agg: u8,
           lam: f64, reweight: bool) -> PyResult<Self> {
        let g = game_from_py(snapshot)?;
        let ft = FanTable::build();
        let acts: Vec<Action> = legal.iter().map(|&(k, t)| Action::new(k, t)).collect();
        Ok(PyDeepSearch {
            inner: DeepSearch::new_fair(&g, p, &acts, n_det, ff_cap, topk, ply, seed,
                                        &ft, fair, agg, lam, reweight),
        })
    }

    /// 逐 determinization 的 Q（[d][动作]，诊断/对拍用）
    fn per_det(&self) -> Vec<Vec<f64>> {
        self.inner.per_det()
    }

    /// **ply=1 专用**：逐 determinization × 动作的终局诊断。
    /// 返回 [d][i] = (terminal, win, delta)，供推荐理由层的
    /// 「N 张假想世界终局统计」（我胡 x / 被胡 y / 终局均分 z）。
    fn det_stats(&self) -> Vec<Vec<(bool, bool, f64)>> {
        self.inner.det_diag()
    }

    fn needs_net(&self) -> bool { self.inner.current().needs_net() }
    fn has_unsettled(&self) -> bool { self.inner.current().has_unsettled() }
    fn stage_done(&self) -> bool { self.inner.stage_done() }
    fn finished(&self) -> bool { self.inner.finished() }
    fn promote(&mut self) -> bool { self.inner.promote() }
    fn stage(&self) -> usize { self.inner.cur }

    fn take_batch<'py>(&self, py: Python<'py>) -> PyResult<(
        Bound<'py, PyArray1<f32>>, Bound<'py, PyArray1<f32>>,
        Bound<'py, PyArray1<u8>>, Bound<'py, PyArray1<u8>>,
        Bound<'py, PyArray1<u8>>,
    )> {
        batch_arrays(py, &self.inner.current().take_batch())
    }

    fn feed(&mut self, actions: Vec<u8>, values: PyReadonlyArray1<f32>) {
        let v = values.as_slice().unwrap_or(&[]);
        self.inner.current_mut().feed(&actions, v);
    }

    /// [(task_idx, root, root_wall, root_pos, root_hands_flat, 完整动作链)]
    /// root 是**第一层** determinization 的编号 —— 重放必须从那里开始。
    fn unsettled(&self) -> Vec<(usize, usize, Vec<u8>, usize, Vec<u8>, Vec<(u8, u8)>)> {
        let s = self.inner.current();
        s.unsettled_tasks().iter().map(|&i| {
            let root = s.root_of(i);
            let o = self.inner.stages[0].origin(root);
            let mut hv = Vec::with_capacity(4 * NUM_TILES);
            for p in 0..4 {
                for t in 0..NUM_TILES {
                    hv.push(o.hands[p][t]);
                }
            }
            (i, root, o.wall.clone(), o.wall_pos, hv,
             s.trace_of(i).iter().map(|a| (a.kind, a.tile)).collect())
        }).collect()
    }

    fn settle(&mut self, i: usize, delta: f64) {
        self.inner.current_mut().settle(i, delta);
    }

    /// Rust 内结算（当前层）：返回 [(task_idx, delta)]。
    fn settle_internal(&mut self) -> Vec<(usize, f64)> {
        self.inner.current_mut().settle_internal()
    }

    fn result(&self) -> Vec<f64> { self.inner.result() }
    fn n_stages(&self) -> usize { self.inner.stages.len() }
}

/// 单局面编码（对拍用）：返回 (tiles[405], glob[70], head, mask[28])
#[pyfunction]
fn encode_one<'py>(py: Python<'py>, snapshot: &Bound<'_, PyDict>, p: u8)
    -> PyResult<(Bound<'py, PyArray1<f32>>, Bound<'py, PyArray1<f32>>, u8, Bound<'py, PyArray1<u8>>)> {
    let g = game_from_py(snapshot)?;
    let ft = FanTable::build();
    let o = crate::encode::encode_v1(&g, p, &ft);
    Ok((
        o.tiles.to_vec().into_pyarray(py),
        o.glob.to_vec().into_pyarray(py),
        o.head,
        o.mask.to_vec().into_pyarray(py),
    ))
}

/// 公平性审计接口：对一个**已知全貌**的局面做一次公平 determinization，
/// 返回 4 家手牌摊平 [p0*27 .. p3*27]。供 `scripts/audit_fair_determinize.py`
/// 验证采样池确实只含「别家手牌 + 未摸牌墙」，不泄漏本家手牌/副露/牌河。
#[pyfunction]
fn determinize_snapshot<'py>(py: Python<'py>, snapshot: &Bound<'_, PyDict>,
                             p: u8, seed: u64) -> PyResult<Bound<'py, PyArray1<i64>>> {
    let g = game_from_py(snapshot)?;
    let mut rng = crate::rng::Rng::new(seed);
    let sim = crate::search::determinize(&g, p as usize, &mut rng);
    let mut v = Vec::with_capacity(4 * NUM_TILES);
    for q in 0..4 {
        for t in 0..NUM_TILES {
            v.push(sim.hands[q][t] as i64);
        }
    }
    Ok(v.into_pyarray(py))
}

#[pymodule]
fn dushan_engine(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PySearch>()?;
    m.add_class::<PyDeepSearch>()?;
    m.add_function(wrap_pyfunction!(encode_one, m)?)?;
    m.add_function(wrap_pyfunction!(determinize_snapshot, m)?)?;
    m.add("NUM_CHANNELS", NUM_CHANNELS)?;
    m.add("GLOBAL_DIM", GLOBAL_DIM)?;
    Ok(())
}
