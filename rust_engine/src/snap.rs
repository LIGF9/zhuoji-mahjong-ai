//! 快照 JSON → `Game` 的**唯一**字段解析实现（2026-11-05）。
//!
//! 两条桥共用本实现，杜绝字段漂移：
//! * Python 桥（pymod）：`json.dumps(snapshot_of(...))` → `game_from_json`
//! * Rust 标注器（bin/annotate）：JSONL 状态文件 → `game_from_json`
//!
//! 字段清单与 `zhuoji/rust_search.py::snapshot_of` 一一对应；
//! 缺失字段走默认值（= 旧行为），供旧快照兼容。

use serde_json::Value;

use crate::actions::PHASE_OVER;
use crate::state::{Config, Game};
use crate::tiles::NUM_TILES;

fn j_i64(v: &Value, k: &str, default: i64) -> i64 {
    v.get(k).and_then(|x| x.as_i64()).unwrap_or(default)
}
fn j_bool(v: &Value, k: &str, default: bool) -> bool {
    v.get(k).and_then(|x| x.as_bool()).unwrap_or(default)
}
fn j_f64(v: &Value, k: &str, default: f64) -> f64 {
    v.get(k).and_then(|x| x.as_f64()).unwrap_or(default)
}
fn j_veci(v: &Value, k: &str) -> Vec<i64> {
    v.get(k)
        .and_then(|x| x.as_array())
        .map(|a| a.iter().filter_map(|e| e.as_i64()).collect())
        .unwrap_or_default()
}
fn j_vecf(v: &Value, k: &str) -> Vec<f64> {
    v.get(k)
        .and_then(|x| x.as_array())
        .map(|a| a.iter().filter_map(|e| e.as_f64()).collect())
        .unwrap_or_default()
}

/// 与 pymod `game_from_py` 同语义（该函数现委托到这里）。
pub fn game_from_json(d: &Value) -> Game {
    let mut g = Game::blank();
    g.cfg = Config {
        enable_pass_lock: j_bool(d, "pass_lock", true),
        enable_qiang_gang: j_bool(d, "qiang_gang", true),
        baojiao_auto: j_bool(d, "baojiao_auto", true),
        ping_hu_needs_dou_for_ron: j_bool(d, "pinghu_needs_dou", true),
        ..Config::default()
    };
    g.wall = j_veci(d, "wall").iter().map(|&x| x as u8).collect();
    g.wall_pos = j_i64(d, "wall_pos", 0).max(0) as usize;
    g.dealer = j_i64(d, "dealer", 0) as u8;
    g.dealer_streak = j_i64(d, "dealer_streak", 0).max(0) as u32;
    g.current = j_i64(d, "current", 0) as u8;
    g.phase = j_i64(d, "phase", PHASE_OVER as i64) as u8;
    g.drawn = j_bool(d, "drawn", false);

    let hands = j_veci(d, "hands");
    for p in 0..4 {
        for t in 0..NUM_TILES {
            g.hands[p][t] = *hands.get(p * NUM_TILES + t).unwrap_or(&0) as u8;
        }
    }
    // melds 扁平化：[player, kind, tile, src] × n
    for p in 0..4 {
        g.melds[p].clear();
    }
    let mv = j_veci(d, "melds");
    let mut i = 0usize;
    while i + 3 < mv.len() {
        let p = mv[i] as usize;
        if p < 4 {
            g.melds[p].push((mv[i + 1] as u8, mv[i + 2] as u8, mv[i + 3] as u8));
        }
        i += 4;
    }
    for p in 0..4 {
        g.discards[p].clear();
    }
    let dv = j_veci(d, "discards");
    let mut i = 0usize;
    while i + 1 < dv.len() {
        let p = dv[i] as usize;
        if p < 4 {
            g.discards[p].push(dv[i + 1] as u8);
        }
        i += 2;
    }
    let pl = j_veci(d, "pass_lock");
    for p in 0..4 {
        g.pass_lock[p] = *pl.get(p).unwrap_or(&0) as u32;
    }
    let flags = |key: &str| -> [bool; 4] {
        let v = j_veci(d, key);
        let mut r = [false; 4];
        for p in 0..4 {
            r[p] = *v.get(p).unwrap_or(&0) != 0;
        }
        r
    };
    g.baojiao = flags("baojiao");
    g.ever_tenpai = flags("ever_tenpai");
    g.first_discard_done = flags("first_discard_done");
    g.gang_discard = flags("gang_discard");
    g.repao_discard = j_bool(d, "repao_discard", false);
    g.after_gang_draw = j_bool(d, "after_gang_draw", false);
    g.any_discard = j_bool(d, "any_discard", false);
    g.first_yaoji_discarded = j_bool(d, "first_yaoji_discarded", false);
    g.chongfeng_player = j_i64(d, "chongfeng_player", -1) as i8;
    g.zeren_player = j_i64(d, "zeren_player", -1) as i8;
    g.zeren_ponger = j_i64(d, "zeren_ponger", -1) as i8;

    let ld = j_veci(d, "last_discard");
    g.last_discard = if ld.len() == 2 && ld[0] >= 0 {
        Some((ld[0] as u8, ld[1] as u8))
    } else {
        None
    };
    let rk = j_veci(d, "rob_kong");
    g.rob_kong = if rk.len() >= 2 && rk[0] >= 0 {
        Some((rk[0] as u8, rk[1] as u8))
    } else {
        None
    };
    g.rob_kong_konger = *rk.get(2).unwrap_or(&0) as u8;
    g.respond_order = j_veci(d, "respond_order").iter().map(|&x| x as u8).collect();
    g.respond_idx = j_i64(d, "respond_idx", 0).max(0) as usize;
    g.respond_hu_votes = j_veci(d, "respond_hu_votes")
        .iter()
        .map(|&x| x as u8)
        .collect();
    g.n_hu = j_i64(d, "n_hu", 0).max(0) as usize;
    g.ji_tiles = j_i64(d, "ji_tiles", 1 << 9) as u32;
    g.cfg.jiesuan_fanji = j_i64(d, "jiesuan_fanji", 3) as u8;
    g.cfg.kaiju_fanji = j_i64(d, "kaiju_fanji", 0) as u8;
    let kf = j_i64(d, "kaiju_flip", -1);
    g.kaiju_flip = if kf >= 0 { Some(kf as u8) } else { None };

    // ---- 鸡历史 + 结算配置全量（2026-11-05）----
    g.last_draw = j_i64(d, "last_draw", 255).clamp(0, 255) as u8;
    g.last_discard_was_gang = j_bool(d, "last_discard_was_gang", false);
    let je = j_veci(d, "ji_events"); // [p, tag] × n
    let jev = j_vecf(d, "ji_events_val");
    let jet = j_veci(d, "ji_event_tiles");
    let n_je = je.len() / 2;
    g.ji_events.clear();
    g.ji_event_tiles.clear();
    for i in 0..n_je {
        let val = jev.get(i).copied().unwrap_or(0.0) as f32;
        g.ji_events.push((je[i * 2] as u8, val, je[i * 2 + 1] as u8));
        let jt = jet.get(i).copied().unwrap_or(27);
        g.ji_event_tiles
            .push(if (0..27).contains(&jt) { jt as u8 } else { 27 });
    }
    let cj = j_veci(d, "claimed_ji"); // [打出者, 碰/杠者, tag, 明杠, 牌] × n
    g.claimed_ji.clear();
    let n_cj = cj.len() / 5;
    for i in 0..n_cj {
        g.claimed_ji.push((
            cj[i * 5] as u8,
            cj[i * 5 + 1] as u8,
            cj[i * 5 + 2] as u8,
            cj[i * 5 + 3] != 0,
            cj[i * 5 + 4] as u8,
        ));
    }
    let cap = j_veci(d, "captured_ji"); // [胡牌者, tag, 牌] × n
    let capv = j_vecf(d, "captured_ji_val");
    g.captured_ji.clear();
    let n_cap = cap.len() / 3;
    for i in 0..n_cap {
        let val = capv.get(i).copied().unwrap_or(0.0) as f32;
        g.captured_ji.push((
            cap[i * 3] as u8,
            val,
            cap[i * 3 + 1] as u8,
            cap[i * 3 + 2] as u8,
        ));
    }
    g.hengji_done = j_i64(d, "hengji_done", 0).max(0) as u32;
    g.hengji_open = j_i64(d, "hengji_open", 0).max(0) as u32;
    let hs = j_veci(d, "hengji_seats");
    for (i, v) in hs.iter().enumerate().take(27) {
        g.hengji_seats[i] = *v as u8;
    }
    let tl = j_veci(d, "ji_tag_of_last");
    for (i, v) in tl.iter().enumerate().take(4) {
        g.ji_tag_of_last[i] = *v as u8;
    }
    let ho = j_veci(d, "hengji_opens");
    for (i, v) in ho.iter().enumerate().take(4) {
        g.hengji_opens[i] = (*v).clamp(0, 255) as u8;
    }
    g.hengji_starter = j_i64(d, "hengji_starter", -1).clamp(-1, 3) as i8;
    g.hengji_event_n = j_i64(d, "hengji_event_n", 0).max(0) as u32;
    // 结算配置（网页可调值表 + 开关）
    g.cfg.mantiangji = j_bool(d, "cfg_mantiangji", g.cfg.mantiangji);
    g.cfg.jin_mult = j_f64(d, "cfg_jin_mult", g.cfg.jin_mult as f64) as f32;
    g.cfg.end_baoji = j_bool(d, "cfg_end_baoji", g.cfg.end_baoji);
    g.cfg.end_baogang = j_bool(d, "cfg_end_baogang", g.cfg.end_baogang);
    g.cfg.huang_baoji = j_bool(d, "cfg_huang_baoji", g.cfg.huang_baoji);
    g.cfg.huang_baogang = j_bool(d, "cfg_huang_baogang", g.cfg.huang_baogang);
    g.cfg.huang_baodapai = j_bool(d, "cfg_huang_baodapai", g.cfg.huang_baodapai);
    g.cfg.baojiao_bonus = j_f64(d, "cfg_baojiao_bonus", g.cfg.baojiao_bonus as f64) as f32;
    g.cfg.ji_hold = j_f64(d, "cfg_ji_hold", g.cfg.ji_hold as f64) as f32;
    g.cfg.ji_discard_plain = j_f64(d, "cfg_ji_discard_plain", g.cfg.ji_discard_plain as f64) as f32;
    g.cfg.ji_chongfeng = j_f64(d, "cfg_ji_chongfeng", g.cfg.ji_chongfeng as f64) as f32;
    g.cfg.ji_heng = j_f64(d, "cfg_ji_heng", g.cfg.ji_heng as f64) as f32;
    g.cfg.ji_fanji = j_f64(d, "cfg_ji_fanji", g.cfg.ji_fanji as f64) as f32;
    g.cfg.kong_chickens = j_f64(d, "cfg_kong_chickens", g.cfg.kong_chickens as f64) as f32;
    g.cfg.zi_mo = j_f64(d, "cfg_zi_mo", g.cfg.zi_mo as f64) as f32;
    g.cfg.dian_pao = j_f64(d, "cfg_dian_pao", g.cfg.dian_pao as f64) as f32;
    g.cfg.gang_kai = j_f64(d, "cfg_gang_kai", g.cfg.gang_kai as f64) as f32;
    g.cfg.gang_pao = j_f64(d, "cfg_gang_pao", g.cfg.gang_pao as f64) as f32;
    g.cfg.da_dui_zi = j_f64(d, "cfg_da_dui_zi", g.cfg.da_dui_zi as f64) as f32;
    g.cfg.qi_dui = j_f64(d, "cfg_qi_dui", g.cfg.qi_dui as f64) as f32;
    g.cfg.long_qi_dui = j_f64(d, "cfg_long_qi_dui", g.cfg.long_qi_dui as f64) as f32;
    g.cfg.dan_diao = j_f64(d, "cfg_dan_diao", g.cfg.dan_diao as f64) as f32;
    g.cfg.qing_yi_se = j_f64(d, "cfg_qing_yi_se", g.cfg.qing_yi_se as f64) as f32;
    g
}
