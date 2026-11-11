//! 结算对拍入口：stdin 吃 JSON 行，stdout 吐结算 JSON 行。
//!
//! 输入：{"wall":[108 牌], "dealer":0, "actions":[[kind,tile],...], "cfg":{...}}
//! 输出：{"ok":true,"deltas":[...],"result":{...}} 或 {"ok":false,"error":"..."}

use dushan_engine::actions::*;
use dushan_engine::fan::FanTable;
use dushan_engine::state::{bit, fanji_mask, Config, Game};
use dushan_engine::tiles::YAOJI;
use dushan_engine::settle;
use serde_json::{json, Value};

fn mode_code(v: Option<&Value>) -> u8 {
    match v.and_then(|x| x.as_str()) {
        Some("up") => 1,
        Some("down") => 2,
        Some("both") => 3,
        _ => 0,
    }
}

fn parse_cfg(d: &Value) -> Config {
    let mut c = Config::default();
    c.jiesuan_fanji = mode_code(d.get("jiesuan_fanji"));
    c.kaiju_fanji = mode_code(d.get("kaiju_fanji"));
    if let Some(x) = d.get("mantiangji").and_then(|v| v.as_bool()) {
        c.mantiangji = x;
    }
    if let Some(x) = d.get("jin_mult").and_then(|v| v.as_f64()) {
        c.jin_mult = x as f32;
    }
    if let Some(x) = d.get("end_baoji").and_then(|v| v.as_bool()) {
        c.end_baoji = x;
    }
    if let Some(x) = d.get("end_baogang").and_then(|v| v.as_bool()) {
        c.end_baogang = x;
    }
    if let Some(x) = d.get("huang_baoji").and_then(|v| v.as_bool()) {
        c.huang_baoji = x;
    }
    if let Some(x) = d.get("huang_baogang").and_then(|v| v.as_bool()) {
        c.huang_baogang = x;
    }
    if let Some(x) = d.get("huang_baodapai").and_then(|v| v.as_bool()) {
        c.huang_baodapai = x;
    }
    // 2026-11-07：第一巡同轮跟打（默认开，对拍定向 case 会显式传开关/赔付值）
    if let Some(x) = d.get("first_round_follow").and_then(|v| v.as_bool()) {
        c.first_round_follow = x;
    }
    if let Some(x) = d.get("follow_pay").and_then(|v| v.as_f64()) {
        c.follow_pay = x as f32;
    }
    if let Some(x) = d.get("follow_hold_chickens").and_then(|v| v.as_u64()) {
        c.follow_hold = x.min(255) as u8;
    }
    // 2026-11-11：追加鸡牌种 + 鸡分值（离线版以前忽略前端 rules，这里做对拍覆盖）
    if let Some(arr) = d.get("extra_ji").and_then(|v| v.as_array()) {
        let mut m = 0u32;
        for x in arr {
            if let Some(t) = x.as_u64() {
                if t < 27 && t as u8 != YAOJI {
                    m |= bit(t as u8);
                }
            }
        }
        c.extra_ji = m;
    }
    if let Some(x) = d.get("ji_chongfeng").and_then(|v| v.as_f64()) { c.ji_chongfeng = x as f32; }
    if let Some(x) = d.get("ji_heng").and_then(|v| v.as_f64()) { c.ji_heng = x as f32; }
    if let Some(x) = d.get("ji_discard_plain").and_then(|v| v.as_f64()) { c.ji_discard_plain = x as f32; }
    if let Some(x) = d.get("ji_hold").and_then(|v| v.as_f64()) { c.ji_hold = x as f32; }
    c
}

fn run_one(line: &str) -> String {
    let d: Value = match serde_json::from_str(line) {
        Ok(v) => v,
        Err(e) => return format!("{{\"ok\":false,\"error\":\"parse:{}\"}}", e),
    };
    let cfg = parse_cfg(d.get("cfg").unwrap_or(&Value::Null));
    let wall: Vec<u8> = d["wall"].as_array().unwrap()
        .iter().map(|x| x.as_u64().unwrap() as u8).collect();
    let dealer = d.get("dealer").and_then(|v| v.as_u64()).unwrap_or(0) as u8;
    let ft = FanTable::build();
    let mut g = Game::new_with_wall(wall, dealer, cfg, &ft);
    for a in d["actions"].as_array().unwrap() {
        let kind = a[0].as_u64().unwrap_or(6) as u8;
        let tile = a[1].as_u64().unwrap_or(27) as u8;
        let tile = if tile >= 27 { 255 } else { tile };
        let act = if kind == PASS { Action::pass() }
        else if kind == HU { Action::hu() }
        else { Action::new(kind, tile) };
        g.step(act, &ft);
        if g.phase == PHASE_OVER {
            break;
        }
    }
    if g.phase != PHASE_OVER {
        return "{\"ok\":false,\"error\":\"not_over\"}".to_string();
    }
    let out = match g.outcome {
        Some(o) if o.n_winners > 0 => settle::settle_win(&mut g, &ft),
        _ => settle::settle_huangzhuang(&g, &ft),
    };
    // 诊断字段（对拍用，前端可忽略）
    let mut out = out;
    if let Value::Object(m) = &mut out {
        m.insert("diag_kaiji_mask".into(),
                 json!(g.kaiju_flip.map(|f| fanji_mask(f, cfg.kaiju_fanji)).unwrap_or(0)));
        m.insert("diag_melds".into(),
                 json!(g.melds.iter().map(|ms| ms.iter()
                     .map(|mm| [mm.0, mm.1, mm.2]).collect::<Vec<_>>())
                     .collect::<Vec<_>>()));
        m.insert("diag_flags".into(), json!({
            "huang_baogang": cfg.huang_baogang, "huang_baoji": cfg.huang_baoji,
            "huang_baodapai": cfg.huang_baodapai,
        }));
        m.insert("diag_follow".into(), json!({
            "dealer": g.follow_dealer,
            "tile": if g.follow_tile < 27 { Some(g.follow_tile) } else { None },
            "total_discards": g.total_discards,
            "first_discard_n": g.first_discard_n,
            "switch": cfg.first_round_follow,
            "pay": cfg.follow_pay,
            "hold": cfg.follow_hold,
        }));
    }
    format!("{{\"ok\":true,\"result\":{}}}", out)
}

fn main() {
    use std::io::BufRead;
    let stdin = std::io::stdin();
    for (idx, line) in stdin.lock().lines().enumerate() {
        let line = line.unwrap();
        if line.trim().is_empty() {
            continue;
        }
        let out = std::panic::catch_unwind(|| run_one(&line))
            .unwrap_or_else(|_| format!("{{\"ok\":false,\"error\":\"panic@{}\"}}", idx));
        println!("{}", out);
    }
}
