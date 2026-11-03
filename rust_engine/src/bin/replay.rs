//! 规则回放对拍入口（支持两种起局方式）。
//!
//! ## 1. 完整对局（随机/模型打的真实牌局）
//! ```text
//! DEALER 2
//! WALL 0 1 2 ...（108 个数）
//! A 0 5          # 动作：kind tile（tile=255 表示无牌）
//! ```
//!
//! ## 2. 状态快照（定向覆盖抢杠/一炮多响/过水/报叫等边界）
//! ```text
//! SNAP
//! PHASE 2
//! CURRENT 1
//! HANDS 00210...
//! ...
//! A 1 255
//! ```
//!
//! 两种都能混用；输出统一为每步动作执行**前**的状态指纹：
//! `S0 phase=1 actor=2 drawn=1 wall=57 hands=... acts=0,5,6`

use std::io::{self, Read, Write};

use dushan_engine::actions::*;
use dushan_engine::fan::FanTable;
use dushan_engine::state::{Config, Game};

fn hands_fp(g: &Game) -> String {
    let mut out = String::with_capacity(4 * 28);
    for p in 0..4 {
        if p > 0 {
            out.push('|');
        }
        for t in 0..27 {
            out.push((b'0' + g.hands[p][t]) as char);
        }
    }
    out
}

fn melds_fp(g: &Game) -> String {
    let mut out = String::new();
    for p in 0..4 {
        if p > 0 {
            out.push('|');
        }
        if g.melds[p].is_empty() {
            out.push('-');
        } else {
            for (i, m) in g.melds[p].iter().enumerate() {
                if i > 0 {
                    out.push(',');
                }
                out.push_str(&format!("{}.{}.{}", m.0, m.1, m.2));
            }
        }
    }
    out
}

fn discards_fp(g: &Game) -> String {
    let mut out = String::new();
    for p in 0..4 {
        if p > 0 {
            out.push('|');
        }
        if g.discards[p].is_empty() {
            out.push('-');
        } else {
            for (i, t) in g.discards[p].iter().enumerate() {
                if i > 0 {
                    out.push(',');
                }
                out.push_str(&t.to_string());
            }
        }
    }
    out
}

fn bitmap(b: [bool; 4]) -> String {
    b.iter().map(|&x| if x { '1' } else { '0' }).collect()
}

fn parse_bools(s: &str) -> [bool; 4] {
    let mut r = [false; 4];
    for (i, ch) in s.chars().take(4).enumerate() {
        r[i] = ch == '1';
    }
    r
}

fn parse_hands(s: &str) -> [[u8; 27]; 4] {
    let mut h = [[0u8; 27]; 4];
    for (p, part) in s.split('|').take(4).enumerate() {
        for (t, ch) in part.chars().take(27).enumerate() {
            h[p][t] = ch.to_digit(10).unwrap_or(0) as u8;
        }
    }
    h
}

fn parse_melds(s: &str) -> [Vec<(u8, u8, u8)>; 4] {
    let mut m: [Vec<(u8, u8, u8)>; 4] = [Vec::new(), Vec::new(), Vec::new(), Vec::new()];
    for (p, part) in s.split('|').take(4).enumerate() {
        if part == "-" || part.is_empty() {
            continue;
        }
        for item in part.split(',') {
            let v: Vec<u8> = item.split('.').filter_map(|x| x.parse().ok()).collect();
            if v.len() >= 2 {
                m[p].push((v[0], v[1], if v.len() > 2 { v[2] } else { 0 }));
            }
        }
    }
    m
}

fn parse_discards(s: &str) -> [Vec<u8>; 4] {
    let mut d: [Vec<u8>; 4] = [Vec::new(), Vec::new(), Vec::new(), Vec::new()];
    for (p, part) in s.split('|').take(4).enumerate() {
        if part == "-" || part.is_empty() {
            continue;
        }
        for item in part.split(',') {
            if let Ok(v) = item.parse::<u8>() {
                d[p].push(v);
            }
        }
    }
    d
}

fn parse_list(s: &str) -> Vec<u8> {
    if s == "-" || s.is_empty() {
        return Vec::new();
    }
    s.split(',').filter_map(|x| x.parse().ok()).collect()
}

fn parse_i8_list(s: &str) -> [i8; 3] {
    let mut r = [-1i8; 3];
    for (i, x) in s.split(',').take(3).enumerate() {
        r[i] = x.parse().unwrap_or(-1);
    }
    r
}

fn apply_snap(g: &mut Game, key: &str, val: &[&str]) {
    match key {
        "DEALER" => g.dealer = val[0].parse().unwrap(),
        "WALL" => {
            g.wall = val.iter().map(|x| x.parse::<u8>().unwrap()).collect();
        }
        "WALLPOS" => g.wall_pos = val[0].parse().unwrap(),
        "CURRENT" => g.current = val[0].parse().unwrap(),
        "PHASE" => g.phase = val[0].parse().unwrap(),
        "DRAWN" => g.drawn = val[0] == "1",
        "HANDS" => g.hands = parse_hands(val[0]),
        "MELDS" => g.melds = parse_melds(val[0]),
        "DISCARDS" => g.discards = parse_discards(val[0]),
        "PASS" => {
            for (p, x) in val[0].split('|').take(4).enumerate() {
                g.pass_lock[p] = x.parse().unwrap_or(0);
            }
        }
        "BAOJIAO" => g.baojiao = parse_bools(val[0]),
        "FIRST" => g.first_discard_done = parse_bools(val[0]),
        "TENPAI" => g.ever_tenpai = parse_bools(val[0]),
        "GANGD" => g.gang_discard = parse_bools(val[0]),
        "REPAO" => g.repao_discard = val[0] == "1",
        "AFTERGANG" => g.after_gang_draw = val[0] == "1",
        "ANY" => g.any_discard = val[0] == "1",
        "LASTD" => {
            let p: i8 = val[0].parse().unwrap_or(-1);
            g.last_discard = if p < 0 {
                None
            } else {
                Some((p as u8, val[1].parse().unwrap()))
            };
        }
        "ROBK" => {
            let p: i8 = val[0].parse().unwrap_or(-1);
            g.rob_kong = if p < 0 {
                None
            } else {
                Some((p as u8, val[1].parse().unwrap()))
            };
            g.rob_kong_konger = val.get(2).and_then(|x| x.parse().ok()).unwrap_or(0);
        }
        "RESP" => g.respond_order = parse_list(val[0]),
        "RESPIDX" => g.respond_idx = val[0].parse().unwrap(),
        "NHU" => g.n_hu = val[0].parse().unwrap(),
        "VOTES" => g.respond_hu_votes = parse_list(val[0]),
        "ZEREN" => {
            let z = parse_i8_list(val[0]);
            g.zeren_player = z[0];
            g.zeren_ponger = z[1];
            g.chongfeng_player = z[2];
        }
        "FIRSTYAOJI" => g.first_yaoji_discarded = val[0] == "1",
        "JI" => {
            let v: Vec<&str> = val[0].split(',').collect();
            if let Ok(m) = v[0].parse::<u32>() {
                g.ji_tiles = m;
            }
        }
        _ => {}
    }
}

fn main() {
    let mut buf = String::new();
    io::stdin().read_to_string(&mut buf).unwrap();
    let ft = FanTable::build();

    let mut dealer = 0u8;
    let mut wall: Vec<u8> = Vec::new();
    let mut cfg = Config::default();
    let mut actions: Vec<Action> = Vec::new();
    let mut snap: Option<Game> = None;

    for line in buf.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let mut it = line.split_whitespace();
        let key = it.next().unwrap_or("");
        let rest: Vec<&str> = it.collect();
        match key {
            "SNAP" => snap = Some(Game::blank()),
            "A" => {
                let k: u8 = rest[0].parse().unwrap();
                let t: u8 = rest[1].parse().unwrap();
                actions.push(Action::new(k, t));
            }
            "CFG" => {
                for kv in rest {
                    let mut sp = kv.split('=');
                    let k = sp.next().unwrap_or("");
                    let v: u8 = sp.next().unwrap_or("1").parse().unwrap_or(1);
                    match k {
                        "pass_lock" => cfg.enable_pass_lock = v != 0,
                        "qiang_gang" => cfg.enable_qiang_gang = v != 0,
                        "baojiao_auto" => cfg.baojiao_auto = v != 0,
                        "pinghu_needs_dou" => cfg.ping_hu_needs_dou_for_ron = v != 0,
                        _ => {}
                    }
                }
            }
            _ => {
                if let Some(g) = snap.as_mut() {
                    apply_snap(g, key, &rest);
                    g.cfg = cfg;
                } else if key == "DEALER" {
                    dealer = rest[0].parse().unwrap();
                } else if key == "WALL" {
                    wall = rest.iter().map(|x| x.parse::<u8>().unwrap()).collect();
                }
            }
        }
    }

    let mut g = match snap {
        Some(mut s) => {
            s.cfg = cfg;
            s
        }
        None => Game::new_with_wall(wall, dealer, cfg, &ft),
    };

    let stdout = io::stdout();
    let mut out = io::BufWriter::new(stdout.lock());

    for (i, a) in actions.iter().enumerate() {
        let mut codes: Vec<u16> = g.legal_actions(&ft).iter().map(|x| x.code()).collect();
        codes.sort_unstable();
        codes.dedup();
        let acts: Vec<String> = codes.iter().map(|c| c.to_string()).collect();
        writeln!(
            out,
            "S{} phase={} actor={} drawn={} wall={} hands={} melds={} discards={} baojiao={} tenpai={} acts={}",
            i,
            g.phase,
            g.actor(),
            if g.drawn { 1 } else { 0 },
            g.wall_left(),
            hands_fp(&g),
            melds_fp(&g),
            discards_fp(&g),
            bitmap(g.baojiao),
            bitmap(g.ever_tenpai),
            acts.join(",")
        )
        .unwrap();
        g.step(*a, &ft);
    }
    match g.outcome {
        Some(o) => {
            let w: Vec<String> = o.winners[..o.n_winners as usize]
                .iter()
                .map(|x| x.to_string())
                .collect();
            writeln!(
                out,
                "Z n_winners={} winners={} tile={} tsumo={} loser={} rob={} steps={}",
                o.n_winners,
                if w.is_empty() { "-".to_string() } else { w.join(",") },
                o.win_tile,
                if o.is_tsumo { 1 } else { 0 },
                o.loser,
                if o.rob_kong { 1 } else { 0 },
                g.steps
            )
            .unwrap();
        }
        None => {
            writeln!(out, "Z unfinished steps={}", g.steps).unwrap();
        }
    }
}
