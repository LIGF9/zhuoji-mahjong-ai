//! 终局结算 —— 与 `zhuoji/dushan.py` 的 `_settle_win_multi` / `_settle_huangzhuang`
//! 逐行对齐（鸡牌计分 / 金鸡 / 四倍金鸡 / 责任鸡溢价 / 包鸡 / 包杠 / 流局包大牌面）。
//!
//! 输出与 Python `game.result` 同键的 JSON（前端结算页直接消费），
//! `deltas` 数值必须与 Python 完全一致（对拍由 `bin/settle_parity.rs` 保证）。

use crate::actions::*;
use crate::fan::{self, FanTable, MELD_KONG_ADDED, MELD_KONG_CONCEALED,
                 MELD_KONG_EXPOSED, Meld};
use crate::state::{fanji_mask, bit, Config, Game, JI_CHONGFENG, JI_HENG, JI_PLAIN};
use crate::tiles::{tile_cn, Counts, NUM_TILES, YAOJI};
use serde_json::{json, Map, Value};

pub const BAO_KONG: f64 = 3.0;

pub fn fan_mode_str(m: u8) -> &'static str {
    match m {
        1 => "up",
        2 => "down",
        3 => "both",
        _ => "off",
    }
}

pub fn fan_cn_of(tp: &str) -> String {
    match tp {
        fan::PING_HU => "平胡".to_string(),
        fan::DA_DUI_ZI => "大对子".to_string(),
        fan::QI_DUI => "小七对".to_string(),
        fan::LONG_QI_DUI => "龙七对".to_string(),
        fan::QING_YI_SE => "清一色".to_string(),
        fan::DAN_DIAO => "单吊".to_string(),
        _ => tp.to_string(),
    }
}

fn ji_tag_cn(tag: u8) -> &'static str {
    match tag {
        JI_CHONGFENG => "冲锋鸡",
        JI_HENG => "横鸡",
        _ => "幺鸡",
    }
}

fn hu_type_value(tp: &str, cfg: &Config) -> f64 {
    match tp {
        fan::DA_DUI_ZI => cfg.da_dui_zi as f64,
        fan::QI_DUI => cfg.qi_dui as f64,
        fan::LONG_QI_DUI => cfg.long_qi_dui as f64,
        fan::QING_YI_SE => cfg.qing_yi_se as f64,
        fan::DAN_DIAO => cfg.dan_diao as f64,
        _ => 0.0,
    }
}

/// 裸计数版可胡张（对齐 Python fan.winning_tiles）：标准牌数才判。
fn winning_tiles_raw(c: &Counts, melds: &[Meld], ft: &FanTable) -> Vec<u8> {
    let n_melds = melds.len();
    if n_melds > 4 {
        return Vec::new();
    }
    let expect = (4 - n_melds) * 3 + 1;
    let mut sum = 0u32;
    for &x in c.iter() {
        sum += x as u32;
    }
    if sum as usize != expect {
        return Vec::new();
    }
    let mut res = Vec::new();
    for t in 0..NUM_TILES as u8 {
        if c[t as usize] >= 4 {
            continue;
        }
        let mut h = *c;
        h[t as usize] += 1;
        if fan::is_winning_hand(&h, melds, ft) {
            res.push(t);
        }
    }
    res
}

/// 终局听牌判定（对齐 DushanGame._tenpai_flag）：标准数直接判，
/// 非标准数（杠后未补牌/被抢杠等）按「打出任意一张后能胡」容错。
pub fn tenpai_flag(g: &Game, p: usize, ft: &FanTable) -> bool {
    if !g.winning_tiles(p, ft).is_empty() {
        return true;
    }
    let n = g.melds[p].len();
    let mut sum = 0u32;
    for &x in g.hands[p].iter() {
        sum += x as u32;
    }
    if sum as usize == (4 - n) * 3 + 1 {
        return false;
    }
    for d in 0..NUM_TILES as u8 {
        if g.hands[p][d as usize] == 0 {
            continue;
        }
        let mut h = g.hands[p];
        h[d as usize] -= 1;
        if !winning_tiles_raw(&h, &g.melds[p], ft).is_empty() {
            return true;
        }
    }
    false
}

/// 听牌者「可达最大牌面」(分值, 牌型中文名)，对齐 _max_hu_type。
fn max_hu_type(g: &Game, p: usize, cfg: &Config, ft: &FanTable) -> (f64, String) {
    let mut tiles = winning_tiles_raw(&g.hands[p], &g.melds[p], ft);
    let n = g.melds[p].len();
    let mut sum = 0u32;
    for &x in g.hands[p].iter() {
        sum += x as u32;
    }
    if tiles.is_empty() && sum as usize != (4 - n) * 3 + 1 {
        for d in 0..NUM_TILES as u8 {
            if g.hands[p][d as usize] == 0 {
                continue;
            }
            let mut h = g.hands[p];
            h[d as usize] -= 1;
            let wt = winning_tiles_raw(&h, &g.melds[p], ft);
            if !wt.is_empty() {
                tiles = wt;
                break;
            }
        }
    }
    let mut best_v = 0.0f64;
    let mut best_cn = String::new();
    for t in tiles {
        let mut h = g.hands[p];
        h[t as usize] += 1;
        let types = fan::win_type_set(&h, &g.melds[p], Some(t), ft);
        let v: f64 = types.iter().map(|tp| hu_type_value(tp, cfg)).sum();
        if v > best_v {
            best_v = v;
            let mut named: Vec<String> = types.iter()
                .filter(|tp| **tp != fan::PING_HU)
                .map(|tp| fan_cn_of(tp)).collect();
            named.sort();
            best_cn = named.join("+");
        }
    }
    (best_v, best_cn)
}

/// 各家「自己打出的鸡」(值/标签/牌种)，对齐 _own_ji_after_claims。
/// 被碰/杠走的鸡从打出者名下抵消；金鸡/四倍金鸡按牌种乘倍率。
#[allow(clippy::type_complexity)]
fn own_ji_after_claims(g: &Game, gold: u32, quad: u32)
    -> (Vec<Vec<f64>>, Vec<Vec<u8>>, Vec<Vec<i64>>)
{
    let jin = if gold != 0 { g.cfg.jin_mult as f64 } else { 1.0 };
    let mut alive = vec![true; g.ji_events.len()];
    for ev in &g.claimed_ji {
        let (discarder, tag) = (ev.0, ev.2);
        for i in (0..g.ji_events.len()).rev() {
            if !alive[i] || g.ji_events[i].0 != discarder || g.ji_events[i].2 != tag {
                continue;
            }
            alive[i] = false;
            break;
        }
    }
    let mut vals: Vec<Vec<f64>> = vec![Vec::new(); 4];
    let mut tags: Vec<Vec<u8>> = vec![Vec::new(); 4];
    let mut tiles: Vec<Vec<i64>> = vec![Vec::new(); 4];
    for (i, ev) in g.ji_events.iter().enumerate() {
        if !alive[i] {
            continue;
        }
        let t = if i < g.ji_event_tiles.len() { g.ji_event_tiles[i] as i64 } else { -1 };
        let tb = if t >= 0 { bit(t as u8) } else { 0 };
        let mult = if tb & quad != 0 {
            jin * 2.0
        } else if tb & gold != 0 {
            jin
        } else {
            1.0
        };
        vals[ev.0 as usize].push(ev.1 as f64 * mult);
        tags[ev.0 as usize].push(ev.2);
        tiles[ev.0 as usize].push(t);
    }
    (vals, tags, tiles)
}

/// 鸡数统计（对齐 _chicken_counts）。返回
/// (c, items, own_vals, own_tags, hold_bd, fanji_bd)。
#[allow(clippy::type_complexity)]
#[allow(clippy::too_many_arguments)]
fn chicken_counts(
    g: &Game,
    tenpai: &[bool; 4],
    fanji: u32,
    gold: u32,          // 2026-11-04：金鸡牌种由 settle 统一算好（含开局已知金鸡）
    jin: f64,
    void: Option<u8>,
    quad: u32,
) -> (
    [f64; 4],
    Vec<Vec<(String, String, f64)>>,
    Vec<Vec<f64>>,
    Vec<Vec<u8>>,
    Vec<Vec<[i64; 2]>>,
    Vec<Vec<[i64; 2]>>,
) {
    let cfg = &g.cfg;
    let mut c = [0.0f64; 4];
    let mut items: Vec<Vec<(String, String, f64)>> = vec![Vec::new(); 4];
    let mut hold_bd: Vec<Vec<[i64; 2]>> = vec![Vec::new(); 4];
    let mut fanji_bd: Vec<Vec<[i64; 2]>> = vec![Vec::new(); 4];

    let mut add = |p: usize, val: f64, label: &str, info: &str,
                   c: &mut [f64; 4], items: &mut Vec<Vec<(String, String, f64)>>| {
        if val <= 0.0 {
            return;
        }
        c[p] += val;
        items[p].push((label.to_string(), info.to_string(), val));
    };

    let quad = quad & gold;
    // 「翻鸡」项只保留非金鸡种
    let fanji_only = fanji & !g.ji_tiles;
    let (own_vals, own_tags, _own_tiles) = own_ji_after_claims(g, gold, quad);

    // 捉炮鸡按牌种扣张数（避免与手中鸡重复）
    let mut cap_by: [Vec<(u8, i64)>; 4] = Default::default();
    for (w, _v, _t, tile) in &g.captured_ji {
        let slot = &mut cap_by[*w as usize];
        if let Some(e) = slot.iter_mut().find(|e| e.0 == *tile) {
            e.1 += 1;
        } else {
            slot.push((*tile, 1));
        }
    }
    let cap_of = |p: usize, t: u8| -> i64 {
        cap_by[p].iter().find(|e| e.0 == t).map(|e| e.1).unwrap_or(0)
    };

    for p in 0..4usize {
        if !tenpai[p] {
            continue;
        }
        // 1) 打出的鸡（冲锋/横/普通，含金鸡倍率）
        for (val, tag) in own_vals[p].iter().zip(&own_tags[p]) {
            add(p, *val, ji_tag_cn(*tag), "打出", &mut c, &mut items);
        }
        // 2) 手中/副露鸡牌：逐牌种统计
        let hold = cfg.ji_hold as f64;
        let mut n_hand = 0i64;
        let mut n_val = 0.0f64;
        let mut n_gold = 0i64;
        let mut n_quad = 0i64;
        for t in 0..NUM_TILES as u8 {
            if g.ji_tiles & bit(t) == 0 {
                continue;
            }
            let n_t = (g.hands[p][t as usize] as i64)
                + (g.melded_count(p, t) as i64)
                - cap_of(p, t);
            let n_t = n_t.max(0);
            if n_t <= 0 {
                continue;
            }
            let mult = if quad & bit(t) != 0 {
                jin * 2.0
            } else if gold & bit(t) != 0 {
                jin
            } else {
                1.0
            };
            n_hand += n_t;
            n_val += n_t as f64 * hold * mult;
            if gold & bit(t) != 0 {
                n_gold += n_t;
            }
            if quad & bit(t) != 0 {
                n_quad += n_t;
            }
            hold_bd[p].push([t as i64, n_t]);
        }
        if n_val > 0.0 {
            let mut info = format!("{} 只", n_hand);
            if n_quad > 0 {
                info += &format!("（含四倍金鸡 {} 张 ×{}）", n_quad, jin * 2.0);
            } else if n_gold > 0 {
                info += &format!("（含金鸡 {} 张 ×{}）", n_gold, jin);
            }
            add(p, n_val, "手中鸡", &info, &mut c, &mut items);
        }
        // 3) 翻鸡牌（非金鸡种；满堂鸡含弃牌张）
        if fanji_only != 0 {
            let mut nf = 0i64;
            let mut extra = String::new();
            for t in 0..NUM_TILES as u8 {
                if fanji_only & bit(t) == 0 {
                    continue;
                }
                nf += g.hands[p][t as usize] as i64 + g.melded_count(p, t) as i64;
            }
            if cfg.mantiangji {
                let mut nd = 0i64;
                for t in 0..NUM_TILES as u8 {
                    if fanji_only & bit(t) == 0 {
                        continue;
                    }
                    nd += g.discards[p].iter().filter(|&&x| x == t).count() as i64;
                }
                if nd > 0 {
                    nf += nd;
                    extra = format!("（含弃牌 {} 张）", nd);
                }
            }
            if nf > 0 {
                add(p, nf as f64 * cfg.ji_fanji as f64, "翻鸡",
                    &format!("共持有 {} 张{}", nf, extra), &mut c, &mut items);
                for t in 0..NUM_TILES as u8 {
                    if fanji_only & bit(t) == 0 {
                        continue;
                    }
                    let mut n_ft = g.hands[p][t as usize] as i64
                        + g.melded_count(p, t) as i64;
                    if cfg.mantiangji {
                        n_ft += g.discards[p].iter().filter(|&&x| x == t).count() as i64;
                    }
                    if n_ft > 0 {
                        fanji_bd[p].push([t as i64, n_ft]);
                    }
                }
            }
        }
    }
    // 4) 捉炮鸡：按打出时类型全额计（金鸡/四倍金鸡乘倍率）
    for (w, val, tag, tile) in &g.captured_ji {
        let tb = bit(*tile);
        let mult = if tb & quad != 0 {
            jin * 2.0
        } else if tb & gold != 0 {
            jin
        } else {
            1.0
        };
        add(*w as usize, *val as f64 * mult, "捉炮鸡",
            &format!("{}·{}", ji_tag_cn(*tag), tile_cn(*tile)), &mut c, &mut items);
    }
    // 2026-11-07：第一巡同轮跟打 —— 被跟打的 4 张牌在结算计鸡时算作庄家手中的鸡。
    // 与「手中鸡」同口径（庄家听牌才计、金鸡牌种按倍率）；放在全烧清零之前，
    // 庄家若恰是全烧者其自身鸡分仍一律作废。触发结果由 Python 权威判定后过桥。
    if g.follow_dealer && g.follow_tile < 27 {
        let d = g.dealer as usize;
        if tenpai[d] {
            let ft = g.follow_tile;
            let tb = bit(ft);
            let mult = if tb & quad != 0 {
                jin * 2.0
            } else if tb & gold != 0 {
                jin
            } else {
                1.0
            };
            let n = cfg.follow_hold as i64;
            let val = n as f64 * cfg.ji_hold as f64 * mult;
            add(d, val, "同轮跟打鸡", &format!("{}×{}", tile_cn(ft), n),
                &mut c, &mut items);
            hold_bd[d].push([ft as i64, n]);
        }
    }
    // 全烧：该玩家自身鸡分作废
    if let Some(v0) = void {
        let v0 = v0 as usize;
        c[v0] = 0.0;
        items[v0].clear();
        hold_bd[v0].clear();
        fanji_bd[v0].clear();
    }
    (c, items, own_vals, own_tags, hold_bd, fanji_bd)
}

// ---------------------------------------------------------------------------
// 有序 pair_det（Python dict 的插入序）
// ---------------------------------------------------------------------------
struct PairDet {
    rows: Vec<([usize; 2], Vec<(String, String, f64, i64)>)>,
}

impl PairDet {
    fn new() -> Self {
        Self { rows: Vec::new() }
    }
    fn push(&mut self, key: [usize; 2], item: (String, String, f64, i64)) {
        if let Some(e) = self.rows.iter_mut().find(|(k, _)| *k == key) {
            e.1.push(item);
        } else {
            self.rows.push((key, vec![item]));
        }
    }
    fn to_json(&self) -> Value {
        Value::Array(self.rows.iter().map(|(k, items)| json!({
            "a": k[0], "b": k[1],
            "items": items.iter().map(|it| json!({
                "label": it.0, "info": it.1, "value": it.2, "who": it.3,
            })).collect::<Vec<_>>(),
        })).collect())
    }
}

// ---------------------------------------------------------------------------
// 胡牌结算
// ---------------------------------------------------------------------------
pub fn settle_win(g: &mut Game, ft: &FanTable) -> Value {
    let out = g.outcome.expect("settle_win called without outcome");
    let cfg = g.cfg;
    let winners: Vec<u8> = out.winners[..out.n_winners.max(1) as usize].to_vec();
    let primary = winners[0];
    let is_tsumo = out.is_tsumo;
    let loser = out.loser;
    let rob = out.rob_kong;
    let wt = if out.win_tile != 255 { Some(out.win_tile) } else { None };

    // ---- 胡牌张归属（对齐 Python：点炮从弃牌区移出；抢杠从被抢者手中移出）----
    if let Some(t) = wt {
        for &w in &winners {
            g.hands[w as usize][t as usize] += 1;
        }
        if !is_tsumo && loser != 255 {
            if rob {
                if g.hands[loser as usize][t as usize] > 0 {
                    g.hands[loser as usize][t as usize] -= 1;
                }
            } else {
                let dl = &mut g.discards[loser as usize];
                if dl.last() == Some(&t) {
                    dl.pop();
                }
                // 捉炮鸡转移已在 settle_win_multi 完成
            }
        }
    }

    let types_by: Vec<Vec<&'static str>> = winners.iter().map(|&w| {
        fan::win_type_set(&g.hands[w as usize], &g.melds[w as usize],
                          if is_tsumo { None } else { wt }, ft)
    }).collect();

    let fan_cn_display = |w_idx: usize| -> String {
        let mut named: Vec<String> = types_by[w_idx].iter()
            .filter(|tp| **tp != fan::PING_HU)
            .map(|tp| fan_cn_of(tp)).collect();
        named.sort();
        if !named.is_empty() {
            named.join("+")
        } else if g.baojiao[winners[w_idx] as usize] {
            "报叫".to_string()
        } else {
            "平胡".to_string()
        }
    };

    // ---- 翻鸡 / 金鸡 / 四倍金鸡 ----
    let fanji: u32 = if cfg.jiesuan_fanji != 0 && g.wall_left() > 0 {
        fanji_mask(g.wall[g.wall_pos], cfg.jiesuan_fanji)
    } else {
        0
    };
    // 2026-11-04 用户口径：开局翻鸡翻到幺鸡（九条）时，幺鸡在**开局**就已是
    // 本局金鸡，结算必须按金鸡计。旧口径只在结算翻鸡再次翻出幺鸡时认金鸡，
    // 用户实测「开局翻到金鸡、结算却没按金鸡算」为 bug。与 Python
    // DushanGame::kaiju_gold_tiles() 同源。
    let kaiju_gold: u32 = if cfg.kaiju_fanji != 0 {
        match g.kaiju_flip {
            Some(kf) if fanji_mask(kf, cfg.kaiju_fanji) & bit(YAOJI) != 0 => bit(YAOJI),
            _ => 0,
        }
    } else {
        0
    };
    let gold = (fanji & g.ji_tiles) | kaiju_gold;
    // 四倍金鸡：开局已知金鸡 + 结算翻鸡再次翻出幺鸡 → 幺鸡在金鸡基础上再 ×2
    let quad = if kaiju_gold != 0 && fanji & bit(YAOJI) != 0 {
        bit(YAOJI)
    } else {
        0
    };
    let jin = if gold != 0 { cfg.jin_mult as f64 } else { 1.0 };

    let mut detail: Vec<(i64, String, String, f64)> = Vec::new();
    let fanji_flip = if fanji != 0 && g.wall_left() > 0 {
        Some(g.wall[g.wall_pos] as i64)
    } else {
        None
    };
    if fanji != 0 {
        let mut info = fanji_flip.map(|t| tile_cn(t as u8)).unwrap_or_default();
        if quad != 0 {
            info += "（四倍金鸡）";
        } else if gold != 0 {
            info += "（金鸡）";
        }
        detail.push((-1, "翻鸡".into(), info, 0.0));
    }

    let mut pair = [[0.0f64; 4]; 4];
    let mut pdet = PairDet::new();
    let mut pflow = |a: usize, b: usize, val: f64, label: &str, info: &str,
                     who: Option<i64>,
                     pair: &mut [[f64; 4]; 4], pdet: &mut PairDet| {
        if val <= 0.0 {
            return;
        }
        pair[a][b] += val;
        pdet.push([a, b], (label.to_string(), info.to_string(), val,
                           who.unwrap_or(a as i64)));
    };

    // 2026-11-07：第一巡同轮跟打 —— 庄家向其余三家各赔 follow_pay 只鸡
    // （首巡已成立的即时约定，与胡牌方式/是否自摸无关）
    if g.follow_dealer && g.cfg.follow_pay > 0.0 {
        let d = g.dealer as usize;
        let pay = g.cfg.follow_pay as f64;
        let info = if g.follow_tile < 27 {
            format!("{}×4", tile_cn(g.follow_tile))
        } else {
            "同轮跟打".to_string()
        };
        for q in 0..4usize {
            if q != d {
                pflow(q, d, pay, "第一巡同轮跟打", &info, Some(d as i64),
                      &mut pair, &mut pdet);
            }
        }
    }

    // ---- 终局听牌（赢家视为听牌）----
    let mut tenpai = [false; 4];
    for p in 0..4 {
        tenpai[p] = tenpai_flag(g, p, ft);
    }
    for &w in &winners {
        tenpai[w as usize] = true;
    }

    // ---- 全烧：热炮放炮者 / 被抢杠者 ----
    let void_player: Option<u8> = if !is_tsumo && loser != 255 && (rob || g.repao_discard) {
        Some(loser)
    } else {
        None
    };

    // ---- 鸡数统计 ----
    let (c, items, _own_vals, _own_tags, hold_bd, fanji_bd) =
        chicken_counts(g, &tenpai, fanji, gold, jin, void_player, quad);
    for p in 0..4usize {
        for (label, info, val) in &items[p] {
            detail.push((p as i64, label.clone(), info.clone(), *val));
        }
    }
    if let Some(vp) = void_player {
        detail.push((vp as i64, "全烧".into(), "鸡分作废".into(), 0.0));
    }

    // ---- 结算页鸡牌三分类明细 ----
    let (own_vals3, tags3, tiles3) = own_ji_after_claims(g, gold, quad);
    let mut discard_detail: Map<String, Value> = Map::new();
    for p in 0..4usize {
        let arr: Vec<Value> = own_vals3[p].iter().zip(&tags3[p]).zip(&tiles3[p])
            .map(|((v, tg), tl)| json!({
                "tile": tl,
                "tag": ji_tag_cn(*tg),
                "val": (v * 1000.0).round() / 1000.0,
            })).collect();
        discard_detail.insert(p.to_string(), Value::Array(arr));
    }
    let mut capture_detail: Map<String, Value> = Map::new();
    for p in 0..4usize {
        capture_detail.insert(p.to_string(), Value::Array(Vec::new()));
    }
    for (w, val, tag, tile) in &g.captured_ji {
        let tb = bit(*tile);
        let mult = if tb & quad != 0 {
            jin * 2.0
        } else if tb & gold != 0 {
            jin
        } else {
            1.0
        };
        capture_detail.entry(w.to_string()).or_insert_with(|| Value::Array(Vec::new()))
            .as_array_mut().unwrap().push(json!({
                "tile": *tile as i64, "tag": ji_tag_cn(*tag),
                "val": ((*val as f64) * mult * 1000.0).round() / 1000.0,
            }));
    }
    if let Some(vp) = void_player {
        discard_detail.insert(vp.to_string(), Value::Array(Vec::new()));
        capture_detail.insert(vp.to_string(), Value::Array(Vec::new()));
    }
    // 责任鸡明细（对齐 _claimed_ji_detail）
    let mut claim_detail: Vec<Value> = Vec::new();
    for ev in &g.claimed_ji {
        let (discarder, claimer, tag, _mg, tile) = *ev;
        let premium_base = match tag {
            JI_CHONGFENG => cfg.ji_chongfeng,
            JI_HENG => cfg.ji_heng,
            _ => cfg.ji_discard_plain,
        } as f64 - cfg.ji_hold as f64;
        let tb = bit(tile);
        let is_gold = tb & gold != 0;
        let premium = if is_gold {
            premium_base * if tb & quad != 0 { jin * 2.0 } else { jin }
        } else {
            premium_base
        };
        if premium > 0.0 {
            claim_detail.push(json!({
                "claimer": claimer as i64, "discarder": discarder as i64,
                "tile": tile as i64, "tag": ji_tag_cn(tag),
                "premium": (premium * 1000.0).round() / 1000.0,
                "gold": is_gold,
            }));
        }
    }
    if let Some(vp) = void_player {
        claim_detail.retain(|d| d["claimer"].as_i64() != Some(vp as i64));
    }

    // ---- 胡牌方式 ----
    let (how, how_val): (&str, f64) = if is_tsumo {
        if g.after_gang_draw {
            ("杠上开花", cfg.gang_kai as f64)
        } else {
            ("自摸", cfg.zi_mo as f64)
        }
    } else if rob {
        ("抢杠", cfg.dian_pao as f64)
    } else if g.repao_discard {
        ("热炮", cfg.gang_pao as f64)
    } else {
        ("点炮", cfg.dian_pao as f64)
    };
    let payers: Vec<u8> = if is_tsumo {
        (0..4u8).filter(|&q| q != primary).collect()
    } else if loser != 255 {
        vec![loser]
    } else {
        Vec::new()
    };
    let how_tag = if winners.len() > 1 {
        format!("{}（{}家）", how, winners.len())
    } else {
        how.to_string()
    };
    for (wi, &w) in winners.iter().enumerate() {
        for &q in &payers {
            if q as usize != w as usize {
                pflow(w as usize, q as usize, how_val, &how_tag, "", None,
                      &mut pair, &mut pdet);
            }
        }
        detail.push((w as i64, how.to_string(), String::new(), how_val));
        let _ = wi;
    }

    // ---- 牌型奖励与报叫 ----
    let mut type_total_by = [0.0f64; 4];
    let mut type_cn_by: [String; 4] = Default::default();
    for (wi, &w) in winners.iter().enumerate() {
        let types = &types_by[wi];
        let type_total: f64 = types.iter().map(|tp| hu_type_value(tp, &cfg)).sum();
        let mut named: Vec<String> = types.iter()
            .filter(|tp| **tp != fan::PING_HU)
            .map(|tp| fan_cn_of(tp)).collect();
        named.sort();
        let type_cn = named.join("+");
        type_total_by[w as usize] = type_total;
        type_cn_by[w as usize] = type_cn.clone();
        if type_total > 0.0 && !payers.is_empty() {
            for &q in &payers {
                if q as usize != w as usize {
                    pflow(w as usize, q as usize, type_total,
                          if type_cn.is_empty() { "平胡" } else { &type_cn },
                          "牌型", None, &mut pair, &mut pdet);
                }
            }
            detail.push((w as i64,
                         if type_cn.is_empty() { "平胡".to_string() } else { type_cn.clone() },
                         "牌型".into(), type_total));
        }
        if g.baojiao[w as usize] && cfg.baojiao_bonus > 0.0 && !payers.is_empty() {
            for &q in &payers {
                if q as usize != w as usize {
                    pflow(w as usize, q as usize, cfg.baojiao_bonus as f64, "报叫", "",
                          None, &mut pair, &mut pdet);
                }
            }
            detail.push((w as i64, "报叫".into(), String::new(),
                         cfg.baojiao_bonus as f64));
        }
    }

    // ---- 杠分 ----
    for p in 0..4usize {
        if !tenpai[p] {
            continue;
        }
        if void_player == Some(p as u8) {
            continue;
        }
        for &(mt, t, src) in &g.melds[p] {
            if !fan::meld_is_kong(mt) {
                continue;
            }
            let rel = ((src as usize) + 4 - p) % 4;
            let who = match rel {
                1 => "下家",
                2 => "对家",
                3 => "上家",
                _ => "",
            };
            let info = match mt {
                MELD_KONG_CONCEALED => format!("暗杠 {}", tile_cn(t)),
                MELD_KONG_ADDED => format!("补杠 {}", tile_cn(t)),
                _ => {
                    if who.is_empty() {
                        format!("明杠 {}", tile_cn(t))
                    } else {
                        format!("明杠 {}（{}点杠）", tile_cn(t), who)
                    }
                }
            };
            let payers_k: Vec<u8> = if mt == MELD_KONG_EXPOSED {
                vec![src]
            } else {
                (0..4u8).filter(|&q| q as usize != p).collect()
            };
            for q in payers_k {
                pflow(p, q as usize, cfg.kong_chickens as f64, "杠分", &info, None,
                      &mut pair, &mut pdet);
            }
            detail.push((p as i64, "杠分".into(), info, cfg.kong_chickens as f64));
        }
    }

    // ---- 责任鸡溢价 ----
    let unt: Vec<usize> = (0..4).filter(|&q| !tenpai[q]).collect();
    for d in &claim_detail.clone() {
        let claimer = d["claimer"].as_u64().unwrap() as usize;
        let discarder = d["discarder"].as_u64().unwrap() as usize;
        let premium = d["premium"].as_f64().unwrap();
        let tag = d["tag"].as_str().unwrap().to_string();
        if Some(claimer as u8) == void_player {
            continue;
        }
        if unt.contains(&claimer) {
            continue;
        }
        pflow(claimer, discarder, premium, "责任鸡", &tag, None, &mut pair, &mut pdet);
        detail.push((claimer as i64, "责任鸡".into(), tag, premium));
    }

    // ---- 包鸡 / 包杠 ----
    let prem_pairs_src: Vec<(usize, usize, f64, String)> = claim_detail.iter()
        .map(|d| (d["claimer"].as_u64().unwrap() as usize,
                  d["discarder"].as_u64().unwrap() as usize,
                  d["premium"].as_f64().unwrap(),
                  d["tag"].as_str().unwrap().to_string()))
        .collect();
    let mut prem_pairs: Vec<([usize; 2], Vec<(String, f64)>)> = Vec::new();
    for (cl, dsc, prem, tg) in &prem_pairs_src {
        if let Some(e) = prem_pairs.iter_mut().find(|(k, _)| k == &[*cl, *dsc]) {
            e.1.push((tg.clone(), *prem));
        } else {
            prem_pairs.push(([*cl, *dsc], vec![(tg.clone(), *prem)]));
        }
    }
    for p in 0..4usize {
        if tenpai[p] {
            continue;
        }
        if void_player == Some(p as u8) {
            continue;
        }
        if cfg.end_baoji {
            // 已亮明的鸡：自己打出的（own_vals 已按金鸡倍率折算）+ 副露里的
            let mut ming: Vec<(String, f64)> = own_vals3[p].iter().zip(&tags3[p])
                .map(|(v, tg)| (ji_tag_cn(*tg).to_string(), *v)).collect();
            let mut n_m = 0i64;
            let mut v_m = 0.0f64;
            for t in 0..NUM_TILES as u8 {
                if g.ji_tiles & bit(t) == 0 {
                    continue;
                }
                let k = g.melded_count(p, t) as i64;
                if k <= 0 {
                    continue;
                }
                n_m += k;
                let mult = if quad & bit(t) != 0 {
                    jin * 2.0
                } else if gold & bit(t) != 0 {
                    jin
                } else {
                    1.0
                };
                v_m += k as f64 * cfg.ji_hold as f64 * mult;
            }
            if v_m > 0.0 {
                ming.push((format!("副露鸡 {} 张", n_m), v_m));
            }
            for (tg, val) in &ming {
                if *val <= 0.0 {
                    continue;
                }
                for q in 0..4usize {
                    if q != p {
                        pflow(q, p, *val, "包鸡", tg, Some(p as i64),
                              &mut pair, &mut pdet);
                    }
                }
                detail.push((p as i64, "包鸡".into(), tg.clone(), *val));
            }
            // 被碰/杠走那张的溢价：原打出者付给碰/杠者（一次性独立成行）
            for q in 0..4usize {
                if q == p {
                    continue;
                }
                if let Some(e) = prem_pairs.iter().find(|(k, _)| k == &[p, q]) {
                    let psum: f64 = e.1.iter().map(|(_, v)| v).sum();
                    let etag = if e.1.len() == 1 {
                        e.1[0].0.clone()
                    } else {
                        e.1.iter().map(|(t, _)| t.clone()).collect::<Vec<_>>().join("、")
                    };
                    pflow(q, p, psum, "包鸡", &format!("{}溢价", etag), Some(p as i64),
                          &mut pair, &mut pdet);
                    detail.push((p as i64, "包鸡".into(),
                                 format!("{}溢价", etag), psum));
                }
            }
        }
        if cfg.end_baogang {
            let kongs = g.melds[p].iter().filter(|m| fan::meld_is_kong(m.0)).count();
            for _ in 0..kongs {
                for q in 0..4usize {
                    if q != p {
                        pflow(q, p, BAO_KONG, "包杠", "", Some(p as i64),
                              &mut pair, &mut pdet);
                    }
                }
                detail.push((p as i64, "包杠".into(), String::new(), BAO_KONG));
            }
        }
    }

    // ---- 两两互减：c[a] − c[b] ----
    for a in 0..4usize {
        for b in 0..4usize {
            if a == b {
                continue;
            }
            let diff = c[a] - c[b];
            if diff > 0.0 {
                pair[a][b] += diff;
                let mut rows: Vec<(String, String, f64, i64)> = Vec::new();
                for (label, info, val) in &items[a] {
                    rows.push((label.clone(), info.clone(), *val, a as i64));
                }
                for (label, info, val) in &items[b] {
                    if *val != 0.0 {
                        rows.push((label.clone(), info.clone(), -*val, b as i64));
                    }
                }
                for r in rows {
                    pdet.push([a, b], r);
                }
            }
        }
    }

    // ---- 汇总 ----
    let mut delta = [0.0f64; 4];
    for a in 0..4usize {
        for b in 0..4usize {
            delta[a] += pair[a][b] - pair[b][a];
        }
    }

    let fan_cn = fan_cn_display(0);
    let fan_total: f64 = pair[primary as usize].iter().sum();
    let wt_display = wt.map(|t| t as i64)
        .or(if is_tsumo && g.last_draw != 255 { Some(g.last_draw as i64) } else { None });

    let fanji_tiles_sorted: Vec<i64> = (0..NUM_TILES as u8)
        .filter(|&t| fanji & bit(t) != 0).map(|t| t as i64).collect();
    let gold_sorted: Vec<i64> = (0..NUM_TILES as u8)
        .filter(|&t| gold & bit(t) != 0).map(|t| t as i64).collect();
    let quad_sorted: Vec<i64> = (0..NUM_TILES as u8)
        .filter(|&t| quad & bit(t) != 0).map(|t| t as i64).collect();

    let p_types = &types_by[0];
    let fan_type_sorted: String = {
        let mut v: Vec<&str> = p_types.iter().map(|x| *x).collect();
        v.sort_unstable();
        v.join("+")
    };
    let winners_fan: Vec<Value> = winners.iter().enumerate().map(|(wi, &w)| {
        let mut v: Vec<&str> = types_by[wi].iter().map(|x| *x).collect();
        v.sort_unstable();
        let ft_str = if v.is_empty() { fan::PING_HU.to_string() } else { v.join("+") };
        json!({
            "seat": w as i64,
            "fan_cn": fan_cn_display(wi),
            "fan_type": ft_str,
            "fan": pair[w as usize].iter().sum::<f64>(),
        })
    }).collect();

    let chicken_labels = ["冲锋鸡", "横鸡", "幺鸡", "手中鸡", "捉炮鸡", "杠牌", "翻鸡",
                          "责任鸡", "包鸡", "包杠", "包大牌面", "自摸", "热炮", "抢杠"];
    let chicken_rows: Vec<Value> = detail.iter()
        .filter(|(_, l, _, _)| chicken_labels.contains(&l.as_str()))
        .map(|(p, l, i, v)| json!([p, l, i, v])).collect();
    let detail_json: Vec<Value> = detail.iter()
        .map(|(p, l, i, v)| json!([p, l, i, v])).collect();

    let mut hold_map = Map::new();
    for p in 0..4usize {
        hold_map.insert(p.to_string(), Value::Array(
            hold_bd[p].iter().map(|e| json!([e[0], e[1]])).collect()));
    }
    let mut fanji_map = Map::new();
    for p in 0..4usize {
        fanji_map.insert(p.to_string(), Value::Array(
            fanji_bd[p].iter().map(|e| json!([e[0], e[1]])).collect()));
    }

    // 胡牌者手牌复原（前端展示用）：把胡牌张从手牌移除回原位
    if let Some(t) = wt {
        for &w in &winners {
            if g.hands[w as usize][t as usize] > 0 {
                g.hands[w as usize][t as usize] -= 1;
            }
        }
        if !is_tsumo && loser != 255 {
            if rob {
                g.hands[loser as usize][t as usize] += 1;
            } else {
                g.discards[loser as usize].push(t);
            }
        }
    }

    json!({
        "type": "win",
        "winner": primary as i64,
        "winners": winners.iter().map(|&w| w as i64).collect::<Vec<_>>(),
        "loser": if loser != 255 { Some(loser as i64) } else { None },
        "is_tsumo": is_tsumo,
        "rob_kong": rob,
        "void_player": void_player.map(|v| v as i64),
        "win_tile": wt_display,
        "fan_type": if fan_type_sorted.is_empty() { fan::PING_HU.to_string() } else { fan_type_sorted },
        "fan_cn": fan_cn,
        "fan": fan_total,
        "total_fan": fan_total,
        "chickens": c.to_vec(),
        "tenpai": (0..4).filter(|&p| tenpai[p]).collect::<Vec<_>>(),
        "fanji_flip": fanji_flip,
        "fanji_tiles": fanji_tiles_sorted,
        "fanji_gold": gold != 0,
        "gold_tiles": gold_sorted,
        "quad_gold_tiles": quad_sorted,
        "ji_hold_detail": hold_map,
        "fanji_detail": fanji_map,
        "ji_discard_detail": discard_detail,
        "ji_capture_detail": capture_detail,
        "ji_claim_detail": claim_detail,
        "jiesuan_fanji": fan_mode_str(cfg.jiesuan_fanji),
        "kaiju_fanji": fan_mode_str(cfg.kaiju_fanji),
        "how": how,
        "how_value": how_val,
        "type_value": type_total_by[primary as usize],
        "winners_fan": winners_fan,
        "dealer_bonus": 0.0,
        "dou": [],
        "chicken": chicken_rows,
        "detail": detail_json,
        "pair": pair.iter().map(|r| r.to_vec()).collect::<Vec<_>>(),
        "pair_detail": pdet.to_json(),
        "wall_left": g.wall_left(),
        "follow_dealer": g.follow_dealer,
        "follow_tile": if g.follow_tile < 27 { Some(g.follow_tile as i64) } else { None },
        "follow_pay": g.cfg.follow_pay as f64,
        "deltas": delta.to_vec(),
    })
}

// ---------------------------------------------------------------------------
// 黄牌（流局）结算
// ---------------------------------------------------------------------------
pub fn settle_huangzhuang(g: &Game, _ft: &FanTable) -> Value {
    let cfg = g.cfg;
    let mut detail: Vec<(i64, String, String, f64)> = Vec::new();
    let mut pair = [[0.0f64; 4]; 4];
    let mut pdet = PairDet::new();

    // 2026-11-07：第一巡同轮跟打 —— 首巡即时约定，流局也照赔
    if g.follow_dealer && g.cfg.follow_pay > 0.0 {
        let d = g.dealer as usize;
        let pay = g.cfg.follow_pay as f64;
        let info = if g.follow_tile < 27 {
            format!("{}×4", tile_cn(g.follow_tile))
        } else {
            "同轮跟打".to_string()
        };
        for q in 0..4usize {
            if q != d {
                pair[q][d] += pay;
                pdet.push([q, d], ("第一巡同轮跟打".into(), info.clone(), pay, d as i64));
            }
        }
    }

    let tenpai = {
        let mut t = [false; 4];
        for p in 0..4 {
            t[p] = tenpai_flag(g, p, _ft);
        }
        t
    };
    let mut baodapai: Vec<Value> = Vec::new();
    for p in 0..4usize {
        if !tenpai[p] {
            continue;
        }
        let (tval, tcn) = max_hu_type(g, p, &cfg, _ft);
        let pay = cfg.zi_mo as f64 + tval;
        baodapai.push(json!({
            "seat": p, "type": if tcn.is_empty() { "平胡" } else { &tcn },
            "type_value": tval, "pay": pay,
        }));
        if !cfg.huang_baodapai {
            continue;
        }
        for q in 0..4usize {
            if !tenpai[q] {
                pair[p][q] += pay;
                pdet.push([p, q], ("包大牌面".into(),
                                   if tcn.is_empty() { "平胡".into() } else { tcn.clone() },
                                   pay, q as i64));
            }
        }
        if (0..4).any(|q| !tenpai[q]) {
            detail.push((p as i64, "包大牌面".into(),
                         format!("{}（自摸鸡）", if tcn.is_empty() { "平胡" } else { &tcn }), pay));
        }
    }

    // 可选：包鸡 / 包杠（默认关；流局不翻鸡 → 无金鸡，倍率恒 1）
    if cfg.huang_baoji || cfg.huang_baogang {
        let (own_vals, own_tags, _tiles) = own_ji_after_claims(g, 0, 0);
        let jin_h = 1.0f64;
        for p in 0..4usize {
            if tenpai[p] {
                continue;
            }
            if cfg.huang_baoji {
                let mut ming: Vec<(String, f64)> = own_tags[p].iter()
                    .zip(&own_vals[p])
                    .map(|(tg, v)| (ji_tag_cn(*tg).to_string(), *v)).collect();
                let n_yao_m = g.melded_count(p, YAOJI) as i64;
                let mut n_oth_m = 0i64;
                for t in 0..NUM_TILES as u8 {
                    if g.ji_tiles & bit(t) != 0 && t != YAOJI {
                        n_oth_m += g.melded_count(p, t) as i64;
                    }
                }
                let v_m = (n_yao_m as f64 * jin_h + n_oth_m as f64) * cfg.ji_hold as f64;
                if v_m > 0.0 {
                    ming.push((format!("副露鸡 {} 张", n_yao_m + n_oth_m), v_m));
                }
                for (tg, v) in &ming {
                    if *v <= 0.0 {
                        continue;
                    }
                    for q in 0..4usize {
                        if q != p {
                            pair[q][p] += *v;
                            pdet.push([q, p], ("包鸡".into(), tg.clone(), *v, p as i64));
                        }
                    }
                    detail.push((p as i64, "包鸡".into(), tg.clone(), *v));
                }
            }
            if cfg.huang_baogang {
                let kongs = g.melds[p].iter().filter(|m| fan::meld_is_kong(m.0)).count();
                for _ in 0..kongs {
                    for q in 0..4usize {
                        if q != p {
                            pair[q][p] += BAO_KONG;
                            pdet.push([q, p], ("包杠".into(), String::new(), BAO_KONG, p as i64));
                        }
                    }
                    detail.push((p as i64, "包杠".into(), String::new(), BAO_KONG));
                }
            }
        }
    }

    let mut delta = [0.0f64; 4];
    for a in 0..4usize {
        for b in 0..4usize {
            delta[a] += pair[a][b] - pair[b][a];
        }
    }

    let detail_json: Vec<Value> = detail.iter()
        .map(|(p, l, i, v)| json!([p, l, i, v])).collect();
    json!({
        "type": "huangzhuang",
        "tenpai": (0..4).filter(|&p| tenpai[p]).collect::<Vec<_>>(),
        "chickens": vec![0.0f64; 4],
        "baodapai": baodapai,
        "baodapai_on": cfg.huang_baodapai,
        "cha_jiao": detail_json.clone(),
        "detail": detail_json,
        "pair": pair.iter().map(|r| r.to_vec()).collect::<Vec<_>>(),
        "pair_detail": pdet.to_json(),
        "wall_left": 0,
        "follow_dealer": g.follow_dealer,
        "follow_tile": if g.follow_tile < 27 { Some(g.follow_tile as i64) } else { None },
        "follow_pay": g.cfg.follow_pay as f64,
        "deltas": delta.to_vec(),
    })
}
