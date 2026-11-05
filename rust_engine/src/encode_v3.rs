//! 状态编码 v3 —— 与 `zhuoji/encoder_v3.py` **逐位一致**（63 通道 / 130 全局维）。
//!
//! v3 = v2 严格超集：前 31 通道 / 前 98 维逐位等于 `encode_v2`，
//! 追加「长时序牌河 d / 横鸡轮 e / 放铳风险与手牌结构 f / 全局扩展 g」。
//! 红线同 v2：只用公开信息，与 Python 版逐位对拍。
//!
//! 依赖的横鸡事件流（hengji_opens / hengji_starter / hengji_event_n）
//! 已于 2026-11-05 补入 `state::Game`，与 DushanGame.hengji_events 同源。

use crate::encode_v2::{self, ObsDyn, NUM_CHANNELS_V2, GLOBAL_DIM_V2};
use crate::fan::{self, FanTable};
use crate::state::{bit, Game};
use crate::tiles::{rank_of, suit_of, tile_from, DECK_SIZE, NUM_TILES, YAOJI};

pub const NUM_CHANNELS_V3: usize = 63;
pub const GLOBAL_DIM_V3: usize = 130;

const V2_CHANNELS: usize = NUM_CHANNELS_V2;
const V2_GLOBAL_DIM: usize = GLOBAL_DIM_V2;

#[inline]
fn rk0(t: u8) -> usize {
    (t % 9) as usize
}

#[inline]
fn tile0(suit: usize, r: usize) -> u8 {
    (suit * 9 + r) as u8
}

#[inline]
fn rel(p: usize, me: usize) -> usize {
    (p + 4 - me) % 4
}

#[inline]
fn seat_of(r: usize, me: usize) -> usize {
    (me + r) % 4
}

fn turn_of(g: &Game) -> f32 {
    let wall_left = g.wall_left() as f32;
    ((DECK_SIZE as f32 - wall_left - 52.0) / 4.0).clamp(0.0, 12.0)
}

fn visible(g: &Game, me: usize) -> [f32; NUM_TILES] {
    let mut vis = [0f32; NUM_TILES];
    for t in 0..NUM_TILES {
        vis[t] = g.hands[me][t] as f32;
    }
    for p in 0..4usize {
        for &t in &g.discards[p] {
            vis[t as usize] += 1.0;
        }
        for m in &g.melds[p] {
            vis[m.1 as usize] += if fan::meld_is_kong(m.0) { 4.0 } else { 3.0 };
        }
    }
    vis
}

#[inline]
fn conf(g: &Game, p: usize, turn: f32) -> f32 {
    (0.2 * g.melds[p].len() as f32 + 0.05 * turn).clamp(0.0, 1.0)
}

/// 自己手牌的 (刻子数, 对子数, 搭子数, 孤张数)。搭子 = 相邻/隔张对（不重复计数）。
fn hand_shape(hand: &[u8; NUM_TILES]) -> (f32, f32, f32, f32) {
    let mut triplets = 0.0f32;
    let mut pairs = 0.0f32;
    let mut partials = 0.0f32;
    let mut lone = 0.0f32;
    for s in 0..3usize {
        for rk in 0..9usize {
            let t = s * 9 + rk;
            if hand[t] == 0 {
                continue;
            }
            if hand[t] >= 3 {
                triplets += 1.0;
            }
            if hand[t] == 2 {
                pairs += 1.0;
            }
            let mut nb1 = [0usize; 2];
            let mut n1 = 0usize;
            for d in [-1i32, 1] {
                let rr = rk as i32 + d;
                if (0..9).contains(&rr) {
                    nb1[n1] = s * 9 + rr as usize;
                    n1 += 1;
                }
            }
            let mut nb2 = [0usize; 2];
            let mut n2 = 0usize;
            for d in [-2i32, 2] {
                let rr = rk as i32 + d;
                if (0..9).contains(&rr) {
                    nb2[n2] = s * 9 + rr as usize;
                    n2 += 1;
                }
            }
            let near = nb1[..n1].iter().any(|&u| hand[u] > 0);
            let far = nb2[..n2].iter().any(|&u| hand[u] > 0);
            if near || far {
                if far && !near {
                    partials += 0.5;
                } else if near && !nb1[..n1].iter().any(|&u| u < t && hand[u] > 0) {
                    partials += 1.0;
                }
            } else if hand[t] == 1 {
                lone += 1.0;
            }
        }
    }
    (triplets, pairs, partials, lone)
}

/// v3 编码：v2 基底 + ch31–62 + glob98–129。
pub fn encode_v3(g: &Game, player: u8, ft: &FanTable) -> ObsDyn {
    let me = player as usize;
    let v2 = encode_v2::encode_v2(g, player, ft);
    let mut tiles = vec![0.0f32; NUM_CHANNELS_V3 * NUM_TILES];
    tiles[..V2_CHANNELS * NUM_TILES].copy_from_slice(&v2.tiles);
    let mut glob = vec![0.0f32; GLOBAL_DIM_V3];
    glob[..V2_GLOBAL_DIM].copy_from_slice(&v2.glob);
    fill_tiles_v3(g, me, &mut tiles);
    fill_glob_v3(g, me, &mut glob, ft);
    ObsDyn {
        tiles,
        glob,
        head: v2.head,
        mask: v2.mask,
        mask_len: v2.mask_len,
    }
}

fn fill_tiles_v3(g: &Game, me: usize, tiles: &mut [f32]) {
    // -- 31–42：四家牌河第 4/5/6 近张 ------------------------------------
    for p in 0..4usize {
        let r = rel(p, me);
        let river = &g.discards[p];
        for k in [3usize, 4, 5] {
            if river.len() >= k + 1 {
                tiles[(31 + (k - 3) * 4 + r) * NUM_TILES + river[river.len() - 1 - k] as usize] =
                    1.0;
            }
        }
    }

    // -- 43–46：四家本局开出横鸡轮的次数 / 2（标量广播整行，与 Python 同口径）
    for p in 0..4usize {
        let v = (g.hengji_opens[p] as f32 / 2.0).min(1.0);
        let row = (43 + rel(p, me)) * NUM_TILES;
        for t in 0..NUM_TILES {
            tiles[row + t] = v;
        }
    }

    // -- 47：当前是否还有横鸡轮开放（广播到整张牌面） -----------------------
    if g.hengji_open != 0 {
        for t in 0..NUM_TILES {
            tiles[47 * NUM_TILES + t] = 1.0;
        }
    }

    // -- 48/49：我现在打这张牌会记成什么鸡（引擎同源函数，不改状态） --------
    for t in 0..NUM_TILES as u8 {
        match g.ji_name_if_discarded(me, t) {
            2 => tiles[48 * NUM_TILES + t as usize] = 1.0, // 横鸡
            1 => tiles[49 * NUM_TILES + t as usize] = 1.0, // 冲锋鸡
            _ => {}
        }
    }

    // -- 50：本局鸡牌种 -----------------------------------------------------
    for t in 0..NUM_TILES as u8 {
        if g.ji_tiles & bit(t) != 0 {
            tiles[50 * NUM_TILES + t as usize] = 1.0;
        }
    }

    // -- 51：我自己手牌里的鸡牌张数 / 4 ------------------------------------
    for t in 0..NUM_TILES as u8 {
        if g.ji_tiles & bit(t) != 0 {
            tiles[51 * NUM_TILES + t as usize] = (g.hands[me][t as usize] as f32 / 4.0).min(1.0);
        }
    }

    // -- 52：已被别家碰/杠走的鸡牌张数 / 4（去重 (碰/杠者, 牌) 键）----------
    let mut seen = [false; 4 * NUM_TILES];
    for ev in &g.claimed_ji {
        let (clmer, tile) = (ev.1 as usize % 4, ev.4);
        if tile >= NUM_TILES as u8 || g.ji_tiles & bit(tile) == 0 {
            continue;
        }
        let key = clmer * NUM_TILES + tile as usize;
        if seen[key] {
            continue;
        }
        seen[key] = true;
        let idx = 52 * NUM_TILES + tile as usize;
        tiles[idx] = (tiles[idx] + 0.25).min(1.0);
    }

    let vis = visible(g, me);
    let mut remain = [0f32; NUM_TILES];
    for t in 0..NUM_TILES {
        remain[t] = (4.0 - vis[t]).clamp(0.0, 4.0);
    }
    let turn = turn_of(g);

    // -- 53：放铳风险 = max_r(听牌置信_r) × (1 − 可见张数/4) ----------------
    let mut cmax = 0.0f32;
    for r in 1..4usize {
        cmax = cmax.max(conf(g, seat_of(r, me), turn));
    }
    for t in 0..NUM_TILES {
        tiles[53 * NUM_TILES + t] = cmax * (1.0 - vis[t] / 4.0);
    }

    // -- 54–57：自己手牌结构 -----------------------------------------------
    for t in 0..NUM_TILES {
        let c = g.hands[me][t];
        if c == 0 {
            continue;
        }
        let s = suit_of(t as u8) as usize;
        let rk = rk0(t as u8);
        let mut nb1 = [0usize; 2];
        let mut n1 = 0usize;
        for d in [-1i32, 1] {
            let rr = rk as i32 + d;
            if (0..9).contains(&rr) {
                nb1[n1] = s * 9 + rr as usize;
                n1 += 1;
            }
        }
        let mut nb2 = [0usize; 2];
        let mut n2 = 0usize;
        for d in [-2i32, 2] {
            let rr = rk as i32 + d;
            if (0..9).contains(&rr) {
                nb2[n2] = s * 9 + rr as usize;
                n2 += 1;
            }
        }
        if nb2[..n2].iter().any(|&u| g.hands[me][u] > 0) {
            tiles[54 * NUM_TILES + t] = 1.0;
        }
        if nb1[..n1].iter().any(|&u| g.hands[me][u] > 0) {
            tiles[55 * NUM_TILES + t] = 1.0;
        }
        let mut run = 1.0f32;
        let mut rr = rk as i32 - 1;
        while rr >= 0 && g.hands[me][s * 9 + rr as usize] > 0 {
            run += 1.0;
            rr -= 1;
        }
        rr = rk as i32 + 1;
        while rr < 9 && g.hands[me][s * 9 + rr as usize] > 0 {
            run += 1.0;
            rr += 1;
        }
        tiles[56 * NUM_TILES + t] = (run / 3.0).min(1.0);
        if c == 1
            && !nb1[..n1].iter().any(|&u| g.hands[me][u] > 0)
            && !nb2[..n2].iter().any(|&u| g.hands[me][u] > 0)
        {
            tiles[57 * NUM_TILES + t] = 1.0;
        }
    }

    // -- 58/59：时效性 —— 最近 3 手内被谁打过（线性衰减 1, 2/3, 1/3）--------
    for r in 0..4usize {
        let river = &g.discards[seat_of(r, me)];
        for k in 0..3usize {
            if river.len() >= k + 1 {
                let w = (1.0 - k as f64 / 3.0) as f32; // f64 中间量，与 Python 浮点口径逐位一致
                let t = river[river.len() - 1 - k] as usize;
                if w > tiles[58 * NUM_TILES + t] {
                    tiles[58 * NUM_TILES + t] = w;
                }
                if r == 3 && w > tiles[59 * NUM_TILES + t] {
                    tiles[59 * NUM_TILES + t] = w;
                }
            }
        }
    }

    // -- 60：自己是否已报叫（广播整行）------------------------------------
    if g.baojiao[me] {
        for t in 0..NUM_TILES {
            tiles[60 * NUM_TILES + t] = 1.0;
        }
    }

    // -- 61：严格现物 —— 对其他三家都是现物 --------------------------------
    for t in 0..NUM_TILES {
        let mut strict = 1.0f32;
        for r in 1..4usize {
            let river = &g.discards[seat_of(r, me)];
            if !river.contains(&(t as u8)) {
                strict = 0.0;
            }
        }
        tiles[61 * NUM_TILES + t] = strict;
    }

    // -- 62：「放铳风险 × 还摸得到」 ----------------------------------------
    for t in 0..NUM_TILES {
        tiles[62 * NUM_TILES + t] = tiles[53 * NUM_TILES + t] * (remain[t] / 4.0);
    }
}

fn fill_glob_v3(g: &Game, me: usize, glob: &mut [f32], ft: &FanTable) {
    let hand = &g.hands[me];
    let turn = turn_of(g);
    let wall_left = g.wall_left() as f32;
    let n_melds_me = g.melds[me].len();

    // -- 98–101：四家「因鸡被碰/杠」次数 / 4 -------------------------------
    let mut claim = [0f32; 4];
    for ev in &g.claimed_ji {
        claim[ev.1 as usize % 4] += 1.0;
    }
    for p in 0..4usize {
        glob[98 + rel(p, me)] = (claim[p] / 4.0).min(1.0);
    }

    // -- 102–105：四家「打出鸡牌」次数 / 4 ---------------------------------
    let mut ev_cnt = [0f32; 4];
    for ev in &g.ji_events {
        ev_cnt[ev.0 as usize % 4] += 1.0;
    }
    for p in 0..4usize {
        glob[102 + rel(p, me)] = (ev_cnt[p] / 4.0).min(1.0);
    }

    // -- 106–108：横鸡轮总体状态 -------------------------------------------
    let species = g.hengji_open.count_ones() as f32;
    glob[106] = (species / 2.0).min(1.0);
    glob[107] = if g.hengji_starter >= 0 { 1.0 } else { 0.0 };
    glob[108] = if g.hengji_starter >= 0 {
        rel(g.hengji_starter as usize, me) as f32 / 3.0
    } else {
        0.0
    };

    // -- 109–112：四家是否已报叫 -------------------------------------------
    for p in 0..4usize {
        glob[109 + rel(p, me)] = if g.baojiao[p] { 1.0 } else { 0.0 };
    }

    // -- 113–117：自己手牌结构 ---------------------------------------------
    let (triplets, pairs, partials, lone) = hand_shape(hand);
    glob[113] = ((3.0 * triplets + 2.0 * pairs + partials) / 8.0).clamp(0.0, 1.0);
    glob[114] = (partials / 6.0).min(1.0);
    glob[115] = (pairs / 6.0).min(1.0);
    glob[116] = (triplets / 4.0).min(1.0);
    glob[117] = (lone / 6.0).min(1.0);

    // -- 118–121：四家副露速度 = 副露组数 / max(1, 巡目) -------------------
    for p in 0..4usize {
        glob[118 + rel(p, me)] = (g.melds[p].len() as f32 / turn.max(1.0)).min(1.0);
    }

    // -- 122–123：自己的听牌质量 -------------------------------------------
    let tenpai = glob[27] > 0.5;
    let mut waits: Vec<u8> = Vec::new();
    if crate::tiles::counts_sum(hand) as usize == (4 - n_melds_me) * 3 + 1 {
        waits = g.winning_tiles(me, ft);
    }
    let vis = visible(g, me);
    if !waits.is_empty() {
        let mut mass = 0.0f32;
        for &t in &waits {
            mass += (4.0 - vis[t as usize]).clamp(0.0, 4.0);
        }
        glob[122] = (mass / 13.0).min(1.0);
        glob[123] = ((mass / waits.len().max(1) as f32) / 4.0).min(1.0);
    }

    // -- 124：全场已亮明张数（弃牌 + 副露折算）/ 64 ------------------------
    let mut shown = 0.0f32;
    for p in 0..4usize {
        shown += g.discards[p].len() as f32;
        for m in &g.melds[p] {
            shown += if fan::meld_is_kong(m.0) { 4.0 } else { 3.0 };
        }
    }
    glob[124] = (shown / 64.0).min(1.0);

    // -- 125：场上最高听牌置信 ---------------------------------------------
    let mut cmax = 0.0f32;
    for r in 1..4usize {
        cmax = cmax.max(conf(g, seat_of(r, me), turn));
    }
    glob[125] = cmax;

    // -- 126：自己可达最大番值 × 是否已听 ---------------------------------
    glob[126] = (encode_v2::max_reachable_fan_pub(g, me, ft) / 30.0).min(1.0)
        * if tenpai { 1.0 } else { 0.0 };

    // -- 127：距黄庄的剩余空间 × 未听牌 ------------------------------------
    glob[127] = (wall_left / DECK_SIZE as f32) * if tenpai { 0.0 } else { 1.0 };

    // -- 128：本局鸡相关事件总数 / 8 ---------------------------------------
    let n_ji = g.ji_events.len() + g.claimed_ji.len() + g.hengji_event_n as usize;
    glob[128] = (n_ji as f32 / 8.0).min(1.0);

    // -- 129：自己手牌 + 副露的总张数 / 14 ---------------------------------
    glob[129] = ((crate::tiles::counts_sum(hand) as f32 + 3.0 * n_melds_me as f32) / 14.0).min(1.0);

    // 引用防误删
    let _ = YAOJI;
    let _ = rank_of;
    let _ = tile_from;
    let _ = tile0;
}
