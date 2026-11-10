//! 状态编码 v2 —— 与 `zhuoji/encoder_v2.py` **逐位一致**（31 通道 / 98 全局维）。
//!
//! v2 是 v1 的严格超集：前 15 通道与前 70 维逐位等于 `encode_v1` 的输出，
//! 只在后面追加「牌河时序 / 危险度 / 番型情境」。所有新特征只来自公开信息。
//!
//! 移植红线（2026-11-05）：与 Python 版逐位对拍（tools/encode_parity.py），
//! 任何特征语义分歧都必须以 Python 版为准修 Rust，不允许"顺手改进"。
//!
//! 位次口径：Python `rank_of` 为 0 基（0..8），Rust `tiles::rank_of` 为 1 基（1..9）。
//! 本文件一律用本地 0 基助手 `rk0/tile0`，避免混用两种口径。

use crate::actions::*;
use crate::encode::{self, HEAD_DISCARD};
use crate::fan::{self, FanTable};
use crate::state::Game;
use crate::tiles::{suit_of, Counts, DECK_SIZE, NUM_TILES, YAOJI};

pub const NUM_CHANNELS_V2: usize = 31;
pub const GLOBAL_DIM_V2: usize = 98;

const V1_CHANNELS: usize = 15;
const V1_GLOBAL_DIM: usize = 70;
/// 「开局每家手上已发的牌」：13×4 + 庄家多摸 1 = 53，取 52 作保守下界（与 Python 同）。
const OPENING_TILES: f32 = 52.0;

/// 动态维度观测（v2/v3 共用；维度由构造处决定）。
pub struct ObsDyn {
    pub tiles: Vec<f32>,
    pub glob: Vec<f32>,
    pub head: u8,
    pub mask: [u8; 28],
    pub mask_len: usize,
}

/// 2026-11-08（方案 D）：把固定尺寸的 v1 `Obs` 无损升维成 `ObsDyn`，
/// 使 `search.rs::take_batch` 可以统一返回 `Vec<ObsDyn>`（v1/v2/v3 同一条路径）。
/// 逐位等价：tiles/glob 按原样搬，不做任何重排或补零。
impl From<crate::encode::Obs> for ObsDyn {
    fn from(o: crate::encode::Obs) -> Self {
        ObsDyn {
            tiles: o.tiles.to_vec(),
            glob: o.glob.to_vec(),
            head: o.head,
            mask: o.mask,
            mask_len: o.mask_len,
        }
    }
}

#[inline]
fn rk0(t: u8) -> usize {
    (t % 9) as usize // 0 基点数，对齐 Python rank_of
}

#[inline]
fn tile0(suit: usize, r: usize) -> u8 {
    (suit * 9 + r) as u8
}

#[inline]
fn rel(p: usize, me: usize) -> usize {
    (p + 4 - me) % 4
}

/// 巡目估计（与 Python `_turn` 完全同口径）。
#[inline]
fn turn_of(g: &Game) -> f32 {
    let wall_left = g.wall_left() as f32;
    ((DECK_SIZE as f32 - wall_left - OPENING_TILES) / 4.0).clamp(0.0, 12.0)
}

/// 每张牌的可见张数（自己手牌 + 四家牌河 + 四家副露，副露按碰 3 / 杠 4 折算）。
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

/// 对手听牌置信（启发式，公开信息）：clamp(0.2×副露组数 + 0.05×巡目, 0, 1)。
#[inline]
fn conf(g: &Game, p: usize, turn: f32) -> f32 {
    (0.2 * g.melds[p].len() as f32 + 0.05 * turn).clamp(0.0, 1.0)
}

/// Python `_max_reachable_fan`：立刻胡牌可达的最大番值（分量番型求和口径）。
pub fn max_reachable_fan_pub(g: &Game, me: usize, ft: &FanTable) -> f32 {
    let hand = &g.hands[me];
    let n_melds = g.melds[me].len();
    if crate::tiles::counts_sum(hand) as usize != (4 - n_melds) * 3 + 1 {
        return 0.0;
    }
    let waits = g.winning_tiles(me, ft);
    if waits.is_empty() {
        return 0.0;
    }
    let mut best = 0.0f32;
    for t in waits {
        let mut h: Counts = *hand;
        h[t as usize] += 1;
        let types = fan::win_type_set(&h, &g.melds[me], Some(t), ft);
        let mut val = 0.0f32;
        for tp in &types {
            if *tp != fan::PING_HU {
                val += fan_value(tp);
            }
        }
        if val <= 0.0 {
            val = 1.0; // 平胡兜底（Python `_FAN_VALUE["ping_hu"]`）
        }
        best = best.max(val);
    }
    best
}

/// 与 Python `fan.FAN_TABLE` 同值（dan_diao 按 `_FAN_VALUE` 覆盖取 1）。
#[inline]
fn fan_value(tp: &str) -> f32 {
    match tp {
        fan::PING_HU | fan::DAN_DIAO => 1.0,
        fan::DA_DUI_ZI => 5.0,
        fan::QI_DUI => 7.0,
        fan::LONG_QI_DUI => 10.0,
        fan::QING_YI_SE => 10.0,
        _ => 0.0,
    }
}

/// v2 编码：v1 基底 + ch15–30 + glob70–97。
pub fn encode_v2(g: &Game, player: u8, ft: &FanTable) -> ObsDyn {
    let me = player as usize;
    let v1 = encode::encode_v1(g, player, ft);
    let mut tiles = vec![0.0f32; NUM_CHANNELS_V2 * NUM_TILES];
    tiles[..V1_CHANNELS * NUM_TILES].copy_from_slice(&v1.tiles);
    let mut glob = vec![0.0f32; GLOBAL_DIM_V2];
    glob[..V1_GLOBAL_DIM].copy_from_slice(&v1.glob);
    fill_tiles_v2(g, me, &mut tiles);
    fill_glob_v2(g, me, &mut glob, ft);
    ObsDyn {
        tiles,
        glob,
        head: v1.head,
        mask: v1.mask,
        mask_len: v1.mask_len,
    }
}

fn fill_tiles_v2(g: &Game, me: usize, tiles: &mut [f32]) {
    // -- 15–26：四家牌河最近 1/2/3 张（时间倒序，第 1 张 = 最近）--------
    for p in 0..4usize {
        let r = rel(p, me);
        let river = &g.discards[p];
        for k in 0..3usize {
            if river.len() >= k + 1 {
                tiles[(15 + k * 4 + r) * NUM_TILES + river[river.len() - 1 - k] as usize] = 1.0;
            }
        }
    }

    let vis = visible(g, me);

    // -- 27：现物安全度 = 已打出过该牌的玩家数 / 3 ------------------------
    for t in 0..NUM_TILES {
        let mut genbutsu = 0.0f32;
        for p in 0..4usize {
            if !g.discards[p].is_empty() && g.discards[p].contains(&(t as u8)) {
                genbutsu += 1.0;
            }
        }
        tiles[27 * NUM_TILES + t] = genbutsu / 3.0;
    }

    // -- 28：筋牌安全度（简化）= 同花色 ±3 点数可见张数 / 4 ----------------
    for t in 0..NUM_TILES {
        let s = suit_of(t as u8) as usize;
        let r = rk0(t as u8);
        let mut acc = 0.0f32;
        if r < 6 {
            acc += vis[s * 9 + r + 3];
        }
        if r >= 3 {
            acc += vis[s * 9 + r - 3];
        }
        tiles[28 * NUM_TILES + t] = acc.clamp(0.0, 4.0) / 4.0;
    }

    // -- 29：生牌危险度 = 1 − 可见张数/4（不 clip，对齐 Python）----------
    for t in 0..NUM_TILES {
        tiles[29 * NUM_TILES + t] = 1.0 - vis[t] / 4.0;
    }

    // -- 30：对手潜在需求 = 副露花色 1.0 + 牌河花色 0.5，归一化 ----------
    let mut meld_suit = [0f32; 3];
    let mut river_suit = [0f32; 3];
    for p in 0..4usize {
        if p == me {
            continue;
        }
        for m in &g.melds[p] {
            meld_suit[suit_of(m.1) as usize] += 1.0;
        }
        for &t in &g.discards[p] {
            river_suit[suit_of(t) as usize] += 1.0;
        }
    }
    let ms: f32 = meld_suit.iter().sum();
    let rs: f32 = river_suit.iter().sum();
    let mut w = [0f32; 3];
    let mut wmax = 0.0f32;
    for s in 0..3 {
        w[s] = meld_suit[s] / ms.max(1.0) + 0.5 * (river_suit[s] / rs.max(1.0));
        wmax = wmax.max(w[s]);
    }
    for t in 0..NUM_TILES {
        tiles[30 * NUM_TILES + t] = w[suit_of(t as u8) as usize] / wmax.max(1e-6);
    }
}

fn fill_glob_v2(g: &Game, me: usize, glob: &mut [f32], ft: &FanTable) {
    let hand = &g.hands[me];
    let n_melds_me = g.melds[me].len();
    let wall_left = g.wall_left() as f32;
    let turn = turn_of(g);

    // -- 70–73：各家估计向听数下界 / 6 ------------------------------------
    for p in 0..4usize {
        let r = rel(p, me);
        let mut counts = [0u32; NUM_TILES];
        for &t in &g.discards[p] {
            counts[t as usize] += 1;
        }
        let mut pair_kinds = 0u32;
        for c in counts.iter() {
            if *c >= 2 {
                pair_kinds += 1;
            }
        }
        let mut adj = 0u32;
        for a in 0..NUM_TILES {
            if counts[a] == 0 {
                continue;
            }
            let s = suit_of(a as u8) as usize;
            let r0 = rk0(a as u8);
            for dr in [1i32, -1] {
                let rr = r0 as i32 + dr;
                if (0..9).contains(&rr) && counts[s * 9 + rr as usize] > 0 {
                    adj += 1;
                }
            }
        }
        adj /= 2;
        let est = (pair_kinds + adj).min(3) as f32;
        let lb = (4.0 - g.melds[p].len() as f32 - est).clamp(0.0, 6.0);
        glob[70 + r] = lb / 6.0;
    }

    // -- 74–77：各家已亮明鸡数 / 4 ---------------------------------------
    // 口径：打出过的鸡（ji_events）+ 碰/杠走的鸡（claimed_ji 的碰/杠者）。
    let mut per_player = [0f32; 4];
    for ev in &g.ji_events {
        per_player[ev.0 as usize % 4] += 1.0;
    }
    for ev in &g.claimed_ji {
        per_player[ev.1 as usize % 4] += 1.0;
    }
    for p in 0..4usize {
        glob[74 + rel(p, me)] = per_player[p].min(4.0) / 4.0;
    }

    // -- 78–81：各家当下听牌置信（启发式）--------------------------------
    for p in 0..4usize {
        glob[78 + rel(p, me)] = conf(g, p, turn);
    }

    // -- 82：巡目 / 12 ----------------------------------------------------
    glob[82] = turn / 12.0;
    // -- 83–85：阶段 one-hot（序盘 ≤4 / 中盘 5–9 / 终盘 ≥10）--------------
    if turn <= 4.0 {
        glob[83] = 1.0;
    } else if turn <= 9.0 {
        glob[84] = 1.0;
    } else {
        glob[85] = 1.0;
    }

    // -- 86–93：八个番型可行性（连续值）----------------------------------
    let mut all_counts = [0f32; NUM_TILES];
    for t in 0..NUM_TILES {
        all_counts[t] = hand[t] as f32;
    }
    for m in &g.melds[me] {
        all_counts[m.1 as usize] += 3.0;
    }
    let total: f32 = all_counts.iter().sum::<f32>().max(1.0);
    let n_kongs = g.melds[me].iter().filter(|m| fan::meld_is_kong(m.0)).count();
    let n_triplets_hand = hand.iter().filter(|&&c| c >= 3).count();
    let mut suit_tot = [0f32; 3];
    for t in 0..NUM_TILES {
        suit_tot[suit_of(t as u8) as usize] += all_counts[t];
    }
    glob[86] = ((n_kongs + n_triplets_hand) as f32 / 2.0).min(1.0);
    glob[87] = suit_tot.iter().cloned().fold(0.0f32, f32::max) / total;
    glob[88] = if n_melds_me == 0 {
        hand.iter().filter(|&&c| c == 2).count() as f32 / 7.0
    } else {
        0.0
    };
    glob[89] = ((n_triplets_hand + n_melds_me) as f32 / 4.0).min(1.0);
    glob[90] = 1.0 / (1.0 + n_triplets_hand as f32 + n_kongs as f32);
    // 将对：2/5/8 点数（0 基 rank 1/4/7）占比
    let mut jiang = 0.0f32;
    for t in 0..NUM_TILES {
        if matches!(rk0(t as u8), 1 | 4 | 7) {
            jiang += all_counts[t];
        }
    }
    glob[91] = jiang / total;
    glob[92] = if n_melds_me == 0 { 1.0 } else { 0.0 };
    let mut best_run = 0.0f32;
    for s in 0..3 {
        let mut n = 0.0f32;
        for r in 0..9 {
            if all_counts[s * 9 + r] > 0.0 {
                n += 1.0;
            }
        }
        best_run = best_run.max(n / 9.0);
    }
    glob[93] = best_run;

    // -- 94：自己可达最大番型值 / 30 --------------------------------------
    glob[94] = max_reachable_fan_pub(g, me, ft) / 30.0;

    // -- 95：黄庄紧迫度 = (1 − wall/108) × 自己未听牌 ---------------------
    let tenpai = glob[27] > 0.5;
    glob[95] = (1.0 - wall_left / DECK_SIZE as f32) * if tenpai { 0.0 } else { 1.0 };
    // -- 96：包鸡风险敞口 -------------------------------------------------
    let others_ji: f32 = per_player.iter().sum::<f32>() - per_player[me];
    glob[96] = if tenpai {
        0.0
    } else {
        1.0
    } * if others_ji > 0.0 { 1.0 } else { 0.0 };
    // -- 97：自己手牌鸡价值 = (幺鸡数 + 已成刻的鸡牌) / 4 -----------------
    let mut ji_val = 0.0f32;
    for t in 0..NUM_TILES as u8 {
        if g.ji_tiles & crate::state::bit(t) != 0 {
            let c = hand[t as usize] as f32;
            ji_val += c;
            if hand[t as usize] >= 3 || g.melded_count(me, t) >= 3 {
                ji_val += c;
            }
        }
    }
    glob[97] = (ji_val / 8.0).min(1.0);
    // YAOJI 参照保留（编译期用一下，防误删 import）
    let _ = YAOJI;
    let _ = HEAD_DISCARD;
}
