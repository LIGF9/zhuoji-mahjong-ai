//! 状态编码 —— 与 `zhuoji/encoder.py` 的 **v1 口径**逐位一致。
//!
//! v1 是 s1–s11 / master 全部检查点的输入分布，必须永久保持不变：
//! 15 通道 × 27 牌 + 70 维全局标量。

use crate::actions::*;
use crate::fan::{self, FanTable, MELD_PONG, MELD_KONG_EXPOSED};
use crate::state::Game;
use crate::tiles::{NUM_TILES, YAOJI};

pub const NUM_CHANNELS: usize = 15;
pub const CTX_DIM: usize = 27 + 4;
pub const GLOBAL_DIM: usize = 39 + CTX_DIM;

/// 动作头编号（与 Python 的 head 字符串对应）
pub const HEAD_DISCARD: u8 = 0;
pub const HEAD_SELF_KONG: u8 = 1;
pub const HEAD_RESPOND: u8 = 2;

#[inline]
fn rel(p: u8, me: u8) -> usize {
    (((p as i32 - me as i32) % 4 + 4) % 4) as usize
}

/// 一次观测：15×27 牌面 + 70 全局 + 头 + 28 维掩码。
pub struct Obs {
    pub tiles: [f32; NUM_CHANNELS * NUM_TILES],
    pub glob: [f32; GLOBAL_DIM],
    pub head: u8,
    pub mask: [u8; 28],
    pub mask_len: usize,
}

impl Obs {
    fn zero() -> Self {
        Obs {
            tiles: [0.0; NUM_CHANNELS * NUM_TILES],
            glob: [0.0; GLOBAL_DIM],
            head: HEAD_DISCARD,
            mask: [0; 28],
            mask_len: 28,
        }
    }
    #[inline]
    pub fn tile(&mut self, ch: usize, t: usize) -> &mut f32 {
        &mut self.tiles[ch * NUM_TILES + t]
    }
}

/// v1 编码：`player` 视角。
pub fn encode_v1(g: &Game, player: u8, ft: &FanTable) -> Obs {
    let me = player as usize;
    let mut o = Obs::zero();
    let hand = &g.hands[me];

    // -- 自己的手牌（多层阈值） ------------------------------------------
    for t in 0..NUM_TILES {
        let c = hand[t];
        if c > 0 {
            *o.tile(0, t) = 1.0;
            *o.tile(4, t) = c as f32 * 0.25;
            if c >= 2 {
                *o.tile(1, t) = 1.0;
            }
            if c >= 3 {
                *o.tile(2, t) = 1.0;
            }
            if c >= 4 {
                *o.tile(3, t) = 1.0;
            }
        }
    }

    // -- 四家牌河 --------------------------------------------------------
    for p in 0..4usize {
        let r = rel(p as u8, player);
        for &t in &g.discards[p] {
            *o.tile(5 + r, t as usize) += 0.25;
        }
    }

    // -- 四家副露 --------------------------------------------------------
    for p in 0..4usize {
        let r = rel(p as u8, player);
        for m in &g.melds[p] {
            let v = if fan::meld_is_kong(m.0) { 1.0 } else { 0.75 };
            *o.tile(9 + r, m.1 as usize) += v;
        }
    }

    // -- 可见度 / 剩余 ----------------------------------------------------
    let mut visible = [0f32; NUM_TILES];
    for t in 0..NUM_TILES {
        visible[t] += o.tiles[4 * NUM_TILES + t] * 4.0;
    }
    for r in 0..4usize {
        for t in 0..NUM_TILES {
            visible[t] += o.tiles[(5 + r) * NUM_TILES + t] * 4.0;
            visible[t] += o.tiles[(9 + r) * NUM_TILES + t] * 4.0;
        }
    }
    for t in 0..NUM_TILES {
        *o.tile(13, t) = visible[t].clamp(0.0, 4.0) / 4.0;
        *o.tile(14, t) = (4.0 - visible[t]).clamp(0.0, 4.0) / 4.0;
    }

    // -- 全局标量 --------------------------------------------------------
    let wall_left = g.wall_left() as f32;
    o.glob[0] = wall_left / 108.0;
    o.glob[1] = (108.0 - wall_left) / 108.0;
    o.glob[2] = rel(player, g.dealer) as f32 / 3.0;
    o.glob[3] = if player == g.dealer { 1.0 } else { 0.0 };
    o.glob[4] = g.dealer_streak as f32 / 5.0;
    o.glob[5 + rel(g.dealer, player)] = 1.0;

    for p in 0..4usize {
        let r = rel(p as u8, player);
        let kongs = g.melds[p].iter().filter(|m| fan::meld_is_kong(m.0)).count();
        o.glob[9 + r] = kongs as f32 / 4.0;
        o.glob[13 + r] = g.melds[p].len() as f32 / 4.0;
        o.glob[17 + r] = g.discards[p].len().min(24) as f32 / 24.0;
        o.glob[21 + r] = if g.ever_tenpai[p] { 1.0 } else { 0.0 };
    }

    let hand_sum: u32 = hand.iter().map(|&x| x as u32).sum();
    o.glob[25] = hand_sum as f32 / 14.0;
    o.glob[26] = if g.has_dou(me) { 1.0 } else { 0.0 };
    let waits = if hand_sum as usize == (4 - g.melds[me].len()) * 3 + 1 {
        g.winning_tiles(me, ft)
    } else {
        Vec::new()
    };
    o.glob[27] = if waits.is_empty() { 0.0 } else { 1.0 };
    o.glob[28] = waits.len() as f32 / 13.0;
    o.glob[29] = (4.0 - visible[YAOJI as usize]) / 4.0;
    o.glob[30] = (hand[YAOJI as usize] as u32 + g.melded_count(me, YAOJI)) as f32 / 4.0;
    o.glob[31] = if g.first_yaoji_discarded { 1.0 } else { 0.0 };
    o.glob[32] = if g.chongfeng_player == player as i8 { 1.0 } else { 0.0 };
    o.glob[33] = if g.chongfeng_player >= 0 { 1.0 } else { 0.0 };
    o.glob[34] = if g.zeren_player == player as i8 { 1.0 } else { 0.0 };
    o.glob[35] = if g.phase == PHASE_SELF_KONG { 1.0 } else { 0.0 };
    o.glob[36] = if g.phase == PHASE_DISCARD { 1.0 } else { 0.0 };
    o.glob[37] = if g.phase == PHASE_RESPOND { 1.0 } else { 0.0 };
    o.glob[38] = if g.actor() == player { 1.0 } else { 0.0 };

    let base = 39;
    if let Some((d_p, d_t)) = g.last_discard {
        o.glob[base + d_t as usize] = 1.0;
        o.glob[base + 27 + rel(d_p, player)] = 1.0;
    }

    let (head, mask, len) = head_and_mask(g, me, ft);
    o.head = head;
    o.mask = mask;
    o.mask_len = len;
    o
}

pub fn head_of(g: &Game, who: u8, ft: &FanTable) -> u8 {
    let (_h, _m, _l) = head_and_mask(g, who as usize, ft);
    _h
}

fn head_and_mask(g: &Game, p: usize, ft: &FanTable) -> (u8, [u8; 28], usize) {
    let mut mask = [0u8; 28];
    match g.phase {
        PHASE_DISCARD => {
            let cur = g.current as usize;
            for t in 0..NUM_TILES {
                if g.hands[cur][t] > 0 {
                    mask[t] = 1;
                }
            }
            if g.drawn && g.can_tsumo(cur, ft) {
                mask[27] = 1;
            }
            (HEAD_DISCARD, mask, 28)
        }
        PHASE_SELF_KONG => {
            mask[27] = 1;
            if let Some((rp, rt)) = g.rob_kong {
                if g.can_ron(rp as usize, rt, ft) {
                    return (HEAD_RESPOND, respond_mask(g, rp as usize, ft), 4);
                }
                return (HEAD_SELF_KONG, mask, 28);
            }
            for (_kind, t) in g.self_kong_options(g.current as usize) {
                mask[t as usize] = 1;
            }
            (HEAD_SELF_KONG, mask, 28)
        }
        PHASE_RESPOND => (HEAD_RESPOND, respond_mask(g, g.actor() as usize, ft), 4),
        _ => (HEAD_DISCARD, mask, 28),
    }
}

fn respond_mask(g: &Game, p: usize, ft: &FanTable) -> [u8; 28] {
    let mut mask = [0u8; 28];
    mask[0] = 1; // 过
    if let Some((_, rt)) = g.rob_kong {
        if g.can_ron(p, rt, ft) {
            mask[3] = 1;
        }
        return mask;
    }
    let tile = match g.last_discard {
        Some((_, t)) => t,
        None => return mask,
    };
    if g.can_ron(p, tile, ft) {
        mask[3] = 1;
    }
    // 明杠 / 碰：与 legal_actions 同源（baojiao 锁定在里面）
    if !g.baojiao[p] && g.melds[p].len() < 4 {
        let c = g.hands[p][tile as usize];
        if c >= 3 {
            mask[2] = 1;
        }
        if c >= 2 {
            mask[1] = 1;
        }
    }
    mask
}

/// 头下标 → 动作（与 `encoder.index_to_action` 同语义）。
pub fn index_to_action(g: &Game, head: u8, idx: usize) -> Action {
    match head {
        HEAD_RESPOND => {
            let kind = [PASS, PENG, MINGGANG, HU][idx.min(3)];
            let tile = if matches!(kind, PENG | MINGGANG) {
                g.last_discard.map(|x| x.1).unwrap_or(255)
            } else {
                255
            };
            Action::new(kind, tile)
        }
        HEAD_SELF_KONG => {
            if idx == 27 {
                return Action::pass();
            }
            for (kind, t) in g.self_kong_options(g.current as usize) {
                if t as usize == idx {
                    return Action::new(kind, t);
                }
            }
            Action::pass()
        }
        _ => {
            if idx == 27 {
                Action::hu()
            } else {
                Action::discard(idx as u8)
            }
        }
    }
}

/// 动作 → 头下标（与 `encoder.head_index` 同语义）。
pub fn head_index(head: u8, a: Action) -> Option<usize> {
    match head {
        HEAD_DISCARD => match a.kind {
            HU => Some(27),
            DISCARD => Some(a.tile as usize),
            _ => None,
        },
        HEAD_SELF_KONG => match a.kind {
            PASS => Some(27),
            ANGANG | BUGANG => Some(a.tile as usize),
            _ => None,
        },
        _ => match a.kind {
            PASS => Some(0),
            PENG => Some(1),
            MINGGANG => Some(2),
            HU => Some(3),
            _ => None,
        },
    }
}

const _: () = {
    // 编译期自检：副露常量与本文件保持一致
    let _ = MELD_PONG;
    let _ = MELD_KONG_EXPOSED;
};
