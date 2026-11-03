//! 胡牌判定与番型 —— 与 `zhuoji/fan.py` 严格对齐。
//!
//! 算法同 Python 侧：**预计算 + 花色分解**。
//! 单花色只有 9 个点数，把「能被完全拆成面子」的所有计数向量（每项 ≤4）
//! 用 5 进制编码预存进 bitset；判胡 = 枚举将牌 → 去 2 张 → 三个花色是否都在表里。

use crate::tiles::{suit_of, Counts, NUM_TILES};

// ---- 番型常量 ------------------------------------------------------------
pub const PING_HU: &str = "ping_hu";
pub const DA_DUI_ZI: &str = "da_dui_zi";
pub const QI_DUI: &str = "qi_dui";
pub const LONG_QI_DUI: &str = "long_qi_dui";
pub const QING_YI_SE: &str = "qing_yi_se";
pub const DAN_DIAO: &str = "dan_diao";

// ---- 副露类型（与 fan.py 的 MELD_* 一致） --------------------------------
pub const MELD_CHOW: u8 = 0;
pub const MELD_PONG: u8 = 1;
pub const MELD_KONG_EXPOSED: u8 = 2;
pub const MELD_KONG_ADDED: u8 = 3;
pub const MELD_KONG_CONCEALED: u8 = 4;

/// 副露：(类型, 代表牌, 来源玩家)。来源 = 被碰/杠的那张牌的打出者，暗杠记自己。
pub type Meld = (u8, u8, u8);

#[inline]
pub fn meld_is_kong(kind: u8) -> bool {
    (MELD_KONG_EXPOSED..=MELD_KONG_CONCEALED).contains(&kind)
}

// ---- 5 进制编码 ----------------------------------------------------------
const POW5: [u32; 9] = [1, 5, 25, 125, 625, 3125, 15625, 78125, 390625];
const TABLE_LEN: usize = 1_953_125; // 5^9

#[inline]
fn encode9(v: &[u8; 9]) -> u32 {
    let mut e = 0u32;
    for i in 0..9 {
        e += v[i] as u32 * POW5[i];
    }
    e
}

/// 单花色面子表（bitset）。
#[derive(Clone)]
pub struct FanTable {
    bits: Vec<u64>,
}

impl FanTable {
    pub fn build() -> Self {
        let mut base: Vec<[u8; 9]> = Vec::with_capacity(16);
        for r in 0..9 {
            let mut v = [0u8; 9];
            v[r] = 3;
            base.push(v);
        }
        for r in 0..7 {
            let mut v = [0u8; 9];
            v[r] = 1;
            v[r + 1] = 1;
            v[r + 2] = 1;
            base.push(v);
        }

        let mut bits = vec![0u64; TABLE_LEN / 64 + 1];
        let mut chosen: Vec<usize> = Vec::with_capacity(4);
        Self::rec(0, &mut chosen, &base, &mut bits);
        Self { bits }
    }

    fn rec(start: usize, chosen: &mut Vec<usize>, base: &[[u8; 9]], bits: &mut [u64]) {
        if chosen.len() <= 4 {
            let mut acc = [0u8; 9];
            for &i in chosen.iter() {
                for k in 0..9 {
                    acc[k] += base[i][k];
                }
            }
            if acc.iter().all(|&x| x <= 4) {
                let e = encode9(&acc) as usize;
                bits[e >> 6] |= 1u64 << (e & 63);
            }
        }
        if chosen.len() == 4 {
            return;
        }
        for i in start..base.len() {
            chosen.push(i);
            Self::rec(i, chosen, base, bits);
            chosen.pop();
        }
    }

    #[inline]
    fn contains(&self, e: u32) -> bool {
        let e = e as usize;
        self.bits[e >> 6] & (1u64 << (e & 63)) != 0
    }

    /// 三花色是否都能被完全拆成面子。
    pub fn melds_ok(&self, c: &Counts) -> bool {
        for s in 0..3 {
            let mut v = [0u8; 9];
            for i in 0..9 {
                let x = c[s * 9 + i];
                if x > 4 {
                    return false;
                }
                v[i] = x;
            }
            if !self.contains(encode9(&v)) {
                return false;
            }
        }
        true
    }
}

// ---- 胡牌判定 ------------------------------------------------------------
pub fn is_seven_pairs(c: &Counts) -> bool {
    if crate::tiles::counts_sum(c) != 14 {
        return false;
    }
    let mut pairs = 0u32;
    for &x in c.iter() {
        match x {
            0 => {}
            2 => pairs += 1,
            4 => pairs += 2,
            _ => return false,
        }
    }
    pairs == 7
}

pub fn is_winning_hand(c: &Counts, melds: &[Meld], t: &FanTable) -> bool {
    let n_melds = melds.len() as i32;
    let need = 4 - n_melds;
    if need < 0 {
        return false;
    }
    if crate::tiles::counts_sum(c) as i32 != need * 3 + 2 {
        return false;
    }
    if melds.is_empty() && is_seven_pairs(c) {
        return true;
    }
    let mut cc = *c;
    for i in 0..NUM_TILES {
        if cc[i] >= 2 {
            cc[i] -= 2;
            let ok = t.melds_ok(&cc);
            cc[i] += 2;
            if ok {
                return true;
            }
        }
    }
    false
}

pub fn all_triplets(c: &Counts, melds: &[Meld], t: &FanTable) -> bool {
    if melds.iter().any(|m| m.0 == MELD_CHOW) {
        return false;
    }
    let n_melds = melds.len() as i32;
    let mut cc = *c;
    for i in 0..NUM_TILES {
        if cc[i] >= 2 {
            cc[i] -= 2;
            let rest = (4 - n_melds) * 3;
            let ok = t.melds_ok(&cc)
                && cc.iter().all(|&x| x % 3 == 0)
                && crate::tiles::counts_sum(&cc) as i32 == rest;
            cc[i] += 2;
            if ok {
                return true;
            }
        }
    }
    false
}

pub fn is_qing_yi_se(c: &Counts, melds: &[Meld]) -> bool {
    let mut suits: Vec<u8> = Vec::with_capacity(3);
    for (t, &x) in c.iter().enumerate() {
        if x > 0 {
            let s = suit_of(t as u8);
            if !suits.contains(&s) {
                suits.push(s);
                if suits.len() > 1 {
                    return false;
                }
            }
        }
    }
    for m in melds {
        let s = suit_of(m.1);
        if !suits.contains(&s) {
            suits.push(s);
            if suits.len() > 1 {
                return false;
            }
        }
    }
    suits.len() == 1
}

/// 全部可叠加牌型（排序后返回，便于逐字对拍）。
pub fn win_type_set(
    c: &Counts,
    melds: &[Meld],
    win_tile: Option<u8>,
    t: &FanTable,
) -> Vec<&'static str> {
    let mut types: Vec<&'static str> = Vec::with_capacity(3);
    let n_melds = melds.len();
    let seven = n_melds == 0 && is_seven_pairs(c);
    let normal = is_winning_hand(c, melds, t);
    if !(seven || normal) {
        return types;
    }
    if seven {
        let dragon = match win_tile {
            Some(w) => c[w as usize] == 4,
            None => c.iter().any(|&x| x == 4),
        };
        types.push(if dragon { LONG_QI_DUI } else { QI_DUI });
    } else {
        let all_triplet_melds = melds.iter().all(|m| m.0 != MELD_CHOW);
        if n_melds == 4 && all_triplet_melds && crate::tiles::counts_sum(c) == 2 {
            types.push(DAN_DIAO);
        } else if all_triplets(c, melds, t) {
            types.push(DA_DUI_ZI);
        }
    }
    if is_qing_yi_se(c, melds) {
        types.push(QING_YI_SE);
    }
    if types.is_empty() {
        types.push(PING_HU);
    }
    types.sort_unstable();
    types
}

#[cfg(test)]
mod tests {
    use super::*;

    fn c(spec: &[(usize, u8)]) -> Counts {
        let mut v = [0u8; NUM_TILES];
        for &(t, n) in spec {
            v[t] = n;
        }
        v
    }

    #[test]
    fn table_builds_and_recognises_meld() {
        let t = FanTable::build();
        let mut v = [0u8; 9];
        v[0] = 3; // 1万1万1万
        assert!(t.contains(encode9(&v)));
        let mut v2 = [0u8; 9];
        v2[0] = 1;
        v2[1] = 1;
        v2[2] = 1; // 123万
        assert!(t.contains(encode9(&v2)));
        let mut v3 = [0u8; 9];
        v3[0] = 1;
        v3[2] = 1;
        assert!(!t.contains(encode9(&v3)));
    }

    #[test]
    fn seven_pairs_and_ping_hu() {
        let t = FanTable::build();
        // 七对：1万2万3万4万5万6万7万 各两张
        let h = c(&[(0, 2), (1, 2), (2, 2), (3, 2), (4, 2), (5, 2), (6, 2)]);
        assert!(is_seven_pairs(&h));
        // 全万字七对 → 同时是清一色
        assert_eq!(win_type_set(&h, &[], Some(6), &t), vec![QI_DUI, QING_YI_SE]);
        // 混花色七对 → 只有七对
        let h3 = c(&[(0, 2), (1, 2), (2, 2), (3, 2), (9, 2), (10, 2), (18, 2)]);
        assert_eq!(win_type_set(&h3, &[], Some(18), &t), vec![QI_DUI]);

        // 平胡：123 456 789 万 + 111 条 + 22 筒
        let h2 = c(&[(0, 1), (1, 1), (2, 1), (3, 1), (4, 1), (5, 1),
                     (6, 1), (7, 1), (8, 1), (9, 3), (18, 2)]);
        assert!(is_winning_hand(&h2, &[], &t));
        assert_eq!(win_type_set(&h2, &[], None, &t), vec![PING_HU]);
    }

    #[test]
    fn all_triplets_scores_da_dui_zi() {
        let t = FanTable::build();
        // 111 222 333 万 + 444 条 + 55 筒
        let h = c(&[(0, 3), (1, 3), (2, 3), (9, 3), (18, 2)]);
        assert!(is_winning_hand(&h, &[], &t));
        let types = win_type_set(&h, &[], None, &t);
        assert!(types.contains(&DA_DUI_ZI), "types={:?}", types);
    }
}
