//! 牌张基础定义 —— 与 `zhuoji/tiles.py` 严格对齐。
//!
//! 27 种牌：0..8 万(W1..W9)、9..17 条(T1..T9)、18..26 筒(B1..B9)，每种 4 张共 108。

pub const NUM_TILES: usize = 27;
pub const TILES_PER_TYPE: u8 = 4;
pub const DECK_SIZE: usize = 108;

/// 幺鸡（一条），捉鸡规则里的常鸡。
pub const YAOJI: u8 = 9;

#[inline]
pub fn suit_of(t: u8) -> u8 {
    t / 9
}

#[inline]
pub fn rank_of(t: u8) -> u8 {
    t % 9 + 1
}

#[inline]
pub fn tile_from(suit: u8, rank: u8) -> u8 {
    suit * 9 + (rank - 1)
}

/// 中文牌名，如「五万」「一条」。对拍日志用。
pub fn tile_cn(t: u8) -> String {
    const CN: [&str; 9] = ["一", "二", "三", "四", "五", "六", "七", "八", "九"];
    const SUIT: [&str; 3] = ["万", "条", "筒"];
    format!("{}{}", CN[(t % 9) as usize], SUIT[(t / 9) as usize])
}

/// 27 维计数向量。
pub type Counts = [u8; NUM_TILES];

/// 一副未洗的完整牌墙。
pub fn full_deck() -> Vec<u8> {
    let mut v = Vec::with_capacity(DECK_SIZE);
    for t in 0..NUM_TILES as u8 {
        for _ in 0..TILES_PER_TYPE {
            v.push(t);
        }
    }
    v
}

#[inline]
pub fn counts_sum(c: &Counts) -> u32 {
    c.iter().map(|&x| x as u32).sum()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn deck_is_108() {
        assert_eq!(full_deck().len(), DECK_SIZE);
    }

    #[test]
    fn tile_names() {
        assert_eq!(tile_cn(0), "一万");
        assert_eq!(tile_cn(9), "一条");
        assert_eq!(tile_cn(26), "九筒");
        assert_eq!(suit_of(26), 2);
        assert_eq!(rank_of(26), 9);
    }
}
