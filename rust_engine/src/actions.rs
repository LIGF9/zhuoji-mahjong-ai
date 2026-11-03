//! 动作定义 —— 与 `zhuoji/rules.py` 的 `Action(kind, tile)` 严格对齐。

/// 动作类型编号（与 Python 字符串一一对应）
pub const DISCARD: u8 = 0;
pub const HU: u8 = 1;
pub const ANGANG: u8 = 2;
pub const BUGANG: u8 = 3;
pub const PENG: u8 = 4;
pub const MINGGANG: u8 = 5;
pub const PASS: u8 = 6;

pub const KIND_NAMES: [&str; 7] = [
    "discard", "hu", "angang", "bugang", "peng", "minggang", "pass",
];

/// 一个动作：`tile = 255` 表示无牌（过 / 胡）。
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Action {
    pub kind: u8,
    pub tile: u8,
}

impl Action {
    #[inline]
    pub fn new(kind: u8, tile: u8) -> Self {
        Self { kind, tile }
    }
    #[inline]
    pub fn pass() -> Self {
        Self { kind: PASS, tile: 255 }
    }
    #[inline]
    pub fn hu() -> Self {
        Self { kind: HU, tile: 255 }
    }
    #[inline]
    pub fn discard(t: u8) -> Self {
        Self { kind: DISCARD, tile: t }
    }
    pub fn label(&self) -> String {
        let cn = match self.kind {
            DISCARD => "打",
            HU => "胡",
            ANGANG => "暗杠",
            BUGANG => "补杠",
            PENG => "碰",
            MINGGANG => "点杠",
            _ => "过",
        };
        if self.tile == 255 {
            cn.to_string()
        } else {
            format!("{}{}", cn, crate::tiles::tile_cn(self.tile))
        }
    }
    /// 紧凑编码：`kind * 27 + tile`（tile=255 记为 27）。
    #[inline]
    pub fn code(&self) -> u16 {
        let t = if self.tile == 255 { 27 } else { self.tile as u16 };
        self.kind as u16 * 28 + t
    }
}

/// 阶段
pub const PHASE_SELF_KONG: u8 = 0;
pub const PHASE_DISCARD: u8 = 1;
pub const PHASE_RESPOND: u8 = 2;
pub const PHASE_OVER: u8 = 3;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn codes_are_unique() {
        let mut seen = std::collections::HashSet::new();
        for k in 0..7u8 {
            for t in 0..28u16 {
                let a = Action::new(k, if t == 27 { 255 } else { t as u8 });
                assert!(seen.insert(a.code()), "dup {:?}", a);
            }
        }
    }
}
