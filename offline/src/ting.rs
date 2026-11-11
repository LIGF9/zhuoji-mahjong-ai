//! 离线 adept 档：**速听**（向听数驱动的规则策略）。
//!
//! 这就是线上 2026-11 之前那套「教师 · 速听」的本机移植：`zhuoji/bots.py`
//! 的 `HeuristicBot` + `TEACHER_STYLES["tenpai_rush"]` 权重，**噪声归零**。
//! 归零不是口味问题：四档难度要能被用户当「水平」理解，同一局面必须每次
//! 打同一张；带高斯噪声的教师只适合造训练数据。
//!
//! 与 Python 的差异只有两处，都是确定性问题：
//!   1. `noise=0`（原 4.0）；
//!   2. 碰后向听不变时不再 25% 概率乱碰（原 `rng.random() < 0.25`）。
//!
//! 其余口径逐项对齐：向听公式、`useful_tile_score`、冲锋鸡 +9、留幺鸡 ±1、
//! 熟张安全 +0.3/张、听牌奖励 +90。

use dushan_engine::actions::{
    self, Action, ANGANG, BUGANG, DISCARD, HU, MINGGANG, PASS, PENG, PHASE_DISCARD,
    PHASE_RESPOND, PHASE_SELF_KONG,
};
use dushan_engine::fan::{Meld, FanTable};
use dushan_engine::state::Game;
use dushan_engine::tiles::{Counts, YAOJI};

const TENPAI_BONUS: f64 = 90.0;
const SAFETY: f64 = 0.3;
const KEEP_YAOJI: f64 = 0.5;
const CHONGFENG: f64 = 9.0;

// ---------------------------------------------------------------------------
// 向听数
// ---------------------------------------------------------------------------

/// 叶子/中间节点打分：面子 2 分、搭子 1 分、将牌 1 分，共 5 块上限。
/// 超限时先裁搭子再裁面子（与 Python `shanten` 的裁剪顺序一致）。
fn score_of(m: usize, p: usize, k: usize, n_melds: usize) -> i32 {
    let total = n_melds + m + p + k;
    let (mm, pp) = if total > 5 {
        let mut cut = total - 5;
        let use_p = cut.min(p);
        cut -= use_p;
        let mm = m as i32 - cut as i32;
        if mm < 0 {
            return i32::MAX / 4;
        }
        (mm as usize, p - use_p)
    } else {
        (m, p)
    };
    8 - (2 * (n_melds + mm) + pp + k) as i32
}

fn dfs(c: &mut Counts, i: usize, m: usize, p: usize, k: usize, n_melds: usize) -> i32 {
    // 注意：`score_of` 已经取过 `8 - val`，所以这里是**越小越好**，全程 min。
    // （第一版写成 max，结果任何手牌都返回 8 —— 向听方向反了。）
    let mut best = score_of(m, p, k, n_melds);
    if i >= 27 {
        return best;
    }
    if c[i] == 0 {
        return dfs(c, i + 1, m, p, k, n_melds);
    }
    let suit_start = (i / 9) * 9;
    let rank = i - suit_start;

    // (1) 当孤张弃掉
    c[i] -= 1;
    best = best.min(dfs(c, i, m, p, k, n_melds));
    c[i] += 1;

    // (2) 刻子
    if c[i] >= 3 {
        c[i] -= 3;
        best = best.min(dfs(c, i, m + 1, p, k, n_melds));
        c[i] += 3;
    }

    // (3) 顺子
    if rank <= 6 && c[i + 1] > 0 && c[i + 2] > 0 {
        c[i] -= 1;
        c[i + 1] -= 1;
        c[i + 2] -= 1;
        best = best.min(dfs(c, i, m + 1, p, k, n_melds));
        c[i] += 1;
        c[i + 1] += 1;
        c[i + 2] += 1;
    }

    // (4) 对子：可作将牌，也可作搭子
    if c[i] >= 2 {
        c[i] -= 2;
        if k == 0 {
            best = best.min(dfs(c, i, m, p, 1, n_melds));
        }
        best = best.min(dfs(c, i, m, p + 1, k, n_melds));
        c[i] += 2;
    }

    // (5) 两面 / 嵌张
    for d in 1..=2usize {
        if rank + d <= 8 && c[i + d] > 0 {
            c[i] -= 1;
            c[i + d] -= 1;
            best = best.min(dfs(c, i, m, p + 1, k, n_melds));
            c[i] += 1;
            c[i + d] += 1;
        }
    }

    best
}

/// 向听数：`-1` 已胡 / `0` 听牌 / 越大越远。
pub fn shanten(c: &Counts, n_melds: usize) -> i32 {
    let mut h = *c;
    dfs(&mut h, 0, 0, 0, 0, n_melds)
}

/// 与 `zhuoji/shanten.py::useful_tile_score` 同式的牌型潜力分。
fn useful_tile_score(c: &Counts) -> f64 {
    let mut s = 0.0;
    for suit in 0..3usize {
        let v = &c[suit * 9..(suit + 1) * 9];
        for i in 0..9usize {
            let x = v[i];
            if x == 0 {
                continue;
            }
            s += x.min(2) as f64 * 2.0;
            if x >= 3 {
                s += 4.0;
            }
            if i > 0 && v[i - 1] > 0 {
                s += 2.0;
            }
            if i > 1 && v[i - 2] > 0 {
                s += 1.0;
            }
        }
    }
    s
}

// ---------------------------------------------------------------------------
// 决策
// ---------------------------------------------------------------------------

fn visible_counts(g: &Game, p: usize) -> [f64; 27] {
    let mut v = [0f64; 27];
    for q in 0..4 {
        for &t in &g.discards[q] {
            v[t as usize] += 1.0;
        }
        for &(_k, t, _s) in &g.melds[q] {
            v[t as usize] += 3.0;
        }
    }
    for t in 0..27 {
        v[t] += g.hands[p][t] as f64;
    }
    v
}

/// 出牌打分：`-100×向听 + 牌型潜力 + 听牌奖励 + 幺鸡/安全项`。
fn discard_scores(g: &Game, p: usize) -> Vec<(f64, u8)> {
    let hand = g.hands[p];
    let n_melds = g.melds[p].len();
    let vis = visible_counts(g, p);
    let first_yaoji = g.chongfeng_player < 0 && !g.first_yaoji_discarded;
    let mut out = Vec::with_capacity(14);
    for t in 0..27u8 {
        if hand[t as usize] == 0 {
            continue;
        }
        let mut h = hand;
        h[t as usize] -= 1;
        let s = shanten(&h, n_melds);
        let mut sc = -100.0 * s as f64 + useful_tile_score(&h);
        if s == 0 {
            sc += TENPAI_BONUS;
        }
        if t == YAOJI {
            if first_yaoji {
                sc += CHONGFENG;
            } else {
                sc -= KEEP_YAOJI * 2.0;
            }
        } else if h[YAOJI as usize] > 0 {
            sc += KEEP_YAOJI;
        }
        sc += SAFETY * vis[t as usize];
        out.push((sc, t));
    }
    // 与 Python 一致：严格大于才替换 ⇒ 同分取牌种序号小的
    out.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal)
        .then(a.1.cmp(&b.1)));
    out
}

fn pick_discard(g: &Game, p: usize) -> Option<Action> {
    discard_scores(g, p).first().map(|&(_, t)| Action::discard(t))
}

fn self_kong(g: &Game, p: usize, ft: &FanTable) -> Action {
    let acts = g.legal_actions(ft);
    if acts.len() == 1 {
        return acts[0];
    }
    let hand = g.hands[p];
    let n_melds = g.melds[p].len();
    let mut best = Action::pass();
    let mut best_s = shanten(&hand, n_melds);
    for a in &acts {
        let after = match a.kind {
            ANGANG => {
                let mut h = hand;
                h[a.tile as usize] = h[a.tile as usize].saturating_sub(4);
                shanten(&h, n_melds + 1)
            }
            BUGANG => {
                let mut h = hand;
                h[a.tile as usize] = h[a.tile as usize].saturating_sub(1);
                shanten(&h, n_melds)
            }
            _ => continue,
        };
        if after <= best_s {
            best_s = after;
            best = *a;
        }
    }
    best
}

fn respond(g: &Game, p: usize, ft: &FanTable) -> Action {
    let acts = g.legal_actions(ft);
    if let Some(a) = acts.iter().find(|a| a.kind == HU) {
        return *a;
    }
    let hand = g.hands[p];
    let n_melds = g.melds[p].len();
    let cur = shanten(&hand, n_melds);
    let tile = g.last_discard.map(|(_, t)| t).unwrap_or(255);

    if let Some(a) = acts.iter().find(|a| a.kind == MINGGANG) {
        let mut h = hand;
        h[a.tile as usize] = h[a.tile as usize].saturating_sub(3);
        if shanten(&h, n_melds + 1) <= cur {
            return *a;
        }
    }
    if let Some(a) = acts.iter().find(|a| a.kind == PENG) {
        let mut h = hand;
        if tile < 27 {
            h[tile as usize] = h[tile as usize].saturating_sub(2);
        }
        if shanten(&h, n_melds + 1) < cur {
            return *a;
        }
    }
    acts.iter().copied().find(|a| a.kind == PASS).unwrap_or_else(Action::pass)
}

/// 一次决策（AI 座位与推荐共用）。
pub fn choose(g: &Game, p: u8, ft: &FanTable) -> Action {
    let p = p as usize;
    match g.phase {
        PHASE_SELF_KONG => self_kong(g, p, ft),
        PHASE_DISCARD => {
            let acts = g.legal_actions(ft);
            if let Some(a) = acts.iter().find(|a| a.kind == HU) {
                return *a;
            }
            pick_discard(g, p).unwrap_or_else(|| Action::pass())
        }
        PHASE_RESPOND => respond(g, p, ft),
        _ => Action::pass(),
    }
}

/// `/api/hint` 用：给出（分数, 动作）降序，分数与 `choose` 同一真源。
pub fn rank(g: &Game, p: u8, ft: &FanTable) -> Vec<(f64, Action)> {
    let pu = p as usize;
    let hand = g.hands[pu];
    let n_melds = g.melds[pu].len();
    let cur = shanten(&hand, n_melds);
    let acts = g.legal_actions(ft);
    let ds: std::collections::HashMap<u8, f64> =
        discard_scores(g, pu).into_iter().map(|(s, t)| (t, s)).collect();
    let mut out: Vec<(f64, Action)> = Vec::new();
    for a in acts {
        let sc = match a.kind {
            HU => 1000.0,
            DISCARD => *ds.get(&a.tile).unwrap_or(&(-100.0 * cur as f64)),
            ANGANG | BUGANG => {
                let mut h = hand;
                h[a.tile as usize] = h[a.tile as usize]
                    .saturating_sub(if a.kind == ANGANG { 4 } else { 1 });
                let after = shanten(&h, n_melds + if a.kind == ANGANG { 1 } else { 0 });
                -100.0 * after as f64 + 40.0
            }
            MINGGANG | PENG => {
                let mut h = hand;
                h[a.tile as usize] = h[a.tile as usize]
                    .saturating_sub(if a.kind == MINGGANG { 3 } else { 2 });
                let after = shanten(&h, n_melds + 1);
                -100.0 * after as f64 + 20.0
            }
            _ => -100.0 * cur as f64 - 10.0,
        };
        out.push((sc, a));
    }
    out.sort_by(|x, y| y.0.partial_cmp(&x.0).unwrap_or(std::cmp::Ordering::Equal));
    out
}

/// 供单测：从 `Counts` 直接算向听（暴露在 crate 内）。
pub fn shanten_of(c: &Counts, n_melds: usize) -> i32 { shanten(c, n_melds) }

#[cfg(test)]
mod tests {
    use super::*;

    fn counts(spec: &[(u8, u8)]) -> Counts {
        let mut c = [0u8; 27];
        for &(t, n) in spec {
            c[t as usize] = n;
        }
        c
    }

    #[test]
    fn shanten_basic_hands() {
        // 123 456 789 万 + 123 条 + 99 筒(将) = 胡
        let win = counts(&[(0, 1), (1, 1), (2, 1), (3, 1), (4, 1), (5, 1),
                           (6, 1), (7, 1), (8, 1), (9, 1), (10, 1), (11, 1),
                           (26, 2)]);
        assert_eq!(shanten(&win, 0), -1);
        // 去掉一张 9 筒 → 听牌
        let mut tenpai = win;
        tenpai[26] = 1;
        assert_eq!(shanten(&tenpai, 0), 0);
    }

    /// 与 Python `zhuoji/shanten.py` 逐手对拍（随机种子 20261111 抽的 6 手）。
    /// 只靠自家单测会「两边一起错」，所以期望值由 Python 侧算出后写死在这里。
    #[test]
    fn shanten_matches_python() {
        let cases: Vec<(Counts, i32)> = vec![
            (counts(&[(0, 1), (4, 1), (6, 1), (9, 2), (12, 2), (16, 1), (17, 2),
                      (22, 1), (23, 1), (25, 1), (26, 1)]), 3),
            (counts(&[(0, 2), (6, 1), (7, 1), (13, 1), (15, 1), (17, 1), (19, 3),
                      (24, 1), (26, 2)]), 2),
            (counts(&[(0, 2), (1, 1), (2, 2), (5, 1), (6, 1), (8, 1), (12, 1),
                      (16, 1), (17, 2), (19, 1), (25, 1)]), 3),
            (counts(&[(2, 1), (3, 1), (4, 1), (7, 2), (8, 1), (9, 1), (11, 1),
                      (12, 2), (16, 1), (17, 1), (24, 1)]), 2),
            (counts(&[(0, 1), (1, 1), (5, 2), (7, 1), (8, 2), (9, 1), (16, 1),
                      (18, 1), (20, 1), (24, 1), (25, 2)]), 3),
            (counts(&[(0, 1), (2, 1), (4, 2), (5, 1), (6, 2), (8, 1), (9, 1),
                      (13, 1), (15, 1), (19, 1), (25, 1)]), 3),
        ];
        for (i, (c, want)) in cases.iter().enumerate() {
            assert_eq!(shanten(c, 0), *want, "case {i} 与 Python 不一致");
        }
    }

    #[test]
    fn shanten_uses_melds() {
        // 副露 2 组 + 手牌 8 张：123 456 万 + 789 条 + 5 筒 对子 → 听
        let c = counts(&[(0, 1), (1, 1), (2, 1), (3, 1), (4, 1), (5, 1),
                         (6, 1), (7, 1), (8, 1), (22, 2)]);
        let s0 = shanten(&c, 2);
        assert!(s0 <= 1, "got {s0}");
    }
}
