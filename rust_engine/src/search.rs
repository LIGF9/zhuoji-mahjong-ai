//! PIMC 搜索内核 —— 与 `SearchAgent._search_fair_batched` + `_expand_joint` 同构。
//!
//! 与 Python 版的两点区别：
//! 1. **网络前向的边界**：状态推进全在 Rust，每推到一个需要网络的位置先攒起来，
//!    等 Python 一次性算完再回填 —— 每层只有一次跨语言往返。
//! 2. **跨叶子合并**（`from_seeds`）：第二/第三层把多个叶子的动作全塞进同一个
//!    Search，避免 Python 版「每个叶子一次前向」造成的碎批。

use crate::actions::*;
use crate::encode::{self, Obs};
use crate::fan::FanTable;
use crate::rng::Rng;
use crate::state::Game;
use crate::tiles::NUM_TILES;

pub const KIND_POLICY: u8 = 0;
pub const KIND_VALUE: u8 = 1;

pub struct Task {
    pub game: Game,
    pub q: f64,
    pub done: bool,
    pub ff: usize,
    /// 从第一层 determinization 起点开始的完整动作序列（终局时供 Python 精确结算）
    pub trace: Vec<Action>,
    /// 该 task 归属的第一层 origin 编号（重放起点）
    pub root: usize,
}

pub struct Batch {
    pub obs: Vec<Obs>,
    pub kind: Vec<u8>,
    pub head: Vec<u8>,
    pub task_index: Vec<usize>,
}

pub struct Search {
    pub p: u8,
    pub tasks: Vec<Task>,
    /// task → origin 编号
    pub groups: Vec<usize>,
    /// 每个 origin 的起点局面（determinization 后 / 上层叶子）
    pub origins: Vec<Game>,
    /// 每个 origin 的 IS 权重（p_uniform/q_proposal；均匀路径恒 1.0）
    pub weights: Vec<f64>,
    pub ff_cap: usize,
    pending: Vec<(usize, u8, u8)>,
    unsettled: Vec<usize>,
    pub ft: FanTable,
}

fn hand_sum(g: &Game, q: usize) -> usize {
    g.hands[q].iter().map(|&x| x as usize).sum()
}

/// 公平 determinization：重排「别家手牌 + 剩余牌墙」，本家与牌河不动。
pub fn determinize(g: &Game, p: usize, rng: &mut Rng) -> Game {
    let (sim, _w) = determinize_lw(g, p, rng, 1.0);
    sim
}

/// 实验 B1（2026-11-03）：信念化 determinization。
///
/// 与均匀版的唯一区别是「别家手牌槽」的分配分布：
///   proposal ∝ λ·uniform + (1−λ)·belief_q(t)
/// belief_q(t)（每张牌、按持有方 q 单独算）：
///   - q 弃过 ≥2 张 t → ×0.25；弃过 1 张 → ×0.6；q 的副露含 t → ×0.5；
///   - 牌墙槽仍均匀（墙的未来顺序没有任何公开证据）。
/// λ=1 退化为均匀（此时直接走原版 shuffle 路径，逐位兼容审计脚本）；
/// λ<1 时返回 IS 权重 = p_uniform(w)/q(w)（逐槽 S_w/S 的连乘），
/// 供上层做自归一化重要性加权聚合（reweight=true 时）。
pub fn determinize_lw(g: &Game, p: usize, rng: &mut Rng, lam: f64) -> (Game, f64) {
    if lam >= 1.0 {
        let mut sim = g.clone();
        let mut totals = [0usize; 4];
        for q in 0..4 {
            if q != p {
                totals[q] = hand_sum(&sim, q);
            }
        }
        let mut pool: Vec<u8> = Vec::with_capacity(80);
        for q in 0..4 {
            if q == p {
                continue;
            }
            for t in 0..NUM_TILES {
                let c = sim.hands[q][t];
                if c > 0 {
                    for _ in 0..c {
                        pool.push(t as u8);
                    }
                    sim.hands[q][t] = 0;
                }
            }
        }
        let wall_n = sim.wall.len() - sim.wall_pos;
        pool.extend_from_slice(&sim.wall[sim.wall_pos..]);
        pool.sort_unstable();
        rng.shuffle(&mut pool);
        sim.wall.truncate(sim.wall_pos);
        sim.wall.extend_from_slice(&pool[..wall_n]);
        let mut k = wall_n;
        for q in 0..4 {
            if q == p {
                continue;
            }
            for _ in 0..totals[q] {
                sim.hands[q][pool[k] as usize] += 1;
                k += 1;
            }
        }
        return (sim, 1.0);
    }

    let mut sim = g.clone();
    let mut totals = [0usize; 4];
    for q in 0..4 {
        if q != p {
            totals[q] = hand_sum(&sim, q);
        }
    }
    // 不可见池（多重集）
    let mut rem = [0u8; NUM_TILES];
    for q in 0..4 {
        if q == p {
            continue;
        }
        for t in 0..NUM_TILES {
            rem[t] += sim.hands[q][t];
            sim.hands[q][t] = 0;
        }
    }
    for t in sim.wall[sim.wall_pos..].iter() {
        rem[*t as usize] += 1;
    }
    // 每个对手、每张牌的持有倾向权重
    let mut w = [[1.0f64; NUM_TILES]; 4];
    for q in 0..4 {
        if q == p {
            continue;
        }
        let mut dc = [0u8; NUM_TILES];
        for t in sim.discards[q].iter() {
            dc[*t as usize] = dc[*t as usize].saturating_add(1);
        }
        let mut md = [false; NUM_TILES];
        for (k, t, _s) in sim.melds[q].iter() {
            let _ = k;
            md[*t as usize] = true;
        }
        for t in 0..NUM_TILES {
            let mut f = 1.0f64;
            if dc[t] >= 2 {
                f *= 0.25;
            } else if dc[t] == 1 {
                f *= 0.6;
            }
            if md[t] {
                f *= 0.5;
            }
            // λ 混合：λ·均匀 + (1−λ)·信念（权重空间线性混合）
            w[q][t] = lam + (1.0 - lam) * f;
        }
    }
    // 依次给每个对手的手牌槽做加权无放回采样；牌墙槽从余量均匀取。
    // IS 权重：每手牌槽 target/proposal = (Σ rem) / (Σ rem·w_eff)，连乘。
    let mut log_w = 0.0f64;
    let mut drawn = [[0u8; NUM_TILES]; 4];
    for q in 0..4 {
        if q == p {
            continue;
        }
        for _ in 0..totals[q] {
            let mut s_all = 0.0f64;
            let mut s_w = 0.0f64;
            for t in 0..NUM_TILES {
                let r = rem[t] as f64;
                s_all += r;
                s_w += r * w[q][t];
            }
            if s_w <= 0.0 {
                // 权重全 0 的极端配置（理论不可达）：整槽退化为均匀，权重记 1
                let s = s_all.max(1.0);
                log_w += s.ln() - s.ln();
                let mut pick = rng.below(s_all as usize) + 1;
                for t in 0..NUM_TILES {
                    if rem[t] == 0 {
                        continue;
                    }
                    if pick <= rem[t] as usize {
                        rem[t] -= 1;
                        drawn[q][t] += 1;
                        break;
                    }
                    pick -= rem[t] as usize;
                }
                continue;
            }
            log_w += s_all.ln() - s_w.ln();
            let mut x = rng.next_u64() as f64 / u64::MAX as f64 * s_w;
            let mut picked = NUM_TILES;
            for t in 0..NUM_TILES {
                if rem[t] == 0 {
                    continue;
                }
                x -= rem[t] as f64 * w[q][t];
                if x <= 0.0 {
                    picked = t;
                    break;
                }
            }
            if picked == NUM_TILES {
                // 浮点兜底：取最后一张有余量的
                for t in (0..NUM_TILES).rev() {
                    if rem[t] > 0 {
                        picked = t;
                        break;
                    }
                }
            }
            rem[picked] -= 1;
            drawn[q][picked] += 1;
        }
    }
    // 牌墙：余量均匀排列
    let mut pool: Vec<u8> = Vec::new();
    for t in 0..NUM_TILES {
        for _ in 0..rem[t] {
            pool.push(t as u8);
        }
    }
    rng.shuffle(&mut pool);
    let wall_n = sim.wall.len() - sim.wall_pos;
    sim.wall.truncate(sim.wall_pos);
    sim.wall.extend_from_slice(&pool[..wall_n]);
    for q in 0..4 {
        if q == p {
            continue;
        }
        for t in 0..NUM_TILES {
            sim.hands[q][t] = drawn[q][t];
        }
    }
    (sim, log_w.exp())
}

impl Search {
    /// 第一层：n_det 个公平 determinization × 每个候选动作。
    pub fn new(game: &Game, p: u8, legal: &[Action], n_det: usize, ff_cap: usize,
               seed: u64, ft: &FanTable) -> Self {
        Self::new_fair(game, p, legal, n_det, ff_cap, seed, ft, true, 1.0, false)
    }

    /// `fair=false` 时不洗牌（用于与 Python 非公平模式做逐位对拍）。
    /// `lam`：信念混合系数（1.0=均匀现状）；`reweight`：是否按 IS 权重聚合。
    pub fn new_fair(game: &Game, p: u8, legal: &[Action], n_det: usize, ff_cap: usize,
                    seed: u64, ft: &FanTable, fair: bool, lam: f64, reweight: bool) -> Self {
        let mut rng = Rng::new(seed);
        let mut ws: Vec<f64> = Vec::with_capacity(n_det);
        let seeds: Vec<(Game, Vec<Action>, usize, Vec<Action>)> = (0..n_det)
            .map(|d| {
                let sim = if !fair {
                    ws.push(1.0);
                    game.clone()
                } else if lam >= 1.0 {
                    ws.push(1.0);
                    determinize(game, p as usize, &mut rng)
                } else {
                    let (s, w) = determinize_lw(game, p as usize, &mut rng, lam);
                    ws.push(if reweight { w } else { 1.0 });
                    s
                };
                (sim, legal.to_vec(), d, Vec::new())
            })
            .collect();
        Self::build_w(seeds, p, ff_cap, ft, ws)
    }

    /// 后续层：把多个叶子的动作合并到一个 Search（跨叶子批量）。
    /// 每个 seed = (叶子局面, 候选动作, 第一层 origin 编号, 到达该叶子的动作前缀)
    pub fn from_seeds(seeds: Vec<(Game, Vec<Action>, usize, Vec<Action>)>, p: u8,
                      ff_cap: usize, ft: &FanTable) -> Self {
        let ws = vec![1.0; seeds.len()];
        Self::build_w(seeds, p, ff_cap, ft, ws)
    }

    fn build_w(seeds: Vec<(Game, Vec<Action>, usize, Vec<Action>)>, p: u8,
               ff_cap: usize, ft: &FanTable, weights: Vec<f64>) -> Self {
        let mut tasks = Vec::new();
        let mut groups = Vec::new();
        let mut origins = Vec::with_capacity(seeds.len());
        for (gi, (sim, legal, root, prefix)) in seeds.into_iter().enumerate() {
            for &a in &legal {
                let mut g2 = sim.clone();
                let over = g2.step(a, ft);
                let done = over || g2.phase == PHASE_OVER;
                let mut tr = prefix.clone();
                tr.push(a);
                tasks.push(Task { game: g2, q: 0.0, done, ff: 0, trace: tr, root });
                groups.push(gi);
            }
            origins.push(sim);
        }
        let mut s = Search {
            p,
            tasks,
            groups,
            origins,
            weights,
            ff_cap,
            pending: Vec::new(),
            unsettled: Vec::new(),
            ft: ft.clone(),
        };
        s.classify();
        s
    }

    fn build(seeds: Vec<(Game, Vec<Action>, usize, Vec<Action>)>, p: u8,
             ff_cap: usize, ft: &FanTable) -> Self {
        let ws = vec![1.0; seeds.len()];
        Self::build_w(seeds, p, ff_cap, ft, ws)
    }

    fn classify(&mut self) {
        self.pending.clear();
        self.unsettled.clear();
        for i in 0..self.tasks.len() {
            if self.tasks[i].done {
                if self.tasks[i].q == 0.0 && self.tasks[i].game.phase == PHASE_OVER {
                    self.unsettled.push(i);
                }
                continue;
            }
            let g = &self.tasks[i].game;
            if g.phase == PHASE_OVER {
                self.tasks[i].done = true;
                self.unsettled.push(i);
                continue;
            }
            let who = g.actor();
            let head = encode::head_of(g, who, &self.ft);
            // Python 语义（`_expand_joint` 的快进循环）：**先 step 一步策略动作，
            // 再判断是否回到本座位**。所以「刚打完碰/杠、仍是自己回合」这种局面
            // 不会被就地估值 —— 模型会先自己再打一手，之后回到本座才取 V。
            // 反过来，第 `ff_cap` 轮之内回到本座都算叶子（Python 的 range 是
            // 0..ff_cap，第 k 轮 step 后 ff=k+1）。
            let is_leaf = who == self.p
                && self.tasks[i].ff >= 1
                && self.tasks[i].ff <= self.ff_cap;
            if is_leaf {
                self.pending.push((i, KIND_VALUE, head));
            } else {
                self.pending.push((i, KIND_POLICY, head));
            }
        }
    }

    #[inline]
    pub fn needs_net(&self) -> bool {
        !self.pending.is_empty()
    }

    #[inline]
    pub fn has_unsettled(&self) -> bool {
        !self.unsettled.is_empty()
    }

    #[inline]
    pub fn finished(&self) -> bool {
        self.pending.is_empty() && self.unsettled.is_empty()
    }

    pub fn take_batch(&self) -> Batch {
        let n = self.pending.len();
        let mut b = Batch {
            obs: Vec::with_capacity(n),
            kind: Vec::with_capacity(n),
            head: Vec::with_capacity(n),
            task_index: Vec::with_capacity(n),
        };
        for &(i, k, h) in &self.pending {
            let g = &self.tasks[i].game;
            let who = if k == KIND_VALUE { self.p } else { g.actor() };
            b.obs.push(encode::encode_v1(g, who, &self.ft));
            b.kind.push(k);
            b.head.push(h);
            b.task_index.push(i);
        }
        b
    }

    pub fn feed(&mut self, actions: &[u8], values: &[f32]) {
        let pend = std::mem::take(&mut self.pending);
        for (j, &(i, k, head)) in pend.iter().enumerate() {
            if k == KIND_VALUE {
                self.tasks[i].q = values[j] as f64;
                self.tasks[i].done = true;
                continue;
            }
            let a = encode::index_to_action(&self.tasks[i].game, head, actions[j] as usize);
            self.tasks[i].trace.push(a);
            let over = self.tasks[i].game.step(a, &self.ft);
            self.tasks[i].ff += 1;
            if over || self.tasks[i].game.phase == PHASE_OVER {
                self.tasks[i].done = true;
            }
        }
        self.classify();
    }

    pub fn unsettled_tasks(&self) -> &[usize] {
        &self.unsettled
    }

    pub fn settle(&mut self, i: usize, delta: f64) {
        self.tasks[i].q = delta;
        self.unsettled.retain(|&x| x != i);
    }

    #[inline]
    pub fn group_of(&self, i: usize) -> usize {
        self.groups[i]
    }

    pub fn origin(&self, gi: usize) -> &Game {
        &self.origins[gi]
    }

    pub fn root_of(&self, i: usize) -> usize {
        self.tasks[i].root
    }

    pub fn trace_of(&self, i: usize) -> &[Action] {
        &self.tasks[i].trace
    }

    /// 叶子局面（已回到 p 决策点、拿到 V 的那条）；终局 / 超 cap 返回 None。
    pub fn leaf(&self, i: usize) -> Option<&Game> {
        let t = &self.tasks[i];
        if t.done && t.game.phase != PHASE_OVER && t.game.actor() == self.p {
            Some(&t.game)
        } else {
            None
        }
    }

    /// 每个原点内第 `pos` 个动作的 Q，跨原点**加权**平均（IS 权重，均匀时退化为等权）。
    /// 第一层专用：每个 origin 都是同一串 legal，位置即动作。
    pub fn action_q(&self) -> Vec<f64> {
        let mut acc: Vec<f64> = Vec::new();
        let mut wsum: Vec<f64> = Vec::new();
        let mut seen = vec![0usize; self.origins.len()];
        for (i, &gi) in self.groups.iter().enumerate() {
            let pos = seen[gi];
            seen[gi] += 1;
            if pos >= acc.len() {
                acc.resize(pos + 1, 0.0);
                wsum.resize(pos + 1, 0.0);
            }
            let w = self.weights.get(gi).copied().unwrap_or(1.0);
            acc[pos] += w * self.tasks[i].q;
            wsum[pos] += w;
        }
        acc.iter()
            .zip(wsum.iter())
            .map(|(&a, &cw)| if cw > 0.0 { a / cw } else { 0.0 })
            .collect()
    }

    /// 某个原点内各动作的 Q。
    pub fn group_vals(&self, gi: usize) -> Vec<f64> {
        self.groups
            .iter()
            .enumerate()
            .filter(|(_, &g)| g == gi)
            .map(|(i, _)| self.tasks[i].q)
            .collect()
    }

    /// 某个原点内 Q 最高的那个 task。
    pub fn group_argmax(&self, gi: usize) -> Option<usize> {
        let mut best: Option<(usize, f64)> = None;
        for (i, &g) in self.groups.iter().enumerate() {
            if g != gi {
                continue;
            }
            let q = self.tasks[i].q;
            if best.map_or(true, |(_, bq)| q > bq) {
                best = Some((i, q));
            }
        }
        best.map(|(i, _)| i)
    }

    /// 某个原点内的第 `pos` 个 task。
    pub fn group_task(&self, gi: usize, pos: usize) -> Option<usize> {
        self.groups
            .iter()
            .enumerate()
            .filter(|(_, &g)| g == gi)
            .map(|(i, _)| i)
            .nth(pos)
    }

    pub fn n_origins(&self) -> usize {
        self.origins.len()
    }
}
