//! 多层搜索编排（ply 1/2/3）—— 严格对齐 `SearchAgent._search_fair_batched` 的
//! `ply >= 2` 分支。
//!
//! 两个容易搞错的地方（都已按 Python 版对齐）：
//! 1. **top-k 是「每个 determinization 各取各的」**，不是先平均再取 top-k
//!    —— `max` 和 `mean` 不可交换，顺序错了会让 Q 与基线反相关。
//! 2. **终局重放要用完整动作链**：第二/三层的起点是上层叶子，重放必须从第一层的
//!    determinization 起点开始，因此每个 task 都携带「根 origin + 完整 trace」。
//!
//! 与 Python 版的性能差异：Python 对每个叶子单独调一次 `_expand`（一次前向），
//! 这里把同层所有叶子的动作合并成一个 Search，一层只前向一次。

use crate::actions::*;
use crate::fan::FanTable;
use crate::search::Search;
use crate::state::Game;

pub struct DeepSearch {
    pub p: u8,
    pub stages: Vec<Search>,
    pub cur: usize,
    pub ply: u8,
    pub topk: usize,
    pub ff_cap: usize,
    /// 第二/三层聚合方式（2026-11-03，实验 A）：
    /// 0 = max（现状）：每个 determinization 内对 topk 分支取 max——
    ///     这是**策略融合**点（搜索在每个假想世界里都按"知道后续"优化）；
    /// 1 = mean（融合无关）：V 与 topk 分支取均值——同一预算、同一分支，
    ///     只把「逐世界优化」换成「逐世界平均」，用于量化融合的净效应。
    pub agg: u8,
    /// 观测编码版本（1/2/3）—— 方案 D，透传到每层 Search
    pub enc: u8,
    pub ft: FanTable,
    pub done: bool,
    /// [determinization][动作] 的当前 Q
    q: Vec<Vec<f64>>,
    /// [d][i] → 第二层 origin 编号
    branch1: Vec<Vec<Option<usize>>>,
    /// [d][i] → 第三层 origin 编号
    branch2: Vec<Vec<Option<usize>>>,
    /// 每个 determinization 的 IS 权重（B1；均匀路径恒 1.0）
    dw: Vec<f64>,
    n_det: usize,
    n_actions: usize,
}

impl DeepSearch {
    #[allow(clippy::too_many_arguments)]
    pub fn new(game: &Game, p: u8, legal: &[Action], n_det: usize, ff_cap: usize,
               topk: usize, ply: u8, seed: u64, ft: &FanTable) -> Self {
        Self::new_fair(game, p, legal, n_det, ff_cap, topk, ply, seed, ft, true, 0,
                       1.0, false)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn new_fair(game: &Game, p: u8, legal: &[Action], n_det: usize, ff_cap: usize,
                    topk: usize, ply: u8, seed: u64, ft: &FanTable, fair: bool,
                    agg: u8, lam: f64, reweight: bool) -> Self {
        Self::new_fair_enc(game, p, legal, n_det, ff_cap, topk, ply, seed, ft, fair,
                           agg, lam, reweight, crate::search::ENC_V1)
    }

    /// 同 `new_fair`，额外指定观测编码版本（方案 D）；`enc` 会透传到每一层 `Search`。
    #[allow(clippy::too_many_arguments)]
    pub fn new_fair_enc(game: &Game, p: u8, legal: &[Action], n_det: usize, ff_cap: usize,
                        topk: usize, ply: u8, seed: u64, ft: &FanTable, fair: bool,
                        agg: u8, lam: f64, reweight: bool, enc: u8) -> Self {
        let stage0 = Search::new_fair_enc(game, p, legal, n_det, ff_cap, seed, ft, fair,
                                          lam, reweight, enc);
        let n = legal.len();
        let dw = stage0.weights.clone();
        DeepSearch {
            p,
            stages: vec![stage0],
            cur: 0,
            ply,
            topk,
            ff_cap,
            agg,
            enc,
            ft: ft.clone(),
            done: false,
            q: vec![vec![0.0; n]; n_det],
            branch1: vec![vec![None; n]; n_det],
            branch2: vec![vec![None; n]; n_det],
            dw,
            n_det,
            n_actions: n,
        }
    }

    #[inline]
    pub fn current(&self) -> &Search {
        &self.stages[self.cur]
    }

    #[inline]
    pub fn current_mut(&mut self) -> &mut Search {
        &mut self.stages[self.cur]
    }

    #[inline]
    pub fn finished(&self) -> bool {
        self.done
    }

    #[inline]
    pub fn stage_done(&self) -> bool {
        self.stages[self.cur].finished()
    }

    fn absorb_stage0(&mut self) {
        for d in 0..self.n_det {
            let vals = self.stages[0].group_vals(d);
            for (i, v) in vals.iter().enumerate().take(self.n_actions) {
                self.q[d][i] = *v;
            }
        }
    }

    /// 上层结果与浅层值合并（实验 A 的聚合开关）。
    /// agg=0：max（现状）；agg=1：V 与各分支取均值（融合无关口径）。
    fn merge_vals(&self, vals: &[f64], base: f64) -> f64 {
        if self.agg == 1 {
            let s: f64 = vals.iter().sum();
            (s + base) / (vals.len() + 1) as f64
        } else {
            vals.iter()
                .cloned()
                .fold(f64::NEG_INFINITY, f64::max)
                .max(base)
        }
    }

    fn topk_of(&self, d: usize) -> Vec<usize> {
        let mut idx: Vec<usize> = (0..self.n_actions).collect();
        idx.sort_by(|&a, &b| {
            self.q[d][b]
                .partial_cmp(&self.q[d][a])
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        idx.truncate(self.topk.min(idx.len()));
        idx
    }

    /// 推进到下一层；返回 false 表示搜索结束。
    pub fn promote(&mut self) -> bool {
        if self.done {
            return false;
        }
        match self.cur {
            0 => {
                self.absorb_stage0();
                if self.ply < 2 || self.n_actions <= 1 {
                    self.done = true;
                    return false;
                }
                let mut seeds: Vec<(Game, Vec<Action>, usize, Vec<Action>)> = Vec::new();
                for d in 0..self.n_det {
                    for i in self.topk_of(d) {
                        let Some(t) = self.stages[0].group_task(d, i) else { continue };
                        let Some(g) = self.stages[0].leaf(t) else { continue };
                        let acts = g.legal_actions(&self.ft);
                        if acts.len() <= 1 {
                            continue;
                        }
                        let prefix = self.stages[0].trace_of(t).to_vec();
                        self.branch1[d][i] = Some(seeds.len());
                        seeds.push((g.clone(), acts, d, prefix));
                    }
                }
                if seeds.is_empty() {
                    self.done = true;
                    return false;
                }
                self.stages
                    .push(Search::from_seeds_enc(seeds, self.p, self.ff_cap, &self.ft, self.enc));
                self.cur = 1;
                true
            }
            1 => {
                let mut best_task: Vec<Vec<Option<usize>>> =
                    vec![vec![None; self.n_actions]; self.n_det];
                for d in 0..self.n_det {
                    for i in 0..self.n_actions {
                        let Some(gi) = self.branch1[d][i] else { continue };
                        let vals = self.stages[1].group_vals(gi);
                        if vals.is_empty() {
                            continue;
                        }
                        let v = self.merge_vals(&vals, self.q[d][i]);
                        self.q[d][i] = v;
                        best_task[d][i] = self.stages[1].group_argmax(gi);
                    }
                }
                if self.ply < 3 {
                    self.done = true;
                    return false;
                }
                let mut seeds: Vec<(Game, Vec<Action>, usize, Vec<Action>)> = Vec::new();
                for d in 0..self.n_det {
                    for i in 0..self.n_actions {
                        let Some(ti) = best_task[d][i] else { continue };
                        let Some(g) = self.stages[1].leaf(ti) else { continue };
                        let acts = g.legal_actions(&self.ft);
                        if acts.len() <= 1 {
                            continue;
                        }
                        let prefix = self.stages[1].trace_of(ti).to_vec();
                        self.branch2[d][i] = Some(seeds.len());
                        seeds.push((g.clone(), acts, d, prefix));
                    }
                }
                if seeds.is_empty() {
                    self.done = true;
                    return false;
                }
                self.stages
                    .push(Search::from_seeds_enc(seeds, self.p, self.ff_cap, &self.ft, self.enc));
                self.cur = 2;
                true
            }
            2 => {
                for d in 0..self.n_det {
                    for i in 0..self.n_actions {
                        let Some(gi) = self.branch2[d][i] else { continue };
                        let vals = self.stages[2].group_vals(gi);
                        if vals.is_empty() {
                            continue;
                        }
                        let v = self.merge_vals(&vals, self.q[d][i]);
                        self.q[d][i] = v;
                    }
                }
                self.done = true;
                false
            }
            _ => {
                self.done = true;
                false
            }
        }
    }

    /// 最终 Q：先按 determinization 求「max(1-ply, 2-ply, 3-ply)」，
    /// 再跨 d **加权**平均（B1 的 IS 权重；均匀路径 = 等权，行为不变）。
    pub fn result(&self) -> Vec<f64> {
        let mut out = vec![0.0; self.n_actions];
        let mut wsum = 0.0f64;
        for (d, row) in self.q.iter().enumerate() {
            let w = self.dw.get(d).copied().unwrap_or(1.0);
            wsum += w;
            for (i, v) in row.iter().enumerate() {
                out[i] += w * v;
            }
        }
        if wsum > 0.0 {
            for v in out.iter_mut() {
                *v /= wsum;
            }
        }
        out
    }

    /// 逐 determinization 的 Q（诊断用）。
    pub fn per_det(&self) -> Vec<Vec<f64>> {
        self.q.clone()
    }

    /// **ply=1 专用**：逐 determinization × 动作的终局诊断
    /// `(terminal, win, delta)`。
    ///
    /// - `terminal`：该假想世界在快进（≤ff_cap 轮）内直通终局；
    /// - `win`：本座位是胡家之一（对应 Python `_note_terminal` 的
    ///   `winner == p` 口径，含一炮多响的多胡家）；
    /// - `delta`：本座位净分差 —— 终局线由 Python 重放精确结算回填，
    ///   残局线是叶子 V（此时 terminal=false，调用方不应采信 delta）。
    ///
    /// ply≥2 时第二/三层只回填 `self.q`，不回写 stage0 的 task，因此此表
    /// 只反映第一层口径；128 路 hint 用 ply=1，正好成立。
    pub fn det_diag(&self) -> Vec<Vec<(bool, bool, f64)>> {
        let s0 = &self.stages[0];
        let mut out = vec![vec![(false, false, 0.0f64); self.n_actions]; self.n_det];
        for d in 0..self.n_det {
            for i in 0..self.n_actions {
                let Some(t) = s0.group_task(d, i) else { continue };
                let task = &s0.tasks[t];
                let term = task.game.phase == PHASE_OVER;
                let win = term
                    && task
                        .game
                        .outcome
                        .map_or(false, |o| {
                            o.winners[..o.n_winners as usize].contains(&self.p)
                        });
                out[d][i] = (term, win, task.q);
            }
        }
        out
    }
}
