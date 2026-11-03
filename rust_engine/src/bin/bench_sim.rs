//! 纯模拟基准：模拟一次 n_det 搜索的「状态推进」开销（不含网络前向）。
//!
//! 流程与 `SearchAgent._search_fair_batched` 一致：
//!   n_det 次 determinization × 每个候选动作 → step → 快进若干步
//! 用固定伪随机动作代替网络的贪心策略（只测引擎吞吐）。

use std::time::Instant;

use dushan_engine::actions::*;
use dushan_engine::fan::FanTable;
use dushan_engine::state::{Config, Game};
use dushan_engine::tiles::{full_deck, NUM_TILES};

struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }
    fn below(&mut self, n: usize) -> usize {
        (self.next() % n as u64) as usize
    }
    fn shuffle<T>(&mut self, v: &mut [T]) {
        for i in (1..v.len()).rev() {
            let j = self.below(i + 1);
            v.swap(i, j);
        }
    }
}

fn hand_sum(g: &Game, q: usize) -> usize {
    g.hands[q].iter().map(|&x| x as usize).sum()
}

/// 公平 determinization：重排「别家手牌 + 剩余牌墙」，本家与已打出的牌不动。
fn determinize(g: &Game, p: usize, rng: &mut Rng) -> Game {
    let mut sim = g.clone();
    let totals: Vec<usize> = (0..4)
        .map(|q| if q == p { 0 } else { hand_sum(&sim, q) })
        .collect();
    let mut pool: Vec<u8> = Vec::with_capacity(70);
    for q in 0..4 {
        if q == p {
            continue;
        }
        for t in 0..NUM_TILES as u8 {
            for _ in 0..sim.hands[q][t as usize] {
                pool.push(t);
            }
            sim.hands[q][t as usize] = 0;
        }
    }
    let wall_n = sim.wall.len() - sim.wall_pos;
    pool.extend_from_slice(&sim.wall[sim.wall_pos..]);
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
    sim
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let ndet: usize = args.get(1).and_then(|x| x.parse().ok()).unwrap_or(32);
    let rounds: usize = args.get(2).and_then(|x| x.parse().ok()).unwrap_or(20);
    let ff: usize = args.get(3).and_then(|x| x.parse().ok()).unwrap_or(4);

    let ft = FanTable::build();
    let mut rng = Rng(0x9E37_79B9_7F4A_7C15);
    let mut clones = 0usize;
    let mut steps = 0usize;
    let mut done_rounds = 0usize;
    let mut legal_sum = 0usize;

    let t0 = Instant::now();
    for r in 0..rounds {
        let mut wall = full_deck();
        rng.shuffle(&mut wall);
        let mut g = Game::new_with_wall(wall, (r % 4) as u8, Config::default(), &ft);
        for _ in 0..(10 + rng.below(35)) {
            if g.phase == PHASE_OVER {
                break;
            }
            let acts = g.legal_actions(&ft);
            let a = acts[rng.below(acts.len())];
            g.step(a, &ft);
        }
        if g.phase == PHASE_OVER {
            continue;
        }
        done_rounds += 1;

        let p = g.actor() as usize;
        let legal = g.legal_actions(&ft);
        legal_sum += legal.len();

        for _ in 0..ndet {
            let sim = determinize(&g, p, &mut rng);
            clones += 1;
            for &a in legal.iter() {
                let mut g2 = sim.clone();
                clones += 1;
                steps += 1;
                if g2.step(a, &ft) {
                    continue;
                }
                for _ in 0..ff {
                    if g2.phase == PHASE_OVER || g2.actor() as usize == p {
                        break;
                    }
                    let acts = g2.legal_actions(&ft);
                    if acts.is_empty() {
                        break;
                    }
                    let b = acts[rng.below(acts.len())];
                    steps += 1;
                    if g2.step(b, &ft) {
                        break;
                    }
                }
            }
        }
    }
    let dt = t0.elapsed().as_secs_f64();
    println!(
        "n_det={} rounds={} ff={} | clones={} steps={} legal_avg={:.1} | \
         {:.3}s total, {:.2} ms/决策点 | {:.0} steps/s",
        ndet,
        done_rounds,
        ff,
        clones,
        steps,
        legal_sum as f64 / done_rounds.max(1) as f64,
        dt,
        dt * 1000.0 / done_rounds.max(1) as f64,
        steps as f64 / dt
    );
}
