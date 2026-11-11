//! 四档强度矩阵：**用离线内核本体**跑，而不是 Python 侧的同名策略。
//!
//! 为什么要单独一个对战场
//! --------------------
//! `tools/tier_matrix.py` 测的是 Python 的 `HeuristicBot`/`NetAgent`；离线版
//! 实际执行的是 `ting.rs` + tract。两边公式是照抄的，但「照抄」不等于「一致」，
//! 而用户看到的差距必须来自**手机里真正跑的那份代码**。所以这里直接：
//!   * 用 `Session` 同款的发牌（`Rng::new(seed).shuffle` + `dealer = seed % 4`）；
//!   * 四家全是 AI，按档位分派（速听走 `ting`，其余按 stem 取 tract 网）；
//!   * 结算走 `settle::settle_win` / `settle_huangzhuang`（已与 Python 逐位对拍）。
//!
//! 协议与 `eval_pairs.py` 一致：每 seed 让候选档轮换坐 0/1/2/3 号位，
//! 对手三家同档；统计量取 `cand_delta − 其余三家均值`，跨 seed 求均值与 95% CI。
//!
//! 用法：`tier_arena --seeds 400 [--out runs/tiers/rust_matrix.json]`

use std::collections::BTreeMap;
use std::time::Instant;

use dushan_engine::actions::PHASE_OVER;
use dushan_engine::fan::FanTable;
use dushan_engine::rng::Rng;
use dushan_engine::settle;
use dushan_engine::state::{Config, Game};
use dushan_engine::tiles;
use offline_core::assets;
use offline_core::policy::Net;
use offline_core::session::{canon_stem, Nets};
use offline_core::ting;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Tier {
    Adept,
    Expert,
    Master,
    Legend,
}

impl Tier {
    fn name(self) -> &'static str {
        match self {
            Tier::Adept => "adept",
            Tier::Expert => "expert",
            Tier::Master => "master",
            Tier::Legend => "legend",
        }
    }
    fn stem(self) -> &'static str {
        match self {
            Tier::Adept => "ting",
            Tier::Expert => "expert_s7",
            Tier::Master => "dushan_master",
            Tier::Legend => "champ_slim_s",
        }
    }
    fn parse(s: &str) -> Option<Tier> {
        Some(match s {
            "adept" => Tier::Adept,
            "expert" => Tier::Expert,
            "master" => Tier::Master,
            "legend" => Tier::Legend,
            _ => return None,
        })
    }
}

fn load_nets() -> Nets {
    let mut map = Vec::new();
    for (stem, bytes, enc) in [
        ("dushan_master", assets::MASTER_ONNX, 1u8),
        ("champ_slim_s", assets::SLIM_S_ONNX, 3),
        ("expert_s7", assets::EXPERT_S7_ONNX, 3),
    ] {
        match Net::from_bytes(bytes, enc) {
            Ok(n) => map.push((stem, n)),
            Err(e) => eprintln!("[arena] {stem} 加载失败: {e}"),
        }
    }
    Nets { map }
}

/// 一局：`cand` 坐 `cand_seat`，其余三家全 `opp`。返回四家分差。
fn play_one(nets: &Nets, ft: &FanTable, seed: u64, cand: Tier, opp: Tier, cand_seat: usize)
    -> [f64; 4] {
    let mut wall = tiles::full_deck();
    Rng::new(seed).shuffle(&mut wall);
    let dealer = (seed % 4) as u8;
    let mut g = Game::new_with_wall(wall, dealer, Config::default(), ft);

    while g.phase != PHASE_OVER {
        let seat = g.actor() as usize;
        let tier = if seat == cand_seat { cand } else { opp };
        let a = if tier == Tier::Adept {
            ting::choose(&g, seat as u8, ft)
        } else {
            match nets.get(canon_stem(tier.stem())) {
                Some(n) => n.choose(&g, seat as u8, ft)
                    .unwrap_or_else(|| offline_core::policy::heuristic_choose(&g, seat as u8, ft)),
                None => offline_core::policy::heuristic_choose(&g, seat as u8, ft),
            }
        };
        g.step(a, ft);
    }

    let res = match g.outcome {
        Some(o) if o.n_winners > 0 => settle::settle_win(&mut g, ft),
        _ => settle::settle_huangzhuang(&g, ft),
    };
    let mut out = [0.0f64; 4];
    if let Some(arr) = res.get("deltas").and_then(|x| x.as_array()) {
        for (i, v) in arr.iter().take(4).enumerate() {
            out[i] = v.as_f64().unwrap_or(0.0);
        }
    }
    out
}

struct Stat {
    n: usize,
    mean: f64,
    se: f64,
    lo: f64,
    hi: f64,
    pos: usize,
    neg: usize,
}

fn stats(vals: &[f64]) -> Stat {
    let n = vals.len();
    let mean = vals.iter().sum::<f64>() / n as f64;
    let var = if n > 1 {
        vals.iter().map(|x| (x - mean) * (x - mean)).sum::<f64>() / (n - 1) as f64
    } else {
        0.0
    };
    let se = (var / n as f64).sqrt();
    Stat {
        n,
        mean,
        se,
        lo: mean - 1.96 * se,
        hi: mean + 1.96 * se,
        pos: vals.iter().filter(|x| **x > 0.0).count(),
        neg: vals.iter().filter(|x| **x < 0.0).count(),
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let seeds: u64 = args.iter().position(|a| a == "--seeds")
        .and_then(|i| args.get(i + 1)).and_then(|v| v.parse().ok()).unwrap_or(300);
    let seed0: u64 = args.iter().position(|a| a == "--seed0")
        .and_then(|i| args.get(i + 1)).and_then(|v| v.parse().ok()).unwrap_or(7_300_000);
    let out_path = args.iter().position(|a| a == "--out")
        .and_then(|i| args.get(i + 1)).cloned()
        .unwrap_or_else(|| "runs/tiers/rust_matrix.json".to_string());

    let ft = FanTable::build();
    let nets = load_nets();
    println!("[arena] nets={} seeds={seeds} seed0={seed0}",
             nets.map.iter().map(|(s, _)| *s).collect::<Vec<_>>().join(","));

    let tiers = [Tier::Adept, Tier::Expert, Tier::Master, Tier::Legend];
    let mut pairs: Vec<(Tier, Tier)> = Vec::new();
    for i in 0..4 {
        for j in (i + 1)..4 {
            pairs.push((tiers[i], tiers[j]));
        }
    }

    let mut rows: Vec<serde_json::Value> = Vec::new();
    let t_all = Instant::now();
    for (a, b) in pairs {
        let t0 = Instant::now();
        let mut per_seed: Vec<f64> = Vec::with_capacity(seeds as usize);
        for k in 0..seeds {
            let seed = seed0 + k;
            // 与 `eval_pairs.py --seats 0,1,2,3` 同构：**同一副牌**让候选轮坐
            // 四座，先把「这把牌好不好打」这个共同因素在 seed 内平均掉，再跨
            // seed 统计。每座单独发牌会让牌墙方差留在残差里，CI 会虚高。
            let mut acc = 0.0;
            for cand_seat in 0..4usize {
                let d = play_one(&nets, &ft, seed, a, b, cand_seat);
                let others = (0..4).filter(|&i| i != cand_seat)
                    .map(|i| d[i]).sum::<f64>() / 3.0;
                acc += d[cand_seat] - others;
            }
            per_seed.push(acc / 4.0);
        }
        let st = stats(&per_seed);
        println!("  {:>6} vs 3×{:<6} {:+.3}±{:.3} CI[{:+.3},{:+.3}]  {}:{}  ({:.0}s)",
                 a.name(), b.name(), st.mean, st.se, st.lo, st.hi, st.pos, st.neg,
                 t0.elapsed().as_secs_f64());
        rows.push(serde_json::json!({
            "a": a.name(), "b": b.name(), "n": st.n,
            "mean": st.mean, "se": st.se, "lo": st.lo, "hi": st.hi,
            "pos": st.pos, "neg": st.neg,
        }));
    }

    let mut dump = BTreeMap::new();
    dump.insert("seeds".to_string(), serde_json::json!(seeds));
    dump.insert("seed0".to_string(), serde_json::json!(seed0));
    dump.insert("engine".to_string(), serde_json::json!("offline-core (ting.rs + tract)"));
    dump.insert("pairs".to_string(), serde_json::json!(rows));
    if let Some(dir) = std::path::Path::new(&out_path).parent() {
        let _ = std::fs::create_dir_all(dir);
    }
    std::fs::write(&out_path,
                   serde_json::to_string_pretty(&dump).unwrap() + "\n").unwrap();
    println!("[arena] 总耗时 {:.1}s → {out_path}", t_all.elapsed().as_secs_f64());
}
