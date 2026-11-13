//! N32 深算档 vs 传奇贪心档 · 配对分差评测（2026-11-13 用户口径）
//!
//! 测得的就是**打包进 App 的那份代码**：`offline_core::search::deep_q`
//! （`search32:champ_slim_s` → `n_det=32, ff_cap=24, topk=3, ply=1`）
//! 对 `offline_core::policy::Net::choose`（`model:champ_slim_s` 贪心），
//! 结算走 `dushan_engine::settle`，与 `tier_arena.rs` / 四档强度矩阵同源。
//!
//! ## 为什么不用 Python 侧评测
//! Python 版 `reports/P0-学生N32搜索红利.md` 的臂 C 是
//! 「学生 N32 vs 3× 学生贪心」，但那是 `web/dushan_server.py` 的
//! `ply=3 / n_det=32` 搜索（10.35 s/局），而桌面/在线宿主里新加的第 5 档
//! 跑的是 Rust 内核 `ply=1 / n_det=32`。两者参数不同、实现不同，不能拿旧报告
//! 回答「新档比传奇强多少」。这里按用户长期规则走 Rust 快路径：
//!   * 前向不再跨进程 JSON → 本进程 tract；
//!   * 搜索内结算不回 Python → `DeepSearch::settle_internal`；
//!   * 每个 worker 独立加载一份权重，seed 分片多线程并行。
//!
//! ## 协议（与 `reports/离线版-四档强度矩阵.md` 完全一致）
//! 每个 seed 用**同一副牌墙**让候选 N32 轮流坐 0/1/2/3 号位，其余三家
//! 全为传奇贪心；统计量 `cand_delta − 其余三家均值`，seed 内四座平均，
//! 跨 seed 求均值与 95% CI；「分/副」= 配对差 × 3/4（四家 deltas 之和为 0）。
//!
//! 用法：`n32_arena --seeds 400 [--seed0 7300000] [--threads 8]
//!                  [--out runs/tiers/n32_vs_legend.json]`
//!       `n32_arena --seeds 400 --reverse` → 反向配对（传奇 vs 3× N32）

use std::time::Instant;

use dushan_engine::actions::{Action, PHASE_OVER};
use dushan_engine::fan::FanTable;
use dushan_engine::rng::Rng;
use dushan_engine::settle;
use dushan_engine::state::{Config, Game};
use dushan_engine::tiles;
use offline_core::assets;
use offline_core::policy::{heuristic_choose, Net};
use offline_core::search::{deep_q, Cfg};

const STEM: &str = "champ_slim_s";
const CFG: Cfg = Cfg { n_det: 32, ff_cap: 24, topk: 3, ply: 1 };

/// 一局：`n32_seat` 跑 32 路 PIMC，其余三家贪心。返回四家 deltas。
fn play_one(net: &Net, ft: &FanTable, seed: u64, cand_seat: usize, reverse: bool) -> [f64; 4] {
    let mut wall = tiles::full_deck();
    Rng::new(seed).shuffle(&mut wall);
    let dealer = (seed % 4) as u8;
    let mut g = Game::new_with_wall(wall, dealer, Config::default(), ft);
    let mut seq: u64 = 0;

    while g.phase != PHASE_OVER {
        let seat = g.actor() as usize;
        // 正向：候选（cand_seat）= N32，对手三家 = 传奇贪心；
        // 反向：候选 = 传奇贪心，其余三家 = N32。
        let this_is_n32 = if reverse { seat != cand_seat } else { seat == cand_seat };
        let a = if this_is_n32 {
            let legal = g.legal_actions(ft);
            if legal.len() <= 1 {
                legal.first().copied().unwrap_or_else(Action::pass)
            } else {
                // 与 Session::decide 同款确定性种子：牌局 seed × 步序。
                let s = seed
                    ^ seq.wrapping_mul(0x9E37_79B9_7F4A_7C15)
                    ^ (g.wall_left() as u64).wrapping_mul(0x2545_F491_4F6C_DD1D);
                match deep_q(net, &g, seat as u8, &legal, &CFG, s, ft) {
                    Ok(q) if !q.is_empty() => {
                        let mut best = 0usize;
                        for (i, v) in q.iter().enumerate() {
                            if *v > q[best] { best = i; }
                        }
                        legal.get(best).copied().unwrap_or_else(Action::pass)
                    }
                    // 搜索失败回落贪心（与线上行为一致）。
                    _ => net.choose(&g, seat as u8, ft)
                        .unwrap_or_else(|| heuristic_choose(&g, seat as u8, ft)),
                }
            }
        } else {
            net.choose(&g, seat as u8, ft)
                .unwrap_or_else(|| heuristic_choose(&g, seat as u8, ft))
        };
        seq += 1;
        if seq > 400 { break; } // 与 Session::advance 上限一致，防病态长局
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

fn stats(vals: &[f64]) -> (f64, f64, f64, f64, usize, usize) {
    let n = vals.len();
    let mean = vals.iter().sum::<f64>() / n as f64;
    let var = if n > 1 {
        vals.iter().map(|x| (x - mean) * (x - mean)).sum::<f64>() / (n - 1) as f64
    } else { 0.0 };
    let se = (var / n as f64).sqrt();
    (mean, se, mean - 1.96 * se, mean + 1.96 * se,
     vals.iter().filter(|x| **x > 0.0).count(),
     vals.iter().filter(|x| **x < 0.0).count())
}

fn arg(args: &[String], key: &str) -> Option<String> {
    args.iter().position(|a| a == key).and_then(|i| args.get(i + 1)).cloned()
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let seeds: u64 = arg(&args, "--seeds").and_then(|v| v.parse().ok()).unwrap_or(300);
    let seed0: u64 = arg(&args, "--seed0").and_then(|v| v.parse().ok()).unwrap_or(7_300_000);
    let threads: usize = arg(&args, "--threads").and_then(|v| v.parse().ok())
        .unwrap_or_else(|| std::thread::available_parallelism().map(|n| n.get()).unwrap_or(4));
    let out_path = arg(&args, "--out")
        .unwrap_or_else(|| "runs/tiers/n32_vs_legend.json".to_string());
    let reverse = args.iter().any(|a| a == "--reverse");

    let label = if reverse { "传奇(贪心) vs 3× N32" } else { "N32 vs 3× 传奇(贪心)" };
    println!("[n32_arena] {label}  seeds={seeds} seed0={seed0} threads={threads}");

    let t_all = Instant::now();
    // 每个 worker 独占一份 Net（tract 执行计划含内部可变缓存，不跨线程共享）。
    let mut handles = Vec::new();
    for w in 0..threads {
        let lo = w as u64 * seeds / threads as u64;
        let hi = (w + 1) as u64 * seeds / threads as u64;
        handles.push(std::thread::spawn(move || {
            let ft = FanTable::build();
            let net = Net::from_bytes(assets::SLIM_S_ONNX, 3)
                .expect("champ_slim_s ONNX 加载失败");
            let mut out = Vec::with_capacity((hi - lo) as usize);
            for k in lo..hi {
                let seed = seed0 + k;
                let mut acc = 0.0;
                for cand_seat in 0..4usize {
                    let d = play_one(&net, &ft, seed, cand_seat, reverse);
                    let others = (0..4).filter(|&i| i != cand_seat)
                        .map(|i| d[i]).sum::<f64>() / 3.0;
                    acc += d[cand_seat] - others;
                }
                out.push((k, acc / 4.0));
            }
            eprintln!("[worker {w}] 完成 {} seeds / {:.0}s", hi - lo,
                      t_all.elapsed().as_secs_f64());
            out
        }));
    }
    let mut vals = vec![0.0f64; seeds as usize];
    for h in handles {
        for (k, v) in h.join().expect("worker panic") {
            vals[k as usize] = v;
        }
    }

    let (mean, se, lo, hi, pos, neg) = stats(&vals);
    let per_hand = mean * 0.75;
    println!("\n=== {label} ===");
    println!("配对差 n={seeds}  mean={mean:+.3}  SE={se:.3}  95%CI=[{lo:+.3},{hi:+.3}]  2×SE={:.3}",
             2.0 * se);
    println!("折合每副（×3/4）: {per_hand:+.3} 分/副   候选领先 seed: {pos} / 落后 {neg}");
    println!("总耗时 {:.1}s", t_all.elapsed().as_secs_f64());

    if let Some(dir) = std::path::Path::new(&out_path).parent() {
        let _ = std::fs::create_dir_all(dir);
    }
    let dump = serde_json::json!({
        "pair": label,
        "cand": if reverse { "model:champ_slim_s (greedy)" } else { "search32:champ_slim_s" },
        "opp": if reverse { "search32:champ_slim_s" } else { "model:champ_slim_s (greedy)" },
        "engine": "offline-core search::deep_q (n_det=32, ff_cap=24, topk=3, ply=1) + tract",
        "seeds": seeds, "seed0": seed0, "seats": 4, "hands": seeds * 4,
        "paired_diff_mean": mean, "se": se, "ci95": [lo, hi],
        "per_hand": per_hand, "pos": pos, "neg": neg,
        "seconds": t_all.elapsed().as_secs_f64(),
        "per_seed": vals,
    });
    std::fs::write(&out_path, serde_json::to_string_pretty(&dump).unwrap() + "\n").unwrap();
    println!("→ {out_path}");
}
