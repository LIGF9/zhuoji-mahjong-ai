//! 设备端 PIMC 搜索耗时基准（宿主机 tract CPU）。
//!
//! 目的：在打包前拿到「N32 / N128 × 两张网」的真实单次决策耗时，
//! 判断手机上默认 `search128:master` 是否可接受（前端默认档写死为它）。

use std::time::Instant;

use dushan_engine::fan::FanTable;
use offline_core::assets;
use offline_core::policy::Net;
use offline_core::search::{deep_q, Cfg};
use offline_core::session::{Nets, Session};

fn main() {
    let ft = FanTable::build();
    let slim = Net::from_bytes(assets::SLIM_S_ONNX, 3).expect("slim 加载失败");
    let master = Net::from_bytes(assets::MASTER_ONNX, 1).expect("master 加载失败");
    let nets = Nets { slim: Some(slim), master: Some(master) };

    // 三个贪心对手，走到「我」的第一个决策点。
    let names = ["我".to_string(), "下家".into(), "对家".into(), "上家".into()];
    let mut s = Session::new(ft, 2026_1111, 0, &names, &[3, 3, 3], &[0, 0, 0]);
    s.advance(&nets, 400);

    let legal = s.game.legal_actions(&s.ft);
    println!("决策点：legal={} 手牌={} 牌墙={}",
             legal.len(),
             s.game.hands[0].iter().filter(|&&x| x > 0).count(),
             s.game.wall_left());

    let cases: [(&str, u8, usize); 4] = [
        ("master  N32", 1, 32),
        ("master  N128", 1, 128),
        ("slim_s  N32", 3, 32),
        ("slim_s  N128", 3, 128),
    ];
    for (label, enc, det) in cases {
        let net = nets.for_enc(enc).expect("网缺失");
        let cfg = Cfg { n_det: det, ff_cap: 24, topk: 3, ply: 1 };
        let t0 = Instant::now();
        match deep_q(net, &s.game, 0, &legal, &cfg, 42, &s.ft) {
            Ok(q) => {
                let ms = t0.elapsed().as_secs_f64() * 1000.0;
                let mut best = 0usize;
                for (i, v) in q.iter().enumerate() {
                    if *v > q[best] { best = i; }
                }
                println!("{label:<14} {ms:8.1} ms  q_span=[{:.2},{:.2}]  首选={} (q={:.2})",
                         q.iter().cloned().fold(f64::INFINITY, f64::min),
                         q.iter().cloned().fold(f64::NEG_INFINITY, f64::max),
                         legal[best].label(), q[best]);
            }
            Err(e) => println!("{label:<14} 失败: {e}"),
        }
    }

    // 贪心基线耗时（单步前向），用于对比「搜索值不值」。
    let net = nets.for_enc(1).unwrap();
    let t0 = Instant::now();
    for _ in 0..50 {
        let _ = net.choose(&s.game, 0, &s.ft);
    }
    println!("贪心单步（master）: {:.2} ms/次", t0.elapsed().as_secs_f64() * 1000.0 / 50.0);

    // 前向成本结构：不同 batch 的每次耗时与每行成本。
    // 搜索每层的批大小是变化的，tract 按形状缓存执行计划，形状种类越多越吃亏。
    println!("\n=== search_batch 成本（master，全 1 mask） ===");
    let net = nets.for_enc(1).unwrap();
    let (ch, gd) = (net.channels(), net.global_dim());
    for n in [1usize, 2, 4, 8, 16, 32, 64, 128, 256] {
        let tiles = vec![0.01f32; n * ch * 27];
        let glob = vec![0.01f32; n * gd];
        let heads = vec![0u8; n];
        let masks = vec![[1u8; 28]; n];
        // 预热一次（tract 首次遇到该形状要编译计划）。
        let _ = net.search_batch(n, &tiles, &glob, &heads, &masks);
        let reps = if n <= 8 { 20 } else { 5 };
        let t0 = Instant::now();
        for _ in 0..reps {
            let _ = net.search_batch(n, &tiles, &glob, &heads, &masks);
        }
        let per_call = t0.elapsed().as_secs_f64() * 1000.0 / reps as f64;
        println!("  n={n:<4} {per_call:8.2} ms/次   {:.3} ms/行", per_call / n as f64);
    }

    // 多线程分块前向：tract 的计划是只读的，理论上可 `&plan` 并发调用。
    // 若 4 线程能拿到 ~2.5× 以上，搜索档在手机上就可用（N128 目标 < 2s）。
    println!("\n=== 多线程分块（master，n=512 拆 K 块） ===");
    let n = 512usize;
    let tiles = vec![0.01f32; n * ch * 27];
    let glob = vec![0.01f32; n * gd];
    let heads = vec![0u8; n];
    let masks = vec![[1u8; 28]; n];
    let _ = net.search_batch(n, &tiles, &glob, &heads, &masks);
    for k in [1usize, 2, 4, 6] {
        let chunk = n / k;
        let t0 = Instant::now();
        for _ in 0..3 {
            std::thread::scope(|sc| {
                let mut hs = Vec::new();
                for c in 0..k {
                    let (t, g, h, m) = (&tiles, &glob, &heads, &masks);
                    let (lo, hi) = (c * chunk, (c + 1) * chunk);
                    let net = &net;
                    hs.push(sc.spawn(move || {
                        let _ = net.search_batch(
                            chunk,
                            &t[(lo * ch * 27)..(hi * ch * 27)],
                            &g[(lo * gd)..(hi * gd)],
                            &h[lo..hi],
                            &m[lo..hi],
                        );
                    }));
                }
                for h in hs {
                    let _ = h.join();
                }
            });
        }
        let ms = t0.elapsed().as_secs_f64() * 1000.0 / 3.0;
        println!("  K={k:<2} {ms:8.2} ms/次（n={n}）");
    }

    // 确定性：并行分块必须与单线程逐位一致（切片是纯数据并行，行内计算顺序不变）。
    // 判据不靠肉眼看数：把 Q 向量按 1e-6 打印，`DUSHAN_MT=1` 与 `DUSHAN_MT=4` 两次
    // 运行必须逐字符相同 —— 这是唯一能证明「4 线程不改变棋力」的证据。
    println!("\n=== 确定性指纹（master N32，DUSHAN_MT={}） ===", offline_core::policy::mt_threads());
    let net = nets.for_enc(1).unwrap();
    let cfg = Cfg { n_det: 32, ff_cap: 24, topk: 3, ply: 1 };
    let q = deep_q(net, &s.game, 0, &legal, &cfg, 42, &s.ft).unwrap();
    let fp: Vec<String> = q.iter().map(|v| format!("{v:.6}")).collect();
    println!("FP {}", fp.join(","));
}
