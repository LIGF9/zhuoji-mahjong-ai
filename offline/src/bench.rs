//! 设备自检：冷启动时把两张网的真实单次决策耗时打进日志（Android 走 logcat）。
//!
//! 为什么要它：宿主机数字不能直接外推到手机（tract 在 AArch64 上用 NEON
//! 矩阵乘，主频 ~2.0-3.2 GHz、内存带宽只有笔电的 1/3）。打包后没有终端，
//! 只能让 App 自己跑一次并把结论写进 `logcat`，才能回答「手机上开深算卡不卡」。
//!
//! 输出形如：
//! ```text
//! [offline] bench greedy slim_s=1.4ms master=1.6ms
//! [offline] bench n32 slim_s=180ms n128 slim_s=690ms K=4
//! ```
//! 时序：在服务起来之后、玩家还没点「开始」之前跑，独占 CPU，读数才干净。

use std::time::Instant;

use dushan_engine::fan::FanTable;

use crate::policy::mt_threads;
use crate::search::{deep_q, Cfg};
use crate::session::{Nets, Session};

/// 走一步到「我」的第一个决策点，然后分别计时贪心 / 32 路 / 128 路。
pub fn log_latency(nets: &Nets) {
    let ft = FanTable::build();
    let names = ["我".to_string(), "下家".into(), "对家".into(), "上家".into()];
    let strats: [String; 3] = std::array::from_fn(|_| "model:champ_slim_s".to_string());
    let mut s = Session::new(ft, 20_261_111, 0, &names, &strats);
    s.advance(nets, 400);
    if s.game.phase == dushan_engine::actions::PHASE_OVER || s.game.actor() != 0 {
        println!("[offline] bench 跳过：未走到我的决策点");
        return;
    }
    let legal = s.game.legal_actions(&s.ft);
    let k = mt_threads();
    println!("[offline] bench K={k} legal={}", legal.len());

    // 速听是纯规则，单独计时（它是 adept 档，手机上应当 <1ms）。
    {
        let _ = crate::ting::choose(&s.game, 0, &s.ft);
        let t0 = Instant::now();
        for _ in 0..50 {
            let _ = crate::ting::choose(&s.game, 0, &s.ft);
        }
        println!("[offline] bench greedy ting={:.2}ms", t0.elapsed().as_secs_f64() * 1000.0 / 50.0);
    }
    for label in ["champ_slim_s", "expert_s7", "dushan_master"] {
        let Some(net) = nets.get(label) else { continue };
        // 贪心：50 次取均值（单次太短，计时噪声大）。
        let t0 = Instant::now();
        for _ in 0..50 {
            let _ = net.choose(&s.game, 0, &s.ft);
        }
        let per = t0.elapsed().as_secs_f64() * 1000.0 / 50.0;
        let mut line = format!("[offline] bench greedy {label}={per:.2}ms");
        // 搜索：先预热一次（tract 首次遇批形状要建计划，不计入）。
        for det in [32usize, 128] {
            let cfg = Cfg { n_det: det, ff_cap: 24, topk: 3, ply: 1 };
            let _ = deep_q(net, &s.game, 0, &legal, &cfg, 42, &s.ft);
            let t0 = Instant::now();
            match deep_q(net, &s.game, 0, &legal, &cfg, 42, &s.ft) {
                Ok(_) => {
                    let ms = t0.elapsed().as_secs_f64() * 1000.0;
                    line.push_str(&format!(" n{det}={ms:.0}ms"));
                }
                Err(e) => line.push_str(&format!(" n{det}=err({e})")),
            }
        }
        println!("{line}");
    }
}
