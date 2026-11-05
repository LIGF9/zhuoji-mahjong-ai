//! Rust 原生 ExIt 标注器 —— 替代 gen_exit.py 的 Python 胶水（rollout + 教师标注 + npz）。
//!
//! 架构（2026-11-05 性能审计后的产物）：
//! * 学生 rollout：Rust 状态机 + v1/v3 编码 + ORT-CoreML 单步前向（B=1）；
//! * 教师标注：DeepSearch（n32-ply3）+ ORT-CoreML 批前向 + Rust 内结算；
//! * 记录观测**恒为 v3**（M1 训练口径），rollout 按学生自己的编码版本决策；
//! * 输出 npz 与 gen_exit.py 完全同构：tiles/glob/head/mask/q。
//!
//! 用法：
//!   annotate --student models/dushan_master.onnx --teacher models/dushan_master.onnx \
//!       --targets 100000 --threads 4 --out runs/exit_r02 --seed0 620000
//! 对拍模式：
//!   annotate --check-encode states.jsonl out.jsonl
//!   annotate --teacher ... --check-q states.jsonl out.jsonl
//!
//! 全部长时路径都在 Rust 内（用户规则：长时任务优先 Rust 降本）。
//! Python 仅剩训练步（torch autograd，无法省）与对拍脚本。

use std::fs::File;
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::{Path, PathBuf};
use std::time::Instant;

use dushan_engine::actions::Action;
use dushan_engine::encode::{self, HEAD_DISCARD, HEAD_RESPOND, HEAD_SELF_KONG};
use dushan_engine::encode_v3::{self, NUM_CHANNELS_V3, GLOBAL_DIM_V3};
use dushan_engine::fan::FanTable;
use dushan_engine::infer::Infer;
use dushan_engine::rng::Rng;
use dushan_engine::search::{Batch, Search};
use dushan_engine::deep::DeepSearch;
use dushan_engine::snap::game_from_json;
use dushan_engine::state::{Config, Game};
use dushan_engine::tiles::full_deck;

const N_SLOTS: usize = 28;

// ---------------------------------------------------------------------------
// npy / npz 写出（零 Python；ZIP_STORED/DEFLATED 均可被 numpy 读取）
// ---------------------------------------------------------------------------

fn npy_header(descr: &str, shape: &[usize]) -> Vec<u8> {
    let shape_str = if shape.len() == 1 {
        format!("({},)", shape[0])
    } else {
        let parts: Vec<String> = shape.iter().map(|x| x.to_string()).collect();
        format!("({})", parts.join(", "))
    };
    let mut header = format!(
        "{{'descr': '{}', 'fortran_order': False, 'shape': {}, }}",
        descr, shape_str
    );
    // 10 = magic(6) + version(2) + hlen(2)；(10 + header_len) % 64 == 0，\n 收尾
    let mut pad = 64 - (10 + header.len() + 1) % 64;
    if pad == 64 {
        pad = 0;
    }
    header.push_str(&" ".repeat(pad));
    header.push('\n');
    let mut out = Vec::with_capacity(10 + header.len());
    out.extend_from_slice(b"\x93NUMPY");
    out.push(1);
    out.push(0);
    out.extend_from_slice(&(header.len() as u16).to_le_bytes());
    out.extend_from_slice(header.as_bytes());
    out
}

fn npy_f32(shape: &[usize], data: &[f32]) -> Vec<u8> {
    let mut b = npy_header("<f4", shape);
    for v in data {
        b.extend_from_slice(&v.to_le_bytes());
    }
    b
}

fn npy_u8(descr: &str, shape: &[usize], data: &[u8]) -> Vec<u8> {
    let mut b = npy_header(descr, shape);
    b.extend_from_slice(data);
    b
}

fn write_npz(path: &Path, arrays: &[(&str, Vec<u8>)]) -> std::io::Result<()> {
    let f = File::create(path)?;
    let mut zip = zip::ZipWriter::new(f);
    let opts: zip::write::SimpleFileOptions = zip::write::SimpleFileOptions::default()
        .compression_method(zip::CompressionMethod::Deflated);
    for (name, bytes) in arrays {
        zip.start_file(*name, opts)?;
        zip.write_all(bytes)?;
    }
    zip.finish()?;
    Ok(())
}

// ---------------------------------------------------------------------------
// 教师搜索驱动（与 zhuoji/rust_search.py::_deep_search_run 同构，无 Python）
// ---------------------------------------------------------------------------

/// 批前向 + `_BatchRunner` 语义：按头取 logits → 28 槽掩码 → argmax。
fn batch_forward(inf: &mut Infer, b: &Batch) -> (Vec<u8>, Vec<f32>) {
    let n = b.obs.len();
    let mut tiles = Vec::with_capacity(n * 15 * 27);
    let mut glob = Vec::with_capacity(n * 70);
    for o in &b.obs {
        tiles.extend_from_slice(&o.tiles);
        glob.extend_from_slice(&o.glob);
    }
    let out = inf.run(&tiles, &glob, n).expect("教师前向失败");
    let mut acts = Vec::with_capacity(n);
    for i in 0..n {
        let head = b.head[i];
        let mask = &b.obs[i].mask;
        let (src, width): (&[f32], usize) = match head {
            HEAD_SELF_KONG => (&out.self_kong[i * 28..(i + 1) * 28], 28),
            HEAD_RESPOND => (&out.respond[i * 4..(i + 1) * 4], 4),
            _ => (&out.discard[i * 28..(i + 1) * 28], 28),
        };
        let mut best = 0usize;
        let mut bestv = f32::NEG_INFINITY;
        for j in 0..width {
            if mask[j] != 0 && src[j] > bestv {
                bestv = src[j];
                best = j;
            }
        }
        acts.push(best as u8);
    }
    (acts, out.value)
}

/// 一次教师搜索（n_det-ply）→ 每个 legal 动作的 Q。
fn teacher_q(
    inf: &mut Infer,
    g: &Game,
    p: u8,
    legal: &[Action],
    n_det: usize,
    ff_cap: usize,
    topk: usize,
    ply: u8,
    seed: u64,
    ft: &FanTable,
) -> Vec<f64> {
    let mut ds = DeepSearch::new_fair(g, p, legal, n_det, ff_cap, topk, ply, seed, ft,
                                     true, 0, 1.0, false);
    let mut guard = 0usize;
    while !ds.finished() {
        guard += 1;
        if guard > 4000 {
            panic!("教师搜索未收敛（疑似死循环）");
        }
        if ds.current().needs_net() {
            let b = ds.current().take_batch();
            let (acts, vals) = batch_forward(inf, &b);
            ds.current_mut().feed(&acts, &vals);
        } else if ds.current().has_unsettled() {
            ds.current_mut().settle_internal();
        } else if !ds.promote() {
            break;
        }
    }
    ds.result()
}

// ---------------------------------------------------------------------------
// rollout 策略（NetAgent.choose 同构：掩码 softmax/T 采样）
// ---------------------------------------------------------------------------

fn softmax_sample(logits: &[f32], mask: &[u8], len: usize, temp: f64, rng: &mut Rng) -> usize {
    let mut maxv = f32::NEG_INFINITY;
    for j in 0..len {
        if mask[j] != 0 && logits[j] > maxv {
            maxv = logits[j];
        }
    }
    if !maxv.is_finite() {
        return 0;
    }
    if temp <= 1e-6 {
        for j in 0..len {
            if mask[j] != 0 && logits[j] >= maxv {
                return j;
            }
        }
        return 0;
    }
    let t = temp as f32;
    let mut sum = 0.0f64;
    let mut probs = vec![0f64; len];
    for j in 0..len {
        if mask[j] != 0 {
            let e = ((logits[j] - maxv) / t).exp() as f64;
            probs[j] = e;
            sum += e;
        }
    }
    let u = (rng.next_u64() as f64 / u64::MAX as f64) * sum;
    let mut acc = 0.0;
    for j in 0..len {
        acc += probs[j];
        if u <= acc && probs[j] > 0.0 {
            return j;
        }
    }
    for j in (0..len).rev() {
        if probs[j] > 0.0 {
            return j;
        }
    }
    0
}

/// 学生单步决策。enc_ver: 1 = v1（master），3 = v3（M1）。
fn student_choose(
    inf: &mut Infer,
    g: &Game,
    ft: &FanTable,
    temp: f64,
    enc_ver: u8,
    rng: &mut Rng,
) -> Action {
    let p = g.actor();
    let (head, tiles, glob, mask, mask_len) = if enc_ver == 3 {
        let o = encode_v3::encode_v3(g, p, ft);
        (o.head, o.tiles, o.glob, o.mask, o.mask_len)
    } else {
        let o = encode::encode_v1(g, p, ft);
        (
            o.head,
            o.tiles.to_vec(),
            o.glob.to_vec(),
            o.mask,
            o.mask_len,
        )
    };
    let out = inf.run_exact(&tiles, &glob, 1).expect("学生前向失败");
    let (src, width): (&[f32], usize) = match head {
        HEAD_SELF_KONG => (&out.self_kong[..28], 28),
        HEAD_RESPOND => (&out.respond[..4], 4),
        _ => (&out.discard[..28], 28),
    };
    let idx = softmax_sample(src, &mask, mask_len.min(width), temp, rng);
    encode::index_to_action(g, head, idx)
}

// ---------------------------------------------------------------------------
// worker：rollout + 标注
// ---------------------------------------------------------------------------

struct Rec {
    tiles: Vec<f32>, // 63×27
    glob: Vec<f32>,  // 130
    head: u8,
    mask: [u8; N_SLOTS],
    q: [f32; N_SLOTS],
}

fn worker(
    env: &ort::environment::Environment,
    tid: usize,
    student_path: &str,
    teacher_path: &str,
    student_enc: u8,
    n_targets: usize,
    sample_rate: f64,
    temperature: f64,
    n_det: usize,
    ff_cap: usize,
    topk: usize,
    ply: u8,
    seed0: u64,
    out_dir: &Path,
    student_coreml: bool,
    use_coreml: bool,
    threads_per_infer: usize,
) {
    let ft = FanTable::build();

    let (s_ch, s_gdim) = if student_enc == 3 {
        (NUM_CHANNELS_V3, GLOBAL_DIM_V3)
    } else {
        (15, 70)
    };
    // M1(v3) 学生 ONNX 会被 CoreML 切成 65 个分区，batch=1 单步在多线程
    // 并发下争用 ANE 可达 0.2s/步（一局上百步 → 23s/局）。CPU EP 单步仅
    // ~5ms 且不与教师抢 ANE，故学生默认可独立走 CPU；教师必须留 CoreML。
    let mut stu = Infer::new(env, student_path, s_ch, s_gdim, threads_per_infer, student_coreml)
        .expect("学生模型加载失败");
    let mut tea = Infer::new(env, teacher_path, 15, 70, threads_per_infer, use_coreml)
        .expect("教师模型加载失败");
    tea.warm_ladder().expect("教师预热失败");

    let mut rng = Rng::new(seed0 + tid as u64 * 10007);
    let mut recs: Vec<Rec> = Vec::with_capacity(n_targets);
    let mut deals = 0usize;
    let t0 = Instant::now();

    while recs.len() < n_targets {
        deals += 1;
        let mut wall = full_deck();
        rng.shuffle(&mut wall);
        let dealer = rng.below(4) as u8;
        let mut g = Game::new_with_wall(wall, dealer, Config::default(), &ft);
        let mut steps = 0usize;
        while g.phase != dushan_engine::actions::PHASE_OVER && steps < 600 {
            let acts = g.legal_actions(&ft);
            if acts.is_empty() {
                break;
            }
            if acts.len() >= 2
                && (rng.next_u64() as f64 / u64::MAX as f64) < sample_rate
                && recs.len() < n_targets
            {
                let p = g.actor();
                let o = encode_v3::encode_v3(&g, p, &ft);
                let q = teacher_q(&mut tea, &g, p, &acts, n_det, ff_cap, topk, ply,
                                  seed0 + tid as u64 * 7919 + deals as u64, &ft);
                let mut q28 = [f32::NAN; N_SLOTS];
                for (a, &qa) in acts.iter().zip(q.iter()) {
                    if let Some(idx) = encode::head_index(o.head, *a) {
                        q28[idx] = qa as f32;
                    }
                }
                let mut mask28 = [0u8; N_SLOTS];
                let mlen = o.mask_len.min(N_SLOTS);
                mask28[..mlen].copy_from_slice(&o.mask[..mlen]);
                recs.push(Rec {
                    tiles: o.tiles,
                    glob: o.glob,
                    head: o.head,
                    mask: mask28,
                    q: q28,
                });
            }
            let a = if acts.len() == 1 {
                acts[0]
            } else {
                student_choose(&mut stu, &g, &ft, temperature, student_enc, &mut rng)
            };
            g.step(a, &ft);
            steps += 1;
        }
        if deals % 10 == 0 {
            let el = t0.elapsed().as_secs_f64();
            let rate = recs.len() as f64 / el.max(1e-9) * 3600.0;
            eprintln!(
                "[w{tid}] deals={deals} targets={}/{n_targets} {el:.0}s ({rate:.0} 标注/h)",
                recs.len()
            );
        }
    }

    // 写 npz（与 gen_exit.py 同构）
    let n = recs.len();
    let mut tiles = Vec::with_capacity(n * NUM_CHANNELS_V3 * 27);
    let mut glob = Vec::with_capacity(n * GLOBAL_DIM_V3);
    let mut head = Vec::with_capacity(n);
    let mut mask = Vec::with_capacity(n * N_SLOTS);
    let mut q = Vec::with_capacity(n * N_SLOTS);
    for r in &recs {
        tiles.extend_from_slice(&r.tiles);
        glob.extend_from_slice(&r.glob);
        head.push(r.head);
        mask.extend_from_slice(&r.mask);
        q.extend_from_slice(&r.q);
    }
    let mask_u8: Vec<u8> = mask.iter().map(|&x| if x != 0 { 1 } else { 0 }).collect();
    let arrays = vec![
        (
            "tiles.npy",
            npy_f32(&[n, NUM_CHANNELS_V3, 27], &tiles),
        ),
        ("glob.npy", npy_f32(&[n, GLOBAL_DIM_V3], &glob)),
        ("head.npy", npy_u8("|u1", &[n], &head)),
        ("mask.npy", npy_u8("|b1", &[n, N_SLOTS], &mask_u8)),
        ("q.npy", npy_f32(&[n, N_SLOTS], &q)),
    ];
    std::fs::create_dir_all(out_dir).ok();
    let out = out_dir.join(format!("exit_w{tid}.npz"));
    write_npz(&out, &arrays).expect("npz 写出失败");
    eprintln!("[w{tid}] saved {}: {n} targets", out.display());
}

// ---------------------------------------------------------------------------
// 对拍模式
// ---------------------------------------------------------------------------

fn check_encode(states: &str, out: &str) {
    let ft = FanTable::build();
    let fin = File::open(states).expect("打开状态文件失败");
    let fout = File::create(out).expect("创建输出失败");
    let mut w = BufWriter::new(fout);
    for line in BufReader::new(fin).lines() {
        let line = line.expect("读行失败");
        if line.trim().is_empty() {
            continue;
        }
        let v: serde_json::Value = serde_json::from_str(&line).expect("JSON 解析失败");
        let g = game_from_json(&v["snap"]);
        let p = v["p"].as_u64().unwrap_or(0) as u8;
        let o = encode_v3::encode_v3(&g, p, &ft);
        let rec = serde_json::json!({
            "tiles": o.tiles,
            "glob": o.glob,
            "head": o.head,
            "mask": o.mask,
            "mask_len": o.mask_len,
        });
        writeln!(w, "{rec}").ok();
    }
    w.flush().ok();
    eprintln!("check-encode done → {out}");
}

fn check_q(teacher: &str, states: &str, out: &str, use_coreml: bool) {
    let ft = FanTable::build();
    let env = ort::init_from(dushan_engine::infer::ort_dylib())
        .expect("ORT dylib")
        .build()
        .expect("ORT env");
    let mut tea = Infer::new(&env, teacher, 15, 70, 1, use_coreml).expect("教师模型");
    tea.warm_ladder().ok();
    let fin = File::open(states).expect("打开状态文件失败");
    let fout = File::create(out).expect("创建输出失败");
    let mut w = BufWriter::new(fout);
    for line in BufReader::new(fin).lines() {
        let line = line.expect("读行失败");
        if line.trim().is_empty() {
            continue;
        }
        let v: serde_json::Value = serde_json::from_str(&line).expect("JSON 解析失败");
        let g = game_from_json(&v["snap"]);
        let p = v["p"].as_u64().unwrap_or(0) as u8;
        let legal: Vec<Action> = v["legal"]
            .as_array()
            .map(|a| {
                a.iter()
                    .map(|c| {
                        let code = c.as_u64().unwrap_or(0) as u16;
                        let kind = (code / 28) as u8;
                        let t = code % 28;
                        Action::new(kind, if t == 27 { 255 } else { t as u8 })
                    })
                    .collect()
            })
            .unwrap_or_default();
        let seed = v["seed"].as_u64().unwrap_or(0);
        let q = teacher_q(&mut tea, &g, p, &legal, 32, 24, 3, 3, seed, &ft);
        let rec = serde_json::json!({ "q": q });
        writeln!(w, "{rec}").ok();
    }
    w.flush().ok();
    eprintln!("check-q done → {out}");
}

// ---------------------------------------------------------------------------
// CLI
// ---------------------------------------------------------------------------

fn arg_map(args: &[String]) -> (std::collections::HashMap<String, String>, Vec<String>) {
    let mut m = std::collections::HashMap::new();
    let mut pos = Vec::new();
    let mut i = 0;
    while i < args.len() {
        if args[i].starts_with("--") && i + 1 < args.len() {
            m.insert(args[i][2..].to_string(), args[i + 1].clone());
            i += 2;
        } else {
            pos.push(args[i].clone());
            i += 1;
        }
    }
    (m, pos)
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let (m, pos) = arg_map(&args);

    if let Some(states) = m.get("check-encode") {
        let out = m
            .get("out")
            .cloned()
            .or_else(|| pos.first().cloned())
            .unwrap_or_else(|| "encode_rust.jsonl".into());
        check_encode(states, &out);
        return;
    }
    if let Some(states) = m.get("check-q") {
        let teacher = m
            .get("teacher")
            .cloned()
            .unwrap_or_else(|| "models/onnx/dushan_master.onnx".into());
        let out = m
            .get("out")
            .cloned()
            .or_else(|| pos.first().cloned())
            .unwrap_or_else(|| "q_rust.jsonl".into());
        let coreml = m.get("device").map(|d| d != "cpu").unwrap_or(true);
        check_q(&teacher, states, &out, coreml);
        return;
    }

    let student = m
        .get("student")
        .cloned()
        .unwrap_or_else(|| "models/onnx/dushan_master.onnx".into());
    let teacher = m
        .get("teacher")
        .cloned()
        .unwrap_or_else(|| student.clone());
    let targets: usize = m
        .get("targets")
        .and_then(|x| x.parse().ok())
        .unwrap_or(100_000);
    let nthreads: usize = m
        .get("threads")
        .and_then(|x| x.parse().ok())
        .unwrap_or(4);
    let sample_rate: f64 = m
        .get("sample-rate")
        .and_then(|x| x.parse().ok())
        .unwrap_or(0.25);
    let temperature: f64 = m
        .get("temperature")
        .and_then(|x| x.parse().ok())
        .unwrap_or(0.4);
    let student_enc: u8 = m
        .get("student-enc")
        .and_then(|x| x.parse().ok())
        .unwrap_or(1);
    let seed0: u64 = m.get("seed0").and_then(|x| x.parse().ok()).unwrap_or(610_000);
    let out_dir = PathBuf::from(m.get("out").cloned().unwrap_or_else(|| "runs/exit_r01".into()));
    let n_det: usize = m.get("n-det").and_then(|x| x.parse().ok()).unwrap_or(32);
    let ff_cap: usize = m.get("ff-cap").and_then(|x| x.parse().ok()).unwrap_or(24);
    let topk: usize = m.get("topk").and_then(|x| x.parse().ok()).unwrap_or(3);
    let ply: u8 = m.get("ply").and_then(|x| x.parse().ok()).unwrap_or(3);
    let use_coreml = m.get("device").map(|d| d != "cpu").unwrap_or(true);
    // 学生设备默认继承 --device；--student-device cpu 可只把学生挪到 CPU EP。
    let student_coreml = m
        .get("student-device")
        .map(|d| d != "cpu")
        .unwrap_or(use_coreml);
    let threads_per_infer: usize = m
        .get("infer-threads")
        .and_then(|x| x.parse().ok())
        .unwrap_or(1);

    // ORT 环境是进程全局的：必须只 init 一次，再把 &Environment 传给各线程
    // （并发 init_from 会死锁 —— 冒烟实测 SIGTERM 无输出）。
    let env = ort::init_from(dushan_engine::infer::ort_dylib())
        .expect("加载 onnxruntime dylib 失败")
        .build()
        .expect("ORT 环境初始化失败");
    eprintln!("[annotate] ORT ready (teacher_coreml={use_coreml}, student_coreml={student_coreml})");

    let per = (targets + nthreads - 1) / nthreads;
    std::thread::scope(|scope| {
        for tid in 0..nthreads {
            let student = student.clone();
            let teacher = teacher.clone();
            let out_dir = out_dir.clone();
            let env = &env;
            scope.spawn(move || {
                worker(
                    env,
                    tid,
                    &student,
                    &teacher,
                    student_enc,
                    per,
                    sample_rate,
                    temperature,
                    n_det,
                    ff_cap,
                    topk,
                    ply,
                    seed0,
                    &out_dir,
                    student_coreml,
                    use_coreml,
                    threads_per_infer,
                );
            });
        }
    });
    eprintln!("[annotate] all done");
}
