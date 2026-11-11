//! O2：纯 Rust `tract` 推理后端。
//!
//! `rust_engine` 的推理后端（`infer.rs`）绑死 ORT + CoreML，是 macOS 专属；
//! 设备端这里用 tract 从**内存中的 ONNX 字节**加载，四头输出顺序与
//! `infer.rs::BatchOut` 完全一致：0=discard(28) / 1=self_kong(28) /
//! 2=respond(4) / 3=value(1)。
//!
//! 与 Python `NetAgent` 的贪心口径一致：按 `head` 取对应头的 logits，
//! 只在 `legal_actions()` 里挑分数最高的动作（temperature=0）。

use std::io::Cursor;

use dushan_engine::actions::Action;
use dushan_engine::encode::{self, HEAD_DISCARD, HEAD_RESPOND, HEAD_SELF_KONG};
use dushan_engine::encode_v3;
use dushan_engine::fan::FanTable;
use dushan_engine::state::Game;
use tract_onnx::prelude::*;

/// tract 各小版本的「可执行计划」具体类型名不稳定，这里用闭包把
/// `plan.run()` 封起来，`Net` 只依赖输入/输出契约，不依赖具体别名。
/// 第一参数是 batch（搜索时一次几十到几千个状态），单步贪心恒为 1。
type ForwardBatch =
    Box<dyn Fn(usize, &[f32], &[f32]) -> Result<Vec<Vec<f32>>, String> + Send + Sync>;

pub struct Net {
    forward_batch: ForwardBatch,
    enc: u8,
    ch: usize,
    gdim: usize,
}

impl Net {
    pub fn from_bytes(bytes: &[u8], enc: u8) -> Result<Self, String> {
        let (ch, gdim) = match enc {
            1 => (15usize, 70usize),
            3 => (63usize, 130usize),
            other => return Err(format!("不支持的 encoder_version={other}")),
        };
        let mut reader = Cursor::new(bytes);
        let model = tract_onnx::onnx()
            .model_for_read(&mut reader)
            .map_err(|e| format!("ONNX 解析失败: {e}"))?;
        let plan = model
            .into_optimized()
            .map_err(|e| format!("ONNX 优化失败（算子不受支持？）: {e}"))?
            .into_runnable()
            .map_err(|e| format!("ONNX 编译失败: {e}"))?;
        let forward: ForwardBatch = Box::new(move |n: usize, tiles: &[f32], glob: &[f32]| {
            if n == 0 {
                return Ok(Vec::new());
            }
            if tiles.len() != n * ch * 27 || glob.len() != n * gdim {
                return Err(format!(
                    "批尺寸不符: n={n} tiles={} glob={} (期望 {} / {})",
                    tiles.len(),
                    glob.len(),
                    n * ch * 27,
                    n * gdim
                ));
            }
            let t = Tensor::from_shape(&[n, ch, 27], tiles)
                .map_err(|e| format!("构造 tiles 张量失败: {e}"))?;
            let g = Tensor::from_shape(&[n, gdim], glob)
                .map_err(|e| format!("构造 glob 张量失败: {e}"))?;
            let outs = plan
                .run(tvec!(t.into_tvalue(), g.into_tvalue()))
                .map_err(|e| format!("前向失败: {e}"))?;
            let mut v = Vec::with_capacity(outs.len());
            for o in outs.iter() {
                let a = o
                    .to_plain_array_view::<f32>()
                    .map_err(|e| format!("读取输出失败: {e}"))?;
                v.push(a.iter().copied().collect::<Vec<f32>>());
            }
            Ok(v)
        });
        Ok(Self { forward_batch: forward, enc, ch, gdim })
    }

    pub fn enc(&self) -> u8 { self.enc }
    pub fn channels(&self) -> usize { self.ch }
    pub fn global_dim(&self) -> usize { self.gdim }

    /// 返回四头 logits（flatten，batch=1）。
    fn forward(&self, tiles: &[f32], glob: &[f32]) -> Result<Vec<Vec<f32>>, String> {
        if tiles.len() != self.ch * 27 || glob.len() != self.gdim {
            return Err(format!(
                "编码尺寸不符: tiles={} glob={} (期望 {} / {})",
                tiles.len(), glob.len(), self.ch * 27, self.gdim
            ));
        }
        (self.forward_batch)(1, tiles, glob)
    }

    /// 搜索用批量前向：`n` 个观测 → `(每行 argmax 动作索引, 每行 value)`。
    ///
    /// 口径与 Python `_BatchRunner.run` 逐项一致：
    ///   1. 四头输出顺序 0=discard(28) / 1=self_kong(28) / 2=respond(4) / 3=value；
    ///   2. 按该行 `head` 只取对应头，其余位置置 `-1e9`；
    ///   3. `mask`（长度 28）为 0 的位置同样置 `-1e9`，再取 argmax；
    ///   4. value 直接取第 3 个输出（形状 [n]）。
    ///
    /// 注意：**不做 64 倍 pad**。Python 侧 pad 是为了让 MPS 的形状种类收敛；
    /// tract 在 CPU 上是按形状缓存执行计划的，pad 反而多做无用计算。
    pub fn search_batch(
        &self,
        n: usize,
        tiles: &[f32],
        glob: &[f32],
        heads: &[u8],
        masks: &[[u8; 28]],
    ) -> Result<(Vec<u8>, Vec<f32>), String> {
        if heads.len() != n || masks.len() != n {
            return Err(format!("批长度不一致: n={n} heads={} masks={}", heads.len(), masks.len()));
        }
        let outs = self.forward_mt(n, tiles, glob)?;
        if outs.len() < 4 {
            return Err(format!("模型输出头不足: {}", outs.len()));
        }
        let mut acts = Vec::with_capacity(n);
        let mut vals = Vec::with_capacity(n);
        for i in 0..n {
            // 头与宽度必须与 feed() 的 index_to_action 口径对应。
            let (src, w): (&Vec<f32>, usize) = match heads[i] {
                HEAD_SELF_KONG => (&outs[1], 28),
                HEAD_RESPOND => (&outs[2], 4),
                _ => (&outs[0], 28),
            };
            let base = i * w;
            let mut lg = [-1e9f32; 28];
            if base + w <= src.len() {
                lg[..w].copy_from_slice(&src[base..base + w]);
            }
            for (j, slot) in lg.iter_mut().enumerate() {
                if masks[i][j] == 0 {
                    *slot = -1e9;
                }
            }
            let mut best = 0usize;
            let mut bv = lg[0];
            for (j, v) in lg.iter().enumerate().skip(1) {
                if *v > bv {
                    bv = *v;
                    best = j;
                }
            }
            acts.push(best as u8);
            vals.push(outs[3].get(i).copied().unwrap_or(0.0));
        }
        Ok((acts, vals))
    }

    /// 并行分块前向：把 `n` 行按 `K = mt_threads()` 切片，各块独立跑同一份
    /// tract 计划（`plan.run` 对 `&plan` 只读，可安全并发），按原序拼回。
    ///
    /// 为什么值得：搜索每一层的批大小是几十到几千行，单线程下 tract 的
    /// 每行成本已稳定在 0.27 ms（见 `search_bench`），降不下去；但块之间
    /// **完全独立**，宿主机 K=4 实测 3.37×、K=6 4.03×。手机 CPU 主频低、
    /// 核更多，收益只会更大。切片是纯数据并行，输出按块序号拼接，
    /// 与单线程**逐位一致**（每行的两次矩阵乘顺序不变）。
    fn forward_mt(&self, n: usize, tiles: &[f32], glob: &[f32]) -> Result<Vec<Vec<f32>>, String> {
        let k = mt_threads();
        // 小批不值得起线程（起停成本 ~50 µs，n=8 的整批才 2.5 ms）。
        if k <= 1 || n < 8 {
            return (self.forward_batch)(n, tiles, glob);
        }
        let chunk = n.div_ceil(k);
        let (ch, gd) = (self.ch, self.gdim);
        let mut parts: Vec<Result<Vec<Vec<f32>>, String>> = Vec::with_capacity(k);
        std::thread::scope(|sc| {
            let mut hs = Vec::with_capacity(k);
            for c in 0..k {
                let lo = (c * chunk).min(n);
                let hi = ((c + 1) * chunk).min(n);
                if lo >= hi { break; }
                let f = &self.forward_batch;
                hs.push(sc.spawn(move || {
                    f(hi - lo,
                      &tiles[lo * ch * 27..hi * ch * 27],
                      &glob[lo * gd..hi * gd])
                }));
            }
            for h in hs {
                parts.push(h.join().unwrap_or_else(|_| Err("前向线程 panic".into())));
            }
        });
        let mut out: Vec<Vec<f32>> = Vec::new();
        for p in parts {
            let v = p?;
            if out.is_empty() {
                out = v;
            } else {
                for (i, head) in v.into_iter().enumerate() {
                    if let Some(dst) = out.get_mut(i) {
                        dst.extend_from_slice(&head);
                    }
                }
            }
        }
        Ok(out)
    }

    /// 一次观测 → (head, mask, tiles, glob)。
    fn observe(&self, g: &Game, p: u8, ft: &FanTable) -> (u8, [u8; 28], Vec<f32>, Vec<f32>) {
        if self.enc == 1 {
            let o = encode::encode_v1(g, p, ft);
            (o.head, o.mask, o.tiles.to_vec(), o.glob.to_vec())
        } else {
            let o = encode_v3::encode_v3(g, p, ft);
            (o.head, o.mask, o.tiles, o.glob)
        }
    }

    /// 合法动作按该头 logits 降序排列；`/api/hint` 用它给候选，`choose` 取首项。
    pub fn rank(&self, g: &Game, p: u8, ft: &FanTable) -> Option<Vec<(f32, Action)>> {
        let (head, _mask, tiles, glob) = self.observe(g, p, ft);
        let outs = self.forward(&tiles, &glob).ok()?;
        let (logits, width) = match head {
            HEAD_SELF_KONG => (outs.get(1)?, 28usize),
            HEAD_RESPOND => (outs.get(2)?, 4usize),
            _ => (outs.get(0)?, 28usize),
        };
        let mut v = Vec::new();
        for a in g.legal_actions(ft) {
            let Some(idx) = encode::head_index(head, a) else { continue };
            if idx < width && idx < logits.len() {
                v.push((logits[idx], a));
            }
        }
        v.sort_by(|x, y| y.0.partial_cmp(&x.0).unwrap_or(std::cmp::Ordering::Equal));
        Some(v)
    }

    /// 贪心决策。推理失败时返回 `None`，由调用方降级到启发式。
    pub fn choose(&self, g: &Game, p: u8, ft: &FanTable) -> Option<Action> {
        self.rank(g, p, ft).and_then(|v| v.first().map(|x| x.1))
    }

    /// 状态价值（`/api/hint` 的 value 字段用；离线版前端不读，保留接口）。
    #[allow(dead_code)]
    pub fn value(&self, g: &Game, p: u8, ft: &FanTable) -> f32 {
        let (_h, _m, tiles, glob) = self.observe(g, p, ft);
        match self.forward(&tiles, &glob) {
            Ok(outs) if outs.len() >= 4 => outs[3].first().copied().unwrap_or(0.0),
            _ => 0.0,
        }
    }
}

/// 无模型/推理失败时的纯规则兜底：优先胡 → 杠 → 碰 → 打孤张。
/// 与服务端 `HeuristicBot` 同方向，保证离线版永远能推进。
pub fn heuristic_choose(g: &Game, _p: u8, ft: &FanTable) -> Action {
    let acts = g.legal_actions(ft);
    for want in [dushan_engine::actions::HU,
                 dushan_engine::actions::MINGGANG,
                 dushan_engine::actions::ANGANG,
                 dushan_engine::actions::BUGANG,
                 dushan_engine::actions::PENG] {
        if let Some(&a) = acts.iter().find(|a| a.kind == want) {
            return a;
        }
    }
    // 打持有数最少的一张（优先保留对子/搭子）
    acts.iter()
        .filter(|a| a.kind == dushan_engine::actions::DISCARD)
        .min_by_key(|a| {
            let p = g.current as usize;
            g.hands[p][a.tile as usize]
        })
        .copied()
        .unwrap_or_else(Action::pass)
}

/// 并行前向线程数：默认 `min(4, 可用核数)`。
///
/// 上限取 4 而不是核数：手机上大核通常 2~4 个，再往上加的线程会落到小核
/// （同频算力只有大核 1/3），实测（宿主机 K=4→3.37×，K=6→4.03×，边际只剩
/// 18%）也不划算；4 线程已经能把 N128 从 1.74 s 压到 ~0.52 s。
/// `DUSHAN_MT=1` 可强制单线程，用于对拍与排障。
pub fn mt_threads() -> usize {
    static K: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *K.get_or_init(|| {
        if let Ok(v) = std::env::var("DUSHAN_MT") {
            if let Ok(n) = v.parse::<usize>() {
                return n.clamp(1, 8);
            }
        }
        std::thread::available_parallelism()
            .map(|n| n.get().clamp(1, 4))
            .unwrap_or(1)
    })
}
