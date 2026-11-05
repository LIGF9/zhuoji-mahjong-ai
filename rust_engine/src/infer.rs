//! ONNX Runtime 原生推理（feature = "annotate"）—— Rust 标注器的前向后端。
//!
//! 与 `zhuoji/forward.py::OrtForward` 同口径：
//! * CoreML EP（ANE）+ CPU 兜底，intra-op 可配；
//! * 教师批：形状量化到 LADDER (128/256/512/1024/2048)，每形状一个会话
//!   （CoreML 换形状重编译 150~240ms，必须避免）；
//! * rollout 单步：`run_exact(1)` 用精确形状会话（B=1 常驻，不 pad 浪费）；
//! * 不足的行补零，前向后只取前 n 行。
//!
//! 会话按线程私有（`Session::run` 签名限制 + ANE 多流并发），与旧的
//! 多进程 Python worker 同构。

use std::collections::HashMap;

use ndarray::{Array2, Array3};

use ort::environment::Environment;
use ort::ep;
use ort::session::builder::GraphOptimizationLevel;
use ort::session::Session;
use ort::value::TensorRef;

/// 形状梯子：32 覆盖教师批（ff_cap≤24 时全部落此档，避免 pad 到 128 浪费 ANE），
/// 128+ 与 `zhuoji/forward.py` 的 LADDER 同口径。
pub const LADDER: [usize; 6] = [32, 128, 256, 512, 1024, 2048];

/// onnxruntime 动态库路径：默认用仓库 .venv 里 Python ORT 的同一份
/// （与 forward.py 数值完全同源）；`ORT_DYLIB_PATH` 可覆盖。
pub fn ort_dylib() -> String {
    std::env::var("ORT_DYLIB_PATH").unwrap_or_else(|_| {
        concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../.venv/lib/python3.13/site-packages/onnxruntime/capi/libonnxruntime.1.31.1.dylib"
        )
        .to_string()
    })
}

/// 一批前向结果（n 行真实数据，pad 行已裁掉）。
pub struct BatchOut {
    pub discard: Vec<f32>,   // n × 28
    pub self_kong: Vec<f32>, // n × 28
    pub respond: Vec<f32>,   // n × 4
    pub value: Vec<f32>,     // n
}

pub struct Infer {
    path: String,
    threads: usize,
    use_coreml: bool,
    env: Environment,
    sessions: HashMap<usize, Session>,
    pub ch: usize,
    pub gdim: usize,
}

impl Infer {
    /// `ch/gdim` = 输入通道与全局维（v1=15/70，v3=63/130）。
    pub fn new(
        env: &Environment,
        path: &str,
        ch: usize,
        gdim: usize,
        threads: usize,
        use_coreml: bool,
    ) -> Result<Self, ort::Error> {
        let mut inf = Infer {
            path: path.to_string(),
            threads,
            use_coreml,
            env: env.clone(),
            sessions: HashMap::new(),
            ch,
            gdim,
        };
        inf.session_for(LADDER[0])?;
        Ok(inf)
    }

    fn session_for(&mut self, key: usize) -> Result<(), ort::Error> {
        if self.sessions.contains_key(&key) {
            return Ok(());
        }
        let mut b = Session::builder(&self.env)?
            .with_optimization_level(GraphOptimizationLevel::Level3)?
            .with_intra_threads(self.threads)?;
        if self.use_coreml {
            b = b.with_execution_providers([ep::CoreML::default().build()])?;
        }
        let s = b.commit_from_file(&self.path)?;
        self.sessions.insert(key, s);
        Ok(())
    }

    /// 预热梯子形状（每形状一次图编译）。
    pub fn warm_ladder(&mut self) -> Result<(), ort::Error> {
        for &rung in LADDER.iter() {
            self.session_for(rung)?;
            let zeros_t = vec![0f32; rung * self.ch * 27];
            let zeros_g = vec![0f32; rung * self.gdim];
            self.run_at(rung, &zeros_t, &zeros_g)?;
        }
        Ok(())
    }

    fn ceil_rung(n: usize) -> usize {
        for &r in LADDER.iter() {
            if n <= r {
                return r;
            }
        }
        LADDER[LADDER.len() - 1]
    }

    /// 教师批前向：内部 pad 到梯子形状。tiles: n×ch×27 平铺；glob: n×gdim 平铺。
    pub fn run(&mut self, tiles: &[f32], glob: &[f32], n: usize) -> Result<BatchOut, ort::Error> {
        let rung = Self::ceil_rung(n);
        self.run_at(rung, tiles, glob)
    }

    /// 精确形状前向（rollout 单步用；会话按形状缓存）。
    pub fn run_exact(&mut self, tiles: &[f32], glob: &[f32], n: usize) -> Result<BatchOut, ort::Error> {
        self.session_for(n)?;
        self.run_at(n, tiles, glob)
    }

    fn run_at(&mut self, rung: usize, tiles: &[f32], glob: &[f32]) -> Result<BatchOut, ort::Error> {
        let ch = self.ch;
        let gdim = self.gdim;
        let n = tiles.len() / (ch * 27);
        let mut tbuf = vec![0f32; rung * ch * 27];
        let mut gbuf = vec![0f32; rung * gdim];
        tbuf[..n * ch * 27].copy_from_slice(&tiles[..n * ch * 27]);
        gbuf[..n * gdim].copy_from_slice(&glob[..n * gdim]);
        let t = Array3::from_shape_vec((rung, ch, 27), tbuf).expect("tiles shape");
        let g = Array2::from_shape_vec((rung, gdim), gbuf).expect("glob shape");

        let sess = self.sessions.get_mut(&rung).expect("会话未预热");
        let outputs = sess.run(ort::inputs![
            TensorRef::from_array_view(&t)?,
            TensorRef::from_array_view(&g)?
        ])?;

        let grab = |i: usize, width: usize| -> Result<Vec<f32>, ort::Error> {
            let (_shape, data) = outputs[i].try_extract_tensor::<f32>()?;
            Ok(data[..n * width].to_vec())
        };
        let discard = grab(0, 28)?;
        let self_kong = grab(1, 28)?;
        let respond = grab(2, 4)?;
        let value = grab(3, 1)?;
        Ok(BatchOut {
            discard,
            self_kong,
            respond,
            value,
        })
    }
}
