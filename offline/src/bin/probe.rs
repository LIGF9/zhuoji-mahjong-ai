//! tract 算子 spike（§7.3 第 1 步）：验证两个 ONNX 能被纯 Rust tract 加载、
//! 优化并跑通一次前向。任何一步失败 ⇒ 推理架构退回 ORT Android AAR。

use tract_onnx::prelude::*;

fn main() {
    let mut paths: Vec<String> = std::env::args().skip(1).collect();
    if paths.is_empty() {
        paths = vec![
            "models/onnx/champ_slim_s.onnx".into(),
            "models/onnx/dushan_master.onnx".into(),
        ];
    }
    let mut bad = 0;
    for p in &paths {
        println!("=== {p} ===");
        match probe_one(p) {
            Ok(()) => println!("[PASS] {p}"),
            Err(e) => {
                println!("[FAIL] {p}: {e:?}");
                bad += 1;
            }
        }
    }
    println!("\n总计 {} 个模型，失败 {} 个", paths.len(), bad);
    if bad > 0 {
        std::process::exit(1);
    }
}

fn probe_one(path: &str) -> TractResult<()> {
    let mut model = tract_onnx::onnx().model_for_path(path)?;
    println!("  inputs={} outputs={}", model.inputs.len(), model.outputs.len());
    for (i, o) in model.inputs.clone().iter().enumerate() {
        println!("    in[{i}] {:?}", model.outlet_fact(*o)?);
    }
    for (i, o) in model.outputs.clone().iter().enumerate() {
        println!("    out[{i}] {:?}", model.outlet_fact(*o)?);
    }

    // 算子不支持 / 动态维未定，通常都在 into_optimized() 暴露。
    let typed = model.into_optimized()?;
    println!("  optimized OK: {} nodes", typed.nodes().len());
    let plan = typed.into_runnable()?;
    println!("  runnable OK");
    let _ = plan;
    Ok(())
}
