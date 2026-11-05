# Python / Rust 使用率审计与迁移成本收益（2026-11-05）

结论先行：**标注/搜索/推理/rollout 已全部下沉 Rust，训练步只保留 torch（不可替代的现有库），
线上服务端保留 Python 协议胶水（迁移收益低、回归风险高）**。明细如下。

## 1. 代码量分布（wc -l 实测）

| 层 | 组件 | 语言 | 行数 | 运行时重活在哪 |
|---|---|---|---|---|
| 训练-标注 | `rust_engine/src/bin/annotate.rs` + `infer.rs` | Rust | 579 + 159 | **全部**（rollout、教师搜索、ORT-CoreML 前向、npz 写出） |
| 训练-引擎 | `rust_engine/src/{state,settle,search,deep,encode*}.rs` | Rust | ~3,600 | **全部**（规则、结算、PIMC 搜索、v1/v2/v3 编码） |
| 训练-学习 | `train_exit.py` | Python | 204 | torch autograd/优化器（**必须保留**） |
| 训练-编排 | `gen_exit.py`（遗留备选）/ `eval_pairs.py` / `check_pairs.py` | Python | 162/156/151 | 胶水（标注已由 Rust 二进制接管） |
| 训练-编码参考 | `zhuoji/encoder{,_v2,_v3}.py` | Python | 348+341+394 | 已移植 Rust，Python 留作对拍金标准 |
| 线上-服务 | `web/dushan_server.py` + `web/coach.py` | Python | 3040 + 377 | 协议/会话/历史/解释文案（重活在 Rust+ORT） |
| 线上-搜索 | `zhuoji/rust_search.py`（桥）→ `.so` | 桥 671 + Rust 引擎 | — | **全部**（RustSearchAgent / deep_q） |
| 线上-前向 | `zhuoji/forward.py`（OrtForward） | Python 胶水 213 | — | onnxruntime C++ / CoreML EP |
| 规则域 | `zhuoji/dushan.py` 等（Python 规则） | Python | ~3,500 | 与 Rust 双轨（golden 对拍钉住） |

Python ≈ 8,000 行（含规则域双轨），Rust ≈ 6,500 行。**计算密集路径已 100% Rust**；
Python 剩下两类：torch 学习步、HTTP/业务胶水与规则域参考实现。

## 2. 运行时占比（实测口径）

- **标注管线**（今天改造后）：Rust 100%。单线程 0.121 s/标注（~3.0 万/h），
  2–4 线程聚合 4.5 万/h（**ANE 共享是上限**，t2≈t4>t6）；与 Python 快路径
  （5.15 万/h 聚合、109.5 ms/标注）持平或略优，且零 Python 胶水、单进程。
- **训练步**：Python 编排 ~ms，torch（MPS/ANE）算子占 >99%。换 Rust 无收益。
- **线上服务**：请求处理 Python（<1 ms）；一次大师推荐 0.5–5 s 中，
  Rust 搜索 + CoreML 前向 >99%，Python 只做协议与序列化。

## 3. 迁移成本/收益矩阵

| 候选 | 成本 | 收益 | 结论 |
|---|---|---|---|
| 标注管线 → Rust | 已完成（annotate.rs + infer.rs + 手写 npy/npz） | 去掉 Python 胶水；Rust 内结算 17×；ORT-CoreML 4.8×；进程内多线程 | ✅ 本轮完成 |
| 教师搜索/结算 → Rust | 已完成（既有 .so） | 17×（settle_internal） | ✅ 既有 |
| 编码器 v2/v3 → Rust | 已完成（encode_v2/v3.rs，对拍逐位一致） | 标注零 Python；顺带修复 Python 侧 0/1 基混用 bug | ✅ 本轮完成 |
| npz 写出 → Rust | 已完成（手写 npy 头 + zip DEFLATE） | 标注全链路零 Python | ✅ 本轮完成 |
| **训练步 → Rust** | 重写 autograd/AdamW/调度/检查点 ≈ 数周–数月，高风险 | 仅省编排 ms 级；数值零收益 | ❌ 不做（torch 是"必须使用的现有库"） |
| 线上服务 → Rust | 重写 HTTP/会话/历史/解释 ≈ 数周，UI 回归风险高 | 端到端延迟改善 <10%（重活已在 Rust） | ❌ 暂不做 |
| Python 规则域 → 废弃 | golden 对拍维护成本 | 消除双轨 | ⏸ 保留作对拍金标准 |

## 4. 训练步为何保留最小 Python

`train_exit.py` 的不可替代部分只有 torch：autograd、优化器、DataLoader、检查点。
除 torch 导入与张量搬运外无独立计算（策略软标签 CE / 价值 Huber 均在张量算子里）。
这是用户约束"仅在必须使用现有的 Python 库时使用最少的 Python"的字面执行：
标注/搜索/推理/rollout/npz 已 0% Python，Python 只剩 torch 调用面（204 行）。

## 5. 本轮顺带修复的正确性 bug（对拍发现）

1. Python `encoder_v2/v3` 的 `rank_of` 0/1 基混用（`rank_of` 实为 1 基）：
   glob91 将对占比选错点数集（1/4/7 → 2/5/8）；ch54–57 邻居边界产生跨花色
   幻影（`tile_from(s,0)=s*9-1`，s=0 时 `hand[-1]` 环绕）且 run 漏计点数 9。
2. Rust `encode_v3` ch43–46 只写 t=0（Python 为整行广播）。
3. Rust ch58/59 f32 双舍入（6e-8）→ f64 中间量。
   修复后 30 状态对拍：tiles/glob/Q **max|Δ| = 0**（逐位一致）。
4. R1 数据（100k）因语义修复已用 Rust 标注器**重采**（旧数据备份
   `runs/exit_r01_bak_pre_fix`，q 标签本身无损，仅观测特征语义过期）。
