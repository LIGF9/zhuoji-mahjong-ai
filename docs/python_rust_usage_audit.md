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

## 6. 复核：加速模式下「Python 是不是耗时主力 / 能否再靠 Rust 提速」（2026-11-07 18:4x）

§1–§3 只覆盖 DAgger 标注管线。P12 起主循环切到「PPO 加速轮」，成本结构变了，
按用户规则重新剖析 + **先试 Rust 后端再决定**，结论：**Python 不是耗时主力，
再写 Rust 收益 ≤7%，且现有 Rust ORT 路径反而更慢，本轮不做 Rust 重写**。

### 6.1 实测口径（与 annotate_r21 四线程并发，数值略保守）

| 阶段 | 实测 | 谁在烧时间 |
|---|---|---|
| 加速轮 P20 合计 | 762s | — |
| ├ PPO rollout | 264s（1920 局 / 90,051 决策 / 4 worker） | M1 9.85M 网络 B=1 前向（torch 原生算子） |
| ├ PPO 更新 | 288s（704 step @ B=256, MPS） | torch autograd/AdamW，**不可替** |
| ├ 闸门 D | 147s（300 seed×4 座，双 M1 贪心） | 同上 B=1 前向 |
| └ 闸门 A | 62s（M1 贪心 vs 冻结小网） | 同上 B=1 前向 |
| DAgger 轮 R12 | 标注 ≈2.4h + 训练 2025s + 闸门 ≈10min | Rust `annotate` + ORT/CoreML 教师搜索 / torch MPS |

- `cProfile` 评测 `eval_pairs.play_deal`（greedy，40 局 / 2076 决策，5.99s）：
  torch forward **5.56s = 93%**（conv2d 2.47 + mish 0.86 + batch_norm 0.59）；
  Python 规则 `rules.step` 0.274s = **4.6%**；Python `encoder_v3` 0.424s = **7.1%**。
- `cProfile` `train_ppo.rollout_shard`（24 局，profiler 下 10.98s）：
  M1 forward 8.32s = **76%**；`encode_v3` 0.765s = **7%**；其余张量搬运/采样。
- `gae()` 纯 Python 双循环，9 万状态仅 **0.42s**（占一轮 0.06%）。

### 6.2 前向后端基准（M1 9.85M / 63×27 / 130，2 线程 CPU 为主）

| 后端 | B=1 | B=8 | B=32 | 备注 |
|---|---|---|---|---|
| torch CPU | **7.05 ms/state** | 3.72 ms/state | 8.92 ms/state | B=8 约 2×；B=32 反而更慢（算力饱和） |
| ORT CPU（Rust `Infer` 同款） | 10.6~13.1 ms/state | 10.8 | 10.9 | 批量零收益 |
| ORT CoreML | 122 ms/state | 19.0 | 8.66 | M1 被切 65 分区，B=1 灾难；B=32 才持平 |

**Rust 路线实测证伪**：把 rollout/eval 换成 Rust 状态机 + ORT（`annotate.rs` 已有
全套设施）单步从 7ms 变 10.6~13.1ms，CoreML 更到 122ms ⇒ **会更慢**。
而 Rust 能替掉的 Python 部分（规则 4.6% + 编码 7.1% + GAE 0.06%）合计约 12%，
即使零桥接开销全下沉，一轮 762s 只省 ≤55s（7%），且要重做规则对拍，风险不划算。

### 6.3 结论与真正可动的杠杆

1. **不是 Python 慢**：加速轮 ~90% 是神经网络算子（torch conv / MPS 训练），
   DAgger 轮 ~75% 是 Rust 标注器里的 ORT 教师搜索 —— 两处都已是原生代码。
2. **Rust 已占住所有值得占的重活**：`annotate.rs`（rollout/搜索/结算/编码/npz）、
   `.so`（线上搜索/结算）、`settle_internal`（17×）。没有可整段搬走的热路径。
3. 真瓶颈是**「每个决策一次 B=1 前向」的调用形态**，不是语言。可选杠杆：
   a) rollout/eval 改成锁步多局 + MPS 批量前向（PPO 更新实测 ≈2.7ms/sample，
      理论 2~3×；本轮**不做**，会改训练数据分布 / 冻结评估协议，须先预登记）；
   b) M1 蒸馏或剪枝（模型侧，非工程侧）；
   c) 减少评测局数或复用同池结果（会动冻结协议，本轮禁止）。
4. 按用户「长时任务先剖析成本、优先试 Rust」的规则，本轮已先剖析、再实测 Rust
   后端并证伪，**因此不新增 Rust 重写**，把算力留给训练主线。

## 7. 资源利用率严格评审：CPU <20% / GPU·ANE <30% 是否「没跑满」（2026-11-07 22:3x）

用户观察到 CPU 利用率 20% 以内、GPU/ANE 30% 以内（偶尔飙到 50%），问机器性能是否没被
完全利用。本轮**只读实测**（`ps` / `top` 被沙箱拒，改用 Mach `host_statistics` +
`proc_pid_rusage` 内核计数器），未启动任何吃 CPU 探针，未改主链一行代码。

### 7.1 硬件与测量口径

- Apple M5，**10 逻辑核 = 4 P + 6 E**，32 GB；load average 1.98 / 2.38 / 3.11。
- macOS「整体 CPU%」分母是 10 核。四次 3s 采样 busy = **10.6 / 15.5 / 10.7 / 15.1%**
  ⇒ **1.06~1.55 个满载核**。用户观察「20% 以内」属实。
- 若按 4 个 P 核口径则是 26~39% —— 账面「空」有相当一部分来自 E 核与分母效应，
  不等于「P 核在闲置」。

### 7.2 当前阶段实测（R24 完整 DAgger 轮的标注段）

| 量 | 实测 |
|---|---|
| 命令 | `annotate --threads 4`，学生 CPU(B=1) / 教师 CoreML-ANE(批 ≤24 → LADDER 32) |
| 教师搜索 | `n_det=32, ply=3, ff_cap=24, topk=3`（`teacher_q` 实参） |
| 全局 CPU | 1.06~1.55 / 10 核 |
| annotate 进程 CPU 时间 | **0.58 核等效**（4 线程合计，4s 窗口） |
| 标注吞吐 | **9.94k/h·worker × 4 = 39.8k/h** |
| 历史基线（§2） | 单线程 0.121s/标注 = 29.8k/h；2~4 线程 45k/h，且 **t2≈t4>t6（ANE 共享是上限）** |

4 线程加速比仅 **1.34×**（线性理想 4×），每线程效率 33%；annotate 只拿到 0.58 核 CPU
时间 ⇒ 约 2/3 墙钟在等 ANE / CPU↔ANE 同步，**不是 CPU 在算，是 CPU 在等**。
交叉验证：8770 无 ESTABLISHED 连接、无并发评估/探针进程 ⇒ 已排除外部 ANE 争抢者。

### 7.3 三层原因（都不是「参数没调满」）

1. **ANE 小图气泡**：教师每次提交 B≤24 的小批（pad 到 LADDER 32），计算量极小，
   CoreML 的 kernel 启动 / 图分区 / CPU↔ANE 拷贝占大头；学生 M1 走 CPU 是**刻意**的
   （CoreML 切 65 分区会崩 5×，见 §6.2 与 `annotate.rs:287`）。ANE「30%」是占空比，
   不是「还有 70% 可以并行喂」。
2. **共享加速器排队**：4 线程各自私有 ORT Session 抢同一 ANE；历史 t6<t4 已实测，
   加线程只会延长队列。当前 39.8k/h 已达历史 45k/h 上限的 88%。
3. **阶段串行、GPU 分时上电**：MPS 只在 PPO 更新（353s / 697s 轮）与 `train_exit`
   （2111s）段工作；标注段 GPU≈0，闸门段是 6 workers 纯 CPU。按整段时间平均，GPU 低是
   设计结果；「偶尔飙到 50%」正对应阶段切换，不是异常。

### 7.4 结论

**是，账面有大量空闲算力；但「把核吃满」在这个冻结配方下既不现实也不该做。**
直接加 workers/threads 已被 t2≈t4>t6 证伪，会让吞吐下降（16:53 两会话抢核双双
SIGTERM 是同类教训）。低利用率是 **ANE 串行等待 + 阶段数据依赖 + B=1/B≤24 调用形态**
的架构性结果，不是配置疏漏。

### 7.5 真正可兑现的杠杆（按是否动配方分类）

| 杠杆 | 预期 | 改配方？ | 处置 |
|---|---|---|---|
| **Rust 跨线程教师批聚合**：把 4 线程各自 B≤24 汇成跨线程 B=128/256 提交，减少 kernel 启动与 ANE 同步次数 | 未知，需 A/B | **否**（前向确定性，逐位可对拍） | 唯一剩余的 Rust 侧工程杠杆；排训练停止后离线基准，**7×24 内不编译、不抢核** |
| 后台 QoS / App Nap 排查（`UserIsActive=0`） | 小 | 否 | 低风险观察项，不动在跑的进程 |
| B=1 → MPS 锁步批量前向 | 理论 2~3×，能喂饱 GPU | **是**（改训练/评估分布、冻结协议） | 与 §6.36「同配方第二周期」冲突 ⇒ 预登记到 D3 之后 / 终评后 |
| 阶段流水线重叠（标注 ∥ 训练） | 中 | 是（`exit_best` 竞态、数据依赖） | 不做 |
| M1 剪枝 / 蒸馏 | 改能力上限 | 是 | 预登记 |
| 用空闲 CPU 跑隔离实验 | — | 否 | 已在做（E1 排 R24 train 段） |

按用户「长时任务先剖析成本、优先试 Rust」规则：本轮已剖析成本并复核 Rust 侧 —— Rust 仍
占住全部热路径（标注/搜索/结算/编码/npz，§1），瓶颈在 ANE 调度与调用形态，Rust 改不了
CoreML 图分区；**唯一值得试的 Rust 改动是 §7.5 第一行**，且必须等训练停止、用离线基准
同时验吞吐收益与逐位一致性后再决定是否落地。本轮不新增任何重写。

### 7.6 观察项（非事故）

R24 9.94k/h·w 比 R21 11.04k/h·w 低约 10%，已排除 Web 对局 / 并发探针争抢。两者学生
权重与每局产出标注数不同（R24 ≈13.3 targets/deal vs R21 ≈11.0），「标注/h」口径不完全
可比；且 R21 是全程 8141s 均值、R24 是前 ~1059s。处置：**只读观察**，不改配置、不重启、
不触发事故；如需对比，用同长度稳态窗口的 `deals/h` + `targets/deal` 双指标。

### 7.7 测量限制

未直接读取 ANE/GPU 硬件计数器（`powermetrics` 需 root、沙箱不允许；`ps` / `top` 被拒）。
GPU/ANE 结论来自阶段耗时、代码中的批形状、吞吐与 CPU 阻塞时间等**间接证据**，不得写成
「实测 ANE 利用率 = 30%」。用户观察值仅作输入，不作本轮测量值。
