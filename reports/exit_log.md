[23:19:40] ============================================================
[23:19:40] 夜间 ExIt 驱动启动（R1-R3，截止 08:00）
[23:19:40] $ /Users/ligf9/zhuoji-mahjong-ai/.venv/bin/python train_exit.py --data runs/exit_r01/exit_w*.npz --out /Users/ligf9/zhuoji-mahjong-ai/models/exit_r01.pt --epochs 30 --window 400000 --batch 256 --seed 20261107  → train_r01.log
[23:37:34]   rc=0  1074s  train_r01
[23:37:34] R1 训练完成 init=冷启动 → exit_r01.pt  (['[done] best val 1.7711 → /Users/ligf9/zhuoji-mahjong-ai/models/exit_r01.pt'])
[23:37:34] $ /Users/ligf9/zhuoji-mahjong-ai/.venv/bin/python eval_pairs.py --a /Users/ligf9/zhuoji-mahjong-ai/models/exit_r01.pt --b /Users/ligf9/zhuoji-mahjong-ai/models/dushan_master.pt --mode greedy --seeds 300 --seed0 710000 --workers 6 --out /Users/ligf9/zhuoji-mahjong-ai/runs/gates/r01_gateA.jsonl  → gateA_r01.log
[23:38:30]   rc=0  56s  gateA_r01
[23:38:30] R1 闸门 A: [summary] seats=[0, 1, 2, 3]  n=300  diff=+0.179±0.406  z=+0.44  p≈0.6599  符号 152:146  → 不通过
[23:38:30] $ /Users/ligf9/zhuoji-mahjong-ai/.venv/bin/python eval_pairs.py --a /Users/ligf9/zhuoji-mahjong-ai/models/exit_r01.pt --b /Users/ligf9/zhuoji-mahjong-ai/models/dushan_master.pt --a-mode greedy --b-mode search32 --seeds 50 --seed0 720000 --workers 6 --out /Users/ligf9/zhuoji-mahjong-ai/runs/gates/r01_gateC.jsonl  → gateC_r01.log
[00:00:29]   rc=0  1319s  gateC_r01
[00:00:29] R1 闸门 C(趋势): [summary] seats=[0, 1, 2, 3]  n=50  diff=-2.480±0.863  z=-2.87  p≈0.0041  符号 22:28
[00:00:29] $ /Users/ligf9/zhuoji-mahjong-ai/.venv/bin/python -c import torch,sys;sys.path.insert(0,'.');from zhuoji.net import build_from_ckpt;from zhuoji.forward import ensure_onnx;ck=torch.load('/Users/ligf9/zhuoji-mahjong-ai/models/exit_r01.pt',map_location='cpu',weights_only=False);net,_=build_from_ckpt(ck);p=ensure_onnx(net,stem='exit_r01');print('onnx:',p)  → export_exit_r01.log
[00:00:31]   rc=0  1s  export_exit_r01
[00:00:31] $ /Users/ligf9/zhuoji-mahjong-ai/rust_engine/target/release/annotate --student /Users/ligf9/zhuoji-mahjong-ai/models/onnx/exit_r01.onnx --teacher models/onnx/dushan_master.onnx --targets 100000 --threads 4 --out /Users/ligf9/zhuoji-mahjong-ai/runs/exit_r02 --seed0 620000  → annotate_r02.log
[00:00:39]   rc=101  8s  annotate_r02
[00:00:39] R2 阶段异常：AssertionError: the following indices\n index: 1 Got: 15 Expected: 63\n Please fix either the inputs/outputs or the model.", ptr: 0x91a5bb520 }
note: run with `RUST_BACKTRACE=1` environment variable to display a backtrace

thread '<unnamed>' (6611696) panicked at src/bin/annotate.rs:237:47:
学生前向失败: Error { code: InvalidArgument, message: "Got invalid dimensions for input: tiles for the following indices\n index: 1 Got: 15 Expected: 63\n Please fix either the inputs/outputs or the model.", ptr: 0x8f8b91fe0 }
Context leak detected, CoreAnalytics returned false
Context leak detected, CoreAnalytics returned false
Context leak detected, CoreAnalytics returned false
Context leak detected, CoreAnalytics returned false
Context leak detected, CoreAnalytics returned false
Context leak detected, CoreAnalytics returned false
Context leak detected, CoreAnalytics returned false
Context leak detected, CoreAnalytics returned false
Context leak detected, CoreAnalytics returned false
Context leak detected, CoreAnalytics returned false
Context leak detected, CoreAnalytics returned false
Context leak detected, CoreAnalytics returned false
Context leak detected, CoreAnalytics returned false
Context leak detected, CoreAnalytics returned false
Context leak detected, CoreAnalytics returned false
Context leak detected, CoreAnalytics returned false

thread '<unnamed>' (6611695) panicked at src/bin/annotate.rs:237:47:
学生前向失败: Error { code: InvalidArgument, message: "Got invalid dimensions for input: tiles for the following indices\n index: 1 Got: 15 Expected: 63\n Please fix either the inputs/outputs or the model.", ptr: 0x8f9669e00 }

thread '<unnamed>' (6611697) panicked at src/bin/annotate.rs:237:47:
学生前向失败: Error { code: InvalidArgument, message: "Got invalid dimensions for input: tiles for the following indices\n index: 1 Got: 15 Expected: 63\n Please fix either the inputs/outputs or the model.", ptr: 0x9175b86e0 }

thread 'main' (6611691) panicked at src/bin/annotate.rs:550:5:
a scoped thread panicked

[00:00:39] $ /Users/ligf9/zhuoji-mahjong-ai/.venv/bin/python -c import torch,sys;sys.path.insert(0,'.');from zhuoji.net import build_from_ckpt;from zhuoji.forward import ensure_onnx;ck=torch.load('/Users/ligf9/zhuoji-mahjong-ai/models/exit_r01.pt',map_location='cpu',weights_only=False);net,_=build_from_ckpt(ck);p=ensure_onnx(net,stem='exit_r01');print('onnx:',p)  → export_exit_r01.log
[00:00:40]   rc=0  1s  export_exit_r01
[00:00:40] $ /Users/ligf9/zhuoji-mahjong-ai/rust_engine/target/release/annotate --student /Users/ligf9/zhuoji-mahjong-ai/models/onnx/exit_r01.onnx --teacher models/onnx/dushan_master.onnx --targets 100000 --threads 4 --out /Users/ligf9/zhuoji-mahjong-ai/runs/exit_r03 --seed0 630000  → annotate_r03.log
[00:00:48]   rc=101  8s  annotate_r03
[00:00:48] R3 阶段异常：AssertionError:  leak detected, CoreAnalytics returned false
Context leak detected, CoreAnalytics returned false
Context leak detected, CoreAnalytics returned false
Context leak detected, CoreAnalytics returned false
Context leak detected, CoreAnalytics returned false
Context leak detected, CoreAnalytics returned false
Context leak detected, CoreAnalytics returned false
Context leak detected, CoreAnalytics returned false
Context leak detected, CoreAnalytics returned false
Context leak detected, CoreAnalytics returned false
Context leak detected, CoreAnalytics returned false

thread '<unnamed>' (6611946) panicked at src/bin/annotate.rs:237:47:
学生前向失败: Error { code: InvalidArgument, message: "Got invalid dimensions for input: tiles for the following indices\n index: 1 Got: 15 Expected: 63\n Please fix either the inputs/outputs or the model.", ptr: 0x949aa15e0 }
note: run with `RUST_BACKTRACE=1` environment variable to display a backtrace

thread '<unnamed>' (6611948) panicked at src/bin/annotate.rs:237:47:
学生前向失败: Error { code: InvalidArgument, message: "Got invalid dimensions for input: tiles for the following indices\n index: 1 Got: 15 Expected: 63\n Please fix either the inputs/outputs or the model.", ptr: 0x92df49860 }
Context leak detected, CoreAnalytics returned false
Context leak detected, CoreAnalytics returned false

thread '<unnamed>' (6611947) panicked at src/bin/annotate.rs:237:47:
学生前向失败: Error { code: InvalidArgument, message: "Got invalid dimensions for input: tiles for the following indices\n index: 1 Got: 15 Expected: 63\n Please fix either the inputs/outputs or the model.", ptr: 0x92d5c6b20 }

thread '<unnamed>' (6611949) panicked at src/bin/annotate.rs:237:47:
学生前向失败: Error { code: InvalidArgument, message: "Got invalid dimensions for input: tiles for the following indices\n index: 1 Got: 15 Expected: 63\n Please fix either the inputs/outputs or the model.", ptr: 0x921f8d5e0 }

thread 'main' (6611942) panicked at src/bin/annotate.rs:550:5:
a scoped thread panicked

[00:00:48] 夜间驱动结束 failures=2  best=无（未晋升）
[00:07:22] ============================================================
[00:07:22] 夜间 ExIt 驱动续跑（R2-R6，截止 08:00；R1 已完成）
[00:07:22] $ /Users/ligf9/zhuoji-mahjong-ai/.venv/bin/python -c import torch,sys;sys.path.insert(0,'.');from zhuoji.net import build_from_ckpt;from zhuoji.forward import ensure_onnx;ck=torch.load('/Users/ligf9/zhuoji-mahjong-ai/models/exit_r01.pt',map_location='cpu',weights_only=False);net,_=build_from_ckpt(ck);p=ensure_onnx(net,stem='exit_r01');print('onnx:',p)  → export_exit_r01.log
[00:07:23]   rc=0  1s  export_exit_r01
[00:07:23] $ /Users/ligf9/zhuoji-mahjong-ai/rust_engine/target/release/annotate --student /Users/ligf9/zhuoji-mahjong-ai/models/onnx/exit_r01.onnx --teacher models/onnx/dushan_master.onnx --student-enc 3 --targets 100000 --threads 4 --out /Users/ligf9/zhuoji-mahjong-ai/runs/exit_r02 --seed0 620000  → annotate_r02.log
[02:22:37] ============================================================
[02:22:37] 夜间 ExIt 驱动续跑（R2-R6，截止 08:00；R1 已完成）
[02:22:37] $ /Users/ligf9/zhuoji-mahjong-ai/.venv/bin/python -c import torch,sys;sys.path.insert(0,'.');from zhuoji.net import build_from_ckpt;from zhuoji.forward import ensure_onnx;ck=torch.load('/Users/ligf9/zhuoji-mahjong-ai/models/exit_r01.pt',map_location='cpu',weights_only=False);net,_=build_from_ckpt(ck);p=ensure_onnx(net,stem='exit_r01');print('onnx:',p)  → export_exit_r01.log
[02:22:38]   rc=0  1s  export_exit_r01
[02:22:38] $ /Users/ligf9/zhuoji-mahjong-ai/rust_engine/target/release/annotate --student /Users/ligf9/zhuoji-mahjong-ai/models/onnx/exit_r01.onnx --teacher models/onnx/dushan_master.onnx --student-enc 3 --student-device cpu --targets 100000 --threads 4 --out /Users/ligf9/zhuoji-mahjong-ai/runs/exit_r02 --seed0 620000  → annotate_r02.log
[2026-11-06 02:33:57] ============================================================
[2026-11-06 02:33:57] 7×24 ExIt 驱动启动：R2 起，截止 2026-11-13 02:30（R1 已完成；断点续跑 + 每小时自检自动化）
[2026-11-06 02:33:57] $ /Users/ligf9/zhuoji-mahjong-ai/.venv/bin/python -c import torch,sys;sys.path.insert(0,'.');from zhuoji.net import build_from_ckpt;from zhuoji.forward import ensure_onnx;ck=torch.load('/Users/ligf9/zhuoji-mahjong-ai/models/exit_r01.pt',map_location='cpu',weights_only=False);net,_=build_from_ckpt(ck);p=ensure_onnx(net,stem='exit_r01');print('onnx:',p)  → export_exit_r01.log
[2026-11-06 02:33:58]   rc=0  1s  export_exit_r01
[2026-11-06 02:33:58] $ /Users/ligf9/zhuoji-mahjong-ai/rust_engine/target/release/annotate --student /Users/ligf9/zhuoji-mahjong-ai/models/onnx/exit_r01.onnx --teacher models/onnx/dushan_master.onnx --student-enc 3 --student-device cpu --targets 100000 --threads 4 --out /Users/ligf9/zhuoji-mahjong-ai/runs/exit_r02 --seed0 620000  → annotate_r02.log
[2026-11-06 02:34:00] ============================================================
[2026-11-06 02:34:00] 运行模式升级：7×24 连续训练（原 08:00 时间盒作废）
[2026-11-06 02:34:00] 新驱动 night_run.py 启动，R2 起、DEADLINE=2026-11-13 02:30、
[2026-11-06 02:34:00] MAX_ROUND=60、断点续跑 + night_run.lock 30s 心跳防重入。
[2026-11-06 02:34:00] 旧驱动（02:22-02:33 的 R2 标注）未落盘，无数据损失；同 seed0
[2026-11-06 02:34:00] 620000 重采，保证可复现。标注超时放宽至 3.5h。
[2026-11-06 02:34:00] 配套：每小时自检自动化（03:32 首次，至 11-13 02:30）。
[2026-11-06 03:35:40] ---- 每小时自检 #1（7×24 起后首次，03:32 触发）----
[2026-11-06 03:35:40] 驱动存活：night_run.lock 心跳 9s、annotate_r02.log 3s 前更新、ps 见
                        night_run.py(PID 79443) + annotate(PID 79445)。判定：在跑 → 只自检/报告/更新计划，
                        不启第二驱动、不打断标注/训练/闸门、不并发跑任何评估。
[2026-11-06 03:35:40] 阶段：R2 DAgger 标注（学生 exit_r01.onnx rollout + N32 教师 Q，100k / 4 分片）。
                        进度 46,762/100,000（w0 11729 / w1 11715 / w2 11691 / w3 11627），已跑 1.03h，
                        吞吐 11,396 标注/h·worker（4 线程聚合 45,584/h），落在 10~12.5k 健康区间；
                        ETA 04:45，距 3.5h 超时上限 06:04 有 1h19m 余量。
                        启动日志 [annotate] ORT ready (teacher_coreml=true, student_coreml=false)，
                        教师 CoreML / 学生 CPU + --student-enc 3，已知坑全部规避。
[2026-11-06 03:35:40] 健康核对：当前 annotate_r02.log panic/NaN/超时 0；磁盘 311Gi 可用(29%)；
                        R1 数据 4×25,000 行、tiles(63,27)、glob(130)、q 的 NaN 与 mask 精确互补
                        （非法动作槽位约定，非 bug）；闸门 jsonl 行数 A=1200(300×4) / C=200(50×4)，
                        seed 池固定 A=710000+300、C=720000+50、四座旋转、冻结基线 master 贪心/N32。
[2026-11-06 03:35:40] R1 复核：train loss 2.4219→1.6633，val 2.2450→1.7711（ep6 最佳，ep9 patience3 早停），
                        ckpt=exit_r01.pt(9.84M M1/v3)；闸门 A diff=+0.179±0.406 z=+0.44 p≈0.6599 152:146
                        → 不通过、未晋升（无 exit_best.pt）；闸门 C diff=-2.480±0.863 z=-2.87 p≈0.0041 22:28
                        → 显著弱于 N32，记为「搜索蒸进网络」初始差距基线。
                        eval_pairs.py:163 判据核对 = p<0.05 且均值>0；night_run._gate_pass("判定: 通过")
                        与之匹配正确（"判定: 不通过" 不含该子串），无晋升逻辑 bug。
[2026-11-06 03:35:40] 决策①（下一步训练）：R2 标注落盘后由驱动自动续跑 R2 训练（200k 窗口，
                        --init exit_r01.pt）→ 闸门 A/C → R3，无需人工干预，本次不启新驱动。
[2026-11-06 03:35:40] 决策②（增速三选一）：不触发。计划要求「连续两轮闸门 A 不显著或均值≤0」，
                        目前仅 R1 一点（p=0.6599、均值 +0.179>0），R2 结果未出；提前改
                        τ/Q-MSE/n_det 会污染对照并违反「先 1 后 2 再 3、禁止同时改两项」。
                        决策点顺延至 R2 闸门 A 出结果后（预计 05:30 前后，下一次自检判断）。
[2026-11-06 03:35:40] 决策③（Oracle Guiding / PPO-lite 分支）：不推进。闸门 C 仅 1 个数据点
                        （-2.480±0.863），不足以判断「差距持续收敛」，需 R2/R3 至少两点成趋势再定。
[2026-11-06 03:35:40] 计划调整：docs/14天训练计划.md 追加 §6「7×24 模式校准」——原 D1–D14 日程
                        （11-19 截止、每天 2 轮 R1–R16）与实际（11-13 02:30、MAX_ROUND=60）已脱节，
                        按实测 ≈2.9h/轮重排；登记驱动与计划的 6 项阶段缺口（闸门 B、正对照、增速开关、
                        aux、lr 5e-4、Oracle/PPO）及补跑时点与优先级；红线：任何评估不得与标注并发
                        （会把 11.4k/h·worker 压到 <8k 触发异常判据）。
[2026-11-06 03:35:40] 下次自检关注：R2 是否 04:45 前落盘 4×25k；R2 train/val 是否继续下降；
                        R2 闸门 A 的 p/均值与闸门 C 差距是否自 -2.48 收敛；若 R2 亦不显著，按 §6.4
                        执行 ① τ 1.0→0.5（仅 train_exit.py --tau，禁止同时动其他项）。
[2026-11-06 04:42:30] ---- 每小时自检 #2（7×24 第 2 次；驱动在跑，只自检/报告/更新计划）----
[2026-11-06 04:42:30] 驱动存活：night_run.lock 心跳 12s、annotate_r02.log 4s 前更新、ps 见
                        night_run.py(PID 79443) + annotate(PID 79445, 74% CPU)。判定：在跑 →
                        不启第二驱动、不打断标注/训练/闸门、不并发跑任何评估。
[2026-11-06 04:42:30] 阶段：R2 DAgger 标注 97.2%（04:41:08 采样 w0 24392 / w1 24231 /
                        w2 24273 / w3 24297 ≈ 97,193/100k），已跑 7626s≈2h07m，
                        吞吐 11,483~11,516 标注/h·worker（4 线程聚合 ≈46,000/h），
                        落在 10~12.5k 健康区间；ETA 04:44~04:45，距 3.5h 超时上限 06:04
                        有 1h19m 余量。自检 #1 的 ETA 04:45 得到验证。
[2026-11-06 04:42:30] 健康核对：当前 annotate_r02.log panic/NaN/失败/Error/timeout 全 0；
                        日志秒级更新（远优于 10min 判据）；磁盘 311Gi 可用(29%)；
                        R1 数据 4×25,000、tiles(63,27)、glob(130)、q 的 NaN 与 mask 4/4 精确互补；
                        闸门 A 1200 行 / C 200 行；驱动代码复核 --student-enc 3、
                        --student-device cpu、教师 CoreML、seed0=620000，已知坑全部规避。
[2026-11-06 04:42:30] 闸门统计独立复算（analyze.py 口径 deltas[cand]-mean(其余三家)，
                        按 seed 四座平均，n=300/50）：A +0.179±0.406 z=+0.44 p≈0.6599；
                        C -2.480±0.863 z=-2.87 p≈0.0041 —— 与驱动 summary 逐位一致，
                        固定 seed 池 / 四座旋转 / 冻结基线协议与 p<0.05 且均值>0 判据实现无误。
[2026-11-06 04:42:30] 最近产出：R1 全齐（100k 标注、exit_r01.pt 39.5MB、
                        train 2.4219→1.6633、val 2.2450→1.7711 ep6 早停、闸门 A/C）。
                        R2 数据尚未落盘属正常（标注器结束一次性落盘）；无 exit_best.pt（R1 未晋升）。
                        产出符合预期。
[2026-11-06 04:42:30] 决策①（下一步训练）：R2 标注落盘后由驱动自动接续 R2 训练
                        （400k 滑窗 / --init exit_r01.pt / 30 epoch patience3 / lr 1e-3 / τ 1.0）
                        → 闸门 A/C → R3。本次不启新驱动、不打断、不并发评估，零人工干预。
[2026-11-06 04:42:30] 决策②（增速三选一）：不触发。R2 闸门 A 未出，目前仅 R1 一点
                        （p≈0.6599、均值 +0.179>0），未达「连续两轮 p>0.2 或均值≤0」。
                        决策点仍为 R2 闸门 A（预计 05:10~05:30）；若届时亦不显著 →
                        执行 ① τ 1.0→0.5，且仅此一项，禁止同时动 Q-MSE / n_det / lr。
[2026-11-06 04:42:30] 决策③（Oracle Guiding / PPO-lite）：不推进。闸门 C 仅 1 点
                        （-2.480±0.863），无法判断「差距持续收敛」，需 R2/R3 至少两点成趋势。
[2026-11-06 04:42:30] 计划调整（§6.6 落地校准，重要）：修正「改 night_run.py 即生效」的隐含错误——
                        运行中驱动是长驻单进程、不重载源码，编辑 night_run.py 只对下次重启生效；
                        τ 1.0→0.5 应改为编辑 train_exit.py:38 的 TAU 常量（每轮训练是新进程，
                        自动重读，零重启、零并发、零打断），这是唯一符合硬约束的落地路径。
                        闸门 B / 正对照改为「驱动重启窗口或终评统一口径」补跑，
                        不在 7×24 标注期内抢 CPU；详见 docs/14天训练计划.md §6.6。
[2026-11-06 04:42:30] 下次自检关注：R2 是否 04:45 前落盘 4×25k；R2 train/val 是否继续下降；
                        R2 闸门 A 的 p/均值、闸门 C 是否自 -2.48 收敛；若连续两轮不显著
                        立即按 §6.4 ① 改 TAU=0.5（仅此一项）。
[2026-11-06 04:44:37]   rc=0  7839s  annotate_r02
[2026-11-06 04:44:37] R2 标注完成 student=exit_r01.onnx seed0=620000 → 4×25k
[2026-11-06 04:44:37] $ /Users/ligf9/zhuoji-mahjong-ai/.venv/bin/python train_exit.py --data runs/exit_r01/exit_w*.npz runs/exit_r02/exit_w*.npz --out /Users/ligf9/zhuoji-mahjong-ai/models/exit_r02.pt --epochs 30 --window 400000 --batch 256 --seed 20261108 --init /Users/ligf9/zhuoji-mahjong-ai/models/exit_r01.pt  → train_r02.log
[2026-11-06 04:45:20] ---- 自检 #2 收尾：捕获 R2 标注→训练交接（全程零干预）----
[2026-11-06 04:45:20] R2 标注 04:44:37 落盘完成 rc=0、耗时 7839s≈2h10m39s，
                        聚合吞吐 45,914/h（≈11,479/h·worker），与自检 #1 预测 04:45 一致。
[2026-11-06 04:45:20] R2 数据合规：exit_w{0..3}.npz 各 25,000 行（合计 100,000），
                        tiles(25000,63,27)、glob(25000,130)、q/mask(25000,28)，
                        q 的 NaN 与 mask 4/4 精确互补，满足 data_ready 的 ≥24k 判据。
[2026-11-06 04:45:20] 下一步训练已由驱动自动启动（本次未干预）：
                        train_exit.py --data R1+R2 两窗口 --out exit_r02.pt
                        --epochs 30 --window 400000 --batch 256 --seed 20261108
                        --init exit_r01.pt（200k 实际窗口、续训自 R1、τ1.0/lr1e-3 未改）。
                        train_exit.py PID 89576 在跑（23% CPU），日志仅 1 条无害
                        UserWarning(float(loss) 的 requires_grad 提示，R1 同款)，
                        panic/NaN/Error/Traceback 全 0；annotate 已退出，lock 心跳持续。
[2026-11-06 04:45:20] 自检竞态说明（非训练异常）：首次读 exit_w3.npz 时该分片尚在写入
                        （4.4MB vs 其余 9.7MB）触发 BadZipFile；标注器四分片逐个落盘，
                        驱动在 annotate 进程退出后才检查 data_ready（内含 try/except），不受影响。
                        等 15s 复核 4/4 全合规。后续自检须等进程退出或分片大小稳定再读 npz。
[2026-11-06 04:45:20] 决策不变：不启第二驱动、不打断、不并发评估；τ/Q-MSE/n_det、
                        Oracle/PPO 全部维持原状，硬决策点仍为 R2 闸门 A（05:10~05:30）。
[2026-11-06 05:23:33]   rc=0  2336s  train_r02
[2026-11-06 05:23:33] R2 训练完成 init=exit_r01.pt → exit_r02.pt  (['[done] best val 1.7661 → /Users/ligf9/zhuoji-mahjong-ai/models/exit_r02.pt'])
[2026-11-06 05:23:33] $ /Users/ligf9/zhuoji-mahjong-ai/.venv/bin/python eval_pairs.py --a /Users/ligf9/zhuoji-mahjong-ai/models/exit_r02.pt --b /Users/ligf9/zhuoji-mahjong-ai/models/dushan_master.pt --mode greedy --seeds 300 --seed0 710000 --workers 6 --out /Users/ligf9/zhuoji-mahjong-ai/runs/gates/r02_gateA.jsonl  → gateA_r02.log
[2026-11-06 05:24:29]   rc=0  56s  gateA_r02
[2026-11-06 05:24:29] R2 闸门 A: [summary] seats=[0, 1, 2, 3]  n=300  diff=+2.374±0.415  z=+5.72  p≈0.0000  符号 179:117  → 通过
[2026-11-06 05:24:29] $ /Users/ligf9/zhuoji-mahjong-ai/.venv/bin/python -c import torch,sys;sys.path.insert(0,'.');from zhuoji.net import build_from_ckpt;from zhuoji.forward import ensure_onnx;ck=torch.load('/Users/ligf9/zhuoji-mahjong-ai/models/exit_best.pt',map_location='cpu',weights_only=False);net,_=build_from_ckpt(ck);p=ensure_onnx(net,stem='exit_best');print('onnx:',p)  → export_exit_best.log
[2026-11-06 05:24:30]   rc=0  1s  export_exit_best
[2026-11-06 05:24:30] 晋升：exit_r02.pt → exit_best.pt
[2026-11-06 05:24:30] $ /Users/ligf9/zhuoji-mahjong-ai/.venv/bin/python eval_pairs.py --a /Users/ligf9/zhuoji-mahjong-ai/models/exit_r02.pt --b /Users/ligf9/zhuoji-mahjong-ai/models/dushan_master.pt --a-mode greedy --b-mode search32 --seeds 50 --seed0 720000 --workers 6 --out /Users/ligf9/zhuoji-mahjong-ai/runs/gates/r02_gateC.jsonl  → gateC_r02.log
[2026-11-06 05:46:44]   rc=0  1334s  gateC_r02
[2026-11-06 05:46:44] R2 闸门 C(趋势): [summary] seats=[0, 1, 2, 3]  n=50  diff=-0.293±1.125  z=-0.26  p≈0.7944  符号 22:28
[2026-11-06 05:46:44] R2 轮次完成  best=exit_best.pt
[2026-11-06 05:46:44] $ /Users/ligf9/zhuoji-mahjong-ai/.venv/bin/python -c import torch,sys;sys.path.insert(0,'.');from zhuoji.net import build_from_ckpt;from zhuoji.forward import ensure_onnx;ck=torch.load('/Users/ligf9/zhuoji-mahjong-ai/models/exit_best.pt',map_location='cpu',weights_only=False);net,_=build_from_ckpt(ck);p=ensure_onnx(net,stem='exit_best');print('onnx:',p)  → export_exit_best.log
[2026-11-06 05:46:45]   rc=0  1s  export_exit_best
[2026-11-06 05:46:45] $ /Users/ligf9/zhuoji-mahjong-ai/rust_engine/target/release/annotate --student /Users/ligf9/zhuoji-mahjong-ai/models/onnx/exit_best.onnx --teacher models/onnx/dushan_master.onnx --student-enc 3 --student-device cpu --targets 100000 --threads 4 --out /Users/ligf9/zhuoji-mahjong-ai/runs/exit_r03 --seed0 630000  → annotate_r03.log
[2026-11-06 05:47:40] ---- 每小时自检 #3（7×24 第 3 次；驱动在跑，只自检/报告/更新计划）----
[2026-11-06 05:47:40] 驱动存活：night_run.lock 心跳 4~5s、annotate_r03.log 秒级更新、ps 见
                        night_run.py(PID 79443，已跑 3h13m) + annotate(PID 94304，85.6% CPU)。
                        判定：在跑 → 不启第二驱动、不打断标注/训练/闸门、不并发跑任何评估。
[2026-11-06 05:47:40] 阶段：R3 DAgger 标注（05:46:45 启动，学生首次用 exit_best.onnx、
                        seed0=630000；教师 CoreML / 学生 CPU + --student-enc 3，已知坑全规避）。
                        吞吐 11,106~11,792 标注/h·worker（健康区间 10~12.5k），
                        panic/NaN/Error/timeout 全 0，日志秒级更新，ETA ≈08:00，
                        距 3.5h 上限 09:16 有 1h16m 余量。
[2026-11-06 05:47:40] R2 完整产出（本次核心新增）：标注 04:44:37 落盘 4×25,000=100,000；
                        训练 05:23:33 完成 init=exit_r01.pt、200k 实际窗口、
                        train 1.7914→1.6924、best val 1.7661(ep6)/ep9 patience3 早停 → exit_r02.pt。
[2026-11-06 05:47:40] R2 闸门 A：+2.374±0.415 z=+5.72 p≈0.0000 符号 179:117 → **通过、晋升**，
                        05:24:30 exit_r02.pt → exit_best.pt（shasum-256 逐位一致），r02.promoted=ok，
                        exit_best.onnx 已导出(tiles[B,63,27]/glob[B,130])。
[2026-11-06 05:47:40] R2 闸门 C：−0.293±1.125 z=−0.26 p≈0.7944 → 自 R1 −2.480 收敛 2.187 分(88%)，
                        已不显著；新贪心基本追平 N32，「搜索蒸进网络」接近达成。
[2026-11-06 05:47:40] 独立复算 4 份闸门 jsonl（deltas[cand]−mean(其余三家) 按 seed 四座平均）
                        与驱动 summary 逐位一致：A r01 +0.179±0.406 p=0.6599 / r02 +2.374±0.415 p=0.0000；
                        C r01 −2.480±0.863 p=0.0041 / r02 −0.293±1.125 p=0.7944；
                        行数 1200 / 200、seats[0,1,2,3]、seed 池 710000+300 / 720000+50 全合规。
[2026-11-06 05:47:40] npz 合规：r01/r02 各 4×25,000，tiles(63,27)/glob(130)/q,mask(28)，
                        q 的 NaN 与 mask 8/8 分片精确互补（非法动作槽位约定，非 bug）；
                        磁盘 311Gi 可用(29%)；TAU=1.0、lr=1e-3 未改。结论：**健康，产出超预期**。
[2026-11-06 05:47:40] 决策①（下一步训练）：R3 标注已由驱动自动启动，本次零干预；
                        后续 R3 训练(300k 窗口、init=exit_best.pt) → 闸门 A/C → R4 由驱动接续。
                        首次以「被认可的最佳 ckpt」作 DAgger rollout 学生，
                        自检 #1/#2 记录的「latest_pt() 回退最新轮」隐患自动闭环。
[2026-11-06 05:47:40] 决策②（增速三选一）：**不触发**。R2 闸门 A 显著通过(p≈0.0000、均值+2.374>0)，
                        不满足「连续两轮不显著或均值≤0」；按 §6.6.3「通过 → 不改任何超参」，
                        τ / Q-MSE / n_det / lr 全部维持原值。明确：R3 单轮回落亦不构成
                        「连续两轮」（R2 已显著），禁止据此提前改 TAU=0.5。
[2026-11-06 05:47:40] 决策③（Oracle Guiding / PPO-lite）：**本次不激活，进入就绪态**。
                        闸门 C 仅 1 段收敛(R1→R2)，§6.4 要求「连续 ≥2 轮收敛」需 R3 点确认；
                        PPO-lite 按 §6.4 约 R10+ 且须先完成 wrapper 定稿+单测，现在太早。
                        P2a 可执行规格、双条件判据与落地时点已固化进计划 §6.7。
[2026-11-06 05:47:40] 决策④（闸门 B）：维持 §6.6.2，R3 末(r%3==0)不在主循环并发补跑，
                        线上等效口径由 11-13 后终评 500 seed 统一补跑；逐轮趋势由闸门 C 承担。
[2026-11-06 05:47:40] 计划调整(§6.7)：①预算修正 R2 实测 3.22h/轮、R4+ ≈3.87h/轮 →
                        168h 实际约 44 轮（非 55），MAX_ROUND=60 不会先触发、DEADLINE 为硬约束；
                        ②val loss 跨轮不可比（窗口 100k→200k 且分布转为学生 rollout），
                        强度只认冻结基线闸门 A/C；③口径澄清：闸门 A +2.374 是
                        「新贪心 vs master 贪心」，非主目标「新+N32 vs master+N32」，
                        后者须闸门 B / 终评确认；闸门 C −0.293 不显著 ≈ 超额目标接近达成。
[2026-11-06 05:47:40] 硬决策点顺延：R3 闸门 C（预计 09:00~09:10 出结果，由 09:47/10:47 自检捕获）
                        与 R6 末增速复核。下次自检(≈06:47)只需核对 R3 标注进度/吞吐/异常，不提前决策。
[2026-11-06 07:00:00] ---- 每小时自检 #4（7×24 第 4 次；06:53 触发、06:54~07:01 采样；驱动在跑，只自检/报告/更新计划）----
[2026-11-06 07:00:00] 驱动存活：night_run.lock 心跳 11~28s、annotate_r03.log 20s 前更新、ps 见
                        night_run.py(PID 79443，已跑 4h20m) + annotate(PID 94304，67.8% CPU)。
                        判定：在跑 → 不启第二驱动、不打断标注、不并发跑任何评估。
[2026-11-06 07:00:00] 阶段：R3 DAgger 标注（05:46:45 启动，学生 exit_best.onnx、seed0=630000；
                        teacher_coreml=true / student_coreml=false、--student-enc 3 / --student-device cpu）。
                        进度 w0 13189 / w1 13066 / w2 13181 / w3 13202 ≈ 52,638/100k = 52.6%，
                        已跑 4195s≈1h10m，吞吐 11,319~11,360 标注/h·worker（聚合 ≈45.3k/h），
                        落在 10~12.5k 健康区间；ETA ≈08:00 落盘，距 3.5h 上限 09:16 有 1h16m 余量。
[2026-11-06 07:00:00] 健康核对：annotate_r03.log panic/NaN/Error/Traceback/timeout/fail 计数 0；
                        日志秒级更新（远优于 10min 判据）；磁盘 311Gi 可用(29%)；
                        R1/R2 共 8/8 npz 分片各 25,000 行、tiles(63,27)/glob(130)/q,mask(28)、
                        q 的 NaN 与 mask 精确互补；runs/exit_r03 尚空属正常（标注器结束一次性落盘）。
[2026-11-06 07:00:00] 最近产出：本次无新轮次产出（R3 仍在标注）。R2 全链路复算——
                        标注 04:44:37 落盘 100k / 7839s；训练 05:23:33 init=exit_r01.pt、200k 窗口、
                        train 1.7914→1.6924、best val 1.7661(ep6)；闸门 A +2.374±0.415 z=+5.72
                        p≈0.0000 179:117 → 05:24:30 晋升 exit_best.pt（shasum 与 exit_r02.pt 一致
                        69f59485…）；闸门 C −0.293±1.125 z=−0.26 p≈0.7944，自 R1 −2.480 收敛 88%。
[2026-11-06 07:00:00] 4 份闸门 jsonl 独立复算（deltas[cand]−mean(其余三家) 按 seed 四座平均）
                        与驱动 summary 逐位一致：A r01 +0.179±0.406 p=0.6599 / r02 +2.374±0.415 p=0.0000；
                        C r01 −2.480±0.863 p=0.0041 / r02 −0.293±1.125 p=0.7944；行数 1200/1200/200/200、
                        四座旋转、固定 seed 池 710000+300 / 720000+50 全合规。train_exit.py:38 TAU=1.0、
                        :85 lr=1e-3 只读核对未改。结论：**健康，产出符合预期**。
[2026-11-06 07:00:00] 决策①（下一步训练）：驱动在跑，R3 标注 → 训练(300k 窗口, init=exit_best.pt)
                        → 闸门 A/C → R4 由驱动自动接续，**本次零干预**。
[2026-11-06 07:00:00] 决策②（增速三选一）：不触发。R2 闸门 A 显著通过 → §6.6.3「通过 → 不改任何
                        超参」；§6.7.2 明确 R3 单轮回落 ≠ 连续两轮，禁止提前改 TAU=0.5。
                        决策③（Oracle / PPO）：不激活、保持就绪态（闸门 C 仅 1 段收敛，需 R3 点
                        确认第 2 段；PPO 约 R10+）。决策④（闸门 B）：维持 §6.6.2 终评统一口径补跑。
[2026-11-06 07:00:00] 本次新增排查（伪风险，已排除，不改代码）：night_run.py:183 seed0=600000+r*10000
                        配合 annotate.rs:294 `Rng::new(seed0+tid*10007)`，R8 起部分 worker 的
                        rollout RNG 种子编号会扫入 710000/720000 评估编号区间（R11/R12 的 w0
                        更直接等于闸门 A/C 的 seed0）。核实：标注用 Rust xorshift64*+Fisher-Yates
                        （rng.rs 注「无需与 Python 对齐」、单条长 RNG 流连续发牌），评估用
                        Python random.Random(seed).shuffle（rules.py:146/168、每 seed 重置），
                        PRNG 不同族且洗牌/发牌流语义不同 ⇒ 编号相同也不会生成相同牌墙，
                        评估 seed 池与 rollout 的隔离在实现层成立。**不修改 night_run.py**
                        （对长驻进程无效，§6.6.1），登记入计划 §6.8 防后续自检误判为协议违规。
[2026-11-06 07:00:00] 计划校准（§6.8）：①窗口截断已核实 train_exit.py:58-60 `x[-window:]`、
                        pats 按轮次升序 ⇒ 400k 滑窗只留最近 4 轮，§6.7.3「R4+ 1.30h 封顶、
                        3.87h/轮、168h≈44 轮」由假设升级为验证结论；②R3 时间线精修：
                        标注 ETA 08:00 → 训练 ≈08:58 → 闸门 A 08:59 → 闸门 C ≈09:21，
                        硬决策点修正为 ≈09:21（原写 09:00~09:10），由 09:47 自检捕获、10:47 兜底；
                        ③长期洁净度：逐轮 A/C 复用同一 300/50 seed 池 44 轮有多重比较假阳性累积，
                        终评须用全新独立 500 seed 池（建议 3_000_000+500），最终显著性以终评为准，
                        逐轮闸门仅作趋势；④磁盘每轮≈118MB×44≈5.2GB，311Gi 余量充足。
[2026-11-06 07:00:00] 下次必做：07:47 核对 R3 标注进度/吞吐/异常（落盘前勿读 r03 npz，避免
                        BadZipFile 竞态）；08:47 核对 R3 训练 --init exit_best.pt / 300k 窗口 /
                        train-val 曲线；09:47 捕获 R3 闸门 A/C，按 §6.7.6 决策树判 P2a 就绪与增速。
[2026-11-06 07:59:09]   rc=0  7944s  annotate_r03
[2026-11-06 07:59:09] R3 标注完成 student=exit_best.onnx seed0=630000 → 4×25k
[2026-11-06 07:59:09] $ /Users/ligf9/zhuoji-mahjong-ai/.venv/bin/python train_exit.py --data runs/exit_r01/exit_w*.npz runs/exit_r02/exit_w*.npz runs/exit_r03/exit_w*.npz --out /Users/ligf9/zhuoji-mahjong-ai/models/exit_r03.pt --epochs 30 --window 400000 --batch 256 --seed 20261109 --init /Users/ligf9/zhuoji-mahjong-ai/models/exit_best.pt  → train_r03.log
[2026-11-06 08:12:26] ---- 每小时自检 #5（7×24 第 5 次；08:12 触发、08:12~08:20 采样；驱动在跑，只自检/报告/更新计划）----
[2026-11-06 08:12:26] 驱动存活：night_run.lock 心跳 8~9s、night_run.py PID 79443（已跑 5h43m）、
                        R3 训练 PID 5006（20min，9.4% CPU）、train_r03.log 08:18:42 更新。
                        判定：在跑 → 不启第二驱动、不打断训练、不并发跑任何评估。
[2026-11-06 08:12:26] 阶段：R3 训练（07:59:09 启动，data 12 files/300000 samples、400k 窗口、
                        --init exit_best.pt、--seed 20261109、30ep/patience3）。R3 标注已于
                        07:59:09 rc=0 完成，7944s≈2h12m，4×25k 一次性落盘。
[2026-11-06 08:12:26] 最近产出①（R3 标注）：r01/r02/r03 共 12/12 npz 分片各 25,000 行、
                        tiles(25000,63,27)/glob(25000,130)/q,mask(25000,28)、q 的 NaN 与 mask
                        4/4 精确互补（r03 qNaN 484,682~488,216）。末次吞吐 11,325~11,372
                        标注/h·worker（聚合 ≈45,317/h），落在 10~12.5k 健康区间；
                        annotate_r03.log panic/NaN/Error/fail 计数 0，日志秒级更新。
[2026-11-06 08:12:26] 最近产出②（R3 训练中期）：ep1 train 1.7647/val 1.7693、ep2 1.7510/1.7687
                        (best)、ep3 1.7433/1.7764 —— train 单调下降，val ep3 微升 0.0077，
                        由 patience3 早停处理，非异常（R2 同款）。每 ep ≈390s，
                        预计 08:32（ep5 早停）~08:58（ep9）结束。
                        exit_r03.pt 已写出（8e29f123…），但 exit_best.pt 仍 = exit_r02.pt
                        （ae506c68…，两者哈希一致）—— **写入 ≠ 晋升**，晋升唯一标志是
                        runs/gates/rNN.promoted + 闸门 A summary 通过。
[2026-11-06 08:12:26] 闸门复算（deltas[cand]−mean(其余三家) 按 seed 四座平均）与驱动逐位一致：
                        A r01 +0.179±0.406 p≈0.6599 / r02 +2.374±0.415 p≈0.0000（已晋升）；
                        C r01 −2.480±0.863 p≈0.0041 / r02 −0.293±1.125 p≈0.7944。
                        train_exit.py:38 TAU=1.0、night_run.py 的 --student-enc 3 /
                        --student-device cpu / A=710000+300 / C=720000+50 只读核对未改。
                        磁盘 311Gi 可用(29%)。结论：**健康，产出符合预期**。
[2026-11-06 08:12:26] 新增排查①（伪异常，已定性）：night_run.log 第 19~79 行有 8 条
                        `学生前向失败 tiles Got:15 Expected:63` panic，均在
                        `[00:00:48] 夜间驱动结束 failures=2` 之前，属 11-06 00:07/02:22 两次
                        早期失败尝试（未带 --student-enc 3）；02:33:57 正式 7×24 启动（第 93 行）
                        之后 panic/Error 计数 0。night_run.log 是 append 累积文件，
                        **后续自检不得因头部历史 panic 误判当前异常**，只按时间戳切片判定。
[2026-11-06 08:12:26] 新增排查②（ONNX 陈旧权重风险，已排除）：promote() 用固定 stem='exit_best'
                        调 ensure_onnx()，其缓存判据是 `onnx.mtime >= {stem}.pt.mtime`
                        （forward.py:73）。核实：promote 用 shutil.copy2 保留源 ckpt mtime ⇒
                        R3 若晋升，exit_best.pt mtime=08:12 > exit_best.onnx 05:24 ⇒ 触发重导出，
                        R4 rollout 拿到 R3 权重；若不通过，两者保持 R2 且 05:24>=05:10 正确复用。
                        **两分支均自洽，无陈旧学生权重风险**。但判据依赖 mtime 而非内容哈希，
                        属脆弱点（异常时间戳会静默复用旧 ONNX 且无报错），登记为低优先级健壮性项，
                        建议终评/自然重启窗口加固为内容哈希校验；**当前不改代码**（避免非必要干预）。
[2026-11-06 08:12:26] 决策①（下一步训练）：驱动在跑，R3 训练 → 闸门 A/C → R4 标注由
                        first_incomplete_round()/for 循环自动接续，**本次零干预、零重启**。
                        决策②（增速三选一）：不触发（R2 A 显著通过，R3 闸门未出；
                        R3+R4 连续两轮 p>0.2 或均值≤0 才仅改 TAU=0.5）。
                        决策③（Oracle/PPO）：不激活、保持就绪态（闸门 C 仅 1 段收敛，等 R3 点）。
                        决策④（闸门 B）：维持 §6.6.2 终评统一口径补跑。
[2026-11-06 08:12:26] 计划校准（§6.9）：R3 训练按实测 390s/ep，早停可能提前到 ≈08:32，
                        ⇒ 闸门 C 结果窗口修正为 ≈09:00~09:21（原单一 ≈09:21）。
                        下次自检（≈09:2x~09:47）捕获；若有 08:47 自检，只记录闸门 A 结果，
                        **不提前决策**（须等闸门 C 配对判第 2 段收敛）。
[2026-11-06 08:12:26] 下次必做：捕获 R3 闸门 A（p/均值/是否晋升）与闸门 C（是否自 −0.293
                        继续收敛 / 反弹显著为负），按 §6.7.6 决策树判 P2a 就绪与增速；
                        核对 R3 train/val 最终曲线与早停 epoch、r03.promoted 是否生成、
                        exit_best.onnx 是否随晋升更新；R6 末复核增速三选一。
[2026-11-06 08:51:03]   rc=0  3114s  train_r03
[2026-11-06 08:51:03] R3 训练完成 init=exit_best.pt → exit_r03.pt  (['[done] best val 1.7683 → /Users/ligf9/zhuoji-mahjong-ai/models/exit_r03.pt'])
[2026-11-06 08:51:03] $ /Users/ligf9/zhuoji-mahjong-ai/.venv/bin/python eval_pairs.py --a /Users/ligf9/zhuoji-mahjong-ai/models/exit_r03.pt --b /Users/ligf9/zhuoji-mahjong-ai/models/dushan_master.pt --mode greedy --seeds 300 --seed0 710000 --workers 6 --out /Users/ligf9/zhuoji-mahjong-ai/runs/gates/r03_gateA.jsonl  → gateA_r03.log
[2026-11-06 08:51:59]   rc=0  56s  gateA_r03
[2026-11-06 08:51:59] R3 闸门 A: [summary] seats=[0, 1, 2, 3]  n=300  diff=+2.843±0.422  z=+6.74  p≈0.0000  符号 186:108  → 通过
[2026-11-06 08:51:59] $ /Users/ligf9/zhuoji-mahjong-ai/.venv/bin/python -c import torch,sys;sys.path.insert(0,'.');from zhuoji.net import build_from_ckpt;from zhuoji.forward import ensure_onnx;ck=torch.load('/Users/ligf9/zhuoji-mahjong-ai/models/exit_best.pt',map_location='cpu',weights_only=False);net,_=build_from_ckpt(ck);p=ensure_onnx(net,stem='exit_best');print('onnx:',p)  → export_exit_best.log
[2026-11-06 08:52:00]   rc=0  1s  export_exit_best
[2026-11-06 08:52:00] 晋升：exit_r03.pt → exit_best.pt
[2026-11-06 08:52:00] $ /Users/ligf9/zhuoji-mahjong-ai/.venv/bin/python eval_pairs.py --a /Users/ligf9/zhuoji-mahjong-ai/models/exit_r03.pt --b /Users/ligf9/zhuoji-mahjong-ai/models/dushan_master.pt --a-mode greedy --b-mode search32 --seeds 50 --seed0 720000 --workers 6 --out /Users/ligf9/zhuoji-mahjong-ai/runs/gates/r03_gateC.jsonl  → gateC_r03.log
[2026-11-06 09:13:37]   rc=0  1296s  gateC_r03
[2026-11-06 09:13:37] R3 闸门 C(趋势): [summary] seats=[0, 1, 2, 3]  n=50  diff=+0.300±0.980  z=+0.31  p≈0.7595  符号 27:23
[2026-11-06 09:13:37] R3 轮次完成  best=exit_best.pt
[2026-11-06 09:13:37] $ /Users/ligf9/zhuoji-mahjong-ai/.venv/bin/python -c import torch,sys;sys.path.insert(0,'.');from zhuoji.net import build_from_ckpt;from zhuoji.forward import ensure_onnx;ck=torch.load('/Users/ligf9/zhuoji-mahjong-ai/models/exit_best.pt',map_location='cpu',weights_only=False);net,_=build_from_ckpt(ck);p=ensure_onnx(net,stem='exit_best');print('onnx:',p)  → export_exit_best.log
[2026-11-06 09:13:38]   rc=0  1s  export_exit_best
[2026-11-06 09:13:38] $ /Users/ligf9/zhuoji-mahjong-ai/rust_engine/target/release/annotate --student /Users/ligf9/zhuoji-mahjong-ai/models/onnx/exit_best.onnx --teacher models/onnx/dushan_master.onnx --student-enc 3 --student-device cpu --targets 100000 --threads 4 --out /Users/ligf9/zhuoji-mahjong-ai/runs/exit_r04 --seed0 640000  → annotate_r04.log
[2026-11-06 09:52:00] ---- 每小时自检 #6（7×24 第 6 次；09:41 触发、09:45~09:55 采样；驱动在跑，只自检/报告/更新计划）----
[2026-11-06 09:52:00] 驱动存活：night_run.lock 心跳 2~7s、night_run.py PID 79443（已跑 7h18m）+
                        annotate PID 10902（64.8% CPU，32min）、annotate_r04.log 秒级更新。
                        判定：在跑 → 不启第二驱动、不打断标注、不并发跑任何评估。
[2026-11-06 09:52:00] 阶段：R4 DAgger 标注（09:13:38 启动，学生 exit_best.onnx=R3 权重、
                        seed0=640000，--student-enc 3 / --student-device cpu、教师 CoreML）。
                        进度 w0 6193 / w1 6223 / w2 6175 / w3 6208 ≈ 24,799/100k（24.8%，1957s），
                        吞吐 11,376~11,481 标注/h·worker（聚合 ≈45.5k/h），ETA ≈11:26，
                        距 3.5h 超时上限 12:43 有 1h17m 余量。annotate_r04.log
                        panic/NaN/Error/timeout/fail 计数 0，日志秒级更新，磁盘 311Gi 可用(29%)。
[2026-11-06 09:52:00] 健康核对：按 02:33:57 正式 7×24 启动时间戳切片，13 个阶段全部 rc=0，
                        panic/Traceback/Timeout/阶段异常 计数 0（文件头部 8 条 tiles Got:15
                        是 00:07/02:22 早期失败启动残留，§6.9.2 已定性，非当前异常）。
                        6 份闸门 jsonl 行数 A 1200×3 / C 200×3、seats[0,1,2,3]、
                        固定 seed 池 A=710000..710299 / C=720000..720049 全合规。
[2026-11-06 09:52:00] 最近产出（R3 全链路，本次核心）：标注 07:59:09 rc=0 落盘 4×25k（7944s≈2h12m）；
                        训练 08:51:03 rc=0（3114s，300k 窗口、--init exit_best.pt、seed 20261109），
                        train 1.7647→1.7184 单调降、best val 1.7683(ep5)、ep8 patience3 早停；
                        闸门 A 08:51:59 +2.843±0.422 z=+6.74 p≈0.0000 186:108 → 通过，
                        08:52:00 晋升 exit_r03.pt → exit_best.pt（哈希 74f313eb… 逐位一致）、
                        r03.promoted=ok、exit_best.onnx 08:52:00 重导出；
                        闸门 C 09:13:37 +0.300±0.980 z=+0.31 p≈0.7595 27:23 → 首次转正，
                        自 R1 −2.480 → R2 −0.293 → R3 +0.300 连续 2 段收敛（累计 +2.780 分）。
[2026-11-06 09:52:00] 独立复算（6 份 jsonl，口径 deltas[cand]−mean(其余三家) 按 seed 四座平均）
                        与驱动 summary 核心统计量逐位一致：A r01 +0.179±0.406 p=0.6599 /
                        r02 +2.374±0.415 p=0.0000 / r03 +2.843±0.422 p=0.0000；
                        C r01 −2.480±0.863 p=0.0041 / r02 −0.293±1.125 p=0.7944 /
                        r03 +0.300±0.980 p=0.7595。符号计数因零值处理口径差 ±3，不影响判定。
[2026-11-06 09:52:00] 数据合规：r01/r02/r03 共 12/12 npz 分片各 25,000 行、tiles(63,27)/
                        glob(130)/q,mask(28)，q-NaN 与 mask 精确互补（NaN ↔ mask=0，
                        overlap=0 / gap=0，12/12 True）。**本次修正自检脚本口径**：历史写的
                        「NaN 与 mask 互补」指 NaN↔mask=0；若按 NaN↔mask=1 校验会全部 False，
                        属校验脚本口径错误而非数据异常，后续自检统一按 mask=0 判定。
[2026-11-06 09:52:00] 新增验证①（ONNX 陈旧权重，§6.9.3 由推断升级为实测）：exit_best.onnx
                        mtime 08:52:00 > exit_best.pt 08:31 ⇒ 触发重导出；用权重数值签名
                        （size+sum+前缀和）比对 174 个 initializer：exit_r03.pt 命中 92/338
                        float 张量、exit_r02.pt 命中 0/338 ⇒ R4 rollout 学生确为 R3 权重，
                        无陈旧权重。ONNX 输入 tiles[0,63,27]/glob[0,130] 动态 batch 合规。
[2026-11-06 09:52:00] 新增验证②（增益斜率，同冻结池逐 seed 配对，仅趋势、不作晋升依据）：
                        R3−R2 闸门 A +0.469±0.563 z=+0.83 p≈0.4053（148/300 seed 更优）、
                        闸门 C +0.593±1.432 z=+0.41 p≈0.6786（26/50 更优）。
                        ⇒ 绝对强度对 master 仍极显著（+2.843），但环比增量已不显著，
                        增益进入平台期；C 三段增量 +2.187 → +0.593 递减，
                        是「教师天花板临近」的定量证据。注意 C 的 +0.300 不显著(n=50)、
                        逐轮复用同一池有多重比较累积，**不宣布「超额目标达成」**，
                        以终评 500 seed 独立池为准。
[2026-11-06 09:52:00] 决策①（下一步训练）：驱动在跑，R4 标注 → 训练(400k 滑窗首次封顶、
                        --init exit_best.pt) → 闸门 A/C → R5 由 for 循环自动接续，
                        **本次零干预、零重启、零并发评估**。不启第二驱动。
[2026-11-06 09:52:00] 决策②（增速三选一）：**不触发**。R2/R3 闸门 A 连续两轮显著为正
                        （p≈0.0000、均值 +2.374/+2.843>0）；触发条件是连续两轮 p>0.2
                        或均值≤0。train_exit.py:38 TAU=1.0、:85 lr=1e-3 只读核对未改。
                        R6 末复核；单轮回落不得提前改 TAU=0.5。
[2026-11-06 09:52:00] 决策③（P2a Oracle Guiding）：§6.7.5 双条件**已满足** ——
                        (a) 闸门 C 连续 ≥2 段收敛且 R3 未反弹为显著负；(b) 闸门 A 显著为正。
                        按 §6.7.6 从「就绪态」升级为**规格冻结 + 离线准备态**，
                        但**不激活、不改代码、不并发、不重启**：长驻驱动无软停 flag
                        （DEADLINE 为模块常量、仅 timebox_ok 拒新阶段），主循环无空档，
                        §6.7.5 只允许自然重启窗口或终评窗口。规格见计划 §6.10.3。
[2026-11-06 09:52:00] 决策④（PPO-lite）：不启动（约 R10+，须先完成奖励 wrapper 定稿+单测；
                        当前增益仍在，只是斜率放缓）。决策⑤（闸门 B / 终评）：
                        终评第一优先 = 500 seed「新+N32 vs master+N32」独立池
                        (3_000_000+500)，优先级高于 P2a 试点；P2a 结论作下一迭代输入。
[2026-11-06 09:52:00] 计划调整（§6.10）：①预算 R3 实测 3.45h/轮（训练 300k 0.865h）、
                        R4+ 400k 封顶 ≈3.76h/轮，剩余 160.6h ⇒ 主循环约 42~45 轮，
                        DEADLINE 为硬约束；②**终评改在 DEADLINE 之后（11-13 02:30 起）执行**，
                        不占 168h 窗口、不与标注并发，脚本/独立 seed 池/闸门 B 等价口径
                        在 11-12 前离线备好（修正 §6.7.3「预留 11-12 晚间」）；
                        ③P2a 结构约束登记：只在终评做则无法「并入主路重跑后续轮」，
                        定位为验证性对照 + 下一迭代输入；④平台期触发规则：
                        R4~R6 若 A 环比增量持续不显著且 C 在 0 附近平台化，
                        R6 末把「天花板」由临近升级为确认，届时切换优先级到 P2a/终评，
                        但仍不改主循环超参；⑤下次自检：10:47 核 R4 标注进度/吞吐，
                        11:47 核 R4 训练 400k 窗口，13:xx 捕获 R4 闸门 A/C，R6 末复核增速。
[2026-11-06 09:52:00] 结论：**健康，产出超预期（连续两轮闸门 A 显著通过、闸门 C 转正）**；
                        下一步训练已由驱动自动接续 R4，本次零干预。
                        落地：reports/exit_log.md（本块）、docs/14天训练计划.md（§6.10）、
                        reports/夜间训练报告-2026-11-06.md（§13）。

================================================================================
[2026-11-06 11:25:00] 自检 #7（7×24 第 7 次，R4 标注落盘 + 训练启动窗口）
================================================================================
[2026-11-06 11:25:00] 状态：驱动全程存活（lock 心跳 6~28s、night_run.py PID 79443 +
                        annotate_r04 PID 10902 → 11:24:47 rc=0 退出 → train_r04 自动接续）。
                        R4 标注 09:13:38→11:24:47（7870s≈2h11m，早于 3.5h 超时上限
                        12:43），吞吐 11,438~11,481/h·worker（聚合 ≈45.8k/h），0 异常。
                        R4 四分片 11:24:47 稳定落盘：4×25,000=100,000 行，tiles(63,27)/
                        glob(130)/q,mask(28)，NaN↔mask overlap=0 gap=0 **全合规（16/16）**。
                        r01~r03 12 分片复核同样全合规。磁盘 311Gi。
[2026-11-06 11:25:00] 驱动自动接续「下一步训练」：11:24:48 `train_exit.py --data r01~r04
                        --window 400000 --init exit_best.pt --seed 20261110`，
                        **400k 滑窗首次封顶**（§6.10.6 预计 ≈1.15h）。按硬约束
                        **零干预、未启第二驱动、未打断、未并发评估**。
[2026-11-06 11:25:00] 闸门独立复算（正确口径：deltas[cand_seat]−mean(其余三家)，
                        四座按 seed 平均）与驱动逐位一致：
                        A: R1 +0.179±0.406 p≈0.6599 / R2 +2.374±0.415 p≈0.0000 /
                           R3 +2.843±0.422 p≈0.0000；
                        C: R1 −2.480±0.863 p≈0.0041 / R2 −0.293±1.125 p≈0.7944 /
                           R3 +0.300±0.980 p≈0.7595。
                        exit_best.pt ≡ exit_r03.pt（sha 74f313eb…），exit_best.onnx 08:52
                        同步；TAU=1.0 / lr=1e-3 / n_det=32 全未改。
                        异常扫描：02:33:57 之后 panic/NaN/timeout/error/rc≠0 **全为 0**；
                        主日志 11 条 panic 与 2 条 rc=101 均为 00:00:48 前早期失败启动
                        历史遗留（缺 --student-enc 3），按时间戳切片排除。
[2026-11-06 11:25:00] 【本次最重要发现 · P0 终评阻断缺陷（前 6 次自检均未发现）】
                        计划书 §4 主目标「新+N32 vs master+N32」的原定命令
                        `eval_pairs.py --a exit_best.pt --a-mode search32 --b-mode search32`
                        **必然形状 panic**，根因坐实（三层静态 + 张量实测）：
                        ① `rust_engine/src/pymod.rs:11,254,319` 的 take_batch/batch_arrays
                           固定用 `encode::NUM_CHANNELS=15 / GLOBAL_DIM=70`（v1）；
                        ② `zhuoji/rust_search.py:290-316` `_BatchRunner` 缓冲按 v1 常量
                           (mp,15,27)/(mp,70) 分配并 reshape；
                        ③ `exit_best.pt` 是 enc3（encoder_version=3, in_channels=63,
                           glob=130），而 `dushan_master.pt` 是 enc1（15/70）。
                        实测把 (1,15,27)+(1,70) 喂 enc3 网络 →
                           `RuntimeError: shape '[1, 63, 3, 9]' is invalid for input of size 405`。
                        历史闸门 A（双 greedy，NetAgent 按 ckpt 自带 enc 版本编码）与
                        闸门 C（A 侧新 greedy + B 侧 master+N32）**都绕开了该组合**，
                        故 7×24 主循环完全健康、产出有效；缺陷只影响闸门 B / 终评主目标。
                        连带修正：§6.3 缺口清单第 1 项闸门 B 原命令
                        `--a-mode search32 --b-mode search32 --seeds 100 --seed0 730000`
                        同样不可执行（幸而 §6.6.2 已改为终评补跑、从未插入主循环）。
[2026-11-06 11:25:00] 决策①（下一步训练）：驱动在跑且已自动接续 R4 训练 →
                        **零干预、不启第二驱动、不打断、不并发评估**；
                        R4 训练 → 闸门 A/C → R5 由 for 循环接续。
[2026-11-06 11:25:00] 决策②（增速三选一）：**不触发**。R2/R3 闸门 A 连续两轮显著为正
                        （p≈0.0000、均值 +2.374/+2.843>0），触发条件「连续两轮 p>0.2
                        或均值≤0」不成立；TAU=1.0 不动，R6 末复核。R4 闸门未出，
                        按 §6.7.6 / §6.10.7 **不提前决策**。
[2026-11-06 11:25:00] 决策③（P2a / PPO / 平台期）：P2a 维持「规格冻结 + 离线准备态」
                        不激活（无合法窗口、不改代码、不重启）；PPO-lite 不启动；
                        平台期结论待 R4 闸门 A/C（≈13:0x）与 R6 末确认，本次不提前判断。
[2026-11-06 11:25:00] 决策④（落地终评离线准备，按 §6.10.6「任一自检均可推进、
                        不需打断训练」授权的本次实质动作）：
                        (a) 新建 `tools/final_eval_precheck.py` —— 只读预检（不跑对局、
                            不占 CPU、不写模型），输出编码兼容性矩阵与 DEADLINE 后
                            冻结口径命令清单；本次 dry-run 通过。
                        (b) 终评**默认主口径改为组合 1**：500 seed 独立池 3_000_000+500、
                            四座旋转、`新贪心 vs master+N32`（同闸门 C 口径放大），
                            对应 §4「超额：新贪心 ≥ 现 N32（搜索等效搬迁）」；
                            辅口径 `新贪心 vs master 贪心` 同池配对降噪。
                        (c) 严格字面「双侧 search32」列为 **P0 待开发**：需让 Rust 搜索
                            内核按模型 enc 版本输出（encode_v3.rs 已存在，63/130），
                            但**绝不在 7×24 窗口内重编译或覆盖在跑的 annotate 二进制**；
                            只能在 DEADLINE 后用独立构建目录补做，完成后补跑组合 2。
                            两者口径差异如实写进终评报告，不以组合 1 冒称字面主目标。
[2026-11-06 11:25:00] 决策⑤（终评预算，首次量化）：闸门 C 实测单局 35~36s（单侧 N32）、
                        闸门 A 0.26s。500 seed×四座=2000 局，6 workers：
                        单侧 N32 ≈3.7h、双侧 N32 ≈7.4h（保守 ×3 ≈11.1h）。
                        ⇒ 终评须留 ≈4~11h，DEADLINE 后执行不占 168h 窗口，与
                        §6.10.6 排程一致；准备资产须 11-12 前备齐。
[2026-11-06 11:25:00] 结论：**健康，产出符合预期**（R4 标注 4×25k 合规落盘、
                        训练自动接续、16/16 分片合规、闸门统计逐位一致）；
                        本次零干预主循环，另发现并登记 P0 终评阻断缺陷、落地预检脚本
                        与修正后的终评口径。落地：reports/exit_log.md（本块）、
                        docs/14天训练计划.md（§6.11）、
                        reports/夜间训练报告-2026-11-06.md（§14）、
                        tools/final_eval_precheck.py（新建）。
[2026-11-06 11:24:47]   rc=0  7870s  annotate_r04
[2026-11-06 11:24:48] R4 标注完成 student=exit_best.onnx seed0=640000 → 4×25k
[2026-11-06 11:24:48] $ /Users/ligf9/zhuoji-mahjong-ai/.venv/bin/python train_exit.py --data runs/exit_r01/exit_w*.npz runs/exit_r02/exit_w*.npz runs/exit_r03/exit_w*.npz runs/exit_r04/exit_w*.npz --out /Users/ligf9/zhuoji-mahjong-ai/models/exit_r04.pt --epochs 30 --window 400000 --batch 256 --seed 20261110 --init /Users/ligf9/zhuoji-mahjong-ai/models/exit_best.pt  → train_r04.log
[2026-11-06 12:07:44]   rc=0  2576s  train_r04
[2026-11-06 12:07:44] R4 训练完成 init=exit_best.pt → exit_r04.pt  (['[done] best val 1.7625 → /Users/ligf9/zhuoji-mahjong-ai/models/exit_r04.pt'])
[2026-11-06 12:07:44] $ /Users/ligf9/zhuoji-mahjong-ai/.venv/bin/python eval_pairs.py --a /Users/ligf9/zhuoji-mahjong-ai/models/exit_r04.pt --b /Users/ligf9/zhuoji-mahjong-ai/models/dushan_master.pt --mode greedy --seeds 300 --seed0 710000 --workers 6 --out /Users/ligf9/zhuoji-mahjong-ai/runs/gates/r04_gateA.jsonl  → gateA_r04.log
[2026-11-06 12:08:39]   rc=0  56s  gateA_r04
[2026-11-06 12:08:39] R4 闸门 A: [summary] seats=[0, 1, 2, 3]  n=300  diff=+3.312±0.418  z=+7.92  p≈0.0000  符号 196:100  → 通过
[2026-11-06 12:08:39] $ /Users/ligf9/zhuoji-mahjong-ai/.venv/bin/python -c import torch,sys;sys.path.insert(0,'.');from zhuoji.net import build_from_ckpt;from zhuoji.forward import ensure_onnx;ck=torch.load('/Users/ligf9/zhuoji-mahjong-ai/models/exit_best.pt',map_location='cpu',weights_only=False);net,_=build_from_ckpt(ck);p=ensure_onnx(net,stem='exit_best');print('onnx:',p)  → export_exit_best.log
[2026-11-06 12:08:40]   rc=0  1s  export_exit_best
[2026-11-06 12:08:41] 晋升：exit_r04.pt → exit_best.pt
[2026-11-06 12:08:41] $ /Users/ligf9/zhuoji-mahjong-ai/.venv/bin/python eval_pairs.py --a /Users/ligf9/zhuoji-mahjong-ai/models/exit_r04.pt --b /Users/ligf9/zhuoji-mahjong-ai/models/dushan_master.pt --a-mode greedy --b-mode search32 --seeds 50 --seed0 720000 --workers 6 --out /Users/ligf9/zhuoji-mahjong-ai/runs/gates/r04_gateC.jsonl  → gateC_r04.log
[2026-11-06 12:30:40]   rc=0  1319s  gateC_r04
[2026-11-06 12:30:40] R4 闸门 C(趋势): [summary] seats=[0, 1, 2, 3]  n=50  diff=-0.347±1.085  z=-0.32  p≈0.7494  符号 21:28
[2026-11-06 12:30:40] R4 轮次完成  best=exit_best.pt
[2026-11-06 12:30:40] $ /Users/ligf9/zhuoji-mahjong-ai/.venv/bin/python -c import torch,sys;sys.path.insert(0,'.');from zhuoji.net import build_from_ckpt;from zhuoji.forward import ensure_onnx;ck=torch.load('/Users/ligf9/zhuoji-mahjong-ai/models/exit_best.pt',map_location='cpu',weights_only=False);net,_=build_from_ckpt(ck);p=ensure_onnx(net,stem='exit_best');print('onnx:',p)  → export_exit_best.log
[2026-11-06 12:30:41]   rc=0  1s  export_exit_best
[2026-11-06 12:30:41] $ /Users/ligf9/zhuoji-mahjong-ai/rust_engine/target/release/annotate --student /Users/ligf9/zhuoji-mahjong-ai/models/onnx/exit_best.onnx --teacher models/onnx/dushan_master.onnx --student-enc 3 --student-device cpu --targets 100000 --threads 4 --out /Users/ligf9/zhuoji-mahjong-ai/runs/exit_r05 --seed0 650000  → annotate_r05.log
================================================================================
[2026-11-06 12:34:43] 自检 #8（7×24 第 8 次 · R4 闸门 A/C 硬决策点）
================================================================================
[2026-11-06 12:34:43] 状态：驱动全程存活（lock 心跳 15~29s、night_run.py PID 79443 自 02:33
                        持续；night_run.log 12:30:41、annotate_r05.log 12:37:37 更新）。
                        当前阶段 = R5 DAgger 标注（12:30:41 启动，seed0=650000，学生
                        exit_best.onnx = R4 权重），12:37 采样 ≈5.4%（w0~w3 各 1.2~1.34k/25k），
                        吞吐 11,508~11,959/h·worker（健康区间 10~12.5k，聚合 ≈46.7k/h），
                        ETA ≈14:40，早于 3.5h 超时上限 16:00；0 异常，磁盘 311Gi。
[2026-11-06 12:34:43] 「下一步训练」已由驱动 12:30:41 自动接续（R5 标注 → R5 训练 →
                        闸门 A/C → R6）。按硬约束零干预：未启第二驱动、未打断、未并发评估。
[2026-11-06 12:34:43] R4 结果（硬决策点，实际 12:08/12:30 完成，早于 §6.11.7 预估 13:0x）：
                        闸门 A +3.312±0.418 z=+7.92 p≈0.0000 符号 196:100 → 通过，
                        12:08:41 晋升 exit_r04.pt → exit_best.pt（sha 947b96be… 一致），
                        r04.promoted=ok、exit_best.onnx 12:08 同步；
                        闸门 C −0.347±1.085 p≈0.7494 符号 21:28（趋势口径，不决策）。
                        R4 训练 400k 封顶 / init=exit_best.pt / 2576s，train 1.7571→1.7301
                        单调降，best val 1.7625(ep2)、ep5 早停；R4 标注 4×25k=100k，
                        r01~r04 16/16 npz（tiles(63,27)/glob(130)/q,mask(28)，
                        NaN↔mask overlap=0 gap=0）全合规。
[2026-11-06 12:34:43] 闸门独立复算（口径 deltas[cand_seat]−mean(其余三家)，四座按 seed 平均）
                        与驱动逐位一致：A R1 +0.179 / R2 +2.374 / R3 +2.843 / R4 +3.312；
                        C R1 −2.480 / R2 −0.293 / R3 +0.300 / R4 −0.347。
                        配对环比：A R3−R2 +0.469±0.563 p≈0.4053、R4−R3 +0.469±0.540
                        p≈0.3854（连续两轮不显著）；C R4−R3 −0.647±1.433 p≈0.6517
                        （未延续收敛，噪声内）。跨两轮累积 A R4−R2 +0.938±0.564 p≈0.0966；
                        C 三轮同池合并 −0.113±0.663 p≈0.8643（新贪心 ≈ N32 稳定成立）。
[2026-11-06 12:34:43] 异常复核：02:33:57 起 26 个阶段全 rc=0，0 panic/NaN/超时；主日志
                        11 条 panic + 2 条 rc=101 全部为 00:00:48 前失败启动历史遗留
                        （缺 --student-enc 3 / --student-device cpu），按时间戳切片排除。
[2026-11-06 12:34:43] 决策（最优推荐直接执行，主循环零干预）：
                        ① 下一步训练 = 驱动已自动接续 R5，不启第二驱动/不打断/不并发评估；
                        ② 增速三选一 **不触发**：闸门 A 连续三轮 p≈0.0000 显著为正且均值>0，
                           TAU=1.0 / Q-MSE 关 / 教师 n_det=32 / lr=1e-3 全维持，R6 末复核；
                           （环比增量不显著 ≠ 闸门 A 不显著，§6.4 触发条件未发生）
                        ③ Oracle Guiding(P2a) **暂停推进**：R4 闸门 C 回落 −0.347 使
                           §6.7.5 条件(a)「与 N32 差距持续收敛」中断 ⇒ 从「规格冻结+
                           离线准备态」降级为「规格冻结+观察态」，不激活/不改代码/不重启；
                           R6 天花板确认后再定。④ PPO-lite 不启动（R10+ 且增益仍累积）。
                        ⑤ 平台期：R4 满足 §6.10.6 首轮条件（A 环比 |Δ|<1.0 且 p>0.2；
                           C 0 附近无显著为负），但按规则 R6 末才升级天花板为「确认」，
                           现不改超参、不提前切资源。
                        ⑥ 终评准备：复核 tools/final_eval_precheck.py dry-run 通过
                           （组合 1/3 可用、组合 2 [P0] 禁用）；剩余导出冒烟清单与报告模板
                           列入 §6.12.5，11-12 前离线完成。
[2026-11-06 12:34:43] 本次最重要量化修正：**「平台期」= 单轮边际增益低于 seed 池分辨率，
                        不是增益停滞**。A 绝对 diff 逐轮单调上升且三点近似线性（+0.469/轮），
                        跨两轮累积 +0.938 p≈0.0966 接近显著 ⇒ 继续主循环仍有正收益，
                        不支持提前切 Oracle/PPO 或调超参。（三点共线使斜率 SE 退化，
                        该拟合仅作描述，不作显著性依据）
[2026-11-06 12:34:43] 新语义登记：train_exit.py:58-60 滑窗 x[-window:]，R5 起 r01~r05
                        共 500k 只取最近 400k ⇒ **master rollout 数据(r01)完全滑出**，
                        训练分布 100% 学生 DAgger；后续 val 跨轮更不可比，强度只认冻结基线闸门。
[2026-11-06 12:34:43] 落地：reports/exit_log.md（本块）、docs/14天训练计划.md（§6.12）、
                        reports/夜间训练报告-2026-11-06.md（§15）。

================================================================================
[2026-11-06 13:50:19] 自检 #9（7×24 第 9 次 · R5 标注期常规巡检 + 终评准备收口）
================================================================================
[2026-11-06 13:50:19] 状态：驱动全程存活（lock 心跳 3~18s、night_run.py PID 79443 自
                        02:33 已跑 11h17m、annotate PID 26706 跑 1h19m、
                        annotate_r05.log 16s 内更新）。当前阶段 = **R5 DAgger 标注**
                        （12:30:41 启动，seed0=650000，学生 exit_best.onnx = R4 权重），
                        13:50 采样 **62.2%**（w0~w3 各 15,526~15,570/25,000，
                        聚合 ≈62.2k/100k），吞吐 11,745~11,824/h·worker（健康区间
                        10~12.5k，聚合 ≈47.2k/h），ETA ≈14:38，远早于 3.5h 上限
                        16:00:41；0 异常，磁盘 311Gi。r05 npz 未落盘（标注结束才写，
                        符合预期，未读以避免 BadZipFile 竞态）。
[2026-11-06 13:50:19] 「下一步训练」：驱动在跑且 12:30:41 已自动接续 R5，按硬约束
                        **零干预** —— 未启第二驱动、未打断标注、未并发跑任何评估。
                        落盘后驱动自动启动 train_r05（r01~r05、窗口 400k 封顶、
                        init=exit_best.pt）→ 闸门 A/C → R6。
[2026-11-06 13:50:19] 产出复核（无新轮次产出，R5 仍在标注）：最近完成轮次 **R4** ——
                        闸门 A `+3.312±0.418 z=+7.92 p≈0.0000`（连续第三轮显著通过），
                        12:08:41 晋升 exit_r04.pt→exit_best.pt、r04.promoted=ok、
                        exit_best.onnx 12:08 同步；闸门 C `−0.347±1.085 p≈0.7494`
                        （趋势口径，噪声内）。R4 训练 400k/init/2576s，
                        train 1.7571→1.7301 单调降、best val 1.7625(ep2)、ep5 早停。
                        ckpt 5 份（r01~r04 + best）、gates 8 份 jsonl、
                        promoted 3 份（r02/r03/r04）。
[2026-11-06 13:50:19] 独立复算（口径 deltas[cand_seat]−mean(其余三家)，四座按 seed
                        平均）与驱动逐位一致：A `+0.179 / +2.374 / +2.843 / +3.312`；
                        C `−2.480 / −0.293 / +0.300 / −0.347`；环比 A
                        R3−R2 `+0.469±0.563 p≈0.4053`、R4−R3 `+0.469±0.540 p≈0.3854`；
                        C R4−R3 `−0.647±1.433 p≈0.6517`。r01~r04 **16/16** npz
                        全合规（25,000 行、tiles(63,27)/glob(130)/q,mask(28)、
                        NaN↔~mask overlap=0 gap=0）。TAU=1.0 未改。
[2026-11-06 13:50:19] 异常复核：02:33:57 起 **26 个阶段全 rc=0**，0 panic/NaN/超时；
                        主日志 11 条 panic + 2 条 rc=101 全为 00:00:48 前失败启动
                        历史遗留（缺 --student-enc 3 / --student-device cpu），
                        按时间戳切片排除。
[2026-11-06 13:50:19] 决策（最优推荐直接执行；按 §6.12.5 本次为常规巡检窗口，
                        **不提前决策**）：① 下一步训练 = 驱动自动接续 R5，零干预；
                        ② 增速三选一 **不触发**（A 连续三轮 `p≈0.0000` 且均值>0，
                           TAU/Q-MSE/n_det/lr 全维持），R6 末复核；
                        ③ P2a 维持「规格冻结 + 观察态（暂停推进）」不激活；
                        ④ PPO-lite 不启动；⑤ 平台期不提前升级天花板；
                        ⑥ 终评准备按 §6.10.6 授权**执行并完成**（见下）。
[2026-11-06 13:50:19] 本次落地（终评离线准备收口，零 CPU 占用、不打断训练）：
                        ① 新建 `tools/final_eval_report.py`（冻结口径统计固化，
                           dry-run 用 r01~r04 八份 jsonl 与驱动逐位一致；明确禁止
                           `deltas[0]` 错误口径 / 自对弈胜率 / val loss 作决策）；
                        ② 新建 `reports/终评执行手册-2026-11-13.md`（导出冒烟清单 +
                           终评三项配对命令 + §4 交付判定对照 + 口径差异强制声明 +
                           终评报告模板 + 禁止事项）；
                        ③ `tools/final_eval_precheck.py` 命令清单回链新资产；
                        ④ 复核预检 dry-run：组合 1/3 可用、组合 2 `[P0]` 禁用。
[2026-11-06 13:50:19] 两项排查（防终评踩坑）：① **`?touch=1` 冒烟项不存在** ——
                        计划 §1 D13–D14 第 2 条字面不可执行：`dushan_server.py`
                        全文件无 touch / parse_qs / urlparse query 处理；
                        `dushan.html:20` 的 `touch` 只是 JS 设备触屏检测变量；
                        模型选择由 `spec.model → resolve_model_stem()`（:854,872）
                        完成。⇒ 手册改用 `--selftest 10` + HTTP 200 +
                        ORT/torch 前向一致性三项等价覆盖。
                        ② 终评命令 `--seats 0,1,2,3` **已核实存在**
                        （`eval_pairs.py:176`），`--seed0 3000000 --seeds 500`
                        参数合法，组合 1 可执行；组合 2 仍因 enc3 内核禁用，
                        **不得冒称 §4 主目标**。
[2026-11-06 13:50:19] 最重要计划校准（§6.13.2）：R4 实测完整一轮 = **3h17m**
                        （标注 2h11m + 训练 43m + 闸门 A 1m + 闸门 C 22m）。
                        据此 R5 时间线：标注落盘 ≈14:38 → 训练 ≈15:21 →
                        闸门 A ≈15:22 → 闸门 C ≈15:44 → R6 标注 ≈15:44 启动。
                        ⇒ **§6.12.5 的「≈16:0x 捕获 R5 闸门」偏晚 25~35min，
                        硬决策点校准为 15:2x~15:5x（由 15:4x 自检捕获）**；
                        R6 末硬决策点 ≈19:0x（原 19~20h 模糊值精确化）。
[2026-11-06 13:50:19] 结论：**健康、产出符合预期、主循环零干预**。当前阶段 R5 标注
                        62.2%、吞吐 11.7~11.8k/h·worker、ETA 14:38、0 异常。
                        最近产出 R4（闸门 A `+3.312±0.418 p≈0.0000` 已晋升
                        exit_best.pt；闸门 C `−0.347±1.085 p≈0.7494`）。
                        下一步动作：驱动自动接续 R5 训练 → 闸门 A/C → R6；
                        本次不调任何超参、不切分支；终评准备已离线完成，
                        DEADLINE 后按手册执行。
[2026-11-06 13:50:19] 落地：reports/exit_log.md（本块）、docs/14天训练计划.md（§6.13）、
                        reports/夜间训练报告-2026-11-06.md（§16）、
                        tools/final_eval_report.py（新建）、
                        reports/终评执行手册-2026-11-13.md（新建）、
                        tools/final_eval_precheck.py（回链）。
[2026-11-06 13:50:19] **下次必做**：14:4x 核 R5 训练 400k 封顶 / init / train-val /
                        早停 + r05 npz 4×25k 合规（落盘后再读）；
                        **15:4x 捕获 R5 闸门 A/C**（A 是否连续第 4 轮显著、C 是否仍
                        0 附近、R5−R3 跨两轮累积、r05.promoted / onnx 同步）；
                        R6 末 ≈19:0x 硬决策点复核增速三选一 + 天花板确认 +
                        P2a 观察态是否解除。切勿因 C 单轮回落改超参。

[2026-11-06 14:37:41]   rc=0  7620s  annotate_r05
[2026-11-06 14:37:41] R5 标注完成 student=exit_best.onnx seed0=650000 → 4×25k
[2026-11-06 14:37:41] $ /Users/ligf9/zhuoji-mahjong-ai/.venv/bin/python train_exit.py --data runs/exit_r01/exit_w*.npz runs/exit_r02/exit_w*.npz runs/exit_r03/exit_w*.npz runs/exit_r04/exit_w*.npz runs/exit_r05/exit_w*.npz --out /Users/ligf9/zhuoji-mahjong-ai/models/exit_r05.pt --epochs 30 --window 400000 --batch 256 --seed 20261111 --init /Users/ligf9/zhuoji-mahjong-ai/models/exit_best.pt  → train_r05.log

================================================================================
[2026-11-06 15:07:30] 自检 #10（7×24 第 10 次 · R5 训练期常规巡检）
================================================================================
[2026-11-06 15:07:30] 状态：驱动全程存活（lock 心跳 13~28s、night_run.py PID 79443
                        自 02:33 跑 12h33m、train_exit.py PID 37357 跑 29m52s/20.3% CPU、
                        train_r05.log 15:04:21 更新）。当前阶段 = **R5 训练**
                        （14:37:41 启动，`[data] 20 files, 400000 samples（窗口 400000）`、
                        init=exit_best.pt、seed 20261111、M1 9.84M/mps）。
                        按硬约束 **零干预**：未启第二驱动、未打断训练、未并发跑任何评估。
[2026-11-06 15:07:30] R5 训练曲线：ep1 train 1.7906/val 1.7886(512s)、ep2 1.7782/1.7908
                        (1027s)、ep3 1.7719/1.7893(1597s) —— train 单调降，val ep2/ep3
                        微升由 patience3 处理，ep1 暂为 best。每 ep ≈512~570s，
                        预计 15:13（ep4 早停）~15:55（ep8/9）结束。日志仅 1 条无害
                        UserWarning(float(loss) requires_grad，R1 起同款)，
                        panic/NaN/Error/Traceback/fail 全 0。
[2026-11-06 15:07:30] R5 标注复核：14:37:41 rc=0 落盘，7620s≈2h07m，四分片逐个
                        saved 25,000 targets + `[annotate] all done`；末段吞吐
                        11,809~11,857/h·worker（聚合 ≈47.3k/h），落在 10~12.5k 健康区间；
                        panic/NaN/Error/timeout/fail 全 0，磁盘 311Gi 可用(29%)。
[2026-11-06 15:07:30] 数据合规：r01~r05 **20/20** npz 分片各 25,000 行、
                        tiles(63,27)/glob(130)/q,mask(28)，q-NaN ↔ mask=0
                        overlap=0 gap=0（r05 四分片已稳定落盘，实测 NaN≈49.1~49.3 万）。
[2026-11-06 15:07:30] 闸门独立复算（tools/final_eval_report.py 冻结口径
                        deltas[cand_seat]−mean(其余三家)、四座按 seed 平均）与驱动逐位一致：
                        A R1 +0.179±0.406 p≈0.6599 / R2 +2.374±0.415 p≈0.0000 /
                        R3 +2.843±0.422 p≈0.0000 / R4 +3.312±0.418 p≈0.0000；
                        C R1 −2.480±0.863 p≈0.0041 / R2 −0.293±1.125 p≈0.7944 /
                        R3 +0.300±0.980 p≈0.7595 / R4 −0.347±1.085 p≈0.7494。
                        8 份 jsonl 行数 A 1200×4 / C 200×4 全合规；R5 闸门**尚未产出**
                        （gates 目录最新为 r04）。
[2026-11-06 15:07:30] 权重链：exit_best.pt ≡ exit_r04.pt（sha 309e05fd…）；
                        exit_r05.pt 已写出（23f2a53e…）但 **无 r05.promoted ⇒ 未晋升**，
                        再次验证 §6.9.4「写入 ≠ 晋升」；exit_best.onnx 12:08 同步；
                        promoted 3 份（r02/r03/r04）。TAU=1.0(train_exit.py:38)、
                        lr=1e-3(:85) 只读核对未改。02:33:57 起 20 条阶段行全 rc=0、
                        异常 0；主日志头部 11 条 panic + 2 条 rc=101 均为 00:00:48 前
                        失败启动历史遗留（按时间戳切片排除）。
[2026-11-06 15:07:30] 决策（最优推荐直接执行；按 §6.12.5 属常规巡检窗口，不提前决策）：
                        ① 下一步训练 = 驱动 14:37:41 已自动接续 R5 训练 → 闸门 A/C →
                           R6 标注，零干预；② 增速三选一 **不触发**（A 连续三轮
                           p≈0.0000 且均值>0，R5 闸门未出；TAU/Q-MSE/n_det/lr 全维持）；
                        ③ P2a 维持「规格冻结 + 观察态（暂停推进）」不激活；
                        ④ PPO-lite 不启动；⑤ 平台期不提前升级天花板（R6 末确认）；
                        ⑥ 终评准备已由 #9 收口，本次复算验证统计脚本仍可用。
[2026-11-06 15:07:30] 本次两项新增（写入计划 §6.14）：
                        (a) **val/train 跨轮不可比再确认**：R4 起点 1.7571/1.7625 →
                            R5 起点 1.7906/1.7886 的抬升，源于 400k 滑窗把 r01
                            （master rollout）完全滑出、训练分布 100% 学生 DAgger
                            （§6.12.4），**不是退化**；强度只认冻结基线闸门 A/C，
                            禁止据 loss 跨轮调超参。
                        (b) **自检采样节奏 vs 闸门窗口对齐**：历史触发间隔 61~101min
                            （中位 ≈72min），而闸门 A+C 窗口仅 ≈23min ⇒ 「准点捕获」
                            不可行；本次 15:07 落在训练 ep3，R5 闸门由 **≈16:1x~16:4x
                            首次捕获**（§6.13.2 的 ≈15:4x 无对应触发）。固化规则：
                            每次自检一律完整复核自上次以来所有新落盘轮次，
                            硬决策点按「首次捕获该轮闸门的自检」判定。
[2026-11-06 15:07:30] 落地资产：新建 `tools/check_npz.py`（只读轻量分片校验：
                        按 npy header version 偏移取 shape、只读 q/mask 验证
                        NaN↔mask=0、BadZipFile 竞态提示；`--round r05` 与全量均通过
                        4/4 与 20/20）。本次内联初版脚本因把 hlen 字节当 dict 文本
                        解析报 SyntaxError(null bytes)，属**自检脚本口径错误非数据异常**，
                        修正后全合规；固化工具避免后续重复踩坑。
[2026-11-06 15:07:30] 结论：**健康、产出符合预期、主循环零干预**。
                        当前 R5 训练 ep3/400k 封顶、train 单调降、0 异常；
                        最近完成轮次 R4 闸门 A +3.312±0.418 p≈0.0000 已晋升。
                        下一步动作：驱动自动接续 R5 闸门 A/C → R6 标注；
                        本次不调任何超参、不切分支、不启第二驱动。
                        落地：reports/exit_log.md（本块）、docs/14天训练计划.md（§6.14）、
                        reports/夜间训练报告-2026-11-06.md（§17）、tools/check_npz.py（新建）。
[2026-11-06 15:07:30] **下次必做（≈16:1x~16:4x，R5 闸门硬决策点）**：完整捕获
                        R5 闸门 A/C（A 是否连续第 4 轮显著、C 是否仍 0 附近、
                        R5−R3 跨两轮累积、环比增量）、`r05.promoted` / exit_best.pt
                        与 onnx 同步、R5 训练最终 train-val 与早停 epoch、R6 标注进度；
                        按 §6.7.6 / §6.10.6 判 P2a 观察态与平台期。R6 末 ≈19:xx
                        复核增速三选一 + 天花板确认。切勿因闸门 C 单轮波动改超参。
[2026-11-06 15:15:09]   rc=0  2248s  train_r05
[2026-11-06 15:15:09] R5 训练完成 init=exit_best.pt → exit_r05.pt  (['[done] best val 1.7886 → /Users/ligf9/zhuoji-mahjong-ai/models/exit_r05.pt'])
[2026-11-06 15:15:09] $ /Users/ligf9/zhuoji-mahjong-ai/.venv/bin/python eval_pairs.py --a /Users/ligf9/zhuoji-mahjong-ai/models/exit_r05.pt --b /Users/ligf9/zhuoji-mahjong-ai/models/dushan_master.pt --mode greedy --seeds 300 --seed0 710000 --workers 6 --out /Users/ligf9/zhuoji-mahjong-ai/runs/gates/r05_gateA.jsonl  → gateA_r05.log
[2026-11-06 15:16:12]   rc=0  63s  gateA_r05
[2026-11-06 15:16:12] R5 闸门 A: [summary] seats=[0, 1, 2, 3]  n=300  diff=+3.398±0.413  z=+8.23  p≈0.0000  符号 209:87  → 通过
[2026-11-06 15:16:12] $ /Users/ligf9/zhuoji-mahjong-ai/.venv/bin/python -c import torch,sys;sys.path.insert(0,'.');from zhuoji.net import build_from_ckpt;from zhuoji.forward import ensure_onnx;ck=torch.load('/Users/ligf9/zhuoji-mahjong-ai/models/exit_best.pt',map_location='cpu',weights_only=False);net,_=build_from_ckpt(ck);p=ensure_onnx(net,stem='exit_best');print('onnx:',p)  → export_exit_best.log
[2026-11-06 15:16:13]   rc=0  1s  export_exit_best
[2026-11-06 15:16:13] 晋升：exit_r05.pt → exit_best.pt
[2026-11-06 15:16:13] $ /Users/ligf9/zhuoji-mahjong-ai/.venv/bin/python eval_pairs.py --a /Users/ligf9/zhuoji-mahjong-ai/models/exit_r05.pt --b /Users/ligf9/zhuoji-mahjong-ai/models/dushan_master.pt --a-mode greedy --b-mode search32 --seeds 50 --seed0 720000 --workers 6 --out /Users/ligf9/zhuoji-mahjong-ai/runs/gates/r05_gateC.jsonl  → gateC_r05.log
[2026-11-06 15:40:42]   rc=0  1468s  gateC_r05
[2026-11-06 15:40:42] R5 闸门 C(趋势): [summary] seats=[0, 1, 2, 3]  n=50  diff=+0.060±0.935  z=+0.06  p≈0.9488  符号 26:23
[2026-11-06 15:40:42] R5 轮次完成  best=exit_best.pt
[2026-11-06 15:40:42] $ /Users/ligf9/zhuoji-mahjong-ai/.venv/bin/python -c import torch,sys;sys.path.insert(0,'.');from zhuoji.net import build_from_ckpt;from zhuoji.forward import ensure_onnx;ck=torch.load('/Users/ligf9/zhuoji-mahjong-ai/models/exit_best.pt',map_location='cpu',weights_only=False);net,_=build_from_ckpt(ck);p=ensure_onnx(net,stem='exit_best');print('onnx:',p)  → export_exit_best.log
[2026-11-06 15:40:43]   rc=0  1s  export_exit_best
[2026-11-06 15:40:43] $ /Users/ligf9/zhuoji-mahjong-ai/rust_engine/target/release/annotate --student /Users/ligf9/zhuoji-mahjong-ai/models/onnx/exit_best.onnx --teacher models/onnx/dushan_master.onnx --student-enc 3 --student-device cpu --targets 100000 --threads 4 --out /Users/ligf9/zhuoji-mahjong-ai/runs/exit_r06 --seed0 660000  → annotate_r06.log
[2026-11-06 15:52:00] 自检 #11（提前捕获 R5 闸门硬决策点，§6.15）：
                        健康、产出符合预期、主循环零干预。唯一驱动 PID 79443，
                        lock 心跳 16s，32 条阶段行全 rc=0、panic/NaN/超时 0；
                        Web 8770 GET /=200、/api/meta=200，单进程无重复启动。
                        R6 标注 15:40:43 起 ≈10.0~10.1k/h·worker（聚合 ≈40.3k/h，
                        >8k 健康线；较 R5 的 11.8k 低 ≈14%，起始即如此，登记观察）。
[2026-11-06 15:52:00] R5 闸门冻结口径复算（final_eval_report.py，与驱动逐位一致）：
                        A +3.398±0.413 z=+8.23 p≈0.0000（连续第 4 轮显著、绝对值新高）；
                        C +0.060±0.935 p≈0.9488（0 附近、无显著为负）。
                        A 配对环比 R5−R4 = +0.086±0.529 p≈0.8715
                        （前两段均 +0.469）；R5−R3 = +0.554±0.545 p≈0.3094；
                        R5−R2 = +1.023±0.545 p≈0.0603；C R2~R5 合并
                        −0.070±0.576 p≈0.9033。r05.promoted / exit_best.pt
                        ≡ exit_r05.pt（SHA a8c708fe…）/ exit_best.onnx 15:16 三者同步。
[2026-11-06 15:52:00] 【最重要】修正 §6.12.2「≈+0.47/轮线性累积」：R5 增量
                        +0.086 使斜率转平，跨两轮累积亦回落；但 A 对 master 仍
                        p≈0.0000 且创新高、R5−R2 接近显著 ⇒ 精确表述为
                        「单轮边际低于分辨率且斜率转平，绝对强度仍缓慢累积」，
                        教师天花板「临近」证据加强，非退化/非停训信号。
[2026-11-06 15:52:00] 决策：① 零干预，R6→R7 由驱动接续；② 增速三选一不触发
                        （A 连续 4 轮 p≈0.0000，TAU/lr/n_det 全维持）；③ 平台期
                        R4/R5 已满足 2/3 条件但按 §6.10.6 仍留 R6 末才升级为「确认」；
                        ④ P2a 维持观察态（C 增量绕 0 震荡、收敛条件未成立）；
                        ⑤ PPO-lite 不启动；⑥ R5 训练 ep1 即 best val、ep4 早停，
                        仅登记分布漂移语义，不改 epochs/窗口/配比；⑦ 终评准备不变。
                        落地：docs/14天训练计划.md §6.15、reports/夜间训练报告-2026-11-06.md §18。
[2026-11-06 15:52:00] 排程：R6 标注 ≈18:0x~18:1x 落盘 → 训练 ≈18:5x →
                        闸门 A ≈18:5x → 闸门 C ≈19:1x~19:2x ⇒ R6 硬决策点
                        ≈19:1x~19:3x。≈16:xx 自动化自检请勿重写 §6.15，
                        只核对结论 + 巡检 R6 进度/吞吐/心跳，不提前决策。
[2026-11-06 16:15:00] 自检 #12（自动化核对，按 §6.15.6 执行，不重写 §6.15）：
                        健康、产出符合预期、主循环零干预、不提前决策。
                        唯一驱动 PID 79443，lock 心跳 17.9s；02:33:57 切片
                        25 个已完成阶段全 rc=0、第 26 条 R6 标注在跑；
                        panic/NaN/Traceback/超时 0（2 条 rc=101 为 00:00:48 前历史遗留）。
                        阶段 R6 标注（15:40:43 起，PID 44330，--student-enc 3
                        --student-device cpu，seed0=660000），进度 ≈27.6%，
                        吞吐 10,268~10,388/h·worker（较 R5 低 ≈12%，起始即如此、
                        全程平稳无掉速），>8k 健康线；超时临界 7,143/h·worker
                        ⇒ 余量 44%，无超时风险。磁盘 311Gi、Web 8770
                        GET / 与 /api/meta 均 200（单进程 PID 43859）、TAU=1.0 未改。
[2026-11-06 16:15:00] R5 结论独立复核：冻结口径复算与驱动 / §6.15.2 逐位一致
                        （A +0.179/+2.374/+2.843/+3.312/+3.398，C −2.480/−0.293/
                        +0.300/−0.347/+0.060；A 环比 +2.196/+0.469/+0.469/+0.086；
                        R5−R3 +0.554±0.545 p≈0.3094；R5−R2 +1.023±0.545 p≈0.0603；
                        C R2~R5 合并 −0.070±0.514 p≈0.8916）。§6.15.3「边际转平、
                        绝对强度仍缓慢累积、天花板临近但非退化」成立。
                        补齐 #11 未做项：tools/check_npz.py 全量 20/20 分片合规
                        （25k 行、tiles(63,27)/glob(130)、NaN↔mask=0 overlap/gap=0）。
[2026-11-06 16:15:00] 【本次新证据】ONNX initializer 数值签名实测：
                        exit_best.onnx 命中 exit_r05.pt 90/336 个浮点权重张量，
                        r02/r03/r04 全 0/336 ⇒ R6 学生确为 R5 权重（推断升实测）。
                        代码走查 promote() = copy2 → export_onnx → ensure_onnx
                        → 写 promoted → 打日志；copy2 保留源 mtime(R5 14:46)
                        晚于上轮 onnx(12:08) 使 mtime 缓存失效而强制重导出，
                        故 15:16 onnx 为 R5。§6.9.3 登记的「mtime 缓存脆弱点」
                        在 promote 路径下是必要机制且实测自洽 ⇒ 降级为已验证的
                        低优先级健壮性项，7×24 期间不改代码（长驻不重载源码）。
[2026-11-06 16:15:00] 决策（§6.15.4 原样维持）：① 下一步训练已由驱动 15:40:43
                        自动接续 R6，零干预（不启第二驱动/不打断/不并发评估）；
                        ② 增速三选一不触发（A 连续 4 轮 p≈0.0000，TAU=1.0 全维持）；
                        ③ 平台期不提前升级，R6 末才确认天花板；④ P2a 维持观察态
                        （C 绕 0 震荡）；⑤ PPO 不启动；⑥ 终评手册 §2 冒烟清单
                        + §6 报告模板已存在 ⇒ §6.15.4⑦「11-12 前补齐」待办关闭。
[2026-11-06 16:15:00] 【本次最重要落地】新建 tools/gate_decision.py（只读/零 CPU/
                        不启对局）：把 §6.4/§6.7.6/§6.10.6/§6.15.4 判据代码化，
                        复用 final_eval_report.py 冻结口径，一键输出逐轮闸门+环比
                        +五项决策建议+健康摘要，杜绝后续自检口径漂移；dry-run
                        与 §6.15.2 逐位一致，--round 6 未落盘时正确回退 R5。
                        时间线校准：R6 标注 ≈18:06 落盘 → 训练 ≈18:44 →
                        闸门 A ≈18:45 → 闸门 C ≈19:09。捕获窗口：自检间隔
                        ≈72min ⇒ 首次捕获 R6 闸门为 ≈19:4x 自检，硬决策点
                        实际落 ≈19:4x（比 §6.15.5 晚 10~30min，按 §6.14.3
                        「首次捕获即决策点」不视为错过）。
                        落地：docs/14天训练计划.md §6.16、reports/夜间训练报告
                        -2026-11-06.md §19、tools/gate_decision.py（新建）。
[2026-11-06 16:15:00] **下次必做（≈19:4x，R6 末硬决策点）**：先跑
                        tools/gate_decision.py --md 得一致判据，再执行 ① 增速
                        三选一（仅当 A 连续两轮 p>0.2 或均值≤0 才仅改
                        train_exit.py:38 TAU=0.5，禁止同时改第二项）；
                        ② 用 R4~R6 三轮把天花板升级为「确认」但仍不改超参；
                        ③ 判 P2a 观察态是否解除；④ 核 r06 4×25k 合规、
                        r06.promoted / exit_best.pt / onnx 三者同步。
                        17:2x 与 18:3x 自检只核对进度/吞吐/落盘，勿提前决策、
                        勿在落盘前读 r06 npz。

[2026-11-06 17:31:00] ===== 自检 #13（7×24 第 13 次 · R6 标注期常规巡检）=====
[2026-11-06 17:31:00] 结论：健康、产出符合预期、主循环零干预。驱动唯一 PID 79443
                        （已跑 14h58m），lock 心跳 1.7~8.1s，annotate_r06.log 秒级
                        更新；R6 标注 PID 44330（--student-enc 3 / --student-device
                        cpu / 教师 CoreML / seed0=660000 / 4 线程 / 100000 targets）。
[2026-11-06 17:31:00] R6 标注进度 81.5%（81,518/100k，w0~w3 各 20,342~20,440/
                        25k，6702s≈1.86h），累计吞吐 10,914~10,960/h·worker
                        （健康区间 10~12.5k，瞬时斜率 ≈11.6k），ETA ≈17:57 落盘
                        （总 2.28h），距 3.5h 超时上限 19:10 余 ≈73min。
                        02:33:57 切片 25 个已完成阶段全 rc=0，第 26 条 R6 标注
                        在跑；panic / NaN / Traceback / 超时 / Error / Killed 全 0
                        （11 条 panic + 2 条 rc=101 均为 00:00:48 前历史遗留、
                        切片外）。磁盘 310Gi、Web 8770 `/` + `/api/meta` 双 200、
                        TAU=1.0 未改。
[2026-11-06 17:31:00] 复核：tools/gate_decision.py --md 与驱动核心统计量逐位
                        一致（A +0.179/+2.374/+2.843/+3.312/+3.398；C −2.480/
                        −0.293/+0.300/−0.347/+0.060；A 环比 +2.196/+0.469/
                        +0.469/+0.086）。tools/check_npz.py 20/20 分片全合规；
                        exit_best.pt ≡ exit_r05.pt（23f2a53e…）、onnx 15:16 同步、
                        r02~r05.promoted = ok。R6 闸门未落盘、r06 npz 未读
                        （避免 BadZipFile 竞态）。最近完整轮仍为 R5：闸门 A
                        +3.398±0.413 p≈0.0000（连续第 4 轮显著通过）、闸门 C
                        +0.060±0.935 p≈0.9488（新贪心 ≈ N32 持平）、环比
                        +0.086±0.529 p≈0.8715（边际转平、绝对强度仍缓慢累积）。
[2026-11-06 17:31:00] 决策（常规巡检，按 §6.16.6 不提前决策）：① 下一步训练由
                        驱动自动接续 R6 标注 → 训练(400k 封顶, init=exit_best.pt)
                        → 闸门 A/C → R7，零干预（不启第二驱动 / 不打断 /
                        不并发评估）；② 增速三选一不触发（A 连续 4 轮 p≈0.0000
                        且均值>0，TAU / Q-MSE / n_det / lr 全维持）；③ 平台期
                        不提前升级（R4/R5 满足 2/3，留 R6 末）；④ P2a 维持观察态
                        （C 最近两段正增量 1/2）；⑤ PPO 不启动（门槛 R10+）；
                        ⑥ 终评离线准备保持收口状态。
[2026-11-06 17:31:00] 【本次最重要发现 · 浮点边界效应】gate_decision.py /
                        final_eval_report.py 复算的「符号」计数与 night_run.log /
                        gateA_r05.log 的 [summary] 差 1~2 个 seed（R5 A 211:87
                        vs 209:87；R5 C 26:24 vs 26:23；R4 C 20:29 vs 21:28）。
                        逐 seed 严格对比坐实：驱动口径（dict 去重 + numpy float64
                        四座平均）与冻结口径（per_seed + Python float）数值差异 0，
                        mean/SE/z/p 完全一致；差异仅来自 3 个 seed 的四座平均分差
                        为 ±2.2e-16 / ±4.4e-16 / −5.6e-17 级浮点残差（理论应为 0），
                        numpy 路径得精确 0（不计符号）、Python 路径得微小非零。
                        ⇒ 纯描述性字段的浮点边界效应；_gate_pass() 只认
                        「判定: 通过」不受影响；终评报告只引用 mean±SE / z / p，
                        不引用符号计数（或必须注明来源口径）。不改代码（长驻不
                        重载且零决策影响），登记低优先级健壮性项。
[2026-11-06 17:31:00] 【本次最重要计划校准 §6.17】① 时间线：R6 标注 ≈17:57
                        （比 §6.16.5(b) 的 18:06 早 9min）→ 训练 ≈18:37 →
                        闸门 A ≈18:38 → 闸门 C ≈19:03 → R7 标注 ≈19:03
                        （闸门 C 耗时 1296→1319→1468s 逐轮递增，外推 1480s）。
                        ② 硬决策点定义纠偏：自检触发 ≈18:43 / 19:55 / 21:07，
                        18:43 落在「R6 闸门 A 已落盘但闸门 C 未落盘」窗口 ⇒
                        明确「首次捕获」指首次捕获完整 A+C 闸门，18:43 只记录
                        闸门 A + 核 r06 落盘/训练，不执行增速 / 天花板 / P2a
                        硬决策；真正的 R6 末硬决策点为 ≈19:55 自检（覆盖
                        §6.16.5(c) 的 ≈19:4x）。③ 19:55 清单：先
                        tools/gate_decision.py --md，再执行「仅当 A 连续两轮
                        p>0.2 或均值≤0 才仅改 TAU=0.5」、R4~R6 天花板升级
                        「确认」但不改超参、P2a 观察态去留、核 r06 4×25k 与
                        r06.promoted / exit_best.pt / onnx 同步。④ 长期：
                        DEADLINE 2026-11-13 02:30 唯一硬约束，约 42~45 轮，
                        终评 DEADLINE 后按手册执行。
[2026-11-06 17:31:00] 落地：docs/14天训练计划.md §6.17、reports/夜间训练报告
                        -2026-11-06.md §20、本日志块。
[2026-11-06 17:31:00] **下次必做**：18:43 自检（若触发）核 R6 标注落盘 4×25k /
                        训练 400k / train-val / 早停 / 闸门 A，勿提前决策；
                        **19:55 硬决策点**完整执行四项清单，勿在闸门 C 未落盘时
                        改任何超参或宣布天花板。

[2026-11-06 17:56:38]   rc=0  8155s  annotate_r06
[2026-11-06 17:56:38] R6 标注完成 student=exit_best.onnx seed0=660000 → 4×25k
[2026-11-06 17:56:38] $ /Users/ligf9/zhuoji-mahjong-ai/.venv/bin/python train_exit.py --data runs/exit_r01/exit_w*.npz runs/exit_r02/exit_w*.npz runs/exit_r03/exit_w*.npz runs/exit_r04/exit_w*.npz runs/exit_r05/exit_w*.npz runs/exit_r06/exit_w*.npz --out /Users/ligf9/zhuoji-mahjong-ai/models/exit_r06.pt --epochs 30 --window 400000 --batch 256 --seed 20261112 --init /Users/ligf9/zhuoji-mahjong-ai/models/exit_best.pt  → train_r06.log
[2026-11-06 18:31:18]   rc=0  2079s  train_r06
[2026-11-06 18:31:18] R6 训练完成 init=exit_best.pt → exit_r06.pt  (['[done] best val 1.7959 → /Users/ligf9/zhuoji-mahjong-ai/models/exit_r06.pt'])
[2026-11-06 18:31:18] $ /Users/ligf9/zhuoji-mahjong-ai/.venv/bin/python eval_pairs.py --a /Users/ligf9/zhuoji-mahjong-ai/models/exit_r06.pt --b /Users/ligf9/zhuoji-mahjong-ai/models/dushan_master.pt --mode greedy --seeds 300 --seed0 710000 --workers 6 --out /Users/ligf9/zhuoji-mahjong-ai/runs/gates/r06_gateA.jsonl  → gateA_r06.log
[2026-11-06 18:32:19]   rc=0  61s  gateA_r06
[2026-11-06 18:32:19] R6 闸门 A: [summary] seats=[0, 1, 2, 3]  n=300  diff=+3.658±0.407  z=+8.99  p≈0.0000  符号 211:88  → 通过
[2026-11-06 18:32:19] $ /Users/ligf9/zhuoji-mahjong-ai/.venv/bin/python -c import torch,sys;sys.path.insert(0,'.');from zhuoji.net import build_from_ckpt;from zhuoji.forward import ensure_onnx;ck=torch.load('/Users/ligf9/zhuoji-mahjong-ai/models/exit_best.pt',map_location='cpu',weights_only=False);net,_=build_from_ckpt(ck);p=ensure_onnx(net,stem='exit_best');print('onnx:',p)  → export_exit_best.log
[2026-11-06 18:32:20]   rc=0  1s  export_exit_best
[2026-11-06 18:32:20] 晋升：exit_r06.pt → exit_best.pt
[2026-11-06 18:32:20] $ /Users/ligf9/zhuoji-mahjong-ai/.venv/bin/python eval_pairs.py --a /Users/ligf9/zhuoji-mahjong-ai/models/exit_r06.pt --b /Users/ligf9/zhuoji-mahjong-ai/models/dushan_master.pt --a-mode greedy --b-mode search32 --seeds 50 --seed0 720000 --workers 6 --out /Users/ligf9/zhuoji-mahjong-ai/runs/gates/r06_gateC.jsonl  → gateC_r06.log

[2026-11-06 18:52:00] === 自检 #14（7×24 第 14 次 · R6 闸门 A 落盘 + 巡检窗口）===
[2026-11-06 18:52:00] 结论：健康、产出符合预期、主循环零干预。驱动唯一 PID 79443
                        （lock 4~16s，已跑 16h06m）；阶段 = R6 闸门 C 评估在跑
                        （eval_pairs PID 59667，18:32:20 启动，18:49 已写 20/50
                        seed，jsonl 未完整）。按 §6.17.4 本次属「A 已落盘 / C
                        未落盘」巡检窗口，未启第二驱动 / 未打断 / 未并发评估 /
                        未调任何超参。
[2026-11-06 18:52:00] 健康：R6 标注 17:56:38 rc=0（8155s≈2h16m，吞吐
                        11,034~11,063/h·worker，聚合 ≈44.2k/h，健康区间 10~12.5k
                        内，未跌破 8k 线）；02:33:57 切片 29 个已完成阶段全 rc=0、
                        rc!=0=0、panic=0、NaN/Traceback/超时=0（11 条 panic +
                        2 条 rc=101 均为 00:00:48 前历史遗留、切片外）；磁盘 310Gi
                        （29% 用）；train_exit.py:38 TAU=1.0 未改。
[2026-11-06 18:52:00] R6 产出：4×25,000=100k 标注全合规，r01~r06 共 **24/24**
                        npz 分片（25k 行、tiles(63,27)/glob(130)/q,mask(28)、
                        NaN↔mask=0 overlap=0 gap=0）；训练 18:31:18 rc=0（2079s，
                        24 files / 400k 滑窗封顶、init=exit_best.pt、seed 20261112），
                        train 1.7894→1.7773→1.7705→1.7649 单调降，best val
                        1.7959(ep1)，ep4 早停（patience3）。val 跨轮不可比，
                        强度只认冻结基线闸门。
[2026-11-06 18:52:00] R6 闸门 A：18:32:19 `+3.658±0.407 z=+8.99 p≈0.0000
                        符号 211:88` → 通过，**连续第 5 轮显著**；18:32:20 晋升
                        exit_r06.pt → exit_best.pt，sha b6a91a59…（exit_best.pt ≡
                        exit_r06.pt）、runs/gates/r06.promoted=ok、exit_best.onnx
                        18:32 同步。独立冻结口径复算与驱动 summary 逐位一致。
[2026-11-06 18:52:00] 【本次最重要发现 · 自检工具 P1 缺陷并修复】
                        `tools/gate_decision.py --md` 在闸门 C 进行中直接崩溃：
                        ValueError `r06_gateC.jsonl 无有效记录`。根因 discover()
                        仅用 c.exists() 判完整轮，而闸门 A 通过后立即起闸门 C，
                        C 需 ≈22min，期间 jsonl 为 0 字节或仅部分 seed（18:49
                        实测 20/50 行）。影响：仅自检工具（驱动不调用），主循环
                        零影响，但 19:55 硬决策点有阻断风险。修复：新增
                        _complete(path,n_min) 按「非空 + 唯一 seed 数达标
                        (A=300/C=50)」判定；discover() 允许 C=None；main 进入
                        「巡检」模式 —— 正常输出 A、标注 C 待落盘、暂缓 ② 天花板
                        / ③ P2a、不崩溃。py_compile + --md + --round 5 回归通过，
                        与 #13 结论逐位一致。仅改只读工具，未碰驱动/训练/超参。
[2026-11-06 18:52:00] 【独立复算 · 边际增益与累积显著性】A 逐轮 +0.179/
                        +2.374/+2.843/+3.312/+3.398/+3.658（R2~R6 连续 5 轮
                        p≈0.0000）；环比 R6−R5 `+0.260±0.518 p≈0.6156`、
                        R5−R4 `+0.086±0.529 p≈0.8715`，单轮增量低于 seed 池
                        分辨率（≈0.52/轮）；跨轮累积 R6−R4 `+0.346±0.540
                        p≈0.5219`、R6−R3 `+0.814±0.548 p≈0.1370`、
                        **R6−R2 `+1.283±0.554 p≈0.0206`（首次跨 4 轮达 p<0.05）**。
                        C R2~R5 合并 `−0.070±0.514 p≈0.8916`（新贪心 ≈ N32 持平，
                        n=200，仍不宣布 §4 达成）。⇒ 平台期 = 单轮边际不可分辨
                        而非停滞；累积增益统计可检出，继续主循环有正收益；
                        不提前切分支 / 不改超参。
[2026-11-06 18:52:00] 决策（巡检窗口，按 §6.17.4 不提前硬决策）：① 下一步训练
                        由驱动自动接续 R6 闸门 C → R7 标注(seed0=670000,
                        student=exit_best.onnx=R6 权重) → 训练(400k 封顶,
                        init=exit_best.pt) → 闸门 A/C → R8，零干预；② 增速三选一
                        **确定不触发**（A 连续 5 轮 p≈0.0000，TAU=1.0/Q-MSE/
                        n_det/lr 全维持；该结论不依赖 C）；③ 天花板确认 / ④ P2a
                        观察态去留**暂缓**（依赖 R6 闸门 C，留首次捕获完整 A+C
                        的自检）；⑤ PPO 不启动（门槛 R10+）；⑥ 终评准备保持收口。
[2026-11-06 18:52:00] 落地：docs/14天训练计划.md §6.18、reports/夜间训练报告
                        -2026-11-06.md §21、tools/gate_decision.py（修复）、本日志块。
[2026-11-06 18:52:00] **下次必做**：先跑 `tools/gate_decision.py --md`（已修复，
                        闸门进行中也不崩）；① 若 R6 C 已完整落盘则本次即硬决策点，
                        否则 ≈19:55；② 用 R4~R6 三轮 A+C 判天花板「确认」但
                        **仍不改超参**；③ 判 P2a 观察态去留；④ 核 R7 标注进度
                        / 吞吐与 exit_best.onnx 是否为 R6 权重；⑤ 勿在 C 未落盘
                        时改 TAU=0.5（不可逆）。

[2026-11-06 18:59:20] === 自检 #14 补记：完整 A+C 落盘 → 本次升级为 R6 末硬决策点 ===
[2026-11-06 18:59:20] 18:58:31 R6 闸门 C 完整落盘（200 行 / 50 seed，1571s）：
                        `+1.347±0.920 z=+1.46 p≈0.1434 符号 27:23`；18:58:31
                        R6 轮次完成（整轮 3h18m），18:58:32 驱动自动启动 R7
                        标注（seed0=670000，学生=exit_best.onnx=R6 权重），
                        起步吞吐 9,438~9,701/h·worker（98s 冷启动瞬时值，未
                        跌破 8k 线，下次自检复核累计值）。按 §6.17.4「硬决策点
                        = 首次捕获完整 A+C 的自检」，**本次即 R6 末硬决策点**
                        （覆盖上面 18:52 块的「暂缓 / ≈19:55」表述，提前 ≈56min），
                        硬决策在 R7 训练前生效。
[2026-11-06 18:59:20] `gate_decision.py --md` 完整模式五项（冻结口径一致）：
                        ① 增速三选一 **不触发**（最近两轮 A 弱=False/False，
                        TAU=1.0/Q-MSE/n_det/lr 全维持，一项都不改）；
                        ② 天花板 **不确认、不升级**（A 三轮环比 3/3 满足，
                        但 C 平台=False：R6 C +1.347 超出 |mean|<1 ⇒ 打断
                        「C 平台化」判据；顺延 R7/R8 复核，仍不改超参）；
                        ③ **P2a 观察态解除 → 规格冻结 + 离线准备态（就绪不激活）**
                        （C 最近两段正增量 2/2：+0.407 / +1.287，A p≈0.0000；
                        不改代码/不启动/不并发/不重启/不并入主路，仅 DEADLINE
                        后或自然重启窗口试点）；④ PPO 不启动（r=6<R10）；
                        ⑤ 下一步训练零干预（驱动已自动接续 R7）。
[2026-11-06 18:59:20] 独立复算 C：全序列 −2.480/−0.293/+0.300/−0.347/
                        +0.060/+1.347；环比 R6−R5 `+1.287±1.368 p≈0.3468`
                        （不显著）；C R2~R6 合并 `+0.213±0.451 p≈0.6361`
                        （n=250，≈0）。⇒ R6 C 上扬是 1 SE 内噪声，
                        **不宣布 §4 超额目标达成**；但足以打断平台化判据。
                        P2a 双条件基于两段均不显著的 C 环比，属「就绪」非
                        「证据充分」，R7 C 若回落立即退回观察态。
[2026-11-06 18:59:20] 时间线校准（§6.18.6）：R6 整轮 3h18m（标注 2h16m +
                        训练 35m + A 1m + C 26m；C 耗时 1296→1319→1468→1571s
                        逐轮递增）；R7 标注 ≈21:15 → 训练 ≈21:50 → A ≈21:51
                        → C ≈22:17 → R8 ≈22:17；自检 ≈20:1x/21:2x/22:3x
                        ⇒ **下一硬决策点 ≈22:3x（首次捕获 R7 完整 A+C）**。
                        02:34 起 16h26m 完成 5 轮，余 ≈151.5h ⇒ 约 42~45 轮。
[2026-11-06 18:59:20] 落地（覆盖 18:52 块「下次必做」）：docs/14天训练计划.md
                        §6.18（完整硬决策版）、reports/夜间训练报告-2026-11-06.md
                        §21.6、tools/gate_decision.py（修复）、本补记块。
[2026-11-06 18:59:20] **下次必做（≈22:3x）**：先跑 `gate_decision.py --md`；
                        ① 增速仍以 A 连续两轮弱为唯一触发；② 用 R5~R7 三轮
                        复核天花板；③ 判 P2a 就绪态是否因 R7 C 回落退回观察态；
                        ④ 核 R7 4×25k 与 r07.promoted/exit_best.pt/onnx 同步；
                        ⑤ 复核 R7 累计吞吐 ≥10k/h·worker。20:1x/21:2x 只巡检，
                        勿提前决策、勿在落盘前读 r07 npz。
[2026-11-06 19:03:52] 补充复核：R7 标注 305s 累计吞吐 `9,266~9,435/h·worker`
                        （98s 为 9,438~9,701），**未回升至 10k**，较 R6 起始
                        低 ≈10%，属连续第 2 轮起始偏低（R6 10.3~10.4k →
                        R7 9.3~9.4k）。仍在 8k 异常线与 7.14k 超时线之上，
                        按 9.3k 估 25k 需 ≈2.69h < 3.5h 上限 ⇒ **不判异常、
                        不干预**，但列入 20:1x / 21:2x 重点复核项；若跌破 8k
                        立即按 annotate_r07.log 定位（负载 / 学生模型 / 热节流）。
                        02:33:57 切片 31 阶段全 rc=0、异常 0；磁盘 310Gi；
                        lock 26s；驱动唯一 PID 79443。
[2026-11-06 18:58:31]   rc=0  1571s  gateC_r06
[2026-11-06 18:58:31] R6 闸门 C(趋势): [summary] seats=[0, 1, 2, 3]  n=50  diff=+1.347±0.920  z=+1.46  p≈0.1434  符号 27:23
[2026-11-06 18:58:31] R6 轮次完成  best=exit_best.pt
[2026-11-06 18:58:31] $ /Users/ligf9/zhuoji-mahjong-ai/.venv/bin/python -c import torch,sys;sys.path.insert(0,'.');from zhuoji.net import build_from_ckpt;from zhuoji.forward import ensure_onnx;ck=torch.load('/Users/ligf9/zhuoji-mahjong-ai/models/exit_best.pt',map_location='cpu',weights_only=False);net,_=build_from_ckpt(ck);p=ensure_onnx(net,stem='exit_best');print('onnx:',p)  → export_exit_best.log
[2026-11-06 18:58:32]   rc=0  1s  export_exit_best
[2026-11-06 18:58:32] $ /Users/ligf9/zhuoji-mahjong-ai/rust_engine/target/release/annotate --student /Users/ligf9/zhuoji-mahjong-ai/models/onnx/exit_best.onnx --teacher models/onnx/dushan_master.onnx --student-enc 3 --student-device cpu --targets 100000 --threads 4 --out /Users/ligf9/zhuoji-mahjong-ai/runs/exit_r07 --seed0 670000  → annotate_r07.log

[2026-11-06 20:30:00] === 自检 #15（7×24 第 15 次 · R7 标注期常规巡检，20:1x 窗口）===
[2026-11-06 20:30:00] 结论：健康、产出符合预期（无新轮次落盘）、主循环零干预。
                        驱动唯一（lock 心跳 5s，annotate_r07.log 秒级更新，
                        night_run.py PID 79443）；阶段 = R7 DAgger 标注 ≈58.3%
                        （18:58:32 启动，seed0=670000，学生 exit_best.onnx=R6
                        权重，--student-enc 3 --student-device cpu ✓）。按硬
                        约束未启第二驱动 / 未打断 / 未并发评估 / 未改超参；
                        r07 npz 未读（落盘前 BadZipFile 竞态）。
[2026-11-06 20:30:00] R7 进度（20:28 实测）：w0 14,549 / w1 14,624 / w2 14,565
                        / w3 14,656（≈58,294/100k），已跑 1h30m；吞吐
                        9,753~9,812/h·worker（聚合 ≈39.1k/h）。ETA ≈21:33，
                        3.5h 超时上限 22:28:32，余量 ≈55min。#14 重点复核项
                        （吞吐）趋势：98s 9.44~9.70k → 305s 9.27~9.44k →
                        401s 9.48~9.60k → 5393s 9.75~9.81k，自低点持续回升
                        但仍未回到 10k，较 R6 同期低 ≈11%；>8k 异常线与
                        >7.14k 超时线余量充足 ⇒ 不判异常、不干预、继续观察；
                        若跌破 8k 立即按 annotate_r07.log 定位（负载/热节流/
                        学生模型）。
[2026-11-06 20:30:00] 健康复核：02:33:57 切片 31 阶段全 rc=0、rc!=0=0、
                        panic/NaN/Traceback/超时=0；磁盘 310Gi；TAU=1.0 未改；
                        check_npz 24/24 合规（r01~r06）；gate_decision.py --md
                        与 #14 逐位一致（A +0.179/+2.374/+2.843/+3.312/
                        +3.398/+3.658；C −2.480/−0.293/+0.300/−0.347/+0.060/
                        +1.347；A 环比 R6−R5 +0.260±0.518 p≈0.6156）；晋升
                        三件套同步（exit_best.pt ≡ exit_r06.pt 39,517,130B、
                        mtime 18:04=copy2 保留；onnx 18:32 重导出（pt mtime
                        >上轮 onnx 15:16，mtime 缓存正确失效）；r06.promoted
                        18:32）。
[2026-11-06 20:30:00] 决策：常规巡检窗口（R7 完整 A+C 未落盘，不提前决策），
                        五项全不触发：① 增速三选一不触发（A 连续 5 轮
                        p≈0.0000）；② 天花板不确认（待 R7 C）；③ P2a 维持
                        「规格冻结+离线准备态（就绪不激活）」（R7 C 若回落
                        即退回观察态）；④ PPO 不启动（r=6<10）；⑤ 下一步
                        训练由驱动自动接续 R7→训练(400k,init=exit_best.pt)
                        →闸门 A/C→R8，零干预。
[2026-11-06 20:30:00] 时间线校准（覆盖 §6.18.6-2 / #14 补记）：R7 吞吐
                        ≈9.8k < R6 11.05k ⇒ 标注落盘 ≈21:33（原估 21:15，
                        +18min）→ 训练 ≈22:08（按 35min）→ A ≈22:09 →
                        C ≈22:35~22:40（C 耗时 1296→1571s 递增，留 26~30min）
                        → R8 ≈22:35。自检节奏 ≈72min（20:26→21:3x→22:4x）
                        ⇒ **下一硬决策点 ≈22:4x（首次捕获 R7 完整 A+C）**；
                        21:3x 即使捕获落盘/训练/A 也只巡检不决策。
[2026-11-06 20:30:00] 落地：docs/14天训练计划.md §6.19、reports/夜间训练报告
                        -2026-11-06.md §22、本块。
[2026-11-06 20:30:00] **下次必做（≈21:3x 巡检；≈22:4x 硬决策）**：21:3x 核
                        R7 落盘 4×25k、训练 400k/init/train-val/早停，只记录
                        不决策；22:4x 先 gate_decision.py --md，执行① 仅当
                        A 连续两轮 p>0.2 或均值≤0 才**仅**改 train_exit.py:38
                        TAU=0.5（禁止同时改第二项）② 用 R5~R7 复核天花板
                        ③ 判 P2a 就绪态去留（R7 C 回落→退回观察态）④ 核
                        r07 4×25k 与 r07.promoted/exit_best.pt/onnx 同步
                        ⑤ 复核 R7 吞吐是否回到 ≥10k/h·worker。DEADLINE 后按
                        终评手册先预检再评估。
[2026-11-06 21:23:44]   rc=0  8712s  annotate_r07
[2026-11-06 21:23:44] R7 标注完成 student=exit_best.onnx seed0=670000 → 4×25k
[2026-11-06 21:23:44] $ /Users/ligf9/zhuoji-mahjong-ai/.venv/bin/python train_exit.py --data runs/exit_r01/exit_w*.npz runs/exit_r02/exit_w*.npz runs/exit_r03/exit_w*.npz runs/exit_r04/exit_w*.npz runs/exit_r05/exit_w*.npz runs/exit_r06/exit_w*.npz runs/exit_r07/exit_w*.npz --out /Users/ligf9/zhuoji-mahjong-ai/models/exit_r07.pt --epochs 30 --window 400000 --batch 256 --seed 20261113 --init /Users/ligf9/zhuoji-mahjong-ai/models/exit_best.pt  → train_r07.log

[2026-11-06 21:33:00] 自检 #16（7×24 第 16 次 · R7 落盘核对窗口）
                        结论：健康、产出符合预期、主循环零干预。驱动存活
                        （lock 2~5s、night_run.py PID 79443 已跑 18h50m），
                        02:33:57 切片 34 阶段全 rc=0、panic/NaN/Traceback 0、
                        磁盘 309Gi、TAU=1.0 未改；未启第二驱动 / 未打断 /
                        未并发评估。
[2026-11-06 21:33:00] R7 标注 **21:23:44 rc=0 落盘**（8712s≈2h25m，聚合
                        ≈41.3k/h = **10.25~10.30k/h·worker**，较 R6 11.05k
                        低 7% 但全程稳定无中途掉速；#15 关注项「吞吐回
                        ≥10k」确认达成，超时余量 44%）；4×25k=100k 行
                        **check_npz 4/4 全合规**（rows=25000、tiles(63,27)、
                        glob(130)、NaN↔mask=0 overlap=0 gap=0），r01~r07
                        累计 **28/28** 分片合规。
[2026-11-06 21:33:00] **21:23:44 驱动自动接续 R7 训练**（400k 滑窗封顶、
                        `--init exit_best.pt`、seed 20261113、数据 r01~r07）⇒
                        「下一步训练」已由驱动启动；ep1 进行中（≈390s/ep），
                        `float(loss)` 的 requires_grad UserWarning 与 R1~R6 同款
                        无害。
[2026-11-06 21:33:00] 复核：`tools/gate_decision.py --md` 与 #14/#15 逐位一致
                        （A +0.179/+2.374/+2.843/+3.312/+3.398/+3.658，
                        C −2.480/−0.293/+0.300/−0.347/+0.060/+1.347；A 环比
                        +2.196/+0.469/+0.469/+0.086/+0.260）；exit_best.pt ≡
                        exit_r06.pt、r06.promoted=ok、onnx 18:32 同步；闸门
                        r07 A/C 未落盘。
[2026-11-06 21:33:00] 决策：本窗口为「只记录不决策」的落盘核对（按 §6.19
                        第 4 条）；五项全不触发：① 增速三选一（A 连续 5 轮
                        p≈0.0000）；② 天花板（待 R7 C）；③ P2a 维持就绪态
                        不激活；④ PPO（r=7<10）；⑤ 零干预。
[2026-11-06 21:33:00] 时间线校准（覆盖 §6.19-3 / #15）：R7 落盘 21:23 比
                        预估 21:33 提前 10min（吞吐回升至 10.3k 高于预估
                        9.8k）⇒ 训练结束 ≈21:59（按 R6 2079s~35min）→
                        闸门 A ≈22:00 → 闸门 C ≈**22:26~22:31**（C 耗时
                        1296→1571s 递增）→ R8 标注 ≈22:30。自检节奏 ≈72min
                        （21:33→22:4x）⇒ **硬决策点 ≈22:4x 不变**（首次捕获
                        R7 完整 A+C）。
[2026-11-06 21:33:00] 落地：docs/14天训练计划.md §6.20、reports/夜间训练
                        报告-2026-11-06.md §23、本块。
[2026-11-06 21:33:00] **下次必做（≈22:4x，R7 完整 A+C 硬决策点）**：先
                        `tools/gate_decision.py --md`；① 仅当 A 连续两轮
                        p>0.2 或均值≤0 才**仅**改 `train_exit.py:38
                        TAU=0.5`（禁止同时改第二项）；② 用 R5~R7 三轮复核
                        天花板（R6 C +1.347 打断平台化，需 R7 C 回 0 附近
                        才重新具备确认条件）；③ 判 P2a 就绪态去留（R7 C
                        回落→退回观察态）；④ 核 r07.promoted /
                        exit_best.pt / onnx 三者同步与 R7 最终 train-val /
                        早停 epoch；⑤ R8 标注进度与吞吐。DEADLINE 后按终评
                        手册先预检再评估。
[2026-11-06 21:56:05]   rc=0  1941s  train_r07
[2026-11-06 21:56:05] R7 训练完成 init=exit_best.pt → exit_r07.pt  (['[done] best val 1.7765 → /Users/ligf9/zhuoji-mahjong-ai/models/exit_r07.pt'])
[2026-11-06 21:56:05] $ /Users/ligf9/zhuoji-mahjong-ai/.venv/bin/python eval_pairs.py --a /Users/ligf9/zhuoji-mahjong-ai/models/exit_r07.pt --b /Users/ligf9/zhuoji-mahjong-ai/models/dushan_master.pt --mode greedy --seeds 300 --seed0 710000 --workers 6 --out /Users/ligf9/zhuoji-mahjong-ai/runs/gates/r07_gateA.jsonl  → gateA_r07.log
[2026-11-06 21:56:56]   rc=0  51s  gateA_r07
[2026-11-06 21:56:56] R7 闸门 A: [summary] seats=[0, 1, 2, 3]  n=300  diff=+3.353±0.428  z=+7.84  p≈0.0000  符号 200:98  → 通过
[2026-11-06 21:56:56] $ /Users/ligf9/zhuoji-mahjong-ai/.venv/bin/python -c import torch,sys;sys.path.insert(0,'.');from zhuoji.net import build_from_ckpt;from zhuoji.forward import ensure_onnx;ck=torch.load('/Users/ligf9/zhuoji-mahjong-ai/models/exit_best.pt',map_location='cpu',weights_only=False);net,_=build_from_ckpt(ck);p=ensure_onnx(net,stem='exit_best');print('onnx:',p)  → export_exit_best.log
[2026-11-06 21:56:58]   rc=0  1s  export_exit_best
[2026-11-06 21:56:58] 晋升：exit_r07.pt → exit_best.pt
[2026-11-06 21:56:58] $ /Users/ligf9/zhuoji-mahjong-ai/.venv/bin/python eval_pairs.py --a /Users/ligf9/zhuoji-mahjong-ai/models/exit_r07.pt --b /Users/ligf9/zhuoji-mahjong-ai/models/dushan_master.pt --a-mode greedy --b-mode search32 --seeds 50 --seed0 720000 --workers 6 --out /Users/ligf9/zhuoji-mahjong-ai/runs/gates/r07_gateC.jsonl  → gateC_r07.log
[2026-11-06 22:17:21]   rc=0  1224s  gateC_r07
[2026-11-06 22:17:21] R7 闸门 C(趋势): [summary] seats=[0, 1, 2, 3]  n=50  diff=-0.460±1.114  z=-0.41  p≈0.6797  符号 25:24
[2026-11-06 22:17:21] R7 轮次完成  best=exit_best.pt
[2026-11-06 22:17:21] $ /Users/ligf9/zhuoji-mahjong-ai/.venv/bin/python -c import torch,sys;sys.path.insert(0,'.');from zhuoji.net import build_from_ckpt;from zhuoji.forward import ensure_onnx;ck=torch.load('/Users/ligf9/zhuoji-mahjong-ai/models/exit_best.pt',map_location='cpu',weights_only=False);net,_=build_from_ckpt(ck);p=ensure_onnx(net,stem='exit_best');print('onnx:',p)  → export_exit_best.log
[2026-11-06 22:17:23]   rc=0  1s  export_exit_best
[2026-11-06 22:17:23] $ /Users/ligf9/zhuoji-mahjong-ai/rust_engine/target/release/annotate --student /Users/ligf9/zhuoji-mahjong-ai/models/onnx/exit_best.onnx --teacher models/onnx/dushan_master.onnx --student-enc 3 --student-device cpu --targets 100000 --threads 4 --out /Users/ligf9/zhuoji-mahjong-ai/runs/exit_r08 --seed0 680000  → annotate_r08.log
[2026-11-06 23:41:16] ============================================================
[2026-11-06 23:41:16] 7×24 ExIt 驱动启动：R8 起，截止 2026-11-13 02:30（R1 已完成；断点续跑 + 每小时自检自动化）
[2026-11-06 23:41:16] $ /Users/ligf9/zhuoji-mahjong-ai/.venv/bin/python -c import torch,sys;sys.path.insert(0,'.');from zhuoji.net import build_from_ckpt;from zhuoji.forward import ensure_onnx;ck=torch.load('/Users/ligf9/zhuoji-mahjong-ai/models/exit_best.pt',map_location='cpu',weights_only=False);net,_=build_from_ckpt(ck);p=ensure_onnx(net,stem='exit_best');print('onnx:',p)  → export_exit_best.log
[2026-11-06 23:41:17]   rc=0  1s  export_exit_best
[2026-11-06 23:41:17] $ /Users/ligf9/zhuoji-mahjong-ai/rust_engine/target/release/annotate --student /Users/ligf9/zhuoji-mahjong-ai/models/onnx/exit_best.onnx --teacher models/onnx/dushan_master.onnx --student-enc 3 --student-device cpu --targets 100000 --threads 4 --out /Users/ligf9/zhuoji-mahjong-ai/runs/exit_r08 --seed0 680000  → annotate_r08.log
[2026-11-06 23:52:00] ========== 自检 #17（7×24 第 17 次 · R7 完整 A+C 硬决策点）==========
[2026-11-06 23:52:00] 【事故·已恢复】原驱动 PID 79443（02:34 启动、连续运行
                        19h45m）于 22:17:23~23:41:16 之间死亡。证据：主日志
                        22:17:23 后无任何写入；无 Python 崩溃报告；pmset 显示
                        全程持有 Electron NoIdleSleepAssertion（**非系统休眠**）；
                        进程树显示驱动是 WorkBuddy 会话 shell 的子进程 ⇒ 判定
                        **会话销毁连带终止进程组**，非程序崩溃、非数据损坏。
                        23:41:16 由自动化按预案重启（唯一 PID 89566），正确
                        识别「R8 起」并从第一个未完成阶段续跑；已完成数据零
                        损失（r01~r07 = 28/28 分片完整）。代价：R8 标注从 0
                        重跑，损失 83.9min 标注进度（annotate 只在结束时落盘）。
[2026-11-06 23:52:00] 【当前状态·健康】新驱动 89566（lock 心跳 1.7~24.5s）+
                        annotate 89568 在跑 R8（23:41:16 起、seed0=680000、
                        学生=exit_best.onnx=R7 权重、enc3/cpu ✓）。R8 进度
                        ≈6.6k/25k·worker（26%），吞吐 11.1~12.9k/h·worker，
                        ETA ≈01:40 < 3.5h 上限 03:11。02:33:57 切片 37 阶段
                        全 rc=0、panic/NaN/超时 0；磁盘 309Gi；Web 8770
                        HTTP 200；TAU=1.0 未改。
[2026-11-06 23:52:00] 【R7 完整一轮产出】标注 21:23:44 rc=0（8712s、
                        10.25~10.30k/h·worker、4×25k、28/28 合规）；训练
                        21:56:05 rc=0（1941s、400k 封顶、init=exit_best=R6，
                        train 1.7806→1.7553、val 1.7765→1.7829、
                        **best val 在 ep1**、ep4 早停）；闸门 A
                        `+3.353±0.428 p≈0.0000` 连续第 6 轮通过 → 21:56:58
                        晋升 exit_r07.pt→exit_best.pt（两侧 sha
                        1629296193c5a9f… 一致、r07.promoted=ok、onnx 21:56
                        同步）；闸门 C `−0.460±1.114 p≈0.6797`；整轮 3h16m。
                        冻结口径独立复算与驱动 [summary] 逐位一致。
[2026-11-06 23:52:00] 【本次最有价值结论：主循环边际增益已归零 4 轮，瓶颈
                        已确认为教师天花板】
                        ① 闸门 A 环比：R5−R4 +0.086±0.529(p=.87)、R6−R5
                        +0.260±0.518(p=.62)、R7−R6 −0.304±0.551(p=.58)、
                        **R7−R4 +0.041±0.545(p=.94)** ⇒ 4 轮累计 +0.04，
                        A 绝对值停在 +3.3~3.7。
                        ② 闸门 C 6 轮持平：−0.29/+0.30/−0.35/+0.06/+1.35/
                        −0.46；R7−R2 `−0.167±1.196(p=.89)`、R7−R5
                        `−0.520±1.448(p=.72)` ⇒ 学生贪心 ≈ N32 搜索，「搜索
                        蒸进网络」已饱和，且不再收敛（无新可蒸）。
                        ③ 训练侧铁证：R5/R6/R7 三轮 **best val 均出现在
                        ep1**、之后单调恶化、ep4~5 早停 ⇒ 从 exit_best 出发
                        的续训在 1 个 epoch 后即过拟合约 400k 窗口，网络已在
                        数据最优点附近，**不是拟合能力不足，而是标签上限**。
                        ④ 标签实测（本次量化）：R7 分片 q 合法值 −26.8~63.2、
                        每样本 max−min 中位 0.81 分、合法动作均值 8.4 个 ⇒
                        τ=1.0 目标熵 1.7006 nats、**τ=0.5 仅降至 1.5151
                        （−11%）**、τ=0.25 至 1.2592。即 τ 杠杆偏小，且
                        q 差异多在噪声量级（<1 分）时锐化会放大 argmax 噪声。
[2026-11-06 23:52:00] 【决策五项】
                        ① 增速三选一 **不触发**：A 绝对强度 6 轮 p≈0.0000 且
                        均值 >0，用户判据（连续两轮 p>0.2 或均值 ≤0）未满足；
                        TAU/Q-MSE/n_det/lr 一项都不改。补充两条反证支撑：
                        (a) ③ n_det=64 依据项目自身实测「N32 略强于 N128」，
                        未必抬天花板而标注成本 ×2；(b) ① τ=0.5 实测仅 −11%
                        熵，EV 低。
                        ② 天花板**不确认但记为「边际停滞」**：A 环比 3/3 不
                        显著已满足，C 平台化按严格定义（连续 3 轮 |C|<1.0）
                        被 R6 +1.347 破坏 ⇒ 不改超参（遵守「44 轮多重比较下
                        不反复调参防 seed 池过拟合」），但主循环已无正收益，
                        转入分支准备。
                        ③ P2a 由「就绪态」按预注册规则**退回观察态**（R7 C
                        回落）；但本次把 P2a 从「需先实现」推进为**开关已落地、
                        可实验态**（见下）。
                        ④ PPO-lite 不启动（r=7 < R10）。
                        ⑤ 主循环**零干预**：R8 已由新驱动自动接续，未启第二
                        驱动、未打断、未与标注并发评估。
[2026-11-06 23:52:00] 【落地：P2a 开关实现（零风险、默认关闭）】查源码确认
                        `search.rs:262` 已存在完美信息分支 `if !fair
                        { game.clone() }`、`annotate.rs` 教师搜索硬编码
                        fair=true；故 `--oracle 1` 只需把 fair 透传（约 12 行）。
                        已在 `rust_engine/src/bin/annotate.rs` 加
                        `teacher_q(.., fair)` 参数 + `--oracle` 解析（值形，
                        `--oracle 0/false` 或省略 ⇒ **与历史行为完全一致**），
                        并用 `CARGO_TARGET_DIR=rust_engine/target_oracle -j 2
                        nice 10` **编译到独立目录**，绝不触碰在跑二进制
                        `target/release/annotate`；主循环正在用的 driver 未
                        重读源码，行为不变。另查实 `--n-det` 已存在（默认 32），
                        教师升级不需要重编译，只需 night_run.py 加参数（但
                        长驻驱动不重载源码 ⇒ 需重启窗口）。
[2026-11-06 23:52:00] 【下一步训练（已启动 + 计划）】R8 标注 → 01:40 训练
                        (400k, init=exit_best) → 闸门 A/C → R9，全自动。
                        分支计划：**R8 训练窗口（≈01:40~02:15，CPU 空闲）**
                        跑 20k oracle 标签对照（α=1.0 纯 Q_oracle，成本远低于
                        n_det=32：单局面搜索而非 32 次 determinize），train
                        20k ckpt，仅用冻结 300 seed 闸门 A 与主路 best 对比；
                        显著更优才考虑并入。**R10 为硬切换点**：若 R8~R10 边际
                        仍 ≈0，则按 P2a 退火（α 1.0→0.5→0.0）并入主循环，需
                        驱动重启窗口改 night_run.py。
[2026-11-06 23:52:00] 【登记缺陷/教训】P1：驱动是会话语义子进程，已致 1 次
                        83.9min 损失（占 168h 的 0.8%）；建议在自然重启窗口改
                        launchd KeepAlive 托管（属系统级变更，需用户确认），
                        当前由每小时自检兜底，最坏损失 ≈1 轮。P1：`annotate`
                        **不支持 --help 且会真启动**，本次误跑 2min（PID
                        90254，占用 16% CPU 使 R8 吞吐瞬时 12.9k→11.1k，已
                        TERM/KILL 并确认恢复、无脏文件），此后查参数一律
                        grep 源码。P0 遗留：enc3 search32 内核（终评主口径）
                        仍未解决。


[2026-11-07 01:30:00] ========== 自检 #18（7×24 第 18 次 · R8 标注期巡检）==========
[2026-11-07 01:30:00] 【当前状态·健康·零干预】驱动唯一 PID 89566（23:41 重启、1h22m），
                        lock 8~30s；R8 标注 69%（17,295/25,000·worker），吞吐
                        11,314~11,348/h·worker，ETA ≈01:55（上限 03:11，余 76min）；
                        37 阶段 rc=0、panic 0、磁盘 309Gi、TAU=1.0、24/24 npz 合规。
                        未启第二驱动、未打断、未与标注并发评估。
[2026-11-07 01:30:00] 【判据修正·必须记住】23:41 重启的驱动 stdout 绑 /dev/ttys002，
                        ⇒ runs/night_run.log 自 22:17 起冻住，**不再是心跳信号**；
                        权威实时日志是 reports/exit_log.md。严禁用 night_run.log 的
                        mtime 判死（会误触发第二个驱动）。存活 = lock<120s ∧ PID 未变
                        ∧ exit_log/annotate 有本轮阶段行。下次重启须 nohup 并核 fd。
[2026-11-07 01:30:00] 【P2a 端到端首次跑通】target_oracle/annotate --oracle 1：
                        800 targets/1 线程/教师 CoreML = 223s（12,890/h·worker），
                        与 PIMC 同量级 ⇒ §6.21.5 的 8m40s 慢**只是教师走了 --device cpu**。
                        配对成立（同 student + 同 seed0=670000，800/800 状态逐位相同）。
[2026-11-07 01:30:00] 【诊断·oracle vs PIMC】argmax 一致率仅 38.8%，但不一致里
                        81.2% 的 gap<1 分、48.4%<0.25 分；≥1 分只占全部状态 11.5%。
                        合法 Q 均值 oracle 比 PIMC **低 0.847**、maxQ 低 0.261
                        ⇒ **是 PIMC 标签系统性偏乐观**，不是 oracle。τ=1 熵几乎相同。
                        ⇒ P2a 下调为「成本可行、证据不足、不进主路」。
[2026-11-07 01:30:00] 【本轮核心·平台期机制证据】教师自身 top1−top2 间隔中位
                        **0.144 分**（oracle 0.099），**63.6% 的状态 <0.25 分**；
                        学生 argmax 与标签只对上 **52.5%**，但闸门 C≈0（≈N32 强度）
                        ⇒ 那 47.5% 分歧几乎全是无价值差异的近平局。
                        **标签在行动维度已到噪声底**，同配方 DAgger 无法再增加信息。
                        另：logit vs centered-Q 斜率 0.840/R² 0.702（已按分标定）、
                        价值头 V*8 ~ maxQ 相关 0.948/残差 0.69 分。
                        ⇒ 增速三选一（τ/ Q-MSE/ n_det）**三项全部量化否决**，此后不再调参。
[2026-11-07 01:30:00] 【已执行·产品】ExIt 学生首次上线：cp exit_best.pt→models/web_best.pt，
                        MASTER_CANDIDATES 前置 web_best（备份 dushan_server.py.bak-20261107），
                        影子 8771 验证（meta/new/state/selftest3 全 PASS，150ms/局），
                        确认 8770 无连接后蓝绿切换 → 双 200、models=web_best(9.85M)、
                        fd 已脱离会话、单步 92ms。回退：删 web_best.pt 或还原 .bak。
                        注：**绝不能覆盖 dushan_master.pt 换代**（它是闸门 A/C 冻结基线）。
[2026-11-07 01:30:00] 【决策五项】① 主循环零干预（R8 → ≈01:55 训练 → 闸门 → R9）；
                        ② 超参/损失一项不改；③ 分支切换点 R10→**R9 末（≈05:3x）**；
                        ④ **PPO-lite 优先、P2a 降级**（终端奖励绕开标签偏差，唯一能超
                        教师上限的路径）；⑤ 切换前置 = PPO 冒烟必须通过，否则不切。
[2026-11-07 01:30:00] 【终评缺陷与修正】闸门 A 只与 master 贪心比、**不与上轮最佳比**
                        ⇒ 每轮必通过、exit_best 在噪声上随机游走（R7−R6=−0.304 即证据）。
                        改为两段式：冻结池排名（当前 r07 +0.341 / r03 +0.281 / r05 +0.044
                        / r01 +0.043 / r06 −0.073 / r04 −0.146 / r02 −0.147，最高−次高
                        =+0.060，合并 SE 0.788 ⇒ 不可分辨）→ top3~5 在独立池 3_000_000+500
                        复赛定交付。已写入终评手册 §8。
[2026-11-07 01:30:00] 【下一步】R8 落盘 → 训练(400k, init=exit_best) → 闸门 A/C → R9，
                        全程自动。下次自检核 R8 4×25k + 训练曲线 + 完整 A+C；R9 末尾为
                        分支硬决策点，先做 PPO 规模测算（2k 局冒烟 >1.5h 则判不可行、
                        退回扩大终评+产品打磨，不硬上）。
[2026-11-07 01:52:34]   rc=0  7876s  annotate_r08
[2026-11-07 01:52:34] R8 标注完成 student=exit_best.onnx seed0=680000 → 4×25k
[2026-11-07 01:52:34] $ /Users/ligf9/zhuoji-mahjong-ai/.venv/bin/python train_exit.py --data runs/exit_r01/exit_w*.npz runs/exit_r02/exit_w*.npz runs/exit_r03/exit_w*.npz runs/exit_r04/exit_w*.npz runs/exit_r05/exit_w*.npz runs/exit_r06/exit_w*.npz runs/exit_r07/exit_w*.npz runs/exit_r08/exit_w*.npz --out /Users/ligf9/zhuoji-mahjong-ai/models/exit_r08.pt --epochs 30 --window 400000 --batch 256 --seed 20261114 --init /Users/ligf9/zhuoji-mahjong-ai/models/exit_best.pt  → train_r08.log
[2026-11-07 02:27:21]   rc=0  2087s  train_r08
[2026-11-07 02:27:21] R8 训练完成 init=exit_best.pt → exit_r08.pt  (['[done] best val 1.7708 → /Users/ligf9/zhuoji-mahjong-ai/models/exit_r08.pt'])
[2026-11-07 02:27:21] $ /Users/ligf9/zhuoji-mahjong-ai/.venv/bin/python eval_pairs.py --a /Users/ligf9/zhuoji-mahjong-ai/models/exit_r08.pt --b /Users/ligf9/zhuoji-mahjong-ai/models/dushan_master.pt --mode greedy --seeds 300 --seed0 710000 --workers 6 --out /Users/ligf9/zhuoji-mahjong-ai/runs/gates/r08_gateA.jsonl  → gateA_r08.log
[2026-11-07 02:28:20]   rc=0  60s  gateA_r08
[2026-11-07 02:28:20] R8 闸门 A: [summary] seats=[0, 1, 2, 3]  n=300  diff=+3.694±0.415  z=+8.90  p≈0.0000  符号 216:82  → 通过
[2026-11-07 02:28:20] $ /Users/ligf9/zhuoji-mahjong-ai/.venv/bin/python -c import torch,sys;sys.path.insert(0,'.');from zhuoji.net import build_from_ckpt;from zhuoji.forward import ensure_onnx;ck=torch.load('/Users/ligf9/zhuoji-mahjong-ai/models/exit_best.pt',map_location='cpu',weights_only=False);net,_=build_from_ckpt(ck);p=ensure_onnx(net,stem='exit_best');print('onnx:',p)  → export_exit_best.log
[2026-11-07 02:28:22]   rc=0  2s  export_exit_best
[2026-11-07 02:28:22] 晋升：exit_r08.pt → exit_best.pt
[2026-11-07 02:28:22] $ /Users/ligf9/zhuoji-mahjong-ai/.venv/bin/python eval_pairs.py --a /Users/ligf9/zhuoji-mahjong-ai/models/exit_r08.pt --b /Users/ligf9/zhuoji-mahjong-ai/models/dushan_master.pt --a-mode greedy --b-mode search32 --seeds 50 --seed0 720000 --workers 6 --out /Users/ligf9/zhuoji-mahjong-ai/runs/gates/r08_gateC.jsonl  → gateC_r08.log
[2026-11-07 02:49:53]   rc=0  1291s  gateC_r08
[2026-11-07 02:49:53] R8 闸门 C(趋势): [summary] seats=[0, 1, 2, 3]  n=50  diff=+0.807±0.964  z=+0.84  p≈0.4028  符号 28:21
[2026-11-07 02:49:53] R8 轮次完成  best=exit_best.pt
[2026-11-07 02:49:53] $ /Users/ligf9/zhuoji-mahjong-ai/.venv/bin/python -c import torch,sys;sys.path.insert(0,'.');from zhuoji.net import build_from_ckpt;from zhuoji.forward import ensure_onnx;ck=torch.load('/Users/ligf9/zhuoji-mahjong-ai/models/exit_best.pt',map_location='cpu',weights_only=False);net,_=build_from_ckpt(ck);p=ensure_onnx(net,stem='exit_best');print('onnx:',p)  → export_exit_best.log
[2026-11-07 02:49:54]   rc=0  1s  export_exit_best
[2026-11-07 02:49:54] $ /Users/ligf9/zhuoji-mahjong-ai/rust_engine/target/release/annotate --student /Users/ligf9/zhuoji-mahjong-ai/models/onnx/exit_best.onnx --teacher models/onnx/dushan_master.onnx --student-enc 3 --student-device cpu --targets 100000 --threads 4 --out /Users/ligf9/zhuoji-mahjong-ai/runs/exit_r09 --seed0 690000  → annotate_r09.log

[2026-11-07 03:00:00] ======== 自检 #19（7×24 第 19 次）R8 完整 A+C 捕获 + PPO 冒烟提前通过 ========
[2026-11-07 03:00:00] 【状态】健康、零干预。驱动唯一 PID 89566（23:41 重启、3h18m），
                        lock 2~30s；R9 标注 02:49:54 起，9.38~9.53k/h·worker
                        （起步 8.0k 低点是本机自检进程抢 CPU，已回升），ETA ≈05:28；
                        41 阶段 rc=0、panic/NaN 0、磁盘 308Gi、TAU=1.0、32/32 npz 合规。
[2026-11-07 03:00:00] 【R8 结果】标注 01:52:34 rc=0（7876s、11.4k/h·worker、4×25k）；
                        训练 02:28:20 rc=0（2082s）train 1.7742→1.7504、
                        best val 1.7708@ep1、early-stop ep4（第 5 轮连续 best-val 落在 ep1）；
                        闸门 A +3.694±0.415 z=+8.90 p≈0.0000（216:82）**连续第 7 轮通过**
                        → 02:28:22 晋升 exit_r08 → exit_best（onnx 同步）；
                        闸门 C +0.807±0.964 p≈0.4028（28:21），较 R7 −0.460 回升；
                        环比 R8−R7 = +0.341±0.552 p≈0.5367（不可分辨，符号为正）；
                        R8 整轮 3h16m（01:52:34 → 02:49:53）。
[2026-11-07 03:00:00] 【PPO 前置条件提前满足】新 tools/ppo_scale_probe.py 成本实测：
                        纯策略自对弈 0.62 s/局、62.6 决策/局（40 局样本）；
                        ⇒ 2k 局 0.35h（单进程）/0.09h（4 进程），远低于 1.5h 预算。
                        新 tools/ppo_smoke.py 跑 **2000 局**（0.93 s/局，含与闸门 C 并发）
                        127,653 决策、熵中位 2.179 nats，四项判据全 PASS：
                        ① 零和 max|Σdeltas|=0；② 终端奖励 std=10.84 分；
                        ③ Φ=8V 塑形后优势 std=7.43 分（中位|A| 3.35）；
                        ④ PPO 更新 79 步 loss=0.399 pg=−0.005 vl=0.807 ratio=1.0004
                        clipfrac=0.115 ent=1.828 ⇒ **R9 末分支切换的技术前置已解除**。
                        注：探针只读、不落盘、不覆盖 exit_best；与闸门 C 并发未拖慢
                        （gateC 1291s，快于 R7 的 1571s）。
[2026-11-07 03:00:00] 【决策】① 主循环零干预：R9 已由驱动 02:49:54 自动接续；
                        ② 超参/损失仍一项不改；③ 分支切换点维持 **R9 末**，
                        但「前置=PPO 冒烟」已通过，届时只需一次 A 复核即可执行；
                        ④ P2a 维持观察态；⑤ 终评两段式 best-of 的排名命令已确认
                        可直接用 final_eval_report.py 多轮并列（R01..R08 同池），无需另写工具。
[2026-11-07 03:00:00] 【计划校准】R9 标注 02:49:54 → ≈05:28（9.45k/h·worker）
                        → 训练 ≈06:13 → 闸门 A ≈06:14 → 闸门 C ≈06:40；
                        **R9 末硬决策点 = 首次捕获完整 A+C 的自检，≈06:4x**
                        （覆盖 §6.22.8 的「≈05:3x」）。剩余 ≈143.5h ⇒ 仍 40+ 轮。
[2026-11-07 03:00:00] 【下一步】R9 → 训练 → 闸门 A/C；06:4x 硬决策：
                        PPO-lite 已具备切换条件（成本 6k 局/轮 ≈0.4h@4进程、
                        状态池上限 2×10^5 float16 ≈670MB），按「PPO 分支 + 保留 DAgger 复核」
                        执行；切换须在阶段边界重启驱动（改 night_run.py），
                        避免杀死进行中的标注。

[2026-11-07 04:15:00] ======== 自检 #20（7×24 第 20 次）PPO 分支工程化落地 + 闸门 D 补齐 ========
[2026-11-07 04:15:00] 【状态】健康、零干预。驱动唯一 PID 89566（23:41 重启、4h20m），lock 1.4~7s；
                        R9 标注 54.3%（w0~w3 各 ≈13.5k/25k、seed0=690000、学生=R8 权重 enc3/cpu），
                        吞吐 11,446~11,544/h·worker（聚合 45.9k/h），ETA ≈05:01；
                        02:33:57 起 37 阶段 rc=0、panic/NaN 0、磁盘 308Gi、TAU=1.0；
                        Web 8770 双 200 且 models=web_best；keepalive PID 99107 活。
[2026-11-07 04:15:00] 【复核】gate_decision 与 §6.23 逐位一致（8 轮 A/C 全同）；
                        check_npz --round r08 4/4 合规；exit_best ≡ exit_r08（6d35b3a6）、
                        r08.promoted=ok、onnx 02:28 同步 ⇒ 三件套一致。
[2026-11-07 04:15:00] 【本轮核心·三个新构件】把 PPO 从「冒烟通过」变成「可运行生产件」：
                        ① 新建 train_ppo.py（393 行）：纯策略自对弈 rollout + 终端分差奖励 +
                          Φ=8V 塑形 + GAE(γ1,λ0.95) + clipped PPO + KL 早停；
                          rollout 4 进程、更新走 mps（实测 mps 0.70s/step vs cpu 3.30s/step，4.7×）；
                          产出 ckpt 与 train_exit 完全同格式，eval_pairs 直读（已实测跑通）。
                          实测标定：960 局≈2min + 60k 状态×2ep≈4.5min ≈ **8min/轮（占一轮 4%）**。
                        ② night_run.py 新增 ppo_round()：PPO_FROM=10、PPO_EVERY=1、PPO_HANDS=960、
                          runs/ppo_disabled 急停开关；位于 gates_round 之后，产出 exit_pNN.pt，
                          **晋升判据 = 闸门 D（vs 现最佳）p<0.05 且均值>0 且闸门 A 通过**；
                          整段独立 try/except，异常不计 consecutive_fail、不影响 DAgger 主线。
                        ③ 补齐规格里一直缺的「上轮最佳」基线：snapshot_prev_best() 在 promote
                          之前快照 exit_best，闸门 A/C 之后跑 300 seed 配对 → rNN_gateD.jsonl。
                          这正是 §6.22.6 缺陷（A 恒过 ⇒ best 在噪声上随机游走，R7−R6=−0.304 仍晋升）
                          的根因，成本仅 60s/轮。D 只记录不改 DAgger 晋升口径（保 R1~R8 可比）。
[2026-11-07 04:15:00] 【切换方案】新建 tools/switch_ppo.py，在「R10 标注启动行 +3s」精确 kill
                        （此刻标注器尚未落盘任何 npz ⇒ 零数据损失），交由 keepalive
                        （start_new_session，本机唯一验证过可跨会话存活的拉起方式）重启。
                        已启动 pid 12419（PPID=1，已脱离会话）。未命中则什么都不做，主线无风险。
[2026-11-07 04:15:00] 【决策】① 主循环零干预（R9 训练 → A/C → R10 自动接续）；
                        ② 超参/损失仍一项不改；③ PPO R10 起每轮启用、保留 DAgger；
                        ④ P2a 维持观察态；⑤ 终评候选池 = r01..rNN ∪ pNN，两段式 best-of 独立池复赛。
[2026-11-07 04:15:00] 【预算】每轮 ≈3h26m（3h17m + PPO 8min + 闸门D 1min），剩余 ≈141h ⇒ ≈40 轮。
[2026-11-07 04:15:00] 【下一步】05:1x 巡检 R9 落盘 4×25k + 训练 400k/init/train-val/早停；
                        ≈06:1x 捕获切换结果（switch_ppo.log + PID 变化 + P10 PPO 完成行）；
                        ≈06:4x 硬决策：R9 完整 A+C 环比 + 首个 PPO 轮 D/A 统计 + 是否晋升。
[2026-11-07 05:01:20]   rc=0  7886s  annotate_r09
[2026-11-07 05:01:20] R9 标注完成 student=exit_best.onnx seed0=690000 → 4×25k
[2026-11-07 05:01:20] $ /Users/ligf9/zhuoji-mahjong-ai/.venv/bin/python train_exit.py --data runs/exit_r01/exit_w*.npz runs/exit_r02/exit_w*.npz runs/exit_r03/exit_w*.npz runs/exit_r04/exit_w*.npz runs/exit_r05/exit_w*.npz runs/exit_r06/exit_w*.npz runs/exit_r07/exit_w*.npz runs/exit_r08/exit_w*.npz runs/exit_r09/exit_w*.npz --out /Users/ligf9/zhuoji-mahjong-ai/models/exit_r09.pt --epochs 30 --window 400000 --batch 256 --seed 20261115 --init /Users/ligf9/zhuoji-mahjong-ai/models/exit_best.pt  → train_r09.log

[2026-11-07 05:15:00] ======== 自检 #21（7×24 第 21 次）R9 落盘核对 + PPO 观察器落地 + 切换窗口收口 ========
[2026-11-07 05:15:00] 【状态】健康、产出符合预期、主线零干预。驱动唯一 PID 89566（23:41 重启、5h34m），
                        lock 5~25s；R9 标注 05:01:20 rc=0 落盘（7886s≈2h11m、11,419~11,446/h·worker、
                        4×25k），check_npz --round r09 4/4、累计 36/36 合规；
                        02:33:57 切片 41 阶段全 rc=0、panic/NaN/Traceback/超时 0；
                        磁盘 308Gi、TAU=1.0、Web 双 200 且 models=web_best(9.85M)、
                        看门狗 99107 活（3h44m）、switch_ppo 12419 武装中（等待 R10 标注启动行）。
[2026-11-07 05:15:00] 【R9 训练（进行中）】05:01:20 起，400k 封顶、init=exit_best.pt(=exit_r08)、
                        seed 20261115；ep01 train 1.7742 / val 1.7681 / 506s。
                        注：**R9 的 ep01 train 1.7742 与 R8 的 ep01 完全相同（4 位小数）**
                        —— 同 init(=exit_r08) + 窗口 400k/400k 高度重叠 ⇒ 每轮新增 100k 标签
                        只换掉 1/4 数据且都是同一噪声底，loss 差异 <1e-4。
                        **这不是异常，是「增量数据已无新信息」的又一条旁证**（§6.25.3）。
[2026-11-07 05:15:00] 【决策五项（gate_decision.py --md 复核，与 #20 逐位一致）】
                        ① 增速三选一 不触发（最近两轮 A 弱=False/False，TAU=1.0 不动）；
                        ② 天花板 不确认（三轮环比 3/3 满足但 C 平台=False，仍不改超参）；
                        ③ P2a 维持观察态；④ PPO 当前轮判「不启动」（r=8，门槛 R10+，符合预期）；
                        ⑤ 下一步训练由驱动自动接续（R9 训练 → A/C → R10），零干预。
[2026-11-07 05:15:00] 【本轮唯一新构件】新建 tools/ppo_report.py（只读/零 CPU/不启对局）：
                        把「PPO 是否在学」的三条同池口径固化 —— P{r} 闸门 A(vs master)、
                        P{r} 闸门 D(vs 现最佳)、以及**关键判据 P{r}−R{r} 逐 seed 配对**
                        （PPO 相对同期 DAgger 的净增量）。复用 final_eval_report 的
                        per_seed/stats 唯一实现，禁止另写口径。已验证：用 r08/r07 跑出
                        +0.341±0.552 p≈0.5367，与 gate_decision 逐位一致；缺失/半截 jsonl
                        正确返回 None 而非拿半截数据下结论；首跑正确输出「尚无 PPO 轮次」。
[2026-11-07 05:15:00] 【切换窗口收口（风险量化，不干预）】R9 训练 ≈05:36 结束（4 ep 早停）
                        → 闸门 A ≈05:37 → 闸门 C ≈05:59 → **R10 标注启动 ≈05:59，
                        switch_ppo 在 +3s 命中**（此刻标注器零落盘）→ keepalive 于 lock
                        过期 120s 后拉起新驱动 ≈06:01~06:03，R10 从头重跑，净损失 ≈3s。
                        switch_ppo 超时点在 07:40 ⇒ 即使闸门 C 打满 5400s 仍有 ≈95min 余量。
                        **回退规则**：若 06:2x 自检发现 switch_ppo 已退出且无【切换】行 ⇒
                        在 R11 标注启动边界重试（命令行已固化，见 §6.25.6）。
[2026-11-07 05:15:00] 【新登记的耦合风险（不阻断，写入纪律）】PPO 晋升会把 exit_pNN 覆盖成
                        exit_best.pt ⇒ **下一轮 DAgger 的学生分布随之改变**。闸门 D(p<0.05)
                        + 闸门 A 双条件已把误晋升概率压到很低，且 rNN_gateD 会记录真实环比，
                        漂移可被终评发现。另：**线上 models/web_best.pt 绝不跟随 PPO 晋升**
                        （只允许从 DAgger rNN 刷新，且须闸门 D 通过 + 影子冒烟 + 蓝绿切换）。
[2026-11-07 05:15:00] 【下一步】R9 闸门 A/C 首次完整捕获的自检（≈06:2x）= R9 硬决策点；
                        同时确认【切换】是否发生、新驱动 PID、首个 PPO 轮 P10 是否跑起来。
[2026-11-07 05:20:00] 【补做·PPO 命令链路实测（原以为已验，查实无任何 ppo 日志 ⇒ 重验）】
                        以 `train_ppo.py --init exit_best.pt --out runs/selftest/... --hands 24
                        --workers 2 --max-states 3000 --epochs 1 --rollout-device cpu --device cpu`
                        端到端跑通（46s）：rollout 24 局/1482 决策/1482 状态、熵中位 2.136、
                        零和 0.00e+00、A_std 8.15 分 → GAE → PPO 6 步 kl=+0.0053
                        ratio=0.998 clipfrac=0.019 → `[save]` + `[metrics]` JSON。
                        存档 9.84M、`encoder_version=3`，**eval_pairs 直读并跑完 2 seed 对局**
                        ⇒ R10 首个 PPO 轮的「命令/存档/闸门读取」三段全部前置验证完毕。
                        注：n<500 状态会 `[FAIL] 状态太少` rc=1（真实轮 60000 状态远超门槛）；
                        `--dry` 与真实跑共用同一路径。验证产物 runs/selftest 已删除、
                        models/ 无 exit_p*.pt 残留。**全程 --device cpu，未占 MPS，训练零影响**。
[2026-11-07 05:35:19]   rc=0  2039s  train_r09
[2026-11-07 05:35:19] R9 训练完成 init=exit_best.pt → exit_r09.pt  (['[done] best val 1.7681 → /Users/ligf9/zhuoji-mahjong-ai/models/exit_r09.pt'])
[2026-11-07 05:35:19] $ /Users/ligf9/zhuoji-mahjong-ai/.venv/bin/python eval_pairs.py --a /Users/ligf9/zhuoji-mahjong-ai/models/exit_r09.pt --b /Users/ligf9/zhuoji-mahjong-ai/models/dushan_master.pt --mode greedy --seeds 300 --seed0 710000 --workers 6 --out /Users/ligf9/zhuoji-mahjong-ai/runs/gates/r09_gateA.jsonl  → gateA_r09.log
[2026-11-07 05:36:12]   rc=0  53s  gateA_r09
[2026-11-07 05:36:12] R9 闸门 A: [summary] seats=[0, 1, 2, 3]  n=300  diff=+3.736±0.399  z=+9.37  p≈0.0000  符号 213:86  → 通过
[2026-11-07 05:36:12] $ /Users/ligf9/zhuoji-mahjong-ai/.venv/bin/python -c import torch,sys;sys.path.insert(0,'.');from zhuoji.net import build_from_ckpt;from zhuoji.forward import ensure_onnx;ck=torch.load('/Users/ligf9/zhuoji-mahjong-ai/models/exit_best.pt',map_location='cpu',weights_only=False);net,_=build_from_ckpt(ck);p=ensure_onnx(net,stem='exit_best');print('onnx:',p)  → export_exit_best.log
[2026-11-07 05:36:13]   rc=0  1s  export_exit_best
[2026-11-07 05:36:13] 晋升：exit_r09.pt → exit_best.pt
[2026-11-07 05:36:13] $ /Users/ligf9/zhuoji-mahjong-ai/.venv/bin/python eval_pairs.py --a /Users/ligf9/zhuoji-mahjong-ai/models/exit_r09.pt --b /Users/ligf9/zhuoji-mahjong-ai/models/dushan_master.pt --a-mode greedy --b-mode search32 --seeds 50 --seed0 720000 --workers 6 --out /Users/ligf9/zhuoji-mahjong-ai/runs/gates/r09_gateC.jsonl  → gateC_r09.log
[2026-11-07 05:57:27]   rc=0  1274s  gateC_r09
[2026-11-07 05:57:27] R9 闸门 C(趋势): [summary] seats=[0, 1, 2, 3]  n=50  diff=-0.893±1.016  z=-0.88  p≈0.3791  符号 23:25
[2026-11-07 05:57:27] R9 轮次完成  best=exit_best.pt
[2026-11-07 05:57:27] $ /Users/ligf9/zhuoji-mahjong-ai/.venv/bin/python -c import torch,sys;sys.path.insert(0,'.');from zhuoji.net import build_from_ckpt;from zhuoji.forward import ensure_onnx;ck=torch.load('/Users/ligf9/zhuoji-mahjong-ai/models/exit_best.pt',map_location='cpu',weights_only=False);net,_=build_from_ckpt(ck);p=ensure_onnx(net,stem='exit_best');print('onnx:',p)  → export_exit_best.log
[2026-11-07 05:57:28]   rc=0  1s  export_exit_best
[2026-11-07 05:57:28] $ /Users/ligf9/zhuoji-mahjong-ai/rust_engine/target/release/annotate --student /Users/ligf9/zhuoji-mahjong-ai/models/onnx/exit_best.onnx --teacher models/onnx/dushan_master.onnx --student-enc 3 --student-device cpu --targets 100000 --threads 4 --out /Users/ligf9/zhuoji-mahjong-ai/runs/exit_r10 --seed0 700000  → annotate_r10.log
[2026-11-07 05:57:41] 【切换】kill 旧驱动（pid=[89564, 89566]）于 R10 标注启动边界；生效新源码：PPO 分支（train_ppo.py）+ 闸门 D（vs 上轮最佳）
[2026-11-07 05:59:30] ============================================================
[2026-11-07 05:59:30] 7×24 ExIt 驱动启动：R10 起，截止 2026-11-13 02:30（R1 已完成；断点续跑 + 每小时自检自动化）
[2026-11-07 05:59:30] $ /Users/ligf9/zhuoji-mahjong-ai/.venv/bin/python -c import torch,sys;sys.path.insert(0,'.');from zhuoji.net import build_from_ckpt;from zhuoji.forward import ensure_onnx;ck=torch.load('/Users/ligf9/zhuoji-mahjong-ai/models/exit_best.pt',map_location='cpu',weights_only=False);net,_=build_from_ckpt(ck);p=ensure_onnx(net,stem='exit_best');print('onnx:',p)  → export_exit_best.log
[2026-11-07 05:59:30]   rc=0  1s  export_exit_best
[2026-11-07 05:59:30] $ /Users/ligf9/zhuoji-mahjong-ai/rust_engine/target/release/annotate --student /Users/ligf9/zhuoji-mahjong-ai/models/onnx/exit_best.onnx --teacher models/onnx/dushan_master.onnx --student-enc 3 --student-device cpu --targets 100000 --threads 4 --out /Users/ligf9/zhuoji-mahjong-ai/runs/exit_r10 --seed0 700000  → annotate_r10.log
[2026-11-07 05:59:31] 【切换】新驱动 pid=[21302] 心跳新鲜=True（R10 标注将从头重跑，无脏数据）

[2026-11-07 06:30:00] ======== 自检 #22（7×24 第 22 次）R9 硬决策点 + PPO 链式修正 ========
[2026-11-07 06:30:00] 【状态】健康、产出符合预期、切换成功、主线零干预。驱动为切换后的唯一新进程
                        PID 21302（05:59:30 起，PPID=keepalive 99107），lock 1~27s；
                        R10 标注 13.9%（3,484/25,000·worker），吞吐 11,870~11,900/h·worker，
                        ETA ≈08:06（3.5h 上限 09:29，余 83min）；异常 0、panic 0、磁盘 307Gi、
                        TAU=1.0、Web 8770 `/`+`/api/meta` 双 200 且 models=web_best(9.85M)、
                        看门狗 99107 活（4h47m）。r01~r09 **36/36** npz 全合规（9 轮×4 片）。
[2026-11-07 06:30:00] 【切换核查（三项证据一致）】① `switch_ppo.log` 05:57:39 命中 → 05:57:41 杀净
                        旧驱动 pid=[89564,89566]；② `exit_log` 05:57:41【切换】+ 05:59:30「驱动启动：
                        R10 起」；③ `keepalive.log` 05:59:29「训练驱动心跳停 → 拉起」→ 06:00:02 恢复。
                        净损失 ≈109s（05:57:28 起的 annotate 被弃，05:59:30 重起，标注只结束落盘 ⇒ 无脏文件，
                        `runs/exit_r10/` 旧目录确实不存在）。新驱动 stdout 去
                        `runs/night_logs/keepalive_night_run.log`（**不是** night_run.log，§6.25.5）。
[2026-11-07 06:30:00] 【R9 完整结果】标注 05:01:20 rc=0（7886s、11,419~11,446/h·worker、4×25k）
                        → 训练 05:35:19 rc=0（2039s、400k 封顶、init=exit_best、seed 20261115；
                        train 1.7742→1.7503、**best val 1.7681@ep01**、ep04 早停，**连续第 6 轮
                        best-val 落在 ep1**）→ 闸门 A **+3.736±0.399 z=+9.37 p≈0.0000 符号 213:86**
                        **连续第 8 轮通过 → 05:36:13 晋升** exit_r09.pt→exit_best.pt（sha `a24e2ee1`、
                        `r09.promoted=ok`、onnx 05:36 同步）→ 闸门 C **−0.893±1.016 p≈0.3791（23:25）**。
                        整轮 3h14m。冻结口径独立复算（gate_decision）与驱动 [summary] 逐位一致。
[2026-11-07 06:30:00] 【硬决策五项】
                        ① 增速三选一 **不触发**：TAU/Q-MSE/n_det/lr 一项不改（§6.22 已把三项量化否决）。
                        ② 天花板 **首次判「确认」**（工具以数据判定：A 环比 3/3 不显著 + C 三轮
                           |mean|<1 且 p>0.05；R9−R8 A 仅 **+0.041±0.570 p≈0.9425**，为连续第 5 个
                           零边际轮）⇒ 不再指望 DAgger 主线增益，**仍不改任何超参**（44 轮多重比较下
                           反复调参只会过拟合 300 seed 池）。
                        ③ P2a **维持观察态**（C 最近两段正增量 1/2，未连续收敛）。
                        ④ **PPO 分支已上线**：PPO_FROM=10，R10 末自动跑首个 PPO 轮（闸门 D 亦自本轮起可用，
                           `r09_gateD` 缺席是因为 R9 的闸门跑在**旧源码**上，符合预期）。
                        ⑤ 下一步训练由驱动自动接续，零干预、未启第二驱动、未打断、未并发评估。
[2026-11-07 06:30:00] 【本轮最重要发现与修复（P1，原设计缺陷）】`ppo_round()` 原写
                        `init = latest_pt()`（= DAgger exit_best）⇒ **每轮 PPO 都从主线权重重启**，
                        而单轮预算只有 960 局/≈6 万决策/2 epoch/lr 5e-5（冒烟实测 kl≈+0.005，
                        权重位移极小）⇒ 40 轮 RL 经验互不叠加，P10/P11/… 全是「一次性微扰」，
                        闸门 D 必然恒 ≈0 —— 这是一个**结构上注定为零**的实验，不是 PPO 无效。
                        已修为 `ppo_init(r)`：**默认续上一条 PPO 链**（`models/exit_p{r-1}.pt` 存在则用之），
                        否则回退 `latest_pt()`；`runs/ppo_rechain` 存在即强制回 DAgger 重启链（远程急停）。
                        40 轮链式后累计 ≈3.8 万局 ≈240 万决策，才是真实 RL 实验。
                        **r=10 时 exit_p09 不存在 ⇒ 行为与修正前完全等价**（已干跑验证 r=10/r=11 均回退
                        exit_best.pt），故该修正零风险；但它**只在下一次进程启动时生效**（长驻不重载源码）。
[2026-11-07 06:30:00] 【重启策略（零数据损失）】新建 `tools/switch_at_train.py`：等驱动 log 出
                        `→ train_r10.log`（= 标注已 rc=0、4×npz 已落盘、训练刚 spawn 尚未写 ckpt/.done）
                        后 2s 内 kill 驱动 + train_exit，交由 keepalive 拉起新源码 ⇒ **净损失 ≈2s**。
                        已启动 **PID 23309（PPID=1，已脱会话）**，超时点 10:19，回退=不切也无损失。
                        选「训练起点」而不是「标注起点」是因为前者此刻上一阶段数据已完整落盘，比
                        §6.25 的 annotate 边界更省（标注边界要重跑 2h 标注）；三者耗时对比：
                        标注期 kill 丢≤2h / 训练期丢≤34min / 闸门期丢≤25min / 训练起点丢≈2s。
[2026-11-07 06:30:00] 【工具修正】`tools/gate_decision.py`：② 天花板文案改为「确认」并区分
                        PPO 已上线/未上线；④ 判据从「是否启动」改为「已上线，判读走 ppo_report 的
                        P{r}−R{r}」，并实读 `night_run.py` 的 PPO_FROM 与 `runs/ppo_disabled`、
                        列出现有 `exit_p*.pt`。py_compile + 两种分支文案均验证通过（r=9 走「未到首发」）。
                        `tools/ppo_report.py` 保持只读不动，仍为 PPO 净增量的唯一判据实现。
[2026-11-07 06:30:00] 【判读纪律（新增，防误判 PPO）】单轮 P10 不晋升属**分辨率限制**
                        （300 seed SE≈0.5 分 > 单轮真实增量），绝不是失败；**绝不因此加 lr 或放宽闸门 D**。
                        PPO 成败只看 `P{r}−R{r}`（同池同口径逐 seed 配对）：连续 ≥3 轮 ≤0 ⇒ 降频或停用；
                        连续 ≥2 轮 >0 且累积 >2 分 ⇒ 加预算（hands 1.5~2×）。首次可用判定点 = **P12/P13**
                        （单轮 n=300 不够，需 3 轮累积 ≈n=900 才把 SE 压到 ≈0.29）。
[2026-11-07 06:30:00] 【时间线（按 R9 实测 3h14m + PPO/闸门D ≈10min 外推）】
                        R10 标注 ≈08:06 落盘 → **08:06 切换点（≈2s）** → 训练 ≈08:07~08:42 →
                        闸门 A ≈08:43 → P10 PPO ≈08:44~08:53 → p10 闸门 D/A ≈08:54~08:56 →
                        闸门 C ≈08:56~09:22 → R11 标注 09:22。自检间隔 ≈72min（06:17→07:29→08:41→09:53）
                        ⇒ **R10 完整 A+C 的首次捕获 = ≈09:53 自检**（= R10 硬决策点）；
                        08:41 自检只会在「R10 训练中/P10 未起」，只巡检不决策。
                        剩余 ≈140h ⇒ 每轮 ≈3h30m（含 PPO）⇒ **≈40 轮**，DEADLINE 仍为唯一硬约束。
[2026-11-07 06:30:00] 【线上服务】`models/web_best.pt` 仍为 R8 快照（sha `16292961`，01:12），
                        **本轮不刷新**：R9 相对 R8 仅 +0.041±0.570（不可分辨），为不可测收益重启在线服务
                        不划算；改为**终评两段式 best-of 定稿后一次性刷新**（闸门 D 通过 + 影子冒烟 +
                        蓝绿），且**绝不跟随 PPO 晋升**（§6.25 纪律 B）。
[2026-11-07 06:30:00] 【下一步】① 07:29 自检只核 R10 吞吐/异常与切换器是否还在武装；
                        ② **08:41 自检**核切换是否真的发生（switch_at_train.log + 【切换】行 + PID 变化），
                        未发生则手工在 R11 训练起点重试 `--round 11`；
                        ③ ≈09:53 硬决策：R10 完整 A+C + R10−R9 环比 + 首个 P10 metrics/gateD/gateA；
                        ④ P12/P13 前不做任何 PPO 预算调整（单轮 n 不够）。

[2026-11-07 06:22:30] ======== 【事故】自检 #22 期间驱动被误杀（P0，已恢复，数据无损）========
[2026-11-07 06:22:30] 【事实链】06:19:30 我启动 tools/switch_at_train.py（PID 23309）用于在 R10 训练
                        起点切换新源码，其触发判据是「reports/exit_log.md 文本里出现
                        `→ train_r10.log`」。06:20:48 我把本轮自检笔记写进 exit_log.md，
                        笔记里**引用了这句话本身** ⇒ 切换器立即误判触发（真实状态：R10 仍在标注 13.9%），
                        lock_age=17.6s 却仍照杀 ⇒ 06:20:50 驱动 PID 21302 被终止。
                        随后脚本在 note_md 的 f-string 里引用了未定义变量 `{r-1}`（应为 `{a.round-1:02d}`），
                        抛 NameError **在「等 keepalive 拉起」分支之前就退出** ⇒ 没有任何兜底重启。
                        更要紧的是脚本只 kill night_run/train_exit，**没杀 annotate**：
                        孤儿 annotate PID 21310（PPID=1）继续跑，若无干预，驱动重启后会再起一个
                        annotate 写同一目录 `runs/exit_r10` ⇒ 双写/覆盖风险。
[2026-11-07 06:22:30] 【处置（4 步，2 分钟内完成）】① 06:21:40 巡检发现 night_run 缺失、annotate PPID=1
                        ⇒ 立即 `pkill -f rust_engine/target/release/annotate` 清掉孤儿（避免双写）；
                        ② 核实 `runs/exit_r10/` **目录不存在** ⇒ 标注器只在结束时落盘，**零脏数据**；
                        ③ 确认切换器已因 NameError 退出（pgrep 空）⇒ 不会二次误杀；
                        ④ 交由 keepalive 在 lock 过期 120s 后拉起（06:23~06:24），不手工抢跑以免双驱动。
[2026-11-07 06:22:30] 【损失】R10 标注已跑 1287s（≈21.5min×4 worker）被弃，R10 从头重跑；
                        停机 ≈2.5~3min（lock 过期 120s + keepalive 30s 轮询）。
                        **数据零损失**（r01~r09 的 36/36 npz 未受影响），**模型零影响**（exit_best 仍
                        = exit_r09 = sha a24e2ee1），Web 8770 未受影响（双 200）。
[2026-11-07 06:22:30] 【正面结果】切换的**目的仍然达成**：重启后的驱动加载新源码，
                        `ppo_init()` 链式生效（R10 时 exit_p09 不存在 ⇒ 首个 P10 仍是 DAgger 起点，
                        与设计等价；P11 起才真正续链）。原计划 08:06 的切换点因此**不再需要**，
                        08:41 自检改为「确认新驱动 PID + R10 重跑进度」，**不必再跑任何切换器**。
[2026-11-07 06:22:30] 【修复（已完成，py_compile 通过）】`tools/switch_at_train.py`：
                        ① 触发判据从「日志文本匹配」改为**文件事实**——`runs/night_logs/train_r{R}.log`
                           存在且 mtime 60s 内（训练进程确实在写）。**任何「监控者自己也会写入日志」的
                           文本触发都不可用**，这是本次事故的根因类别；
                        ② `{r-1}` → `{a.round-1:02d}`（NameError 消除）；
                        ③ 保留 `pkill train_exit` 并**必须同时纳入 annotate**（本次孤儿进程的直接原因）。
                        **该工具暂不再武装**：切换已在 06:23 由 keepalive 完成。
[2026-11-07 06:22:30] 【教训（写入长期纪律）】A) 触发器只能依赖**不可由自检文本产生**的信号；
                        B) 一切 kill 脚本必须把「拉起」放进 finally/必达路径，不能放在可能抛异常的行之后；
                        C) kill 阶段必须枚举**该阶段全部子进程**（annotate/train_exit/train_ppo），
                        否则留孤儿造成双跑写同一输出目录。三条已同步写入 §6.26.10。
[2026-11-07 06:22:33] ============================================================
[2026-11-07 06:22:33] 7×24 ExIt 驱动启动：R10 起，截止 2026-11-13 02:30（R1 已完成；断点续跑 + 每小时自检自动化）
[2026-11-07 06:22:33] $ /Users/ligf9/zhuoji-mahjong-ai/.venv/bin/python -c import torch,sys;sys.path.insert(0,'.');from zhuoji.net import build_from_ckpt;from zhuoji.forward import ensure_onnx;ck=torch.load('/Users/ligf9/zhuoji-mahjong-ai/models/exit_best.pt',map_location='cpu',weights_only=False);net,_=build_from_ckpt(ck);p=ensure_onnx(net,stem='exit_best');print('onnx:',p)  → export_exit_best.log
[2026-11-07 06:22:34]   rc=0  1s  export_exit_best
[2026-11-07 06:22:34] $ /Users/ligf9/zhuoji-mahjong-ai/rust_engine/target/release/annotate --student /Users/ligf9/zhuoji-mahjong-ai/models/onnx/exit_best.onnx --teacher models/onnx/dushan_master.onnx --student-enc 3 --student-device cpu --targets 100000 --threads 4 --out /Users/ligf9/zhuoji-mahjong-ai/runs/exit_r10 --seed0 700000  → annotate_r10.log

[2026-11-07 07:26:00] ======== 【自检 #23】R10 标注期巡检 · 健康、零干预 ========
[2026-11-07 07:26:00] 【状态】驱动唯一 PID 23674（PPID=99107 keepalive，1h02m），lock 22s；
                        R10 标注 50.2%（12,435~12,554 / 25,000·worker），seed0=700000，enc3/cpu；
                        吞吐 11,795~11,824/h·worker（起始 12.5k 缓降，仍在 10~12.5k 健康区间）；
                        ETA ≈08:29（3.5h 上限 09:52，余量 83min）；06:22:33 起异常 0。
                        磁盘 307Gi（models 615M）、TAU=1.0、Web 8770 双 200（models=web_best R8 快照）、
                        看门狗 99107 活（05:59 / 06:22 两次拉起均成功）。36/36 npz 合规。
[2026-11-07 07:26:00] 【R9 复核】A +3.736±0.399 p≈0.0000（213:86）第 8 轮通过已晋升；
                        C −0.893±1.016 p≈0.3791；A R9−R8 +0.041±0.570 p≈0.9425（第 5 个零边际轮）；
                        exit_best≡exit_r09（a24e2ee1）+ r09.promoted + onnx 三件套一致。
[2026-11-07 07:26:00] 【新发现 · 闸门 D 从未执行】runs/gates/ 下无任何 *_prevbest.pt 与 *_gateD.jsonl。
                        根因：闸门 D 于 04:15 写入源码，但在跑的 PID 89566 是 23:41 启动的旧进程
                        ⇒ R06~R09 四轮只跑 A+C。06:22 重启的 23674 才首次载入。
                        ⇒ **R10 是闸门 D 与 PPO 的双首秀**；r06~r09 真环比永久缺失（终评事后配对补）。
[2026-11-07 07:26:00] 【PPO 判读预登记】单轮 960 局 ≈6万决策 / kl≈0.005 ⇒ 必然低于 300 seed 分辨率
                        （SE≈0.5）。「P10 未晋升」不构成证据，禁止据此加 lr / 放宽闸门。唯一判据
                        = 同池配对 P{r}−R{r}（tools/ppo_report.py）。P12 用 P10~P12 累积 n≈900
                        判：Σ>0 且 p<.05 → PPO_HANDS 960→3840（单变量）；~0 → 维持至 P16/P20；
                        Σ<0 且 p<.10 → 放 runs/ppo_disabled 冻结 PPO。
[2026-11-07 07:26:00] 【结构事实】PPO 晋升在结构上不可达（需 +1.0 分 > SE 0.5）⇒ exit_best 恒为
                        DAgger rNN、web_best 永不跟随 PPO；PPO 价值只在终评 best-of 兑现。
                        「PPO 一直没晋升」不得解读为无效。
[2026-11-07 07:26:00] 【结构风险 · 本轮不动】keepalive(99107) 父链仍是 WorkBuddy 会话（zsh 99106）；
                        Web 服务 PID 97638 PPID=1 已存活 6h11m 证明脱管可行，但重启 keepalive 会
                        连带杀掉子进程驱动 23674（R10 标注 50% 丢失）⇒ 判定收益<风险，不处置。
                        处置窗口只能在训练阶段且须人工会话确认。
[2026-11-07 07:26:00] 【决策】零干预（R10→训练→A/C/D→P10→R11 自动接续）；超参一项不改；
                        天花板维持「确认」；P2a 观察态；web_best 保持 R8 快照待终评一次性刷新。
                        CPU 有 ≈7 核空闲（标注仅 59%，瓶颈在教师 CoreML），但实证并发会压垮吞吐 ⇒ 弃用。
[2026-11-07 07:26:00] 【时间线】R10 落盘 ≈08:29 → 训练 ≈09:11 → A ≈09:13 → C ≈09:36 → D ≈09:37 →
                        P10 ≈09:48 → p10 双闸门 ≈09:51 → R11 标注。08:41 只巡检；
                        **硬决策点 = ≈09:53**。剩余 139.1h / 3h33m ≈ 39 轮（R10~R48）。
[2026-11-07 07:26:00] ============================================================
[2026-11-07 08:29:45]   rc=0  7632s  annotate_r10
[2026-11-07 08:29:45] R10 标注完成 student=exit_best.onnx seed0=700000 → 4×25k
[2026-11-07 08:29:45] $ /Users/ligf9/zhuoji-mahjong-ai/.venv/bin/python train_exit.py --data runs/exit_r01/exit_w*.npz runs/exit_r02/exit_w*.npz runs/exit_r03/exit_w*.npz runs/exit_r04/exit_w*.npz runs/exit_r05/exit_w*.npz runs/exit_r06/exit_w*.npz runs/exit_r07/exit_w*.npz runs/exit_r08/exit_w*.npz runs/exit_r09/exit_w*.npz runs/exit_r10/exit_w*.npz --out /Users/ligf9/zhuoji-mahjong-ai/models/exit_r10.pt --epochs 30 --window 400000 --batch 256 --seed 20261116 --init /Users/ligf9/zhuoji-mahjong-ai/models/exit_best.pt  → train_r10.log
[2026-11-07 08:36:00] ============================================================
[2026-11-07 08:36:00] 【自检 #24】R10 落盘核对 + best-of 第一段工具化（只核对，不决策）
[2026-11-07 08:36:00] 健康：驱动唯一 PID 23674(PPID=99107 keepalive, 2h13m) lock 1.9s；无 switch_* 残留；
                        ps 仅 1 driver + 1 train_exit(33596)。06:22:33 起异常 0；磁盘 307Gi；TAU=1.0；
                        Web 8770 / =200、/api/meta models=web_best(R8 快照)；看门狗 99107 活。
[2026-11-07 08:29:45]   rc=0  7632s  annotate_r10   （11,786~11,797/h·worker；4×25k；check_npz 4/4，
                        累计 40/40；NaN↔mask overlap=gap=0）
[2026-11-07 08:29:45] R10 训练自动接续（400k 封顶 / --init exit_best.pt / seed 20261116 / PID 33596）
[2026-11-07 08:36:00] 【判活口径补充】train_r10.log 起始只有 1 条 UserWarning(365B) 不是卡住：
                        train_exit.py 仅 epoch/early-stop/done 带 flush=True(:181/:198/:200)，
                        [data]/[init]/[net](:61/:126/:128) 不带 flush，留在 4KB 缓冲直至首个 epoch。
                        判活只能看 ps -o time 累计 CPU（8min→39s 单调增）。与 §6.22「night_run.log 冻结」
                        同类陷阱：一律不得据日志 mtime 判死。
[2026-11-07 08:36:00] 【新工具】tools/bestof_rank.py（只读/零 CPU/不启对局，复用 final_eval_report
                        的 per_seed+stats 唯一实现）：自动发现 r*/p*_gateA.jsonl（仅收 ≥300 唯一 seed
                        的完整文件）→ 排名 + top1 对其余逐 seed 配对差分 + 选择偏差 SE·√(2lnN) +
                        top-k 第二段复赛命令 + --md/--json。此前 §8「第一段」只能人工看全表，
                        无排名/无 top-k/无显著性，现一条命令可复现。
[2026-11-07 08:36:00] 【R01~R09 排名】R09 +3.736±.399 | R08 +3.694±.415 | R06 +3.658±.407 |
                        R05 +3.398±.413 | R07 +3.353±.428 | R04 +3.312±.418 | R03 +2.843±.422 |
                        R02 +2.374±.415 | R01 +0.179±.406(ns)
[2026-11-07 08:36:00] 【核心结论】R09 对其余：−R08 +0.041±0.570 p=.94 / −R06 +0.078±0.536 p=.88 /
                        −R05 +0.338±0.556 p=.54 / −R07 +0.382±0.537 p=.48 / −R04 +0.423±0.533 p=.43
                        ⇒ R04~R09 六轮**全部并列(<1SE)**；仅 −R02 +1.361±0.540 p=.012 显著。
                        平台期由「环比不显著」升级为「交付维度完全不可分辨」。
[2026-11-07 08:36:00] 【定量发现】SE(R09−R08)=0.570 ≈ √(.399²+.415²)=0.576 ⇒ 逐 seed 配对
                        几乎不降方差(ρ≈0，同一牌墙一着分叉后两局独立)。⇒ 独立池 500 seed 配对
                        SE≈0.44 仍大于实际差距，第二段**大概率同样判并列**（预期而非失败）。
                        ⇒ §8.1 的「P0 交付缺陷」降级为 P2 方法论完备性：平台期 exit_best 逐轮
                        覆盖不带来交付损失（任取 R04~R09 统计等价）。
[2026-11-07 08:36:00] 【预登记·交付规则】top1 显著(p<.05 且 gap>1SE)→交付 top1；否则交付
                        「并列集合中轮次最晚者」。理由：数据最多 / = exit_best 零切换 / 取并列集合
                        mean 最高者正是要消除的选择偏差本身（N=9 纯噪声 max 期望上偏 0.836 分≈2.10 SE）。
                        规则在看到独立池结果之前固化，禁止事后调整。
[2026-11-07 08:36:00] 【成本校准】闸门 A 实测 53~60s/1200 局 ⇒ 第一段 0 成本；第二段 greedy
                        500 seed ≈90~100s/候选 ⇒ top-5 ≈8min；仅交付模型的 N32 验收 ≈3.7h（选择项
                        不比、验收项才比）⇒ 终评总计 ≈4h，原「4~11h」上限撤销、预留 4~5h 足够。
[2026-11-07 08:36:00] 【决策】零干预（R10 训练→A/C/D→P10→R11 自动接续）；超参一项不改；天花板
                        维持「确认」；P2a 观察态；web_best 保持 R8 快照待终评一次性刷新。
                        硬决策点 = ≈09:53（首次捕获 R10 完整 A+C+D 与 P10 首轮 + P10−R10 配对）。
[2026-11-07 08:36:00] ============================================================
[2026-11-07 09:12:01]   rc=0  2536s  train_r10
[2026-11-07 09:12:01] R10 训练完成 init=exit_best.pt → exit_r10.pt  (['[done] best val 1.7695 → /Users/ligf9/zhuoji-mahjong-ai/models/exit_r10.pt'])
[2026-11-07 09:12:01] $ /Users/ligf9/zhuoji-mahjong-ai/.venv/bin/python eval_pairs.py --a /Users/ligf9/zhuoji-mahjong-ai/models/exit_r10.pt --b /Users/ligf9/zhuoji-mahjong-ai/models/dushan_master.pt --mode greedy --seeds 300 --seed0 710000 --workers 6 --out /Users/ligf9/zhuoji-mahjong-ai/runs/gates/r10_gateA.jsonl  → gateA_r10.log
[2026-11-07 09:12:55]   rc=0  53s  gateA_r10
[2026-11-07 09:12:55] R10 闸门 A: [summary] seats=[0, 1, 2, 3]  n=300  diff=+3.864±0.411  z=+9.40  p≈0.0000  符号 214:83  → 通过
[2026-11-07 09:12:55] $ /Users/ligf9/zhuoji-mahjong-ai/.venv/bin/python -c import torch,sys;sys.path.insert(0,'.');from zhuoji.net import build_from_ckpt;from zhuoji.forward import ensure_onnx;ck=torch.load('/Users/ligf9/zhuoji-mahjong-ai/models/exit_best.pt',map_location='cpu',weights_only=False);net,_=build_from_ckpt(ck);p=ensure_onnx(net,stem='exit_best');print('onnx:',p)  → export_exit_best.log
[2026-11-07 09:12:56]   rc=0  1s  export_exit_best
[2026-11-07 09:12:56] 晋升：exit_r10.pt → exit_best.pt
[2026-11-07 09:12:56] $ /Users/ligf9/zhuoji-mahjong-ai/.venv/bin/python eval_pairs.py --a /Users/ligf9/zhuoji-mahjong-ai/models/exit_r10.pt --b /Users/ligf9/zhuoji-mahjong-ai/models/dushan_master.pt --a-mode greedy --b-mode search32 --seeds 50 --seed0 720000 --workers 6 --out /Users/ligf9/zhuoji-mahjong-ai/runs/gates/r10_gateC.jsonl  → gateC_r10.log
[2026-11-07 09:33:54]   rc=0  1258s  gateC_r10
[2026-11-07 09:33:54] R10 闸门 C(趋势): [summary] seats=[0, 1, 2, 3]  n=50  diff=+2.227±1.009  z=+2.21  p≈0.0273  符号 30:20
[2026-11-07 09:33:54] $ /Users/ligf9/zhuoji-mahjong-ai/.venv/bin/python eval_pairs.py --a /Users/ligf9/zhuoji-mahjong-ai/models/exit_r10.pt --b /Users/ligf9/zhuoji-mahjong-ai/runs/gates/r10_prevbest.pt --mode greedy --seeds 300 --seed0 710000 --workers 6 --out /Users/ligf9/zhuoji-mahjong-ai/runs/gates/r10_gateD.jsonl  → gateD_r10.log
[2026-11-07 09:36:00]   rc=0  126s  gateD_r10
[2026-11-07 09:36:00] R10 闸门 D(vs 上轮最佳): [summary] seats=[0, 1, 2, 3]  n=300  diff=+0.481±0.420  z=+1.14  p≈0.2523  符号 150:142
[2026-11-07 09:36:00] P10 PPO 链起点 = exit_best.pt
[2026-11-07 09:36:00] $ /Users/ligf9/zhuoji-mahjong-ai/.venv/bin/python train_ppo.py --init /Users/ligf9/zhuoji-mahjong-ai/models/exit_best.pt --out /Users/ligf9/zhuoji-mahjong-ai/models/exit_p10.pt --hands 960 --workers 4 --threads 2 --max-states 60000 --epochs 2 --lr 5e-5 --seed0 1000000  → ppo_p10.log
[2026-11-07 09:41:34]   rc=0  334s  ppo_p10
[2026-11-07 09:41:34] P10 PPO 完成 init=exit_best.pt → exit_p10.pt {"hands": 960, "states": 60000, "A_std": 7.602680206298828, "ret_std": 11.13347055803655, "kl": [0.011250697744841826, 0.013262783052360123], "secs": 333.0}
[2026-11-07 09:41:34] $ /Users/ligf9/zhuoji-mahjong-ai/.venv/bin/python eval_pairs.py --a /Users/ligf9/zhuoji-mahjong-ai/models/exit_p10.pt --b /Users/ligf9/zhuoji-mahjong-ai/models/exit_best.pt --mode greedy --seeds 300 --seed0 710000 --workers 6 --out /Users/ligf9/zhuoji-mahjong-ai/runs/gates/p10_gateD.jsonl  → gateD_p10.log
[2026-11-07 09:43:34]   rc=0  121s  gateD_p10
[2026-11-07 09:43:34] P10 闸门 D(vs 现最佳): [summary] seats=[0, 1, 2, 3]  n=300  diff=-0.097±0.367  z=-0.26  p≈0.7924  符号 144:146  → 不通过
[2026-11-07 09:43:34] $ /Users/ligf9/zhuoji-mahjong-ai/.venv/bin/python eval_pairs.py --a /Users/ligf9/zhuoji-mahjong-ai/models/exit_p10.pt --b /Users/ligf9/zhuoji-mahjong-ai/models/dushan_master.pt --mode greedy --seeds 300 --seed0 710000 --workers 6 --out /Users/ligf9/zhuoji-mahjong-ai/runs/gates/p10_gateA.jsonl  → gateA_p10.log
[2026-11-07 09:44:31]   rc=0  57s  gateA_p10
[2026-11-07 09:44:31] P10 闸门 A(vs master): [summary] seats=[0, 1, 2, 3]  n=300  diff=+3.796±0.414  z=+9.16  p≈0.0000  符号 207:91
[2026-11-07 09:44:31] P10 不晋升（未显著优于现最佳），ckpt 留档供终评 best-of
[2026-11-07 09:44:31] R10 轮次完成  best=exit_best.pt
[2026-11-07 09:44:31] $ /Users/ligf9/zhuoji-mahjong-ai/.venv/bin/python -c import torch,sys;sys.path.insert(0,'.');from zhuoji.net import build_from_ckpt;from zhuoji.forward import ensure_onnx;ck=torch.load('/Users/ligf9/zhuoji-mahjong-ai/models/exit_best.pt',map_location='cpu',weights_only=False);net,_=build_from_ckpt(ck);p=ensure_onnx(net,stem='exit_best');print('onnx:',p)  → export_exit_best.log
[2026-11-07 09:44:32]   rc=0  1s  export_exit_best
[2026-11-07 09:44:32] $ /Users/ligf9/zhuoji-mahjong-ai/rust_engine/target/release/annotate --student /Users/ligf9/zhuoji-mahjong-ai/models/onnx/exit_best.onnx --teacher models/onnx/dushan_master.onnx --student-enc 3 --student-device cpu --targets 100000 --threads 4 --out /Users/ligf9/zhuoji-mahjong-ai/runs/exit_r11 --seed0 710000  → annotate_r11.log

[2026-11-07 09:58:00] ============================================================
[2026-11-07 09:58:00] 【自检 #25】健康、产出符合预期、主线零干预。驱动唯一 PID 23674（PPID=99107
                        keepalive，3h35m），lock 心跳 ~1s，唯一 annotate(39622)，无 switch_* 残留；
                        阶段 = R11 标注（09:44:32 起，11.6k/h·worker，ETA ≈11:55）；42 阶段 rc=0、
                        panic 0；磁盘 307Gi；TAU=1.0；40/40 npz；Web 双 200（models=web_best）；
                        看门狗 99107 活。
[2026-11-07 09:58:00] 【R10 结算】标注 08:29:45 rc=0（7632s、11.79k/h·worker、4×25k）→ 训练 09:12:01
                        rc=0（2536s、400k、init=r09）train 1.7739→1.7463、best val 1.7695 **@ep02**
                        （6 轮后首次非 ep01）、ep05 早停 → 闸门 A `+3.864±0.411 p≈0.0000（214:83）`
                        第 9 轮通过 → 晋升 exit_r10.pt（sha b013ed75…、onnx 09:12 同步）→
                        闸门 C `+2.227±1.009 p≈0.0273（29:20）` **首次名义显著为正** →
                        闸门 D 首秀 `+0.481±0.420 p≈0.2523` 不显著 → P10 → R11。整轮 3h15m。
[2026-11-07 09:58:00] 【闸门 C 跃升：先证真、再证不可采信】(a) 同池配对 R10−R09 C `+3.120±1.178
                        p=0.0081`，bootstrap CI[+0.83,+5.33] P(>0)=.996、trim10/20% +3.26/+3.62、
                        LOO[+2.78,+3.59]、胜负 33:16 ⇒ **不是离群点**。(b) 但 **族系不显著**：
                        C 已做 10 次，仿真 P(max|z|≥2.213,10 tests)=**0.239**，Bonferroni 需 p<.005
                        ⇒ 24% 纯运气即可出现；且与闸门 D（vs r09 仅 +0.481 p=.25）内部矛盾、
                        与 R2~R10 合并 `+0.305±0.441 p=.4887` 矛盾；固定池 720000+50 已被用 10 次。
[2026-11-07 09:58:00] 【预登记 · 交付判据】R10 的 C =「名义显著、族系不显著」。① 不宣布 §4
                        「超额」（新贪心 ≥ 现 N32）达成；② 不改任何超参；③ 唯一确认途径 = 终评
                        独立池 3_000_000+500（500 seed）跑「r10 贪心 vs master+N32」，同时充当
                        §4「N32 验收」⇒ 一次计算双重用途，判据固化进终评手册 §9。
[2026-11-07 09:58:00] 【P10 PPO 首轮】960 局/60k states/333s；kl 0.0113→0.0133、ratio 1.0001、
                        clipfrac 0.136、A_std 7.60 —— 全部健康；闸门 D `−0.097±0.367 p=.79` 不晋升
                        （预期内）；**P10−R10 `−0.069±0.419 p=.87`（1/3 轮）**；P12/P13 前不调预算、
                        不放宽闸门、不加 lr。
[2026-11-07 09:58:00] 【决策】①增速三选一不触发（A p≈0.0000，TAU/Q-MSE/n_det/lr 一项不改）；
                        ②天花板按规则「不确认」但不撤销 A 侧结论（两分支处置一致）；③P2a 维持观察态；
                        ④PPO 维持 960 局；⑤R11 已自动接续，零干预。
[2026-11-07 09:58:00] 【时间线】剩余 136.5h ⇒ ≈41 轮（R11~R51），MAX_ROUND=60 不触发、DEADLINE 唯一
                        硬约束；R11 落盘 ≈11:55 → 训练 ≈12:37 → A ≈12:38 → C ≈12:59 → D ≈13:01 →
                        P11 ≈13:07 → R12。**下一硬决策点 = 首次捕获 R11 完整 A+C+D+P11（≈13:0x）**。
[2026-11-07 09:58:00] 【防误判】闸门 C 单轮值此后不得单独作证据；exit_r10.pt mtime 08:46 是 best-val
                        落盘时刻（非训练结束），判 ckpt 看 train_r*.done + [done]。
[2026-11-07 09:58:00] ============================================================

[2026-11-07 11:03:00] ============================================================
[2026-11-07 11:03:00] 【自检 #26】健康、产出符合预期、主线零干预。驱动唯一 PID 23674（PPID=99107
                        keepalive，4h40m），lock 23s；阶段 = R11 标注 59.2%（14,808/25k·worker，
                        seed0=710000，enc3/cpu ✓），吞吐 11,318/h·worker，ETA 11:59（上限 13:11，
                        余 72min）；42 阶段 rc=0、panic/NaN 超时 0；40/40 npz 合规；TAU=1.0；
                        Web 8770 双 200（models=web_best R8 快照）；看门狗 99107 活；磁盘 307Gi。
[2026-11-07 11:03:00] 【闸门复核】gate_decision / ppo_report / bestof_rank 三工具与 #25 逐位一致：
                        A 十轮 +0.179→+3.864 全 p≈0.0000；C 十轮 −2.480…+2.227；
                        A 环比 +2.196/+0.469/+0.469/+0.086/+0.260/−0.304/+0.341/+0.041/+0.129；
                        P10 A +3.796±0.414、P10 D −0.097±0.367、P10−R10 −0.069±0.419 p≈0.87。
[2026-11-07 11:03:00] 【新 · 收官排期精算】轮长 202min（标注 127+训练 42+闸门 24+PPO 9）。
                        剩余 135.4h ÷ 202 = 40.2 轮 ⇒ R11~R50；MAX_ROUND 不触发、DEADLINE 唯一硬约束。
                        时间盒 can_start(130/35/15) ⇒ 能跑完「标注+训练」的最后一轮须在
                        11-12 23:41 前启动；能跑完标注的可在 11-13 00:20 前启动 ⇒ **39min 判定带
                        （占轮长 19%）会白跑 2h07m 标注**（数据落盘但永不训练，驱动随即 break）。
                        预登记：11-13 00:25 后若当前 annotate 所属轮次无 gateC ⇒ 3min 内干净终止
                        （kill 驱动 + pkill annotate + 确认无残留分片）并立即转终评，
                        终评起点 02:30→00:30、报告 07:30→05:30。红线：只在最后 2h 且判定带成立时执行。
[2026-11-07 11:03:00] 【新 · 末轮无闸门】训练跨 DEADLINE 结束时 can_start(15) 会跳掉 A/C/D/PPO
                        ⇒ 该轮 ckpt 存在但无 rNN_gateA.jsonl，bestof_rank 静默看不到它（可能丢最强模型）。
                        处置已写入终评手册 §10.3：终评第一步扫「有 ckpt 无 gateA」并补跑闸门 A
                        （1200 局 57s/轮，全补 ≤5min），再跑排名。
[2026-11-07 11:03:00] 【新 · PPO 链核对】ppo_init(11)→exit_p10.pt ✓；ppo_init(12/13)→exit_best 是
                        p11 尚未生成所致，非 bug。真实脆弱点：任一 PPO 轮失败 ⇒ 下一轮静默回退
                        DAgger 现最佳（只写「链起点」不报错）。监控项：每轮查 exit_p*.pt 是否连续，
                        缺号后样本不得并入 P12 累积判读。
[2026-11-07 11:03:00] 【交付诚实陈述】R10/P10/R09/R08/R06 两两配对全部 <1SE（R10−R08 +0.170±0.541
                        p=.75）⇒ 交付强度自 R08 起已冻结；剩余 40 轮 DAgger 不改变交付物强度。
                        终评不为「多跑几轮」预留时间。唯一变量 = PPO 链。
[2026-11-07 11:03:00] 【决策】五项全无变更：①增速三选一不触发（A p≈0.0000，TAU/Q-MSE/n_det/lr
                        一项不改）②天花板不确认亦不撤销 A 侧结论 ③P2a 维持观察态 ④PPO 维持 960 局，
                        P12 前不动预算、不放宽闸门 D ⑤R11 已由驱动自动接续（11:59 落盘 → 12:41 训练
                        → 13:1x 闸门 → P11 → R12），零干预、不启第二驱动、不打断、不并发评估。
[2026-11-07 11:03:00] 【时间线】R11 落盘 ≈11:59 → 训练 ≈12:41 → A ≈12:42 → C ≈13:03 → D ≈13:05
                        → P11 ≈13:11 → R12。**下一硬决策点 ≈13:2x**。
[2026-11-07 11:03:00] ============================================================
[2026-11-07 11:56:44]   rc=0  7932s  annotate_r11
[2026-11-07 11:56:45] R11 标注完成 student=exit_best.onnx seed0=710000 → 4×25k
[2026-11-07 11:56:45] $ /Users/ligf9/zhuoji-mahjong-ai/.venv/bin/python train_exit.py --data runs/exit_r01/exit_w*.npz runs/exit_r02/exit_w*.npz runs/exit_r03/exit_w*.npz runs/exit_r04/exit_w*.npz runs/exit_r05/exit_w*.npz runs/exit_r06/exit_w*.npz runs/exit_r07/exit_w*.npz runs/exit_r08/exit_w*.npz runs/exit_r09/exit_w*.npz runs/exit_r10/exit_w*.npz runs/exit_r11/exit_w*.npz --out /Users/ligf9/zhuoji-mahjong-ai/models/exit_r11.pt --epochs 30 --window 400000 --batch 256 --seed 20261117 --init /Users/ligf9/zhuoji-mahjong-ai/models/exit_best.pt  → train_r11.log

[2026-11-07 12:22:00] ============================================================
[2026-11-07 12:22:00] 【自检 #27】健康、产出符合预期、主线零干预。驱动唯一 PID 23674（PPID=99107
                        keepalive，5h58m），lock 5.9s，子进程只有 train_exit PID 52339；
                        阶段 = R11 训练（11:56:45 起、400k 封顶、init=exit_best.pt、
                        sha 校验 init≡exit_r10.pt b013ed75、seed 20261117）。
                        ep01 train 1.7672 / val 1.7613 / 510s，ep02 进行中，预计 12:38 结束。
[2026-11-07 12:22:00] 【R11 标注】11:56:44 rc=0（7932s≈2h12m、11.3k/h·worker），4×25,000=100,000 行，
                        check_npz 4/4 全合规（tiles(63,27)/glob(130)/q,mask(28)，
                        NaN↔mask overlap=0 gap=0），r01~r11 累计 **44/44**。
                        无新闸门（最新仍 R10 09:12）⇒ 本轮属训练窗口巡检。
[2026-11-07 12:22:00] 【复核】gate_decision / ppo_report / bestof_rank 三工具与 #26 逐位一致：
                        A 十轮 +0.179→+3.864 全 p≈0.0000；C 十轮 −2.480…+2.227；
                        R10−R09 A +0.129±0.535 p≈0.8098；P10 A +3.796±0.414、
                        P10 D −0.097±0.367 p≈0.7924、P10−R10 −0.069±0.419 p≈0.8694；
                        bestof 第一段 R10/P10/R09/R08/R06 两两 <1SE。
                        exit_best.pt≡exit_r10.pt（b013ed75）、r10.promoted=ok、r11 未晋升（正常）。
                        运行态：Web 8770 `/`+`/api/meta` 双 200（models=web_best R8 快照）；
                        看门狗 99107 活（最近两次拉起 05:59/06:22 均成功）；磁盘 307Gi；
                        TAU=1.0；06:22 起异常行 0、今日 rc=0 阶段行 29 条。
[2026-11-07 12:22:00] 【本轮最有价值 · 新增 tools/plateau_test.py】把「最近几轮闸门 A 环比不显著」
                        升级为**有零假设理论值可比对的漂移检验**，并顺带实测出跨轮 pool 的真实 SE。
                        模型：若各轮真值相同、只差一份 iid 扰动 δ，则
                        D_k = M[k+1]−M[k] = δ_{k+1}−δ_k ⇒ Var(D)=2σ²，
                        **lag-1 相关 = −0.50**；若存在系统性漂移 μ，lag-1 被推向 0 甚至为正。
                        实测（r01~r10，300 seed 共同池、逐 seed 配对）8 对相邻差分 lag-1：
                        −0.484 / −0.478 / −0.480 / −0.469 / −0.541 / −0.547 / −0.543 / −0.522，
                        均值 **−0.508**；非相邻 28 对均值 **+0.004**（≈0，各段独立）。
                        ⇒ 轮间波动与「固定中心 + iid 扰动」**不可区分** ⇒ R04~R10 无系统性漂移。
[2026-11-07 12:22:00] 【该检验的意义 · 第 6 条独立证据】前 5 条证据（5 个零边际轮 / 6 轮 best-val@ep01 /
                        R9、R10 ep01 train loss 四位小数相同 / 教师 top1−top2 间隔中位 0.144 分 /
                        学生 argmax 与标签 52.5% 但闸门 C≈0）都是「**不显著**」类的阴性结论；
                        本条第一次给出**可对的阳性理论值**（−0.50），实测 −0.508 ⇒ 平台期由
                        「未能拒绝无差异」升级为「与无漂移模型定量吻合」。
                        边界声明：该检验灵敏度受噪声限制（每 seed 单轮 sd≈7.16 分），
                        只能排除「每轮 ≥0.5 分」量级的漂移，不能排除 0.1 分级微增益；
                        但 0.1 分本就低于终评交付分辨率，不影响任何决策。
[2026-11-07 12:22:00] 【pool SE 实测校准 · 影响 P12 判读功效】相邻段共用中间轮扰动 ⇒ 噪声部分相消，
                        cluster-robust SE（按 seed 聚簇）**小于** naive iid SE：
                          3 段 pool：mean +0.170、naive SE 0.319（n=900）、cluster SE **0.186**、膨胀 ×0.59
                          2 段 pool：naive SE 0.391（n=600）、cluster SE **0.271**、膨胀 ×0.69
                        ⇒ P12 预登记的「累积 n≈900 ⇒ SE≈0.29」是**保守**估计，实际 0.19，
                        可分辨下限（2σ）±0.37 分。**判读窗口维持预登记的 3 段不变（不做事后放宽）**，
                        但已知其功效高于预期：真增益 ≥0.4 分/轮 即可在 P12 被检出，P13 降为确认。
[2026-11-07 12:22:00] 【决策】五项全无变更：①增速三选一不触发（A 第 9 轮后仍 p≈0.0000，
                        weak=False/False ⇒ TAU / Q-MSE / n_det / lr 一项不改）
                        ②天花板不确认亦不撤销 A 侧结论（两分支处置一致，不改超参）
                        ③P2a 维持观察态 ④PPO 维持 960 局 / lr 5e-5 / 闸门 D 不放宽，
                        P12 前不动预算 ⑤**下一步训练无需启动**——R11 训练已由驱动 11:56:45 自动接续，
                        随后 12:38→闸门 A→C→D→P11→R12 全自动；零干预、不启第二驱动、
                        不打断、不并发评估。
[2026-11-07 12:22:00] 【新增预登记 · 尾部预算再分配备选（§6.31.4）】不现在执行，只在 P12 按数据决定：
                        若 P10~P12 pool 判 PPO 无净收益（|Σ|<0.37 或 Σ≤0），则剩余 ≈115h 的
                        40 轮同配方 DAgger 期望增益为 0，而 PPO-only 轮 ≈8~9min/轮（rollout 333s +
                        双闸门 120s + 导出）⇒ 可容纳 ≈700 轮 ≈67 万局自对弈（现模式仅 3.8 万局）。
                        触发条件、护栏（每 10 轮从 exit_best 重锚、闸门 A/D 不放宽、
                        web_best 只走终评 best-of）与回退路径见计划 §6.31.4。
[2026-11-07 12:22:00] 【时间线】R11 训练 ≈12:38 结束 → A ≈12:39 → C ≈13:01 → D ≈13:03
                        → P11 ≈13:09 → R12 ≈13:12。**下一硬决策点 ≈13:2x**
                        （首次捕获 R11 完整 A+C+D + P11）。
[2026-11-07 12:22:00] ============================================================
[2026-11-07 12:39:19]   rc=0  2555s  train_r11
[2026-11-07 12:39:19] R11 训练完成 init=exit_best.pt → exit_r11.pt  (['[done] best val 1.7596 → /Users/ligf9/zhuoji-mahjong-ai/models/exit_r11.pt'])
[2026-11-07 12:39:19] $ /Users/ligf9/zhuoji-mahjong-ai/.venv/bin/python eval_pairs.py --a /Users/ligf9/zhuoji-mahjong-ai/models/exit_r11.pt --b /Users/ligf9/zhuoji-mahjong-ai/models/dushan_master.pt --mode greedy --seeds 300 --seed0 710000 --workers 6 --out /Users/ligf9/zhuoji-mahjong-ai/runs/gates/r11_gateA.jsonl  → gateA_r11.log
[2026-11-07 12:40:12]   rc=0  53s  gateA_r11
[2026-11-07 12:40:12] R11 闸门 A: [summary] seats=[0, 1, 2, 3]  n=300  diff=+4.199±0.424  z=+9.90  p≈0.0000  符号 212:88  → 通过
[2026-11-07 12:40:12] $ /Users/ligf9/zhuoji-mahjong-ai/.venv/bin/python -c import torch,sys;sys.path.insert(0,'.');from zhuoji.net import build_from_ckpt;from zhuoji.forward import ensure_onnx;ck=torch.load('/Users/ligf9/zhuoji-mahjong-ai/models/exit_best.pt',map_location='cpu',weights_only=False);net,_=build_from_ckpt(ck);p=ensure_onnx(net,stem='exit_best');print('onnx:',p)  → export_exit_best.log
[2026-11-07 12:40:13]   rc=0  1s  export_exit_best
[2026-11-07 12:40:13] 晋升：exit_r11.pt → exit_best.pt
[2026-11-07 12:40:13] $ /Users/ligf9/zhuoji-mahjong-ai/.venv/bin/python eval_pairs.py --a /Users/ligf9/zhuoji-mahjong-ai/models/exit_r11.pt --b /Users/ligf9/zhuoji-mahjong-ai/models/dushan_master.pt --a-mode greedy --b-mode search32 --seeds 50 --seed0 720000 --workers 6 --out /Users/ligf9/zhuoji-mahjong-ai/runs/gates/r11_gateC.jsonl  → gateC_r11.log
[2026-11-07 13:01:51]   rc=0  1297s  gateC_r11
[2026-11-07 13:01:51] R11 闸门 C(趋势): [summary] seats=[0, 1, 2, 3]  n=50  diff=-0.880±0.831  z=-1.06  p≈0.2897  符号 24:26
[2026-11-07 13:01:51] $ /Users/ligf9/zhuoji-mahjong-ai/.venv/bin/python eval_pairs.py --a /Users/ligf9/zhuoji-mahjong-ai/models/exit_r11.pt --b /Users/ligf9/zhuoji-mahjong-ai/runs/gates/r11_prevbest.pt --mode greedy --seeds 300 --seed0 710000 --workers 6 --out /Users/ligf9/zhuoji-mahjong-ai/runs/gates/r11_gateD.jsonl  → gateD_r11.log
[2026-11-07 13:03:56]   rc=0  125s  gateD_r11
[2026-11-07 13:03:56] R11 闸门 D(vs 上轮最佳): [summary] seats=[0, 1, 2, 3]  n=300  diff=+0.430±0.378  z=+1.14  p≈0.2549  符号 149:148
[2026-11-07 13:03:56] P11 PPO 链起点 = exit_p10.pt
[2026-11-07 13:03:56] $ /Users/ligf9/zhuoji-mahjong-ai/.venv/bin/python train_ppo.py --init /Users/ligf9/zhuoji-mahjong-ai/models/exit_p10.pt --out /Users/ligf9/zhuoji-mahjong-ai/models/exit_p11.pt --hands 960 --workers 4 --threads 2 --max-states 60000 --epochs 2 --lr 5e-5 --seed0 1010000  → ppo_p11.log
[2026-11-07 13:09:14]   rc=0  318s  ppo_p11
[2026-11-07 13:09:14] P11 PPO 完成 init=exit_p10.pt → exit_p11.pt {"hands": 960, "states": 57696, "A_std": 8.279669761657715, "ret_std": 12.586057927987884, "kl": [0.00788864276688175, 0.010773299556056581], "secs": 317.5}
[2026-11-07 13:09:14] $ /Users/ligf9/zhuoji-mahjong-ai/.venv/bin/python eval_pairs.py --a /Users/ligf9/zhuoji-mahjong-ai/models/exit_p11.pt --b /Users/ligf9/zhuoji-mahjong-ai/models/exit_best.pt --mode greedy --seeds 300 --seed0 710000 --workers 6 --out /Users/ligf9/zhuoji-mahjong-ai/runs/gates/p11_gateD.jsonl  → gateD_p11.log
[2026-11-07 13:11:21]   rc=0  127s  gateD_p11
[2026-11-07 13:11:21] P11 闸门 D(vs 现最佳): [summary] seats=[0, 1, 2, 3]  n=300  diff=-0.378±0.347  z=-1.09  p≈0.2769  符号 141:156  → 不通过
[2026-11-07 13:11:21] $ /Users/ligf9/zhuoji-mahjong-ai/.venv/bin/python eval_pairs.py --a /Users/ligf9/zhuoji-mahjong-ai/models/exit_p11.pt --b /Users/ligf9/zhuoji-mahjong-ai/models/dushan_master.pt --mode greedy --seeds 300 --seed0 710000 --workers 6 --out /Users/ligf9/zhuoji-mahjong-ai/runs/gates/p11_gateA.jsonl  → gateA_p11.log
[2026-11-07 13:12:18]   rc=0  57s  gateA_p11
[2026-11-07 13:12:18] P11 闸门 A(vs master): [summary] seats=[0, 1, 2, 3]  n=300  diff=+4.269±0.393  z=+10.86  p≈0.0000  符号 230:67
[2026-11-07 13:12:18] P11 不晋升（未显著优于现最佳），ckpt 留档供终评 best-of
[2026-11-07 13:12:18] R11 轮次完成  best=exit_best.pt
[2026-11-07 13:12:18] $ /Users/ligf9/zhuoji-mahjong-ai/.venv/bin/python -c import torch,sys;sys.path.insert(0,'.');from zhuoji.net import build_from_ckpt;from zhuoji.forward import ensure_onnx;ck=torch.load('/Users/ligf9/zhuoji-mahjong-ai/models/exit_best.pt',map_location='cpu',weights_only=False);net,_=build_from_ckpt(ck);p=ensure_onnx(net,stem='exit_best');print('onnx:',p)  → export_exit_best.log
[2026-11-07 13:12:19]   rc=0  1s  export_exit_best
[2026-11-07 13:12:19] $ /Users/ligf9/zhuoji-mahjong-ai/rust_engine/target/release/annotate --student /Users/ligf9/zhuoji-mahjong-ai/models/onnx/exit_best.onnx --teacher models/onnx/dushan_master.onnx --student-enc 3 --student-device cpu --targets 100000 --threads 4 --out /Users/ligf9/zhuoji-mahjong-ai/runs/exit_r12 --seed0 720000  → annotate_r12.log
[2026-11-07 13:16:00] 【自检 #28 · R11 完整结算（A+C+D）+ P11 双闸门 · 硬决策点】
                        结论：健康、产出符合预期、主线零干预。驱动唯一 PID 23674
                        （PPID=99107 keepalive，06:22:33 起已 6h50m），lock 17.7s；
                        子进程唯一 annotate_r12；02:33:57 切片无 panic/NaN/超时；
                        磁盘 307Gi；TAU=1.0；Web 8770 `/`+`/api/meta` 双 200；
                        看门狗 99107 活；44/44 npz 合规。
[2026-11-07 13:16:00] 【R11 全流程（整轮 3h16m）】标注 11:56:44 rc=0（7932s、11.3k/h·worker、
                        4×25k）→ 训练 12:39:19 rc=0（2555s、400k 封顶、init=exit_best、
                        seed 20261117，train 1.7672→1.7425、**best val 1.7596@ep02**、
                        ep05 早停）→ **闸门 A +4.199±0.424 p≈0.0000（212:87）第 10 轮通过
                        → 12:40:13 晋升 exit_r11（md5 4251dd26…）**、r11.promoted=ok、
                        onnx 12:40 同步（三件套一致）→ 闸门 C −0.880±0.831 p≈0.2897
                        → 闸门 D +0.430±0.378 p≈0.2549（ns）→ P11 → 13:12:19 自动接续 R12。
[2026-11-07 13:16:00] 【本轮最重要 · R10 闸门 C 的「首次名义显著」被反向同量级完全回吐】
                        R10 C = +2.227±1.009 p=0.0273；R11 C = −0.880±0.831；
                        **环比 C R11−R10 = −3.107±1.255 p≈0.0133**（反向且同量级，同池同 seed）。
                        这不是"退步"，是**教科书式的回归均值**：R10 那个点是 10 次里挑出的最大值。
                        ⇒ #25 当时判「族系不显著（仿真 P(max|z|≥2.213,10)=0.239）、
                        拒绝宣布 §4 达成、拒绝把该点写入交付结论」的纪律**已被数据完整验证**。
                        若当时据 +2.227 宣布"搜索已蒸进网络"，今天必须公开撤稿。
                        固化：**闸门 C 的任何单轮值永久不得单独作证据**（含本轮 −0.880）；
                        C 的族系显著性只能由终评独立池 500 seed 一次性检验。
[2026-11-07 13:16:00] 【第二次最有价值 · 闸门 D 与 P−R 两个估计量的口径差首次被量化】
                        P11 两个数看着矛盾，实为**不同 estimand**，均已独立复算（300 seed 全交集）：
                          · p11_gateD  head-to-head：p11 坐庄 vs r11×3，−0.378±0.347
                          · P11−R11    vs-master 差分：+0.070±0.541
                        差的 0.448 ≈ 0.64(合并 SE) 的 0.7σ，属噪声内，**不是 bug**
                        （我逐 seed 用 `deltas[cand]−mean(其余三家)` 四座平均重算，两数逐位复现）。
                        根因：gateD 是同一副牌里两模型互相对打；P−R 是两场"各自 vs master×3"
                        的游戏相减 ⇒ 对手语境不同，估计量本来就不同。
                        **新判读纪律（收紧方向，预登记于 P12）**：P12 起两个估计量并报，
                        若符号相反且均不显著 ⇒ 直接判「PPO 无净收益」，不因 P−R 为正面放行。
                        理由：P−R 在"双方都远超 master"时会向 0 压缩（天花板效应），
                        head-to-head 才是"谁更强"的直问；两者打架时取保守。
[2026-11-07 13:16:00] 【P11 结果】960 局 / 57,696 状态 / 317.5s；kl [0.0079, 0.0108]
                        （较 P10 的 [0.0113, 0.0133] **下降 30%**，更新更保守，健康）、
                        A_std 8.28、ret_std 12.59。**闸门 A +4.269±0.393（231:68）=
                        历史最高绝对强度**，但闸门 D −0.378 → 不晋升（预期内，非失败）。
                        PPO 两轮累积：P10−R10 −0.069、P11−R11 +0.070 ⇒ **Σ≈+0.001，完全抵消**。
                        这不是"PPO 学会了"，是两轮各 ±0.07 的噪声；按预登记 P12 才够 n≈900。
[2026-11-07 13:16:00] 【平台期第 7 条证据】plateau_test 扩到 9 对：lag-1 = −0.484/−0.478/
                        −0.480/−0.469/−0.541/−0.547/−0.543/−0.522/−0.469，**均值 −0.504**
                        （理论值 −0.50），非相邻 28→36 对均值 +0.002。3 段 pool
                        mean +0.168、cluster SE 0.174、2σ=±0.348 分。
                        ⇒ R04~R11 与「固定中心 + iid 扰动」**继续不可区分**；第 7 个零边际轮
                        （A R11−R10 +0.334±0.522 p=0.5219）。best-of 第一段：R11 居首但与
                        R10/R09/R08/P10/R06 **全部并列(<1SE)** ⇒ 交付维度仍未拉开。
[2026-11-07 13:16:00] 【决策五项】①增速三选一**不触发**（A 最近两轮弱=False/False，
                        p≈0.0000 ⇒ TAU/Q-MSE/n_det/lr 一项不改）
                        ②天花板按工具规则**不确认**（C 平台=False，因 R10 的 +2.227 仍在带外），
                        但本轮 −3.107 p=.0133 已**证伪 R10 那个点**，实质支持"平台"；
                        按预登记不因单轮改判，顺延 R12 复核，**仍不改超参**
                        ③P2a 维持观察态（C 正增量 1/2，R11 回落）
                        ④PPO 维持 960 局/lr 5e-5/闸门 D 不放宽，**P12 前不动预算**
                        ⑤**下一步训练无需启动** —— R12 已由驱动 13:12:19 自动接续
                        （seed0=720000，起步 10.7k/h·worker）。零干预、不启第二驱动。
[2026-11-07 13:16:00] 【事实纠正 · 低影响】`models/web_best.pt` 实为 **exit_r07** 快照
                        （md5 1418f1fb ≡ exit_r07.pt），**不是此前记忆所写的 R08**
                        （R08 是 11846758）。线上跑的是 R07 权重（A +3.353），
                        较现最佳 R11 差 +0.846±0.533 p=0.1127（不显著）。
                        ⇒ 玩家体验无可测损失，**维持"终评 best-of 后一次性蓝绿刷新"不改**
                        （线上有真人，无统计可证收益就不动）。终评报告须按 R07 标注。
[2026-11-07 13:16:00] 【时间线/预算】R12 标注 13:12:19 起（10.7k/h）⇒ 落盘 ≈15:25
                        → 训练 ≈16:07 → A 16:08 → C 16:30 → D 16:32 → P12 16:38 → R13。
                        **P12 硬判读点 ≈16:4x**（自检 ≈72min 间隔：14:2x/15:3x/16:4x）。
                        剩余 133.2h ÷ 202min ⇒ **≈39 轮（R12~R50）**，DEADLINE 唯一硬约束。
                        终评另留 4.5h（含 §10.3 末轮补闸门），终评起始不晚于 11-12 22:00。
[2026-11-07 13:16:00] ============================================================
[2026-11-07 14:34:00] 【自检 #29 · 7×24 第 29 次 · R12 标注期巡检 + PPO 加速模式工程化（默认关闭）】
                        结论：**健康、产出符合预期、主线零干预**。驱动唯一 PID 23674
                        （PPID=99107 keepalive，06:22 起 8h09m），lock 0.5~29s；
                        R12 标注 62%（15.4~15.6k/25k·worker），11.7k/h·worker，
                        ETA ≈15:24（上限 16:42）；42 阶段 rc=0、panic/NaN 0；
                        44/44 npz；TAU=1.0；磁盘 307Gi；Web 双 200（models=web_best=r07）；
                        看门狗 99107 活。未启第二驱动/未打断/未并发评估。
[2026-11-07 14:34:00] 【本轮新测量 · PPO 链内配对（低噪声估计量）】同池 300 seed 逐 seed 配对：
                        P11−P10 = **+0.473±0.363 p=0.1923**（单轮 PPO 链内推进）
                        R11−R10 = +0.334±0.522 p=0.5219（单轮 DAgger 推进）
                        P11−R11 = +0.070±0.541 p=0.8971（PPO 链 vs DAgger 链）
                        **SE 结构本身是信息**：PPO 两代只差 960 局、高度相关 ⇒ 配对 SE 0.363
                        低于 DAgger 两代的 0.522。这是独立于 P−R 的「PPO 有没有在推进」估计量。
                        诚实限定：+0.473 仅 1.30σ，与 0 不可区分；8min vs 202min 的 25× 时间比
                        **没有推断意义**（两个噪声估计量之比），只能作「预算向 PPO 倾斜是
                        零风险探索」的理由，不得写成强度结论。
[2026-11-07 14:34:00] 【本轮落地 · PPO 加速模式（night_run.py，默认 RAPID_FROM=None）】
                        动机：DAgger 闸门 A 环比连续 8 轮为 0、lag-1 自相关 −0.504（纯 iid 噪声）
                        ⇒ 剩余 125h 同配方 DAgger 期望增益≈0，PPO 是唯一非零方差项。
                        设计：从 RAPID_FROM 起，每 12 轮里 11 轮只跑 PPO（≈13min，含闸门 D/A），
                        第 12 轮跑完整 DAgger（重锚+刷新交付物+维持 A/C 序列）。
                        安全：PPO 停用/失败 ⇒ ppo_round 返回 False ⇒ 该轮自动回落 DAgger；
                        加速轮写 rNN.rapid 标记，first_incomplete_round 跳过 ⇒ 断点续跑安全；
                        晋升口径一字不改（闸门 D + 闸门 A 双通过）。
                        验证：py_compile ✓；模块级单测（默认态 is_rapid 全 False、
                        first_incomplete_round=12 与改动前一致；RAPID_FROM=13 时
                        加速轮=R13~23/25~35/37~…、DAgger 轮=R24/R36）；train_ppo --dry 端到端 ✓。
                        **运行中的 PID 23674 不重载源码 ⇒ 行为逐位不变，零影响。**
[2026-11-07 14:34:00] 【预登记 · P12 决策规则（P12 数据尚未生成，先锁死）】
                        主判据（不变）：Σ=Σ_{r=10,11,12}(P_r−R_r)，三段落 cluster SE。
                        辅判据（新增、仅支持不得覆盖）：链内 D 池化 = mean(P11−P10, P12−P11)。
                        H1（Σ≥+0.30 且 p<0.10 或 D池化≥+0.25）→ 开加速模式；
                        H0（|Σ|<2σ 且 P 的 A 无退化）→ **同样开加速**（DAgger 期望 0）；
                        H−1（Σ≤−0.30 且 p<0.10 或 P 最新 A 掉 >2σ）→ 不加速、PPO_EVERY=2。
                        冻结：TAU/Q-MSE/n_det/lr/闸门判据一字不改；web_best 仍待终评一次性刷新。
[2026-11-07 14:34:00] 【切换协议（不武装任何触发器，§6.26.10 教训）】在 ≈16:46 自检
                        （R13 标注刚起 ≈5min）手工执行：先杀 annotate 再杀驱动 → 确认
                        runs/exit_r13 无已落盘分片 → 改 RAPID_FROM=13 且 MAX_ROUND 60→400
                        （否则 2h 内跑满 60 轮空转 5 天）→ py_compile → keepalive ≤120s 拉起
                        → 复核新 PID/lock/「R13 加速轮」字样。禁止日志文本触发器、禁止
                        标注中途重启、禁止同时改 PPO_HANDS 与 RAPID_HANDS。
[2026-11-07 14:34:00] 【时间线】R12 落盘 ≈15:24 → 训练 16:06 → A 16:07 → C 16:29 → D 16:31
                        → P12 16:38 → R13 16:41。剩余 132.5h：现状口径 ≈39 轮（R12~R50）；
                        若 16:46 切加速模式 ≈570 轮≈110 万局 PPO。终评起始不晚于 11-12 22:00。
[2026-11-07 14:34:00] ============================================================
[2026-11-07 15:20:25]   rc=0  7686s  annotate_r12
[2026-11-07 15:20:25] R12 标注完成 student=exit_best.onnx seed0=720000 → 4×25k
[2026-11-07 15:20:25] $ /Users/ligf9/zhuoji-mahjong-ai/.venv/bin/python train_exit.py --data runs/exit_r01/exit_w*.npz runs/exit_r02/exit_w*.npz runs/exit_r03/exit_w*.npz runs/exit_r04/exit_w*.npz runs/exit_r05/exit_w*.npz runs/exit_r06/exit_w*.npz runs/exit_r07/exit_w*.npz runs/exit_r08/exit_w*.npz runs/exit_r09/exit_w*.npz runs/exit_r10/exit_w*.npz runs/exit_r11/exit_w*.npz runs/exit_r12/exit_w*.npz --out /Users/ligf9/zhuoji-mahjong-ai/models/exit_r12.pt --epochs 30 --window 400000 --batch 256 --seed 20261118 --init /Users/ligf9/zhuoji-mahjong-ai/models/exit_best.pt  → train_r12.log
[2026-11-07 15:54:17]   rc=0  2032s  train_r12
[2026-11-07 15:54:17] R12 训练完成 init=exit_best.pt → exit_r12.pt  (['[done] best val 1.7527 → /Users/ligf9/zhuoji-mahjong-ai/models/exit_r12.pt'])
[2026-11-07 15:54:17] $ /Users/ligf9/zhuoji-mahjong-ai/.venv/bin/python eval_pairs.py --a /Users/ligf9/zhuoji-mahjong-ai/models/exit_r12.pt --b /Users/ligf9/zhuoji-mahjong-ai/models/dushan_master.pt --mode greedy --seeds 300 --seed0 710000 --workers 6 --out /Users/ligf9/zhuoji-mahjong-ai/runs/gates/r12_gateA.jsonl  → gateA_r12.log
[2026-11-07 15:55:10]   rc=0  53s  gateA_r12
[2026-11-07 15:55:10] R12 闸门 A: [summary] seats=[0, 1, 2, 3]  n=300  diff=+4.100±0.408  z=+10.05  p≈0.0000  符号 218:76  → 通过
[2026-11-07 15:55:10] $ /Users/ligf9/zhuoji-mahjong-ai/.venv/bin/python -c import torch,sys;sys.path.insert(0,'.');from zhuoji.net import build_from_ckpt;from zhuoji.forward import ensure_onnx;ck=torch.load('/Users/ligf9/zhuoji-mahjong-ai/models/exit_best.pt',map_location='cpu',weights_only=False);net,_=build_from_ckpt(ck);p=ensure_onnx(net,stem='exit_best');print('onnx:',p)  → export_exit_best.log
[2026-11-07 15:55:11]   rc=0  1s  export_exit_best
[2026-11-07 15:55:11] 晋升：exit_r12.pt → exit_best.pt
[2026-11-07 15:55:11] $ /Users/ligf9/zhuoji-mahjong-ai/.venv/bin/python eval_pairs.py --a /Users/ligf9/zhuoji-mahjong-ai/models/exit_r12.pt --b /Users/ligf9/zhuoji-mahjong-ai/models/dushan_master.pt --a-mode greedy --b-mode search32 --seeds 50 --seed0 720000 --workers 6 --out /Users/ligf9/zhuoji-mahjong-ai/runs/gates/r12_gateC.jsonl  → gateC_r12.log
[2026-11-07 16:17:02]   rc=0  1311s  gateC_r12
[2026-11-07 16:17:02] R12 闸门 C(趋势): [summary] seats=[0, 1, 2, 3]  n=50  diff=-0.340±0.765  z=-0.44  p≈0.6567  符号 20:29
[2026-11-07 16:17:02] $ /Users/ligf9/zhuoji-mahjong-ai/.venv/bin/python eval_pairs.py --a /Users/ligf9/zhuoji-mahjong-ai/models/exit_r12.pt --b /Users/ligf9/zhuoji-mahjong-ai/runs/gates/r12_prevbest.pt --mode greedy --seeds 300 --seed0 710000 --workers 6 --out /Users/ligf9/zhuoji-mahjong-ai/runs/gates/r12_gateD.jsonl  → gateD_r12.log
[2026-11-07 16:19:05]   rc=0  123s  gateD_r12
[2026-11-07 16:19:05] R12 闸门 D(vs 上轮最佳): [summary] seats=[0, 1, 2, 3]  n=300  diff=+0.401±0.398  z=+1.01  p≈0.3138  符号 154:142
[2026-11-07 16:19:05] P12 PPO 链起点 = exit_p11.pt
[2026-11-07 16:19:05] $ /Users/ligf9/zhuoji-mahjong-ai/.venv/bin/python train_ppo.py --init /Users/ligf9/zhuoji-mahjong-ai/models/exit_p11.pt --out /Users/ligf9/zhuoji-mahjong-ai/models/exit_p12.pt --hands 960 --workers 4 --threads 2 --max-states 60000 --epochs 2 --lr 5e-5 --seed0 1020000  → ppo_p12.log
[2026-11-07 16:24:02]   rc=0  297s  ppo_p12
[2026-11-07 16:24:02] P12 PPO 完成 init=exit_p11.pt → exit_p12.pt {"hands": 960, "states": 54453, "A_std": 8.732075691223145, "ret_std": 13.361636626301934, "kl": [0.006081004927105362, 0.009022425029955875], "secs": 295.7}
[2026-11-07 16:24:02] $ /Users/ligf9/zhuoji-mahjong-ai/.venv/bin/python eval_pairs.py --a /Users/ligf9/zhuoji-mahjong-ai/models/exit_p12.pt --b /Users/ligf9/zhuoji-mahjong-ai/models/exit_best.pt --mode greedy --seeds 300 --seed0 710000 --workers 6 --out /Users/ligf9/zhuoji-mahjong-ai/runs/gates/p12_gateD.jsonl  → gateD_p12.log
[2026-11-07 16:26:09]   rc=0  127s  gateD_p12
[2026-11-07 16:26:09] P12 闸门 D(vs 现最佳): [summary] seats=[0, 1, 2, 3]  n=300  diff=+0.596±0.407  z=+1.46  p≈0.1438  符号 158:141  → 不通过
[2026-11-07 16:26:09] $ /Users/ligf9/zhuoji-mahjong-ai/.venv/bin/python eval_pairs.py --a /Users/ligf9/zhuoji-mahjong-ai/models/exit_p12.pt --b /Users/ligf9/zhuoji-mahjong-ai/models/dushan_master.pt --mode greedy --seeds 300 --seed0 710000 --workers 6 --out /Users/ligf9/zhuoji-mahjong-ai/runs/gates/p12_gateA.jsonl  → gateA_p12.log
[2026-11-07 16:27:06]   rc=0  57s  gateA_p12
[2026-11-07 16:27:06] P12 闸门 A(vs master): [summary] seats=[0, 1, 2, 3]  n=300  diff=+3.957±0.441  z=+8.97  p≈0.0000  符号 212:85
[2026-11-07 16:27:06] P12 不晋升（未显著优于现最佳），ckpt 留档供终评 best-of
[2026-11-07 16:27:06] R12 轮次完成  best=exit_best.pt
[2026-11-07 16:27:06] $ /Users/ligf9/zhuoji-mahjong-ai/.venv/bin/python -c import torch,sys;sys.path.insert(0,'.');from zhuoji.net import build_from_ckpt;from zhuoji.forward import ensure_onnx;ck=torch.load('/Users/ligf9/zhuoji-mahjong-ai/models/exit_best.pt',map_location='cpu',weights_only=False);net,_=build_from_ckpt(ck);p=ensure_onnx(net,stem='exit_best');print('onnx:',p)  → export_exit_best.log
[2026-11-07 16:27:07]   rc=0  1s  export_exit_best
[2026-11-07 16:27:07] $ /Users/ligf9/zhuoji-mahjong-ai/rust_engine/target/release/annotate --student /Users/ligf9/zhuoji-mahjong-ai/models/onnx/exit_best.onnx --teacher models/onnx/dushan_master.onnx --student-enc 3 --student-device cpu --targets 100000 --threads 4 --out /Users/ligf9/zhuoji-mahjong-ai/runs/exit_r13 --seed0 730000  → annotate_r13.log
[2026-11-07 16:35:16] ============================================================
[2026-11-07 16:35:16] 7×24 ExIt 驱动启动：R13 起，截止 2026-11-13 02:30（R1 已完成；断点续跑 + 每小时自检自动化）
[2026-11-07 16:35:16] P13 PPO 链起点 = exit_p12.pt  hands=1920 max_states=120000
[2026-11-07 16:35:16] $ /Users/ligf9/zhuoji-mahjong-ai/.venv/bin/python train_ppo.py --init /Users/ligf9/zhuoji-mahjong-ai/models/exit_p12.pt --out /Users/ligf9/zhuoji-mahjong-ai/models/exit_p13.pt --hands 1920 --workers 4 --threads 2 --max-states 120000 --epochs 2 --lr 5e-5 --seed0 1030000  → ppo_p13.log
[2026-11-07 16:45:04]   rc=0  588s  ppo_p13
[2026-11-07 16:45:04] P13 PPO 完成 init=exit_p12.pt → exit_p13.pt {"hands": 1920, "states": 106698, "A_std": 8.880213737487793, "ret_std": 13.379827789125937, "kl": [0.00633978738245159, 0.008732129231053997], "vl": 1.1985846178280077, "ent": 1.4788974483522008, "gnorm": 1.421008334171286, "clipfrac": 0.08727000454525581, "ent_rollout": 1.8204447467242617, "kl_stop": false, "secs": 587.0}
[2026-11-07 16:45:04] $ /Users/ligf9/zhuoji-mahjong-ai/.venv/bin/python eval_pairs.py --a /Users/ligf9/zhuoji-mahjong-ai/models/exit_p13.pt --b /Users/ligf9/zhuoji-mahjong-ai/models/exit_best.pt --mode greedy --seeds 300 --seed0 710000 --workers 6 --out /Users/ligf9/zhuoji-mahjong-ai/runs/gates/p13_gateD.jsonl  → gateD_p13.log
[2026-11-07 16:47:10]   rc=0  127s  gateD_p13
[2026-11-07 16:47:10] P13 闸门 D(vs 现最佳): [summary] seats=[0, 1, 2, 3]  n=300  diff=+0.369±0.414  z=+0.89  p≈0.3728  符号 147:151  → 不通过
[2026-11-07 16:47:10] $ /Users/ligf9/zhuoji-mahjong-ai/.venv/bin/python eval_pairs.py --a /Users/ligf9/zhuoji-mahjong-ai/models/exit_p13.pt --b /Users/ligf9/zhuoji-mahjong-ai/models/dushan_master.pt --mode greedy --seeds 300 --seed0 710000 --workers 6 --out /Users/ligf9/zhuoji-mahjong-ai/runs/gates/p13_gateA.jsonl  → gateA_p13.log
[2026-11-07 16:48:15]   rc=0  64s  gateA_p13
[2026-11-07 16:48:15] P13 闸门 A(vs master): [summary] seats=[0, 1, 2, 3]  n=300  diff=+4.218±0.403  z=+10.45  p≈0.0000  符号 218:79
[2026-11-07 16:48:15] P13 不晋升（未显著优于现最佳），ckpt 留档供终评 best-of
[2026-11-07 16:48:15] R13 加速轮完成（PPO-only：无标注/无训练，含闸门 D/A）
[2026-11-07 16:48:15] P14 PPO 链起点 = exit_p13.pt  hands=1920 max_states=120000
[2026-11-07 16:48:15] $ /Users/ligf9/zhuoji-mahjong-ai/.venv/bin/python train_ppo.py --init /Users/ligf9/zhuoji-mahjong-ai/models/exit_p13.pt --out /Users/ligf9/zhuoji-mahjong-ai/models/exit_p14.pt --hands 1920 --workers 4 --threads 2 --max-states 120000 --epochs 2 --lr 5e-5 --seed0 1040000  → ppo_p14.log
[2026-11-07 16:47:00] 【自检 #30 · 7×24 第 30 次 · R12 结算 + P12=H0 → PPO 加速模式已生效 + 首个加速轮 R13】
                        结论：**健康、产出符合预期、主线零干预**。驱动唯一 PID 74896
                        （PPID=99107 keepalive，16:35:11 起），lock 16.5s；阶段 = **R14 加速轮**
                        （PPO-only，train_ppo 76141 + 4 workers，16:48:15 起）；
                        R13 加速轮 16:35:16→16:48:15 **13min 整**，已写 runs/gates/r13.rapid；
                        48/48 npz 全合规；磁盘 306Gi；TAU=1.0/lr=1e-3/PPO_HANDS=960 未改；
                        Web 8770 双 200（models=web_best=r07）；看门狗 99107 活。
                        未启第二驱动/未打断/未并发评估（唯一的 1 核探针为只读代理探测，nice 10）。
[2026-11-07 16:47:00] 【R12 结算 · 整轮 3h15m】标注 13:12:19→15:20:25（7686s、11.7k/h·worker、4×25k）
                        → 训练 15:20:25→15:54:17（2032s、400k、init=exit_best、best val 1.7527、
                        ep04 早停）→ **闸门 A +4.100±0.408（220:76）p≈0.0000 第 11 轮通过
                        → 15:55:11 晋升 exit_r12（md5 332ba6d2 ≡ exit_best，onnx 同步）**
                        → 闸门 C −0.340±0.765 p≈0.6567 → 闸门 D +0.401±0.398 p≈0.3138
                        → P12 960 局/54,453 状态/kl[0.0061,0.0090]/D +0.596±0.407 p=.1438 不晋升
                        → A +3.957±0.441。环比 A R12−R11 **−0.099±0.521 p=.8494 = 第 8 个零边际轮**。
[2026-11-07 16:47:00] 【P12 判读（预登记 §6.33.3 口径，16:28:34 已执行）】Σ_{10..12}(P−R) =
                        **−0.142±1.050（cluster SE）z=−0.14 p=0.8923**；各段 −0.069/+0.070/−0.143；
                        链内 D 池化 +0.081（辅，仅支持）⇒ **H0 无差别 ⇒ 按预登记同样开加速模式**
                        （DAgger 期望已 ≈0，PPO 是唯一非零方差项）。输出
                        runs/gates/P12_decision_20261107_162834.txt，阈值在看数据前锁死。
[2026-11-07 16:47:00] 【切换已生效 · 本节补记缺失文档】16:27 旧驱动按 DAgger 起 R13 标注 →
                        16:33~16:34 kill annotate+驱动 → keepalive 16:35:15 拉起 / 16:35:48 恢复
                        → 新驱动首轮直接走加速分支。**数据零损失**：runs/exit_r13/ 不存在（已核），
                        48/48 npz 完好，损失=8min 标注 + 停机 76s。配置：RAPID_FROM=13、
                        MAX_ROUND 60→400、RAPID_HANDS=1920、RAPID_DAAGGER_EVERY=12；
                        **PPO_HANDS 保持 960 未动**（单变量）。断点续跑安全性实测：
                        加速轮无 npz/无 exit_rNN.pt，靠 rNN.rapid + exit_pNN.pt 双条件跳过，R13 已正确跳过。
                        **流程缺口**：该次执行会话在写 docs/日志前结束 ⇒ 本次补齐 §6.34。
                        此后任何切换必须在同一会话内落文档。
[2026-11-07 16:47:00] 【R13 首个加速轮实测 · 13min/轮】PPO 1920 局/106,698 状态/588s/
                        A_std 8.88/kl[0.0063,0.0087]/clipfrac 0.0872/kl_stop=false ⇒ 四项健康判据全过；
                        闸门 D **+0.369±0.414 p=.3728（147:151）不晋升**（单轮分辨率不足，预期内）；
                        闸门 A **+4.218±0.403 p≈0.0000（218:79）历史第 2 高**（仅次 P11 +4.269）。
                        成本拆解：PPO 588s + D 127s + A 64s + 开销 ≈ 13min（DAgger 整轮 202min）。
                        经济性：12 轮周期 = 11×13min + 3h15m = **5h38m**；剩余 129h ⇒ **≈23 周期 ≈275 轮
                        ≈2,930 万决策状态**（纯 DAgger 同时间仅 ≈38 轮 / 380 万目标）。经验量放大约 7.7×，
                        但性质不同（无教师标签），**不得换算成「学得更多」**。
[2026-11-07 16:47:00] 【新事实 · 榜单头两名已是 PPO 候选】best-of 第一段：
                        P11 +4.269 > R11 +4.199 > R12 +4.100 > P12 +3.957 > R10 +3.864 > P10 +3.796 >
                        R09 +3.736…；top1−top2 = +0.070±0.541 **<1SE ⇒ 仍全并列**（R04 及以后全并列）。
                        P13 +4.218 vs R12 +4.100 差 +0.118±0.57 ns ⇒ **PPO 链绝对强度已与 DAgger 链持平**。
                        平台期第 8 条证据：lag-1 10 对均值 −0.501、非相邻 +0.001；3 段 pool +0.121
                        cluster SE 0.164 ⇒ 可分辨下限 ±0.328 分。
[2026-11-07 16:47:00] 【教师-学生差距分解 tools/ts_gap.py（3,200 状态，in-sample 偏差已知）】
                        agree@1 0.568 / regret 0.128 分 / top3 覆盖 0.880 / 价值 RMSE 1.335 分。
                        分头：discard 0.509（n=2722，85%）、self_kong 0.774（53）、respond 0.920（425）；
                        分阶段：early 0.486 / mid 0.568 / late 0.644，regret late 0.176 > early 0.090，
                        价值 RMSE early 0.42 → late 2.04（终局波动更大，符合预期）。
                        预登记裁决 = **③ 证据不足/混合（rare 1%、head spread 41pp）⇒ 不改配方**。
                        可读结论：差距**不在稀有局面**，而在主决策头 discard 且全阶段均匀；
                        regret 仅 0.14 分 ⇒ 分歧绝大多数落在近平局，与 #18「top1−top2 中位 0.144 分、
                        63.6% 状态 <0.25 分」一致。想突破只能换信息源（PPO/更强搜索/更强网络），
                        不是调 τ、Q-MSE、n_det。
[2026-11-07 16:47:00] 【N64 教师代理探测 tools/n64_probe.py（32 状态，本次新跑）】
                        同 seed/ply3/topk3/ff_cap24，只改 n_det 32→64（N64 前 32 个 determinization
                        与 N32 完全相同 ⇒ 配对噪声最小）。**性质限定：决策级差异上界代理，不是强度判定，
                        不进任何闸门**。预登记：①一致率≥0.95 且 mean|ΔQ|<0.30 ⇒ 天花板已到；
                        ②一致率<0.85 或 mean|ΔQ|≥0.50 ⇒ 含实质新信息（R24 重锚点考虑切 N64）；
                        ③其余 ⇒ 按排期跑正式 50-seed，不据代理改配方。结果由下次自检读
                        runs/night_logs/n64_probe.log 判定（写本行时仍在跑）。
[2026-11-07 16:47:00] 【新登记风险 P2 · 加速模式 267 次闸门 D 的多重比较】剩余 129h 约 267 次
                        检验，α=.05 下**期望约 13 次假晋升**。不定级 P0/P1 的理由：exit_best 本就被
                        「A vs master 每轮必过」不断覆盖（#18 已降 P2）；web_best 不跟随 PPO（零线上风险）；
                        终评两段式 best-of 才是交付判定。**处置=事件式审计**（不改代码、不重启）：
                        自检发现新增 pNN.promoted 时用独立池 730000+300 复核，若 |Δ|<1SE 或 Δ≤0
                        ⇒ 从 exit_r{最近 DAgger}.pt 恢复 exit_best（p ckpt 保留仍可 best-of）。
[2026-11-07 16:47:00] 【下一硬决策点（预登记，看数据前锁死）≈19:00 · P23 vs R12】R13~R23 是同锚
                        链式续训 ⇒ 天然剂量-反应，**不做「11 次 D 谁先过 0.05」**（多重比较必假阳性），
                        只做**一次**累积检验 P23−R12（同池 300 seed 配对）。判据：Δ≥+1.5 且 p<.05 ⇒ 有累积
                        增益，下一周期维持 1920 局（不加预算）；|Δ|<1.5 ns ⇒ 无累积，第二周期同配方继续；
                        Δ≤−1.5 且 p<.05 ⇒ 写 runs/ppo_disabled 停用 PPO，DAgger-only 至 DEADLINE，
                        剩余预算转终评扩容（1000 seed 第二段）+ 产品打磨。
[2026-11-07 16:47:00] 【决策五项】①增速三选一**不触发**（A 最近两轮弱=False/False，p≈0.0000
                        ⇒ TAU/Q-MSE/n_det/lr 一项不改）②天花板按工具规则**不确认**（C 平台=False），
                        但 §6.34.6 差距分解与 lag-1 证据实质支持「平台」，**仍不改超参**
                        ③P2a 维持观察态（C 正增量 1/2）④PPO 已升级为加速模式，判据改为
                        P23−R12 单次累积检验 + rapid_guard 三轮 D 负向护栏，**闸门 D 判据不放宽**
                        ⑤**下一步无需启动** —— R14 已由驱动 16:48:15 自动接续，零干预。
[2026-11-07 16:47:00] 【时间线】R14 16:48 起，每 13min ⇒ R23 收口 ≈18:58；R24 完整 DAgger：
                        18:58 标注 → ≈21:05 落盘 → 训练 ≈21:47 → A 21:48 → C ≈22:10 → D ≈22:12
                        → P24 ≈22:20 → R25（第二周期）。剩余 129h。终评仍留 4.5h，起始不晚于 11-12 22:00。
[2026-11-07 16:47:00] ============================================================
[2026-11-07 16:59:49]   rc=0  695s  ppo_p14
[2026-11-07 16:59:49] P14 PPO 完成 init=exit_p13.pt → exit_p14.pt {"hands": 1920, "states": 102286, "A_std": 9.026036262512207, "ret_std": 13.36069133378459, "kl": [0.006836862885465962, 0.008937923765915911], "vl": 1.2439688631147146, "ent": 1.3949881824851036, "gnorm": 1.419979564845562, "clipfrac": 0.08480895136483013, "ent_rollout": 1.7031686187398762, "kl_stop": false, "secs": 693.2}
[2026-11-07 16:59:49] $ /Users/ligf9/zhuoji-mahjong-ai/.venv/bin/python eval_pairs.py --a /Users/ligf9/zhuoji-mahjong-ai/models/exit_p14.pt --b /Users/ligf9/zhuoji-mahjong-ai/models/exit_best.pt --mode greedy --seeds 300 --seed0 710000 --workers 6 --out /Users/ligf9/zhuoji-mahjong-ai/runs/gates/p14_gateD.jsonl  → gateD_p14.log
[2026-11-07 17:02:24]   rc=0  155s  gateD_p14
[2026-11-07 17:02:24] P14 闸门 D(vs 现最佳): [summary] seats=[0, 1, 2, 3]  n=300  diff=+0.220±0.402  z=+0.55  p≈0.5846  符号 144:156  → 不通过
[2026-11-07 17:02:24] $ /Users/ligf9/zhuoji-mahjong-ai/.venv/bin/python eval_pairs.py --a /Users/ligf9/zhuoji-mahjong-ai/models/exit_p14.pt --b /Users/ligf9/zhuoji-mahjong-ai/models/dushan_master.pt --mode greedy --seeds 300 --seed0 710000 --workers 6 --out /Users/ligf9/zhuoji-mahjong-ai/runs/gates/p14_gateA.jsonl  → gateA_p14.log
[2026-11-07 17:03:32]   rc=0  67s  gateA_p14
[2026-11-07 17:03:32] P14 闸门 A(vs master): [summary] seats=[0, 1, 2, 3]  n=300  diff=+4.032±0.421  z=+9.59  p≈0.0000  符号 214:85
[2026-11-07 17:03:32] P14 不晋升（未显著优于现最佳），ckpt 留档供终评 best-of
[2026-11-07 17:03:32] R14 加速轮完成（PPO-only：无标注/无训练，含闸门 D/A）
[2026-11-07 17:03:32] P15 PPO 链起点 = exit_p14.pt  hands=1920 max_states=120000
[2026-11-07 17:03:32] $ /Users/ligf9/zhuoji-mahjong-ai/.venv/bin/python train_ppo.py --init /Users/ligf9/zhuoji-mahjong-ai/models/exit_p14.pt --out /Users/ligf9/zhuoji-mahjong-ai/models/exit_p15.pt --hands 1920 --workers 4 --threads 2 --max-states 120000 --epochs 2 --lr 5e-5 --seed0 1050000  → ppo_p15.log
[2026-11-07 17:05:00] 【补记 · N64 教师代理探测结果（32 状态，seed0=2400000，17:02 完成）】
                        平均合法动作 7.7；**top1 一致率 0.781（25/32）**；**mean|Q32−Q64| = 0.143 分/动作**
                        （max 1.745）；7 例换动作中 N64 所选动作在 N32 口径下**平均落后 top1 0.201 分**。
                        预登记机械触发 ②（一致率<0.85），但三条实测把措辞往下压：
                        ①Q 差异 0.143 比 ② 的另一阈值（≥0.50）低 3.5 倍；
                        ②若 N64 真更强，换动作后应在 N32 口径下领先/持平，实测平均**落后 0.201 分**
                        ⇒ 一致的读法是「两个噪声估计在近平局附近抖」，不是「N64 找到 N32 没找到的好棋」；
                        ③**成本硬约束**：N64 教师让标注教师算力翻倍 ⇒ 100k 从 2.2h 变 ≈4.4h **超过 3.5h 超时线**，
                        超时会整轮丢数据 ⇒ **N64 教师在本轮主循环事实上不可用**（除非每轮砍到 50k，
                        那会让每周期唯一一次 DAgger 重锚的信息量减半）。
                        **处置（含对预登记行动项的显式修正）**：①不改任何阈值（② 触发条件留档可审计）；
                        ②正式 50-seed 对局判定（--a-n-det 64 --b-n-det 32，35~40min）从「R24 前」推迟到
                        **DEADLINE 后终评窗口**（训练已停、零机会成本；且它不可能改变剩余 129h 的任何动作）；
                        ③提前条件：若 ≈19:00 P23−R12 |Δ|<1.5，说明「换信息源」是唯一剩下的杠杆，
                        则在下一周期 PPO 窗口内提前跑（4 workers ≈55min）；
                        ④**禁止**据此宣称「已验证 N64 更强 / 已验证天花板」——这是 n=32 的决策级代理。
                        另：本次发现 16:53 起有**另一个会话在并行跑 n64_probe --states 80**（PID 76661），
                        与本探测写同一输出文件；下一自检读到的 JSON 可能是 80 状态版本，**以文件内容为准、记 n**。
[2026-11-07 17:05:00] 【R14 实测（加速轮节奏确认）】PPO 16:48:15→16:59:49 = **694s**（比 R13 的 588s 慢 18%，
                        原因是两个 1 核代理探测并发占核，非 PPO 本身退化）；闸门 D **+0.220±0.402 p=.5846
                        （144:156）不通过**（第 2 个加速轮，仍在 ±1SE 内）；闸门 A 17:02 开跑。
                        结论：13min/轮 的估计在**无外部探测**时成立，自检期若跑探针会把单轮拉长到 ~15~18min，
                        规划时间线时按 13min 排、不按 18min 排（探针是偶发）。
[2026-11-07 17:05:00] ============================================================
[2026-11-07 17:06:00] 【R14 收口】P14 闸门 A **+4.032±0.421 p≈0.0000（214:85）**（历史第 4 高，次于 P11 4.269 /
                        P13 4.218 / R11 4.199）；不晋升；R14 加速轮 16:48:15→17:03:32 ≈ **15min**
                        （含探测并发拖慢 2min），17:03:32 驱动自动接续 **R15**。
                        两个加速轮 D 汇总：R13 +0.369±0.414、R14 +0.220±0.402 —— **均为正、均不可分辨**
                        （vs 冻结 R12），符合「单轮分辨率不足」预登记；rapid_guard 需 R13~R15 三轮才判读。
                        下一硬点仍为 ≈19:00 的 **P23−R12 单次累积检验**（不在 11 次 D 里挑显著）。
[2026-11-07 17:14:16]   rc=0  644s  ppo_p15
[2026-11-07 17:14:16] P15 PPO 完成 init=exit_p14.pt → exit_p15.pt {"hands": 1920, "states": 101538, "A_std": 8.843988418579102, "ret_std": 13.099031850356981, "kl": [0.0067509579254075025, 0.009254365626626392], "vl": 1.2027166072157227, "ent": 1.3048329617574774, "gnorm": 1.4060331426129233, "clipfrac": 0.08697835279051543, "ent_rollout": 1.5682381191605004, "kl_stop": false, "secs": 643.1}
[2026-11-07 17:14:16] $ /Users/ligf9/zhuoji-mahjong-ai/.venv/bin/python eval_pairs.py --a /Users/ligf9/zhuoji-mahjong-ai/models/exit_p15.pt --b /Users/ligf9/zhuoji-mahjong-ai/models/exit_best.pt --mode greedy --seeds 300 --seed0 710000 --workers 6 --out /Users/ligf9/zhuoji-mahjong-ai/runs/gates/p15_gateD.jsonl  → gateD_p15.log
[2026-11-07 17:16:58]   rc=0  162s  gateD_p15
[2026-11-07 17:16:58] P15 闸门 D(vs 现最佳): [summary] seats=[0, 1, 2, 3]  n=300  diff=+0.750±0.393  z=+1.91  p≈0.0566  符号 160:137  → 不通过
[2026-11-07 17:16:58] $ /Users/ligf9/zhuoji-mahjong-ai/.venv/bin/python eval_pairs.py --a /Users/ligf9/zhuoji-mahjong-ai/models/exit_p15.pt --b /Users/ligf9/zhuoji-mahjong-ai/models/dushan_master.pt --mode greedy --seeds 300 --seed0 710000 --workers 6 --out /Users/ligf9/zhuoji-mahjong-ai/runs/gates/p15_gateA.jsonl  → gateA_p15.log
[2026-11-07 17:18:18]   rc=0  79s  gateA_p15
[2026-11-07 17:18:18] P15 闸门 A(vs master): [summary] seats=[0, 1, 2, 3]  n=300  diff=+4.880±0.435  z=+11.22  p≈0.0000  符号 219:80
[2026-11-07 17:18:18] P15 不晋升（未显著优于现最佳），ckpt 留档供终评 best-of
[2026-11-07 17:18:18] R15 加速轮完成（PPO-only：无标注/无训练，含闸门 D/A）
[2026-11-07 17:18:18] P16 PPO 链起点 = exit_p15.pt  hands=1920 max_states=120000
[2026-11-07 17:18:18] $ /Users/ligf9/zhuoji-mahjong-ai/.venv/bin/python train_ppo.py --init /Users/ligf9/zhuoji-mahjong-ai/models/exit_p15.pt --out /Users/ligf9/zhuoji-mahjong-ai/models/exit_p16.pt --hands 1920 --workers 4 --threads 2 --max-states 120000 --epochs 2 --lr 5e-5 --seed0 1060000  → ppo_p16.log
[2026-11-07 17:28:29]   rc=0  611s  ppo_p16
[2026-11-07 17:28:29] P16 PPO 完成 init=exit_p15.pt → exit_p16.pt {"hands": 1920, "states": 97527, "A_std": 8.724430084228516, "ret_std": 12.63110413489916, "kl": [0.005468159043422614, 0.008332580858884315], "vl": 1.1677942581846332, "ent": 1.2298695348066295, "gnorm": 1.3742068782253216, "clipfrac": 0.08112335310676905, "ent_rollout": 1.436437017032301, "kl_stop": false, "secs": 610.0}
[2026-11-07 17:28:29] $ /Users/ligf9/zhuoji-mahjong-ai/.venv/bin/python eval_pairs.py --a /Users/ligf9/zhuoji-mahjong-ai/models/exit_p16.pt --b /Users/ligf9/zhuoji-mahjong-ai/models/exit_best.pt --mode greedy --seeds 300 --seed0 710000 --workers 6 --out /Users/ligf9/zhuoji-mahjong-ai/runs/gates/p16_gateD.jsonl  → gateD_p16.log
[2026-11-07 17:31:04]   rc=0  155s  gateD_p16
[2026-11-07 17:31:04] P16 闸门 D(vs 现最佳): [summary] seats=[0, 1, 2, 3]  n=300  diff=+1.073±0.410  z=+2.62  p≈0.0088  符号 168:130  → 通过
[2026-11-07 17:31:04] $ /Users/ligf9/zhuoji-mahjong-ai/.venv/bin/python eval_pairs.py --a /Users/ligf9/zhuoji-mahjong-ai/models/exit_p16.pt --b /Users/ligf9/zhuoji-mahjong-ai/models/dushan_master.pt --mode greedy --seeds 300 --seed0 710000 --workers 6 --out /Users/ligf9/zhuoji-mahjong-ai/runs/gates/p16_gateA.jsonl  → gateA_p16.log
[2026-11-07 17:32:11]   rc=0  67s  gateA_p16
[2026-11-07 17:32:11] P16 闸门 A(vs master): [summary] seats=[0, 1, 2, 3]  n=300  diff=+4.191±0.410  z=+10.21  p≈0.0000  符号 218:80
[2026-11-07 17:32:11] $ /Users/ligf9/zhuoji-mahjong-ai/.venv/bin/python -c import torch,sys;sys.path.insert(0,'.');from zhuoji.net import build_from_ckpt;from zhuoji.forward import ensure_onnx;ck=torch.load('/Users/ligf9/zhuoji-mahjong-ai/models/exit_best.pt',map_location='cpu',weights_only=False);net,_=build_from_ckpt(ck);p=ensure_onnx(net,stem='exit_best');print('onnx:',p)  → export_exit_best.log
[2026-11-07 17:32:12]   rc=0  1s  export_exit_best
[2026-11-07 17:32:12] 晋升：exit_p16.pt → exit_best.pt
[2026-11-07 17:32:12] R16 加速轮完成（PPO-only：无标注/无训练，含闸门 D/A）
[2026-11-07 17:32:12] P17 PPO 链起点 = exit_p16.pt  hands=1920 max_states=120000
[2026-11-07 17:32:12] $ /Users/ligf9/zhuoji-mahjong-ai/.venv/bin/python train_ppo.py --init /Users/ligf9/zhuoji-mahjong-ai/models/exit_p16.pt --out /Users/ligf9/zhuoji-mahjong-ai/models/exit_p17.pt --hands 1920 --workers 4 --threads 2 --max-states 120000 --epochs 2 --lr 5e-5 --seed0 1070000  → ppo_p17.log
[2026-11-07 17:42:47]   rc=0  635s  ppo_p17
[2026-11-07 17:42:47] P17 PPO 完成 init=exit_p16.pt → exit_p17.pt {"hands": 1920, "states": 94713, "A_std": 8.599404335021973, "ret_std": 12.501364508857955, "kl": [0.005254426160024957, 0.006975035843867305], "vl": 1.1336257104938094, "ent": 1.1682388313718743, "gnorm": 1.3790825356502792, "clipfrac": 0.07756083454634692, "ent_rollout": 1.3255188098502035, "kl_stop": false, "secs": 633.6}
[2026-11-07 17:42:47] $ /Users/ligf9/zhuoji-mahjong-ai/.venv/bin/python eval_pairs.py --a /Users/ligf9/zhuoji-mahjong-ai/models/exit_p17.pt --b /Users/ligf9/zhuoji-mahjong-ai/models/exit_best.pt --mode greedy --seeds 300 --seed0 710000 --workers 6 --out /Users/ligf9/zhuoji-mahjong-ai/runs/gates/p17_gateD.jsonl  → gateD_p17.log
[2026-11-07 17:45:20]   rc=0  153s  gateD_p17
[2026-11-07 17:45:20] P17 闸门 D(vs 现最佳): [summary] seats=[0, 1, 2, 3]  n=300  diff=+0.056±0.326  z=+0.17  p≈0.8647  符号 122:149  → 不通过
[2026-11-07 17:45:20] $ /Users/ligf9/zhuoji-mahjong-ai/.venv/bin/python eval_pairs.py --a /Users/ligf9/zhuoji-mahjong-ai/models/exit_p17.pt --b /Users/ligf9/zhuoji-mahjong-ai/models/dushan_master.pt --mode greedy --seeds 300 --seed0 710000 --workers 6 --out /Users/ligf9/zhuoji-mahjong-ai/runs/gates/p17_gateA.jsonl  → gateA_p17.log
[2026-11-07 17:46:25]   rc=0  66s  gateA_p17
[2026-11-07 17:46:25] P17 闸门 A(vs master): [summary] seats=[0, 1, 2, 3]  n=300  diff=+4.150±0.400  z=+10.37  p≈0.0000  符号 218:78
[2026-11-07 17:46:25] P17 不晋升（未显著优于现最佳），ckpt 留档供终评 best-of
[2026-11-07 17:46:25] R17 加速轮完成（PPO-only：无标注/无训练，含闸门 D/A）
[2026-11-07 17:46:25] P18 PPO 链起点 = exit_p17.pt  hands=1920 max_states=120000
[2026-11-07 17:46:25] $ /Users/ligf9/zhuoji-mahjong-ai/.venv/bin/python train_ppo.py --init /Users/ligf9/zhuoji-mahjong-ai/models/exit_p17.pt --out /Users/ligf9/zhuoji-mahjong-ai/models/exit_p18.pt --hands 1920 --workers 4 --threads 2 --max-states 120000 --epochs 2 --lr 5e-5 --seed0 1080000  → ppo_p18.log
[2026-11-07 17:56:00]   rc=0  575s  ppo_p18
[2026-11-07 17:56:00] P18 PPO 完成 init=exit_p17.pt → exit_p18.pt {"hands": 1920, "states": 93366, "A_std": 8.552409172058105, "ret_std": 12.28716725558282, "kl": [0.004889505054986691, 0.006819080331758277], "vl": 1.125518311784692, "ent": 1.1270752799021055, "gnorm": 1.3674750353375527, "clipfrac": 0.0729109824265111, "ent_rollout": 1.2808365332203713, "kl_stop": false, "secs": 573.6}
[2026-11-07 17:56:00] $ /Users/ligf9/zhuoji-mahjong-ai/.venv/bin/python eval_pairs.py --a /Users/ligf9/zhuoji-mahjong-ai/models/exit_p18.pt --b /Users/ligf9/zhuoji-mahjong-ai/models/exit_best.pt --mode greedy --seeds 300 --seed0 710000 --workers 6 --out /Users/ligf9/zhuoji-mahjong-ai/runs/gates/p18_gateD.jsonl  → gateD_p18.log
[2026-11-07 17:58:28]   rc=0  148s  gateD_p18
[2026-11-07 17:58:28] P18 闸门 D(vs 现最佳): [summary] seats=[0, 1, 2, 3]  n=300  diff=+0.391±0.342  z=+1.14  p≈0.2525  符号 154:134  → 不通过
[2026-11-07 17:58:28] $ /Users/ligf9/zhuoji-mahjong-ai/.venv/bin/python eval_pairs.py --a /Users/ligf9/zhuoji-mahjong-ai/models/exit_p18.pt --b /Users/ligf9/zhuoji-mahjong-ai/models/dushan_master.pt --mode greedy --seeds 300 --seed0 710000 --workers 6 --out /Users/ligf9/zhuoji-mahjong-ai/runs/gates/p18_gateA.jsonl  → gateA_p18.log
[2026-11-07 17:59:39]   rc=0  71s  gateA_p18
[2026-11-07 17:59:39] P18 闸门 A(vs master): [summary] seats=[0, 1, 2, 3]  n=300  diff=+4.932±0.418  z=+11.81  p≈0.0000  符号 226:73
[2026-11-07 17:59:39] P18 不晋升（未显著优于现最佳），ckpt 留档供终评 best-of
[2026-11-07 17:59:39] R18 加速轮完成（PPO-only：无标注/无训练，含闸门 D/A）
[2026-11-07 17:59:39] P19 PPO 链起点 = exit_p18.pt  hands=1920 max_states=120000
[2026-11-07 17:59:39] $ /Users/ligf9/zhuoji-mahjong-ai/.venv/bin/python train_ppo.py --init /Users/ligf9/zhuoji-mahjong-ai/models/exit_p18.pt --out /Users/ligf9/zhuoji-mahjong-ai/models/exit_p19.pt --hands 1920 --workers 4 --threads 2 --max-states 120000 --epochs 2 --lr 5e-5 --seed0 1090000  → ppo_p19.log
[2026-11-07 18:09:39]   rc=0  599s  ppo_p19
[2026-11-07 18:09:39] P19 PPO 完成 init=exit_p18.pt → exit_p19.pt {"hands": 1920, "states": 92614, "A_std": 8.443048477172852, "ret_std": 11.919478106583918, "kl": [0.005830282060244861, 0.007188522705849445], "vl": 1.0945906632512972, "ent": 1.0614353923837125, "gnorm": 1.3585495745444167, "clipfrac": 0.07547499990677306, "ent_rollout": 1.162902358985781, "kl_stop": false, "secs": 597.9}
[2026-11-07 18:09:39] $ /Users/ligf9/zhuoji-mahjong-ai/.venv/bin/python eval_pairs.py --a /Users/ligf9/zhuoji-mahjong-ai/models/exit_p19.pt --b /Users/ligf9/zhuoji-mahjong-ai/models/exit_best.pt --mode greedy --seeds 300 --seed0 710000 --workers 6 --out /Users/ligf9/zhuoji-mahjong-ai/runs/gates/p19_gateD.jsonl  → gateD_p19.log
[2026-11-07 18:11:56]   rc=0  137s  gateD_p19
[2026-11-07 18:11:56] P19 闸门 D(vs 现最佳): [summary] seats=[0, 1, 2, 3]  n=300  diff=+0.166±0.331  z=+0.50  p≈0.6166  符号 140:153  → 不通过
[2026-11-07 18:11:56] $ /Users/ligf9/zhuoji-mahjong-ai/.venv/bin/python eval_pairs.py --a /Users/ligf9/zhuoji-mahjong-ai/models/exit_p19.pt --b /Users/ligf9/zhuoji-mahjong-ai/models/dushan_master.pt --mode greedy --seeds 300 --seed0 710000 --workers 6 --out /Users/ligf9/zhuoji-mahjong-ai/runs/gates/p19_gateA.jsonl  → gateA_p19.log
[2026-11-07 18:12:57]   rc=0  61s  gateA_p19
[2026-11-07 18:12:57] P19 闸门 A(vs master): [summary] seats=[0, 1, 2, 3]  n=300  diff=+4.727±0.400  z=+11.82  p≈0.0000  符号 217:79
[2026-11-07 18:12:57] P19 不晋升（未显著优于现最佳），ckpt 留档供终评 best-of
[2026-11-07 18:12:57] R19 加速轮完成（PPO-only：无标注/无训练，含闸门 D/A）
[2026-11-07 18:12:57] P20 PPO 链起点 = exit_p19.pt  hands=1920 max_states=120000
[2026-11-07 18:12:57] $ /Users/ligf9/zhuoji-mahjong-ai/.venv/bin/python train_ppo.py --init /Users/ligf9/zhuoji-mahjong-ai/models/exit_p19.pt --out /Users/ligf9/zhuoji-mahjong-ai/models/exit_p20.pt --hands 1920 --workers 4 --threads 2 --max-states 120000 --epochs 2 --lr 5e-5 --seed0 1100000  → ppo_p20.log
[2026-11-07 18:22:12]   rc=0  554s  ppo_p20
[2026-11-07 18:22:12] P20 PPO 完成 init=exit_p19.pt → exit_p20.pt {"hands": 1920, "states": 90051, "A_std": 8.163689613342285, "ret_std": 11.671078035040294, "kl": [0.005374118705565872, 0.00727833327023704], "vl": 1.026660376821052, "ent": 1.0318067219447007, "gnorm": 1.3816995890811086, "clipfrac": 0.06955828189595857, "ent_rollout": 1.1261207413770224, "kl_stop": false, "secs": 553.1}
[2026-11-07 18:22:12] $ /Users/ligf9/zhuoji-mahjong-ai/.venv/bin/python eval_pairs.py --a /Users/ligf9/zhuoji-mahjong-ai/models/exit_p20.pt --b /Users/ligf9/zhuoji-mahjong-ai/models/exit_best.pt --mode greedy --seeds 300 --seed0 710000 --workers 6 --out /Users/ligf9/zhuoji-mahjong-ai/runs/gates/p20_gateD.jsonl  → gateD_p20.log
[2026-11-07 18:24:38]   rc=0  147s  gateD_p20
[2026-11-07 18:24:38] P20 闸门 D(vs 现最佳): [summary] seats=[0, 1, 2, 3]  n=300  diff=+0.212±0.330  z=+0.64  p≈0.5198  符号 155:135  → 不通过
[2026-11-07 18:24:38] $ /Users/ligf9/zhuoji-mahjong-ai/.venv/bin/python eval_pairs.py --a /Users/ligf9/zhuoji-mahjong-ai/models/exit_p20.pt --b /Users/ligf9/zhuoji-mahjong-ai/models/dushan_master.pt --mode greedy --seeds 300 --seed0 710000 --workers 6 --out /Users/ligf9/zhuoji-mahjong-ai/runs/gates/p20_gateA.jsonl  → gateA_p20.log
[2026-11-07 18:25:41]   rc=0  62s  gateA_p20
[2026-11-07 18:25:41] P20 闸门 A(vs master): [summary] seats=[0, 1, 2, 3]  n=300  diff=+4.962±0.413  z=+12.02  p≈0.0000  符号 224:73
[2026-11-07 18:25:41] P20 不晋升（未显著优于现最佳），ckpt 留档供终评 best-of
[2026-11-07 18:25:41] R20 加速轮完成（PPO-only：无标注/无训练，含闸门 D/A）
[2026-11-07 18:25:41] P21 PPO 链起点 = exit_p20.pt  hands=1920 max_states=120000
[2026-11-07 18:25:41] $ /Users/ligf9/zhuoji-mahjong-ai/.venv/bin/python train_ppo.py --init /Users/ligf9/zhuoji-mahjong-ai/models/exit_p20.pt --out /Users/ligf9/zhuoji-mahjong-ai/models/exit_p21.pt --hands 1920 --workers 4 --threads 2 --max-states 120000 --epochs 2 --lr 5e-5 --seed0 1110000  → ppo_p21.log
[2026-11-07 18:30:05]   rc=1  264s  ppo_p21
[2026-11-07 18:30:05] P21 PPO 训练失败 rc=1（主线不受影响）
[2026-11-07 18:30:05] R21 加速轮 PPO 无产出 → 回落到 DAgger 路径
[2026-11-07 18:30:05] $ /Users/ligf9/zhuoji-mahjong-ai/.venv/bin/python -c import torch,sys;sys.path.insert(0,'.');from zhuoji.net import build_from_ckpt;from zhuoji.forward import ensure_onnx;ck=torch.load('/Users/ligf9/zhuoji-mahjong-ai/models/exit_best.pt',map_location='cpu',weights_only=False);net,_=build_from_ckpt(ck);p=ensure_onnx(net,stem='exit_best');print('onnx:',p)  → export_exit_best.log
[2026-11-07 18:30:06]   rc=0  1s  export_exit_best
[2026-11-07 18:30:06] $ /Users/ligf9/zhuoji-mahjong-ai/rust_engine/target/release/annotate --student /Users/ligf9/zhuoji-mahjong-ai/models/onnx/exit_best.onnx --teacher models/onnx/dushan_master.onnx --student-enc 3 --student-device cpu --targets 100000 --threads 4 --out /Users/ligf9/zhuoji-mahjong-ai/runs/exit_r21 --seed0 810000  → annotate_r21.log

## 自检 #31（2026-11-07 19:05，7×24 第 31 次 · 加速模式第一周期结算 + 硬决策点）

**结论：健康、主线零干预、产出「持平但不达预期」。** 唯一驱动 PID 74896（PPID=99107 keepalive，
16:35:11 起 2h30m），lock 1~18s；单实例（1 driver + 1 annotate 85665），无 switch_* 残留；
02:33:57 起 panic/NaN/超时 0；磁盘 305Gi；`health_watch --last 12` 全绿；Web 8770 `/`+`/api/meta` 双 200。

**硬决策点（P23−R12 → P20−R12，链在 P21 断，只做这一次）**：
`+0.862±0.554  z=+1.56  p≈0.1194  符号 156:139` ⇒ **|Δ|<1.5 且不显著 ⇒ 预登记分支 ②：
第二周期同配方继续，一个参数都不改。** 不加 1920 局预算，不写 `ppo_disabled`。

**R13~R20 八轮**：闸门 A +4.032~+4.962 全 p≈0（与 R12 的 +4.100 同级，无退化也无提升）；
闸门 D 8 次仅 P16 显著（+1.073±0.410 p=.0088）⇒ 17:32 晋升 `exit_best=exit_p16`(md5 21c15f5b)。
**按多重比较尺子：8 次取一，Bonferroni 阈值 .00625，p=.0088 未过 ⇒ 族系不显著**；后续
p17/p18/p19/p20 对 p16 的 D 全 ns（+0.056/+0.391/+0.166/+0.212）⇒ 无「晋升后更好」的延续。

**事故（已修复）**：P21 `rc=1` 静默失败，根因 = 环境 safe-delete 批量守卫（50 次/turn）
拦截 `os.unlink` 的分片清理，崩在 rollout 之前 ⇒ 无脏样本、无 `exit_p21.pt`。
并行会话 18:55 已改为 `runs/ppo_shards/shard_<wid>.npz` 固定名覆盖写、全链路零删除，
`night_run.py` 两处同类删除同步改掉；`py_compile` OK + 18:56 两 worker 实测写出分片 ⇒ R22 生效。
代价：R21 由 13min 加速轮退化为 3h15m 完整 DAgger 轮（净损 ≈2h），但因此获得一次**以 p16 为学生的
重锚**（PPO 策略 → 教师标签蒸馏），`r21_gateD`(vs p16) 将成为「增益能否被蒸馏保住」的免费检验。

**新证据（PPO 饱和）**：`ent_rollout` 1.820→1.126（−38%）、决策熵 1.479→1.032（−30%）单调降、
跌幅逐轮收窄；vl 1.199→1.027 单调降；而 A 停在 +4.0~+4.9 ⇒ **策略在锐化，不是变强**。

**产品侧（用户口径，本自动化不再推动）**：主力恒为 `dushan_master`，`web_best`(R07) 改为
`champions.json` 附加冠军；`/api/meta` 实测 `[dushan_master, web_best]` 双 200。覆盖 #18。

**下一步**：R21 → 训练 ≈21:41 → A/C/D ≈22:12 → R22/R23（PPO 13min）→ R24 完整 DAgger ≈02:00。
预登记一次性测量（计划 §6.36.7）：CPU 空闲窗口跑 `P20 vs R12` 全新池 730000+600，判读三分支锁死。

**[19:0x 用户问「Python 是否耗时主力 / 能否 Rust 加速」——只读剖析，配方零改动]**
加速轮 P20 762s = rollout 264s + MPS 更新 288s + 闸门 D 147s + A 62s；cProfile：
评测 greedy 40 局 torch 前向 93%（Python 规则 4.6% / 编码 7.1%），PPO rollout
M1 前向 76%（编码 7%），gae 9 万状态 0.42s。前向基准：torch CPU B=1 **7.05ms**，
Rust `Infer` 同款 ORT CPU B=1 10.6~13.1ms、CoreML B=1 122ms ⇒ 换 Rust+ORT 更慢；
Python 全下沉也只省一轮 ≤7%。**结论：Python 非主力、Rust 已占住所有重活，
本轮不新增 Rust 重写**；真瓶颈是「每决策 B=1 前向」，MPS 锁步批量/剪枝会动冻结
协议，7×24 窗口内不动。完整数据 `docs/python_rust_usage_audit.md` §6、计划 §6.37。
同批：线上主力切回 master（冠军只增不替）+ PPO 零删除修复，见 §6.36/#31。

**[20:2x 第二份外部专家建议（GRP/Oracle/AWR/ACH/课程+对手池）——主链零改动，开影子轨道]**
用户要求「7×24 内合适时机做实验，目标训练能力超强的模型」。此时 §6.36 已判
「同配方第二周期，一参数不改」⇒ 直接落地这些建议会毁掉第二周期可比性。
处置：**主链一字不改**（train_ppo / night_run / 超参 / 闸门判据全部未动），
开完全隔离的影子实验轨道，判据看结果前锁死，产物只留档、只增不替。研判见计划 §6.38。

**本轮实际完成的两个实验（零对局、零 CPU 压力，可与标注并发）**：
1. `tools/exploitability_diag.py` — ACH 可利用度诊断，p13~p20 两门 19,200 条逐 seed
   记录（同 300 seed 池、四座、零和）。J1 原式零假设误设（把基线差当座位偏差）已
   作废留档，收紧为 J1′ 逐 seed 相对偏差。**J1′ 无座位弱点**（A 最大 +1.419±1.085
   p=.19）；**J2 无风格敏感**（波动比 1.009，p=.80）；J3 名义 slope +0.1047±0.0508
   分/轮 p=.039、同 seed r=0.370<0.70 ⇒ 按判据「仍在补强」但仅探索性辅证，
   不改配方、不宣布未饱和。D 门（同链对手）放炮率 31.5% vs A 门 20.4%、胜率
   26.2% vs 37.1% ⇒ **自对弈对手明显更强**，「只学会赢自己/对手池坍缩」证据不足，
   维持 §6.35.2④ 驳回。产物 `runs/shadow/exploitability.json`。
2. `tools/grp_probe.py` — G1 前置探针。R12 4000 状态上 PPO 链跨 8 轮
   `mean|ΔV|×8 = 1.9216 分`（阈值 0.50，p90 4.34、r 0.9448）⇒ **V_θ 轮间漂移可观，
   E1 必须跑**；冻结源 r12 与 p13/p20 `r≈0.70` 却差 3.9~4.2 分 ⇒ 确是不同信号；
   前向一致性 max|重算−落盘|=0.000378。注意这不推翻 §6.35.2②「rollout 内冻结」，
   量的是轮间。产物 `runs/shadow/grp_probe.json`。

**研判**：优先级一 GRP **采纳为影子实验**；优先级二 Oracle 价值蒸馏、三 AWR 预热、
课程学习、历史对手池 **驳回本轮**（改配方，与分支②冲突），登记第三周期后方向；
优先级四 ACH **采纳已完成**。Suphx 的 GRU 跨局摘要不适用（本项目每局独立结算），
项目化最小实现 = `Φ_GRP=8·V_frozen(exit_r12)`，零训练成本、**批量前向零 rollout 开销**
（刻意绕开 §6.37 的 B=1 瓶颈；Rust 路线已证伪，不新增重写）。

**待执行（只做一次，预登记）**：`tools/shadow_lab.py --e1` 严格配对对照
（同 init=p20 / 同 320 局 rollout 池 1_100_000 / 同超参，唯一差异 Φ 源
A=8·V_θ vs B=8·V_frozen），两组各跑冻结闸门 A 逐 seed 配对；
判据 Δ≥+1.5&p<.05 有效 / ns 无证据（低功效，不等于已证无效）/ Δ≤−1.5&p<.05 有害。
`--dry` 已验全管道（669 状态、Φ 差 2.859 分、A 相关 0.9375、ppo_update 通）。
**时机**：R21 train 段已被 §6.36.7 的 600-seed 测量独占，R22/R23 加速轮占满 ⇒
第一个合法窗口是 **R24 完整 DAgger 的 train 段 ≈11-08 02:4x**，由每小时自检触发，
不挂长驻进程。守卫四场景实测通过（annotate 在跑 / 最新已 done / 陈旧残留 均拒绝，
R24 活跃段放行）；修掉 `train_r01.log`（45h 未配对 .done）会被误判成活跃窗口的隐患。
自动化 prompt 已同步（§6.38 必读、E1 状态输出、勿重复跑诊断/探针）。

**状态**：20:39 驱动 74896 / 标注 85665（≈23.3k/25k，吞吐 11.1k/h·w）/ Web 88725 /
看门狗 99107 均存活，lock 20:36，8770 `/` 200；本轮未启动任何吃 CPU 的实验。
[2026-11-07 20:45:58]   rc=0  8152s  annotate_r21
[2026-11-07 20:45:58] R21 标注完成 student=exit_best.onnx seed0=810000 → 4×25k
[2026-11-07 20:45:58] $ /Users/ligf9/zhuoji-mahjong-ai/.venv/bin/python train_exit.py --data runs/exit_r01/exit_w*.npz runs/exit_r02/exit_w*.npz runs/exit_r03/exit_w*.npz runs/exit_r04/exit_w*.npz runs/exit_r05/exit_w*.npz runs/exit_r06/exit_w*.npz runs/exit_r07/exit_w*.npz runs/exit_r08/exit_w*.npz runs/exit_r09/exit_w*.npz runs/exit_r10/exit_w*.npz runs/exit_r11/exit_w*.npz runs/exit_r12/exit_w*.npz runs/exit_r21/exit_w*.npz --out /Users/ligf9/zhuoji-mahjong-ai/models/exit_r21.pt --epochs 30 --window 400000 --batch 256 --seed 20261127 --init /Users/ligf9/zhuoji-mahjong-ai/models/exit_best.pt  → train_r21.log

## 自检 #32（2026-11-07 21:0x，7×24 第 32 次 · 第三周期预算决策：全新池 600-seed 首测为正）

**结论：健康、产出符合预期、主循环零干预。** 唯一驱动 PID 74896（PPID=99107 keepalive，16:35 起），
lock 20:03:22（<120s）；单实例 1 driver + 1 train_exit(97630)，无 switch_* 残留；
02:33:57 起 panic/NaN/超时 0；磁盘 305Gi；`health_watch --last 12` **全绿**；
Web 8770 `/`+`/api/meta` 双 200，master=`dushan_master`、冠军表仅 `web_best`(R07 快照)；看门狗 99107 活。

**R21 完整 DAgger 轮**（P21 PPO 事故回落，非配方问题）：标注 20:45:58 rc=0（8152s、11.0k/h·worker、
4×25k、`check_npz r21` **4/4 合规**、r01~r21 累计 48/48），学生=exit_best.onnx(=p16)、seed0=810000；
20:45:58 驱动自动接续训练（400k 窗口、`--init exit_best.pt`、seed 20261127），ep01 train 1.7480/val 1.7455，
ETA ≈21:28 → A ≈21:29 → C ≈21:52 → D ≈21:54 → R22（加速 13min）。

**【本轮核心】§6.36.7 唯一剩余预登记测量已执行并首次为正**：窗口=R21 train 段，
池 **730000+600 全新**（确认从未使用），双侧 greedy 四座，2400 局，744s，6 workers，与 annotate 零重叠。
**`P20 − R12 = +0.632 ± 0.271，z=+2.33，p=0.0199`（308/6/286）**；bootstrap CI [+0.09,+1.15]、
P(>0)=0.991；trim10/20% +0.51/+0.42（非离群驱动）；逐座 +0.06/+0.32/+0.98/+1.17 **四座同号**。
⇒ 全项目第一次在**未使用过的池**上取得 PPO 链累积增益**显著为正**（≈+0.63 分/手，约为对 master
边际 +4.1~+5.0 的 13~15%）。

**决策（预登记三分支的字面落在中间支，但前提被同一数据证伪 ⇒ 判「维持」）**：
预登记原文 `|Δ|<1.0 ⇒ 1920→960 且 EVERY 12→8`；`Δ=+0.632` 字面命中，但该支的前提是
「PPO 无净收益」，而实测符号为正且 p=0.0199。**决定：维持 `RAPID_HANDS=1920` / `DAAGGER_EVERY=12`，
不重启驱动**。理由：①中间支两条动作都降低期望强度（960 砍掉唯一正边际臂；EVERY 8 把墙钟进一步压向
零边际的 DAgger，轮速 28.2→35.8min/轮）；②落地需重启（历史两次事故），而两分支交付差异在闸门 A
本就不可分辨（P20 +4.96 vs R12 +4.10 ns）；③**自缚条款**：偏差只许向第一支靠拢，禁止借机加预算/
改任何超参/放宽任何闸门；④与 §6.36.2 的 `+0.862±0.554 p=.1194` 并列报告，不翻案、不择一引用。

**新增（防误判）**：`health_watch` 全绿但 `ent_rollout` 1.820→1.126（−38%）、决策熵 1.479→1.032、
states/轮 106698→**90051**（−15.6%，同 hands 下对局变短=更果断）——策略在锐化不是变强。
登记：`ent_rollout` 跌破 **0.8** 时探索实质停止（按当前斜率 ≈9 轮），届时仅作"PPO 信息源枯竭"
的判读依据，**不据此改任何参数**。

**顺带修正终评口径**：`--a` 必须取 best-of 第一段**胜者**（`tools/bestof_rank.py` #1），不是
`exit_best`（现为 p16 +4.19，而 P20 +4.96 且本轮拿到独立池正效应）。已给
`tools/final_eval_precheck.py` 加 `--model` 与 `[!]` 警示。23 轮 A 均值 top1=P20 此前只能判
"全并列（P20−P10 p=.014、16 次比较族系 ns）"，现在 P20 对 R12 有独立池证据 ⇒ "top1 不是纯噪声最大值"
第一次获外部支持；**仍不得提前宣布 P20 更强**，第二段 1000-seed 复赛照旧。

**复现测量预登记**：`740000+600` 全新池同规格，排 **R36 train 段**（≈11-08 06:2x），
**不与 E1 同窗**（E1 已排 R24，两者都吃 CPU）；同号且 p<0.10 ⇒ 复现成功；反号 ⇒ 只写不可分辨，不改主线。
时间线修正：R21 是完整 DAgger 轮 ⇒ R22/R23 加速 13min、**R24 才是下一个完整 DAgger（train 段 ≈00:4x~01:2x）**，
§6.38.5 的"≈02:4x"作废。

**下一步**：R21 训练 → 闸门 A/C/D（`r21_gateD` vs p16 = 「PPO 增益能否被蒸馏保住」的免费检验，**务必捕获**）
→ R22（PPO 链起点 = exit_best，若 r21 晋升则 = exit_r21）→ R23 → R24 完整 DAgger。
仍不动任何超参、不放宽闸门、`web_best` 不跟 PPO、`dushan_master` 不覆盖。

**[21:0x 追加 · 结构性发现 §6.40]** PPO 链增益在 DAgger 重锚处被丢弃：`ppo_init` 续 `exit_p{r−1}`，
但 12 轮一次的完整 DAgger 轮 `--init` 恒为 `exit_best.pt`，而 `exit_best` 只由闸门 A 决定
（PPO 轮需 D+A 双过，8 次 D 按族系尺子一次都不显著）⇒ R21 的 init 是 p16，p17~p20 的增益
（末端 p20 = +0.632，§6.39.2）既没进 exit_best 也没进下一周期。**后果：每周期 +0.6 不累积。**
判定依据 = `r21_gateD`(≈21:54) + `r24_gateD`(≈01:2x) 同号；判定点 = **R36 边界（≈11-08 06:2x，
第三周期合法配方节点）**，方案 B（DAgger `--init` 改为本周期链末端，确定性规则）通过才执行，
不在 R24 硬上、不改其他参数、`exit_best` 语义不变。
[2026-11-07 21:21:17]   rc=0  2119s  train_r21
[2026-11-07 21:21:17] R21 训练完成 init=exit_best.pt → exit_r21.pt  (['[done] best val 1.7455 → /Users/ligf9/zhuoji-mahjong-ai/models/exit_r21.pt'])
[2026-11-07 21:21:17] $ /Users/ligf9/zhuoji-mahjong-ai/.venv/bin/python eval_pairs.py --a /Users/ligf9/zhuoji-mahjong-ai/models/exit_r21.pt --b /Users/ligf9/zhuoji-mahjong-ai/models/dushan_master.pt --mode greedy --seeds 300 --seed0 710000 --workers 6 --out /Users/ligf9/zhuoji-mahjong-ai/runs/gates/r21_gateA.jsonl  → gateA_r21.log
[2026-11-07 21:22:12]   rc=0  54s  gateA_r21
[2026-11-07 21:22:12] R21 闸门 A: [summary] seats=[0, 1, 2, 3]  n=300  diff=+3.717±0.420  z=+8.86  p≈0.0000  符号 201:96  → 通过
[2026-11-07 21:22:12] $ /Users/ligf9/zhuoji-mahjong-ai/.venv/bin/python -c import torch,sys;sys.path.insert(0,'.');from zhuoji.net import build_from_ckpt;from zhuoji.forward import ensure_onnx;ck=torch.load('/Users/ligf9/zhuoji-mahjong-ai/models/exit_best.pt',map_location='cpu',weights_only=False);net,_=build_from_ckpt(ck);p=ensure_onnx(net,stem='exit_best');print('onnx:',p)  → export_exit_best.log
[2026-11-07 21:22:13]   rc=0  1s  export_exit_best
[2026-11-07 21:22:13] 晋升：exit_r21.pt → exit_best.pt
[2026-11-07 21:22:13] $ /Users/ligf9/zhuoji-mahjong-ai/.venv/bin/python eval_pairs.py --a /Users/ligf9/zhuoji-mahjong-ai/models/exit_r21.pt --b /Users/ligf9/zhuoji-mahjong-ai/models/dushan_master.pt --a-mode greedy --b-mode search32 --seeds 50 --seed0 720000 --workers 6 --out /Users/ligf9/zhuoji-mahjong-ai/runs/gates/r21_gateC.jsonl  → gateC_r21.log
[2026-11-07 21:44:06]   rc=0  1313s  gateC_r21
[2026-11-07 21:44:06] R21 闸门 C(趋势): [summary] seats=[0, 1, 2, 3]  n=50  diff=+4.047±0.937  z=+4.32  p≈0.0000  符号 36:12
[2026-11-07 21:44:06] $ /Users/ligf9/zhuoji-mahjong-ai/.venv/bin/python eval_pairs.py --a /Users/ligf9/zhuoji-mahjong-ai/models/exit_r21.pt --b /Users/ligf9/zhuoji-mahjong-ai/runs/gates/r21_prevbest.pt --mode greedy --seeds 300 --seed0 710000 --workers 6 --out /Users/ligf9/zhuoji-mahjong-ai/runs/gates/r21_gateD.jsonl  → gateD_r21.log
[2026-11-07 21:46:12]   rc=0  127s  gateD_r21
[2026-11-07 21:46:12] R21 闸门 D(vs 上轮最佳): [summary] seats=[0, 1, 2, 3]  n=300  diff=-0.464±0.397  z=-1.17  p≈0.2419  符号 138:161
[2026-11-07 21:48:23] ============================================================
[2026-11-07 21:48:23] 7×24 ExIt 驱动启动：R22 起，截止 2026-11-13 02:30（R1 已完成；断点续跑 + 每小时自检自动化）
[2026-11-07 21:48:23] P22 PPO 链起点 = exit_best.pt  hands=1920 max_states=120000
[2026-11-07 21:48:23] $ /Users/ligf9/zhuoji-mahjong-ai/.venv/bin/python train_ppo.py --init /Users/ligf9/zhuoji-mahjong-ai/models/exit_best.pt --out /Users/ligf9/zhuoji-mahjong-ai/models/exit_p22.pt --hands 1920 --workers 4 --threads 2 --max-states 120000 --epochs 2 --lr 5e-5 --seed0 1120000  → ppo_p22.log
[2026-11-07 22:00:08]   rc=0  705s  ppo_p22
[2026-11-07 22:00:08] P22 PPO 完成 init=exit_best.pt → exit_p22.pt {"hands": 1920, "states": 120000, "A_std": 7.898251533508301, "ret_std": 11.62375665393938, "kl": [0.013579710607113106, 0.014300375017272963], "vl": 0.8187045511215735, "ent": 1.8249168565024192, "gnorm": 1.2154860830764527, "clipfrac": 0.157571517415583, "ent_rollout": 2.17864698625369, "kl_stop": false, "secs": 703.6}
[2026-11-07 22:00:08] $ /Users/ligf9/zhuoji-mahjong-ai/.venv/bin/python eval_pairs.py --a /Users/ligf9/zhuoji-mahjong-ai/models/exit_p22.pt --b /Users/ligf9/zhuoji-mahjong-ai/models/exit_best.pt --mode greedy --seeds 300 --seed0 710000 --workers 6 --out /Users/ligf9/zhuoji-mahjong-ai/runs/gates/p22_gateD.jsonl  → gateD_p22.log
[2026-11-07 22:02:24]   rc=0  136s  gateD_p22
[2026-11-07 22:02:24] P22 闸门 D(vs 现最佳): [summary] seats=[0, 1, 2, 3]  n=300  diff=-0.024±0.350  z=-0.07  p≈0.9442  符号 138:155  → 不通过
[2026-11-07 22:02:24] $ /Users/ligf9/zhuoji-mahjong-ai/.venv/bin/python eval_pairs.py --a /Users/ligf9/zhuoji-mahjong-ai/models/exit_p22.pt --b /Users/ligf9/zhuoji-mahjong-ai/models/dushan_master.pt --mode greedy --seeds 300 --seed0 710000 --workers 6 --out /Users/ligf9/zhuoji-mahjong-ai/runs/gates/p22_gateA.jsonl  → gateA_p22.log
[2026-11-07 22:03:26]   rc=0  62s  gateA_p22
[2026-11-07 22:03:26] P22 闸门 A(vs master): [summary] seats=[0, 1, 2, 3]  n=300  diff=+4.169±0.439  z=+9.50  p≈0.0000  符号 204:91
[2026-11-07 22:03:26] P22 不晋升（未显著优于现最佳），ckpt 留档供终评 best-of
[2026-11-07 22:03:26] R22 加速轮完成（PPO-only：无标注/无训练，含闸门 D/A）
[2026-11-07 22:03:26] P23 PPO 链起点 = exit_p22.pt  hands=1920 max_states=120000
[2026-11-07 22:03:26] $ /Users/ligf9/zhuoji-mahjong-ai/.venv/bin/python train_ppo.py --init /Users/ligf9/zhuoji-mahjong-ai/models/exit_p22.pt --out /Users/ligf9/zhuoji-mahjong-ai/models/exit_p23.pt --hands 1920 --workers 4 --threads 2 --max-states 120000 --epochs 2 --lr 5e-5 --seed0 1130000  → ppo_p23.log

## 自检 #33（2026-11-07 22:0x，7×24 第 33 次 · R21 结算 + r21_gateD 捕获 + P22 双闸门）

**状态：健康、零干预。** 驱动 PID 3078（PPID=99107 看门狗；21:48:23 拉起），lock mtime 22:02（<120s）；
单实例（1 driver + 1 train_ppo）；阶段 = **R23 加速轮**（22:03:26 起，PPO init=exit_p22，链连续）；
panic/NaN/超时 0；npz 52/52（r21 4×25k，tiles(25000,63,27)/glob(25000,130)/q(25000,28)，NaN 仅非法槽位）；
磁盘 305Gi；`health_watch --last 12` 全绿；Web 8770 `/`+`/api/meta` 双 200，master=dushan_master、
models 首位 champion=false、champions.json 仍只含 web_best；看门狗 99107 活；TAU=1.0 / PPO_HANDS=960 未改。

**R21 结算**：train 2119s（best val **1.7455**，项目最低）→ 闸门 A **+3.717±0.420 p<1e-4**（201:96）通过
→ 晋升 `exit_best ≡ exit_r21`（md5 185133f0）→ 闸门 C **+4.047±0.937 z=+4.32 p≈0.0000**（36:12）
→ 闸门 D(vs p16) **−0.464±0.397 p=0.2419**。

**【本轮最重要】`r21_gateD` 已捕获，但它不能回答 §6.40.3 让它回答的问题。** r21 的 `--init`=p16，
且标注数据由 p16 策略生成；p17~p20 的增益**从未进入 r21 的初始化或数据流**。故 r21 vs p16 测的是
「DAgger 轮自身边际」（≈0，第 9 个零边际轮），**不是**「PPO 增益能否被蒸馏保住」。同一结构缺陷对 r24
同样成立（r24 init=exit_r21，而链末端是 p23）。⇒ **§6.40.3 第一行读法作废（结构性无效，非数据驱动改判），
`r21_gateD=−0.464` 不得当作「方案 B 不可行」的证据。** 替代预登记见计划 §6.41。

**P22**：init=exit_best(≡exit_r21)、1920 局 / 120,000 状态 / 703.6s，kl[0.01358,0.01430]、clipfrac 0.1576、
gnorm 1.215、ent 1.825、ent_rollout 2.179、vl 0.819 —— 重锚后熵由 1.032 回升到 1.825 属预期（新 init 分布），
非异常；闸门 D **−0.024±0.350 p=0.9442 不通过**、闸门 A +4.169±0.439 p<1e-4 通过 ⇒ **D 未过不晋升**，
`exit_best` 仍 = exit_r21。

**看门狗拉起 21:48:23**：发生在 gateD_r21 完成（21:46:12）之后、R22 启动之前，**零数据损失**
（R21 三件套齐全、exit_best 未被覆盖、P22 正确以 exit_best 为起点）。本日第 4 次同类（05:59/06:22/16:35/21:48），
模式 = 长闸门（C 1313s）后锁心跳滞后，**已知良性，不改看门狗参数**。

**E1 影子实验：未执行**（`runs/shadow/e1.done` 与结果文件均不存在；20:36/20:39 两次 rc=2 跳过都是 annotate
在跑）。下一合法窗口 = **R24 train 段（预计 00:4x~01:1x）**。守卫 `MAX_TRAIN_LOG_AGE=7200s` 但 `.done`
一出现即永久失效 ⇒ **必须在 train 段内触发**；本自动化下一轮巡检若落在 01:0x 即在窗口内。

**下一步**：R23（≈22:16 结束）→ R24 完整 DAgger（annotate ≈22:19 起、≈2h15m）→ train 段跑 E1
→ 捕获 r24_gateD → R25。仍不动任何超参、不重启、不放宽闸门；**不注册新冠军**（闸门 C 单轮值按 #28
永久规则不得单独作证据，产品侧维持 dushan_master + 只增不替）。

[2026-11-07 22:15:04]   rc=0  698s  ppo_p23
[2026-11-07 22:15:04] P23 PPO 完成 init=exit_p22.pt → exit_p23.pt {"hands": 1920, "states": 117755, "A_std": 8.213330268859863, "ret_std": 12.683948599838038, "kl": [0.008764655118790435, 0.011699697460078269], "vl": 1.0038117145066676, "ent": 1.708357535497002, "gnorm": 1.299837759266729, "clipfrac": 0.10745582548172578, "ent_rollout": 2.1017382752753697, "kl_stop": false, "secs": 696.7}
[2026-11-07 22:15:04] $ /Users/ligf9/zhuoji-mahjong-ai/.venv/bin/python eval_pairs.py --a /Users/ligf9/zhuoji-mahjong-ai/models/exit_p23.pt --b /Users/ligf9/zhuoji-mahjong-ai/models/exit_best.pt --mode greedy --seeds 300 --seed0 710000 --workers 6 --out /Users/ligf9/zhuoji-mahjong-ai/runs/gates/p23_gateD.jsonl  → gateD_p23.log
[2026-11-07 22:17:33]   rc=0  149s  gateD_p23
[2026-11-07 22:17:33] P23 闸门 D(vs 现最佳): [summary] seats=[0, 1, 2, 3]  n=300  diff=-0.122±0.360  z=-0.34  p≈0.7341  符号 134:161  → 不通过
[2026-11-07 22:17:33] $ /Users/ligf9/zhuoji-mahjong-ai/.venv/bin/python eval_pairs.py --a /Users/ligf9/zhuoji-mahjong-ai/models/exit_p23.pt --b /Users/ligf9/zhuoji-mahjong-ai/models/dushan_master.pt --mode greedy --seeds 300 --seed0 710000 --workers 6 --out /Users/ligf9/zhuoji-mahjong-ai/runs/gates/p23_gateA.jsonl  → gateA_p23.log
[2026-11-07 22:18:41]   rc=0  68s  gateA_p23
[2026-11-07 22:18:41] P23 闸门 A(vs master): [summary] seats=[0, 1, 2, 3]  n=300  diff=+4.393±0.418  z=+10.52  p≈0.0000  符号 211:87
[2026-11-07 22:18:41] P23 不晋升（未显著优于现最佳），ckpt 留档供终评 best-of
[2026-11-07 22:18:41] R23 加速轮完成（PPO-only：无标注/无训练，含闸门 D/A）
[2026-11-07 22:18:41] $ /Users/ligf9/zhuoji-mahjong-ai/.venv/bin/python -c import torch,sys;sys.path.insert(0,'.');from zhuoji.net import build_from_ckpt;from zhuoji.forward import ensure_onnx;ck=torch.load('/Users/ligf9/zhuoji-mahjong-ai/models/exit_best.pt',map_location='cpu',weights_only=False);net,_=build_from_ckpt(ck);p=ensure_onnx(net,stem='exit_best');print('onnx:',p)  → export_exit_best.log
[2026-11-07 22:18:42]   rc=0  1s  export_exit_best
[2026-11-07 22:18:42] $ /Users/ligf9/zhuoji-mahjong-ai/rust_engine/target/release/annotate --student /Users/ligf9/zhuoji-mahjong-ai/models/onnx/exit_best.onnx --teacher models/onnx/dushan_master.onnx --student-enc 3 --student-device cpu --targets 100000 --threads 4 --out /Users/ligf9/zhuoji-mahjong-ai/runs/exit_r24 --seed0 840000  → annotate_r24.log


**[22:2x 追加 · 用户指令：周期映射 + 提前终评（§6.42 / §6.43）]**
① 用户要求补「§6.41 结构周期 ↔ 决策周期映射表」；§6.41 已被并发自检 #33 占用
   （免费检验失效 + R36 替代预登记），故**顺延为 §6.42**，内容按原要求：S/D 双编号、
   S1~S22 映射表、三处历史编号冲突的读法（§6.40.4 的「第二周期 R13~R23」轮次写错，
   读作「D2 期间配方未改」）。D3 = R36 边界 ≈11-08 06:2x（S3=R37~R48）。纯文档。
② 用户授权「不要求强行用完所有时间，一旦发现训练循环无法提升性能即可终评」⇒
   DEADLINE 11-13 02:30 降级为**最晚终点**；§6.43 把「发现无法提升」写成看结果前锁死的
   预登记判据（S 周期边界 r=36/48/60…，独立池 750000+600k 头对头主判据 M：
   Δ≤+0.50 且 p≥0.10 ⇒ 无提升 / Δ≥+1.0 且 p<.05 ⇒ 继续 / 中间带证据不足继续；
   辅证 S1 闸门 A 无趋势、S2 ent<0.80 或持平、S3 重锚不超 1SE、S4 链健康，四条全过才停；
   最早 R36，方案 B 生效则推迟 R48）。ns = 「边际增益低于检测分辨率」，不是「已证明到顶」。
   用户手工通道（--manual）保留 owner 覆盖权。
落地机制（只改停止条件，配方/超参/闸门一字未动）：tools/early_stop_guard.py（只读判读 +
   --arm 一次性写 runs/early_stop，零删除、不跑对局）；night_run.py 轮首防御检查；
   tools/keepalive.py 见标记不再拉起驱动但 Web 守护继续（22:20 安全重启 PID 6189）；
   终评手册 §0.1 早停路径（全量执行，跳过 §10.2、保留 §10.3）。
验证：py_compile 全过；守卫 dry 实测 not_ready rc=2 且拒绝 --arm、无标记；mock 独立池
   （750600+600 全零）M 计算 Δ=0/p=1/fresh=True；manual 通道重定向到 /tmp 验证写标记逻辑；
   boundary_ok 对 R24/R25/R36 均正确拒绝。Web 双 200、驱动 3078、lock 正常、R24 标注在跑。
**更正（§6.41.2）**：r21_gateD(−0.464 ns) 的「免费检验」构造性失效，量的是 DAgger 轮自身
   边际，不得再用作「方案 B 不可行」的证据；本会话上一条回答里的该读法作废。

**[22:3x 追加 · 用户「CPU<20% / GPU·ANE<30% 是否没跑满」严格评审]** 纯只读，未改主链一行，
   未启动任何吃 CPU 探针（ps/top 被沙箱拒 ⇒ 改用 Mach host_statistics + proc_pid_rusage）。
   结论：**账面确有空闲算力，但不是「参数没调满」，加并行度会被实测证伪**。
   硬件 Apple M5 10 核（4P+6E）/32GB；四次 3s 采样全局 busy 10.6/15.5/10.7/15.1%
   = 1.06~1.55 核，用户观察属实；按 4 个 P 核口径为 26~39%。load avg 1.98 交叉印证。
   当前 R24 标注（`annotate --threads 4`）：annotate 进程仅 **0.58 核等效 CPU 时间**，
   4 线程加速比 1.34×（每线程效率 33%），≈2/3 墙钟在等 ANE；吞吐 9.94k/h·w ×4
   = 39.8k/h，已达历史 4 线程上限 45k/h 的 88%，而历史实测 t2≈t4>t6（ANE 共享是上限）。
   三层原因：① 教师每批 B≤24（pad 到 LADDER 32）小图 + 学生 B=1 走 CPU（刻意避开
   CoreML 65 分区崩 5×）⇒ ANE 占空比低是 kernel 启动/同步气泡，不是「还有 70% 可喂」；
   ② 4 线程私有 ORT Session 抢同一 ANE；③ MPS 只在 PPO 更新(353s/697s)与 train_exit
   (2111s) 段上电，标注段 GPU≈0，「偶尔 50%」是阶段切换。8770 无活跃连接、无并发探针
   ⇒ 已排除外部 ANE 争抢者。
   唯一剩余的**不改配方**杠杆 = Rust 跨线程教师批聚合（B≤24 汇成 B=128/256，前向确定性
   可逐位对拍），按用户「先剖析成本、优先试 Rust」规则登记为训练停止后的离线基准，
   **7×24 内不编译、不抢核**；B=1→MPS 锁步批量（理论 2~3×）会改数据分布/冻结协议，
   排 D3 之后预登记。R24 吞吐比 R21 低 ~10% 属观察项（学生权重不同、targets/deal
   13.3 vs 11.0、窗口长度不同），不触发事故、不重启、不改配置。
   完整数据与测量限制（未直读 ANE/GPU 硬件计数器，结论为间接证据）见
   `docs/python_rust_usage_audit.md` §7。自检 automation prompt 已加资源利用率口径。

## [自检 #34 · 2026-11-07 23:29] S1 收官轮 R24 标注中 · P22/P23 双零边际 · 两处预登记澄清

**结论：健康、产出符合预期（平台期延续）、主线零干预。** 唯一驱动 PID 3078（PPID=6189 keepalive），
单实例（1 driver + 1 annotate 5941），lock 1~2s；R24 = 完整 DAgger 轮，标注 12,995/25,000·worker
（52.0%）@4,190s、11,166/h·worker、ETA ≈00:33；panic/NaN/超时 0；npz 59；磁盘 305Gi；
health_watch --last 12 全绿；Web `/`=200 + `/api/meta` master=dushan_master + models[0].champion=false；
看门狗 6189 活、22:20 重启后无新拉起；PPO 链 p22→p23 连续。

**R22/R23 结算**：P22(init=exit_best≡r21 重锚) 120,000 状态/703.6s，D −0.024±0.350 p=.9442，
A +4.169±0.439，ent 由 1.03 回升 1.825（重锚预期）；P23 117,755 状态/696.7s，D −0.122±0.360
p=.7341，A +4.393±0.418；两轮均不晋升 ⇒ exit_best 仍 = exit_r21。重锚后两轮 D 为负但 <1SE，
不触发 rapid_abort（rapid_guard 未写标记）。

**best-of 不变**：P20 +4.962 仍 #1，P20−P23 +0.569±0.513 p=.267、P20−R11 +0.763±0.539 p=.157
⇒ top1 与其后全部 ns，交付强度自 R08 冻结。

**新登记（均在看 R36 数据之前锁定）**：
① 方案 B 条件②「链无缺号」改按**段**判定：当前段 = p22…p36（重锚点 exit_r21），p21 缺口属上一段
   已归档事故、不重复计入；其余两项条件不放松（docs §6.44.3）。
② E1 窗口按小时巡检会结构性错过 R24 train 段（00:34–01:08）⇒ 单建一次性 automation 于 00:45
   触发 `shadow_lab.py --e1`；守卫 rc=2/3 则不重试、顺延 R36；判据一字不改（docs §6.44.4）。

**早停状态：未到边界（最早 R36）**，`runs/early_stop` 不存在，`early_stop_guard --boundary 24`
按设计 not_ready（R24 不检查）。

**下一步**：R24 train 段内跑 E1（00:45）→ 捕 r24_gateD/三件套/4×25k → R25 起 S2 同配方 →
**R36 边界（≈07:0x）三件事**：740000+600 复现测量（p20 vs r12）、boundary=36 早停判读、方案 B 三项判定。
未改任何超参/判据/线上档。

## [自检 #35 · 2026-11-08 00:31] S1 收官轮标注末段 · 平台期定量证据入库 · 排期实测修订

**结论：健康、主线零干预、产出符合「平台期」预期。** 唯一驱动 PID 3078（PPID=6189 看门狗），
单实例（1 driver + 1 annotate 5941），`runs/night_run.lock` mtime 1s；阶段 = **R24 = S1 收官轮
（完整 DAgger）**，annotate 23,100/25,000·worker（92.4%）@7,930s、10,500/h·worker，
**落盘 ETA ≈00:42~00:45**（比 §6.44.5 预估 00:33 慢 ~9min，吞吐从 R21 的 11.2k/h 降到 10.5k/h）。
panic/NaN/Traceback 0；npz 70；磁盘 305Gi；`health_watch --last 12` 12/12 全绿
（kl .0117、clip .107、gnorm 1.30、ent 1.708、vl 1.004、states 117,755、secs 697）；
Web `/`=200、`/api/meta` master=dushan_master、models=[dushan_master(false), web_best(true)]；
看门狗 6189 活、22:20 后无新拉起；PPO 链 p22→p23 连续（p21 缺号为已归档事故）。

**平台期定量证据（新入库，供 R36 判读对照）**：自 P16（17:32 D +1.073 p=.0088 晋升）之后
**8 个决策轮无一在闸门 D 上通过**：P17 +0.056(.86) / P18 +0.391(.25) / P19 +0.166(.62) /
P20 +0.212(.52) / r21 −0.464(.24，零边际读法) / P22 −0.024(.94) / P23 −0.122(.73)。
闸门 A 在 4.15~4.96 横盘（P20 +4.962 为区间最高，P22 +4.169、P23 +4.393 无趋势）。
⇒ 与 §6.43 早停语义一致，但**不是结论**：M 判据只能由 600-seed 独立池在 R36 边界裁决。

**附带观测（仅登记，不改规则）**：`exit_best ≡ exit_r21`(A +3.717) 而 A 均值最高的是
`exit_p20`(+4.962) —— `exit_best` 语义为「最近一次 A 门通过者」，非「A 最高者」；这解释 P22 起
重锚后 ent 1.03→1.825。交付侧无风险（终评 best-of 覆盖全部 ckpt）；**晋升规则不动**。

**早停：未到边界**。`runs/early_stop` 无、`runs/gates/plan_b_enabled` 无；
`early_stop_guard --boundary 24` = `not_ready`（R24 按设计不检查，最早 R36）。
本周期最强=P20 +4.962±0.413，上周期最强=R11 +4.269±0.393（守卫自动选出，仅展示）。

**E1：仍待执行**。`runs/shadow/` 只有 early_stop_check.json / early_stop_mock.jsonl /
exploitability.json / grp_probe.json，无 `e1.done`、无结果文件。一次性定时器
85e191d7（11-08 **00:45**）ACTIVE，将自动撞 R24 train 段；若守卫在 train_r24.log 落盘前触发
⇒ rc=3/2 不重试、顺延 R36。**本轮不手动触发**（避免与定时器双跑抢核）。

**排期实测修订（已写入 docs §6.45.3）**：R24 train 段 ≈00:45–01:22 → R24 收口 ≈01:50~02:05 →
R25 起 S2 → R36 annotate 起跑 ≈04:30、落盘 ≈06:40、train 段 ≈06:41–07:15、
**R36 三件事（740000+600 复现测量 / boundary=36 早停判读 / 方案 B 三项）≈07:2x~07:5x**。

**下一硬决策点 = R36 边界**。未改任何超参、判据、线上档；无重启、无 kill、无编译、无并发探针。
[2026-11-08 00:44:31]   rc=0  8749s  annotate_r24
[2026-11-08 00:44:31] R24 标注完成 student=exit_best.onnx seed0=840000 → 4×25k
[2026-11-08 00:44:31] $ /Users/ligf9/zhuoji-mahjong-ai/.venv/bin/python train_exit.py --data runs/exit_r01/exit_w*.npz runs/exit_r02/exit_w*.npz runs/exit_r03/exit_w*.npz runs/exit_r04/exit_w*.npz runs/exit_r05/exit_w*.npz runs/exit_r06/exit_w*.npz runs/exit_r07/exit_w*.npz runs/exit_r08/exit_w*.npz runs/exit_r09/exit_w*.npz runs/exit_r10/exit_w*.npz runs/exit_r11/exit_w*.npz runs/exit_r12/exit_w*.npz runs/exit_r21/exit_w*.npz runs/exit_r24/exit_w*.npz --out /Users/ligf9/zhuoji-mahjong-ai/models/exit_r24.pt --epochs 30 --window 400000 --batch 256 --seed 20261130 --init /Users/ligf9/zhuoji-mahjong-ai/models/exit_best.pt  → train_r24.log
[2026-11-08 00:44:31]   rc=1  0s  train_r24
[2026-11-08 00:44:31] R24 阶段异常：AssertionError: Fatal Python error: init_sys_streams: can't initialize sys standard streams
Python runtime state: core initialized
OSError: [Errno 9] Bad file descriptor

Current thread 0x00000001efd4a1c0 (most recent call first):
  <no Python frame>

[2026-11-08 00:44:31] P25 PPO 链起点 = exit_best.pt  hands=1920 max_states=120000
[2026-11-08 00:44:31] $ /Users/ligf9/zhuoji-mahjong-ai/.venv/bin/python train_ppo.py --init /Users/ligf9/zhuoji-mahjong-ai/models/exit_best.pt --out /Users/ligf9/zhuoji-mahjong-ai/models/exit_p25.pt --hands 1920 --workers 4 --threads 2 --max-states 120000 --epochs 2 --lr 5e-5 --seed0 1150000  → ppo_p25.log
[2026-11-08 00:44:31]   rc=1  0s  ppo_p25
[2026-11-08 00:44:31] P25 PPO 训练失败 rc=1（主线不受影响）
[2026-11-08 00:44:31] R25 加速轮 PPO 无产出 → 回落到 DAgger 路径
[2026-11-08 00:44:31] $ /Users/ligf9/zhuoji-mahjong-ai/.venv/bin/python -c import torch,sys;sys.path.insert(0,'.');from zhuoji.net import build_from_ckpt;from zhuoji.forward import ensure_onnx;ck=torch.load('/Users/ligf9/zhuoji-mahjong-ai/models/exit_best.pt',map_location='cpu',weights_only=False);net,_=build_from_ckpt(ck);p=ensure_onnx(net,stem='exit_best');print('onnx:',p)  → export_exit_best.log
[2026-11-08 00:44:31]   rc=1  0s  export_exit_best
[2026-11-08 00:44:31] R25 阶段异常：AssertionError: Fatal Python error: init_sys_streams: can't initialize sys standard streams
Python runtime state: core initialized
OSError: [Errno 9] Bad file descriptor

Current thread 0x00000001efd4a1c0 (most recent call first):
  <no Python frame>

[2026-11-08 00:44:31] P26 PPO 链起点 = exit_best.pt  hands=1920 max_states=120000
[2026-11-08 00:44:31] $ /Users/ligf9/zhuoji-mahjong-ai/.venv/bin/python train_ppo.py --init /Users/ligf9/zhuoji-mahjong-ai/models/exit_best.pt --out /Users/ligf9/zhuoji-mahjong-ai/models/exit_p26.pt --hands 1920 --workers 4 --threads 2 --max-states 120000 --epochs 2 --lr 5e-5 --seed0 1160000  → ppo_p26.log
[2026-11-08 00:44:31]   rc=1  0s  ppo_p26
[2026-11-08 00:44:31] P26 PPO 训练失败 rc=1（主线不受影响）
[2026-11-08 00:44:31] R26 加速轮 PPO 无产出 → 回落到 DAgger 路径
[2026-11-08 00:44:31] $ /Users/ligf9/zhuoji-mahjong-ai/.venv/bin/python -c import torch,sys;sys.path.insert(0,'.');from zhuoji.net import build_from_ckpt;from zhuoji.forward import ensure_onnx;ck=torch.load('/Users/ligf9/zhuoji-mahjong-ai/models/exit_best.pt',map_location='cpu',weights_only=False);net,_=build_from_ckpt(ck);p=ensure_onnx(net,stem='exit_best');print('onnx:',p)  → export_exit_best.log
[2026-11-08 00:44:32]   rc=1  0s  export_exit_best
[2026-11-08 00:44:32] R26 阶段异常：AssertionError: Fatal Python error: init_sys_streams: can't initialize sys standard streams
Python runtime state: core initialized
OSError: [Errno 9] Bad file descriptor

Current thread 0x00000001efd4a1c0 (most recent call first):
  <no Python frame>

[2026-11-08 00:44:32] 连续 3 轮失败 → 退出，等待每小时自检定位后重启
[2026-11-08 00:44:32] 7×24 驱动结束 failures=3  best=exit_best.pt
[2026-11-08 00:51:10] ============================================================
[2026-11-08 00:51:10] 7×24 ExIt 驱动启动：R24 起，截止 2026-11-13 02:30（R1 已完成；断点续跑 + 每小时自检自动化）
[2026-11-08 00:51:10] R24 标注已完成（4×25k 分片齐），跳过
[2026-11-08 00:51:10] $ /Users/ligf9/zhuoji-mahjong-ai/.venv/bin/python train_exit.py --data runs/exit_r01/exit_w*.npz runs/exit_r02/exit_w*.npz runs/exit_r03/exit_w*.npz runs/exit_r04/exit_w*.npz runs/exit_r05/exit_w*.npz runs/exit_r06/exit_w*.npz runs/exit_r07/exit_w*.npz runs/exit_r08/exit_w*.npz runs/exit_r09/exit_w*.npz runs/exit_r10/exit_w*.npz runs/exit_r11/exit_w*.npz runs/exit_r12/exit_w*.npz runs/exit_r21/exit_w*.npz runs/exit_r24/exit_w*.npz --out /Users/ligf9/zhuoji-mahjong-ai/models/exit_r24.pt --epochs 30 --window 400000 --batch 256 --seed 20261130 --init /Users/ligf9/zhuoji-mahjong-ai/models/exit_best.pt  → train_r24.log

## [自检 #35b · 2026-11-08 00:55] 事故处置：R24 训练段瞬时崩溃 → 已修复并恢复（零数据损失）

**触发**：00:44:31 R24 annotate 落盘 4×25k 后，驱动启动 `train_exit.py` **0s 内 rc=1**，报
`Fatal Python error: init_sys_streams / OSError: [Errno 9] Bad file descriptor`；R25/R26 同因 1 秒内连锁失败，
`consecutive_fail>=3` ⇒ 驱动 00:44:32 自退；看门狗每 30s 拉起仍瞬崩（12 行「仍无心跳」）。

**根因**：驱动/看门狗进程的 **fd 0 已失效**（会话边界回收继承流），而 `subprocess` 未显式指定 stdin ⇒
子进程继承坏 fd ⇒ CPython `init_sys_streams` dup 失败 EBADF 直接 abort。与训练代码、数据、权重无关。

**修复（只动可靠性，零配方变更）**：
1. `night_run.py::run()` 与 `tools/keepalive.py::start_driver/start_web` 一律 `stdin=subprocess.DEVNULL`；
2. `night_run.py::main()` 增启动探针（`python -c` 子进程自检，失败则 `return 2` 不进轮次循环）；
3. 轮次异常且墙钟 <60s ⇒ 记「疑似环境故障」并退避 180s（不再 1 秒烧 3 轮）。
4. 看门狗以干净 stdin 重启 **PID 19486**（旧 6189 已 TERM，Web 进程未受影响）。
5. 驱动 00:51:10 断点续跑：`R24 标注已完成（4×25k 分片齐），跳过` → 直接进 train 段（**PID 19488**，
   单实例：1 driver + 1 train_exit 19489；lock 心跳 1s；无第二个驱动）。

**损失**：数据 0（R24 4×25k 完好、exit_p22/p23/exit_best/冠军表/线上档位全未动）；轮号 0（R25/R26 无任何
ckpt/闸门产物，`first_incomplete_round()` 仍为 24，p24/p25/p26 将正常生成，无链缺口）；墙钟 ≈7min
（00:44–00:51，占剩余 121h 的 0.1%）。

**复核**：Web `/`=200 全程未断（88725 独立于驱动）；看门狗 19486 活、30s 一轮；`health_watch` 12/12 绿；
磁盘 305Gi。**未触碰早停判据、闸门、候选规则、线上默认档。**

**排期（实测）**：R24 train 00:51–01:27 → A/C/D ≈01:27–02:10 → **R24 收口 ≈02:10** → R25 起 S2；
R36 三件事（740000+600 复现测量 / boundary=36 早停 / 方案 B 三项）顺延至 **≈07:3x–08:0x**。

**E1**：00:45 定时器到点未执行（当时窗口实为空窗）；现窗口 = R24 train 段 00:51–01:27，定时器若在窗口内
补触发即正常执行，否则按预登记顺延 R36。**不手动双跑。**

[E1] 2026-11-08 01:38 rc=3 硬上限中止（30min 上限；实测 42.9min，其中 rollout 161s、双更新各 118 步）→ 闸门 A 未跑、无 e1_result.json、结果作废不得判读；e1.done 未写，一次性机会未被消耗；R24 窗口未捕获，顺延 R36/R48，判据不变
[2026-11-08 01:47:12]   rc=0  3363s  train_r24
[2026-11-08 01:47:12] R24 训练完成 init=exit_best.pt → exit_r24.pt  (['[done] best val 1.7366 → /Users/ligf9/zhuoji-mahjong-ai/models/exit_r24.pt'])
[2026-11-08 01:47:12] $ /Users/ligf9/zhuoji-mahjong-ai/.venv/bin/python eval_pairs.py --a /Users/ligf9/zhuoji-mahjong-ai/models/exit_r24.pt --b /Users/ligf9/zhuoji-mahjong-ai/models/dushan_master.pt --mode greedy --seeds 300 --seed0 710000 --workers 6 --out /Users/ligf9/zhuoji-mahjong-ai/runs/gates/r24_gateA.jsonl  → gateA_r24.log
[2026-11-08 01:48:36]   rc=0  83s  gateA_r24
[2026-11-08 01:48:36] R24 闸门 A: [summary] seats=[0, 1, 2, 3]  n=300  diff=+3.501±0.406  z=+8.63  p≈0.0000  符号 206:91  → 通过
[2026-11-08 01:48:36] $ /Users/ligf9/zhuoji-mahjong-ai/.venv/bin/python -c import torch,sys;sys.path.insert(0,'.');from zhuoji.net import build_from_ckpt;from zhuoji.forward import ensure_onnx;ck=torch.load('/Users/ligf9/zhuoji-mahjong-ai/models/exit_best.pt',map_location='cpu',weights_only=False);net,_=build_from_ckpt(ck);p=ensure_onnx(net,stem='exit_best');print('onnx:',p)  → export_exit_best.log
[2026-11-08 01:48:37]   rc=0  1s  export_exit_best
[2026-11-08 01:48:37] 晋升：exit_r24.pt → exit_best.pt
[2026-11-08 01:48:37] $ /Users/ligf9/zhuoji-mahjong-ai/.venv/bin/python eval_pairs.py --a /Users/ligf9/zhuoji-mahjong-ai/models/exit_r24.pt --b /Users/ligf9/zhuoji-mahjong-ai/models/dushan_master.pt --a-mode greedy --b-mode search32 --seeds 50 --seed0 720000 --workers 6 --out /Users/ligf9/zhuoji-mahjong-ai/runs/gates/r24_gateC.jsonl  → gateC_r24.log
[2026-11-08 02:14:28]   rc=0  1551s  gateC_r24
[2026-11-08 02:14:28] R24 闸门 C(趋势): [summary] seats=[0, 1, 2, 3]  n=50  diff=+3.053±0.998  z=+3.06  p≈0.0022  符号 32:17
[2026-11-08 02:14:28] $ /Users/ligf9/zhuoji-mahjong-ai/.venv/bin/python eval_pairs.py --a /Users/ligf9/zhuoji-mahjong-ai/models/exit_r24.pt --b /Users/ligf9/zhuoji-mahjong-ai/runs/gates/r24_prevbest.pt --mode greedy --seeds 300 --seed0 710000 --workers 6 --out /Users/ligf9/zhuoji-mahjong-ai/runs/gates/r24_gateD.jsonl  → gateD_r24.log
[2026-11-08 02:17:31]   rc=0  183s  gateD_r24
[2026-11-08 02:17:31] R24 闸门 D(vs 上轮最佳): [summary] seats=[0, 1, 2, 3]  n=300  diff=-0.037±0.404  z=-0.09  p≈0.9276  符号 128:169
[2026-11-08 02:17:31] P24 PPO 链起点 = exit_p23.pt  hands=960 max_states=60000
[2026-11-08 02:17:31] $ /Users/ligf9/zhuoji-mahjong-ai/.venv/bin/python train_ppo.py --init /Users/ligf9/zhuoji-mahjong-ai/models/exit_p23.pt --out /Users/ligf9/zhuoji-mahjong-ai/models/exit_p24.pt --hands 960 --workers 4 --threads 2 --max-states 60000 --epochs 2 --lr 5e-5 --seed0 1140000  → ppo_p24.log

## [自检 #36 · 2026-11-08 02:2x] S1 收官结算：R24 全通过（A +3.501 / C +3.053 / D ns）+ DAgger 重锚低于链末端 + E1 rc=3 处置

**健康**：驱动 19488（PPID 19486，00:51:10 起）单实例；lock<120s；Web `/`=200、`master=dushan_master`、
`models[0].champion=false`、冠军表仅 `web_best`；磁盘 305Gi；`health_watch` 12/12 绿；00:51 起 panic/NaN/超时 0。

**R24（S1 收官 · 完整 DAgger）**：annotate 4×25k（97k 张，10.5k/h·w，`check_npz` 4/4）→ train 3363s
（init=exit_best、window 400k、**best val 1.7366 项目最低**、ep05 早停）→ **A +3.501±0.406**（206:91，p≈0）
⇒ 晋升 `exit_r24`≡`exit_best`(52ad219f) → **C +3.053±0.998**（32:17，p=0.0022，26min）→
**D(vs exit_p23) −0.037±0.404**（128:169，p=0.9276，ns ⇒ §6.41.2 只记零边际）→ 02:17:31 起 P24
（init=exit_p23、hands=960，链 p23→p24 连续）→ **R25 起 S2**。

**S1 结算**：gate A 均值 **+4.323**（上一周期 +3.226），内部 OLS slope **−0.034 分/轮**（§6.43.2 S1 辅证成立）。
S1 是一次**水平跃迁**（R13–R20 PPO 链 4.03→4.96），不是逐轮爬升。best-of 仍是 **P20 +4.962**，与
P18/P15/P19/P23 全部 <1SE 并列 ⇒ 交付强度仍冻结在 R08 以来的水平（终评 best-of 覆盖）。

**新证据（第三处，支持方案 B 但不显著）**：同池配对 **R24−P23 = −0.892±0.548（p=.104）**、
R24−R21 −0.216±0.524、P23−R21 +0.677±0.450 ⇒ 点估计「exit_best 重锚的 DAgger 比链末端低 ≈0.9 分」，
与 §6.40 结构性损失一致。**不改 R36 三项条件、不提前启用方案 B。**

**E1 rc=3**：00:45 定时器在 R24 train 段触发，rollout 320 局/15,020 状态 161s，双更新各 118 步，
**01:38:54 达 30min 硬上限 abort** ⇒ 闸门未跑、无结果、`e1.done` 未写，**本次机会未消耗且不得判读**。
副作用：抢 MPS 使 R24 train 3363s（正常 2032–2555s）。**处置**：E1 重跑窗口改到训练停止后的终评窗口
（零机会成本），判据一字不改；不在 R36 train 段与复现测量抢资源。

**早停**：未到边界（最早 R36），`runs/early_stop` 与 `plan_b_enabled` 均无。**R36 三件事执行手册见计划 §6.47.5**
（含新发现的口径冲突：M 的 A 候选含 R36 自身 gateA ⇒ M 不可能落在 R36 train 段；锁死「先复现 740000+600、
后 M 750000+600、严禁并发」）。

**排期（实测重算）**：R24 收口 02:17 → P24 ≈02:27 → R25–R35 11 加速轮 ≈143min → R36 annotate ≈04:50 →
落盘 ≈07:10 → train ≈07:10–07:45 → A ≈07:47 → C ≈08:15 → **边界判读 ≈08:2x–08:4x**。

**未动**：TAU/Q-MSE/n_det/lr/PPO_HANDS/Φ/对手分布/闸门判据/线上默认档/`dushan_master.pt`/冠军表。
[2026-11-08 02:24:31]   rc=0  420s  ppo_p24
[2026-11-08 02:24:31] P24 PPO 完成 init=exit_p23.pt → exit_p24.pt {"hands": 960, "states": 55181, "A_std": 8.641499519348145, "ret_std": 13.15833069453214, "kl": [0.006066428264333539, 0.008663322808354645], "vl": 1.1389570861226983, "ent": 1.6037212974495358, "gnorm": 1.4115479841000504, "clipfrac": 0.08563737440164443, "ent_rollout": 1.984221621133959, "kl_stop": false, "secs": 419.2}
[2026-11-08 02:24:31] $ /Users/ligf9/zhuoji-mahjong-ai/.venv/bin/python eval_pairs.py --a /Users/ligf9/zhuoji-mahjong-ai/models/exit_p24.pt --b /Users/ligf9/zhuoji-mahjong-ai/models/exit_best.pt --mode greedy --seeds 300 --seed0 710000 --workers 6 --out /Users/ligf9/zhuoji-mahjong-ai/runs/gates/p24_gateD.jsonl  → gateD_p24.log
[2026-11-08 02:27:30]   rc=0  179s  gateD_p24
[2026-11-08 02:27:30] P24 闸门 D(vs 现最佳): [summary] seats=[0, 1, 2, 3]  n=300  diff=-0.027±0.389  z=-0.07  p≈0.9454  符号 139:160  → 不通过
[2026-11-08 02:27:30] $ /Users/ligf9/zhuoji-mahjong-ai/.venv/bin/python eval_pairs.py --a /Users/ligf9/zhuoji-mahjong-ai/models/exit_p24.pt --b /Users/ligf9/zhuoji-mahjong-ai/models/dushan_master.pt --mode greedy --seeds 300 --seed0 710000 --workers 6 --out /Users/ligf9/zhuoji-mahjong-ai/runs/gates/p24_gateA.jsonl  → gateA_p24.log
[2026-11-08 02:28:49]   rc=0  79s  gateA_p24
[2026-11-08 02:28:49] P24 闸门 A(vs master): [summary] seats=[0, 1, 2, 3]  n=300  diff=+3.972±0.408  z=+9.73  p≈0.0000  符号 213:84
[2026-11-08 02:28:49] P24 不晋升（未显著优于现最佳），ckpt 留档供终评 best-of
[2026-11-08 02:28:49] R24 轮次完成  best=exit_best.pt
[2026-11-08 02:28:49] P25 PPO 链起点 = exit_p24.pt  hands=1920 max_states=120000
[2026-11-08 02:28:49] $ /Users/ligf9/zhuoji-mahjong-ai/.venv/bin/python train_ppo.py --init /Users/ligf9/zhuoji-mahjong-ai/models/exit_p24.pt --out /Users/ligf9/zhuoji-mahjong-ai/models/exit_p25.pt --hands 1920 --workers 4 --threads 2 --max-states 120000 --epochs 2 --lr 5e-5 --seed0 1150000  → ppo_p25.log

[P24 · 2026-11-08 02:28] A +3.972±0.408（213:84，通过）/ D −0.027±0.389（139:160，ns 不晋升）；960 局、55,181 状态、420s，
kl[.0061,.0087]、clipfrac .086、ent 1.604、ent_rollout 1.984（重锚后熵回落属预期）；链 p23→p24 连续。
**R24 轮次完成 → 02:28:49 起 R25 = S2 第一个加速轮**（P25 init=exit_p24、hands=1920）⇒ 下一步训练已自动接续，零干预。
[2026-11-08 02:42:01]   rc=0  792s  ppo_p25
[2026-11-08 02:42:01] P25 PPO 完成 init=exit_p24.pt → exit_p25.pt {"hands": 1920, "states": 108270, "A_std": 8.803433418273926, "ret_std": 13.195475471413172, "kl": [0.006101539146719857, 0.00843485102253409], "vl": 1.1795306290295107, "ent": 1.489754940460196, "gnorm": 1.359075913122078, "clipfrac": 0.09188638887330713, "ent_rollout": 1.8405747562265509, "kl_stop": false, "secs": 791.3}
[2026-11-08 02:42:01] $ /Users/ligf9/zhuoji-mahjong-ai/.venv/bin/python eval_pairs.py --a /Users/ligf9/zhuoji-mahjong-ai/models/exit_p25.pt --b /Users/ligf9/zhuoji-mahjong-ai/models/exit_best.pt --mode greedy --seeds 300 --seed0 710000 --workers 6 --out /Users/ligf9/zhuoji-mahjong-ai/runs/gates/p25_gateD.jsonl  → gateD_p25.log
[2026-11-08 02:44:57]   rc=0  175s  gateD_p25
[2026-11-08 02:44:57] P25 闸门 D(vs 现最佳): [summary] seats=[0, 1, 2, 3]  n=300  diff=+0.081±0.387  z=+0.21  p≈0.8339  符号 144:153  → 不通过
[2026-11-08 02:44:57] $ /Users/ligf9/zhuoji-mahjong-ai/.venv/bin/python eval_pairs.py --a /Users/ligf9/zhuoji-mahjong-ai/models/exit_p25.pt --b /Users/ligf9/zhuoji-mahjong-ai/models/dushan_master.pt --mode greedy --seeds 300 --seed0 710000 --workers 6 --out /Users/ligf9/zhuoji-mahjong-ai/runs/gates/p25_gateA.jsonl  → gateA_p25.log
[2026-11-08 02:46:13]   rc=0  76s  gateA_p25
[2026-11-08 02:46:13] P25 闸门 A(vs master): [summary] seats=[0, 1, 2, 3]  n=300  diff=+4.620±0.410  z=+11.26  p≈0.0000  符号 211:87
[2026-11-08 02:46:13] P25 不晋升（未显著优于现最佳），ckpt 留档供终评 best-of
[2026-11-08 02:46:13] R25 加速轮完成（PPO-only：无标注/无训练，含闸门 D/A）
[2026-11-08 02:46:13] P26 PPO 链起点 = exit_p25.pt  hands=1920 max_states=120000
[2026-11-08 02:46:13] $ /Users/ligf9/zhuoji-mahjong-ai/.venv/bin/python train_ppo.py --init /Users/ligf9/zhuoji-mahjong-ai/models/exit_p25.pt --out /Users/ligf9/zhuoji-mahjong-ai/models/exit_p26.pt --hands 1920 --workers 4 --threads 2 --max-states 120000 --epochs 2 --lr 5e-5 --seed0 1160000  → ppo_p26.log
[2026-11-08 02:58:57]   rc=0  764s  ppo_p26
[2026-11-08 02:58:57] P26 PPO 完成 init=exit_p25.pt → exit_p26.pt {"hands": 1920, "states": 103594, "A_std": 8.772665023803711, "ret_std": 13.038884154712012, "kl": [0.006814566495638616, 0.008438087108453023], "vl": 1.1778702677032094, "ent": 1.404359851943122, "gnorm": 1.3765516682171528, "clipfrac": 0.08663092321819729, "ent_rollout": 1.7183167492113238, "kl_stop": false, "secs": 762.7}
[2026-11-08 02:58:57] $ /Users/ligf9/zhuoji-mahjong-ai/.venv/bin/python eval_pairs.py --a /Users/ligf9/zhuoji-mahjong-ai/models/exit_p26.pt --b /Users/ligf9/zhuoji-mahjong-ai/models/exit_best.pt --mode greedy --seeds 300 --seed0 710000 --workers 6 --out /Users/ligf9/zhuoji-mahjong-ai/runs/gates/p26_gateD.jsonl  → gateD_p26.log
[2026-11-08 03:01:51]   rc=0  174s  gateD_p26
[2026-11-08 03:01:51] P26 闸门 D(vs 现最佳): [summary] seats=[0, 1, 2, 3]  n=300  diff=+0.861±0.371  z=+2.32  p≈0.0203  符号 152:143  → 通过
[2026-11-08 03:01:51] $ /Users/ligf9/zhuoji-mahjong-ai/.venv/bin/python eval_pairs.py --a /Users/ligf9/zhuoji-mahjong-ai/models/exit_p26.pt --b /Users/ligf9/zhuoji-mahjong-ai/models/dushan_master.pt --mode greedy --seeds 300 --seed0 710000 --workers 6 --out /Users/ligf9/zhuoji-mahjong-ai/runs/gates/p26_gateA.jsonl  → gateA_p26.log
[2026-11-08 03:03:10]   rc=0  79s  gateA_p26
[2026-11-08 03:03:10] P26 闸门 A(vs master): [summary] seats=[0, 1, 2, 3]  n=300  diff=+4.294±0.390  z=+11.03  p≈0.0000  符号 216:81
[2026-11-08 03:03:10] $ /Users/ligf9/zhuoji-mahjong-ai/.venv/bin/python -c import torch,sys;sys.path.insert(0,'.');from zhuoji.net import build_from_ckpt;from zhuoji.forward import ensure_onnx;ck=torch.load('/Users/ligf9/zhuoji-mahjong-ai/models/exit_best.pt',map_location='cpu',weights_only=False);net,_=build_from_ckpt(ck);p=ensure_onnx(net,stem='exit_best');print('onnx:',p)  → export_exit_best.log
[2026-11-08 03:03:11]   rc=0  1s  export_exit_best
[2026-11-08 03:03:11] 晋升：exit_p26.pt → exit_best.pt
[2026-11-08 03:03:11] R26 加速轮完成（PPO-only：无标注/无训练，含闸门 D/A）
[2026-11-08 03:03:11] P27 PPO 链起点 = exit_p26.pt  hands=1920 max_states=120000
[2026-11-08 03:03:11] $ /Users/ligf9/zhuoji-mahjong-ai/.venv/bin/python train_ppo.py --init /Users/ligf9/zhuoji-mahjong-ai/models/exit_p26.pt --out /Users/ligf9/zhuoji-mahjong-ai/models/exit_p27.pt --hands 1920 --workers 4 --threads 2 --max-states 120000 --epochs 2 --lr 5e-5 --seed0 1170000  → ppo_p27.log
[2026-11-08 03:15:30]   rc=0  739s  ppo_p27
[2026-11-08 03:15:30] P27 PPO 完成 init=exit_p26.pt → exit_p27.pt {"hands": 1920, "states": 100785, "A_std": 8.5932035446167, "ret_std": 12.750306368868161, "kl": [0.0051467257614193134, 0.007122264162998949], "vl": 1.135447898643271, "ent": 1.320854568239396, "gnorm": 1.3254595399659297, "clipfrac": 0.08192527297940956, "ent_rollout": 1.60227934046844, "kl_stop": false, "secs": 737.6}
[2026-11-08 03:15:30] $ /Users/ligf9/zhuoji-mahjong-ai/.venv/bin/python eval_pairs.py --a /Users/ligf9/zhuoji-mahjong-ai/models/exit_p27.pt --b /Users/ligf9/zhuoji-mahjong-ai/models/exit_best.pt --mode greedy --seeds 300 --seed0 710000 --workers 6 --out /Users/ligf9/zhuoji-mahjong-ai/runs/gates/p27_gateD.jsonl  → gateD_p27.log
[2026-11-08 03:18:25]   rc=0  176s  gateD_p27
[2026-11-08 03:18:25] P27 闸门 D(vs 现最佳): [summary] seats=[0, 1, 2, 3]  n=300  diff=+0.698±0.308  z=+2.26  p≈0.0236  符号 153:121  → 通过
[2026-11-08 03:18:25] $ /Users/ligf9/zhuoji-mahjong-ai/.venv/bin/python eval_pairs.py --a /Users/ligf9/zhuoji-mahjong-ai/models/exit_p27.pt --b /Users/ligf9/zhuoji-mahjong-ai/models/dushan_master.pt --mode greedy --seeds 300 --seed0 710000 --workers 6 --out /Users/ligf9/zhuoji-mahjong-ai/runs/gates/p27_gateA.jsonl  → gateA_p27.log
[2026-11-08 03:19:43]   rc=0  77s  gateA_p27
[2026-11-08 03:19:43] P27 闸门 A(vs master): [summary] seats=[0, 1, 2, 3]  n=300  diff=+4.414±0.402  z=+10.98  p≈0.0000  符号 217:82
[2026-11-08 03:19:43] $ /Users/ligf9/zhuoji-mahjong-ai/.venv/bin/python -c import torch,sys;sys.path.insert(0,'.');from zhuoji.net import build_from_ckpt;from zhuoji.forward import ensure_onnx;ck=torch.load('/Users/ligf9/zhuoji-mahjong-ai/models/exit_best.pt',map_location='cpu',weights_only=False);net,_=build_from_ckpt(ck);p=ensure_onnx(net,stem='exit_best');print('onnx:',p)  → export_exit_best.log
[2026-11-08 03:19:44]   rc=0  1s  export_exit_best
[2026-11-08 03:19:44] 晋升：exit_p27.pt → exit_best.pt
[2026-11-08 03:19:44] R27 加速轮完成（PPO-only：无标注/无训练，含闸门 D/A）
[2026-11-08 03:19:44] P28 PPO 链起点 = exit_p27.pt  hands=1920 max_states=120000
[2026-11-08 03:19:44] $ /Users/ligf9/zhuoji-mahjong-ai/.venv/bin/python train_ppo.py --init /Users/ligf9/zhuoji-mahjong-ai/models/exit_p27.pt --out /Users/ligf9/zhuoji-mahjong-ai/models/exit_p28.pt --hands 1920 --workers 4 --threads 2 --max-states 120000 --epochs 2 --lr 5e-5 --seed0 1180000  → ppo_p28.log

[自检 #37 · 2026-11-08 03:35]  S2 中段：P26/P27 连续通过闸门 D 并晋升 exit_best（S1 以来首次）
├ 健康：绿。P27 kl .0071 / clip .082 / gnorm 1.325 / ent 1.321 / ent_rollout 1.602 / states 100,785 / 738s；
│  P24/P25 的 vl 三连升 WARN（0.819→1.180）已由 P27 vl 1.135 自发解除；链 p24→p27 连续；rapid_abort/ppo_disabled 无。
├ 闸门：P24 A +3.972 D −0.027（不晋升）｜P25 A +4.620 D +0.081（不晋升）
│  P26 A +4.294 D **+0.861±0.371 p=.0203** → 晋升 exit_best=p26（03:03）
│  P27 A +4.414 D **+0.698±0.308 p=.0236** → 晋升 exit_best=p27（03:19，md5 382cf726）
├ S2 gateA 均值 +4.325（SE .136）vs S1 +4.323；ent_rollout 三轮 −19.3%（1.984→1.602）。
├ 解释（弱阳性待确认）：D 固定池十余次比较，α=.05 期望假阳 0.5~0.75 个，两连正超出期望但不足以定论；
│  **禁止**据此改配方/加预算/挑显著/宣称可累积。确认路径只有三条已预登记的全新池：740000+600 复现（R36 train 段）、
│  750000+600（M，gateA_r36 之后）、终评 best-of + 1000-seed 复赛。阈值一字不改。
├ 结构含义：exit_best 已从 r24 重锚点移到 PPO 链末端 p27 ⇒ §6.40「重锚丢弃链增益」被 D 门机制自动部分修复；
│  若边界时 exit_best 仍=链末端，方案 B 的 1 行改动在 R36 等价 no-op（仍按预登记判定留档，不改判据）。
├ 早停：**未到边界**（最早 R36），runs/early_stop 与 plan_b_enabled 均无。预测 verdict=continue（S2 辅证不成立：
│  熵仍在下降通道），但必须到场执行 guard，不得据预测跳过。
├ 排期修正：实测 16.7min/加速轮（PPO 740-790s + D 176s + A 78s）⇒ R36 annotate ≈05:35、落盘 ≈08:0x、
│  边界判读 ≈09:2x–09:4x（原 08:2x 估计作废）。复现测量窗口挪到 ≈08:05–08:45。
└ 零干预：未重启/未改超参/未注册冠军/未跑 E1（rc=3 已作废，窗口=终评）/未跑吃 CPU 探针。
[2026-11-08 03:31:32]   rc=0  708s  ppo_p28
[2026-11-08 03:31:32] P28 PPO 完成 init=exit_p27.pt → exit_p28.pt {"hands": 1920, "states": 96199, "A_std": 8.615372657775879, "ret_std": 12.551539580200245, "kl": [0.005436735338236681, 0.007204608200984868], "vl": 1.1399171067837706, "ent": 1.2476772136510688, "gnorm": 1.3382236428875873, "clipfrac": 0.0770438952975847, "ent_rollout": 1.4807058967491065, "kl_stop": false, "secs": 707.5}
[2026-11-08 03:31:32] $ /Users/ligf9/zhuoji-mahjong-ai/.venv/bin/python eval_pairs.py --a /Users/ligf9/zhuoji-mahjong-ai/models/exit_p28.pt --b /Users/ligf9/zhuoji-mahjong-ai/models/exit_best.pt --mode greedy --seeds 300 --seed0 710000 --workers 6 --out /Users/ligf9/zhuoji-mahjong-ai/runs/gates/p28_gateD.jsonl  → gateD_p28.log
[2026-11-08 03:34:31]   rc=0  179s  gateD_p28
[2026-11-08 03:34:31] P28 闸门 D(vs 现最佳): [summary] seats=[0, 1, 2, 3]  n=300  diff=+0.262±0.268  z=+0.98  p≈0.3288  符号 135:129  → 不通过
[2026-11-08 03:34:31] $ /Users/ligf9/zhuoji-mahjong-ai/.venv/bin/python eval_pairs.py --a /Users/ligf9/zhuoji-mahjong-ai/models/exit_p28.pt --b /Users/ligf9/zhuoji-mahjong-ai/models/dushan_master.pt --mode greedy --seeds 300 --seed0 710000 --workers 6 --out /Users/ligf9/zhuoji-mahjong-ai/runs/gates/p28_gateA.jsonl  → gateA_p28.log
[2026-11-08 03:35:48]   rc=0  77s  gateA_p28
[2026-11-08 03:35:48] P28 闸门 A(vs master): [summary] seats=[0, 1, 2, 3]  n=300  diff=+4.519±0.426  z=+10.61  p≈0.0000  符号 220:79
[2026-11-08 03:35:48] P28 不晋升（未显著优于现最佳），ckpt 留档供终评 best-of
[2026-11-08 03:35:48] R28 加速轮完成（PPO-only：无标注/无训练，含闸门 D/A）
[2026-11-08 03:35:48] P29 PPO 链起点 = exit_p28.pt  hands=1920 max_states=120000
[2026-11-08 03:35:48] $ /Users/ligf9/zhuoji-mahjong-ai/.venv/bin/python train_ppo.py --init /Users/ligf9/zhuoji-mahjong-ai/models/exit_p28.pt --out /Users/ligf9/zhuoji-mahjong-ai/models/exit_p29.pt --hands 1920 --workers 4 --threads 2 --max-states 120000 --epochs 2 --lr 5e-5 --seed0 1190000  → ppo_p29.log
[2026-11-08 03:47:44]   rc=0  716s  ppo_p29
[2026-11-08 03:47:44] P29 PPO 完成 init=exit_p28.pt → exit_p29.pt {"hands": 1920, "states": 96933, "A_std": 8.537837028503418, "ret_std": 12.213327334651002, "kl": [0.004848265157745891, 0.007120465743048289], "vl": 1.1206384152567166, "ent": 1.177596491842597, "gnorm": 1.333793732059977, "clipfrac": 0.07216555603575896, "ent_rollout": 1.3700389497750902, "kl_stop": false, "secs": 714.8}
[2026-11-08 03:47:44] $ /Users/ligf9/zhuoji-mahjong-ai/.venv/bin/python eval_pairs.py --a /Users/ligf9/zhuoji-mahjong-ai/models/exit_p29.pt --b /Users/ligf9/zhuoji-mahjong-ai/models/exit_best.pt --mode greedy --seeds 300 --seed0 710000 --workers 6 --out /Users/ligf9/zhuoji-mahjong-ai/runs/gates/p29_gateD.jsonl  → gateD_p29.log
[2026-11-08 03:50:40]   rc=0  175s  gateD_p29
[2026-11-08 03:50:40] P29 闸门 D(vs 现最佳): [summary] seats=[0, 1, 2, 3]  n=300  diff=+0.511±0.303  z=+1.68  p≈0.0920  符号 142:135  → 不通过
[2026-11-08 03:50:40] $ /Users/ligf9/zhuoji-mahjong-ai/.venv/bin/python eval_pairs.py --a /Users/ligf9/zhuoji-mahjong-ai/models/exit_p29.pt --b /Users/ligf9/zhuoji-mahjong-ai/models/dushan_master.pt --mode greedy --seeds 300 --seed0 710000 --workers 6 --out /Users/ligf9/zhuoji-mahjong-ai/runs/gates/p29_gateA.jsonl  → gateA_p29.log
[2026-11-08 03:51:56]   rc=0  76s  gateA_p29
[2026-11-08 03:51:56] P29 闸门 A(vs master): [summary] seats=[0, 1, 2, 3]  n=300  diff=+4.107±0.398  z=+10.31  p≈0.0000  符号 223:74
[2026-11-08 03:51:56] P29 不晋升（未显著优于现最佳），ckpt 留档供终评 best-of
[2026-11-08 03:51:56] R29 加速轮完成（PPO-only：无标注/无训练，含闸门 D/A）
[2026-11-08 03:51:56] P30 PPO 链起点 = exit_p29.pt  hands=1920 max_states=120000
[2026-11-08 03:51:56] $ /Users/ligf9/zhuoji-mahjong-ai/.venv/bin/python train_ppo.py --init /Users/ligf9/zhuoji-mahjong-ai/models/exit_p29.pt --out /Users/ligf9/zhuoji-mahjong-ai/models/exit_p30.pt --hands 1920 --workers 4 --threads 2 --max-states 120000 --epochs 2 --lr 5e-5 --seed0 1200000  → ppo_p30.log
[2026-11-08 04:03:21]   rc=0  686s  ppo_p30
[2026-11-08 04:03:21] P30 PPO 完成 init=exit_p29.pt → exit_p30.pt {"hands": 1920, "states": 93725, "A_std": 8.277425765991211, "ret_std": 11.866499150268934, "kl": [0.005759253620064744, 0.007628280181743519], "vl": 1.0559105257572206, "ent": 1.121682475795538, "gnorm": 1.3195482460776857, "clipfrac": 0.07049949938172541, "ent_rollout": 1.2862200794650416, "kl_stop": false, "secs": 684.7}
[2026-11-08 04:03:21] $ /Users/ligf9/zhuoji-mahjong-ai/.venv/bin/python eval_pairs.py --a /Users/ligf9/zhuoji-mahjong-ai/models/exit_p30.pt --b /Users/ligf9/zhuoji-mahjong-ai/models/exit_best.pt --mode greedy --seeds 300 --seed0 710000 --workers 6 --out /Users/ligf9/zhuoji-mahjong-ai/runs/gates/p30_gateD.jsonl  → gateD_p30.log
[2026-11-08 04:06:18]   rc=0  177s  gateD_p30
[2026-11-08 04:06:18] P30 闸门 D(vs 现最佳): [summary] seats=[0, 1, 2, 3]  n=300  diff=+0.378±0.295  z=+1.28  p≈0.2003  符号 149:135  → 不通过
[2026-11-08 04:06:18] $ /Users/ligf9/zhuoji-mahjong-ai/.venv/bin/python eval_pairs.py --a /Users/ligf9/zhuoji-mahjong-ai/models/exit_p30.pt --b /Users/ligf9/zhuoji-mahjong-ai/models/dushan_master.pt --mode greedy --seeds 300 --seed0 710000 --workers 6 --out /Users/ligf9/zhuoji-mahjong-ai/runs/gates/p30_gateA.jsonl  → gateA_p30.log
[2026-11-08 04:07:35]   rc=0  77s  gateA_p30
[2026-11-08 04:07:35] P30 闸门 A(vs master): [summary] seats=[0, 1, 2, 3]  n=300  diff=+4.634±0.396  z=+11.71  p≈0.0000  符号 224:74
[2026-11-08 04:07:35] P30 不晋升（未显著优于现最佳），ckpt 留档供终评 best-of
[2026-11-08 04:07:35] R30 加速轮完成（PPO-only：无标注/无训练，含闸门 D/A）
[2026-11-08 04:07:35] P31 PPO 链起点 = exit_p30.pt  hands=1920 max_states=120000
[2026-11-08 04:07:35] $ /Users/ligf9/zhuoji-mahjong-ai/.venv/bin/python train_ppo.py --init /Users/ligf9/zhuoji-mahjong-ai/models/exit_p30.pt --out /Users/ligf9/zhuoji-mahjong-ai/models/exit_p31.pt --hands 1920 --workers 4 --threads 2 --max-states 120000 --epochs 2 --lr 5e-5 --seed0 1210000  → ppo_p31.log
[2026-11-08 04:18:48]   rc=0  673s  ppo_p31
[2026-11-08 04:18:48] P31 PPO 完成 init=exit_p30.pt → exit_p31.pt {"hands": 1920, "states": 90651, "A_std": 8.243407249450684, "ret_std": 11.713974026776736, "kl": [0.004563244704557189, 0.005867220863247369], "vl": 1.0454724352964213, "ent": 1.0746011097666244, "gnorm": 1.304388032916566, "clipfrac": 0.066964169276852, "ent_rollout": 1.21096595738899, "kl_stop": false, "secs": 671.8}
[2026-11-08 04:18:48] $ /Users/ligf9/zhuoji-mahjong-ai/.venv/bin/python eval_pairs.py --a /Users/ligf9/zhuoji-mahjong-ai/models/exit_p31.pt --b /Users/ligf9/zhuoji-mahjong-ai/models/exit_best.pt --mode greedy --seeds 300 --seed0 710000 --workers 6 --out /Users/ligf9/zhuoji-mahjong-ai/runs/gates/p31_gateD.jsonl  → gateD_p31.log
[2026-11-08 04:21:43]   rc=0  175s  gateD_p31
[2026-11-08 04:21:43] P31 闸门 D(vs 现最佳): [summary] seats=[0, 1, 2, 3]  n=300  diff=+0.301±0.311  z=+0.97  p≈0.3327  符号 149:138  → 不通过
[2026-11-08 04:21:43] $ /Users/ligf9/zhuoji-mahjong-ai/.venv/bin/python eval_pairs.py --a /Users/ligf9/zhuoji-mahjong-ai/models/exit_p31.pt --b /Users/ligf9/zhuoji-mahjong-ai/models/dushan_master.pt --mode greedy --seeds 300 --seed0 710000 --workers 6 --out /Users/ligf9/zhuoji-mahjong-ai/runs/gates/p31_gateA.jsonl  → gateA_p31.log
[2026-11-08 04:22:59]   rc=0  75s  gateA_p31
[2026-11-08 04:22:59] P31 闸门 A(vs master): [summary] seats=[0, 1, 2, 3]  n=300  diff=+4.389±0.371  z=+11.84  p≈0.0000  符号 216:81
[2026-11-08 04:22:59] P31 不晋升（未显著优于现最佳），ckpt 留档供终评 best-of
[2026-11-08 04:22:59] R31 加速轮完成（PPO-only：无标注/无训练，含闸门 D/A）
[2026-11-08 04:22:59] P32 PPO 链起点 = exit_p31.pt  hands=1920 max_states=120000
[2026-11-08 04:22:59] $ /Users/ligf9/zhuoji-mahjong-ai/.venv/bin/python train_ppo.py --init /Users/ligf9/zhuoji-mahjong-ai/models/exit_p31.pt --out /Users/ligf9/zhuoji-mahjong-ai/models/exit_p32.pt --hands 1920 --workers 4 --threads 2 --max-states 120000 --epochs 2 --lr 5e-5 --seed0 1220000  → ppo_p32.log
[2026-11-08 04:34:06]   rc=0  667s  ppo_p32
[2026-11-08 04:34:06] P32 PPO 完成 init=exit_p31.pt → exit_p32.pt {"hands": 1920, "states": 89591, "A_std": 8.113615989685059, "ret_std": 11.62476478256657, "kl": [0.00451033119195407, 0.006647277711558023], "vl": 1.0158315122979027, "ent": 1.031415219306946, "gnorm": 1.3215753912074226, "clipfrac": 0.06532540033438376, "ent_rollout": 1.1477663933523166, "kl_stop": false, "secs": 665.8}
[2026-11-08 04:34:06] $ /Users/ligf9/zhuoji-mahjong-ai/.venv/bin/python eval_pairs.py --a /Users/ligf9/zhuoji-mahjong-ai/models/exit_p32.pt --b /Users/ligf9/zhuoji-mahjong-ai/models/exit_best.pt --mode greedy --seeds 300 --seed0 710000 --workers 6 --out /Users/ligf9/zhuoji-mahjong-ai/runs/gates/p32_gateD.jsonl  → gateD_p32.log

## 2026-11-08 04:36（自检 #38，7×24 第 38 次 · S2 中后段：D 门 7/7 符号全正，链上边际仍在爬）
- 结论：**健康、产出符合预期（平台期 + 弱阳性延续）、主线零干预**。唯一驱动 PID 19488（PPID 19486，00:51:10 起）单实例，
  lock 心跳 1~2s；当前阶段 = **R32（S2 第 8 个加速轮，P32 init=exit_p31，04:22:59 起，ep1/2 steps=350）**；panic/NaN/超时 0；
  磁盘 304Gi；Web `/`=200、`/api/meta` master=dushan_master、models[0].champion=false、冠军表仅 web_best；
  看门狗 19486 活（00:51 后无新拉起）；`runs/early_stop`/`plan_b_enabled`/`ppo_disabled`/`rapid_abort` 全部不存在。
- **S2 部分结算（P25–P31，n=7）**：gateA 均值 **+4.425（SE 0.151）** vs S1 +4.323 ⇒ **+0.102（<1SE，噪声内）**；
  S2 内部 OLS slope **−0.0114/轮**（|slope|<0.10 ⇒ §6.43.2 S1 辅证成立）。
- **D 门 7/7 点估计全为正**：+0.081 / **+0.861(p=.0203)** / **+0.698(p=.0236)** / +0.262 / +0.511 / +0.378 / +0.301，
  均值 **+0.442**；其中 P28–P31 是**对晋升后的 p27** 比（+0.363 均值）⇒ 链上仍在缓慢上爬而非停滞。
  符号检验 7/7 单侧 p≈0.0078，但**固定池 710000 已比较十余次、且 exit_best 为移动靶**（winner's curse），
  **不得据此改配方/加预算/挑显著/宣称可累积**；确认路径仍只有三条预登记全新池（740000+600 / 750000+600 / 终评 1000-seed）。
- **熵通道（重要观察，非判据改动）**：ent_rollout **1.984→1.211（P24→P31，−39.0%）**，最近 6 轮相对变化 −29.5%，
  仍远未触及 §6.43.2 的 S2 辅证阈值（<0.80 或 <5%）⇒ **R36 预期 verdict=continue**（与 §6.48.5 预测一致）。
  若按当前斜率（≈−5%/轮）继续，ent_rollout 将在 R44–R48 逼近 0.80 ⇒ **R48 是比 R36 更可能的早停边界**（预测，不是判读）。
- 早停：**未到边界**（最早 R36）。`early_stop_guard --boundary 36` 需 gateA_r36 落盘后才能判，本轮不执行、不 --arm。
- **E1：仍无结果，按 §6.47 处置维持「终评窗口补跑」**，本轮不重跑（rc=3 机会已用掉一次，判据不变）。
- 排期（实测 16.7min/轮仍成立）：R32 收口 ≈04:36 → R33 04:36 / R34 04:53 / R35 05:09 → **R36 annotate 起 ≈05:25** →
  4×25k 落盘 ≈07:45 → train ≈07:45–08:25（复现测量窗口）→ A ≈08:26 → C ≈08:27–08:53 → D ≈08:55 ⇒ **边界判读 ≈09:0x–09:2x**。
- **动作（按最优推荐，已执行）**：新建一次性 automation 于 **07:47** 触发 §6.39.5 复现测量（`--wait-log train_r36.log --wait-max-min 25`
  兜住窗口漂移；annotate 未退则等日志，天然避免并发），只跑该测量、判据一字不改、错过则按 §6.47.5 在 gateD_r36 后补跑并标注偏差。
- 零干预：未重启驱动/看门狗/Web、未改超参、未注册冠军、未动 dushan_master.pt、未跑任何吃 CPU 的探针。

**[04:36 补记]** P32 已收口：states **89,591**、665.8s、kl[.0045,.0066]、clip .065、gnorm 1.322、ent 1.031、ent_rollout **1.148**、vl 1.016 —— 全部在预登记阈值内，链 p31→p32 连续；熵仍在 −5%/轮通道内下行（与本节 6.49.3 外推一致）。R32 闸门 D/A 随后 ~4min 落盘，R33 自动接续。
[2026-11-08 04:37:07]   rc=0  182s  gateD_p32
[2026-11-08 04:37:07] P32 闸门 D(vs 现最佳): [summary] seats=[0, 1, 2, 3]  n=300  diff=+0.728±0.345  z=+2.11  p≈0.0348  符号 148:134  → 通过
[2026-11-08 04:37:07] $ /Users/ligf9/zhuoji-mahjong-ai/.venv/bin/python eval_pairs.py --a /Users/ligf9/zhuoji-mahjong-ai/models/exit_p32.pt --b /Users/ligf9/zhuoji-mahjong-ai/models/dushan_master.pt --mode greedy --seeds 300 --seed0 710000 --workers 6 --out /Users/ligf9/zhuoji-mahjong-ai/runs/gates/p32_gateA.jsonl  → gateA_p32.log
[2026-11-08 04:38:22]   rc=0  74s  gateA_p32
[2026-11-08 04:38:22] P32 闸门 A(vs master): [summary] seats=[0, 1, 2, 3]  n=300  diff=+4.573±0.392  z=+11.66  p≈0.0000  符号 222:75
[2026-11-08 04:38:22] $ /Users/ligf9/zhuoji-mahjong-ai/.venv/bin/python -c import torch,sys;sys.path.insert(0,'.');from zhuoji.net import build_from_ckpt;from zhuoji.forward import ensure_onnx;ck=torch.load('/Users/ligf9/zhuoji-mahjong-ai/models/exit_best.pt',map_location='cpu',weights_only=False);net,_=build_from_ckpt(ck);p=ensure_onnx(net,stem='exit_best');print('onnx:',p)  → export_exit_best.log
[2026-11-08 04:38:23]   rc=0  1s  export_exit_best
[2026-11-08 04:38:23] 晋升：exit_p32.pt → exit_best.pt
[2026-11-08 04:38:23] R32 加速轮完成（PPO-only：无标注/无训练，含闸门 D/A）
[2026-11-08 04:38:23] P33 PPO 链起点 = exit_p32.pt  hands=1920 max_states=120000
[2026-11-08 04:38:23] $ /Users/ligf9/zhuoji-mahjong-ai/.venv/bin/python train_ppo.py --init /Users/ligf9/zhuoji-mahjong-ai/models/exit_p32.pt --out /Users/ligf9/zhuoji-mahjong-ai/models/exit_p33.pt --hands 1920 --workers 4 --threads 2 --max-states 120000 --epochs 2 --lr 5e-5 --seed0 1230000  → ppo_p33.log
[2026-11-08 04:49:15]   rc=0  653s  ppo_p33
[2026-11-08 04:49:15] P33 PPO 完成 init=exit_p32.pt → exit_p33.pt {"hands": 1920, "states": 88414, "A_std": 8.05906867980957, "ret_std": 11.502456712227465, "kl": [0.004696490332407532, 0.006019176908222301], "vl": 1.0033501917227154, "ent": 0.9854615660588866, "gnorm": 1.3126961137518027, "clipfrac": 0.0660808893092881, "ent_rollout": 1.078852163993056, "kl_stop": false, "secs": 651.5}
[2026-11-08 04:49:15] $ /Users/ligf9/zhuoji-mahjong-ai/.venv/bin/python eval_pairs.py --a /Users/ligf9/zhuoji-mahjong-ai/models/exit_p33.pt --b /Users/ligf9/zhuoji-mahjong-ai/models/exit_best.pt --mode greedy --seeds 300 --seed0 710000 --workers 6 --out /Users/ligf9/zhuoji-mahjong-ai/runs/gates/p33_gateD.jsonl  → gateD_p33.log
[2026-11-08 04:52:04]   rc=0  168s  gateD_p33
[2026-11-08 04:52:04] P33 闸门 D(vs 现最佳): [summary] seats=[0, 1, 2, 3]  n=300  diff=-0.227±0.277  z=-0.82  p≈0.4134  符号 124:139  → 不通过
[2026-11-08 04:52:04] $ /Users/ligf9/zhuoji-mahjong-ai/.venv/bin/python eval_pairs.py --a /Users/ligf9/zhuoji-mahjong-ai/models/exit_p33.pt --b /Users/ligf9/zhuoji-mahjong-ai/models/dushan_master.pt --mode greedy --seeds 300 --seed0 710000 --workers 6 --out /Users/ligf9/zhuoji-mahjong-ai/runs/gates/p33_gateA.jsonl  → gateA_p33.log
[2026-11-08 04:53:18]   rc=0  74s  gateA_p33
[2026-11-08 04:53:18] P33 闸门 A(vs master): [summary] seats=[0, 1, 2, 3]  n=300  diff=+4.474±0.391  z=+11.44  p≈0.0000  符号 219:79
[2026-11-08 04:53:18] P33 不晋升（未显著优于现最佳），ckpt 留档供终评 best-of
[2026-11-08 04:53:18] R33 加速轮完成（PPO-only：无标注/无训练，含闸门 D/A）
[2026-11-08 04:53:18] P34 PPO 链起点 = exit_p33.pt  hands=1920 max_states=120000
[2026-11-08 04:53:18] $ /Users/ligf9/zhuoji-mahjong-ai/.venv/bin/python train_ppo.py --init /Users/ligf9/zhuoji-mahjong-ai/models/exit_p33.pt --out /Users/ligf9/zhuoji-mahjong-ai/models/exit_p34.pt --hands 1920 --workers 4 --threads 2 --max-states 120000 --epochs 2 --lr 5e-5 --seed0 1240000  → ppo_p34.log
[2026-11-08 05:04:03]   rc=0  645s  ppo_p34
[2026-11-08 05:04:03] P34 PPO 完成 init=exit_p33.pt → exit_p34.pt {"hands": 1920, "states": 87821, "A_std": 8.080801010131836, "ret_std": 11.44331591658059, "kl": [0.004713738582511216, 0.00596971830592063], "vl": 1.005685661854439, "ent": 0.9456876231314137, "gnorm": 1.328430472764858, "clipfrac": 0.06883542739981136, "ent_rollout": 1.0217064913933525, "kl_stop": false, "secs": 644.1}
[2026-11-08 05:04:03] $ /Users/ligf9/zhuoji-mahjong-ai/.venv/bin/python eval_pairs.py --a /Users/ligf9/zhuoji-mahjong-ai/models/exit_p34.pt --b /Users/ligf9/zhuoji-mahjong-ai/models/exit_best.pt --mode greedy --seeds 300 --seed0 710000 --workers 6 --out /Users/ligf9/zhuoji-mahjong-ai/runs/gates/p34_gateD.jsonl  → gateD_p34.log
[2026-11-08 05:06:51]   rc=0  168s  gateD_p34
[2026-11-08 05:06:51] P34 闸门 D(vs 现最佳): [summary] seats=[0, 1, 2, 3]  n=300  diff=-0.218±0.295  z=-0.74  p≈0.4596  符号 125:148  → 不通过
[2026-11-08 05:06:51] $ /Users/ligf9/zhuoji-mahjong-ai/.venv/bin/python eval_pairs.py --a /Users/ligf9/zhuoji-mahjong-ai/models/exit_p34.pt --b /Users/ligf9/zhuoji-mahjong-ai/models/dushan_master.pt --mode greedy --seeds 300 --seed0 710000 --workers 6 --out /Users/ligf9/zhuoji-mahjong-ai/runs/gates/p34_gateA.jsonl  → gateA_p34.log
[2026-11-08 05:08:08]   rc=0  77s  gateA_p34
[2026-11-08 05:08:08] P34 闸门 A(vs master): [summary] seats=[0, 1, 2, 3]  n=300  diff=+4.471±0.446  z=+10.02  p≈0.0000  符号 215:84
[2026-11-08 05:08:08] P34 不晋升（未显著优于现最佳），ckpt 留档供终评 best-of
[2026-11-08 05:08:08] R34 加速轮完成（PPO-only：无标注/无训练，含闸门 D/A）
[2026-11-08 05:08:08] P35 PPO 链起点 = exit_p34.pt  hands=1920 max_states=120000
[2026-11-08 05:08:08] $ /Users/ligf9/zhuoji-mahjong-ai/.venv/bin/python train_ppo.py --init /Users/ligf9/zhuoji-mahjong-ai/models/exit_p34.pt --out /Users/ligf9/zhuoji-mahjong-ai/models/exit_p35.pt --hands 1920 --workers 4 --threads 2 --max-states 120000 --epochs 2 --lr 5e-5 --seed0 1250000  → ppo_p35.log
[2026-11-08 05:18:40]   rc=0  632s  ppo_p35
[2026-11-08 05:18:40] P35 PPO 完成 init=exit_p34.pt → exit_p35.pt {"hands": 1920, "states": 85967, "A_std": 8.121204376220703, "ret_std": 11.30183659484894, "kl": [0.0044513235713696175, 0.006250985642206312], "vl": 1.0169307028963452, "ent": 0.9108734478553137, "gnorm": 1.316216304188683, "clipfrac": 0.061332960635246264, "ent_rollout": 0.9640558700586014, "kl_stop": false, "secs": 631.2}
[2026-11-08 05:18:40] $ /Users/ligf9/zhuoji-mahjong-ai/.venv/bin/python eval_pairs.py --a /Users/ligf9/zhuoji-mahjong-ai/models/exit_p35.pt --b /Users/ligf9/zhuoji-mahjong-ai/models/exit_best.pt --mode greedy --seeds 300 --seed0 710000 --workers 6 --out /Users/ligf9/zhuoji-mahjong-ai/runs/gates/p35_gateD.jsonl  → gateD_p35.log
[2026-11-08 05:22:01]   rc=0  201s  gateD_p35
[2026-11-08 05:22:01] P35 闸门 D(vs 现最佳): [summary] seats=[0, 1, 2, 3]  n=300  diff=+0.108±0.305  z=+0.35  p≈0.7240  符号 132:149  → 不通过
[2026-11-08 05:22:01] $ /Users/ligf9/zhuoji-mahjong-ai/.venv/bin/python eval_pairs.py --a /Users/ligf9/zhuoji-mahjong-ai/models/exit_p35.pt --b /Users/ligf9/zhuoji-mahjong-ai/models/dushan_master.pt --mode greedy --seeds 300 --seed0 710000 --workers 6 --out /Users/ligf9/zhuoji-mahjong-ai/runs/gates/p35_gateA.jsonl  → gateA_p35.log
[2026-11-08 05:23:01]   rc=0  60s  gateA_p35
[2026-11-08 05:23:01] P35 闸门 A(vs master): [summary] seats=[0, 1, 2, 3]  n=300  diff=+4.696±0.431  z=+10.91  p≈0.0000  符号 222:75
[2026-11-08 05:23:01] P35 不晋升（未显著优于现最佳），ckpt 留档供终评 best-of
[2026-11-08 05:23:01] R35 加速轮完成（PPO-only：无标注/无训练，含闸门 D/A）
[2026-11-08 05:23:01] $ /Users/ligf9/zhuoji-mahjong-ai/.venv/bin/python -c import torch,sys;sys.path.insert(0,'.');from zhuoji.net import build_from_ckpt;from zhuoji.forward import ensure_onnx;ck=torch.load('/Users/ligf9/zhuoji-mahjong-ai/models/exit_best.pt',map_location='cpu',weights_only=False);net,_=build_from_ckpt(ck);p=ensure_onnx(net,stem='exit_best');print('onnx:',p)  → export_exit_best.log
[2026-11-08 05:23:02]   rc=0  1s  export_exit_best
[2026-11-08 05:23:02] $ /Users/ligf9/zhuoji-mahjong-ai/rust_engine/target/release/annotate --student /Users/ligf9/zhuoji-mahjong-ai/models/onnx/exit_best.onnx --teacher models/onnx/dushan_master.onnx --student-enc 3 --student-device cpu --targets 100000 --threads 4 --out /Users/ligf9/zhuoji-mahjong-ai/runs/exit_r36 --seed0 960000  → annotate_r36.log

## 2026-11-08 05:36（自检 #39，7×24 第 39 次 · S2 收官前最后一轮已完成：P35 为 S2 最高 A 门；R36 完整 DAgger 标注中）
- 结论：**健康、产出符合预期（S2 平台期 + 弱阳性延续）、主线零干预**。驱动 19488（PPID 19486 看门狗）单实例，
  lock 心跳 <2s；当前阶段 = **R36 完整 DAgger 的 annotate**（05:23:02 起，`--targets 100000`＝4 分片×25000，
  现 ~2.7k/25k 每片、吞吐 **11.72–11.82k/h·worker**，四片齐头）；panic/NaN/超时 0；磁盘 304Gi；
  `runs/early_stop`、`runs/gates/plan_b_enabled`、`runs/ppo_disabled`、`runs/rapid_abort` 全部不存在。
- **S2（P25–P35，n=11）部分结算**：闸门 A 均值 **+4.472**（轮间 SD 0.169 / SE 0.051；单轮 SE≈0.40），
  vs S1 +4.323 ⇒ **+0.149**；OLS slope **+0.0151/轮**（|slope|<0.10 ⇒ §6.43.2 S1 辅证成立）。
  注意：轮间 SD 只有 0.17 而单轮 SE 是 0.40 ⇒ **同一 300-seed 池共用导致的强相关**，+0.149 只能表述为
  「平台中心略抬升」，**不得当显著性或当增益幅度引用**（真正的增益幅度只能由全新池测量给）。
- **闸门 D（P25–P35，11 轮）**：+0.081 / **+0.861(p=.0203)** / **+0.698(p=.0236)** / +0.262 / +0.511 / +0.378 /
  +0.301 / **+0.728(p=.0348)** / −0.227 / −0.218 / +0.108 ⇒ **9/11 点估计为正、3 次晋升**（P26/P27/P32），
  均值 **+0.317**；对晋升后 p27 的 8 轮均值 **+0.230**。`exit_best ≡ exit_p32`（04:38:23 晋升，md5 待终评核）。
  统计纪律不变：固定池 + 移动靶（winner's curse）⇒ 只记「弱阳性延续」，**不作决策依据**、不据此改配方。
- **健康度**（`health_watch --last 12`）：12/12 在预登记阈值内 —— kl .0059–.0087、clip .061–.092、gnorm 1.30–1.38、
  ent .911–1.604、states 85,967–108,270、secs 631–791（≈同配置中位 1.0×）。
  新增 1 条 **WARN：P35 vl 连续三轮上升 1.0034→1.0057→1.0169（+1.3%）** ⇒ 幅度极小、同期 A 门 +4.696 为 S2 最高，
  与此前 P24/P25 同类告警（P27 自行解除）同型 ⇒ **仅观察、零动作**（不改 lr/τ、不重启）。
- **熵通道**：ent_rollout 1.984→0.964（P24→P35，**−51.4%**），最近 6 轮 **−29.6%** ⇒ 距 §6.43.2 的 S2 阈值
  （<0.80 **或** 最近 6 轮 <5%）**仍远** ⇒ **R36 预期 verdict=continue**（预测，不是判读）。
  按 ≈−5.4%/轮 斜率，0.80 将在 **P39–P40** 触及；但只要仍在快速下降，S2 就不成立 ⇒
  「熵触底后横盘」才会让 S2 成立，**R48 仍是比 R36 更可能的早停边界**。
- Web/看门狗：`/`=200；`/api/meta` master=**dushan_master**、models[0].champion=**false**、冠军表仍仅 web_best；
  用户在线（sid 3b834f05，search128:web_best）⇒ **未重启 Web**；看门狗 19486 活、05:23 后无新拉起。
- **排期（本次实测重算，比 §6.49.4 提前约 1h）**：annotate 05:23:02 起 + 11.76k/h·片 ⇒ 4×25k 落盘
  **≈07:30–07:32** → train_exit ≈**07:32–08:12**（R24 为 3363s 但含 E1 抢 MPS；无争抢按历史 2032–2555s）
  → gateA_r36 ≈**08:14** → gateC_r36 ≈08:16–08:42（26min）→ gateD_r36 ≈**08:44** ⇒ **边界判读窗口 ≈08:45–09:15**。
- **下一步（已锁定，判据一字不改）**：① **07:47 一次性 automation** 已在册，跑 §6.39.5 复现测量
  （`p20 vs r12`，池 740000+600，`--wait-log train_r36.log --wait-max-min 25` 兜窗口漂移；07:35 巡检**不得重复跑**）；
  ② `gateA_r36` 落盘后 M 测量（池 750000+600，A=S2 最强〔当前 P35 +4.696，R36 若更高则改 R36〕、B=S1 最强 P20 +4.962），
  **必须避开 gateC/gateD 的 eval_pairs**（6 worker 互斥）；③ `early_stop_guard.py --boundary 36`，**仅 verdict=early_stop 才 `--arm`**；
  ④ 方案 B 三项（复现同号 p<0.10 / 段内 p25…p36 无缺号 / 链末端存在）⇒ 全过才写 `plan_b_enabled`，最早边界自动推 R48。
- **E1**：维持 §6.47.4 处置 —— **终评窗口补跑**，判据不变；本轮不重跑、R36 train 段只留给复现测量。
- 零干预：未重启驱动/看门狗/Web、未改超参、未注册冠军、未动 `dushan_master.pt`、未跑任何吃 CPU 的探针。


---

## [06:2x 追加 · 用户线上 519 局复盘，纯只读诊断] 详见 `docs/python_rust_usage_audit.md` §8

用户问「为什么一名 N128 玩家得分远超我（最强模型）与另两个同模型玩家？是不是没用相对
名次作训练目标？」**结论：不是训练目标问题，也不是模型弱，是评估口径 + 零和随机游走。**
数据源 `web/history/records.jsonl`（519 局，23:39→05:21，稳定配置 518 局），全程离线读记录。

**配置**：我方 `search128:web_best`（R07 冠军快照），三家对手 `search128:master`（同 N128 档）；
`dealer=0` 恒 519/519 局，494/518 局身份→座位固定 ⇒ **身份与座位共线，结构上测不出强弱**；
519/519 局 `sum(deltas)=0` 严格零和。

**关键数字**：按身份累计 王昭君(seat0) +606 / 云天明(seat1) +171 / 大乔 −295 / 妲己 −482。
- 王−我 逐局配对 **+0.768±1.019, p=0.451 不显著**（435 分差仅 0.82 SD）；
- 按权重 web_best vs 三家 master 均值 +0.440±0.824 p=0.593（方向为正、ns）；
- **唯一显著量**：同策略同档的 seat0 − (seat2+seat3)/2 = **+1.967±0.855, p=0.0214** ⇒ 座位效应；
- 反事实 2000 次随机四座旋转：「对手最高 − 我」中位数 **+299**，5~95% [−461,+1022]，
  观测 435 分在第 61.7 百分位，**P(纯随机≥435)=0.384** ⇒ 零和四玩家取极值的统计常态；
- 前 24 局我方坐庄家位则我方领先(+40)，后 494 局换成王昭君坐庄家位则王领先(+606)。

**名次目标问题**：训练目标确实全是 cardinal 分差（教师 Q / Φ=8·V_θ / 闸门 Δ±SE），但本作
严格零和**按分结算**（番数/鸡分/杠分/跟打赔付），非日麻顺位点 ⇒ 分差是信息更全的正确目标；
名次是分差的秩变换，本数据名次排序与分差排序一致，换目标不改变结论。厚尾 SD 14.0 vs 均分
0.3~1.2 ⇒ 名次/裁剪可作**辅助降方差**方向，登记 **D3 之后 E2 影子对照**（同 §6.38 E1 规格），
**S2/D2 配方一字不改**。

**改善清单**：P0 四座旋转 + 修 dealer 恒 0（需动服务端并蓝绿重启，等用户点头）、改看逐局
配对 Δ±SE 而非累计分、走 `eval_pairs.py` 冻结池（300 seed×四座 SE≈0.55，线上随机池需 ~1.1
万局）；P1 `champions.json` 只有 R07 的 web_best，而 `exit_best`(p35, 闸门 A +4.696±0.431
p≈0) 已落盘 ⇒ 用户线上选不到当前最强权重，建议复制成固定快照后 `register_champion.py`
只增不替（无需重启 Web；须先做 web_best vs exit_best 冻结头对头才能宣称更强）；P2 终评定最强。

**防腐（自检必读）**：这批 519 局**不构成**模型弱 / 配方失败 / 早停证据，**禁止**据此改配方、
加重训练、触发 §6.43 早停或写 `rapid_abort`/`ppo_disabled`。归档时 R36 标注在跑，本轮零对局、
零 CPU 探针、主链与线上配置零改动。
