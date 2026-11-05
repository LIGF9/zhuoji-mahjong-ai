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
