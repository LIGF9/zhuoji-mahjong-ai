"""M1 分段训练的标准起动入口：内置各段配方 + 起动前检查 + 守护进程一并拉起。

为什么需要它
------------
分段执行（S1/S2/S3）每段有 20 来个参数，其中只有 4 个随段变化（`--init`/`--out`/
`--anchor-warmup`/超参与 lr）。手敲一次是运气，敲三次迟早漏一个 —— 而漏掉
`--anchor-warmup 0` 这种错误**不会报错**，只会静默烧掉 200 轮（≈2.8 h）算力。

所以把配方固化在代码里，并把"起动前检查"做成硬闸（磁盘 / 残留进程 / 上游检查点）。

用法::

    python scripts/m1_launch_segment.py --segment s1 --dry-run   # 只打印，不起
    python scripts/m1_launch_segment.py --segment s1             # 检查 + 起跑（**跟随**调用方）
    python scripts/m1_launch_segment.py --segment s1 --detach    # 脱离会话（仅真实终端可用）
    python scripts/m1_launch_segment.py --segment s2 --workers 8 # 内存吃紧时下调

⚠️ 起动方式怎么选（2026-10-15 实测定案）
----------------------------------------
**沙箱型执行环境（如 WorkBuddy 的工具调用）在每次调用结束时无条件清理整个进程树**，
与子进程用什么姿势起动**无关**。2026-10-15 逐档实测（子进程每 5 s 写一行 tick）：

| 起动方式 | 命令返回后 |
|---|---|
| `DETACHED` + `CREATE_BREAKAWAY_FROM_JOB` + `CREATE_NEW_PROCESS_GROUP` | 一个字节都没写出 |
| `DETACHED` + `CREATE_NEW_PROCESS_GROUP` | 一个字节都没写出 |
| 无 flags | 一个字节都没写出 |
| `nohup … & disown` | 只活到写了第 1 行 |
| `setsid nohup … &` | 文件从未创建 |
| `Popen`（父进程**继续存活** 120 s） | ✅ 正常，跑出 iter 1 |

结论：**"脱离会话自动保命"在本环境不存在**。所以：

- 默认 `--foreground`：训练进程跟随调用方，**必须把这条命令挂在平台自己的后台
  任务机制里** —— 命令行不结束，进程就不会被清；
- `--detach` 只留给**真实终端**（用户自己开 cmd/bash，关掉终端仍要继续跑）。
  在沙箱里用它会**静默零输出即死**。

配套的第二个坑（同一天发现）：脱离/跟随都**不要**依赖"父进程传下来的 stdout 句柄"。
`with log.open(...)` 块一结束父进程就把句柄关了；虽然 Windows 上子进程持有的是
独立内核句柄、关掉父进程的 FD 不影响它（实测 t=35 关句柄后训练照样产出 iter 1），
但一旦父进程真的退出，句柄就成为唯一悬空引用。现在训练进程用 `--log-file`
**自己**打开日志，父进程只负责写头部，任何一侧退出都不影响另一侧。

起动后会写：
- `reports/_{run}.out`          训练 stdout（判读脚本读它补字段）
- `reports/_res_monitor_{run}.csv` 资源打点（**独立**进程，被冻结时仍按 wall clock 写）
- `reports/_{run}.pid`          训练主进程 pid
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable

# ---- 公共配方（S1–S3 共用；与 docs/P3-M1训练执行计划.md §0 保持一致）----
COMMON = [
    "--rules", "dushan",
    "--arch", "m1", "--encoder", "v3", "--m1-preset", "m1",
    "--games-per-iter", "192", "--epochs-per-iter", "1", "--batch", "512",
    "--past-prob", "0.3", "--kl-coef", "0.02", "--ent-coef", "0.01",
    "--eval-every", "20", "--eval-rounds", "40", "--eval-anchor", "dushan_master",
    "--select", "guard", "--select-metric", "vs_anchor",
    "--save-every", "20", "--snapshot-pool", "dushan_master", "--snapshot-every", "16",
    "--threads", "24", "--vectorized", "--seed", "2026",
    "--pool-mem-pct", "75", "--pool-rebuild-every", "150",
]

# ---- M2 配方（A1–A8；见 docs/P3-M2-追赶并超越S11训练计划.md §4/§5）----
# 与 COMMON 的差异都用**显式参数覆盖**表达，靠 argparse 的"后者覆盖前者"生效，
# 而不是在这里删改 COMMON —— 保证 S1–S3 的历史配方一个字都不变（可复现）。
M2_COMMON = [
    # A1 熵比例控制：H* = 0.95·H₀（★ 不是计划里的 0.75 —— 实测熵的自然稳态只有
    # 段首的 91.6%（H₀=1.0253 → 稳态 0.9389），0.75·H₀=0.769 在数学上**不可达**
    # → H>H* 恒成立 → ec 只会往下调、而起点就是下限 0.01 → 60/60 轮零变化（缺陷 C5）。
    # 0.95 → H*≈0.974，落在稳态上方 0.036 > 死区 0.03，控制器真正接管。
    # 标定：scripts/ent_gain_probe.py 实测 dH/d(ec)≈1.18 → 0.95 对应 ec≈0.05
    #（上限 0.5 的 10%、下限 0.01 的 5 倍），环路增益 η·g·ec≈0.03 ≪ 1 无震荡。
    #
    # η 从计划的 4.0 改成 0.5：η=4 时 ec 在 clamp 两端继电器式来回（2–4 轮撞一次端点、
    # 67% 的轮次贴着端点），等于给训练注入高频探索噪声。标定见
    # scripts/test_coef_controller.py 的 T8（玩具被控对象 + η 扫描）。
    "--ent-target-ratio", "0.95", "--ent-baseline-iters", "25",
    "--ent-eta", "0.5", "--ent-deadzone", "0.03",
    # A2 ε-均匀探索：3% → 1%
    "--explore-eps", "0.03", "--explore-eps-end", "0.01", "--explore-anneal-iters", "300",
    # A3 滚动 KL 锚 + **率**判据
    # --kl-target 的单位是「每轮漂移率」（= kl ÷ 锚点年龄），不是裸的 kl：
    # 锚点每 150 轮才刷新，裸 kl 是"距锚点的距离"、会随锚点变旧单调增大，
    # 实测率却是稳的（0.023/0.031/0.028/0.025 每轮）。用距离判据会在 25 轮内
    # 把 kl_coef 顶到 clamp 上限 0.5，把模型锁在过期锚上。
    "--kl-target", "0.05", "--kl-relax-ratio", "0.25",
    "--kl-anchor-rolling", "150", "--kl-anchor-alpha", "0.3",
    # A4 KLT 教师蒸馏（β 上限由分段指定）
    "--master-ckpt", "dushan_master",
    # A6 GAE + 按座位标准化
    "--adv-mode", "gae", "--gae-lambda", "0.95", "--gamma", "1.0", "--adv-norm", "seat",
    # A7 吞吐：KL 锚降频 + 池重建走进程树 RSS + 快照留 4 份 + eval 降频提档
    # --pool-mem-pct 必须抬到 92（COMMON 里是 75）：它只是"RSS 判据没触发时的次级兜底"
    # 系统口径，而本机桌面基线已占 63.7%（31.8 GB 里 20.2 GB），75% 只剩 ~4 GB 余量
    # → 正是计划 R6 那次"重建风暴"的成因。M2 的主判据是 --pool-rss-gb 16。
    "--kl-anchor-every", "4", "--pool-rss-gb", "16", "--pool-mem-pct", "92",
    "--pool-rebuild-every", "250",
    "--snapshot-keep", "4", "--eval-every", "40", "--eval-rounds", "120",
    # A5 弱规则教师信息量低（M2-0 已证分歧主要在"跟会打牌的对手博弈"）
    "--teacher-prob", "0.0",
    # 快照池改成由 --master-ckpt 承担一等对手，旧的 1/N 均分写法作废
    "--snapshot-pool", "",
]

# ---- 分段配方（只有这几项随段变化）----
SEGMENTS = {
    "s1": dict(iters=800, init="models/dushan_master.pt", out="models/dushan_m1_s1.pt",
               warmup=200, teacher_prob=0.15, temperature=1.2, lr=1.2e-5,
               needs=""),
    "s2": dict(iters=800, init="models/dushan_m1_s1.pt", out="models/dushan_m1_s2.pt",
               warmup=0, teacher_prob=0.08, temperature=1.0, lr=1.2e-5,
               needs="models/dushan_m1_s1.pt"),
    "s3": dict(iters=850, init="models/dushan_m1_s2.pt", out="models/dushan_m1_s3.pt",
               warmup=0, teacher_prob=0.05, temperature=1.2, lr=6e-6,
               needs="models/dushan_m1_s2.pt"),
    # ---- M2-A：修复 + 能力移植（600 轮 / ≈6.8 h）----
    "s4a": dict(iters=600, init="models/dushan_m1_s3.pt", out="models/dushan_m2_a.pt",
                warmup=0, teacher_prob=0.0, temperature=1.2, lr=1.2e-5,
                needs="models/dushan_m1_s3.pt", m2=True,
                extra=["--klt-beta", "0.4", "--klt-anneal-iters", "300",
                       "--master-seat-prob", "0.30",
                       "--lr-min", "4e-6", "--save-every", "20"]),
    # ---- M2-B：超越（1200 轮 / ≈12–16 h）----
    # 名人堂池：M2-A 的产出 + 现任 master（一等对手权重下调到 0.20，奖励自博弈爬升）。
    # ⚠️ G1 判读后若 M2-A 明显强于 master，应把 --snapshot-pool 换成
    #    `dushan_m2_a,dushan_m1_s3,dushan_master`（按 G1 实测强度排序）。
    "s4b": dict(iters=1200, init="models/dushan_m2_a.pt", out="models/dushan_m2_b.pt",
                warmup=0, teacher_prob=0.0, temperature=1.2, lr=1.2e-5,
                needs="models/dushan_m2_a.pt", m2=True,
                extra=["--klt-beta", "0.0",
                       "--master-seat-prob", "0.20",
                       "--snapshot-pool", "dushan_m1_s3",
                       "--lr-min", "3e-6", "--save-every", "20"]),
    # ---- M2-B2：用 **A 段已验证配方** 重跑 B（600 轮 / ≈7 h）----
    # 起因（2026-10-15 C8）：s4b 跑到 500 轮时被判读 **比 A 段末低 2.18 ± 0.65（3.3σ）**，
    # 内部评测自段首起 480 轮平坦无改善（reports/M2B_塌陷诊断与配方修正.md）。
    # 关键洞察：**A 段 it300→it600 本身就是 β=0 的 300 轮、且涨了 +0.36**
    # ⇒ "撤教师"不是病因，**对手场强度与熵目标**才是。s4b 相对 A 段的三处退化：
    #   ① `--snapshot-pool`（★ 是**常驻对手模型**，不是自身快照！）由「**空**」
    #      换成更弱的 `dushan_m1_s3`；② master 座位比 0.30 → 0.20；
    #      ⚠️ 2026-10-15 复核更正：原文写的"A 段用 dushan_master"是**误读** ——
    #      s4a 的命令行末尾是 `--snapshot-pool`（空值，argparse 取最后一个），
    #      A 段实际是**纯自博弈**，日志无 `[pool] 常驻对手` 行为证。
    #   ③ 熵目标随"续训模型更尖"漂移：H* = 0.95×H₀，而 H₀ 由 1.026 掉到 0.880
    #      → H* 0.975 → 0.836（低 0.14 nats，政策被推得更尖）。
    # 因此 s4b2 = A 段配方原样 + β=0：池换回 master、座位比回 0.30、
    # **用 `--ent-baseline 1.026` 把 H₀ 钉成绝对值**（不必改代码，H* 复位到 0.975）、
    # lr-min 回 4e-6、ε 只做轻退火（0.015→0.01，不在续训段重开 3% 探索）。
    # ⚠️ 2026-10-15 18:30 实测结果：**s4b2 同样在 it0300 判负**
    # （vs_master −2.402 ± 0.393，比 A 段末低 1.356 ± 0.596 = 2.28σ），已自动止损。
    # 全量复核后确认：熵目标（两段 H\* 都是 0.975，日志逐字）与 β（A 段 it301 起
    # 即为 0）**都不是**病因；`--snapshot-pool` 只影响约 2% 的 master 出场
    # （见 zhuoji/vcollect.py::_make_seats —— 每局只抽一次分支 + arange 新近加权，
    # 且 snapshot_keep 裁剪只作用于自快照列表），量级解释不了 −1.34。
    # 剩余主导嫌疑 = **lr 余弦被冷重启**：A 段收敛区跑在 8e-6→4e-6，而 s4b2 从
    # 1.2e-5 起步，整个前半段是 A 段收敛区的 1.5–3 倍步长。
    # ⇒ 下一段（s4b3）应把 lr / ent_coef / kl_coef 按"续训档位"暖启动。
    # 详见 reports/M2B2_退化根因复核.md。
    "s4b2": dict(iters=600, init="models/dushan_m2_a.pt", out="models/dushan_m2_b2.pt",
                 warmup=0, teacher_prob=0.0, temperature=1.2, lr=1.2e-5,
                 needs="models/dushan_m2_a.pt", m2=True,
                 extra=["--klt-beta", "0.0",
                        "--master-seat-prob", "0.30",
                        "--snapshot-pool", "dushan_master",
                        "--ent-baseline", "1.026",
                        "--lr-min", "4e-6",
                        "--explore-eps", "0.015", "--explore-eps-end", "0.01",
                        "--explore-anneal-iters", "200",
                        "--save-every", "20"]),
    # ---- M2-B3：从 A 段末按**收敛区档位**续训（600 轮 / ≈7.5 h）----
    # 起因（2026-10-15 复查，见 reports/M2B2_退化根因复核.md）：s4b2 也在 it0300 判负
    # （vs_master −2.402 ± 0.393，比 A 段末低 1.356 ± 0.596 = 2.28σ），已自动止损。
    # 全量核对**两段启动命令行原文**后，把嫌疑逐个排除：
    #   * **熵目标相同**：A 段日志逐字 `[coef] 熵基线 H0=1.026（段首 25 轮） → H*=0.975`；
    #     s4b2 的 `--ent-baseline 1.026` 钉的就是这个值。熵实测 A 段 it301-600 = 0.9728、
    #     s4b2 it101-300 = 0.9772，都在带内 ⇒ **熵不是病因**（C8 的机制②被推翻）。
    #   * **β 相同**：A 段 `--klt-anneal-iters 300` ⇒ it301 起 β=0，与 s4b2 一致，
    #     且 A 段就在这个窗口里涨了 +0.353。
    #   * **`--snapshot-pool` 量级不够**：vcollect._make_seats 是**每局只抽一次分支**
    #     （r<0.30 master / 0.30≤r<0.60 past 自快照 / 否则纯自弈），pool_extra 被放在
    #     snap_sd 的**索引 0**，而权重 arange(1,n+1) 是"越新越大" ⇒ 常驻对手只拿
    #     1/15 的 past 分支质量 ≈ **全场的 2%**；snapshot_keep 裁剪也只作用于自快照列表。
    #     （上一版注释说"A 段用的是 dushan_master"是**误读** —— s4a 命令行末尾的
    #      `--snapshot-pool` 是空值，A 段实为纯自博弈，日志无 `[pool] 常驻对手` 行为证。）
    # ⇒ 剩下唯一量级够大的差异是 **lr 余弦被冷重启**：`_cur_lr` 每段都从 lr0 重来，
    #   A 段收敛区（it300→600）实际跑在 8.0e-6 → 4.0e-6，而 s4b2 的 it1 就是 1.2e-5，
    #   整个前半段在用 A 段收敛区 1.5–3 倍的步长 —— 正好对应实测到的
    #   "头 20 轮掉 0.57 / 100 轮累计 −1.19 / 之后 200 轮走平"的侵蚀形状。
    # 因此 s4b3 = **A 段收敛区的全部档位**，并把控制器状态一并暖启动
    # （s4b2 的 ec 从下限 0.010 起、12 轮内被顶到 0.066，A 段同窗口已稳在 0.039）：
    #   lr 8e-6→3e-6、ent_coef 初值 0.040、kl_coef 初值 0.016、ε 恒 0.010、池留空。
    # 预注册闸（另挂 gate_watch + m2_stoploss_watch）：
    #   it0100 `vs_master ≤ −1.60` 即停；it0300 `≤ −1.05` 即停；it0600 走 G3。
    "s4b3": dict(iters=600, init="models/dushan_m2_a.pt", out="models/dushan_m2_b3.pt",
                 warmup=0, teacher_prob=0.0, temperature=1.2, lr=8e-6,
                 needs="models/dushan_m2_a.pt", m2=True,
                 extra=["--klt-beta", "0.0",
                        "--snapshot-pool", "",
                        "--ent-baseline", "1.026",
                        "--master-seat-prob", "0.30",
                        "--lr-min", "3e-6",
                        "--ent-coef", "0.040",
                        "--kl-coef", "0.016",
                        "--explore-eps", "0.010", "--explore-eps-end", "0.010",
                        "--explore-anneal-iters", "1",
                        "--save-every", "20"]),
    # ---- M2-B4：修掉「冷 Adam」后续训（600 轮 / ≈7 h）----
    # 起因（2026-10-15 23:26，见 reports/M2B4_冷Adam根因与机制修复.md）：
    # B3 也在 it0100 判负（−1.657 ± 0.396，阈值 −1.60，差 0.057 = 0.14σ），已自动止损。
    # 但早期轨迹给出了**关键形状**：A末 −1.060 → it20 −1.826 → it40 −1.500
    # → it60 −1.733 → it80 −1.736 → it100 −1.657，是**一次性位移后走平**；
    # 而 B2 是**单调下滑**（−1.63/−1.89/−2.20/−2.25/−2.40）。
    #   ⇒ lr 减半（1.2e-5→8e-6）止住了"持续下滑"，但止不住"重启位移"。
    #
    # **决定性自然实验（证伪"lr 数值"假说）**：算两段真实 lr 轨迹 ——
    # B3 重启 lr = 8.000e-6，A 段 it300 处 lr = 8.010e-6，**只差 0.13%**。
    # 同等 lr、同流水线、同牌池下：A 段从更差的 it300(−1.413) 继续涨 +0.353；
    # B3 从更好的 A末(−1.060) 掉 −0.60。⇒ 步长数值不足以解释，
    # 唯一剩余差异 = **优化器状态（AdamW 动量/二阶矩）跨段被丢弃**。
    #
    # 取证：`_dump_ckpt`（selfplay.py:739）只写 state_dict/config/net/encoder_version/
    # iter/eval/encoder_packs；`--init` 只 load_into(model, ck["state_dict"])。
    # 机制：冷 Adam 首步 Δθ ≈ lr·sign(g)，**与梯度大小无关**的 lr 量级扰动，
    # 打在已退火到 lr_min 的收敛模型上 = 全局随机踢一脚；损伤 ∝ lr。
    # 旁证：A 段是单次 600 轮、从未续训 ⇒ 从未冷启动 ⇒ 安然无恙。
    #
    # 因此 B4 三管齐下（本段起）：
    #   ① **机制修复**（selfplay.py）：`--lr-warmup N`（前 N 轮 lr 由 lr0/N 线性升到 lr0）
    #      + `--save-opt`（检查点带 AdamW 状态、`--init` 自动恢复）——已冒烟通过；
    #   ② **起点档位下压**：lr 4e-6→2e-6（A 段收敛区下沿 4e-6 起，而非 it300 的 8e-6）；
    #   ③ **warmup 50**：先把二阶矩喂起来再放大步长。
    #   注意 A 段末模型里没有历史动量（老检查点无 opt 字段），所以 ③ 是必需项。
    # 预注册闸（2400 副，单点 CI≈±0.28）：it0100 `≤ −1.45` 即停（粗大失败闸）；
    # it0300 `≤ −1.05` 即停；it0600 走 G3（追平 ≥ +0.5 且 CI 下界 > 0；超越 ≥ +1.5）。
    "s4b4": dict(iters=600, init="models/dushan_m2_a.pt", out="models/dushan_m2_b4.pt",
                 warmup=0, teacher_prob=0.0, temperature=1.2, lr=4e-6,
                 needs="models/dushan_m2_a.pt", m2=True,
                 extra=["--klt-beta", "0.0",
                        "--snapshot-pool", "",
                        "--ent-baseline", "1.026",
                        "--master-seat-prob", "0.30",
                        "--lr-min", "2e-6",
                        "--lr-warmup", "50",
                        "--save-opt",
                        "--ent-coef", "0.040",
                        "--kl-coef", "0.016",
                        "--explore-eps", "0.010", "--explore-eps-end", "0.010",
                        "--explore-anneal-iters", "1",
                        "--save-every", "20"]),
    # ============================================================================
    # ---- M2-B5 / B6 / C1：夜间无人值守接力段（2026-10-16 用户授权自主决策）----
    # ============================================================================
    # 授权背景：用户夜间不做选择，要求「按推荐选项直接执行、不要闲置算力」，
    # 目标 = 10M 模型（9.85M 参数）追平并超越 S11（= dushan_master，0.77M 参数）。
    #
    # 核心杠杆：**B4 起检查点已带 opt**（`--save-opt`，见 M2B4 报告 §4）
    #   ⇒ 从 B4 的 `_ckpts/*_itNNNN.pt` 续训时 AdamW 的 m/v **自动恢复**，
    #     不会再有「冷启动位移」（B/B2/B3 三次续训失败的真因）。
    #   ⇒ 因此止损后的最优动作 **不是**换档位从 A 段末冷启动重跑，
    #     而是**从 B4 断点无缝续训**：已训轮次不浪费，位移不再付第二次。
    #   这正是「不闲置算力」在算法层的正解。
    #
    # 档位设计（相对 B4 的四条改动，其余口径与 B4 逐字相同）：
    #   ① 续训 lr 取上一段末 lr 的下沿（B4 末 lr=2e-6），再留一档余量 → 收敛区继续细磨；
    #   ② `--lr-warmup 10`（不是 B4 的 50）——opt 已恢复，只需极短缓冲吸收
    #      「检查点周期 20 轮 vs 优化器步 256 组」的时序错配，不需要长爬坡；
    #   ③ `--save-opt` 必带（保证本段产出的 ckpt 同样可无缝续训，链条不断）；
    #   ④ 其余（纯自博弈 / H*=0.975 绝对钉死 / ε 恒 0.010 / ec₀=0.040 / kc₀=0.016）
    #      与 B4 完全一致 —— 单变量原则，只改「起点 + 步长」。
    #
    # 配方增强（相对 B4 的**有意**改动，不是单变量 —— 因为 B4 若失败，
    # 说明"纯自博弈 + β=0"这条路信号不足，必须换杠杆）：
    #   ① `--master-seat-prob` 0.30 → **0.45**：要打赢谁就跟谁多打（P3-M2 D6）；
    #   ② `--klt-beta` 0 → **0.10（退火）**：A 段 −3.1→−1.06 的改善有一半来自
    #      reverse-KL 蒸馏提供的**密集信号**（P3-M2 D3：纯自博弈的 pi 损失只剩
    #      ~2.5% 解释力）。B 段撤教师后信号衰减，是 B/B2/B3 三个段"走平/下滑"的
    #      另一条主线。轻量蒸馏（0.10 ≪ A 段的 0.40）既能补信号又不会锁死在教师上
    #      （Kickstarting DRL 结论：蒸馏起点可反超教师 40%）。
    # s4b5：B4 在 it0100 早停（粗大失败闸 va ≤ −1.45）后的接力，从 it0100 续 500 轮
    "s4b5": dict(iters=500, init="models/_ckpts/dushan_m2_b4_it0100.pt",
                 out="models/dushan_m2_b5.pt",
                 warmup=0, teacher_prob=0.0, temperature=1.2, lr=3e-6,
                 needs="models/_ckpts/dushan_m2_b4_it0100.pt", m2=True,
                 extra=["--klt-beta", "0.10", "--klt-anneal-iters", "200",
                        "--snapshot-pool", "",
                        "--ent-baseline", "1.026",
                        "--master-seat-prob", "0.45",
                        "--lr-min", "1.5e-6",
                        "--lr-warmup", "10",
                        "--save-opt",
                        "--ent-coef", "0.040",
                        "--kl-coef", "0.016",
                        "--explore-eps", "0.010", "--explore-eps-end", "0.010",
                        "--explore-anneal-iters", "1",
                        "--save-every", "20"]),
    # s4b6：B4 在 it0300 早停（va ≤ −1.05）后的接力，从 it0300 续 300 轮
    "s4b6": dict(iters=300, init="models/_ckpts/dushan_m2_b4_it0300.pt",
                 out="models/dushan_m2_b6.pt",
                 warmup=0, teacher_prob=0.0, temperature=1.2, lr=2.5e-6,
                 needs="models/_ckpts/dushan_m2_b4_it0300.pt", m2=True,
                 extra=["--klt-beta", "0.10", "--klt-anneal-iters", "150",
                        "--snapshot-pool", "",
                        "--ent-baseline", "1.026",
                        "--master-seat-prob", "0.45",
                        "--lr-min", "1.2e-6",
                        "--lr-warmup", "10",
                        "--save-opt",
                        "--ent-coef", "0.040",
                        "--kl-coef", "0.016",
                        "--explore-eps", "0.010", "--explore-eps-end", "0.010",
                        "--explore-anneal-iters", "1",
                        "--save-every", "20"]),
    # s4c1：B4 正常跑满 600 轮（或 G3 未达标）后的**长跑巩固段**，
    #       从 B4 主输出（带 opt，因 B4 用 --save-opt 写盘）续 1200 轮。
    #       这是「把样本/参数比推上去」的主战场：10M 模型当前只有 ~5 样本/参数，
    #       而 Mortal 量级（10.8M）用 1e8–1e9 样本 ⇒ 瓶颈是数据量而非算力/容量。
    #       长跑需要**持续的信号**：β=0.06 的轻蒸馏 + master 座位 0.40。
    "s4c1": dict(iters=1200, init="models/dushan_m2_b4.pt", out="models/dushan_m2_c1.pt",
                 warmup=0, teacher_prob=0.0, temperature=1.2, lr=2e-6,
                 needs="models/dushan_m2_b4.pt", m2=True,
                 extra=["--klt-beta", "0.06", "--klt-anneal-iters", "300",
                        "--snapshot-pool", "",
                        "--ent-baseline", "1.026",
                        "--master-seat-prob", "0.40",
                        "--lr-min", "8e-7",
                        "--lr-warmup", "10",
                        "--save-opt",
                        "--ent-coef", "0.040",
                        "--kl-coef", "0.016",
                        "--explore-eps", "0.010", "--explore-eps-end", "0.010",
                        "--explore-anneal-iters", "1",
                        "--save-every", "40"]),
    # auto4：s4b6 跑满后的动态长跑段（= autopilot dynamic_cfg("auto4", b6_it0300, 800, 1.5e-6)）。
    # 单独入表以便用独立后台 shell 起跑（总控 spawn 的子进程会被平台连坐，auto4 事故实证）。
    "auto4": dict(iters=800, init="models/_ckpts/dushan_m2_b6_it0300.pt",
                  out="models/dushan_m2_auto4.pt",
                  warmup=0, teacher_prob=0.0, temperature=1.2, lr=1.5e-6,
                  needs="models/_ckpts/dushan_m2_b6_it0300.pt", m2=True,
                  extra=["--klt-beta", "0.06", "--klt-anneal-iters", "200",
                         "--snapshot-pool", "",
                         "--ent-baseline", "1.026",
                         "--master-seat-prob", "0.40",
                         "--lr-min", "6.7e-7",
                         "--lr-warmup", "10",
                         "--save-opt",
                         "--ent-coef", "0.040",
                         "--kl-coef", "0.016",
                         "--explore-eps", "0.010", "--explore-eps-end", "0.010",
                         "--explore-anneal-iters", "1",
                         "--save-every", "20"]),

    # ---- M2 阶段 P1：lr-restart 旋钮的干净评测 ----
    # 前情：auto6 系列被运行时崩溃打断 5 次（均在 it48–62），lr-restart 从未
    # 得到过 2400 副的正式评测。本段从已判读的 auto5_it0800（va −0.702）出发，
    # 只改「lr 重启 + 不退到 lr-min」这一个变量，跑到 320 轮。
    "m2p1": dict(iters=320, init="models/_ckpts/dushan_m2_auto5_it0800.pt",
                 out="models/dushan_m2_p1.pt",
                 warmup=0, teacher_prob=0.0, temperature=1.2, lr=2.2e-6,
                 needs="models/_ckpts/dushan_m2_auto5_it0800.pt", m2=True,
                 extra=["--klt-beta", "0.06", "--klt-anneal-iters", "150",
                        "--snapshot-pool", "",
                        "--ent-baseline", "1.026",
                        "--master-seat-prob", "0.40",
                        "--lr-min", "1.4e-6",      # 关键：不退到 lr-min
                        "--lr-warmup", "40",
                        "--save-opt",
                        "--ent-coef", "0.040",
                        "--kl-coef", "0.016",
                        "--explore-eps", "0.010", "--explore-eps-end", "0.010",
                        "--explore-anneal-iters", "1",
                        "--save-every", "20"]),

    # ---- M2 阶段 P1：epochs 1→3（「欠训练」假设的干净验证）----
    # 依据（2026-10-17 m2p1 段末判读）：kl_rate 全程 0.001–0.006，而 --kl-target 是 0.05
    # —— 策略每轮位移比目标小一个数量级；同时 pi 损失在变大（−0.136→−0.177）、V 稳定、熵未坍缩
    # ⇒ 不是「没有梯度可算」，是 **每轮只走 16 个梯度步、走得太少**。
    # 只改一个变量：--epochs-per-iter 1→3（每轮 16→47 步，梯度噪声 ↓√3）；
    # 轮数 320→150 让总壁钟相近（≈69 s/轮 × 150 ≈ 2.9 h），其余配方与 m2p1 逐项相同。
    # 预期：kl_rate 从 0.005 抬到 0.012–0.02（仍远低于 0.05，安全性有余量）。
    "m2p2": dict(iters=150, init="models/_ckpts/dushan_m2_p1_it0240.pt",
                 out="models/dushan_m2_p2.pt",
                 warmup=0, teacher_prob=0.0, temperature=1.2, lr=2.2e-6,
                 needs="models/_ckpts/dushan_m2_p1_it0240.pt", m2=True,
                 extra=["--epochs-per-iter", "3",
                        "--klt-beta", "0.06", "--klt-anneal-iters", "150",
                        "--snapshot-pool", "",
                        "--ent-baseline", "1.026",
                        "--master-seat-prob", "0.40",
                        "--lr-min", "1.4e-6",
                        "--lr-warmup", "40",
                        "--save-opt",
                        "--ent-coef", "0.040",
                        "--kl-coef", "0.016",
                        "--explore-eps", "0.010", "--explore-eps-end", "0.010",
                        "--explore-anneal-iters", "1",
                        "--save-every", "10"]),

    # ---- m2p3：同一批数据多走几遍 = 反向（m2p2 实测 −0.310±0.252，显著劣）----
    # m2p2 把 epochs 1→3 后：kl 漂移 0.061(121 轮) → 0.267(150 轮) = **4.4×**，
    # 但 2400 副判读 **−0.310±0.252 显著劣**，且 it0050/0080/0110/0150 单调下降。
    # ⇒ 移动量确实上去了，**方向是错的**：8000 样本重复 3 遍 ≠ 3 倍信息，
    #   等于把 192 局的终局噪声（单局制、奖励稀疏）放大 3 倍去拟合。
    # 所以本轮**唯一变量换成数据量**：--games-per-iter 192→384（epochs 回 1）。
    # 期望：每次更新看到 384 局独立终局 → 优势估计方差 ↓√2，梯度步 16→32。
    # 代价：collect 11→22 s、update 19→38 s ⇒ ≈60–65 s/轮（m2p1 的 2×）。
    # 100 轮 ≈ 1.8 h；起点仍是 m2p1 的最佳点 it0240（m2p2 全程没有超过它）。
    "m2p3": dict(iters=100, init="models/_ckpts/dushan_m2_p1_it0240.pt",
                 out="models/dushan_m2_p3.pt",
                 warmup=0, teacher_prob=0.0, temperature=1.2, lr=2.2e-6,
                 needs="models/_ckpts/dushan_m2_p1_it0240.pt", m2=True,
                 extra=["--games-per-iter", "384",
                        "--epochs-per-iter", "1",
                        "--klt-beta", "0.06", "--klt-anneal-iters", "100",
                        "--snapshot-pool", "",
                        "--ent-baseline", "0.990",
                        "--master-seat-prob", "0.40",
                        "--lr-min", "1.6e-6",
                        "--lr-warmup", "20",
                        "--save-opt",
                        "--ent-coef", "0.040",
                        "--kl-coef", "0.016",
                        "--explore-eps", "0.010", "--explore-eps-end", "0.010",
                        "--explore-anneal-iters", "1",
                        # 384 局 × 2 份缓冲：把池重建判据抬到 20 GB，避免频繁重建吃掉收益
                        "--pool-rss-gb", "20",
                        "--save-every", "10"]),

    # ---- m2d1：先「对齐到 master 水平」，再谈超越 ----------------------------
    # 三次消融（m2p1 堆轮次 ±0 / m2p2 加 epoch −0.310 / m2p3 加数据量 −0.082）
    # 都没动过水平 ⇒ 停滞不在优化器侧。回到事实：
    #   S11(master, 769K 参) = 5.680；10M 模型从 M1 至今稳定停在 5.0 附近，
    #   **起点就比 master 低 0.6**，而且 7 个段一次都没接近过。
    # 关键差异：S11 训练时 teacher_prob 0.55→0.40（大量专家示范），
    #   M1/M2 全程 teacher_prob **0.0**（纯自博弈），KLT 只给了 β=0.06 的弱蒸馏。
    # ⇒ 假设：10M 网络**从未被要求去复制一个 5.68 的策略**，自博弈只能在自己的
    #   5.0 水平上自我强化，永远上不去。
    # 本段只做一件事：把 KLT 拉到 β=0.8 且**不退火**（anneal=0 = 恒定），
    #   验证「10M 有没有能力表示 master 的策略」。这是超越的前提条件。
    # 判据：2400 副 level ≥ 5.40（≈ master 的 95%）→ 对齐成功，下一段从 5.4–5.6 起跑自博弈；
    #       level < 5.2 → 说明是**容量/优化**问题，不是数据问题，方案转向缩模型+蒸馏。
    # 其余配方与 m2p3 逐项相同（192 局/轮、epochs 1、lr 2.2e-6），壁钟 ≈55 s/轮。
    "m2d1": dict(iters=100, init="models/_ckpts/dushan_m2_p1_it0240.pt",
                 out="models/dushan_m2_d1.pt",
                 warmup=0, teacher_prob=0.0, temperature=1.2, lr=2.2e-6,
                 needs="models/_ckpts/dushan_m2_p1_it0240.pt", m2=True,
                 extra=["--epochs-per-iter", "1",
                        # ★ β 恒定 0.8：anneal=0 语义见 selfplay.py:1430（n_an<=0 → 不退火）
                        "--klt-beta", "0.8", "--klt-anneal-iters", "0",
                        "--snapshot-pool", "",
                        "--ent-baseline", "0.990",
                        "--master-seat-prob", "0.40",
                        "--lr-min", "1.8e-6",
                        "--lr-warmup", "20",
                        "--save-opt",
                        "--ent-coef", "0.040",
                        "--kl-coef", "0.016",
                        "--explore-eps", "0.010", "--explore-eps-end", "0.010",
                        "--explore-anneal-iters", "1",
                        "--save-every", "10"]),

    # ★ m2d2（2026-10-17 22:00）：d1 的 β 0.8 把 klt 从 0.645 压到 0.29 后**卡住不动**
    #   （iter 46–100 全程 0.286–0.33），同时 ec 被熵控制器顶到 0.29–0.33 与蒸馏对抗。
    #   这一段只改 **β 0.8 → 5.0**（其余与 m2d1 逐项相同），问一个问题：
    #   klt 卡在 0.29 是"自博弈拉力 = 蒸馏拉力"的平衡点（⇒ 加大 β 会继续降），
    #   还是优化器/容量到顶（⇒ β 再大 klt 也不动）。
    #   结果读法：klt ≤0.20 且 level 上 5.4 ⇒ 平衡点假设成立，下一步做长对齐段；
    #            klt 仍 ~0.29 且 ec 顶到 0.5 ⇒ 熵控制器是绑腿，下一段改钉死 ent-coef。
    "m2d2": dict(iters=80, init="models/_ckpts/dushan_m2_d1_it0100.pt",
                 out="models/dushan_m2_d2.pt",
                 warmup=0, teacher_prob=0.0, temperature=1.2, lr=2.2e-6,
                 needs="models/_ckpts/dushan_m2_d1_it0100.pt", m2=True,
                 extra=["--epochs-per-iter", "1",
                        "--klt-beta", "5.0", "--klt-anneal-iters", "0",
                        "--snapshot-pool", "",
                        "--ent-baseline", "0.990",
                        "--master-seat-prob", "0.40",
                        "--lr-min", "1.8e-6",
                        "--lr-warmup", "20",
                        "--save-opt",
                        "--ent-coef", "0.040",
                        "--kl-coef", "0.016",
                        "--explore-eps", "0.010", "--explore-eps-end", "0.010",
                        "--explore-anneal-iters", "1",
                        "--save-every", "10"]),

    # ★ m2d3（2026-10-17）：d2 把 β 抬到 5.0（d1 的 6.25 倍），klt 只从 0.290 降到 0.228，
    #   而 `ec` 在 iter 51 就**顶到上限 0.5**、熵被压到 0.83 —— 熵控制器在与蒸馏正面对抗。
    #   这一段只改一件事：**--no-adaptive**（关掉 CoefController，ec 恒 0.040 / kc 恒 0.016），
    #   其余与 m2d2 逐项相同。问：klt 卡在 0.23 是"控制器在拉"还是"优化器/容量到顶"。
    #   读法：klt ≤0.13 且 level ≥5.4 ⇒ 控制器是绑腿，做长对齐段（β5 / 150 轮 / no-adaptive）；
    #        klt 仍 ≈0.23 ⇒ 10M 在 lr 1.8e-6 下的蒸馏极限，转 P4「缩到 3–4M」或提 lr。
    #   ⚠️ 关控制器后熵会掉——这是**对齐段的设计意图**（把策略压向 master），
    #      不是训练事故；判读时 A1b/A1c 会 FAIL，属预期，按 klt 与 level 读。
    "m2d3": dict(iters=60, init="models/_ckpts/dushan_m2_d2_it0080.pt",
                 out="models/dushan_m2_d3.pt",
                 warmup=0, teacher_prob=0.0, temperature=1.2, lr=2.2e-6,
                 needs="models/_ckpts/dushan_m2_d2_it0080.pt", m2=True,
                 extra=["--epochs-per-iter", "1",
                        "--klt-beta", "5.0", "--klt-anneal-iters", "0",
                        "--no-adaptive",
                        "--snapshot-pool", "",
                        "--ent-baseline", "0.990",
                        "--master-seat-prob", "0.40",
                        "--lr-min", "1.8e-6",
                        "--lr-warmup", "20",
                        "--save-opt",
                        "--ent-coef", "0.040",
                        "--kl-coef", "0.016",
                        "--explore-eps", "0.010", "--explore-eps-end", "0.010",
                        "--explore-anneal-iters", "1",
                        "--save-every", "10"]),

    # ---- M2-AUX · m2a1（2026-10-18 01:40）：完美信息辅助头（P3，PTIE 式）----
    # 证据链（见 docs/训练决策铁律_20261017.md §5.2/R11 与 m2p1/p2/p3/d1/d2/d3 判读）：
    #   优化器侧四旋钮（轮次/epoch/数据量/蒸馏β）全部受控排除 → 剩余机制假设只有一个：
    #   **稀疏终局奖励是唯一梯度来源**。m2a1 把「训练时可见、执行时不可见」的完美
    #   信息变成 4 个稠密、**静止**的监督目标挂共享主干（UNREAL 式辅助头 +
    #   PerfectDou PTIE 框架）：自家向听(CE) / 三对手听牌(BCE) / 牌墙余量(smooth_l1) /
    #   27 牌危险度地图(BCE，仅 discard 点)。对手头教的正是对 defensive 落后
    #   43.4% 的那条短板 —— 隐藏状态推断。
    # 配方 = m2p1 逐项相同（R1 单变量），只加 --aux-heads on --aux-coef 0.4；
    #   ent-baseline 不显式指定 → 控制器在前 25 轮自测 H₀（d3 系起点熵 ~0.8，
    #   与 m2p1 的 1.026 不可混用）。起点 = M2 全程最高点 d3_it0020（2400 副 5.325）。
    # 判据（预先定死）：2400 副 vs it0020 锚，Δ≥+0.40 且 CI 下界>0 ⇒ P3 成立，
    #   随后做 aux-coef 消融/拉长段；Δ∈±0.2 ⇒ 辅助信号强度不足，试 coef 1.0 或
    #   加对手手牌计数通道；Δ<0 ⇒ 稠密梯度与稀疏奖励在打架，检查 aux 与 pi 的
    #   梯度范数占比再定。
    "m2a1": dict(iters=240, init="models/_ckpts/dushan_m2_d3_it0020.pt",
                 out="models/dushan_m2_a1.pt",
                 warmup=0, teacher_prob=0.0, temperature=1.2, lr=2.2e-6,
                 needs="models/_ckpts/dushan_m2_d3_it0020.pt", m2=True,
                 extra=["--aux-heads", "on", "--aux-coef", "0.4",
                        "--epochs-per-iter", "1",
                        "--klt-beta", "0.06", "--klt-anneal-iters", "150",
                        "--snapshot-pool", "",
                        "--master-seat-prob", "0.40",
                        "--lr-min", "1.4e-6",      # 关键：不退到 lr-min（同 m2p1）
                        "--lr-warmup", "40",
                        "--save-opt",
                        "--ent-coef", "0.040",
                        "--kl-coef", "0.016",
                        "--explore-eps", "0.010", "--explore-eps-end", "0.010",
                        "--explore-anneal-iters", "1",
                        "--save-every", "20"]),

    # ---- M2-AUX · m2a2（2026-10-18 上午）：m2a1 判负后的修复段 ---------------
    # m2a1 结果（2400 副 vs it0020 锚）：it0240 = −0.991±0.262 **显著劣**，
    #   且 it0080 就已 −1.450（损伤集中在前 80 轮 = aux 梯度最大的窗口），
    #   之后仅缓慢回升（+0.29/100 轮）。
    # 归因（scripts/aux_grad_probe.py，段首权重真实轨迹实测）：
    #   aux×0.4 对共享主干的梯度 = RL 主干的 **0.56**；合成总范数 22.8 ≫ clip 5.0
    #   → 全局裁剪把 RL 有效步长压掉 ~4.5×、方向被 aux 污染约 1/3。
    #   段末权重比值仍 0.31 —— 干涉全程存在。健康检查 11 项全 PASS ⇒ 真信号非事故。
    # 修复（单变量：只动 aux 系数曲线，其余与 m2a1 逐位相同）：
    #   coef 0.4 → **0.1**（trunk 占比 0.56→预期 ~0.14）+ **--aux-warmup 80**
    #   线性预热（全新辅助头损失 4.5 的窗口内系数从 0 爬升）。
    # 判据（预先定死）：Δ≥+0.40 且 CI 下界>0 ⇒ P3 成立 → 拉长段+coef 扫描；
    #   Δ∈±0.15 ⇒ 信号太弱，coef 回 0.2 折中；Δ<0 ⇒ aux 信号本身有害，P3 判死，
    #   转录 docs/训练决策铁律 新条目并回退到 m2p1 基线推进其他方向。
    "m2a2": dict(iters=240, init="models/_ckpts/dushan_m2_d3_it0020.pt",
                 out="models/dushan_m2_a2.pt",
                 warmup=0, teacher_prob=0.0, temperature=1.2, lr=2.2e-6,
                 needs="models/_ckpts/dushan_m2_d3_it0020.pt", m2=True,
                 extra=["--aux-heads", "on", "--aux-coef", "0.1",
                        "--aux-warmup", "80",
                        "--epochs-per-iter", "1",
                        "--klt-beta", "0.06", "--klt-anneal-iters", "150",
                        "--snapshot-pool", "",
                        "--master-seat-prob", "0.40",
                        "--lr-min", "1.4e-6",
                        "--lr-warmup", "40",
                        "--save-opt",
                        "--ent-coef", "0.040",
                        "--kl-coef", "0.016",
                        "--explore-eps", "0.010", "--explore-eps-end", "0.010",
                        "--explore-anneal-iters", "1",
                        "--save-every", "20"]),

    # ---- P2 · m2b1（2026-10-18 08:45）：对手池扩容 + 温度采样 ----------------
    # P3 判死（m2a1 −0.991 / m2a2 −0.295，铁律 R12）后按 §5.4 备选顺序转 P2。
    # 假设：自博弈多样性不足——现在池里只有 master + 最近 4 份自己，
    #   线性新近度加权下老快照几乎不出场，自博弈在同一条策略上互相强化。
    # 单一变量包（一个假设=一个变量）：**对手池方案**
    #   * --snapshot-keep 4→12（240 轮 × 每 16 轮一份 ≈ 覆盖 it048–240 全历史）
    #   * --snapshot-temp 0→1.0（softmax(exp(新近度/T))，最新:最老 ≈ e:1，
    #     保证 12 份里最老的也有 ~8% 出场率而非线性的 1/78）
    # 其余与 m2a1/a2 逐位相同（除 aux 外的唯一差异就是上面两项）；起点仍
    # = d3_it0020（5.325，2400 副判读锚）。ent-baseline 不指定（d3 系起点熵 ~0.8）。
    # 内存账：+8 快照 × 39MB × 12 worker ≈ +3.8GB 树峰值 → pool-rss-gb 16→20
    #   （资源旋钮随段配置，非科学旋钮）；系统峰值预计 24→28GB，95% 看门狗之内。
    # 判据（预先定死）：2400 副 vs d3_it0020 锚，Δ≥+0.40 且 CI 下界>0 ⇒ P2 成立
    #   → 拉长段 + 池规模/温度消融；Δ∈±0.15 ⇒ 多样性假设按此实现不成立；
    #   Δ<0 ⇒ 显著劣判负，P2 关闭转 P4。
    "m2b1": dict(iters=240, init="models/_ckpts/dushan_m2_d3_it0020.pt",
                 out="models/dushan_m2_b1.pt",
                 warmup=0, teacher_prob=0.0, temperature=1.2, lr=2.2e-6,
                 needs="models/_ckpts/dushan_m2_d3_it0020.pt", m2=True,
                 extra=["--epochs-per-iter", "1",
                        "--klt-beta", "0.06", "--klt-anneal-iters", "150",
                        "--snapshot-pool", "",
                        "--master-seat-prob", "0.40",
                        "--snapshot-keep", "12",
                        "--snapshot-temp", "1.0",
                        "--pool-rss-gb", "20",
                        "--lr-min", "1.4e-6",
                        "--lr-warmup", "40",
                        "--save-opt",
                        "--ent-coef", "0.040",
                        "--kl-coef", "0.016",
                        "--explore-eps", "0.010", "--explore-eps-end", "0.010",
                        "--explore-anneal-iters", "1",
                        "--save-every", "20"]),

    # ---- P4-A · m2c1（2026-10-18 12:30）：蒸馏瘦身 10M → 4M -------------------
    # 动机：M2 的 17 段里没有任何旋钮在 d3_it0020（5.325）起点上产生正增益，
    #   指向「10M 在 2.7 样本/参数下过参数化」。P4 要回答的问题不是「能不能更强」，
    #   而是「**4M 能不能接住 10M 已有的水平**」：
    #   * 接得住 ⇒ 容量不是瓶颈、10M 确实过参数化 → 之后按 4M 继续爬，训练/推理成本砍 60%；
    #   * 接不住 ⇒ 瓶颈在样本规模而非参数量 → 只剩换硬件/换算法两条路。
    # 为什么用蒸馏而不是纯冷启动（B 路径）：纯冷启动要爬 2000+ 轮才能追到 −3，
    #   600 轮内必然还没上桌，得到的「差」无法归因（分不清是容量不足还是没练够）。
    #   用 d3_it0020 当 KLT 教师，学生**从一开始就在正确分布上学**，
    #   600 轮内到达的水平才有解释力。
    #
    # 变量账（这是新起点，不是单旋钮续训）：
    #   * 学生 = m0 预设 128ch×12 ≈ 4.07M（COMMON 的 m1 → 用 extra 覆盖为 m0）
    #   * --force-cold：10M 与 4M 同 arch 但宽度不同，必须整网重来（新参数）
    #   * --init 仍指 d3_it0020：**只**取它的编码口径（v3/七包）与超参，不装权重
    #   * 教师 = d3_it0020（--master-ckpt 覆盖 M2_COMMON 的 dushan_master），
    #     同时以 0.40 概率当一等对手 —— 学生始终跟「会打牌的自己人」博弈
    #   * β₀ 0.8 → 400 轮退火到 0：前 400 轮蒸馏主导（冷启动初期没有别的有效信号），
    #     后 200 轮纯自博弈，避免把学生钉死在教师上（退火到 0 是硬要求）
    #   * --anchor-warmup 100：前 100 轮**不挂** KL 锚（那会儿策略还接近随机，
    #     挂锚等于把噪声当参考）；第 100 轮把当时策略定为锚，之后按 kl-target 控漂移
    #   * lr 1.2e-5 + lr-warmup 60：冷 Adam 必须缓冲（B4 血案：冷启位移）
    #   * save-every 60 → 600 轮共 10 份存档，判读能画出完整爬升曲线
    #
    # 判据（预先定死，看**段末 level 与 dushan_master 的差**，同牌池口径可比）：
    #   master 5.680 / d3_it0020 5.325 是已知锚。
    #   * it0600 ≥ 5.00（差 master ≤0.68、差 d3 ≤0.33）⇒ **P4 成立**：
    #       4M 容量足够 → 加长段（+1200 轮）继续爬，并把主模型切到 4M
    #   * 4.60 ≤ it0600 < 5.00 ⇒ **部分成立**：容量接近够；若 it0480→0600 仍 >+0.10
    #       （还在涨）则加长段，否则视为 4M 天花板
    #   * it0600 < 4.60 ⇒ **容量不足**：P4 关闭（注明与「600 轮训练量不够」的混淆，
    #       用 it0480→0600 的斜率做区分：斜率显著为正 ⇒ 是训练量问题不是容量问题）
    "m2c1": dict(iters=600, init="models/_ckpts/dushan_m2_d3_it0020.pt",
                 out="models/dushan_m2_c1.pt",
                 warmup=100, teacher_prob=0.0, temperature=1.2, lr=1.2e-5,
                 needs="models/_ckpts/dushan_m2_d3_it0020.pt", m2=True,
                 extra=["--m1-preset", "m0", "--force-cold",
                        "--master-ckpt", "_ckpts/dushan_m2_d3_it0020",
                        "--klt-beta", "0.8", "--klt-anneal-iters", "400",
                        "--master-seat-prob", "0.40",
                        "--snapshot-keep", "4", "--snapshot-temp", "0.0",
                        "--epochs-per-iter", "1",
                        "--lr-min", "4e-6",
                        "--lr-warmup", "60",
                        # 冷启动熵从 ln(动作数) 附近起步并快速下滑，M2 热启动档的
                        # ent-coef 0.04 配 0.95·H₀ 会在前 25 轮把 H₀ 定到「已经掉下来
                        # 的位置」，等于开局就压熵。s1（本项目唯一成功冷启动）用的是
                        # COMMON 的 0.01，这里沿用 —— 熵只当健康指标，不当强度代理（R11）。
                        "--ent-coef", "0.010",
                        "--kl-coef", "0.016",
                        "--explore-eps", "0.010", "--explore-eps-end", "0.010",
                        "--explore-anneal-iters", "1",
                        "--save-every", "60"]),

    # ---- P4-A2 · m2c2（2026-10-18 19:00）：修掉 m2c1 的三个信号杀手 -----------
    # m2c1 判读：峰值 it0300=3.019，之后掉回 2.239（master 5.680）。归因不是容量，
    # 而是三处「把学习信号掐死」的配置（详见 reports/M2C1_蒸馏瘦身判读与归因_20261018.md）：
    #
    #   ① **熵控制器反向工作**：冷启动前 25 轮策略≈随机 → H0=2.057 → H*=1.954
    #      ≈ 随机策略自身的熵。策略一学习（H 掉到 1.72）控制器就判「熵太低」，
    #      ent_coef 从 0.010 乘到 clamp 上限 **0.500** 并贴顶 300 轮，把熵反推回 1.92
    #      —— 整整 300 轮的熵奖励在对抗收敛。修：--no-ent-control（只关熵侧，KL 侧保留）。
    #   ② **β 400 轮撤净**：KLT 从 it120 的 0.390 单调反弹到 it400 的 0.729，
    #      学生从 it120 起就不再像教师；后 480 轮等于裸跑自博弈（10M 自博弈已证零增益）。
    #      修：--klt-beta-end 0.2（退火 600 轮，全程保留弱蒸馏当「不忘教师」的常驻约束）。
    #   ③ **lr 衰减到 4e-6 下限**：后 200 轮几乎不学。修：--lr-min = --lr = 1.2e-5（恒定）。
    #
    # 其余与 m2c1 逐位相同（m0 4.07M / 教师 d3_it0020 / anchor-warmup 100 / 192 局/轮）。
    # 这三条是**同一个假设**：m2c1 的天花板来自「学习信号被掐死」，不是 4M 容量不足。
    # 若修正后仍停在 3.0 附近，则容量不足才真正坐实。
    #
    # 判据（锚 = m2c1 峰值 3.019，同牌池口径）：
    #   * it0600 ≥ 4.6 ⇒ **容量足够**，P4 成立 → 加长段 + 主模型切 4M
    #   * 3.8 – 4.6 ⇒ 部分成立，看 it0480→0600 斜率决定是否加长
    #   * < 3.4（连 m2c1 的峰值都没超过）⇒ **容量不足坐实** → P4 关闭，回 10M 转算法/硬件路线
    "m2c2": dict(iters=600, init="models/_ckpts/dushan_m2_d3_it0020.pt",
                 out="models/dushan_m2_c2.pt",
                 warmup=100, teacher_prob=0.0, temperature=1.2, lr=1.2e-5,
                 needs="models/_ckpts/dushan_m2_d3_it0020.pt", m2=True,
                 extra=["--m1-preset", "m0", "--force-cold",
                        "--master-ckpt", "_ckpts/dushan_m2_d3_it0020",
                        "--klt-beta", "0.8", "--klt-beta-end", "0.2",
                        "--klt-anneal-iters", "600",
                        "--master-seat-prob", "0.40",
                        "--no-ent-control", "--ent-coef", "0.010",
                        "--snapshot-keep", "4", "--snapshot-temp", "0.0",
                        "--epochs-per-iter", "1",
                        "--lr-min", "1.2e-5",
                        "--lr-warmup", "60",
                        "--kl-coef", "0.016",
                        "--explore-eps", "0.010", "--explore-eps-end", "0.010",
                        "--explore-anneal-iters", "1",
                        "--save-every", "60"]),
    # m2c3（P4-A3 加长段）：从 m2c2 末点无缝续训 850 轮（编号 601→1450）。
    # 目的：区分「4M 容量天花板 ≈4.5」vs「训练量不足」（c2 全程仅 1.25 样本/参数）。
    # 与 m2c2 的差异只剩三点，全是续训语义而非新变量：
    #   ① init = c2_it0600（4.494）；② β 恒定 0.2（c2 的退火已在 it600 走完，
    #      公式自带 min(1,·) clamp，直接给 start=end=0.2）；③ warmup=0
    #      （KL 锚 = c2_it0600 立即生效，成熟策略不需要冷启动缓冲期）。
    # opt 状态未存档 → 沿用 lr-warmup 60 缓冲冷 Adam（铁律 B4）。
    "m2c3": dict(iters=850, init="models/_ckpts/dushan_m2_c2_it0600.pt",
                 out="models/dushan_m2_c3.pt",
                 warmup=0, teacher_prob=0.0, temperature=1.2, lr=1.2e-5,
                 needs="models/_ckpts/dushan_m2_d3_it0020.pt", m2=True,
                 offset=600,
                 extra=["--m1-preset", "m0",
                        "--master-ckpt", "_ckpts/dushan_m2_d3_it0020",
                        "--klt-beta", "0.2", "--klt-beta-end", "0.2",
                        "--klt-anneal-iters", "1",
                        "--master-seat-prob", "0.40",
                        "--no-ent-control", "--ent-coef", "0.010",
                        "--snapshot-keep", "4", "--snapshot-temp", "0.0",
                        "--epochs-per-iter", "1",
                        "--lr-min", "1.2e-5",
                        "--lr-warmup", "60",
                        "--kl-coef", "0.016",
                        "--explore-eps", "0.010", "--explore-eps-end", "0.010",
                        "--explore-anneal-iters", "1",
                        "--save-every", "60"]),
    # m2c4（P4-A4 换教师）：从 c3 末点续训 600 轮（编号 1441→2040），
    # **唯一样本信号变化** = KLT 教师 d3_it0020（5.325，v3/9.84M）→ dushan_master
    # （5.680，v1/64ch/0.77M）。
    # 动机：master 只有 0.77M 却打出 5.68 —— 容量显然不是约束，强教师的策略才是；
    # 学生此前只模仿过 5.325 的教师（klt 已降到 0.11，说明"该学的都学完了"）。
    # 已知耦合：--master-ckpt 一物两用（教师 + 0.40 座位的对手），换 master 会同时
    # 把对手从 5.325 提到 5.680。若结果为正，需下一段拆开归因（当前无独立 teacher 参数）。
    # opt 未存档 → lr-warmup 60 缓冲冷 Adam（同 c3）。
    # 850 轮而非 600（2026-10-19 08:58 起跑前调整）：换 v1 教师后实测 17.3 s/轮
    # （教师 769k vs 旧 9.84M，前向便宜近 40%），850 轮仅 ~4.1 h；且**与 c3 轮数逐位对齐**
    # ——c3 用 850 轮从 4.494 爬到 5.004 并在 it1260（段内第 660 轮）才走平，600 轮会在
    # 平台之前截断，使「训练量」成为与 c3 对比的混淆变量。
    "m2c4": dict(iters=850, init="models/_ckpts/dushan_m2_c3_it1440.pt",
                 out="models/dushan_m2_c4.pt",
                 warmup=0, teacher_prob=0.0, temperature=1.2, lr=1.2e-5,
                 needs="models/dushan_master.pt", m2=True,
                 offset=1440,
                 extra=["--m1-preset", "m0",
                        "--master-ckpt", "dushan_master",
                        "--klt-beta", "0.2", "--klt-beta-end", "0.2",
                        "--klt-anneal-iters", "1",
                        "--master-seat-prob", "0.40",
                        "--no-ent-control", "--ent-coef", "0.010",
                        "--snapshot-keep", "4", "--snapshot-temp", "0.0",
                        "--epochs-per-iter", "1",
                        "--lr-min", "1.2e-5",
                        "--lr-warmup", "60",
                        "--kl-coef", "0.016",
                        "--explore-eps", "0.010", "--explore-eps-end", "0.010",
                        "--explore-anneal-iters", "1",
                        "--save-every", "60"]),
    # m2c5（P4-A5 加长续训）：从 c4 末点 c4_it2280（5.434）再续 850 轮（编号 2281→3130）。
    # 依据：m2c4 段末三对相邻配对仍是 +0.152/+0.139/+0.142 连续同向，**未走平**，
    # 与 c3「660 轮后走平」的形态不同 ⇒ 换强教师把平台推高了，需要再一段确认新天花板。
    # 除轮次外（R1 单旋钮）与 m2c4 完全一致：同教师 dushan_master / β 0.2 / 熵控关 / lr 恒 1.2e-5。
    # 编号说明：m2c4 计划 1441→2290，但 save-every=60 使最后存档落在 2280，
    # it2281–2290 的状态未落盘 ⇒ 本段从 2280 起、编号 2281 起（重复训练 10 轮，无害且必须登记）。
    # 判据：锚 = 5.434±0.244；**Δ≥+0.20 且 CI 下界>0** ⇒ 新平台更进一步；
    #       |Δ|<0.20 ⇒ 4M 配方在 5.43 附近收敛，主模型定版、转算法/规则侧；
    #       Δ<0 ⇒ 过训/教师信号耗尽，回滚 c4_it2280。
    "m2c5": dict(iters=850, init="models/_ckpts/dushan_m2_c4_it2280.pt",
                 out="models/dushan_m2_c5.pt",
                 warmup=0, teacher_prob=0.0, temperature=1.2, lr=1.2e-5,
                 needs="models/dushan_master.pt", m2=True,
                 offset=2280,
                 extra=["--m1-preset", "m0",
                        "--master-ckpt", "dushan_master",
                        "--klt-beta", "0.2", "--klt-beta-end", "0.2",
                        "--klt-anneal-iters", "1",
                        "--master-seat-prob", "0.40",
                        "--no-ent-control", "--ent-coef", "0.010",
                        "--snapshot-keep", "4", "--snapshot-temp", "0.0",
                        "--epochs-per-iter", "1",
                        "--lr-min", "1.2e-5",
                        "--lr-warmup", "60",
                        "--kl-coef", "0.016",
                        "--explore-eps", "0.010", "--explore-eps-end", "0.010",
                        "--explore-anneal-iters", "1",
                        "--save-every", "60"]),
    # m2d1（P4-B 归因 B 格）：**教师保持不变（master），只把一等对手座位退回旧陪练**。
    # 目的：拆 m2c4 的 +0.386 —— 那是「换教师」与「换陪练」同时发生的结果，此前无法归因。
    # 对照关系（与 m2c4 逐项相同，唯一差异 = 座位对手）：
    #   m2c4 = 教师 master × 座位 master   → 5.434（+0.386）
    #   m2d1 = 教师 master × 座位 d3_it0020 → ?
    # 座位选 d3_it0020（9.84M / 5.325）而不是 c3_it1440：**c3 段真实用的陪练就是 d3_it0020**
    # （m2c3 的 --master-ckpt 即它），这才是"旧配方原样"的陪练，能保证单旋钮。
    # 代价：d3 是 9.84M v3，前向比 master(767k) 贵，实测 c4 的 20.4 s/轮会回到 ~24-26 s/轮，
    # 850 轮 ≈ 5.5-6.5 h。
    # 判据（锚 = c4_it2280 = 5.434±0.244，2400 副，看 B 末点）：
    #   B ≥ 5.23（Δ vs c4 ≥ −0.20）⇒ 陪练贡献 ≤0.20，**教师是主因**
    #   B ≤ 5.03（Δ ≤ −0.40）     ⇒ 陪练贡献 ≥0.40，**陪练是主因，教师≈0**
    #   5.03–5.23                  ⇒ 两者各半（各 ~0.2）
    # 注意：本格的期望不是"更高"，而是"低多少"——**低得越少说明教师项越重要**。
    "m2e1": dict(iters=850, init="models/_ckpts/dushan_m2_c3_it1440.pt",
                 out="models/dushan_m2_e1.pt",
                 warmup=0, teacher_prob=0.0, temperature=1.2, lr=1.2e-5,
                 needs="models/dushan_master.pt", m2=True,
                 offset=1440,
                 extra=["--m1-preset", "m0",
                        "--master-ckpt", "_ckpts/dushan_m2_d3_it0020",
                        "--klt-teacher", "dushan_master",
                        "--klt-beta", "0.2", "--klt-beta-end", "0.2",
                        "--klt-anneal-iters", "1",
                        "--master-seat-prob", "0.40",
                        "--no-ent-control", "--ent-coef", "0.010",
                        "--snapshot-keep", "4", "--snapshot-temp", "0.0",
                        "--epochs-per-iter", "1",
                        "--lr-min", "1.2e-5",
                        "--lr-warmup", "60",
                        "--kl-coef", "0.016",
                        "--explore-eps", "0.010", "--explore-eps-end", "0.010",
                        "--explore-anneal-iters", "1",
                        "--save-every", "60"]),
    # ---- P4-C · m2f1（2026-10-20 01:17）：10M 容量实验 -----------------------
    # 问题：4.07M 在 c2 配方下的上限 ≈5.0（c2_it0600 4.494 / c3_it1260 5.004）。
    # 同配方、同教师（d3_it0020）、同冷启，唯一变量 = 预设 m0→m1（4.07M→9.84M）。
    # **必须 --force-cold**：init=d3_it0020 与 preset m1 形状完全匹配，不加会静默
    # 热启动（整段变成"从 5.325 续训"，容量问题被完全污染）。
    # 判据（锚 = c3_it1260 = 5.004±0.239，4M 同配方 1450 轮的产物）：
    #   * it1200 显著 > 5.00（Δ≥+0.20 且 CI 下界>0）⇒ 容量有剩，10M 路线打开
    #   * |Δ| < 0.20 ⇒ 容量说排除（同配方下宽度不带来强度），转算法侧
    #   * 末点 < 4.8 且斜率仍正 ⇒ 训练量不足（10M 仅 1.04 样本/参数 vs 4M 的 2.7），
    #     不判死，续训段再定
    # 速度参照旧 10M 段（m2d1 legacy）：~38 s/轮 ⇒ 1200 轮 ≈ 12.7 h。
    # β 退火保持 600 轮（c2 原样）：后半程 β 恒 0.2，与 c2 的退火终点一致。
    "m2f1": dict(iters=1200, init="models/_ckpts/dushan_m2_d3_it0020.pt",
                 out="models/dushan_m2_f1.pt",
                 warmup=100, teacher_prob=0.0, temperature=1.2, lr=1.2e-5,
                 needs="models/_ckpts/dushan_m2_d3_it0020.pt", m2=True,
                 extra=["--m1-preset", "m1", "--force-cold",
                        "--master-ckpt", "_ckpts/dushan_m2_d3_it0020",
                        "--klt-beta", "0.8", "--klt-beta-end", "0.2",
                        "--klt-anneal-iters", "600",
                        "--master-seat-prob", "0.40",
                        "--no-ent-control", "--ent-coef", "0.010",
                        "--snapshot-keep", "4", "--snapshot-temp", "0.0",
                        "--epochs-per-iter", "1",
                        "--lr-min", "1.2e-5",
                        "--lr-warmup", "60",
                        "--kl-coef", "0.016",
                        "--explore-eps", "0.010", "--explore-eps-end", "0.010",
                        "--explore-anneal-iters", "1",
                        "--save-every", "60"]),
    # ---- P4-C2 · m2f2（2026-10-20 11:40）：10M 后半程续训（601→1200）---------
    # 由来：m2f1 在 it0600 的中途判读（2400 副，anchor=c2_it0600）
    #   f1_it0600 = 4.983±0.236 vs c2_it0600 4.494 → **+0.490±0.238 显著优**
    #   ⇒ 预登记判据（Δ≥+0.20 且 CI 下界>0）命中：容量有剩，跑满 1200 轮。
    #   另：10M 用 600 轮就追平了 4M 用 1450 轮才到的 5.004（c3_it1260）。
    # 中断事故：m2f1 主进程在 iter 604 被 OOM 杀掉（当时并发了两份 2400 副判读，
    #   系统剩余内存 0.18 GB）。最后一个完好检查点 = it0600，故本段从它续训，
    #   重复 4 轮（无害，按 R7 留痕）。**铁律补记：判读与训练永不并发，10M 尤其。**
    # 配方与 m2f1 的差异全是续训语义：① init=it0600；② β 恒 0.2（退火已走完）；
    #   ③ warmup=0（成熟策略，KL 锚 = it0600 立即生效）；④ opt 未存档 → lr-warmup 60。
    # 判据（锚 = f1_it0600 = 4.983±0.236）：it1200 ≥ 5.30 且 CI 下界>0 ⇒ 容量还有剩，
    #   继续加长；|Δ|<0.20 ⇒ 10M 也在 5.0–5.2 附近收敛，转算法侧。
    "m2f2": dict(iters=600, init="models/_ckpts/dushan_m2_f1_it0600.pt",
                 out="models/dushan_m2_f2.pt",
                 warmup=0, teacher_prob=0.0, temperature=1.2, lr=1.2e-5,
                 needs="models/_ckpts/dushan_m2_d3_it0020.pt", m2=True,
                 offset=600,
                 extra=["--m1-preset", "m1",
                        "--master-ckpt", "_ckpts/dushan_m2_d3_it0020",
                        "--klt-beta", "0.2", "--klt-beta-end", "0.2",
                        "--klt-anneal-iters", "1",
                        "--master-seat-prob", "0.40",
                        "--no-ent-control", "--ent-coef", "0.010",
                        "--snapshot-keep", "4", "--snapshot-temp", "0.0",
                        "--epochs-per-iter", "1",
                        "--lr-min", "1.2e-5",
                        "--lr-warmup", "60",
                        "--kl-coef", "0.016",
                        "--explore-eps", "0.010", "--explore-eps-end", "0.010",
                        "--explore-anneal-iters", "1",
                        "--save-every", "60"]),
    # m2g1（P4-D β 末端退火）：**唯一还没被干净试过的机制** —— 把 KLT 的拉力在末段解绑。
    # 依据（2026-10-20 的 m2f2 判读）：四段同池落点全在各自教师下方 0.16–0.39
    # （c3 −0.321 / c4 −0.246 / e1 −0.390 / f2 −0.162），与参数量无关 ⇒ 上限由教师决定，
    # 「加容量 / 加轮数 / 加强陪练」三路失效共用这一条解释。β=0.2 的 KLT 一直在把策略
    # 往回拽，唯一能越过教师的手段就是撤掉这根绳子。
    # 与 m2c5 的关键差异：m2c1 那次 β→0 是废的，因为 `master_model = teacher_model`
    # 让座位对手跟着教师一起消失；2026-10-19 的 `--klt-teacher` 已把两者拆成独立对象，
    # 本段显式写 --klt-teacher，β=0 后 0.40 权重的一等对手仍然在打。
    # 配方（除 β 计划外与 m2c4/c5 逐项相同，R1 单旋钮）：
    #   300 轮保持 β=0.2（继续贴教师）→ 250 轮线性退到 0 → 末 150 轮纯 PPO 打 master。
    # 对照基准不是「起点」而是「同配方再多跑 700 轮」：m2c5 已实测 +850 轮 = −0.027（ns），
    # 所以任何 Δ≥+0.20 都只能由 β 退火解释。
    # 判据（锚 = c4_it2280 = 5.434±0.244）：末点 − 5.434 ≥ +0.20 且 CI 下界>0 ⇒ 天花板可破；
    #   |Δ|<0.20 ⇒ KL 拉力不是瓶颈（教师分布本身就是上限），转 PPO 目标/采样侧；
    #   Δ<0 ⇒ 解绑即失去学习信号（m2c1 的定性结论复现），回滚 c4。
    "m2g1": dict(iters=700, init="models/_ckpts/dushan_m2_c4_it2280.pt",
                 out="models/dushan_m2_g1.pt",
                 warmup=0, teacher_prob=0.0, temperature=1.2, lr=1.2e-5,
                 needs="models/dushan_master.pt", m2=True,
                 offset=2280,
                 extra=["--m1-preset", "m0",
                        "--master-ckpt", "dushan_master",
                        "--klt-teacher", "dushan_master",
                        "--klt-beta", "0.2", "--klt-beta-end", "0.0",
                        "--klt-hold-iters", "300",
                        "--klt-anneal-iters", "250",
                        "--master-seat-prob", "0.40",
                        "--no-ent-control", "--ent-coef", "0.010",
                        "--snapshot-keep", "4", "--snapshot-temp", "0.0",
                        "--epochs-per-iter", "1",
                        "--lr-min", "1.2e-5",
                        "--lr-warmup", "60",
                        "--kl-coef", "0.016",
                        "--explore-eps", "0.010", "--explore-eps-end", "0.010",
                        "--explore-anneal-iters", "1",
                        "--save-every", "60"]),

    # m2g1r：m2g1 睡眠中断续跑（2026-10-21 20:10）。机器 ~00:35 睡眠，训练死于 abs 2668
    # （段内 388/700），最后完好 ckpt = it2640。原计划 abs 2281→2980（hold300+anneal250+纯PPO150）。
    # 段内 it 从 0 重计 ⇒ β 计划须按新坐标系重写：死亡时 β≈0.130（段内 it388），
    # 原斜率 0.2/250=0.0008/轮；用 β0=0.13 / anneal=162 / hold=0 可在 it162（abs 2802）
    # 退到 0（原设计 abs 2831，差 29 轮），斜率 0.13/162≈0.000802 与原线几乎重合。
    # 末 150 轮（abs 2803→2952）纯 PPO。共 312 轮，编号 2641→2952。
    "m2g1r": dict(iters=312, init="models/_ckpts/dushan_m2_g1_it2640.pt",
                 out="models/dushan_m2_g1r.pt",
                 warmup=0, teacher_prob=0.0, temperature=1.2, lr=1.2e-5,
                 needs="models/dushan_master.pt", m2=True,
                 offset=2640,
                 extra=["--m1-preset", "m0",
                        "--master-ckpt", "dushan_master",
                        "--klt-teacher", "dushan_master",
                        "--klt-beta", "0.13", "--klt-beta-end", "0.0",
                        "--klt-hold-iters", "0",
                        "--klt-anneal-iters", "162",
                        "--master-seat-prob", "0.40",
                        "--no-ent-control", "--ent-coef", "0.010",
                        "--snapshot-keep", "4", "--snapshot-temp", "0.0",
                        "--epochs-per-iter", "1",
                        "--lr-min", "1.2e-5",
                        "--lr-warmup", "60",
                        "--kl-coef", "0.016",
                        "--explore-eps", "0.010", "--explore-eps-end", "0.010",
                        "--explore-anneal-iters", "1",
                        "--save-every", "60"]),

    # m2g2（2026-10-21 22:12）：m2g1r 判负后的单旋钮对照——β=0 纯 PPO 但**熵控制器开启**。
    # g1r 证明：撤净 KLT ⇒ 熵自由跌（0.63→0.39）⇒ 强度 150 轮崩 −1.15。
    # 本段唯一变量 = 恢复熵控（H*≈0.63 由段首 H0 标定），检验"崩因是熵还是缺 KL 锚"。
    # init=c4_it2280（5.434），600 轮，编号 2281→2880；座位 master 0.40 在场；无 KLT 教师。
    "m2g2": dict(iters=600, init="models/_ckpts/dushan_m2_c4_it2280.pt",
                 out="models/dushan_m2_g2.pt",
                 warmup=0, teacher_prob=0.0, temperature=1.2, lr=1.2e-5,
                 needs="models/dushan_master.pt", m2=True,
                 offset=2280,
                 extra=["--m1-preset", "m0",
                        "--master-ckpt", "dushan_master",
                        "--klt-beta", "0.0",
                        "--master-seat-prob", "0.40",
                        "--ent-coef", "0.010",
                        "--snapshot-keep", "4", "--snapshot-temp", "0.0",
                        "--epochs-per-iter", "1",
                        "--lr-min", "1.2e-5",
                        "--lr-warmup", "60",
                        "--kl-coef", "0.016",
                        "--explore-eps", "0.010", "--explore-eps-end", "0.010",
                        "--explore-anneal-iters", "1",
                        "--save-every", "60"]),

    # m2g3（2026-10-21 夜间自动链预登记）：m2g2 通过（β=0+熵控不崩且末点 ≥5.40）
    # 之后按决策树自动起跑的「受控超越」段 —— 夜间授权（用户 22:45 明示）。
    # 假设：0.05 的小 KL 拉力足以防漂移塌缩，又给策略留出超越教师分布的空间；
    # 熵侧由控制器（g2 验证）负责，不再依赖 β 保熵。
    # init=g2 末点模型（判读通过时 ≥ c4_it2280）；β 0.05 常数（无退火）；
    # 教师=座位=master（显式 --klt-teacher，c6 修的解耦）；编号 2881→3480。
    "m2g3": dict(iters=600, init="models/dushan_m2_g2.pt",
                 out="models/dushan_m2_g3.pt",
                 warmup=0, teacher_prob=0.0, temperature=1.2, lr=1.2e-5,
                 needs="models/dushan_m2_g2.pt", m2=True,
                 offset=2880,
                 extra=["--m1-preset", "m0",
                        "--master-ckpt", "dushan_master",
                        "--klt-teacher", "dushan_master",
                        "--klt-beta", "0.05",
                        "--master-seat-prob", "0.40",
                        "--ent-coef", "0.010",
                        "--snapshot-keep", "4", "--snapshot-temp", "0.0",
                        "--epochs-per-iter", "1",
                        "--lr-min", "1.2e-5",
                        "--lr-warmup", "60",
                        "--kl-coef", "0.016",
                        "--explore-eps", "0.010", "--explore-eps-end", "0.010",
                        "--explore-anneal-iters", "1",
                        "--save-every", "60"]),

    # m2g3b（2026-10-22 凌晨预登记）：m2g2 判负（β=0+熵控没站住 5.40）时的
    # 回退实验 —— 从 c4_it2280 出发把 KL 拉力减半（β 0.2→0.05），**其余逐项
    # 复刻 c4**（熵控关、无 --klt-teacher = 教师/座位共用 master、β 恒定不退火）。
    # R1 单旋钮：对 c4 的唯一变量 = β。教师与座位同源，β 恒 0.05 不触 c6 对手消失 bug。
    # 编号 2281→2880（存档 stem 不同，与 g2 无文件冲突）。
    # 判据：末点 − c4_it2280 ≥ +0.20 且 CI 下界>0 ⇒ 小拉力即可受控超越；
    #       |Δ|<0.20 ⇒ 拉力梯度上 0.05 与 0.2 等价，天花板再确认；
    #       塌陷（<5.0）⇒ 连 0.05 都保不住 ⇒ KL 锚不可撤终审判决，"突破教师"线关闭。
    "m2g3b": dict(iters=600, init="models/_ckpts/dushan_m2_c4_it2280.pt",
                  out="models/dushan_m2_g3b.pt",
                  warmup=0, teacher_prob=0.0, temperature=1.2, lr=1.2e-5,
                  needs="models/dushan_master.pt", m2=True,
                  offset=2280,
                  extra=["--m1-preset", "m0",
                         "--master-ckpt", "dushan_master",
                         "--klt-beta", "0.05", "--klt-beta-end", "0.05",
                         "--klt-anneal-iters", "1",
                         "--master-seat-prob", "0.40",
                         "--no-ent-control", "--ent-coef", "0.010",
                         "--snapshot-keep", "4", "--snapshot-temp", "0.0",
                         "--epochs-per-iter", "1",
                         "--lr-min", "1.2e-5",
                         "--lr-warmup", "60",
                         "--kl-coef", "0.016",
                         "--explore-eps", "0.010", "--explore-eps-end", "0.010",
                         "--explore-anneal-iters", "1",
                         "--save-every", "60"]),
}


def find_latest_ckpt(stem: str) -> tuple[Path, int] | None:
    """本段 ``_ckpts/{stem}_itNNNN.pt`` 里全局轮次最大的那一份。

    续训的起点：段中途崩溃（如 S1 的 iter 268 契约崩溃）时，主输出
    ``models/{stem}.pt`` 可能停在更早的选点轮，而周期存档保留了更近的权重。
    """
    d = ROOT / "models" / "_ckpts"
    if not d.exists():
        return None
    best: tuple[Path, int] | None = None
    for p in d.glob(f"{stem}_it*.pt"):
        m = re.search(r"_it(\d+)\.pt$", p.name)
        if not m:
            continue
        it = int(m.group(1))
        if best is None or it > best[1]:
            best = (p, it)
    return best


def preflight(seg: str, cfg: dict, workers: int) -> list[str]:
    """起动前检查。返回致命问题列表（空 = 放行）。"""
    fatal = []

    init_p = ROOT / cfg["init"]
    if not init_p.exists():
        fatal.append(f"--init 不存在：{init_p}")
    if cfg["needs"]:
        need_p = ROOT / cfg["needs"]
        if not need_p.exists():
            fatal.append(f"上游检查点缺失：{need_p}（上一段还没跑完？）")

    free_gb = shutil.disk_usage(ROOT).free / 1024 ** 3
    if free_gb < 50:
        fatal.append(f"磁盘可用仅 {free_gb:.1f} GB（< 50 GB）")
    else:
        print(f"[preflight] 磁盘可用 {free_gb:.1f} GB ✅")

    # 残留训练进程：同时起两个 selfplay 会互相争抢 CPU 与内存，指标全废
    try:
        import psutil
        leftovers = []
        for p in psutil.process_iter(["cmdline"]):
            cl = list(p.info.get("cmdline") or [])
            if any(a.endswith("selfplay.py") for a in cl[1:]) and p.pid != 0:
                leftovers.append(p.pid)
        if leftovers:
            fatal.append(f"已有 selfplay 进程在跑：{leftovers} —— 先停掉再起新段")
        else:
            print("[preflight] 无残留训练进程 ✅")
        vm = psutil.virtual_memory()
        print(f"[preflight] 内存基线：已用 {vm.used / 1024 ** 3:.2f} GB "
              f"({vm.percent}%)，可用 {vm.available / 1024 ** 3:.2f} GB")
    except ImportError:
        print("[preflight] ⚠️ psutil 不可用，跳过内存/残留检查")

    print(f"[preflight] 配方：{seg}  iters={cfg['iters']}  warmup={cfg['warmup']}  "
          f"teacher_prob={cfg['teacher_prob']}  temperature={cfg['temperature']}  "
          f"lr={cfg['lr']}  workers={workers}")
    if cfg["warmup"] == 0:
        print("[preflight] 注意：warmup=0 → KL 锚点 = --init 的那份权重（锚点不持久化）")
    return fatal


def build_cmd(seg: str, cfg: dict, workers: int) -> list[str]:
    out = ROOT / cfg["out"]
    hist = ROOT / "reports" / f"m1_{seg}_history.json"
    log = ROOT / "reports" / f"_m1_{seg}.out"
    extra: list[str] = []
    if cfg.get("offset"):
        # 续训：轮次编号从 offset+1 起（不覆盖第一段已存的 _ckpts 同名文件），
        # 且 history 追加写 —— 否则段末判读只能看到后半截。
        extra += ["--iter-offset", str(cfg["offset"]), "--history-append"]
    return [PY, "-u", str(ROOT / "scripts" / "selfplay.py"),
            # ★ 训练进程**自己**开日志（append）。不要依赖父进程传下来的 stdout 句柄：
            #   父进程一退出，继承来的句柄就成了悬空引用，输出会**静默消失**。
            "--log-file", str(log),
            "--init", str(ROOT / cfg["init"]), "--out", str(out),
            "--history", str(hist), "--iters", str(cfg["iters"]),
            "--anchor-warmup", str(cfg["warmup"]),
            "--teacher-prob", str(cfg["teacher_prob"]),
            "--temperature", str(cfg["temperature"]),
            "--lr", str(cfg["lr"]),
            "--workers", str(workers)] + extra + COMMON \
        + (M2_COMMON if cfg.get("m2") else []) + list(cfg.get("extra") or [])


def _spawn(cmd: list[str], stdout=None, label: str | None = None) -> subprocess.Popen:
    """启动子进程，并**尽量**让它脱离调用方的 Job Object（真实终端下有用）。

    ⚠️ 沙箱型环境（WorkBuddy 工具调用）**无视**这些 flags —— 每次调用结束都会
    清理整棵进程树，逐档实测见模块 docstring。所以本函数只负责"起得起来"，
    "活得下去"靠 `--foreground` + 平台后台任务，不要指望这里。

    Windows 上 ``CREATE_BREAKAWAY_FROM_JOB`` 只在 Job 允许 breakaway 时才生效，
    沙箱常常不允许（``CreateProcess`` 报 ERROR_ACCESS_DENIED）→ 逐级回退。
    本项目吃过两次长跑事故（22.2 h 空闲冻结、s9 因 nohup 脱管失联），
    脱离行为值得多试几档、并且把实际用的是哪一档打出来。

    ``stdin=DEVNULL`` 是**必需**的：调用方（工具链/CI 的 shell）给的多半是 pipe，
    它自己一退出 pipe 就断，而训练进程还持着那个失效句柄。守护进程本就不该
    持有标准输入。
    """
    DETACHED = getattr(subprocess, "DETACHED_PROCESS", 0)
    BREAKAWAY = getattr(subprocess, "CREATE_BREAKAWAY_FROM_JOB", 0)
    NEWGRP = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
    out = stdout if stdout is not None else subprocess.DEVNULL
    for name, flags in (("DETACHED+BREAKAWAY+NEWGRP", DETACHED | BREAKAWAY | NEWGRP),
                        ("DETACHED+NEWGRP", DETACHED | NEWGRP),
                        ("DETACHED", DETACHED),
                        ("（无 flags，退化为跟随）", 0)):
        try:
            p = subprocess.Popen(cmd, stdin=subprocess.DEVNULL,
                                 stdout=out, stderr=subprocess.STDOUT,
                                 cwd=str(ROOT), creationflags=flags)
            if label:
                print(f"[launch] {label}起动方式：{name}", flush=True)
            return p
        except OSError:
            continue
    raise RuntimeError("子进程启动失败（所有 creationflags 档位都被拒绝）")


def launch(seg: str, cfg: dict, workers: int, foreground: bool = True,
           settle: float = 8.0, wait: bool = True, guards: bool = True) -> int:
    """起动一段训练。

    ``foreground`` 管**怎么起**（跟随调用方 vs 脱离会话），``wait`` 管**起完要不要等**。
    两者必须分开（2026-10-15 C7）：M2 自动链要在**训练跑着的同时**在 400/800 轮跑门禁
    判读（那才是"撤教师塌陷"的早警点），所以它需要 ``foreground=True, wait=False`` ——
    沿用旧签名只能二选一，结果门禁被排到 1200 轮之后，早警功能形同虚设。
    """
    run = f"m1_{seg}"
    log = ROOT / "reports" / f"_{run}.out"
    mon = ROOT / "reports" / f"_res_monitor_{run}.csv"
    svc = ROOT / "scripts"

    cmd = build_cmd(seg, cfg, workers)
    print(f"[launch] 起动 {run}\n  {' '.join(cmd)}\n  日志 → {log}", flush=True)

    # 守护 1：防系统空闲睡眠（Windows 下长跑被冻结过一次，22.2 h 未被日志发现）
    # 守护 2：独立 wall clock 打点（训练被挂起时它仍在写 → 事后能定位冻结区间）
    # ★ 用户 2026-10-17 明确取消过这两个脚本（keep_awake / res_monitor），
    #   所以 --no-guards 时必须完全不拉起 —— 不是"起了再关"，是一个进程都不建。
    if guards:
        _spawn([PY, str(svc / "keep_awake.py")])
        _spawn([PY, str(svc / "res_monitor.py"), "--interval", "60", "--out", str(mon)])
    else:
        print("[launch] --no-guards：不拉起 keep_awake / res_monitor", flush=True)

    # 头部写盘后**立即释放句柄**。训练进程用 `--log-file` 以 append 方式自写同一文件：
    # 头部在前、训练输出在后，顺序天然衔接；两侧互不持有对方的引用。
    with log.open("a", encoding="utf-8") as fh:
        fh.write(f"\n===== {run} 起动 @ {time.strftime('%Y-%m-%d %H:%M:%S')} =====\n")
        fh.write(" ".join(cmd) + "\n")

    if foreground:
        proc = subprocess.Popen(cmd, stdin=subprocess.DEVNULL,
                                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                                cwd=str(ROOT))
        print("[launch] 起动方式：前台跟随（调用方存活多久，训练就活多久）", flush=True)
    else:
        proc = _spawn(cmd, None, label="训练进程")

    (ROOT / "reports" / f"_{run}.pid").write_text(str(proc.pid), encoding="utf-8")

    # 存活确认：立即死掉的（缺依赖 / 参数错 / 权重路径错）在几秒内就会 poll≠None，
    # 早点报出来，别让人等 6 小时才发现日志没长。
    rc = None
    if settle > 0:
        time.sleep(settle)
        rc = proc.poll()
        if rc is None:
            print(f"[launch] ✅ 主进程存活确认（pid {proc.pid}，已在跑 {settle:.0f}s）", flush=True)
        else:
            tail = ""
            try:
                tail = "\n".join(log.read_text(encoding="utf-8", errors="replace").splitlines()[-12:])
            except Exception:                                   # noqa: BLE001
                pass
            print(f"[launch] ❌ {run} 起动后 {settle:.0f}s 内就退出了（exit={rc}）。"
                  f"日志尾部：\n{tail}", flush=True)
            return rc or 1
    else:
        print(f"[launch] 主进程 pid {proc.pid}（未做存活确认）", flush=True)

    _sz0 = log.stat().st_size if log.exists() else 0

    print(f"[launch] {'前台跟随' if foreground else '已脱离会话'}；"
          f"看进度：tail -f {log}", flush=True)
    print(f"[launch] 段末收尾：python scripts/m1_segment_finalize.py "
          f"--run {run} --ckpt-prefix {Path(cfg['out']).stem}", flush=True)

    if foreground and wait:
        rc = proc.wait()
        _sz1 = log.stat().st_size if log.exists() else 0
        print(f"[launch] {run} 结束，exit={rc}；日志 {_sz0} → {_sz1} 字节（{log}）", flush=True)
        return rc
    if foreground and not wait:
        print(f"[launch] {run} 非阻塞起跑（pid {proc.pid}）；调用方自行决定何时等",
              flush=True)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--segment", required=True, choices=sorted(SEGMENTS))
    ap.add_argument("--workers", type=int, default=12,
                    help="采集进程数；内存吃紧时下调（12→8）。降它不伤吞吐 —— 采集只占单轮 1/4")
    ap.add_argument("--dry-run", action="store_true", help="只做检查并打印命令，不启动")
    ap.add_argument("--detach", action="store_true",
                    help="脱离会话起训练（仅真实终端适用）。沙箱型环境（工具调用）会"
                         "在调用结束清理整棵进程树 → 用它会**静默零输出即死**，"
                         "所以默认关：不加此参数时训练进程跟随调用方。")
    ap.add_argument("--foreground", action="store_true",
                    help="（已废弃，等价于默认行为）保留仅为兼容旧命令行。")
    ap.add_argument("--settle", type=float, default=8.0,
                    help="起动后等待多少秒做一次存活确认（poll 非 None 即报错退出）。"
                         "0 = 跳过确认。默认 8。")
    ap.add_argument("--resume", action="store_true",
                    help="段中途崩溃后续跑：自动从 models/_ckpts/ 取本段最新存档当 "
                         "--init，只补跑剩余轮次，并用 --iter-offset 让编号续上"
                         "（不覆盖第一段的周期存档、history 追加而非覆盖）。")
    ap.add_argument("--no-guards", action="store_true",
                    help="不拉起 keep_awake.py / res_monitor.py（用户 2026-10-17 已明确"
                         "取消这两个脚本）。默认仍拉，历史配方不受影响。")
    args = ap.parse_args()

    cfg = dict(SEGMENTS[args.segment])
    if args.resume:
        hit = find_latest_ckpt(Path(cfg["out"]).stem)
        if hit is None:
            print(f"[resume] ❌ models/_ckpts/{Path(cfg['out']).stem}_it*.pt 一个都没有，"
                  "无处可续 —— 请按新段正常起动。", flush=True)
            return 2
        p, it = hit
        total = cfg["iters"]
        # 编号起点取「周期存档轮次」与「history 已完成轮次」的较大者：
        # history 往往比最后一个周期存档更靠前（崩溃点与 save-every 不对齐），
        # 若只按存档编号续，续训的首几轮会与既有记录**重号**，段末判读
        # 的对齐合并（按 iter 建索引）会张冠李戴。
        done = it
        hist_p = ROOT / "reports" / f"m1_{args.segment}_history.json"
        if hist_p.exists():
            try:
                _h = json.loads(hist_p.read_text(encoding="utf-8"))
                if isinstance(_h, list) and _h:
                    done = max(done, int(_h[-1].get("iter") or 0))
            except Exception:                                # noqa: BLE001
                pass
        if done >= total:
            print(f"[resume] ❌ 已完成 {done} 轮 ≥ 目标 {total}，本段其实已跑完。", flush=True)
            return 2
        cfg.update(init=str(p.relative_to(ROOT)).replace("\\", "/"),
                   iters=total - done, warmup=0, offset=done)
        print(f"[resume] 权重起点 {p.name}（iter {it}），编号已到 {done} → "
              f"再跑 {cfg['iters']} 轮至 {total}；KL 锚点 = 该存档（warmup=0），"
              f"history 追加写。", flush=True)

    fatal = preflight(args.segment, cfg, args.workers)
    if fatal:
        print("\n[preflight] ❌ 未通过：", flush=True)
        for f in fatal:
            print(f"  - {f}", flush=True)
        return 2

    if args.dry_run:
        print("\n[dry-run] 将要执行：\n  " + " ".join(build_cmd(args.segment, cfg, args.workers)))
        return 0
    return launch(args.segment, cfg, args.workers,
                  foreground=not args.detach, settle=args.settle,
                  guards=not args.no_guards)


if __name__ == "__main__":
    raise SystemExit(main())
