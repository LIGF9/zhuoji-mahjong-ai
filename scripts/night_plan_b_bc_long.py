"""方案 B 续篇：**BC 预热（阶段 1）+ 长轮次单臂对照（A）连跑**。

为什么是这两步
--------------
2026-10-11 的两批消融给出了一个干净但尴尬的结论（`docs/P1-阶段2-消融结论.md` §7）：

* **迁移方式 > 特征质量**：`v1→v2` 热启动必须用 `--new-param-init zero`
  （C1 证明随机初始化光挂权重就掉 2.34 分，C3 证明零初始化能把 P1b 的 −0.67 收回噪声）。
* 但零初始化之后，`a`/`c`/`ac` 三个包**相对 `p_none`（纯多训 400 轮）的增量 ≈ 0**，
  而 `p_none` 自己也是 −0.114 —— **400 轮 × lr 8e-6 对 s11 是过浅预算**。

问题出在零初始化的代价：新权重**整段为 0**，等于"新特征一开始完全没被使用"，
而"学会使用它"这件事被整个丢给了 RL。这正是 400 轮不够用的原因。
本脚本按用户选定的顺序补两步：

* **B（阶段 1，BC 预热）**：用 s11 自己 1 万局对局做监督（`gen_bc_from_model.py`，
  v2 编码），把新增的 16 输入通道 / 28 列 `fc_in` 从 0 抬到"已经会用"的位置，
  再进 RL。验收由 `scripts/verify_bc_warm.py` 把关：① 三头准确率相对起点降 ≤0.5pp；
  ② 新参数非零占比高 + 范数明显抬升（否则 BC 空转，等于没热身）。
* **A（长轮次）**：把臂从 400 轮拉到 1200 轮，看"包的效果是否只是来得慢"。

轮次为什么定在 1200
-------------------
前一批实测 **8.7 – 11.8 s/轮**（`p_ac_z` 400 轮 / 0.968 h 等），本机复测（`packs=none`
臂的前 22 轮）为 **7.8 – 8.8 s/轮**。取 8.7 估算：三臂各 1200 轮 ≈ 8.7 h，加上复评
预留 1.96 h（7 检查点 × 0.28）≈ 10.7 h，装得进 11.5 h 窗口 → 三臂**等长**跑完。
若实测比预估慢（spi > ~9.3），预算闸门会把第三条臂（**BC 对照臂，正是本夜的头条
对比**）截断；截断点强制落在 100 的整数倍上，复评可退到等长快照对拍，不会静默地
拿不同长度的检查点互比。

臂的设计（前置声明，不许看到结果再改）
--------------------------------------
同 seed（20270）、同超参、同 1200 轮，**全部零初始化 / 全部 v2**：

| 臂 | init | packs | 作用 |
|---|---|---|---|
| `p_none_long` | s11 | none | **纯长度对照**：三包全关 = v1 s11 多训 1200 轮（新权重永久为 0） |
| `p_ac_z_long` | s11 | ac | 零初始化 + a+c，1200 轮 |
| `p_ac_bc` | **BC 预热产物** | ac | BC 相对 `p_ac_z_long` 的**单变量增量** |

三条差值各自只动一个变量，因此可直接归因：

* `p_ac_z_long` − `p_none_long` = **a+c 在长预算下的纯净贡献**
* `p_ac_bc` − `p_ac_z_long` = **BC 预热的增量**
* `p_ac_z_long` − s11（锚）= 长训练 + 包的总效果

复评清单里额外放两个**零 RL / 浅 RL**参照，全部丢进同一批牌池 → 配对可比：

* `dushan_b2_bc_ac`（BC 产物本身，0 轮 RL）—— BC 预热单独值多少
* `dushan_b2_p_ac_z`（上一批同一臂的 400 轮版）—— **同一条轨迹上的长度截面**
  （同 seed / 同 init / 同 packs，所以 1200 轮版的前 400 轮就是它）

判读口径
--------
共用牌池 + 共用对手的 **1200 副**配对净分差（`score_checkpoints.py`），
门槛 **净分差 ≥ +0.40 且 95% CI 下界 > 0**（与两批历史实验一致，前置写死）。
注意 CI 半宽约 ±0.27 —— 它只能分辨 ≥0.40 的效应，**小于门槛的差异一律记为"读不出来"，
不许当趋势解释**。

用法::

    python scripts/night_plan_b_bc_long.py --iters 20 --bc-games 400 --skip-score   # 试跑
    python scripts/night_plan_b_bc_long.py                                          # 正式
"""
from __future__ import annotations

import argparse
import ctypes
import json
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = r"C:\Users\LIGF\.workbuddy\binaries\python\envs\zhuoji\Scripts\python.exe"

LOG = ROOT / "reports" / "plan_b_bc_long.log"
SUMMARY = ROOT / "reports" / "plan_b_bc_long_summary.json"
RESCORE_OUT = ROOT / "reports" / "plan_b_bc_long_rescore.json"
BC_VERIFY = ROOT / "reports" / "bc_warm_verify.json"

DATA = ROOT / "data" / "bc_s11_v2.npz"
BC_OUT = ROOT / "models" / "dushan_b2_bc_ac.pt"
BC_HISTORY = ROOT / "reports" / "bc_s11_v2_ac_history.json"

INIT = "models/dushan_big_s11.pt"
ANCHOR = "dushan_big_s11"
POOL = "dushan_big_s3,dushan_big_s7,dushan_big_s11"
REF = "net:dushan_big_s11|teacher:balanced|teacher:tenpai_rush"
SEED = 20270
GATE = 0.40
DEALS = 1200
BC_PACKS = "ac"

#: 训练臂（按执行顺序）。第一条必须是最便宜的对照臂 —— 它的 s/轮 用来
#: 估后面两条的时间，并在预算不足时把轮次缩到 50 的整数倍。
ARM_ORDER = ["p_none_long", "p_ac_z_long", "p_ac_bc"]
ARMS: dict[str, dict] = {
    "p_none_long": {"init": INIT, "packs": "none",
                    "desc": "纯长度对照：三包全关，= v1 s11 多训 1200 轮"},
    "p_ac_z_long": {"init": INIT, "packs": "ac",
                    "desc": "零初始化 + a+c，1200 轮（包在长预算下的纯净效果）"},
    "p_ac_bc": {"init": "models/dushan_b2_bc_ac.pt", "packs": "ac",
                "desc": "BC 预热(a+c) → 1200 轮（BC 相对 p_ac_z_long 的单变量增量）"},
}

#: 复评清单（同一批牌池，配对可比）
RESCORE_EXTRA = [
    "dushan_b2_bc_ac",      # BC 产物本身：0 轮 RL，BC 预热单独值多少
    "dushan_b2_p_ac_z",     # 上一批同一臂的 400 轮版 → 同轨迹的长度截面
    "dushan_b2_p_none",     # 上一批"纯长度对照"的 400 轮版 → 控制臂的长度截面
]

ES_CONTINUOUS = 0x80000000
ES_SYSTEM_REQUIRED = 0x00000001


def awake(on: bool = True) -> bool:
    try:
        ctypes.windll.kernel32.SetThreadExecutionState(
            (ES_CONTINUOUS | ES_SYSTEM_REQUIRED) if on else ES_CONTINUOUS)
        return True
    except Exception:
        return False


def ck_exists(stem: str) -> bool:
    return (ROOT / "models" / f"{stem}.pt").exists()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=1200, help="每臂轮次")
    ap.add_argument("--min-iters", type=int, default=400,
                    help="预算不足时允许缩到的最小轮次；低于它就放弃该臂")
    ap.add_argument("--threads", type=int, default=24)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--deals", type=int, default=DEALS)
    ap.add_argument("--save-every", type=int, default=100,
                    help="中间检查点间隔；臂被截断时用等长快照对拍")
    ap.add_argument("--bc-games", type=int, default=10000)
    ap.add_argument("--bc-epochs", type=int, default=3)
    ap.add_argument("--bc-lr", type=float, default=5.0e-4,
                    help="BC 的 lr（freeze_old 下只动新增切片，可以开大些）；"
                         "验收不过会自动降到 1/3、1/10 重试")
    ap.add_argument("--bc-batch", type=int, default=1024)
    ap.add_argument("--verify-samples", type=int, default=30000)
    ap.add_argument("--max-hours", type=float, default=11.5)
    ap.add_argument("--arms", default=",".join(ARM_ORDER))
    ap.add_argument("--skip-bc", action="store_true", help="跳过 BC（数据/训练都跳过）")
    ap.add_argument("--skip-score", action="store_true")
    ap.add_argument("--ignore-verify-fail", action="store_true",
                    help="BC 验收没过也继续（仅用于试跑）")
    ap.add_argument("--force-retrain", action="store_true")
    args = ap.parse_args()

    log = open(LOG, "a", encoding="utf-8", buffering=1)

    def say(msg: str) -> None:
        print(f"[{time.strftime('%m-%d %H:%M:%S')}] {msg}", file=log)

    say("=" * 68)
    say(f"方案 B 续篇：BC 预热（阶段 1）+ {args.iters} 轮长训练 × 3 臂"
        f"（threads={args.threads} workers={args.workers}，预算 {args.max_hours} h）")
    for a in ARM_ORDER:
        say(f"  · {a:<12} init={ARMS[a]['init']:<28} packs={ARMS[a]['packs']:<4} "
            f"{ARMS[a]['desc']}")

    want = [a.strip() for a in args.arms.split(",") if a.strip()]
    unknown = [a for a in want if a not in ARMS]
    if unknown:
        raise SystemExit(f"未知的臂：{unknown}；可选 {list(ARMS)}")

    say(f"防休眠={'OK' if awake(True) else '失败'}")
    t_start = time.time()

    state: dict = {
        "iters": args.iters, "gate": GATE, "anchor": ANCHOR, "ref": REF,
        "deals": args.deals, "seed": SEED, "new_param_init": "zero",
        "bc_packs": BC_PACKS, "bc_games": args.bc_games,
        "note": "阶段 1 BC 预热 + 长轮次 3 臂；三条差值各只动一个变量",
        "arms": {}, "recipe": {a: {"init": ARMS[a]["init"], "packs": ARMS[a]["packs"],
                                   "desc": ARMS[a]["desc"]} for a in ARM_ORDER},
    }
    if SUMMARY.exists():
        try:
            old = json.loads(SUMMARY.read_text(encoding="utf-8"))
            if old.get("arms"):
                state["arms"] = old["arms"]
                say(f"已有 {len(old['arms'])} 臂历史结果（完成臂将跳过）")
            # 断点续跑：BC 产物 + 验收记录也一并继承，避免重训（一次 ~7 min）。
            # 只认**验收通过**的记录；否则宁可重跑，别把脏产物带进长训练。
            for k in ("data_check", "bc", "bc_verify", "bc_rejected"):
                if old.get(k) is not None:
                    state[k] = old[k]
            if state.get("bc"):
                say(f"  继承 BC 记录（rc={state['bc'].get('rc')}，"
                    f"验收 ok={(state.get('bc_verify') or {}).get('ok')}）")
        except Exception:
            pass

    # ---- 1. BC 数据 ---------------------------------------------------------
    bc_ok = True
    if not args.skip_bc:
        say("—— 阶段 1a：用 s11 生成 v2 编码的 BC 数据 ——")
        if DATA.exists() and not args.force_retrain:
            say(f"  已有 {DATA.name}，跳过生成")
        else:
            cmd = [PY, "scripts/gen_bc_from_model.py",
                   "--model", INIT, "--games", str(args.bc_games),
                   "--encoder", "v2", "--out", str(DATA.relative_to(ROOT)),
                   "--threads", str(args.threads), "--seed", str(SEED)]
            say("  cmd: " + " ".join(cmd[2:]))
            t0 = time.time()
            with open(ROOT / "reports" / "dushan_bc_s11_v2_gen.log", "w",
                      encoding="utf-8") as f:
                rc = subprocess.run(cmd, cwd=str(ROOT), stdout=f,
                                    stderr=subprocess.STDOUT).returncode
            say(f"  数据生成 rc={rc} 用时 {(time.time() - t0) / 60:.1f} min")
            bc_ok &= (rc == 0 and DATA.exists())
        # 数据体检：**先查形状**（曾有 --encoder v2 被模型自带属性盖掉、
        # 生成出"标着 v2 实为 15 通道"的数据），再查两个包的特征是否真有信号。
        if DATA.exists():
            try:
                import numpy as np
                d = np.load(DATA)
                enc = int(np.asarray(d["encoder_version"]).ravel()[0]) \
                    if "encoder_version" in d.files else 1
                n = len(d["head"])
                shape = (int(d["tiles"].shape[1]), int(d["glob"].shape[1]))
                say(f"  数据体检：声明编码器 v{enc}，形状 {shape}，样本 {n:,}")
                assert shape == (31, 98), (
                    f"v2 数据必须是 (31, 98)，实际 {shape} —— 编码器版本没生效")
                assert enc == 2, f"数据声明编码器 v{enc}，应为 v2"
                sub = d["tiles"][:5000]
                gsub = d["glob"][:5000]
                a_nz = float((sub[:, 15:27] != 0).mean())
                c_nz = float((gsub[:, 82:98] != 0).mean())
                say(f"  P1a 通道非零占比 {a_nz:.4f}，P1c 全局维非零占比 {c_nz:.4f}")
                state["data_check"] = {"encoder": enc, "samples": n,
                                       "shape": list(shape),
                                       "a_nonzero": a_nz, "c_nonzero": c_nz}
                if not (a_nz > 0.01 and c_nz > 0.01):
                    say("  ⚠ 有包的特征近乎整段为零 → 该包的 BC 热身必然空转")
                    bc_ok = False
            except Exception as e:                                   # noqa: BLE001
                say(f"  ⚠ 数据体检不通过：{e}")
                bc_ok = False

        # ---- 2. BC 训练 + 3. 验收（lr 阶梯自动降档重试） ---------------------
        # 只训练新增参数（--freeze-old）：目标是"让新特征学会按教师的分布说话"，
        # 而不是把已经调好的共享策略重训一遍（实测 1 epoch lr=3e-4 就掉 1.1pp）。
        # 若验收没过就**降 lr 重试**（1/3、1/10），而不是直接中止 —— 验收门槛偏严
        # 不该让整晚空转；三次都不过才认输。
        lr_ladder = [args.bc_lr, args.bc_lr / 3, args.bc_lr / 10]
        v_ok = False
        if not bc_ok:
            say("  ⚠ 阶段 1a 的数据体检未通过 → 不做 BC，直接判定失败")
            if not args.ignore_verify_fail:
                awake(False)
                state["aborted"] = "bc_data_check_failed"
                state["elapsed_hours"] = round((time.time() - t_start) / 3600, 2)
                SUMMARY.write_text(json.dumps(state, ensure_ascii=False, indent=2),
                                   encoding="utf-8")
                return 1
        if BC_OUT.exists() and BC_VERIFY.exists() and not args.force_retrain \
                and state.get("bc"):
            state["bc_verify"] = json.loads(BC_VERIFY.read_text(encoding="utf-8"))
            v_ok = bool(state["bc_verify"].get("ok")) or args.ignore_verify_fail
            say(f"已有 BC 产物与验收记录（ok={v_ok}），跳过阶段 1b/1c")
        for attempt, lr in enumerate(lr_ladder):
            if v_ok:
                break
            say(f"—— 阶段 1b：BC 预热（packs={BC_PACKS}，freeze_old=on，"
                f"lr={lr:.2e}，{args.bc_epochs} epochs，第 {attempt + 1} 次）——")
            cmd = [PY, "scripts/train_bc.py",
                   "--data", str(DATA.relative_to(ROOT)),
                   "--out", str(BC_OUT.relative_to(ROOT)),
                   "--history", str(BC_HISTORY.relative_to(ROOT)),
                   "--encoder", "v2", "--encoder-packs", BC_PACKS,
                   "--init", INIT, "--new-param-init", "zero", "--freeze-old",
                   "--epochs", str(args.bc_epochs), "--batch", str(args.bc_batch),
                   "--lr", str(lr), "--threads", str(args.threads),
                   "--seed", str(SEED)]
            say("  cmd: " + " ".join(cmd[2:]))
            t0 = time.time()
            with open(ROOT / "reports" / "dushan_b2_bc_ac_train.log", "w",
                      encoding="utf-8") as f:
                rc = subprocess.run(cmd, cwd=str(ROOT), stdout=f,
                                    stderr=subprocess.STDOUT).returncode
            say(f"  BC 训练 rc={rc} 用时 {(time.time() - t0) / 60:.1f} min")
            state["bc"] = {"cmd": cmd[2:], "rc": rc, "epochs": args.bc_epochs,
                           "lr": lr, "freeze_old": True, "games": args.bc_games,
                           "hours": round((time.time() - t0) / 3600, 3)}
            SUMMARY.write_text(json.dumps(state, ensure_ascii=False, indent=2),
                               encoding="utf-8")
            if rc != 0 or not BC_OUT.exists():
                bc_ok = False
                break

            say("—— 阶段 1c：BC 验收（不退化 + 新参数真装上）——")
            cmd = [PY, "scripts/verify_bc_warm.py",
                   "--data", str(DATA.relative_to(ROOT)), "--init", INIT,
                   "--bc", str(BC_OUT.relative_to(ROOT)),
                   "--encoder", "v2", "--encoder-packs", BC_PACKS,
                   "--new-param-init", "zero",
                   "--samples", str(args.verify_samples), "--out",
                   str(BC_VERIFY.relative_to(ROOT))]
            with open(ROOT / "reports" / "bc_warm_verify.log", "w",
                      encoding="utf-8") as f:
                vrc = subprocess.run(cmd, cwd=str(ROOT), stdout=f,
                                     stderr=subprocess.STDOUT).returncode
            if BC_VERIFY.exists():
                v = json.loads(BC_VERIFY.read_text(encoding="utf-8"))
                state["bc_verify"] = v
                # 注意：accuracy 条目**只有数值**（n/before/after/drop_pp），没有 ok
                # —— 一致率降幅是描述性指标，门槛在 new_params 与顶层 ok 上。
                for name, r in (v.get("accuracy") or {}).items():
                    say(f"  acc(描述) {name:10s} {r['before']:.4f} → {r['after']:.4f}"
                        f"（{r['drop_pp']:+.2f}pp）")
                for k, r in (v.get("new_params") or {}).items():
                    if k.startswith("_"):
                        continue
                    say(f"  新权重 {k:12s} l2 {r['before']['l2']:.4f} → "
                        f"{r['after']['l2']:.4f}"
                        f"（旧权重的 {r.get('absmean_vs_old', 0) * 100:.1f}%），"
                        f"非零 {r['before']['nonzero_frac']:.3f} → "
                        f"{r['after']['nonzero_frac']:.3f}"
                        f"  {'OK' if r.get('ok') else 'FAIL'}")
                say(f"  高置信决策被改坏 {v.get('conf_flip_rate', float('nan')) * 100:.2f}%"
                    f"（门槛 ≤ {v.get('max_conf_flip', 0.03) * 100:.1f}%）"
                    f"  {'OK' if v.get('ok_drift') else 'FAIL'}")
            v_ok = (vrc == 0) and bool(
                (state.get("bc_verify") or {}).get("ok"))
            say(f"  BC 验收 {'通过 ✅' if v_ok else '未通过 ❌'}")
            SUMMARY.write_text(json.dumps(state, ensure_ascii=False, indent=2),
                               encoding="utf-8")

        if not v_ok and not args.ignore_verify_fail:
            # 不整晚作废：BC 过不了就把 p_ac_bc 这一臂从队列里摘掉，
            # 剩下的两条对照臂（纯长度 / 包效果）照跑 —— 它们不依赖 BC。
            say("  ⚠ BC 验收三次都没过 → **放弃 p_ac_bc 臂**，"
                "只跑 p_none_long / p_ac_z_long 两条对照臂"
                "（BC 失败的结论本身也是结果，会记进汇总）")
            state["bc_rejected"] = True
            want = [a for a in want if a != "p_ac_bc"]
            if not want:
                awake(False)
                state["aborted"] = "bc_verify_failed"
                state["elapsed_hours"] = round((time.time() - t_start) / 3600, 2)
                SUMMARY.write_text(json.dumps(state, ensure_ascii=False, indent=2),
                                   encoding="utf-8")
                return 1
        bc_ok &= v_ok or args.ignore_verify_fail
    else:
        say("--skip-bc：跳过阶段 1")

    # ---- 4. 逐臂长训练 ------------------------------------------------------
    rescore_ckpts = [ANCHOR] + [f"dushan_b2_{a}" for a in ARM_ORDER
                                if a in want] + RESCORE_EXTRA
    # 复评预留按**计划**清单算（不能只看当下存在的文件：新臂还没训出来，
    # 按存在数算会把预留压到 0.3 h，等真到复评时才发现预算不够）
    n_score = len(rescore_ckpts)
    reserve_h = 0.0 if args.skip_score else 0.28 * n_score
    say(f"复评计划 {n_score} 个检查点 → 预留 {reserve_h:.2f} h")

    spi_seen: list[float] = []
    for i, arm in enumerate(want):
        out = ROOT / "models" / f"dushan_b2_{arm}.pt"
        rec0 = state["arms"].get(arm) or {}
        if out.exists() and rec0.get("finished") and not args.force_retrain:
            say(f"[skip] {arm} 已完成，跳过")
            if rec0.get("s_per_iter"):
                spi_seen.append(float(rec0["s_per_iter"]))
            continue

        # 预算闸门：按已实测的 s/轮 反推本臂能跑多少轮。**必须落在
        # save_every 的整数倍上**，否则截断后的臂拿不到「等长快照」，
        # 复评会静默地拿不同长度的检查点互比（曾经踩过）。
        it_arm = args.iters
        if i > 0 and spi_seen:
            spi = max(spi_seen)
            left = args.max_hours * 3600 - (time.time() - t_start) - reserve_h * 3600
            step = max(1, args.save_every)
            it_arm = int((left - 420) / spi / step) * step
            it_arm = max(0, min(args.iters, it_arm))
            if it_arm < args.min_iters:
                say(f"[budget] 剩余 {left / 3600:.2f} h 只够 {it_arm} 轮 "
                    f"（< 下限 {args.min_iters}），停止启动新臂")
                break
            if it_arm < args.iters:
                say(f"[budget] 本臂轮次缩为 {it_arm}（原 {args.iters}，"
                    f"落在 {step} 的整数倍上），"
                    f"复评时与其它臂用等长快照对拍")

        spec = ARMS[arm]
        glog = ROOT / "reports" / f"dushan_b2_{arm}.log"
        cmd = [PY, "scripts/selfplay.py",
               "--init", spec["init"], "--out", f"models/dushan_b2_{arm}.pt",
               "--history", f"reports/rl_history_b2_{arm}.json",
               "--iters", str(it_arm), "--games-per-iter", "96",
               "--batch", "512", "--lr", "8.0e-6", "--temperature", "0.95",
               "--teacher-prob", "0.08",
               "--threads", str(args.threads), "--vectorized",
               "--workers", str(args.workers),
               "--eval-every", "20", "--eval-rounds", "60",
               "--select", "guard", "--select-metric", "vs_anchor",
               "--eval-anchor", ANCHOR,
               "--save-every", str(args.save_every), "--snapshot-pool", POOL,
               "--encoder", "v2", "--encoder-packs", spec["packs"],
               "--spatial-head", "off",
               "--new-param-init", "zero",
               "--seed", str(SEED), "--rules", "dushan"]
        say(f"—— {arm}：{spec['desc']} ——")
        say("  cmd: " + " ".join(cmd[2:]))
        t0 = time.time()
        with open(glog, "w", encoding="utf-8") as f:
            rc = subprocess.run(cmd, cwd=str(ROOT), stdout=f,
                                stderr=subprocess.STDOUT).returncode
        dt = time.time() - t0
        text = glog.read_text(encoding="utf-8", errors="ignore")
        rec = {"desc": spec["desc"], "init": spec["init"], "packs": spec["packs"],
               "cmd": cmd[2:], "rc": rc, "hours": round(dt / 3600, 3),
               "iters": it_arm,
               "finished": "完成。" in text,
               "s_per_iter": round(dt / max(1, it_arm), 2),
               "head": [ln for ln in text.splitlines()
                        if ln.startswith(("[warm]", "[编码]", "[pool]"))],
               "select_lines": [ln for ln in text.splitlines()
                                if ln.startswith("[select]")]}
        state["arms"][arm] = rec
        state["elapsed_hours"] = round((time.time() - t_start) / 3600, 2)
        SUMMARY.write_text(json.dumps(state, ensure_ascii=False, indent=2),
                           encoding="utf-8")
        for ln in rec["head"]:
            say("  " + ln)
        say(f"  {arm} rc={rc} 完成={rec['finished']} {dt / 3600:.2f} h"
            f"（{rec['s_per_iter']} s/轮 × {it_arm}）")
        if rc != 0 or not out.exists():
            say(f"  ⚠ {arm} 未正常产出模型 → 本臂作废（rc≠0 的臂不许沿用）")
        else:
            spi_seen.append(rec["s_per_iter"])

    # ---- 5. 同一批牌池复评 --------------------------------------------------
    if not args.skip_score:
        ckpts = [c for c in rescore_ckpts if ck_exists(c)]
        # 有臂被截断 → 补等长快照，保证"同长度"这件事有数据支撑
        short = [r["iters"] for r in state["arms"].values()
                 if r.get("iters") and r["iters"] < args.iters]
        if short:
            step = max(1, args.save_every)
            m = (int(min(short)) // step) * step      # 必须落在快照边界上
            got = []
            for a in ARM_ORDER:
                sn = f"_ckpts/dushan_b2_{a}_it{m:04d}"
                if ck_exists(sn):
                    ckpts.append(sn)
                    got.append(a)
            say(f"有臂被截断到 {int(min(short))} 轮 → 追加等长快照 {m} 轮"
                f"（拿到 {len(got)}/{len(ARM_ORDER)} 臂：{','.join(got) or '无'}）")
            if len(got) < len(ARM_ORDER):
                say("  ⚠ 等长快照不齐 → 截断臂的差值**只可作点估计参考**，"
                    "不可当门槛依据")
        say(f"复评：{len(ckpts)} 个检查点 × {args.deals} 副"
            f"（共用牌池 + 共用对手，配对可比）")
        subprocess.run(
            [PY, "scripts/score_checkpoints.py",
             "--ckpts", ",".join(ckpts), "--ref", REF,
             "--deals", str(args.deals), "--threads", str(args.threads),
             "--anchor", ANCHOR, "--out", str(RESCORE_OUT.relative_to(ROOT)),
             "--no-per-deal"],
            cwd=str(ROOT), stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
        if RESCORE_OUT.exists():
            sc = json.loads(RESCORE_OUT.read_text(encoding="utf-8"))
            rows = {r["ckpt"]: r for r in sc.get("rows", [])}
            state["rescore"] = rows
            state["rescore_ckpts"] = ckpts
            say("—— 复评（相对 s11 的 1200 副配对净分差；门槛 ≥ +0.40 且 CI 下界 > 0）——")
            for c in ckpts:
                r = rows.get(c) or {}
                val, ci = r.get("vs_anchor"), r.get("vs_anchor_ci95")
                if not isinstance(val, (int, float)):
                    say(f"  {c:<38} （无数据）")
                    continue
                lo = val - (ci or 0.0)
                say(f"  {c:<38} {val:+.3f} ± {(ci or 0):.3f}  CI下界 {lo:+.3f}  → "
                    f"{'过门槛 ✅' if (val >= GATE and lo > 0) else '未过门槛'}")
            play = {k.replace("dushan_b2_", ""): v.get("vs_anchor")
                    for k, v in rows.items()}

            def d(a: str, b: str):
                x, y = play.get(a), play.get(b)
                return None if x is None or y is None else round(x - y, 4)

            state["deltas"] = {
                "ac_vs_none_long": d("p_ac_z_long", "p_none_long"),
                "bc_vs_ac_long": d("p_ac_bc", "p_ac_z_long"),
                "ac_long_vs_s11": d("p_ac_z_long", "dushan_big_s11"),
                "ac_long_vs_ac_400": d("p_ac_z_long", "p_ac_z"),
                "bc0_vs_s11": d("bc_ac", "dushan_big_s11"),
            }
            say("—— 三条单变量差值（点估计；小于 ±0.27 一律记为读不出来）——")
            for k, v in state["deltas"].items():
                say(f"  {k:<20} " + ("（缺数据）" if v is None else f"{v:+.4f}"))
            SUMMARY.write_text(json.dumps(state, ensure_ascii=False, indent=2),
                               encoding="utf-8")
        else:
            say("复评未产出 json")

    awake(False)
    state["elapsed_hours"] = round((time.time() - t_start) / 3600, 2)
    SUMMARY.write_text(json.dumps(state, ensure_ascii=False, indent=2),
                       encoding="utf-8")
    say(f"队列结束，累计 {state['elapsed_hours']} h，汇总 -> {SUMMARY}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
