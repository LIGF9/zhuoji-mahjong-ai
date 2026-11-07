#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""教师天花板探测：同一批固定决策状态下 N32 vs N64 的搜索输出差异。

**状态（2026-11-07 17:02）：一次性诊断，已跑完并归档，不进自检常规循环。**
结果见 ``runs/gates/n64_probe.json`` / ``runs/night_logs/n64_probe.log``，
研判见 ``docs/14天训练计划.md`` §6.34.7。此后**不要重复跑**：
16:53 曾有两个会话并行跑本工具（80 状态版）与主训练抢核，双双被 SIGTERM；
加速轮 PPO 满占 CPU 时本工具本就不该启动。

外部评审 §6 要求「用 30min 测 master 在 N64 下是否更强」来判断「教师天花板
是否真到顶」。本项目的现实约束是 7×24 加速模式把 CPU 占满，**完整对局评估
（50 seed × 四座 × 搜索）约 35~40min 且与训练/闸门抢核**，会污染正在跑的
配对评估。因此先做一个零对局、低并发的**代理探测**：

  在固定 seed 的真实对局轨迹上采样 N 个决策点，对**同一个 game 状态**
  分别跑 ``deep_q(..., n_det=32 / 64, ply=3, topk=3, ff_cap=24)``
  （与线上 N32 口径完全同参，只改 n_det，且两次用同一 seed ⇒ N64 的前
  32 个 determinization 与 N32 相同，差异只来自新增的 32 个，配对噪声最小）。

**这不是强度判定**，只是「N64 是否包含 N32 没有的信息」的上界代理：
若连决策/Q 都几乎不变，则对局强度显著提升的可能性极低；若明显分叉，
才值得花 40min 跑正式 ``eval_pairs.py --a-n-det 64 --b-n-det 32`` 对局配对。
本工具输出**不进任何闸门、不改任何晋升/分支判定**。

**口径边界（2026-11-07 用户提醒后补，务必区分两种「N128」）**：
线上「⚡ 极算 · 128 路推演」（``web/dushan_server.py`` 的 ``search128``）是
**ply=1 × n_det=128**；而本工具与评估器的基线是 **ply=3 × n_det=32**（训练教师
master 的同参口径）。前者更便宜是**少两层前瞻**换来的，不是「128 路比 32 路便宜」。
所以：
  · 线上 240 副实测「N128 与 32 路同级、略优但不显著、且更便宜」——是**另一种
    搜索配置**的支持性证据，**不能**直接推成「ply3 下把 n_det 32→64 无收益」；
  · 「N64 教师让标注算力翻倍、超 3.5h 超时线」讲的是 **ply3 × n_det=64**，
    与线上 ply1 N128 不矛盾（见 §6.34.7 成本硬约束）；
  · 本工具只回答 ply3 下的窄问题，正式对局判定仍按 §6.34.7 排在 DEADLINE 后
    终评窗口（训练已停、零机会成本），**不在 7×24 窗口内跑**。

**预登记判读规则**（看到结果前锁死，禁止事后调阈值）：
  ① 一致率 ≥ 0.95 且 mean|ΔQ| < 0.30 分 ⇒ 「N64 与 N32 近乎同源」，
     教师天花板大概率已到；正式 50-seed 对局按排期执行，加速模式不改。
  ② 一致率 < 0.85 或 mean|ΔQ| ≥ 0.50 分 ⇒ 「N64 含实质新信息」，
     正式对局判定提前到最近维护窗口；若正式判定 diff ≥ +2SE 且 p<0.05，
     在 R24 DAgger 重锚点启用 N64 教师（分支见 docs/14天训练计划.md §6.34）。
  ③ 其余 ⇒ 证据不足，按排期跑正式版，不据代理指标改任何配方。

**② 的实际处置修正（记录在案，不删阈值、不改历史 JSON）**：本次 32 状态跑出
一致率 0.781，机械触发 ②，``runs/gates/n64_probe.json`` 里的 ``verdict`` 就是工具
按上述字面打印的原文，**保留不改以供审计**。但自检 #30（§6.34.7）已据三条实测
把行动项改掉：mean|ΔQ| 仅 0.143 分（比 ② 另一阈值低 3.5 倍）、换动作的 7 例中
N64 所选在 N32 口径下平均**落后** 0.201 分（更像平局抖动而非找到好棋）、
且 ply3×64 教师标注成本翻倍会超 3.5h 超时线整轮丢数据 ⇒
**正式对局推迟到 DEADLINE 后终评窗口**，7×24 主循环不启用 N64 教师。
读 JSON 时请以 §6.34.7 为准，**禁止**据该 verdict 宣称「已验证 N64 更强 / 天花板」。

低优先级运行（nice 10、torch 1 线程），默认 40 个状态。

用法::
    .venv/bin/python tools/n64_probe.py                 # 40 状态
    .venv/bin/python tools/n64_probe.py --states 80 --json
"""
from __future__ import annotations

import argparse
import gc
import json
import math
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
try:
    sys.stdout.reconfigure(line_buffering=True)   # 重定向到日志时也能实时看到进度
except (AttributeError, OSError):
    pass
try:
    os.nice(10)
except (OSError, AttributeError):
    pass

# ---- 预登记阈值 ----
HOMO_AGREE = 0.95
HOMO_DQ = 0.30
NOVEL_AGREE = 0.85
NOVEL_DQ = 0.50


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="N32/N64 教师天花板代理探测（只读）")
    ap.add_argument("--states", type=int, default=40)
    ap.add_argument("--seed0", type=int, default=2_400_000,
                    help="探测专用牌墙池，与闸门 A/C/D 及终评池不相交")
    ap.add_argument("--model", default="models/dushan_master.pt")
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args(argv)

    import torch
    torch.set_num_threads(1)
    from eval_pairs import _load_net
    from zhuoji.agent import NetAgent
    from zhuoji.dushan import DushanConfig, DushanGame
    from zhuoji.rules import Phase
    from zhuoji.rust_search import deep_q

    net = _load_net(str(ROOT / a.model))
    rng = np.random.default_rng(20261107)
    rows = []
    tried = 0
    print("=" * 96)
    print(f"教师天花板代理探测 ｜ 固定 {a.states} 个决策状态 ｜ "
          f"N32 vs N64（同 seed、同 ply3/topk3/ff_cap24）｜ 牌墙池 {a.seed0}+")
    print("性质：决策级差异上界代理，**不是对局强度判定**，不进任何闸门。")
    print("=" * 96, flush=True)

    for h in range(400):
        if len(rows) >= a.states:
            break
        g = DushanGame(DushanConfig(), seed=a.seed0 + h)
        agents = [NetAgent(net, seed=(a.seed0 + h) * 131 + s, temperature=0.0)
                  for s in range(4)]
        steps = 0
        while g.phase != Phase.OVER and steps < 600 and len(rows) < a.states:
            acts = g.legal_actions()
            if not acts:
                break
            p = g.actor()
            if len(acts) >= 2 and steps > 4 and rng.random() < 0.28:
                tried += 1
                seed = (a.seed0 + h) * 1000 + steps
                try:
                    q32, _ = deep_q(net, g, p, acts, n_det=32, ff_cap=24,
                                    topk=3, ply=3, seed=seed, device="cpu")
                    q64, _ = deep_q(net, g, p, acts, n_det=64, ff_cap=24,
                                    topk=3, ply=3, seed=seed, device="cpu")
                except Exception as e:            # noqa: BLE001
                    print(f"  [skip] 状态 {tried}: {type(e).__name__}: {e}", flush=True)
                else:
                    q32 = np.asarray(q32, dtype=float)
                    q64 = np.asarray(q64, dtype=float)
                    if q32.shape == q64.shape and np.isfinite(q32).all() and np.isfinite(q64).all():
                        i32, i64 = int(q32.argmax()), int(q64.argmax())
                        dq = np.abs(q32 - q64)
                        rows.append({
                            "n_act": len(acts), "same": i32 == i64,
                            "dq_mean": float(dq.mean()), "dq_max": float(dq.max()),
                            "n32_gap_of_n64_pick": float(q32[i32] - q32[i64]),
                            "q32_top": float(q32[i32]), "q64_top": float(q64[i64]),
                        })
                        if len(rows) % 10 == 0:
                            print(f"  ... 已采 {len(rows)}/{a.states}", flush=True)
                            gc.collect()
            a_act = agents[p].choose(g) if len(acts) >= 2 else acts[0]
            if a_act is None:
                a_act = acts[0]
            g.step(a_act)
            steps += 1
        del agents, g
        gc.collect()

    if len(rows) < min(10, a.states):
        print(f"[FAIL] 有效状态仅 {len(rows)} 个（目标 {a.states}），不下结论")
        return 1

    agree = float(np.mean([r["same"] for r in rows]))
    dqm = float(np.mean([r["dq_mean"] for r in rows]))
    dqx = float(np.max([r["dq_max"] for r in rows]))
    gaps = [r["n32_gap_of_n64_pick"] for r in rows if not r["same"]]
    print(f"\n有效状态 {len(rows)} ｜ 平均合法动作 {np.mean([r['n_act'] for r in rows]):.1f}")
    print(f"  top1 一致率            = {agree:.3f}  （{int(agree*len(rows))}/{len(rows)}）")
    print(f"  mean|Q32−Q64| / 动作   = {dqm:.3f} 分   max = {dqx:.3f} 分")
    if gaps:
        print(f"  换动作的 {len(gaps)} 例中，N64 所选动作在 N32 下落后 top1 "
              f"平均 {np.mean(gaps):.3f} 分（越小越像平局抖动）")
    else:
        print("  两次搜索从未换动作")

    hom = agree >= HOMO_AGREE and dqm < HOMO_DQ
    nov = agree < NOVEL_AGREE or dqm >= NOVEL_DQ
    print("\n" + "-" * 96)
    print("【预登记判读】（阈值在看到结果前锁定）")
    print(f"  ① 同源：一致率 ≥ {HOMO_AGREE} 且 mean|ΔQ| < {HOMO_DQ}")
    print(f"  ② 新信息：一致率 < {NOVEL_AGREE} 或 mean|ΔQ| ≥ {NOVEL_DQ}")
    if hom:
        verdict = ("N64 与 N32 近乎同源 ⇒ 教师天花板大概率已到；"
                   "正式 50-seed 对局按排期执行，加速模式不改")
    elif nov:
        verdict = ("N64 含实质新信息 ⇒ 正式对局判定提前到最近维护窗口；"
                   "若显著更强，在 R24 重锚点启用 N64 教师")
    else:
        verdict = "证据不足 ⇒ 按排期跑正式 50-seed 对局，不据代理指标改配方"
    print(f"  ⇒ {verdict}")

    if a.json:
        out = ROOT / "runs/gates/n64_probe.json"
        out.write_text(json.dumps({
            "states": len(rows), "seed0": a.seed0, "model": a.model,
            "agree": agree, "dq_mean": dqm, "dq_max": dqx,
            "n_changed": len(gaps), "n32_gap_of_n64_pick_mean":
                float(np.mean(gaps)) if gaps else None,
            "verdict": verdict,
            "note": "决策级代理，非对局强度判定；不进任何闸门",
        }, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"  JSON → {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
