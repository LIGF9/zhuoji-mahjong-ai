#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""教师-学生差距分解（外部评审 §1）：DAgger 饱和到底是哪一种饱和？

外部建议原文要求「按决策类型 / 局面阶段算学生 Q 与 master Q 的 MSE 或动作一致率」。
本实现按项目实际口径做了一个**必要修正**：

* 学生策略头输出的是 **logits**，训练目标是 ``CE(logp_S, softmax(Q_T/τ))``
  软标签，**不是 Q 值回归**。因此直接算 ``MSE(logits, Q_T)`` 没有物理意义
  （两者不同量纲、不同自由度）。
* 真正与 Q 同量纲的是 **value head**：它的训练目标就是 ``maxQ/8``，
  所以 ``RMSE(8·V_S − maxQ_T)`` 才是合法的「Q 值误差」（分差单位）。
* 策略侧改用四个可解释指标：``agree@1``、``teacher_CE``、``q_regret``
  （教师最优 Q − 学生所选动作的 Q，分差单位）、``top3_cover``。

分组维度：决策头（discard / self_kong / respond）× 局面阶段
（用 ``glob[0] = wall_left/108`` 的样本 33%/67% 分位切早/中/晚，
保证每组样本量均衡）；respond 另按教师最优动作索引细分。

**预登记解读规则**（写在看到任何结果之前，禁止事后调阈值）：
  ① 若最大 q_regret 组样本占比 < 5% 且该组 regret ≥ 3× 中位组
     ⇒ 「数据覆盖饱和」信号（差距集中在稀有局面）；
  ② 若各决策头 agree@1 极差 < 8pp 且整体 agree@1 < 0.75
     ⇒ 「容量 / 搜索深度饱和」信号（差距均匀，无单点可补）；
  ③ 其余 ⇒ 证据不足/混合，只报数不下结论。
  这是**弱诊断**（组间相对比较），不能替代受控消融；in-sample 偏差在输出中显式标注。

只读、不改任何训练/闸门状态；单次前向 + 统计。低优先级运行（nice 10、
torch 1 线程），避免与加速轮 PPO / 评估抢核。

用法::
    .venv/bin/python tools/ts_gap.py                        # 默认 r12 数据 + exit_best
    .venv/bin/python tools/ts_gap.py --round 11 --student models/exit_r11.pt
    .venv/bin/python tools/ts_gap.py --per-file 2000 --json
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
try:
    os.nice(10)  # 低优先级：加速模式期间不与训练/评估抢核
except (OSError, AttributeError):
    pass

HEADS = ("discard", "self_kong", "respond")
STAGES = ("early", "mid", "late")

# ---- 预登记阈值 ----
RARE_FRACTION = 0.05
CONCENTRATION_RATIO = 3.0
UNIFORM_SPREAD = 0.08
LOW_AGREE = 0.75


def load_samples(round_no: int, per_file: int, seed: int):
    """从某轮 npz 每片随机取 per_file 条（同一 rng，可复现）。"""
    files = sorted((ROOT / f"runs/exit_r{round_no:02d}").glob("exit_w*.npz"))
    if not files:
        raise SystemExit(f"[FAIL] runs/exit_r{round_no:02d}/exit_w*.npz 不存在")
    rng = np.random.default_rng(seed)
    parts = []
    for f in files:
        d = np.load(f)
        n = len(d["head"])
        idx = rng.choice(n, size=min(per_file, n), replace=False)
        parts.append({k: d[k][idx] for k in ("tiles", "glob", "head", "mask", "q")})
    return {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}, files


def pad28(t):
    import torch
    if t.shape[-1] == 28:
        return t
    o = torch.full((*t.shape[:-1], 28), -1e9, device=t.device, dtype=t.dtype)
    o[..., :t.shape[-1]] = t
    return o


def student_logits(net, tiles, glob_, head, batch: int):
    """逐批前向，返回 (n,28) 的策略 logits 与 (n,) 的 value（×8 前）。"""
    import torch
    net.eval()
    lg_out, v_out = [], []
    with torch.no_grad():
        for k0 in range(0, len(head), batch):
            sl = slice(k0, k0 + batch)
            t = torch.from_numpy(tiles[sl].astype(np.float32))
            g = torch.from_numpy(glob_[sl].astype(np.float32))
            out = net(t, g)
            lg = torch.stack([pad28(out[h]) for h in HEADS], dim=1)
            sel = lg[torch.arange(len(head[sl])), torch.from_numpy(head[sl].astype(np.int64))]
            lg_out.append(sel.numpy())
            v_out.append(out["value"].reshape(-1).numpy())
    return np.concatenate(lg_out), np.concatenate(v_out)


def metrics(q, logits, mask, value, tau: float):
    """返回逐样本的指标数组（非法动作一律屏蔽）。"""
    n = len(q)
    qv = np.where(mask, q, -np.inf)
    qv = np.nan_to_num(qv, nan=-np.inf, posinf=-np.inf, neginf=-np.inf)
    lv = np.where(mask, logits, -np.inf)
    t_arg = qv.argmax(1)
    s_arg = lv.argmax(1)
    ok = np.isfinite(qv[np.arange(n), t_arg]) & np.isfinite(lv[np.arange(n), s_arg])

    # 教师软标签（唯一实现，与 train_exit.policy_targets 同源）
    from train_exit import policy_targets
    pt = policy_targets(q.astype(np.float32), mask, tau)
    # 学生分布
    x = lv - np.max(lv, axis=1, keepdims=True)
    ex = np.exp(np.where(np.isfinite(x), x, -np.inf))
    ex = np.nan_to_num(ex, nan=0.0, posinf=0.0, neginf=0.0)
    ps = ex / np.maximum(ex.sum(axis=1, keepdims=True), 1e-12)
    ce = -(pt * np.log(np.clip(ps, 1e-12, None))).sum(1)

    q_top = qv[np.arange(n), t_arg]
    q_chosen = np.where(mask[np.arange(n), s_arg], q[np.arange(n), s_arg], np.nan)
    regret = q_top - q_chosen
    # 教师 top3（合法动作按 Q 降序）
    top3 = np.argsort(-qv, axis=1)[:, :3]
    cover = (top3 == s_arg[:, None]).any(1).astype(float)
    # value 头的 Q 误差（分差量纲）
    qmax = np.nanmax(np.where(mask, q, np.nan), axis=1)
    vgap = 8.0 * value - qmax

    return {"agree": (t_arg == s_arg).astype(float), "ce": ce,
            "regret": regret, "cover": cover, "vgap2": vgap ** 2,
            "ok": ok, "t_arg": t_arg}


def summarize(m: dict, sel: np.ndarray) -> dict | None:
    if sel.sum() < 20:
        return None
    keep = sel & m["ok"] & np.isfinite(m["regret"]) & np.isfinite(m["vgap2"])
    if keep.sum() < 20:
        return None
    r = m["regret"][keep]
    return {"n": int(keep.sum()),
            "agree": float(m["agree"][keep].mean()),
            "ce": float(m["ce"][keep].mean()),
            "regret": float(r.mean()),
            "cover": float(m["cover"][keep].mean()),
            "q_rmse": float(math.sqrt(max(0.0, float(m["vgap2"][keep].mean()))))}


def table(rows: list[tuple[str, dict]]) -> str:
    out = ["| 分组 | n | agree@1 | teacher_CE | q_regret(分) | top3_cover | Q_RMSE(分) |",
           "|---|---|---|---|---|---|---|"]
    for name, s in rows:
        out.append(f"| {name} | {s['n']} | {s['agree']:.3f} | {s['ce']:.3f} | "
                   f"{s['regret']:+.3f} | {s['cover']:.3f} | {s['q_rmse']:.3f} |")
    return "\n".join(out)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="教师-学生差距分解（只读）")
    ap.add_argument("--round", type=int, default=12, help="数据轮次（exit_rNN）")
    ap.add_argument("--student", default="models/exit_best.pt")
    ap.add_argument("--per-file", type=int, default=800)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--tau", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=20261107)
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args(argv)

    import torch
    torch.set_num_threads(1)
    from train_ppo import build_net

    data, files = load_samples(a.round, a.per_file, a.seed)
    n = len(data["head"])
    wall = data["glob"][:, 0].astype(float)
    q33, q67 = np.quantile(wall, [1 / 3, 2 / 3])
    stage = np.where(wall <= q33, 2, np.where(wall <= q67, 1, 0))  # 0=late(牌少) 1=mid 2=early
    print("=" * 100)
    print(f"教师-学生差距分解 ｜ 数据 R{a.round:02d}（{len(files)} 片 × {a.per_file}）"
          f"｜ 学生 {a.student} ｜ n={n}")
    print(f"阶段切分 wall_left/108：late ≤ {q33:.3f} < mid ≤ {q67:.3f} < early"
          "（按样本分位，保证组内样本量均衡）")
    print("口径说明：学生策略头是 logits（软标签 CE，不是 Q 回归）⇒ 不报 MSE(logits,Q)；"
          "Q_RMSE 来自 value head 的 8·V − maxQ_T（分差量纲）。")
    print("=" * 100)

    net, _cfg = build_net(str(ROOT / a.student), "cpu")
    logits, value = student_logits(net, data["tiles"], data["glob"],
                                   data["head"], a.batch)
    m = metrics(data["q"].astype(np.float64), logits.astype(np.float64),
                data["mask"].astype(bool), value.astype(np.float64), a.tau)

    allsel = np.ones(n, bool)
    head_sel = [(f"head={HEADS[h]}", (data["head"] == h)) for h in range(3)]
    stage_sel = [(f"stage={STAGES[s]}", (stage == s)) for s in range(3)]
    cross = [(f"{HEADS[h]}×{STAGES[s]}", (data["head"] == h) & (stage == s))
             for h in range(3) for s in range(3)]
    resp_sub = [(f"respond·教师最优动作={k}", (data["head"] == 2) & (m["t_arg"] == k))
                for k in range(4)]

    groups: dict[str, list[tuple[str, dict]]] = {}
    for title, pairs in (("总计", [("ALL", allsel)]),
                         ("按决策头", head_sel), ("按局面阶段", stage_sel),
                         ("交叉（用于集中性判据）", cross),
                         ("respond 细分", resp_sub)):
        rows = [(name, summarize(m, sel)) for name, sel in pairs]
        rows = [(name, s) for name, s in rows if s is not None]
        groups[title] = rows
        print(f"\n【{title}】")
        print(table(rows))

    # ---- 预登记解读（阈值跑前锁死）----
    cr = dict(groups["交叉（用于集中性判据）"])
    hr = dict(groups["按决策头"])
    worst = max(cr.items(), key=lambda kv: kv[1]["regret"]) if cr else None
    med_reg = float(np.median([v["regret"] for v in cr.values()])) if cr else 0.0
    frac = (worst[1]["n"] / n) if worst else 1.0
    spread = (max(v["agree"] for v in hr.values()) -
              min(v["agree"] for v in hr.values())) if hr else 1.0
    overall = groups["总计"][0][1]["agree"] if groups["总计"] else 0.0
    print("\n" + "-" * 100)
    print("【预登记判据裁决】（阈值在看到结果前锁定；弱诊断，不替代受控消融）")
    print(f"  最大 q_regret 组 = {worst[0] if worst else 'NA'}  "
          f"regret={worst[1]['regret'] if worst else float('nan'):+.3f}  "
          f"样本占比={frac:.1%}  中位组 regret={med_reg:+.3f}")
    print(f"  决策头 agree@1 极差={spread:.3f}  整体 agree@1={overall:.3f}")
    c1 = bool(worst) and frac < RARE_FRACTION and worst[1]["regret"] >= CONCENTRATION_RATIO * max(med_reg, 1e-6)
    c2 = spread < UNIFORM_SPREAD and overall < LOW_AGREE
    if c1:
        verdict = "数据覆盖饱和信号（差距集中在稀有局面 ⇒ 加数据/重采样比换教师更有效）"
    elif c2:
        verdict = "容量 / 搜索深度饱和信号（差距均匀 ⇒ 该查 N64 天花板或模型容量）"
    else:
        verdict = "证据不足 / 混合：只报分组数据，不据此改训练配方"
    print(f"  ⇒ {verdict}")
    print("  [限定] in-sample（该轮 npz 参与了训练），只能作组间相对比较；"
          "不得写成绝对强度或增长率结论。")

    # ---- 探索性观察：明确标注「非预登记、不参与上面裁决」 ----
    # 判据缺陷记录：跨决策头直接比 agree@1 会被合法动作数混淆
    #（respond 4 槽且多为 PASS，discard 最多 28 槽），故 ② 的「极差」在
    # 结构上几乎不可能 <8pp。按预登记纪律**不追溯修改本次裁决**，只把
    # 归一化口径登记为下一次诊断的候选判据。
    nlegal = np.maximum(data["mask"].sum(1), 1)
    print("\n【探索性 · 非预登记，不参与裁决】动作空间归一化与判据缺陷记录")
    base_all = float((1.0 / nlegal).mean())
    print(f"  随机基线 mean(1/n_legal)={base_all:.3f}  "
          f"实际 agree@1={overall:.3f}  提升 ×{overall / max(base_all, 1e-9):.1f}")
    for h in range(3):
        sel = data["head"] == h
        if sel.sum() < 5:
            continue
        base = float((1.0 / nlegal[sel]).mean())
        ag = float(m["agree"][sel & m["ok"]].mean())
        print(f"    {HEADS[h]:>11}: n={int(sel.sum()):5d}  "
              f"随机基线={base:.3f}  agree@1={ag:.3f}  提升 ×{ag / max(base, 1e-9):.1f}")
    print("  head 内阶段 agree@1 极差（不受动作空间基数混淆）：")
    for h in range(3):
        vals = [cr[f"{HEADS[h]}×{STAGES[s]}"]["agree"] for s in range(3)
                if f"{HEADS[h]}×{STAGES[s]}" in cr]
        if vals:
            print(f"    {HEADS[h]:>11}: {min(vals):.3f}~{max(vals):.3f}  "
                  f"极差={max(vals) - min(vals):.3f}")
    resp_counts = {k: int(((data["head"] == 2) & (m["t_arg"] == k)).sum())
                   for k in range(4)}
    print(f"  respond 教师最优动作分布（原始计数）= {resp_counts}"
          "  ← 若几乎全落在同一个动作（PASS），说明稀有响应局面在 100k 状态里覆盖不足")
    print("  [判据缺陷] ② 用跨 head 的 agree@1 极差判「均匀」，但 respond/discard 的"
          "合法动作数不同 ⇒ 结构上不可能 <8pp。按预登记纪律**不追溯改本次结论**；"
          "「head 内阶段极差 + 随机基线归一化」登记为下一次诊断的候选判据。")

    if a.json:
        out = ROOT / f"runs/gates/ts_gap_r{a.round:02d}.json"
        out.write_text(json.dumps({
            "round": a.round, "student": a.student, "n": n,
            "stage_cuts": [float(q33), float(q67)],
            "groups": {k: {name: s for name, s in v} for k, v in groups.items()},
            "verdict": verdict, "rare_fraction": frac,
            "head_agree_spread": spread, "overall_agree": overall,
        }, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"  JSON → {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
