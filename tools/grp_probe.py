#!/usr/bin/env python3
"""优先级一（GRP / 冻结密集信号源）的**前置信号探针**：V_θ 到底漂移多少？

只读、零对局、零训练、零落盘 ckpt。单线程 torch 批量前向 + nice，可与
annotate / train_exit / 闸门并发（实测 <10s，不抢核）。

为什么先做这个
--------------
外部建议要用「专门训练、相对冻结的 GRP」替代 Φ=8·V_θ，理由是 V_θ 在训练中漂移。
但本项目 §6.35.2② 已用 `tests_ppo_stationarity.py` 证实 **V_θ 在每次 rollout 内冻结**，
只在轮边界随 ckpt 更换；且 8 轮 PPO 的 kl 仅 0.006~0.009、lr 5e-5、2 epochs。
⇒ 真正需要量的是「**跨 8 轮**的 value 漂移有多大」，而不是有没有漂移。
若漂移远小于闸门 SE(≈0.55 分) 与 A 门绝对强度(≈+4.96 分)，则 GRP 的锚定收益
可忽略，不值得为它占用 7×24 的 CPU 窗口 —— 先测再决定要不要跑贵实验。

项目化说明（Suphx GRP → 本架构的最小实现）
------------------------------------------
Suphx 用 GRU 聚合「本局 + 之前所有局」预测最终奖励；本项目每局独立结算、无半庄/
跨局延续，GRU 跨局摘要不适用。但项目**已有天然的冻结价值源**：`exit_r12.pt`
（DAgger 主线交付物）或 `dushan_master.pt` 的 value head，全链量纲一致
（V = maxQ/8，Φ=8·V 还原到分差量纲），且相对 PPO Actor-Critic 完全冻结。
因此 GRP 在此的最小实现是 `Φ_GRP(s) = 8·V_frozen(s)`，**零训练成本**；
且 V_frozen 可在合并分片后批量前向一次算出，**零 rollout 额外开销**
（§6.37 已实测「逐决策 B=1 前向」才是瓶颈，批量前向很便宜）。

预登记判据（**2026-11-07 20:2x，看到任何数值前锁死**）
------------------------------------------------------
G1（决定是否跑 E1 对照）：在 R12 的 4000 个留出状态上，取 PPO 链两端
   `exit_p13.pt` / `exit_p20.pt` 的 value，报告 `mean|ΔV| × 8`（分差量纲）。
   - < 0.50 分 ⇒ 「V_θ 跨 8 轮漂移不足以构成问题，GRP 锚定收益可忽略」⇒
     **不占用 CPU 窗口跑 E1 对照**，只登记离线结论，CPU 预算留主链；
   - >= 0.50 分 ⇒ 「漂移可观，冻结密集信号源值得实测」⇒ 按 §6.38 规格
     在下一个完整 DAgger 轮的 train 段（CPU 空闲）执行 E1 严格配对对照。
   0.50 分 ≈ 闸门 SE(0.55)，不是可调超参，不得事后改。
G2（辅证）：冻结源 `exit_r12.pt` 与 PPO 链 value 的 Pearson r 与 mean|Δ|×8。
   r>0.90 且差小 ⇒ 换源不带来新信息；r 低 ⇒ 冻结源确是不同信号。
G3（辅证，小样本）：用 `runs/ppo_shards/` 的冒烟 16 局（含 rets），比较
   V 对该局最终 R/8 的预测精度（Pearson r、RMSE(V×8, R)），仅作量级参考，
   **不参与 G1 决策**；同时校验重算 V_p20 与 rollout 落盘 `val` 逐位一致。
G1/G2 的 R12 状态参与过训练（in-sample）且为 DAgger 分布，与 PPO 纯策略分布有偏，
   只能作**相对比较**，不得写成绝对精度或强度结论（同 §6.34.6 限定）。

用法：.venv/bin/python tools/grp_probe.py
输出：runs/shadow/grp_probe.json + 控制台结论
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
OUT = ROOT / "runs" / "shadow" / "grp_probe.json"
N_R12 = 4000
G1_THRESHOLD = 0.50          # 分差量纲，锁死
G2_R = 0.90                  # 锁死


def values(ckpt: str, tiles: np.ndarray, glob: np.ndarray,
           batch: int = 256) -> np.ndarray:
    """批量前向取 value head（cpu / 1 线程 / no_grad）。"""
    import torch
    from train_ppo import build_net

    torch.set_num_threads(1)
    net, _ = build_net(str(ROOT / ckpt), "cpu")
    net.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(tiles), batch):
            t = torch.from_numpy(np.ascontiguousarray(tiles[i:i + batch], dtype=np.float32))
            g = torch.from_numpy(np.ascontiguousarray(glob[i:i + batch], dtype=np.float32))
            out.append(net(t, g)["value"].reshape(-1).numpy())
    return np.concatenate(out).astype(np.float64)


def pearson(a, b) -> float:
    x, y = np.asarray(a, np.float64), np.asarray(b, np.float64)
    if x.std() < 1e-12 or y.std() < 1e-12:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def main() -> int:
    import torch
    torch.set_num_threads(1)

    d = np.load(ROOT / "runs/exit_r12/exit_w0.npz")
    rng = np.random.default_rng(20261107)
    idx = np.sort(rng.choice(len(d["tiles"]), size=N_R12, replace=False))
    tiles, glob = d["tiles"][idx], d["glob"][idx]

    print(f"[probe] 状态 {N_R12}（R12 w0 留出抽样，固定 seed 20261107）")
    v13 = values("models/exit_p13.pt", tiles, glob)
    v20 = values("models/exit_p20.pt", tiles, glob)
    vr12 = values("models/exit_r12.pt", tiles, glob)

    def report(a, b, label):
        diff = (a - b) * 8.0
        q = np.percentile(np.abs(diff), [50, 90, 99])
        return {"label": label, "mean_abs_diff_pts": round(float(np.abs(diff).mean()), 4),
                "std_diff_pts": round(float(diff.std()), 4),
                "p50": round(float(q[0]), 4), "p90": round(float(q[1]), 4),
                "p99": round(float(q[2]), 4), "pearson_r": round(pearson(a, b), 4)}

    g1 = report(v20, v13, "p20 - p13 (PPO 链跨 8 轮漂移)")
    g2 = [report(vr12, v13, "r12 - p13"), report(vr12, v20, "r12 - p20")]
    drift = g1["mean_abs_diff_pts"]
    g1_verdict = ("漂移可观，冻结密集信号源值得实测（按 §6.38 规格排 E1）"
                  if drift >= G1_THRESHOLD else
                  "漂移远小于闸门 SE 与绝对强度，GRP 锚定收益可忽略 ⇒ 不占 CPU 窗口跑 E1")
    g2_verdict = ("换冻结源几乎不带来新信息" if
                  all(x["pearson_r"] > G2_R for x in g2) else
                  "冻结源与 PPO 链 value 确有差异")

    print("\n=== G1 主判据：V_θ 跨 8 轮漂移（分差量纲 ×8）===")
    print(f"  mean|ΔV|×8 = {g1['mean_abs_diff_pts']:.4f} 分  "
          f"std={g1['std_diff_pts']:.4f}  p90={g1['p90']:.4f}  "
          f"p99={g1['p99']:.4f}  r={g1['pearson_r']:.4f}")
    print(f"  阈值 {G1_THRESHOLD:.2f} 分 ⇒ {g1_verdict}")
    print("\n=== G2 冻结源 vs PPO 链 ===")
    for x in g2:
        print(f"  {x['label']}: mean|Δ|×8={x['mean_abs_diff_pts']:.4f}  "
              f"r={x['pearson_r']:.4f}")
    print(f"  ⇒ {g2_verdict}")

    # ---- G3：小样本价值精度 + 前向一致性校验 ----
    g3 = {}
    shards = sorted((ROOT / "runs/ppo_shards").glob("shard_*.npz"))
    if shards:
        s = np.load(shards[0])
        T, G, hand, seat, rets, val_saved = (s["T"], s["G"], s["hand"],
                                             s["seat"], s["rets"], s["val"])
        R = np.array([rets[int(h)][int(seat[i])] if int(h) < len(rets) else 0.0
                      for i, h in enumerate(hand)], dtype=np.float64)
        v_re = values("models/exit_p20.pt", T, G)
        v_fr = values("models/exit_r12.pt", T, G)
        for name, v in (("rollout_val_saved", val_saved.astype(np.float64)),
                        ("V_p20_recomputed", v_re), ("V_r12_frozen", v_fr)):
            g3[name] = {"pearson_r_with_R_over8": round(pearson(v, R / 8.0), 4),
                        "rmse_pts": round(float(np.sqrt(np.mean((v * 8.0 - R) ** 2))), 4),
                        "mean_v": round(float(v.mean()), 4)}
        g3["forward_check_max_abs"] = round(float(np.max(np.abs(v_re - val_saved))), 6)
        g3["n_states"] = int(len(T))
        g3["caveat"] = "16 局冒烟分片，n 小且为 in-sample 分布，仅量级参考，不参与 G1"
        print("\n=== G3 价值精度（16 局冒烟，仅辅证）===")
        for name in ("rollout_val_saved", "V_p20_recomputed", "V_r12_frozen"):
            x = g3[name]
            print(f"  {name}: r={x['pearson_r_with_R_over8']:+.4f} "
                  f"RMSE={x['rmse_pts']:.3f} 分")
        print(f"  前向一致性 max|重算−落盘| = {g3['forward_check_max_abs']:.6f}")

    result = {"protocol": {"locked_at": "2026-11-07 20:2x (before seeing any numbers)",
                           "n_r12": N_R12, "G1_threshold_pts": G1_THRESHOLD,
                           "G2_r": G2_R,
                           "caveat": "R12 状态 in-sample 且为 DAgger 分布，只作相对比较"},
              "G1": {**g1, "threshold": G1_THRESHOLD, "verdict": g1_verdict,
                     "run_e1": bool(drift >= G1_THRESHOLD)},
              "G2": {**{x["label"]: x for x in g2}, "verdict": g2_verdict},
              "G3": g3}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[save] {OUT.relative_to(ROOT)}")
    print(f"[decision] run_e1 = {result['G1']['run_e1']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
