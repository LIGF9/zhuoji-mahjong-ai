#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""PPO 加速轮训练健康度跨轮告警（**只读**：不写标志、不改任何闸门/晋升判据）。

背景：外部专家评审 §7「训练健康度监控缺失」。train_ppo.py 自 2026-11-07 起
在每轮末尾打印一行 ``[metrics] {json}``，已含 kl / clipfrac / ent / gnorm /
vl / ent_rollout / kl_stop / states / secs。本工具把它汇总成跨轮趋势并做
**阈值告警**，让异常在胜率掉之前可见。

阈值全部在**看到跨轮告警结果之前**锁定（2026-11-07 17:3x），依据 R13~P15
的实测区间（kl 0.005~0.009、clipfrac 0.081~0.087、ent 1.23~1.48、
gnorm 1.37~1.42、states 97k~107k），见 docs/14天训练计划.md §6.35。

关于 Q-MSE（评审点名）：Q-MSE 属 **DAgger 训练轮**（train_exit.py 的
CE(softmax(Q/τ)) + 0.5·SmoothL1(V, maxQ/8)），加速轮是纯自对弈 PPO、
没有教师 Q 标签 ⇒ 加速轮**不存在 Q-MSE**，硬造一个才是错误指标。
DAgger 轮由 train_exit 的 val loss 代理，已在 §6.34.2 逐轮记录，本工具不重复。
「Q-MSE 持续降但胜率不涨」的过拟合信号，在加速模式里对应的是
**闸门 A 持平而 vl 持续下降**，由本工具的 vl 趋势 + 闸门 A 联合看。

用法::
    .venv/bin/python tools/health_watch.py            # 全轮次 + 告警汇总
    .venv/bin/python tools/health_watch.py --last 12  # 最近 12 轮
    .venv/bin/python tools/health_watch.py --md       # 附 markdown 表

退出码恒 0：告警只是给人看，绝不自动停训/改配方（那会破坏预登记）。
"""
from __future__ import annotations

import argparse
import json
import re
import statistics as st
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LOGS = ROOT / "runs" / "night_logs"

# ---- 预登记阈值（改这里 = 改预登记，必须同步 docs/14天训练计划.md §6.35）----
KL_MAX = 0.03          # 单轮最大 kl 上界（自检 #30 既有口径）
KL_MIN = 0.0           # kl≤0 = 没更新 / 数值异常
CLIPFRAC_MAX = 0.30    # 过高 ⇒ 步长太大（自检 #30 既有口径）
GNORM_MAX = 5.0        # clip 前原始梯度范数；实测 ~1.4
ENT_DROP = 0.30        # 相对前 5 轮中位下跌 >30% ⇒ 过早收敛
VL_RISE_STREAK = 3     # 连续 3 轮 vl 上升 ⇒ 价值头退化
STATES_MIN = 50000     # 单轮有效状态下限
SECS_RATIO = 2.0       # 超过前 5 轮中位 2 倍 ⇒ 吞吐异常
WARN_WINDOW = 5        # 熵/耗时的参照窗口

NUM = re.compile(r"ppo_p(\d+)\.log$")


def read_all() -> list[dict]:
    rows: dict[int, dict] = {}
    for p in sorted(LOGS.glob("ppo_p*.log")):
        m = NUM.search(p.name)
        if not m:
            continue
        r = int(m.group(1))
        last = None
        for line in p.read_text(encoding="utf-8", errors="replace").splitlines():
            if line.startswith("[metrics] "):
                try:
                    last = json.loads(line[len("[metrics] "):])
                except json.JSONDecodeError:
                    pass
        if last:
            # P10~P12 是旧版 metrics（只有 kl/states/secs/A_std）；P13 起才有完整健康指标。
            # 缺失一律记 NaN ⇒ 展示为 nan、告警时跳过，绝不把「字段缺失」误报成 0 或异常。
            nan = float("nan")
            rows[r] = {
                "round": r,
                "kl": float(last["kl"][-1]),
                "clipfrac": float(last.get("clipfrac", nan)),
                "ent": float(last.get("ent", nan)),
                "ent_rollout": float(last.get("ent_rollout", nan)),
                "gnorm": float(last.get("gnorm", nan)),
                "vl": float(last.get("vl", nan)),
                "states": int(last["states"]),
                "secs": float(last["secs"]),
                "hands": int(last.get("hands", 0)),
                "kl_stop": bool(last.get("kl_stop", False)),
                "full": "clipfrac" in last,
            }
    return [rows[k] for k in sorted(rows)]


def judge(rows: list[dict]) -> list[str]:
    """跨轮告警；返回人类可读的告警行（空 = 全绿）。"""
    out: list[str] = []
    if not rows:
        return ["[WARN] 未找到任何 [metrics] 行"]
    for i, r in enumerate(rows):
        tag = f"P{r['round']:02d}"
        cf, ent, gn, vl = r["clipfrac"], r["ent"], r["gnorm"], r["vl"]
        if not (KL_MIN < r["kl"] <= KL_MAX):
            out.append(f"[WARN] {tag} kl={r['kl']:.4f} 越界 (0, {KL_MAX}]")
        if cf == cf and cf > CLIPFRAC_MAX:
            out.append(f"[WARN] {tag} clipfrac={cf:.3f} > {CLIPFRAC_MAX}"
                       " ⇒ 步长过大，考虑调小 lr/clip")
        if gn == gn and gn > GNORM_MAX:
            out.append(f"[WARN] {tag} gnorm={gn:.2f} > {GNORM_MAX} ⇒ 梯度爆炸风险")
        if r["kl_stop"]:
            out.append(f"[WARN] {tag} kl_stop=True（KL 早停触发）")
        if r["states"] < STATES_MIN:
            out.append(f"[WARN] {tag} states={r['states']} < {STATES_MIN} ⇒ 采样异常")
        # 参照窗口必须**同配置**：P10~P12 是 960 局（~320s），P13 起加速轮 1920 局
        # （~600s）。混在一起比会把「配置切换」误报成吞吐异常（已实测，勿删此过滤）。
        hist = [x for x in rows[max(0, i - WARN_WINDOW):i]
                if x["hands"] == r["hands"]]
        if ent == ent and len(hist) >= 3:
            ents = [x["ent"] for x in hist if x["ent"] == x["ent"]]
            if len(ents) >= 3:
                ent_med = st.median(ents)
                if ent_med > 0 and ent < ent_med * (1 - ENT_DROP):
                    out.append(f"[WARN] {tag} ent={ent:.3f} 较前 {len(ents)} 轮中位 "
                               f"{ent_med:.3f} 跌 >{ENT_DROP:.0%} ⇒ 探索不足/过早收敛")
        if len(hist) >= 3:
            sec_med = st.median(x["secs"] for x in hist)
            if sec_med > 0 and r["secs"] > sec_med * SECS_RATIO:
                out.append(f"[WARN] {tag} [{r['hands']} 局配置] secs={r['secs']:.0f}s > "
                           f"{SECS_RATIO}× 中位 {sec_med:.0f}s ⇒ 吞吐异常"
                           "（先查是否外部探针并发占核）")
        if vl == vl and i >= VL_RISE_STREAK - 1:
            vls = [x["vl"] for x in rows[i - VL_RISE_STREAK + 1:i + 1]]
            if all(v == v for v in vls) and all(b > a for a, b in zip(vls, vls[1:])):
                out.append(f"[WARN] {tag} vl 连续 {VL_RISE_STREAK} 轮上升 {vls} ⇒ 价值头退化")
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="PPO 加速轮训练健康度跨轮告警（只读）")
    ap.add_argument("--last", type=int, default=0, help="只看最近 N 轮（0=全部）")
    ap.add_argument("--md", action="store_true", help="附 markdown 表")
    a = ap.parse_args(argv)

    rows = read_all()
    shown = rows[-a.last:] if a.last else rows
    print("=" * 100)
    print(f"PPO 训练健康度 ｜ 共 {len(rows)} 轮有 [metrics]，本次展示 {len(shown)} 轮"
          f" ｜ 阈值预登记 2026-11-07 §6.35")
    print(f"判据：kl∈(0,{KL_MAX}] · clipfrac<{CLIPFRAC_MAX} · gnorm<{GNORM_MAX} · "
          f"ent 跌幅<{ENT_DROP:.0%} · states≥{STATES_MIN} · secs<{SECS_RATIO}×中位 · "
          f"vl 不连续 {VL_RISE_STREAK} 轮上升")
    print("=" * 100)
    print(f"{'轮':>5} {'kl':>8} {'clip':>7} {'ent':>7} {'entRoll':>8} "
          f"{'gnorm':>7} {'vl':>7} {'states':>8} {'secs':>7} {'klStop':>7}")
    for r in shown:
        print(f"P{r['round']:02d}   {r['kl']:8.4f} {r['clipfrac']:7.3f} "
              f"{r['ent']:7.3f} {r['ent_rollout']:8.3f} {r['gnorm']:7.3f} "
              f"{r['vl']:7.3f} {r['states']:8d} {r['secs']:7.0f} "
              f"{str(r['kl_stop']):>7}")

    warns = judge(rows)
    print("-" * 100)
    if warns:
        print(f"告警 {len(warns)} 条：")
        for w in warns:
            print("  " + w)
    else:
        print("✅ 全部指标在阈值内，无告警")
    print("\n[说明] 本工具只读、不写任何标志、不停训、不改闸门判据；"
          "告警需人工结合闸门 A/D 判断是否行动。")
    print("[说明] Q-MSE 属 DAgger 轮（train_exit val loss 代理），加速轮无教师 Q 标签，"
          "故不在此表；过拟合信号看「闸门 A 持平而 vl 持续降」。")

    if a.md:
        print("\n```markdown")
        print("| 轮 | kl | clipfrac | ent | gnorm | vl | states | secs |")
        print("|---|---|---|---|---|---|---|---|")
        for r in shown:
            print(f"| P{r['round']:02d} | {r['kl']:.4f} | {r['clipfrac']:.3f} | "
                  f"{r['ent']:.3f} | {r['gnorm']:.3f} | {r['vl']:.3f} | "
                  f"{r['states']} | {r['secs']:.0f} |")
        print("```")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
