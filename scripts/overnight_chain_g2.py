# -*- coding: utf-8 -*-
"""夜间自主决策链（2026-10-21 夜，用户 22:45 明示授权）。

职责单一：守 m2g2 判读 verdict → 按预登记决策树决策 → PASS 则自动起
m2g3（β0.05 受控超越）并链式判读归档，全程留痕。

预登记判据（写死，不临场改；用户 2026-10-22 00:15 追加授权"失败分支也别浪费夜间"）：
  m2g2 末点 2400 副 level ≥ 5.40（c4_it2280 = 5.434 的噪声带内）
      ⇒ PASS：崩因=熵而非缺 KL 锚 → 起 m2g3（β0.05 常数、教师=座位=master、熵控开）
  末点 < 5.40
      ⇒ FAIL：仍起 **m2g3b**（回退实验）——从 c4_it2280 出发，唯一变量 = β 0.2→0.05
        （其余逐项复刻 c4：熵控关、教师/座位共用 master、β 恒定），判 KL 拉力减半后
        能否受控超越；连 0.05 都塌 ⇒ KL 锚不可撤终审，"突破教师"线关闭。
  两分支跑完都 finalize 2400 副判读归档，之后收手等晨会（不做第三次自动实验）。

安全栏：
  - 只在无 selfplay / 无 finalize 进程时才动手；
  - m2g3 用 --no-guards（keep_awake / res_monitor 永久禁止）；
  - 超时（08:00 还没有 verdict）或判读文件损坏 → 记日志退出，不赌。
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(r"C:\Users\LIGF\WorkBuddy\2026-09-30-12-49-43\zhuoji-mahjong")
PY = Path(r"C:\Users\LIGF\.workbuddy\binaries\python\envs\zhuoji\Scripts\python.exe")
JUDGE_JSON = ROOT / "reports" / "m1_m2g2_judge.json"
LOG = ROOT / "reports" / "_overnight_g2_chain.out"
DECISION_MD = ROOT / "reports" / "_overnight_decision.md"
DEADLINE = time.mktime(time.strptime("2026-10-22 08:00:00", "%Y-%m-%d %H:%M:%S"))
PASS_FLOOR = 5.40          # 预登记：m2g2 末点 ≥ 5.40 才放行 m2g3
C4_LEVEL = 5.4343          # 已知参照


def log(msg: str) -> None:
    line = f"[{time.strftime('%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    with LOG.open("a", encoding="utf-8") as fh:
        fh.write(line + "\n")


def decision_md(msg: str) -> None:
    with DECISION_MD.open("a", encoding="utf-8") as fh:
        fh.write(msg.rstrip() + "\n")


def procs_matching(*needles: str) -> list[int]:
    """返回 cmdline 含任一关键词的存活进程 pid（排除自己）。"""
    import os
    import psutil
    me = os.getpid()
    hits = []
    for p in psutil.process_iter(["pid", "cmdline"]):
        if p.info["pid"] == me:
            continue
        cl = " ".join(p.info.get("cmdline") or [])
        if any(n in cl for n in needles):
            hits.append(p.info["pid"])
    return hits


def parse_verdict() -> tuple[float, float, float, str] | None:
    """返回 (末点 level, c4 level, master level, 末点 ckpt 名)；损坏返回 None。"""
    try:
        v = json.loads(JUDGE_JSON.read_text(encoding="utf-8"))
        rows = v["duel"]["rows"]
    except Exception as e:                                    # noqa: BLE001
        log(f"verdict 解析失败：{e}")
        return None
    g2_rows = [(r["ckpt"], r["level"], r.get("level_ci95")) for r in rows
               if "dushan_m2_g2_it" in r["ckpt"]]
    if not g2_rows:
        log(f"verdict 里没有 m2g2 存档行：rows={[r['ckpt'] for r in rows]}")
        return None
    end_name, end_lv, end_ci = max(g2_rows, key=lambda x: int(x[0].rsplit("_it", 1)[1]))
    c4_lv = next((r["level"] for r in rows if "c4_it2280" in r["ckpt"]), C4_LEVEL)
    master_lv = next((r["level"] for r in rows if r["ckpt"] == "dushan_master"), float("nan"))
    return end_lv, c4_lv, master_lv, end_name


def wait_for_verdict() -> tuple[float, float, float, str] | None:
    """等链式 finalize 产出的 judge json（且 finalize 进程已退）。"""
    log("开始守 m2g2 verdict（每 120s 一查，08:00 截止）")
    while time.time() < DEADLINE:
        if JUDGE_JSON.exists():
            # finalize 写完 json 后还要归档；等它进程退出再读，避免半截文件
            for _ in range(60):
                if not procs_matching("m1_segment_finalize.py"):
                    break
                time.sleep(60)
            time.sleep(30)                                    # 文件落盘余量
            hit = parse_verdict()
            if hit:
                return hit
        elif not procs_matching("selfplay.py"):
            # 训练没了但 verdict 也没来 → finalize 可能失败，先记一笔再继续等
            log("⚠️ selfplay 已退出但 verdict 未出现（finalize 可能失败/仍在跑），继续守")
        time.sleep(120)
    log("❌ 08:00 截止仍未拿到 verdict，放弃夜间决策")
    decision_md(f"\n## 超时（{time.strftime('%m-%d %H:%M')}）\n\n"
                "08:00 前 m1_m2g2_judge.json 未产出，夜间链未做任何决策。\n")
    return None


def main() -> int:
    log("=== 夜间链启动：守 m2g2 verdict → 预登记决策树 ===")
    hit = wait_for_verdict()
    if hit is None:
        return 2
    end_lv, c4_lv, master_lv, end_name = hit
    delta = end_lv - c4_lv
    log(f"m2g2 末点 {end_name}: level={end_lv:.3f} (c4={c4_lv:.3f}, Δ={delta:+.3f}, "
        f"master={master_lv:.3f})")

    seg = "m2g3" if end_lv >= PASS_FLOOR else "m2g3b"
    if seg == "m2g3b":
        log(f"⚠️ FAIL：末点 {end_lv:.3f} < 门槛 {PASS_FLOOR} → 回退实验 m2g3b"
            f"（c4 起点、唯一变量 β 0.2→0.05、熵控关复刻 c4）")
        decision_md(
            f"\n## 决策：FAIL → LAUNCH m2g3b（{time.strftime('%m-%d %H:%M')}）\n\n"
            f"- m2g2 末点 **{end_lv:.3f}** < 预登记门槛 {PASS_FLOOR}（Δ vs c4 = {delta:+.3f}）"
            f" ⇒ β=0+熵控没站住 → 按预登记回退起 **m2g3b**：init=c4_it2280、"
            f"唯一变量 β 0.2→0.05（熵控关、教师/座位共用 master，逐项复刻 c4）、"
            f"600 轮（编号 2281→2880）。\n"
            f"- m2g3b 判据：Δ≥+0.20 显著 ⇒ 小拉力受控超越；|Δ|<0.20 ⇒ 天花板再确认；"
            f"塌陷 <5.0 ⇒ KL 锚不可撤终审，「突破教师」线关闭。\n"
            f"- 参照：c4_it2280={c4_lv:.3f}，master={master_lv:.3f}\n")
    else:
        log(f"✅ PASS：末点 {end_lv:.3f} ≥ {PASS_FLOOR} → 启动 m2g3（β0.05 受控超越）")
        decision_md(
            f"\n## 决策：PASS → LAUNCH m2g3（{time.strftime('%m-%d %H:%M')}）\n\n"
            f"- m2g2 末点 **{end_lv:.3f}** ≥ 预登记门槛 {PASS_FLOOR}（Δ vs c4 = {delta:+.3f}）"
            f" ⇒ 崩因=熵、β=0+熵控站住了 → 按预登记自动起 **m2g3**。\n"
            f"- m2g3 配方：init=m2g2 末点、β **0.05 常数**、教师=座位=master、熵控开、"
            f"600 轮（编号 2881→3480）；判据：末点 − c4_it2280 ≥ +0.20 且 CI 下界>0。\n"
            f"- 参照：c4_it2280={c4_lv:.3f}，master={master_lv:.3f}\n")

    if procs_matching("selfplay.py", "m1_segment_finalize.py"):
        log("❌ 异常：动手前发现残留训练/判读进程，放弃（内存铁律）")
        return 2

    rc = subprocess.call([str(PY), "-u", str(ROOT / "scripts" / "m1_launch_segment.py"),
                          "--segment", seg, "--no-guards"], cwd=str(ROOT))
    log(f"{seg} 训练退出 rc={rc}")
    if rc != 0:
        decision_md(f"\n- ⚠️ {seg} 训练异常退出 rc={rc}，未跑 finalize 判读。\n")
        return rc

    rc2 = subprocess.call([str(PY), "-u", str(ROOT / "scripts" / "m1_segment_finalize.py"),
                           "--run", f"m1_{seg}", "--ckpt-prefix", f"dushan_m2_{seg[1:]}",
                           "--extra", "dushan_master,_ckpts/dushan_m2_c4_it2280",
                           "--deals", "2400"], cwd=str(ROOT))
    log(f"{seg} finalize 退出 rc={rc2}")
    decision_md(f"\n- {seg} finalize rc={rc2}；晨会看 `reports/m1_{seg}_judge.json`。\n")
    return rc2


if __name__ == "__main__":
    sys.exit(main())
