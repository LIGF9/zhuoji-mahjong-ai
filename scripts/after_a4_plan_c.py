"""等方案 A4（s9 → s12）跑完后，自动执行 P1 阶段 2 的决定性对照 **C3**。

为什么需要它
------------
P1 阶段 2 的第一次消融给出全负结果：`p_abc`（三包全开）−0.670、`p_b`（危险度包）
−0.635，两者都"显著劣"（1200 副共用牌池）。但这里有一个**必须先排除的混淆因素**：

v1 → v2 热启动时，新增的 16 个输入通道权重与 28 列 `fc_in` 权重是**随机初始化**的。
实测（2026-10-10，`reports/_diag_c1.json` 同口径）这一随机扰动会让热启动瞬间的
discard logits 最大偏离 **16.7** —— 足以完全改变 argmax 决策。而同一批实验里
唯一"零初始化"的改动（空间头，`sp_on` +0.167）是唯一没有掉分的臂。

于是"特征本身有害"（H2）与"迁移方式带毒"（H1）还没分开。C3 就是那个单变量对照：
**其余超参与 `p_abc` 完全一致，只把 `--new-param-init` 从 random 换成 zero。**

判读方式
--------
在同一次复评里**同时**评 `p_abc` 与 `p_abc_zero`（外加 `p_none` / `sp_off` / s11），
于是两臂是同一批牌、同一对手的配对观测，差值的 CI 直接可比，不需要跨文件拼数据。

前置声明（看到结果之前写死）
--------------------------
* 若 `p_abc_zero` 相对 `p_abc` 显著为正（CI 下界 > 0）→ **H1 成立**：掉分主要是迁移扰动，
  特征本身并未被证伪，应重跑"零初始化"版本的三臂；
* 若两者无显著差异且都显著负于 s11 → **H2 成立**：特征本身有害，P1b 需要返工；
* 若 `p_abc_zero` 仍显著负但幅度明显缩小 → 两个因素都在起作用，按幅度定权重。

用法::

    python scripts/after_a4_plan_c.py          # 等 A4 结束再跑
    python scripts/after_a4_plan_c.py --now    # 立即跑（调试用）
"""
from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = r"C:\Users\LIGF\.workbuddy\binaries\python\envs\zhuoji\Scripts\python.exe"
A4_SUMMARY = ROOT / "reports" / "shift_plan_a_s12_summary.json"
LOG = ROOT / "reports" / "after_a4_plan_c.log"
SUMMARY = ROOT / "reports" / "plan_c_summary.json"

REF = "net:dushan_big_s11|teacher:balanced|teacher:tenpai_rush"
TAG = "b2c_p_abc_zero"
SEED = 20270
POOL = "dushan_big_s3,dushan_big_s7,dushan_big_s11"

#: 复评同批比较的检查点：C3 本体 + 第一次消融的几个关键臂 + 纯扰动对照 + 锚点。
#: `_diag_c1_v2warm` 是"只热启动、零训练"的模型 —— 它给出**纯扰动造成的分数损失**，
#: 与 `p_abc`（扰动 + 400 轮训练后）一比，就知道训练到底修回来多少。
#:
#: 注意文件名口径：`score_checkpoints.py` 按 `models/{stem}.pt` 直接拼路径，
#: 拿的是**磁盘上的真实文件名**；而 `TAG` 是"训练产物逻辑名"（脚本自己补 `dushan_` 前缀）。
#: 2026-10-11 踩坑：这里原先直接写 `TAG`，导致找 `models/b2c_p_abc_zero.pt` 失败，
#: 复评进程秒退、错误又被 stderr 重定向吞掉，最后只在汇总里留下一个空 `rescore: {}`。
TAG_FILE = f"dushan_{TAG}"                       # = models/dushan_b2c_p_abc_zero.pt
RESCORE_CKPTS = [TAG_FILE, "_diag_c1_v2warm", "dushan_b2_p_abc", "dushan_b2_p_none",
                 "dushan_b2_sp_off", "dushan_big_s11"]


def say(msg: str) -> None:
    with open(LOG, "a", encoding="utf-8", buffering=1) as f:
        f.write(f"[{time.strftime('%m-%d %H:%M:%S')}] {msg}\n")


def a4_done() -> bool:
    """A4 完成的判据：summary 已不是我们放的那个占位符。

    （不能只看文件存在 —— 为了放行夜跑队列，那份占位文件早在 23:46 就写下了。）
    """
    if not A4_SUMMARY.exists():
        return False
    try:
        d = json.loads(A4_SUMMARY.read_text(encoding="utf-8"))
    except Exception:
        return False
    return not d.get("_placeholder")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--now", action="store_true", help="不等 A4，立即开始")
    ap.add_argument("--max-hours", type=float, default=8.0,
                    help="最长等待 A4 的小时数（默认 8h）")
    ap.add_argument("--iters", type=int, default=400)
    # 新机（i7-14700K：20 物理核 / 28 逻辑核 / 32 GB）实测选点：
    # 更新阶段 threads=24 最快（171ms/batch，比 20 快 11%，28 反而回落）；
    # 采集阶段 workers>10 后收益饱和（96 局 collect 1.2s → 0.9s）。
    ap.add_argument("--threads", type=int, default=24,
                    help="torch 更新阶段线程数（14700K 实测 24 最优）")
    ap.add_argument("--workers", type=int, default=12,
                    help="向量化采集进程数（14700K 实测 >10 后饱和）")
    args = ap.parse_args()

    say("=" * 62)
    say("Plan C 接续监控启动（等 A4 结束后跑 C3：零初始化 + 400 轮）")

    if not args.now:
        time.sleep(60)
        t0 = time.time()
        while not a4_done():
            waited = (time.time() - t0) / 3600
            if waited > args.max_hours:
                say(f"等待 A4 超过 {args.max_hours:.0f}h，放弃（请人工检查）")
                return 1
            if int(time.time() - t0) % 600 < 30:
                say(f"A4 仍在运行… 已等 {(time.time() - t0) / 60:.0f} 分钟")
            time.sleep(60)
        say(f"A4 已完成（等待 {(time.time() - t0) / 60:.0f} 分钟）")
    else:
        say("--now：跳过等待")

    # ---- 1. C3 训练：与 p_abc 只差 --new-param-init ----
    out = ROOT / "models" / f"dushan_{TAG}.pt"
    glog = ROOT / "reports" / f"dushan_{TAG}.log"
    cmd = [PY, "scripts/selfplay.py",
           "--init", "models/dushan_big_s11.pt", "--out", f"models/dushan_{TAG}.pt",
           "--history", f"reports/rl_history_{TAG}.json",
           "--iters", str(args.iters), "--games-per-iter", "96", "--batch", "512",
           "--lr", "8.0e-6", "--temperature", "0.95", "--teacher-prob", "0.08",
           "--threads", str(args.threads), "--vectorized", "--workers", str(args.workers),
           "--eval-every", "10", "--eval-rounds", "40",
           "--select", "guard", "--select-metric", "vs_anchor",
           "--eval-anchor", "dushan_big_s11",
           "--save-every", "50", "--snapshot-pool", POOL,
           "--encoder", "v2", "--encoder-packs", "abc", "--spatial-head", "off",
           "--new-param-init", "zero",          # ← 全场唯一的变量
           "--seed", str(SEED), "--rules", "dushan"]
    say("C3 训练启动：" + " ".join(cmd[2:]))
    t0 = time.time()
    with open(glog, "w", encoding="utf-8") as f:
        rc = subprocess.run(cmd, cwd=str(ROOT), stdout=f,
                            stderr=subprocess.STDOUT).returncode
    dt = time.time() - t0
    text = glog.read_text(encoding="utf-8", errors="ignore")
    train = {
        "rc": rc, "hours": round(dt / 3600, 2),
        "finished": "完成。" in text,
        "warm_line": [ln for ln in text.splitlines() if ln.startswith("[warm]")],
        "head": [ln for ln in text.splitlines()
                 if ln.startswith(("[编码]", "[pool]", "[eval] 锚点"))],
        "select_lines": [ln for ln in text.splitlines()
                         if ln.startswith("[select]")],
    }
    for ln in train["warm_line"] + train["head"] + train["select_lines"][:2]:
        say("  " + ln)
    say(f"C3 训练结束 rc={rc} 完成={train['finished']} 用时 {dt / 3600:.2f}h")
    if rc != 0 or not out.exists():
        say("C3 未正常产出模型，跳过复评")
        SUMMARY.write_text(json.dumps({"train": train, "rescore": None},
                                      ensure_ascii=False, indent=2),
                           encoding="utf-8")
        return 1

    # ---- 2. 同批复评（含对照臂，配对可比） ----
    out_json = ROOT / "reports" / "plan_c_rescore.json"
    err_log = ROOT / "reports" / "_plan_c_rescore_stderr.log"
    say(f"复评：{len(RESCORE_CKPTS)} 个检查点 × 1200 副（同批牌池 + 同对手）")
    # stderr 落盘而不是丢弃：复评失败时能直接看到原因（原先 DEVNULL 让失败静默）
    with open(err_log, "wb") as errf:
        subprocess.run(
            [PY, "scripts/score_checkpoints.py",
             "--ckpts", ",".join(RESCORE_CKPTS), "--ref", REF,
             "--deals", "1200", "--threads", str(args.threads),
             "--anchor", "dushan_big_s11", "--out", "reports/plan_c_rescore.json",
             "--no-per-deal"],
            cwd=str(ROOT), stdout=subprocess.DEVNULL, stderr=errf)
    rows = {}
    if out_json.exists():
        sc = json.loads(out_json.read_text(encoding="utf-8"))
        rows = {r["ckpt"]: r for r in sc.get("rows", [])}
        say("—— 结果（1200 副配对净分差 vs s11）——")
        for k in RESCORE_CKPTS:
            r = rows.get(k) or {}
            say(f"  {k:<24} {r.get('vs_anchor', float('nan')):+.3f} "
                f"± {r.get('vs_anchor_ci95', float('nan')):.3f}  {r.get('verdict', '')}")

        # H1 / H2 判读（前置声明见本文件 docstring）
        a = rows.get(TAG_FILE) or {}
        b = rows.get("dushan_b2_p_abc") or {}
        if a and b:
            diff = (a.get("vs_anchor") or 0.0) - (b.get("vs_anchor") or 0.0)
            se = (((a.get("vs_anchor_ci95") or 0) / 1.96) ** 2 +
                  ((b.get("vs_anchor_ci95") or 0) / 1.96) ** 2) ** 0.5
            lo = diff - 1.96 * se
            if lo > 0:
                verdict = "H1 成立：零初始化相对随机初始化显著更好 → 掉分主要来自迁移扰动"
            elif lo < 0 < diff + 1.96 * se:
                verdict = "差值为正但不显著：两个因素都可能起作用，按幅度定权重"
            else:
                verdict = "H2 更可能：零初始化没有救回来 → 特征本身有害，P1b 需返工"
            say(f"  [判读] C3 − p_abc = {diff:+.3f}（保守 95% CI [{lo:+.3f}, "
                f"{diff + 1.96 * se:+.3f}]）→ {verdict}")
    else:
        say("复评未产出 json —— 子进程 stderr 末尾：")
        try:
            tail = err_log.read_text(encoding="utf-8", errors="replace").strip().splitlines()[-6:]
            for ln in tail:
                say("    " + ln)
        except Exception as e:
            say(f"    （读取 {err_log.name} 失败：{e}）")

    SUMMARY.write_text(json.dumps(
        {"train": train, "rescore": rows, "anchor": "dushan_big_s11",
         "ref": REF, "ckpts": RESCORE_CKPTS, "seed": SEED,
         "note": "C3 与 p_abc 只差 --new-param-init（zero vs random）"},
        ensure_ascii=False, indent=2), encoding="utf-8")
    say(f"全部结束，汇总 -> {SUMMARY}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
