#!/usr/bin/env python
"""M2 夜间无人值守自主接力（2026-10-16 用户授权自主决策）。

授权与目标
----------
用户夜间不能及时做选择，要求：**按推荐选项直接执行训练、不要闲置算力**；
目标 = 让 10M 模型（9,849,902 参数）追平并超越 S11（= ``dushan_master``，769,422 参数）。
所以本脚本的全部设计原则只有两条：

1. **任何时刻必须有训练在跑**（止损不是终点，是「分段验收 + 无痕续训」的一环）；
2. **接力必须无缝** —— 从上一代**含优化器状态**的检查点续训，不重付冷启动位移。

为什么"从断点续训"是夜间的最优动作
--------------------------------
B/B2/B3 三次续训全部失败，根因是「冷 Adam」：老检查点不含 ``opt``，
``--init`` 只恢复 ``state_dict`` ⇒ AdamW 的 m/v 归零 ⇒ 首步 ``Δθ ≈ lr·sign(g)``
是**与梯度大小无关**的全局随机踢一脚（见 ``reports/M2B4_冷Adam根因与机制修复.md``）。

从 B4 起 ``--save-opt`` 让每个检查点都带 ``opt``；于是：

* 上一代 **早停** ⇒ 从它的 ``_ckpts/<stem>_itNNNN.pt`` 续训（含 opt，**零位移**）；
* 上一代 **跑满** ⇒ 从它的主输出续训（同样含 opt）。

两种情况都**不必**退回 A 段末冷启动重跑 —— 那会白扔已训轮次，
还要再付一次 warmup 位移的代价。这是「不闲置算力」在算法层的正解。

预注册决策树
-----------
    B4 (s4b4, 600 轮)
      ├─ it0100 止损触发 (va ≤ −1.45) → s4b5：从 b4_it0100 续 500 轮
      ├─ it0300 止损触发 (va ≤ −1.05) → s4b6：从 b4_it0300 续 300 轮
      └─ 跑满 600（G3 出判决）        → s4c1：从 b4 主输出续 1200 轮（长跑）
    接力段同样被判读 + 止损；若接力段也结束 → 自动再接力（动态档位，lr 逐代下压）。
    崩溃（无判决且轮次未跑满）→ 从本段最新 ckpt 续跑剩余轮次（不浪费）。

    G3 达标（追平 va ≥ +0.5 且 CI 下界 > 0）→ **备份并晋升 master**
    （严格预注册判据 + 先备份 ``dushan_master.pt`` → ``dushan_master_pre_m2.pt``）。

止损线一律**动态**计算：``起点实测 va − 1.5σ``（σ = 2400 副下的单点 CI ≈ 0.28）。
—— 用 A 段基线当固定锚会在起点本就偏低时造成纯噪声误杀（B4 it0300 的 −1.05 就是这类）。

起动方式（★ 关键）
----------------
**必须挂平台后台任务**（``run_in_background=true``）：本脚本作为前台进程持有整棵
训练树；命令一返回、树就被沙箱清理。``--dry-run`` 只打印计划不动作。

用法::

    python scripts/m2_night_autopilot.py --dry-run     # 只打印决策表
    python scripts/m2_night_autopilot.py               # 托管（挂后台任务）
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable
sys.path.insert(0, str(ROOT / "scripts"))

from m1_launch_segment import SEGMENTS, build_cmd                      # noqa: E402

REPORTS = ROOT / "reports"
LOG = REPORTS / "_night_autopilot.out"
STATE = REPORTS / "night_autopilot_state.json"

SIGMA = 0.28            # 2400 副下 score_checkpoints 单点 CI 半宽
STOP_LINES = 1.5        # 止损线 = 起点 va − STOP_LINES×σ

try:
    import psutil
except ImportError:     # pragma: no cover
    sys.exit("需要 psutil：用 zhuoji venv 的 python 运行本脚本")


# --------------------------------------------------------------------------- #
# 日志 / 状态
# --------------------------------------------------------------------------- #
def ts() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def say(msg: str) -> None:
    line = f"[{ts()}] {msg}"
    print(line, flush=True)
    REPORTS.mkdir(exist_ok=True)
    with LOG.open("a", encoding="utf-8") as fh:
        fh.write(line + "\n")


def load_state() -> dict:
    if STATE.exists():
        try:
            return json.loads(STATE.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            pass
    return {"current": "s4b4", "history": [], "relay_count": 0}


def save_state(st: dict) -> None:
    st["updated_at"] = ts()
    STATE.write_text(json.dumps(st, ensure_ascii=False, indent=2), encoding="utf-8")


# --------------------------------------------------------------------------- #
# 段元信息 / 进程 / 产物
# --------------------------------------------------------------------------- #
def seg_meta(seg: str, cfg: dict) -> dict:
    return {
        "seg": seg,
        "stem": Path(cfg["out"]).stem,
        "out": cfg["out"],
        "iters": int(cfg["iters"]),
        "init": cfg["init"],
        "log": REPORTS / f"_m1_{seg}.out",
        "pid_file": REPORTS / f"_m1_{seg}.pid",
    }


def read_pid(p: Path) -> int | None:
    try:
        return int(p.read_text(encoding="utf-8").strip())
    except Exception:                                      # noqa: BLE001
        return None


def pid_alive(pid: int | None) -> bool:
    if not pid:
        return False
    try:
        proc = psutil.Process(pid)
        return proc.is_running() and proc.status() != psutil.STATUS_ZOMBIE
    except psutil.Error:
        return False


def any_selfplay_alive() -> list[int]:
    """全机扫描：还在跑的 selfplay 训练进程（排除本脚本的亲属）。"""
    me = Path(__file__).name
    out = []
    for pr in psutil.process_iter(["pid", "cmdline"]):
        try:
            cl = " ".join(pr.info.get("cmdline") or [])
            if me in cl:
                continue
            if "selfplay.py" in cl and "--iters" in cl:
                out.append(pr.pid)
        except psutil.Error:
            pass
    return out


def kill_tree(pid: int) -> int:
    try:
        root = psutil.Process(pid)
    except psutil.NoSuchProcess:
        return 0
    procs = []
    try:
        procs = root.children(recursive=True)
    except psutil.Error:
        pass
    procs.append(root)
    for p in procs:
        try:
            p.terminate()
        except psutil.Error:
            pass
    gone, alive = psutil.wait_procs(procs, timeout=20)
    for p in alive:
        try:
            p.kill()
        except psutil.Error:
            pass
    _, alive2 = psutil.wait_procs(alive, timeout=10)
    say(f"  终止旧训练 {pid}：{len(gone)} 个已退出，存活 {len(alive2)}")
    return len(gone)


def ensure_stopped(pid: int | None, tag: str, timeout_s: int = 600) -> None:
    """起接力段之前，确保旧训练树完全退出（避免双训练互抢 CPU/内存）。"""
    if not pid_alive(pid):
        return
    say(f"  等待旧训练 {pid} 退出（最多 {timeout_s // 60} 分）…")
    t0 = time.time()
    while (time.time() - t0) < timeout_s and pid_alive(pid):
        time.sleep(10)
    if pid_alive(pid):
        say(f"  ⚠️ 旧训练仍在跑 → 主动终止（{tag} 需要独占算力）")
        kill_tree(pid)
    left = any_selfplay_alive()
    if left:
        say(f"  ⚠️ 仍有 selfplay 进程：{left} → 一并终止")
        for p in left:
            kill_tree(p)


def latest_ckpt(stem: str) -> tuple[Path, int] | None:
    best = None
    for p in (ROOT / "models" / "_ckpts").glob(f"{stem}_it*.pt"):
        m = re.search(r"_it(\d+)\.pt$", p.name)
        if not m:
            continue
        it = int(m.group(1))
        if best is None or it > best[1]:
            best = (p, it)
    return best


def has_opt(path: Path) -> bool:
    try:
        import torch
        ck = torch.load(str(path), map_location="cpu", weights_only=False)
        return isinstance(ck, dict) and "opt" in ck
    except Exception:                                      # noqa: BLE001
        return False


def log_reached(log: Path) -> int:
    if not log.exists():
        return 0
    try:
        txt = log.read_text(encoding="utf-8", errors="replace")
    except Exception:                                      # noqa: BLE001
        return 0
    best = 0
    for m in re.finditer(r"^iter\s+(\d+)\s", txt, flags=re.M):
        best = max(best, int(m.group(1)))
    return best


# --------------------------------------------------------------------------- #
# 判决读取
# --------------------------------------------------------------------------- #
def read_verdict(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        v = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    return v if isinstance(v, dict) and v.get("vs_anchor") is not None else None


def read_stoploss_decision(stem: str, gate: str) -> dict | None:
    slug = "".join(ch if ch.isalnum() else "_" for ch in gate).strip("_")
    p = REPORTS / f"{stem}_{slug}_stoploss_decision.json"
    if not p.exists():
        return None
    try:
        d = json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    return d if isinstance(d, dict) and d.get("triggered") else None


# --------------------------------------------------------------------------- #
# 起一段训练
# --------------------------------------------------------------------------- #
def clear_old_watchdogs(keep_gate: bool = False) -> None:
    """清掉上一代的 gate_watch / m2_stoploss_watch（不动训练进程、不动本脚本）。

    注意 Windows 上 venv 的 ``Scripts\\python.exe`` 是启动器桩：一个逻辑守望在
    psutil 里会出现两条 —— 所以必须 terminate 后再 wait/kill，只 terminate 会残留。

    ``keep_gate=True`` 时**保留 gate_watch（判读类）**，只清 stoploss。
    原因（2026-10-16 AUTO4@800 事故实证）：判读是异步长任务——它要等本段兜底快照
    落盘、再跑 2400 副（约 35 min）。而总控用"日志到达 N 轮"当作段结束信号，
    快照刚落盘的那一刻就清理守望并接力下一段，会把**尚未启动/尚未跑完的判读腰斩**，
    导致该段最终水平的正式判读永久缺失。判读应与新段训练**并行**，不能被清理。
    """
    me = Path(__file__).name
    victims = []
    for pr in psutil.process_iter(["pid", "cmdline"]):
        try:
            cl = " ".join(pr.info.get("cmdline") or [])
            if me in cl:
                continue
            if "gate_watch.py" in cl and not keep_gate:
                victims.append(pr)
            elif "m2_stoploss_watch.py" in cl:
                victims.append(pr)
        except psutil.Error:
            pass
    if not victims:
        return
    pids = [p.pid for p in victims]
    for p in victims:
        try:
            p.terminate()
        except psutil.Error:
            pass
    gone, alive = psutil.wait_procs(victims, timeout=15)
    for p in alive:
        try:
            p.kill()
        except psutil.Error:
            pass
    psutil.wait_procs(alive, timeout=8)
    say(f"  清理旧守望 {len(gone) + len(alive)} 个（pids {pids[:8]}…）")


def spawn_segment(seg: str, cfg: dict, workers: int) -> int:
    run = f"m1_{seg}"
    log = REPORTS / f"_{run}.out"
    mon = REPORTS / f"_res_monitor_{run}.csv"
    cmd = build_cmd(seg, cfg, workers)

    with log.open("a", encoding="utf-8") as fh:
        fh.write(f"\n===== {run} 起动 @ {ts()}（夜间自主接力）=====\n")
        fh.write(" ".join(cmd) + "\n")

    # DETACHED + BREAKAWAY：训练必须脱离本脚本的进程树/Job，
    # 否则总控所在 shell 被平台终止时会把刚 spawn 的训练连坐杀掉（auto4 事故实证）。
    _flags = (subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.CREATE_BREAKAWAY_FROM_JOB
              | subprocess.DETACHED_PROCESS)
    proc = subprocess.Popen(cmd, stdin=subprocess.DEVNULL,
                            stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT,
                            cwd=str(ROOT), creationflags=_flags)

    for helper in ("keep_awake.py", "res_monitor.py"):
        hcmd = [PY, str(ROOT / "scripts" / helper)]
        if helper == "res_monitor.py":
            hcmd += ["--interval", "60", "--out", str(mon)]
        subprocess.Popen(hcmd, stdin=subprocess.DEVNULL,
                         stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT,
                         cwd=str(ROOT))

    (REPORTS / f"_{run}.pid").write_text(str(proc.pid), encoding="utf-8")
    time.sleep(15)
    rc = proc.poll()
    if rc is None:
        say(f"  ✅ {run} 存活确认（pid {proc.pid}）")
        return 0
    tail = ""
    try:
        tail = "\n".join(log.read_text(encoding="utf-8", errors="replace").splitlines()[-15:])
    except Exception:                                      # noqa: BLE001
        pass
    say(f"  ❌ {run} 起动 15s 内退出（exit={rc}）；日志尾部：\n{tail}")
    return rc or 1


def attach_watchdogs(meta: dict, start_va: float, checkpoints: list[int], tag: str) -> None:
    """为新段挂「判读 + 止损」守望。止损线动态 = start_va − 1.5σ；判读 2400 副。"""
    stem = meta["stem"]
    thr = round(start_va - STOP_LINES * SIGMA, 3)
    svc = ROOT / "scripts"
    for at in checkpoints:
        out = REPORTS / f"{tag}_it{at:04d}.json"
        gw = [PY, str(svc / "gate_watch.py"),
              "--stem", stem, "--at", str(at), "--gate", f"{tag}@{at}",
              "--threshold", str(thr),
              "--baseline-name", "本段起点（实测）",
              "--baseline", f"{start_va:.3f}", "--baseline-ci", "0.414",
              "--deals", "2400", "--seed", "31337",
              "--workers", "6", "--threads", "2", "--worker-threads", "1",
              "--log", str(meta["log"].relative_to(ROOT)),
              "--stall-min", "30", "--max-wait-h", "14",
              "--out", str(out.relative_to(ROOT))]
        subprocess.Popen(gw, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                         stderr=subprocess.STDOUT, cwd=str(ROOT))
        say(f"  挂判读 {tag}@{at}（阈值 {thr:+.3f}）")

        sl_log = REPORTS / f"_stoploss_{tag}_{at}.log"
        sl = [PY, str(svc / "m2_stoploss_watch.py"),
              "--verdict", str(out.with_suffix(".verdict.json").relative_to(ROOT)),
              "--pid-file", str(meta["pid_file"].relative_to(ROOT)),
              "--threshold", str(thr), "--gate", f"{tag}@{at}", "--stem", stem,
              "--reports", "reports", "--log", str(sl_log.relative_to(ROOT)),
              "--poll-s", "60", "--max-wait-h", "14"]
        subprocess.Popen(sl, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                         stderr=subprocess.STDOUT, cwd=str(ROOT))
        say(f"  挂止损 {tag}@{at}（≤ {thr:+.3f} 则停跑保留存档）")


# --------------------------------------------------------------------------- #
# 动态档位（第 3+ 代接力 / 崩溃续跑）
# --------------------------------------------------------------------------- #
def dynamic_cfg(name: str, init_ckpt: Path, iters: int, lr: float) -> dict:
    return dict(iters=iters, init=str(init_ckpt.relative_to(ROOT)),
                out=f"models/dushan_m2_{name}.pt",
                warmup=0, teacher_prob=0.0, temperature=1.2, lr=lr,
                needs=str(init_ckpt.relative_to(ROOT)), m2=True,
                extra=["--klt-beta", "0.06", "--klt-anneal-iters", "200",
                       "--snapshot-pool", "",
                       "--ent-baseline", "1.026",
                       "--master-seat-prob", "0.40",
                       "--lr-min", f"{lr * 0.45:.2g}",
                       "--lr-warmup", "10",
                       "--save-opt",
                       "--ent-coef", "0.040",
                       "--kl-coef", "0.016",
                       "--explore-eps", "0.010", "--explore-eps-end", "0.010",
                       "--explore-anneal-iters", "1",
                       "--save-every", "20"])


# --------------------------------------------------------------------------- #
# 探索计划（AUTO4 平台期之后的短段试错）
# --------------------------------------------------------------------------- #
# 事实依据：AUTO4@800 = −0.7926 ± 0.2754（2400 副），相对 @400 的 −0.8072 只动 +0.015
# ⇒ 原配方（lr 1.5e-6 长退火到 lr-min 6.7e-7、KLT β 0.06→0、master-seat 0.40）已饱和。
# 策略：auto6 起改成 300 轮短段，**逐段只动一个旋钮**，配 20 的倍数判读点。
#   命中判据：本段 va − 上一段 va ≥ HIT_DELTA 且 CI 下界 > 上一段 va
#     → 命中：沿用同旋钮并把段长加到 800 轮（稳扎稳打，训练量优先）
#     → 未命中：换下一个旋钮；连续 3 次未命中 → 收敛长段（历史最优旋钮，1200 轮慢退火）
# 判读点必须落在段长**之前**（240/300），因为 gate_watch 判读要跑约 18 min，
# 排在段末才能让总控在检测到"跑满"时已经读到结果；同时必须是 save-every(20) 的倍数。
EXPLORE_START_VA = -0.85        # auto6 起点代理（auto4@800 实测 −0.793，留安全余量）
HIT_DELTA = 0.20                # 命中门槛（≈0.7×CI 半宽）
EXPLORE_LEN = 300               # 短段基准长度
EXPLORE_LEN_HIT = 800           # 命中后的加长（稳扎稳打：训练量优先）
EXPLORE_LEN_CONS = 1200         # 三旋钮全未命中 → 收敛长段（历史最优旋钮 + 慢退火）

EXPLORE_KNOBS: list[dict] = [
    dict(name="lr-restart", why="lr 重启：2.2e-6 + 100 轮短退火（不退到 lr-min）",
         over=dict(lr=2.2e-6, lr_min_ratio=0.64, klt_anneal=100, lr_warmup=20)),
    dict(name="explore", why="探索抬升：temperature 1.35 + ent-coef 0.055",
         over=dict(lr=2.0e-6, temperature=1.35, ent_coef=0.055, lr_min_ratio=0.70,
                   klt_anneal=100, lr_warmup=20)),
    dict(name="pool-anchor", why="锚定加强：master-seat 0.25 + KLT β 0.10",
         over=dict(lr=2.0e-6, master_seat=0.25, klt_beta=0.10, lr_min_ratio=0.70,
                   klt_anneal=100, lr_warmup=20)),
]


def explore_cfg(name: str, init_ckpt: Path, iters: int, over: dict) -> dict:
    """按旋钮覆盖表构造探索段配置（模板与 dynamic_cfg 对齐，仅覆盖指定旋钮）。"""
    lr = over.get("lr", 2.0e-6)
    cfg = dict(iters=iters, init=str(init_ckpt.relative_to(ROOT)),
               out=f"models/dushan_m2_{name}.pt",
               warmup=0, teacher_prob=0.0,
               temperature=over.get("temperature", 1.2), lr=lr,
               needs=str(init_ckpt.relative_to(ROOT)), m2=True,
               extra=[])
    ex: list[str] = cfg["extra"]

    def add(flag: str, val=None) -> None:
        ex.append(flag)
        if val is not None:
            ex.append(str(val))

    add("--klt-beta", over.get("klt_beta", 0.06))
    add("--klt-anneal-iters", over.get("klt_anneal", 100))
    add("--snapshot-pool", "")                  # 空串覆盖 COMMON 的 dushan_master（与既有段一致）
    add("--ent-baseline", 1.026)
    add("--master-seat-prob", over.get("master_seat", 0.40))
    add("--lr-min", f"{lr * over.get('lr_min_ratio', 0.70):.2g}")
    add("--lr-warmup", over.get("lr_warmup", 20))
    add("--save-opt")
    add("--ent-coef", over.get("ent_coef", 0.040))
    add("--kl-coef", 0.016)
    add("--explore-eps", 0.010)
    add("--explore-eps-end", 0.010)
    add("--explore-anneal-iters", 1)
    add("--save-every", 20)
    return cfg


def read_gate_row(tag: str, at: int) -> dict | None:
    """读 gate_watch 判读报告，返回目标 ckpt 那一行（含 vs_anchor / vs_anchor_ci95）。"""
    p = REPORTS / f"{tag}_it{at:04d}.json"
    if not p.exists():
        return None
    try:
        d = json.loads(p.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    rows = d.get("rows") or []
    anchor = d.get("anchor")
    for r in rows:
        if r.get("ckpt") != anchor and f"it{at:04d}" in str(r.get("ckpt", "")):
            return r
    return next((r for r in rows if r.get("ckpt") != anchor), None)


def explore_plan(st: dict, gen: int, src: Path) -> dict | None:
    """第 gen 段（auto{gen}）的探索配方。

    gen == 6 时没有可参考的探索结果 → 直接用第一个旋钮。
    gen > 6 时需要上一段的段末判读（AUTO{gen-1}@{prev_at}，由 gate_watch 异步产出）；
    判读点排在段长之前，正常情况下总控检测到"跑满"时结果已落盘。
    **若尚未落盘 → 返回 None，主循环保持等待（不触发闲置保底）。**

    分支（稳扎稳打口径，2026-10-16 夜间版）：
      命中（Δva ≥ +0.20 且 CI 下界 > 基线）→ 沿用旋钮，加长到 800 轮
      未命中 → 换下一个旋钮（miss +1，同时刷新历史最优旋钮记录）
      连续 3 次未命中（一整轮旋钮试完）→ **收敛长段**：历史最优旋钮 + 慢退火，
        1200 轮大训练量；跑完后 miss 清零、继续下一轮探索循环。
    """
    idx = int(st.get("explore_idx", 0))
    miss = int(st.get("explore_miss", 0))
    best_knob = st.get("explore_best_knob")           # None = 尚无记录
    best_va = float(st.get("explore_best_va", -99.0))
    iters = EXPLORE_LEN
    prev_at = int(st.get("explore_verdict_at", 240))
    start_va = EXPLORE_START_VA
    hit = False
    consolidate = False

    if gen > 6:
        row = read_gate_row(f"AUTO{gen - 1}", prev_at)
        if row is None:
            return None                                     # 判读未落盘 → 等
        start_va = float(row.get("vs_anchor", EXPLORE_START_VA))
        ci = float(row.get("vs_anchor_ci95", 0.28))
        base = float(st.get("explore_base", EXPLORE_START_VA))
        if (start_va - base) >= HIT_DELTA and (start_va - ci) > base:
            hit = True                                      # 命中 → 沿用该旋钮并加长
            iters = EXPLORE_LEN_HIT
            miss = 0
        else:
            idx += 1                                        # 未命中 → 换下一个旋钮
            miss += 1
        # 刷新历史最优旋钮（判读 va 最高的那个）
        if start_va > best_va:
            best_knob = (idx - 1) % len(EXPLORE_KNOBS)
            best_va = start_va
    idx %= len(EXPLORE_KNOBS)
    knob = EXPLORE_KNOBS[idx]

    if miss >= len(EXPLORE_KNOBS) and best_knob is not None:
        # 一整轮旋钮全部试过且无一命中 → 收敛：用历史最优旋钮跑 1200 轮长段
        consolidate = True
        idx = best_knob
        knob = EXPLORE_KNOBS[idx]
        iters = EXPLORE_LEN_CONS
        miss = 0                                            # 收敛段跑完后重开一轮探索

    if consolidate:
        checkpoints, verdict_at = [600, 1000], 1000
    elif iters >= EXPLORE_LEN_HIT:
        checkpoints, verdict_at = [400, 640], 640
    else:
        checkpoints, verdict_at = [160, 240], 240

    name = f"auto{gen}"
    over = dict(knob["over"])
    if consolidate:                                         # 收敛段：放慢节奏（稳扎稳打）
        over["klt_anneal"] = 300
        over["lr_warmup"] = 40
        over["lr"] = over.get("lr", 2.0e-6) * 0.9
    cfg = explore_cfg(name, src, iters, over)
    tagname = ("收敛长段" if consolidate else
               f"探索[{knob['name']}]" + ("（命中→加长）" if hit else ""))
    return dict(dyn=cfg, tag=f"AUTO{gen}", iters=iters, checkpoints=checkpoints,
                start_va=start_va, knob=idx, verdict_at=verdict_at,
                miss=miss, best_knob=best_knob, best_va=best_va,
                why=(tagname + f" {knob['why']}"
                     + f"｜{iters} 轮，起点 va={start_va:+.3f}"))


def promote_master(cand: Path) -> None:
    master = ROOT / "models" / "dushan_master.pt"
    backup = ROOT / "models" / "dushan_master_pre_m2.pt"
    if not backup.exists():
        shutil.copy2(master, backup)
        say(f"  master 备份 → {backup.relative_to(ROOT)}")
    shutil.copy2(cand, master)
    say(f"  ★ master 晋升：{cand.relative_to(ROOT)} → dushan_master.pt")


# --------------------------------------------------------------------------- #
# 决策
# --------------------------------------------------------------------------- #
def decide(st: dict, meta: dict, alive: bool, it_done: int) -> dict | None:
    """返回 {'seg'|'dyn', 'tag', 'checkpoints', 'start_va', 'why', 'init_hint'} 或 None。"""
    seg, stem = meta["seg"], meta["stem"]
    iters = meta["iters"]

    if alive:
        return None                                        # 还在跑 → 不动

    if seg == "s4b4":
        v100 = read_verdict(REPORTS / "B4_s4b4_it0100.verdict.json")
        v300 = read_verdict(REPORTS / "B4_s4b4_it0300.verdict.json")
        v600 = read_verdict(REPORTS / "G3_s4b4_it0600.verdict.json")
        d100 = read_stoploss_decision("dushan_m2_b4", "B4@100")
        d300 = read_stoploss_decision("dushan_m2_b4", "B4@300")

        if d100 and d100.get("triggered"):
            va = (v100 or {}).get("vs_anchor", -1.45)
            return dict(seg="s4b5", tag="B5", checkpoints=[250, 500], start_va=va,
                        init_hint="models/_ckpts/dushan_m2_b4_it0100.pt",
                        why=f"B4 it0100 止损触发（va={va:+.3f}）→ 从 b4_it0100 无缝续训")

        if d300 and d300.get("triggered"):
            va = (v300 or {}).get("vs_anchor", -1.05)
            return dict(seg="s4b6", tag="B6", checkpoints=[150, 300], start_va=va,
                        init_hint="models/_ckpts/dushan_m2_b4_it0300.pt",
                        why=f"B4 it0300 止损触发（va={va:+.3f}）→ 从 b4_it0300 无缝续训")

        if v600 is not None or it_done >= iters:
            va = (v600 or {}).get("vs_anchor", -1.0)
            passed = bool((v600 or {}).get("passed"))
            return dict(seg="s4c1", tag="C1", checkpoints=[400, 800, 1200],
                        start_va=va, init_hint="models/dushan_m2_b4.pt",
                        promote=passed, g3=v600,
                        why=(f"B4 跑满 {iters} 轮且 G3 达标（va={va:+.3f}）→ 晋升 master + 长跑巩固"
                             if passed else
                             f"B4 跑满 {iters} 轮但 G3 未达标（va={va:+.3f}）→ 长跑继续爬"))

        # 有判决但 stoploss 决策文件还没落盘（poll 周期 60s）→ 等，不要误判为崩溃
        if v100 is not None or v300 is not None:
            return None

        # 崩溃（无任何判决、轮次未跑满、日志已停滞）→ 从本段最新 ckpt 续跑剩余
        lc = latest_ckpt(stem)
        stalled = True
        if meta["log"].exists():
            stalled = (time.time() - meta["log"].stat().st_mtime) > 600
        if lc and it_done < iters and stalled:
            remain = max(60, iters - lc[1])
            return dict(dyn=dynamic_cfg(f"{stem[10:]}_r1", lc[0], remain, 3e-6),
                        tag="B4R", checkpoints=[remain], start_va=-1.06,
                        why=f"B4 异常中断（跑到 it{it_done}）→ 从 it{lc[1]} 续跑 {remain} 轮")
        return None

    # ---- 接力段（s4b5 / s4b6 / s4c1 / autoN）----
    if it_done >= iters:
        lc = latest_ckpt(stem)
        out_p = ROOT / meta["out"]
        if lc is None and not out_p.exists():
            return None                        # 既无快照也无主输出 → 再等一轮
        src = lc[0] if lc else out_p
        gen = st.get("relay_count", 0) + 3
        if gen >= 6:
            # auto6 起进入探索计划：可能返回 None（= 等段末判读落盘），主循环会继续 poll
            return explore_plan(st, gen, src)
        return dict(dyn=dynamic_cfg(f"auto{gen}", src, 800, 1.5e-6),
                    tag=f"AUTO{gen}", checkpoints=[400, 800], start_va=-1.0,
                    why=f"{seg} 跑满 {iters} 轮 → 从 {src.name} 续 800 轮（lr 下压）")

    # 接力段被止损
    for gate in ("B5@500", "B6@300", "C1@400", "C1@800", "C1@1200"):
        d = read_stoploss_decision(stem, gate)
        if d and d.get("triggered"):
            va = d.get("vs_anchor", -1.3)
            lc = latest_ckpt(stem)
            if lc:
                gen = st.get("relay_count", 0) + 3
                return dict(dyn=dynamic_cfg(f"auto{gen}", lc[0], 600, 2e-6),
                            tag=f"AUTO{gen}", checkpoints=[300, 600], start_va=va,
                            why=f"{seg} 在 {gate} 止损（va={va:+.3f}）→ 从 it{lc[1]} 续 600 轮")

    # 接力段异常中断
    lc = latest_ckpt(stem)
    if lc and it_done < iters:
        remain = max(60, iters - lc[1])
        return dict(dyn=dynamic_cfg(f"{seg}_r1", lc[0], remain, 2e-6),
                    tag=f"{seg.upper()}R", checkpoints=[remain], start_va=-1.2,
                    why=f"{seg} 异常中断（跑到 it{it_done}）→ 从 it{lc[1]} 续跑 {remain} 轮")
    return None


# --------------------------------------------------------------------------- #
# 主循环
# --------------------------------------------------------------------------- #
def main() -> int:
    ap = argparse.ArgumentParser(description="M2 夜间自主接力（无人值守）")
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--poll-s", type=int, default=60)
    ap.add_argument("--dry-run", action="store_true", help="只打印决策，不动作")
    ap.add_argument("--max-hours", type=float, default=16.0)
    args = ap.parse_args()

    say("=" * 74)
    say("M2 夜间自主接力启动（用户授权：按推荐选项执行，不闲置算力）")
    say("目标：10M 模型（9,849,902 参数）追平并超越 S11 = dushan_master（769,422 参数）")
    say("核心杠杆：从含 opt 的检查点无缝续训（不再有冷 Adam 位移）")
    say(f"止损线口径：起点实测 va − {STOP_LINES}σ（σ={SIGMA}，2400 副单点 CI）")
    say(f"探索计划：已启用（auto6 起 {EXPLORE_LEN} 轮短段，逐段只动一个旋钮；"
        f"命中判据 Δva ≥ {HIT_DELTA:+.2f} → 加长到 {EXPLORE_LEN_HIT} 轮）")
    say(f"  旋钮序列：{' / '.join(k['name'] for k in EXPLORE_KNOBS)}")
    say("=" * 74)

    t0 = time.time()
    idle_since: float | None = None

    while (time.time() - t0) / 3600.0 < args.max_hours:
        st = load_state()
        seg = st["current"]
        cfg = SEGMENTS.get(seg) or (st.get("dyn_segs") or {}).get(seg)
        if cfg is None:
            say(f"⚠️ 未知段 {seg}，退出")
            return 2
        meta = seg_meta(seg, cfg)

        pid = read_pid(meta["pid_file"])
        alive = pid_alive(pid)
        it_done = log_reached(meta["log"])

        plan = decide(st, meta, alive, it_done)

        if plan is None:
            # 运行期自检（一次性）：接力策略的命门 = 检查点必须含 AdamW 状态。
            # 若这里发现不含 opt，则"无缝续训"假设不成立，日志里必须留下硬证据。
            verific = st.setdefault("opt_verified", {})
            if alive and seg not in verific:
                lc = latest_ckpt(meta["stem"])
                if lc and lc[1] >= 20:
                    ok = has_opt(lc[0])
                    say(f"🔍 自检 {lc[0].name}："
                        + ("含 AdamW 状态 ✅ 接力可零位移" if ok
                           else "❌ 不含 opt —— 接力会冷启动，需复查 --save-opt！"))
                    verific[seg] = {"ckpt": lc[0].name, "has_opt": ok, "at": ts()}
                    save_state(st)

            if not alive:
                # 探索段跑满后会进入「等段末判读落盘」状态（explore_plan 返回 None）：
                # 判读点排在段长之前，正常在总控检测到跑满时结果已落盘。这段等待**不算闲置**，
                # 不能被 900 s 保底逻辑打断 —— 否则会用兜底配方顶掉探索计划。
                seg_await = (it_done >= meta["iters"]
                             and st.get("relay_count", 0) + 3 >= 6
                             and str(seg).startswith("auto"))
                if seg_await:
                    if idle_since is None:
                        idle_since = time.time()
                        say(f"⏳ {seg} 已跑满 {it_done} 轮，等段末判读落盘后决定下一段配方…")
                    elif time.time() - idle_since > 3000:
                        say("⏳ 判读等待超时 50 分 → 以兜底配方接力（不闲置）")
                        lc = latest_ckpt(meta["stem"])
                        if lc:
                            plan = dict(dyn=dynamic_cfg(f"{meta['stem'][10:]}_keep", lc[0],
                                                        600, 2e-6),
                                        tag=f"{seg.upper()}K", checkpoints=[300, 540],
                                        start_va=-1.0,
                                        why=f"判读超时兜底：从 it{lc[1]} 续 600 轮")
                elif idle_since is None:
                    idle_since = time.time()
                    say(f"⚠️ 段 {seg} 训练已停、暂无可执行接力（等待判决/产物）")
                elif time.time() - idle_since > 900:
                    say(f"⚠️ 已闲置 {(time.time() - idle_since) / 60:.0f} 分 —— "
                        f"启动保底续训，避免夜间空转")
                    lc = latest_ckpt(meta["stem"])
                    if lc:
                        plan = dict(dyn=dynamic_cfg(f"{meta['stem'][10:]}_keep", lc[0],
                                                    600, 2e-6),
                                    tag=f"{seg.upper()}K", checkpoints=[300, 600],
                                    start_va=-1.2,
                                    why=f"保底：{seg} 停跑但无判决 → 从 it{lc[1]} 续 600 轮")
            else:
                idle_since = None
            if plan is None:
                if args.dry_run:
                    say(f"（dry-run）段 {seg}：alive={alive}，已跑到 it{it_done}/{meta['iters']}，"
                        f"暂无动作 —— 决策树待判决触发")
                    return 0
                time.sleep(args.poll_s)
                continue
        else:
            idle_since = None

        say("-" * 74)
        say(f"【决策】{plan['why']}")

        if args.dry_run:
            say(f"  （dry-run）将起 {plan.get('seg') or plan['tag']}，"
                f"判读点 {plan['checkpoints']}，起点 va≈{plan['start_va']:+.3f}，"
                f"init≈{plan.get('init_hint')}")
            return 0

        # 0) 确保旧训练已完全退出
        ensure_stopped(pid, seg)

        # 1) 起点检查（接力必须从"含 opt 的检查点"起 —— 这是零位移的前提）
        init_hint = plan.get("init_hint")
        if init_hint:
            ip = ROOT / init_hint
            if not ip.exists():
                lc = latest_ckpt(meta["stem"])
                if lc:
                    say(f"  ⚠️ {init_hint} 不存在 → 回退到最新 ckpt {lc[0].name}")
                    init_hint = str(lc[0].relative_to(ROOT))
                else:
                    say(f"  ❌ {init_hint} 与任何 ckpt 都不存在 → 无法接力，等 30 分重试")
                    time.sleep(1800)
                    continue
            hasopt = has_opt(ROOT / init_hint)
            say(f"  起点 {init_hint}：{'含 AdamW 状态 ✅（零位移）' if hasopt else '⚠️ 无 opt → 冷启动，靠 warmup 缓冲'}")
            if plan.get("seg"):
                SEGMENTS[plan["seg"]]["init"] = init_hint
                plan["seg_cfg"] = SEGMENTS[plan["seg"]]
            else:
                plan["dyn"]["init"] = init_hint
                plan["dyn"]["needs"] = init_hint

        # 2) 晋升（严格判据）
        if plan.get("promote"):
            v = plan["g3"] or {}
            va, lo = v.get("vs_anchor"), v.get("lower")
            if va is not None and va >= 0.5 and (lo is None or lo > 0):
                src = ROOT / meta["out"]
                if not src.exists():
                    lc = latest_ckpt(meta["stem"])
                    src = lc[0] if lc else src
                if src.exists():
                    promote_master(src)
            else:
                say(f"  G3 未达严格晋升判据（va={va}，CI 下界 {lo}）→ 只继续训练，不动 master")

        # 3) 清旧守望 → 起接力段 → 挂新守望
        #    keep_gate=True：只清 stoploss，**保留判读守望**——旧段的兜底快照刚
        #    落盘，它的 2400 副判读还没跑完（AUTO4@800 事故）；判读与新段并行。
        clear_old_watchdogs(keep_gate=True)
        if plan.get("dyn"):
            nseg = plan["tag"].lower().replace("@", "_")
            ncfg = plan["dyn"]
        else:
            nseg = plan["seg"]
            ncfg = plan.get("seg_cfg") or SEGMENTS[nseg]

        rc = spawn_segment(nseg, ncfg, args.workers)
        if rc != 0:
            say(f"  ❌ 接力段 {nseg} 起动失败（rc={rc}）—— 30 分钟后重试")
            time.sleep(1800)
            continue

        nmeta = seg_meta(nseg, ncfg)
        attach_watchdogs(nmeta, plan["start_va"], plan["checkpoints"], plan["tag"])

        st.setdefault("history", []).append(
            {"from": seg, "to": nseg, "why": plan["why"], "at": ts()})
        if plan.get("dyn"):
            st.setdefault("dyn_segs", {})[nseg] = ncfg  # 动态段配置入档，主循环才能继续监护
        if plan.get("knob") is not None:                # 探索计划状态：旋钮指针 + 命中判定基线
            st["explore_idx"] = plan["knob"]
            st["explore_base"] = plan.get("start_va", EXPLORE_START_VA)
            st["explore_verdict_at"] = plan.get("verdict_at", 240)
            st["explore_miss"] = plan.get("miss", 0)
            if plan.get("best_knob") is not None:
                st["explore_best_knob"] = plan["best_knob"]
                st["explore_best_va"] = plan.get("best_va", -99.0)
        st["current"] = nseg
        st["relay_count"] = st.get("relay_count", 0) + 1
        save_state(st)
        say(f"✅ 接力完成：{seg} → {nseg}；守望已挂，继续监护")

    say("达到 --max-hours 上限，自主接力结束（训练仍在跑则不受影响）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
