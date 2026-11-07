"""独山麻将 ExIt 无人值守主驱动（7×24 连续训练，2026-11-06 → 11-13）。

主循环（docs/14天训练计划.md §0）：
  每轮 r：DAgger 标注（学生 rollout + N32 教师 Q，100k）
        → 训练（400k 滑窗，init=exit_best/最新轮）
        → 闸门 A（300 seed 四座，贪心 vs master 贪心，p<0.05 且均值>0 才晋升）
        → 闸门 C（50 seed 四座，新贪心 vs N32，只记趋势）
        → 下一轮
时间盒：2026-11-13 02:30 之前持续运行（7 天 24 小时，无每日 08:00 截止）。

断点续跑：每个阶段完成后写 .done 标记 / jsonl 行数判据；重启自动从第一个
未完成阶段继续，绝不重复覆盖已落盘数据。防重复启动：runs/night_run.lock
心跳（30s touch，>120s 视为无驱动）——每小时自检的自动化据此决定是否重启。

硬约束：
  * 评估固定牌墙 seed 池（A: 710000+300 / C: 720000+50）、四座旋转、
    冻结基线池 {master 贪心, N32, 上轮最佳}；自对弈胜率零决策权重。
  * M1(v3) 学生标注必须 --student-enc 3 + --student-device cpu；教师保持 CoreML。
"""
from __future__ import annotations

import datetime as dt
import os
import re
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DEADLINE = dt.datetime(2026, 11, 13, 2, 30, 0)   # 7×24 训练终点
PY = str(ROOT / ".venv/bin/python")
ANNOTATE = str(ROOT / "rust_engine/target/release/annotate")
ORT_DYLIB = str(ROOT / ".venv/lib/python3.13/site-packages/onnxruntime/capi/"
                     "libonnxruntime.1.31.1.dylib")
TEACHER_ONNX = "models/onnx/dushan_master.onnx"
MASTER_PT = "models/dushan_master.pt"
LOG = ROOT / "reports/exit_log.md"
NIGHT_LOG_DIR = ROOT / "runs/night_logs"
LOCK = ROOT / "runs/night_run.lock"
MAX_ROUND = 60                                    # 7 天约 55~60 轮，留余量

# ---- PPO-lite 分支（docs/14天训练计划.md §6.24）----
# 依据：8 轮 DAgger 后闸门 A 环比 4 轮为 0、闸门 C≈0、best-val 连续 5 轮落在 ep1
# ⇒ 瓶颈是教师标签的信息天花板。PPO 用「谁赢」这个 DAgger 拿不到的信号。
# 设计：**不替换** DAgger 主线，只在每轮闸门之后追加一个 PPO 候选 exit_pNN.pt，
# 与主候选竞争；晋升要求「比现最佳显著更好」（闸门 D），否则只存档留作终评 best-of。
# 单轮成本实测 ≈8min（rollout 2min@4进程 + 更新 4~5min@mps），占一轮 4%。
PPO_FROM = 10                                     # 首个 PPO 轮次（R10）
PPO_EVERY = 1                                     # 每 N 轮跑一次（1 = 每轮）
PPO_HANDS = 960                                   # ≈6 万决策，与 max-states 匹配
PPO_DISABLE = ROOT / "runs/ppo_disabled"          # 存在即停用（远程急停开关）


def log(msg: str) -> None:
    line = f"[{dt.datetime.now():%F %H:%M:%S}] {msg}"
    print(line, flush=True)
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def heartbeat() -> None:
    """每 30s touch 锁文件；自检自动化据此判断驱动存活（不依赖 ps/TCC）。"""
    while True:
        try:
            LOCK.touch()
        except OSError:
            pass
        time.sleep(30)


def driver_alive() -> bool:
    if not LOCK.exists():
        return False
    return time.time() - LOCK.stat().st_mtime < 120


def can_start(minutes: float) -> bool:
    ok = dt.datetime.now() + dt.timedelta(minutes=minutes) < DEADLINE
    if not ok:
        log(f"时间盒拒绝：剩余不足 {minutes:.0f}min（截止 {DEADLINE:%F %H:%M}）")
    return ok


def run(cmd: list[str], tag: str, env_extra: dict | None = None,
        timeout_s: int = 9000) -> tuple[int, str]:
    """跑一条命令，输出落盘 runs/night_logs/<tag>.log，返回 (rc, 全量输出)。"""
    NIGHT_LOG_DIR.mkdir(parents=True, exist_ok=True)
    out_path = NIGHT_LOG_DIR / f"{tag}.log"
    env = os.environ.copy()
    env["ORT_DYLIB_PATH"] = ORT_DYLIB
    if env_extra:
        env.update(env_extra)
    log(f"$ {' '.join(cmd)}  → {out_path.name}")
    t0 = time.perf_counter()
    with open(out_path, "w", encoding="utf-8") as f:
        proc = subprocess.run(cmd, cwd=str(ROOT), env=env, stdout=f,
                              stderr=subprocess.STDOUT, timeout=timeout_s)
    text = out_path.read_text(encoding="utf-8", errors="replace")
    log(f"  rc={proc.returncode}  {time.perf_counter()-t0:.0f}s  {tag}")
    return proc.returncode, text


def latest_pt() -> Path:
    """exit_best 优先，否则最新 exit_rNN.pt。"""
    best = ROOT / "models/exit_best.pt"
    if best.exists():
        return best
    cands = sorted((ROOT / "models").glob("exit_r*.pt"))
    if not cands:
        raise RuntimeError("没有任何学生 ckpt")
    return cands[-1]


def export_onnx(ckpt: Path) -> Path:
    code = (
        "import torch,sys;"
        "sys.path.insert(0,'.');"
        "from zhuoji.net import build_from_ckpt;"
        "from zhuoji.forward import ensure_onnx;"
        f"ck=torch.load({str(ckpt)!r},map_location='cpu',weights_only=False);"
        "net,_=build_from_ckpt(ck);"
        f"p=ensure_onnx(net,stem={ckpt.stem!r});"
        "print('onnx:',p)"
    )
    rc, out = run([PY, "-c", code], f"export_{ckpt.stem}", timeout_s=300)
    assert rc == 0, out[-2000:]
    m = re.search(r"onnx: (\S+)", out)
    assert m, out[-2000:]
    return Path(m.group(1))


# ---------------- 阶段完成判据（断点续跑） ----------------

def data_ready(r: int) -> bool:
    import numpy as np
    fs = sorted((ROOT / f"runs/exit_r{r:02d}").glob("exit_w*.npz"))
    if len(fs) < 4:
        return False
    try:
        return all(int(np.load(f)["q"].shape[0]) >= 24_000 for f in fs)
    except Exception:
        return False


def trained(r: int) -> bool:
    ck = ROOT / f"models/exit_r{r:02d}.pt"
    if (NIGHT_LOG_DIR / f"train_r{r:02d}.done").exists():
        return ck.exists()
    # 兼容 08:00 前的旧驱动（无 marker）：训练日志出现 [done] 且 ckpt 存在
    p = NIGHT_LOG_DIR / f"train_r{r:02d}.log"
    if p.exists() and ck.exists():
        try:
            return "[done]" in p.read_text(encoding="utf-8", errors="replace")
        except OSError:
            pass
    return False


def _jsonl_rows(p: Path) -> int:
    if not p.exists():
        return 0
    with open(p, encoding="utf-8") as f:
        return sum(1 for line in f if line.strip())


def gate_a_done(r: int) -> bool:
    return _jsonl_rows(ROOT / f"runs/gates/r{r:02d}_gateA.jsonl") >= 1200  # 300 seed × 4 座


def gate_c_done(r: int) -> bool:
    return _jsonl_rows(ROOT / f"runs/gates/r{r:02d}_gateC.jsonl") >= 200    # 50 seed × 4 座


def gate_d_done(r: int) -> bool:
    return _jsonl_rows(ROOT / f"runs/gates/r{r:02d}_gateD.jsonl") >= 1200   # 300 seed × 4 座


def snapshot_prev_best(r: int):
    """闸门 D 的基线 = 本轮晋升**之前**的 exit_best（规格里的「上轮最佳」）。

    必须在闸门 A 触发 promote 之前调用，否则退化成自己比自己。
    本轮已晋升过（说明快照窗口已错过）则返回 None，本轮跳过 D。
    """
    dst = ROOT / f"runs/gates/r{r:02d}_prevbest.pt"
    if dst.exists():
        return dst
    if (ROOT / f"runs/gates/r{r:02d}.promoted").exists():
        return None
    src = ROOT / "models/exit_best.pt"
    if not src.exists():
        return None
    try:
        shutil.copy2(src, dst)           # 40MB，1s
        return dst
    except OSError:
        return None


def first_incomplete_round() -> int:
    """R1 已完成；找第一个「标注/训练/闸门 A/C 未全齐」的轮次。"""
    for r in range(2, MAX_ROUND + 1):
        if not (data_ready(r) and trained(r) and gate_a_done(r) and gate_c_done(r)):
            return r
    return MAX_ROUND + 1


# ---------------- 各阶段 ----------------

def annotate_round(r: int) -> None:
    """DAgger 标注：学生 rollout + N32 教师 Q（100k，4 分片）。"""
    if data_ready(r):
        log(f"R{r} 标注已完成（4×25k 分片齐），跳过")
        return
    out_dir = ROOT / f"runs/exit_r{r:02d}"
    seed0 = 600_000 + r * 10_000
    stu = export_onnx(latest_pt())
    cmd = [ANNOTATE, "--student", str(stu), "--teacher", TEACHER_ONNX,
           "--student-enc", "3",          # M1=v3/63 通道（默认 1 会形状 panic）
           "--student-device", "cpu",    # M1 CoreML 切 65 分区 → 吞吐崩 5 倍
           "--targets", "100000", "--threads", "4",
           "--out", str(out_dir), "--seed0", str(seed0)]
    # 4 线程聚合实测 4~5 万/h → 100k 需 2~2.5h；给 3.5h 余量，避免超时丢数据
    rc, text = run(cmd, f"annotate_r{r:02d}", timeout_s=12600)
    if rc != 0:                          # 瞬时故障重试一次
        log(f"R{r} 标注失败（rc={rc}），清理残留后重试一次")
        shutil.rmtree(out_dir, ignore_errors=True)
        rc, text = run(cmd, f"annotate_r{r:02d}_retry", timeout_s=12600)
    assert rc == 0 and data_ready(r), text[-2000:]
    log(f"R{r} 标注完成 student={stu.name} seed0={seed0} → 4×25k")


def train_round(r: int) -> None:
    """训练第 r 轮：400k 滑窗，init=exit_best（缺则最新轮）。"""
    if trained(r):
        log(f"R{r} 训练已完成（有 .done 标记），跳过")
        return
    out = ROOT / f"models/exit_r{r:02d}.pt"
    pats = [f"runs/exit_r{i:02d}/exit_w*.npz" for i in range(1, r + 1)
            if (ROOT / f"runs/exit_r{i:02d}").is_dir()]
    cmd = [PY, "train_exit.py", "--data", *pats, "--out", str(out),
           "--epochs", "30", "--window", "400000",
           "--batch", "256", "--seed", str(20261106 + r)]
    init = None
    if r > 1:
        init = latest_pt()
        cmd += ["--init", str(init)]
    rc, text = run(cmd, f"train_r{r:02d}", timeout_s=7200)
    assert rc == 0 and out.exists(), text[-2000:]
    (NIGHT_LOG_DIR / f"train_r{r:02d}.done").write_text("ok\n", encoding="utf-8")
    log(f"R{r} 训练完成 init={(init.name if init else '冷启动')} → {out.name}  "
        f"({[l for l in text.splitlines() if '[done]' in l] or ['no done line']})")


def _gate_pass(text: str) -> bool:
    return "判定: 通过" in text


def _gate_line(text: str) -> str:
    for l in text.splitlines():
        if l.startswith("[summary] n=") or l.startswith("[summary] seats="):
            return l
    return "(no summary)"


def gates_round(r: int) -> None:
    """闸门 A（晋升判据）+ 闸门 C（趋势）+ 闸门 D（环比上轮最佳），各自幂等跳过。"""
    ck = ROOT / f"models/exit_r{r:02d}.pt"
    base = str(ROOT / MASTER_PT)
    prevb = snapshot_prev_best(r)        # 必须在下面的 promote 之前取快照
    if not gate_a_done(r):
        out_a = ROOT / f"runs/gates/r{r:02d}_gateA.jsonl"
        cmd_a = [PY, "eval_pairs.py", "--a", str(ck), "--b", base,
                 "--mode", "greedy", "--seeds", "300", "--seed0", "710000",
                 "--workers", "6", "--out", str(out_a)]
        rc_a, txt_a = run(cmd_a, f"gateA_r{r:02d}", timeout_s=5400)
        assert rc_a == 0, txt_a[-2000:]
        passed = _gate_pass(txt_a)
        log(f"R{r} 闸门 A: {_gate_line(txt_a)}  → {'通过' if passed else '不通过'}")
        if passed and not (ROOT / f"runs/gates/r{r:02d}.promoted").exists():
            promote(r)
    if not gate_c_done(r):
        out_c = ROOT / f"runs/gates/r{r:02d}_gateC.jsonl"
        cmd_c = [PY, "eval_pairs.py", "--a", str(ck), "--b", base,
                 "--a-mode", "greedy", "--b-mode", "search32",
                 "--seeds", "50", "--seed0", "720000",
                 "--workers", "6", "--out", str(out_c)]
        rc_c, txt_c = run(cmd_c, f"gateC_r{r:02d}", timeout_s=5400)
        if rc_c == 0:
            log(f"R{r} 闸门 C(趋势): {_gate_line(txt_c)}")
        else:
            log(f"R{r} 闸门 C 运行失败（不影响晋升）rc={rc_c}")
    # 闸门 D：与「上轮最佳」的配对对比 —— 这才是真实环比进步，闸门 A 只比 master 恒过
    if prevb is not None and not gate_d_done(r):
        out_d = ROOT / f"runs/gates/r{r:02d}_gateD.jsonl"
        cmd_d = [PY, "eval_pairs.py", "--a", str(ck), "--b", str(prevb),
                 "--mode", "greedy", "--seeds", "300", "--seed0", "710000",
                 "--workers", "6", "--out", str(out_d)]
        rc_d, txt_d = run(cmd_d, f"gateD_r{r:02d}", timeout_s=5400)
        if rc_d == 0:
            log(f"R{r} 闸门 D(vs 上轮最佳): {_gate_line(txt_d)}")
            try:
                prevb.unlink(missing_ok=True)      # 只在成功后清理快照
            except OSError:
                pass
        else:
            log(f"R{r} 闸门 D 运行失败（不影响晋升）rc={rc_d}")


def promote_ckpt(src: Path, tag: str) -> None:
    dst = ROOT / "models/exit_best.pt"
    shutil.copy2(src, dst)
    export_onnx(dst)                     # 同步 ONNX 缓存 stem
    (ROOT / f"runs/gates/{tag}.promoted").write_text("ok\n", encoding="utf-8")
    log(f"晋升：{src.name} → exit_best.pt")


def promote(r: int) -> None:
    promote_ckpt(ROOT / f"models/exit_r{r:02d}.pt", f"r{r:02d}")


def ppo_init(r: int) -> Path:
    """PPO 链式初始化：默认续上一条 PPO 链，让每轮 960 局的少量经验跨轮累积。

    为什么必须链式：960 局 ≈6 万决策 / 2 epoch / lr 5e-5 的更新量极小（冒烟实测
    kl≈+0.005），若每轮都从 DAgger 现最佳重新起步，则每轮都是「一次几乎零位移的
    微扰」⇒ 40 轮的 RL 经验互不叠加，等于安慰剂，闸门 D 必然恒 ≈0。
    链式后 40 轮累计 ≈3.8 万局 ≈240 万决策，才是一次真实的 RL 实验。

    r=10 时 exit_p09.pt 不存在 ⇒ 回退 latest_pt()，与修正前行为**完全等价**（零风险）。
    runs/ppo_rechain 存在（远程急停标志）或上一轮链缺 ckpt ⇒ 从 DAgger 现最佳重启链。
    """
    if not (ROOT / "runs/ppo_rechain").exists():
        prev = ROOT / f"models/exit_p{r - 1:02d}.pt"
        if prev.exists():
            return prev
    return latest_pt()


def ppo_round(r: int) -> None:
    """PPO-lite 候选轮（§6.24）：纯策略自对弈 + 终端分差奖励 → exit_p{r:02d}.pt。

    晋升判据 = 闸门 D（vs 现最佳）p<0.05 且均值>0 **同时** 闸门 A（vs master）通过。
    300 seed 的池分辨率约 0.5 分/SE，因此单轮 PPO 增量多半达不到显著 —— 那就不晋升，
    但 ckpt 与 jsonl 全部留档，供终评两段式 best-of 在独立池上复赛。
    异常绝不外泄：主线 DAgger 不受影响。
    """
    if PPO_DISABLE.exists():
        return
    if r < PPO_FROM or (r - PPO_FROM) % PPO_EVERY:
        return
    out = ROOT / f"models/exit_p{r:02d}.pt"
    if out.exists():
        log(f"P{r:02d} PPO 已完成（ckpt 在），跳过")
        return
    init = ppo_init(r)
    log(f"P{r:02d} PPO 链起点 = {init.name}")
    cmd = [PY, "train_ppo.py", "--init", str(init), "--out", str(out),
           "--hands", str(PPO_HANDS), "--workers", "4", "--threads", "2",
           "--max-states", "60000", "--epochs", "2", "--lr", "5e-5",
           "--seed0", str(900_000 + r * 10_000)]
    rc, text = run(cmd, f"ppo_p{r:02d}", timeout_s=5400)
    if rc != 0 or not out.exists():
        log(f"P{r:02d} PPO 训练失败 rc={rc}（主线不受影响）")
        return
    mt = [ln for ln in text.splitlines() if ln.startswith("[metrics]")]
    log(f"P{r:02d} PPO 完成 init={init.name} → {out.name} "
        f"{mt[-1][len('[metrics] '):] if mt else ''}")

    best = str(ROOT / "models/exit_best.pt")
    # 闸门 D 先跑：基线是「此刻的」exit_best（本轮 DAgger 候选可能刚晋升）
    out_d = ROOT / f"runs/gates/p{r:02d}_gateD.jsonl"
    rc_d, txt_d = run([PY, "eval_pairs.py", "--a", str(out), "--b", best,
                       "--mode", "greedy", "--seeds", "300", "--seed0", "710000",
                       "--workers", "6", "--out", str(out_d)],
                      f"gateD_p{r:02d}", timeout_s=5400)
    d_pass = rc_d == 0 and _gate_pass(txt_d)
    if rc_d == 0:
        log(f"P{r:02d} 闸门 D(vs 现最佳): {_gate_line(txt_d)}  → "
            f"{'通过' if d_pass else '不通过'}")
    # 闸门 A：绝对强度记录（与 r 轮同池同口径，保证可比）
    out_a = ROOT / f"runs/gates/p{r:02d}_gateA.jsonl"
    rc_a, txt_a = run([PY, "eval_pairs.py", "--a", str(out),
                       "--b", str(ROOT / MASTER_PT), "--mode", "greedy",
                       "--seeds", "300", "--seed0", "710000",
                       "--workers", "6", "--out", str(out_a)],
                      f"gateA_p{r:02d}", timeout_s=5400)
    a_pass = rc_a == 0 and _gate_pass(txt_a)
    if rc_a == 0:
        log(f"P{r:02d} 闸门 A(vs master): {_gate_line(txt_a)}")
    if d_pass and a_pass and not (ROOT / f"runs/gates/p{r:02d}.promoted").exists():
        promote_ckpt(out, f"p{r:02d}")
    elif not d_pass:
        log(f"P{r:02d} 不晋升（未显著优于现最佳），ckpt 留档供终评 best-of")


def main() -> int:
    if driver_alive():
        print("已有驱动在跑（night_run.lock 心跳 <120s），本次不启动", flush=True)
        return 0
    LOCK.parent.mkdir(parents=True, exist_ok=True)
    LOCK.touch()
    threading.Thread(target=heartbeat, daemon=True).start()
    start = first_incomplete_round()
    log("=" * 60)
    log(f"7×24 ExIt 驱动启动：R{start} 起，截止 {DEADLINE:%F %H:%M}"
        f"（R1 已完成；断点续跑 + 每小时自检自动化）")
    failures = 0
    consecutive_fail = 0
    for r in range(start, MAX_ROUND + 1):
        try:
            if not can_start(130):
                break
            annotate_round(r)
            if not can_start(35):
                break
            train_round(r)
            if can_start(15):
                gates_round(r)
            else:
                log(f"R{r} 闸门被时间盒跳过（下次自检补跑）")
            if can_start(15):
                try:
                    ppo_round(r)
                except Exception as e:       # noqa: BLE001 —— PPO 失败绝不拖累主线
                    log(f"R{r} PPO 轮次异常（主线不受影响）："
                        f"{type(e).__name__}: {e}")
            log(f"R{r} 轮次完成  best="
                f"{'exit_best.pt' if (ROOT/'models/exit_best.pt').exists() else '无（未晋升）'}")
            consecutive_fail = 0
        except subprocess.TimeoutExpired as e:
            failures += 1
            consecutive_fail += 1
            log(f"R{r} 阶段超时：{e}")
        except Exception as e:  # noqa: BLE001
            failures += 1
            consecutive_fail += 1
            log(f"R{r} 阶段异常：{type(e).__name__}: {e}")
        if consecutive_fail >= 3:
            log("连续 3 轮失败 → 退出，等待每小时自检定位后重启")
            break
    log(f"7×24 驱动结束 failures={failures}  best="
        f"{'exit_best.pt' if (ROOT/'models/exit_best.pt').exists() else '无（未晋升）'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
