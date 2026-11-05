"""夜间自主 ExIt 驱动（2026-11-05 夜 → 11-06 晨）。

无人值守跑 14 天计划的 ExIt 主循环前几轮：
  R1（数据已采）→ 训练 → 闸门 A+C →（通过则晋升 exit_best）
  → R2/R3：DAgger rollout+标注（教师=N32）→ 训练 → 闸门 → 晋升
时间盒：08:00 后不再启动新的标注/训练阶段；全程写 reports/exit_log.md。

约定（docs/14天训练计划.md §0）：
  * 闸门 A（300 seed 四座，贪心 vs master 贪心）决定晋升；
  * 闸门 C（50 seed 四座，贪心 vs N32）只记录趋势，不拦晋升；
  * 评估 seed 池固定（A: 710000+300 / C: 720000+50），各轮同池可比；
  * rollout 用「当前最佳学生」（exit_best，缺则最新轮 ckpt）；
  * 训练链：init = exit_best（缺则上一轮 ckpt），数据窗 400k 滑动；
  * 自对弈指标不参与任何决策。
"""
from __future__ import annotations

import datetime as dt
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DEADLINE = dt.datetime(2026, 11, 6, 8, 0, 0)     # 此后不再启动长阶段
PY = str(ROOT / ".venv/bin/python")
ANNOTATE = str(ROOT / "rust_engine/target/release/annotate")
ORT_DYLIB = str(ROOT / ".venv/lib/python3.13/site-packages/onnxruntime/capi/"
                     "libonnxruntime.1.31.1.dylib")
TEACHER_ONNX = "models/onnx/dushan_master.onnx"
MASTER_PT = "models/dushan_master.pt"
LOG = ROOT / "reports/exit_log.md"
NIGHT_LOG_DIR = ROOT / "runs/night_logs"

# (轮次, 标注 seed0) —— R1 数据+训练+闸门已完成（23:19–00:00 首跑）
# 续跑从 R2 起；多排几轮，靠时间盒（08:00）自然截断。
ROUNDS = [
    (2, 620_000),
    (3, 630_000),
    (4, 640_000),
    (5, 650_000),
    (6, 660_000),
]


def log(msg: str) -> None:
    line = f"[{dt.datetime.now():%H:%M:%S}] {msg}"
    print(line, flush=True)
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def can_start(minutes: float) -> bool:
    ok = dt.datetime.now() + dt.timedelta(minutes=minutes) < DEADLINE
    if not ok:
        log(f"时间盒拒绝：剩余不足 {minutes:.0f}min（截止 {DEADLINE:%H:%M}）")
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


def train_round(r: int) -> Path:
    """训练第 r 轮：窗口滑动 400k，init=exit_best（缺则上一轮；R1 冷启动）。"""
    out = ROOT / f"models/exit_r{r:02d}.pt"
    pats = [f"runs/exit_r{i:02d}/exit_w*.npz" for i in range(1, r + 1)]
    cmd = [PY, "train_exit.py", "--data", *pats, "--out", str(out),
           "--epochs", "30", "--window", "400000",
           "--batch", "256", "--seed", str(20261106 + r)]
    init = None
    if r > 1:
        init = latest_pt()
        cmd += ["--init", str(init)]
    rc, text = run(cmd, f"train_r{r:02d}", timeout_s=7200)
    assert rc == 0 and out.exists(), text[-2000:]
    log(f"R{r} 训练完成 init={(init.name if init else '冷启动')} → {out.name}  "
        f"({[l for l in text.splitlines() if '[done]' in l] or ['no done line']})")
    return out


def annotate_round(r: int, seed0: int) -> None:
    """DAgger 标注：学生 rollout + N32 教师 Q。学生均为 M1（v3 编码，63 通道）
    —— 必须带 --student-enc 3，否则 Rust 默认按 v1(15 通道) 喂学生前向而炸。"""
    out_dir = ROOT / f"runs/exit_r{r:02d}"
    stu = export_onnx(latest_pt())
    cmd = [ANNOTATE, "--student", str(stu), "--teacher", TEACHER_ONNX,
           "--student-enc", "3",
           # M1(v3) 学生在 CoreML 上被切成 65 个分区，batch=1 单步争用 ANE
           # 会把吞吐打到 R1 的 1/5（实测 2.3k/h·worker）；CPU EP 恢复
           # 到 ~12.8k/h·worker。教师仍走 CoreML（CPU 上教师慢 18 倍）。
           "--student-device", "cpu",
           "--targets", "100000", "--threads", "4",
           "--out", str(out_dir), "--seed0", str(seed0)]
    rc, text = run(cmd, f"annotate_r{r:02d}", timeout_s=9000)
    if rc != 0:                      # 瞬时故障（ORT/CoreML 偶发）重试一次
        log(f"R{r} 标注失败（rc={rc}），清理残留后重试一次")
        shutil.rmtree(out_dir, ignore_errors=True)
        rc, text = run(cmd, f"annotate_r{r:02d}_retry", timeout_s=9000)
    n = len(list(out_dir.glob("exit_w*.npz")))
    assert rc == 0 and n >= 1, text[-2000:]
    log(f"R{r} 标注完成 student={stu.name} → {out_dir.name} ({n} 分片)")


def _gate_pass(text: str) -> bool:
    return "判定: 通过" in text


def _gate_line(text: str) -> str:
    for l in text.splitlines():
        if l.startswith("[summary] n=") or l.startswith("[summary] seats="):
            return l
    return "(no summary)"


def gates_round(r: int, with_c: bool = True) -> bool:
    """闸门 A（晋升判据）+ 闸门 C（趋势记录）。返回 A 是否通过。"""
    ck = ROOT / f"models/exit_r{r:02d}.pt"
    base = str(ROOT / MASTER_PT)
    out_a = ROOT / f"runs/gates/r{r:02d}_gateA.jsonl"
    cmd_a = [PY, "eval_pairs.py", "--a", str(ck), "--b", base,
             "--mode", "greedy", "--seeds", "300", "--seed0", "710000",
             "--workers", "6", "--out", str(out_a)]
    rc_a, txt_a = run(cmd_a, f"gateA_r{r:02d}", timeout_s=5400)
    assert rc_a == 0, txt_a[-2000:]
    passed = _gate_pass(txt_a)
    log(f"R{r} 闸门 A: {_gate_line(txt_a)}  → {'通过' if passed else '不通过'}")

    if with_c:
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
    return passed


def promote(r: int) -> None:
    src = ROOT / f"models/exit_r{r:02d}.pt"
    dst = ROOT / "models/exit_best.pt"
    shutil.copy2(src, dst)
    # 同步 ONNX 缓存 stem
    export_onnx(dst)
    log(f"晋升：{src.name} → exit_best.pt")


def main() -> int:
    log("=" * 60)
    log("夜间 ExIt 驱动续跑（R2-R6，截止 08:00；R1 已完成）")
    failures = 0
    for r, seed0 in ROUNDS:
        try:
            # ---- 标注（R1 数据已就绪）----
            if not can_start(130):
                break
            annotate_round(r, seed0)
            # ---- 训练（实测 ~18-30min/轮，35min 预算）----
            if not can_start(35):
                break
            train_round(r)
            # ---- 闸门（A 实测 ~1min，C ~25min）----
            if can_start(15):
                with_c = can_start(25)
                if gates_round(r, with_c=with_c):
                    promote(r)
            else:
                log(f"R{r} 闸门被时间盒跳过（留给晨间补跑）")
        except subprocess.TimeoutExpired as e:
            failures += 1
            log(f"R{r} 阶段超时：{e}")
        except Exception as e:  # noqa: BLE001
            failures += 1
            log(f"R{r} 阶段异常：{type(e).__name__}: {e}")
            if r == 1:
                log("R1（冷启动首轮）失败 → 终止，需人工检查训练链路")
                return 1
    log(f"夜间驱动结束 failures={failures}  best="
        f"{'exit_best.pt' if (ROOT/'models/exit_best.pt').exists() else '无（未晋升）'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
