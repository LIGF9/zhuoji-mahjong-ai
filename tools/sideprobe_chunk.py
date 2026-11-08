#!/usr/bin/env python3
"""独立旁证 E-SIDE 分块驱动（§6.67 预登记；探索性 · 降样本量）。

与组合② 的唯一差异是 **seed 池与样本量**；搜索口径逐位不变：
    A = models/exit_p48.pt       (enc=3, search32)
    B = models/dushan_master.pt  (enc=1, search32)
    n_det=32 / ply=3 / topk=3 / ff_cap=24 / fair=true, 四座 0,1,2,3,
    device=mps, workers=6
池 6_000_000 .. 6_000_119（120 seed），输出 480 行 = 120 × 4 座。
§6.64.5 禁并发两片评估 ⇒ 本脚本只由 tools/keepalive.py 在
combo2_done() 为真之后拉起，绝不自行启动。

工程要点（沿用 §6.66.1 事故教训）：
  * 每片 20 seed 再切 4 个 5-seed 块，共用同一 --out ⇒ 必须 --no-truncate
    追加写 + 按 (seed,cand_seat) dedupe()，否则前序块会被清空抹掉。
  * PIDFILE 防重；分片幂等跳过；零删除、原地重写。
  * 合并前强制校验 480 行且 480 个唯一 (seed,cand_seat)；不完整则**不写**
    sideprobe_report_section.md（该文件是 keepalive 的完成判据之一），
    避免守护误判收口。

判据见 docs/14天训练计划.md §6.67.3（看结果前锁死），本脚本只产出数据，
不作任何判定、不改报告 §3、不动线上模型。
用法：.venv/bin/python tools/sideprobe_chunk.py
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PY = str(ROOT / ".venv/bin/python")
GATES = ROOT / "runs" / "gates"
LOG = GATES / "sideprobe.log"
PIDFILE = GATES / "sideprobe_chunk.pid"
MAIN_OUT = GATES / "sideprobe_main.jsonl"
REPORT_OUT = GATES / "sideprobe_report_section.md"
INCOMPLETE_FLAG = GATES / "sideprobe_incomplete.flag"

BASE = 6_000_000
N_SEEDS = 120
SHARD = 20          # 每片 seed 数（与组合② 分片结构对齐）
CHUNK = 5           # 每块 seed 数（≈13 min，被回收最多丢 1 块）
SEATS = "0,1,2,3"
NEED_PER_SEED = 4
NEED_ROWS = N_SEEDS * NEED_PER_SEED       # 480

REPORT_HEADER = """# 独立旁证 E-SIDE · 探索性结果段（§6.67）

> **探索性实验，不是 §4 主目标判定。** 本段不得写入
> `reports/终评报告-2026-11-13.md` §3；不得改判主目标；不得与组合② 数据合并；
> 不得用于任何模型/线上决策（主力恒 `dushan_master`，冠军只增不替）。

- 口径：双侧 `search32`（`n_det=32 / ply=3 / topk=3 / ff_cap=24 / fair`），四座旋转，
  `exit_p48` vs `dushan_master`，mps / 6 workers —— 与 §6.64.4 逐位一致。
- 池 `6_000_000 .. 6_000_119`（120 seed，全新独立池，与已耗池零重叠）。
- **判据（看结果前锁死，§6.67.3）**：`SE(120)=0.440`，双侧 `p<.05` 需 `|mean|≥0.862`。

  | 观测 | 读数 | 允许 / 禁止 |
  |---|---|---|
  | `p<.05 且 mean≥+0.862` | 正方向且过分辨率线 | 可写「支持正方向的旁证」；禁写「§4 达成」 |
  | `p≥.05` 或 `\\|mean\\|<+0.862` | **分辨率不足**（默认预期） | 只写「无足够证据」；禁写「无差异 / 并列 / 到顶」 |
  | `p<.05 且 mean≤−0.862` | 负方向且显著 | 只记「跨池方向冲突」告警，不改决策 |

- 必须同时报 Δ±SE、z、双侧 p、符号胜:负、完整 seed 数（=120，四座齐全）。
- 与组合② 结论不一致 ⇒ 只写「跨池稳健性存疑，需第三池」；不重跑、不调判据、不合并。
- 统计脚本：`tools/final_eval_report.py primary=...`（§6.11.7 唯一正确口径：
  `deltas[cand_seat] − mean(其余三家)` → 四座按 seed 平均 → seed 维度配对 t）。

## 实测统计输出

"""


def say(msg: str) -> None:
    line = f"=== {msg} {time.strftime('%F %H:%M:%S')} ==="
    print(line, flush=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def dedupe(path: Path) -> int:
    """按 (seed,cand_seat) 去重，保留首次顺序，零删除原地重写。"""
    if not path.exists() or path.stat().st_size == 0:
        return 0
    seen, order, bad = {}, [], 0
    for line in open(path, encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        try:
            r = json.loads(line)
        except Exception:
            bad += 1
            continue
        k = (r.get("seed"), r.get("cand_seat"))
        if k not in seen:
            order.append(k)
        seen[k] = r
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        for k in order:
            f.write(json.dumps(seen[k], ensure_ascii=False) + "\n")
    os.replace(tmp, path)
    if bad:
        print(f"[dedupe] {path.name}: kept {len(order)}, dropped {bad} bad", flush=True)
    return len(order)


def have_seeds(path: Path) -> set:
    """返回四座齐全的 seed 集合。"""
    if not path.exists() or path.stat().st_size == 0:
        return set()
    counts: dict = {}
    for line in open(path, encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        try:
            r = json.loads(line)
        except Exception:
            continue
        counts.setdefault(r.get("seed"), set()).add(r.get("cand_seat"))
    return {s for s, seats in counts.items() if len(seats) >= NEED_PER_SEED}


def run_chunk(seed0: int, out: Path) -> int:
    cmd = [PY, "eval_pairs.py",
           "--a", "models/exit_p48.pt", "--b", "models/dushan_master.pt",
           "--a-mode", "search32", "--b-mode", "search32",
           "--seeds", str(CHUNK), "--seed0", str(seed0),
           "--seats", SEATS, "--workers", "6", "--out", str(out),
           "--no-truncate"]
    with open(LOG, "a", encoding="utf-8") as lf:
        return subprocess.call(cmd, cwd=str(ROOT), stdout=lf, stderr=lf)


def main() -> int:
    GATES.mkdir(parents=True, exist_ok=True)
    if PIDFILE.exists():
        try:
            os.kill(int(PIDFILE.read_text().strip()), 0)
            print(f"another instance alive (pid={PIDFILE.read_text().strip()}); exit", flush=True)
            return 1
        except (ValueError, OSError):
            pass
    PIDFILE.write_text(str(os.getpid()), encoding="utf-8")

    shards = [GATES / f"sideprobe_s{i:02d}.jsonl" for i in range(N_SEEDS // SHARD)]
    t_all = time.time()
    for si, out in enumerate(shards):
        done = have_seeds(out)
        for k in range(SHARD // CHUNK):
            seed0 = BASE + si * SHARD + k * CHUNK
            need = set(range(seed0, seed0 + CHUNK))
            if need <= done:
                continue
            for attempt in (1, 2, 3):
                say(f"s{si:02d} chunk seed0={seed0} try{attempt} "
                    f"(have {len(done & need)}/{CHUNK} seeds)")
                rc = run_chunk(seed0, out)
                dedupe(out)
                done = have_seeds(out)
                say(f"s{si:02d} chunk seed0={seed0} rc={rc} "
                    f"have={len(done & need)}/{CHUNK}")
                if need <= done:
                    break
                time.sleep(5)
        print(f"[shard {si:02d}] done, complete seeds in file={len(done)}, "
              f"elapsed={(time.time() - t_all) / 60:.1f}min", flush=True)

    # 收口：合并 + 完整性校验 + 报告段
    say("merge")
    rows, keys, valid = [], set(), []
    for out in shards:
        if not out.exists():
            continue
        dedupe(out)
        for line in open(out, encoding="utf-8"):
            line = line.strip()
            if not line:
                continue
            rows.append(line)
            try:
                r = json.loads(line)
                k = (r.get("seed"), r.get("cand_seat"))
            except Exception:
                continue
            if k not in keys:
                keys.add(k)
                valid.append(line)
    if len(valid) != NEED_ROWS or len(keys) != NEED_ROWS:
        INCOMPLETE_FLAG.write_text(
            f"rows={len(rows)} valid={len(valid)} unique={len(keys)} need={NEED_ROWS}\n"
            f"at {time.strftime('%F %T')}\n", encoding="utf-8")
        say(f"INCOMPLETE rows={len(rows)} unique={len(keys)} need={NEED_ROWS} "
            f"→ 不写报告段，等待守护重拉补齐")
        return 2
    if INCOMPLETE_FLAG.exists():
        INCOMPLETE_FLAG.unlink()

    with open(MAIN_OUT, "w", encoding="utf-8") as f:
        f.write("\n".join(valid) + "\n")
    say(f"rows={len(valid)} -> {MAIN_OUT.name}")

    with open(REPORT_OUT, "w", encoding="utf-8") as f:
        f.write(REPORT_HEADER)
        rc = subprocess.call([PY, "tools/final_eval_report.py",
                              f"primary={MAIN_OUT}", "--md"],
                             cwd=str(ROOT), stdout=f, stderr=f)
    say(f"report rc={rc} -> {REPORT_OUT.name}")
    say("SIDEPROBE COMPLETE (chunk driver)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
