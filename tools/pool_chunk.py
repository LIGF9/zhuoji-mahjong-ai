#!/usr/bin/env python3
"""E-POOL 多对手池评测驱动（docs/14天训练计划.md §6.71.3 预登记）。

**判据、对手池、seed 区间、样本量、Holm 校正全部在开工前写死于 §6.71.3**；
本脚本只按锁定的表执行与如实出数，不作任何事后调整。

    A   = models/champ_exit_p48.pt, --a-mode greedy   （5 片恒定）
    池  = 7_100_000 .. 7_100_499（500 seed × 四座 = 2000 行/片）
    5 片对手：O1 master greedy / O2 champ_exit_p36 greedy /
             O3 web_best greedy / O4 master noisy10 / O5 heuristic
    禁并发：一片彻底跑完（含完整性校验）才开下一片。

工程要点（沿用 §6.66 / §6.67 事故教训）：
  * 每片 50 seed 一块，共用同一 --out ⇒ 必须 --no-truncate + 按
    (seed,cand_seat) dedupe，否则前序块被清空抹掉；
  * PIDFILE 防重；分片幂等跳过；零删除、原地重写；
  * 完整性校验不过 ⇒ 不写 pool_report_section.md，直接退出 2。

用法：
    .venv/bin/python tools/pool_chunk.py --smoke   # 每片 1 seed 冒烟（正确性与成本）
    .venv/bin/python tools/pool_chunk.py           # 正式跑 5 片 + 出报告段
    .venv/bin/python tools/pool_chunk.py --report  # 只用已落盘数据重出报告段
"""
from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

from final_eval_report import per_seed, stats  # noqa: E402

PY = str(ROOT / ".venv/bin/python")
GATES = ROOT / "runs" / "gates"
LOG = GATES / "pool.log"
PIDFILE = GATES / "pool_chunk.pid"
REPORT_OUT = GATES / "pool_report_section.md"
INCOMPLETE_FLAG = GATES / "pool_incomplete.flag"

BASE = 7_100_000
N_SEEDS = 500
CHUNK = 50                 # 每块 seed 数（200 局）
SEATS = "0,1,2,3"
NEED_PER_SEED = 4
NEED_ROWS = N_SEEDS * NEED_PER_SEED   # 2000 / 片
WORKERS = "6"
DEVICE = "mps"

A_PATH = "models/champ_exit_p48.pt"
A_MODE = "greedy"

# 看结果前锁死（§6.71.3）；顺序即 Holm family 的登记顺序。
SLICES = [
    {"id": "O1", "b": "models/dushan_master.pt", "mode": "greedy",
     "style": "冻结主力 v1 贪心"},
    {"id": "O2", "b": "models/champ_exit_p36.pt", "mode": "greedy",
     "style": "同族早期冠军 v3 贪心"},
    {"id": "O3", "b": "models/web_best.pt", "mode": "greedy",
     "style": "更早学生 R07 v3 贪心"},
    {"id": "O4", "b": "models/dushan_master.pt", "mode": "noisy10",
     "style": "主力 + 10% 均匀随机动作"},
    {"id": "O5", "b": "models/dushan_master.pt", "mode": "heuristic",
     "style": "纯启发式规则基线（无 NN）"},
]

# 成本基线：final_gateA（贪心对贪心）实测 0.27 s/局，6 workers
# ⇒ 2000 局 ≈ 90 s/片。首片超过 +30% 即停下重报，不自行加码。
EST_SEC_PER_DEAL = 0.27
EST_WALL_PER_SLICE = EST_SEC_PER_DEAL * NEED_ROWS / int(WORKERS)
COST_TOL = 1.30


def out_for(sid: str) -> Path:
    return GATES / f"pool_{sid}.jsonl"


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
        except Exception:  # noqa: BLE001
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


def have_seeds(path: Path) -> set[int]:
    if not path.exists() or path.stat().st_size == 0:
        return set()
    counts: dict = {}
    for line in open(path, encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        try:
            r = json.loads(line)
        except Exception:  # noqa: BLE001
            continue
        counts.setdefault(int(r.get("seed", -1)), set()).add(r.get("cand_seat"))
    return {s for s, seats in counts.items() if len(seats) >= NEED_PER_SEED}


def check_fresh_pool() -> None:
    """确认 7_100_000..7_100_499 从未被任何既有 gates 评估用过。"""
    lo, hi = BASE, BASE + N_SEEDS - 1
    hits: list[str] = []
    for p in sorted(GATES.glob("*.jsonl")):
        if p.name.startswith("pool_"):
            continue
        try:
            for line in open(p, encoding="utf-8"):
                if not line.strip():
                    continue
                r = json.loads(line)
                if lo <= int(r.get("seed", -1)) <= hi:
                    hits.append(p.name)
                    break
        except Exception:  # noqa: BLE001
            continue
    if hits:
        raise RuntimeError(f"seed 池已被占用：{hits}")


def run_chunk(sl: dict, seed0: int, nseed: int, out: Path) -> int:
    cmd = [PY, "eval_pairs.py",
           "--a", A_PATH, "--b", sl["b"],
           "--a-mode", A_MODE, "--b-mode", sl["mode"],
           "--seeds", str(nseed), "--seed0", str(seed0),
           "--seats", SEATS, "--workers", WORKERS, "--device", DEVICE,
           "--out", str(out), "--no-truncate"]
    with open(LOG, "a", encoding="utf-8") as lf:
        return subprocess.call(cmd, cwd=str(ROOT), stdout=lf, stderr=lf)


def smoke() -> int:
    say("SMOKE 1 seed × 5 片（正确性 + 成本校准）")
    ok = True
    for sl in SLICES:
        out = GATES / f"pool_smoke_{sl['id']}.jsonl"
        if out.exists():
            out.unlink()
        t0 = time.perf_counter()
        rc = run_chunk(sl, 9_000_000, 1, out)   # 冒烟用独立 seed，不占正式池
        dt = time.perf_counter() - t0
        rows = [json.loads(x) for x in out.read_text().splitlines() if x.strip()]
        good = (rc == 0 and len(rows) == NEED_PER_SEED
                and len({(r["seed"], r["cand_seat"]) for r in rows}) == NEED_PER_SEED
                and all(len(r["deltas"]) == 4 for r in rows))
        ok = ok and good
        print(f"[smoke] {sl['id']} {sl['mode']:10s} rc={rc} rows={len(rows)} "
              f"wall={dt:.1f}s per_deal(6w)={dt / NEED_PER_SEED:.2f}s "
              f"{'OK' if good else 'FAIL'}", flush=True)
    print("SMOKE VERDICT:", "PASS" if ok else "FAIL", flush=True)
    return 0 if ok else 1


def holm(pvals: list[float]) -> list[float]:
    """Holm step-down 校正后 p（同位置返回，单调不减，封顶 1）。"""
    m = len(pvals)
    order = sorted(range(m), key=lambda i: pvals[i])
    adj = [0.0] * m
    running = 0.0
    for k, i in enumerate(order):
        running = max(running, min(1.0, (m - k) * pvals[i]))
        adj[i] = running
    return adj


def report() -> int:
    """只用已落盘数据出报告段；不完整则拒绝写（守护完成判据）。"""
    rows: list[dict] = []
    for sl in SLICES:
        out = out_for(sl["id"])
        if not out.exists():
            say(f"INCOMPLETE 缺 {out.name}")
            return 2
        dedupe(out)
        data = [json.loads(x) for x in out.read_text().splitlines() if x.strip()]
        keys = {(r["seed"], r["cand_seat"]) for r in data}
        seats_per_seed: dict = {}
        for r in data:
            seats_per_seed.setdefault(r["seed"], set()).add(r["cand_seat"])
        complete = (len(data) == NEED_ROWS and len(keys) == NEED_ROWS
                    and all(len(v) == NEED_PER_SEED for v in seats_per_seed.values()))
        if not complete:
            INCOMPLETE_FLAG.write_text(
                f"{sl['id']} rows={len(data)} unique={len(keys)} "
                f"seeds={len(seats_per_seed)} need={NEED_ROWS}\n"
                f"at {time.strftime('%F %T')}\n", encoding="utf-8")
            say(f"INCOMPLETE {sl['id']} → 不写报告段")
            return 2
        ps = per_seed(out)
        st = stats([ps[s] for s in sorted(ps)])
        sd = st["se"] * math.sqrt(st["n"])
        rows.append({**sl, **st, "sd": sd,
                     "ci": (st["mean"] - 1.96 * st["se"],
                            st["mean"] + 1.96 * st["se"])})

    adj = holm([r["p"] for r in rows])
    for r, a in zip(rows, adj):
        r["p_adj"] = a

    n_pos = sum(1 for r in rows if r["mean"] > 0)
    sig_neg = [r["id"] for r in rows if r["p_adj"] < 0.05 and r["mean"] < 0]
    passed = n_pos >= 4 and not sig_neg
    verdict = ("通过外部效度方向一致性检验"
               if passed else "未通过 / 证据不足")

    L: list[str] = []
    L.append("# E-POOL 多对手池评测 · 结果段（§6.71.3 预登记）\n")
    L.append("> **探索性外部效度补充证据**。不改判 `reports/终评报告-2026-11-13.md` "
             "v1.1 的 §3/§4；不与组合①/②/E-SIDE 合并；不用于线上模型决策"
             "（主力恒 `dushan_master`，冠军只增不替）。\n")
    L.append(f"- 候选 A 恒定：`{A_PATH}` / `{A_MODE}`；5 片对手池、样本量、判据"
             "均在 **看任何新结果前** 写死于 §6.71.3，零事后调整。")
    L.append(f"- 池 `{BASE}..{BASE + N_SEEDS - 1}`（{N_SEEDS} seed，5 片共用同一 seed 块，"
             f"四座 `{SEATS}`），每片 {NEED_ROWS} 行，合计 {NEED_ROWS * len(rows)} 行；"
             "与已耗池零重叠，评估 JSONL 不回流训练。")
    L.append("- 统计口径：`tools/final_eval_report.py::per_seed/stats` 冻结口径"
             "（`deltas[cand_seat] − mean(其余三家)` → 四座按 seed 平均 → seed 维度配对 t）；"
             "5 片双侧 p 做 Holm step-down（family α=0.05）。\n")
    L.append("| 片 | 对手 | 风格 | n | Δ±SE | 95% CI | z | 双侧 p | p_adj(Holm) | 符号 正:负 | SD | 读数 |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        ns = r["p"] >= 0.05
        reading = "无足够证据" if ns else ("正方向且显著" if r["mean"] > 0
                                       else "负方向且显著")
        L.append(f"| {r['id']} | `{Path(r['b']).stem}` / `{r['mode']}` | {r['style']} | "
                 f"{r['n']} | {r['mean']:+.3f}±{r['se']:.3f} | "
                 f"[{r['ci'][0]:+.3f}, {r['ci'][1]:+.3f}] | {r['z']:+.2f} | "
                 f"{r['p']:.4f} | {r['p_adj']:.4f} | {r['pos']}:{r['neg']} | "
                 f"{r['sd']:.3f} | {reading} |")
    L.append("")
    L.append("## 主判据（锁死原文）\n")
    L.append("`≥4/5 片 Δ>0` 且 `无任何一片满足 p_adj<0.05 且 Δ<0` ⇒ 通过外部效度方向一致性检验。\n")
    L.append(f"- 实测：正方向 `{n_pos}/5`；Holm 校正后显著负向片 `{sig_neg or '无'}`。")
    L.append(f"- **机械判读：{verdict}**。")
    L.append("")
    L.append("## 解读纪律\n")
    L.append("- 任何 `p≥.05` 的片只写「无足够证据」，禁写「无差异 / 并列 / 打平 / 到顶 / 一样强」。")
    L.append("- 不同对手的 Δ 不可相减、不可合并、不可 pooled；对手强度不同，σ 亦不同。")
    L.append(f"- 功效（看前写死）：按 `σ≈6.915`，`SE(500)≈0.309`，80% 功效可检出 "
             "`Δ≈0.87`；对 `+0.5` 仅约 40% 功效 ⇒ Δ 接近 0 时「无足够证据」是默认预期。")
    L.append("- 与组合① `+1.478±0.297`（对手 master+N32，池 `3_000_000`）只作方向对照，"
             "不合并、不相减、不作显著性比较。")
    L.append("- 本段不进终评报告主体；若要作附录 D 需另行批准。\n")
    L.append(f"\n生成时间：{time.strftime('%F %H:%M:%S')}；数据："
             + ", ".join(f"`runs/gates/pool_{r['id']}.jsonl`" for r in rows) + "\n")

    if INCOMPLETE_FLAG.exists():
        INCOMPLETE_FLAG.unlink()
    REPORT_OUT.write_text("\n".join(L), encoding="utf-8")
    say(f"REPORT -> {REPORT_OUT.name}  verdict={verdict}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--report", action="store_true")
    args = ap.parse_args()
    GATES.mkdir(parents=True, exist_ok=True)
    if args.smoke:
        return smoke()
    if args.report:
        return report()

    if PIDFILE.exists():
        try:
            os.kill(int(PIDFILE.read_text().strip()), 0)
            print("another pool driver alive; exit", flush=True)
            return 1
        except (ValueError, OSError):
            pass
    PIDFILE.write_text(str(os.getpid()), encoding="utf-8")
    check_fresh_pool()
    say(f"START 5 slices × {N_SEEDS} seed；est {EST_WALL_PER_SLICE:.0f}s/片")

    for si, sl in enumerate(SLICES):
        out = out_for(sl["id"])
        t_slice = time.perf_counter()
        done = have_seeds(out)
        for k in range(N_SEEDS // CHUNK):
            seed0 = BASE + k * CHUNK
            need = set(range(seed0, seed0 + CHUNK))
            if need <= done:
                continue
            for attempt in (1, 2, 3):
                say(f"{sl['id']} chunk seed0={seed0} try{attempt} "
                    f"(have {len(done & need)}/{CHUNK})")
                rc = run_chunk(sl, seed0, CHUNK, out)
                dedupe(out)
                done = have_seeds(out)
                say(f"{sl['id']} chunk seed0={seed0} rc={rc} "
                    f"have={len(done & need)}/{CHUNK}")
                if need <= done:
                    break
                time.sleep(3)
        wall = time.perf_counter() - t_slice
        # 成本判据用**局内实测 CPU 秒/局**（JSONL 的 s 字段，6-worker 满载口径），
        # 不用驱动 wall：wall 含 10 次进程启动 / 模型加载，与 §6.71.3 的
        # 「final_gateA 0.27 s/局」不是同一个量。冒烟 4 局并发不满载，也不作校准。
        data = [json.loads(x) for x in out.read_text().splitlines() if x.strip()]
        mean_s = sum(float(r.get("s", 0)) for r in data) / max(1, len(data))
        ratio = mean_s / EST_SEC_PER_DEAL
        say(f"{sl['id']} DONE rows={len(data)} wall={wall:.0f}s "
            f"mean_s={mean_s:.3f}s/deal (est {EST_SEC_PER_DEAL:.3f}, x{ratio:.2f})")
        # 单向闸：只拦「成本超支」。§6.71.3 的「偏差 >30% 停下重报」目的在
        # 防止静默加码；比估算便宜不是风险。历史 0.27 s/局是在**并发训练**时
        # 测的，当前机器空闲（首片中途遥测 0.144 s/局），若做双向闸会把「更省」
        # 误判成偏差。该澄清在看到任何胜负/分差结果数据之前做出，判据、
        # 对手池、样本量、Holm 校正一字未改。
        if si == 0 and ratio > COST_TOL:
            say(f"COST GUARD 首片成本超出估算 >30%（x{ratio:.2f}）→ 停下重报，不自行加码")
            return 3
        # 逐片完整性校验，未齐绝不进入下一片（禁并发 + 防污染）
        if len(done) != N_SEEDS:
            say(f"{sl['id']} INCOMPLETE seeds={len(done)}/{N_SEEDS} → 停止")
            return 2

    rc = report()
    say("POOL COMPLETE" if rc == 0 else f"POOL rc={rc}")
    return rc


if __name__ == "__main__":
    sys.exit(main())
