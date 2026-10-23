"""M2-0 · T0.3 单轮耗时剖析（剖析 collect / update / eval / pool-rebuild 四段归属）。

为什么需要它
------------
S3 段实测 **65.6 s/轮**，其中采集 ~10 s、更新 ~33 s、剩下的 ~22 s 一直叫"固定开销"
——没有任何人知道这 22 s 到底花在哪，所以 M2-A 的吞吐修复（A7）此前只是在猜。
本脚本把"固定开销"拆开归因，并给出修复后的可期望值。

两个数据源（互补，互不依赖）
----------------------------
* **A · 历史反演（零算力）**：训练进度行已经写了 ``collect=`` 与累计 ``total=``，
  日志里还有 ``[pool] iter N 重建进程池`` / ``[eval]`` / ``[snap]`` 事件行。
  于是可以逐轮算 ``dt_total = total_i − total_{i−1}``，再按"这一轮有没有 eval /
  有没有 pool 重建"分组做差 —— 差值就是该段的**真实成本**。样本量 850 轮，
  统计上远比单点测量可靠。
* **B · 实测微基准（``--micro``）**：加载真实检查点，用真实 batch/线程数测
  M1 的 forward、forward+backward、以及 master 前向（KLT 蒸馏要付的那部分），
  再用真实样本数/批次数换算成"单轮 update 预期秒数"，与 A 的实测量互证。
  顺带测纯引擎对局与 M1 对局，拆出采集里"引擎 vs 网络"的比例。

判读口径提醒：本脚本只产出**耗时**结论，不产出强度结论。

用法::

    python scripts/iter_profile.py \
        --log reports/_m1_s3.out --history reports/m1_s3_history.json

    # 加实测微基准（约 1–3 分钟，24 线程）
    python scripts/iter_profile.py --log ... --history ... --micro \
        --ckpt dushan_m1_s3 --threads 24
"""
from __future__ import annotations

import argparse
import json
import re
import statistics as st
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

#: 进度行：``iter  12  samples=... collect=10.1s total=63s``
RE_ITER = re.compile(r"^iter\s+(\d+)\s")
RE_POOL = re.compile(r"^\[pool\]\s+iter\s+(\d+)\s+重建进程池[：:](.*)$")
RE_EVAL = re.compile(r"^\s*\[eval\]\s*\{")
RE_SNAP = re.compile(r"^\s*\[snap\]\s")


# ---------------------------------------------------------------------------
# A. 历史反演
# ---------------------------------------------------------------------------
def parse_log(path: Path) -> dict[int, dict]:
    """解析训练日志，返回 ``{轮号: {"eval":bool,"snap":bool,"pool":str|None}}``。

    对齐规则：``[pool]`` / ``[eval]`` / ``[snap]`` 三种事件行都**先于**它所属那一轮的
    进度行出现（pool 在循环头、eval/snap 在进度行打印之前），所以统一挂到
    "其后第一条进度行"的轮号上。脚本会用两条已知事实自检这个对齐：
    eval 轮号必须全是 ``eval_every`` 的倍数、snap 轮号也必须是 —— 不满足就整体偏移 1。
    """
    events: dict[int, dict] = {}
    pending: list[tuple[str, str]] = []
    text = Path(path).read_text(encoding="utf-8", errors="replace")
    for line in text.splitlines():
        m = RE_ITER.match(line)
        if m:
            it = int(m.group(1))
            d = events.setdefault(it, {"eval": False, "snap": False, "pool": None})
            for kind, why in pending:
                if kind == "pool":
                    d["pool"] = why
                elif kind == "eval":
                    d["eval"] = True
                elif kind == "snap":
                    d["snap"] = True
            pending = []
            continue
        mp = RE_POOL.match(line)
        if mp:
            pending.append(("pool", mp.group(2).strip()))
            continue
        if RE_EVAL.match(line):
            pending.append(("eval", ""))
            continue
        if RE_SNAP.match(line):
            pending.append(("snap", ""))
            continue
    return events


def _align_events(events: dict[int, dict], eval_every: int, snap_every: int) -> tuple[dict, int]:
    """自检事件轮号的对齐；返回 ``(修正后事件表, 采用的偏移)``。

    偏移 0 = 直接挂到"下一条进度行"；偏移 1 = 再往后挪一轮。
    """
    def score(off: int) -> int:
        ok = 0
        for it, d in events.items():
            if d["eval"] and (it + off) % eval_every == 0:
                ok += 1
            if d["snap"] and (it + off) % snap_every == 0:
                ok += 1
        return ok

    n_eval = sum(1 for d in events.values() if d["eval"])
    n_snap = sum(1 for d in events.values() if d["snap"])
    s0, s1 = score(0), score(1)
    off = 0 if s0 >= s1 else 1
    if off:
        events = {it + 1: d for it, d in events.items()}
    return events, off, n_eval, n_snap, (s0, s1)


def _agg(xs: list[float]) -> dict:
    if not xs:
        return {"n": 0, "mean": 0.0, "median": 0.0, "sd": 0.0}
    return {
        "n": len(xs),
        "mean": round(st.mean(xs), 3),
        "median": round(st.median(xs), 3),
        "sd": round(st.stdev(xs), 3) if len(xs) > 1 else 0.0,
    }


def analyze_history(history_path: Path, events: dict[int, dict]) -> dict:
    """逐轮算 total / collect / non-collect，并按事件类型分组归因。

    **成本归属必须错开一轮**（本脚本踩到的坑，值得记一笔）：
    ``rec["elapsed"]`` 是在 ``run_eval()`` / ``[snap]`` **之前**采样的
    （selfplay.py 里 rec 先构造、eval 后执行），所以第 N 轮的 eval/snap 耗时
    落在 ``dt_total(N+1)`` 里；而 ``[pool]`` 重建发生在循环头、采集之前，
    落在 ``dt_total(N+1)``（``[pool] iter N`` 打印的是 it，进度行是 it+1）。
    于是：pool 用 ``events[it]``，eval/snap 用 ``events[it-1]``。
    """
    rows = json.loads(Path(history_path).read_text(encoding="utf-8"))
    if not isinstance(rows, list) or not rows:
        raise SystemExit(f"{history_path} 不是非空列表")

    per: list[dict] = []
    prev_total = 0.0
    for rec in rows:
        it = int(rec["iter"])
        total = float(rec["elapsed"])
        dt = total - prev_total
        prev_total = total
        collect = float(rec.get("collect_s") or 0.0)
        ev_now = events.get(it, {"eval": False, "snap": False, "pool": None})
        ev_prev = events.get(it - 1, {"eval": False, "snap": False, "pool": None})
        per.append({
            "iter": it,
            "dt_total": round(dt, 3),
            "collect": round(collect, 3),
            "non_collect": round(dt - collect, 3),
            "eval": bool(ev_prev["eval"]),      # 上一轮的 eval 成本落在本轮
            "snap": bool(ev_prev["snap"]),      # 上一轮的 snap 同理
            "pool": ev_now["pool"],             # 本轮的 pool 重建落在本轮
            "samples": int(rec.get("samples") or 0),
        })

    # 丢掉极端异常点（例如重启/挂起造成的空洞）——用中位数的 3 倍做鲁棒截断
    dts = [p["dt_total"] for p in per]
    med = st.median(dts)
    clean = [p for p in per if p["dt_total"] < max(3.0 * med, med + 300.0)]
    dropped = len(per) - len(clean)

    def pick(eval_f: bool, pool_f: bool) -> list[float]:
        return [p["non_collect"] for p in clean
                if p["eval"] == eval_f and (p["pool"] is not None) == pool_f]

    g = {
        "plain": _agg(pick(False, False)),
        "pool_only": _agg(pick(False, True)),
        "eval_only": _agg(pick(True, False)),
        "both": _agg(pick(True, True)),
    }
    base = g["plain"]["median"]
    cost = {
        "collect_median": round(st.median([p["collect"] for p in clean]), 3),
        "non_collect_plain": base,
        "pool_rebuild": round(g["pool_only"]["median"] - base, 3),
        "eval_40deals": round(g["eval_only"]["median"] - base, 3),
        "pool_plus_eval_check": round(
            g["both"]["median"] - base - (g["pool_only"]["median"] - base)
            - (g["eval_only"]["median"] - base), 3) if g["both"]["n"] and g["eval_only"]["n"] else None,
    }
    # 全段平均（按事件实际频率加权）——这才是"每轮固定开销 22 s"的正确拆法
    n = len(clean)
    share_pool = sum(1 for p in clean if p["pool"] is not None) / max(1, n)
    share_eval = sum(1 for p in clean if p["eval"]) / max(1, n)
    avg = {
        "dt_total": round(st.mean([p["dt_total"] for p in clean]), 3),
        "collect": round(st.mean([p["collect"] for p in clean]), 3),
        "non_collect": round(st.mean([p["non_collect"] for p in clean]), 3),
        "pool_share": round(share_pool, 4),
        "eval_share": round(share_eval, 4),
        "pool_avg_contrib": round(share_pool * max(0.0, cost["pool_rebuild"]), 3),
        "eval_avg_contrib": round(share_eval * max(0.0, cost["eval_40deals"]), 3),
    }
    avg["update_plus_rest"] = round(
        avg["non_collect"] - avg["pool_avg_contrib"] - avg["eval_avg_contrib"], 3)

    # ---- 最小二乘分解（主口径）-------------------------------------------
    # dt_total ~ β0 + β1·collect + β2·1[pool] + β3·1[eval(上一轮)] + β4·1[snap(上一轮)]
    # 850 个样本、4 个自变量、且共线性可控（pool 触发率 80% 但 eval 只有 5%），
    # 系数就是各段的**边际成本**，比"分组中位数做差"稳健得多。
    ols = None
    try:
        import numpy as np
        y = np.asarray([p["dt_total"] for p in clean], dtype=float)
        e_vec = np.asarray([1.0 if p["eval"] else 0.0 for p in clean])
        s_vec = np.asarray([1.0 if p["snap"] else 0.0 for p in clean])
        # eval-every 与 save-every 相同时两个指示向量**逐位相同**，设计矩阵列线性相关，
        # lstsq 的最小范数解会把效应平分到两列（实测两列系数都是 33.07 s），
        # 于是"eval 0.5 s、snap 66 s"这种读法是错的。合并成一列才是可识别的。
        same_freq = bool(np.array_equal(e_vec, s_vec))
        cols = [np.ones(len(clean)),
                np.asarray([p["collect"] for p in clean], dtype=float),
                np.asarray([1.0 if p["pool"] else 0.0 for p in clean])]
        names = ["const_s", "per_collect_s_slope", "pool_s"]
        if same_freq:
            cols.append(np.maximum(e_vec, s_vec))
            names.append("eval_or_snap_s")
        else:
            cols += [e_vec, s_vec]
            names += ["eval_s", "snap_s"]
        X = np.column_stack(cols)
        coef, *_ = np.linalg.lstsq(X, y, rcond=None)
        pred = X @ coef
        ss_res = float(((y - pred) ** 2).sum())
        ss_tot = float(((y - y.mean()) ** 2).sum())
        ols = {n: round(float(c), 3) for n, c in zip(names, coef)}
        ols["eval_snap_same_freq"] = same_freq
        ols["r2"] = round(1.0 - ss_res / ss_tot, 4) if ss_tot else None
        share_ev = (max(float(e_vec.mean()), float(s_vec.mean())) if same_freq
                    else float(e_vec.mean()) + float(s_vec.mean()))
        ols["fixed_avg_s"] = round(
            float(coef[0]) + share_pool * float(coef[2]) + share_ev * ols[
                "eval_or_snap_s" if same_freq else "eval_s"], 3)
        if not same_freq:
            ols["fixed_avg_s"] = round(
                ols["fixed_avg_s"] + share_ev * float(ols["snap_s"]) / 2, 3)
    except Exception as e:                                     # noqa: BLE001
        ols = {"error": str(e)}

    # collect 随轮次的走势（内存压力下是否劣化）
    def band(a: int, b: int) -> dict:
        xs = [p for p in clean if a <= p["iter"] <= b]
        return {"iters": f"{a}-{b}", "n": len(xs),
                "dt_total": _agg([p["dt_total"] for p in xs])["median"],
                "collect": _agg([p["collect"] for p in xs])["median"],
                "non_collect": _agg([p["non_collect"] for p in xs])["median"]}

    trend = [band(1, 50), band(400, 450), band(800, 850)]

    # pool 重建的触发原因分布
    whys: dict[str, int] = {}
    for p in clean:
        if p["pool"]:
            key = "系统内存阈值" if "系统内存" in p["pool"] else (
                "轮数兜底" if "轮数兜底" in p["pool"] else p["pool"][:24])
            whys[key] = whys.get(key, 0) + 1

    return {
        "history": str(history_path),
        "iters": len(rows),
        "dropped_outliers": dropped,
        "groups": g,
        "cost_estimate": cost,
        "ols": ols,
        "per_iter_average": avg,
        "trend": trend,
        "pool_reasons": whys,
        "per_iter": per,
    }


# ---------------------------------------------------------------------------
# B. 实测微基准
# ---------------------------------------------------------------------------
def _load_model(stem: str):
    import torch
    from zhuoji.net import build_from_ckpt
    ck = torch.load(ROOT / "models" / f"{stem}.pt", map_location="cpu", weights_only=False)
    m, ver = build_from_ckpt(ck)
    return m.eval(), ver


def _mk_batch(batch: int, in_ch: int, glob_dim: int):
    import torch
    return torch.randn(batch, in_ch, 27), torch.randn(batch, glob_dim)


def _step_loss(out):
    loss = out["value"].mean()
    for k in ("discard", "self_kong", "respond"):
        loss = loss + out[k].mean()
    return loss


def bench_net(model, batch: int, reps: int, threads: int, tag: str) -> dict:
    """测单步 forward 与 forward+backward+AdamW 的中位耗时（毫秒）。"""
    import torch
    torch.set_num_threads(threads)
    in_ch = int(getattr(model, "in_channels"))
    glob_dim = int(getattr(model, "global_dim"))
    tiles, gl = _mk_batch(batch, in_ch, glob_dim)
    out = model(tiles, gl)
    out["discard"].sum().backward()
    model.zero_grad(set_to_none=True)
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-5)

    fwd_ts, bwd_ts = [], []
    for _ in range(reps):
        with torch.no_grad():
            t0 = time.perf_counter()
            model(tiles, gl)
            fwd_ts.append((time.perf_counter() - t0) * 1e3)
        t0 = time.perf_counter()
        out = model(tiles, gl)
        loss = _step_loss(out)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()
        bwd_ts.append((time.perf_counter() - t0) * 1e3)
    model.eval()
    return {
        "tag": tag,
        "batch": batch,
        "threads": threads,
        "params": int(sum(p.numel() for p in model.parameters())),
        "fwd_ms_median": round(st.median(fwd_ts), 2),
        "fwdwd_ms_median": round(st.median(bwd_ts), 2),
    }


def bench_engine_game(n: int, seed: int, with_net, model=None) -> dict:
    """单局 wall time：纯引擎（4×启发式）vs 引擎+网络（4×M1）。

    单线程跑，用来拆"采集里引擎占多少"。注意这是**单进程串行**口径，
    与 selfplay 里 12 个 worker 并行的 10 s/轮不可直接相除，
    但两者的**比值**（引擎 : 网络）是可比的结构性信息。
    """
    import torch
    from zhuoji.agent import NetAgent
    from zhuoji.bots import HeuristicBot
    from zhuoji.dushan import DushanConfig, DushanGame

    torch.set_num_threads(1)
    ts = []
    obs_n = 0
    for i in range(n):
        g = DushanGame(DushanConfig(), seed=seed + i * 1013)
        if with_net:
            agents = [NetAgent(model, seed=i * 4 + k, temperature=0.0, name="m1")
                      for k in range(4)]
        else:
            agents = [HeuristicBot(seed=i * 4 + k) for k in range(4)]
        t0 = time.perf_counter()
        g.play([(lambda a: (lambda gm: a.choose(gm)))(a) for a in agents])
        ts.append(time.perf_counter() - t0)
        obs_n += len(g.discards[0]) + len(g.discards[1]) + len(g.discards[2]) + len(g.discards[3])
    return {
        "n_games": n,
        "median_s_per_game": round(st.median(ts), 3),
        "mean_s_per_game": round(st.mean(ts), 3),
    }


def analyze_micro(args, hist: dict) -> dict:
    m1, ver = _load_model(args.ckpt)
    batch = args.batch
    reps = args.reps
    m1_bench = bench_net(m1, batch, reps, args.threads, args.ckpt)

    ref_bench = None
    m_bench = None
    try:
        owner, _ = _load_model(args.ref)
        ref_bench = bench_net(owner, batch, reps, args.threads, args.ref)
    except SystemExit:
        pass
    if args.klt_ckpt:
        try:
            small, _ = _load_model(args.klt_ckpt)
            m_bench = bench_net(small, batch, reps, args.threads, args.klt_ckpt)
        except SystemExit:
            pass

    # 用真实样本数/批次数换算单轮 update
    avg = hist.get("per_iter_average", {})
    samples = int(st.median([p["samples"] for p in hist.get("per_iter", [])
                             if p["samples"] > 0] or [0])) or args.samples
    nb = max(1, -(-samples // batch)) * args.epochs
    pred = {
        "samples_per_iter": samples,
        "batches_per_iter": nb,
        "update_predicted_s": round(nb * m1_bench["fwdwd_ms_median"] / 1e3, 2),
        # 优势估计的前向：selfplay.update() 按 4096 一块跑 no_grad 前向，
        # 折成 batch 等效数 = ceil(n/4096) × (4096/batch)
        "adv_batches_equiv": (-(-samples // 4096)) * (4096 / batch),
        "adv_forward_predicted_s": round(
            (-(-samples // 4096)) * (4096 / batch) * m1_bench["fwd_ms_median"] / 1e3, 2),
    }
    if ref_bench is not None:
        pred["ref_forward_predicted_s"] = round(
            nb * ref_bench["fwd_ms_median"] / 1e3, 2)
        pred["update_with_ref_predicted_s"] = round(
            pred["update_predicted_s"] + pred["ref_forward_predicted_s"], 2)
    if m_bench is not None:
        pred["klt_forward_predicted_s"] = round(nb * m_bench["fwd_ms_median"] / 1e3, 3)
        pred["klt_overhead_pct"] = round(
            100.0 * m_bench["fwd_ms_median"] / m1_bench["fwdwd_ms_median"], 2)

    engine = None
    if not args.no_engine_bench:
        engine = {
            "teacher_only": bench_engine_game(args.engine_games, 777, False),
            "m1_all_seats": bench_engine_game(args.engine_games, 777, True, m1),
        }
        t = engine["teacher_only"]["median_s_per_game"]
        n = engine["m1_all_seats"]["median_s_per_game"]
        engine["engine_share"] = round(t / n, 4) if n else None

    return {
        "ckpt": args.ckpt,
        "encoder_version": ver,
        "m1_bench": m1_bench,
        "ref_bench": ref_bench,
        "klt_bench": m_bench,
        "predict": pred,
        "game_bench": engine,
        "measured_update_s": avg.get("update_plus_rest"),
    }


# ---------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", default=str(ROOT / "reports" / "_m1_s3.out"))
    ap.add_argument("--history", default=str(ROOT / "reports" / "m1_s3_history.json"))
    ap.add_argument("--eval-every", type=int, default=20)
    ap.add_argument("--snap-every", type=int, default=20)
    ap.add_argument("--out", default=str(ROOT / "reports" / "_iter_profile.json"))
    ap.add_argument("--micro", action="store_true", help="跑实测微基准（加载检查点）")
    ap.add_argument("--ckpt", default="dushan_m1_s3")
    ap.add_argument("--ref", default="dushan_m1_s2",
                    help="KL 锚点模型（update 里 ref_model 的整批前向成本来源）")
    ap.add_argument("--klt-ckpt", default="dushan_master",
                    help="KLT 蒸馏的教师（0.77M），用于估算 +x%% 的蒸馏开销")
    ap.add_argument("--threads", type=int, default=24)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--reps", type=int, default=8)
    ap.add_argument("--samples", type=int, default=11000)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--engine-games", type=int, default=3)
    ap.add_argument("--no-engine-bench", action="store_true")
    args = ap.parse_args()

    log_p = Path(args.log)
    hist_p = Path(args.history)
    if not log_p.exists():
        raise SystemExit(f"找不到日志 {log_p}")
    if not hist_p.exists():
        raise SystemExit(f"找不到 history {hist_p}")

    raw = parse_log(log_p)
    events, off, n_eval, n_snap, sc = _align_events(raw, args.eval_every, args.snap_every)
    print(f"[A] 解析 {log_p.name}：事件轮 {len(raw)} 个（eval {n_eval} / snap {n_snap}），"
          f"对齐偏移 = {off}（自检得分 {sc[0]} vs {sc[1]}）", flush=True)

    hist = analyze_history(hist_p, events)
    avg, cost, ols = hist["per_iter_average"], hist["cost_estimate"], hist["ols"]
    total = avg["dt_total"]
    print("\n【A · 历史反演】逐轮总耗时 {:.1f} s = 采集 {:.1f} s + 非采集 {:.1f} s"
          .format(total, avg["collect"], avg["non_collect"]))
    if ols and "error" not in ols:
        print(f"  最小二乘分解（R²={ols['r2']}，n={len(hist['per_iter']) - hist['dropped_outliers']}）：")
        print(f"    常数/固定项（含 update）  {ols['const_s']:7.2f} s/轮")
        print(f"    采集斜率                  {ols['per_collect_s_slope']:7.2f}"
              f"（s 每 s 采集；≈1 = 采集耗时本身，无额外传导）")
        print(f"    进程池重建                {ols['pool_s']:7.2f} s/次"
              f"（触发率 {avg['pool_share'] * 100:.0f}%）")
        if ols.get("eval_snap_same_freq"):
            print(f"    eval(40副)+周期快照        {ols['eval_or_snap_s']:7.2f} s/次"
                  f"（eval-every == save-every，两列逐位同频 → 不可分离，已合并）")
        else:
            print(f"    eval 40 副                {ols['eval_s']:7.2f} s/次")
            print(f"    周期快照写盘              {ols['snap_s']:7.2f} s/次")
        print(f"    → 平均固定开销 ≈ {ols['fixed_avg_s']:.1f} s/轮")
    print(f"  （分组中位数交叉验证）非采集 {avg['non_collect']:.1f} s = "
          f"池 {avg['pool_avg_contrib']:.2f} + 评测 {avg['eval_avg_contrib']:.2f} + "
          f"其余 {avg['update_plus_rest']:.2f}")
    print(f"  pool 触发原因：{hist['pool_reasons']}")
    print("  分段趋势（中位）：")
    for b in hist["trend"]:
        print(f"    iter {b['iters']:<9} total={b['dt_total']:7.1f}  "
              f"collect={b['collect']:6.2f}  non_collect={b['non_collect']:6.2f}")

    out = {"A_history": hist}
    if args.micro:
        print("\n[B] 实测微基准（加载检查点，可能需要 1–3 分钟）…", flush=True)
        micro = analyze_micro(args, hist)
        out["B_micro"] = micro
        b = micro["m1_bench"]
        print(f"  {b['tag']}：{b['params']:,} 参数  batch={b['batch']}  "
              f"fwd={b['fwd_ms_median']:.0f} ms  fwd+bwd={b['fwdwd_ms_median']:.0f} ms"
              f"（{args.threads} 线程）")
        if micro["ref_bench"]:
            print(f"  {micro['ref_bench']['tag']}：fwd={micro['ref_bench']['fwd_ms_median']:.0f} ms")
        if micro["klt_bench"]:
            print(f"  {micro['klt_bench']['tag']}：fwd={micro['klt_bench']['fwd_ms_median']:.0f} ms")
        p = micro["predict"]
        print(f"  换算单轮 update：{p['batches_per_iter']} batch × {b['fwdwd_ms_median']:.0f} ms"
              f" = {p['update_predicted_s']:.1f} s（实测常数项 {micro['measured_update_s']} s）")
        print(f"  + 优势估计前向 {p['adv_batches_equiv']:.0f} batch 等效"
              f" = {p['adv_forward_predicted_s']:.1f} s")
        if "update_with_ref_predicted_s" in p:
            print(f"  + KL 锚点整批前向 = {p['ref_forward_predicted_s']:.1f} s"
                  f" → 三段合计 {p['update_predicted_s'] + p['adv_forward_predicted_s'] + p['ref_forward_predicted_s']:.1f} s")
        if "klt_forward_predicted_s" in p:
            print(f"  KLT 蒸馏的教师前向 ≈ {p['klt_forward_predicted_s']:.2f} s/轮"
                  f"（相对 fwd+bwd {p['klt_overhead_pct']:.1f}%）")
        if micro["game_bench"]:
            gb = micro["game_bench"]
            print(f"  单局（单线程）：纯引擎 {gb['teacher_only']['median_s_per_game']:.2f} s，"
                  f"M1 四座 {gb['m1_all_seats']['median_s_per_game']:.2f} s"
                  f" → 引擎占 {gb['engine_share'] * 100:.1f}%")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n→ {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
