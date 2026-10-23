"""M1 训练计划的实测测算（2026-10-12，32 GB / i7-14700K / 仅 AVX2）。

回答四个问题，全部用**实测**而不是估算：

1. M1 在多少线程下性价比最高？（train step / eval batch）
2. 采集端 worker 数取多少？（同时给出每 worker 的真实内存占用）
3. 真实一轮（collect + update）要多久？各占多少？
4. 据此，一夜（约 11.5 h）能跑多少轮、喂多少样本？

用法：
  python scripts/m1_plan_bench.py                 # 全部
  python scripts/m1_plan_bench.py --quick         # 跳过内存与真实一轮
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from zhuoji.dushan import DushanConfig, DushanGame  # noqa: E402
from zhuoji.encoder import dims_for_version  # noqa: E402
from zhuoji.net import NetConfig, build_model, ck_from_model, net_meta  # noqa: E402
from zhuoji.vcollect import collect_parallel  # noqa: E402

M1_CFG = dict(channels=160, blocks=20, hidden=512, global_dim=130,
              in_channels=63, arch="m1")
S11_CFG = dict(channels=64, blocks=8, hidden=320, global_dim=98,
               in_channels=31, arch="s")


def make(cfg: dict):
    return build_model(NetConfig(**cfg))


def time_train(model, batch, threads, rep=5, warm=2) -> float:
    torch.set_num_threads(threads)
    cin = model.in_channels
    gd = model.global_dim
    x = torch.randn(batch, cin, 3, 9)
    g = torch.randn(batch, gd)
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-6)

    def step():
        opt.zero_grad(set_to_none=True)
        loss = model(x, g)["value"].square().mean()
        loss.backward()
        opt.step()

    for _ in range(warm):
        step()
    t0 = time.perf_counter()
    for _ in range(rep):
        step()
    dt = (time.perf_counter() - t0) / rep
    model.eval()
    return dt


def time_eval(model, batch, threads, rep=8, warm=2) -> float:
    torch.set_num_threads(threads)
    x = torch.randn(batch, model.in_channels, 3, 9)
    g = torch.randn(batch, model.global_dim)
    model.eval()
    with torch.no_grad():
        for _ in range(warm):
            model(x, g)
        t0 = time.perf_counter()
        for _ in range(rep):
            model(x, g)
        dt = (time.perf_counter() - t0) / rep
    return dt


def rss_mb() -> float:
    try:
        import psutil
        p = psutil.Process(os.getpid())
        tot = p.memory_info().rss
        for c in p.children(recursive=True):
            try:
                tot += c.memory_info().rss
            except Exception:
                pass
        return tot / 1e6
    except Exception:
        return float("nan")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--sweep", default="8,12,16,20,24,28")
    args = ap.parse_args()

    rep: dict = {"machine": {}, "thread_sweep": [], "memory": {}, "iter": {}}
    try:
        import psutil
        vm = psutil.virtual_memory()
        rep["machine"] = {
            "cpu": "i7-14700K",
            "logical": os.cpu_count(),
            "physical": psutil.cpu_count(logical=False),
            "ram_total_gb": round(vm.total / 1e9, 1),
            "ram_avail_gb": round(vm.available / 1e9, 1),
            "torch": torch.__version__,
            "cpu_capability": torch.backends.cpu.get_cpu_capability(),
        }
    except Exception as e:
        rep["machine"] = {"error": str(e)}
    print("== 机器 ==")
    for k, v in rep["machine"].items():
        print(f"  {k}: {v}")

    m1 = make(M1_CFG)
    s11 = make(S11_CFG)
    print(f"\n== 参数量 ==\n  s11 = {s11.n_params() / 1e6:.3f}M\n"
          f"  M1  = {m1.n_params() / 1e6:.3f}M")

    print("\n== 线程扩展：M1 train(batch 512) / eval(batch 64) ==")
    print(f"  {'threads':>7} {'train ms':>9} {'样本/s':>8} {'eval64 ms':>10}")
    best = None
    for t in [int(x) for x in args.sweep.split(",")]:
        tt = time_train(m1, 512, t)
        te = time_eval(m1, 64, t)
        print(f"  {t:>7} {tt * 1000:>9.1f} {512 / tt:>8.1f} {te * 1000:>10.2f}")
        rep["thread_sweep"].append({"threads": t, "train_s": tt, "eval64_s": te,
                                    "sample_per_s": 512 / tt})
        if best is None or tt < best[1]:
            best = (t, tt)
    print(f"  → 最快训练线程数 = {best[0]}（{best[1] * 1000:.1f} ms/step）")
    rep["best_train_threads"] = best[0]

    if not args.quick:
        print("\n== 每 worker 内存（含主模型 + 8 份同级快照） ==")
        base = rss_mb()
        m = make(M1_CFG)
        snaps = [ck_from_model(make(M1_CFG), encoder_version=3) for _ in range(8)]
        after = rss_mb()
        n_nets = 9                     # 主模型 + 8 份快照
        per_net_mb = (after - base) / n_nets
        param_mb = m.n_params() * 4 / 1e6
        print(f"  主进程 RSS：建 M1 + 8 快照后 {after:.0f} MB（净增 {after - base:.0f} MB）")
        print(f"  单个 M1 网络权重 {param_mb:.1f} MB；实测每份净增 ≈ {per_net_mb:.1f} MB")
        print(f"  → 单 worker 粗估 ≈ 运行时 ~200 MB + 快照 {8 * param_mb:.0f} MB"
              f" + 主模型 {param_mb:.0f} MB ≈ {200 + 9 * param_mb:.0f} MB")
        for w in (8, 12, 16):
            est_gb = (200 + 9 * param_mb) * w / 1000
            print(f"     workers={w:>2} → 子进程合计 ≈ {est_gb:.1f} GB")
        rep["memory"] = {"main_rss_mb_with_snaps": round(after, 1),
                         "delta_mb": round(after - base, 1),
                         "per_net_mb": round(per_net_mb, 1),
                         "est_per_worker_mb": round(200 + 9 * param_mb, 0)}
        del m, snaps
        import gc
        gc.collect()

        print("\n== 真实一轮：collect + update（M1, 48 局, 12 workers, epochs 1） ==")
        nc = net_meta(NetConfig(**M1_CFG))
        torch.set_num_threads(best[0])
        t0 = time.time()
        traj, rewards, dt_collect, _aux = collect_parallel(
            m1.state_dict(), nc, 48, 777, 1.2, 8.0, 0.15, workers=12,
            past_prob=0.3, snapshots_sd=[], rules="dushan", want_traj=True,
            encoder_version=3, encoder_packs=None, snap_key=None)
        t_collect_wall = time.time() - t0
        print(f"  采集：worker 内耗时 {dt_collect:.2f}s，主进程墙钟 {t_collect_wall:.2f}s，"
              f"样本 {len(traj)}")

        # update：复用 selfplay 的实现，保证口径一致
        sys.path.insert(0, str(ROOT / "scripts"))
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "sp_selfplay", ROOT / "scripts" / "selfplay.py")
        sp = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(sp)
        head_arr = np.array([t[3] for t in traj], dtype=np.int64)
        act_arr = np.array([t[4] for t in traj], dtype=np.int64)

        class A:
            pass
        a = A()
        a.epochs_per_iter = 1
        a.batch = 512
        a.value_coef = 0.5
        a.ent_coef = 0.01
        a.kl_coef = 0.0

        opt = torch.optim.AdamW(m1.parameters(), lr=1.2e-5, weight_decay=1e-4)
        t1 = time.time()
        upd = sp.update(m1, None, opt, traj, rewards, a)
        t_update = time.time() - t1
        print(f"  更新：{t_update:.2f}s（{len(traj)} 样本 × 1 epoch, batch 512）")
        print(f"  熵 H={upd['entropy']:.3f}  V={upd['value_loss']:.4f}  "
              f"reward={upd['reward']:+.3f}")
        per_iter = t_collect_wall + t_update
        samples = 48 * (len(traj) / 48)
        print(f"  → 单轮 ≈ {per_iter:.2f}s（采集 {t_collect_wall:.2f} + 更新 {t_update:.2f}）")
        night_h = 11.5
        iters = int(night_h * 3600 / per_iter)
        print(f"  → 一夜 {night_h}h ≈ {iters} 轮，累计样本 ≈ {iters * len(traj) / 1e6:.1f}M")
        rep["iter"] = {"games": 48, "workers": 12, "epochs": 1,
                       "collect_wall_s": round(t_collect_wall, 2),
                       "update_s": round(t_update, 2),
                       "samples": len(traj),
                       "per_iter_s": round(per_iter, 2),
                       "night_iters": iters,
                       "night_samples_m": round(iters * len(traj) / 1e6, 2)}

    out = ROOT / "reports" / "m1_plan_bench.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rep, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n结果 -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
