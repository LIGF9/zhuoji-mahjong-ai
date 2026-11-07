#!/usr/bin/env python3
"""E1 严格配对对照：冻结密集信号源（Suphx GRP 的项目化最小实现）vs Φ=8·V_θ。

与 7×24 主链的关系（**这是全部设计的前提**）
--------------------------------------------
§6.36 已判 P20−R12 ⇒ 分支②「同配方第二周期，一参数不改」。因此本实验
**绝不改 train_ppo.py / night_run.py 的任何默认行为，不写 exit_best.pt /
web_best / dushan_master.pt，不并发启动第二驱动**。它是完全隔离的影子轨道：
独立分片目录 `runs/shadow/`、独立 ckpt 前缀 `models/shadow_e1_*.pt`、
独立闸门 jsonl、独立 `.done` 标记。产物只留档，不自动晋升、不自动上线。

实验设计（**严格单变量配对**）
------------------------------
- 共同 init：`models/exit_p20.pt`（PPO 链末端，固定文件，主链永不覆盖）
- 共同 rollout：本脚本自采 320 局纯策略自对弈（seed0=1_100_000，独立池），
  写 `runs/shadow/e1_shard_<wid>.npz`（不碰主链 `runs/ppo_shards/`）
- 共同 PPO 超参：epochs=2 / lr=5e-5 / clip=0.2 / batch=256 / ent=0 / kl_target=0.03，
  两组 torch 种子相同、同一份数据、同一子样本
- **唯一差异**：GAE 里的 Φ 源
    A 组（对照）：Φ = 8·V_θ，即 rollout 落盘 `val`（主链同款）
    B 组（实验）：Φ = 8·V_frozen，`models/exit_r12.pt` 的 value head
      （全链冻结、enc3 同架构、V=maxQ/8 量纲一致；批量前向一次算出，零 rollout 开销）
- 评估：两组各跑一次冻结闸门 A（710000+300、四座旋转、vs 冻结 master 贪心），
  逐 (seed, seat) 严格配对，按 seed 跨四座平均得 300 个独立观测。

预登记判据（**2026-11-07 20:3x，看到任何闸门结果前锁死，不得事后改**）
-----------------------------------------------------------------------
Δ = mean_seed(d_B − d_A)，d = deltas[cand]−mean(其余三家)，±SE 与双侧 p：
  * Δ >= +1.5 且 p<0.05 ⇒ 「冻结密集信号源有效」：写
    `runs/shadow/e1_champion_candidate.json`，登记为**第三周期**候选方向，
    仍不改第二周期主链；走只增不替需人工/自检按 register_champion 决定。
  * |Δ| < 1.5 或 ns ⇒ 「无证据」。**低功效声明**：320 局 + 一次更新的差异很小，
    300 seed 的 SE≈0.55，ns 只能写「本实验功效不足以检出 <1.5 分的效应」，
    **不得写成「已证明 GRP 无效」**。
  * Δ <= −1.5 且 p<0.05 ⇒ 「冻结源有害」，排除该方向。
单次预登记配对，不在多轮里挑显著；不做任何超参/阈值扫描。

为什么现在值得跑（G1 探针，runs/shadow/grp_probe.json）
------------------------------------------------------
`tools/grp_probe.py` 实测 PPO 链跨 8 轮 `mean|ΔV|×8 = 1.92 分`（阈值 0.50，
p90=4.34），冻结源与 PPO 链 r≈0.70 却差 3.9 分 ⇒ V_θ 的**跨轮**漂移可观，
且冻结源确实是不同信号。注意：这不推翻 §6.35.2②「V_θ 在 rollout 内冻结」——
两者量的是不同东西（轮内平稳性 vs 轮间漂移）。

窗口守卫（**绝不与主链抢核**）
------------------------------
只在「完整 DAgger 轮的 train 段」执行：最新 `train_rNN.log` 的 `.done` 未出现
（train_exit 走 MPS、CPU 相对空闲）**且 mtime 在 2h 内活跃**，且 annotate 已退出、
600-seed 预登记测量（tools/run_fresh_pool_measure.py）不在跑、`e1.done` 不存在。
新鲜度是硬条件：`train_r01.log` 是 45h 前未完成的历史残留，只按「无 .done」判断会
误判成当前训练段，进而与 R21 train 段的 600-seed 测量撞车。由每小时自检自动化
在窗口内调用，无需长驻后台进程。全程 `--workers 2 / threads 1 / nice`。

用法：
  .venv/bin/python tools/shadow_lab.py --dry     # 用现有冒烟分片验管道，不写 ckpt/不跑闸门
  .venv/bin/python tools/shadow_lab.py --e1      # 正式（自带窗口守卫，一次性）
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import math
import multiprocessing as mp
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
SHADOW = ROOT / "runs" / "shadow"
LOGS = ROOT / "runs" / "night_logs"
PY = sys.executable

INIT = "models/exit_p20.pt"        # 共同 init，固定文件
FROZEN = "models/exit_r12.pt"      # 冻结价值源
MASTER = "models/dushan_master.pt"  # 冻结基线（闸门 A）
HANDS, WORKERS, MAX_STATES, THREADS = 320, 2, 20_000, 1
EPOCHS, LR, CLIP, BATCH, ENT, KL = 2, 5e-5, 0.2, 256, 0.0, 0.03
SEED0 = 1_100_000                 # rollout 独立池
POOL_SEED0, POOL_SEEDS = 710_000, 300
D_THRESHOLD, D_P = 1.5, 0.05
SEED = 20261107
MAX_SECONDS = 1800               # 单次实验硬上限 30min
MAX_TRAIN_LOG_AGE = 7200         # 最新 train 日志 mtime 必须 2h 内活跃（防历史残留误判）


def log(msg: str) -> None:
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with (LOGS / "shadow_e1.log").open("a", encoding="utf-8") as fh:
        fh.write(line + "\n")


# ---------------- rollout（独立目录，零删除） ----------------

def shadow_rollout_shard(ckpt: str, hands: int, seed0: int,
                         max_states: int, wid: int) -> str:
    """与 train_ppo.rollout_shard 同构，但写 runs/shadow/，不碰主链分片。"""
    import torch
    from zhuoji.dushan import DushanConfig, DushanGame
    from zhuoji.encoder import encode, index_to_action
    from train_ppo import HEADS, build_net

    torch.set_num_threads(THREADS)
    net, _ = build_net(ckpt, "cpu")
    net.eval()
    T, G, HEAD, IDX, LOGP, VAL, SEAT, HAND, MASK = [], [], [], [], [], [], [], [], []
    rets, n_dec = [], 0
    rng = np.random.default_rng(seed0)
    t0 = time.perf_counter()

    for h in range(hands):
        g = DushanGame(DushanConfig(), seed=seed0 + h)
        steps = 0
        while not g.result and steps < 600:
            p = g.actor()
            obs = encode(g, p, 3, getattr(net, "encoder_packs", None))
            n_act = int(obs.mask.shape[0])
            with torch.no_grad():
                out = net(torch.from_numpy(obs.tiles[None].astype(np.float32)),
                          torch.from_numpy(obs.glob[None].astype(np.float32)))
                logits = out[obs.head][0][:n_act].float().cpu()
                logits[~torch.from_numpy(obs.mask)] = -1e9
                logp_all = torch.log_softmax(logits, dim=-1)
                prob = logp_all.exp().numpy().astype(np.float64)
                s = prob.sum()
                prob = prob / s if s > 0 else np.full(n_act, 1.0 / n_act)
                k = int(rng.choice(n_act, p=prob))
                lp = float(logp_all[k])
                val = float(out["value"].reshape(-1)[0].cpu())
            if len(T) < max_states:
                mk = np.zeros(28, dtype=np.uint8)
                mk[:n_act] = obs.mask.astype(np.uint8)
                T.append(obs.tiles.astype(np.float16))
                G.append(obs.glob.astype(np.float16))
                HEAD.append(HEADS.index(obs.head))
                IDX.append(k)
                LOGP.append(lp)
                VAL.append(val)
                SEAT.append(p)
                HAND.append(h)
                MASK.append(mk)
            g.step(index_to_action(obs.head, k, g))
            steps += 1
            n_dec += 1
        d = (g.result or {}).get("deltas") or [0.0] * 4
        rets.append([float(x) for x in d])

    el = time.perf_counter() - t0
    path = SHADOW / f"e1_shard_{wid}.npz"
    np.savez_compressed(
        str(path),
        T=np.stack(T) if T else np.zeros((0, 63, 27), np.float16),
        G=np.stack(G) if G else np.zeros((0, 130), np.float16),
        head=np.array(HEAD, np.int8), idx=np.array(IDX, np.int16),
        logp=np.array(LOGP, np.float32), val=np.array(VAL, np.float32),
        seat=np.array(SEAT, np.int8), hand=np.array(HAND, np.int32),
        mask=np.stack(MASK) if MASK else np.zeros((0, 28), np.uint8),
        rets=np.array(rets, np.float64),
        stats=np.array([n_dec, len(T), el, hands], np.float64),
    )
    return str(path)


def merge(paths: list[str]) -> dict:
    """与 train_ppo.main 的合并逻辑一致（hand offset 保证 GAE 按局回溯正确）。"""
    parts, rets_all, off = [], [], 0
    for p in paths:
        d = np.load(p)
        st = d["stats"]
        parts.append({"T": d["T"], "G": d["G"], "head": d["head"], "idx": d["idx"],
                      "logp": d["logp"], "val": d["val"], "seat": d["seat"],
                      "hand": d["hand"].astype(np.int64) + off, "mask": d["mask"]})
        rets_all.append(d["rets"])
        off += int(st[3])
    out = {k: np.concatenate([q[k] for q in parts]) for k in parts[0]}
    out["rets"] = np.concatenate(rets_all)
    return out


# ---------------- Φ 源 / GAE / 更新 ----------------

def frozen_values(tiles: np.ndarray, glob: np.ndarray, batch: int = 256) -> np.ndarray:
    """冻结价值源批量前向（一次算完，零 rollout 开销）。"""
    import torch
    from train_ppo import build_net

    torch.set_num_threads(THREADS)
    net, _ = build_net(str(ROOT / FROZEN), "cpu")
    net.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(tiles), batch):
            t = torch.from_numpy(np.ascontiguousarray(tiles[i:i + batch], np.float32))
            g = torch.from_numpy(np.ascontiguousarray(glob[i:i + batch], np.float32))
            out.append(net(t, g)["value"].reshape(-1).numpy())
    return np.concatenate(out).astype(np.float64)


def gae_with_phi(st: dict, phi: np.ndarray, gamma: float = 1.0,
                 lam: float = 0.95) -> tuple[np.ndarray, np.ndarray]:
    """train_ppo.gae 的参数化版：phi 已是分差量纲（8·V），逐 (局,座) 回溯。"""
    n = len(st["head"])
    A = np.zeros(n, dtype=np.float32)
    hand, seat, rets = st["hand"], st["seat"], st["rets"]
    for hid in np.unique(hand):
        for s in range(4):
            idxs = np.where((hand == hid) & (seat == s))[0]
            if idxs.size == 0:
                continue
            R = float(rets[int(hid)][s]) if int(hid) < len(rets) else 0.0
            last = 0.0
            for j in range(len(idxs) - 1, -1, -1):
                phi_next = phi[idxs[j + 1]] if j + 1 < len(idxs) else 0.0
                r = phi_next - phi[idxs[j]] + (R if j == len(idxs) - 1 else 0.0)
                last = r + gamma * lam * last
                A[idxs[j]] = last
    return A, A + phi


def run_update(tag: str, st: dict, A: np.ndarray, RT: np.ndarray,
               device: str = "cpu") -> dict:
    import torch
    from train_ppo import build_net, ppo_update, save_ckpt

    torch.set_num_threads(THREADS)
    torch.manual_seed(SEED)
    net, cfg = build_net(str(ROOT / INIT), device)
    s = ppo_update(net, st, A, RT, device, epochs=EPOCHS, lr=LR, clip=CLIP,
                   batch_size=BATCH, ent_coef=ENT, kl_target=KL)
    save_ckpt(net, cfg, ROOT / f"models/shadow_e1_{tag}.pt", {
        "init": INIT, "phi_source": "V_theta" if tag == "a" else "V_frozen(r12)",
        "hands": HANDS, "states": len(st["head"]), "epochs": EPOCHS,
        "lr": LR, "seed": SEED, "kind": "shadow_e1", "done": True, "iter": 1,
    })
    return {"steps": s.get("steps"), "kl": s.get("kl_epoch"),
            "vl": s.get("vl"), "clipfrac": s.get("clipfrac"),
            "gnorm": s.get("gnorm"), "kl_stop": bool(s.get("kl_stop"))}


def gate_a(tag: str) -> int:
    out = SHADOW / f"e1_{tag}_gateA.jsonl"
    cmd = [PY, "-u", "eval_pairs.py",
           "--a", f"models/shadow_e1_{tag}.pt", "--b", MASTER,
           "--mode", "greedy", "--seeds", str(POOL_SEEDS),
           "--seed0", str(POOL_SEED0), "--workers", "4", "--device", "cpu",
           "--out", str(out)]
    log(f"闸门 A [{tag}]: " + " ".join(cmd[2:]))
    logf = (LOGS / f"shadow_e1_gateA_{tag}.log").open("a", encoding="utf-8")
    rc = subprocess.call(cmd, cwd=str(ROOT), stdout=logf, stderr=subprocess.STDOUT)
    logf.close()
    log(f"闸门 A [{tag}] rc={rc} rows={sum(1 for _ in out.open()) if out.exists() else 0}")
    return rc


def d_from_jsonl(path: Path) -> dict:
    out = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        j = json.loads(line)
        ds = j["deltas"]
        s = int(j["cand_seat"])
        out[(int(j["seed"]), s)] = ds[s] - (sum(ds) - ds[s]) / 3.0
    return out


def paired_delta() -> dict:
    a, b = d_from_jsonl(SHADOW / "e1_a_gateA.jsonl"), d_from_jsonl(SHADOW / "e1_b_gateA.jsonl")
    seeds = sorted({k[0] for k in a} & {k[0] for k in b})
    diff = [np.mean([b[(sd, s)] - a[(sd, s)] for s in range(4)]) for sd in seeds]
    d = np.asarray(diff, dtype=np.float64)
    m, se = float(d.mean()), float(d.std(ddof=1) / math.sqrt(len(d)))
    p = math.erfc(abs(m / se) / math.sqrt(2)) if se > 0 else 1.0
    if m >= D_THRESHOLD and p < D_P:
        verdict = "冻结密集信号源有效 ⇒ 登记第三周期候选方向（不改第二周期主链）"
    elif m <= -D_THRESHOLD and p < D_P:
        verdict = "冻结源有害 ⇒ 排除该方向"
    else:
        verdict = "无证据（低功效：ns 不等于已证明无效）"
    return {"n_seed": len(seeds), "delta": round(m, 4), "se": round(se, 4),
            "p": round(p, 6), "threshold": D_THRESHOLD, "alpha": D_P,
            "verdict": verdict,
            "low_power_note": "320 局 + 一次更新，300 seed 的 SE≈0.55；|Δ|<1.5 只能写功效不足"}


# ---------------- 窗口守卫 ----------------

def window_guard() -> tuple[bool, str]:
    if (SHADOW / "e1.done").exists():
        return False, "e1.done 已存在（一次性实验，防重复）"
    if (SHADOW / "e1.skip").exists():
        return False, "e1.skip 已存在"
    # annotate 必须已退出
    r = subprocess.run(["pgrep", "-f", "target/release/annotate"],
                       capture_output=True, text=True)
    if r.stdout.strip():
        return False, "annotate 仍在跑（严禁并发）"
    # 600-seed 预登记测量不能同时在跑
    r = subprocess.run(["pgrep", "-f", "run_fresh_pool_measure.py"],
                       capture_output=True, text=True)
    if r.stdout.strip():
        return False, "600-seed 预登记测量仍在跑"
    # 必须落在**当前活跃**的完整 DAgger 轮 train 段：只看 mtime 最新的那份日志。
    # 只按「无 .done」判断会被 train_r01.log（45h 前未配对残留）误判成活跃窗口。
    trains = sorted(LOGS.glob("train_r*.log"), key=lambda p: p.stat().st_mtime)
    if not trains:
        return False, "无 train_rNN.log（不在任何 DAgger 轮）"
    latest = trains[-1]
    if (LOGS / (latest.stem + ".done")).exists():
        return False, f"最新 {latest.name} 已 .done（已进入闸门/加速轮阶段）"
    age = time.time() - latest.stat().st_mtime
    if age > MAX_TRAIN_LOG_AGE:
        return False, (f"最新 {latest.name} 已 {age / 3600:.1f}h 未写入，"
                       f"是历史残留而非活跃 train 段")
    return True, f"合法窗口：{latest.name} 活跃中（{age / 60:.0f}min 内有写入），annotate 已退出"


# ---------------- 主流程 ----------------

def do_dry() -> int:
    """用现有冒烟分片验完整管道（不写 ckpt、不跑闸门、不占 CPU）。"""
    paths = sorted(str(p) for p in (ROOT / "runs/ppo_shards").glob("shard_*.npz"))
    if not paths:
        log("[dry][FAIL] 没有冒烟分片")
        return 1
    st = merge(paths)
    log(f"[dry] 合并 {len(paths)} 分片 → {len(st['head'])} 状态 "
        f"{len(st['rets'])} 局")
    phi_a = 8.0 * st["val"].astype(np.float64)
    phi_b = 8.0 * frozen_values(st["T"], st["G"])
    A_a, RT_a = gae_with_phi(st, phi_a)
    A_b, RT_b = gae_with_phi(st, phi_b)
    log(f"[dry] Φ 差 mean|Δ|={np.abs(phi_b - phi_a).mean():.4f} 分  "
        f"A_a.std={A_a.std():.4f} A_b.std={A_b.std():.4f}  "
        f"corr(A_a,A_b)={np.corrcoef(A_a, A_b)[0, 1]:.4f}")
    import torch
    from train_ppo import build_net, ppo_update
    torch.set_num_threads(THREADS)
    torch.manual_seed(SEED)
    net, _ = build_net(str(ROOT / INIT), "cpu")
    s = ppo_update(net, st, A_a, RT_a, "cpu", epochs=1, lr=LR, clip=CLIP,
                   batch_size=64, ent_coef=ENT, kl_target=KL)
    log(f"[dry] ppo_update 通: steps={s.get('steps')} loss={s.get('loss'):.4f} "
        f"kl={s.get('kl_epoch')}")
    log("[dry] OK（未写任何 ckpt、未跑闸门、未碰主链分片）")
    return 0


def do_e1() -> int:
    ok, why = window_guard()
    if not ok:
        log(f"[e1] 窗口不成立，跳过：{why}")
        return 2
    log(f"[e1] 窗口成立：{why}")
    SHADOW.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()

    # 1) 自采 rollout（独立池、独立分片、限核）
    per = [HANDS // WORKERS + (1 if i < HANDS % WORKERS else 0) for i in range(WORKERS)]
    cap = max(1, MAX_STATES // WORKERS)
    ctx = mp.get_context("spawn")
    with cf.ProcessPoolExecutor(max_workers=WORKERS, mp_context=ctx) as ex:
        futs = [ex.submit(shadow_rollout_shard, str(ROOT / INIT), per[i],
                          SEED0 + i * 100_000, cap, i) for i in range(WORKERS)]
        paths = [f.result() for f in cf.as_completed(futs)]
    st = merge(sorted(paths))
    log(f"[e1] rollout {HANDS} 局 → {len(st['head'])} 状态 "
        f"{time.perf_counter() - t0:.0f}s")
    if len(st["head"]) < 500:
        log("[e1][FAIL] 状态太少")
        return 1

    # 2) 两个 Φ 源 → 两组 GAE（唯一差异）
    phi_a = 8.0 * st["val"].astype(np.float64)
    phi_b = 8.0 * frozen_values(st["T"], st["G"])
    A_a, RT_a = gae_with_phi(st, phi_a)
    A_b, RT_b = gae_with_phi(st, phi_b)
    log(f"[e1] Φ 差 mean|Δ|={np.abs(phi_b - phi_a).mean():.3f} 分  "
        f"corr(A)={np.corrcoef(A_a, A_b)[0, 1]:.4f}")

    # 3) 两次同种子更新
    ma = run_update("a", st, A_a, RT_a)
    mb = run_update("b", st, A_b, RT_b)
    log(f"[e1] 更新完成 A={ma} B={mb}")
    if time.perf_counter() - t0 > MAX_SECONDS:
        log("[e1][abort] 超过硬上限，放弃闸门（不并发拖累主链）")
        return 3

    # 4) 两组冻结闸门 A
    if gate_a("a") != 0 or gate_a("b") != 0:
        log("[e1][FAIL] 闸门 A 未完整，结果作废不判读")
        return 1

    # 5) 判读（判据已锁死）
    res = paired_delta()
    res.update({"phi_mean_abs_diff": round(float(np.abs(phi_b - phi_a).mean()), 4),
                "states": int(len(st["head"])), "hands": HANDS,
                "init": INIT, "frozen": FROZEN, "pool": f"{POOL_SEED0}+{POOL_SEEDS}",
                "secs": round(time.perf_counter() - t0, 1),
                "update_a": ma, "update_b": mb})
    (SHADOW / "e1_result.json").write_text(
        json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    (SHADOW / "e1.done").write_text(time.strftime("%F %H:%M:%S") + "\n", encoding="utf-8")
    log(f"[e1] Δ={res['delta']:+.3f}±{res['se']:.3f} p={res['p']:.4g} ⇒ {res['verdict']}")
    if res["delta"] >= D_THRESHOLD and res["p"] < D_P:
        (SHADOW / "e1_champion_candidate.json").write_text(json.dumps({
            "ckpt": "models/shadow_e1_b.pt", "delta": res["delta"], "p": res["p"],
            "note": "仅方向筛选阳性；走只增不替须按 register_champion 另行决定，"
                    "绝不覆盖 dushan_master/exit_best/web_best"},
            ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry", action="store_true")
    ap.add_argument("--e1", action="store_true")
    a = ap.parse_args()
    SHADOW.mkdir(parents=True, exist_ok=True)
    if a.dry:
        return do_dry()
    if a.e1:
        return do_e1()
    log("用法：--dry（验管道）或 --e1（正式，自带窗口守卫）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
