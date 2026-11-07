"""PPO-lite 训练轮次（§6.22.7 / §6.24）：在 DAgger 主线之外产出一个「结局优化」候选。

设计要点（与 DAgger 互不替代，只竞争同一个闸门）：

  * 初始化：`--init models/exit_best.pt`（enc3 / 63×27 / glob 130）；
  * rollout：**纯策略**自对弈（无搜索、无教师），单局 = 一个 episode，
    终端奖励 = 该局四座净分差 `result["deltas"]`（零和）；
  * 势函数塑形（Ng 1999，policy-invariant）：Φ(s)=8·V_θ(s)、终局 Φ≡0，
    逐步 r = Φ(s_{t+1}) − Φ(s_t)，价值头量纲即 maxQ/8（与 train_exit.VALUE_SCALE 一致）；
  * GAE(γ=1, λ=0.95) 按 (局, 座) 回溯；
  * clipped PPO + 价值 MSE + **KL 早停**（防止一次更新把蒸馏策略带崩）。

为什么 PPO：8 轮 DAgger 后闸门 A 环比已 4 轮为 0、闸门 C≈0、best-val 连续 5 轮落在
ep1 —— 瓶颈是教师标签的信息天花板，不是拟合能力。PPO 用「谁赢」这个 DAgger 永远
拿不到的信号，是当前唯一的新信息源。

产出 ckpt 格式与 `train_exit.py` 完全一致（`net_meta` + `state_dict` + `encoder_version=3`），
因此 `eval_pairs.py` / 闸门 A·C·D 可直接评估，无需任何转换。

用法：
  .venv/bin/python train_ppo.py --init models/exit_best.pt --out models/exit_p10.pt \
      --hands 2400 --workers 4 --epochs 2 --lr 5e-5
  .venv/bin/python train_ppo.py --hands 16 --max-states 1500 --dry   # 冒烟（不改 ckpt）
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

PHI_SCALE = 8.0          # 与 zhuoji/reward.PHI_SCALE、train_exit.VALUE_SCALE 同量纲
GAMMA = 1.0
LAM = 0.95
HEAD_DIM = {"discard": 28, "self_kong": 28, "respond": 4}
HEADS = ("discard", "self_kong", "respond")


# ---------------- 网络 ----------------

def build_net(ckpt: str | None, device: str):
    import torch
    from zhuoji.net_m1 import M1Net
    from train_exit import STUDENT_CFG

    cfg = STUDENT_CFG
    net = M1Net(cfg.channels, cfg.blocks, cfg.hidden,
                global_dim=cfg.global_dim, in_channels=cfg.in_channels,
                spatial_head=cfg.spatial_head, se=cfg.se, bn_eps=cfg.bn_eps,
                aux=cfg.aux, dueling=cfg.dueling, se_dual=cfg.se_dual,
                bn_momentum=cfg.bn_momentum)
    if ckpt:
        ck = torch.load(ckpt, map_location="cpu", weights_only=False)
        net.load_state_dict(ck["state_dict"])
        print(f"[init] {ckpt}  encoder_version={ck.get('encoder_version')}", flush=True)
    net.to(device)
    return net, cfg


def save_ckpt(net, cfg, out: Path, extra: dict) -> None:
    import torch
    from zhuoji.net import net_meta

    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "net": net_meta(cfg),
        "state_dict": net.state_dict(),
        "encoder_version": 3,
        "encoder_packs": ["a", "b", "c", "d", "e", "f", "g"],
        "iter": extra.get("iter", 1),
        "eval": {"val_loss": float("nan"), "n_train": int(extra.get("states", 0)),
                 "tau": 1.0, "kind": "ppo"},
        "distill": {"kind": "ppo", "from": extra.get("init", ""),
                    "hands": int(extra.get("hands", 0))},
        "ppo": extra,
    }, str(out))


# ---------------- rollout（子进程） ----------------

def rollout_shard(ckpt: str, device: str, hands: int, seed0: int,
                  max_states: int, threads: int, wid: int = 0) -> str:
    """纯策略自对弈一个分片，写 npz 到**固定路径**并返回路径（避免大数组走 pickle）。

    2026-11-07 P21 事故：原先每片用 tempfile.mkstemp 生成随机名，主进程合并后
    os.unlink 删除。WorkBuddy 的 safe-delete 批量守卫（50 次/turn 阈值）会拦截
    os.unlink 并让子进程 rc=1 ⇒ P21 静默失败、R21 回落到 2.4h 完整 DAgger。
    现改为 runs/ppo_shards/shard_<wid>.npz 固定名覆盖写，全链路**零删除**：
    每轮只有 workers 个文件，旧轮次自动被覆盖，不会堆积。
    """
    import torch
    from zhuoji.dushan import DushanConfig, DushanGame
    from zhuoji.encoder import encode, index_to_action

    torch.set_num_threads(threads)
    net, _cfg = build_net(ckpt, device)
    net.eval()
    packs = getattr(net, "encoder_packs", None)
    ver = 3

    T, G, HEAD, IDX, LOGP, VAL, SEAT, HAND, MASK = [], [], [], [], [], [], [], [], []
    rets, ent_all, n_dec = [], [], 0
    rng = np.random.default_rng(seed0)
    t0 = time.perf_counter()

    for h in range(hands):
        g = DushanGame(DushanConfig(), seed=seed0 + h)
        steps = 0
        while not g.result and steps < 600:
            p = g.actor()
            obs = encode(g, p, ver, packs)
            n_act = int(obs.mask.shape[0])
            with torch.no_grad():
                out = net(torch.from_numpy(obs.tiles[None].astype(np.float32)).to(device),
                          torch.from_numpy(obs.glob[None].astype(np.float32)).to(device))
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
            ent_all.append(float(-(prob * np.log(np.clip(prob, 1e-12, None))).sum()))
            g.step(index_to_action(obs.head, k, g))
            steps += 1
            n_dec += 1
        d = (g.result or {}).get("deltas") or [0.0] * 4
        rets.append([float(x) for x in d])

    el = time.perf_counter() - t0
    shard_dir = ROOT / "runs" / "ppo_shards"
    shard_dir.mkdir(parents=True, exist_ok=True)
    path = str(shard_dir / f"shard_{wid}.npz")
    np.savez_compressed(
        path,
        T=np.stack(T) if T else np.zeros((0, 63, 27), np.float16),
        G=np.stack(G) if G else np.zeros((0, 130), np.float16),
        head=np.array(HEAD, np.int8), idx=np.array(IDX, np.int16),
        logp=np.array(LOGP, np.float32), val=np.array(VAL, np.float32),
        seat=np.array(SEAT, np.int8), hand=np.array(HAND, np.int32),
        mask=np.stack(MASK) if MASK else np.zeros((0, 28), np.uint8),
        rets=np.array(rets, np.float64),
        stats=np.array([n_dec, len(T), el, hands,
                        float(np.median(ent_all)) if ent_all else 0.0], np.float64),
    )
    return path


# ---------------- GAE / 更新 ----------------

def gae(states: dict, gamma: float = GAMMA, lam: float = LAM):
    n = len(states["head"])
    A = np.zeros(n, dtype=np.float32)
    hand, seat, val, rets = states["hand"], states["seat"], states["val"], states["rets"]
    for hid in np.unique(hand):
        for s in range(4):
            idxs = np.where((hand == hid) & (seat == s))[0]
            if idxs.size == 0:
                continue
            R = float(rets[hid][s]) if hid < len(rets) else 0.0
            phi = PHI_SCALE * val[idxs].astype(np.float64)
            last = 0.0
            for j in range(len(idxs) - 1, -1, -1):
                phi_next = phi[j + 1] if j + 1 < len(idxs) else 0.0
                r = phi_next - phi[j] + (R if j == len(idxs) - 1 else 0.0)
                last = r + gamma * lam * last
                A[idxs[j]] = last
    RT = A + PHI_SCALE * val
    return A, RT


def ppo_update(net, st: dict, A: np.ndarray, RT: np.ndarray, device: str, *,
               epochs: int, lr: float, clip: float, batch_size: int,
               ent_coef: float, kl_target: float) -> dict:
    import torch

    n = len(st["head"])
    if n == 0:
        return {"steps": 0}
    A_t = torch.from_numpy(((A - A.mean()) / (A.std() + 1e-8)).astype(np.float32))
    RT_t = torch.from_numpy((RT / PHI_SCALE).astype(np.float32))
    idx_t = torch.from_numpy(st["idx"].astype(np.int64))
    head_t = torch.from_numpy(st["head"].astype(np.int64))
    logp_old = torch.from_numpy(st["logp"].astype(np.float32))
    mask_t = torch.from_numpy(st["mask"].astype(bool))

    opt = torch.optim.Adam(net.parameters(), lr=lr)
    net.train()
    hist, kl_epoch = [], []
    stop = False

    def pad28(t):
        if t.shape[-1] == 28:
            return t
        o = torch.full((t.shape[0], 28), -1e9, device=t.device, dtype=t.dtype)
        o[:, :t.shape[-1]] = t
        return o

    for ep in range(epochs):
        perm = torch.randperm(n)
        kl_acc, nb = 0.0, 0
        for k0 in range(0, n, batch_size):
            b = perm[k0:k0 + batch_size]
            bi = b.numpy()
            tiles = torch.from_numpy(st["T"][bi].astype(np.float32)).to(device)
            glob = torch.from_numpy(st["G"][bi].astype(np.float32)).to(device)
            out = net(tiles, glob)
            lg = torch.stack((pad28(out["discard"]), pad28(out["self_kong"]),
                              pad28(out["respond"])), dim=1)          # (B,3,28)
            sel = lg[torch.arange(len(bi), device=device), head_t[b].to(device)]
            sel = sel.masked_fill(~mask_t[b].to(device), -1e9)
            lp = torch.log_softmax(sel, dim=-1)
            lp_new = lp.gather(1, idx_t[b].to(device).unsqueeze(1)).squeeze(1)
            lp_old = logp_old[b].to(device)
            ent = -(lp.exp() * lp).sum(dim=-1).mean()
            ratio = torch.exp(lp_new - lp_old)
            ai = A_t[b].to(device)
            pg = -torch.min(ratio * ai, torch.clamp(ratio, 1 - clip, 1 + clip) * ai).mean()
            v = out["value"].reshape(-1)
            vl = torch.nn.functional.mse_loss(v, RT_t[b].to(device))
            loss = pg + 0.5 * vl - ent_coef * ent
            opt.zero_grad(set_to_none=True)
            loss.backward()
            gnorm = float(torch.nn.utils.clip_grad_norm_(net.parameters(), 0.5))
            opt.step()
            with torch.no_grad():
                kl_acc += float((lp_old - lp_new).mean())
                nb += 1
            hist.append({"loss": float(loss), "pg": float(pg), "vl": float(vl),
                         "ratio": float(ratio.mean()), "gnorm": gnorm,
                         "clipfrac": float(((ratio - 1).abs() > clip).float().mean()),
                         "ent": float(ent)})
        kl = kl_acc / max(nb, 1)
        kl_epoch.append(kl)
        print(f"[ppo ep {ep+1}/{epochs}] steps={nb} kl={kl:+.5f} "
              f"loss={hist[-1]['loss']:.4f} pg={hist[-1]['pg']:+.4f} "
              f"vl={hist[-1]['vl']:.4f} ratio={hist[-1]['ratio']:.4f} "
              f"clip={hist[-1]['clipfrac']:.3f} ent={hist[-1]['ent']:.3f}", flush=True)
        if kl > kl_target:
            print(f"[ppo] KL {kl:.4f} > {kl_target} → 提前停止（防策略带崩）", flush=True)
            stop = True
            break

    net.eval()
    agg = {k: float(np.mean([h[k] for h in hist])) for k in hist[0]} if hist else {}
    agg.update({"steps": len(hist), "kl_epoch": kl_epoch, "kl_stop": stop, "n": n})
    return agg


# ---------------- 主流程 ----------------

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--init", default=str(ROOT / "models/exit_best.pt"))
    ap.add_argument("--out", default=str(ROOT / "models/exit_ppo.pt"))
    ap.add_argument("--hands", type=int, default=960)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--max-states", type=int, default=60000, help="状态池上限（控制内存）")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--clip", type=float, default=0.2)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--ent-coef", type=float, default=0.0)
    ap.add_argument("--kl-target", type=float, default=0.03)
    ap.add_argument("--seed0", type=int, default=900000, help="rollout 牌墙起点（避开评估池）")
    ap.add_argument("--rollout-device", default="cpu",
                    help="rollout 永远用 cpu：逐决策小 batch，GPU 启动开销会拖慢 5 倍")
    ap.add_argument("--device", default="mps",
                    help="PPO 更新用的设备；实测 mps 0.70s/step vs cpu 3.3s/step")
    ap.add_argument("--dry", action="store_true", help="冒烟：不写 ckpt")
    a = ap.parse_args()

    import torch
    torch.set_num_threads(a.threads)
    t_all = time.perf_counter()

    n_w = max(1, min(a.workers, a.hands))
    per = [a.hands // n_w + (1 if i < a.hands % n_w else 0) for i in range(n_w)]
    cap = max(1, a.max_states // n_w)
    print(f"[plan] hands={a.hands} workers={n_w} 每片={per} 状态上限/片={cap} "
          f"init={Path(a.init).name}", flush=True)

    # 1) 并行 rollout
    import concurrent.futures as cf
    import multiprocessing as mp
    t0 = time.perf_counter()
    paths = []
    ctx = mp.get_context("spawn")
    with cf.ProcessPoolExecutor(max_workers=n_w, mp_context=ctx) as ex:
        futs = [ex.submit(rollout_shard, a.init, a.rollout_device, per[i],
                          a.seed0 + i * 100_000, cap, a.threads, i)
                for i in range(n_w)]
        for f in cf.as_completed(futs):
            paths.append(f.result())
    roll_s = time.perf_counter() - t0

    # 2) 合并分片
    parts, rets_all, n_dec, ent_med, off = [], [], 0.0, [], 0
    for p in paths:
        d = np.load(p)
        st = d["stats"]
        n_dec += float(st[0])
        ent_med.append(float(st[4]))
        hand = d["hand"].astype(np.int64) + off
        off += int(st[3])
        parts.append({"T": d["T"], "G": d["G"], "head": d["head"], "idx": d["idx"],
                      "logp": d["logp"], "val": d["val"], "seat": d["seat"],
                      "hand": hand, "mask": d["mask"]})
        rets_all.append(d["rets"])
        # 不删除：分片固定路径下轮覆盖写，删除会触发 safe-delete 守卫（见函数头）
    st = {k: np.concatenate([q[k] for q in parts]) for k in parts[0]}
    rets = np.concatenate(rets_all)
    st["rets"] = rets
    n = len(st["head"])
    print(f"[rollout] {a.hands} 局 {int(n_dec)} 决策 {n} 状态 {roll_s:.0f}s "
          f"({roll_s/a.hands:.3f}s/局) 熵中位={np.median(ent_med):.3f}", flush=True)
    print(f"[奖励] 零和 max|Σdeltas|={np.abs(rets.sum(1)).max():.2e} "
          f"std={rets.std():.3f} |R|中位={np.median(np.abs(rets)):.2f} "
          f"max|R|={np.abs(rets).max():.1f}", flush=True)
    if n < 500:
        print("[FAIL] 状态太少，放弃", flush=True)
        return 1

    # 3) GAE + PPO
    A, RT = gae(st)
    print(f"[优势] std={A.std():.3f}分 中位|A|={np.median(np.abs(A)):.3f}", flush=True)
    net, cfg = build_net(a.init, a.device)
    t1 = time.perf_counter()
    s = ppo_update(net, st, A, RT, a.device, epochs=a.epochs, lr=a.lr, clip=a.clip,
                   batch_size=a.batch, ent_coef=a.ent_coef, kl_target=a.kl_target)
    upd_s = time.perf_counter() - t1
    ok = (math.isfinite(s.get("loss", float("nan"))) and s.get("steps", 0) > 0)
    print(f"[ppo] {s.get('steps')} 步 {upd_s:.0f}s loss={s.get('loss'):.4f} "
          f"ratio={s.get('ratio'):.4f} clipfrac={s.get('clipfrac'):.3f} "
          f"kl={s.get('kl_epoch')}", flush=True)

    if not a.dry and ok:
        save_ckpt(net, cfg, Path(a.out), {
            "init": a.init, "hands": a.hands, "states": n,
            "epochs": a.epochs, "lr": a.lr, "clip": a.clip,
            "ent_coef": a.ent_coef, "kl_target": a.kl_target,
            "A_std": float(A.std()), "ret_std": float(rets.std()),
            "shaped": True, "gamma": GAMMA, "lam": LAM, "phi_scale": PHI_SCALE,
            "secs_rollout": round(roll_s, 1), "secs_update": round(upd_s, 1),
            "done": True, "iter": 1,
        })
        print(f"[save] {a.out}", flush=True)
    print(f"[done] total {time.perf_counter()-t_all:.0f}s  states={n} "
          f"kl_ok={not s.get('kl_stop', False)}", flush=True)
    print("[metrics] " + json.dumps({"hands": a.hands, "states": n,
                                     "A_std": float(A.std()),
                                     "ret_std": float(rets.std()),
                                     "kl": s.get("kl_epoch"),
                                     "vl": s.get("vl"), "ent": s.get("ent"),
                                     "gnorm": s.get("gnorm"),
                                     "clipfrac": s.get("clipfrac"),
                                     "ent_rollout": float(np.median(ent_med)),
                                     "kl_stop": bool(s.get("kl_stop")),
                                     "secs": round(time.perf_counter() - t_all, 1)}),
          flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
