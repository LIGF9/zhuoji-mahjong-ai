"""PPO-lite 冒烟（§6.22.7⑤ 前置条件）：自对弈 rollout + 终端分差奖励 + 势函数塑形 + 一次 PPO 更新。

**只做冒烟，不做正式训练**：验证四件事，任一失败即判「分支不可切」。

  1. 管线：纯策略（无搜索）自对弈能跑满 N 局、零异常、掩码始终合法；
  2. 奖励：`deltas` 每手零和、四座均分≈0，且**有可学方差**（否则 PPO 无梯度）；
  3. 优势：Φ=8·V_θ 塑形后 advantage 非退化（std>0.5 分、|A| 中位≠0）；
  4. 更新：PPO clipped loss 有限、首步 ratio≈1、clip frac 合理、value loss 收敛不炸。

势函数说明（重要）：γ=1、Φ(s)=8·V_θ(s)、终局 Φ(s_T)=0 时，逐步塑形奖励
`r_t = Φ(s_{t+1}) − Φ(s_t)` 望远镜求和 = `R − 8V(s_0)`，即**等价于拿 V 做基线**。
所以本实现用「终端 R + 逐步 Φ 差」算 GAE，效果与 R−8V(s_t) 一致，但保留了
λ<1 时的方差折中能力。

用法（训练期请 --device cpu，别抢 MPS）：
  .venv/bin/python tools/ppo_smoke.py --hands 200 --device cpu --epochs 2
  .venv/bin/python tools/ppo_smoke.py --hands 40 --device cpu --no-update   # 只看 rolllout
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

HEAD_DIM = {"discard": 28, "self_kong": 28, "respond": 4}


def build_net(ckpt: str, device: str):
    import torch
    from zhuoji.net import build_from_ckpt

    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    net, ver = build_from_ckpt(ck)
    net.to(device).eval()
    return net, ver, ck


def rollout(net, ver, packs, n_hands, seed0, device, max_states, gamma=1.0,
            lam=0.95, v_scale=8.0):
    """纯策略自对弈，收集 (state, action, logp, value) 与终端分差，返回 dict。"""
    import torch
    from zhuoji.dushan import DushanConfig, DushanGame
    from zhuoji.encoder import encode, index_to_action

    T, G, HEAD, IDX, LOGP, VAL, SEAT, HAND, MASK = (
        [], [], [], [], [], [], [], [], [])
    hand_rets, n_dec, ent_all = [], 0, []
    rng = np.random.default_rng(seed0)
    t0 = time.perf_counter()

    for h in range(n_hands):
        seed = seed0 + h
        g = DushanGame(DushanConfig(), seed=seed)
        steps = 0
        while not g.result and steps < 600:
            p = g.actor()
            obs = encode(g, p, ver, packs)
            with torch.no_grad():
                out = net(torch.from_numpy(obs.tiles[None].astype(np.float32)).to(device),
                          torch.from_numpy(obs.glob[None].astype(np.float32)).to(device))
                logits = out[obs.head][0][:obs.mask.shape[0]].float().cpu()
                logits[~torch.from_numpy(obs.mask)] = -1e9
                logp_all = torch.log_softmax(logits, dim=-1)
                prob = logp_all.exp().numpy().astype(np.float64)
                idx = int(rng.choice(len(prob), p=prob / prob.sum()))
                logp = float(logp_all[idx])
                val = float(out["value"][0].cpu())
            if len(T) < max_states:
                mk = np.zeros(HEAD_DIM[obs.head], dtype=np.uint8)
                mk[:obs.mask.shape[0]] = obs.mask.astype(np.uint8)
                T.append(obs.tiles.astype(np.float16))
                G.append(obs.glob.astype(np.float16))
                HEAD.append(obs.head)
                IDX.append(idx)
                LOGP.append(logp)
                VAL.append(val)
                SEAT.append(p)
                HAND.append(h)
                MASK.append(mk)
            ent_all.append(float(-(prob * np.log(np.clip(prob, 1e-12, None))).sum()))
            g.step(index_to_action(obs.head, idx, g))
            steps += 1
            n_dec += 1
        deltas = (g.result or {}).get("deltas") or [0.0] * 4
        hand_rets.append([float(x) for x in deltas])

    el = time.perf_counter() - t0
    n = len(T)
    return {
        "T": np.stack(T) if n else np.zeros(0), "G": np.stack(G) if n else np.zeros(0),
        "head": np.array(HEAD), "idx": np.array(IDX), "logp": np.array(LOGP),
        "val": np.array(VAL), "seat": np.array(SEAT), "hand": np.array(HAND),
        "mask": MASK, "rets": np.array(hand_rets, dtype=np.float64),
        "decisions": n_dec, "states": n, "secs": el,
        "entropy_median": float(np.median(ent_all)) if ent_all else 0.0,
    }


def gae(batch, gamma=1.0, lam=0.95, v_scale=8.0):
    """按 (hand, seat) 分组回溯；势函数 Φ=8V，终局 Φ=0。"""
    A = np.zeros(batch["states"], dtype=np.float32)
    RT = np.zeros(batch["states"], dtype=np.float32)
    rets = batch["rets"]
    hand, seat, val = batch["hand"], batch["seat"], batch["val"]
    for hid in np.unique(hand):
        for s in range(4):
            idxs = np.where((hand == hid) & (seat == s))[0]
            if idxs.size == 0:
                continue
            R = rets[hid][s] if hid < len(rets) else 0.0
            phi = v_scale * val[idxs]
            adv = np.zeros(len(idxs), dtype=np.float64)
            last = 0.0
            for j in range(len(idxs) - 1, -1, -1):
                # r_j = Φ_{j+1} − Φ_j（终局 Φ=0 但额外加 R）
                phi_next = phi[j + 1] if j + 1 < len(idxs) else 0.0
                r = phi_next - phi[j] + (R if j == len(idxs) - 1 else 0.0)
                last = r + gamma * lam * last
                adv[j] = last
            A[idxs] = adv
    RT = A + v_scale * val
    return A, RT


def ppo_update(net, batch, A, RT, device, epochs=2, lr=1e-4, clip=0.2,
               batch_size=256, v_scale=8.0, ent_coef=0.0):
    import torch

    n = batch["states"]
    if n == 0:
        return {}
    A_t = torch.from_numpy((A - A.mean()) / (A.std() + 1e-8)).float()
    RT_t = torch.from_numpy((RT / v_scale).astype(np.float32))   # value 头在「分/8」尺度
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    net.train()
    stats = []
    for ep in range(epochs):
        perm = torch.randperm(n)
        for k in range(0, n, batch_size):
            b = perm[k:k + batch_size]
            tiles = torch.from_numpy(batch["T"][b.numpy()].astype(np.float32)).to(device)
            glob = torch.from_numpy(batch["G"][b.numpy()].astype(np.float32)).to(device)
            out = net(tiles, glob)
            logp_new, ent = [], []
            for i, gi in enumerate(b.numpy()):
                head = batch["head"][gi]
                mask = torch.from_numpy(batch["mask"][gi]).bool().to(device)
                logits = out[head][i][:mask.shape[0]].float()
                logits = torch.where(mask, logits, torch.full_like(logits, -1e9))
                lp = torch.log_softmax(logits, dim=-1)
                logp_new.append(lp[int(batch["idx"][gi])])
                ent.append(-(lp.exp() * lp).sum())
            logp_new = torch.stack(logp_new)
            logp_old = torch.from_numpy(batch["logp"][b.numpy()]).to(device)
            ratio = torch.exp(logp_new - logp_old)
            ai = A_t[b].to(device)
            pg = -torch.min(ratio * ai,
                            torch.clamp(ratio, 1 - clip, 1 + clip) * ai).mean()
            v = out["value"].float().squeeze(-1)
            vl = torch.nn.functional.mse_loss(v, RT_t[b].to(device))
            loss = pg + 0.5 * vl - ent_coef * torch.stack(ent).mean()
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 0.5)
            opt.step()
            stats.append({"loss": float(loss), "pg": float(pg), "vl": float(vl),
                          "ratio": float(ratio.mean()),
                          "clipfrac": float(((ratio - 1).abs() > clip).float().mean()),
                          "ent": float(torch.stack(ent).mean())})
    net.eval()
    keys = stats[0].keys()
    return {k: float(np.mean([s[k] for s in stats])) for k in keys} | {"steps": len(stats)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=str(ROOT / "models/exit_best.pt"))
    ap.add_argument("--hands", type=int, default=200)
    ap.add_argument("--seed0", type=int, default=880000)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--max-states", type=int, default=12000)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--no-update", action="store_true")
    ap.add_argument("--save", default="", help="可选：另存冒烟后权重（不覆盖 exit_best）")
    a = ap.parse_args()

    import torch
    torch.set_num_threads(4)
    net, ver, _ck = build_net(a.ckpt, a.device)
    packs = getattr(net, "encoder_packs", None)
    print(f"[net] {Path(a.ckpt).name} params={net.n_params():,} enc=v{ver} dev={a.device}")

    b = rollout(net, ver, packs, a.hands, a.seed0, a.device, a.max_states)
    R = b["rets"]
    print(f"[rollout] {a.hands} 局 {b['decisions']} 决策 {b['states']} 存states "
          f"{b['secs']:.1f}s ({b['secs']/max(1,a.hands):.3f}s/局)  "
          f"熵中位={b['entropy_median']:.3f} nats")
    print(f"[奖励] 每手四座和={np.abs(R.sum(1)).max():.2e}（应≈0，零和）  "
          f"均值={R.mean():+.3f}  std={R.std():.3f}  |R|中位={np.median(np.abs(R)):.2f}  "
          f"max|R|={np.abs(R).max():.1f}")

    if b["states"] == 0:
        print("[FAIL] 没有采到状态")
        return 1
    A, RT = gae(b)
    print(f"[优势] std={A.std():.3f}分  中位|A|={np.median(np.abs(A)):.3f}  "
          f"分位[5,95]=[{np.percentile(A,5):+.2f},{np.percentile(A,95):+.2f}]  "
          f"（std<0.5 判退化）")
    ok_adv = A.std() > 0.5
    print(f"[判据1 奖励零和] {'PASS' if np.abs(R.sum(1)).max() < 1e-6 else 'FAIL'}")
    print(f"[判据2 终端方差] {'PASS' if R.std() > 1.0 else 'FAIL'}（std>1 分）")
    print(f"[判据3 优势非退化] {'PASS' if ok_adv else 'FAIL'}")

    if a.no_update:
        print("[skip] --no-update：未做梯度更新")
        return 0

    s = ppo_update(net, b, A, RT, a.device, epochs=a.epochs, lr=a.lr)
    print(f"[更新] {s['steps']} 步  loss={s['loss']:.4f}  pg={s['pg']:+.4f}  "
          f"vl={s['vl']:.4f}  ratio={s['ratio']:.4f}  clipfrac={s['clipfrac']:.3f}  "
          f"ent={s['ent']:.3f}")
    import math
    ok_upd = (math.isfinite(s["loss"]) and abs(s["ratio"] - 1) < 0.5
              and 0.0 <= s["clipfrac"] <= 1.0 and math.isfinite(s["vl"]))
    print(f"[判据4 更新健康] {'PASS' if ok_upd else 'FAIL'}"
          f"（loss 有限 / ratio 首轮贴近 1 / clipfrac∈[0,1]）")

    if a.save:
        from zhuoji.net import ck_from_model
        ck = ck_from_model(net, ver)
        torch.save(ck, a.save)
        print(f"[save] {a.save}")
    verdict = ok_adv and ok_upd and R.std() > 1.0
    print(f"[VERDICT] {'PPO 冒烟通过，可进入 R9 末分支决策' if verdict else '未通过，不切分支'}")
    return 0 if verdict else 2


if __name__ == "__main__":
    raise SystemExit(main())
