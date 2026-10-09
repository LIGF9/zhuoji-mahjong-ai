"""训练循环「不改算法语义」的提速候选基准。

背景
----
`scripts/selfplay.py::update()` 的真实开销构成（每轮）：
    adv 价值前向（no_grad，全样本分块）        ≈ 9 个 batch 当量
    epochs_per_iter(2) × 主前向                ≈ 18 个 batch 当量
    epochs_per_iter(2) × 参考前向（KL 锚点）    ≈ 18 个 batch 当量  ← 见下
    epochs_per_iter(2) × 反向(≈2× 前向)        ≈ 36 个 batch 当量

本脚本要验证的两个「零语义变化」的提速点：

  A. `ref_hoist`：当前参考前向写在 `for hid, name in enumerate(HEAD_NAMES)`
     循环里，每个头各调一次 `ref_model(...)`（对子集）。整条 backbone 被调用
     4 次、且每次只用 1/4 的样本 —— 小批量 BLAS 效率差，而且 4 个头的
     Linear 被各算一遍。把参考前向提到头循环外、一次算全 batch 即可。

  B. `ref_cache`：`ref_model` 是 BC 初始策略的冻结副本且始终 `.eval()`
     （BatchNorm 走 running stats），所以**同一样本的参考输出与 batch 组成无关**。
     于是整轮的参考 logits 可以在轮首算一次（n 个样本 ≈ 9 个 batch 当量），
     两个 epoch 复用 —— 参考前向开销直接减半。

用法：
    python scripts/bench_tweaks.py                       # 0.77M 全矩阵 + 10M 关键项
    python scripts/bench_tweaks.py --configs 64,8,320
    python scripts/bench_tweaks.py --variants baseline ref_cache --json out.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from zhuoji.encoder import GLOBAL_DIM, HEAD_DIMS, NUM_CHANNELS  # noqa: E402
from zhuoji.net import NetConfig, build_model  # noqa: E402

HEAD_NAMES = ("discard", "self_kong", "respond", "value")
ADV_CHUNK = 4096


def make_data(n: int, g: int):
    """造一批与真实 traj 同形的假样本（动作索引必须是该头合法范围内的）。"""
    tiles = torch.randn(n, NUM_CHANNELS, 27)
    glob = torch.randn(n, g)
    mask = torch.ones(n, 28)
    head = torch.randint(0, len(HEAD_NAMES) - 1, (n,))   # 三个动作头；value 不产生动作
    act = torch.tensor([int(torch.randint(0, HEAD_DIMS[HEAD_NAMES[h]], (1,)).item())
                        for h in head.tolist()])
    ret = torch.randn(n)
    return tiles, glob, mask, head, act, ret


def ref_logits_full(ref_model, tiles, glob, n, g):
    """一次性把所有样本的参考 logits 算出来（ref_model 恒定 → 可跨 epoch 复用）。

    只缓存三个动作头：`value` 头不是动作、不参与 KL，宽度也不在 HEAD_DIMS 里。
    """
    act_heads = HEAD_NAMES[:-1]
    out = None
    with torch.no_grad():
        for i in range(0, n, ADV_CHUNK):
            sl = slice(i, i + ADV_CHUNK)
            r = ref_model(tiles[sl], glob[sl])
            if out is None:
                out = {name: torch.zeros(n, r[name].shape[1]) for name in act_heads}
            for name in act_heads:
                out[name][sl] = r[name]
    return out


def run_variant(model, ref_model, opt, data, variant: str, batch: int,
                epochs: int) -> float:
    tiles, glob, mask, head, act, ret = data
    n = tiles.shape[0]
    t0 = time.time()

    with torch.no_grad():
        adv = torch.zeros(n)
        for i in range(0, n, ADV_CHUNK):
            sl = slice(i, i + ADV_CHUNK)
            adv[sl] = ret[sl] - model(tiles[sl], glob[sl])["value"]
        if adv.std() > 1e-6:
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        adv = adv.clamp(-4.0, 4.0)

    ref_full = None
    if variant == "ref_cache":
        ref_full = ref_logits_full(ref_model, tiles, glob, n, glob.shape[1])

    model.train()
    perm = torch.randperm(n)
    for _ep in range(epochs):
        for i in range(0, n, batch):
            sel = perm[i:i + batch]
            if sel.numel() < 16:
                continue
            out = model(tiles[sel], glob[sel])
            logp_all = torch.zeros(sel.numel())
            ent_all = torch.zeros(sel.numel())
            kl_all = torch.zeros(sel.numel())

            # --- 参考 logits 的取法：这就是本脚本要对比的差异 ---
            ref_hoisted = None
            if variant == "baseline":
                pass                        # 保持"每头各调一次"的原始写法
            elif variant in ("ref_hoist", "cl"):
                with torch.no_grad():
                    ref_hoisted = ref_model(tiles[sel], glob[sel])
            elif variant == "ref_cache":
                ref_hoisted = "cached"

            for hid, name in enumerate(HEAD_NAMES[:-1]):
                sub = (head[sel] == hid).nonzero(as_tuple=True)[0]
                if sub.numel() == 0:
                    continue
                dim = HEAD_DIMS[name]
                lg = out[name][sub][:, :dim]
                m = mask[sel][sub][:, :dim].bool()
                lg = lg.masked_fill(~m, -1e9)
                lp = F.log_softmax(lg, dim=-1)
                p = lp.exp()
                ent_all[sub] = -(p * lp).nan_to_num(0).sum(-1)
                logp_all[sub] = lp.gather(1, act[sel][sub].unsqueeze(1)).squeeze(1)

                if ref_hoisted == "cached":
                    rlg = ref_full[name][sel][sub][:, :dim]
                elif ref_hoisted is not None:
                    rlg = ref_hoisted[name][sub][:, :dim]
                else:
                    with torch.no_grad():      # 原始写法：每个头各跑一次 backbone
                        rlg = ref_model(tiles[sel][sub], glob[sel][sub])[name][:, :dim]
                rlg = rlg.masked_fill(~m, -1e9)
                rlp = F.log_softmax(rlg, dim=-1)
                kl_all[sub] = (p * (lp - rlp)).nan_to_num(0).sum(-1)

            a = adv[sel]
            loss = -(logp_all * a).mean() \
                + 0.5 * F.smooth_l1_loss(out["value"], ret[sel]) \
                - 0.01 * ent_all.mean() \
                + 0.02 * kl_all.mean()
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
    return time.time() - t0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", nargs="*", default=["64,8,320", "160,20,512"])
    ap.add_argument("--samples", type=int, default=4300)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--variants", nargs="*", default=None)
    ap.add_argument("--reps", type=int, default=2,
                    help="每个变体重复测几轮，取最小值（笔记本上单次波动可达 ±50%%）")
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    all_variants = {
        "baseline":  "现状（参考前向写在头循环里，每头各调一次）",
        "ref_hoist": "参考前向提到头循环外，一次算全 batch",
        "ref_cache": "参考 logits 轮首算一次，两个 epoch 复用",
        "cl":        "ref_hoist + channels_last 内存格式",
        "batch1024": "baseline + batch 提到 1024",
        "threads4":  "baseline + 线程数降到 4（超线程争用）",
    }
    full_list = ["baseline", "ref_hoist", "ref_cache", "cl", "batch1024", "threads4"]
    if args.variants:
        variants = args.variants
    else:
        variants = full_list

    torch.set_num_threads(args.threads)
    ac = None
    try:  # 电池模式下笔记本会大幅降功耗墙，单次测量波动可达 ±50%
        import ctypes
        class _SPS(ctypes.Structure):
            _fields_ = [("ACLineStatus", ctypes.c_byte),
                        ("BatteryFlag", ctypes.c_byte),
                        ("BatteryLifePercent", ctypes.c_byte),
                        ("SystemStatusFlag", ctypes.c_byte),
                        ("BatteryLifeTime", ctypes.c_uint32),
                        ("BatteryFullLifeTime", ctypes.c_uint32)]
        _s = _SPS()
        ctypes.windll.kernel32.GetSystemPowerStatus(ctypes.byref(_s))
        ac = _s.ACLineStatus
        print(f"[电源] {'插电' if ac == 1 else '电池'}"
              f"（电量 {_s.BatteryLifePercent}%）"
              + ("" if ac == 1 else "  ← 强烈建议插电后再测，否则结果不可比"))
    except Exception:
        pass
    print(f"torch {torch.__version__}  threads={args.threads}  samples={args.samples} "
          f"batch={args.batch} epochs={args.epochs}  reps={args.reps}")

    rows = []
    for spec in args.configs:
        ch, bl, hd = (int(v) for v in spec.split(","))
        if args.variants:
            variants = args.variants
        else:
            variants = full_list if spec != "160,20,512" else \
                ["baseline", "ref_hoist", "ref_cache"]
        data = make_data(args.samples, GLOBAL_DIM)
        base_model = build_model(NetConfig(ch, bl, hd))
        n_params = base_model.n_params()
        ref_src = build_model(NetConfig(ch, bl, hd))
        ref_src.load_state_dict(base_model.state_dict())
        ref_src.eval()
        for p in ref_src.parameters():
            p.requires_grad_(False)

        print(f"\n== {spec}  {n_params / 1e6:.2f}M ==")
        print(f"{'变体':<12}{'秒/轮':>9}{'相对 baseline':>14}   说明")

        best: dict[str, float] = {v: float("inf") for v in variants}
        for _r in range(args.reps):
            for v in variants:
                batch = 1024 if v == "batch1024" else args.batch
                nthr = 4 if v == "threads4" else args.threads
                torch.set_num_threads(nthr)

                model = build_model(NetConfig(ch, bl, hd))
                model.load_state_dict(base_model.state_dict())
                ref = build_model(NetConfig(ch, bl, hd))
                ref.load_state_dict(ref_src.state_dict())
                ref.eval()
                for p in ref.parameters():
                    p.requires_grad_(False)
                if v == "cl":
                    model = model.to(memory_format=torch.channels_last)
                    ref = ref.to(memory_format=torch.channels_last)
                opt = torch.optim.AdamW(model.parameters(), lr=8e-6)

                run_variant(model, ref, opt, data, "baseline", 128, 1)   # warmup
                best[v] = min(best[v], run_variant(model, ref, opt, data, v,
                                                   batch, args.epochs))

        base_s = best["baseline"]
        for v in variants:
            gain = f"{base_s / best[v]:.3f}x"
            print(f"{v:<12}{best[v]:>9.1f}{gain:>14}   {all_variants[v]}")
            rows.append(dict(config=spec, params=n_params, variant=v,
                             seconds=best[v],
                             batch=(1024 if v == "batch1024" else args.batch),
                             threads=(4 if v == "threads4" else args.threads),
                             speedup=base_s / best[v]))

    if args.json:
        Path(args.json).write_text(
            json.dumps(dict(threads=args.threads, samples=args.samples,
                            rows=rows), ensure_ascii=False, indent=2),
            encoding="utf-8")
        print(f"\n-> {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
