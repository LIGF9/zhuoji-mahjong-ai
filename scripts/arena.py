"""
对战评测：把待测 Agent 和一组基准放在同一副牌上轮换座位对打。

用法::

    python scripts/arena.py --model models/bc.pt --rounds 150
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from zhuoji import RulesConfig  # noqa: E402
from zhuoji.agent import NetAgent, play_duplicate, print_table  # noqa: E402
from zhuoji.bots import (  # noqa: E402
    TEACHER_STYLES, HeuristicBot, RandomBot,
)
from zhuoji.net import NetConfig, build_model  # noqa: E402


def load_model(path: str, threads: int | None = None):
    if threads:
        # 之前这里收了 threads 却没用，导致 --threads 静默失效
        torch.set_num_threads(threads)
    ck = torch.load(path, map_location="cpu", weights_only=False)
    nc = ck.get("net") or {"channels": 48, "blocks": 8, "hidden": 256}
    model = build_model(NetConfig(nc["channels"], nc["blocks"], nc["hidden"]))
    model.load_state_dict(ck["state_dict"])
    model.eval()
    return model, ck


def det_seed(name: str, i: int) -> int:
    """由「对手名 + 第几个」推出确定性的种子。

    之前用 ``torch.randint`` 现场抽种子，意味着每次运行对手都是不同实例，
    两份 arena.json（比如 bc.pt 与 rl.pt）虽然吃的是一模一样的牌，
    对手行为却不同 —— 于是**没法做逐副牌配对比较**，白白丢掉最大的降方差手段。
    改成确定性种子后，同一条命令的两次运行完全可复现，也让跨模型配对成为可能。
    """
    h = hashlib.sha256(f"{name}#{i}".encode("utf-8")).hexdigest()
    return int(h[:8], 16) % (10 ** 6)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=str(ROOT / "models" / "bc.pt"))
    ap.add_argument("--rounds", type=int, default=120)
    ap.add_argument("--seed", type=int, default=777)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--out", default=str(ROOT / "reports" / "arena.json"))
    ap.add_argument("--only", default="", help="逗号分隔的对手列表，默认全部")
    ap.add_argument("--baseline", default="",
                    help="再载入一个模型作为对手（例如 models/bc.pt），"
                         "用来在同一份结果里对比强化前后的两代模型")
    args = ap.parse_args()

    model, ck = load_model(args.model, args.threads)
    me = NetAgent(model, seed=0, temperature=args.temperature,
                  name=f"net:{Path(args.model).stem}")

    pool = {
        "random": lambda i: RandomBot(seed=det_seed("random", i)),
        **{f"teacher:{k}": (lambda kk: (lambda i: HeuristicBot(
            seed=det_seed(f"teacher:{kk}", i), weights=TEACHER_STYLES[kk])))(k)
           for k in TEACHER_STYLES},
    }
    if args.baseline and Path(args.baseline).exists():
        bmodel, _bck = load_model(args.baseline, args.threads)
        # 名字别用 "bc:bc" 这种自我重复的写法，报告里读起来像笔误
        bname = f"prev:{Path(args.baseline).stem}"

        def baseline_factory(i, _m=bmodel, _n=bname):
            # 基线是确定性策略（temperature=0），不吃 seed，但签名要和其它工厂一致
            return NetAgent(_m, seed=0, temperature=args.temperature, name=_n)

        pool[bname] = baseline_factory
        print(f"基线对手已加入：{bname}（{args.baseline}）")
    if args.only:
        want = set(args.only.split(","))
        pool = {k: v for k, v in pool.items() if k in want}

    print(f"载入模型 {args.model}  (epoch={ck.get('epoch')}, val_loss={ck.get('val_loss')})")
    print(f"复式评测: {args.rounds} 副牌 × 4 座位轮换\n")

    all_results = {}
    all_per_deal = {}
    for opp_name, factory in pool.items():
        def make(factory=factory, opp_name=opp_name):
            # 三个对手共用 opp_name：play_duplicate 按 name 聚合统计，
            # 同名即得到"我方 vs 3×该对手"的合并结果（这正是评测想看的量）。
            bots = [me]
            for i in range(3):
                b = factory(i)
                b.name = opp_name
                bots.append(b)
            return bots

        res, per_deal = play_duplicate(make, args.rounds, seed0=args.seed,
                                       return_per_deal=True)
        print(f"--- vs 3×{opp_name} ---")
        print_table(res)
        print()
        all_results[opp_name] = res
        all_per_deal[opp_name] = {k: [round(x, 4) for x in v]
                                  for k, v in per_deal.items()}

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps({
        "model": args.model,
        "rounds": args.rounds,
        "seed": args.seed,
        "epoch": ck.get("epoch"),
        "val_loss": ck.get("val_loss"),
        "results": all_results,
        # 逐副牌得分：供跨模型做配对比较（compare_arena.py）
        "per_deal": all_per_deal,
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print("结果已写入", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
