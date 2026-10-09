"""
跨模型配对比较两份 arena.json 的结果。

为什么需要它
------------
``arena.py`` 单跑一份，只能给出"被测模型 vs 固定对手"的均值与标准误。
但捉鸡单局得分方差极大，几百副牌算出来的标准误常在 0.15 上下，
而两代模型之间的真实差距可能只有 0.1~0.2 —— **单看两份评测根本分不出胜负**。

关键点在于：两份评测吃的是**同一批牌**（``--seed`` 与 ``--rounds`` 相同），
只要对手也用确定性种子生成（``arena.det_seed``），那么每一副牌上
"牌运"这个最大的噪声源在两次运行里是同一个值，逐副牌做差就能把它消掉。
这正是复式思想的延伸：**不只是对同一手牌轮换座位，还要让两个模型去坐同一把椅子**。

用法::

    python scripts/compare_arena.py --a reports/arena.json --b reports/arena_bc.json
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def load(path: str) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def net_name(results: dict) -> str | None:
    for n in results:
        if n.startswith("net"):
            return n
    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", default=str(ROOT / "reports" / "arena.json"),
                    help="第一份评测（通常是新一代模型）")
    ap.add_argument("--b", default=str(ROOT / "reports" / "arena_bc.json"),
                    help="第二份评测（通常是基线）")
    ap.add_argument("--out", default=str(ROOT / "reports" / "compare.json"))
    ap.add_argument("--label-a", default="", help="A 的显示名，默认取模型文件名")
    ap.add_argument("--label-b", default="", help="B 的显示名")
    args = ap.parse_args()

    A, B = load(args.a), load(args.b)
    la = args.label_a or Path(A["model"]).name
    lb = args.label_b or Path(B["model"]).name

    if (A.get("seed"), A.get("rounds")) != (B.get("seed"), B.get("rounds")):
        print("[warn] 两份评测的 seed/rounds 不一致，逐副牌配对不成立，结果仅供参考",
              file=sys.stderr)
    if not (A.get("per_deal") and B.get("per_deal")):
        raise SystemExit("两份评测里至少一份没有 per_deal 字段；"
                         "请用带逐副牌记录的 arena.py 重新生成。")

    common = [k for k in A["results"] if k in B["results"]
              and not k.startswith("prev:")]
    rows = []
    for opp in common:
        na, nb = net_name(A["results"][opp]), net_name(B["results"][opp])
        da = (A["per_deal"].get(opp) or {}).get(na or "")
        db = (B["per_deal"].get(opp) or {}).get(nb or "")
        if not da or not db or len(da) != len(db):
            continue
        diffs = [x - y for x, y in zip(da, db)]
        n = len(diffs)
        mean = sum(diffs) / n
        var = sum((d - mean) ** 2 for d in diffs) / (n - 1) if n > 1 else 0.0
        se = math.sqrt(var / n) if n > 1 else 0.0
        t = mean / se if se > 0 else 0.0
        rows.append({"opponent": opp, "n_deals": n, "mean_diff": mean,
                     "se": se, "t": t,
                     # 配对差的标准误比"两份独立均值各算 SE 再相加"小得多，
                     # 顺带把方差缩减比例也记下来，能直观看出配对值不值
                     "var_reduction": 1.0 - (var / n) / max(
                         1e-12, (A["results"][opp][na]["avg_score_se"] ** 2
                                 + B["results"][opp][nb]["avg_score_se"] ** 2))})

    if not rows:
        raise SystemExit("没有可配对的对手（两份评测的对手列表没有交集）。")

    # 逆方差加权合并：每个对手一个估计，方差小的权重高
    wsum = sum(1.0 / r["se"] ** 2 for r in rows if r["se"] > 0)
    comb = (sum(r["mean_diff"] / r["se"] ** 2 for r in rows if r["se"] > 0) / wsum
            if wsum > 0 else 0.0)
    comb_se = math.sqrt(1.0 / wsum) if wsum > 0 else 0.0
    comb_t = comb / comb_se if comb_se > 0 else 0.0

    print(f"A = {la}   B = {lb}   （同一批牌，逐副牌配对）\n")
    print(f"{'对手':<26}{'副牌数':>8}{'差值(A-B)':>12}{'标准误':>10}{'t':>8}{'方差缩减':>10}")
    for r in rows:
        print(f"{r['opponent']:<26}{r['n_deals']:>8}{r['mean_diff']:>+12.3f}"
              f"{r['se']:>10.3f}{r['t']:>8.2f}{r['var_reduction']:>9.0%}")
    print(f"\n逆方差加权合并：{comb:+.3f} ± {comb_se:.3f}  (t = {comb_t:+.2f})")
    print("（注意：各对手共用同一批牌，合并估计假设各对手间近似独立，"
          "只能当参考值，别当精确置信区间用）")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps({
        "label_a": la, "label_b": lb,
        "seed": A.get("seed"), "rounds": A.get("rounds"),
        "rows": rows,
        "combined": {"mean_diff": comb, "se": comb_se, "t": comb_t},
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print("\n对比结果已写入", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
