#!/usr/bin/env python3
"""候选池 best-of 第一段：用冻结池给 r01~rNN / pNN 排名并挑出复赛候选（只读 · 零 CPU）。

为什么需要（reports/终评执行手册-2026-11-13.md §8.1）：
    闸门 A 只判「新贪心 vs master 贪心」，**不含「必须优于上轮最佳」**⇒ 每轮都通过、
    exit_best 每轮被覆盖、强度在 seed 池噪声上随机游走。跑 40+ 轮后交付的 exit_best 只是
    「最后一轮」，不是「历史最佳」。必须从全部轮次里挑。

为什么可以离线零成本做：
    rNN_gateA.jsonl 全部在**同一个** 300 seed 池（710000+300）、**同一个**对手
    （models/dushan_master.pt 贪心）、**同样**四座旋转下产出 ⇒ 轮与轮之间逐 seed **天然配对**，
    直接做配对差分即可，不需要重跑任何对局。

口径（唯一正确，自检 #7 踩坑后固化）：
    d = deltas[cand_seat] − mean(其余三家)，同一 seed 四座平均 ⇒ 1 个独立样本
    再在 seed 维度算 mean / SE / z / 双侧正态近似 p
  ❌ deltas[0]、不做四座平均 ⇒ 全错。

输出：
    1) 全轮次排名表（mean±SE）
    2) top-k 候选 + 第二段独立池复赛命令（逐条可直接粘贴）
    3) top1 对其余候选的逐 seed 配对差分（判断「真领先」还是「并列」）
    4) 选择偏差量级估计（N 个候选取 max 的期望上偏 ≈ SE·√(2lnN)）
  top1−top2 < 1 SE ⇒ 必须写「并列」，不得宣称某轮显著更优。

用法::

    ./.venv/bin/python tools/bestof_rank.py --md                # 全部轮次
    ./.venv/bin/python tools/bestof_rank.py --topk 5 --md
    ./.venv/bin/python tools/bestof_rank.py --rounds 3-9 --md
    ./.venv/bin/python tools/bestof_rank.py --json runs/gates/bestof.json

只读 jsonl，不加载模型、不启对局、可在 7×24 标注期安全运行。
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
from final_eval_report import per_seed, stats  # noqa: E402  唯一实现复用

GATES = ROOT / "runs" / "gates"
FROZEN_SEEDS = 300      # 闸门 A 冻结池 710000+300
FROZEN_SEED0 = 710000
INDEP_SEEDS = 500       # 独立复赛池 3_000_000+500
INDEP_SEED0 = 3_000_000
MASTER = "models/dushan_master.pt"


def _parse_rounds(spec: str | None) -> set[int] | None:
    if not spec:
        return None
    out: set[int] = set()
    for part in spec.split(","):
        part = part.strip()
        if "-" in part:
            a, b = part.split("-", 1)
            out.update(range(int(a), int(b) + 1))
        elif part:
            out.add(int(part))
    return out


def _complete(path: Path, n_min: int) -> bool:
    """只接受已跑完的 jsonl：唯一 seed 数 ≥ n_min（闸门 A = 300）。"""
    if not path.exists() or path.stat().st_size == 0:
        return False
    seeds: set[int] = set()
    try:
        with path.open(encoding="utf-8") as f:
            for raw in f:
                raw = raw.strip()
                if not raw:
                    continue
                try:
                    seeds.add(int(json.loads(raw)["seed"]))
                except (KeyError, TypeError, ValueError):
                    continue
    except OSError:
        return False
    return len(seeds) >= n_min


def discover(rounds: set[int] | None, n_min: int) -> list[tuple[str, int, Path]]:
    """返回 [(label, round, path)]，含 DAgger rNN 与 PPO pNN 两类闸门 A。"""
    found: list[tuple[str, int, Path]] = []
    for path in sorted(GATES.glob("*_gateA.jsonl")):
        m = re.fullmatch(r"([rp])(\d{2})_gateA\.jsonl", path.name)
        if not m:
            continue
        kind, rtxt = m.group(1), int(m.group(2))
        if rounds is not None and rtxt not in rounds:
            continue
        if not _complete(path, n_min):
            print(f"  [跳过] {path.name}: 未跑完（seed < {n_min}）")
            continue
        found.append((f"{kind.upper()}{rtxt:02d}", rtxt, path))
    return found


def pair(a: dict[int, float], b: dict[int, float], label: str) -> dict | None:
    common = sorted(set(a) & set(b))
    if len(common) < 2:
        return None
    s = stats([a[k] - b[k] for k in common])
    s["label"] = label
    s["common"] = len(common)
    return s


def fmt(s: dict, name_w: int = 26) -> str:
    return (f"{s['label']:<{name_w}} n={s['n']:<4} diff={s['mean']:+.3f}±{s['se']:.3f}  "
            f"z={s['z']:+.2f}  p≈{s['p']:.4f}  符号 {s['pos']}:{s['neg']}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="候选池 best-of 第一段（冻结池排名，只读）")
    ap.add_argument("--rounds", default=None, help="限定轮次，如 3-9 或 4,6,8")
    ap.add_argument("--topk", type=int, default=5, help="进入第二段的候选数（默认 5）")
    ap.add_argument("--min-seeds", type=int, default=FROZEN_SEEDS)
    ap.add_argument("--json", default=None, help="把排名结果写成 json（可选）")
    ap.add_argument("--md", action="store_true")
    ap.add_argument("--include-ppo", action="store_true", default=True,
                    help="包含 PPO 轮（pNN_gateA，同池同对手，默认包含）")
    ap.add_argument("--no-ppo", dest="include_ppo", action="store_false")
    args = ap.parse_args(argv)

    rounds = _parse_rounds(args.rounds)
    entries = discover(rounds, args.min_seeds)
    if not args.include_ppo:
        entries = [e for e in entries if not e[0].startswith("P")]
    if not entries:
        print("[ERROR] 没有可用的完整闸门 A jsonl", file=sys.stderr)
        return 2

    print("=" * 96)
    print("候选池 best-of · 第一段（冻结池 710000+300 · 对手 master 贪心 · 四座旋转 · 只读）")
    print("=" * 96)

    series: list[dict] = []
    for label, r, path in entries:
        d = per_seed(path)
        s = stats(list(d.values()))
        s["label"] = label
        s["round"] = r
        s["kind"] = label[0]
        s["path"] = str(path.relative_to(ROOT))
        series.append({"label": label, "round": r, "kind": label[0],
                       "path": path, "seeds": d, "stat": s})

    series.sort(key=lambda x: -x["stat"]["mean"])
    n = len(series)
    print(f"-- 排名（共 {n} 轮完整闸门 A）--")
    for i, e in enumerate(series, 1):
        print(f"  #{i:<2} " + fmt(e["stat"]))

    top = series[0]
    runner = series[1] if n > 1 else None
    print("\n-- top1 对其余候选的逐 seed 配对差分（正数=top1 更强）--")
    proc: list[dict] = []
    for e in series[1:]:
        s = pair(top["seeds"], e["seeds"], f"{top['label']}−{e['label']}")
        if s is None:
            continue
        proc.append(s)
        print("  " + fmt(s) + f"   共同 seed={s['common']}")

    # 选择偏差：N 个独立 ~N(0,SE) 候选取 max 的期望上偏 ≈ SE·√(2ln N)
    se_ref = top["stat"]["se"]
    bias = se_ref * math.sqrt(2 * math.log(max(n, 2)))
    print(f"\n-- 选择偏差量级 --  N={n} 候选在纯噪声下取 max 的期望上偏 ≈ "
          f"{bias:.3f} 分 ≈ {bias / se_ref:.2f} SE（对 {top['label']} 的 {top['stat']['mean']:+.3f}）")

    verdict = None
    if runner is not None:
        d12 = pair(top["seeds"], runner["seeds"], f"{top['label']}−{runner['label']}")
        if d12 is not None:
            gap = d12["mean"]
            if gap < d12["se"]:
                verdict = (f"并列：{top['label']} − {runner['label']} = {gap:+.3f}±{d12['se']:.3f}"
                           f"（< 1 SE, p≈{d12['p']:.3f}）⇒ 禁止宣称某轮显著更优，"
                           f"第二段必须用独立池复赛定交付")
            elif d12["p"] < 0.05:
                verdict = (f"显著领先：{top['label']} − {runner['label']} = "
                           f"{gap:+.3f}±{d12['se']:.3f}（p≈{d12['p']:.3f}）；仍需独立池复核选择偏差")
            else:
                verdict = (f"领先但未显著：{top['label']} − {runner['label']} = "
                           f"{gap:+.3f}±{d12['se']:.3f}（p≈{d12['p']:.3f}）⇒ 按并列处理")
    print("\n-- 判定 --")
    print("  " + (verdict or "仅 1 轮可用，无法比较"))

    topk = series[:args.topk]
    print(f"\n-- 第二段：独立池 {INDEP_SEED0}+{INDEP_SEEDS} 复赛（串行，禁止并发抢核）--")
    for e in topk:
        ck = ROOT / ("models/exit_p%02d.pt" % e["round"] if e["kind"] == "P"
                     else "models/exit_r%02d.pt" % e["round"])
        out = ROOT / f"runs/gates/final_cand_{e['label'].lower()}.jsonl"
        flag = "" if ck.exists() else "   # [警告] ckpt 不存在"
        print(f"  ./.venv/bin/python eval_pairs.py --a {ck.relative_to(ROOT)} --b {MASTER} "
              f"--mode greedy --seeds {INDEP_SEEDS} --seed0 {INDEP_SEED0} --workers 6 "
              f"--out {out.relative_to(ROOT)}{flag}")

    if args.md:
        rows = ["| 名次 | 候选 | n | diff ± SE (vs master) | z | p | 符号 |",
                "|---|---|---|---|---|---|---|"]
        for i, e in enumerate(series, 1):
            s = e["stat"]
            rows.append(f"| {i} | {e['label']} | {s['n']} | {s['mean']:+.3f}±{s['se']:.3f} | "
                        f"{s['z']:+.2f} | {s['p']:.4f} | {s['pos']}:{s['neg']} |")
        print("\n```markdown\n" + "\n".join(rows) + "\n```")
        if proc:
            mrows = ["| 配对 | n | diff ± SE | z | p | 结论 |", "|---|---|---|---|---|---|"]
            for s in proc:
                concl = "并列(<1SE)" if s["mean"] < s["se"] else (
                    "显著" if s["p"] < 0.05 else "未显著")
                mrows.append(f"| {s['label']} | {s['n']} | {s['mean']:+.3f}±{s['se']:.3f} | "
                             f"{s['z']:+.2f} | {s['p']:.4f} | {concl} |")
            print("\n```markdown\n" + "\n".join(mrows) + "\n```")

    if args.json:
        out = Path(args.json)
        if not out.is_absolute():
            out = ROOT / out
        out.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "pool": {"seed0": FROZEN_SEED0, "seeds": FROZEN_SEEDS, "baseline": MASTER},
            "generated_by": "tools/bestof_rank.py",
            "ranking": [{"rank": i, "label": e["label"], "round": e["round"],
                         "kind": e["kind"], "path": e["path"], "stat": e["stat"]}
                        for i, e in enumerate(series, 1)],
            "top_pairwise": proc,
            "selection_bias_est": bias,
            "topk_for_stage2": [{"label": e["label"], "round": e["round"], "kind": e["kind"]}
                                for e in topk],
            "verdict": verdict,
            "caveat": "第一段复用晋升判据同一 300 seed 池 ⇒ 存在多重比较选择偏差，"
                      "交付前必须用独立池 3000000+500 复赛并由 final_eval_report.py --pair 判定",
        }
        out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n[JSON] {out.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
