"""回归闸门：`score_checkpoints.py` 的按副并行必须与单进程**逐位一致**。

为什么需要这个闸门
------------------
段末判读（`m1_segment_judge.py`）与「vs master 配对比」的**全部结论**都出自
`score_checkpoints.py`。我们在里面加了「按副切分的进程池」（`--workers`，
默认自动 = 核数上限 16），把 1200 副 6 模型从 139.6 分压到十几分钟。

并行本身是安全的：第 r 副牌只依赖 ``seed0 + r*1013`` 与对手名字推出的种子，
副与副之间没有任何共享状态。但这句话一旦哪天被改动破坏（比如有人在副之间
引入共享的 RNG、缓存或模型状态），**判读结论会静默偏移** —— 而这种偏移不会
有任何报错。所以用本闸门把它钉死：

    同一副牌池、同一牌组，`--workers 1` 与 `--workers 4` 的
    ``rows``（含 CI 与 verdict）与 ``per_deal``（逐副明细）必须**完全相同**。

用法::

    python scripts/score_parallel_check.py            # 48 局 + 进程池启动，约 25 s
    python scripts/score_parallel_check.py --deals 8  # 更快

退出码 0 = 全部通过。
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable
REF = "teacher:balanced|teacher:tenpai_rush|teacher:defensive"

# 必须选**两个真正不同**的模型：若选到同权重的两个 stem（例如 master 与 big_s11
# 其实是同一份），分差恒为 0，闸门就退化成「什么都没验」。下面第 5 条专门守这一点。
CKPTS = ["dushan_big_s11", "dushan_big_s4"]
ANCHOR = "dushan_big_s11"

_results: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    _results.append((name, ok, detail))
    print(f"  {'PASS' if ok else 'FAIL'}  {name}" + (f"：{detail}" if detail else ""))


def run(deals: int, workers: int, out: Path) -> dict:
    cmd = [PY, str(ROOT / "scripts" / "score_checkpoints.py"),
           "--ckpts", ",".join(CKPTS), "--ref", REF,
           "--deals", str(deals), "--threads", "1",
           "--workers", str(workers), "--worker-threads", "1",
           "--anchor", ANCHOR, "--out", str(out)]
    r = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True)
    if r.returncode != 0 or not out.exists():
        print(r.stdout[-2000:])
        print(r.stderr[-2000:])
        raise SystemExit(f"score_checkpoints 退出码 {r.returncode}")
    return json.loads(out.read_text(encoding="utf-8"))


def main() -> int:
    ap = argparse.ArgumentParser(description="按副并行 vs 单进程 一致性闸门")
    ap.add_argument("--deals", type=int, default=6)
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    print(f"\n{'=' * 72}\n判读并行一致性：--workers 1  vs  --workers {args.workers}"
          f"（{args.deals} 副）\n{'=' * 72}")

    with tempfile.TemporaryDirectory() as td:
        p1, pw = Path(td) / "w1.json", Path(td) / f"w{args.workers}.json"
        a = run(args.deals, 1, p1)
        b = run(args.deals, args.workers, pw)

    check("两个产物的 run 元信息一致（deals/seed0/anchor/ref_field）",
          all(a[k] == b[k] for k in ("deals", "seed0", "anchor", "ref_field")),
          f"deals={a['deals']} seed0={a['seed0']}")
    check("rows 逐字段相同（含 level / CI / verdict）", a["rows"] == b["rows"],
          "; ".join(f"{r['ckpt'].split('/')[-1]} {r['vs_anchor']:+.3f}"
                    for r in a["rows"]))
    same_pd = all(a["per_deal"][c] == b["per_deal"][c] for c in a["per_deal"])
    n_deals = len(next(iter(a["per_deal"].values())))
    check(f"per_deal 逐副逐模型完全相同（{len(a['per_deal'])} 模型 × {n_deals} 副）",
          same_pd)
    mx = max((abs(x - y) for c in a["per_deal"]
              for x, y in zip(a["per_deal"][c], b["per_deal"][c])), default=0.0)
    check("最大逐副绝对差 = 0", mx == 0.0, f"max|Δ| = {mx}")
    check("并行路径确实记录了 workers", b.get("workers") == args.workers,
          f"workers={b.get('workers')}")
    # 守「闸门本身有分辨力」：两个检查点要是同一份权重，上面的比较全是 0，等于空跑。
    spread = max((abs(x - y)
                  for c in a["per_deal"] for x, y in zip(a["per_deal"][c],
                                                         a["per_deal"][ANCHOR])
                  if c != ANCHOR), default=0.0)
    check("牌组有分辨力（检查点之间存在非零分差）", spread > 0.0,
          f"最大分差 {spread:.3f}")
    check("锚点行恒为 0（自比自身）",
          next(r for r in a["rows"] if r["ckpt"] == ANCHOR)["vs_anchor"] == 0.0)

    ok = all(x[1] for x in _results)
    print(f"\n{'=' * 72}")
    print(f"结果：{sum(1 for x in _results if x[1])}/{len(_results)} 通过"
          + ("" if ok else "  ⛔ 并行与单进程不一致！"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
