"""逐代检查点「共用牌池阶梯检验」：S3/S7/S11 是不是真的逐代变强。

回答的问题
----------
``dushan_big_s3 → s7 → s11`` 是一条连续训练链上的三代权重（s3 iter 1775 →
s4 → s5 → s6 → s7 280 → s8 → s9 → s10 20 → s11 120）。用户体感"一代比一代强"，
但既有测量都不足以支持或否定这个结论，而且各自有硬伤：

* ``duel.py``（s11 vs s7，2400 副）= +0.319 ± 0.288；``duel.py``（s3 vs s7，600 副）
  = +0.404 ± 0.556。方向互相矛盾且 CI 巨大。根因：``play_duplicate`` 把座位 shift
  混进了牌 seed（``seed0 + r * 1013 + shift``），四轮吃的是**四副不同的牌**，
  座位偏差根本没被抵消 —— 它根本不是复式。
* ``score_checkpoints.py`` 天梯（200 副）用 ``dushan_big_s7`` 当对手池的一员，
  **让 s7 打自己的镜像**，s7 的分被系统性压低（天梯里 s7 排第 9「显著劣」，
  但 s7−s3 只有 +0.156）。对手池含被测模型是致命伤。

两种评测设计（本脚本都支持，实测方差差 3 倍）
---------------------------------------------
``--mode coop``（默认，推荐）
    同一副牌、同一座位、同一对手种子，**每个检查点各自打一局**（三个检查点不坐在
    同一张桌上）。于是两个检查点的分差是**共享牌运、彼此不竞争**的配对观测
    （分数正相关，ρ≈0.43）→ 配对差 sd 实测 **≈6.0**。代价是每副牌要跑
    ``4 座位 × N 检查点`` 局。
``--mode arena``
    三个检查点 + 一个中立对手**同坐一桌**。每副牌只要 4 局，但三者在同一局里
    争夺同一份分（负相关）→ 配对差 sd 实测 **≈10.4–11.0**，比 coop 大 1.75×。
    每副局数少 3 倍，恰好抵消方差劣势，总成本相当；coop 的数值更干净，
    且能顺带给出"跨对手层"的稳健性证据。

**无论哪种模式，对手池都禁止包含被测检查点**（脚本会直接断言拦下）——
这是天梯踩过的坑。

判据
----
"逐代稳定变强"成立需要**两个相邻增量都显著为正**：
``S7 − S3 > 0`` 且 ``S11 − S7 > 0``（各自 95% CI 不含 0），
并且逐副严格单调比例显著高于随机（精确符号检验）。

用法::

    # pilot：先测方差与速度
    python scripts/ladder_test.py --ckpts dushan_big_s3,dushan_big_s7,dushan_big_s11 \
        --deals 200 --out reports/_ladder_pilot.json

    # 正式检验
    python scripts/ladder_test.py --ckpts dushan_big_s3,dushan_big_s7,dushan_big_s11 \
        --deals 6000 --out reports/ladder_s3_s7_s11.json

    # 换一套对手场 / 换牌池 seed 做稳健性复验
    python scripts/ladder_test.py --ckpts dushan_big_s3,dushan_big_s7,dushan_big_s11 \
        --deals 3000 --opp teacher:defensive,teacher:gambler,teacher:chicken_lover \
        --seed 90210 --out reports/ladder_altfield.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics as st
import sys
import time

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from zhuoji.agent import NetAgent  # noqa: E402
from zhuoji.bots import TEACHER_STYLES, HeuristicBot  # noqa: E402
from zhuoji.dushan import DushanConfig, DushanGame  # noqa: E402
from zhuoji.net import NetConfig, build_model  # noqa: E402

DEFAULT_OPPS = ["teacher:balanced", "teacher:aggressive", "teacher:tenpai_rush"]

_MODELS: dict[str, tuple] = {}


def load(stem: str):
    """按裸文件名加载；返回 (model, ckpt_meta)。同一 stem 只读一次。"""
    if stem in _MODELS:
        return _MODELS[stem]
    path = ROOT / "models" / f"{stem}.pt"
    if not path.exists():
        raise SystemExit(f"找不到模型 {path}")
    ck = torch.load(path, map_location="cpu", weights_only=False)
    nc = ck.get("net") or {"channels": 48, "blocks": 8, "hidden": 256}
    m = build_model(NetConfig(nc["channels"], nc["blocks"], nc["hidden"]))
    m.load_state_dict(ck["state_dict"])
    m = m.eval()
    for p in m.parameters():
        p.requires_grad_(False)
    _MODELS[stem] = (m, ck)
    return _MODELS[stem]


def _seed(*parts) -> int:
    """由名字推出确定性种子 —— 保证对手行为跨副可复现、跨检查点一致。"""
    h = hashlib.sha256("#".join(str(p) for p in parts).encode("utf-8")).hexdigest()
    return int(h[:8], 16) % (10 ** 9)


def make_opponent(spec: str, seat_seed: int):
    if spec.startswith("teacher:"):
        style = spec.split(":", 1)[1]
        if style not in TEACHER_STYLES:
            raise SystemExit(f"未知教师风格 {style!r}，可选：{list(TEACHER_STYLES)}")
        b = HeuristicBot(seed=seat_seed, weights=TEACHER_STYLES[style])
        b.name = spec
        return b
    stem = spec.split(":", 1)[1] if spec.startswith("net:") else spec
    m, _ = load(stem)
    return NetAgent(m, seed=seat_seed, temperature=0.0, name=spec)


def _wrap(bot):
    return lambda game: bot.choose(game)


def opp_stem_of(spec: str) -> str:
    return spec.split(":", 1)[1] if spec.startswith("net:") else spec


# --------------------------------------------------------------------------- #
# 统计工具
# --------------------------------------------------------------------------- #
def binom_p(k: int, n: int, p0: float = 0.5) -> float:
    """精确二项检验（method of small p-values）：X ~ Bin(n, p0) 观测到 k 的 p 值。

    ``p0=0.5`` 用于逐副胜负方向（符号检验）；
    ``p0=1/6`` 用于「三个模型逐副严格递增」的比例 —— 随机基准不是 1/2 而是 1/6
    （三个模型的全排列有 6 种等可能，只有 1 种是严格递增）。

    用对数实现，n 上万也稳定（直接算 ``comb`` 会溢出到无法比较）。
    """
    if n <= 0:
        return 1.0
    if not 0.0 < p0 < 1.0:
        return 1.0
    lp0, lq0 = math.log(p0), math.log1p(-p0)
    lg = math.lgamma

    def logpmf(i: int) -> float:
        return (lg(n + 1) - lg(i + 1) - lg(n - i + 1)
                + i * lp0 + (n - i) * lq0)

    obs = logpmf(k)
    tot = 0.0
    for i in range(n + 1):
        lp = logpmf(i)
        if lp <= obs + 1e-9:
            tot += math.exp(lp)
    return min(1.0, tot)


def binom_ge(k: int, n: int, p0: float) -> float:
    """单侧 P(X >= k)，X ~ Bin(n, p0) —— 用于「高于随机基准」的方向性检验。"""
    if n <= 0:
        return 1.0
    lp0, lq0 = math.log(p0), math.log1p(-p0)
    lg = math.lgamma
    tot = 0.0
    for i in range(k, n + 1):
        tot += math.exp(lg(n + 1) - lg(i + 1) - lg(n - i + 1)
                        + i * lp0 + (n - i) * lq0)
    return min(1.0, tot)


def paired_stats(a: np.ndarray, b: np.ndarray, boot: int = 10000) -> dict:
    """配对净分差 a − b：均值、正态 CI、bootstrap CI、符号检验。"""
    d = np.asarray(a, float) - np.asarray(b, float)
    n = d.size
    mean = float(d.mean())
    sd = float(d.std(ddof=1)) if n > 1 else 0.0
    se = sd / math.sqrt(n) if n else 0.0
    if n > 1 and boot > 0:
        rng = np.random.default_rng(20261009)
        idx = rng.integers(0, n, size=(boot, n))
        bs = d[idx].mean(axis=1)
        blo, bhi = (float(x) for x in np.percentile(bs, [2.5, 97.5]))
    else:
        blo = bhi = mean
    pos = int((d > 1e-12).sum())
    neg = int((d < -1e-12).sum())
    verdict = ("显著优" if mean - 1.96 * se > 0 else
               "显著劣" if mean + 1.96 * se < 0 else "无显著差异")
    return {
        "n_deals": int(n),
        "mean": round(mean, 4),
        "sd": round(sd, 4),
        "se": round(se, 4),
        "ci95": round(1.96 * se, 4),
        "boot_ci95": [round(blo, 4), round(bhi, 4)],
        "wins_deals": pos,
        "loss_deals": neg,
        "win_share": round(pos / max(1, pos + neg), 4),
        "sign_p": round(binom_p(pos, pos + neg, 0.5), 6),
        "verdict": verdict,
        "significant": bool(mean - 1.96 * se > 0 or mean + 1.96 * se < 0),
    }


def level_stats(v: np.ndarray) -> dict:
    n = v.size
    sd = float(v.std(ddof=1)) if n > 1 else 0.0
    se = sd / math.sqrt(n) if n else 0.0
    return {"mean": round(float(v.mean()), 4), "sd": round(sd, 4),
            "se": round(se, 4), "ci95": round(1.96 * se, 4)}


def deals_for_ci(sd: float, target_ci: float) -> int:
    """达到目标 95% CI 半宽所需的副数。"""
    se = target_ci / 1.96
    return int(math.ceil((sd / se) ** 2)) if sd > 0 and se > 0 else 0


# --------------------------------------------------------------------------- #
# 主评测
# --------------------------------------------------------------------------- #
def run(ckpts: list[str], opps: list[str], deals: int, seed0: int, threads: int,
        mode: str, layout: int, progress_every: int = 100,
        boot: int = 10000) -> dict:
    # 对手池不得包含任何被测检查点（天梯踩过的坑：s7 打自己的镜像）
    bad = [o for o in opps if opp_stem_of(o) in set(ckpts)]
    if bad:
        raise SystemExit(f"对手池里不能有被测检查点：{bad}")

    torch.set_num_threads(threads)
    loaded = {c: load(c) for c in ckpts}

    per_deal: dict[str, list[float]] = {c: [] for c in ckpts}
    dealt_opp: list[str] = []
    seat_hist: dict[str, list[int]] = {c: [0] * 4 for c in ckpts}
    games = 0

    t0 = time.time()
    for r in range(deals):
        deal_seed = seed0 + r * 1013          # 同一副牌：与检查点无关
        acc: dict[str, list[float]] = {c: [] for c in ckpts}

        if mode == "coop":
            # 同副牌 / 同座位 / 同对手种子，每个检查点各打一局（互不在同一桌）
            opp_order = [opps[(i + r) % len(opps)] for i in range(3)]
            for shift in range(4):
                free = [s for s in range(4) if s != shift]
                for c in ckpts:
                    seat_agents = [None] * 4
                    for k, slot in enumerate(free):
                        spec = opp_order[k]
                        seat_agents[slot] = make_opponent(
                            spec, _seed(r, shift, slot, spec))
                    seat_agents[shift] = NetAgent(
                        loaded[c][0], seed=0, temperature=0.0, name=c)
                    game = DushanGame(DushanConfig(), seed=deal_seed)
                    res = game.play([_wrap(a) for a in seat_agents])
                    acc[c].append(res["deltas"][shift])
                    seat_hist[c][shift] += 1
                    games += 1
        else:  # arena：三个检查点 + 一个中立对手同坐一桌
            opp_spec = opps[r % len(opps)]
            if layout == 12:
                combos = [(os_, pm) for os_ in range(4) for pm in range(len(ckpts))]
            else:
                combos = [((k + r) % 4, (k + r) % len(ckpts)) for k in range(4)]
            for opp_seat, perm in combos:
                remaining = [s for s in range(4) if s != opp_seat]
                order = ckpts[perm:] + ckpts[:perm]
                seat_agents = [None] * 4
                seat_agents[opp_seat] = make_opponent(
                    opp_spec, _seed(r, opp_seat, perm, opp_spec))
                for seat, cname in zip(remaining, order):
                    seat_agents[seat] = NetAgent(
                        loaded[cname][0], seed=0, temperature=0.0, name=cname)
                    seat_hist[cname][seat] += 1
                game = DushanGame(DushanConfig(), seed=deal_seed)
                res = game.play([_wrap(a) for a in seat_agents])
                for seat, cname in zip(remaining, order):
                    acc[cname].append(res["deltas"][seat])
                games += 1
            dealt_opp.append(opp_spec)

        for c in ckpts:
            per_deal[c].append(st.mean(acc[c]))

        if progress_every and (r + 1) % progress_every == 0:
            el = time.time() - t0
            eta = el / (r + 1) * (deals - r - 1) / 60
            last = [per_deal[ckpts[-1]][i] - per_deal[ckpts[0]][i] for i in range(r + 1)]
            lead = max(ckpts, key=lambda c: st.mean(per_deal[c]))
            print(f"[进度 {r + 1}/{deals}] 末代−首代 {st.mean(last):+.3f}；"
                  f"暂列第一 {lead}；已用 {el / 60:.1f} 分，预计剩余 {eta:.0f} 分",
                  flush=True)

    arr = {c: np.array(per_deal[c], float) for c in ckpts}

    result: dict = {
        "ckpts": ckpts,
        "opp_field": opps,
        "mode": mode,
        "layout": layout if mode == "arena" else 4,
        "anchor": ckpts[0],
        "seed0": seed0,
        "deals": deals,
        "games_total": games,
        "elapsed_min": round((time.time() - t0) / 60, 2),
        "ckpt_meta": {c: {"net": loaded[c][1].get("net"),
                          "iter": loaded[c][1].get("iter"),
                          "params": loaded[c][0].n_params()} for c in ckpts},
        "seat_balance": seat_hist,
        "levels": {c: level_stats(arr[c]) for c in ckpts},
        "pairs": {},
        "increments": {},
        "by_opp": {},
        "split_half": {},
        "quarter": {},
        "per_deal": {c: [round(x, 4) for x in per_deal[c]] for c in ckpts},
    }
    if dealt_opp:
        result["dealt_opp"] = dealt_opp

    # --- 两两配对比较 ---
    for i in range(len(ckpts)):
        for j in range(i + 1, len(ckpts)):
            a, b = ckpts[i], ckpts[j]
            result["pairs"][f"{a} - {b}"] = paired_stats(arr[a], arr[b], boot)

    # --- 阶梯判据：相邻代增量是否都显著为正 ---
    ok = True
    for i in range(len(ckpts) - 1):
        a, b = ckpts[i + 1], ckpts[i]      # 新一代 − 老一代
        s = paired_stats(arr[a], arr[b], boot)
        result["increments"][f"{a} - {b}"] = s
        ok = ok and s["mean"] > 0 and s["significant"]
    result["ladder_verdict"] = {
        "monotone_by_increment": bool(ok),
        "note": "每个相邻代增量都显著为正（μ>0 且 95% CI 不含 0）才判定为『逐代稳定变强』",
    }

    # --- 逐副严格单调比例 + 三者名次分布 ---
    if len(ckpts) == 3:
        a, b, c = arr[ckpts[0]], arr[ckpts[1]], arr[ckpts[2]]
        strict = int(((c > b) & (b > a)).sum())
        loose = int(((c >= b) & (b >= a)).sum())
        rev = int(((c < b) & (b < a)).sum())
        # 每副牌上谁第一、谁垫底
        best_cnt = {x: 0 for x in ckpts}
        worst_cnt = {x: 0 for x in ckpts}
        for i in range(deals):
            vals = {x: float(arr[x][i]) for x in ckpts}
            best_cnt[max(vals, key=vals.get)] += 1
            worst_cnt[min(vals, key=vals.get)] += 1
        result["per_deal_monotone"] = {
            "strict": strict, "loose": loose, "reversed": rev, "deals": deals,
            "strict_share": round(strict / deals, 4),
            "expected_if_noise": round(1 / 6, 4),
            "p_two_sided": round(binom_p(strict, deals, 1 / 6), 6),
            "p_greater": round(binom_ge(strict, deals, 1 / 6), 6),
            "best_first_count": best_cnt,
            "worst_count": worst_cnt,
        }

    # --- 分对手层（仅 arena 模式有按副轮换的对手） ---
    if dealt_opp:
        opp_arr = np.array(dealt_opp)
        for spec in sorted(set(dealt_opp)):
            mask = opp_arr == spec
            if result["mode"] == "arena" and mask.sum() < 20:
                continue
            sub = {c: arr[c][mask] for c in ckpts}
            entry: dict = {"deals": int(mask.sum())}
            for i in range(len(ckpts)):
                for j in range(i + 1, len(ckpts)):
                    a, b = ckpts[i], ckpts[j]
                    s = paired_stats(sub[a], sub[b], max(2000, boot // 4))
                    entry[f"{a} - {b}"] = {"mean": s["mean"], "ci95": s["ci95"],
                                           "verdict": s["verdict"]}
            result["by_opp"][spec] = entry

    # --- 分半 / 分四段稳定性（防时间漂移） ---
    segs = [("first_half", [slice(0, deals // 2)]),
            ("second_half", [slice(deals // 2, deals)])]
    for tag, slices in segs:
        entry = {}
        for i in range(len(ckpts)):
            for j in range(i + 1, len(ckpts)):
                a, b = ckpts[i], ckpts[j]
                d = np.concatenate([arr[a][sl] for sl in slices])
                e = np.concatenate([arr[b][sl] for sl in slices])
                s = paired_stats(d, e, max(2000, boot // 4))
                entry[f"{a} - {b}"] = {"mean": s["mean"], "ci95": s["ci95"]}
        result["split_half"][tag] = entry

    q = max(1, deals // 4)
    for qi in range(4):
        sl = slice(qi * q, min(deals, (qi + 1) * q))
        entry = {}
        for i in range(len(ckpts) - 1):
            a, b = ckpts[i + 1], ckpts[i]
            s = paired_stats(arr[a][sl], arr[b][sl], max(2000, boot // 8))
            entry[f"{a} - {b}"] = {"mean": s["mean"], "ci95": s["ci95"]}
        result["quarter"][f"q{qi + 1}"] = entry

    # --- 样本量规划（用实测 sd 推算） ---
    plan = {}
    for k, s in result["pairs"].items():
        plan[k] = {
            "sd": s["sd"],
            "deals_for_ci_0.30": deals_for_ci(s["sd"], 0.30),
            "deals_for_ci_0.15": deals_for_ci(s["sd"], 0.15),
            "current_ci95": s["ci95"],
        }
    result["sample_size_plan"] = plan

    return result


def render(res: dict) -> str:
    ck = res["ckpts"]
    L = []
    L.append(f"模式 {res['mode']}｜牌池 seed={res['seed0']}，{res['deals']} 副"
             f"（共 {res['games_total']} 局，真复式：同副牌换座位）")
    L.append(f"中立对手场：{' + '.join(res['opp_field'])}")
    L.append(f"用时 {res['elapsed_min']:.1f} 分")
    L.append("")
    L.append("== 分值水平（含对手固定偏倚，仅供参考） ==")
    L.append(f"{'检查点':<22}{'参数':>9}{'iter':>7}{'分值':>10}{'±95%CI':>10}{'per-deal sd':>13}")
    for c in ck:
        m = res["ckpt_meta"][c]
        lv = res["levels"][c]
        L.append(f"{c:<22}{m['params'] / 1e6:>8.2f}M{m['iter']:>7}"
                 f"{lv['mean']:>+10.3f}{lv['ci95']:>10.3f}{lv['sd']:>13.3f}")
    L.append("")
    L.append("== 配对净分差（同副牌配对观测量；判强看这个） ==")
    L.append(f"{'比较':<34}{'净分差':>10}{'±95%CI':>10}{'bootstrap 95%':>24}"
             f"{'逐副胜率':>10}{'符号p':>11}   结论")
    for k, s in res["pairs"].items():
        L.append(f"{k:<34}{s['mean']:>+10.3f}{s['ci95']:>10.3f}"
                 f"{'[%+.3f, %+.3f]' % tuple(s['boot_ci95']):>24}"
                 f"{s['win_share']:>10.3f}{s['sign_p']:>11.4f}   {s['verdict']}")
    L.append("")
    L.append("== 阶梯判据（相邻代增量，新一代 − 老一代） ==")
    for k, s in res["increments"].items():
        L.append(f"  {k:<34}{s['mean']:>+9.3f} ± {s['ci95']:.3f}"
                 f"   sd={s['sd']:.2f}   {s['verdict']}")
    v = res["ladder_verdict"]
    L.append(f"  → 逐代稳定变强：{'【成立】' if v['monotone_by_increment'] else '【不成立】'}")
    if "per_deal_monotone" in res:
        pm = res["per_deal_monotone"]
        L.append(f"  逐副严格单调（末>中>首）：{pm['strict']}/{pm['deals']} 副 "
                 f"= {pm['strict_share']:.1%}（随机基准 16.7%），"
                 f"单侧 p = {pm['p_greater']:.4f}；反向 {pm['reversed']} 副")
        bs = pm["best_first_count"]
        ws = pm["worst_count"]
        L.append("  逐副名次：拿第一 " + "／".join(f"{_short_name(c)} {bs[c]}"
                                                 for c in res["ckpts"])
                 + f"（共 {pm['deals']} 副）；垫底 "
                 + "／".join(f"{_short_name(c)} {ws[c]}" for c in res["ckpts"]))
    L.append("")
    if res.get("by_opp"):
        L.append("== 分对手层 ==")
        for spec, e in res["by_opp"].items():
            parts = [f"{_short(k)} {val['mean']:+.3f}±{val['ci95']:.3f}"
                     for k, val in e.items() if k != "deals"]
            L.append(f"  {spec:<26}（{e['deals']} 副）  " + "； ".join(parts))
        L.append("")
    L.append("== 分半稳定性 ==")
    for tag in ("first_half", "second_half"):
        parts = [f"{_short(k)} {val['mean']:+.3f}±{val['ci95']:.3f}"
                 for k, val in res["split_half"][tag].items()]
        L.append(f"  {tag:<14}" + "； ".join(parts))
    if res.get("quarter"):
        L.append("")
        L.append("== 分四段（看增量的时间漂移） ==")
        for tag, e in res["quarter"].items():
            parts = [f"{_short(k)} {val['mean']:+.3f}" for k, val in e.items()]
            L.append(f"  {tag:<6}" + "； ".join(parts))
    L.append("")
    L.append("== 样本量规划（按实测 sd 推算，达到目标 CI 半宽所需副数） ==")
    for k, p in res["sample_size_plan"].items():
        L.append(f"  {_short(k):<18} sd={p['sd']:.2f}  当前±{p['current_ci95']:.3f}"
                 f"  → ±0.30 需 {p['deals_for_ci_0.30']:,} 副，"
                 f"±0.15 需 {p['deals_for_ci_0.15']:,} 副")
    return "\n".join(L)


def _short(pair_key: str) -> str:
    a, b = pair_key.split(" - ")
    return f"{_short_name(a)}vs{_short_name(b)}"


def _short_name(c: str) -> str:
    return c.replace("dushan_big_", "").replace("dushan_", "")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpts", required=True, help="按代际顺序排列，逗号分隔")
    ap.add_argument("--opp", default=",".join(DEFAULT_OPPS),
                    help="中立对手池，逗号分隔；不得包含被测检查点")
    ap.add_argument("--mode", default="coop", choices=("coop", "arena"),
                    help="coop=各自打（方差小，推荐）；arena=同场竞争")
    ap.add_argument("--deals", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=31337)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--layout", type=int, default=4, choices=(4, 12),
                    help="仅 arena 模式：每副 4 或 12 局")
    ap.add_argument("--boot", type=int, default=10000)
    ap.add_argument("--progress-every", type=int, default=100)
    ap.add_argument("--out", default="reports/ladder_test.json")
    args = ap.parse_args()

    ckpts = [c.strip() for c in args.ckpts.split(",") if c.strip()]
    opps = [o.strip() for o in args.opp.split(",") if o.strip()]
    if len(ckpts) < 2:
        raise SystemExit("--ckpts 至少 2 个")

    res = run(ckpts, opps, args.deals, args.seed, args.threads,
              args.mode, args.layout, args.progress_every, args.boot)
    print()
    print(render(res))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n结果 -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
