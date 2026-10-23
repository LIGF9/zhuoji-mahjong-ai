"""M2-0 · T0.1 动作一致率分解（M1 vs master，按头 / 危险度 / 鸡牌分层）。

要回答的问题
------------
M1-S3 与 master 的 **1200 副配对净分差 −3.105**，差距究竟落在哪类决策上？

* 若整体一致率 **高（>80%）** 而强度差 3 分 → 差距集中在**少数高价值决策**，
  指向价值头 / 信念质量（"平时都对，关键处错"）。
* 若整体一致率 **低（~50%）** → 是**系统性风格差异**，指向策略覆盖不足
  （"处处都不一样"）。

这个区分直接决定 M2-A 的配方：前者优先修 critic / 加辅助头（A9），
后者优先扩探索 / 教师蒸馏（A2 / A4）。

做法（严格配对）
----------------
1. 用指定的 driver 模型跑自博弈，在**每个决策点**前 `deepcopy` 冻结局面
   —— 于是同一局面可以喂给两个模型，输入分布完全一致（不存在"两边各自
   对局导致局面分岔"的问题，这是本脚本与"各自跑 1200 副再比"的本质区别）。
2. 对每个冻结局面，两个模型各自算 argmax 与完整分布（temperature=0，确定性）。
   v1 与 v3 编码器共用同一套 ``head``/``mask``（``encode_v3`` 复用 v1 的
   ``action_head_and_mask``），所以下标语义逐位可比 —— 脚本会实测断言这一点。
3. 按 头 / 对手听牌压力 / 巡目 / 自己听牌 / 可胡 / 鸡牌相关性 分层统计一致率，
   并给出分布层面的 JS 散度与"master 的选择在 M1 分布里的概率质量"。

用法::

    python scripts/agreement_probe.py \
        --m1 dushan_m1_s3 --master dushan_master \
        --drivers dushan_master,dushan_m1_s3 --games-per-driver 12 --max-states 2000
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import statistics as st
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from zhuoji.agent import NetAgent  # noqa: E402
from zhuoji.dushan import DushanConfig, DushanGame  # noqa: E402
from zhuoji.net import build_from_ckpt  # noqa: E402
from zhuoji.tiles import YAOJI  # noqa: E402

HEADS = ("discard", "self_kong", "respond")
HU_IDX = {"discard": 27, "respond": 3}
TILE_IN_DISCARD_MAX = 27


def load(stem: str):
    ck = torch.load(ROOT / "models" / f"{stem}.pt", map_location="cpu", weights_only=False)
    m, ver = build_from_ckpt(ck)
    return m.eval(), ver


class Recorder:
    """包一层 agent：被调用前把局面深拷贝冻结下来，再照常决策。

    deepcopy 的是**只读快照**，之后两个模型各自在上面编码/前向，互不影响。
    """

    def __init__(self, agent, sink: list, driver: str, deal: int):
        self.agent = agent
        self.sink = sink
        self.driver = driver
        self.deal = deal

    def choose(self, game):
        self.sink.append((copy.deepcopy(game), game.actor(), self.driver, self.deal))
        return self.agent.choose(game)


def collect_states(drivers: dict, games_per_driver: int, seed0: int) -> list:
    states: list = []
    for di, (name, model) in enumerate(drivers.items()):
        t_ver = getattr(model, "encoder_version", 1)
        for gi in range(games_per_driver):
            g = DushanGame(DushanConfig(), seed=seed0 + di * 100003 + gi * 6151)
            recs = [Recorder(NetAgent(model, seed=gi * 4 + k, temperature=0.0, name=name,
                                      encoder_version=t_ver), states, name, gi)
                    for k in range(4)]
            g.play([r.choose for r in recs])
    return states


def stratified_sample(states: list, cap: int, seed: int = 20261014) -> list:
    """按 (driver, 局面阶段) 分层抽样。

    名额按**桶的自然占比**分配（不是每桶平分）：这样抽样后的 head 分布仍接近
    真实对局分布，"整体一致率"不会被重加权污染；同时刻意给每个桶至少 1 个名额，
    罕见的 self_kong / 抢杠局面也能出现在分层统计里。
    """
    if len(states) <= cap:
        return states
    rng = np.random.default_rng(seed)
    buckets: dict[tuple, list[int]] = {}
    for i, s in enumerate(states):
        ph = getattr(s[0].phase, "value", s[0].phase)
        buckets.setdefault((s[2], str(ph)), []).append(i)

    picked: list[int] = []
    n = len(states)
    for k in sorted(buckets):
        idx = buckets[k]
        quota = max(1, int(round(cap * len(idx) / n)))
        quota = min(quota, len(idx))
        sel = rng.permutation(len(idx))[:quota]
        picked.extend(idx[i] for i in sel)
    if len(picked) < cap:                     # 余量按大桶补，保持自然比例
        used = set(picked)
        pool = [i for i in range(n) if i not in used]
        rng.shuffle(pool)
        picked.extend(pool[:cap - len(picked)])
    rng.shuffle(picked)
    return [states[i] for i in picked[:cap]]


def js_divergence(p: np.ndarray, q: np.ndarray) -> float:
    m = 0.5 * (p + q)
    def kl(a, b):
        keep = a > 1e-12
        return float(np.sum(a[keep] * np.log(a[keep] / np.maximum(b[keep], 1e-12))))
    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def _self_tenpai(g, me: int) -> bool:
    fn = getattr(g, "_tenpai_flag", None)
    if callable(fn):
        try:
            return bool(fn(me))
        except Exception:                                       # noqa: BLE001
            pass
    return bool(g.ever_tenpai[me])


@torch.no_grad()
def probe_all(states: list, agents: dict) -> list[dict]:
    rows: list[dict] = []
    mask_mismatch = 0
    head_mismatch = 0
    for g, me, driver, deal in states:
        got = {}
        for key, ag in agents.items():
            head, mask, probs, val, obs = ag.distribution(g)
            got[key] = (head, np.asarray(mask), probs.numpy().astype(np.float64), float(val))

        h = got["m1"][0]
        if h != got["master"][0]:
            head_mismatch += 1
            continue
        m1_mask, ms_mask = got["m1"][1], got["master"][1]
        if m1_mask.shape != ms_mask.shape or not np.array_equal(m1_mask, ms_mask):
            mask_mismatch += 1
        p1, p2 = got["m1"][2], got["master"][2]
        v1, v2 = got["m1"][3], got["master"][3]
        a1, a2 = int(np.argmax(p1)), int(np.argmax(p2))

        others = [(me + i) % 4 for i in (1, 2, 3)]
        opp_tenpai = sum(1 for p in others if g.ever_tenpai[p])
        opp_melds = sum(len(g.melds[p]) for p in others)
        hi = HU_IDX.get(h)
        hu_legal = bool(m1_mask[hi]) if hi is not None else False
        tile1 = a1 if (h == "discard" and a1 < TILE_IN_DISCARD_MAX) else -1
        tile2 = a2 if (h == "discard" and a2 < TILE_IN_DISCARD_MAX) else -1
        hand_yaoji = int(g.hands[me][YAOJI])
        chongfeng_open = (getattr(g, "chongfeng_player", None) is None
                          and not getattr(g, "_first_yaoji_discarded", True))
        try:
            hengji = bool(g.hengji_active)
        except Exception:                                       # noqa: BLE001
            hengji = False

        rows.append({
            "head": h,
            "agree": a1 == a2,
            "a1": a1, "a2": a2,
            "tile1": tile1, "tile2": tile2,
            "p1_at_a2": round(float(p1[a2]), 5),
            "rank_of_a2_in_p1": int((p1 > p1[a2]).sum()) + 1,
            "conf1": round(float(p1[a1]), 5),
            "conf2": round(float(p2[a2]), 5),
            "js": round(js_divergence(p1, p2), 5),
            "v1": round(v1, 4), "v2": round(v2, 4),
            "opp_tenpai": opp_tenpai,
            "opp_melds": min(opp_melds, 4),
            "wall": int(g.wall_left),
            "self_tenpai": _self_tenpai(g, me),
            "hu_legal": hu_legal,
            "discard_yaoji": tile1 == YAOJI,
            "hand_yaoji": hand_yaoji,
            "chongfeng_open": bool(chongfeng_open),
            "hengji": hengji,
            "n_legal": int(m1_mask.sum()),
            "driver": driver,
            "deal": deal,
            "me": int(me),
        })
    return rows, {"head_mismatch": head_mismatch, "mask_mismatch": mask_mismatch}


def _rate(rows: list[dict]) -> dict:
    n = len(rows)
    if not n:
        return {"n": 0, "agree": None, "ci95": None}
    k = sum(1 for r in rows if r["agree"])
    p = k / n
    se = math.sqrt(max(p * (1 - p), 1e-12) / n)
    return {"n": n, "agree": round(p, 4), "ci95": round(1.96 * se, 4),
            "js": round(st.mean([r["js"] for r in rows]), 4),
            "p1_at_a2": round(st.mean([r["p1_at_a2"] for r in rows]), 4),
            "top1_in_3": round(sum(1 for r in rows if r["rank_of_a2_in_p1"] <= 3) / n, 4)}


def summarize(rows: list[dict]) -> dict:
    def subset(fn):
        return _rate([r for r in rows if fn(r)])

    out: dict = {"overall": _rate(rows), "by_head": {}, "strata": {}}
    for h in HEADS:
        out["by_head"][h] = subset(lambda r, h=h: r["head"] == h)

    def add(name, fn):
        out["strata"][name] = subset(fn)

    add("对手无人听牌", lambda r: r["opp_tenpai"] == 0)
    add("对手至少一家听牌", lambda r: r["opp_tenpai"] >= 1)
    add("对手两家以上听牌", lambda r: r["opp_tenpai"] >= 2)
    add("对手无副露", lambda r: r["opp_melds"] == 0)
    add("对手副露≥2", lambda r: r["opp_melds"] >= 2)
    add("巡目早(wall>60)", lambda r: r["wall"] > 60)
    add("巡目中(30–60)", lambda r: 30 <= r["wall"] <= 60)
    add("巡目晚(wall<30)", lambda r: r["wall"] < 30)
    add("自己已听牌", lambda r: r["self_tenpai"])
    add("自己未听牌", lambda r: not r["self_tenpai"])
    add("本步可胡", lambda r: r["hu_legal"])
    add("打出幺鸡", lambda r: r["discard_yaoji"])
    add("手中有幺鸡", lambda r: r["hand_yaoji"] > 0)
    add("冲锋鸡未定", lambda r: r["chongfeng_open"])
    add("横鸡轮", lambda r: r["hengji"])
    add("对手听牌+巡目晚", lambda r: r["opp_tenpai"] >= 1 and r["wall"] < 30)

    hi_value = lambda r: (r["hu_legal"] or r["opp_tenpai"] >= 1
                          or r["wall"] < 35 or r["discard_yaoji"] or r["hand_yaoji"] > 0)
    lo_value = lambda r: not hi_value(r)
    out["strata"]["★高价值子集"] = subset(hi_value)
    out["strata"]["★低价值子集"] = subset(lo_value)

    # 分歧结构：不一致时 master 选的动作在 M1 分布里排第几 / 概率多少
    dis = [r for r in rows if not r["agree"]]
    out["disagreement"] = {
        "n": len(dis),
        "share": round(len(dis) / max(1, len(rows)), 4),
        "mean_p1_at_a2": round(st.mean([r["p1_at_a2"] for r in dis]), 4) if dis else None,
        "top3_rate": round(sum(1 for r in dis if r["rank_of_a2_in_p1"] <= 3) / len(dis), 4) if dis else None,
        "mean_js": round(st.mean([r["js"] for r in dis]), 4) if dis else None,
        "by_head": {h: sum(1 for r in dis if r["head"] == h) for h in HEADS},
        "master_argmax_is_hu": sum(1 for r in dis
                                   if HU_IDX.get(r["head"]) == r["a2"]),
        "m1_argmax_is_hu": sum(1 for r in dis
                               if HU_IDX.get(r["head"]) == r["a1"]),
    }
    out["by_driver"] = {}
    for d in sorted({r["driver"] for r in rows}):
        out["by_driver"][d] = subset(lambda r, d=d: r["driver"] == d)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--m1", default="dushan_m1_s3")
    ap.add_argument("--master", default="dushan_master")
    ap.add_argument("--drivers", default="dushan_master,dushan_m1_s3",
                    help="用来产生自博弈状态的模型（逗号分隔）")
    ap.add_argument("--games-per-driver", type=int, default=12)
    ap.add_argument("--max-states", type=int, default=2000)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--seed", type=int, default=20261014)
    ap.add_argument("--out", default=str(ROOT / "reports" / "_agreement_probe.json"))
    ap.add_argument("--dump-rows", action="store_true", help="JSON 里带上逐条明细")
    args = ap.parse_args()

    torch.set_num_threads(args.threads)
    m1, m1_ver = load(args.m1)
    ms, ms_ver = load(args.master)
    print(f"[模型] {args.m1}（enc v{m1_ver}，{m1.n_params():,} 参数）"
          f" vs {args.master}（enc v{ms_ver}，{ms.n_params():,} 参数）")

    drivers = {}
    for stem in [s.strip() for s in args.drivers.split(",") if s.strip()]:
        d, v = load(stem)
        print(f"[driver] {stem}（enc v{v}）")
        drivers[stem] = d

    print("[1/3] 采集自博弈冻结局面 …", flush=True)
    states = collect_states(drivers, args.games_per_driver, args.seed)
    print(f"      原始 {len(states)} 个决策点"
          f"（{len(drivers)} 个 driver × {args.games_per_driver} 局 × 4 座）")
    states = stratified_sample(states, args.max_states, args.seed)
    print(f"      分层抽样后 {len(states)} 个")

    print("[2/3] 双模型对同一冻结局面前向 …", flush=True)
    agents = {"m1": NetAgent(m1, temperature=0.0, name="m1", encoder_version=m1_ver),
              "master": NetAgent(ms, temperature=0.0, name="master", encoder_version=ms_ver)}
    rows, chk = probe_all(states, agents)
    print(f"      head 不一致 {chk['head_mismatch']} 条（理论恒 0），"
          f"mask 不一致 {chk['mask_mismatch']} 条（v1/v3 共用 mask，理论恒 0）")

    print("[3/3] 汇总 …\n", flush=True)
    rep = summarize(rows)
    ov = rep["overall"]
    print(f"整体 argmax 一致率 {ov['agree'] * 100:.1f}% ± {ov['ci95'] * 100:.1f}"
          f"（n={ov['n']}）  平均 JS={ov['js']:.3f}")
    print("\n按动作头：")
    for h, v in rep["by_head"].items():
        if v["n"]:
            print(f"  {h:<10} n={v['n']:<5} 一致率 {v['agree'] * 100:5.1f}%  JS={v['js']:.3f}")
    print("\n分层：")
    for k, v in rep["strata"].items():
        if v["n"]:
            print(f"  {k:<16} n={v['n']:<5} 一致率 {v['agree'] * 100:5.1f}%"
                  f"  JS={v['js']:.3f}  master选择落M1前三={v['top1_in_3'] * 100:4.1f}%")
    d = rep["disagreement"]
    print(f"\n分歧结构：{d['n']} 条（占 {d['share'] * 100:.1f}%）；"
          f"其中 master 的选择在 M1 分布里平均概率 {d['mean_p1_at_a2']}，"
          f"落进前三 {d['top3_rate'] * 100:.1f}%")
    print(f"  按头分布 {d['by_head']}；不一致时 master 选胡 {d['master_argmax_is_hu']} 次"
          f"、M1 选胡 {d['m1_argmax_is_hu']} 次")

    hi, lo = rep["strata"]["★高价值子集"], rep["strata"]["★低价值子集"]
    if hi["n"] and lo["n"]:
        print(f"\n【判据】高价值子集一致率 {hi['agree'] * 100:.1f}% vs 低价值 {lo['agree'] * 100:.1f}%"
              f"（差 {(lo['agree'] - hi['agree']) * 100:+.1f} pt）")
        if ov["agree"] >= 0.80 and hi["agree"] < ov["agree"] - 0.03:
            print("  → 一致率整体高、高价值段明显更低 ⇒ 差距集中在少数关键决策"
                  "（指向 critic/信念质量，优先 A9 辅助头 + 价值头）")
        elif ov["agree"] < 0.60:
            print("  → 一致率整体偏低 ⇒ 系统性风格差异（指向策略覆盖，优先 A2 探索/A4 蒸馏）")
        else:
            print("  → 介于两者之间：既有系统性差异也有关键处集中，两条线都要走")

    out = {"m1": args.m1, "master": args.master, "n_states": len(rows),
           "checks": chk, "report": rep}
    if args.dump_rows:
        out["rows"] = rows
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n→ {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
