#!/usr/bin/env python3
"""结算对拍：Python DushanGame 随机自走 N 局，把牌墙+动作流喂给 Rust，
逐局对比双方 delta / pair / chickens 是否一致。

策略：优先胡/杠/碰（提高胡牌局占比），其余随机出牌。

2026-11-07 新增「第一巡同轮跟打」定向 case：随机牌墙里挑一张四家各持一张的
牌种，强制四家首张弃牌都打它（中间只允许 SELF_KONG/RESPOND 的 PASS，保证
Python/Rust 动作流逐位一致），覆盖：
  * 开关开 / 关（触发与不触发）；
  * 默认赔付 3 鸡、4 张手中鸡，以及非默认 follow_pay / follow_hold_chickens；
  * 胡牌局 / 黄庄局；
  * 双方 result.follow_* 与 Rust 的 diag_follow（首巡判定进度 + cfg 过桥）。

用法：
  .venv/bin/python tools/settle_parity.py [随机局数] [定向局数]
"""
import json
import random
import subprocess
import sys
from pathlib import Path

# 旧脚本把仓库路径写死在 2026-11-03 的 WorkBuddy 临时目录里，项目迁移后
# 直接跑会 ModuleNotFoundError / 找不到二进制。改成相对本文件定位仓库根目录。
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from zhuoji.dushan import DushanConfig, DushanGame  # noqa: E402
from zhuoji.rules import (Action, DISCARD, PASS, Phase, HU, ANGANG, BUGANG,  # noqa: E402
                          MINGGANG, PENG)

# Python 侧 kind 常量是字符串（"discard"/"hu"/…），映射到 Rust 数值编号
KIND_CODE = {"discard": 0, "hu": 1, "angang": 2, "bugang": 3,
             "peng": 4, "minggang": 5, "pass": 6}

# 定向 case 的参数轮转：(开关, 每家赔付鸡数, 手中鸡张数)
FOLLOW_SPECS = [
    (True, 3.0, 4),
    (True, 3.0, 4),
    (True, 5.0, 3),
    (True, 1.5, 4),
    (False, 3.0, 4),
    (False, 3.0, 4),
]


def deltas_from_pair(pair):
    n = len(pair)
    return [sum(pair[a][b] - pair[b][a] for b in range(n)) for a in range(n)]


def act_code(a):
    return [KIND_CODE[a.kind], 27 if a.tile is None else int(a.tile)]


def result_row(g):
    r = g.result
    pair = r["pair"]
    return {
        "type": r["type"],
        "deltas": deltas_from_pair(pair),
        "pair": pair,
        "chickens": r.get("chickens"),
        "tenpai": r.get("tenpai"),
        "melds": [[list(m) for m in g.melds[p]] for p in range(4)],
        "flags": {"huang_baogang": g.cfg.huang_baogang,
                  "huang_baoji": g.cfg.huang_baoji},
        # 2026-11-07：同轮跟打的触发结果双轨逐位对拍
        "follow": (bool(r.get("follow_dealer")),
                   r.get("follow_tile"),
                   round(float(r.get("follow_pay") or 0.0), 6)),
    }


def finish_random(g, actions, rng, limit=800):
    """按原激进策略走到终局；返回是否成功结算。"""
    n = 0
    while not g.result and n < limit:
        acts = g.legal_actions()
        if not acts:
            break
        hot = [a for a in acts if a.kind in (HU, ANGANG, BUGANG, MINGGANG, PENG)]
        a = rng.choice(hot) if hot and rng.random() < 0.9 else rng.choice(acts)
        actions.append(act_code(a))
        g.step(a)
        n += 1
    return g.result is not None


def random_cases(n_games, rng):
    cases, results = [], []
    made = 0
    while made < n_games:
        kaiju = ("off", "up", "down", "both")[made % 4]
        cfg = DushanConfig(kaiju_fanji=kaiju, jiesuan_fanji="both",
                           end_baogang=rng.random() < 0.3,
                           huang_baogang=rng.random() < 0.2)
        g = DushanGame(cfg, seed=rng.randrange(1 << 30))
        wall = list(g.wall)
        actions = []
        if not finish_random(g, actions, rng):
            continue
        cases.append({"wall": wall, "dealer": int(g.dealer), "actions": actions,
                      "cfg": {"jiesuan_fanji": cfg.jiesuan_fanji,
                              "kaiju_fanji": cfg.kaiju_fanji,
                              "end_baogang": cfg.end_baogang,
                              "huang_baogang": cfg.huang_baogang,
                              "first_round_follow": cfg.first_round_follow,
                              "follow_pay": cfg.follow_pay,
                              "follow_hold_chickens": cfg.follow_hold_chickens}})
        results.append(result_row(g))
        made += 1
    return cases, results


# 定向胡牌局用的庄家听牌手型：000 111 222 333 4（13 张，单钓 4 成 44 将）
WIN_HAND_13 = [0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4]
WIN_TILE = 4


def force_prefix(cfg, seed, wall, t, dealer=None, hands_override=None):
    """在给定牌墙上强制四家首张弃牌都打 t；返回 (game, actions, ok)。"""
    g = DushanGame(cfg, seed=seed, dealer=dealer)
    g.wall = list(wall)          # 前 53 张（初始发牌）不变，后面可控
    if hands_override:
        for p, h in hands_override.items():
            g.hands[p] = list(h)
        # Rust 的 new_with_wall 在发牌后会重算 enter_self_phase；Python 的
        # __init__ 已经按原手牌定过 phase，覆写手牌后必须同步重算，否则
        # 四张相同牌触发的 SELF_KONG 判定会双轨分叉（Rust 多等一个 PASS）。
        g._enter_self_phase(int(g.dealer))
    actions = []
    forced, guard = 0, 0
    while forced < 4 and guard < 40:
        guard += 1
        if g.result is not None or g.phase == Phase.OVER:
            return g, actions, False
        # 摸牌阶段可能出现暗杠/补杠选项；响应阶段保守起见也只过牌。
        # 这两种 PASS 在 Python/Rust 双轨语义一致，动作流仍逐位相同。
        if g.phase in (Phase.SELF_KONG, Phase.RESPOND):
            actions.append([KIND_CODE[PASS], 27])
            g.step(Action(PASS))
            continue
        p = g.current
        if g.hands[p][t] < 1:
            return g, actions, False
        # 预检：这张打出去无人可胡/碰/杠，避免响应阶段导致流分叉
        others = [q for q in range(4) if q != p]
        if any(g.can_ron(q, t) or g._can_peng(q, t) or
               g._can_minggang(q, t) for q in others):
            return g, actions, False
        actions.append([KIND_CODE[DISCARD], t])
        g.step(Action(DISCARD, t))
        forced += 1
    return g, actions, forced == 4 and g.result is None


def crafted_tsumo_win(cfg, seed, wall, t, dealer, rng):
    """构造一副「庄家首巡跟打后自摸胡」的牌墙，让定向 case 覆盖胡牌结算。

    做法（Python/Rust 拿到的是同一副墙，动作流仍逐位一致）：
      * 把 wall[0..51] 按四家手牌交错写回：庄家换成固定听牌 13 张
        ``WIN_HAND_13``，其余三家保持该 seed 的原始手牌；
      * wall[52]（庄家第 14 张）= t，庄家首张弃牌就打它 → 手牌回到 13 张听牌；
      * wall[56]（庄家第二巡摸牌）= ``WIN_TILE`` → 自摸成胡。
    Python 侧同步覆写 ``g.hands[dealer]``（构造后、未 step 前，安全）。
    """
    natural = DushanGame(cfg, seed=seed, dealer=dealer)
    tiles = [[] for _ in range(4)]
    for p in range(4):
        if p == dealer:
            tiles[p] = list(WIN_HAND_13)
        else:
            for tt, c in enumerate(natural.hands[p]):
                tiles[p].extend([tt] * c)
    if any(len(h) != 13 for h in tiles):
        return None
    wall2 = list(natural.wall)
    for r in range(13):
        for p in range(4):
            wall2[r * 4 + p] = tiles[p][r]
    wall2[52] = t
    wall2[56] = WIN_TILE

    b = [0] * 27
    for x in WIN_HAND_13:
        b[x] += 1
    dealer_hand = list(b)
    dealer_hand[t] += 1                      # 第 14 张 = 首张弃牌 t
    g, actions, ok = force_prefix(cfg, seed, wall2, t, dealer=dealer,
                                  hands_override={dealer: dealer_hand})
    if not ok:
        return None
    guard = 0
    while g.phase == Phase.SELF_KONG and guard < 8:
        actions.append([KIND_CODE[PASS], 27])
        g.step(Action(PASS))
        guard += 1
    if (g.phase != Phase.DISCARD or not g.drawn
            or not g.can_tsumo(g.current)):
        return None
    actions.append([KIND_CODE[HU], 27])
    g.step(Action(HU))
    if g.result is None or g.result["type"] != "win":
        return None
    return wall2, g, actions


def follow_cases(n_cases, rng):
    """构造「四家首张弃牌同牌种」的定向 case（触发或不触发同轮跟打）。

    一半 case 走随机续局（多为黄庄），一半强制自摸胡（覆盖结算计鸡路径）。
    """
    cases, results, tried = [], [], set()
    while len(cases) < n_cases:
        idx = len(cases)
        spec = FOLLOW_SPECS[idx % len(FOLLOW_SPECS)]
        on, pay, hold = spec
        want_win = (idx % 2 == 0)
        seed = rng.randrange(1 << 30)
        if seed in tried:
            continue
        tried.add(seed)
        cfg = DushanConfig(kaiju_fanji="off", jiesuan_fanji="both",
                           first_round_follow=on, follow_pay=pay,
                           follow_hold_chickens=hold)
        probe = DushanGame(cfg, seed=seed)
        wall = list(probe.wall)
        dealer = int(probe.dealer)
        # 四家各持至少一张的牌种（4 张全在手里 → 牌墙里没有该种，永远不会被摸到）
        cands = [t for t in range(27)
                 if all(probe.hands[q][t] >= 1 for q in range(4))]
        rng.shuffle(cands)
        for t in cands:
            if want_win:
                made = crafted_tsumo_win(cfg, seed, wall, t, dealer, rng)
                if made is None:
                    continue
                wall_used, g, actions = made
                # 胡牌动作不增加弃牌数，前缀的「前 4 次恰是四家首张」仍须成立
                if int(g.total_discards) != 4:
                    continue
            else:
                g, actions, ok = force_prefix(cfg, seed, wall, t, dealer=dealer)
                if not ok:
                    continue
                # 判定必须在随机续局**之前**验：续局会把 total_discards 涨到几十
                if (bool(g.follow_dealer) != on
                        or int(g.total_discards) != 4):
                    continue
                if not finish_random(g, actions, rng):
                    continue
                wall_used = wall
            if bool(g.follow_dealer) != on or g.result is None:
                continue
            cases.append({"wall": wall_used, "dealer": dealer, "actions": actions,
                          "cfg": {"jiesuan_fanji": cfg.jiesuan_fanji,
                                  "kaiju_fanji": cfg.kaiju_fanji,
                                  "end_baogang": cfg.end_baogang,
                                  "huang_baogang": cfg.huang_baogang,
                                  "first_round_follow": on,
                                  "follow_pay": pay,
                                  "follow_hold_chickens": hold}})
            row = result_row(g)
            row["spec"] = spec
            row["want_win"] = want_win
            results.append(row)
            break
        if len(tried) > 6000:  # 防止生成器异常时死循环
            break
    return cases, results


def main():
    n_games = int(sys.argv[1]) if len(sys.argv) > 1 else 300
    n_follow = int(sys.argv[2]) if len(sys.argv) > 2 else 12
    rng = random.Random(20261104)

    dcases, dresults = follow_cases(n_follow, rng)
    n_dir = len(dcases)
    if n_dir < min(6, n_follow):
        print(f"FAIL: 定向同轮跟打 case 只造出 {n_dir} 局（目标 {n_follow}）")
        sys.exit(1)
    n_trig = sum(1 for r in dresults if r["follow"][0])
    n_follow_win = sum(1 for r in dresults
                       if r["type"] == "win" and r["follow"][0])
    n_follow_huang = sum(1 for r in dresults
                         if r["type"] == "huangzhuang" and r["follow"][0])
    print(f"follow-cases={n_dir} triggered={n_trig} disabled={n_dir - n_trig} "
          f"win={n_follow_win} huangzhuang={n_follow_huang}")
    if n_trig < 3 or n_follow_win < 2 or n_follow_huang < 1:
        print("FAIL: 触发/胡牌/黄庄定向 case 不足，无法验证赔付与计鸡路径")
        sys.exit(1)

    rcases, rresults = random_cases(n_games, rng)
    cases = dcases + rcases
    results = dresults + rresults

    inp = "\n".join(json.dumps(c) for c in cases) + "\n"
    bin_path = str(ROOT / "rust_engine" / "target" / "release" / "settle_parity")
    proc = subprocess.run([bin_path], input=inp, capture_output=True, text=True,
                          timeout=600)
    if proc.returncode != 0:
        print("RUST FAIL:", proc.stderr[:2000])
        sys.exit(1)
    outs = [json.loads(l) for l in proc.stdout.strip().splitlines()]

    bad = 0
    stats = {"win": 0, "huangzhuang": 0}
    for i, (res, out) in enumerate(zip(results, outs)):
        stats[res["type"]] = stats.get(res["type"], 0) + 1
        if not out.get("ok"):
            print(f"case {i}: rust error {out.get('error')}")
            bad += 1
            continue
        r = out["result"]
        rd = [round(x, 6) for x in r["deltas"]]
        pd = [round(x, 6) for x in res["deltas"]]
        if rd != pd:
            bad += 1
            if bad <= 3:
                print(f"case {i} [{res['type']}] deltas mismatch")
                print("  py:", pd)
                print("  rs:", rd)
                print("  py tenpai:", res.get("tenpai"), "rs tenpai:", r.get("tenpai"))
                print("  py melds:", res.get("melds"))
                print("  rs melds:", r.get("diag_melds"))
                print("  flags:", res.get("flags"), r.get("diag_flags"))
                print("  follow:", res["follow"], r.get("diag_follow"))
                print("  py pair:", res["pair"])
                print("  rs pair:", r["pair"])
        if res["type"] == "win":
            rc = [round(x, 6) for x in r["chickens"]]
            pc = [round(x, 6) for x in res["chickens"]]
            if rc != pc:
                bad += 1
                if bad <= 3:
                    print(f"case {i} chickens mismatch py={pc} rs={rc}")
        # 同轮跟打：触发结果（dealer / tile / pay）双轨逐位一致
        rf = (bool(r.get("follow_dealer")), r.get("follow_tile"),
              round(float(r.get("follow_pay") or 0.0), 6))
        if rf != res["follow"]:
            bad += 1
            if bad <= 3:
                print(f"case {i} follow mismatch py={res['follow']} rs={rf} "
                      f"diag={r.get('diag_follow')}")
        # 定向 case 还校验开关/赔付/手中鸡张数的 cfg 过桥
        if i < n_dir:
            diag = r.get("diag_follow") or {}
            spec = res["spec"]
            if (diag.get("switch") != spec[0]
                    or round(float(diag.get("pay") or 0.0), 6) != round(spec[1], 6)
                    or diag.get("hold") != spec[2]
                    or int(diag.get("first_discard_n") or 0) != 4):
                bad += 1
                if bad <= 3:
                    print(f"case {i} follow cfg bridge mismatch spec={spec} "
                          f"diag={diag}")
    print(f"games={len(results)} (follow={n_dir}) "
          f"win={stats.get('win', 0)} huangzhuang={stats.get('huangzhuang', 0)} "
          f"mismatch={bad}")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
