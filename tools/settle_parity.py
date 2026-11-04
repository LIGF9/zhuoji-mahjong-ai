#!/usr/bin/env python3
"""结算对拍：Python DushanGame 随机自走 N 局，把牌墙+动作流喂给 Rust，
逐局对比双方 delta / pair / chickens 是否一致。

策略：优先胡/杠/碰（提高胡牌局占比），其余随机出牌。"""
import json
import random
import subprocess
import sys

sys.path.insert(0, "/Users/ligf9/WorkBuddy/2026-11-03-18-40-11/zhuoji-mahjong-ai")

from zhuoji.dushan import DushanConfig, DushanGame  # noqa: E402
from zhuoji.rules import HU, ANGANG, BUGANG, MINGGANG, PENG  # noqa: E402

# Python 侧 kind 常量是字符串（"discard"/"hu"/…），映射到 Rust 数值编号
KIND_CODE = {"discard": 0, "hu": 1, "angang": 2, "bugang": 3,
             "peng": 4, "minggang": 5, "pass": 6}


def deltas_from_pair(pair):
    n = len(pair)
    return [sum(pair[a][b] - pair[b][a] for b in range(n)) for a in range(n)]


def main():
    n_games = int(sys.argv[1]) if len(sys.argv) > 1 else 300
    rng = random.Random(20261104)
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
        n = 0
        while not g.result and n < 800:
            acts = g.legal_actions()
            if not acts:
                break
            # 激进策略：优先胡/杠/碰 → 大幅提高胡牌局与碰杠鸡覆盖
            hot = [a for a in acts if a.kind in (HU, ANGANG, BUGANG, MINGGANG, PENG)]
            a = rng.choice(hot) if hot and rng.random() < 0.9 else rng.choice(acts)
            actions.append([KIND_CODE[a.kind], 27 if a.tile is None else int(a.tile)])
            g.step(a)
            n += 1
        if g.result is None:
            continue
        cases.append({"wall": wall, "dealer": int(g.dealer), "actions": actions,
                      "cfg": {"jiesuan_fanji": cfg.jiesuan_fanji,
                              "kaiju_fanji": cfg.kaiju_fanji,
                              "end_baogang": cfg.end_baogang,
                              "huang_baogang": cfg.huang_baogang}})
        pair = g.result["pair"]
        results.append({
            "type": g.result["type"],
            "deltas": deltas_from_pair(pair),
            "pair": pair,
            "chickens": g.result.get("chickens"),
            "tenpai": g.result.get("tenpai"),
            "melds": [[list(m) for m in g.melds[p]] for p in range(4)],
            "flags": {"huang_baogang": cfg.huang_baogang,
                      "huang_baoji": cfg.huang_baoji},
        })
        made += 1

    inp = "\n".join(json.dumps(c) for c in cases) + "\n"
    bin_path = ("/Users/ligf9/WorkBuddy/2026-11-03-18-40-11/zhuoji-mahjong-ai/"
                "rust_engine/target/release/settle_parity")
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
                print("  py pair:", res["pair"])
                print("  rs pair:", r["pair"])
        if res["type"] == "win":
            rc = [round(x, 6) for x in r["chickens"]]
            pc = [round(x, 6) for x in res["chickens"]]
            if rc != pc:
                bad += 1
                if bad <= 3:
                    print(f"case {i} chickens mismatch py={pc} rs={rc}")
    print(f"games={len(results)} win={stats.get('win',0)} "
          f"huangzhuang={stats.get('huangzhuang',0)} mismatch={bad}")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
