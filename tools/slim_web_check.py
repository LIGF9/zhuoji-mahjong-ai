#!/usr/bin/env python3
"""线上「真档」核验：验证 champ_slim_s 的搜索档真的用学生权重（E-WEIGHT 防回归）。

为什么单独一个脚本
------------------
E-WEIGHT 事故的根因是 `ensure_onnx` 回落 `adhoc` 别名，导致**选任何冠军，
搜索档 Q 都来自冻结主力**——纯看分数完全正常，只有核对权重身份才能发现。
服务端已在 `/api/hint` 返回 `src`（ORT 实际加载的 stem）与 `src_ok`；
本脚本把它连同耗时一起打出来，作为「学生上 web」的验收证据。

用法::

    .venv/bin/python tools/slim_web_check.py --base http://127.0.0.1:8770
    .venv/bin/python tools/slim_web_check.py --strategies search32:champ_slim_s,model:champ_slim_s

注意：耗时只在机器空载时有意义。若同时有训练/评测在跑，请只看 `src_ok`
（确定性判据），延迟仅作相对参考。
"""
from __future__ import annotations

import argparse
import json
import statistics as st
import time
import urllib.parse
import urllib.request

DEFAULT_STRATS = (
    "model:champ_slim_s",
    "search32:champ_slim_s",
    "search128:champ_slim_s",
    "model:master",
    "search32:master",
    "search128:master",
    "search32:champ_exit_p48",
)


def _get(url: str, timeout: float = 600.0) -> dict:
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


def _post(url: str, payload: dict, timeout: float = 120.0) -> dict:
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=body,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


def new_session(base: str, stem: str, seed: int) -> str:
    r = _post(base + "/api/new", {
        "opponents": [{"type": "model", "model": stem} for _ in range(3)],
        "my_seat": 0, "seed": seed, "speed": 0.0,
    })
    if not r.get("ok"):
        raise SystemExit(f"new 失败: {r}")
    return r["sid"]


def wait_my_turn(base: str, sid: str, tries: int = 40) -> bool:
    """轮询到玩家决策点（对手走完 / 轮到自己）。"""
    for _ in range(tries):
        s = _get(f"{base}/api/state?sid={urllib.parse.quote(sid)}")
        st = s.get("state") or s
        if st.get("pending") or st.get("phase") in ("discard", "self_kong", "respond"):
            return True
        time.sleep(0.25)
    return False


def hint(base: str, sid: str, strategy: str) -> tuple[dict, float]:
    q = urllib.parse.urlencode({"sid": sid, "strategy": strategy})
    t0 = time.perf_counter()
    r = _get(f"{base}/api/hint?{q}")
    return r, time.perf_counter() - t0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:8770")
    ap.add_argument("--strategies", default=",".join(DEFAULT_STRATS))
    ap.add_argument("--seeds", type=int, default=3, help="每个策略换几局重复测")
    ap.add_argument("--opp", default="champ_slim_s", help="三家对手模型")
    ap.add_argument("--json-out", default="")
    args = ap.parse_args()

    strats = [s.strip() for s in args.strategies.split(",") if s.strip()]
    rows, bad = [], 0
    for strat in strats:
        times, srcs, oks = [], set(), []
        for i in range(args.seeds):
            sid = new_session(args.base, args.opp, 8_600_000 + i * 7 + len(rows))
            if not wait_my_turn(args.base, sid):
                print(f"[warn] {strat} seed#{i}: 未到玩家回合，跳过")
                continue
            r, dt = hint(args.base, sid, strat)
            if not r.get("ok"):
                print(f"[warn] {strat} seed#{i}: hint 失败 {r.get('error')}")
                continue
            times.append(dt)
            srcs.add(str(r.get("src")))
            oks.append(bool(r.get("src_ok")))
        if not times:
            rows.append({"strategy": strat, "ok": False, "note": "无有效样本"})
            bad += 1
            continue
        med = st.median(times)
        all_ok = all(oks)
        if not all_ok:
            bad += 1
        want = strat.split(":")[1]
        rows.append({"strategy": strat, "want_stem": want, "src": sorted(srcs),
                     "src_ok": all_ok, "n": len(times),
                     "median_s": round(med, 3), "times": [round(t, 3) for t in times]})
        flag = "OK " if all_ok else "BAD"
        print(f"[{flag}] {strat:28s} src={sorted(srcs)} 中位 {med:6.2f}s  n={len(times)}")

    print()
    if bad:
        print(f"VERDICT: FAIL —— {bad}/{len(rows)} 项来源不符或无效（搜索档串用了别的权重）")
    else:
        print(f"VERDICT: PASS —— 全部 {len(rows)} 个档位均由所选 stem 的权重计算")
    if args.json_out:
        from pathlib import Path

        Path(args.json_out).write_text(
            json.dumps({"rows": rows, "all_pass": bad == 0},
                       ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"→ {args.json_out}")


if __name__ == "__main__":
    main()
