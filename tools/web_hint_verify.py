#!/usr/bin/env python3
"""线上 E-WEIGHT 验收：/api/hint 的「来源核验」是否恢复为真。

对每个策略**各开一个同 seed 的新会话**，保证比较的是同一个决策点。
只读接口，不改模型/标志；会创建临时会话（不影响用户正在玩的会话）。

用法：
    .venv/bin/python tools/web_hint_verify.py [--host http://127.0.0.1:8770]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.parse
import urllib.request

BASE = "http://127.0.0.1:8770"
SEED = 909090

# (策略, 期望真实 stem, 期望 enc, 期望 engine 前缀)
CASES = [
    ("search128:champ_exit_p48", "champ_exit_p48", 3, "rust-"),
    ("search32:champ_exit_p48", "champ_exit_p48", 3, "rust-"),
    ("search128:master", "dushan_master", 1, "rust-"),
    ("model:champ_exit_p48", "champ_exit_p48", 3, "torch"),
    ("model:master", "dushan_master", 1, "torch"),
]


def post(path: str, payload: dict) -> dict:
    req = urllib.request.Request(
        BASE + path, data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=20) as r:
        return json.loads(r.read())


def get(path: str, params: dict) -> tuple[dict, float]:
    url = BASE + path + "?" + urllib.parse.urlencode(params)
    t0 = time.perf_counter()
    with urllib.request.urlopen(url, timeout=180) as r:
        d = json.loads(r.read())
    return d, time.perf_counter() - t0


def main() -> int:
    global BASE
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default=BASE)
    args = ap.parse_args()
    BASE = args.host.rstrip("/")

    ok = True
    tops: dict[str, list] = {}
    for strat, want_stem, want_enc, want_engine in CASES:
        new = post("/api/new", {"seed": SEED, "hold": False, "speed": 0,
                                "my_seat": 0})
        sid = new.get("sid")
        if not sid:
            print(f"[FAIL] /api/new 失败 {new}", flush=True)
            return 1
        # /api/new 之后对局线程要先把庄家到「我」之间的几手走完才写 pending。
        # speed=0 时应当瞬间完成，但这里仍显式轮询，避免竞态造成假阴性。
        ready = False
        for _ in range(40):
            st, _ = get("/api/state", {"sid": sid})
            if st.get("pending"):
                ready = True
                break
            time.sleep(0.25)
        if not ready:
            print(f"[FAIL] {strat:28s} 等不到我的决策点（pending 未写）", flush=True)
            ok = False
            continue
        t0 = time.perf_counter()
        try:
            r, _ = get("/api/hint", {"sid": sid, "strategy": strat})
        except Exception as exc:  # noqa: BLE001
            print(f"[ERROR] {strat:28s} {type(exc).__name__}: {exc} "
                  f"({time.perf_counter() - t0:.1f}s)", flush=True)
            ok = False
            continue
        dt = time.perf_counter() - t0
        src = r.get("src") or {}
        items = (r.get("items") or [])[:3]
        tops[strat] = [(x.get("label"), round(float(x.get("q", x.get("prob", 0))), 4))
                       for x in items]
        good = (r.get("ok") is True and r.get("src_ok") is True
                and src.get("stem") == want_stem
                and src.get("enc") == want_enc
                and src.get("enc_expected") == want_enc
                and str(src.get("engine", "")).startswith(want_engine))
        ok = ok and bool(good)
        print(f"[{' OK ' if good else 'FAIL'}] {strat:28s} {dt:6.2f}s "
              f"src_ok={r.get('src_ok')} stem={src.get('stem')} "
              f"engine={src.get('engine')} enc={src.get('enc')}/"
              f"{src.get('enc_expected')} top={tops[strat]}", flush=True)
        if not r.get("ok"):
            print(f"        error={r.get('error')}", flush=True)

    a = tops.get("search128:champ_exit_p48", [])
    b = tops.get("search128:master", [])
    if a and b:
        diff = a != b
        print(f"[区分度] 冠军搜索 vs 主力搜索 top3 "
              f"{'不同（真档）' if diff else '相同（疑似仍是假档）'}", flush=True)
        ok = ok and diff
    print("VERDICT:", "PASS" if ok else "FAIL", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
