#!/usr/bin/env python3
"""离线服务端到端冒烟：真实打完一手，核对 16 个接口与 /api/state 契约。

前置：cargo run --bin offline_server --manifest-path offline/Cargo.toml -- 8771
用法：.venv/bin/python tools/offline_smoke.py [base_url]
"""
from __future__ import annotations

import json
import sys
import time
import urllib.error
import urllib.request

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8771"


def api(path: str, body: dict | None = None, timeout: float = 30.0):
    url = BASE + path
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(url, data=data,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        raw = r.read()
        if not raw:
            return None
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return raw


def check(name: str, cond: bool, extra: str = "") -> bool:
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"  {extra}" if extra else ""))
    return cond


def main() -> int:
    ok = True
    print("=== 静态与元数据 ===")
    meta = api("/api/meta")
    ok &= check("/api/meta: 27 张牌", len(meta.get("tiles", [])) == 27)
    ok &= check("/api/meta: master 非空", bool(meta.get("master")))
    ok &= check("/api/meta: 冠军已注册",
                any(m.get("champion") for m in meta.get("models", [])))
    with urllib.request.urlopen(BASE + "/", timeout=10) as r:
        html = r.read()
    ok &= check("GET / 返回 dushan.html", html[:15].startswith(b"<!DOCTYPE html") or b"<html" in html[:200],
                f"{len(html)} bytes")

    print("=== 开局 ===")
    new = api("/api/new", {
        "opponents": [
            {"type": "model", "model": "champ_slim_s", "temperature": 0},
            {"type": "model", "model": "champ_slim_s", "temperature": 0},
            {"type": "model", "model": "champ_slim_s", "temperature": 0},
        ],
        "my_seat": 0,
        "speed": 5,
        "names": ["我", "下家", "对家", "上家"],
        "hold": False,
    })
    ok &= check("/api/new", bool(new.get("ok")), str(new))

    print("=== 打完一手 ===")
    st = None
    acts = 0
    for i in range(1200):
        st = api("/api/state?sid=offline")
        if st.get("error"):
            ok &= check("state 无服务端异常", False, str(st["error"]))
            break
        if st.get("over"):
            break
        pend = st.get("pending")
        if not pend:
            time.sleep(0.01)
            continue
        if pend.get("discardables"):
            a = {"kind": "discard", "tile": pend["discardables"][0]}
        elif pend.get("buttons"):
            b = pend["buttons"][0]
            a = {"kind": b["kind"], "tile": b.get("tile", -1)}
        else:
            ok &= check("决策点有可选项", False, json.dumps(pend, ensure_ascii=False))
            break
        r = api("/api/act", {"sid": "offline", **a})
        if not r.get("ok"):
            ok &= check("/api/act", False, f"{a} → {r}")
            break
        acts += 1
        # 契约不变量
        if i % 25 == 0:
            ok &= check(f"step {i}: players=4 / counter=27",
                        len(st.get("players", [])) == 4 and len(st.get("counter", [])) == 27)
    else:
        ok &= check("一手在 1200 次轮询内结束", False)

    if st:
        ok &= check("本局已结束", bool(st.get("over")), f"共 {acts} 次人类动作")
        res = st.get("result")
        ok &= check("result 非空", isinstance(res, dict))
        if isinstance(res, dict):
            ok &= check("result.deltas 长度 4", len(res.get("deltas") or []) == 4,
                        str(res.get("deltas")))
            ok &= check("result.pair 4x4", len(res.get("pair") or []) == 4)
            ok &= check("result.type 合法", res.get("type") in ("win", "huang"),
                        str(res.get("type")))
        ok &= check("scores 长度 4", len(st.get("scores") or []) == 4)

    print("=== 其余接口（离线版可返回空，但必须合法 JSON） ===")
    for path in ["/api/history?limit=30", "/api/stats?sid=offline",
                 "/api/replay?rid=x", "/api/ting?sid=offline",
                 "/api/ting?sid=offline&mode=now",
                 "/api/hint?sid=offline&strategy=model%3Achamp_slim_s"]:
        try:
            v = api(path)
            ok &= check(f"GET {path.split('?')[0]}", isinstance(v, dict) and "ok" in v)
        except Exception as e:
            ok &= check(f"GET {path.split('?')[0]}", False, str(e))
    for path, body in [("/api/hold", {"sid": "offline", "on": False}),
                       ("/api/speed", {"sid": "offline", "speed": 1.0}),
                       ("/api/history_clear", {}),
                       ("/api/quit", {"sid": "offline"})]:
        try:
            v = api(path, body)
            ok &= check(f"POST {path}", isinstance(v, dict) and v.get("ok") is True)
        except Exception as e:
            ok &= check(f"POST {path}", False, str(e))

    print("=== 下一局 ===")
    v = api("/api/next", {"sid": "offline"})
    ok &= check("/api/next", isinstance(v, dict) and v.get("ok") is True, str(v))
    st2 = api("/api/state?sid=offline")
    ok &= check("新局 hand_index=2", st2.get("hand_index") == 2, str(st2.get("hand_index")))

    print("\n" + ("全部通过" if ok else "存在失败项"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
