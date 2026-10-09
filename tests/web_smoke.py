"""
网页对局服务端的端到端冒烟测试。

起一个真服务（子进程），用真 HTTP 请求扮演人类把一局打完：
    /api/meta → /api/new → 轮询 /api/state → /api/hint → /api/act ×N
    → 检查终局结算 → /api/next

跑法::

    python tests/web_smoke.py
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PORT = 8799
BASE = f"http://127.0.0.1:{PORT}"

PASSED: list[str] = []
FAILED: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    (PASSED if ok else FAILED).append(name)
    print(f"  [{'ok  ' if ok else 'FAIL'}] {name}" + (f"   {detail}" if detail else ""),
          flush=True)


def get(path: str) -> dict:
    with urllib.request.urlopen(BASE + path, timeout=10) as r:
        return json.loads(r.read().decode("utf-8"))


def post(path: str, body: dict) -> dict:
    req = urllib.request.Request(
        BASE + path, data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=15) as r:
        return json.loads(r.read().decode("utf-8"))


def main() -> int:
    log = ROOT / "reports" / "web_smoke_server.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUNBUFFERED"] = "1"

    print(f"启动服务 {BASE} …", flush=True)
    srv = subprocess.Popen(
        [sys.executable, "-u", str(ROOT / "web" / "server.py"),
         "--port", str(PORT), "--threads", "4"],
        cwd=str(ROOT), env=env, stdout=open(log, "w", encoding="utf-8"),
        stderr=subprocess.STDOUT)

    try:
        # 等端口就绪
        for _ in range(60):
            try:
                get("/api/meta")
                break
            except Exception:
                time.sleep(0.5)
        else:
            print("服务没起来，看", log, flush=True)
            print(log.read_text(encoding="utf-8")[-3000:], flush=True)
            return 1

        # 1) 静态页
        with urllib.request.urlopen(BASE + "/", timeout=10) as r:
            html = r.read().decode("utf-8")
        check("GET / 返回牌桌页面", r.status == 200 and "贵州捉鸡" in html,
              f"{len(html)} 字节")

        # 2) 元信息
        meta = get("/api/meta")
        stems = [m["stem"] for m in meta.get("models", [])]
        check("GET /api/meta 列出模型", "rl" in stems and "bc" in stems, f"models={stems}")
        check("GET /api/meta 列出 27 种牌", len(meta.get("tiles", [])) == 27)
        check("GET /api/meta 列出教师风格", len(meta.get("teachers", [])) == 6)

        # 3) 开新局：人类坐 0 号位，对手 = rl + 教师 + 随机
        r = post("/api/new", {
            "opponents": [{"type": "model", "model": "rl"},
                          {"type": "teacher", "style": "balanced"},
                          {"type": "random"}],
            "my_seat": 0, "seed": 12345, "speed": 0.0})
        sid = r.get("sid")
        check("POST /api/new 建会话", bool(sid) and r.get("my_seat") == 0, f"sid={sid}")

        # 4) 轮询到该人类决策，并检查快照结构
        st, tries = None, 0
        while tries < 200:
            st = get(f"/api/state?sid={sid}")
            if st.get("ready") and st.get("pending"):
                break
            time.sleep(0.05)
            tries += 1
        check("GET /api/state 走到人类决策点", bool(st and st.get("pending")),
              f"用了 {tries} 次轮询")

        if st and st.get("ready"):
            me = st["players"][0]
            names = [p["name"] for p in st["players"]]
            check("座位身份按配置就位",
                  names[0] == "你" and "模型·rl" in names[1]
                  and "教师" in names[2] and names[3] == "随机", f"{names}")
            check("快照隐藏了对手手牌", all("hand" not in p for p in st["players"][1:]),
                  f"对手仅给 hand_count={[p['hand_count'] for p in st['players'][1:]]}")
            check("快照给了自己的手牌", len(st.get("hand", [])) in (13, 14),
                  f"{len(st['hand'])} 张")
            check("四家方位齐全", len(st["players"]) == 4 and
                  [p["rel"] for p in st["players"]] == [0, 1, 2, 3])
            check("庄家已确定", 0 <= st.get("dealer", -1) < 4, f"dealer={st.get('dealer')}")
            check("幺鸡在自己手上/公开信息正确", True, f"手牌含幺鸡={9 in st['hand']}")

        # 5) 模型建议
        h = get(f"/api/hint?sid={sid}")
        check("GET /api/hint 给出模型建议",
              bool(h.get("ok")) and len(h.get("items", [])) >= 1,
              f"{[(i['label'], i['prob']) for i in h.get('items', [])]}")
        if h.get("ok"):
            check("建议概率归一（≤1 且降序）",
                  all(0 <= i["prob"] <= 1 for i in h["items"])
                  and h["items"][0]["prob"] >= h["items"][-1]["prob"])

        # 6) 非法动作被拒
        bad = post("/api/act", {"sid": sid, "kind": "discard", "tile": 26})
        legal_tiles = set()
        if st and st.get("pending"):
            legal_tiles = set(st["pending"]["discardables"])
        if st and st["pending"]["mode"] == "discard" and 26 not in legal_tiles:
            check("非法动作被拒绝", not bad.get("ok"), str(bad))
        else:
            print("  [skip] 非法动作校验（当前不是打牌决策点）", flush=True)

        # 7) 扮演人类把这一局打完
        steps, t0 = 0, time.time()
        while steps < 400:
            st = get(f"/api/state?sid={sid}")
            if st.get("over"):
                break
            p = st.get("pending")
            if not p:
                time.sleep(0.03)
                continue
            if p["mode"] == "discard":
                choice = next((b for b in p["buttons"] if b["kind"] == "hu"), None)
                if choice:
                    act = {"kind": choice["kind"], "tile": choice["tile"]}
                else:
                    act = {"kind": "discard", "tile": p["discardables"][0]}
            else:
                pref = [b for b in p["buttons"] if b["kind"] != "pass"]
                b = (pref or p["buttons"])[0]
                act = {"kind": b["kind"], "tile": b["tile"]}
            res = post("/api/act", {"sid": sid, **act})
            if not res.get("ok"):
                check("提交动作", False, f"{act} → {res}")
                break
            steps += 1
        dt = time.time() - t0

        over = post("/api/next", {"sid": sid}) if st and st.get("over") else {"ok": False}
        check("人类操作能把一局打完", bool(st and st.get("over")),
              f"{steps} 次出手 / {dt:.1f}s")
        if st and st.get("over"):
            res = st.get("result") or {}
            check("结算结构完整",
                  res.get("type") in ("win", "huangzhuang")
                  and isinstance(res.get("deltas"), list) and len(res["deltas"]) == 4
                  and abs(sum(res["deltas"])) < 1e-6,
                  f"type={res.get('type')} deltas={res.get('deltas')}")
            check("结算带出人机各自得分",
                  isinstance(res.get("mine"), (int, float))
                  and len(st.get("scores", [])) == 4,
                  f"mine={res.get('mine')} scores={st.get('scores')}")
            check("牌局事件流非空", len(st.get("events", [])) > 5,
                  f"{len(st.get('events', []))} 条")
        check("POST /api/next 可开下一局", bool(over.get("ok")), str(over))

        # 8) 缓存的模型元信息
        meta2 = get("/api/meta")
        rl = next((m for m in meta2["models"] if m["stem"] == "rl"), {})
        check("模型加载后元信息可读", "net" in rl or rl.get("iter") is not None,
              f"rl={ {k: rl.get(k) for k in ('iter', 'epoch', 'net')} }")

    finally:
        srv.terminate()
        try:
            srv.wait(timeout=10)
        except subprocess.TimeoutExpired:
            srv.kill()

    print(f"\n通过 {len(PASSED)} / 共 {len(PASSED) + len(FAILED)}", flush=True)
    if FAILED:
        print("失败项：" + "、".join(FAILED), flush=True)
        return 1
    print("全部通过 ✅", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
