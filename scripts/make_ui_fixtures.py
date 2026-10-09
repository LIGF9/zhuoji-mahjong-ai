"""采集真实对局快照，生成前端（jsdom）校验用的夹具。

产出（reports/）::

    ui_fixtures_meta.json   /api/meta 原样
    ui_fixtures_over.json   若干「终局」快照 [{"seed": n, "state": {...}}]
    ui_fixtures_mid.json    若干「我的决策点」快照 [{"tag": "midN", "state": {...}}]

所有快照都来自真实服务端（新进程、新代码），因此天然包含结算新增字段
（winners / winner_names / winner_hands / winner_melds_map / winners_fan /
void_name / rob_kong / how 等），前端校验才有意义。

跑法::

    python scripts/make_ui_fixtures.py [局数] [端口]
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PORT = int(sys.argv[2]) if len(sys.argv) > 2 else 8788
HANDS = int(sys.argv[1]) if len(sys.argv) > 1 else 14
BASE = f"http://127.0.0.1:{PORT}"

OPP_MIXES = [
    [{"type": "model", "model": "master", "temperature": 0},
     {"type": "teacher", "style": "balanced"},
     {"type": "random"}],
    [{"type": "teacher", "style": "tenpai_rush"},
     {"type": "teacher", "style": "aggressive"},
     {"type": "random"}],
    [{"type": "model", "model": "master", "temperature": 0},
     {"type": "teacher", "style": "chicken_lover"},
     {"type": "teacher", "style": "gambler"}],
]


def call(path: str, body: dict | None = None, timeout: float = 60.0):
    data = json.dumps(body).encode("utf-8") if body is not None else None
    req = urllib.request.Request(
        BASE + path, data=data,
        headers={"Content-Type": "application/json"} if data else {})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


def play_session(seed: int, opps: list[dict], want: int,
                 overs: list[dict], mids: list[dict], want_mid: bool) -> int:
    """打完一场（playing until `want` hands or session ends）；返回完成局数。"""
    r = call("/api/new", {"opponents": opps, "my_seat": 0, "speed": 0.0,
                          "auto": False, "seed": seed})
    if not r.get("ok"):
        print("  开局失败", r)
        return 0
    sid = r["sid"]
    mid_saved = False
    done = 0
    deadline = time.time() + 300
    while done < want and time.time() < deadline:
        st = call(f"/api/state?sid={sid}", timeout=30)
        if st.get("error"):
            print("  服务端异常：", str(st["error"])[:200])
            break
        if st.get("over"):
            res = st.get("result") or {}
            overs.append({"seed": seed * 1000 + st["hand_index"], "state": st})
            tag = res.get("type")
            if tag == "win":
                ws = res.get("winners") or [res.get("winner")]
                tag += f"/{res.get('how')}/赢家{len(ws)}"
                if res.get("void_name"):
                    tag += "/全烧"
            print(f"  局 {st['hand_index']} 结束：{tag} 我 {res.get('mine'):+.1f}")
            done += 1
            call("/api/next", {"sid": sid})
            time.sleep(0.15)
            continue
        pend = st.get("pending")
        if not mid_saved and want_mid and pend and st.get("turn") == st.get("my_seat"):
            mids.append({"tag": f"mid{len(mids)}", "state": st})
            mid_saved = True
        if not pend or st.get("turn") != st.get("my_seat"):
            time.sleep(0.1)
            continue
        h = call(f"/api/hint?sid={sid}&strategy=model:master", timeout=60)
        legal = {("discard", int(t)) for t in (pend.get("discardables") or [])}
        legal |= {(b["kind"], int(b["tile"])) for b in (pend.get("buttons") or [])}
        act = None
        if h.get("ok") and h.get("items"):
            a = h["items"][0]
            key = (a["kind"], int(a["tile"]))
            if key in legal:
                act = key
        if act is None:
            act = (("discard", pend["discardables"][0])
                   if pend.get("mode") == "discard" and pend.get("discardables")
                   else ("pass", -1))
        call("/api/act", {"sid": sid, "kind": act[0], "tile": act[1]})
    try:
        call("/api/quit", {"sid": sid})
    except Exception:
        pass
    return done


def main() -> int:
    log = ROOT / "reports" / "_ui_fixtures_server.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUNBUFFERED"] = "1"

    print(f"启动服务 {BASE} …", flush=True)
    srv = subprocess.Popen(
        [sys.executable, "-u", str(ROOT / "web" / "dushan_server.py"),
         "--port", str(PORT), "--threads", "4"],
        cwd=str(ROOT), env=env, stdout=open(log, "w", encoding="utf-8"),
        stderr=subprocess.STDOUT)
    try:
        for _ in range(120):
            try:
                meta = call("/api/meta", timeout=5)
                break
            except Exception:
                time.sleep(0.5)
        else:
            print("服务没起来，看", log)
            return 1

        overs: list[dict] = []
        mids: list[dict] = []
        base_seed = 700
        i = 0
        while len(overs) < HANDS and i < 40:
            seed = base_seed + i * 7
            opps = OPP_MIXES[i % len(OPP_MIXES)]
            print(f"会话 {i}（seed={seed}）…", flush=True)
            play_session(seed, opps, want=HANDS - len(overs), overs=overs, mids=mids,
                         want_mid=len(mids) < 2)
            i += 1

        (ROOT / "reports" / "ui_fixtures_meta.json").write_text(
            json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
        (ROOT / "reports" / "ui_fixtures_over.json").write_text(
            json.dumps(overs, ensure_ascii=False), encoding="utf-8")
        (ROOT / "reports" / "ui_fixtures_mid.json").write_text(
            json.dumps(mids, ensure_ascii=False), encoding="utf-8")

        wins = [o for o in overs if o["state"]["result"]["type"] == "win"]
        multi = [o for o in wins if len(o["state"]["result"].get("winners") or []) > 1]
        void = [o for o in wins if o["state"]["result"].get("void_name")]
        hows = sorted({o["state"]["result"].get("how") for o in wins})
        print(f"\n夹具：终局 {len(overs)}（胡 {len(wins)} / 黄 {len(overs) - len(wins)}）· "
              f"决策点 {len(mids)}")
        print(f"  胡牌方式集合 {hows}｜一炮多响 {len(multi)} 局｜鸡分全烧 {len(void)} 局")
        return 0
    finally:
        srv.terminate()
        try:
            srv.wait(timeout=10)
        except Exception:
            srv.kill()


if __name__ == "__main__":
    raise SystemExit(main())
