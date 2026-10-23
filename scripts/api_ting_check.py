"""``/api/ting``（打牌听牌提示）冒烟 + 正确性校验。

做两件事：

1. **接口冒烟**：起一个临时服务，用「真人座位」打若干手牌，每个决策点都调
   ``/api/ting``，校验返回结构（``map[tile] = {tiles:[{t,left}], types:[...]}``），
   且 map 的键必须落在真正的可打牌集合里。
2. **正确性对拍**：把同一局面用**引擎**重算一遍（``DushanGame.ting_info``），
   「还剩几张」用**本脚本内独立实现**的计数公式（不 import 服务端代码，避免同源
   自证），与接口返回值逐字段比对。

跑法::

    python scripts/api_ting_check.py [局数] [端口]
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
sys.path.insert(0, str(ROOT))

from zhuoji.dushan import DushanConfig, DushanGame  # noqa: E402
from zhuoji.tiles import NUM_TILE_TYPES, tile_cn  # noqa: E402

HANDS = int(sys.argv[1]) if len(sys.argv) > 1 else 3
PORT = int(sys.argv[2]) if len(sys.argv) > 2 else 8791
BASE = f"http://127.0.0.1:{PORT}"

KONG_MELDS = (2, 3, 4)     # 明杠 / 补杠 / 暗杠（碰 = 1）

ok = 0
bad = 0


def check(name: str, cond: bool, detail: str = "") -> None:
    global ok, bad
    if cond:
        ok += 1
    else:
        bad += 1
    print(f"  [{'ok  ' if cond else 'FAIL'}] {name}" + (f"   {detail}" if detail else ""))


def call(path: str, body: dict | None = None, timeout: float = 60.0):
    data = json.dumps(body).encode("utf-8") if body is not None else None
    req = urllib.request.Request(
        BASE + path, data=data,
        headers={"Content-Type": "application/json"} if data else {})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


def left_count(hands, melds, discards, me: int, drop: int, tile: int) -> int:
    """独立实现的「还剩几张」：4 −（四家弃牌 + 四家副露 + 我的暗手）。

    ``discards`` 必须是**真实牌河**（被碰/明杠吃走的牌已剔除），否则会与副露重复计张。
    """
    seen = sum(int(discards[p].count(tile)) for p in range(4))
    for p in range(4):
        for mt, t, _s in melds[p]:
            if int(t) == tile:
                seen += 4 if int(mt) in KONG_MELDS else 3
    seen += int(hands[me][tile]) - (1 if drop == tile else 0)
    return max(0, 4 - seen)


def mirror(st: dict) -> dict:
    """用服务端下发的公开/私有牌面还原局面，再按引擎口径重算提示。

    ⚠️ 2026-10-13 修正：``players[p]["discards"]`` 是**展示视图**
    （``discard_view``，被碰/明杠吃走的牌**原地保留**、配 ``discard_claimed`` 标记），
    而引擎的权威 ``game.discards`` 里那几张**已经移除**。若照搬展示视图去重算，
    被吃走的牌会被「弃牌 + 副露」**重复计一次**，于是同一张牌本算 3 张已见
    却算成 4 张 → ``left`` 少 1。故这里先剔掉 claimed 的牌再喂给引擎。
    """
    me = int(st["my_seat"])
    game = DushanGame(DushanConfig(), seed=0)
    hands, melds, discards = [], [], []
    for p in range(4):
        pl = st["players"][p]
        h = [0] * NUM_TILE_TYPES
        for x in (pl.get("hand_tiles") or []):
            h[int(x)] += 1
        hands.append(h)
        melds.append([(int(m["type"]), int(m["tile"]), int(m["src"]))
                      for m in (pl.get("melds") or [])])
        view = [int(x) for x in (pl.get("discards") or [])]
        claimed = [bool(b) for b in (pl.get("discard_claimed") or [])]
        if claimed and len(claimed) == len(view):
            view = [t for t, c in zip(view, claimed) if not c]
        discards.append(view)
    game.hands, game.melds, game.discards = hands, melds, discards
    out: dict[str, dict] = {}
    for t in range(NUM_TILE_TYPES):
        if hands[me][t] <= 0:
            continue
        tiles, types = game.ting_info(me, drop=t)
        if not tiles:
            continue
        out[str(t)] = {
            "tiles": [{"t": int(w), "left": left_count(hands, melds, discards, me, t, int(w))}
                      for w in tiles],
            "types": [str(x) for x in types if str(x) != "平胡"],
        }
    return out


def main() -> int:
    log = ROOT / "reports" / "_api_ting_server.log"
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
                call("/api/meta", timeout=5)
                break
            except Exception:
                time.sleep(0.5)
        else:
            print("服务没起来，看", log)
            return 1

        r = call("/api/new", {"opponents": [
            {"type": "teacher", "style": "balanced"},
            {"type": "teacher", "style": "tenpai_rush"},
            {"type": "random"}], "my_seat": 0, "speed": 0.0, "seed": 4242})
        sid = r["sid"]
        print("会话", sid)

        points = 0
        ting_points = 0
        samples: list[str] = []
        compared = 0
        done = 0
        deadline = time.time() + 300

        while done < HANDS and time.time() < deadline:
            st = call(f"/api/state?sid={sid}", timeout=30)
            if st.get("error"):
                print("  服务端异常：", str(st["error"])[:200])
                break
            if st.get("over"):
                done += 1
                print(f"  局 {st['hand_index']} 结束：{(st.get('result') or {}).get('type')}")
                call("/api/next", {"sid": sid})
                time.sleep(0.15)
                continue
            pend = st.get("pending")
            if not pend or st.get("turn") != st.get("my_seat"):
                time.sleep(0.08)
                continue

            # ---- 调用被测接口 ----
            t = call(f"/api/ting?sid={sid}", timeout=30)
            points += 1
            if not t.get("ok"):
                check("接口返回 ok", False, json.dumps(t, ensure_ascii=False)[:160])
                break
            mp = t.get("map") or {}
            legal = set(int(x) for x in (pend.get("discardables") or []))
            struct_ok = isinstance(mp, dict)
            for k, v in mp.items():
                struct_ok &= (k.isdigit() and int(k) in legal
                              and isinstance(v, dict)
                              and isinstance(v.get("tiles"), list) and v["tiles"]
                              and all(isinstance(x, dict) and isinstance(x.get("t"), int)
                                      and isinstance(x.get("left"), int)
                                      and 0 <= x["left"] <= 4 for x in v["tiles"])
                              and isinstance(v.get("types"), list))
            check(f"决策点 {points}：结构合法且键⊆可打牌", bool(struct_ok),
                  f"可打 {len(legal)} 张 / 打出可听 {len(mp)} 张")
            if mp:
                ting_points += 1
            if mp and len(samples) < 3:
                k, v = next(iter(mp.items()))
                samples.append(f"打 {tile_cn(int(k))} → 听 " +
                               "、".join(f"{tile_cn(x['t'])}剩{x['left']}" for x in v["tiles"]) +
                               (f"（{'/'.join(v['types'])}）" if v["types"] else ""))

            # ---- 与本地引擎逐字段对拍 ----
            mine = mirror(st)
            same = json.dumps(mine, sort_keys=True) == json.dumps(mp, sort_keys=True)
            compared += 1
            check(f"决策点 {points}：与引擎逐字段一致", same,
                  "" if same else f"\n        local={json.dumps(mine, ensure_ascii=False)}\n"
                                  f"        api  ={json.dumps(mp, ensure_ascii=False)}")

            # ---- 出牌推进 ----
            if pend.get("mode") == "discard" and pend.get("discardables"):
                act = ("discard", pend["discardables"][0])
            elif any(b["kind"] == "pass" for b in (pend.get("buttons") or [])):
                act = ("pass", -1)
            else:
                b = (pend.get("buttons") or [])[0]
                act = (b["kind"], b["tile"])
            call("/api/act", {"sid": sid, "kind": act[0], "tile": act[1]})

        print("\n样例：")
        for s in samples:
            print("   ", s)
        print(f"\n决策点 {points}（打出可听 {ting_points}）· 对拍 {compared} 次 · "
              f"通过 {ok} · 失败 {bad}")
        try:
            call("/api/quit", {"sid": sid})
        except Exception:
            pass
        return 0 if bad == 0 and points > 0 else 2
    finally:
        srv.terminate()
        try:
            srv.wait(timeout=10)
        except Exception:
            srv.kill()


if __name__ == "__main__":
    raise SystemExit(main())
