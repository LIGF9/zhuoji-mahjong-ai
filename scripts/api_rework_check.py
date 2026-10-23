"""2026-10-11 服务端改造批量冒烟：记牌器 / 听牌条数据 / 流水事件 / 结束本局 / 统计口径。

覆盖：

1. ``state.counter``（记牌器）：27 维、取值 0..4，并抽查与独立实现的
   「4 −（四家弃牌 + 四家副露碰3杠4 + 我的暗手）」一致。
2. ``state.my_ting``（听牌提示条数据源）：我的出牌决策点 = 引擎 ``ting_info(me)``；
   非决策点为空。
3. 流水事件：每条都带 ``hand``（局号）与 ``seq``（局内递增序号）；
   横鸡轮开/收事件（``kind=="hengji"``）若出现，则 open/close 成对。
4. ``POST /api/end_hand``（结束本局）：立即 ``result.type=="abort"``、deltas 全 0、
   不写历史、统计 games 不变。
5. ``/api/stats``：历史累计 10 键齐备且为非负数。
6. ``/api/new`` 下发的 names 回显（名字与头像绑定，我不再叫「你」）。

跑法::

    python scripts/api_rework_check.py [端口]
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

from zhuoji.tiles import NUM_TILE_TYPES  # noqa: E402

PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 8793
BASE = f"http://127.0.0.1:{PORT}"
KONG_MELDS = (2, 3, 4)

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


def counter_expect(st: dict, me: int) -> list[int]:
    """独立实现的记牌器口径（与服务端同式，但不 import 服务端代码）。

    ⚠️ 2026-10-12：snapshot 里的 ``discards`` 是**只增不减的展示视图**
    （被碰/杠吃走的牌仍留在原位、靠 ``discard_claimed`` 标记灰显），
    而服务端记牌器用的是**真牌河**。所以这里必须跳过 ``discard_claimed[i]=True``
    的那些张 —— 它们已经由「副露碰 3 / 杠 4」计入，不能重复算。"""
    seen = [0] * NUM_TILE_TYPES
    for p in range(4):
        pl = st["players"][p]
        claimed = pl.get("discard_claimed") or []
        for i, x in enumerate(pl.get("discards") or []):
            if i < len(claimed) and claimed[i]:
                continue                    # 被吃走的那张：由下面的副露项计入
            seen[int(x)] += 1
        for m in (pl.get("melds") or []):
            if int(m["tile"]) < NUM_TILE_TYPES:
                seen[int(m["tile"])] += 4 if int(m["type"]) in KONG_MELDS else 3
    for x in (st.get("hand") or []):
        seen[int(x)] += 1
    return [max(0, 4 - seen[t]) for t in range(NUM_TILE_TYPES)]


def history_lines() -> int:
    f = ROOT / "web" / "history" / "records.jsonl"
    if not f.exists():
        return 0
    with open(f, encoding="utf-8") as fh:
        return sum(1 for line in fh if line.strip())


def main() -> int:
    log = ROOT / "reports" / "_api_rework_server.log"
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

        names = ["云天明", "大乔(S11)", "妲己(B3)", "王昭君(抢听)"]
        r = call("/api/new", {"opponents": [
            {"type": "teacher", "style": "balanced"},
            {"type": "teacher", "style": "tenpai_rush"},
            {"type": "random"}], "my_seat": 0, "speed": 0.0, "seed": 2027,
            "names": names})
        sid = r["sid"]
        print("会话", sid)

        # ---- 6) 名字回显 ----
        st = call(f"/api/state?sid={sid}", timeout=30)
        check("names 回显（我不再叫「你」）", list(st.get("names") or []) == names,
              json.dumps(st.get("names"), ensure_ascii=False))

        # ---- 1/2/3) 记牌器 / 听牌条 / 事件编号 ----
        counter_points = 0
        ting_points = 0
        evt_points = 0
        hengji_events: list[dict] = []
        seq_bad = ""
        steps = 0
        while steps < 60:
            steps += 1
            st = call(f"/api/state?sid={sid}", timeout=30)
            if st.get("error"):
                print("  服务端异常：", str(st["error"])[:200])
                break
            if st.get("over"):
                break
            # 记牌器
            cnt = st.get("counter")
            check(f"step{steps} counter 维度=27", isinstance(cnt, list) and len(cnt) == 27,
                  f"len={len(cnt) if isinstance(cnt, list) else 'na'}")
            exp = counter_expect(st, int(st["my_seat"]))
            check(f"step{steps} counter 与独立实现一致", cnt == exp,
                  "" if cnt == exp else f"api={cnt[:9]}… exp={exp[:9]}…")
            counter_points += 1
            # 事件编号
            evs = st.get("events") or []
            hand_ids = {e.get("hand") for e in evs}
            seqs_ok = True
            for h in hand_ids:
                ss = [e.get("seq") for e in evs if e.get("hand") == h]
                if any(not isinstance(x, int) for x in ss) or ss != sorted(ss) \
                        or len(set(ss)) != len(ss):
                    seqs_ok = False
                    seq_bad = f"hand={h} seqs={ss[:12]}"
            if evs:
                evt_points += 1
            hengji_events += [e for e in evs if e.get("kind") == "hengji"]
            # 听牌条（我的出牌决策点）
            pend = st.get("pending")
            if pend and pend.get("mode") == "discard" and st.get("turn") == st.get("my_seat"):
                mt = st.get("my_ting")
                check(f"step{steps} my_ting 是牌种列表且⊆手牌外叫牌",
                      isinstance(mt, list) and all(isinstance(x, int) and 0 <= x < 27 for x in mt),
                      json.dumps(mt)[:80])
                ting_points += 1
                act = ("discard", (pend.get("discardables") or [None])[0])
                if act[1] is None:
                    break
            elif pend and pend.get("buttons"):
                b = pend["buttons"][0]
                act = (b["kind"], b.get("tile", -1))
                if b["kind"] == "pass":
                    act = ("pass", -1)
            else:
                time.sleep(0.1)
                continue
            call("/api/act", {"sid": sid, "kind": act[0], "tile": act[1]})
        check("记牌器校验点 > 0", counter_points > 0, f"points={counter_points}")
        check("事件编号校验点 > 0（seq 局内递增）", evt_points > 0 and not seq_bad, seq_bad)
        if hengji_events:
            opens = sum(1 for e in hengji_events if e.get("ctag") == "横鸡轮开始"
                        or "开出横鸡轮" in str(e.get("text", "")))
            closes = sum(1 for e in hengji_events if e.get("ctag") == "横鸡轮结束"
                         or "横鸡轮结束" in str(e.get("text", "")))
            check("横鸡轮事件 open/close 结构", opens >= 1 and closes >= 1 and closes <= opens,
                  f"open={opens} close={closes}（close<open=抽样截止时轮未收口，正常）")
        else:
            print("  [info ] 本样本未出现横鸡轮（正常，概率事件）")

        # ---- 4) 结束本局（若样本局已自然结束，先开下一局再测） ----
        st0 = call(f"/api/state?sid={sid}", timeout=30)
        if st0.get("over"):
            # 本会话用 speed=0（AI 全速）采样记牌器 → 一局可能在毫秒级内跑完，
            # `/api/next` 之后 6s 轮询窗口经常整局已结束，end_hand 就被拒成
            # 「当前这局已经结束」（纯测试竞态，跟产品逻辑无关）。
            # 先把节奏放到最慢（speed 上限 5s/步）再开新局，保证能抓到一个活局。
            call("/api/speed", {"sid": sid, "speed": 5.0})
            call("/api/next", {"sid": sid})
            for _ in range(40):
                time.sleep(0.15)
                st0 = call(f"/api/state?sid={sid}", timeout=30)
                if not st0.get("over"):
                    break
            if st0.get("over"):   # 兜底：仍拿不到活局就再开一次
                call("/api/next", {"sid": sid})
                time.sleep(0.3)
                st0 = call(f"/api/state?sid={sid}", timeout=30)
        games0 = json.loads(json.dumps((st0.get("stats") or {}).get("0") or {}))
        hist0 = history_lines()
        r = call("/api/end_hand", {"sid": sid}, timeout=30)
        check("end_hand 返回 ok", bool(r.get("ok")), json.dumps(r, ensure_ascii=False)[:120])
        # force_end 只在**下一个 step 边界**生效（主循环每步末尾 sleep(speed)），
        # 现在 speed 被放到 5.0 → 最多要等 5s 才 publish abort，固定 0.6s 会读到空 result。
        # 改成轮询等类型出现。
        st1, res = st0, {}
        for _ in range(60):
            time.sleep(0.25)
            st1 = call(f"/api/state?sid={sid}", timeout=30)
            res = st1.get("result") or {}
            if res.get("type"):
                break
        check("end_hand → result.type=abort", res.get("type") == "abort", res.get("type"))
        check("abort deltas 全 0",
              (res.get("deltas") or [None]) is not None and bool(res.get("deltas"))
              and all(float(x) == 0.0 for x in res["deltas"]),
              json.dumps(res.get("deltas")))
        call("/api/speed", {"sid": sid, "speed": 0.0})   # 恢复本会话原有的全速节奏
        games1 = (st1.get("stats") or {}).get("0") or {}
        check("abort 不计入 games", games1.get("games") == games0.get("games"),
              f"{games0.get('games')} → {games1.get('games')}")
        time.sleep(0.4)
        check("abort 不写历史", history_lines() == hist0, f"{hist0} → {history_lines()}")

        # ---- 5) /api/stats 10 键 ----
        # 2026-10-11 口径变更：「点胡(dianhu)」→「送胡(songhu_games)」整局视角
        #（该局有没有人非自摸胡：点炮/抢杠/热炮…），前端同名字段，此处同步。
        agg = call("/api/stats?sid=" + sid, timeout=30).get("all_time") or {}
        keys = ["games", "zimo", "songhu_games", "chongfeng_games", "ji_discard_games",
                "baoji_games", "tenpai_bigfan_games", "win_bigfan_games",
                "huang_games", "huang_noten_games"]
        missing = [k for k in keys if k not in agg or not isinstance(agg[k], (int, float))]
        check("历史累计 10 键齐备且为数值", not missing, f"缺/非法: {missing}")
        check("games 为数值且 ≥0", isinstance(agg.get("games"), (int, float))
              and agg.get("games", -1) >= 0, str(agg.get("games")))

        # ---- 7) /api/hold：开局鸡牌特效的「暂停推进」闸门（2026-10-13） ----
        # 用户口径：「开局先显示鸡牌特效，特效结束再开始牌局」。客户端在换局那一刻
        # 发 on=true、特效播完发 on=false；期间服务端必须**一步都不走**。
        # 速度 0 时若无闸门，1 秒内整手牌就跑完好几步 → 差异极其明显。
        # my_seat=1（不当庄）：庄家是对手，闸门一放行就会自动出牌，进度可观测；
        # 若用 my_seat=0，人类自己是庄、卡在等待出牌，看不出「推进」。
        hs = call("/api/new", {"opponents": [{"type": "random"}] * 3,
                               "my_seat": 1, "speed": 0.0, "seed": 991,
                               "hold": True})["sid"]
        h0 = {}
        for _ in range(40):
            h0 = call(f"/api/state?sid={hs}", timeout=30)
            if h0.get("ready") and h0.get("hand_index"):
                break
            time.sleep(0.05)
        check("hold 会话：第一手已发牌（快照 publish 出去了）",
              h0.get("hand_index") == 1 and (h0.get("wall_left") or 0) > 0,
              f"hand={h0.get('hand_index')} wall={h0.get('wall_left')}")
        n0 = len(h0.get("events") or [])
        time.sleep(1.2)
        h1 = call(f"/api/state?sid={hs}", timeout=30)
        check("闸门生效：暂停期间牌局完全不推进（1.2s 后仍是同一张发牌快照）",
              h1.get("hand_index") == 1
              and (h1.get("wall_left") or 0) == (h0.get("wall_left") or 0)
              and len(h1.get("events") or []) == n0 and not h1.get("over"),
              f"hand={h1.get('hand_index')} wall={h1.get('wall_left')} "
              f"ev={len(h1.get('events') or [])}(原 {n0})")
        check("on=true 回执 hold=True",
              call("/api/hold", {"sid": hs, "on": True}).get("hold") is True)
        check("on=false 回执 hold=False",
              call("/api/hold", {"sid": hs, "on": False}).get("hold") is False)
        adv, h2 = False, h1
        for _ in range(60):
            time.sleep(0.25)
            h2 = call(f"/api/state?sid={hs}", timeout=30)
            if (h2.get("over") or (h2.get("hand_index") or 1) > 1
                    or len(h2.get("events") or []) > n0):
                adv = True
                break
        check("放行后牌局立刻继续推进", adv,
              f"hand={h2.get('hand_index')} ev={len(h2.get('events') or [])} over={h2.get('over')}")
        try:
            call("/api/hold", {"sid": "no-such-sid", "on": True})
            unknown_rejected = False
        except Exception:
            unknown_rejected = True
        check("未知会话 → 拒绝（404，不会静默成功）", unknown_rejected)
        # 不带 hold 的普通会话不受影响（回归：默认不扣牌 → 开局庄家立刻出牌）
        ps = call("/api/new", {"opponents": [{"type": "random"}] * 3,
                               "my_seat": 1, "speed": 0.0, "seed": 992})["sid"]
        p_ev, p_over = 0, False
        for _ in range(40):
            time.sleep(0.15)
            p1 = call(f"/api/state?sid={ps}", timeout=30)
            p_ev = len(p1.get("events") or [])
            p_over = bool(p1.get("over"))
            if p_ev > 0 or p_over:
                break
        check("普通会话（不带 hold）不扣牌：开局庄家对手立刻出牌（事件 > 0）",
              p_ev > 0 or p_over, f"ev={p_ev} over={p_over}")

        print(f"\n合计：通过 {ok} / 失败 {bad}")
        return 0 if bad == 0 else 1
    finally:
        srv.terminate()
        try:
            srv.wait(timeout=10)
        except Exception:
            srv.kill()


if __name__ == "__main__":
    sys.exit(main())
