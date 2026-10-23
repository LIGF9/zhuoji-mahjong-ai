"""replay_e2e.py —— 对局回放的端到端校验

它做四件事：
  1) 用 HTTP 接口真打完整一局（人类座位由脚本代打：只在合法时点提交「打第一张可打的牌」），
     触发服务端落盘记录 + 回放数据
  2) 校验记录里带 rid、/api/history 会标 has_replay
  3) 校验 /api/replay 的事件流形状：seq 递增、rel 在 0..3、巡号非降、
     每条带 4 个手牌张数 hc、text 非空
  4) **重建一致性**：把事件流重放一遍（打牌→牌河、碰/杠→副露、hc 直接用服务端值），
     断言重建出的牌河总数与「打牌事件数」一致、最后一帧的 hc 与引擎终局手牌张数一致
     （终局手牌张数用记录里的 tenpai/winner_hand 反推校核）

用法：
    <venv>/python.exe scripts/replay_e2e.py [--port 8770] [-v]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request

ROOT_URL = "http://127.0.0.1:{port}"
SID = "replay-e2e"


def call(base: str, path: str, body: dict | None = None, timeout: float = 20.0):
    url = base + path
    data = json.dumps(body).encode("utf-8") if body is not None else None
    req = urllib.request.Request(
        url, data=data,
        headers={"Content-Type": "application/json"} if data else {})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8770)
    ap.add_argument("-v", "--verbose", action="store_true")
    ap.add_argument("--fast", action="store_true",
                    help="对手全用 random（不加载教师模型）——算力紧张时用，验证的是回放链路本身")
    a = ap.parse_args()
    base = ROOT_URL.format(port=a.port)

    bad = []

    def check(name, ok, extra=""):
        print(f"  {'PASS' if ok else 'FAIL'}  {name}{('  ' + extra) if extra else ''}")
        if not ok:
            bad.append(name)

    # ---- 1. 打完整一局：对手用瞬间速度，人类座位由脚本代打 ----
    # 注意：/api/new **不使用**客户端传的 sid —— 会话 id 由服务端现生成并回传，
    # 后续所有请求必须用 new["sid"]（早期脚本用自己编的 sid 会一路 404）。
    print("=== 1. 真打一局 ===")
    opps = ([{"type": "random"}, {"type": "random"}, {"type": "random"}] if a.fast else
            [{"type": "teacher", "style": "balanced"},
             {"type": "teacher", "style": "defensive"},
             {"type": "random"}])
    t0_ms = int(time.time() * 1000)   # 开局时刻：第 2 段用它校验「记录确实是本局新落盘的」
    new = call(base, "/api/new", {
        "opponents": opps,
        "my_seat": 0,
        "speed": 0,
    })
    check("开局返回 ok", bool(new.get("ok")), str(new)[:120])
    sid = str(new.get("sid") or "")
    check("服务端回传 sid", bool(sid), sid)
    seat = int(new.get("my_seat") or 0)
    check("我的座位回传", 0 <= seat <= 3, f"seat={seat}")

    deadline = time.time() + 240
    acted = 0
    last_sig = None
    st = {}
    last_prog = None
    stuck_since = time.time()
    while time.time() < deadline:
        try:
            st = call(base, f"/api/state?sid={sid}", timeout=30)
        except Exception as exc:
            check("轮询 /api/state", False, str(exc))
            break
        if st.get("over"):
            break
        # 停滞检测：引擎侧可见量（牌墙 / 事件数 / 行动者 / 阶段）长时间不变 → 推进停了。
        # 实测这台机器上训练占满 CPU 时，教师模型推理会被拖到几十秒一次，表现就是
        # 「提交几手后就不再轮到我」。给出诊断比死等到超时有用得多。
        prog = (st.get("wall_left"), len(st.get("events") or []), st.get("turn"), st.get("phase"))
        if prog != last_prog:
            last_prog = prog
            stuck_since = time.time()
        elif time.time() - stuck_since > 60:
            print(f"    [诊断] 引擎停滞 {int(time.time() - stuck_since)}s："
                  f"wall_left={st.get('wall_left')} turn={st.get('turn')} phase={st.get('phase')} "
                  f"error={st.get('error')} 事件={len(st.get('events') or [])}")
            break
        pend = st.get("pending")
        if pend and st.get("turn") == seat:
            mode = pend.get("mode")
            sig = json.dumps([mode, pend.get("buttons"), pend.get("discardables")], sort_keys=True)
            if sig != last_sig:
                last_sig = sig
                if mode == "discard":
                    dis = pend.get("discardables") or []
                    tile = dis[0] if dis else (st.get("hand") or [None])[0]
                    if tile is not None:
                        call(base, "/api/act", {"sid": sid, "kind": "discard", "tile": tile})
                        acted += 1
                else:
                    btns = pend.get("buttons") or []
                    if btns:
                        b = btns[-1]          # 默认选「过」
                        call(base, "/api/act", {"sid": sid,
                                                "kind": b.get("kind"), "tile": b.get("tile")})
                        acted += 1
        time.sleep(0.25)
    finished = bool(st.get("over"))
    check("在这一局结束前完成推进", finished, f"over={st.get('over')} 我共提交 {acted} 次动作")
    res = st.get("result") or {}
    print(f"    结果：type={res.get('type')} winners={res.get('winners')} deltas={res.get('deltas')}")
    if not finished:
        # 关键：没打完就**不能**往下走 —— 否则第 2 段读到的是上局的记录，
        # 后面三段会拿旧回放数据假装通过（早期就是这么假绿的）。
        print("    本局未打完 → 后续校验跳过（避免拿上局的旧记录冒充本局）")
        print(f"\n失败 {len(bad)} 项：" + "、".join(bad))
        return 1

    # ---- 2. 记录与 has_replay ----
    print("\n=== 2. 记录带 rid / has_replay ===")
    time.sleep(0.6)                      # 等 _settle 落盘
    hist = call(base, "/api/history?limit=3")
    recs = hist.get("records") or []
    rec = recs[0] if recs else {}
    rid_str = str(rec.get("rid") or "")
    _parts = rid_str.split("-")
    rid_ms = int(_parts[0]) if _parts and _parts[0].isdigit() else 0
    check("历史里有这一局", bool(rec), f"{len(recs)} 条")
    check("记录带 rid", bool(rid_str), rid_str)
    check("最新记录确实是本局新落盘的（不是上局残留）",
          rid_ms >= t0_ms - 2000, f"rid={rid_str} 开局={t0_ms} 差={rid_ms - t0_ms}ms")
    check("/api/history 标记 has_replay=True", rec.get("has_replay") is True)

    # ---- 3. 回放数据形状 ----
    print("\n=== 3. /api/replay 形状 ===")
    rp = call(base, f"/api/replay?rid={rec.get('rid')}").get("replay") or {}
    ev = rp.get("ev") or []
    check("事件流非空", len(ev) > 10, f"{len(ev)} 条")
    check("回放带 names / my_seat / dealer",
          isinstance(rp.get("names"), list) and rp.get("my_seat") is not None
          and rp.get("dealer") is not None, f"dealer={rp.get('dealer')}")
    shape_bad = []
    for i, e in enumerate(ev):
        if len(e) < 8:
            shape_bad.append(f"#{i} 字段数 {len(e)}"); continue
        if e[0] != i + 1:
            shape_bad.append(f"#{i} seq={e[0]}")
        # result / fanji 是「全场」事件，rel 固定 -1（不是某家的动作）
        if not (0 <= e[2] <= 3) and e[1] not in ("result", "fanji"):
            shape_bad.append(f"#{i} rel={e[2]}")
        if not isinstance(e[6], list) or len(e[6]) != 4:
            shape_bad.append(f"#{i} hc={e[6]}")
        if not isinstance(e[5], int) or e[5] < 1:
            shape_bad.append(f"#{i} r={e[5]}")
        if not e[7]:
            shape_bad.append(f"#{i} text 空")
    check("每条事件：seq 递增 / rel∈0..3（result/fanji 为全场 -1）/ r≥1 / hc 四元 / text 非空",
          not shape_bad, "; ".join(shape_bad[:3]))
    rounds = [e[5] for e in ev]
    check("巡号单调不降",
          bool(rounds) and all(rounds[i] <= rounds[i + 1] for i in range(len(rounds) - 1)),
          f"{rounds[0] if rounds else '-'} → {rounds[-1] if rounds else '-'}")

    # ---- 4. 重建一致性 ----
    print("\n=== 4. 重放重建一致性 ===")
    rivers = [[], [], [], []]
    melds = [[], [], [], []]
    hc = [13, 13, 13, 13]
    if isinstance(rp.get("dealer"), int):
        hc[rp["dealer"]] = 14
    n_discard = 0
    for e in ev:
        kind, rel, tile, txt = e[1], e[2], e[3], e[7]
        if kind == "discard" and tile >= 0:
            rivers[rel].append(tile)
            n_discard += 1
        elif kind == "peng" and tile >= 0:
            melds[rel].append(("peng", tile))
        elif kind == "kong" and tile >= 0:
            # 补杠（独山叫「补杠」）：把已有的同牌「碰」升级为 4 张，不新增一组。
            # ⚠️ 不能用文案判——事件文案是「随机 补杠一万（得通行证）」，
            # 早先的 `"补杠" in txt` 永远不成立，会把一次补杠算成「碰 + 杠」两组，
            # 使「手牌 + 副露」张数对不上（2026-10-12 由本脚本实测暴露）。
            # 结构性判据：同牌 4 张，碰用 3 张，第 4 张到手只可能是补杠。
            merged = False
            for j in range(len(melds[rel]) - 1, -1, -1):
                if melds[rel][j] == ("peng", tile):
                    melds[rel][j] = ("kong", tile)
                    merged = True
                    break
            if not merged:
                melds[rel].append(("kong", tile))
        hc = list(e[6])
    check("重放出的牌河总数 == 打牌事件数",
          sum(len(r) for r in rivers) == n_discard, f"{sum(len(r) for r in rivers)}/{n_discard}")
    # 「手牌 + 副露（杠算 4 张）」= 13 / 14，**只有「点杠（明杠）」会 +1**：
    #   · 点杠（明杠，别人打出第 4 张）：手上 3 张出去 + 补摸 1 张 → 总张数 +1
    #   · 暗杠（暗杠，手上 4 张全出）→ 总张数不变
    #   · 补杠（补杠，从已有的碰升级）→ 总张数不变
    # 早先按「每个杠都 +1」算是错的（下界会虚高，把只有暗杠/补杠的家误判为 FAIL）。
    tot = [hc[p] + sum(4 if m[0] == "kong" else 3 for m in melds[p]) for p in range(4)]
    nk = [sum(1 for m in melds[p] if m[0] == "kong") for p in range(4)]
    win_kong = [sum(1 for e in ev if e[1] == "kong" and e[2] == p
                    and "点杠" in (e[7] or "")) for p in range(4)]
    check("各家「手牌 + 副露」张数与点杠数吻合",
          all(13 + win_kong[p] <= tot[p] <= 14 + win_kong[p] for p in range(4)),
          f"张数={tot} 点杠={win_kong} 全部杠={nk}")
    # 与记录里的终局信息交叉核对：winner_hand 是「暗手（已并入胡牌张）」、不含副露，
    # 所以应恰好等于重放最后一帧里赢家的手牌张数 —— 两条独立路径（记录 / 事件流）互证。
    if rec.get("type") == "win" and rec.get("winner") is not None:
        w = int(rec["winner"])
        wh = rec.get("winner_hand") or []
        mw = sum(4 if m[0] == "kong" else 3 for m in melds[w])
        print(f"    赢家 {w}：记录暗手 {len(wh)} 张；重放末帧手牌 {hc[w]} 张 + 副露 {mw} 张")
        check("赢家暗手张数与重放末帧一致", len(wh) == hc[w], f"{len(wh)} vs {hc[w]}")
        check("赢家「暗手 + 副露」≥ 14（含胡牌张）", len(wh) + mw >= 14, f"{len(wh)} + {mw}")
    # 每家手牌张数不得为负、不得大于 14
    check("各家手牌张数合理（0..14）", all(0 <= x <= 14 for x in hc), str(hc))

    print("\n" + ("全部通过" if not bad else f"失败 {len(bad)} 项：" + "、".join(bad)))
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
