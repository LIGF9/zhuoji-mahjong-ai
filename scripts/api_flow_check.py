"""真实 HTTP 链路 e2e：像网页那样用「大师推荐」推进对局，任何动作被拒即为失败。

覆盖历史故障：碰牌后成胡（drawn=False）时 /api/hint 曾返回非法的 hu，
前端自动提交 → 服务端回「非法动作 hu:-1」→ 对局卡死。

用法：python scripts/api_flow_check.py [局数] [端口]
"""
from __future__ import annotations

import json
import sys
import time
import urllib.request

PORT = int(sys.argv[2]) if len(sys.argv) > 2 else 8770
HANDS = int(sys.argv[1]) if len(sys.argv) > 1 else 10
BASE = f"http://127.0.0.1:{PORT}"


def call(path: str, body: dict | None = None, timeout: float = 60.0):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(
        BASE + path, data=data,
        headers={"Content-Type": "application/json"} if data else {})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def main() -> int:
    r = call("/api/new", {
        "opponents": [{"type": "model", "model": "master", "temperature": 0},
                      {"type": "teacher", "style": "tenpai_rush"},
                      {"type": "model", "model": "dushan_s7", "temperature": 0}],
        "my_seat": 0, "speed": 0, "auto": False,
        "names": ["你", "大乔(抢听)", "妲己(B3)", "王昭君(S7)"],
    })
    if not r.get("ok"):
        print("开局失败", r)
        return 1
    sid = r["sid"]
    print(f"会话 {sid} 开局成功（目标 {HANDS} 局）")

    rejects: list[tuple] = []
    acts = 0
    hands_done = 0
    hints = 0
    illegal_hint = 0
    deadline = time.time() + 900
    while hands_done < HANDS and time.time() < deadline:
        st = call(f"/api/state?sid={sid}", timeout=30)
        if st.get("error"):
            print("服务端异常：", st["error"][:400])
            break
        pend = st.get("pending")
        if st.get("over"):
            if st.get("hand_index") and st["hand_index"] > hands_done:
                hands_done = st["hand_index"]
                print(f"  第 {hands_done} 局结束：{st['result']['type']} "
                      f"我的分 {st['result']['mine']:+.1f}")
            call("/api/next", {"sid": sid})
            time.sleep(0.2)
            continue
        if not pend or st.get("turn") != st.get("my_seat"):
            time.sleep(0.15)
            continue
        # 我的决策点：取推荐（与网页一致），并校验推荐确实属于当前决策点
        h = call(f"/api/hint?sid={sid}&strategy=model:master", timeout=60)
        hints += 1
        acts_ok = {(b["kind"], int(b["tile"])) for b in (pend.get("buttons") or [])}
        legal = {("discard", int(t)) for t in (pend.get("discardables") or [])} | acts_ok
        act = None
        if h.get("ok") and h.get("items"):
            a = h["items"][0]
            key = (a["kind"], int(a["tile"]))
            if key in legal:
                act = (a["kind"], int(a["tile"]))
            else:
                illegal_hint += 1
                print(f"  ⚠ 推荐不属于当前决策点：{key} 合法集={sorted(legal)[:6]}")
        if act is None:                      # 兜底（与前端一致）
            act = (("discard", pend["discardables"][0]) if pend.get("mode") == "discard"
                   and pend.get("discardables") else ("pass", -1))
        rr = call("/api/act", {"sid": sid, "kind": act[0], "tile": act[1]})
        acts += 1
        if not rr.get("ok"):
            rejects.append((act, rr.get("error"), st.get("phase"), st.get("hand_index")))
            print(f"  ✗ 动作被拒 {act}: {rr.get('error')}")
            time.sleep(0.3)
    try:
        call("/api/quit", {"sid": sid})
    except Exception:
        pass

    print(f"\n完成 {hands_done} 局 · 提交动作 {acts} 次 · 推荐 {hints} 次 · "
          f"越界推荐 {illegal_hint} · 被拒 {len(rejects)}")
    if rejects:
        for w in rejects[:8]:
            print("  ", w)
        return 1
    print("全部动作被服务端接受 ✅（无「非法动作」卡死）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
