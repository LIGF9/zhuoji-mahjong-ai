"""验证 2026-10-12 修复：会话在牌局进行中被停止（/api/quit）时**不再落盘空白记录**。

背景：`Session._play_hand` 的 while 循环被 `self.stop` 打断时 `game.result` 仍是空，
旧版 `_settle` 照常走 `_make_record` → records/replays 各多一条 type=None 的脏记录
（实测 14:01 的 api_huang_check 就造了 3 条）。修后应早退、不落盘。

用法：python scripts/verify_settle_guard.py    （需要网页服务在线）
"""
import json
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BASE = "http://127.0.0.1:8770"


def call(path, body=None, timeout=30):
    data = None if body is None else json.dumps(body).encode("utf-8")
    req = urllib.request.Request(BASE + path, data=data,
                                 headers={"Content-Type": "application/json"} if data else {})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


def lines(p):
    f = ROOT / "web" / "history" / p
    return len([l for l in f.read_text(encoding="utf-8").splitlines() if l.strip()]) if f.exists() else 0


def main() -> int:
    r_before, p_before = lines("records.jsonl"), lines("replays.jsonl")
    print(f"起始：records={r_before}  replays={p_before}")

    new = call("/api/new", {
        "opponents": [{"type": "random"}, {"type": "random"}, {"type": "random"}],
        "my_seat": 0, "speed": 0,
    })
    sid = new.get("sid")
    assert sid, f"未拿到 sid: {new}"
    print(f"开新会话 sid={sid}")

    # 提交若干手（每碰到人类决策点就打一张），确保牌局确实在推进中。
    # 注意：打牌阶段 pending.buttons 是**空**的（可打牌在 discardables 里，
    # 因为「打某张」由任意一张手牌生成），所以两种都要处理。
    acted = 0
    deadline = time.time() + 60
    while acted < 5 and time.time() < deadline:
        st = call(f"/api/state?sid={sid}")
        if st.get("over"):
            print("这一局已意外结束，重来")
            return 1
        pend = st.get("pending")
        if not pend or st.get("turn") != 0:
            time.sleep(0.05)
            continue
        btns = pend.get("buttons") or []
        kind = tile = None
        if btns:
            kind, tile = btns[0].get("kind"), btns[0].get("tile")
        elif pend.get("discardables"):
            kind, tile = "discard", pend["discardables"][0]
        if kind is None:
            time.sleep(0.05)
            continue
        call("/api/act", {"sid": sid, "kind": kind, "tile": tile})
        acted += 1
    print(f"牌局推进中，已提交 {acted} 手（此时 phase 未结束）")
    assert acted >= 3, f"只推进了 {acted} 手，不足以覆盖「中途停止」路径"

    # 关键动作：牌局**没打完**就停止会话
    call("/api/quit", {"sid": sid})
    time.sleep(1.5)   # 给服务端线程一点收尾时间（旧版就在这段时间落盘）

    r_after, p_after = lines("records.jsonl"), lines("replays.jsonl")
    print(f"停止后：records={r_after}  replays={p_after}")
    ok = (r_after == r_before) and (p_after == p_before)
    if ok:
        print("PASS：未打完就停止 → 没有落盘空白记录 ✅")
        return 0
    print(f"FAIL：脏记录又出现了（records +{r_after - r_before}, replays +{p_after - p_before}）")
    # 打印新增行帮助定位
    recs = (ROOT / "web/history/records.jsonl").read_text(encoding="utf-8").splitlines()
    for ln in recs[r_before:]:
        r = json.loads(ln)
        print("   新增:", {k: r.get(k) for k in ("hand", "type", "rid")})
    return 1


if __name__ == "__main__":
    sys.exit(main())
