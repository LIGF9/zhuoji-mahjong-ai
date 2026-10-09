"""真实 HTTP 链路 e2e：荒牌流局新口径结算校验。

规则（2026-10 用户确认）：
- 流局不再结算鸡与杠（手中鸡/打出鸡/翻鸡/杠分/责任鸡一律不计分，chickens 全 0）；
- 只在「未听牌者 → 听牌者」之间结算：每个未听者向每个听牌者赔
  「自摸分 + 该听牌者可达最大牌型分」（平胡时大牌面为 0 → 正好一个自摸分）；
- 听牌者之间互不结算。

用法：python scripts/api_huang_check.py [端口] [目标黄庄局数]
"""
from __future__ import annotations

import json
import sys
import time
import urllib.request

PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 8770
TARGET = int(sys.argv[2]) if len(sys.argv) > 2 else 3
BASE = f"http://127.0.0.1:{PORT}"

ALLOWED_LABELS = {"包大牌面", "包鸡", "包杠"}   # 流局 detail 里允许出现的行
BAD_LABELS = {"手中鸡", "杠分", "责任鸡", "翻鸡", "冲锋鸡", "横鸡", "幺鸡", "自摸", "点炮"}


def call(path: str, body: dict | None = None, timeout: float = 60.0):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(
        BASE + path, data=data,
        headers={"Content-Type": "application/json"} if data else {})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def verify(res: dict, st: dict, bad: list[str], tag: str) -> None:
    d = [float(x) for x in (res.get("deltas") or [])]
    if abs(sum(d)) > 1e-6:
        bad.append(f"{tag} 非零和 Σ={sum(d)}")
    if res.get("chickens") != [0.0, 0.0, 0.0, 0.0]:
        bad.append(f"{tag} chickens={res.get('chickens')}（流局应全 0）")
    labels = {str(r.get("label") if isinstance(r, dict) else r[1])
              for r in (res.get("detail") or [])}
    hit = labels & BAD_LABELS
    if hit:
        bad.append(f"{tag} 流局出现了不该结算的行：{hit}")
    if not labels <= ALLOWED_LABELS:
        bad.append(f"{tag} detail 含未知行：{labels - ALLOWED_LABELS}")

    tenpai = list(res.get("tenpai") or [])
    n_payers = 4 - len(tenpai)
    pair = res.get("pair") or [[0.0] * 4 for _ in range(4)]
    bds = {int(b["seat"]): b for b in (res.get("baodapai") or [])}
    if set(bds) != set(tenpai):
        bad.append(f"{tag} baodapai 座位 {sorted(bds)} ≠ tenpai {tenpai}")
    zm = float(res.get("zi_mo") or 0.0)
    for p, b in bds.items():
        if abs(float(b["pay"]) - (zm + float(b["type_value"]))) > 1e-9:
            bad.append(f"{tag} p{p} pay={b['pay']} ≠ 自摸{zm}+牌型{b['type_value']}")
        recv = sum(float(x) for x in pair[p])
        if abs(recv - float(b["pay"]) * n_payers) > 1e-6:
            bad.append(f"{tag} p{p} 收进 {recv} ≠ pay×未听数 {b['pay']}×{n_payers}")
    # 听牌者之间互不结算
    for a in tenpai:
        for b_ in tenpai:
            if a != b_ and abs(float(pair[a][b_])) > 1e-9:
                bad.append(f"{tag} 听牌者 {a}->{b_} 出现互算 {pair[a][b_]}")
    # delta 与两两矩阵一致
    delta2 = [sum(float(pair[a][b_]) - float(pair[b_][a]) for b_ in range(4))
              for a in range(4)]
    if any(abs(d[i] - delta2[i]) > 1e-6 for i in range(4)):
        bad.append(f"{tag} deltas {d} 与 pair 推导 {delta2} 不一致")
    tp_names = res.get("tenpai_names") or []
    print(f"  {tag} 黄庄：听牌 {tp_names} · "
          + "；".join(f"{b['name']} {b['type']} 每家赔 {b['pay']:g}"
                      for b in (res.get("baodapai") or []))
          + f" · deltas {[round(x, 1) for x in d]}")


def main() -> int:
    r = call("/api/new", {
        "opponents": [{"type": "model", "model": "master", "temperature": 0},
                      {"type": "teacher", "style": "tenpai_rush"},
                      {"type": "model", "model": "dushan_s7", "temperature": 0}],
        "my_seat": 0, "speed": 0, "auto": False,   # 我的座位由本脚本按推荐推进
        "names": ["你", "大乔(抢听)", "妲己(B3)", "王昭君(S7)"],
    })
    if not r.get("ok"):
        print("开局失败", r)
        return 1
    sid = r["sid"]
    print(f"会话 {sid}（纯机器人视角，目标 {TARGET} 个黄庄局）")
    bad: list[str] = []
    checked = 0
    hands = 0
    deadline = time.time() + 600
    while checked < TARGET and time.time() < deadline:
        st = call(f"/api/state?sid={sid}", timeout=30)
        if st.get("error"):
            bad.append("服务端异常：" + str(st["error"])[:300])
            break
        if st.get("over"):
            hands += 1
            res = st["result"] or {}
            if res.get("type") == "huangzhuang":
                checked += 1
                verify(res, st, bad, f"第{hands}局")
            else:
                print(f"  第{hands}局 {res.get('type')}（跳过）")
            call("/api/next", {"sid": sid})
            time.sleep(0.15)
            continue
        # 我的决策点：按推荐推进（与前端的 autoAct 兜底一致）
        pend = st.get("pending")
        if pend and st.get("turn") == st.get("my_seat"):
            h = call(f"/api/hint?sid={sid}&strategy=model:master", timeout=60)
            legal = {("discard", int(t)) for t in (pend.get("discardables") or [])} | \
                    {(b["kind"], int(b["tile"])) for b in (pend.get("buttons") or [])}
            act = None
            if h.get("ok") and h.get("items"):
                a = h["items"][0]
                if (a["kind"], int(a["tile"])) in legal:
                    act = (a["kind"], int(a["tile"]))
            if act is None:
                act = (("discard", pend["discardables"][0]) if pend.get("mode") == "discard"
                       and pend.get("discardables") else ("pass", -1))
            call("/api/act", {"sid": sid, "kind": act[0], "tile": act[1]})
            continue
        time.sleep(0.05)
    try:
        call("/api/quit", {"sid": sid})
    except Exception:
        pass

    print(f"\n完成 {hands} 局，其中黄庄 {checked} 局")
    if checked < TARGET:
        bad.append(f"仅 {checked}/{TARGET} 个黄庄样本（超时）")
    if bad:
        print("FAIL\n  " + "\n  ".join(bad[:12]))
        return 1
    print("流局新口径全部校验通过 ✅（鸡与杠不结算；未听者按 自摸+大牌面 包听牌者；听牌者互不结算）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
