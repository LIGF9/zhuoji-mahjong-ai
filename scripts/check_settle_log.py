"""结算流水行口径校验（2026-10-12）。

复现用户实机反馈的两个「全局流水结算不对」：
  ① 荒牌（黄庄）大家都听牌，流水却写成「我（你）包了大家的牌」；
  ② 上家自摸，结算页 +6 正确，流水却写 +9。

根因与修法（见 web/dushan_server.py::Session._emit_result 的注释）：
  · 胡牌行原用 res['fan']（首位赢家的**总进账**）当分数 → 自摸局多算了
    赢家付给他人的部分；现在取 deltas[seat] 净分。
  · 荒牌行原硬编码 rel=0，前端加「你」前缀又吃掉首词 → 读成「我包了大家」；
    现在结算行统一 rel=-1，前端按「结算」行原样渲染。

断言：
  A. 所有 kind=result 事件 rel == -1（前端才会走「结算」分支，不加席位前缀）；
  B. 胡牌行里写出的分数 == 引擎 deltas[winner] 净分；
  C. 荒牌行不出现「包大牌面」当且仅当无人未听牌（四家听牌/四家未听都不得包赔）；
  D. 荒牌行里的未听家分数 == deltas[seat]。

运行：
  C:/Users/LIGF/.workbuddy/binaries/python/envs/zhuoji/Scripts/python.exe \
    scripts/check_settle_log.py [局数]
"""
from __future__ import annotations

import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from web.dushan_server import Session  # noqa: E402

N_HANDS = int(sys.argv[1]) if len(sys.argv) > 1 else 60
SPECS = [{"type": "teacher", "style": "balanced"},
         {"type": "teacher", "style": "aggressive"},
         {"type": "teacher", "style": "defensive"}]

bad: list[str] = []
stats = {"win": 0, "huang": 0, "tenpai_huang": 0, "all_tenpai_huang": 0}
seen: list[dict] = []

s = Session("settlelog", SPECS, my_seat=0, seed=20261012, speed=0.0, auto=True)


def on_end(sess, game):
    res = dict(game.result or {})
    res["_hand"] = sess.hand_index
    res["_ev"] = [dict(e) for e in sess.events if e.get("hand") == sess.hand_index
                  and e.get("kind") == "result"]
    seen.append(res)


s.on_hand_end = on_end
t0 = time.time()
while len(seen) < N_HANDS and not s.error and time.time() - t0 < 300:
    time.sleep(0.05)
s.stop = True
if s.error:
    print(s.error)
    sys.exit(1)

score_re = re.compile(r"([+-]\d+(?:\.\d+)?)")
tail_score_re = re.compile(r"([+-]\d+(?:\.\d+)?) 分\s*$")


def tail_score(text: str):
    m = tail_score_re.search(text or "")
    return float(m.group(1)) if m else None


def noten_scores(text: str) -> list[float]:
    seg = (text or "").split("· 未听 ", 1)
    if len(seg) < 2:
        return []
    return [float(x) for x in score_re.findall(seg[1])]


for r in seen:
    hand = r.get("_hand")
    evs = r.get("_ev") or []
    if not evs:
        bad.append(f"第{hand}局 没有 result 流水行")
        continue
    txt = evs[-1].get("text") or ""
    if any(e.get("rel") != -1 for e in evs):
        bad.append(f"第{hand}局 result 行 rel != -1：{[e.get('rel') for e in evs]}")
    d = [float(x) for x in (r.get("deltas") or [0.0] * 4)]

    if r.get("type") == "win":
        stats["win"] += 1
        w = int(r["winner"])
        got = tail_score(txt)
        if got is None:
            bad.append(f"第{hand}局 胡牌行没写出分数：{txt}")
        elif abs(got - d[w]) > 1e-6:
            bad.append(f"第{hand}局 胡牌行分数 {got} != 净分 deltas[{w}]={d[w]}：{txt}")
        if "包" in txt:
            bad.append(f"第{hand}局 胡牌行误含「包」字：{txt}")
    else:
        stats["huang"] += 1
        tenpai = [int(x) for x in (r.get("tenpai") or [])]
        noten = [q for q in range(4) if q not in tenpai]
        if 0 < len(tenpai) < 4:
            stats["tenpai_huang"] += 1
        if len(tenpai) == 4:
            stats["all_tenpai_huang"] += 1
        # 只有「一部分听、一部分未听」才发生包大牌面
        expect_bao = bool(tenpai) and bool(noten)
        if ("包大牌面" in txt) != expect_bao:
            bad.append(f"第{hand}局 荒牌行包赔口径错（听 {tenpai} / 未听 {noten}）：{txt}")
        if not expect_bao:
            if any(abs(x) > 1e-6 for x in d):
                bad.append(f"第{hand}局 无人/全员未听却非零和局分：{d}")
            if "未听 " in txt:
                bad.append(f"第{hand}局 无未听赔付却写了未听家分数：{txt}")
        else:
            nets = noten_scores(txt)
            want = [round(d[q], 2) for q in noten]
            if sorted(nets) != sorted(want):
                bad.append(f"第{hand}局 荒牌未听家分数 {nets} != {want}：{txt}")

print(f"[settle-log] 抽样 {len(seen)} 局：胡 {stats['win']} / 黄庄 {stats['huang']}"
      f"（其中未听黄庄 {stats['tenpai_huang']}、四家听牌 {stats['all_tenpai_huang']}）")

# ---- 合成用例：用户实机报的那两个具体局面（真机里很罕见，抽样打不出来）----
synth_bad: list[str] = []


class _Fake:
    def __init__(self, res):
        self.result = res
        self.cfg = None


def emit(res):
    s2 = Session.__new__(Session)
    s2.names = ["你", "上家", "对家", "下家"]
    s2.my_seat = 0
    s2.hand_index = 1
    s2._ev_seq = 0
    s2.events = []
    s2.lock = __import__("threading").RLock()
    s2._hand_ev = []
    s2._cur_game = None
    s2._ev_draws = 0
    s2._ev_discards = 0
    Session._emit_result(s2, _Fake(res))
    return s2.events[-1]


# ① 上家自摸：结算页 +6（deltas[1]=6），但总进账 fan=9 —— 流水必须写 +6
e = emit({"type": "win", "winner": 1, "winners": [1], "is_tsumo": True, "loser": None,
          "fan_cn": "平胡", "fan": 9.0, "how": "自摸",
          "deltas": [-6.0, 6.0, -6.0, 6.0]})
if e["rel"] != -1:
    synth_bad.append(f"自摸行 rel={e['rel']}（应 -1）")
if tail_score(e["text"]) != 6.0:
    synth_bad.append(f"自摸行写的是 {tail_score(e['text'])} 分，应为净分 +6：{e['text']}")
if "9" in e["text"]:
    synth_bad.append(f"自摸行仍出现总进账 9：{e['text']}")
print("   合成①:", e["text"])

# ② 荒牌四家听牌：谁都不赔，流水不得出现「包」
e = emit({"type": "huangzhuang", "tenpai": [0, 1, 2, 3], "baodapai": [
    {"seat": 0, "type": "平胡", "pay": 3}, {"seat": 1, "type": "平胡", "pay": 3},
    {"seat": 2, "type": "平胡", "pay": 3}, {"seat": 3, "type": "平胡", "pay": 3}],
    "baodapai_on": True, "deltas": [0.0, 0.0, 0.0, 0.0]})
if e["rel"] != -1:
    synth_bad.append(f"四家听牌荒牌行 rel={e['rel']}（应 -1）")
if "包" in e["text"] or "未听" in e["text"]:
    synth_bad.append(f"四家听牌荒牌行不该出现包赔/未听：{e['text']}")
print("   合成②:", e["text"])

bad += synth_bad


for r in seen[-3:]:
    for e in (r.get("_ev") or []):
        print("   例:", e.get("text"))
if bad:
    print(f"[settle-log] FAIL  {len(bad)} 项")
    for b in bad[:20]:
        print("   -", b)
    sys.exit(1)
print("[settle-log] PASS — 结算行 rel=-1；胡牌分数 = 净分；荒牌包赔口径正确")
