"""「下一局生效」规则下发链路契约测试。

Bug（2026-10-12 用户反馈）
------------------------
「我在设置中增加的鸡牌并没有在下一局默认生效。」

根因
----
前端只有 ``/api/new`` 带 ``rules``；点「下一局」走的 ``/api/next`` **只传 sid**，
服务端 ``Session.rules`` 冻在开局那一刻 → 同一 session 的后续每一手
（``_play_hand`` 里 ``config_from_rules(self.rules)``）都用旧规则。
受影响的**不止鸡牌**：翻鸡策略（开局/结算）、满堂鸡、未听包赔、整张分值表
在前端都标着「下一局生效」，实际全部要重开一局才认。

修复
----
* 前端抽 ``collectRules()``，``/api/new`` 与 ``/api/next`` 共用；
* 服务端 ``Session.next_hand(rules)`` 在唤醒下一手**之前**替换 ``self.rules``；
* ``/api/next`` 把 body 里的 ``rules`` 透传下去。

本脚本锁住这条链路（4 组断言）：
  ① rules 载荷 → DushanConfig：extra_ji 清洗（去重/越界丢弃/忽略幺鸡）
  ② DushanConfig.extra_ji → DushanGame.ji_tiles：追加牌真的算鸡
  ③ Session 契约：next_hand(rules) 更新 self.rules；未 over 时不动
  ④ HTTP：/api/next 把 body.rules 交给 session.next_hand（不传则为 None，向后兼容）

运行::

    python tests/test_rules_next.py
"""
from __future__ import annotations

import importlib.util
import json
import sys
import threading
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "web"))

# web/ 不是包（没有 __init__.py），按路径加载模块
_spec = importlib.util.spec_from_file_location("dushan_server", ROOT / "web" / "dushan_server.py")
assert _spec and _spec.loader
S = importlib.util.module_from_spec(_spec)
sys.modules["dushan_server"] = S
_spec.loader.exec_module(S)

from zhuoji.dushan import DushanConfig, DushanGame, YAOJI  # noqa: E402

PASS = FAIL = 0


def check(name: str, cond: bool, extra: str = "") -> None:
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  PASS  {name}{'  ' + extra if extra else ''}")
    else:
        FAIL += 1
        print(f"  FAIL  {name}{'  ' + extra if extra else ''}")


# ---------------------------------------------------------------------------
# ① rules 载荷 → DushanConfig
# ---------------------------------------------------------------------------
print("\n=== ① rules.extra_ji → DushanConfig ===")
cfg = S.config_from_rules({"extra_ji": [8, 15, 8, -1, 99, "7", None, YAOJI]})
check("extra_ji 清洗：去重 + 忽略越界 + 忽略幺鸡",
      cfg.extra_ji == (7, 8, 15), f"extra_ji={cfg.extra_ji}")
check("extra_ji 存的是 tuple（dataclass 字段）", isinstance(cfg.extra_ji, tuple))
check("缺省（前端没传）→ 空元组", S.config_from_rules({}).extra_ji == ())
check("extra_ji 非列表 → 空元组", S.config_from_rules({"extra_ji": "8筒"}).extra_ji == ())
check("翻鸡策略随 rules 一起改", 
      (S.config_from_rules({"kaiju": "both"}).kaiju_fanji
       != S.config_from_rules({"kaiju": "off"}).kaiju_fanji))

# ---------------------------------------------------------------------------
# ② DushanConfig.extra_ji → DushanGame.ji_tiles
# ---------------------------------------------------------------------------
print("\n=== ② 追加牌真的进了本局鸡牌集合 ===")
g_after = DushanGame(S.config_from_rules({"extra_ji": [8, 15]}), seed=1)
g_before = DushanGame(S.config_from_rules({}), seed=1)
check("追加牌并入 ji_tiles", {8, 15} <= g_after.ji_tiles, f"ji_tiles={sorted(g_after.ji_tiles)}")
check("没追加时只有幺鸡（+开局翻鸡种）", g_before.ji_tiles == {YAOJI},
      f"ji_tiles={sorted(g_before.ji_tiles)}")
check("同一 seed 下两局鸡牌集合确实不同", g_after.ji_tiles != g_before.ji_tiles)

# ---------------------------------------------------------------------------
# ③ Session 契约
# ---------------------------------------------------------------------------
print("\n=== ③ Session.next_hand(rules) ===")
sess = S.Session("t1", [{"type": "random"}] * 3, my_seat=0, seed=7, speed=0.0,
                 auto=True, rules={"extra_ji": [8]})
check("开局 rules 已装载", sess.rules == {"extra_ji": [8]}, str(sess.rules))
check("未结束时不放行", sess.next_hand({"extra_ji": [15]})["ok"] is False)
check("未结束时不改规则（当前手不受影响）", sess.rules == {"extra_ji": [8]}, str(sess.rules))

sess.state = {"ready": True, "over": True}
res = sess.next_hand({"extra_ji": [15], "kaiju": "both"})
check("结束后放行", res.get("ok") is True)
check("rules 被整体替换（下一局用新设置）",
      sess.rules == {"extra_ji": [15], "kaiju": "both"}, str(sess.rules))
check("唤醒信号已发出", sess.next_event.is_set())
check("替换后的 rules 能译成新配置",
      S.config_from_rules(sess.rules).extra_ji == (15,) and
      S.config_from_rules(sess.rules).kaiju_fanji == "both")

sess2 = S.Session("t2", [{"type": "random"}] * 3, my_seat=0, seed=7, speed=0.0,
                  auto=True, rules={"extra_ji": [8]})
sess2.state = {"ready": True, "over": True}
sess2.next_hand()                       # 不带 rules（旧前端 / 其它调用点）
check("不传 rules → 保留原规则（向后兼容）", sess2.rules == {"extra_ji": [8]}, str(sess2.rules))

# ---------------------------------------------------------------------------
# ④ HTTP 层透传
# ---------------------------------------------------------------------------
print("\n=== ④ /api/next 透传 body.rules ===")
captured: dict = {}


class _FakeSession:
    def next_hand(self, rules=None):
        captured["rules"] = rules
        return {"ok": True}


_orig_get = S.get_session
S.get_session = lambda sid: _FakeSession()
srv = ThreadingHTTPServer(("127.0.0.1", 8791), S.Handler)
threading.Thread(target=srv.serve_forever, daemon=True).start()


def post(path: str, payload: dict) -> dict:
    req = urllib.request.Request(
        "http://127.0.0.1:8791" + path,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=10) as r:
        return json.loads(r.read().decode())


try:
    r = post("/api/next", {"sid": "x", "rules": {"extra_ji": [15]}})
    check("HTTP 200 + ok", r.get("ok") is True, str(r))
    check("body.rules 原样交给 next_hand", captured.get("rules") == {"extra_ji": [15]},
          str(captured.get("rules")))
    captured.clear()
    post("/api/next", {"sid": "x"})
    check("不带 rules 时传 None（旧前端兼容）", captured.get("rules") is None,
          str(captured.get("rules")))
    # 路由名/方法没写错：GET 同一个路径应当 404（/api/next 是 POST 专属）
    try:
        urllib.request.urlopen("http://127.0.0.1:8791/api/next", timeout=5)
        check("/api/next 仅 POST（GET 应 404）", False, "GET 竟然成功了")
    except urllib.error.HTTPError as e:
        check("/api/next 仅 POST（GET 应 404）", e.code == 404, f"code={e.code}")
finally:
    srv.shutdown()
    S.get_session = _orig_get

print(f"\n合计：{PASS} PASS / {FAIL} FAIL")
sys.exit(1 if FAIL else 0)
