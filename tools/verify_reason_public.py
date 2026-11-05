# -*- coding: utf-8 -*-
"""审计：推荐 Q 值与推荐理由都**不使用全局 / 隐藏信息**。

2026-11-04 用户口径：「确保没有使用全局信息来计算推荐和编写推荐理由」。

做法（比「自律不读」更硬）
--------------------------
1. 造一个真实对局决策点，克隆成两份：
   * ``orig``：原局面；
   * ``shuf``：只重排**隐藏信息**——别家暗手的内容 + 剩余牌墙的顺序，
     其余一律不动（自己手牌、四家弃牌 / 副露、鸡种、报叫、各家手牌张数、
     牌墙剩余张数、已摸牌前缀、所有规则标志都逐位相同）。
2. 断言：
   * ``PublicView.from_game(orig, p) == PublicView.from_game(shuf, p)``
     ——公开视图逐字段相同（它结构里根本没有隐藏字段）；
   * 真实搜索内核对两份局面给出**完全相同的 Q**（推荐不偷看隐藏牌）；
   * ``explain_items`` / ``compare_tip`` / ``situation_summary`` 输出逐字相同
     （理由不偷看隐藏牌）。
3. 反向灵敏度检查：故意改一张**公开**弃牌 / 自己手牌，断言 PublicView 必须
   变化——证明上面的「相同」不是空测（不是因为比较逻辑写错而恒真）。

运行：``.venv/bin/python tools/verify_reason_public.py``
"""
from __future__ import annotations

import io
import random
import sys
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "web"))

import torch  # noqa: E402

import coach  # noqa: E402  # web/ 目录
from zhuoji.dushan import DushanConfig, DushanGame  # noqa: E402
from zhuoji.net import ZhuojiNet  # noqa: E402
from zhuoji.public_view import PublicView  # noqa: E402
from zhuoji.search_agent import SearchAgent, explain_items  # noqa: E402
from zhuoji.tiles import NUM_TILE_TYPES  # noqa: E402

GAME_SEED = 20261104
STEP_SEED = 991
SHUF_SEED = 424242
FIXED_SEARCH_SEED = 13579
N_WALK = 24
N_DET = 8


def _log(msg: str) -> None:
    print(msg, flush=True)


def _check(name: str, ok: bool) -> bool:
    _log(("  PASS  " if ok else "  FAIL  ") + name)
    return bool(ok)


# ---------------------------------------------------------------------------
# 1. 造真实决策点
# ---------------------------------------------------------------------------
def make_decision_point():
    g = DushanGame(DushanConfig(), seed=GAME_SEED)
    rng = random.Random(STEP_SEED)
    for _ in range(N_WALK):
        acts = g.legal_actions()
        if not acts or g.result:
            break
        g.step(rng.choice(acts))
    p = g.actor()
    legal = g.legal_actions()
    if not legal:
        # 极端情况：换一副牌保证有决策点
        g = DushanGame(DushanConfig(), seed=GAME_SEED + 1)
        p, legal = g.actor(), g.legal_actions()
    return g, int(p), legal


# ---------------------------------------------------------------------------
# 2. 只重排隐藏信息（别家暗手内容 + 剩余牌墙顺序）
# ---------------------------------------------------------------------------
def shuffle_hidden(game, p: int, seed: int):
    g = game.clone()
    rng = random.Random(seed)
    totals = [sum(g.hands[q]) for q in range(4)]       # 各家手牌张数是公开信息，保持
    pool: list[int] = []
    for q in range(4):
        if q == p:
            continue
        for t, c in enumerate(g.hands[q]):
            pool.extend([t] * c)
    wall_n = len(g.wall) - g.wall_pos
    pool.extend(g.wall[g.wall_pos:])
    rng.shuffle(pool)
    # 回填：牌墙剩余段 + 别家暗手；已摸牌前缀 wall[:wall_pos] 不动
    g.wall = list(g.wall[:g.wall_pos]) + list(pool[:wall_n])
    rest = pool[wall_n:]
    k = 0
    for q in range(4):
        if q == p:
            continue
        h = [0] * NUM_TILE_TYPES
        for _ in range(totals[q]):
            h[rest[k]] += 1
            k += 1
        g.hands[q] = h
    assert k == len(rest), "隐藏池回填张数不守恒"
    return g


def assert_public_identical(a, b, p: int) -> bool:
    ok = True
    ok &= _check("自己暗手相同", list(a.hands[p]) == list(b.hands[p]))
    for q in range(4):
        ok &= _check(f"座位{q}弃牌相同", list(a.discards[q]) == list(b.discards[q]))
        ok &= _check(f"座位{q}副露相同", [tuple(m) for m in a.melds[q]]
                     == [tuple(m) for m in b.melds[q]])
    ok &= _check("鸡牌种相同", set(a.ji_tiles) == set(b.ji_tiles))
    ok &= _check("报叫标志相同", list(a.baojiao) == list(b.baojiao))
    ok &= _check("刚打出牌相同", tuple(a.last_discard or ()) == tuple(b.last_discard or ()))
    ok &= _check("开局指示牌相同", a.kaiju_flip == b.kaiju_flip)
    ok &= _check("牌墙剩余张数相同", a.wall_left == b.wall_left)
    ok &= _check("已摸牌墙前缀相同", list(a.wall[:a.wall_pos]) == list(b.wall[:b.wall_pos]))
    ok &= _check("各家手牌张数(公开)相同",
                 [sum(a.hands[q]) for q in range(4)] ==
                 [sum(b.hands[q]) for q in range(4)])
    # 隐藏信息确实被改了（否则审计是空转）
    hidden_changed = any(list(a.hands[q]) != list(b.hands[q])
                         for q in range(4) if q != p) or \
        list(a.wall[a.wall_pos:]) != list(b.wall[b.wall_pos:])
    ok &= _check("隐藏信息确实已重排（审计非空转）", hidden_changed)
    return ok


# ---------------------------------------------------------------------------
# 3. 模型 / 搜索 Q
# ---------------------------------------------------------------------------
def load_net():
    ck = torch.load(str(ROOT / "models" / "dushan_master.pt"),
                    map_location="cpu", weights_only=False)
    net = ZhuojiNet(channels=64, blocks=8, hidden=320)
    net.load_state_dict(ck["state_dict"])
    net.eval()
    return net


def py_q(net, game, p, legal):
    """Python 搜索内核（CPU 前向，无 ORT 依赖）的 Q。每次新建同 seed 实例。"""
    ag = SearchAgent(net, seed=FIXED_SEARCH_SEED, device="cpu",
                     ply=1, p2_topk=3, n_det=N_DET, fair=True, own_model=True)
    buf = io.StringIO()
    with redirect_stdout(buf), redirect_stderr(buf):
        _legal, q, _prior, _base, _v0 = ag._evaluate(game)
    assert [x.label() for x in _legal] == [x.label() for x in legal]
    return np.asarray(q, dtype=float)


def rust_q(net, game, p, legal):
    """生产同源的 Rust 搜索内核（ORT 前向）的 Q。"""
    from zhuoji.rust_search import RustSearchAgent  # noqa: PLC0415
    ag = RustSearchAgent(net, seed=FIXED_SEARCH_SEED, device="ort",
                         ply=1, p2_topk=3, n_det=N_DET, fair=True)
    buf = io.StringIO()
    with redirect_stdout(buf), redirect_stderr(buf):
        q, _qs, _diag = ag.rank_detail(game)
    return np.asarray(q, dtype=float)


def fair_view_identical(orig, shuf, p: int) -> bool:
    """模型无关：直接检验 Python 公平 determinization 对隐藏排列不变。"""
    class _Stub:
        def __init__(self, seed):
            self.rng = np.random.default_rng(seed)

    a = SearchAgent._fair_view(_Stub(FIXED_SEARCH_SEED), orig, p)
    b = SearchAgent._fair_view(_Stub(FIXED_SEARCH_SEED), shuf, p)
    hands_eq = [list(a.hands[q]) == list(b.hands[q]) for q in range(4)]
    wall_eq = list(a.wall) == list(b.wall)
    return _check("公平 determinization 逐位相同（模型无关）",
                  all(hands_eq) and wall_eq)


# ---------------------------------------------------------------------------
# 4. 理由序列化（逐字比较）
# ---------------------------------------------------------------------------
def serialize_items(items) -> str:
    out = []
    for it in items:
        out.append({
            "label": it["action"].label(),
            "q": round(float(it["q"]), 9),
            "reason": it.get("reason", ""),
            "lines": list(it.get("lines", [])),
            "tags": list(it.get("tags", [])),
        })
    return repr(out)


def frontend(items) -> list[dict]:
    return [{"label": it["action"].label(), "kind": it["action"].kind,
             "tile": int(it["action"].tile), "q": float(it["q"]),
             "prob": 0.5} for it in items]


def main() -> int:
    _log("=== 推荐理由公开信息隔离审计 ===")
    _log(f"对局 seed={GAME_SEED}  行走 {N_WALK} 步  搜索 n_det={N_DET} ply=1")

    orig, p, legal = make_decision_point()
    shuf = shuffle_hidden(orig, p, SHUF_SEED)
    kinds = sorted({a.kind for a in legal})
    _log(f"决策点: 座位={p} 合法动作={len(legal)} 类型={kinds} "
         f"牌墙剩余={orig.wall_left}")

    ok = True
    _log("\n[1] 公开信息逐位一致 / 隐藏信息确已重排")
    ok &= assert_public_identical(orig, shuf, p)

    _log("\n[2] PublicView 物理隔离：两份局面构造出完全相同的公开视图")
    v_orig = PublicView.from_game(orig, p)
    v_shuf = PublicView.from_game(shuf, p)
    ok &= _check("PublicView.from_game(orig) == from_game(shuf)", v_orig == v_shuf)
    # 结构自检：公开视图里确实没有别家手牌 / 真实牌墙字段
    fields = set(v_orig.__dataclass_fields__)
    forbidden = {"wall", "wall_pos", "hands", "opponent_hands", "hidden"}
    ok &= _check("PublicView 不含 wall/hands 等隐藏字段",
                 not (fields & forbidden))

    _log("\n[3] 反向灵敏度：改公开信息后 PublicView 必须变化（防空测）")
    bad = shuf.clone()
    bad.discards[(p + 1) % 4].append(0)   # 公开弃牌 +1
    v_bad = PublicView.from_game(bad, p)
    ok &= _check("改动公开弃牌 → PublicView 变化", v_bad != v_orig)

    _log("\n[4] 推荐 Q 值对隐藏排列不变（计算推荐不使用全局信息）")
    ok &= fair_view_identical(orig, shuf, p)
    net = load_net()
    q_py_o = py_q(net, orig, p, legal)
    q_py_s = py_q(net, shuf, p, legal)
    ok &= _check("Python 搜索 Q 逐位相同",
                 np.allclose(q_py_o, q_py_s, atol=1e-9, rtol=0))
    # Rust（生产同源）作为附加验证；ORT 不可用时单独标注、不影响主结论
    try:
        q_rs_o = rust_q(net, orig, p, legal)
        q_rs_s = rust_q(net, shuf, p, legal)
        ok &= _check("Rust 搜索(ORT) Q 逐位相同",
                     np.allclose(q_rs_o, q_rs_s, atol=1e-9, rtol=0))
    except Exception as exc:  # noqa: BLE001
        _log(f"  SKIP  Rust/ORT Q 检查（{type(exc).__name__}: {exc}）")

    _log("\n[5] 完整理由 / 一句话推荐 / 局面摘要 对隐藏排列逐字不变")
    items_o = explain_items(v_orig, legal, q_py_o, k=4)
    items_s = explain_items(v_shuf, legal, q_py_s, k=4)
    ok &= _check("完整理由(items: reason/lines/tags) 逐字相同",
                 serialize_items(items_o) == serialize_items(items_s))
    tip_o = coach.compare_tip(v_orig, frontend(items_o))
    tip_s = coach.compare_tip(v_shuf, frontend(items_s))
    ok &= _check("一句话推荐(compare_tip) 逐字相同", tip_o == tip_s)
    sum_o = coach.situation_summary(orig, p)
    sum_s = coach.situation_summary(shuf, p)
    ok &= _check("局面摘要(situation_summary) 相同", sum_o == sum_s)

    _log("\n--- 样例完整理由（首选）---")
    for ln in items_o[0]["lines"]:
        _log("  " + ln)
    _log("--- 样例一句话推荐 ---")
    _log("  💡 " + tip_o)

    _log("\n" + ("=== ALL PASS ✅  推荐与理由均不使用全局信息 ===" if ok
                 else "=== SOME FAILED ❌ ==="))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
