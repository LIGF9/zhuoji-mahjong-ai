#!/usr/bin/env python3
"""O0 黄金样本：用 Python 的 DushanGame + Session._snapshot 生成 /api/state 快照。

离线版的状态视图必须与联网服务端逐字段一致。做法是把 Python 洗好的牌墙
原样交给 Rust `Game::new_with_wall`（两边发牌顺序/默认配置已核对相同），
再按同一串动作回放，逐个快照比对。

用法：.venv/bin/python tools/offline_golden.py
输出：runs/offline_golden.json
"""
from __future__ import annotations

import json
import sys
from collections import deque
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from zhuoji.dushan import DushanGame  # noqa: E402
from zhuoji.rules import Phase  # noqa: E402
from web.dushan_server import Session  # noqa: E402

SEEDS = (12345, 777, 20261111)
STEPS = 12


def make_session(me: int) -> Session:
    """绕过 __init__ 构造最小会话壳，_snapshot 只依赖这些字段。"""
    s = Session.__new__(Session)
    s.my_seat = me
    s.names = ["我", "下家", "对家", "上家"]
    s.kinds = ["human", "model", "model", "model"]
    s.cum = [0.0] * 4
    s.dealer_streak = 0
    s.hand_index = 1
    s.sid = "golden"
    s.error = None
    s.events = deque(maxlen=600)
    s.stats = {p: {} for p in range(4)}
    s.opp_think = {}
    s.state = {"pending": None}
    return s


def main() -> int:
    cases = []
    for seed in SEEDS:
        g = DushanGame(seed=seed, dealer=0)
        wall = [int(t) for t in g.wall]
        s = make_session(0)
        snaps, acts = [], []
        for _ in range(STEPS):
            if g.phase == Phase.OVER:
                break
            legal = g.legal_actions()
            s.state["pending"] = s._pending_payload(g, legal)
            snaps.append(s._snapshot(g))
            a = legal[0]  # 确定性：取第一个合法动作
            acts.append([a.kind, int(a.tile)])
            g.step(a)
        cases.append({
            "seed": seed,
            "wall": wall,
            "dealer": int(g.dealer),
            "me": 0,
            "actions": acts,
            "snapshots": snaps,
        })
        print(f"seed={seed}: {len(snaps)} 个快照 / {len(acts)} 步")
    out = ROOT / "runs" / "offline_golden.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"cases": cases}, ensure_ascii=False), encoding="utf-8")
    print(f"已写 {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
