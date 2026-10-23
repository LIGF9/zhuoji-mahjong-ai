"""临时探针：拿一条**胡牌局**的终局快照（final），给前端回放结算页做定向验证。

把 Session._persist 换成只捕获不落盘（不污染 history），反复 _play_hand 直到
出现 type == "win"，把 replay["final"] 存成 reports/_probe_final_win.json。
逐步打点写文件，避免长跑被 timeout 掐掉后看不到进度。
"""
from __future__ import annotations

import importlib.util
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "web"))

LOG = ROOT / "reports" / "_probe_final_win.log"
OUT = ROOT / "reports" / "_probe_final_win.json"


def log(msg: str) -> None:
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(msg + "\n")


LOG.write_text("", encoding="utf-8")

_spec = importlib.util.spec_from_file_location("dushan_server", ROOT / "web" / "dushan_server.py")
S = importlib.util.module_from_spec(_spec)
sys.modules["dushan_server"] = S
_spec.loader.exec_module(S)

captured = {"record": None, "replay": None}


def fake_persist(self, record, replay=None):   # noqa: ANN001
    captured["record"] = record
    captured["replay"] = replay


S.Session._persist = fake_persist

kinds = {"win": 0, "huangzhuang": 0, "other": 0}
OPPS = [{"type": "teacher", "style": "aggressive"},
        {"type": "teacher", "style": "balanced"},
        {"type": "teacher", "style": "tenpai_rush"}]
t_all = time.time()
for trial in range(60):
    sess = S.Session("probe", [dict(o) for o in OPPS], 0, seed=1000 + trial,
                     speed=0.0, auto=True)
    for h in range(8):
        captured["record"] = captured["replay"] = None
        t0 = time.time()
        sess._play_hand()
        dt = time.time() - t0
        rp = captured["replay"]
        res = ((rp or {}).get("final") or {}).get("result") or {}
        t = res.get("type") or "none"
        kinds[t if t in kinds else "other"] = kinds.get(t, 0) + 1
        log(f"trial={trial} hand={h} {dt:.2f}s type={t}")
        if t == "win" and rp:
            OUT.write_text(json.dumps({
                "trial": trial, "hand": rp.get("hand"), "my_seat": rp.get("my_seat"),
                "names": rp.get("names"), "ji_tiles": rp.get("ji_tiles"),
                "ev_n": len(rp.get("ev") or []),
                "final": rp.get("final"),
            }, ensure_ascii=False), encoding="utf-8")
            log(f"WIN saved deltas={res.get('deltas')} how={res.get('how')} "
                f"fan={res.get('fan_cn')} winners={res.get('winners')} "
                f"size={OUT.stat().st_size}")
            log(f"total {time.time() - t_all:.1f}s kinds={kinds}")
            sys.exit(0)
log(f"no win; total {time.time() - t_all:.1f}s kinds={kinds}")
sys.exit(1)
