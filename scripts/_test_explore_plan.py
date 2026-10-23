"""探索计划离线仿真：验证 explore_plan 的配方选择、命中判定与等待语义。

用法：C:/Users/LIGF/.workbuddy/binaries/python/envs/zhuoji/Scripts/python.exe scripts/_test_explore_plan.py
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import m2_night_autopilot as ap          # noqa: E402

REP = ROOT / "reports"
MOCK = REP / "AUTO6_it0240.json"
SRC = ROOT / "models" / "_ckpts" / "dushan_m2_auto5_it0800.pt"

ok = True


def check(label, cond, extra=""):
    global ok
    mark = "✅" if cond else "❌"
    if not cond:
        ok = False
    print(f"  {mark} {label}{('  ' + extra) if extra else ''}")


def mock_verdict(va, ci=0.27):
    MOCK.write_text(json.dumps({
        "anchor": "dushan_master",
        "rows": [
            {"ckpt": "dushan_master", "vs_anchor": 0.0, "vs_anchor_ci95": 0.0},
            {"ckpt": "_ckpts/dushan_m2_auto6_it0240", "vs_anchor": va,
             "vs_anchor_ci95": ci},
        ],
    }, ensure_ascii=False), encoding="utf-8")


base_st = dict(current="auto5", relay_count=3,
               dyn_segs={"auto4": {}, "auto5": {}})

print("=" * 72)
print("① gen=6（首次进入探索，应取第 0 个旋钮 lr-restart / 300 轮）")
p = ap.explore_plan(dict(base_st), 6, SRC)
check("返回计划非空", p is not None)
check("tag=AUTO6", p["tag"] == "AUTO6", p["tag"])
check("iters=300", p["iters"] == 300, str(p["iters"]))
check("旋钮=lr-restart(idx 0)", p["knob"] == 0, str(p["knob"]))
check("判读点 [160,240]", p["checkpoints"] == [160, 240], str(p["checkpoints"]))
check("verdict_at=240", p["verdict_at"] == 240, str(p["verdict_at"]))
check("起点 va=-0.85", abs(p["start_va"] + 0.85) < 1e-9, f"{p['start_va']}")
check("lr=2.2e-6", abs(p["dyn"]["lr"] - 2.2e-6) < 1e-12, str(p["dyn"]["lr"]))
print(f"      why: {p['why']}")

print("=" * 72)
print("② gen=7 且判读缺失 → 必须返回 None（等待，不换旋钮）")
MOCK.unlink(missing_ok=True)
p = ap.explore_plan(dict(base_st), 7, SRC)
check("返回 None", p is None)

print("=" * 72)
print("③ gen=7 且命中（va −0.45 vs 基线 −0.85，改善 +0.40 且 CI 下界 > 基线）")
st = dict(base_st, explore_idx=0, explore_base=-0.85, explore_verdict_at=240)
mock_verdict(-0.45)
p = ap.explore_plan(dict(st), 7, SRC)
check("返回计划非空", p is not None)
check("沿用旋钮 idx 0", p["knob"] == 0, str(p["knob"]))
check("加长到 800 轮", p["iters"] == 800, str(p["iters"]))
check("判读点 [400,640]", p["checkpoints"] == [400, 640], str(p["checkpoints"]))
check("verdict_at=640", p["verdict_at"] == 640, str(p["verdict_at"]))
check("起点取实测 -0.45", abs(p["start_va"] + 0.45) < 1e-9, f"{p['start_va']}")
check("`命中` 出现在 why", "命中" in p["why"], p["why"])

print("=" * 72)
print("④ gen=7 且未命中（va −0.80，改善 +0.05 < 0.20）→ 换下一个旋钮 explore")
mock_verdict(-0.80)
p = ap.explore_plan(dict(st), 7, SRC)
check("返回计划非空", p is not None)
check("换成旋钮 idx 1（explore）", p["knob"] == 1, str(p["knob"]))
check("保持 300 轮", p["iters"] == 300, str(p["iters"]))
check("temperature=1.35", abs(p["dyn"]["temperature"] - 1.35) < 1e-9, str(p["dyn"]["temperature"]))
check("ent-coef 0.055", "0.055" in p["dyn"]["extra"], str(p["dyn"]["extra"][p["dyn"]["extra"].index("--ent-coef") + 1]))

print("=" * 72)
print("⑤ 旋钮指针越界回绕（idx=2 未命中 → 回 idx 0）")
st2 = dict(st, explore_idx=2)
mock_verdict(-0.80)
p = ap.explore_plan(dict(st2), 7, SRC)
check("回绕到 idx 0", p["knob"] == 0, str(p["knob"]))
check("miss=1", p.get("miss") == 1, str(p.get("miss")))
check("记录历史最优（best_knob=2, best_va=-0.80）",
      p.get("best_knob") == 2 and abs(p.get("best_va", 0) + 0.80) < 1e-9,
      f"{p.get('best_knob')}/{p.get('best_va')}")

print("=" * 72)
print("⑤b 收敛长段：连续第 3 次未命中 → 历史最优旋钮 + 1200 轮慢退火")
st3 = dict(st, explore_idx=2, explore_miss=2,
           explore_best_knob=1, explore_best_va=-0.70)
mock_verdict(-0.85, ci=0.27)          # 改善 +0.15 < 0.20 → 未命中（且低于 best −0.70）
p = ap.explore_plan(dict(st3), 7, SRC)
check("进入收敛长段", p["iters"] == 1200, str(p["iters"]))
check("用历史最优旋钮 idx 1（explore）", p["knob"] == 1, str(p["knob"]))
check("判读点 [600,1000]", p["checkpoints"] == [600, 1000], str(p["checkpoints"]))
check("verdict_at=1000", p["verdict_at"] == 1000, str(p["verdict_at"]))
check("miss 清零（收敛后重开一轮）", p.get("miss") == 0, str(p.get("miss")))
check("why 含 `收敛长段`", "收敛长段" in p["why"], p["why"])
ex = p["dyn"]["extra"]
ia = ex.index("--klt-anneal-iters")
check("收敛段慢退火 klt-anneal=300", ex[ia + 1] == "300", ex[ia + 1])
iw = ex.index("--lr-warmup")
check("收敛段 warmup=40", ex[iw + 1] == "40", ex[iw + 1])
check("收敛段 lr 打 9 折（1.8e-06）", abs(p["dyn"]["lr"] - 1.8e-6) < 1e-9, str(p["dyn"]["lr"]))

print("=" * 72)
print("⑥ 命令行构建：覆盖项必须在末尾生效（argparse 取最后一个）")
cmd = ap.build_cmd("auto6", p["dyn"], 12)
i = len(cmd) - 1 - cmd[::-1].index("--snapshot-pool")       # 找**最后一个**（覆盖 COMMON 的那个）
check("末尾 --snapshot-pool 紧邻空串", cmd[i + 1] == "", repr(cmd[i + 1]))
check("其后是合法选项", i + 2 >= len(cmd) or cmd[i + 2].startswith("--"),
      cmd[i + 2] if i + 2 < len(cmd) else "(end)")
j = len(cmd) - 1 - cmd[::-1].index("--ent-coef")
_exp = p["dyn"]["extra"]
_expect = _exp[_exp.index("--ent-coef") + 1]
check(f"末尾 --ent-coef 覆盖为计划值 {_expect}", cmd[j + 1] == _expect, cmd[j + 1])
k = len(cmd) - 1 - cmd[::-1].index("--kl-coef")
check("末尾 --kl-coef 覆盖为 0.016", cmd[k + 1] == "0.016", cmd[k + 1])
print(f"      命令长度 {len(cmd)}；末尾覆盖项：")
for flag in ("--lr", "--lr-min", "--lr-warmup", "--klt-beta", "--klt-anneal-iters",
             "--temperature", "--master-seat-prob", "--ent-coef", "--snapshot-pool"):
    if flag in cmd:
        m = len(cmd) - 1 - cmd[::-1].index(flag)
        nxt = cmd[m + 1] if m + 1 < len(cmd) else "(end)"
        print(f"        {flag} {'<空串>' if nxt == '' else ('' if nxt.startswith('--') else nxt)}")

print("=" * 72)
print("⑦ 端到端：用真实 state 走 decide（把 auto5 当作已跑满 800 轮）")
st_real = ap.load_state()
dyn5 = (st_real.get("dyn_segs") or {}).get("auto5")
check("state 里有 auto5 的段配置", dyn5 is not None,
      str(list((st_real.get("dyn_segs") or {}).keys())))
check("state.current=auto5", st_real.get("current") == "auto5", str(st_real.get("current")))
check("state.relay_count=3", st_real.get("relay_count") == 3, str(st_real.get("relay_count")))
if dyn5:
    meta5 = ap.seg_meta("auto5", dyn5)
    d = ap.decide(dict(st_real), meta5, alive=False, it_done=800)
    check("decide 返回接力计划", d is not None, str(d and d.get("tag")))
    if d:
        check("接力到 AUTO6", d["tag"] == "AUTO6", d["tag"])
        check("走探索配方（300 轮）", d["iters"] == 300, str(d["iters"]))
        check("旋钮=lr-restart", d["knob"] == 0, str(d["knob"]))

MOCK.unlink(missing_ok=True)
print("=" * 72)
print("全部通过 ✅" if ok else "存在失败 ❌")
sys.exit(0 if ok else 1)
