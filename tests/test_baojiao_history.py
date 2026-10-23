"""
2026-10-12 晚批次的服务端/引擎口径回归。

覆盖：
  A. 报叫显示名（引擎 `_fan_cn_of`）：报叫胡牌显示「报叫」而不是「平胡」，
     但 `fan_type` 与统计口径不变；
  B. 报叫分默认值 13 → **10**（引擎默认 + 前端 rules 载荷两个入口）；
  C. 「具名大牌型」判定把「报叫」与「平胡」一起排除（`is_win_bigfan` / `is_bigfan_name`）；
  D. 对局记录删除（`_rewrite_jsonl` / `delete_history_record`）—— 含 rid 安全校验、
     行号越界、缓存剔除；删除后 `/api/stats` 的口径来源（records.jsonl）自然同步。

跑法::

    python tests/test_history_delete.py
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "web"))

from zhuoji.dushan import (  # noqa: E402
    DushanConfig,
    DushanGame,
    PING_HU,
    ZI_MO_JI,
)
from zhuoji.rules import Phase  # noqa: E402
from zhuoji.tiles import B, T, W, tile_from  # noqa: E402

import dushan_server as DS  # noqa: E402

PASSED: list[str] = []
FAILED: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    (PASSED if ok else FAILED).append(name)
    print(f"  [{'ok  ' if ok else 'FAIL'}] {name}" + (f"   {detail}" if detail else ""),
          flush=True)


def counts(*tiles: int) -> list[int]:
    c = [0] * 27
    for t in tiles:
        c[t] += 1
    return c


# ---------------------------------------------------------------------------
# A / B：报叫显示名 + 默认分
# ---------------------------------------------------------------------------
def _plain_win(baojiao: bool) -> DushanGame:
    """构造「p0 平胡自摸」的盘子：其余三家散牌、无人持鸡、翻鸡牌无人持有。

    与 tests/test_dushan.py::test_tsumo_payment 同构 —— 那里已验证该盘面
    除「自摸 +3×3」外没有别的进账。"""
    g = DushanGame(DushanConfig(), seed=7)
    g.hands[0] = counts(*[tile_from(W, r) for r in (1, 2, 3, 4, 5, 6, 7, 8, 9)],
                        *[tile_from(T, r) for r in (2, 3, 4, 5, 5)])
    g.melds[0] = []
    for p in (1, 2, 3):
        g.hands[p] = [0] * 27
        for r in (1, 3, 5):
            g.hands[p][tile_from(B, r)] += 1
        g.melds[p] = []
    g.wall[g.wall_pos] = tile_from(T, 7)      # 翻鸡牌：无人持有
    g.current = 0
    g.phase = Phase.DISCARD
    g._any_discard = True
    g.baojiao[0] = baojiao
    g._settle_win(0, -1, is_tsumo=True)
    return g


def test_baojiao_display_name() -> None:
    # 未报叫 → 平胡
    g0 = _plain_win(False)
    check("A1 未报叫平胡局 fan_cn=平胡", g0.result["fan_cn"] == "平胡", g0.result["fan_cn"])
    check("A2 winners_fan[0].fan_cn=平胡",
          g0.result["winners_fan"][0]["fan_cn"] == "平胡",
          str(g0.result["winners_fan"]))
    # 报叫 → 显示名变「报叫」
    g1 = _plain_win(True)
    check("A3 报叫平胡局 fan_cn=报叫（不再显示平胡）",
          g1.result["fan_cn"] == "报叫", g1.result["fan_cn"])
    check("A4 winners_fan[0].fan_cn=报叫",
          g1.result["winners_fan"][0]["fan_cn"] == "报叫",
          str(g1.result["winners_fan"]))
    # 统计口径不变：fan_type 仍是 ping_hu
    check("A5 fan_type 仍是 ping_hu（只改显示名，统计口径不动）",
          g1.result["fan_type"] == PING_HU and
          g1.result["winners_fan"][0]["fan_type"] == PING_HU,
          f"{g1.result['fan_type']} / {g1.result['winners_fan'][0]['fan_type']}")
    # 具名牌型优先：报叫 + 清一色/大对子时显示具名牌型
    g2 = DushanGame(DushanConfig(), seed=7)
    # W1×3 W2×3 W3×3 W4×3 W5×2 → 清一色 + 大对子（具名），且已报叫
    g2.hands[0] = [0] * 27
    for r in (1, 2, 3, 4):
        g2.hands[0][tile_from(W, r)] = 3
    g2.hands[0][tile_from(W, 5)] = 2
    g2.melds[0] = []
    for p in (1, 2, 3):
        g2.hands[p] = [0] * 27
        for r in (1, 3, 5):
            g2.hands[p][tile_from(B, r)] += 1
        g2.melds[p] = []
    g2.wall[g2.wall_pos] = tile_from(T, 7)
    g2.current = 0
    g2.phase = Phase.DISCARD
    g2._any_discard = True
    g2.baojiao[0] = True
    g2._settle_win(0, -1, is_tsumo=True)
    cn2 = g2.result["fan_cn"]
    check("A6 报叫 + 具名牌型 → 显示具名牌型（不是「报叫」）",
          cn2 != "报叫" and "清一色" in cn2, cn2)


def test_baojiao_bonus_default() -> None:
    cfg = DushanConfig()
    check("B1 引擎默认 baojiao_bonus = 10（13 → 10）",
          abs(float(cfg.baojiao_bonus) - 10.0) < 1e-9, str(cfg.baojiao_bonus))
    g = _plain_win(True)
    rows = [d for d in g.result["detail"] if d[1] == "报叫"]
    check("B2 结算明细里有「报叫」项且值 = 10",
          bool(rows) and abs(float(rows[0][3]) - 10.0) < 1e-9,
          str(rows[:2]))
    # 报叫 10 × 3 家 + 自摸 3 × 3 家 = 39
    check("B3 报叫自摸净收 = 自摸9 + 报叫30 = 39",
          abs(g.result["deltas"][0] - (3 * ZI_MO_JI + 30.0)) < 1e-9,
          str(g.result["deltas"]))
    # 服务端 rules 载荷（前端设置面板的默认值入口）
    cfg2 = DS.config_from_rules({})
    check("B4 服务端 config_from_rules({}) 默认 10",
          abs(float(cfg2.baojiao_bonus) - 10.0) < 1e-9, str(cfg2.baojiao_bonus))
    cfg3 = DS.config_from_rules({"scores": {"baojiao": 13}})
    check("B5 rules 显式给 13 时仍然生效（可配置）",
          abs(float(cfg3.baojiao_bonus) - 13.0) < 1e-9, str(cfg3.baojiao_bonus))


# ---------------------------------------------------------------------------
# C：具名大牌型判定
# ---------------------------------------------------------------------------
def test_bigfan_predicate() -> None:
    check("C1 报叫不算具名大牌型",
          DS.is_win_bigfan([{"fan_cn": "报叫"}]) is False)
    check("C2 平胡不算具名大牌型",
          DS.is_win_bigfan([{"fan_cn": "平胡"}]) is False)
    check("C3 空 / None 不算",
          DS.is_win_bigfan([{"fan_cn": ""}, {}]) is False and
          DS.is_win_bigfan(None) is False)
    check("C4 清一色算具名大牌型",
          DS.is_win_bigfan([{"fan_cn": "清一色"}]) is True)
    check("C5 一炮多响：任一人具名即算",
          DS.is_win_bigfan([{"fan_cn": "报叫"}, {"fan_cn": "大对子"}]) is True)
    check("C6 一炮多响：全报叫/平胡 → 不算",
          DS.is_win_bigfan([{"fan_cn": "报叫"}, {"fan_cn": "平胡"}]) is False)
    # 旧记录兜底（按字符串）
    check("C7 兜底 is_bigfan_name：报叫/平胡/空 排除，清一色保留",
          DS.is_bigfan_name("报叫") is False and DS.is_bigfan_name("平胡") is False and
          DS.is_bigfan_name(None) is False and DS.is_bigfan_name("清一色") is True)
    check("C8 BIGFAN_EXCLUDE 含 报叫",
          "报叫" in DS.BIGFAN_EXCLUDE and "平胡" in DS.BIGFAN_EXCLUDE)


# ---------------------------------------------------------------------------
# D：对局记录删除
# ---------------------------------------------------------------------------
def _write(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows),
                    encoding="utf-8")


def _read(path: Path) -> list[dict]:
    return [json.loads(s) for s in path.read_text(encoding="utf-8").splitlines() if s.strip()]


def test_delete_history_record() -> None:
    with tempfile.TemporaryDirectory() as td:
        recs = Path(td) / "records.jsonl"
        reps = Path(td) / "replays.jsonl"
        rows = [{"rid": f"r{i}", "type": "win", "fan_cn": "清一色"} for i in range(3)]
        _write(recs, rows)
        _write(reps, [{"rid": f"r{i}", "ev": []} for i in range(3)])

        # D1 按行号删中间一条
        out = DS.delete_history_record(recs, reps, 1, "r1")
        left = _read(recs)
        check("D1 按行号删中间一条 → ok，剩 2 条且 r1 消失",
              out["ok"] and out["removed_rec"] == 1 and out["removed_replay"] == 1 and
              [r["rid"] for r in left] == ["r0", "r2"],
              f"{out} left={[r['rid'] for r in left]}")
        check("D1b 同 rid 的回放也被删掉",
              [r["rid"] for r in _read(reps)] == ["r0", "r2"],
              str([r["rid"] for r in _read(reps)]))

        # D2 rid 与行号不匹配 → 拒删，文件不动
        before = recs.read_text(encoding="utf-8")
        out = DS.delete_history_record(recs, reps, 0, "r2")
        check("D2 rid 与行号不匹配 → 拒删且文件不变",
              (not out["ok"]) and "不匹配" in out["error"] and
              recs.read_text(encoding="utf-8") == before, f"{out}")

        # D3 行号越界 → 拒删
        out = DS.delete_history_record(recs, reps, 99, None)
        check("D3 行号越界 → 拒删", (not out["ok"]) and "越界" in out["error"], f"{out}")

        # D4 只给 rid（不给行号）→ 自己找行删除
        out = DS.delete_history_record(recs, reps, None, "r0")
        check("D4 只按 rid 删 → ok，剩 r2",
              out["ok"] and [r["rid"] for r in _read(recs)] == ["r2"], f"{out}")

        # D5 rid 找不到
        out = DS.delete_history_record(recs, reps, None, "nope")
        check("D5 rid 不存在 → 报「已不在文件里」",
              (not out["ok"]) and "已不在文件里" in out["error"], f"{out}")

        # D6 参数都没给
        out = DS.delete_history_record(recs, reps, None, None)
        check("D6 缺定位参数 → 拒删", (not out["ok"]) and "缺少定位参数" in out["error"], f"{out}")

        # D7 记录文件不存在
        out = DS.delete_history_record(Path(td) / "nope.jsonl", reps, 0, "x")
        check("D7 记录文件不存在 → 友好报错",
              (not out["ok"]) and "还没有对局记录" in out["error"], f"{out}")

        # D8 内存回放缓存里的同 rid 也被剔除
        DS._REPLAY_CACHE.clear()
        DS._REPLAY_CACHE["r2"] = {"rid": "r2"}
        out = DS.delete_history_record(recs, reps, 0, "r2")
        check("D8 删除后内存回放缓存剔除该 rid",
              out["ok"] and "r2" not in DS._REPLAY_CACHE and _read(recs) == [],
              f"{out} cache={list(DS._REPLAY_CACHE)}")

        # D9 坏行（非 JSON）不影响其它行删除；重写用 .tmp → 无残留
        recs.write_text('{"rid":"a"}\nnot-json\n{"rid":"b"}\n', encoding="utf-8")
        out = DS.delete_history_record(recs, reps, 2, "b")
        raw = recs.read_text(encoding="utf-8")
        check("D9 坏行保留、目标行删除、无 .tmp 残留",
              out["ok"] and "not-json" in raw and '"a"' in raw and '"b"' not in raw and
              not (Path(td) / "records.jsonl.tmp").exists(),
              repr(raw))


def main() -> int:
    print("=== A/B 报叫显示名与默认分 ===")
    test_baojiao_display_name()
    test_baojiao_bonus_default()
    print("\n=== C 具名大牌型判定 ===")
    test_bigfan_predicate()
    print("\n=== D 对局记录删除 ===")
    test_delete_history_record()
    print(f"\n通过 {len(PASSED)} / 共 {len(PASSED) + len(FAILED)}", flush=True)
    if FAILED:
        print("失败项：" + "、".join(FAILED), flush=True)
        return 1
    print("全部通过 ✅", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
