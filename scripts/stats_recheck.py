"""stats_recheck.py —— 「历史累计」统计口径的独立复核器

为什么需要它
------------
`/api/stats` 的口径是本项目最容易「修了但没人敢信」的部分：每项都要求
分母正确、且 2026-10-12 之后新增了两项以「全程没开过模型大师推荐」为分母的
统计。这里**不调用服务端聚合代码**，而是按口径说明从 `web/history/records.jsonl`
原始记录重新算一遍，再跟 `/api/stats` 回传的结果逐字段对照——
两边对不上就是真的有问题（而不是"看起来对"）。

口径（截至 2026-10-12，与 dushan_server.py /api/stats 的注释一致）
----------------------------------------------------------------
* 分母一律是「对局数 games」＝ records.jsonl 里所有 `type != "abort"` 的行。
  「结束本局」提前终止的局不写历史、不计任何一项。
* 自摸 zimo            : 我自摸胡牌的局数（我的座位 my_seat 在 winners 里 且 is_tsumo）
* 送胡 songhu_games    : **整局视角** —— 该局只要有人不是自摸胡（点炮 / 抢杠 / 热炮…）
                         就计一局，与"我"的输赢无关
* 冲锋鸡 chongfeng_games / 出鸡 ji_discard_games / 包鸡 baoji_games
                      : 该局是否出现过对应事件的局数（flags 优先，旧记录用 detail 标签兜底）
* 非平胡·听牌 tenpai_bigfan_games / 非平胡·胡牌 win_bigfan_games
                      : 「非平胡」= 既不是「平胡」也不是「报叫」。
                        2026-10-12 用户口径：报叫胡牌时引擎把 fan_cn 显示成「报叫」
                        （不再是平胡），但它并不是具名大牌型 —— 统计口径要跟「平胡」
                        一样排除，否则报叫局会被误计成非平胡局。
* 黄庄·局占率 huang_games    : type != "win" 的局
* 黄庄·未听牌 huang_noten_games : 黄庄且至少一家没听牌
* 本人胡牌 solo_win_games / 本人累计 solo_score
                      : 只在「本局全程没调过 /api/hint」的局里统计；
                        **分母是 solo_games（这些局数），不是全部对局**
                        flags.hint_used 缺失的旧记录按"用过推荐"处理、不计入
* score               : 我在所有对局里的输赢总分（每局取 deltas[my_seat]）

用法
----
    <venv>/python.exe scripts/stats_recheck.py            # 只看结论
    <venv>/python.exe scripts/stats_recheck.py -v         # 逐局明细
"""
from __future__ import annotations

import json
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
HIST = ROOT / "web" / "history" / "records.jsonl"
API = "http://127.0.0.1:8770/api/stats?sid=__recheck__"

FIELDS = ["games", "zimo", "songhu_games", "score",
          "chongfeng_games", "ji_discard_games", "baoji_games",
          "tenpai_bigfan_games", "win_bigfan_games",
          "huang_games", "huang_noten_games",
          "solo_games", "solo_win_games", "solo_score"]


def recompute() -> tuple[dict, list[str]]:
    agg = {k: 0 for k in FIELDS}
    agg["score"] = 0.0
    agg["solo_score"] = 0.0
    lines: list[str] = []
    if not HIST.exists():
        return agg, ["（无历史文件）"]

    for ln, raw in enumerate(HIST.read_text(encoding="utf-8").splitlines(), 1):
        raw = raw.strip()
        if not raw:
            continue
        r = json.loads(raw)
        if r.get("type") == "abort":
            lines.append(f"#{ln} abort（提前结束，不计任何项）")
            continue

        me = int(r.get("my_seat", 0))
        d = r.get("deltas") or [0.0] * 4
        dme = float(d[me]) if me < len(d) else 0.0
        fl = r.get("flags") or {}
        labels = [str(x[1]) for x in (r.get("detail") or [])]
        winners = [int(x) for x in (r.get("winners") or [r.get("winner")]) if x is not None]

        agg["games"] += 1
        agg["score"] += dme

        chongfeng = bool(fl.get("chongfeng", "冲锋鸡" in labels))
        ji_discard = bool(fl.get("ji_discard",
                                 any(x in labels for x in ("冲锋鸡", "横鸡", "幺鸡"))))
        baoji = bool(fl.get("baoji", "包鸡" in labels))
        tenpai_big = bool(fl.get("tenpai_bigfan", False))

        # 本人胡牌 / 本人累计：分母 = 全程没看过大师推荐的局
        solo = not fl.get("hint_used", True)

        hit = []
        if chongfeng:
            agg["chongfeng_games"] += 1
            hit.append("冲锋鸡")
        if ji_discard:
            agg["ji_discard_games"] += 1
            hit.append("出鸡")
        if baoji:
            agg["baoji_games"] += 1
            hit.append("包鸡")
        if tenpai_big:
            agg["tenpai_bigfan_games"] += 1
            hit.append("非平胡·听牌")
        if solo:
            agg["solo_games"] += 1
            agg["solo_score"] += dme
            hit.append(f"solo(本人累计 {dme:+.1f})")

        if r.get("type") == "win":
            if not r.get("is_tsumo"):
                agg["songhu_games"] += 1
                hit.append("送胡")
            big = fl.get("win_bigfan")
            if big is None:
                # 与 dushan_server.is_bigfan_name 同口径（本脚本刻意不 import，保持独立复核）
                big = str(r.get("fan_cn") or "") not in ("", "平胡", "报叫")
            if big:
                agg["win_bigfan_games"] += 1
                hit.append("非平胡·胡牌")
            if me in winners:
                if r.get("is_tsumo"):
                    agg["zimo"] += 1
                    hit.append("自摸")
                if solo:
                    agg["solo_win_games"] += 1
                    hit.append("本人胡牌")
        else:
            agg["huang_games"] += 1
            hit.append("黄庄")
            if fl.get("huang_noten", len(r.get("tenpai") or []) < 4):
                agg["huang_noten_games"] += 1
                hit.append("未听牌")

        lines.append(
            # type 可能为 None（历史脏数据 / 未完成即中止的旧记录），先兜底成 "-"
            f"#{ln} type={str(r.get('type') or '-'):<4} fan={str(r.get('fan_cn') or '-'):<6} "
            f"my_seat={me} 我得{dme:+.1f} hint_used={fl.get('hint_used')} "
            f"winners={winners} -> {', '.join(hit) or '—'}")

    return agg, lines


def main() -> int:
    verbose = "-v" in sys.argv or "--verbose" in sys.argv
    mine, lines = recompute()
    if verbose:
        print("逐局明细：")
        for ln in lines:
            print("  " + ln)
        print()

    try:
        with urllib.request.urlopen(API, timeout=5) as resp:
            srv = json.loads(resp.read().decode("utf-8"))["all_time"]
    except Exception as exc:  # 服务没起也能跑本地复算
        print(f"[warn] 取不到 {API}（{exc}），只输出本地复算结果：")
        srv = None

    print(f"{'字段':<24}{'本地复算':>12}{'服务端':>12}   一致")
    print("-" * 60)
    bad = 0
    for k in FIELDS:
        a, b = mine[k], (srv or {}).get(k, "—")
        if isinstance(a, float):
            a, b = round(a, 4), (round(float(b), 4) if b != "—" else b)
        ok = (b == "—") or (a == b)
        if not ok:
            bad += 1
        print(f"{k:<24}{str(a):>12}{str(b):>12}   {'OK' if ok else 'MISMATCH'}")
    print("-" * 60)
    if srv is None:
        print("服务端未取到，无法判定（本地复算已输出）")
        return 0
    if bad:
        print(f"{bad} 个字段不一致 —— 口径可能被改坏，请对照 /api/stats 代码复查")
        return 1
    print("全部字段一致 —— 统计口径自洽")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
