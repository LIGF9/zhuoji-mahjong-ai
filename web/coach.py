# -*- coding: utf-8 -*-
"""独山麻将教练解释引擎：把模型候选动作翻译成可读、可教学的中文解释。

原则：
1. 只用公开信息（各家弃牌、副露、自己的手牌）推断，不做透视；
2. 分数口径与引擎结算完全一致（分值读 DushanConfig，设置里改了自动同步）；
3. 每条解释带标签（听牌/鸡分/风险/牌型），前端渲染成彩色小标签 + 多行文字。
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from zhuoji.dushan import DushanGame, JI_CHONGFENG, JI_HENG, JI_YAO  # noqa: E402
from zhuoji.fan import (  # noqa: E402
    meld_is_kong,
    winning_tiles,
)
from zhuoji.rules import (  # noqa: E402
    ANGANG,
    BUGANG,
    DISCARD,
    HU,
    MINGGANG,
    PASS,
    PENG,
    Action,
)
from zhuoji.shanten import shanten  # noqa: E402
from zhuoji.tiles import NUM_TILE_TYPES, suit_of, tile_cn  # noqa: E402

REL_CN = {1: "下家", 2: "对家", 3: "上家"}


# ---------------------------------------------------------------------------
# 基础分析工具
# ---------------------------------------------------------------------------
def _live_count(game: DushanGame, me: int, t: int, hand: list[int]) -> int:
    """牌 t 在公开信息之外剩余的张数（可供摸/胡）。"""
    used = 0
    for p in range(4):
        used += game.discards[p].count(t)
    for p in range(4):
        for mt, tt, _s in game.melds[p]:
            if tt == t:
                used += 4 if meld_is_kong(mt) else 3
    used += hand[t]
    return max(0, 4 - used)


def _hand_ji(game: DushanGame, me: int) -> int:
    """我手中（暗手）持有的鸡牌张数。"""
    return sum(game.hands[me][t] for t in game.ji_tiles)


def _ji_name_of_discard(game: DushanGame, me: int, tile: int) -> str | None:
    """这张鸡牌按引擎口径打出后会成为什么：冲锋鸡/横鸡/幺鸡。

    与引擎同源（``game.ji_name_if_discarded``）：横鸡轮内跟打同种鸡也算横鸡。
    """
    return game.ji_name_if_discarded(me, tile) or None


def _winning_line(game: DushanGame, me: int, hand13: list[int]) -> str:
    """听牌描述：叫 X 剩 n、Y 剩 m（含牌型提示）。"""
    calls = winning_tiles(hand13, game.melds[me])
    if not calls:
        return ""
    parts: list[str] = []
    for t in calls[:4]:
        parts.append(f"{tile_cn(t)}(剩{_live_count(game, me, t, hand13)}张)")
    if len(calls) > 4:
        parts.append("…")
    # 牌型提示：听的牌里有没有值钱牌型
    premium: set[str] = set()
    cfg = game.cfg
    type_names = {"da_dui_zi": "大对子", "qi_dui": "小七对", "long_qi_dui": "龙七对",
                  "qing_yi_se": "清一色", "dan_diao": "单吊", "ping_hu": "平胡"}
    for t in calls:
        h = list(hand13)
        h[t] += 1
        for tp in game._win_type_set(h, game.melds[me], t):
            name = type_names.get(tp, tp)
            v = game._hu_type_value(tp)
            if name != "平胡" and v > 0:
                premium.add(f"{name}(+{v:g})")
    line = "打出后即听牌：叫 " + "、".join(parts)
    if premium:
        line += "，可听 " + "、".join(sorted(premium))
    return line


def _ukeire_line(game: DushanGame, me: int, hand: list[int], n_melds: int) -> str:
    """未听牌时的进张分析：向听数 + 有效进张。"""
    base = shanten(hand, n_melds)
    if base < 0:
        return "已成胡"
    if base == 0:
        return _winning_line(game, me, hand)
    kinds, total, names = 0, 0, []
    for t in range(NUM_TILE_TYPES):
        if hand[t] >= 4:
            continue
        hand[t] += 1
        better = shanten(hand, n_melds) < base
        hand[t] -= 1
        if better:
            live = _live_count(game, me, t, hand)
            if live > 0:
                kinds += 1
                total += live
                if len(names) < 5:
                    names.append(tile_cn(t))
    if kinds == 0:
        return f"打出后向听 {base} 进，暂无有效进张（牌形僵）"
    more = f" 等{kinds}种" if kinds > 5 else ""
    return f"打出后向听 {base} 进，进张 {kinds} 种共 {total} 张：{'、'.join(names)}{more}"


def _danger_line(game: DushanGame, me: int, t: int) -> str:
    """生熟张与点炮敞口。"""
    vis = 4 - _live_count(game, me, t, [0] * NUM_TILE_TYPES)
    cfg = game.cfg
    cap = cfg.dian_pao + max(cfg.qing_yi_se, cfg.qi_dui, cfg.da_dui_zi,
                             cfg.long_qi_dui, cfg.dan_diao)
    if vis >= 3:
        return f"【风险】已现 {vis}/4 张，几乎安全（点炮最多赔 {cap:g} 分）"
    if vis >= 1:
        return f"【风险】已现 {vis}/4 张，点炮风险中等"
    return f"【风险】生张，点炮需赔方式分+牌型分（最高 {cap:g}）"


def _potential_line(game: DushanGame, me: int, hand: list[int]) -> str | None:
    """手牌结构提示（对子数、清一色潜质）。"""
    total = sum(hand)
    if total < 10:
        return None
    suit_n = [0, 0, 0]
    for t in range(NUM_TILE_TYPES):
        if hand[t]:
            suit_n[suit_of(t)] += hand[t]
    pairs = sum(1 for t in range(NUM_TILE_TYPES) if hand[t] == 2)
    tri = sum(1 for t in range(NUM_TILE_TYPES) if hand[t] >= 3)
    bits = []
    mx = max(suit_n)
    if mx >= 10 and total - mx <= 3:
        bits.append("清一色潜质（单花色已占大半）")
    if tri:
        bits.append(f"{tri} 个刻子坯")
    if pairs:
        bits.append(f"{pairs} 个对子")
    if not bits:
        return None
    return "【牌形】" + "，".join(bits)


# ---------------------------------------------------------------------------
# 主入口：给动作生成结构化解释
# ---------------------------------------------------------------------------
def explain_action_full(game: DushanGame, me: int, a: Action) -> dict:
    """返回 {tags: [...], lines: [str, ...], reason: str}。"""
    cfg: "DushanConfig" = game.cfg  # type: ignore[assignment]
    tags: list[str] = []
    lines: list[str] = []
    k = a.kind
    hand = list(game.hands[me])
    n_melds = len(game.melds[me])

    if k == HU:
        h = list(hand)
        wt = a.tile if a.tile is not None and a.tile >= 0 else (
            game.last_discard[1] if game.last_discard else -1)
        if wt >= 0:
            h[wt] += 1
        types = game._win_type_set(h, game.melds[me], wt if wt >= 0 else None)
        names = {"da_dui_zi": "大对子", "qi_dui": "小七对", "long_qi_dui": "龙七对",
                 "qing_yi_se": "清一色", "dan_diao": "单吊", "ping_hu": "平胡"}
        tv = sum(game._hu_type_value(tp) for tp in types)
        cn = "+".join(names.get(tp, tp) for tp in sorted(types) if tp != "ping_hu")
        if game.current == me and game.phase.name in ("DISCARD", "SELF_KONG"):
            how, val = ("杠上开花", cfg.gang_kai) if game._after_gang_draw \
                else ("自摸", cfg.zi_mo)
            payer = "其他每家各付"
        else:
            src = game.last_discard[0] if game.last_discard else None
            if game.repao_discard:
                how, val = "热炮", cfg.gang_pao
            else:
                how, val = "点炮", cfg.dian_pao
            payer = f"{REL_CN.get((src - me) % 4, '对方')}独付" if src is not None else "点炮者独付"
        tags += ["得分"]
        lines.append(f"【方式】{how}：{payer} {val:g} 分")
        if cn:
            lines.append(f"【牌型】{cn}（+{tv:g}）")
        lines.append(f"【鸡分】你已持有 {_hand_ji(game, me)} 只鸡（金鸡时计分×2）")
        return _pack(tags, lines)

    if k in (ANGANG, BUGANG, MINGGANG):
        pay = ("点杠者独付" if k == MINGGANG else "其他每家各付")
        tags += ["杠分"]
        lines.append(f"【杠分】+{cfg.kong_chickens:g}（{pay}），并获点胡通行证")
        lines.append("【后续】杠后再摸一张，有机会杠上开花（每家 +"
                     f"{cfg.gang_kai:g}）")
        if k == BUGANG:
            lines.append(f"【注意】补杠可能被抢杠——被抢则按抢杠赔 "
                         f"{cfg.dian_pao:g}+牌型，且自己鸡分全烧")
        tags.append("通行证")
        return _pack(tags, lines)

    if k == PENG:
        t = a.tile
        h2 = list(hand)
        h2[t] -= 2
        s0 = shanten(hand, n_melds)
        s1 = shanten(h2, n_melds + 1)
        tags += ["组牌"]
        if s1 < s0:
            lines.append(f"【进度】碰后向听 {s0}→{s1}，显著加速")
        elif s1 == 0:
            wl = _winning_line(game, me, h2)
            lines.append("【进度】碰后即听牌。" + wl if wl else "【进度】碰后即听牌")
        else:
            lines.append(f"【进度】碰后向听 {s0}→{s1}（副露固定，灵活度下降）")
        if t in game.ji_tiles:
            tags.append("鸡分")
            lines.append("【鸡分】碰鸡牌：对方赔溢价（冲锋+2/横+1），"
                         "且 3 只鸡面计入你手中鸡、对每家生效")
        lines.append("【注意】副露会暴露牌型信息；报叫玩家不能碰杠")
        return _pack(tags, lines)

    if k == PASS:
        src, t = game.last_discard if game.last_discard else (None, -1)
        who = REL_CN.get((src - me) % 4, "") if src is not None else ""
        tags.append("放弃")
        lines.append(f"【放弃】不响应 {who} 的 {tile_cn(t) if t >= 0 else ''}"
                     "，保持暗手不暴露")
        return _pack(tags, lines)

    if k != DISCARD:
        return _pack(tags, lines)

    # ---- 打牌（核心教学场景） ----
    t = a.tile
    h2 = list(hand)
    h2[t] -= 1
    wl = _winning_line(game, me, h2) or _ukeire_line(game, me, h2, n_melds)
    tags.append("听牌" if shanten(h2, n_melds) == 0 else "进张")
    lines.append("【进度】" + wl)

    if t in game.ji_tiles:
        tags.append("鸡分")
        name = _ji_name_of_discard(game, me, t)
        if name == "冲锋鸡":
            lines.append(f"【鸡分】首张打鸡＝冲锋鸡：终局对其他每家 +{JI_CHONGFENG:g}；"
                         f"若被碰/杠走，那张牌归对方（面 3/4 张），你另赔溢价 "
                         f"{JI_CHONGFENG - JI_YAO:g}")
        elif name == "横鸡":
            lines.append(f"【鸡分】打鸡＝横鸡：终局对其他每家 +{JI_HENG:g}；"
                         f"若被碰/杠走，那张牌归对方，你另赔溢价 {JI_HENG - JI_YAO:g}")
        else:
            n_keep = _hand_ji(game, me) - 1
            if n_keep > 0:
                lines.append(f"【鸡分】普通鸡打出仍算 1 只（归你）；若被人碰/杠走则转移"
                             f"给对方。手中仍持 {n_keep} 只鸡")
            else:
                lines.append("【鸡分】普通鸡打出仍算 1 只（归你）；若被人碰/杠走则转移给对方。"
                             "打完手中无鸡")
    else:
        n_keep = _hand_ji(game, me)
        if n_keep:
            lines.append(f"【鸡分】保留手中 {n_keep} 只鸡（每只终局对三家各 +1）")
        else:
            tags.append("鸡分")
            lines.append("【鸡分】手中无鸡，注意鸡牌种：" +
                         "、".join(tile_cn(x) for x in sorted(game.ji_tiles)))

    # 未听牌时打冲锋/横鸡会被包鸡
    if t in game.ji_tiles and not winning_tiles(h2, game.melds[me]):
        name = _ji_name_of_discard(game, me, t)
        if name in ("冲锋鸡", "横鸡"):
            lines.append(f"【警告】你未听牌，打出的{name}终局要包鸡（赔其他每家）")

    lines.append(_danger_line(game, me, t))
    pot = _potential_line(game, me, h2)
    if pot:
        lines.append(pot)
    return _pack(tags, lines)


def _pack(tags: list[str], lines: list[str]) -> dict:
    # 去重保序
    seen: set[str] = set()
    tags = [x for x in tags if not (x in seen or seen.add(x))]
    return {"tags": tags, "lines": lines, "reason": "；".join(
        l.split("】", 1)[1] if "】" in l else l for l in lines[:2])}


# ---------------------------------------------------------------------------
# 局面教学摘要 + 对比提示
# ---------------------------------------------------------------------------
def situation_summary(game: DushanGame, me: int) -> dict:
    """我方局面速览：向听/听牌/鸡/报叫/通行证。"""
    hand = list(game.hands[me])
    n_melds = len(game.melds[me])
    # 14 张手牌（本回合刚摸牌）时，向听数取"打出一张后"的最优值，与教学口径一致
    if sum(hand) % 3 == 2:
        best, best_t = 8, -1
        for t in range(NUM_TILE_TYPES):
            if hand[t] > 0:
                hand[t] -= 1
                v = shanten(hand, n_melds)
                if v < best:
                    best, best_t = v, t
                hand[t] += 1
        s = best
        # 听牌时用最优打张后的 13 张来列叫牌
        if s <= 0 and best_t >= 0:
            hand[best_t] -= 1
            calls = winning_tiles(hand, game.melds[me])
            hand[best_t] += 1
        else:
            calls = []
    else:
        s = shanten(hand, n_melds)
        calls = winning_tiles(hand, game.melds[me])
    ji_n = _hand_ji(game, me)
    has_kong = any(meld_is_kong(mt) for mt, _, _ in game.melds[me])
    passport = bool(calls) or has_kong or game.baojiao[me]
    # 通行证细化：听的是否非平胡
    premium_pass = False
    for t in calls:
        h = list(hand)
        h[t] += 1
        if game._win_type_set(h, game.melds[me], t) - {"ping_hu"}:
            premium_pass = True
            break
    d = {
        "shanten": int(s),
        "tenpai": bool(calls),
        "calls": [tile_cn(t) for t in calls[:6]],
        "ji_held": ji_n,
        "ji_tiles": [tile_cn(x) for x in sorted(game.ji_tiles)],
        "baojiao": bool(game.baojiao[me]),
        "has_kong": has_kong,
        "passport": passport,
        "passport_by_premium": premium_pass,
    }
    if calls:
        d["note"] = "已听牌，可安心打鸡收鸡分" if ji_n else "已听牌"
    elif game.baojiao[me]:
        d["note"] = "报叫状态：已有点胡通行证，但不能再碰杠"
    elif has_kong:
        d["note"] = "有杠＝通行证，可打冲锋鸡不惧点胡限制"
    elif s <= 1:
        d["note"] = f"向听 {s} 进，先组牌；无通行证时避免乱打鸡被责任鸡"
    else:
        d["note"] = f"向听 {s} 进，距离听牌尚远，优先保留搭子与鸡牌"
    return d


def compare_tip(game: DushanGame, me: int, items: list[dict]) -> str:
    """首选 vs 次选的对比教学点。"""
    if len(items) < 2:
        return ""
    top, second = items[0], items[1]
    t1 = top.get("kind")
    t2 = second.get("kind")
    if t1 == DISCARD and t2 == DISCARD:
        a, b = top["tile"], second["tile"]
        ja, jb = a in game.ji_tiles, b in game.ji_tiles
        if ja and not jb:
            return ("模型弃鸡保鸡？不对——这里模型认为打这张鸡更优："
                    "注意打出的首张鸡＝冲锋鸡有收入，且该牌安全度更高")
        if jb and not ja:
            return ("模型弃安全牌保鸡牌：鸡牌每只终局对三家各 +1，"
                    "收益常常大于一点生张风险")
        return ""
    if t1 == PASS and t2 == PENG:
        return "模型选择不碰：暗手灵活度与隐蔽性有时比快一步组牌更值钱"
    if t1 == PENG and t2 == PASS:
        return "模型选择碰：加速组牌（或碰鸡收溢价）压过了暴露信息的代价"
    if t1 == HU:
        return "能胡就胡：独山规则下平胡也有稳定鸡分，落袋为安"
    return ""
