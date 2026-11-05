# -*- coding: utf-8 -*-
"""公开信息推荐理由生成器（``zhuoji.reason``）。

2026-11-04 用户口径：
  「优化搜索 agent 的推荐出牌理由，确保没有使用全局信息来计算推荐和编写
   推荐理由，重点突出为什么应该出这张牌/采取这个操作（与其他牌或者操作的
   对比），包括完整理由和一句话推荐。」

物理隔离
--------
本模块的**唯一**输入是 :class:`zhuoji.public_view.PublicView`（外加 Q 值与
动作表）。PublicView 里没有别家手牌、没有真实牌墙内容、没有任何隐藏诊断——
所以理由层在数据结构上就无从访问全局信息，不是靠「自律不读」。

产出
----
* ``build_items(view, legal, q, k)`` → 每个候选的完整理由（``lines`` 多行、
  ``reason`` 摘要、``tags``、以及给一句话推荐用的 ``cmp``）；
* ``one_line(view, items)`` → 一句话推荐（教练面板底部 💡 与简要模式）。

两种文字都**围绕「对比」**组织：首选回答「为什么是它、比次优强在哪」，
其余候选回答「为什么落后、比首选差在哪」。所有「为什么」都落到公开指标
（打出后的向听 / 进张 / 听牌 / 已现张数 / 鸡分）与 Q 差值，不臆测别家手牌、
不写「且这张更安全」这类无依据断言。
"""
from __future__ import annotations

import numpy as np

from .public_view import PublicView
from .rules import ANGANG, BUGANG, DISCARD, HU, MINGGANG, PASS, PENG
from .shanten import shanten
from .tiles import NUM_TILE_TYPES, tile_cn

# 前端 TAG_CLS 会着色的标签优先用这几个；其余标签走默认样式，也能正常渲染。
TAG_TOP = "首选"
TAG_CMP = "对比"
TAG_ALT = "备选"


# ---------------------------------------------------------------------------
# 公开指标小工具（全部只读 view）
# ---------------------------------------------------------------------------
def _tile_cn(t: int) -> str:
    return tile_cn(t) if 0 <= t < NUM_TILE_TYPES else "?"


def _progress(view: PublicView, t: int) -> str:
    """打出 t 后自己牌的进度（向听 / 进张 / 听牌）。"""
    s = view.shanten_after(t)
    calls = view.ting_after(t)
    if calls:
        return f"打出后即听牌：叫 {view.calls_cn(t)}"
    kinds, total = view.ukeire_after(t)
    if total:
        return f"打出后向听 {s} 进，有效进张 {kinds} 种共 {total} 张"
    return f"打出后向听 {s} 进，暂无有效进张（牌形偏僵）"


def _ji_line(view: PublicView, t: int) -> str | None:
    """打出 t 这张鸡牌的公开计分说明（非鸡牌返回 None）。"""
    if t not in view.ji_tiles:
        return None
    mult = view.ji_mult(t)
    gold = f"（金鸡计分 ×{mult:g}）" if mult > 1 else ""
    name = view.ji_name(t)
    if name == "冲锋鸡":
        val = view.score.get("ji_chongfeng", 3.0)
        return (f"【鸡分】{_tile_cn(t)}是冲锋鸡{gold}：打出即记，"
                f"终局对其他每家 +{val:g} 分档；但未听牌打出要包鸡")
    if name == "横鸡":
        val = view.score.get("ji_heng", 2.0)
        return (f"【鸡分】{_tile_cn(t)}是横鸡{gold}：打出即记，"
                f"终局对其他每家 +{val:g} 分档；但未听牌打出要包鸡")
    left = view.hand_ji_count() - 1
    tail = f"手中还剩 {left} 只鸡" if left > 0 else "打完手中无鸡"
    return f"【鸡分】{_tile_cn(t)}是普通鸡{gold}：打出仍算 1 只归你，{tail}"


def _action_brief(view: PublicView, a) -> str:
    """非出牌动作的一句话公开说明（用于动作级对比）。"""
    k = a.kind
    t = a.tile
    if k == HU:
        return "直接胡牌、立即结算得分，落袋为安"
    if k == PASS:
        return "过牌保持暗手不暴露、维持牌型灵活"
    if k == PENG:
        h2 = list(view.hand)
        if 0 <= t < NUM_TILE_TYPES and h2[t] >= 2:
            h2[t] -= 2
        s0 = shanten(list(view.hand), view.n_melds)
        s1 = shanten(h2, view.n_melds + 1)
        return f"碰后向听 {s0}→{s1}，副露固定、会暴露牌型"
    if k in (ANGANG, BUGANG, MINGGANG):
        return f"杠分 +{view.score.get('kong_chickens', 2):g} 并获点胡通行证，杠后再摸一张"
    if k == DISCARD:
        return _progress(view, t)
    return ""


def _diff(view: PublicView, ta: int, tb: int) -> tuple[list[str], list[str]]:
    """「打 ta 相对打 tb」在公开指标上的差异 → (优势句, 劣势句)。

    只比较四家可见 + 自己手牌能推出的量：打出后向听 / 有效进张 / 听牌 /
    已现张数（点炮敞口）/ 鸡分。绝不触碰别家手牌或真实牌墙。
    """
    pros: list[str] = []
    cons: list[str] = []
    ca, cb = _tile_cn(ta), _tile_cn(tb)
    sa, sb = view.shanten_after(ta), view.shanten_after(tb)
    tinga, tingb = view.ting_after(ta), view.ting_after(tb)
    ka, na = view.ukeire_after(ta)
    kb, nb = view.ukeire_after(tb)
    va, vb = view.visible(ta), view.visible(tb)
    ja, jb = ta in view.ji_tiles, tb in view.ji_tiles

    # 听牌 / 向听
    if tinga and not tingb:
        pros.append(f"打{ca}即听牌（叫 {view.calls_cn(ta)}），打{cb}仍是 {sb} 进")
    elif tingb and not tinga:
        cons.append(f"打{cb}才听牌，打{ca}要退回 {sa} 进")
    elif sa < sb:
        pros.append(f"向听更近（打{ca} {sa} 进 vs 打{cb} {sb} 进）")
    elif sa > sb:
        cons.append(f"向听更远（打{ca} {sa} 进 vs 打{cb} {sb} 进）")

    # 有效进张（同向听、都未听时才有可比性）
    if sa == sb and not (tinga or tingb):
        if na > nb:
            pros.append(f"有效进张更多（{na} 张/{ka} 种 vs 打{cb} {nb} 张/{kb} 种）")
        elif na < nb:
            cons.append(f"有效进张更少（{na} 张 vs 打{cb} {nb} 张）")

    # 点炮敞口（只是已现张数，不臆测谁听牌）
    if va > vb:
        pros.append(f"更安全（{ca}已现 {va}/4 张，{cb} {vb}/4 张）")
    elif vb > va:
        cons.append(f"点炮敞口更大（{ca}已现 {va}/4 张，{cb} {vb}/4 张）")

    # 鸡分不塞进 pro/con：打鸡收冲锋分 / 留鸡收手中分谁更划算由 Q 与
    # 各自【鸡分】行决定，硬判方向会在首选与备选之间自相矛盾（见 _ji_note）。
    _ = (ja, jb)
    return pros, cons


def _ji_note(view: PublicView, t: int, other: int) -> str | None:
    """中性的「鸡牌差异」说明，且与 (t, other) 顺序无关，保证各候选措辞一致。"""
    jt, jo = t in view.ji_tiles, other in view.ji_tiles
    if jt == jo:
        return None
    ji = t if jt else other
    plain = other if jt else t
    nm = view.ji_name(ji) or "普通鸡"
    return (f"{_tile_cn(ji)}是{nm}、{_tile_cn(plain)}不是鸡牌，"
            "两者鸡分归属不同（详见各自【鸡分】）")


def _ji_short(view: PublicView, t: int, other: int) -> str | None:
    """给一句话推荐用的简短鸡分差异（中性）。"""
    jt, jo = t in view.ji_tiles, other in view.ji_tiles
    if jt == jo:
        return None
    ji = t if jt else other
    nm = view.ji_name(ji) or "普通鸡"
    return f"{_tile_cn(ji)}是{nm}（鸡分归属不同）"


# ---------------------------------------------------------------------------
# 主入口：完整理由
# ---------------------------------------------------------------------------
def build_items(view: PublicView, legal, q, k: int = 4) -> list[dict]:
    """把 Q 排序翻成「对比型」完整理由；返回 items（含 action/q/reason/lines/tags/cmp）。"""
    q = np.asarray(q, dtype=float)
    n = len(legal)
    if n == 0:
        return []
    if view is None:
        # 公开视图不可用时降级为「纯 Q 排名」：只讲收益差，不碰任何牌面指标，
        # 更不会去读隐藏信息。宁可少说，也不越界。
        return _q_only_items(legal, q, k)
    order = np.argsort(-q)[:max(1, min(int(k), n))]
    items: list[dict] = []
    for rank_i, idx in enumerate(order):
        i = int(idx)
        a = legal[i]
        qv = float(q[i])
        kind = a.kind
        tile = int(a.tile) if a.tile is not None else -1
        label = a.label()
        lines: list[str] = []
        tags: list[str] = []
        cmp: dict = {}

        # 参照对象：首选拿次优比，其余拿首选比
        if rank_i == 0:
            ref_i = int(order[1]) if len(order) > 1 else None
        else:
            ref_i = int(order[0])
        ref = legal[ref_i] if ref_i is not None else None
        ref_label = ref.label() if ref is not None else ""
        ref_q = float(q[ref_i]) if ref is not None else qv

        both_discard = (kind == DISCARD and ref is not None
                        and ref.kind == DISCARD and tile >= 0
                        and int(ref.tile) >= 0)

        # ---- ① 结论：名次 + 与参照的 Q 差（收益） --------------------------
        if ref is None:
            lines.append(f"【唯一选择】{label}，期望 {qv:+.2f} 分")
            tags.append(TAG_TOP)
        elif rank_i == 0:
            lines.append(f"【对比·首选】{label} 期望 {qv:+.2f} 分，"
                         f"领先次优「{ref_label}」{qv - ref_q:+.2f} 分")
            tags += [TAG_TOP, TAG_CMP]
            cmp = {"dq": qv - ref_q, "ref": ref_label, "ref_q": ref_q,
                   "kind": kind}
        else:
            lines.append(f"【对比·第{rank_i + 1}选】{label} 期望 {qv:+.2f} 分，"
                         f"比首选「{ref_label}」落后 {ref_q - qv:.2f} 分")
            tags += [TAG_ALT, TAG_CMP]
            cmp = {"dq": qv - ref_q, "ref": ref_label, "ref_q": ref_q,
                   "kind": kind}

        # ---- ② 为什么：与参照的公开指标对比（核心） ------------------------
        if both_discard:
            rt = int(ref.tile)
            if rank_i == 0:
                pros, cons = _diff(view, tile, rt)
                if pros:
                    lines.append("【为什么是它】" + "；".join(pros))
                    if cons:
                        lines.append("【代价/取舍】" + "；".join(cons))
                elif cons:
                    # 牌面反而是次优占优：如实说明，别把「推演期望」伪装成牌面优势
                    lines.append(
                        f"【为什么是它】单看公开牌面，「{ref_label}」在"
                        f"（{'；'.join(cons)}）上更占优；本手仍居首，"
                        "靠的是公平推演给出的更高终局期望")
                else:
                    # 两张牌公开指标打平时也要回答「为什么」，不能空着
                    lines.append(
                        f"【为什么是它】与「{ref_label}」在公开牌面"
                        "（向听/进张/已现张数/鸡分）上几乎无差，"
                        "领先来自公平推演的终局期望，而非牌面可见的优势")
                cmp["pros"] = pros
                cmp["cons"] = cons
                cmp["ref_tile"] = rt
            else:
                # 以「本牌相对首选」取**劣势句**，读起来才是「它为什么落后」；
                # 若本牌在公开指标上并不差，就如实说差距主要来自推演期望分。
                _pros, cons = _diff(view, tile, rt)
                if cons:
                    lines.append("【为什么落后】" + "；".join(cons))
                else:
                    lines.append(f"【为什么落后】单看公开牌面与「{ref_label}」接近，"
                                 "差距主要来自推演期望分")
        elif ref is not None:
            ba = _action_brief(view, a) or label
            br = _action_brief(view, ref) or ref_label
            if rank_i == 0:
                lines.append(f"【为什么是它】{ba}；而「{ref_label}」{br}")
            else:
                lines.append(f"【为什么落后】「{ref_label}」{br}；本手{ba}")
            cmp["brief"] = ba

        # ---- ③ 这张牌 / 这个动作自身 --------------------------------------
        if kind == DISCARD and tile >= 0:
            s = view.shanten_after(tile)
            tags.append("听牌" if view.ting_after(tile) else "进张")
            lines.append("【这张牌】" + _progress(view, tile))
            # 鸡牌差异（中性、与比较方向无关，首选与各选措辞一致）
            if ref is not None and ref.kind == DISCARD and int(ref.tile) >= 0:
                note = _ji_note(view, tile, int(ref.tile))
                if note:
                    lines.append("【鸡分对比】" + note)
            # 风险
            danger = view.danger(tile)
            lines.append("【风险】" + danger)
            if view.visible(tile) == 0:
                tags.append("风险")
            # 鸡分
            ji = _ji_line(view, tile)
            if ji:
                lines.append(ji)
                tags.append("鸡分")
        elif kind != DISCARD:
            b = _action_brief(view, a)
            if b:
                lines.append("【这一手】" + b)
            if kind == HU:
                tags.append("得分")

        # 摘要（brief 模式 / reason 兜底）
        reason = "；".join(l.split("】", 1)[-1] if "】" in l else l
                          for l in lines[:2])
        items.append({
            "action": a, "q": qv, "reason": reason, "lines": lines,
            "tags": _dedup(tags), "cmp": cmp,
        })
    return items


def _q_only_items(legal, q, k: int = 4) -> list[dict]:
    """公开视图缺失时的安全降级：只按 Q 排名讲收益差，不引用任何牌面/隐藏信息。"""
    order = np.argsort(-q)[:max(1, min(int(k), len(legal)))]
    items = []
    for rank_i, idx in enumerate(order):
        i = int(idx)
        a, qv = legal[i], float(q[i])
        if rank_i == 0 and len(order) > 1:
            j = int(order[1])
            line = (f"【对比·首选】{a.label()} 期望 {qv:+.2f} 分，"
                    f"领先次优「{legal[j].label()}」{qv - float(q[j]):+.2f} 分")
        elif len(order) > 1:
            j0 = int(order[0])
            line = (f"【对比·第{rank_i + 1}选】{a.label()} 期望 {qv:+.2f} 分，"
                    f"比首选落后 {float(q[j0]) - qv:.2f} 分")
        else:
            line = f"【唯一选择】{a.label()}，期望 {qv:+.2f} 分"
        items.append({"action": a, "q": qv, "reason": line,
                      "lines": [line], "tags": [TAG_TOP if rank_i == 0 else TAG_ALT,
                                                TAG_CMP],
                      "cmp": {"dq": 0.0, "ref": "", "kind": a.kind}})
    return items


def _dedup(seq: list[str]) -> list[str]:
    seen: set[str] = set()
    return [x for x in seq if not (x in seen or seen.add(x))]


# ---------------------------------------------------------------------------
# 一句话推荐（教练面板底部 / 简要模式）
# ---------------------------------------------------------------------------
def one_line(view: PublicView, items: list[dict]) -> str:
    """从公开信息 + Q 差值压成一句话，突出「为什么是它（对比次选）」。

    ``items`` 为最终送给前端的候选表（含 ``kind``/``tile``/``label``，
    搜索档还带 ``q``/``cmp``；模型档只有 ``prob``）。不编造分数：没有 Q
    时只说名次，不说「高 X 分」。
    """
    if len(items) < 2:
        return ""
    top, second = items[0], items[1]
    ka, kb = top.get("kind"), second.get("kind")
    la, lb = top.get("label", ""), second.get("label", "")
    qa, qb = top.get("q"), second.get("q")
    gap = ""
    if isinstance(qa, (int, float)) and isinstance(qb, (int, float)):
        gap = f"（期望高 {qa - qb:+.2f} 分）"

    if view is None:  # 公开视图缺失：只报排名，不编造牌面理由
        return f"建议{la}{gap}，优于「{lb}」。"

    # 出牌 vs 出牌：用公开指标对比
    if ka == DISCARD and kb == DISCARD:
        ta, tb = int(top.get("tile", -1)), int(second.get("tile", -1))
        if ta < 0 or tb < 0:
            return ""
        pros, cons = _diff(view, ta, tb)
        has_q = bool(gap)   # 搜索档有 Q 差值；模型档只有 prob，不谎报分数
        if pros:
            bits = ["；".join(pros[:2])]
            if cons:
                bits.append(f"代价是{cons[0]}")
        elif cons:
            bits = [f"牌面不占优：{cons[0]}，但"
                    f"{'推演期望更高' if has_q else '仍在推荐排序中领先'}"]
        else:
            bits = [f"公开牌面与{lb}接近，"
                    f"{'胜在推演期望' if has_q else '按推荐指数排序'}"]
        jn = _ji_short(view, ta, tb)
        if jn:
            bits.append(jn)
        return f"建议{la}比{lb}更优{gap}：" + "；".join(bits) + "。"

    # 动作级对比
    if ka == HU:
        return f"建议{la}{gap}：能胡就胡、落袋为安，别为做大牌错过。"
    if ka == PENG and kb == PASS:
        return f"建议{la}{gap}：碰后向听加速，代价是副露固定、暴露牌型。"
    if ka == PASS and kb == PENG:
        return f"建议{la}{gap}：不碰保持暗手灵活，不暴露牌型。"
    ba = _action_brief(view, _PseudoAction(ka, int(top.get("tile", -1)))) or la
    return f"建议{la}{gap}：{ba}。"


class _PseudoAction:
    """给 one_line 复用 _action_brief 用的轻量动作壳（只有 kind/tile）。"""
    __slots__ = ("kind", "tile")

    def __init__(self, kind, tile):
        self.kind = kind
        self.tile = tile
