"""模型选点与对位评分口径的单元测试。

背景（为什么要专门测这两个函数）
--------------------------------
训练内评测（对固定三位启发式教师、默认 160 局）的 95% 置信区间约 ±1.9 分，
而真实的代际提升只有 ~0.3 分。用这个指标做 ``argmax`` 选点会系统性选错
（winner's curse）。所以选点逻辑（``selfplay.choose_final``）和可信的
对位评分口径（``score_checkpoints.summarize``）都必须有回归测试，
免得哪天又悄悄退回"看评测分挑最好的"。

直接运行::

    python tests/test_selection.py
"""
from __future__ import annotations

import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from score_checkpoints import summarize  # noqa: E402
from selfplay import choose_final  # noqa: E402


def test_choose_final() -> None:
    log = [(10, 6.25, 0.95), (20, 5.69, 0.97), (30, 5.66, 0.96),
           (40, 7.23, 0.97), (50, 5.49, 0.97)]

    which, note = choose_final(log, "max")
    assert which == "best" and "select=max" in note, (which, note)
    print("ok max 保持旧行为（保留历史最佳，不覆盖）")

    which, note = choose_final(log, "last")
    assert which == "last" and "select=last" in note, (which, note)
    print("ok last 采用最后一轮")

    # 最近 3 次（7.23 / 5.49 里含 3 次）均值 = (7.23 + 5.49)/2 ? 不是——取最近 3 个
    # 最近 3 个 = 5.66, 7.23, 5.49 -> 均值 6.13；中位数 = 5.69
    # 2×SE = 2×√(0.96²+0.97²+0.97²)/3 = 2×1.66/3 = 1.11
    # 6.13 + 1.11 = 7.24 > 5.69 → 不算退化 → last
    which, note = choose_final(log, "guard")
    assert which == "last", (which, note)
    print("ok guard 在无实质退化时采用最后一轮")

    # 构造实质退化：最后 3 次都远低于全程中位数
    degraded = [(10, 8.0, 0.5), (20, 8.1, 0.5), (30, 8.0, 0.5),
                (40, 8.2, 0.5), (50, 2.0, 0.5), (60, 2.1, 0.5), (70, 2.0, 0.5)]
    which, note = choose_final(degraded, "guard")
    assert which == "best", (which, note)
    assert "退化" in note, note
    print("ok guard 在实质退化时回退历史最佳")

    which, note = choose_final([], "guard")
    assert which == "best", (which, note)
    print("ok 没有评测记录时退回 best（不覆盖已有存档）")


def test_summarize() -> None:
    rng = random.Random(7)
    n = 400
    base = [rng.gauss(0.0, 5.0) for _ in range(n)]        # 牌运噪声很大
    per_deal = {
        "anchor": base,
        "twin": [x + 0.5 for x in base],                  # 完全相关，只差 0.5
        "better": [x + 1.0 + rng.gauss(0, 0.5) for x in base],
        "worse": [x - 1.0 + rng.gauss(0, 0.5) for x in base],
    }
    rows = summarize(per_deal, "anchor")
    d = {r["ckpt"]: r for r in rows}

    # 排序：better > twin > anchor > worse
    assert [r["ckpt"] for r in rows] == ["better", "twin", "anchor", "worse"], rows
    print("ok summarize 按相对锚点的配对净分差降序排名")

    assert d["anchor"]["vs_anchor"] == 0.0 and d["anchor"]["verdict"] == "无显著差异"
    print("ok 锚点自身净分差为 0")

    assert abs(d["twin"]["vs_anchor"] - 0.5) < 1e-9, d["twin"]
    assert d["twin"]["vs_anchor_ci95"] == 0.0, d["twin"]
    assert d["twin"]["verdict"] == "显著优", d["twin"]
    print("ok 完全相关的配对：差值精确、CI 为 0（配对消掉了牌运噪声）")

    # 配对设计的意义：水平值的 CI 远大于配对差值的 CI
    assert d["twin"]["level_ci95"] > 0.5, d["twin"]
    assert d["twin"]["vs_anchor_ci95"] < d["twin"]["level_ci95"], d["twin"]
    print("ok 配对净分差的 CI 明显小于分值水平自身的 CI")

    assert d["better"]["verdict"] == "显著优" and d["worse"]["verdict"] == "显著劣"
    print("ok 1 分差距在 400 副下被判为显著")

    # 样本量不足时应当判不出显著——不能把噪声当结论。
    # 用实测量级：代际真实差距 ~0.3 分，逐副配对差的标准差约 7 分
    # （由 S11 vs S7 的 2400 副结果 +0.319 ± 0.288 反推：0.288/1.96×√2400 ≈ 7.2）。
    rng2 = random.Random(11)
    sd = 7.2
    tiny = [x + 0.3 + rng2.gauss(0, sd) for x in base]
    per2 = {"anchor": base, "tiny": tiny}
    few = {r["ckpt"]: r for r in summarize({k: v[:40] for k, v in per2.items()}, "anchor")}
    many = {r["ckpt"]: r for r in summarize(per2, "anchor")}
    assert few["tiny"]["verdict"] == "无显著差异", few["tiny"]
    print("ok 小样本（40 副）下 0.3 分的真实差距判不出显著")
    assert many["tiny"]["vs_anchor_ci95"] < few["tiny"]["vs_anchor_ci95"] * 0.5, \
        (few["tiny"], many["tiny"])
    print("ok CI 随样本量按 √n 收缩")

    # 需要多少副才能把 0.3 分的差距测到显著——这个量级解释了为什么
    # 训练内 160 局的评测不可能用来挑检查点。
    need = (1.96 * sd / 0.3) ** 2
    assert 1500 < need < 4000, need
    print(f"ok 测出 0.3 分差距约需 {need:.0f} 副牌（与实测 2400 副 CI ±0.288 相符）")


if __name__ == "__main__":
    test_choose_final()
    test_summarize()
    print("选点与对位评分口径测试全部通过 ✅")
