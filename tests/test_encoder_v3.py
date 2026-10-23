"""v3 编码器的契约测试（P3）。

守四条线：

1. **维度表一致性**：``dims_for_version(3)`` 必须等于 ``encoder_v3`` 里的常量。
   历史上出现过"标 v2 实为 15 通道"的数据事故，所以维度必须有单一事实来源并断言。
2. **严格超集**：``v3[:31] == v2``、``v3[:98] == v2``、``v2[:15] == v1`` 逐位相同。
   这条性质是「纯容量 vs 容量+新特征」单变量消融的前提。
3. **公开信息红线**：篡改任意对手的手牌**不得**改变编码（不得偷看别人牌）。
4. **包掩码语义**：关掉某包只把那一段置零，其余逐位不变；七包全关时前 15 通道
   与 v1 逐位相同（即"退化成纯 v1 输入"）。

用法：``python tests/test_encoder_v3.py``（也兼容 pytest）。
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from zhuoji.dushan import DushanConfig, DushanGame  # noqa: E402
from zhuoji.encoder import dims_for_version, encode  # noqa: E402
from zhuoji.encoder_v3 import (ALL_PACKS_V3, GLOBAL_DIM_V3, NUM_CHANNELS_V3,  # noqa: E402
                               PACK_GLOB_SLICES_V3, PACK_TILE_SLICES_V3,
                               parse_packs_v3)

FAIL = 0


def check(name: str, ok: bool, detail: str = "") -> None:
    global FAIL
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}{(' — ' + detail) if detail else ''}")
    if not ok:
        FAIL += 1


def random_states(n_games: int = 3, steps: int = 40):
    """跑若干局随机对局，产出 (game, 每个座位) 的状态快照。"""
    rng = np.random.default_rng(20261012)
    out = []
    for g in range(n_games):
        game = DushanGame(DushanConfig(), seed=int(1000 + g))
        for _ in range(steps):
            if game.phase.value == "over":
                break
            acts = game.legal_actions()
            if not acts:
                break
            game.step(acts[int(rng.integers(0, len(acts)))])
        out.append(game)
    return out


# ---------------------------------------------------------------------------
def test_dims() -> None:
    print("== 维度表 ==")
    check("dims_for_version(3) == encoder_v3 常量",
          dims_for_version(3) == (NUM_CHANNELS_V3, GLOBAL_DIM_V3),
          f"{dims_for_version(3)} vs {(NUM_CHANNELS_V3, GLOBAL_DIM_V3)}")
    check("v3 = 63 通道 / 130 全局维",
          (NUM_CHANNELS_V3, GLOBAL_DIM_V3) == (63, 130))
    check("包切片不重叠且覆盖 31–62",
          sorted(list(PACK_TILE_SLICES_V3.values())) == [(31, 43), (43, 53), (53, 63)])
    check("glob 包切片覆盖 98–129", PACK_GLOB_SLICES_V3["g"] == (98, 130))
    check("parse_packs_v3(None) 全开且顺序固定",
          parse_packs_v3(None) == ALL_PACKS_V3 == ("a", "b", "c", "d", "e", "f", "g"))
    check("parse_packs_v3('de') 只留 d/e", parse_packs_v3("de") == ("d", "e"))
    check("parse_packs_v3('') 全关", parse_packs_v3("") == ())


def test_shapes_and_superset() -> None:
    print("== 形状 / 超集 / 数值健康 ==")
    games = random_states()
    ok_shape = ok_sup21 = ok_sup32 = ok_finite = ok_range = ok_no_nan = True
    worst = 0.0
    for game in games:
        for seat in range(4):
            o1 = encode(game, seat, 1)
            o2 = encode(game, seat, 2)
            o3 = encode(game, seat, 3)
            ok_shape &= (o3.tiles.shape == (63, 27) and o3.glob.shape == (130,))
            ok_sup21 &= np.array_equal(o2.tiles[:15], o1.tiles) and \
                np.array_equal(o2.glob[:70], o1.glob)
            ok_sup32 &= np.array_equal(o3.tiles[:31], o2.tiles) and \
                np.array_equal(o3.glob[:98], o2.glob)
            ok_no_nan &= not (np.isnan(o3.tiles).any() or np.isnan(o3.glob).any())
            ok_finite &= bool(np.isfinite(o3.tiles).all() and np.isfinite(o3.glob).all())
            ok_range &= bool(o3.tiles.min() >= -1e-6 and o3.tiles.max() <= 1.0 + 1e-6
                             and o3.glob.min() >= -1e-6 and o3.glob.max() <= 1.0 + 1e-6)
            worst = max(worst, float(np.abs(o3.tiles).max()))
    check("v3 形状 (63,27)/(130,)，dtype float32",
          ok_shape, f"max|tiles|={worst:.3f}")
    check("v2 ⊃ v1（逐位）", ok_sup21)
    check("v3 ⊃ v2（逐位）", ok_sup32)
    check("无 NaN / 全有限", ok_no_nan and ok_finite)
    check("取值落在 [0,1]", ok_range)


def test_determinism() -> None:
    print("== 确定性 ==")
    game = random_states(1, 30)[0]
    a = encode(game, 1, 3)
    b = encode(game, 1, 3)
    check("同一局同座位两次编码逐位相同",
          np.array_equal(a.tiles, b.tiles) and np.array_equal(a.glob, b.glob))


def test_no_hand_peeking() -> None:
    """红线：篡改对手手牌不得改变编码。"""
    print("== 公开信息红线：不得偷看对手手牌 ==")
    ok = True
    for game in random_states():
        for seat in range(4):
            before = encode(game, seat, 3)
            saved = [np.array(game.hands[p], copy=True) for p in range(4)]
            rng = np.random.default_rng(7 + seat)
            for p in range(4):
                if p == seat:
                    continue
                # 把对手的手牌改成任意其他分布（总数保持不变，避免触发形状相关分支）
                h = np.array(game.hands[p], copy=True)
                flat = h.reshape(-1)
                rng.shuffle(flat)
                game.hands[p] = flat.reshape(h.shape)
            after = encode(game, seat, 3)
            ok &= np.array_equal(before.tiles, after.tiles) and \
                np.array_equal(before.glob, after.glob)
            for p in range(4):
                game.hands[p] = saved[p]
    check("篡改三家对手手牌 → 编码逐位不变", ok)


def test_pack_mask() -> None:
    print("== 包掩码（单变量消融） ==")
    game = random_states(1, 35)[0]
    full = encode(game, 0, 3)
    ok_zero = True
    ok_rest = True
    for pack, (lo, hi) in PACK_TILE_SLICES_V3.items():
        o = encode(game, 0, 3, packs="".join(p for p in ALL_PACKS_V3 if p != pack))
        ok_zero &= np.all(o.tiles[lo:hi] == 0.0)
        keep = np.ones(63, dtype=bool)
        keep[lo:hi] = False
        ok_rest &= np.array_equal(o.tiles[keep], full.tiles[keep])
    for pack, (lo, hi) in PACK_GLOB_SLICES_V3.items():
        o = encode(game, 0, 3, packs="".join(p for p in ALL_PACKS_V3 if p != pack))
        ok_zero &= np.all(o.glob[lo:hi] == 0.0)
        keep = np.ones(130, dtype=bool)
        keep[lo:hi] = False
        ok_rest &= np.array_equal(o.glob[keep], full.glob[keep])
    check("关掉某包 → 该段恰好为 0", ok_zero)
    check("关掉某包 → 其余各段逐位不变", ok_rest)

    none_o = encode(game, 0, 3, packs="")
    v1 = encode(game, 0, 1)
    check("七包全关 → 前 15 通道/前 70 维与 v1 逐位相同",
          np.array_equal(none_o.tiles[:15], v1.tiles)
          and np.array_equal(none_o.glob[:70], v1.glob))
    check("七包全关 → 其余段全 0",
          np.all(none_o.tiles[15:] == 0.0) and np.all(none_o.glob[70:] == 0.0))

    # a/b/c 必须仍然透传给 v2（否则 v2 消融在 v3 下静默失效）
    only_abc = encode(game, 0, 3, packs="abc")
    check("packs='abc' → v3 新增段全 0、v2 段与 v2 全开一致",
          np.all(only_abc.tiles[31:] == 0.0) and np.all(only_abc.glob[98:] == 0.0)
          and np.array_equal(only_abc.tiles[:31], encode(game, 0, 2).tiles))


def test_mask_and_head() -> None:
    print("== 头与掩码 ==")
    ok = True
    for game in random_states():
        for seat in range(4):
            o = encode(game, seat, 3)
            from zhuoji.encoder import HEAD_DIMS
            ok &= o.head in HEAD_DIMS and o.mask.shape == (HEAD_DIMS[o.head],)
    check("head 合法且 mask 形状匹配", ok)


# ---------------------------------------------------------------------------
def main() -> int:
    test_dims()
    test_shapes_and_superset()
    test_determinism()
    test_no_hand_peeking()
    test_pack_mask()
    test_mask_and_head()
    print(f"\n{'全部通过' if not FAIL else str(FAIL) + ' 项失败'}")
    return 1 if FAIL else 0


if __name__ == "__main__":
    raise SystemExit(main())
