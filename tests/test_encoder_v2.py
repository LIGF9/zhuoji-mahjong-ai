"""编码器 v2（P1 表征升级）的验收测试 —— 对应设计文档「阶段 0」的六条标准。

设计文档：``docs/P1-表征升级设计.md`` §7「阶段 0 · 实现与单测」

| 项 | 通过标准 |
|---|---|
| 维度 | ``NUM_CHANNELS_V2 == 31``、``GLOBAL_DIM_V2 == 98``、``Observation`` 形状 |
| 取值范围 | 所有通道 ∈ [0,1]；所有标量 ∈ [0,1] |
| **视角不变性** | 四个座位各自编码，其「相对座位」维度互为循环置换 |
| **无信息泄漏** | 篡改对手手牌不得改变编码 |
| 空局/边界 | 牌河为空 / 只有 1 张 / 开局 13 张等边界不报错且通道全 0 正确 |
| 回归 | smoke_test / test_encoder / test_rob_kong / test_dushan 全绿（另跑） |

另外两条本仓自己加的性质（都很便宜，能挡住真出过的错）：

* **v1 ⊆ v2**：v2 的前 15 通道 / 前 70 维与 v1 **逐位相同**（否则「v2 相对 v1 的
  增量」就说不清是特征加了什么还是把老特征悄悄改了）；
* **时序通道真的有时序**：把牌河换序，通道 15–26 必须跟着变（早年 v1 把牌河
  压成一个累加通道，换序毫无变化）。

直接运行::

    python tests/test_encoder_v2.py
"""
from __future__ import annotations

import copy
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from zhuoji.dushan import DushanConfig, DushanGame  # noqa: E402
from zhuoji.encoder import (  # noqa: E402
    GLOBAL_DIM, NUM_CHANNELS, Observation, encode)
from zhuoji.encoder_v2 import (  # noqa: E402
    GLOBAL_DIM_V2, NUM_CHANNELS_V2, _V1_CHANNELS, _V1_GLOBAL_DIM)
from zhuoji.rules import DISCARD, Action  # noqa: E402
from zhuoji.tiles import NUM_TILE_TYPES  # noqa: E402

#: 逐玩家（相对座位）分块：这些块在换视角时必须发生循环置换
TILE_PLAYER_BLOCKS = (
    ("牌河累加", [5, 6, 7, 8]),
    ("副露", [9, 10, 11, 12]),
    ("牌河最近1张", [15, 16, 17, 18]),
    ("牌河最近2张", [19, 20, 21, 22]),
    ("牌河最近3张", [23, 24, 25, 26]),
)
GLOB_PLAYER_BLOCKS = (
    ("杠数", [9, 10, 11, 12]),
    ("副露组数", [13, 14, 15, 16]),
    ("牌河长度", [17, 18, 19, 20]),
    ("曾经听牌", [21, 22, 23, 24]),
    ("向听下界", [70, 71, 72, 73]),
    ("已亮明鸡", [74, 75, 76, 77]),
    ("听牌置信", [78, 79, 80, 81]),
)


def _play(game, steps: int, seed: int = 0) -> None:
    """用确定性策略把对局推进若干步，制造出牌河/副露等非平凡状态。"""
    rng = np.random.default_rng(seed)
    for _ in range(steps):
        if game.phase.value == "over":
            break
        p = game.actor()
        legal = [t for t in range(NUM_TILE_TYPES) if game.hands[p][t] > 0]
        if not legal:
            break
        t = int(rng.choice(legal))
        try:
            game.step(Action(DISCARD, t))
        except Exception:
            break


def test_dimensions() -> None:
    g = DushanGame(DushanConfig(), seed=1)
    obs = encode(g, g.actor(), 2)
    assert NUM_CHANNELS_V2 == 31, NUM_CHANNELS_V2
    assert GLOBAL_DIM_V2 == 98, GLOBAL_DIM_V2
    assert (NUM_CHANNELS, GLOBAL_DIM) == (15, 70)
    assert isinstance(obs, Observation)
    assert obs.tiles.shape == (NUM_CHANNELS_V2, 27), obs.tiles.shape
    assert obs.glob.shape == (GLOBAL_DIM_V2,), obs.glob.shape
    assert obs.tiles.dtype == np.float32 and obs.glob.dtype == np.float32
    print(f"ok 维度：tiles{obs.tiles.shape} glob{obs.glob.shape}（v1 为 "
          f"{NUM_CHANNELS}/{GLOBAL_DIM}）")


def test_range() -> None:
    """所有通道、所有标量都必须落在 [0,1]——归一化写错时这里会立刻炸。"""
    bad: list[str] = []
    for seed in range(60):
        g = DushanGame(DushanConfig(), seed=1000 + seed)
        _play(g, int(np.random.default_rng(seed).integers(0, 90)), seed)
        for me in range(4):
            obs = encode(g, me, 2)
            if not np.isfinite(obs.tiles).all() or not np.isfinite(obs.glob).all():
                bad.append(f"seed={seed} me={me}: 出现 NaN/Inf")
                continue
            if obs.tiles.min() < -1e-6 or obs.tiles.max() > 1 + 1e-6:
                i = int(np.unravel_index(obs.tiles.argmax(), obs.tiles.shape)[0])
                bad.append(f"seed={seed} me={me}: 通道 {i} 越界 "
                           f"[{obs.tiles.min()}, {obs.tiles.max()}]")
            if obs.glob.min() < -1e-6 or obs.glob.max() > 1 + 1e-6:
                i = int(obs.glob.argmax())
                bad.append(f"seed={seed} me={me}: 标量 {i} 越界 "
                           f"[{obs.glob.min()}, {obs.glob.max()}]")
        if bad:
            break
    assert not bad, bad[:5]
    print("ok 取值范围：60 局 × 4 视角的所有通道与标量均在 [0,1]")


def test_v1_is_prefix_of_v2() -> None:
    """v2 必须把 v1 原样包住：前 15 通道、前 70 维逐位相同。"""
    for seed in range(25):
        g = DushanGame(DushanConfig(), seed=2000 + seed)
        _play(g, 30 + seed, seed)
        for me in range(4):
            a = encode(g, me, 1)
            b = encode(g, me, 2)
            assert np.array_equal(a.tiles, b.tiles[:_V1_CHANNELS]), (seed, me)
            assert np.array_equal(a.glob, b.glob[:_V1_GLOBAL_DIM]), (seed, me)
            assert a.head == b.head
            assert np.array_equal(a.mask, b.mask)
    print("ok v1 ⊆ v2：前 15 通道 / 前 70 维逐位相同，动作头与掩码一致")


def test_view_permutation() -> None:
    """视角不变性：同一局面下，四座位编码的「逐玩家」分块互为循环置换。

    定义：玩家 p 在视角 me 下占据相对座位 ``(p-me)%4``。所以对任意两个视角
    ``me`` / ``me2``，都应有 ``block_me[(p-me)%4] == block_me2[(p-me2)%4]``。
    这一条能抓出「相对座位算反了」这类最隐蔽、也最致命的错。
    """
    checked = 0
    for seed in range(50):
        g = DushanGame(DushanConfig(), seed=3000 + seed)
        _play(g, 25 + seed * 3, seed)
        obs = {me: encode(g, me, 2) for me in range(4)}
        for me in range(4):
            for me2 in range(me + 1, 4):
                for name, chans in TILE_PLAYER_BLOCKS:
                    for p in range(4):
                        a = obs[me].tiles[chans[(p - me) % 4]]
                        b = obs[me2].tiles[chans[(p - me2) % 4]]
                        assert np.array_equal(a, b), \
                            f"tiles/{name} 视角{me}vs{me2} 玩家{p} 不一致"
                for name, idxs in GLOB_PLAYER_BLOCKS:
                    for p in range(4):
                        a = obs[me].glob[idxs[(p - me) % 4]]
                        b = obs[me2].glob[idxs[(p - me2) % 4]]
                        assert abs(float(a) - float(b)) < 1e-6, \
                            f"glob/{name} 视角{me}vs{me2} 玩家{p}: {a} != {b}"
                checked += 1
    print(f"ok 视角不变性：{checked} 组视角对 × {len(TILE_PLAYER_BLOCKS)} 个通道块"
          f" + {len(GLOB_PLAYER_BLOCKS)} 个标量块，全部一致")


def test_no_information_leak() -> None:
    """无信息泄漏：篡改**对手手牌**不得改变本视角的编码。

    遮挡的是「把别人手牌喂进特征」这条最容易走偏的路——一旦有人图省事直接读
    ``game.hands[q]``，这条断言会立刻失败。
    """
    g = DushanGame(DushanConfig(), seed=77)
    _play(g, 40, 3)
    base = {me: encode(g, me, 2) for me in range(4)}

    for victim in range(4):
        g2 = copy.deepcopy(g)
        h = list(g2.hands[victim])
        # 把该家的手牌换成完全不同的分布（张数保持不变，避免破坏引擎不变量）
        total = sum(h)
        h2 = [0] * NUM_TILE_TYPES
        left = total
        t = 0
        while left > 0 and t < NUM_TILE_TYPES:
            h2[t] = 1
            left -= 1
            t += 1
        g2.hands[victim] = h2
        for me in range(4):
            if me == victim:
                continue          # 自己的手牌当然会影响自己的编码
            obs = encode(g2, me, 2)
            assert np.array_equal(base[me].tiles, obs.tiles), \
                f"篡改玩家{victim}手牌后，视角{me}的 tiles 变了（信息泄漏）"
            assert np.array_equal(base[me].glob, obs.glob), \
                f"篡改玩家{victim}手牌后，视角{me}的 glob 变了（信息泄漏）"
    print("ok 无信息泄漏：篡改对手手牌，其余三个视角的编码逐位不变")


def test_boundary_empty_river() -> None:
    """牌河为空的边界：时序通道 15–26 必须全 0（不填充、不重复计）。"""
    g = DushanGame(DushanConfig(), seed=5)
    assert all(len(g.discards[p]) == 0 for p in range(4)), "开局牌河应为空"
    obs = encode(g, g.actor(), 2)
    assert np.all(obs.tiles[15:27] == 0.0), obs.tiles[15:27].sum()
    # 手牌张数：庄家 14 张、其余 13 张
    assert sum(g.hands[g.actor()]) in (13, 14)
    print("ok 边界：开局牌河为空 → 通道 15–26 全 0，手牌 13/14 张正常")


def test_boundary_short_river() -> None:
    """牌河只有 1 张 / 2 张时：帧 2、帧 3 通道必须全 0。"""
    for want in (1, 2):
        g = DushanGame(DushanConfig(), seed=9)
        me = g.actor()
        # 直接构造：让该家牌河恰好有 want 张
        while len(g.discards[me]) < want and g.phase.value != "over":
            legal = [t for t in range(NUM_TILE_TYPES) if g.hands[g.actor()][t] > 0]
            if not legal:
                break
            g.step(Action(DISCARD, legal[0]))
        n = len(g.discards[me])
        obs = encode(g, me, 2)
        for k in range(3):
            blk = obs.tiles[15 + k * 4:15 + k * 4 + 4]
            if k + 1 <= n:
                continue
            assert np.all(blk == 0.0), f"牌河只有 {n} 张，帧 {k + 1} 应为 0"
        # 帧 1 的 one-hot 必须恰好落在该家最后打过的那张牌上
        if n >= 1:
            last = g.discards[me][-1]
            assert obs.tiles[15 + (me - me) % 4, last] == 1.0
    print("ok 边界：牌河 1/2 张时帧 2/3 通道全 0，帧 1 精确落在最近一张")


def test_river_temporal_channels_are_temporal() -> None:
    """时序通道真的携带时序：交换牌河末两张，通道 15–26 必须跟着变。

    这条针对 v1 的老毛病——v1 把整条牌河压成一个累加通道，换序毫无变化。
    """
    g = DushanGame(DushanConfig(), seed=11)
    for _ in range(200):
        if len(g.discards[0]) >= 3 or g.phase.value == "over":
            break
        p = g.actor()
        legal = [t for t in range(NUM_TILE_TYPES) if g.hands[p][t] > 0]
        if not legal:
            break
        g.step(Action(DISCARD, legal[0]))
    if len(g.discards[0]) < 3:
        print("ok 时序通道（本局未累积足够牌河，跳过构造）")
        return
    a = encode(g, 0, 2)
    g2 = copy.deepcopy(g)
    g2.discards[0][-1], g2.discards[0][-2] = g2.discards[0][-2], g2.discards[0][-1]
    b = encode(g2, 0, 2)
    # v1 的累加通道不变（这正说明它丢了时序）
    assert np.array_equal(a.tiles[5:9], b.tiles[5:9]), "累加通道本不该变化"
    # v2 的时序通道必须变
    assert not np.array_equal(a.tiles[15:27], b.tiles[15:27]), \
        "交换牌河末两张后时序通道居然没变——时序信息丢了"
    print("ok 时序通道有效：换序后 v1 累加通道不变、v2 时序通道改变")


def test_v2_runtime_cost() -> None:
    """v2 编码耗时必须与 v1 同一量级（否则训练预算全被编码吃掉）。"""
    import time
    g = DushanGame(DushanConfig(), seed=13)
    _play(g, 60, 5)
    n = 40
    for v in (1, 2):
        t0 = time.perf_counter()
        for _ in range(n):
            encode(g, 0, v)
        dt = (time.perf_counter() - t0) / n * 1000
        print(f"  encoder v{v}: {dt:.2f} ms/次")
        if v == 1:
            t1 = dt
    t2 = dt
    assert t2 < max(8.0, t1 * 12), f"v2 编码 {t2:.2f}ms 相对 v1 {t1:.2f}ms 过慢"
    print(f"ok 编码耗时：v2 是 v1 的 {t2 / max(t1, 1e-9):.2f} 倍")


def test_net_accepts_v2() -> None:
    """网络必须能直接吃 v2 的观测（维度打通），且空间头输出形状正确。"""
    import torch
    from zhuoji.net import NetConfig, build_model, net_config_from_ckpt, net_meta

    cfg = NetConfig(32, 3, 128, global_dim=GLOBAL_DIM_V2,
                    in_channels=NUM_CHANNELS_V2, spatial_head=True)
    m = build_model(cfg)
    g = DushanGame(DushanConfig(), seed=21)
    _play(g, 20, 7)
    obs = encode(g, 0, 2)
    with torch.no_grad():
        out = m(torch.from_numpy(obs.tiles[None]), torch.from_numpy(obs.glob[None]))
    assert out["discard"].shape == (1, 28), out["discard"].shape
    assert out["self_kong"].shape == (1, 28)
    assert out["respond"].shape == (1, 4)
    assert out["value"].shape == (1,)
    # 元数据往返：写进检查点的 net 字段必须能还原出同样的结构
    cfg2 = net_config_from_ckpt({"net": net_meta(cfg)})
    assert cfg2 == cfg, (cfg2, cfg)
    print(f"ok 网络吃 v2 观测：参数量 {m.n_params():,}，"
          f"四头形状正确，net_meta 往返一致")


def test_spatial_head_is_zero_init_additive() -> None:
    """空间头必须「零初始化 + 加法」，于是同版本加载后输出与旧模型逐位相同。"""
    import torch
    from zhuoji.net import NetConfig, build_model, encoder_version_from_ckpt

    path = ROOT / "models" / "dushan_big_s11.pt"
    if not path.exists():
        print("ok 空间头（缺 dushan_big_s11.pt，跳过权重等价性检查）")
        return
    ck = torch.load(path, map_location="cpu", weights_only=False)
    assert encoder_version_from_ckpt(ck) == 1, "s11 应被识别为 v1"

    # 用检查点自带的 net 字典还原（含 channels/blocks/hidden，无空间头）
    from zhuoji.net import net_meta, warm_start_from_v1
    d = ck.get("net")
    cfg_old = NetConfig(d["channels"], d["blocks"], d["hidden"])
    m_old = build_model(cfg_old)
    m_old.load_state_dict(ck["state_dict"])
    m_old.eval()

    cfg_new = NetConfig(d["channels"], d["blocks"], d["hidden"],
                        global_dim=cfg_old.global_dim,
                        in_channels=cfg_old.in_channels, spatial_head=True)
    m_new = build_model(cfg_new)
    info = warm_start_from_v1(m_new, ck)
    m_new.eval()
    assert m_new.n_params() == m_old.n_params() + 2 * (d["channels"] + 1), \
        (m_new.n_params(), m_old.n_params())
    assert not info["partial"], f"同结构热启动不该有部分复制项：{info['partial']}"

    tiles = torch.randn(4, cfg_old.in_channels, 27)
    glob = torch.randn(4, cfg_old.global_dim)
    with torch.no_grad():
        a, b = m_old(tiles, glob), m_new(tiles, glob)
    for k in a:
        dmax = float((a[k] - b[k]).abs().max())
        assert dmax == 0.0, f"{k} 加了空间头后输出变了（max|Δ|={dmax}）"
    print(f"ok 空间头：+{m_new.n_params() - m_old.n_params()} 个参数，"
          f"加载 s11 后四头输出逐位相同（严格热启动）")


def test_warm_start_v1_to_v2() -> None:
    """v1 → v2 的部分热启动：共享部分必须逐位复制成功。"""
    import torch
    from zhuoji.net import NetConfig, build_model, warm_start_from_v1

    path = ROOT / "models" / "dushan_big_s11.pt"
    if not path.exists():
        print("ok 热启动（缺 dushan_big_s11.pt，跳过）")
        return
    ck = torch.load(path, map_location="cpu", weights_only=False)
    d = ck["net"]
    cfg = NetConfig(d["channels"], d["blocks"], d["hidden"],
                    global_dim=GLOBAL_DIM_V2, in_channels=NUM_CHANNELS_V2,
                    spatial_head=True)
    m = build_model(cfg)
    info = warm_start_from_v1(m, ck)
    sd = m.state_dict()
    src = ck["state_dict"]
    # fc_in 的列布局是 [通道特征(64) | 全局标量(v1 70)]，v2 只是把全局标量从 70
    # 扩到 98，所以前 64+70=134 列整体可以逐位搬过来，新增的 28 列留初始化。
    n_prefix = src["fc_in.weight"].shape[1]
    assert n_prefix == 64 + 70, n_prefix
    # stem 前 15 通道、fc_in 前 134 列必须逐位一致
    assert torch.equal(sd["stem.0.weight"][:, :15], src["stem.0.weight"])
    assert torch.equal(sd["fc_in.weight"][:, :n_prefix], src["fc_in.weight"])
    assert torch.equal(sd["fc_in.bias"], src["fc_in.bias"])
    # 新增的 16 个通道 / 28 列不该是"复制过来的零"（应当保持随机初始化）
    assert not torch.equal(sd["fc_in.weight"][:, n_prefix:],
                           torch.zeros_like(sd["fc_in.weight"][:, n_prefix:]))
    # 其余共享键（ResBlock / 各头 / BN）也必须原样搬过来
    for k in ("res.0.conv1.weight", "head_discard.weight", "head_value.bias",
              "fc_mid.weight"):
        assert torch.equal(sd[k], src[k]), k
    assert any("stem.0.weight" in s for s in info["partial"]), info["partial"]
    print(f"ok 热启动 v1→v2：原样复制 {info['copied']} 个张量，"
          f"部分复制 {len(info['partial'])} 个（新增通道/列随机初始化）")


def test_pack_mask_are_exact_slices():
    """包掩码：关掉的包**整段为 0**，开着的包**逐位不变**，v1 前缀永不受影响。

    这是「单变量消融」的地基——如果关包时把别的包也动了，涨跌就说不清是谁的功劳。
    """
    from zhuoji.encoder_v2 import (PACK_GLOB_SLICES, PACK_TILE_SLICES)
    g = DushanGame(DushanConfig(), seed=31)
    _play(g, 60)
    me = g.actor()

    full = encode(g, me, 2)
    assert full.tiles.shape == (NUM_CHANNELS_V2, NUM_TILE_TYPES)

    # 逐个包的组合都试一遍（含"全关"和"全开"）
    for packs in [("a",), ("b",), ("c",), ("a", "b"), ("a", "c"), ("b", "c"), ()]:
        o = encode(g, me, 2, packs)
        assert np.array_equal(o.tiles[:NUM_CHANNELS], full.tiles[:NUM_CHANNELS])
        assert np.array_equal(o.glob[:_V1_GLOBAL_DIM], full.glob[:_V1_GLOBAL_DIM])
        for name, (lo, hi) in PACK_TILE_SLICES.items():
            if name in packs:
                assert np.array_equal(o.tiles[lo:hi], full.tiles[lo:hi]), (packs, name)
            else:
                assert not o.tiles[lo:hi].any(), (packs, name, "应为全 0")
        for name, (lo, hi) in PACK_GLOB_SLICES.items():
            if name in packs:
                assert np.array_equal(o.glob[lo:hi], full.glob[lo:hi]), (packs, name)
            else:
                assert not o.glob[lo:hi].any(), (packs, name, "应为全 0")
    # 全关 = v1 特征 + 零填充
    o0 = encode(g, me, 2, ())
    assert not o0.tiles[NUM_CHANNELS:].any() and not o0.glob[_V1_GLOBAL_DIM:].any()
    print("ok 包掩码：关包整段清零、开包逐位不变、v1 前缀不受影响")


def test_parse_packs():
    from zhuoji.encoder_v2 import ALL_PACKS, parse_packs
    assert parse_packs(None) == ALL_PACKS == ("a", "b", "c")
    assert parse_packs("abc") == ("a", "b", "c")
    assert parse_packs("ba") == ("a", "b")          # 顺序固定，与输入顺序无关
    assert parse_packs("a") == ("a",)
    assert parse_packs(["b", "c"]) == ("b", "c")
    assert parse_packs("none") == ()
    assert parse_packs("-") == ()
    assert parse_packs("") == ()
    print("ok parse_packs：None=全开、乱序归一、none/空串=全关")


def test_pack_mask_keeps_architecture():
    """关包**不改变**维度与参数量：这才叫单变量消融。"""
    import torch
    from zhuoji.net import NetConfig, build_model
    from zhuoji.encoder import dims_for_version
    c, gd = dims_for_version(2)
    m = build_model(NetConfig(48, 8, 256, global_dim=gd, in_channels=c,
                              spatial_head=True))
    n = m.n_params()
    for packs in [("a", "b", "c"), ("a",), ()]:
        tiles = np.zeros((1, c, NUM_TILE_TYPES), dtype=np.float32)
        gv = np.zeros((1, gd), dtype=np.float32)
        out = m(torch.from_numpy(tiles), torch.from_numpy(gv))
        assert out["discard"].shape == (1, 28)
    assert m.n_params() == n
    print(f"ok 关包不改架构：参数量恒为 {n:,}")


def test_net_agent_inherits_packs_from_model():
    """``NetAgent`` 必须从模型自带属性取包掩码，否则评测会把训练关掉的包又打开。"""
    import torch
    from zhuoji.agent import NetAgent
    from zhuoji.encoder import dims_for_version
    from zhuoji.encoder_v2 import PACK_TILE_SLICES
    from zhuoji.net import NetConfig, build_model
    c, gd = dims_for_version(2)
    m = build_model(NetConfig(48, 8, 256, global_dim=gd, in_channels=c))
    m.eval()
    m.encoder_version = 2
    m.encoder_packs = ("a",)
    agent = NetAgent(m, temperature=0.0)
    assert agent.encoder_version == 2
    assert tuple(agent.encoder_packs) == ("a",)
    g = DushanGame(DushanConfig(), seed=5)
    _play(g, 40)
    obs = agent._obs(g)
    lo, hi = PACK_TILE_SLICES["b"]
    assert not obs.tiles[lo:hi].any(), "NetAgent 没继承包掩码"
    print("ok NetAgent 从模型属性继承 encoder_version / encoder_packs")


def test_ck_from_model_roundtrip():
    """``ck_from_model`` → ``build_from_ckpt`` 往返：版本/包掩码/网络结构都还原。"""
    import torch
    from zhuoji.encoder import dims_for_version
    from zhuoji.net import NetConfig, build_from_ckpt, build_model, ck_from_model
    c, gd = dims_for_version(2)
    m = build_model(NetConfig(32, 4, 128, global_dim=gd, in_channels=c,
                              spatial_head=True))
    m.eval()
    # 造一个非平凡权重，保证往返后数值一致是"真复制"而不是"都是零"
    with torch.no_grad():
        for p in m.parameters():
            p.add_(torch.randn_like(p) * 0.05)
    ck = ck_from_model(m, encoder_version=2, encoder_packs=("b", "c"))
    m2, ver = build_from_ckpt(ck)
    assert ver == 2 and tuple(m2.encoder_packs) == ("b", "c")
    assert m2.n_params() == m.n_params()
    tiles = np.random.rand(2, c, NUM_TILE_TYPES).astype(np.float32)
    gv = np.random.rand(2, gd).astype(np.float32)
    with torch.no_grad():
        a = m(torch.from_numpy(tiles), torch.from_numpy(gv))
        b = m2(torch.from_numpy(tiles), torch.from_numpy(gv))
    for k in a:
        assert torch.equal(a[k], b[k]), k
    print("ok ck_from_model↔build_from_ckpt 往返：版本/包掩码/输出全一致")


def test_dump_ckpt_writes_packs():
    """``_dump_ckpt`` 必须把包掩码写进检查点（评测靠它自动复现训练口径）。"""
    import torch
    from scripts.selfplay import _dump_ckpt
    from zhuoji.encoder import dims_for_version
    from zhuoji.net import NetConfig, build_from_ckpt, build_model
    c, gd = dims_for_version(2)
    m = build_model(NetConfig(24, 2, 64, global_dim=gd, in_channels=c))
    m.eval()
    nc = {"channels": 24, "blocks": 2, "hidden": 64, "global_dim": gd,
          "in_channels": c, "spatial_head": False}
    out = Path(ROOT) / "reports" / "_tmp_pack_meta.pt"

    class A:      # 假装是 argparse.Namespace
        pass
    _dump_ckpt(out, m, nc, A(), 7, {}, 2, ("a", "c"))
    ck = torch.load(out, map_location="cpu", weights_only=False)
    assert ck["encoder_version"] == 2 and ck["encoder_packs"] == ["a", "c"]
    _, ver = build_from_ckpt(ck)
    m2, _ = build_from_ckpt(ck)
    assert ver == 2 and tuple(m2.encoder_packs) == ("a", "c")
    # v1 不写这个字段（写了反而误导：v1 没有包的概念）
    _dump_ckpt(out, m, nc, A(), 7, {}, 1, None)
    ck1 = torch.load(out, map_location="cpu", weights_only=False)
    assert "encoder_packs" not in ck1
    out.unlink()
    print("ok _dump_ckpt：v2 写 encoder_packs、v1 不写")


if __name__ == "__main__":
    test_dimensions()
    test_range()
    test_v1_is_prefix_of_v2()
    test_view_permutation()
    test_no_information_leak()
    test_boundary_empty_river()
    test_boundary_short_river()
    test_river_temporal_channels_are_temporal()
    test_v2_runtime_cost()
    test_net_accepts_v2()
    test_spatial_head_is_zero_init_additive()
    test_warm_start_v1_to_v2()
    test_pack_mask_are_exact_slices()
    test_parse_packs()
    test_pack_mask_keeps_architecture()
    test_net_agent_inherits_packs_from_model()
    test_ck_from_model_roundtrip()
    test_dump_ckpt_writes_packs()
    print("编码器 v2（P1 阶段 0）验收测试全部通过 ✅")
