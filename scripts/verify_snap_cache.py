# -*- coding: utf-8 -*-
"""快照缓存回归 —— 两个阶段。

阶段 A（直接断言，不依赖对局随机性）
    用 spy 拦住 ``vcollect._make_seats``，分别走「全量发送」与「缓存命中」两条
    worker 路径，读每次建局时**实际拿到几个快照对手**。
    命中路径必须与全量路径一样多 —— 这是回归点。

阶段 B（端到端等价）
    同 seed / 同快照跑两次 ``collect_parallel``：第二次只发 key（真走缓存），
    轨迹必须与第一次逐元素一致；同时断言「快照确实上过桌」。

为什么要重写
------------
初版脚本只做了阶段 B，而且有**统计盲区**：``GAMES=8`` 分给 3 个 worker 每个
只建 2–3 局，而快照座位分支的概率是 ``past_prob=0.3`` —— 很可能两次调用都
一局没抽中快照，于是轨迹天然相等，**测不出任何问题**。
2026-10-12 发现的真实缺陷正落在这个盲区里：``_worker`` 在缓存命中分支末尾
多写了一行重建 ``snaps``，把刚取出的缓存覆盖成空列表，导致默认每 4 轮里有
3 轮**完全没有历史快照在桌上**（league 自博弈静默失效）。
阶段 A 用 spy 直接读事实，不受随机性影响；阶段 B 把局数与断言都补上。
"""
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import multiprocessing as mp  # noqa: E402

import zhuoji.vcollect as V  # noqa: E402
from zhuoji.encoder import dims_for_version  # noqa: E402
from zhuoji.encoder_v2 import parse_packs  # noqa: E402
from zhuoji.net import (build_model, ck_from_model, encoder_version_from_ckpt,  # noqa: E402
                        load_into, net_config_from_ckpt, net_meta)
from zhuoji.vcollect import collect_parallel  # noqa: E402

RULES = "dushan"
GAMES = 24          # 必须够大，否则 30% 的快照分支抽不到（初版 GAMES=8 的盲区）
WORKERS = 3

FAIL = 0


def check(name: str, ok: bool, detail: str = "") -> None:
    global FAIL
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}{(' — ' + detail) if detail else ''}")
    if not ok:
        FAIL += 1


def build(stem: str):
    """按 selfplay 的口径从 models/{stem}.pt 构建模型与 net 元数据。"""
    ck = torch.load(ROOT / "models" / f"{stem}.pt", map_location="cpu",
                    weights_only=False)
    ver = encoder_version_from_ckpt(ck)
    cfg = net_config_from_ckpt(ck)
    cfg.in_channels, cfg.global_dim = dims_for_version(ver)
    cfg.spatial_head = bool((ck.get("net") or {}).get("spatial_head", False))
    packs = parse_packs("") if ver < 2 else None
    model = build_model(cfg)
    load_into(model, ck["state_dict"])
    model.eval()
    model.encoder_version = ver
    model.encoder_packs = packs if ver >= 2 else None
    return model, ver, net_meta(cfg), packs


# ---------------------------------------------------------------------------
# 阶段 A：缓存命中路径必须保留对手池
# ---------------------------------------------------------------------------
def _tiny_snap(seed: int) -> dict:
    from zhuoji.net import ZhuojiNet
    torch.manual_seed(seed)
    m = ZhuojiNet(channels=16, blocks=2, hidden=32).eval()
    return ck_from_model(m, encoder_version=1, encoder_packs=None)


def stage_a() -> None:
    print("=== 阶段 A：缓存命中路径的对手池 ===")
    from zhuoji.net import ZhuojiNet
    snaps = [_tiny_snap(s) for s in (1, 2, 3)]
    seen: list[int] = []
    orig = V._make_seats

    def spy(rng, teacher_prob, snapshots, past_prob, *rest, **kw):   # rest/kw：master_prob、snap_temp 等
        seen.append(len(snapshots))
        return orig(rng, teacher_prob, snapshots, past_prob, *rest, **kw)

    V._make_seats = spy
    V._SNAP_CACHE["key"] = None
    sd = ZhuojiNet(channels=16, blocks=2, hidden=32).state_dict()
    nc = {"channels": 16, "blocks": 2, "hidden": 32,
          "global_dim": 70, "in_channels": 15, "spatial_head": False}

    def call(snapshots_sd, key):
        seen.clear()
        try:
            # 签名对齐 vcollect._worker（2026-10-19）：snap_key 之后还有
            # explore_eps/teacher_sd/teacher_ver/teacher_packs/master_prob/
            # aux_flag/snap_temp，再追加 master_sd/master_ver/master_shared
            # （M2-C5 教师与座位对手解耦，共 24 元）
            V._worker((sd, nc, 8, 1, 1.0, 8.0, 0.15, 0.3, snapshots_sd, RULES,
                       False, 1, None, key, 0.0, None, None, None, 0.0, False,
                       0.0, None, None, True))
            return sorted(set(seen)), None
        except V._SnapCacheMiss as e:
            return None, e

    got, err = call(snaps, 7)
    check("全量发送 → 建局时看到 3 个快照对手", got == [3], f"实得 {got}；err={err}")

    got, err = call(None, 7)
    check("缓存命中 → 建局时**仍**看到 3 个快照对手（回归点）", got == [3],
          f"实得 {got}；err={err}（实得 [0] 即缺陷复现：每 4 轮有 3 轮无对手池）")

    cached = V._SNAP_CACHE.get("snaps")
    check("缓存里确实是那 3 份快照",
          isinstance(cached, list) and len(cached) == 3,
          f"{type(cached).__name__} len={len(cached) if cached else 0}")

    got, err = call(None, 99)
    check("换快照代数 → 抛 SnapCacheMiss（上层兜底全量重发）",
          isinstance(err, V._SnapCacheMiss), f"got={got} err={err}")

    V._make_seats = orig


# ---------------------------------------------------------------------------
# 阶段 B：端到端 A/B（缓存只省 IPC，不改训练数据）
# ---------------------------------------------------------------------------
def stage_b() -> None:
    """端到端 A/B。**每次调用都传真实的 snapshots_sd**，与 selfplay 的用法一致。

    初版脚本在这里踩了两个坑，都记下来：

    1. 只传一次 ``snapshots_sd``、第二次传 ``None`` —— 但 ``collect_parallel``
       的兜底重发（缓存未命中）当时用 ``(snapshots_sd or [])`` 取载荷，传 ``None``
       就变成**空对手池**，于是"缓存命中"那一轮实际跑的是"无对手池"，
       A/B 看似通过、其实什么都没测到。
    2. ``GAMES=8`` 太小：快照座位分支概率 0.3，8 局里常常一局都没抽中。
    """
    print("=== 阶段 B：缓存的行为等价性（端到端，用法与 selfplay 一致） ===")
    master, mv, nc, mpacks = build("dushan_master")
    for q in master.parameters():
        q.requires_grad_(False)
    snaps = []
    for s in (3, 7):
        m, v, _nc, pk = build(f"dushan_big_s{s}")
        snaps.append(ck_from_model(m, encoder_version=v, encoder_packs=pk))

    def run(key, sd, pool):
        return collect_parallel(
            master.state_dict(), nc, GAMES, 20270, 0.95, 8.0, 0.08,
            workers=WORKERS, past_prob=0.3, snapshots_sd=sd, rules=RULES,
            want_traj=True, pool=pool, encoder_version=mv,
            encoder_packs=mpacks, snap_key=key)

    def diff(t1, t2) -> str:
        """返回 ""（完全一致）或一段差异描述（含首个不一致样本与最大浮点偏差）。

        不直接用 ``array_equal`` 判死：多进程池里每个 worker 的浮点归约顺序
        （oneDNN/OpenMP 分块）不保证逐位可复现，末位噪声会让 ``array_equal``
        永远为 False，从而把"是否存在真实逻辑差异"这个问题淹没掉。
        """
        if len(t1) != len(t2):
            return f"样本数不同 {len(t1)} vs {len(t2)}"
        worst = 0.0
        for i, (x, y) in enumerate(zip(t1, t2)):
            for j in (0, 1, 2):
                d = float(np.abs(np.asarray(x[j], dtype=np.float64)
                                 - np.asarray(y[j], dtype=np.float64)).max())
                worst = max(worst, d)
            for j in (3, 4):
                if x[j] != y[j]:
                    return f"样本 {i} 的动作/头不同：{x[j]} vs {y[j]}"
        if worst > 1e-6:
            return f"特征最大偏差 {worst:.3e}（超过 1e-6，可疑）"
        return "" if worst == 0.0 else f"仅浮点末位噪声 {worst:.2e}"

    ctx = mp.get_context("spawn")
    pool = ctx.Pool(processes=WORKERS)
    ts = [time.time()]
    a = run(31, snaps, pool)      # 全量发送 #1
    ts.append(time.time())
    b = run(32, snaps, pool)      # 全量发送 #2（换代数）
    ts.append(time.time())
    c = run(32, snaps, pool)      # 同代数 → **真走 worker 缓存**
    ts.append(time.time())
    e = run(34, [], pool)         # 空对手池（正向对照）
    ts.append(time.time())
    pool.close()
    pool.join()

    n = lambda r: len(r[0])  # noqa: E731
    print("  全量#1 %s=%d | 全量#2 %s=%d | 缓存 %s=%d | 空池 %s=%d"
          % (f"{ts[1] - ts[0]:.2f}s", n(a), f"{ts[2] - ts[1]:.2f}s", n(b),
             f"{ts[3] - ts[2]:.2f}s", n(c), f"{ts[4] - ts[3]:.2f}s", n(e)))

    d_ab = diff(a[0], b[0])
    d_bc = diff(b[0], c[0])
    d_be = diff(b[0], e[0])
    check("池路径自身确定：两次全量发送的轨迹一致", not d_ab or "噪声" in d_ab, d_ab)
    check("缓存命中与全量发送的轨迹一致（缓存只省 IPC）",
          not d_bc or "噪声" in d_bc, d_bc)
    check("缓存命中的 reward 与全量一致", bool(np.allclose(b[1], c[1])))
    check("空对手池 → 轨迹明显不同（证明快照确实在影响对局）",
          bool(d_be) and "噪声" not in d_be, d_be)


if __name__ == "__main__":
    stage_a()
    stage_b()
    print(f"\n{'全部通过：缓存行为保持不变 ✔' if not FAIL else str(FAIL) + ' 项失败 ✘'}")
    raise SystemExit(1 if FAIL else 0)
