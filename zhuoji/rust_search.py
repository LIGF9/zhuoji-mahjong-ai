"""Rust 搜索内核的 Python 胶水层。

分工
----
* **Rust**（`dushan_engine.PySearch`）：determinization、克隆、状态推进、快进、编码
* **Python**：批量网络前向 + 终局精确结算（重放 Rust 给出的动作轨迹）

外部只需要 `rust_q(...)` 或 `RustSearchAgent`。
"""
from __future__ import annotations

import copy

import numpy as np
import torch

from .encoder import GLOBAL_DIM, NUM_CHANNELS
from .rules import Action

KIND_DISCARD, KIND_HU, KIND_ANGANG, KIND_BUGANG, KIND_PENG, KIND_MINGGANG, KIND_PASS = range(7)
_KIND_NAME = ["discard", "hu", "angang", "bugang", "peng", "minggang", "pass"]
_HEAD_NAME = {0: "discard", 1: "self_kong", 2: "respond"}


def _code(a) -> tuple[int, int]:
    from .encoder import action_to_index  # noqa: PLC0415

    _ = action_to_index
    k = {
        "discard": KIND_DISCARD, "hu": KIND_HU, "angang": KIND_ANGANG,
        "bugang": KIND_BUGANG, "peng": KIND_PENG, "minggang": KIND_MINGGANG,
        "pass": KIND_PASS,
    }[a.kind]
    return k, (255 if a.tile < 0 else int(a.tile))


def snapshot_of(game, p: int) -> dict:
    """把 DushanGame 的可变状态摊平成 Rust 认识的 dict。"""
    melds = []
    for q in range(4):
        for mt, t, src in game.melds[q]:
            melds.extend((q, int(mt), int(t), int(src)))
    discards = []
    for q in range(4):
        for t in game.discards[q]:
            discards.append(q)
            discards.append(int(t))
    hands = []
    for q in range(4):
        hands.extend(int(x) for x in game.hands[q])
    pass_lock = [sum(1 << t for t in game.pass_lock[q]) for q in range(4)]
    cfg = game.cfg
    return {
        "wall": [int(t) for t in game.wall],
        "wall_pos": int(game.wall_pos),
        "dealer": int(game.dealer),
        "dealer_streak": int(getattr(game, "dealer_streak", 0)),
        "current": int(game.current),
        "phase": {"self_kong": 0, "discard": 1, "respond": 2, "over": 3}[game.phase.value],
        "drawn": bool(game.drawn),
        "hands": hands,
        "melds": melds,
        "discards": discards,
        "pass_lock": pass_lock,
        "baojiao": [int(x) for x in game.baojiao],
        "ever_tenpai": [int(x) for x in game.ever_tenpai],
        "first_discard_done": [int(x) for x in game.first_discard_done],
        "gang_discard": [int(x) for x in game.gang_discard],
        "repao_discard": bool(getattr(game, "repao_discard", False)),
        "after_gang_draw": bool(getattr(game, "_after_gang_draw", False)),
        "any_discard": bool(getattr(game, "_any_discard", False)),
        "first_yaoji_discarded": bool(getattr(game, "_first_yaoji_discarded", False)),
        "chongfeng_player": -1 if game.chongfeng_player is None else int(game.chongfeng_player),
        "zeren_player": -1 if getattr(game, "zeren_player", None) is None
                        else int(game.zeren_player),
        "zeren_ponger": -1 if getattr(game, "zeren_ponger", None) is None
                        else int(game.zeren_ponger),
        "last_discard": [-1, -1] if game.last_discard is None
                        else [int(game.last_discard[0]), int(game.last_discard[1])],
        "rob_kong": [-1, -1, 0] if getattr(game, "_rob_kong", None) is None
                    else [int(game._rob_kong[0]), int(game._rob_kong[1]),
                          int(getattr(game, "_rob_kong_konger", 0))],
        "respond_order": [int(x) for x in game._respond_order],
        "respond_idx": int(game._respond_idx),
        "respond_hu_votes": [int(x) for x in game._respond_hu_votes],
        "n_hu": int(game._n_hu),
        "ji_tiles": sum(1 << t for t in game.ji_tiles),
        "pass_lock_cfg": bool(cfg.enable_pass_lock),
        "qiang_gang": bool(cfg.enable_qiang_gang),
        "baojiao_auto": bool(cfg.baojiao_auto),
        "pinghu_needs_dou": bool(cfg.ping_hu_needs_dou_for_ron),
    }


def replay_delta(base_game, wall, wall_pos, hands_flat, trace, p: int) -> float:
    """按 Rust 给的 determinization 起点 + 动作轨迹重放，返回 p 的净分差。"""
    g = base_game.clone()
    g.hands = [list(hands_flat[q * 27:(q + 1) * 27]) for q in range(4)]
    g.wall = list(wall)
    g.wall_pos = int(wall_pos)
    for kind, tile in trace:
        g.step(Action(_KIND_NAME[kind], -1 if tile == 255 else int(tile)))
    res = g.result or {}
    deltas = res.get("deltas")
    if not deltas:
        return 0.0
    return float(deltas[p])


_PAD_CHOICES = (32, 64, 128, 256, 512, 1024, 2048, 4096)


def _pad_to(n: int) -> int:
    for c in _PAD_CHOICES:
        if n <= c:
            return c
    return 8192


class _BatchRunner:
    """批前向运行器：预分配缓冲 + 64 倍数量化 pad + CPU 端 mask/argmax。

    2026-11-02 实测推翻了两个旧前提：
    1. **MPS 前向对这本网是线性的（~14.5μs/状态，无批处理经济性）**——所以
       pad 到 2 的幂会白烧 44% 算力；pad 唯一的意义是控制形状种类数。
    2. 「变形状触发 kernel 重编译 ~15ms」只对**每个新形状的第一次**成立，
       MPS 图缓存按形状缓存且进程内常驻。所以把形状量化到 64 的倍数
       （全部形状 ≤64 个，进程很快焐热）比 2 的幂 pad 快 ~1.7×。
    3. mask/argmax 放 GPU 反而更慢——bool 索引/where 每批多次强制同步，
       CPU 端 numpy 组合更便宜。

    缓冲全部**每次决策分配一次**、批次间复用（旧版每批 np.zeros 6.6MB）。
    """

    _MAX_PAD = 4096

    def __init__(self, model, device: str, v_model=None):
        self.model = model
        self.device = device
        # 叶子 V ensemble（2026-11-03）：v_model 提供第二个 V，与前向取均值；
        # 策略头仍只用主网络。None = 关闭（逐位同旧行为）。
        self.v_ort = None
        self.v_model_t = None
        if v_model is not None:
            if str(device).lower().startswith("ort"):
                from .forward import build_forward  # noqa: PLC0415

                self.v_ort = build_forward(v_model, device,
                                           stem=getattr(v_model, "onnx_stem", None))
            else:
                self.v_model_t = copy.deepcopy(v_model).to(device).eval()
        mp = self._MAX_PAD
        self.tt = np.zeros((mp, NUM_CHANNELS, 27), dtype=np.float32)
        self.gg = np.zeros((mp, GLOBAL_DIM), dtype=np.float32)
        self.lg = np.empty((mp, 28), dtype=np.float32)
        # device 以 "ort" 开头 → 走 onnxruntime（CoreML/CPU）后端：
        # 输入直接喂 numpy 缓冲，不再需要 torch 张量与显存拷贝（2026-11-02）。
        self.ort = None
        if str(device).lower().startswith("ort"):
            from .forward import build_forward  # noqa: PLC0415

            self.ort = build_forward(model, device,
                                     stem=getattr(model, "onnx_stem", None))
        else:
            self.t_t = torch.from_numpy(self.tt)          # 与 numpy 共享内存
            self.t_g = torch.from_numpy(self.gg)
            self.gpu_t = torch.empty((mp, NUM_CHANNELS, 27), device=device)
            self.gpu_g = torch.empty((mp, GLOBAL_DIM), device=device)

    def run(self, tiles, glob, head, mask):
        n = int(head.shape[0])
        pad = (n + 63) // 64 * 64
        self.tt[:n] = tiles.reshape(n, NUM_CHANNELS, 27)
        self.gg[:n] = glob.reshape(n, GLOBAL_DIM)
        if pad > n:                                    # 补齐行清零（缓冲是复用的）
            self.tt[n:pad] = 0
            self.gg[n:pad] = 0
        if self.ort is not None:
            out = self.ort.raw(self.tt[:pad], self.gg[:pad])
            vals = out["value"][:n].reshape(-1)
        else:
            self.gpu_t[:pad].copy_(self.t_t[:pad])
            self.gpu_g[:pad].copy_(self.t_g[:pad])
            o = self.model(self.gpu_t[:pad], self.gpu_g[:pad])
            out = {k: v.detach().cpu().numpy() for k, v in o.items()}
            vals = out["value"][:n].reshape(-1)
        if self.v_ort is not None:
            v2 = self.v_ort.raw(self.tt[:pad], self.gg[:pad])["value"][:n].reshape(-1)
            vals = (vals + v2) * 0.5
        elif self.v_model_t is not None:
            with torch.no_grad():
                v2 = self.v_model_t(self.gpu_t[:pad], self.gpu_g[:pad])
            vals = (vals + v2["value"].detach().cpu().numpy()[:n].reshape(-1)) * 0.5
        lg = self.lg[:n]
        lg.fill(-1e9)
        head_np = head
        for hidx, hname in ((0, "discard"), (1, "self_kong"), (2, "respond")):
            src = out[hname]                           # (pad, w)
            rows = np.flatnonzero(head_np == hidx)
            if rows.size:
                lg[rows, :src.shape[1]] = src[rows]
        lg[~mask.reshape(n, 28).view(np.bool_)] = -1e9
        acts = lg.argmax(axis=1).astype(np.uint8)
        return acts, vals.astype(np.float32)


@torch.no_grad()
def rust_q(model, game, p: int, legal, n_det: int = 32, ff_cap: int = 4,
           seed: int = 0, device: str = "cpu", stats: dict | None = None):
    """用 Rust 内核跑一次公平 PIMC，返回每个候选动作的平均 Q（分差单位）。

    与 `SearchAgent._search_fair_batched` 同构：n_det 个 determinization × 每个
    候选动作 → step → 贪心快进到本座位决策点 → V 估值；终局用真实结算。
    """
    import dushan_engine as de  # noqa: PLC0415

    snap = snapshot_of(game, p)
    sn = de.PySearch(snap, p, [_code(a) for a in legal], n_det, ff_cap, seed)
    stats = stats if stats is not None else {}
    stats.setdefault("batches", 0)
    stats.setdefault("states", 0)
    stats.setdefault("terminals", 0)
    runner = _BatchRunner(model, device)

    guard = 0
    while not sn.finished():
        guard += 1
        if guard > 4000:
            raise RuntimeError("rust_q 未收敛（疑似死循环）")
        if sn.needs_net():
            tiles, glob, head, mask, kind = sn.take_batch()
            n = head.shape[0]
            stats["batches"] += 1
            stats["states"] += n
            if n:
                acts, vals = runner.run(tiles, glob, head, mask)
                sn.feed(acts.tolist(), vals)
            continue
        if sn.has_unsettled():
            for (i, _d, wall, wpos, hands, trace) in sn.unsettled():
                stats["terminals"] += 1
                try:
                    d = replay_delta(game, wall, wpos, hands, trace, p)
                except Exception:  # noqa: BLE001 - 重放异常按 0 处理并计数
                    stats["replay_fail"] = stats.get("replay_fail", 0) + 1
                    d = 0.0
                sn.settle(i, d)
            continue
        break
    return np.array(sn.action_q(), dtype=np.float64), stats


@torch.no_grad()
def _deep_search_run(model, game, p: int, legal, n_det: int, ff_cap: int,
                     topk: int, ply: int, seed: int, device: str,
                     stats: dict | None, fair: bool, want_diag: bool,
                     agg: int = 0, lam: float = 1.0, reweight: bool = False,
                     v_model=None):
    """`deep_q` 的实现体：`(q, stats, diag, qs)`。

    ``agg``（2026-11-03，实验 A）：ply≥2 的聚合方式。
    0 = max（现状，每个 determinization 逐世界取 max = 策略融合）；
    1 = mean（V 与 topk 分支取均值 = 融合无关口径）。
    其余参数与返回值不变。
    ``diag`` / ``qs`` 仅在 ``want_diag=True`` 时有内容（否则为 [] / []）：
    - ``diag[d][i] = {"terminal": bool, "win": bool, "delta": float}``
      —— 第 d 个假想世界里打动作 i 的终局诊断（ply=1 口径，来自 Rust
      `det_stats`；ply≥2 时 diag 为空）。
    - ``qs[d][i]`` —— 逐 determinization 的 Q（与 Python 版对拍同义）。
    """
    import dushan_engine as de  # noqa: PLC0415

    snap = snapshot_of(game, p)
    ds = de.PyDeepSearch(snap, p, [_code(a) for a in legal], n_det, ff_cap, topk,
                         ply, seed, fair, int(agg), float(lam), bool(reweight))
    stats = stats if stats is not None else {}
    for k, v in (("batches", 0), ("states", 0), ("terminals", 0), ("stages", 0)):
        stats.setdefault(k, v)
    runner = _BatchRunner(model, device, v_model=v_model)

    def run_batch():
        tiles, glob, head, mask, kind = ds.take_batch()
        n = head.shape[0]
        if not n:
            return
        stats["batches"] += 1
        stats["states"] += n
        acts, vals = runner.run(tiles, glob, head, mask)
        ds.feed(acts.tolist(), vals)

    def settle_all():
        for (i, _root, wall, wpos, hands, trace) in ds.unsettled():
            stats["terminals"] += 1
            try:
                d = replay_delta(game, wall, wpos, hands, trace, p)
            except Exception:  # noqa: BLE001
                stats["replay_fail"] = stats.get("replay_fail", 0) + 1
                d = 0.0
            ds.settle(i, d)

    guard = 0
    while not ds.finished():
        guard += 1
        if guard > 20000:
            raise RuntimeError("deep_q 未收敛")
        if ds.needs_net():
            run_batch()
        elif ds.has_unsettled():
            settle_all()
        elif ds.stage_done():
            stats["stages"] = max(stats["stages"], ds.stage() + 1)
            if not ds.promote():
                break
        else:
            break
    stats["stages"] = max(stats["stages"], ds.n_stages())
    q = np.array(ds.result(), dtype=np.float64)
    if not want_diag:
        return q, stats, [], []
    diag = [[{"terminal": bool(t), "win": bool(w), "delta": float(x)}
             for (t, w, x) in row] for row in ds.det_stats()] if ply == 1 else []
    qs = [list(map(float, row)) for row in ds.per_det()] if ply == 1 else []
    return q, stats, diag, qs


@torch.no_grad()
def deep_q(model, game, p: int, legal, n_det: int = 32, ff_cap: int = 24,
           topk: int = 3, ply: int = 1, seed: int = 0, device: str = "cpu",
           stats: dict | None = None, fair: bool = True, agg: int = 0,
           lam: float = 1.0, reweight: bool = False, v_model=None):
    """多层搜索（ply 1/2/3），与 `SearchAgent._deep` 同口径。

    与 `rust_q` 的区别：第二/三层把多个叶子的动作合并成一个批次，
    Python 版是每个叶子单独一次前向。
    ``agg``：0=max（现状），1=mean（融合无关聚合，实验 A）。
    ``lam``：信念混合系数（1.0=均匀现状，<1=B1 信念化采样）；
    ``reweight``：是否按 IS 权重做无偏聚合（实验 B1）。
    """
    q, stats, _diag, _qs = _deep_search_run(model, game, p, legal, n_det, ff_cap,
                                            topk, ply, seed, device, stats,
                                            fair, want_diag=False, agg=agg,
                                            lam=lam, reweight=reweight,
                                            v_model=v_model)
    return q, stats


@torch.no_grad()
def deep_q_detail(model, game, p: int, legal, n_det: int = 32, ff_cap: int = 24,
                  topk: int = 3, ply: int = 1, seed: int = 0, device: str = "cpu",
                  stats: dict | None = None, fair: bool = True, agg: int = 0,
                  lam: float = 1.0, reweight: bool = False):
    """`deep_q` + 逐 determinization 诊断：返回 `(q, stats, diag, qs)`。

    供推荐理由层复用 Python 版 `explain_items` 的「N 张假想世界终局统计」
    （直通终局几条 / 我胡几次 / 被胡几次 / 终局均分）。ply≥2 时 diag/qs 为空
    （第二/三层的 max 语义不回写第一层 task，diag 只反映 1-ply 口径，不导出）。
    """
    return _deep_search_run(model, game, p, legal, n_det, ff_cap, topk, ply,
                            seed, device, stats, fair, want_diag=True, agg=agg,
                            lam=lam, reweight=reweight)


class RustSearchAgent:
    """`SearchAgent` 的 Rust 内核版本 —— 可直接用于 `play_duplicate` 的牌力对战。

    与 `SearchAgent` 的差别只在**实现**（状态推进/克隆/快进在 Rust），决策口径
    与 `_search_fair_batched` 已做过逐位对拍（`scripts/check_search_parity.py`），
    所以两者可以直接对打，不会引入「换了 agent」的混淆。

    平手裁决沿用 `SearchAgent.choose`：与 max 相差 <1e-9 的动作里取先验最高者。
    先验需要额外一次前向，只在真的出现平手时才付这个钱（罕见）。
    """

    def __init__(self, model, seed: int = 0, device: str = "cpu", name: str = "rust",
                 ply: int = 1, topk: int = 3, n_det: int = 32, ff_cap: int = 24,
                 fair: bool = True, own_model: bool = False, p2_topk: int | None = None,
                 agg: int = 0, lam: float = 1.0, reweight: bool = False,
                 v_ens_model=None):
        # 与 SearchAgent 同口径：持有**设备就位的独立副本**。两个原因：
        # ① 不能 .to(device) 移动共享对象——同进程里的 CPU NetAgent 会挂；
        # ② 不能把 CPU 模型直接喂给 deep_q(device="mps")——前向必崩
        #    （2026-11-02 修复：此前网页 Rust 路径每次都在这里崩、静默回落
        #    Python SearchAgent，等于 128 路一直没真正用上 Rust）。
        self.model = (copy.deepcopy(model) if str(device).lower().startswith("ort")
                      else copy.deepcopy(model).to(device))
        self.seed = seed
        self.device = device
        self.name = name
        self.ply = ply
        self.topk = int(p2_topk) if p2_topk is not None else topk
        self.n_det = n_det
        self.ff_cap = ff_cap
        self.fair = fair
        self.agg = int(agg)              # ply≥2 聚合：0=max（现状）, 1=mean（融合无关）
        self.lam = float(lam)            # B1：1.0=均匀现状，<1=信念化采样
        self.reweight = bool(reweight)   # B1：IS 无偏聚合开关
        self.v_ens_model = v_ens_model   # 叶子 V ensemble（None=关闭）
        self._own_model = own_model      # 仅为兼容 SearchAgent 的签名，无实际作用
        self.stats: dict = {}
        self._call = 0           # 决策计数：RNG 必须**跨决策递进**
        self._helper = None      # 惰性构造：只用来算先验（平手裁决）

    def _next_seed(self) -> int:
        """RNG 跨决策递进（splitmix64 打散）。

        Rust 侧 `Rng::new(seed)` 是 xorshift64*（`seed|1`），若是每次决策都传同一个
        常数，则每个决策点都在复用同一串随机流 —— 与 `SearchAgent` 的行为不一致
        （它的 numpy Generator 是持续前进的）。而且 xorshift 对小整数种子前几个
        输出高度相关，所以这里必须先用 splitmix64 终混再喂。
        """
        self._call += 1
        m = (1 << 64) - 1
        x = (self.seed * 0x9E3779B97F4A7C15 + self._call * 0xD1B54A32D192ED03) & m
        x ^= x >> 30
        x = (x * 0xBF58476D1CE4E5B9) & m
        x ^= x >> 27
        x = (x * 0x94D049BB133111EB) & m
        return (x ^ (x >> 31)) & m

    def _prior(self, game, p: int, legal):
        if self._helper is None:
            from .search_agent import SearchAgent  # noqa: PLC0415

            self._helper = SearchAgent(self.model, seed=self.seed, device=self.device,
                                       own_model=False, fair=False, n_det=1)
        from .encoder import index_to_action  # noqa: PLC0415

        o = self._helper._obs_np(game, p)  # noqa: SLF001
        lg, _v = self._helper._batch_out(  # noqa: SLF001
            [(o.tiles, o.glob, o.head, o.mask)])
        lab2i = {}
        for idx in range(o.mask.shape[0]):
            if o.mask[idx]:
                lab2i[index_to_action(o.head, idx, game).label()] = idx
        pr = torch.softmax(lg[0], dim=-1).cpu().numpy()
        return np.array([pr[lab2i[a.label()]] for a in legal], dtype=np.float64)

    @torch.no_grad()
    def choose(self, game):
        legal = game.legal_actions()
        if not legal:
            return None
        if len(legal) <= 1:
            return legal[0]
        p = game.actor()
        q, st = deep_q(self.model, game, p, legal, n_det=self.n_det,
                       ff_cap=self.ff_cap, topk=self.topk, ply=self.ply,
                       seed=self._next_seed(), device=self.device, stats=self.stats,
                       fair=self.fair, agg=self.agg, lam=self.lam,
                       reweight=self.reweight, v_model=self.v_ens_model)
        qmax = float(q.max())
        tie = np.flatnonzero(q >= qmax - 1e-9)
        if len(tie) == 1:
            return legal[int(tie[0])]
        prior = self._prior(game, p, legal)
        return legal[int(tie[int(np.argmax(prior[tie]))])]

    def rank(self, game, k: int = 4):
        legal = game.legal_actions()
        if not legal:
            return [], 0.0
        p = game.actor()
        q, _st = deep_q(self.model, game, p, legal, n_det=self.n_det,
                        ff_cap=self.ff_cap, topk=self.topk, ply=self.ply,
                        seed=self._next_seed(), device=self.device, stats=self.stats,
                        fair=self.fair, agg=self.agg, lam=self.lam, reweight=self.reweight)
        order = np.argsort(-q)[:max(1, min(int(k), len(legal)))]
        return [(legal[int(i)], float(q[int(i)])) for i in order], 0.0

    def rank_detail(self, game):
        """`rank` 的诊断版：返回 `(q, qs, diag)`（对齐 legal 顺序）。

        - ``q``：逐动作平均 Q；
        - ``qs[d][i]``：第 d 个假想世界里动作 i 的 Q；
        - ``diag[d][i] = {"terminal","win","delta"}``：终局诊断（仅 ply=1）。
        供推荐理由层 `explain_items` 的「N 张假想世界终局统计」。
        """
        legal = game.legal_actions()
        if not legal:
            return np.zeros(0), [], []
        p = game.actor()
        q, _st, diag, qs = deep_q_detail(
            self.model, game, p, legal, n_det=self.n_det, ff_cap=self.ff_cap,
            topk=self.topk, ply=self.ply, seed=self._next_seed(),
            device=self.device, stats=self.stats, fair=self.fair, agg=self.agg,
            lam=self.lam, reweight=self.reweight)
        return q, qs, diag
