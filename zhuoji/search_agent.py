"""搜索增强：1-ply / 2-ply 价值前瞻搜索 Agent（策略改进算子）。

机制：
    1-ply（B-④，已上线）在每个决策点，对每个合法动作 a：
      1. deepcopy 当前局面，step(a)（若终局，直接用真实分差，精确）；
      2. 用基准策略（贪心）快进其余座位，直到回到**本座位**的下一个决策点；
      3. 用它自己的价值头评估 V（预测本座位最终分差）。
      Q(a) = 终局 ? 实际分差 : V(后继状态)

    2-ply（B-⑤，2026-10-31 实现）在 1-ply 之后再做一次策略改进：
      对 1-ply 估值最高的 ``p2_topk`` 个动作，取它们回到本座位决策点时的局面，
      再展开一层（每个合法动作 a2 → 快进 → V），令
          Q(a) ← max( V(s'), max_{a2} V(s'_{a2}) )
      也就是把「1-ply 截断在 V」换成「再走一步看 V」。理论上 V 越准，两者越接近；
      实测分歧率与收益是判定 2-ply 是否值得的判据（见 reports/）。

关键工程点：
    - 所有克隆局面的前向**逐迭代批处理**（一次 MPS 调用喂整批状态），
      小网络 + MPS 延迟 ~2-5ms/批，单决策点 ~30-60 个前向摊成 ~6 次批调用；
    - 所有克隆共享同一 RNG 流（deepcopy 复制 rng 状态）→ 跨动作公共随机数，
      牌运方差在动作间天然对消；
    - 快进无步数上限的硬保护（默认 24 子步 > 一整圈），超限兜底用该克隆
      最近的已知信息继续推进到终局（罕见）；
    - 2-ply 只深评 top-k，其余保留 1-ply 值：全展开是 k=n 的特例，成本约 1-ply 的 3~4 倍。

用法：见 scripts/duel_agents.py / scripts/duel_search.py。
"""
from __future__ import annotations

import copy
import time

import numpy as np
import torch

from .agent import Bot
from .encoder import encode, index_to_action
from .rules import (ANGANG, BUGANG, DISCARD, HU, MINGGANG, PASS, PENG, Phase)

REL_CN = ("", "下家", "对家", "上家")

# 牌在手里的结构角色（推演理由用）：只看该牌与同花色邻牌的关系
#   刻子/对子：同种 3 / 2 张；两面搭子：±1 有牌；嵌张搭子：±2 有牌；
#   边搭/孤张：都没有。判断很轻，只用于解释措辞，不参与决策。
def _tile_struct(hand, t: int) -> str:
    c = hand[t] if 0 <= t < len(hand) else 0
    if c >= 3:
        return "刻子"
    if c == 2:
        return "对子"
    rank, suit = t % 9, t // 9
    for d in (1, -1):
        r = rank + d
        if 0 <= r <= 8 and hand[suit * 9 + r] > 0:
            return "两面搭子" if 0 < rank < 8 else "边搭"
    for d in (2, -2):
        r = rank + d
        if 0 <= r <= 8 and hand[suit * 9 + r] > 0:
            return "嵌张搭子"
    return "孤张"


class SearchAgent(Bot):
    """价值前瞻搜索 Agent（基准策略 = 同一个网络的贪心）。"""

    name = "search"

    def __init__(self, model, seed: int = 0, name: str | None = None,
                 device: str = "mps", ff_cap: int = 24,
                 prior_weight: float = 0.0, encoder_version: int | None = None,
                 encoder_packs=None, verbose: bool = False,
                 sample_tau: float = 0.0, ply: int = 1, p2_topk: int = 3,
                 own_model: bool = True, fair: bool = False, n_det: int = 1,
                 batched: bool = True):
        # 独立副本：不能 .to(device) 移动共享对象，否则同进程里的 CPU NetAgent 会挂
        _dev = str(device or "cpu").lower()
        self.ort = _dev.startswith("ort")
        if self.ort:
            # onnxruntime 后端（CoreML/CPU）：网络留 CPU，前向走 ONNX（见 forward.py）
            self.model = (copy.deepcopy(model) if own_model else model)
            self.model.eval()
            from .forward import build_forward  # noqa: PLC0415

            self.fwd = build_forward(self.model, _dev,
                                     stem=getattr(self.model, "onnx_stem", None))
        else:
            self.model = (copy.deepcopy(model) if own_model else model).to(device)
            self.model.eval()
            self.fwd = None
        self.device = device
        self.rng = np.random.default_rng(seed)
        self.ff_cap = ff_cap
        # fair=True：搜索前把「本座位不可见」的牌（后续牌墙 + 别家手牌）按当前
        # rng 洗成一张均匀 determinization，再展开动作。保证决策只依赖可观察
        # 信息 + 「其余未知牌均匀分布」的假设——不用上帝视角的真实牌墙/手牌。
        # 一次决策只洗一次（先洗再逐动作 clone）→ 跨动作公共随机数仍然成立。
        self.fair = bool(fair)
        # n_det>1（仅 fair 模式生效）：同一决策抽 n_det 张独立 determinization，
        # 各自跑完整棵搜索树后对 Q **逐动作取平均**再排序。单样本 + max 会
        # 系统性偏向「幸运假想世界」（乐观偏差），是 determinization 搜索的
        # 经典陷阱；多次采样平均把它压回去（代价 = n_det 倍搜索时间）。
        self.n_det = max(1, int(n_det))
        # batched=True：n_det 个 determinization 的叶子合并成大批前向（默认）。
        # 关掉 = 逐 determinization 串行跑（老路径，做 A/B 对照用）。
        self.batched = bool(batched)
        # 最近一次 _evaluate 的逐 determinization Q 列表（PIMC 冲突率诊断用）
        self._last_qs: list = []
        # 逐 determinization 的推演诊断（聚合 N 张假想世界的终局统计）
        self._last_diags: list = []
        # prior_weight>0 时动作分 = Q + prior_weight·log π_prior（先验正则）；
        # 0 = 纯 argmax Q（最干净的政策改进算子）。
        self.prior_weight = float(prior_weight)
        # sample_tau>0：按 softmax(score/τ) 采样动作（B4D2 扩状态覆盖用）；
        # 0 = 纯 argmax Q（B4D1 蒸馏数据用）。
        self.sample_tau = float(sample_tau)
        self.ply = int(ply)
        self.p2_topk = int(p2_topk)
        self.encoder_version = int(
            encoder_version if encoder_version is not None
            else getattr(model, "encoder_version", 1))
        self.encoder_packs = (
            encoder_packs if encoder_packs is not None
            else getattr(model, "encoder_packs", None))
        if name:
            self.name = name
        self.verbose = verbose
        self.stats = {"decisions": 0, "agreed": 0, "search_s": 0.0,
                      "forwards": 0, "clones": 0, "deep": 0}

    def reset(self, seed: int | None = None) -> None:
        if seed is not None:
            self.rng = np.random.default_rng(seed)

    # ---- 内部工具 -------------------------------------------------------
    def _obs_np(self, game, p):
        return encode(game, p, self.encoder_version, self.encoder_packs)

    def _fair_view(self, game, p: int):
        """把 ``game`` 克隆成 p 视角的公平 determinization。

        本座位不可见的信息只有两块：后续牌墙（wall[wall_pos:]）和别家手牌。
        把这两块的牌（类型多重集）合并后均匀重排，按原槽位回填——别家手牌
        总张数不变、墙长不变，重排后的世界与原世界在 p 的全部可观察量上
        **逐位相同**，但不再携带「真实墙序/真实别家手牌」的上帝信息。
        """
        g2 = game.clone()
        totals = []
        pool: list[int] = []
        for q in range(4):
            if q == p:
                totals.append(None)
                continue
            totals.append(sum(g2.hands[q]))
            for t, c in enumerate(g2.hands[q]):
                if c:
                    pool.extend([t] * c)
                    g2.hands[q][t] = 0
        wall_n = len(g2.wall) - g2.wall_pos
        pool.extend(g2.wall[g2.wall_pos:])
        # 排序后洗牌：公平视图只依赖「不可见池的多重集 + rng 状态」，
        # 与隐藏牌的原始排列无关（可被审计脚本精确检验决策不变性）。
        pool.sort()
        # 用 agent 自己的 rng：同一决策内所有动作共享同一个洗牌结果（CRN）；
        # 且同一 agent 跨决策的洗牌流也可复现（审计可重放）。
        order = self.rng.permutation(len(pool))
        shuffled = [pool[i] for i in order]
        g2.wall = g2.wall[:g2.wall_pos] + shuffled[:wall_n]
        rest = shuffled[wall_n:]
        k = 0
        for q in range(4):
            if q == p:
                continue
            for _ in range(totals[q]):
                g2.hands[q][rest[k]] += 1
                k += 1
        return g2

    @torch.no_grad()
    def _batch_out(self, states):
        """states: [(tiles,glob,head,mask)] → 每条的 (logits(仅合法位), value)。"""
        if self.fwd is not None:                       # onnxruntime 后端
            tiles_np = np.stack([s[0] for s in states]).astype(np.float32)
            glob_np = np.stack([s[1] for s in states]).astype(np.float32)
            raw = self.fwd.raw(tiles_np, glob_np)
            self.stats["forwards"] += len(states)
            logits_list = []
            for i, h in enumerate([s[2] for s in states]):
                m = states[i][3]
                lg = torch.from_numpy(raw[h][i][:m.shape[0]].copy())
                lg[~torch.from_numpy(m)] = -1e9
                logits_list.append(lg)
            return logits_list, raw["value"].reshape(-1)
        tiles = torch.from_numpy(
            np.stack([s[0] for s in states]).astype(np.float32)).to(self.device)
        glob = torch.from_numpy(
            np.stack([s[1] for s in states]).astype(np.float32)).to(self.device)
        out = self.model(tiles, glob)
        self.stats["forwards"] += len(states)
        heads = [s[2] for s in states]
        masks = [s[3] for s in states]
        # 各头的 logits 一次性搬回 CPU，再在 CPU 上做 mask。
        # 原来是在 device 上「逐样本 from_numpy(mask).to(device) + 布尔赋值」——
        # CPU 上无所谓，但在 MPS 上每次布尔索引都要同步，262 样本就是 262 次
        # GPU→CPU 往返（实测吃掉 n32 单点 12 秒里的 10 秒，而 conv 只占 0.25 秒）。
        cpu_out = {k: v.detach().cpu() for k, v in out.items() if k != "value"}
        logits_list = []
        for i, h in enumerate(heads):
            dim = masks[i].shape[0]
            lg = cpu_out[h][i][:dim].clone()
            lg[~torch.from_numpy(masks[i])] = -1e9
            logits_list.append(lg)
        vals = out["value"].detach().cpu().numpy()
        return logits_list, vals

    @staticmethod
    def _note_terminal(g, p: int) -> dict:
        """终局诊断（供推荐理由）：谁胡、几番、本座位净分差。"""
        r = g.result or {}
        return {"terminal": True, "win": r.get("winner") == p,
                "winner": r.get("winner"), "loser": r.get("loser"),
                "fan_cn": r.get("fan_cn"), "fan": r.get("fan"),
                "delta": float(r["deltas"][p]) if r.get("deltas") else 0.0}

    @torch.no_grad()
    def _expand_joint(self, tasks, p):
        """**联合**展开一组「(局面, 首个动作)」任务，再按贪心快进到 p 的下一个决策点。

        ``tasks`` 可以是同一局面的多个候选动作（= 老的单树展开），也可以是
        **多个 determinization 的候选动作全放在一起**（= 跨世界批量）。后者的
        意义：同样的算法、同样的状态数，但前向从「一批十几个状态」变成「一批几百个」，
        conv 的算力利用率差 3~5 倍——这正是 n32 延迟的主刀口（2026-11-01 实测）。

        返回 ``(q, ends, diags)``：
          - ``q[i]``：终局真实分差，或 V(回到 p 决策点时的局面)；
          - ``ends[i]``：回到 p 决策点时的 game 对象（终局 / 超 cap 的为 None）；
          - ``diags[i]``：该条推演是否直通终局、谁胡、几番。
        """
        n = len(tasks)
        q: list[float] = [0.0] * n
        ends: list = [None] * n
        diag: list[dict] = [{} for _ in range(n)]

        live: list[tuple[int, object]] = []
        for i, (gm, a) in enumerate(tasks):
            g2 = gm.clone()
            over = g2.step(a)
            self.stats["clones"] += 1
            if over:
                q[i] = float(g2.result["deltas"][p])
                diag[i] = self._note_terminal(g2, p)
            else:
                live.append((i, g2))

        # ---- 快进：批量推进，直到各自回到本座位决策点或终局 ---------------
        valued: list[tuple[int, object]] = []
        for _ in range(self.ff_cap):
            if not live:
                break
            states = []
            for _i, g in live:
                ap = g.actor()
                o = self._obs_np(g, ap)
                states.append((o.tiles, o.glob, o.head, o.mask))
            lgs, _ = self._batch_out(states)
            nxt: list[tuple[int, object]] = []
            for k, (i, g) in enumerate(live):
                a_idx = int(torch.argmax(lgs[k]).item())
                a = index_to_action(states[k][2], a_idx, g)
                if g.step(a):
                    q[i] = float(g.result["deltas"][p])
                    diag[i] = self._note_terminal(g, p)
                elif g.actor() == p:
                    valued.append((i, g))
                else:
                    nxt.append((i, g))
            live = nxt
        # 兜底：超过 cap 还没回到本座位的，继续推进到终局（无价值评估，纯策略）
        for _ in range(200):
            if not live:
                break
            states = []
            for _i, g in live:
                o = self._obs_np(g, g.actor())
                states.append((o.tiles, o.glob, o.head, o.mask))
            lgs, _ = self._batch_out(states)
            nxt = []
            for k, (i, g) in enumerate(live):
                a_idx = int(torch.argmax(lgs[k]).item())
                a = index_to_action(states[k][2], a_idx, g)
                if g.step(a):
                    q[i] = float(g.result["deltas"][p])
                    diag[i] = self._note_terminal(g, p)
                else:
                    nxt.append((i, g))
            live = nxt

        # ---- 价值批评估（所有回到 p 决策点的叶子一起前向） -----------------
        if valued:
            vstates = []
            for _i, g in valued:
                o = self._obs_np(g, p)
                vstates.append((o.tiles, o.glob, o.head, o.mask))
            _, vals = self._batch_out(vstates)
            for (i, g), v in zip(valued, vals):
                q[i] = float(v)
                ends[i] = g
        return q, ends, diag

    @torch.no_grad()
    def _expand(self, game, p, actions):
        """单棵树展开（= ``_expand_joint`` 只喂一局面的所有候选动作）。"""
        q, ends, diag = self._expand_joint([(game, a) for a in actions], p)
        return np.array(q, dtype=np.float64), ends, diag

    def _search_fair_batched(self, game, p: int, legal, reps: int):
        """n_det 个公平 determinization **合并成一次大批前向**的搜索。

        与「逐 determinization 跑完整棵树」逐位等价（同样的公平洗牌、同样的贪心
        快进、同样的 max/平均），区别只在批的形状：
          串行：32 棵树 × 每批 ~13 个状态 → conv 空转
          批量：每层把 32 棵树的叶子拼成一个 ~400 状态的批 → conv 吃饱
        返回 ``(qs, diags)``：每个 determinization 的 Q 向量与推演诊断。
        """
        n = len(legal)
        sims = [self._fair_view(game, p) for _ in range(reps)]
        tasks = [(sim, a) for sim in sims for a in legal]
        q1, e1, d1 = self._expand_joint(tasks, p)
        q = np.array(q1, dtype=np.float64).reshape(reps, n)
        ends = [[e1[d * n + j] for j in range(n)] for d in range(reps)]
        diags = [d1[d * n:(d + 1) * n] for d in range(reps)]

        if self.ply >= 2 and n > 1:
            k = max(1, min(int(self.p2_topk), n))
            # 第二层：每个 determinization 的 top-k 动作全部展开，仍走联合批
            tasks2: list = []
            owner: list[tuple[int, int, int]] = []   # (det, 动作下标, tasks2 下标)
            for d in range(reps):
                for i in np.argsort(-q[d])[:k]:
                    i = int(i)
                    g = ends[d][i]
                    if g is None:
                        continue
                    a2s = g.legal_actions()
                    if len(a2s) <= 1:
                        continue
                    for a2 in a2s:
                        tasks2.append((g, a2))
                        owner.append((d, i, len(tasks2) - 1))
            if tasks2:
                self.stats["deep"] += 1
                q2, e2, _d2 = self._expand_joint(tasks2, p)
                best: dict[tuple[int, int], tuple[float, int]] = {}
                for (d, i, t), v in zip(owner, q2):
                    cur = best.get((d, i))
                    if cur is None or float(v) > cur[0]:
                        best[(d, i)] = (float(v), t)
                for (d, i), (v2, _t) in best.items():
                    if v2 > q[d][i]:
                        q[d][i] = v2
                # 第三层（ply==3 只沿每条线的最优第二层局面再展开一层；ply>=4 全展开）
                if self.ply >= 3:
                    tasks3: list = []
                    owner3: list[tuple[int, int, int]] = []
                    for (d, i), (_v2, t) in best.items():
                        group = ([t] if self.ply == 3 else
                                 [tt for (dd, ii, tt) in owner if (dd, ii) == (d, i)])
                        for tt in group:
                            g3 = e2[tt]
                            if g3 is None:
                                continue
                            a3s = g3.legal_actions()
                            if len(a3s) <= 1:
                                continue
                            for a3 in a3s:
                                tasks3.append((g3, a3))
                                owner3.append((d, i, len(tasks3) - 1))
                    if tasks3:
                        q3, _e3, _d3 = self._expand_joint(tasks3, p)
                        for (d, i, _t), v in zip(owner3, q3):
                            if float(v) > q[d][i]:
                                q[d][i] = float(v)
        return [q[d].copy() for d in range(reps)], diags

    @torch.no_grad()
    def _deep(self, q, ends, n: int):
        """第二层（ply≥2）：对 1-ply 估值最高的 ``p2_topk`` 个动作再展开一层。

        叶子处的 V 由 ``_expand`` 统一批评估。取值用 ``max(q1, max_a2 V)``：
        2-ply 是对 1-ply 的细化，不应因为单点 V 噪声而比 1-ply 更悲观。

        第三层（ply≥3）：
          - ply=3「贪心三层」：沿每个候选**已选中的**最优第二层局面 s'' 再展开一层，
            取 ``max(·, max_a3 V)``。实测比 2-ply 强 +2.62±1.47（200 副），已上线；
          - ply=4「完整三层」（2026-10-31 夜间实验）：对第二层的**每个** a2 都再展开
            一层取全局 max。成本再乘 n2（≈2~4s/步），收益见夜间对照。
        """
        k = max(1, min(int(self.p2_topk), n))
        order = np.argsort(-q)[:k]
        self.stats["deep"] += 1
        for i in order:
            i = int(i)
            g = ends[i]
            if g is None:            # 该分支已终局或超 cap，没有可展开的后续
                continue
            acts2 = g.legal_actions()
            if len(acts2) <= 1:      # 唯一动作：V(s') 已经是它的值
                continue
            q2, e2, _d2 = self._expand(g, g.actor(), acts2)
            if len(q2):
                v2 = float(q2.max())
                if self.ply >= 3:
                    # ply=3 只展开 argmax 那条；ply=4 全部展开
                    branch = ([int(np.argmax(q2))] if self.ply == 3
                              else range(len(q2)))
                    for j in branch:
                        g3 = e2[j]
                        if g3 is None:
                            continue
                        a3 = g3.legal_actions()
                        if len(a3) <= 1:
                            continue
                        q3, _e3, _d3 = self._expand(g3, g3.actor(), a3)
                        if len(q3) and float(q3.max()) > v2:
                            v2 = float(q3.max())
                if v2 > q[i]:
                    q[i] = v2
        return q

    @torch.no_grad()
    def _evaluate(self, game):
        """跑完搜索，返回 (legal, q, prior_p, base_best, V(当前局面))。

        ``choose()`` 与 ``rank()`` 共用这一份逻辑：前者取 argmax，后者取 top-k 供
        网页「大师建议」面板展示。
        """
        p = game.actor()
        base_obs = self._obs_np(game, p)
        lg0, v0 = self._batch_out([(base_obs.tiles, base_obs.glob,
                                    base_obs.head, base_obs.mask)])
        legal = game.legal_actions()
        n = len(legal)
        self.stats["decisions"] += 1

        # 先验：把 mask 内合法位逐一映射回 Action，按 label 对齐 legal_actions
        dim = base_obs.mask.shape[0]
        lab2i = {}
        for idx in range(dim):
            if base_obs.mask[idx]:
                lab2i[index_to_action(base_obs.head, idx, game).label()] = idx
        prior_idx_of = [lab2i[a.label()] for a in legal]   # 每个合法动作的 mask 下标
        probs_all = torch.softmax(lg0[0], dim=-1).cpu().numpy()
        prior_p = np.array([probs_all[j] for j in prior_idx_of], dtype=np.float64)
        base_best = int(np.argmax(prior_p))

        if n <= 1:
            return (legal, np.zeros(n, dtype=np.float64), prior_p,
                    base_best, float(v0[0]))

        reps = self.n_det if self.fair else 1
        qs, diags = [], []
        if reps > 1 and getattr(self, "batched", True):
            # 2026-11-01：n_det 个 determinization 合并成大批前向（延迟主优化）
            qs, diags = self._search_fair_batched(game, p, legal, reps)
            diag = diags[-1] if diags else None
        else:
            diag = None
            for _ in range(reps):
                sim = self._fair_view(game, p) if self.fair else game
                q_d, ends_d, diag_d = self._expand(sim, p, legal)
                if self.ply >= 2:
                    q_d = self._deep(q_d, ends_d, n)
                qs.append(q_d)
                diag = diag_d
                diags.append(diag_d)
        self._last_diag = diag
        self._last_qs = qs          # 逐 determinization 的 Q（PIMC 冲突率诊断用）
        self._last_diags = diags    # 逐 determinization 的推演诊断（聚合理由用）
        q = np.mean(qs, axis=0) if reps > 1 else qs[0]
        return legal, q, prior_p, base_best, float(v0[0])

    def choose(self, game):
        t0 = time.time()
        legal, q, prior_p, base_best, _v0 = self._evaluate(game)
        n = len(legal)
        if n <= 1:
            self.stats["search_s"] += time.time() - t0
            return legal[0]

        # ---- 决策：argmax Q（先验仅作平手裁决 / 可选正则）------------------
        score = q.copy()
        if self.prior_weight > 0.0:
            score = q + self.prior_weight * np.log(np.clip(prior_p, 1e-9, None))
        qmax = float(score.max())
        best = int(np.argmax(score))
        # 平手裁决：与 argmax Q 相差 <1e-9 的动作里取先验最高者
        tie = [i for i in range(n) if score[i] >= qmax - 1e-9]
        if len(tie) > 1:
            best = tie[int(np.argmax(prior_p[tie]))]
        if self.sample_tau > 0.0:
            # τ-采样：策略改进算子的随机化版本（扩自博弈状态覆盖，B4D2）
            z = (score - score.max()) / self.sample_tau
            p = np.exp(z)
            p /= p.sum()
            best = int(self.rng.choice(n, p=p))
        # 与基准策略的一致率统计（诊断项）
        if best == base_best:
            self.stats["agreed"] += 1
        self.stats["search_s"] += time.time() - t0
        if self.verbose:
            spread = float(q.max() - q.min())
            print(f"  [search] n={n} qmax={q.max():+.3f} spread={spread:.3f} "
                  f"chosen={legal[best].label()} base={legal[base_best].label()}")
        return legal[best]

    def rank(self, game, k: int = 4):
        """按 Q 从高到低返回前 k 个合法动作：``([(Action, q), ...], V(当前局面))``。

        供 ``/api/hint`` 展示「搜索·大师」的建议；与 ``choose()`` 同源同料，
        所以面板里的第一项 == 搜索 agent 真正会打的那张牌。
        """
        t0 = time.time()
        legal, q, _prior_p, _base_best, v0 = self._evaluate(game)
        self.stats["search_s"] += time.time() - t0
        if not legal:
            return [], float(v0)
        order = np.argsort(-q)[:max(1, min(int(k), len(legal)))]
        return [(legal[int(i)], float(q[int(i)])) for i in order], float(v0)

    # ---- 推荐理由（算法可解释性） ----------------------------------------
    def rank_explain(self, game, k: int = 4):
        """与 ``rank`` 同源，但为每个候选附**分层推演理由**（面向学牌）。

        理由只用推演事实 + 数值比较，不编造：每行回答一个问题——
          ① 结论：这张牌值多少、比次优高多少；
          ② 依据：N 种假想世界里胡了几次、被胡几次、其余按残局估值；
          ③ 牌面：这张牌在手里是什么结构（孤张/搭子/对子）、牌河现了几张；
          ④ 学习点：一句话把上面的数字翻成可迁移的打法原则。
        """
        legal, q, _prior_p, _base_best, v0 = self._evaluate(game)
        items = explain_items(game, game.actor(), legal, q,
                              getattr(self, "_last_qs", None) or [],
                              getattr(self, "_last_diags", None) or [], k)
        return items, float(v0)


    @staticmethod
    def _lesson(game, p, a, i, order, q, legal, hand, wins, losses, term) -> str:
        """把推演数字翻译成一句可迁移的打法原则（只在证据支持时给出）。"""
        tile = getattr(a, "tile", None)
        kind = str(getattr(a, "kind", ""))
        if kind == HU:
            return "能胡就先胡：这一手是确定的真金白银，别为了做大牌错过"
        if kind in (PENG, MINGGANG, ANGANG, BUGANG):
            return "碰/杠买的是速度，卖的是灵活：手牌还散时别急着定死牌型"
        if kind == PASS:
            return "过牌＝保持现状，通常是等更好的机会"
        if tile is None or tile < 0:
            return "按期望分差最大的走，长期下来每一分都会累积"
        struct = _tile_struct(hand, tile)
        seen = sum(1 for q2 in range(4) for t in game.discards[q2] if t == tile)
        if struct == "孤张" and seen >= 2:
            return "孤张 + 已现多张：先清废牌，既不损牌效又降低被胡风险"
        if struct == "孤张":
            return "孤张优先出手，但它是生张——留意别家已经听牌的迹象"
        if struct in ("对子", "刻子") and term and wins:
            return "宁可拆对子也不冒险：这里是在用牌效换「不被胡」的确定性"
        if struct.endswith("搭子"):
            return "拆搭子是拿听牌速度换安全/番型，看上面的终局数据判断值不值"
        return "按期望分差最大的走，长期下来每一分都会累积"


def explain_items(game, p: int, legal, q, qs, diags, k: int = 4):
    """把 (Q 向量, 逐 determinization 的 Q, 推演诊断) 翻成可读理由。

    2026-11-03 用户口径：**只保留必要的收益与风险说明**——
      · 收益：这张动作的期望分差、与次优/最优的差距；
      · 风险：打牌时的生熟张（点炮敞口）与鸡牌提示。
    不再输出推演统计、V 值解释、牌面结构、学习点等教学层。
    """
    order = np.argsort(-q)[:max(1, min(int(k), len(legal)))]
    items = []
    for rank_i, i in enumerate(order):
        i = int(i)
        a, qv = legal[i], float(q[i])
        lines: list[str] = []
        # ---- 收益：期望分差 + 与次优/最优的差距 -------------------------
        if rank_i == 0 and len(order) > 1:
            j = int(order[1])
            lines.append(f"最优：{a.label()}，期望 {qv:+.2f} 分"
                         f"（次优「{legal[j].label()}」{q[j]:+.2f}）")
        else:
            j0 = int(order[0])
            lines.append(f"第{rank_i + 1}选：{a.label()}，期望 {qv:+.2f} 分"
                         f"（比最优低 {float(q[j0]) - qv:+.2f}）")
        # ---- 风险：仅打牌有——生熟张（点炮敞口）+ 鸡牌 ------------------
        tile = getattr(a, "tile", None)
        kind = str(getattr(a, "kind", ""))
        if tile is not None and tile >= 0 and kind == DISCARD:
            seen = sum(1 for q2 in range(4) for t in game.discards[q2] if t == tile)
            shown = [t for q2 in range(4) for m in game.melds[q2] for t in m[:1]]
            seen += sum(1 for t in shown if t == tile)
            risk = ("已现 %d 张，较安全" % seen if seen >= 2 else
                    "生张，点炮风险较高" if seen == 0 else "已现 1 张")
            if tile in getattr(game, "ji_tiles", ()):
                risk += "；鸡牌"
            lines.append(risk)
        items.append({"action": a, "q": qv, "reason": "；".join(lines),
                      "lines": lines})
    return items
