"""搜索进程池：把一次决策的 N 个 determinization 摊到多个核上跑。

为什么需要它
------------
n32 建议的算力本质上是「32 棵互相独立的树」：每棵树各自洗牌、各自展开、
各自快进。它们之间没有任何共享状态——天然适合并行，但 Python 的 GIL 让
线程池只能吃到 torch 释放 GIL 的那部分（实测加速 <1.3×）。

所以用**进程池**：主进程只做「快照 + 拆活 + 平均」，worker 进程各自持有
一份模型副本，用 ``torch.set_num_threads(1)`` 避免线程超订。

设计要点
--------
* 快照进、Q 数组出：worker 拿到的是 ``ZhuojiGame`` 的 pickle 快照，
  完全独立于对局进程——搜索期间玩家出牌、对局推进都不受影响（这正是
  「点了打不出去」那个 bug 的根治办法）。
* 结果与串行等价：每个 worker 内部仍走 ``_search_fair_batched``，
  determinization 仍是「均匀重排不可见牌」的公平世界，最后对 Q 逐动作平均。
* 任何一步失败（spawn 不可用 / worker 崩了 / 超时）都回落到进程内搜索，
  绝不把「池子坏了」变成「没有建议」。
"""
from __future__ import annotations

import math
import multiprocessing as mp
import pickle
import threading
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

_WORKER_AGENTS: dict = {}


def _worker_agent(stem: str, ply: int, topk: int, n_det: int, seed: int, model_dir: str,
                  threads: int = 1):
    """worker 内按 (模型, 参数) 缓存一个 SearchAgent（避免每次重建网络）。"""
    key = (stem, ply, topk)          # n_det 由每次任务的 chunk 决定，不进缓存键
    ag = _WORKER_AGENTS.get(key)
    if ag is None:
        import torch

        from zhuoji.net import build_from_ckpt
        from zhuoji.search_agent import SearchAgent

        torch.set_num_threads(max(1, int(threads)))
        path = Path(model_dir) / f"{stem}.pt"
        ck = torch.load(path, map_location="cpu", weights_only=False)
        net, _ = build_from_ckpt(ck)
        net.eval()
        ag = SearchAgent(net, seed=seed, device="cpu", ply=ply, p2_topk=topk,
                         fair=True, n_det=1, batched=True)
        _WORKER_AGENTS.clear()            # 只留一份网络副本
        _WORKER_AGENTS[key] = ag
    else:
        ag.reset(seed)
    return ag


def _worker_warm(stem: str, ply: int, topk: int, model_dir: str, threads: int = 1) -> bool:
    _worker_agent(stem, ply, topk, 1, 0, model_dir, threads)
    return True


def _worker_search(task: dict) -> dict:
    """跑一段 determinization（chunk 个），返回 Q 矩阵 + 诊断。"""
    t0 = time.perf_counter()
    game = pickle.loads(task["game"])
    ag = _worker_agent(task["stem"], task["ply"], task["topk"], task["n_det"],
                       task["seed"], task["model_dir"], task.get("threads", 1))
    legal = game.legal_actions()
    p = game.actor()
    qs, diags = ag._search_fair_batched(game, p, legal, task["n_det"])
    return {"qs": qs, "diags": diags,
            "labels": [a.label() for a in legal],
            "secs": time.perf_counter() - t0}


class SearchPool:
    """懒启动的搜索进程池（spawn）。"""

    def __init__(self, workers: int | None = None, model_dir: str | Path = "models",
                 threads_per_worker: int = 1):
        cpu = mp.cpu_count() or 4
        if workers is None:
            # 留 2 个核给对局主循环 / HTTP / 前端轮询；上限 8（每 worker 约 200 MB）。
            # 实测 10 核机器：8×1 线程最快（n32 ≈ 1.2 s），6×1 稍慢，4×2 居中。
            workers = max(1, min(8, cpu - 2))
        self.workers = int(workers)
        self.threads_per_worker = max(1, int(threads_per_worker))
        self.model_dir = str(model_dir)
        self._exe: ProcessPoolExecutor | None = None
        self._lock = threading.Lock()
        self._broken = False
        self._warm = False
        self.stats = {"calls": 0, "pool_ms": 0.0, "tasks": 0}

    # -- 生命周期 -----------------------------------------------------------
    def _ensure(self) -> ProcessPoolExecutor | None:
        if self._broken:
            return None
        with self._lock:
            if self._exe is None:
                try:
                    self._exe = ProcessPoolExecutor(
                        max_workers=self.workers,
                        mp_context=mp.get_context("spawn"))
                except Exception:
                    self._broken = True
                    return None
            return self._exe

    def warm(self, stem: str, ply: int = 3, topk: int = 3) -> None:
        """预热点亮全部 worker：把「第一次建议」的 3~5 s 启动成本挪到后台。"""
        exe = self._ensure()
        if exe is None:
            return
        try:
            futs = [exe.submit(_worker_warm, stem, ply, topk, self.model_dir,
                               self.threads_per_worker)
                    for _ in range(self.workers)]
            for f in futs:
                f.result(timeout=180)
            self._warm = True
        except Exception:
            self._broken = True

    def warm_async(self, stem: str, ply: int = 3, topk: int = 3) -> None:
        if self._warm or self._broken:
            return
        threading.Thread(target=self.warm, args=(stem, ply, topk), daemon=True).start()

    def shutdown(self) -> None:
        with self._lock:
            if self._exe is not None:
                self._exe.shutdown(wait=False, cancel_futures=True)
                self._exe = None

    # -- 主入口 -------------------------------------------------------------
    def search(self, game, stem: str, ply: int = 3, topk: int = 3,
               n_det: int = 32, seed: int = 7, timeout: float = 60.0) -> dict | None:
        """把 ``n_det`` 个 determinization 摊到 worker 上，返回聚合后的结果。

        返回 ``{"q": np.ndarray, "qs": [...], "diags": [...], "labels": [...]}``；
        池子不可用时返回 ``None``（调用方回落到进程内搜索）。
        """
        if self._broken or self.workers <= 1:
            return None
        exe = self._ensure()
        if exe is None:
            return None
        n_chunks = self.workers
        per = max(1, math.ceil(n_det / n_chunks))
        chunks = []
        left = n_det
        while left > 0:
            k = min(per, left)
            chunks.append(k)
            left -= k
        snap = pickle.dumps(game, protocol=pickle.HIGHEST_PROTOCOL)
        t0 = time.perf_counter()
        tasks = [{"game": snap, "stem": stem, "ply": ply, "topk": topk,
                  "n_det": k, "seed": seed * 1000 + i * 17, "model_dir": self.model_dir,
                  "threads": self.threads_per_worker}
                 for i, k in enumerate(chunks)]
        try:
            futs = [exe.submit(_worker_search, t) for t in tasks]
            outs = [f.result(timeout=timeout) for f in futs]
        except Exception:
            self._broken = True
            return None
        labels = outs[0]["labels"]
        n = len(labels)
        qs: list = []
        diags: list = []
        for o in outs:
            if o["labels"] != labels:      # 快照不一致（理论上不可能）→ 回落
                return None
            qs.extend(o["qs"])
            diags.extend(o["diags"])
        qs = qs[:n_det]
        diags = diags[:n_det]
        if not qs or len(qs[0]) != n:
            return None
        self.stats["calls"] += 1
        self.stats["tasks"] += len(tasks)
        self.stats["pool_ms"] += (time.perf_counter() - t0) * 1000.0
        return {"q": np.mean(np.asarray(qs, dtype=np.float64), axis=0),
                "qs": qs, "diags": diags, "labels": labels,
                "secs": time.perf_counter() - t0}
