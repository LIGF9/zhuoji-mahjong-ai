#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""PPO 势函数平稳性 / 量纲一致性的防回归测试（只读 AST + 常量核对）。

外部评审 §2 担心「Φ=8·V_θ 的 V_θ 每轮更新会让奖励信号非平稳」。
核查 train_ppo.py 的实际实现后可以确认**该隐患在本项目中不存在**，
且是架构级保证而非约定：

1. rollout 在 ``spawn`` 子进程中执行：``rollout_shard(ckpt, ...)`` 从**磁盘
   固定的 init 检查点** ``build_net(ckpt)`` → ``net.eval()``，逐状态把
   ``out["value"]`` 写进 npz 的 ``val`` 数组；子进程里没有任何 optimizer。
2. 父进程在 **所有分片 future 都收齐之后** 才 ``build_net`` + ``ppo_update``。
   整个 rollout 期间父进程根本还没有可更新的网络对象。
3. ``gae()`` 的 ``phi = PHI_SCALE * val[idxs]`` 直接读 npz 落盘的固定值，
   不重算、不回读更新后的网络。
4. 每轮 PPO 只有「一次 rollout + 一次更新」；链式初始化 ``ppo_init(r)``
   换的是**轮边界**的 init 检查点，正是评审要求的「只在轮边界更新 V_θ」。

本测试用 AST 锁死上述顺序与数据来源，防止以后重构（例如把更新挪进
rollout 循环、改成在线 V 重算）悄悄引入非平稳性。

另外核对 ``PHI_SCALE`` 与 ``train_exit.VALUE_SCALE`` / ``zhuoji.reward.PHI_SCALE``
三者相等——这是驳回「Φ 系数 4/8/16 敏感性扫描」的代码级依据：8 不是可调
超参，而是 value head 训练目标 maxQ/8 的**逆量纲常数**；单独改 Φ 会让
``RT = A + Φ·V`` 与 ``mse(V, RT/8)`` 不再自洽。

用法::
    .venv/bin/python tests_ppo_stationarity.py
"""
from __future__ import annotations

import ast
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
SRC = ROOT / "train_ppo.py"


def calls_in(node):
    for x in ast.walk(node):
        if isinstance(x, ast.Call):
            yield x


def fname(c: ast.Call) -> str:
    f = c.func
    if isinstance(f, ast.Name):
        return f.id
    if isinstance(f, ast.Attribute):
        return f.attr
    return ""


def arg0(c: ast.Call) -> str:
    if not c.args:
        return ""
    x = c.args[0]
    if isinstance(x, ast.Name):
        return x.id
    if isinstance(x, ast.Attribute) and isinstance(x.value, ast.Name):
        return f"{x.value.id}.{x.attr}"
    return ast.dump(x)


def func(tree, name):
    for n in tree.body:
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == name:
            return n
    raise AssertionError(f"未找到函数 {name}")


def const(tree, name):
    for n in tree.body:
        if isinstance(n, ast.Assign) and any(
                isinstance(t, ast.Name) and t.id == name for t in n.targets):
            return ast.literal_eval(n.value)
    raise AssertionError(f"未找到常量 {name}")


def main() -> int:
    tree = ast.parse(SRC.read_text(encoding="utf-8"))

    # ---- 1) rollout_shard：V 来自自身固定 net，且无 optimizer ----
    shard = func(tree, "rollout_shard")
    assert shard.args.args and shard.args.args[0].arg == "ckpt", (
        "rollout_shard 首形参必须是 ckpt（固定检查点路径）")
    builds = [c for c in calls_in(shard) if fname(c) == "build_net"]
    assert builds, "rollout_shard 未 build_net"
    assert arg0(builds[0]) == "ckpt", (
        f"rollout 必须从形参 ckpt 建网（实际 {arg0(builds[0])}）；"
        "若改用全局/共享 net，rollout 内 V 会随更新漂移")
    names = [fname(c) for c in calls_in(shard)]
    # 注意：游戏状态机也有 g.step(Action)，不能按函数名误判为 optimizer.step
    bad = []
    for c in calls_in(shard):
        if fname(c) in ("backward", "zero_grad"):
            bad.append(fname(c))
        if (fname(c) == "step" and isinstance(c.func, ast.Attribute)
                and isinstance(c.func.value, ast.Name)
                and c.func.value.id in ("opt", "optimizer")):
            bad.append("optimizer.step")
    assert not bad, f"rollout_shard 内出现参数更新 {bad} ⇒ 势函数非平稳"
    assert "eval" in names, "rollout 网络必须 eval()（BN/Dropout 也会影响 V）"

    # ---- 2) main：更新网络在所有 rollout future 之后才创建 ----
    mn = func(tree, "main")
    seq = [(c.lineno, fname(c), arg0(c)) for c in calls_in(mn)]
    submits = [x for x in seq if x[1] == "submit"]
    assert submits, "main 未提交 rollout 分片"
    assert any("rollout_shard" in ast.dump(c) for c in calls_in(mn)
               if fname(c) == "submit"), "submit 的目标不是 rollout_shard"
    upd_builds = [x for x in seq if x[1] == "build_net" and x[2] == "a.init"]
    assert upd_builds, "main 未找到用于 PPO 更新的 build_net(a.init, ...)"
    upd_line = min(x[0] for x in upd_builds)
    assert max(x[0] for x in submits) < upd_line, (
        "PPO 更新网络在 rollout 提交之前创建 ⇒ rollout 期间 V 可能漂移")
    upd_calls = [x for x in seq if x[1] == "ppo_update"]
    assert upd_calls and upd_line < min(x[0] for x in upd_calls), (
        "ppo_update 顺序异常")
    print(f"[OK] 顺序：submit(rollout) L{max(x[0] for x in submits)} "
          f"< build_net(更新网) L{upd_line} < ppo_update L{min(x[0] for x in upd_calls)}")

    # ---- 3) gae：phi 只读落盘的 val，不重算网络 ----
    gae = func(tree, "gae")
    gnames = [fname(c) for c in calls_in(gae)]
    assert "build_net" not in gnames and "forward" not in gnames, (
        "gae() 内不得重算 V；Φ 必须来自 rollout 落盘的固定 val 数组")
    src_gae = ast.get_source_segment(SRC.read_text(encoding="utf-8"), gae) or ""
    assert "PHI_SCALE * val" in src_gae, "Φ 未按固定 val 计算"
    print("[OK] gae(): phi = PHI_SCALE * val[idxs]，val 来自 npz（rollout 时固定）")

    # ---- 4) 量纲一致性：Φ 不是可调超参 ----
    phi = const(tree, "PHI_SCALE")
    from train_exit import VALUE_SCALE
    from zhuoji.reward import PHI_SCALE as REWARD_PHI
    assert phi == VALUE_SCALE == REWARD_PHI == 8.0, (
        f"量纲失配 PHI_SCALE={phi} VALUE_SCALE={VALUE_SCALE} reward={REWARD_PHI}")
    print(f"[OK] 量纲一致：PHI_SCALE = VALUE_SCALE = reward.PHI_SCALE = {phi}")

    print("\n结论：V_θ 在每次 rollout 内冻结（spawn 子进程 + 磁盘 ckpt + 更新在收齐之后），"
          "\n      轮边界才随 ppo_init 更换 ⇒ 评审 §2 的非平稳性隐患不成立；"
          "\n      Φ=8 是 maxQ/8 的逆量纲常数，不是可扫描超参。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
