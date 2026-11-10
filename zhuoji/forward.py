"""前向后端抽象：torch(MPS/CPU) 与 onnxruntime(CoreML / CPU)。

背景（2026-11-02 实测）
----------------------
Rust 内核把「模拟 + 编码 + 克隆」压到 3~11 ms 之后，**网络前向占单决策 96~97%**
（n8 117ms 里 114ms、n32 318ms 里 307ms）。同一批状态上比过三个后端：

    B=209   torch-MPS 3.26ms(15.6µs/状态)   ORT-CoreML 1.51ms(7.2µs)   ORT-CPU 84ms
    B=832   torch-MPS 10.7ms(12.9µs)        ORT-CoreML 5.98ms(7.2µs)   ORT-CPU 339ms

CoreML 的收益集中在**小批**（B=1 时 0.18ms vs 2.13ms，11.7×）——这正是搜索里
最常见的形状（一批几十到几百个叶子）。CPU 前向只在极小批（B≤32）时可用。

精度：CoreML 走 fp16，真实局面上 max|ΔV| 1.2e-3（mean 4.8e-4）、
discard argmax 100% 一致；ORT-CPU 是 fp32 精确（4.8e-7）。

用法
----
    fwd = build_forward(net, "ort")          # CoreML
    out = fwd.raw(tiles_np, glob_np)         # {"discard","self_kong","respond","value"}
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
ONNX_DIR = ROOT / "models" / "onnx"

# ONNX 输出顺序（与 export_onnx 的 output_names 一致）
HEAD_NAMES = ("discard", "self_kong", "respond", "value")


# ---------------------------------------------------------------- 导出 ----
def export_onnx(net, path: str | Path, batch: int = 8, opset: int = 17) -> Path:
    """把一个 ZhuojiNet 导出成动态批的 ONNX。"""
    import torch

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    net_cpu = net.to("cpu").eval()

    class _Wrap(torch.nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m

        def forward(self, tiles, glob):
            o = self.m(tiles, glob)
            return tuple(o[n] for n in HEAD_NAMES)

    tiles = torch.zeros(batch, net_cpu.in_channels, 27)
    glob = torch.zeros(batch, net_cpu.global_dim)
    # dynamo=False：走传统 TorchScript 导出器。torch>=2.15 默认 dynamo=True，
    # 只支持 opset>=18 且会把 dynamic_axes 降级转换（实测 opset 17 转换直接失败），
    # 还会牵出一串 torchvision 未安装的注册警告；旧导出器对 opset 17 + 动态批稳定。
    wrapper = _Wrap(net_cpu).eval()   # 包装器本身也要 eval，否则报 training mode 警告
    torch.onnx.export(wrapper, (tiles, glob), str(path),
                      input_names=["tiles", "glob"], output_names=list(HEAD_NAMES),
                      dynamic_axes={k: {0: "B"} for k in
                                    ("tiles", "glob", *HEAD_NAMES)},
                      opset_version=opset, do_constant_folding=True,
                      dynamo=False)
    return path


def ensure_onnx(net, stem: str | None = None) -> Path:
    """取（必要时生成）该网络的 ONNX；``stem`` 为模型名，决定缓存文件名。"""
    stem = stem or getattr(net, "onnx_stem", None) or "adhoc"
    path = ONNX_DIR / f"{stem}.onnx"
    pt = ROOT / "models" / f"{stem}.pt"
    if path.exists() and (not pt.exists() or path.stat().st_mtime >= pt.stat().st_mtime):
        return path
    export_onnx(net, path)
    try:
        net.onnx_stem = stem
    except Exception:  # noqa: BLE001
        pass
    return path


# ---------------------------------------------------------------- 后端 ----
class TorchForward:
    """torch 后端（MPS/CPU）——保持既有行为，供 A/B 对照。"""

    mode = "torch"

    def __init__(self, net, device: str = "mps"):
        # 不在这里 .to(device)：net 可能是调用方共享的（同进程多个 agent 都用它），
        # 搬走会让别人的 CPU 前向直接崩（NNPack "Mismatched Tensor types"）。
        self.net = net
        self.device = device
        self._torch = __import__("torch")

    def raw(self, tiles_np: np.ndarray, glob_np: np.ndarray) -> dict:
        t = self._torch.from_numpy(np.ascontiguousarray(tiles_np)).to(self.device)
        g = self._torch.from_numpy(np.ascontiguousarray(glob_np)).to(self.device)
        with self._torch.no_grad():
            o = self.net(t, g)
        return {k: v.detach().cpu().numpy() for k, v in o.items()}


class OrtForward:
    """onnxruntime 后端。``provider``: "coreml" | "cpu"。

    两个必须踩过才知道的坑（2026-11-02 实测）：

    1. **会话必须缓存**：CoreML 建会话要编译图（~150~220 ms）。搜索里
       ``_BatchRunner`` 是「每次决策建一个」，不缓存会把 300 ms 的搜索拖成 1.5 s。
    2. **一个会话只能见一个形状**：CoreML EP 在**输入形状变化时重新编译**
       （B=64 稳态 0.6 ms，换到别的形状再回来又变 54 ms；轮换 8 次全程 54~143 ms）。
       所以按形状缓存会话（LRU，上限 8 个，每个约 6 MB）——同一形状第二次起就是
       0.6~2.3 ms，比 torch-MPS 快 3~4 倍。
    """

    mode = "ort"
    _SHAPES: "OrderedDict[tuple, object]" = None      # (路径, provider, 线程, 形状) → 会话
    _CAP = 8
    # 搜索的批形状必须**收敛到极少几个**：CoreML EP 每次遇到新形状都要重编译
    # （150~240 ms）。实测 n32 一次决策会出现 22 种形状 → 1.6 s 全花在编译上。
    # 固定梯子后只剩 5 种形状，每种编一次；不足的补零，最多浪费 ~2× 行数。
    LADDER = (128, 256, 512, 1024, 2048)

    def __init__(self, onnx_path: str | Path, provider: str = "coreml", threads: int = 1,
                 warm: bool = False, ch: int = 15, gdim: int = 70):
        self.path = str(Path(onnx_path).resolve())
        self.provider = provider
        self.threads = max(1, int(threads))
        self.mode = f"ort-{provider}"
        # 2026-11-10：输入维度随编码器版本变（v1=15/70，v3=63/130）。
        # 预热必须按权重真实维度造张量，否则 CoreML 会拒绝全部形状。
        self.ch = int(ch)
        self.gdim = int(gdim)
        if OrtForward._SHAPES is None:
            from collections import OrderedDict  # noqa: PLC0415
            OrtForward._SHAPES = OrderedDict()
        if warm:
            self.warm(self.LADDER)   # 显式预热：建会话会编译，别放在决策路径里

    # -- 会话管理 -----------------------------------------------------------
    def _new_session(self):
        import onnxruntime as ort

        so = ort.SessionOptions()
        so.log_severity_level = 3        # 静音 CoreML GetCapability 的信息级输出（仅 ERROR 以上）
        so.intra_op_num_threads = self.threads
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        provs = (["CoreMLExecutionProvider", "CPUExecutionProvider"]
                 if self.provider == "coreml" else ["CPUExecutionProvider"])
        return ort.InferenceSession(self.path, so, providers=provs)

    def warm(self, shapes) -> None:
        """按形状预热（每形状一次编译）；形状与真实模型不符时静默跳过。"""
        for b in shapes:
            try:
                self.raw(np.zeros((b, self.ch, 27), np.float32),
                         np.zeros((b, self.gdim), np.float32))
            except Exception:  # noqa: BLE001
                pass

    def _session(self, b: int):
        key = (self.path, self.provider, self.threads, b)
        s = OrtForward._SHAPES.get(key)
        if s is None:
            s = self._new_session()
            OrtForward._SHAPES[key] = s
            if len(OrtForward._SHAPES) > self._CAP:      # LRU：丢掉最久未用的
                OrtForward._SHAPES.popitem(last=False)
        else:
            OrtForward._SHAPES.move_to_end(key)
        return s

    def raw(self, tiles_np: np.ndarray, glob_np: np.ndarray) -> dict:
        n = int(tiles_np.shape[0])
        step = self.LADDER[-1]
        if n <= step:                                   # 常见情况：一次调用
            return self._run_padded(tiles_np, glob_np, self._ceil_shape(n))
        out = None                                      # 超大批：按 step 切块
        for s in range(0, n, step):
            part = self._run_padded(tiles_np[s:s + step], glob_np[s:s + step], step)
            if out is None:
                out = {k: np.empty((n,) + v.shape[1:], dtype=v.dtype) for k, v in part.items()}
            m = min(step, n - s)
            for k, v in part.items():
                out[k][s:s + m] = v[:m]
        return out

    def _ceil_shape(self, n: int) -> int:
        for cand in self.LADDER:
            if n <= cand:
                return cand
        return self.LADDER[-1]

    def _run_padded(self, tiles_np, glob_np, target: int) -> dict:
        n = int(tiles_np.shape[0])
        if n == target:
            t_in, g_in = tiles_np, glob_np
        else:
            t_in = np.zeros((target, tiles_np.shape[1], tiles_np.shape[2]), np.float32)
            g_in = np.zeros((target, glob_np.shape[1]), np.float32)
            t_in[:n] = tiles_np
            g_in[:n] = glob_np
        sess = self._session(target)
        outs = sess.run(None, {
            "tiles": np.ascontiguousarray(t_in, dtype=np.float32),
            "glob": np.ascontiguousarray(g_in, dtype=np.float32)})
        return {k: v[:n] for k, v in zip(HEAD_NAMES, outs)}


def build_forward(net, device: str = "mps", stem: str | None = None, threads: int = 1):
    """device: "cpu" / "mps" / "ort" / "ort-cpu"（"ort" = CoreML）。"""
    d = (device or "cpu").lower()
    if d.startswith("ort"):
        prov = "cpu" if d.endswith("cpu") else "coreml"
        return OrtForward(ensure_onnx(net, stem), prov, threads)
    return TorchForward(net, d)
