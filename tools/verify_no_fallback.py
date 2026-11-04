"""一次性验证：确认 Rust 搜索 + ORT-CoreML 前向在全决策路径上无回落。"""
import io
import random
import sys
from contextlib import redirect_stderr, redirect_stdout

sys.path.insert(0, "/Users/ligf9/zhuoji-mahjong-ai")

import torch
from zhuoji.dushan import DushanConfig, DushanGame
from zhuoji.net import ZhuojiNet
from zhuoji.rust_search import RustSearchAgent

buf_out, buf_err = io.StringIO(), io.StringIO()
with redirect_stdout(buf_out), redirect_stderr(buf_err):
    ck = torch.load("/Users/ligf9/zhuoji-mahjong-ai/models/dushan_master.pt",
                    map_location="cpu", weights_only=False)
    net = ZhuojiNet(channels=64, blocks=8, hidden=320)
    net.load_state_dict(ck["state_dict"])
    net.eval()
    ag = RustSearchAgent(net, seed=1, device="ort", name="大师", ply=2,
                         p2_topk=3, n_det=32, fair=True)

n_ok = 0
with redirect_stdout(buf_out), redirect_stderr(buf_err):
    for g_seed in range(3):                      # 3 副牌
        g = DushanGame(DushanConfig(), seed=g_seed)
        rng = random.Random(7 + g_seed)
        for step in range(400):                  # 每副牌最多 400 步
            acts = g.legal_actions()
            if not acts or g.result:
                break
            if g.actor() in (1, 2):              # 让搜索 AI 替座位 1/2 决策
                a = ag.choose(g)
                assert a is not None and a in acts, "AI 给出非法动作"
                n_ok += 1
                g.step(a)
            else:
                g.step(rng.choice(acts))

log = buf_out.getvalue() + buf_err.getvalue()
bad = [ln for ln in log.splitlines()
       if ("回落" in ln or "hint]" in ln or "失败" in ln or "Error" in ln)]
print(f"决策次数: {n_ok}  |  模式: {ag.stats.get('mode', 'n/a')}")
print(f"搜索统计: states={ag.stats.get('states')} batches={ag.stats.get('batches')}")
print(f"回落/失败痕迹: {bad if bad else '无 ✅'}")
