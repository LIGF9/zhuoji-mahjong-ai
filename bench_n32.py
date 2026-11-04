"""N32 单决策成本剖析：states / terminals / 各阶段耗时。"""
import sys, time, random
sys.path.insert(0, "/Users/ligf9/WorkBuddy/2026-11-03-18-40-11/zhuoji-mahjong-ai")
import numpy as np
import torch
from zhuoji.dushan import DushanConfig, DushanGame
from zhuoji.net import ZhuojiNet, NetConfig
from zhuoji.rust_search import deep_q, snapshot_of
import dushan_engine as de

# 1) 加载模型（CPU）
ck = torch.load("/Users/ligf9/WorkBuddy/2026-11-03-18-40-11/zhuoji-mahjong-ai/models/dushan_master.pt",
                map_location="cpu", weights_only=False)
cfg = ck["config"]
net = ZhuojiNet(channels=64, blocks=8, hidden=320)
net.load_state_dict(ck["state_dict"])
net.eval()

# 2) 随机推进一局到中局，取一个真实决策点（弃牌、候选多）
g = DushanGame(DushanConfig(), seed=42)
rng = random.Random(7)
for _ in range(200):
    acts = g.legal_actions()
    if not acts or g.result:
        break
    g.step(rng.choice(acts))
    if g.actor() == g.dealer and len(g.legal_actions()) >= 8 and g.wall_left < 40:
        break
p = g.actor()
legal = g.legal_actions()
assert len(legal) > 3, "bad decision point"
print("decision point: actor", p, "n_legal", len(legal), "wall_left", g.wall_left)

# 3) 剖析 deep_q(n32, ply3)
stats = {}
t0 = time.perf_counter()
q, stats = deep_q(net, g, p, legal, n_det=32, ff_cap=24, topk=3, ply=3,
                  seed=1, device="cpu", stats=stats)
t1 = time.perf_counter()
print(f"deep_q n32 ply3: {t1-t0:.3f}s  states={stats.get('states')} "
      f"batches={stats.get('batches')} terminals={stats.get('terminals')}")

# 4) 再跑两次看方差
for i in range(2):
    stats = {}
    t0 = time.perf_counter()
    q, stats = deep_q(net, g, p, legal, n_det=32, ff_cap=24, topk=3, ply=3,
                      seed=100+i, device="cpu", stats=stats)
    print(f"run{i}: {time.perf_counter()-t0:.3f}s states={stats.get('states')} terminals={stats.get('terminals')}")

# 5) 纯前向基准：512 状态一批
tiles = np.random.rand(512, 15, 27).astype(np.float32)
glob = np.random.rand(512, 70).astype(np.float32)
tt = torch.from_numpy(tiles).view(512, 15, 3, 9)
gg = torch.from_numpy(glob)
with torch.no_grad():
    for _ in range(3):
        out = net(tt, gg)
    t0 = time.perf_counter()
    for _ in range(10):
        out = net(tt, gg)
    t1 = time.perf_counter()
print(f"torch CPU forward 512 states: {(t1-t0)/10*1000:.1f} ms")

# 6) Python 重放单条终局的成本（replay_delta 路径）
from zhuoji.rust_search import replay_delta, _KIND_NAME
from zhuoji.rules import Action
snap = snapshot_of(g, p)
wall = snap["wall"]; wpos = snap["wall_pos"]; hands = snap["hands"]
fake_trace = [(0, t % 27) for t in range(60)]
t0 = time.perf_counter()
for _ in range(20):
    replay_delta(g, wall, wpos, hands, fake_trace, p)
t1 = time.perf_counter()
print(f"python replay 60-step terminal: {(t1-t0)/20*1000:.2f} ms")
