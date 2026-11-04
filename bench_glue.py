"""胶水成本剖析：snapshot_of / take_batch+feed 往返 / replay_delta。"""
import sys, time, random
sys.path.insert(0, "/Users/ligf9/WorkBuddy/2026-11-03-18-40-11/zhuoji-mahjong-ai")
import numpy as np
import torch
from zhuoji.dushan import DushanConfig, DushanGame
from zhuoji.net import ZhuojiNet
from zhuoji.rust_search import deep_q, snapshot_of, _BatchRunner
import dushan_engine as de

ck = torch.load("/Users/ligf9/WorkBuddy/2026-11-03-18-40-11/zhuoji-mahjong-ai/models/dushan_master.pt",
                map_location="cpu", weights_only=False)
net = ZhuojiNet(channels=64, blocks=8, hidden=320)
net.load_state_dict(ck["state_dict"]); net.eval()

g = DushanGame(DushanConfig(), seed=42)
rng = random.Random(7)
for _ in range(200):
    acts = g.legal_actions()
    if not acts or g.result: break
    g.step(rng.choice(acts))
    if g.actor() == g.dealer and len(g.legal_actions()) >= 8 and g.wall_left < 40:
        break
p = g.actor(); legal = g.legal_actions()

# snapshot_of 成本
t0 = time.perf_counter()
for _ in range(200):
    snap = snapshot_of(g, p)
t1 = time.perf_counter()
print(f"snapshot_of: {(t1-t0)/200*1000:.2f} ms")

# take_batch/feed 往返成本（空转 deep search 的一层）
sn = de.PySearch(snap, p, [(0, 5), (0, 6), (0, 7)], 32, 24, 1)
nb = 0
t0 = time.perf_counter()
runner = _BatchRunner(net, "cpu")
t_init = time.perf_counter() - t0
print(f"_BatchRunner init: {t_init*1000:.1f} ms")
t0 = time.perf_counter()
guard = 0
while not sn.finished() and guard < 500:
    guard += 1
    if sn.needs_net():
        tiles, glob, head, mask, kind = sn.take_batch()
        n = head.shape[0]
        nb += 1
        acts_ = np.zeros(n, dtype=np.uint8); vals = np.zeros(n, dtype=np.float32)
        sn.feed(acts_.tolist(), vals)
        continue
    break
t1 = time.perf_counter()
print(f"PySearch glue-only loop: {(t1-t0)*1000:.1f} ms, batches={nb}")

# torch.set_num_threads 对前向的影响
for nth in (1, 2, 4):
    torch.set_num_threads(nth)
    tiles = np.random.rand(256, 15, 27).astype(np.float32)
    tt = torch.from_numpy(tiles).view(256, 15, 3, 9)
    gg = torch.from_numpy(np.random.rand(256, 70).astype(np.float32))
    with torch.no_grad():
        net(tt, gg)
        t0 = time.perf_counter()
        for _ in range(10):
            net(tt, gg)
        dt = (time.perf_counter()-t0)/10
    print(f"torch threads={nth}: 256 states {dt*1000:.1f} ms ({dt/256*1e6:.0f} us/state)")
