"""MPS 前向吞吐基准：单进程 & 多进程并发（顶层 worker，spawn 安全）。"""
import sys, time
sys.path.insert(0, "/Users/ligf9/WorkBuddy/2026-11-03-18-40-11/zhuoji-mahjong-ai")
import numpy as np
import torch


def worker(q, idx, batch, iters):
    torch.set_num_threads(2)
    dev = "mps"
    x_t = torch.randn(batch, 15, 3, 9, device=dev)
    x_g = torch.randn(batch, 70, device=dev)
    ck = torch.load("/Users/ligf9/WorkBuddy/2026-11-03-18-40-11/zhuoji-mahjong-ai/models/dushan_master.pt",
                    map_location="cpu", weights_only=False)
    from zhuoji.net import ZhuojiNet
    net = ZhuojiNet(channels=64, blocks=8, hidden=320)
    net.load_state_dict(ck["state_dict"])
    net.eval().to(dev)
    with torch.no_grad():
        net(x_t, x_g)
        torch.mps.synchronize()
        t0 = time.perf_counter()
        for _ in range(iters):
            out = net(x_t, x_g)
        torch.mps.synchronize()
        dt = time.perf_counter() - t0
    q.put((idx, dt))


def bench(tag, nproc=1, batch=256, iters=30):
    import multiprocessing as mp
    q = mp.Queue()
    ps = [mp.Process(target=worker, args=(q, i, batch, iters)) for i in range(nproc)]
    t0 = time.perf_counter()
    for p in ps:
        p.start()
    res = [q.get() for _ in range(nproc)]
    for p in ps:
        p.join()
    wall = time.perf_counter() - t0
    total = nproc * iters * batch
    print(f"{tag}: wall={wall:.2f}s agg={total/wall:.0f} states/s "
          f"({total/wall*32.4e-6:.2f} TFLOPS eq)", flush=True)


if __name__ == "__main__":
    bench("MPS x1 batch256", 1, 256, 40)
    bench("MPS x1 batch512", 1, 512, 40)
    bench("MPS x4 batch256", 4, 256, 40)
    bench("MPS x8 batch256", 8, 256, 40)
