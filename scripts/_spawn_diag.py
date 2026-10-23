import multiprocessing as mp
import sys

print("step1 启动", flush=True)
import os
print("step2 os ok", flush=True)
import numpy
print("step3 numpy ok pid=%d" % os.getpid(), flush=True)

import torch
print("step4 torch ok  threads=%d" % torch.get_num_threads(), flush=True)


def _sq(x):
    return x * x


if __name__ == "__main__":
    print("step5 进 main", flush=True)
    ctx = mp.get_context("spawn")
    with ctx.Pool(4) as p:
        r = p.map(_sq, range(8))
    print("step6 mp Pool ok", r, flush=True)
    print("step7 全部通过", flush=True)
