# -*- coding: utf-8 -*-
"""m2b1 巡检探针：识别 selfplay 训练树与判读进程，排除自身祖先链。"""
import os, sys, time
import psutil

SELF = os.getpid()


def ancestors(pid):
    out = set()
    try:
        p = psutil.Process(pid)
        while p is not None:
            out.add(p.pid)
            try:
                p = p.parent()
            except Exception:
                break
    except Exception:
        pass
    return out


EXCL = ancestors(SELF)
EXCL.add(0)


def cmdline_of(p):
    try:
        return " ".join(p.cmdline())
    except Exception:
        return ""


procs = []
for p in psutil.process_iter(["pid", "name", "create_time", "memory_info"]):
    pid = p.info["pid"]
    if pid in EXCL:
        continue
    cl = cmdline_of(p)
    if not cl:
        continue
    low = cl.lower()
    if "selfplay.py" in low or "m1_launch_segment" in low or "m1_segment_finalize" in low \
            or "m1_segment_judge" in low or "score_checkpoints" in low:
        rss = 0.0
        try:
            rss = p.info["memory_info"].rss / 1024 ** 3
        except Exception:
            pass
        procs.append((pid, p.info["name"], p.info["create_time"], rss, cl[:220]))

print("=== matched processes: %d ===" % len(procs))
now = time.time()
for pid, name, ct, rss, cl in sorted(procs):
    print("pid=%d name=%s age=%.1fmin rss=%.2fGB" % (pid, name, (now - ct) / 60.0, rss))
    print("   " + cl)

# selfplay 树
sp = [x for x in procs if "selfplay.py" in x[4].lower()]
print("\n=== selfplay count: %d ===" % len(sp))
for pid, name, ct, rss, cl in sp:
    kids = []
    try:
        kids = psutil.Process(pid).children(recursive=True)
    except Exception:
        pass
    tot = rss + sum((k.memory_info().rss / 1024 ** 3) for k in kids if k.is_running())
    print("selfplay pid=%d tree_rss=%.2fGB descendants=%d" % (pid, tot, len(kids)))

vm = psutil.virtual_memory()
print("\n=== memory: %.1f%% used, avail %.2fGB / total %.2fGB ==="
      % (vm.percent, vm.available / 1024 ** 3, vm.total / 1024 ** 3))
