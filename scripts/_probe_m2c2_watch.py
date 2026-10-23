"""巡检探针：识别 selfplay 训练进程树（排除自身祖先链），并报告内存。"""
import os
import sys
import psutil

SELF = os.getpid()


def ancestors(pid):
    out = set()
    try:
        p = psutil.Process(pid)
        while True:
            out.add(p.pid)
            pp = p.parent()
            if pp is None:
                break
            p = pp
    except Exception:
        pass
    return out


BAD = ancestors(SELF) | {SELF}

hits = []
for p in psutil.process_iter(["pid", "name", "cmdline", "create_time"]):
    try:
        cl = " ".join(p.info["cmdline"] or [])
    except Exception:
        continue
    if "selfplay.py" in cl and p.info["pid"] not in BAD:
        hits.append(p)

print("== selfplay candidates ==")
print("count:", len(hits))
tree_rss = 0.0
for p in hits:
    try:
        rss = p.memory_info().rss / 2**30
        age = (psutil.time.time() - p.info["create_time"]) / 60.0
        print(f"pid={p.pid} rss={rss:.2f}GB age={age:.1f}min ppid={p.ppid()}")
        print("  cmd:", " ".join(p.cmdline())[:400])
        tree_rss += rss
        for c in p.children(recursive=True):
            try:
                crss = c.memory_info().rss / 2**30
                tree_rss += crss
                print(f"   child pid={c.pid} rss={crss:.2f}GB name={c.name()}")
            except Exception:
                pass
    except psutil.NoSuchProcess:
        print(f"pid={p.pid} process PID not found (probe echo)")

print(f"tree_rss_total={tree_rss:.2f}GB")

print("== other relevant ==")
for p in psutil.process_iter(["pid", "cmdline"]):
    try:
        cl = " ".join(p.info["cmdline"] or [])
    except Exception:
        continue
    if p.info["pid"] in BAD:
        continue
    for kw in ("m1_segment_finalize", "m1_segment_judge", "score_checkpoints", "keep_awake", "res_monitor", "m1_launch_segment"):
        if kw in cl:
            try:
                print(f"pid={p.pid} rss={p.memory_info().rss/2**30:.2f}GB kw={kw}")
                print("  cmd:", cl[:300])
            except Exception:
                pass
            break

vm = psutil.virtual_memory()
print(f"== mem == total={vm.total/2**30:.2f}GB used={vm.used/2**30:.2f}GB avail={vm.available/2**30:.2f}GB percent={vm.percent}%")
