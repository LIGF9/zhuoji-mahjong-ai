import os, sys, psutil, time

SELF = os.getpid()
anc = set()
p = psutil.Process(SELF)
while True:
    try:
        anc.add(p.pid)
        p = p.parent()
        if p is None: break
    except Exception:
        break

def cmd(pr):
    try: return " ".join(pr.cmdline())
    except Exception: return ""

selfplays, judges, others = [], [], []
for pr in psutil.process_iter(['pid','name']):
    try:
        c = cmd(pr)
    except Exception:
        continue
    if not c: continue
    if pr.pid in anc: continue
    if 'selfplay.py' in c:
        selfplays.append(pr)
    elif ('m1_segment_finalize' in c or 'score_checkpoints' in c or 'm1_segment_judge' in c) and 'grep' not in c:
        judges.append(pr)

print("=== SELFTREE ANCESTORS:", sorted(anc))
print("=== SELFPLAY COUNT:", len(selfplays))
for pr in selfplays:
    try:
        with pr.oneshot():
            tree = [pr] + pr.children(recursive=True)
            rss = 0
            for t in tree:
                try: rss += t.memory_info().rss
                except Exception: pass
            print(f"  pid={pr.pid} ppid={pr.ppid()} age={time.time()-pr.create_time():.0f}s rss={pr.memory_info().rss/2**30:.2f}GB tree={len(tree)}procs treeRSS={rss/2**30:.2f}GB")
            print("   cmd:", " ".join(pr.cmdline())[:300])
    except Exception as e:
        print("  err", e)

print("=== JUDGE-LIKE COUNT:", len(judges))
for pr in judges:
    print(f"  pid={pr.pid} cmd=", " ".join(pr.cmdline())[:200])

vm = psutil.virtual_memory()
print(f"=== MEM total={vm.total/2**30:.2f}GB used={vm.used/2**30:.2f}GB avail={vm.available/2**30:.2f}GB percent={vm.percent}%")
