import os, sys, psutil, time
me = psutil.Process(os.getpid())
anc = set()
try:
    for p in [me] + me.parents():
        anc.add(p.pid)
except Exception:
    pass
rows = []
for p in psutil.process_iter(['pid','name','cmdline','create_time']):
    try:
        cl = " ".join(p.info['cmdline'] or [])
    except Exception:
        continue
    if not cl or p.info['pid'] in anc:
        continue
    if 'selfplay.py' in cl or 'm1_segment_finalize' in cl or 'm1_launch_segment' in cl or 'score_checkpoints' in cl or 'keep_awake' in cl or 'res_monitor' in cl:
        age = (time.time() - p.info['create_time'])/60.0
        try:
            rss = p.memory_info().rss/1e9
        except Exception:
            rss = -1
        rows.append((p.info['pid'], p.ppid(), round(age,1), round(rss,2), p.info['name'], cl[:150]))
for r in rows:
    print(r)
print("COUNT", len(rows))
vm = psutil.virtual_memory()
print("MEM pct=%.1f avail=%.2fGB total=%.2fGB" % (vm.percent, vm.available/1e9, vm.total/1e9))
sw = psutil.swap_memory()
print("SWAP pct=%.1f" % sw.percent)
