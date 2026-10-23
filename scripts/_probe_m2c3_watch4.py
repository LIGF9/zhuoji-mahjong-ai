import psutil, os, sys

me = os.getpid()
ancestors = set()
p = psutil.Process(me)
while p is not None:
    ancestors.add(p.pid)
    try:
        p = p.parent()
    except Exception:
        p = None

tree_pids, seen = [], set()
selfplay_pids = []
for proc in psutil.process_iter(['pid', 'ppid', 'name', 'cmdline', 'memory_info']):
    try:
        cl = proc.info['cmdline'] or []
        cmd = ' '.join(cl)
        if 'selfplay.py' in cmd or 'launch_segment' in cmd or 'segment_finalize' in cmd or 'judge' in cmd:
            if proc.pid in ancestors:
                continue
            mi = proc.info['memory_info']
            rss = mi.rss if mi else 0
            is_sp = 'selfplay.py' in cmd
            entry = (proc.pid, proc.info['ppid'], rss, 'SELFPLAY' if is_sp else 'OTHER', cmd[:160])
            if proc.pid not in seen:
                seen.add(proc.pid)
                tree_pids.append(entry)
                if is_sp:
                    selfplay_pids.append(proc.pid)
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        pass

print('=== MATCHED PROCESSES (excluding self ancestors) ===')
for e in tree_pids:
    print(f'pid={e[0]} ppid={e[1]} rss={e[2]/1e9:.2f}GB type={e[3]} cmd={e[4]}')

# dedupe tree: sum unique pids under selfplay main (exclude double counting parent-child dupes)
uniq_rss = {}
for e in tree_pids:
    if e[0] not in uniq_rss:
        uniq_rss[e[0]] = e[2]
total = sum(uniq_rss.values())
print(f'unique pids={len(uniq_rss)} total_rss={total/1e9:.2f}GB selfplay_count={len(selfplay_pids)}')

vm = psutil.virtual_memory()
print(f'SYS mem: {vm.percent}% avail={vm.available/1e9:.2f}GB total={vm.total/1e9:.2f}GB')

if vm.percent > 95 and selfplay_pids:
    print('ACTION_REQUIRED=RESTART_TREE')
else:
    print('ACTION_REQUIRED=NONE')
