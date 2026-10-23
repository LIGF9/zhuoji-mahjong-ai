# -*- coding: utf-8 -*-
"""m2a2 巡检探测：selfplay 进程树 + 系统内存（排除自身）"""
import os, sys
import psutil

me = os.getpid()
ancestors = set()
p = psutil.Process(me)
while p is not None:
    ancestors.add(p.pid)
    try:
        p = p.parent()
    except Exception:
        break

tree, others = [], []
for proc in psutil.process_iter(['pid', 'ppid', 'name', 'cmdline', 'memory_info', 'create_time']):
    try:
        if proc.pid == me or proc.pid in ancestors:
            continue
        info = proc.info
        cl = ' '.join(info['cmdline'] or [])
        if 'selfplay.py' not in cl and 'finalize' not in cl and 'score_checkpoints' not in cl and 'm1_segment_judge' not in cl and 'm1_launch_segment' not in cl:
            continue
        rss = info['memory_info'].rss / (1 << 30) if info['memory_info'] else 0.0
        entry = f"pid={proc.pid} ppid={info['ppid']} rss={rss:.2f}GB start={info['create_time']} cmd={cl[:160]}"
        tree.append(entry)
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        pass

vm = psutil.virtual_memory()
print(f"MEM pct={vm.percent:.1f} avail={vm.available/(1<<30):.2f}GB total={vm.total/(1<<30):.2f}GB")
print(f"TREE count={len(tree)}")
for t in tree:
    print(t)
