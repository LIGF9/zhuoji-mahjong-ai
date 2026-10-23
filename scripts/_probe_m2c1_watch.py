# -*- coding: utf-8 -*-
"""m2c1 巡检探针：识别 selfplay 训练进程树 / 判读进程 / 系统内存。排除探测脚本自身 pid 与祖先链。"""
import os
import sys
import datetime
import psutil

ME = os.getpid()
ancestors = set()
p = psutil.Process(ME)
try:
    while p is not None:
        ancestors.add(p.pid)
        p = p.parent()
except Exception:
    pass

now = datetime.datetime.now().isoformat(timespec="seconds")
vm = psutil.virtual_memory()
print(f"now={now} mem_pct={vm.percent} avail_gb={vm.available/2**30:.2f} total_gb={vm.total/2**30:.2f}")

train_pids = []
judge_pids = []
for proc in psutil.process_iter(["pid", "ppid", "name", "cmdline", "create_time", "memory_info"]):
    try:
        pid = proc.info["pid"]
        if pid in ancestors or pid == ME:
            continue
        cmd = " ".join(proc.info["cmdline"] or [])
        if not cmd:
            continue
        low = cmd.lower()
        if "probe_m2c1" in low or "_probe" in low:
            continue
        if "selfplay.py" in low and "m1_launch" not in low.split("selfplay.py")[0][-30:]:
            pass
        if "selfplay.py" in low:
            rss = proc.info["memory_info"].rss / 2**30 if proc.info["memory_info"] else 0.0
            ct = datetime.datetime.fromtimestamp(proc.info["create_time"]).strftime("%H:%M:%S")
            train_pids.append((pid, proc.info["ppid"], rss, ct, cmd[:160]))
        elif ("score_checkpoints" in low) or ("m1_segment_finalize" in low) or ("m1_segment_judge" in low):
            ct = datetime.datetime.fromtimestamp(proc.info["create_time"]).strftime("%H:%M:%S")
            judge_pids.append((pid, proc.info["ppid"], ct, cmd[:160]))
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        continue

# 计算训练树全部后代 RSS
def descendants(root_pid):
    out = []
    for proc in psutil.process_iter(["pid", "ppid", "memory_info", "cmdline"]):
        try:
            out.append((proc.info["pid"], proc.info["ppid"],
                        proc.info["memory_info"].rss / 2**30 if proc.info["memory_info"] else 0.0,
                        " ".join(proc.info["cmdline"] or [])[:120]))
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return out

print(f"train_count={len(train_pids)}")
for pid, ppid, rss, ct, cmd in train_pids:
    print(f"TRAIN pid={pid} ppid={ppid} rss={rss:.2f}GB start={ct} cmd={cmd}")

print(f"judge_count={len(judge_pids)}")
for pid, ppid, ct, cmd in judge_pids:
    print(f"JUDGE pid={pid} ppid={ppid} start={ct} cmd={cmd}")

# 对每个 selfplay 进程枚举后代
allprocs = None
if train_pids:
    allprocs = descendants(None)
for pid, ppid, rss, ct, cmd in train_pids:
    # 构建父链映射，收集以 pid 为根的整棵树
    parent_map = {a: b for a, b, _, _ in allprocs}
    tree = [pid]
    changed = True
    tree_set = {pid}
    while changed:
        changed = False
        for a, b, _, _ in allprocs:
            if b in tree_set and a not in tree_set:
                tree_set.add(a)
                tree.append(a)
                changed = True
    total = 0.0
    for a, b, r, c in allprocs:
        if a in tree_set:
            total += r
            print(f"  TREE pid={a} ppid={b} rss={r:.2f}GB cmd={c}")
    print(f"TREE_TOTAL root={pid} procs={len(tree)} rss={total:.2f}GB")
