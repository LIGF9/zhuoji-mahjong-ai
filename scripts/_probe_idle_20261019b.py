"""Idle-state inspection probe (2nd run of 2026-10-19): residual selfplay/training processes + memory.

Excludes own ancestor chain (bash/python -c echo false positives seen before).
Read-only. Does NOT start or kill anything.
"""
import psutil, os, time, glob

ROOT = r"C:\Users\LIGF\WorkBuddy\2026-09-30-12-49-43\zhuoji-mahjong"

me = os.getpid()
ancestors = set()
p = psutil.Process(me)
while p is not None:
    ancestors.add(p.pid)
    try:
        p = p.parent()
    except Exception:
        p = None

KEYS = ('selfplay.py', 'launch_segment', 'segment_finalize', 'm1_segment_judge',
        'score_checkpoints', 'keep_awake', 'res_monitor', 'relay_', 'autopilot',
        'night_train', 'train.py', 'duel.py')

rows = []
allpy = []
for proc in psutil.process_iter(['pid', 'ppid', 'name', 'cmdline', 'create_time', 'memory_info']):
    try:
        pid = proc.info['pid']
        if pid in ancestors:
            continue
        cl = proc.info['cmdline'] or []
        cmd = ' '.join(cl)
        name = (proc.info['name'] or '').lower()
        mi = proc.info['memory_info']
        rss = mi.rss if mi else 0
        is_py = 'python' in name or 'python' in cmd.lower()
        if is_py:
            age_min = (time.time() - (proc.info['create_time'] or time.time())) / 60.0
            allpy.append((pid, proc.info['ppid'], rss, age_min, cmd[:180]))
        if any(k in cmd for k in KEYS):
            rows.append((pid, proc.info['ppid'], rss,
                         (time.time() - (proc.info['create_time'] or time.time())) / 60.0,
                         cmd[:180]))
    except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
        pass

print('=== TRAINING/JUDGE-LIKE PROCESSES (self ancestors excluded) ===')
if not rows:
    print('(none)')
for r in rows:
    print(f'pid={r[0]} ppid={r[1]} rss={r[2]/1e9:.2f}GB age={r[3]:.1f}min cmd={r[4]}')

print('\n=== ALL PYTHON PROCESSES (for orphan detection) ===')
if not allpy:
    print('(none)')
for r in sorted(allpy, key=lambda x: -x[2]):
    print(f'pid={r[0]} ppid={r[1]} rss={r[2]/1e9:.2f}GB age={r[3]:.1f}min cmd={r[4]}')

vm = psutil.virtual_memory()
sw = psutil.swap_memory()
print(f'\nSYS mem: {vm.percent}% avail={vm.available/1e9:.2f}GB total={vm.total/1e9:.2f}GB')
print(f'swap: {sw.percent}% used={sw.used/1e9:.2f}GB')

print('\n=== LATEST LOG/ARTIFACT MTIMES ===')
pats = [
    r"reports\_m1_m2c4.out", r"reports\_finalize_m2c4.out",
    r"reports\m1_m2c4_judge.json", r"reports\m1_m2c4_judge_duel.json",
    r"reports\_m2_chain_state.json", r"reports\*.out",
]
for pat in pats:
    for f in sorted(glob.glob(os.path.join(ROOT, pat)), key=os.path.getmtime)[-4:]:
        mt = os.path.getmtime(f)
        age_min = (time.time() - mt) / 60.0
        print(f'{os.path.basename(f):<44} mtime={time.strftime("%m-%d %H:%M", time.localtime(mt))} age={age_min:.1f}min size={os.path.getsize(f)/1024:.0f}KB')

print(f'\nnow={time.strftime("%Y-%m-%d %H:%M:%S")}')
print('ACTION=' + ('REVIEW_ORPHAN' if rows else 'NONE'))
