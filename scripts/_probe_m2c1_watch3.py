import psutil, os, json
self_pids = set()
p = psutil.Process(os.getpid())
try:
    for a in p.parents():
        self_pids.add(a.pid)
except Exception:
    pass
self_pids.add(os.getpid())
rows = []
for pr in psutil.process_iter(['pid','name','cmdline','memory_info','create_time']):
    try:
        cl = pr.info.get('cmdline') or []
        cmd = ' '.join(cl)
    except Exception:
        continue
    if pr.info['pid'] in self_pids:
        continue
    if '_probe_m2c1_watch' in cmd or 'psutil' in cmd:
        continue
    low = cmd.lower()
    if any(k in low for k in ['selfplay.py','m1_launch_segment.py','m1_segment_finalize.py','score_checkpoint','m1_segment_judge']):
        try:
            rss = pr.info['memory_info'].rss/1024**3
            ct = pr.info['create_time']
        except Exception:
            rss = -1; ct = 0
        rows.append({'pid':pr.info['pid'],'ppid':pr.ppid(),'rss_gb':round(rss,2),'created':ct,'cmd':cmd[:220]})
vm = psutil.virtual_memory()
print(json.dumps({'mem_percent':vm.percent,'avail_gb':round(vm.available/1024**3,2),'total_gb':round(vm.total/1024**3,2),'procs':rows}, ensure_ascii=False, indent=1))
