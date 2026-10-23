"""临时巡检脚本：不参与训练，仅打印真实状态。用完可删。"""
import psutil, datetime, os, time, json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
me = os.getpid()

LABELS = [
    ("m2_night_" + "autopilot.py", "总控"),
    ("gate_" + "watch.py", "判读守望"),
    ("m2_stoploss_" + "watch.py", "止损守望"),
    ("score_" + "checkpoints.py", "判读进程"),
    ("self" + "play.py", "训练"),
    ("keep_" + "awake.py", "防睡眠"),
    ("res_" + "monitor.py", "资源监控"),
]

print("=== 进程清单（真实） ===")
for p in psutil.process_iter(["pid", "cmdline", "create_time", "memory_info"]):
    try:
        if p.pid == me:
            continue
        cl = " ".join(p.info["cmdline"] or [])
        if not cl:
            continue
        sig = None
        for key, lab in LABELS:
            if key in cl:
                sig = lab
                break
        if not sig:
            continue
        extra = ""
        if "--at" in cl:
            extra += " at=" + cl.split("--at")[1].split()[0]
        for flag in ("--out", "--init"):
            if flag + " " in cl:
                extra += " " + flag.strip("-") + "=" + os.path.basename(cl.split(flag)[1].split()[0])
        st = datetime.datetime.fromtimestamp(p.info["create_time"]).strftime("%H:%M:%S")
        print(f"  {sig:<8} PID {p.info['pid']:<7} {st}  {p.info['memory_info'].rss/1024**3:5.2f}G{extra}")
    except Exception:
        pass

print()
print("=== pid 文件 ===")
for f in sorted((ROOT / "reports").glob("_m1_*.pid")):
    try:
        val = f.read_text(encoding="utf-8").strip()
        alive = psutil.pid_exists(int(val)) if val.isdigit() else None
        print(f"  {f.name} = {val}  存活={alive}")
    except Exception as e:
        print(f"  {f.name} 读取失败 {e}")

print()
print("=== 训练日志新鲜度 ===")
for name in ("_m1_auto4.out", "_m1_auto5.out"):
    p = ROOT / "reports" / name
    if p.exists():
        age = time.time() - p.stat().st_mtime
        print(f"  {name}: {age:.0f}s 前更新")

print()
print("=== 总控 state ===")
try:
    st = json.loads((ROOT / "reports" / "night_autopilot_state.json").read_text(encoding="utf-8"))
    print(f"  current={st.get('current')} relay={st.get('relay_count')} updated={st.get('updated_at')}")
except Exception as e:
    print("  ", e)

print()
print("=== auto5 最新轮 ===")
p = ROOT / "reports" / "_m1_auto5.out"
if p.exists():
    lines = [l for l in p.read_text(encoding="utf-8", errors="replace").splitlines() if l.startswith("iter")]
    for l in lines[-2:]:
        print("  " + l[:150])
    print(f"  共 {len(lines)} 轮")
