"""应急重启 auto5（原进程被误回收，仅跑到 it7、无 ckpt）。

设计要点：
* 命令行直接从 `reports/_auto5_cmd.txt`（由 _m1_auto5.out 头部提取）解析，
  避免手抄 60+ 参数出错；空参数 `""` 用占位符还原。
* 本脚本作为**长驻启动器**运行（外层 run_in_background），Popen 训练后立即 wait()，
  保证启动器不退出 ⇒ 训练不被平台回收。不要 kill 本启动器！
* 写入 _m1_auto5.pid，供总控探活。
"""
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW = (ROOT / "reports" / "_auto5_cmd.txt").read_text(encoding="utf-8").strip()

# 还原空参数：Popen 传的是空串，日志文本里只剩双空格 ⇒ `--snapshot-pool  --klt-beta`
# 若某选项的下一个 token 以 `--` 开头，说明它的值原本是空串。
tokens = RAW.split()
argv: list[str] = []
for i, tok in enumerate(tokens):
    argv.append(tok)
    if tok == "--snapshot-pool" and (i + 1 >= len(tokens) or tokens[i + 1].startswith("--")):
        argv.append("")

LOG = ROOT / "reports" / "_m1_auto5.out"
PIDF = ROOT / "reports" / "_m1_auto5.pid"
RUNOUT = ROOT / "reports" / "_m1_auto5_relaunch.out"

assert len(argv) > 10, f"命令解析异常：{argv}"
empty_at = [i for i, a in enumerate(argv) if a == ""]
print(f"[relaunch] argv 长度 {len(argv)}；空参数位置 {empty_at}", flush=True)
print(f"[relaunch] 可执行 {argv[0]}", flush=True)

flags = (subprocess.CREATE_NEW_PROCESS_GROUP
         | subprocess.CREATE_BREAKAWAY_FROM_JOB
         | subprocess.DETACHED_PROCESS)

with LOG.open("a", encoding="utf-8") as fh:
    fh.write(f"\n===== m1_auto5 重挂 @ {time.strftime('%Y-%m-%d %H:%M:%S')}（原进程被误回收，从 it0800 重跑）=====\n")

ff = RUNOUT.open("a", encoding="utf-8")
proc = subprocess.Popen(
    argv,
    stdin=subprocess.DEVNULL,
    stdout=ff,
    stderr=subprocess.STDOUT,
    cwd=str(ROOT),
    creationflags=flags,
)
print(f"[relaunch] 训练已 spawn，pid={proc.pid}", flush=True)
time.sleep(20)
rc = proc.poll()
if rc is None:
    PIDF.write_text(str(proc.pid), encoding="utf-8")
    print(f"[relaunch] ✅ 存活确认，pid 文件已更新为 {proc.pid}；转入 wait()", flush=True)
else:
    print(f"[relaunch] ❌ 20s 内退出 rc={rc}", flush=True)
    tail = RUNOUT.read_text(encoding="utf-8", errors="replace").splitlines()[-20:]
    print("\n".join(tail), flush=True)

rc = proc.wait()
print(f"[relaunch] 训练结束 rc={rc}", flush=True)
