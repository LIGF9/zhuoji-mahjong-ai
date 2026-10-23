#!/usr/bin/env python
"""M2 专用自动链：M2-A 收尾 → 判读 → G1 分支 → M2-B → G2×3 + G3 → 总报告。

为什么不用 ``m1_autopilot``
--------------------------
它的 ``SEG_ORDER = ["s1","s2","s3"]`` 与 ``dushan_m1_{seg}`` 命名是**硬编码**的
（2026-10-12 为 M1 三段写死），M2 的段名（``s4a/s4b``）与产出名
（``dushan_m2_a/b``）都对不上；硬扩会碰坏已经跑通三段的 M1 链条。
所以 M2 走独立 runner，判据直接对齐 ``docs/P3-M2-追赶并超越S11训练计划.md`` §8。

链路（全部阶段幂等 —— 中断后重跑同一条命令即可续上）
--------------------------------------------------
1. **等 M2-A 到 600 轮**（体检：Traceback / NaN / 停滞 45 min 判死；
   进程死亡允许**自动 resume 一次** —— 从最近周期存档续跑，重跑再死则停链报错）
2. **G1 判决**：``reports/G1_s4a_it300.verdict.json``（正常应已由守望任务在
   第 300 轮出好；缺失则现场 ``gate_watch --now`` 补跑）
3. **监护进程去重** → **段末收尾**（``m1_segment_finalize.py``，含 A1–A9 健康度
   + 1200 副段内配对）→ **写 ``reports/M2A_阶段结果与解读.md``**
4. **G1 分支**：
   - 通过（``va ≥ −1.6``）→ M2-B 从 ``dushan_m2_a`` 起；
     对手池按「A 段是否已显著强于 master」自动二选一；
   - 未通过 → **按预注册方案关 β 重跑 A 段**（新段 ``s4a2``，600 轮），
     收尾后 M2-B 改从 ``dushan_m2_a2`` 起（对手池取保守默认）。
5. **M2-B（1200 轮）**，随后 G2@400 / G2@800 / G3@1200 三次 1200 副门禁
   （判据：G2 单调不减；G3 追平 = ``va ≥ +0.5`` 且 CI 下界 > 0，
   超越 = ``va ≥ +1.5`` 且 CI 下界 > 0；it1200 一次判读同时出 G2 与 G3 两档）
6. **写 ``reports/M2_阶段结果与解读.md``**（含 M1+M2 总账与 M2+ 建议）

起动方式（★ C6 教训）
--------------------
**必须挂平台后台任务**：``python scripts/m2_chain.py`` —— 链条本身作为前台进程
持有整棵训练树；命令一返回、树就被清。``--dry-run`` 只打印计划不执行。

用法::

    python scripts/m2_chain.py --dry-run     # 打印将要做的事
    python scripts/m2_chain.py               # 全链（挂后台任务）
    python scripts/m2_chain.py --resume      # 中断后续跑（按幂等性自动跳过已完成阶段）
"""
from __future__ import annotations

import argparse
import json
import re
import statistics
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import psutil                                              # noqa: E402

from m1_launch_segment import SEGMENTS, find_latest_ckpt, launch  # noqa: E402

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

PY = sys.executable
GATE = str(ROOT / "scripts" / "gate_watch.py")
FINALIZE = str(ROOT / "scripts" / "m1_segment_finalize.py")

# ---- 固定口径（与计划 §6/§8 逐字对齐，改这里 = 改实验设计）----
DEALS = 1200
SEED = 31337
G1_THRESHOLD = -1.6            # 「vs master ≥ −1.6」= 相对 S3 改善 ≥ +1.5
S3_BASELINE, S3_BASELINE_CI = -3.105, 0.437
G3_TRACE, G3_SURPASS = 0.5, 1.5
REF = "teacher:balanced|teacher:tenpai_rush|teacher:defensive"

RUN_A, STEM_A = "m1_s4a", "dushan_m2_a"
RUN_B, STEM_B = "m1_s4b", "dushan_m2_b"
SEG_RERUN = "s4a2"             # G1 未通过时关 β 重跑的 A 段
STEM_RERUN = "dushan_m2_a2"

LOG_A = ROOT / "reports" / f"_{RUN_A}.out"
PID_A = ROOT / "reports" / f"_{RUN_A}.pid"
MON_A = ROOT / "reports" / f"_res_monitor_{RUN_A}.csv"
G1_JSON = ROOT / "reports" / "G1_s4a_it300.json"
G1_VERDICT = G1_JSON.with_suffix(".verdict.json")
JUDGE_A = ROOT / "reports" / f"{RUN_A}_judge.json"
REPORT_A = ROOT / "reports" / "M2A_阶段结果与解读.md"
REPORT_M2 = ROOT / "reports" / "M2_阶段结果与解读.md"
STATE = ROOT / "reports" / "_m2_chain_state.json"

STALL_GIVEUP_MIN = 45          # 日志停滞多久判死（长跑冻结过 22.2 h 没被发现）
JUDGE_WORKERS = 8              # 判读与训练并发时降内存（共用牌池 → 结果与并行度无关）
TRAIN_WORKERS = 12             # 与 M2-A 实跑一致


def say(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def save_state(**kw) -> None:
    st = {}
    if STATE.exists():
        try:
            st = json.loads(STATE.read_text(encoding="utf-8"))
        except Exception:
            st = {}
    st.update(kw, when=datetime.now().isoformat(timespec="seconds"))
    STATE.write_text(json.dumps(st, ensure_ascii=False, indent=2), encoding="utf-8")


# ---------------------------------------------------------------------------
# 训练等待 / 崩溃恢复
# ---------------------------------------------------------------------------
def _count_iters(log: Path) -> int:
    if not log.exists():
        return 0
    return len(re.findall(r"^iter", log.read_text(encoding="utf-8",
                                               errors="replace"), re.M))


def _pid_alive(pid_file: Path, stem: str) -> bool:
    if not pid_file.exists():
        return False
    try:
        pid = int(pid_file.read_text(encoding="utf-8").strip())
        p = psutil.Process(pid)
        return stem in " ".join(p.cmdline() or [])
    except (psutil.NoSuchProcess, ValueError, psutil.AccessDenied):
        return False


def wait_train(run: str, stem: str, target: int, max_wait_h: float,
               allow_resume: bool, workers: int) -> str:
    """等训练到 target 轮。返回 "done" / "crashed" / "stalled"。

    "crashed" 只在**自动 resume 也失败**后才会返回；resume 成功则继续等。
    """
    log = ROOT / "reports" / f"_{run}.out"
    pid_file = ROOT / "reports" / f"_{run}.pid"
    resumed = False
    t0 = time.time()
    while True:
        n = _count_iters(log)
        if n >= target:
            say(f"{run}: 已到 {n}/{target} 轮 ✅")
            return "done"
        if time.time() - t0 > max_wait_h * 3600:
            say(f"❌ {run}: 等待超时（{max_wait_h:g} h），停在 {n}/{target} 轮")
            return "stalled"
        alive = _pid_alive(pid_file, stem)
        if not alive:
            if n >= target:
                return "done"
            if allow_resume and not resumed:
                resumed = True
                say(f"⚠️ {run}: 进程消失但只到 {n}/{target} 轮 —— 自动 resume 一次")
                if _resume(run, stem, workers):
                    t0 = time.time()
                    continue
            say(f"❌ {run}: 进程已死（{n}/{target} 轮），尾部：")
            _print_tail(log, 15)
            return "crashed"
        # 体检：Traceback / NaN
        body = log.read_text(encoding="utf-8", errors="replace") if log.exists() else ""
        if "Traceback" in body:
            say(f"❌ {run}: 日志出现 Traceback，尾部：")
            _print_tail(log, 20)
            return "crashed"
        if re.search(r"=\s*nan\b", body, re.I):
            say(f"❌ {run}: 日志出现 NaN")
            return "crashed"
        time.sleep(60)


def _resume(run: str, stem: str, workers: int) -> bool:
    """复刻 m1_launch_segment.main() 的 --resume 逻辑（launch() 不管 resume）。"""
    seg = run[len("m1_"):]
    cfg = dict(SEGMENTS[seg])
    hit = find_latest_ckpt(Path(cfg["out"]).stem)
    if hit is None:
        say(f"[resume] ❌ {cfg['out']} 的周期存档一个都没有，无处可续")
        return False
    p, it = hit
    total = cfg["iters"]
    done = it
    hist = ROOT / "reports" / f"{run}_history.json"
    if hist.exists():
        try:
            _h = json.loads(hist.read_text(encoding="utf-8"))
            if isinstance(_h, list) and _h:
                done = max(done, int(_h[-1].get("iter") or 0))
        except Exception:
            pass
    if done >= total:
        say(f"[resume] 已完成 {done} ≥ {total}，视作跑完")
        return True
    cfg.update(init=str(p.relative_to(ROOT)).replace("\\", "/"),
               iters=total - done, warmup=0, offset=done)
    say(f"[resume] 从 {p.name}（iter {it}）续跑，再跑 {cfg['iters']} 轮至 {total}")
    try:
        launch(seg, cfg, workers, foreground=True, settle=8.0)
        return True
    except Exception as e:                                 # noqa: BLE001
        say(f"[resume] ❌ 起动失败：{e}")
        return False


def _print_tail(log: Path, n: int) -> None:
    if not log.exists():
        print("    （日志不存在）", flush=True)
        return
    for line in log.read_text(encoding="utf-8", errors="replace").splitlines()[-n:]:
        print("    " + line, flush=True)


# ---------------------------------------------------------------------------
# 门禁 / 收尾
# ---------------------------------------------------------------------------
def kill_guardians(*tags: str) -> int:
    """杀掉上一段留下的 keep_awake / res_monitor（重复起会劈开打点文件）。

    ``keep_awake.py`` 起动时**不带标签**（launcher 里就没给它 --out 之类），
    所以按脚本名匹配 —— 本项目同一时刻只该有一个；``res_monitor`` 按
    ``--out reports/_res_monitor_{run}.csv`` 里的 run 标签匹配。
    """
    n = 0
    for p in psutil.process_iter(["pid", "cmdline"]):
        try:
            cl = " ".join(p.info["cmdline"] or [])
        except Exception:
            continue
        if "gate_watch" in cl or "m2_chain" in cl:
            continue                      # 别误杀自己这条链上的判读守望
        hit = ("keep_awake.py" in cl
               or ("res_monitor.py" in cl and any(t in cl for t in tags)))
        if hit:
            try:
                p.terminate()
                n += 1
                say(f"  已停守护进程 pid={p.info['pid']}  {cl[-80:]}")
            except Exception:
                pass
    return n


def ensure_g1(workers: int) -> dict:
    """G1 判决（正常应已由守望任务出好；缺失则现场补跑）。"""
    if G1_VERDICT.exists():
        v = json.loads(G1_VERDICT.read_text(encoding="utf-8"))
        say(f"G1 判决已就绪：va={v['vs_anchor']:+.3f} ± {v['vs_anchor_ci95']:.3f} "
            f"→ {'通过' if v['passed'] else '未通过'}")
        return v
    say("⚠️ G1 判决缺失 —— 现场补跑（it0300 应已存在）")
    r = subprocess.run([PY, GATE, "--stem", STEM_A, "--at", "300", "--gate", "G1",
                        "--threshold", str(G1_THRESHOLD),
                        "--baseline-name", "M1-S3",
                        "--baseline", str(S3_BASELINE), "--baseline-ci",
                        str(S3_BASELINE_CI),
                        "--deals", str(DEALS), "--seed", str(SEED),
                        "--workers", str(workers), "--threads", "2",
                        "--worker-threads", "1",
                        "--out", str(G1_JSON)], cwd=str(ROOT))
    if r.returncode != 0 or not G1_VERDICT.exists():
        raise RuntimeError("G1 补跑失败")
    return json.loads(G1_VERDICT.read_text(encoding="utf-8"))


def finalize(run: str, ckpt_prefix: str, deals: int, threads: int) -> Path:
    js = ROOT / "reports" / f"{run}_judge.json"
    model = ROOT / "models" / f"{ckpt_prefix}.pt"
    if js.exists() and model.exists() and js.stat().st_mtime > model.stat().st_mtime:
        say(f"{run}: 判读 JSON 已比模型新 → 跳过收尾（幂等）")
        return js
    say(f"{run}: 段末收尾（{deals} 副段内配对 + A1–A9 健康度）…")
    r = subprocess.run([PY, FINALIZE, "--run", run, "--ckpt-prefix", ckpt_prefix,
                        "--extra", "dushan_master", "--deals", str(deals),
                        "--threads", str(threads)], cwd=str(ROOT))
    if r.returncode != 0:
        raise RuntimeError(f"{run} 段末收尾 exit={r.returncode}")
    return js


# ---------------------------------------------------------------------------
# 报告
# ---------------------------------------------------------------------------
def _train_stats(log: Path, mon: Path) -> dict:
    st: dict = {}
    if log.exists():
        rows = []
        for l in log.read_text(encoding="utf-8", errors="replace").splitlines():
            m = re.match(r"^iter\s+(\d+)\s+(.*)$", l)
            if not m:
                continue
            d = {k: float(v) for k, v in
                 re.findall(r"(\w+)=\s*([\d.eE+-]+)", m.group(2))}
            if "total" in d:
                rows.append(d)
        if rows:
            st["iters"] = len(rows)
            # ★ total 是**累计**秒数（iter 行的 total=… 从训练起点累加），
            #   "每轮耗时"必须用相邻差分；直接取中位会得到一个无意义的大数。
            tots = [r["total"] for r in rows]
            deltas = [b - a for a, b in zip(tots, tots[1:]) if 0 < b - a < 3600]
            if deltas:
                st["total_med"] = statistics.median(deltas)
            st["collect_med"] = statistics.median(
                r["collect"] for r in rows if "collect" in r)
            st["samples_med"] = statistics.median(
                r["samples"] for r in rows if "samples" in r)
            st["samples_last"] = rows[-1].get("samples")
            st["ec_last"] = rows[-1].get("ec")
            st["H_last"] = rows[-1].get("H")
            st["beta_last"] = rows[-1].get("beta")
            st["klr_max"] = max((r.get("klr", 0.0) for r in rows), default=0.0)
            st["klr_last"] = rows[-1].get("klr")
            m = re.search(r"熵基线\s*H0=([\d.]+).*?H\*=([\d.]+)", log.read_text(
                encoding="utf-8", errors="replace"))
            if m:
                st["h0"], st["hstar"] = float(m.group(1)), float(m.group(2))
    if mon.exists():
        peaks = [float(m.group(1)) for l in mon.read_text(errors="replace").splitlines()
                 if (m := re.search(r"tree ([\d.]+)G", l))]
        if peaks:
            st["rss_peak"] = max(peaks)
    return st


def _fmt(v, pat="{:.3f}", na="—") -> str:
    return na if v is None else pat.format(v)


def write_m2a_report(judge: dict, g1: dict, rerun: bool) -> None:
    h = judge.get("health") or {}
    vp = judge.get("verdict_progress")
    st = _train_stats(LOG_A, MON_A)
    ok = "✅ 通过" if g1["passed"] else "❌ 未通过"
    lines = [
        "# M2-A（修复 + 能力移植，600 轮）阶段结果与解读",
        "",
        f"> 自动生成于 {datetime.now():%Y-%m-%d %H:%M:%S}（`scripts/m2_chain.py`）；"
        f"段 = `s4a`{'（G1 未达标 → 关 β 重跑 `s4a2` 后本报告对应重跑段）' if rerun else ''}",
        "",
        "## 一句话结论",
        "",
        f"- **G1（vs master，1200 副配对）**：`{g1['ckpt']}` "
        f"**{g1['vs_anchor']:+.3f} ± {g1['vs_anchor_ci95']:.3f}** "
        f"（下界 {g1['lower']:+.3f}）→ **{ok}**（门槛 ≥ {G1_THRESHOLD:+.1f}）；",
        f"  相对 M1-S3 基线（{S3_BASELINE:+.3f} ± {S3_BASELINE_CI:.3f}）改善 "
        f"**{g1['vs_anchor'] - S3_BASELINE:+.3f}**。",
        f"- **段内进展**：{_fmt((vp or {}).get('delta'), '{:+.3f}')} ± "
        f"{_fmt((vp or {}).get('ci95'))} "
        f"（{'显著上升' if (vp or {}).get('significant_up') else '未达显著'}，"
        f"末点相对本段首点）。",
        f"- **健康度**：{'全部通过' if h.get('all_ok') else '有未通过项'}"
        f"（{h.get('n_skip', 0)} 项跳过）—— 见下表。",
        "",
        "## 健康度检查（阶段 A，A1–A9）",
        "",
        "| 项 | 状态 | 细节 |",
        "|---|---|---|",
    ]
    for c in h.get("checks", []):
        lines.append(f"| {c['name']} | {c['status']} | {c['detail']} |")
    lines += [
        "",
        "## 吞吐 / 资源 / 控制系数（实跑）",
        "",
        "| 项 | 值 |",
        "|---|---|",
        f"| 轮数 | {st.get('iters', '—')} |",
        f"| total 中位 | {_fmt(st.get('total_med'), '{:.1f}')} s/轮 |",
        f"| collect 中位 | {_fmt(st.get('collect_med'), '{:.1f}')} s/轮 |",
        f"| samples 中位/末 | {_fmt(st.get('samples_med'), '{:.0f}')} / "
        f"{_fmt(st.get('samples_last'), '{:.0f}')} |",
        f"| 熵基线 H₀ → H* | {_fmt(st.get('h0'))} → {_fmt(st.get('hstar'))} |",
        f"| ec 末值 | {_fmt(st.get('ec_last'))}（闭环平衡点预测 ≈0.0515）|",
        f"| β 末值 | {_fmt(st.get('beta_last'))}（退火至 0）|",
        f"| KL 率 max / 末 | {_fmt(st.get('klr_max'))} / {_fmt(st.get('klr_last'))}"
        f"（危险线 0.05）|",
        f"| 进程树 RSS 峰值 | {_fmt(st.get('rss_peak'), '{:.2f}')} GB |",
        "",
        "## 判读",
        "",
    ]
    if g1["passed"]:
        lines += [
            "- G1 达标 → **蒸馏（KLT reverse-KL）在修复后的底座上有效**，"
            "按计划进入 M2-B（β=0，撤教师靠自博弈继续爬）。",
            f"- `ec` 末值 {_fmt(st.get('ec_last'))}：对照闭环预测（第 300 轮 ≈0.046、"
            "平衡点 0.0515）—— 若明显偏低，说明 A1 环路未真正接管，需回看 §5.4。",
        ]
    else:
        lines += [
            f"- G1 未达标（{g1['vs_anchor']:+.3f} < {G1_THRESHOLD:+.1f}）→ "
            "按预注册方案**关 β 重跑 A 段**（`s4a2`，600 轮）："
            "隔离「蒸馏的锅」vs「A1–A8 修复无效」。",
            "- 若重跑段显著更好 → 归因给 KLT/β；若重跑同样不动 → 瓶颈在表征/数据预算，"
            "转 M2+（结构改进）。",
        ]
    lines += ["", "---", "",
              "相关：`reports/G1_s4a_it300.json`（G1 原始判读）、"
              f"`reports/{RUN_A}_judge.json`（本段判读）、"
              "`docs/P3-M2-追赶并超越S11训练计划.md` §5.4/§5.5。", ""]
    REPORT_A.write_text("\n".join(lines), encoding="utf-8")
    say(f"报告 → {REPORT_A.name}")


def _criteria_desc(g: dict) -> str:
    """把一个门禁判决用到的判据缩写成短标签（总表用）。"""
    parts = []
    if g.get("threshold", -99) > -99:
        parts.append(f"≥{g['threshold']:+.1f}")
    if g.get("ci_lower_min") is not None:
        parts.append("CI下界>0")
    if g.get("monotonic_prev") is not None:
        parts.append("单调不减")
    return "；".join(parts) or "—"


def write_m2_report(judge_a: dict, gates: list[dict], st_b: dict) -> None:
    g1 = gates[0]
    final = gates[-1]
    lines = [
        "# M2 阶段结果与解读（追赶并超越 S11）",
        "",
        f"> 自动生成于 {datetime.now():%Y-%m-%d %H:%M:%S}（`scripts/m2_chain.py`）。",
        "",
        "## 门禁总表（1200 副共用牌池配对，anchor = `dushan_master`，seed 31337）",
        "",
        "| 门禁 | 检查点 | vs master | ±95%CI | 下界 | 判据 | 结果 |",
        "|---|---|---|---|---|---|---|",
    ]
    for g in gates:
        lines.append(
            f"| {g['gate']} | `{g['ckpt']}` | {g['vs_anchor']:+.3f} | "
            f"{g['vs_anchor_ci95']:.3f} | {g['lower']:+.3f} | "
            f"{_criteria_desc(g)} | "
            f"{'✅' if g['passed'] else '❌'}"
            f"{'（超越 ✅）' if g.get('passed2') else ''} |")
    lines += [
        "",
        "## M1 + M2 总账（vs master 配对净分差）",
        "",
        f"| 节点 | vs master |",
        "|---|---|",
        f"| M1-S3 段末（追赶起点） | {S3_BASELINE:+.3f} ± {S3_BASELINE_CI:.3f} |",
        f"| G1（M2-A 第 300 轮） | {g1['vs_anchor']:+.3f} ± {g1['vs_anchor_ci95']:.3f} |",
        f"| G3（M2-B 第 1200 轮） | {final['vs_anchor']:+.3f} ± "
        f"{final['vs_anchor_ci95']:.3f} |",
        "",
        f"**全链改善**：{final['vs_anchor'] - S3_BASELINE:+.3f}"
        f"（S3 基线 → M2-B 末点）。",
        "",
        "## 判读",
        "",
    ]
    va, lo = final["vs_anchor"], final["lower"]
    if final.get("passed2"):
        lines.append(f"- **超越 S11 达成**：va = {va:+.3f}，CI 下界 {lo:+.3f} > 0，"
                     f"且 va ≥ +{G3_SURPASS}。")
    elif final["passed"]:
        lines.append(f"- **追平达成**：va = {va:+.3f}（≥ +{G3_TRACE}），"
                     f"CI 下界 {lo:+.3f} > 0；未到超越档（+1.5）。")
    else:
        lines.append(f"- **未追平**：va = {va:+.3f}（需 ≥ +{G3_TRACE} 且 CI 下界 > 0，"
                     f"实测下界 {lo:+.3f}）。")
        if va < -2.0:
            lines.append("- va < −2 → 落空：10M 容量在 CPU 数据预算下无法兑现 → "
                         "按计划转 **M2+**（oracle critic / PTIE + 信念头 + 测试时搜索）。")
    lines += [
        "",
        "## 防 Goodhart 独立口径（G3 强制并列）",
        "",
        "- vs 三个教师风格 + roundrobin：见段末判读 JSON 的 refs 行"
        "（`reports/m1_s4b_judge.json` / G3 判读 JSON）；"
        "**绝不能只对 master 变强**。",
        "",
        "## M2-B 资源与吞吐（实跑）",
        "",
        f"- total 中位 {_fmt(st_b.get('total_med'), '{:.1f}')} s/轮 · "
        f"samples 中位 {_fmt(st_b.get('samples_med'), '{:.0f}')} · "
        f"KL 率 max {_fmt(st_b.get('klr_max'))} · "
        f"RSS 峰值 {_fmt(st_b.get('rss_peak'), '{:.2f}')} GB。",
        "",
        "---", "",
        "相关：`reports/M2A_阶段结果与解读.md`、`reports/G1_s4a_it300.json`、"
        "`reports/_m2_chain_state.json`（链路各阶段时点）。",
    ]
    REPORT_M2.write_text("\n".join(lines), encoding="utf-8")
    say(f"报告 → {REPORT_M2.name}")


# ---------------------------------------------------------------------------
# 训练起动（幂等：已跑完 / 已在跑 则不重复起）
# ---------------------------------------------------------------------------
def ensure_training(seg: str, cfg: dict, workers: int, block: bool = True) -> str:
    """确认训练在跑（已在跑则采纳，跑满则跳过，否则起动）。

    ``block=False`` 时**起完就返回**——门禁（G2@400/800）要在训练进行中跑，
    那才是"撤教师塌陷"的早警点（2026-10-15 C7：早先默认阻塞，门禁被排到 1200 轮之后）。
    """
    run = f"m1_{seg}"
    stem = Path(cfg["out"]).stem
    pid_file = ROOT / "reports" / f"_{run}.pid"
    if _pid_alive(pid_file, stem):
        say(f"{run}: 训练进程已在跑（pid 文件存在且存活）→ 采纳，不重复起")
        return "adopted"
    model = ROOT / cfg["out"]
    last = find_latest_ckpt(stem)
    if model.exists() and last and last[1] >= cfg["iters"]:
        say(f"{run}: 已跑满（{last[1]} 轮存档 + 主权重在）→ 跳过起动")
        return "done"
    say(f"{run}: 起动（{cfg['iters']} 轮，out={cfg['out']}，"
        f"{'阻塞等待' if block else '非阻塞（门禁将在跑动中插入）'}）")
    return launch(seg, cfg, workers, foreground=True, settle=8.0, wait=block)


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description="M2 专用自动链（A 收尾 → G1 → B → 门禁）")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--resume", action="store_true",
                    help="中断后续跑（各阶段幂等，其实不加也安全；此参数只为语义明确）")
    ap.add_argument("--workers", type=int, default=TRAIN_WORKERS)
    ap.add_argument("--judge-workers", type=int, default=JUDGE_WORKERS)
    ap.add_argument("--threads", type=int, default=24)
    args = ap.parse_args()

    gates: list[dict] = []
    rerun = False
    say("=== M2 自动链起动 ===")

    if args.dry_run:
        print(f"""
[dry-run] 链路：
  1. 等 {RUN_A} 到 600 轮（体检 + 崩溃自动 resume 一次）
  2. G1 判决 ← {G1_VERDICT.name}（缺失则补跑；门槛 va ≥ {G1_THRESHOLD:+.1f}）
  3. 杀旧守护 → 段末收尾（{DEALS} 副 + 健康度）→ {REPORT_A.name}
  4. G1 通过 → M2-B（s4b，1200 轮，β=0；对手池按 A 段强度二选一）
     G1 未通过 → 关 β 重跑 A 段（{SEG_RERUN}，600 轮）→ 再进 M2-B
  5. G2@400 / G2@800 / G3@1200（{DEALS} 副；G3 双档 {G3_TRACE:+.1f}/{G3_SURPASS:+.1f}）
  6. → {REPORT_M2.name}
[dry-run] 训练 workers={args.workers}  判读 workers={args.judge_workers}  threads={args.threads}
""")
        return 0

    # ---- 1. 等 M2-A ----
    say(f"[阶段1] 等 {RUN_A} 到 600 轮（当前 {_count_iters(LOG_A)} 轮）")
    save_state(phase="wait_a")
    st = wait_train(RUN_A, STEM_A, 600, max_wait_h=12,
                    allow_resume=True, workers=args.workers)
    if st != "done":
        save_state(phase="wait_a_failed", status=st)
        raise RuntimeError(f"{RUN_A} 未能跑完（{st}）—— 停链，请人工排查")

    # ---- 2. G1 判决 ----
    say("[阶段2] G1 判决")
    save_state(phase="g1")
    g1 = ensure_g1(args.judge_workers)
    gates.append(dict(g1, gate="G1"))

    # ---- 3. 收尾 + A 段报告 ----
    say("[阶段3] 段末收尾 + M2A 报告")
    save_state(phase="finalize_a")
    kill_guardians("_m1_s4a")
    js_a = finalize(RUN_A, STEM_A, DEALS, args.threads)
    write_m2a_report(json.loads(js_a.read_text(encoding="utf-8")), g1, rerun=False)

    # ---- 4. G1 分支 ----
    init_model = f"models/{STEM_A}.pt"
    if not g1["passed"]:
        rerun = True
        say(f"[阶段4] G1 未通过 → 关 β 重跑 A 段（{SEG_RERUN}，600 轮，β=0）")
        save_state(phase="rerun_a")
        cfg2 = dict(SEGMENTS["s4a"])
        cfg2["out"] = f"models/{STEM_RERUN}.pt"
        cfg2["extra"] = ["--klt-beta", "0.0", "--master-seat-prob", "0.30",
                         "--lr-min", "4e-6", "--save-every", "20"]
        SEGMENTS[SEG_RERUN] = cfg2
        kill_guardians(f"_res_monitor_{RUN_A}")
        ensure_training(SEG_RERUN, cfg2, args.workers)
        st = wait_train(f"m1_{SEG_RERUN}", STEM_RERUN, 600, max_wait_h=12,
                        allow_resume=True, workers=args.workers)
        if st != "done":
            save_state(phase="rerun_failed", status=st)
            raise RuntimeError(f"{SEG_RERUN} 未能跑完（{st}）—— 停链")
        kill_guardians(f"_res_monitor_m1_{SEG_RERUN}")
        js_r = finalize(f"m1_{SEG_RERUN}", STEM_RERUN, DEALS, args.threads)
        write_m2a_report(json.loads(js_r.read_text(encoding="utf-8")), g1, rerun=True)
        init_model = f"models/{STEM_RERUN}.pt"
        say("[阶段4] 重跑完成 → M2-B 改从 dushan_m2_a2 起（对手池取保守默认）")
    else:
        say("[阶段4] G1 通过 → M2-B 从 dushan_m2_a 起")

    # ---- 5. M2-B + 门禁 ----
    say("[阶段5] M2-B（1200 轮）")
    save_state(phase="m2b", init=init_model)
    cfgb = dict(SEGMENTS["s4b"])
    cfgb["init"] = init_model
    ex = list(cfgb["extra"])
    if "--snapshot-pool" in ex:
        va, ci = g1["vs_anchor"], g1["vs_anchor_ci95"]
        strong = va - ci > 0                       # A 段已显著强于 master
        pool = ("dushan_m2_a,dushan_m1_s3,dushan_master" if strong
                else "dushan_m1_s3")
        ex[ex.index("--snapshot-pool") + 1] = pool
        say(f"  对手池 = {pool}（A 段{'显著强于' if strong else '未显著强于'} master）")
    cfgb["extra"] = ex
    kill_guardians(f"_res_monitor_{RUN_B}")
    # ★ 非阻塞起跑：门禁 G2@400/G2@800 必须在**训练进行中**跑（撤教师塌陷的早警点）。
    #   早先用阻塞式起动，门禁被排到 1200 轮之后 —— 判据还在、但早警功能失效（C7）。
    ensure_training("s4b", cfgb, args.workers, block=False)

    stem_b = STEM_B
    log_b = ROOT / "reports" / f"_{RUN_B}.out"
    # G2@400 / G2@800：判「单调不减」（读数沿 G1 → 400 → 800 传递）
    prev_va, prev_ci = g1["vs_anchor"], g1["vs_anchor_ci95"]
    for at in (400, 800):
        out = ROOT / "reports" / f"G2_s4b_it{at:04d}.json"
        say(f"[阶段5] G2@{at}（等存档 → 判读；上次读数 {prev_va:+.3f}）")
        save_state(phase=f"g2_{at}")
        r = subprocess.run([PY, GATE, "--stem", stem_b, "--at", str(at),
                            "--gate", f"G2-{at}",
                            "--threshold", "-99",
                            "--monotonic-prev", str(prev_va),
                            "--prev-ci", str(prev_ci),
                            "--deals", str(DEALS), "--seed", str(SEED),
                            "--workers", str(args.judge_workers), "--threads", "2",
                            "--worker-threads", "1",
                            "--log", str(log_b.relative_to(ROOT)),
                            "--stall-min", "25", "--max-wait-h", "16",
                            "--out", str(out)], cwd=str(ROOT))
        if r.returncode != 0:
            save_state(phase=f"g2_{at}_failed")
            raise RuntimeError(f"G2@{at} 失败 —— 停链（训练可能已死，见 gate_watch 输出）")
        v = json.loads(out.with_suffix(".verdict.json").read_text(encoding="utf-8"))
        gates.append(dict(v, gate=f"G2-{at}"))
        prev_va, prev_ci = v["vs_anchor"], v["vs_anchor_ci95"]
        if not v["passed"]:
            say(f"⚠️ G2@{at} 未通过（撤教师塌陷信号）—— 按计划继续到下一门禁，"
                "但要在报告里标注")

    # 等 M2-B 跑满 → G3@1200（一次判读同时出 G2 第三点与 G3 双档）
    say("[阶段5] 等 M2-B 到 1200 轮")
    st = wait_train(RUN_B, stem_b, 1200, max_wait_h=20,
                    allow_resume=True, workers=args.workers)
    if st != "done":
        save_state(phase="m2b_failed", status=st)
        raise RuntimeError(f"{RUN_B} 未能跑完（{st}）—— 停链")
    kill_guardians(f"_res_monitor_{RUN_B}")
    js_b = finalize(RUN_B, stem_b, DEALS, args.threads)

    out3 = ROOT / "reports" / "G3_s4b_it1200.json"
    say("[阶段5] G3@1200（追平 +0.5 / 超越 +1.5，均要求 CI 下界 > 0）")
    save_state(phase="g3")
    r = subprocess.run([PY, GATE, "--stem", stem_b, "--at", "1200", "--gate", "G3",
                        "--threshold", str(G3_TRACE), "--threshold2", str(G3_SURPASS),
                        "--ci-lower-min", "0",
                        "--monotonic-prev", str(prev_va), "--prev-ci", str(prev_ci),
                        "--baseline-name", "M1-S3",
                        "--baseline", str(S3_BASELINE), "--baseline-ci",
                        str(S3_BASELINE_CI),
                        "--deals", str(DEALS), "--seed", str(SEED),
                        "--workers", str(args.judge_workers), "--threads", "2",
                        "--worker-threads", "1",
                        "--out", str(out3)], cwd=str(ROOT))
    if r.returncode != 0:
        save_state(phase="g3_failed")
        raise RuntimeError("G3 判读失败 —— 停链")
    v3 = json.loads(out3.with_suffix(".verdict.json").read_text(encoding="utf-8"))
    gates.append(dict(v3, gate="G3"))

    # ---- 6. 总报告 ----
    say("[阶段6] 写 M2 总报告")
    save_state(phase="report", done=True)
    write_m2_report(json.loads(js_a.read_text(encoding="utf-8")), gates,
                    _train_stats(log_b, ROOT / "reports" / f"_res_monitor_{RUN_B}.csv"))
    say("=== 全链完成 ===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
