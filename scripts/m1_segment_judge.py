"""段末判读：把「训练健康度」与「真实强度」两道门合成一条命令。

为什么需要这个脚本
------------------
`docs/P3-M1训练执行计划.md` 把长跑切成 S0..S3 四段，每段末都要做两件手工活：
① 看日志判健康度（熵是否坍缩、collect 是否尖峰、是否被系统睡眠冻结、内存是否失控）；
② 用 `score_checkpoints.py` 在**共用牌池**上做配对赛，再自己去 JSON 里对数字。

手工做的代价是：判据散在脑子里，换个段就忘；而且两次事故（22.2 h 睡眠冻结、
s9 因 nohup 脱管失联）都发生在"没人去看日志"的长时间窗口里。

本脚本把两道门固化成可复现的一条命令，输出一段结论 + 一份 JSON。

阶段 A：健康度（秒级，纯读日志/CSV）
    A1 熵带宽       —— 末段均值必须仍在 [1.6, 2.4]，且相对首段跌幅 < 0.25
    A2 collect 稳定 —— 中位数 ±60% 内无离群（采集进程池是否被拖垮）
    A3 每轮 wall 时间 —— 相对中位数的离群倍数（>3× 判定"疑似冻结"）
    A4 冻结交叉验证 —— `_res_monitor.csv` 的打点间隔是否出现 >> interval 的空洞
    A5 内存         —— 主进程 RSS / 系统内存峰值
    A6 对手池活体检查 —— 快照池首次生效后，samples/轮 不得塌陷（这是 vcollect
                        两个缓存缺陷的**运行时**守卫：缺陷在时样本数会掉 ~4%）
    A7 KL 锚点预热  —— `anchor-warmup` 期间 kc 必须为 0（冷启动锚点不能是随机策略）

阶段 B：真实强度（分钟级，跑 `score_checkpoints.py`）
    取 `models/_ckpts/` 里本段各存档点，加参照模型，在共用牌池上做配对赛，
    输出「各点 vs 锚点」的净分差与 95% CI —— 这才是有分辨力的口径。

用法::

    # S0 冒烟（只判健康度，省时间）
    python scripts/m1_segment_judge.py --run m1_smoke --skip-duel

    # 段末完整判读（健康度 + 配对赛）
    python scripts/m1_segment_judge.py --run m1_smoke --deals 600 --threads 24

    # S1 末：与上一段对比，并追加 s11 作外部参照
    python scripts/m1_segment_judge.py --run m1_s1 --extra dushan_big_s11 --deals 1200
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import statistics as st
import subprocess
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

#: 健康度判据（集中在这里，改判据只改这一处）
HEALTH = dict(
    # A1a：熵带**按本段首 k 轮的均值标定**（R5 修正）。
    # 为什么不再用固定 [1.6, 2.4]：那是一段**冷启动**（随机初始化 → 熵≈2.2）标定的，
    # 而 M1-S2 起的每段都是**热启动**（段首熵≈1.5）。固定带会让每一段的 A1a 必假 FAIL
    # —— 2026-10-13 的 S2/S3 判读都栽在这里。改成比例带后判据变成
    # 「熵是否守住段首基线的健康比例」，与启动方式无关。
    entropy_lo_ratio=0.68, entropy_hi_ratio=1.20,
    entropy_lo=1.6, entropy_hi=2.4,          # 仅作参考信息打印，不再参与判定
    entropy_drop_max=0.25,
    # A3：冻结尖峰检测跳过启动窗口（R5）—— 进程池 spawn、首次 eval、
    # torch 算子选择都会落在头几轮，那时的 wall 尖峰是**结构性的**，不是冻结。
    wall_skip_iters=25,
    collect_outlier=0.60, wall_outlier=3.0, monitor_gap_factor=3.0,
    # A1c：判定"熵控制器是否有效"时，允许"熵本来就在死区里"这一情形。
    # ★ 0.95 必须与 `CoefController` 的默认值 / M2_COMMON 一致 —— 这个数字是
    #   按实测标定的（熵的自然稳态只有段首的 91.6%，0.75 不可达 → C5），
    #   三处任一处写旧值，A1c 就会按错的 H* 判"离死区多远"。
    ent_target_ratio=0.95, ent_deadzone=0.03,
    # A8：samples/轮 —— **不再用绝对区间**（2026-10-17 修）。
    # 历史：原口径 [8000,13000] 按"teacher 0.05 / past 0.30"标定；M2-A 起 master
    # 占 30% → 天然降到 ~8000，放宽成 [5000,14000]；m2p3 把 games/iter 翻倍到 384
    # （15.9k 样本）又把 [5000,14000] 打穿 ⇒ 每加一档数据量就假 FAIL 一次。
    # 现口径：绝对带只做"量级级"兜底（防止塌到几百或爆到几万），真正的故障判据是
    # **段内漂移**（末 10% 中位 vs 首 10% 中位）—— 它抓的是"对手池中途静默消失"，
    # 那才是这条检查想防的事。显式给 lo/hi 时仍按显式值判（向后兼容）。
    samples_lo=None, samples_hi=None,
    samples_abs_lo=2000, samples_abs_hi=60000, samples_drift_max=0.40,
)

_ITER_RE = re.compile(
    r"^iter\s+(?P<iter>\d+)\s+samples=\s*(?P<samples>\d+)\s+"
    r"R=(?P<reward>[-+][\d.]+)\s+pi=(?P<policy_loss>[-+][\d.]+)\s+"
    r"V=(?P<value_loss>[\d.]+)\s+H=(?P<entropy>[\d.]+)\s+KL=(?P<kl>[\d.]+)\s+"
    r"ec=(?P<ent_coef>[\d.]+)\s+kc=(?P<kl_coef>[\d.]+)\s+lr=(?P<lr>[\d.eE+-]+)\s+"
    r"collect=(?P<collect_s>[\d.]+)s\s+total=(?P<elapsed>[\d.]+)s"
    # M2-A：β/ε/klt 追加在**行尾**（必须放在 lr 之后，否则会切断上面
    # ec→kc→lr 的连续匹配）。整段可选，老日志照样解析。
    # klr = KL 的**每轮漂移率**（kl ÷ 锚点年龄）—— A3 的判据用率不用距离，
    # 因为 kl 是"距锚点的距离"，会随锚点变旧而单调增大，与"漂移是否异常"无关。
    r"(?:\s+beta=(?P<klt_beta>[\d.]+)\s+eps=(?P<explore_eps>[\d.]+)"
    r"\s+klt=(?P<klt>[-\d.]+))?"
    r"(?:\s+klr=(?P<kl_rate>[-\d.]+))?"
)

# 训练启动后由 CoefController 打出的**权威**熵基线（窗口与训练一致）。
# 形如：``[coef] 熵基线 H0=1.025（段首 25 轮） → H*=0.769；ent_coef 由 0.010 起控``
_COEF_H0_RE = re.compile(
    r"\[coef\]\s*熵基线\s*H0=(?P<h0>[\d.]+)\s*（段首\s*(?P<n>\d+)\s*轮）")


# --------------------------------------------------------------------------
# 读取
# --------------------------------------------------------------------------
def parse_log(path: Path) -> list[dict]:
    """从训练 stdout 解析逐轮记录。用于 history.json 还没落盘（<50 轮）时。"""
    out = []
    if not path.exists():
        return out
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        m = _ITER_RE.match(line.strip())
        if not m:
            continue
        g = m.groupdict()
        rec = {
            "iter": int(g["iter"]), "samples": int(g["samples"]),
            "reward": float(g["reward"]), "policy_loss": float(g["policy_loss"]),
            "value_loss": float(g["value_loss"]), "entropy": float(g["entropy"]),
            "kl": float(g["kl"]), "ent_coef": float(g["ent_coef"]),
            "kl_coef": float(g["kl_coef"]), "lr": float(g["lr"]),
            "collect_s": float(g["collect_s"]), "elapsed": float(g["elapsed"]),
        }
        for key in ("klt_beta", "explore_eps", "klt", "kl_rate"):   # M2-A（老日志没有 → 不写）
            if g.get(key) is not None:
                rec[key] = float(g[key])
        out.append(rec)
    return out


def train_h0(stdout: Path) -> tuple[float | None, int]:
    """从训练 stdout 里读**控制器实际使用的**熵基线 H₀ 与它的窗口长度。

    为什么不能用判读自己算的 ``mean(H[:k])``：两者的窗口口径不同 ——
    训练用 ``--ent-baseline-iters``（默认 25 轮），判读用 ``k = min(25, n//4)``
    （60 轮时 k=15）。热启动段前 25 轮熵还在往下走，两个窗口的均值能差 0.03+，
    于是判读会报 ``H*=0.790`` 而训练日志写 ``H*=0.769`` —— **报告与实测对不上**，
    而 A1c"控制器是否有效"恰恰要用训练那个 H*。所以直接读日志里的权威值。
    """
    if not stdout.exists():
        return None, 0
    for line in stdout.read_text(encoding="utf-8", errors="replace").splitlines():
        m = _COEF_H0_RE.search(line)
        if m:
            return float(m.group("h0")), int(m.group("n"))
    return None, 0


def load_history(run: str, explicit: str = "") -> tuple[list[dict], str]:
    """优先读 history.json，并用 stdout 解析结果**补齐它缺的字段**。

    2026-10-12：history.json 早期不写 ent_coef / kl_coef / lr（只在 stdout 打印），
    而 A7 要查 kl_coef → 只读 history 会取不到值（旧版直接 IndexError 崩掉）。
    这里改成按 iter 对齐后合并：history 的字段优先，stdout 补它缺的。
    """
    stdout = ROOT / "reports" / f"_{run}.out"
    log = {r["iter"]: r for r in parse_log(stdout)}
    hjson = Path(explicit) if explicit else ROOT / "reports" / f"{run}_history.json"
    if hjson.exists() and hjson.stat().st_size > 0:
        try:
            data = json.loads(hjson.read_text(encoding="utf-8"))
            if isinstance(data, list) and data:
                merged = []
                for r in data:
                    base = dict(log.get(r.get("iter"), {}))
                    base.update(r)                      # history 优先，stdout 补空缺
                    merged.append(base)
                note = str(hjson) + (f"（+ stdout 补 {len(log)} 轮）" if log else "")
                return merged, note
        except Exception:
            pass
    recs = list(log.values())
    return recs, str(stdout) if recs else "(无记录)"


def total_ram_gb() -> float:
    """机器物理内存（GB）。取不到就退回标定值，避免硬编码写死在判据里。"""
    try:
        import psutil
        return psutil.virtual_memory().total / 1024 ** 3
    except Exception:                                  # noqa: BLE001
        return 31.8


def load_monitor(run: str, explicit: str = "") -> tuple[list[dict], str]:
    """读资源打点 CSV。

    **不做静默回退**：早期版本在找不到 ``_res_monitor_{run}.csv`` 时会退到共享的
    ``_res_monitor.csv``，于是把上一次长跑的旧数据当成本次读数报了 A4/A5
    （实测 417 个陈旧打点）。宁可报"缺数据"，也不要拿别人的数据下结论。

    ``tree_rss_gb`` 是 2026-10-12 之后新增的列（目标进程树含 spawn worker）；
    老 CSV 没有这列时该字段为 0，A5 退回全系统口径。
    """
    cand = Path(explicit) if explicit else ROOT / "reports" / f"_res_monitor_{run}.csv"
    if not cand.exists():
        return [], ""
    rows = []
    with cand.open(encoding="utf-8") as f:
        for r in csv.DictReader(f):
            try:
                rows.append({"ts": float(r["ts"]), "iso": r["iso"],
                             "cpu_pct": float(r["cpu_pct"]),
                             "mem_used_gb": float(r["mem_used_gb"]),
                             "max_rss_gb": float(r["max_rss_gb"] or 0),
                             "tree_rss_gb": float(r.get("tree_rss_gb") or 0)})
            except (KeyError, ValueError):
                continue
    return rows, str(cand)


# --------------------------------------------------------------------------
# 阶段 A：健康度
# --------------------------------------------------------------------------
def _seg(recs: list[dict], key: str) -> list[float]:
    return [float(r[key]) for r in recs if key in r and r[key] is not None]


def health(recs: list[dict], mon: list[dict], mon_interval: float,
           snapshot_every: int, mon_src: str = "", anchor_warmup: int = 0,
           ent_target_ratio: float = None,
           ent_deadzone: float = None,
           train_h0: float | None = None,
           train_h0_iters: int = 0) -> dict:
    out: dict = {"checks": [], "n_iters": len(recs)}
    ratio = (HEALTH["ent_target_ratio"] if ent_target_ratio is None
             else float(ent_target_ratio))
    dead = (HEALTH["ent_deadzone"] if ent_deadzone is None
            else float(ent_deadzone))

    def chk(name, ok, detail, skip=False):
        """三态：PASS / FAIL / SKIP。

        SKIP 表示"数据还不够，现在还判不了"，**不计入**总判 —— 否则跑到一半时
        会报出一堆虚假的 FAIL，把真正的问题淹掉。
        """
        out["checks"].append({"name": name, "ok": bool(ok), "detail": detail,
                              "status": "skip" if skip else ("pass" if ok else "fail")})

    if not recs:
        chk("A0 数据可得性", False, "没有任何轮记录")
        out["all_ok"] = False
        return out

    n = len(recs)
    run_it = recs[-1]["iter"]

    # --- A1 熵带宽与坍缩 ---
    # 判据按**本段**首 k 轮标定（R5）：热启动段的健康熵本来就比冷启动低，
    # 用固定的 [1.6, 2.4] 判会让每一段都假 FAIL。
    H = _seg(recs, "entropy")
    k = max(1, min(25, n // 4))          # 首/末各取约 1/4（上限 25 轮）
    h_head, h_tail = st.mean(H[:k]), st.mean(H[-k:])
    # ★ H*（控制器的目标）必须用**训练实际用的 H₀** —— 它的窗口是
    #   --ent-baseline-iters（25 轮），而本判读窗口是 k=min(25, n//4)（60 轮时 15）。
    #   热启动段前 25 轮熵还在下滑，两个窗口均值能差 0.03+ → 判读会报 H*=0.790
    #   而训练日志写 0.769，报告与实测对不上。优先级：日志里的权威值 > 本判读窗口。
    h0_ctrl = float(train_h0) if train_h0 else h_head
    hstar = ratio * h0_ctrl
    band_lo = HEALTH["entropy_lo_ratio"] * h_head
    band_hi = HEALTH["entropy_hi_ratio"] * h_head
    out["entropy"] = {"head": round(h_head, 4), "tail": round(h_tail, 4),
                      "min": round(min(H), 4), "max": round(max(H), 4),
                      "drop": round(h_head - h_tail, 4),
                      "band": [round(band_lo, 3), round(band_hi, 3)],
                      "h0_ctrl": round(h0_ctrl, 4),
                      "h0_ctrl_iters": int(train_h0_iters or 0),
                      "h0_from_log": bool(train_h0),
                      "target_h_star": round(hstar, 3)}
    in_band = band_lo <= h_tail <= band_hi
    not_collapsed = (h_head - h_tail) < HEALTH["entropy_drop_max"]
    chk("A1a 熵在带内", in_band,
        f"末 {k} 轮均值 H={h_tail:.3f}，按段首基线标定的带 "
        f"[{band_lo:.2f}, {band_hi:.2f}]（= {HEALTH['entropy_lo_ratio']}×/{HEALTH['entropy_hi_ratio']}×"
        f"H₀, H₀={h_head:.3f}）")
    chk("A1b 熵未坍缩", not_collapsed,
        f"首 {k} 轮 {h_head:.3f} → 末 {k} 轮 {h_tail:.3f}，跌幅 "
        f"{h_head - h_tail:+.3f}（阈值 <{HEALTH['entropy_drop_max']}）")

    # --- A1c 熵控制器是否真的在起作用（M2-A · A1） ---
    # 为什么必须单独查：旧阈值规则（ent<floor→×1.8）在 S1–S3 共 2450 轮里
    # 把 ec 全程钉在 0.03，**从未动过一次** —— 这才是熵坍缩的直接机制。
    #
    # ★ 三态判定（2026-10-14 修）。原来的二选一会在**短跑**上假 FAIL：
    #   H 全程高于 H* 时控制器**按设计就该一动不动**（熵还太高，不需要加探索奖励），
    #   但 `settled` 也不成立（H 没停在死区里）→ 报 FAIL，把"正确地闲着"当成"失灵"。
    #   实测：修好版的 60 轮验证跑里 H≈0.96 而 H*=0.83，整段都没跨过目标。
    #   现在分三态：
    #     PASS  ec 动过（真在控制）          或  熵已停在 H*±2×死区（没活干，正常收敛）
    #     SKIP  熵**全程**高于 H*+死区（控制器按设计不动，本窗口无法判定其有效性）
    #     FAIL  其余（熵曾进入 H* 下方却仍不动 → 真失灵）
    EC = _seg(recs, "ent_coef")
    if EC:
        span = max(EC) - min(EC)
        settled = abs(H[-1] - hstar) <= 2.0 * dead
        idle_ok = min(H) > hstar + dead          # 熵从来没低到需要控制器出手
        sk = not (span > 1e-6 or settled) and idle_ok
        out["ent_controller"] = {"ec_min": round(min(EC), 4), "ec_max": round(max(EC), 4),
                                 "span": round(span, 4), "h_star": round(hstar, 3),
                                 "h_min": round(min(H), 3), "h_last": round(H[-1], 3),
                                 "idle_ok": bool(idle_ok),
                                 "reach_iters": sum(1 for e in EC if e > min(EC) + 1e-9)}
        # SKIP 的语义要说足：它不是"没问题"，而是"**没验证**"。
        # C5 就是这么躲过判读的 —— H 全程高于 H*（控制器按设计闲着），
        # 但根因是 --ent-target-ratio 不可达 → A1 事实上从未生效。
        # 所以 SKIP 文案里直接点出这一可能，并给出该改哪个参数。
        _src = (f"H₀={h0_ctrl:.3f}，取自训练日志（段首 {train_h0_iters or '?'} 轮）"
                if train_h0 else f"H₀={h0_ctrl:.3f}（判读窗口 {k} 轮，日志里未找到基线行）")
        chk("A1c 熵控制器有效", span > 1e-6 or settled,
            f"ec 区间 [{min(EC):.3f}, {max(EC):.3f}]（跨度 {span:.4f}，"
            f"{out['ent_controller']['reach_iters']}/{len(EC)} 轮离开过起点）；"
            f"H*={hstar:.3f}（{_src}），H 最低 {min(H):.3f} / 末轮 {H[-1]:.3f}，"
            + ("熵全程高于 H* → 控制器按设计保持不动，**本窗口无法判定 A1 是否有效**"
               "（若整段一直如此，检查 --ent-target-ratio 是否低到不可达）" if sk
               else "已停在死区内（控制器无需动作）" if settled else "远离死区"),
            skip=sk)

    # --- A2 collect 稳定性 ---
    C = _seg(recs, "collect_s")
    medC = st.median(C)
    worst = max(range(n), key=lambda i: abs(recs[i]["collect_s"] - medC))
    dev = abs(recs[min(worst, len(C) - 1)]["collect_s"] - medC) / medC if medC else 0
    out["collect_s"] = {"median": round(medC, 2), "min": round(min(C), 2),
                        "max": round(max(C), 2), "worst_iter": recs[worst]["iter"]}
    chk("A2 collect 无离群", dev <= HEALTH["collect_outlier"],
        f"中位 {medC:.1f}s，最差 iter {recs[worst]['iter']} 偏离 {dev * 100:.0f}%"
        f"（阈值 ≤{HEALTH['collect_outlier'] * 100:.0f}%）")

    # --- A3 每轮真实 wall 时间 ---
    # 跳过启动窗口（R5）：进程池 spawn、首次 eval、torch 算子选择都落在头几轮，
    # 那里的 wall 尖峰是**结构性的**（S3 判读时 A3 的尖峰全部落在启动窗口内，
    # 每次都误报"疑似冻结"）。
    skip_w = min(HEALTH["wall_skip_iters"], max(1, n // 4))
    # ★ 剔除 eval 记账错位（2026-10-17 修）：selfplay 里 `rec` 先构造、`run_eval()` 后执行，
    # eval 的耗时被记进**下一轮**的 elapsed。于是每 `--eval-every` 轮就会凭空出现一个
    # 200–300 s 的尖峰，被 A3 判成"疑似冻结"——m2p1/m2p2/m2p3/m2d1 四段连续 100% 复现
    # （尖峰恒在 `it%eval_every==1`），属于**固定假 FAIL**，会把真冻结信号埋掉。
    # 精确判据：上一轮带 `eval` 字段 ⇒ 本轮的 delta 含 eval 成本，不计入 A3。
    eval_shift = set()
    for i in range(1, n):
        if recs[i - 1].get("eval"):
            eval_shift.add(recs[i]["iter"])
    walls = []
    for i in range(1, n):
        d = recs[i]["elapsed"] - recs[i - 1]["elapsed"]
        if d > 0 and recs[i]["iter"] > skip_w and recs[i]["iter"] not in eval_shift:
            walls.append((recs[i]["iter"], d))
    if walls:
        medW = st.median([w for _, w in walls])
        susp = [(it, w) for it, w in walls if w > medW * HEALTH["wall_outlier"]]
        out["wall_s"] = {"median": round(medW, 1),
                         "skipped_head_iters": skip_w,
                         "skipped_eval_shifted": sorted(eval_shift),
                         "suspect": [{"iter": it, "s": round(w, 1)} for it, w in susp[:10]],
                         "n_suspect": len(susp)}
        chk("A3 无冻结尖峰", not susp,
            f"前 {skip_w} 轮不计（启动窗口）；已剔除 {len(eval_shift)} 轮 eval 记账错位；"
            f"中位 {medW:.1f}s/轮；疑似冻结 {len(susp)} 处"
            + (f"（如 iter {susp[0][0]} = {susp[0][1]:.0f}s）" if susp else ""))
    else:
        out["wall_s"] = {"note": f"有效轮数不足（前 {skip_w} 轮被跳过）"}
        chk("A3 无冻结尖峰", True, f"可判轮数不足（已跳过前 {skip_w} 轮），不判", skip=True)

    # --- A4 监视器打点空洞（旁路交叉验证） ---
    if len(mon) >= 3 and mon_interval > 0:
        gaps = [(mon[i]["iso"], mon[i]["ts"] - mon[i - 1]["ts"])
                for i in range(1, len(mon))]
        worst_gap = max(gaps, key=lambda x: x[1])
        out["monitor"] = {"points": len(mon), "interval_s": mon_interval,
                          "max_gap_s": round(worst_gap[1], 1),
                          "max_gap_at": worst_gap[0],
                          "mem_used_max_gb": round(max(m["mem_used_gb"] for m in mon), 2),
                          "rss_max_gb": round(max(m["max_rss_gb"] for m in mon), 2)}
        chk("A4 监视器无空洞", worst_gap[1] <= mon_interval * HEALTH["monitor_gap_factor"],
            f"{len(mon)} 点，最大间隔 {worst_gap[1]:.0f}s @ {worst_gap[0]}"
            f"（阈值 ≤{mon_interval * HEALTH['monitor_gap_factor']:.0f}s）")
        sys_max = out["monitor"]["mem_used_max_gb"]
        rss_max = out["monitor"]["rss_max_gb"]
        tree_max = max((m["tree_rss_gb"] for m in mon), default=0.0)
        ram = total_ram_gb()
        out["monitor"]["ram_total_gb"] = round(ram, 1)
        # 优先用「训练进程树」（含 spawn worker）判；否则退回全系统口径。
        # 为什么不能只看全系统：机器上还跑着浏览器/编辑器，会把它们的占用算到训练头上
        # （2026-10-12：系统峰值 30.13 GB 里只有 ≲9 GB 是训练，A5 因此误报 FAIL）。
        if tree_max > 0:
            out["monitor"]["tree_rss_max_gb"] = round(tree_max, 2)
            base = max(0.0, sys_max - tree_max)
            chk("A5 内存可控", tree_max < 0.70 * ram,
                f"训练进程树峰值 {tree_max:.2f} GB／机器 {ram:.1f} GB"
                f"（主进程 {rss_max:.2f} GB）；系统峰值 {sys_max:.2f} GB，"
                f"非训练占用 ≈{base:.2f} GB")
        else:
            chk("A5 内存可控", sys_max < 0.85 * ram,
                f"主进程峰值 RSS {rss_max:.2f} GB，系统内存峰值 {sys_max:.2f} / "
                f"{ram:.1f} GB（CSV 无 tree_rss_gb 列 → 按全系统口径判）")
    else:
        chk("A4 监视器无空洞", False,
            f"只有 {len(mon)} 个打点（需 ≥3）；数据源：{mon_src or '缺失'}",
            skip=True)
        chk("A5 内存可控", False, "同上，监视数据缺失", skip=True)

    # --- A6 对手池活体检查 ---
    # 快照池在第 snapshot_every 轮首次生效；其后 samples/轮 必须与之前同量级。
    # vcollect 的两个缓存缺陷在**运行时**的表现正是：池里明明有对手，样本数却掉 ~4%
    # （实测 1163 → 1116）。这里用中位数 + 0.95 阈值作粗筛，权威判据仍是
    # scripts/verify_snap_cache.py 里用 spy 直接断言"命中时对手池仍是 3 份"。
    need = snapshot_every + 8
    if snapshot_every > 0 and run_it > need:
        # 续训段（--iter-offset）编号不从 1 起：快照池在本段第 snapshot_every 轮
        # （局部计数）生效，全局编号 = 本段首轮前一行 + snapshot_every。
        # 旧代码硬编码 `iter <= snapshot_every+8`，对 m2c3（编号 601 起）窗口为空，
        # st.median(empty) 直接崩 —— m2c3 判读 2026-10-19 实测踩中。
        it0 = int(recs[0].get("iter") or 1) - 1
        before = [r["samples"] for r in recs
                  if it0 + snapshot_every < r["iter"] <= it0 + snapshot_every + 8]
        after = [r["samples"] for r in recs if r["iter"] > it0 + snapshot_every + 8]
        if not before or not after:
            out["snapshot_pool"] = {"first_active_iter": it0 + snapshot_every + 1,
                                    "note": "窗口内无记录，跳过（记录覆盖不足）"}
            chk("A6 对手池生效", False,
                f"窗口 ({it0 + snapshot_every},{it0 + snapshot_every + 8}] 内记录 "
                f"{len(before)} 条 / 之后 {len(after)} 条 —— 无法判定", skip=True)
        else:
            mb, ma = st.median(before), st.median(after)
            ratio = ma / mb if mb else 0
            out["snapshot_pool"] = {"median_before": mb, "median_after": ma,
                                    "ratio": round(ratio, 4),
                                    "first_active_iter": it0 + snapshot_every + 1}
            chk("A6 对手池生效", ratio >= 0.95,
                f"快照池自 iter {it0 + snapshot_every + 1} 起生效；samples/轮中位 "
                f"{mb:.0f} → {ma:.0f}（比值 {ratio:.3f}，阈值 ≥0.95）")
    elif snapshot_every > 0:
        out["snapshot_pool"] = {"first_active_iter": snapshot_every + 1,
                                "note": f"尚未跑满 iter {need}，暂不可判"}
        chk("A6 对手池生效", False,
            f"需 iter > {need} 才能判（当前 {run_it}）—— 样本不足", skip=True)

    # --- A7 KL 锚点预热 ---
    # 冷启动时 --anchor-warmup N 必须真的把 kc 压到 0：否则 KL 惩罚会把策略往一份
    # **随机**锚点上拽（旧代码正是因为把 adapt_coefs 放在赋值之后，kc 被顶回 3.0）。
    # R5 修正：``--anchor-warmup 0``（热启动，M1-S2 起都是）**本就不该**有 kc=0 的轮次
    # —— 旧版在这里必假 FAIL，等于给每一段热启动训练都记一笔"必输"。
    KC = _seg(recs, "kl_coef")
    if not KC:
        out["anchor_warmup"] = {"kc_zero_iters": None, "kc_tail": None, "iters": run_it}
        chk("A7 锚点预热生效", False,
            "记录里没有 kc 字段（history.json 不含 kl_coef 且 stdout 无逐轮行）—— 无法判定",
            skip=True)
    elif anchor_warmup <= 0:
        out["anchor_warmup"] = {"kc_zero_iters": 0, "kc_tail": round(KC[-1], 4),
                                "iters": run_it, "warmup": 0}
        chk("A7 锚点预热生效", True,
            f"--anchor-warmup=0（热启动）：锚点直接用 --init 的已有策略，"
            f"**不要求**出现 kc=0 的轮次（当前 kc={KC[-1]:.4f}）", skip=True)
    else:
        n_zero = sum(1 for kc in KC if kc == 0.0)
        out["anchor_warmup"] = {"kc_zero_iters": n_zero, "kc_tail": round(KC[-1], 4),
                                "iters": run_it, "warmup": anchor_warmup}
        chk("A7 锚点预热生效", n_zero > 0,
            f"{n_zero}/{len(KC)} 轮 kc=0（预热期不把随机策略当锚点），当前 kc={KC[-1]:.3f}")

    # --- A9 KLT / ε 退火方向（M2-A · A2/A4） ---
    # 这两项都是"必须单调朝一个方向走"的调度量。方向写反（例如 ε 越训越大、
    # β 不降反升）不会报错，只会静默毁掉整段 —— 所以把它做成断言。
    B, E = _seg(recs, "klt_beta"), _seg(recs, "explore_eps")
    if B or E:
        parts, ok = [], True
        if B:
            mono = all(B[i] <= B[i - 1] + 1e-9 for i in range(1, len(B)))
            ok = ok and mono
            parts.append(f"β {B[0]:.3f} → {B[-1]:.3f}"
                         f"（{'单调不增' if mono else '**非单调**'}）")
        if E:
            mono = all(E[i] <= E[i - 1] + 1e-9 for i in range(1, len(E)))
            ok = ok and mono
            parts.append(f"ε {E[0]:.3f} → {E[-1]:.3f}"
                         f"（{'单调不增' if mono else '**非单调**'}）")
        out["anneal"] = {"klt_beta": [B[0], B[-1]] if B else None,
                         "explore_eps": [E[0], E[-1]] if E else None}
        chk("A9 β/ε 退火方向", ok, "；".join(parts) + "（两者都必须单调不增）")
    else:
        chk("A9 β/ε 退火方向", False, "记录里没有 beta/eps 字段（M2-A 之前的分段）—— 不判",
            skip=True)

    # --- A8 样本量量级 + 段内漂移 ---
    S = _seg(recs, "samples")
    medS = st.median(S)
    k = max(1, len(S) // 10)
    headS, tailS = st.median(S[:k]), st.median(S[-k:])
    drift = abs(tailS - headS) / max(1.0, headS)
    lo, hi = HEALTH["samples_lo"], HEALTH["samples_hi"]
    if lo is not None and hi is not None:            # 显式区间（旧口径，向后兼容）
        band_ok, band_txt = lo <= medS <= hi, f"显式区间 {lo}–{hi}"
    else:                                            # 自动口径
        band_ok = HEALTH["samples_abs_lo"] <= medS <= HEALTH["samples_abs_hi"]
        band_txt = f"量级兜底 {HEALTH['samples_abs_lo']}–{HEALTH['samples_abs_hi']}"
    drift_ok = drift <= HEALTH["samples_drift_max"]
    out["samples"] = {"median": medS, "min": min(S), "max": max(S),
                      "head10_median": headS, "tail10_median": tailS,
                      "drift": round(drift, 3), "band": band_txt}
    chk("A8 样本量正常", band_ok and drift_ok,
        f"中位 {medS:.0f} 样本/轮（{band_txt}）；段内漂移 "
        f"{drift * 100:.0f}%（首 10% {headS:.0f} → 末 10% {tailS:.0f}，阈值 ≤"
        f"{HEALTH['samples_drift_max'] * 100:.0f}%）")

    out["all_ok"] = all(c["ok"] for c in out["checks"] if c["status"] != "skip")
    out["n_skip"] = sum(1 for c in out["checks"] if c["status"] == "skip")
    out["source"] = {"iters": run_it}
    return out


# --------------------------------------------------------------------------
# 阶段 B：共用牌池配对赛
# --------------------------------------------------------------------------
def list_ckpts(run: str, prefix: str, max_n: int) -> tuple[list[str], str]:
    """列出本段存档点（裸名，相对 models/），均匀抽样到 max_n 个。

    命名约定：``_dump_ckpt`` 写出的是 ``{模型名}_it{NNNN}.pt``。这里**必须**用
    ``{prefix}_it*.pt`` 精确匹配，不能用 ``{prefix}*.pt`` —— 后者会让
    ``dushan_b2_p_a`` 顺手把 ``dushan_b2_p_abc_it*`` 也吞进来，把两个消融臂
    混进同一场配对赛，结论直接错掉。
    """
    d = ROOT / "models" / "_ckpts"
    if not d.exists():
        return [], ""
    pat = f"{prefix}*.pt" if prefix.endswith("_it") else f"{prefix}_it*.pt"
    stems = sorted((p.stem for p in d.glob(pat)), key=_iter_of)
    stems = [s for s in stems if _iter_of(s) >= 0]
    if not stems:
        return [], ""
    if max_n and len(stems) > max_n:      # 均匀抽样，保证首尾都在
        idx = [round(i * (len(stems) - 1) / (max_n - 1)) for i in range(max_n)]
        stems = sorted({stems[i] for i in idx}, key=_iter_of)
    return [f"_ckpts/{s}" for s in stems], stems[0]


def _iter_of(stem: str) -> int:
    m = re.search(r"_it(\d+)$", stem)
    return int(m.group(1)) if m else -1


def run_duel(ckpts: list[str], anchor: str, ref: str, deals: int,
             threads: int, out_json: Path) -> dict | None:
    cmd = [sys.executable, str(ROOT / "scripts" / "score_checkpoints.py"),
           "--ckpts", ",".join(ckpts), "--ref", ref, "--deals", str(deals),
           "--threads", str(threads), "--anchor", anchor,
           "--out", str(out_json)]
    print("  $ " + " ".join(cmd[1:]), flush=True)
    r = subprocess.run(cmd, cwd=str(ROOT))
    if r.returncode != 0 or not out_json.exists():
        return None
    return json.loads(out_json.read_text(encoding="utf-8"))


# --------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(
        description="段末判读：健康度 + 共用牌池配对赛")
    ap.add_argument("--run", required=True,
                    help="运行名：用于定位 reports/{run}_history.json、reports/_{run}.out、"
                         "models/_ckpts/{run}_it*.pt、reports/_res_monitor_{run}.csv")
    ap.add_argument("--history", default="", help="显式指定 history.json 路径")
    ap.add_argument("--ckpt-prefix", default="", help="存档点前缀，默认取 --run")
    ap.add_argument("--ref", default="teacher:balanced|teacher:tenpai_rush|teacher:defensive",
                    help="三个对手，用 | 分隔")
    ap.add_argument("--extra", default="",
                    help="额外参评的模型裸名（逗号分隔），如 dushan_big_s11,dushan_master")
    ap.add_argument("--deals", type=int, default=600)
    ap.add_argument("--max-ckpts", type=int, default=5, help="均匀抽样上限")
    ap.add_argument("--threads", type=int, default=24)
    ap.add_argument("--monitor-interval", type=float, default=60.0)
    ap.add_argument("--monitor-csv", default="",
                    help="显式指定资源打点 CSV（默认 reports/_res_monitor_{run}.csv）")
    ap.add_argument("--snapshot-every", type=int, default=16)
    ap.add_argument("--anchor-warmup", type=int, default=-1,
                    help="本段的 --anchor-warmup 值。-1=自动推断：若最前几轮出现 kc=0，"
                         "说明是冷启动预热段（A7 按严格口径判）；否则按热启动处理"
                         "（A7 不要求 kc=0，避免给每段热启动都记一笔假 FAIL，见 R5）。")
    ap.add_argument("--ent-target-ratio", type=float, default=HEALTH["ent_target_ratio"],
                    help="与 selfplay 的 --ent-target-ratio 对齐；用于 A1c 的 H* 与"
                         "「控制器是否只是在死区内静止」的判定。")
    ap.add_argument("--ent-deadzone", type=float, default=HEALTH["ent_deadzone"])
    ap.add_argument("--skip-duel", action="store_true", help="只跑阶段 A")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    out_json = Path(args.out) if args.out else ROOT / "reports" / f"{args.run}_judge.json"

    print(f"\n{'=' * 72}\n段末判读：{args.run}\n{'=' * 72}")

    # ---------------- 阶段 A ----------------
    recs, src = load_history(args.run, args.history)
    mon, mon_src = load_monitor(args.run, args.monitor_csv)
    print(f"\n[阶段 A] 健康度  记录来源：{src}（{len(recs)} 轮）"
          f"{f'，监视点 {len(mon)} 个 @ {mon_src}' if mon else '，无监视数据（A4/A5 将跳过）'}")
    warm = args.anchor_warmup
    if warm < 0:
        # 自动推断：冷启动预热段的特征就是**前若干轮 kc 被强制成 0**。
        head_kc = [float(r["kl_coef"]) for r in recs[:5] if r.get("kl_coef") is not None]
        warm = 1 if (head_kc and head_kc[0] == 0.0) else 0
        print(f"  [A7] --anchor-warmup 自动推断 = {warm}"
              f"（前 5 轮 kc={head_kc}）")
    # 训练实际使用的熵基线（从 stdout 的 `[coef] 熵基线 H0=...` 行读）。
    # 必须优先于判读窗口自算的均值 —— 两者窗口不同（25 vs n//4），
    # 会让报告里的 H* 与训练日志对不上，而 A1c 要用训练那个 H*。
    _stdout = ROOT / "reports" / f"_{args.run}.out"
    h0_log, h0_log_n = train_h0(_stdout)
    if h0_log:
        print(f"  [A1] 训练日志里的熵基线 H₀={h0_log:.4f}"
              f"（段首 {h0_log_n} 轮）→ 判读统一用这个口径")
    hres = health(recs, mon, args.monitor_interval, args.snapshot_every, mon_src,
                  anchor_warmup=warm, ent_target_ratio=args.ent_target_ratio,
                  ent_deadzone=args.ent_deadzone,
                  train_h0=h0_log, train_h0_iters=h0_log_n)
    tag = {"pass": "PASS", "fail": "FAIL", "skip": "SKIP"}
    for c in hres["checks"]:
        print(f"  {tag[c['status']]}  {c['name']}：{c['detail']}")
    verdict = "全部通过" if hres.get("all_ok") else "有未通过项"
    if hres.get("n_skip"):
        verdict += f"（{hres['n_skip']} 项因数据不足跳过）"
    print(f"  → 阶段 A 总判：{verdict}")

    result = {"run": args.run, "when": datetime.now().isoformat(timespec="seconds"),
              "health": hres}

    # ---------------- 阶段 B ----------------
    if not args.skip_duel:
        prefix = args.ckpt_prefix or args.run
        ckpts, _ = list_ckpts(args.run, prefix, args.max_ckpts)
        if not ckpts and not args.ckpt_prefix:
            # 兼容常见命名：训练脚本的 --out 多带 dushan_ 前缀（run=m1_smoke →
            # 落盘是 models/_ckpts/dushan_m1_smoke_it*.pt）。不自动回退的话，
            # 阶段 B 会误判「一个存档点都没有」而整段跳过。
            alt = f"dushan_{prefix}"
            ckpts, _ = list_ckpts(args.run, alt, args.max_ckpts)
            if ckpts:
                prefix = alt
        extra = [e.strip() for e in args.extra.split(",") if e.strip()]
        if not ckpts:
            print(f"\n[阶段 B] 跳过：models/_ckpts/{prefix}*.pt 一个都没有"
                  f"（还没到第一个 --save-every 点？）")
        else:
            anchor = ckpts[0]                 # 以本段最早的点为基准 → 看"有没有在涨"
            allc = ckpts + [e for e in extra if e not in ckpts]
            print(f"\n[阶段 B] 共用牌池配对赛  {len(allc)} 个模型 × {args.deals} 副"
                  f"（锚点 = {anchor}）")
            duel = run_duel(allc, anchor, args.ref, args.deals, args.threads,
                            out_json.with_name(out_json.stem + "_duel.json"))
            if duel is None:
                print("  → 配对赛失败（见上方子进程输出）")
                result["duel"] = None
            else:
                print(f"\n  {'模型':<34}{'水平':>9}{'±95%CI':>9}{'vs 锚点':>10}{'±95%CI':>9}  结论")
                for r in duel["rows"]:
                    print(f"  {r['ckpt'].split('/')[-1]:<34}{r['level']:>+9.3f}"
                          f"{r['level_ci95']:>9.3f}{r['vs_anchor']:>+10.3f}"
                          f"{r['vs_anchor_ci95']:>9.3f}  {r['verdict']}")
                result["duel"] = duel
                # 结论行：末点相对首点
                last = next((r for r in duel["rows"] if r["ckpt"] == ckpts[-1]), None)
                if last:
                    result["verdict_progress"] = {
                        "from": ckpts[0], "to": ckpts[-1],
                        "delta": last["vs_anchor"], "ci95": last["vs_anchor_ci95"],
                        "significant_up": last["vs_anchor"] - last["vs_anchor_ci95"] > 0,
                    }
                    print(f"\n  → 本段进展：{ckpts[-1].split('/')[-1]} 相对 "
                          f"{ckpts[0].split('/')[-1]} {last['vs_anchor']:+.3f} "
                          f"± {last['vs_anchor_ci95']:.3f}  "
                          f"（{'显著上升' if result['verdict_progress']['significant_up'] else '未达显著'}）")

    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n判读结果 -> {out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
