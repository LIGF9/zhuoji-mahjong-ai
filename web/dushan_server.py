"""
网页版独山麻将对局服务 —— 你 vs 大师模型 / 教师 / 随机。

启动::

    python web/dushan_server.py --port 8770
    # 然后浏览器打开 http://127.0.0.1:8770

与 zhuoji 版 server.py 的差异
----------------------------
* 规则引擎换成 :class:`zhuoji.dushan.DushanGame`；
* 快照公开**报叫**、**横鸡轮**、**热炮**等独山公开信息（隐藏手牌依旧只给张数）；
* ``/api/hint`` 返回带**理由**的 Top-3 推荐（牌效率 + 安全性 + 鸡分解释）；
* ``/api/stats``：本会话 + 历史累计的 胡牌/点炮/冲锋鸡/包鸡/责任鸡 统计；
* 每局记录落盘 ``web/history/records.jsonl``，可在前端查看对局记录；
* 服务 ``/assets/**``（牌面/桌布/精灵图/音效，均来自 DushanMajiang 工程）。
"""
from __future__ import annotations

import argparse
import io
import json
import os
import pickle
import random
import re
import socket
import sys
import threading
import time
import traceback
import uuid
import zipfile
from collections import OrderedDict, deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch  # noqa: E402

import coach  # noqa: E402  # web/ 目录下的教练解释引擎
from zhuoji.agent import NetAgent  # noqa: E402
from zhuoji.bots import TEACHER_STYLES, Bot, HeuristicBot, RandomBot  # noqa: E402
from zhuoji.dushan import (  # noqa: E402
    DushanConfig,
    DushanGame,
    fanji_tiles_from,
    norm_fanji,
)
from zhuoji.encoder import index_to_action  # noqa: E402
from zhuoji.fan import (  # noqa: E402
    MELD_KONG_ADDED,
    MELD_KONG_CONCEALED,
    MELD_KONG_EXPOSED,
    MELD_PONG,
    winning_tiles,
)
from zhuoji.net import (  # noqa: E402
    build_from_ckpt, net_config_from_ckpt, net_meta)
from zhuoji.rules import (  # noqa: E402
    ANGANG,
    BUGANG,
    DISCARD,
    HU,
    MINGGANG,
    PASS,
    PENG,
    Action,
    Phase,
)
from zhuoji.tiles import (  # noqa: E402
    NUM_TILE_TYPES, YAOJI, tile_cn, tile_name,
)

STATIC = Path(__file__).resolve().parent / "static"
ASSETS = STATIC / "assets"


def _page_ver() -> str:
    """dushan.html 的轻量指纹（mtime+size）。旧标签页用它发现页面已更新并提示刷新——
    本页是长驻 SPA，改动前端后不刷新就永远跑旧 JS（曾因此出现"修了但没生效"的误会）。"""
    try:
        st = (STATIC / "dushan.html").stat()
        return f"{st.st_mtime_ns:x}-{st.st_size:x}"
    except OSError:
        return ""
MODEL_DIR = ROOT / "models"
# 历史目录（records.jsonl / replays.jsonl）。默认 web/history/；
# ``DUSHAN_HISTORY_DIR`` 可覆盖它 —— 只为让端到端测试跑在**隔离的临时目录**里，
# 不必去动用户真实的战绩文件（默认值不变，正常启动行为完全一致）。
HISTORY_DIR = Path(os.environ.get("DUSHAN_HISTORY_DIR")
                   or (Path(__file__).resolve().parent / "history"))
# ---- 回放缓存（2026-10-12）------------------------------------------------
# 对局记录面板点某一局就要回放，所以按 rid 随机取。文件是追加写的，
# 随机取要么全量读进来（文件大了很浪费），要么建索引。
# 这里取折中：**只读文件尾部 8MB** 建一个「最近 N 局」的内存字典。
# 记录面板最多列 30 局，够用；再早的局点开会明确提示「回放已不在缓存」。
_REPLAY_CACHE_MAX = 300
_REPLAY_TAIL_BYTES = 8 * 1024 * 1024
_REPLAY_CACHE: "OrderedDict[str, dict]" = OrderedDict()
_REPLAY_READY = False

#: 回放手牌的紧凑编码（2026-10-12）：27 个牌种各映射成 1 个字符。
#: 回放要「像实盘一样」把四家手牌都画出来，逐事件带 4 个整数数组会让
#: replays.jsonl 涨 3 倍以上；编码成字符串后一手 13 张 = 13 字符，四家 ≈ 55 字节。
_TILE_CHARS = "0123456789abcdefghijklmnopqrstuvwxyz"


def pack_hand(counts) -> str:
    """``hands[p]``（27 个计数）→ 升序牌种字符串（每张 1 字符）。"""
    out = []
    for t, c in enumerate(counts):
        c = int(c)
        if c > 0:
            out.append(_TILE_CHARS[t] * c)
    return "".join(out)


def remember_replay(rp: dict) -> None:
    """新落盘的回放直接进缓存（顺序写 → 顺序取，命中率最高）。"""
    rid = rp.get("rid")
    if not rid:
        return
    _REPLAY_CACHE[rid] = rp
    while len(_REPLAY_CACHE) > _REPLAY_CACHE_MAX:
        _REPLAY_CACHE.popitem(last=False)


def _load_replays() -> None:
    global _REPLAY_READY
    if _REPLAY_READY:
        return
    _REPLAY_READY = True
    p = HISTORY_DIR / "replays.jsonl"
    try:
        if not p.exists():
            return
        with open(p, "rb") as fh:
            size = p.stat().st_size
            if size > _REPLAY_TAIL_BYTES:
                fh.seek(size - _REPLAY_TAIL_BYTES)
                fh.readline()          # 丢掉可能被截断的首行
            data = fh.read().decode("utf-8", "ignore")
        for line in data.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                remember_replay(json.loads(line))
            except Exception:
                continue
    except Exception:
        traceback.print_exc()


def load_replay(rid: str) -> dict | None:
    if not rid:
        return None
    if rid not in _REPLAY_CACHE:
        _load_replays()
    return _REPLAY_CACHE.get(rid)


#: 「具名大牌型」显示名的**排除表**（2026-10-12 用户口径）
#: —— 「报叫」只是平胡局的显示名（引擎 ``_fan_cn_of``），不是具名大牌型，
#:    所以 非平胡·听牌/胡牌 的统计口径要把它和「平胡」一样排除掉。
BIGFAN_EXCLUDE = ("", "平胡", "报叫")


def is_win_bigfan(winners_fan) -> bool:
    """该局赢家里有没有人拿到**具名大牌型**（清一色 / 七对 / 大对子 …）。

    只影响统计口径（``flags.win_bigfan`` / ``/api/stats`` 的非平胡·胡牌局占比），
    不改变任何计分。"""
    return any(str((x or {}).get("fan_cn") or "") not in BIGFAN_EXCLUDE
               for x in (winners_fan or []))


def is_bigfan_name(fan_cn) -> bool:
    """旧记录（没有 flags）的兜底：按字符串判是否具名大牌型。"""
    return str(fan_cn or "") not in BIGFAN_EXCLUDE


# ---- 删除对局记录（2026-10-12 用户需求）----------------------------------
# 记录面板每行都有删除按钮；删掉一条记录必须同时：
#   ① 从 records.jsonl 里删掉这一行（/api/stats 就是逐行扫这个文件，
#      所以统计会自动跟着记录走 —— 用户口径「数据统计仅统计对局记录里的对局数据」）；
#   ② 从 replays.jsonl 里删掉同 rid 的回放，并从内存缓存里剔除（否则同一局还能点开回放）。
# 写临时文件再 replace，中途崩溃不会把记录文件写坏。
def _rewrite_jsonl(path: Path, drop_lines: set[int] | None = None,
                   drop_rids: set[str] | None = None) -> int:
    """重写 jsonl 删行。``drop_lines`` = 0 基行号集合；``drop_rids`` = 按 rid 删。
    返回删掉的条数（文件不存在返回 0）。写临时文件再 replace，中途崩不会写坏。"""
    if not path.exists():
        return 0
    drop_lines = drop_lines or set()
    drop_rids = drop_rids or set()
    tmp = path.with_suffix(path.suffix + ".tmp")
    removed = 0
    with open(path, encoding="utf-8") as src, open(tmp, "w", encoding="utf-8") as dst:
        for i, line in enumerate(src):
            hit = i in drop_lines
            if not hit and drop_rids and line.strip():
                try:
                    if str(json.loads(line).get("rid") or "") in drop_rids:
                        hit = True
                except Exception:
                    pass
            if hit:
                removed += 1
                continue
            dst.write(line if line.endswith("\n") else line + "\n")
    tmp.replace(path)
    return removed


def delete_history_records(rec_file: Path, replay_file: Path, items: list) -> dict:
    """删掉一批对局记录（及其回放）。``items`` = [(ln|None, rid|None), ...]。

    一次重写完成（比逐条重写快，也避免中途状态不一致）。``rid`` 兼作**安全校验**：
    行号对应的记录 rid 与调用方给的不一致时跳过该条（并发新局只在尾部追加，
    行号不会漂移；这里只是双保险）。返回 ``{ok, removed_rec, removed_replay,
    error, skipped}``。"""
    out = {"ok": False, "removed_rec": 0, "removed_replay": 0,
           "error": "", "skipped": []}
    if not rec_file.exists():
        out["error"] = "还没有对局记录"
        return out
    lines = rec_file.read_text(encoding="utf-8").splitlines()
    by_rid: dict[str, int] = {}
    for i, s in enumerate(lines):
        try:
            r = str(json.loads(s).get("rid") or "")
        except Exception:
            continue
        if r and r not in by_rid:
            by_rid[r] = i

    drop_lines: set[int] = set()
    drop_rids: set[str] = set()
    for ln, rid in items:
        rid = (str(rid).strip() or None) if rid else None
        i: int | None = None
        bad_int = False
        if ln is not None and str(ln) != "":
            try:
                i = int(ln)
            except (TypeError, ValueError):
                i = None
                bad_int = True
        if i is not None and not (0 <= i < len(lines)):
            out["skipped"].append({"ln": ln, "rid": rid, "why": f"行号越界：{ln}"})
            continue
        if i is None and not bad_int:
            if not rid:
                out["skipped"].append({"ln": ln, "rid": rid, "why": "缺少定位参数"})
                continue
            i = by_rid.get(rid)
            if i is None:
                out["skipped"].append(
                    {"ln": ln, "rid": rid, "why": f"记录已不在文件里：{rid}"})
                continue
        elif bad_int:
            out["skipped"].append({"ln": ln, "rid": rid, "why": f"行号越界：{ln}"})
            continue
        try:
            rec = json.loads(lines[i])
        except Exception:
            rec = {}
        rec_rid = str(rec.get("rid") or "")
        if rid and rec_rid not in ("", rid):
            out["skipped"].append(
                {"ln": ln, "rid": rid, "why": f"行号与记录不匹配（第 {i} 行是 {rec_rid}）"})
            continue
        drop_lines.add(i)
        if rec_rid:
            drop_rids.add(rec_rid)
        elif rid:
            drop_rids.add(rid)

    out["removed_rec"] = _rewrite_jsonl(rec_file, drop_lines=drop_lines)
    if drop_rids:
        out["removed_replay"] = _rewrite_jsonl(replay_file, drop_rids=drop_rids)
        for r in drop_rids:
            _REPLAY_CACHE.pop(r, None)
    out["ok"] = out["removed_rec"] > 0
    if not out["ok"]:
        # 一条都没删成：把**具体原因**冒到 error，前端/命令行直接可读
        # （不然只回「记录已被删除」，用户以为是并发撞车，其实是行号对不上）。
        whys: list[str] = []
        for s in out["skipped"]:
            w = s.get("why") or "定位不到"
            if w not in whys:
                whys.append(w)
        out["error"] = "；".join(whys) if whys else (out["error"] or "记录已被删除")
    return out


def delete_history_record(rec_file: Path, replay_file: Path,
                          ln: int | None, rid: str | None) -> dict:
    """删掉**一条**记录（旧调用点兼容壳）。

    与批量接口 ``delete_history_records`` 的区别：这里保留**逐项前置校验**和
    旧版的具体报错文案（「缺少定位参数 / 行号越界 / 记录已不在文件里 /
    还没有对局记录 / 行号与记录不匹配」），任何校验失败都**不动文件**。
    批量接口走的是「能删的删、删不了的进 skipped」的宽容口径，两者互补。
    """
    miss = {"ok": False, "removed_rec": 0, "removed_replay": 0, "skipped": []}
    rid = (str(rid).strip() or None) if rid else None
    if (ln is None or str(ln) == "") and not rid:
        return {**miss, "error": "缺少定位参数"}
    if not rec_file.exists():
        return {**miss, "error": "还没有对局记录"}
    lines = rec_file.read_text(encoding="utf-8").splitlines()
    i: int | None = None
    if ln is not None and str(ln) != "":
        try:
            i = int(ln)
        except (TypeError, ValueError):
            i = None
        if i is None or not (0 <= i < len(lines)):
            return {**miss, "error": f"行号越界：{ln}（共 {len(lines)} 行）"}
    else:
        for j, s in enumerate(lines):
            try:
                if str(json.loads(s).get("rid") or "") == rid:
                    i = j
                    break
            except Exception:
                continue
        if i is None:
            return {**miss, "error": f"记录已不在文件里（rid={rid}）"}
    try:
        rec_rid = str(json.loads(lines[i]).get("rid") or "")
    except Exception:
        rec_rid = ""
    if rid and rec_rid not in ("", rid):
        return {**miss, "error": f"行号与记录不匹配：第 {i} 行是 {rec_rid!r}，期望 {rid!r}"}
    return delete_history_records(rec_file, replay_file, [(i, rid)])


# ---- 导出对局记录（2026-10-12 用户需求）----------------------------------
# 一键把 records.jsonl 导出成可下载的 JSON / CSV。字段按「结算页能看到的东西」
# 挑一遍（时间/局号/座位/四家/胡牌方式/牌型/番/输赢分/听牌/鸡牌/标记），
# 老记录缺字段一律留空，不做任何补算 —— 导出既不改文件也不锁文件。
EXPORT_FIELDS = [
    ("time", "时间"), ("rid", "回放ID"), ("hand", "局号"), ("type", "结果"),
    ("my_seat", "我的座位"), ("dealer", "庄家"),
    ("names", "四家"), ("winner_names", "赢家"), ("loser_name", "点炮者"),
    ("how", "胡牌方式"), ("fan_cn", "牌型"), ("fan", "番数"),
    ("is_tsumo", "自摸"), ("rob_kong", "抢杠"),
    ("tenpai_names", "听牌者"), ("deltas", "四家分"),
    ("ji_tiles", "本局鸡牌"), ("fanji", "翻鸡"),
    ("void_name", "烧鸡者"), ("detail", "结算明细"),
]


def _export_rows(rec_file: Path) -> list[dict]:
    rows: list[dict] = []
    if not rec_file.exists():
        return rows
    with open(rec_file, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception:
                continue
    return rows


def export_history(rec_file: Path, fmt: str) -> tuple[bytes, str, str]:
    """返回 ``(body, content_type, filename)``。``fmt`` = json | csv。"""
    rows = _export_rows(rec_file)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    if fmt == "csv":
        import csv as _csv
        buf = io.StringIO()
        w = _csv.writer(buf)
        w.writerow([cn for _k, cn in EXPORT_FIELDS])
        for r in rows:
            line = []
            for k, _cn in EXPORT_FIELDS:
                v = r.get(k)
                if isinstance(v, (list, dict)):
                    v = json.dumps(v, ensure_ascii=False)
                elif v is None:
                    v = ""
                line.append(v)
            w.writerow(line)
        # Excel 认 UTF-8 BOM，不加会中文乱码
        body = ("\ufeff" + buf.getvalue()).encode("utf-8")
        return body, "text/csv; charset=utf-8", f"dushan_records_{stamp}.csv"
    body = json.dumps({"exported_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                       "count": len(rows), "records": rows},
                      ensure_ascii=False, indent=1).encode("utf-8")
    return body, "application/json; charset=utf-8", f"dushan_records_{stamp}.json"


MELD_LABEL = {
    MELD_PONG: "碰",
    MELD_KONG_EXPOSED: "明杠",
    MELD_KONG_ADDED: "补杠",
    MELD_KONG_CONCEALED: "暗杠",
}

BUTTON_ORDER = {HU: 0, ANGANG: 1, BUGANG: 1, MINGGANG: 1, PENG: 2, PASS: 3}

HUMAN_SEAT_NAME = "你"
MAX_STEPS = 3000
VERBOSE = False
#: 客户端「开局特效暂停」最长时长（秒）：客户端播完会立即放行，
#: 这个上限只用于兜底（页面被关掉 / 断连 / 崩溃时不让会话永久停摆）。
HOLD_MAX = 20.0

#: "大师" 别名 → 实际模型 stem（按优先级探测）
MASTER_CANDIDATES = ["dushan_master", "dushan_s1", "dushan_s2", "dushan_s3", "rl"]


#: 未听牌包赔开关的网页默认（与引擎 DushanConfig 默认一致）
RULE_DEFAULTS: dict[str, bool] = {
    "end_baoji": True,        # 有人胡牌结束：未听者包鸡
    "end_baogang": False,     # 有人胡牌结束：未听者包杠
    "huang_baoji": False,     # 流局：未听者包鸡
    "huang_baogang": False,   # 流局：未听者包杠
    "huang_baodapai": True,   # 流局：未听者包大牌面（非平胡）
}

#: 翻鸡策略 → 流水文案（2026-10-12 #85：开局/结束翻鸡写进流水）
FANJI_MODE_CN = {"up": "上鸡", "down": "下鸡", "both": "上下鸡", "off": "关闭"}


def config_from_rules(rules: dict | None) -> DushanConfig:
    """前端 rules 字典 → 引擎配置（规则开关 + 分值表 + 未听包赔开关）。"""
    r = rules if isinstance(rules, dict) else {}
    sc = r.get("scores") if isinstance(r.get("scores"), dict) else {}

    def _f(key: str, default: float) -> float:
        try:
            v = float(sc.get(key, default))
            return v if v >= 0 else default
        except (TypeError, ValueError):
            return default

    flags = {k: bool(r.get(k, v)) for k, v in RULE_DEFAULTS.items()}
    # 2026-10-12：追加鸡牌种（如贵阳玩的八筒）——前端传牌种索引数组。
    # 清洗口径：丢弃越界 / 非整数 / 重复项；**幺鸡恒为鸡**，出现在这里直接忽略
    # （引擎 DushanGame._init_dushan_state 也会兜一次，这里保证 rules 载荷本身干净）。
    extra_ji: list[int] = []
    if isinstance(r.get("extra_ji"), (list, tuple)):
        for x in r["extra_ji"]:
            try:
                t = int(x)
            except (TypeError, ValueError):
                continue
            if 0 <= t < NUM_TILE_TYPES and t != YAOJI and t not in extra_ji:
                extra_ji.append(t)
    extra_ji.sort()          # 升序：与前端 SET.extraJi 一致，载荷稳定可预测
    return DushanConfig(
        # 结算翻鸡自 2026-10-11 起支持 off（网页设置新增「关闭」，默认仍 both）
        jiesuan_fanji=norm_fanji(r.get("jiesuan", r.get("jiesuan_fanji", "both")),
                                 allow_off=True),
        kaiju_fanji=norm_fanji(r.get("kaiju", r.get("kaiju_fanji", "off")),
                               allow_off=True),
        mantiangji=bool(r.get("mantiangji", False)),
        # 可调分值表（网页设置）
        zi_mo=_f("zimo", 3.0), dian_pao=_f("dianpao", 3.0),
        gang_kai=_f("gangkai", 5.0), gang_pao=_f("gangpao", 5.0),
        da_dui_zi=_f("dadui", 5.0), qing_yi_se=_f("qing", 10.0),
        qi_dui=_f("qidui", 10.0), long_qi_dui=_f("long", 20.0),
        dan_diao=_f("dudiao", 10.0), baojiao_bonus=_f("baojiao", 10.0),
        # 鸡的分值表（教程 10.1，全部可在设置里调）
        kong_chickens=_f("kong", 3.0),
        jin_ji_multiplier=_f("jinji", 2.0),
        ji_hold=_f("jihold", 1.0),
        ji_discard_plain=_f("jiplain", 1.0),
        ji_chongfeng=_f("jicf", 3.0),
        ji_heng=_f("jiheng", 2.0),
        ji_fanji=_f("jifanji", 1.0),
        # 追加鸡牌种（如贵阳八筒）；幺鸡自动忽略
        extra_ji=tuple(extra_ji),
        **flags,
    )


def resolve_model_stem(name: str) -> str | None:
    """把前端传来的模型名解析成实际存在的 stem。"""
    if name in ("master", "best"):
        for cand in MASTER_CANDIDATES:
            if (MODEL_DIR / f"{cand}.pt").exists():
                return cand
        return None
    if (MODEL_DIR / f"{name}.pt").exists():
        return name
    return None


# ---------------------------------------------------------------------------
# 模型元信息（轻量读取，用于前端展示的版本号/规模）
# ---------------------------------------------------------------------------
_PEEK_CACHE: dict[str, tuple[float, dict]] = {}
_PEEK_LOCK = threading.Lock()


def peek_checkpoint(path: Path) -> dict:
    """只读出检查点里的 ``config`` / ``net`` / 参数量，不解码权重张量。

    ``torch.save`` 产出的 zip 内 ``*/data.pkl`` 是纯 pickle；把张量的重建指令
    替换成占位对象即可毫秒级读出元信息（实测 ~3ms/文件，完整 ``torch.load`` 需
    数十毫秒且要构造网络）。失败一律返回 ``{}``，调用方需容错。
    """
    try:
        mtime = path.stat().st_mtime
    except OSError:
        return {}
    with _PEEK_LOCK:
        hit = _PEEK_CACHE.get(path.name)
        if hit and hit[0] == mtime:
            return hit[1]

    info: dict = {}
    try:
        sizes: list[int] = []

        class _Stub:                       # torch 张量的占位（不读数据）
            def __init__(self, *a, **_k):
                if len(a) >= 3 and isinstance(a[2], (tuple, list)) and a[2]:
                    try:
                        n = 1
                        for x in a[2]:
                            n *= int(x)
                        sizes.append(n)
                    except (TypeError, ValueError):
                        pass

            def __setstate__(self, _s):
                pass

        class _Unpickler(pickle.Unpickler):
            #: 只允许解析元信息所需的全局符号，其余一律拒绝
            #: （本函数只用来读 config/net，不该具备执行代码的能力）
            _ALLOWED = ("builtins", "collections", "collections.abc",
                        "numpy", "numpy.core.multiarray")

            def find_class(self, mod, name):      # noqa: A003
                if mod.startswith("torch"):
                    return _Stub
                if mod in self._ALLOWED:
                    return super().find_class(mod, name)
                raise pickle.UnpicklingError(f"unexpected global {mod}.{name}")

            def persistent_load(self, _pid):
                return None

        with zipfile.ZipFile(path) as z:
            pkl = next((n for n in z.namelist() if n.endswith("data.pkl")), None)
            obj = _Unpickler(io.BytesIO(z.read(pkl))).load() if pkl else None
        if isinstance(obj, dict):
            info = {"config": obj.get("config") or {},
                    "net": obj.get("net") or {},
                    "iter": obj.get("iter"),
                    "params": sum(sizes)}
    except Exception:
        info = {}
    with _PEEK_LOCK:
        _PEEK_CACHE[path.name] = (mtime, info)
    return info


def _origin_stem(stem: str, cfg: dict) -> str:
    """模型「出身代号」：优先取权重内记录的训练来源（config.out / config.init），
    这样 ``dushan_master.pt`` 这类冠军副本能报出它真正的代次（如 big_s7）。"""
    for key in ("out", "init"):
        raw = str(cfg.get(key) or "").strip().replace("\\", "/")
        if raw:
            s = Path(raw).stem
            if s:
                return s
    return stem


def model_identity(stem: str) -> dict:
    """把模型 stem 变成前端展示用的「网络线别 + 版本号 + 规模」。

    ``dushan_master``(=big_s7 副本) → ``{"line": "大网", "ver": "s7",
    "abbr": "B7", "size": "769K", "label": "大师 s7（769K）"}``
    """
    info = peek_checkpoint(MODEL_DIR / f"{stem}.pt")
    origin = _origin_stem(stem, info.get("config") or {}).lower()
    if origin.startswith("dushan_"):
        origin = origin[len("dushan_"):]
    big = origin.startswith("big_")
    if big:
        origin = origin[len("big_"):]
    line = "大网" if big else "小网"
    prefix = "B" if big else "S"
    if re.fullmatch(r"s\d+", origin):
        ver, abbr = origin.lower(), prefix + origin[1:]
    elif origin in ("bc", "rl"):
        ver, abbr = origin, origin.upper()
    else:
        ver, abbr = origin, (prefix + origin.upper() if origin else "?")
    params = int(info.get("params") or 0)
    size = f"{params / 1000:.0f}K" if params else ""
    return {"stem": stem, "origin": _origin_stem(stem, info.get("config") or {}),
            "line": line, "ver": ver, "abbr": abbr, "size": size}


_MASTER_ID_CACHE: dict = {"key": None, "val": None}


def master_identity() -> dict | None:
    """当前冠军模型的展示身份（按文件 mtime 缓存，权重换代后自动更新）。"""
    stem = resolve_model_stem("master")
    if not stem:
        return None
    path = MODEL_DIR / f"{stem}.pt"
    key = (stem, path.stat().st_mtime if path.exists() else 0)
    if _MASTER_ID_CACHE["key"] == key:
        return _MASTER_ID_CACHE["val"]
    try:
        info = model_identity(stem)
    except Exception:
        info = {"stem": stem, "origin": stem, "line": "", "ver": "", "abbr": "M", "size": ""}
    # 前端选项文案：与兄弟选项同构（名字 + 版本号（规模））；排在第一即当前最强
    name = "大师" + (f" {info['ver']}" if info.get("ver") else "")
    info["label"] = f"{name}（{info['size']}）" if info.get("size") else name
    _MASTER_ID_CACHE.update(key=key, val=info)
    return info


# ---------------------------------------------------------------------------
# 模型仓库
# ---------------------------------------------------------------------------
_MODELS: dict[str, dict] = {}
_MODELS_LOCK = threading.Lock()


def get_model(stem: str) -> dict:
    path = MODEL_DIR / f"{stem}.pt"
    if not path.exists():
        raise FileNotFoundError(f"找不到模型 {path}")
    mtime = path.stat().st_mtime
    with _MODELS_LOCK:
        if stem in _MODELS and _MODELS[stem]["mtime"] == mtime:
            return _MODELS[stem]
    ck = torch.load(path, map_location="cpu", weights_only=False)
    # 用检查点自带的元数据还原结构（v1/v2、有无空间头、编码器版本都由它声明）
    net, enc_v = build_from_ckpt(ck)
    cfg = net_config_from_ckpt(ck)
    params = sum(int(v.numel()) for v in ck["state_dict"].values() if hasattr(v, "numel"))
    entry = {
        "net": net,
        "mtime": mtime,
        "config": ck.get("config") or {},
        "params": params,
        "meta": {"epoch": ck.get("epoch"), "val_loss": ck.get("val_loss"),
                 "iter": ck.get("iter"), "eval": ck.get("eval"),
                 "net": net_meta(cfg),
                 "encoder_version": enc_v,
                 "params": params},
        "path": str(path),
    }
    with _MODELS_LOCK:
        _MODELS[stem] = entry
    print(f"[model] {path.name} → iter={entry['meta'].get('iter')} "
          f"eval={entry['meta'].get('eval')}", flush=True)
    return entry


def available_models() -> list[dict]:
    """模型清单（含展示身份 ident）。只读检查点元信息，不解码权重、不构造网络。"""
    out = []
    for p in sorted(MODEL_DIR.glob("*.pt")):
        peek = peek_checkpoint(p)
        info = {"stem": p.stem, "size_kb": round(p.stat().st_size / 1024, 1),
                "mtime": time.strftime("%m-%d %H:%M", time.localtime(p.stat().st_mtime)),
                "iter": peek.get("iter"), "net": peek.get("net"),
                "params": peek.get("params")}
        try:
            info["ident"] = model_identity(p.stem)
        except Exception:
            info["bad"] = True
        out.append(info)
    return out


# ---------------------------------------------------------------------------
# 弃牌鸡类标记：冲锋鸡 / 横鸡（前端用专用牌面 冲.png / 横.png）
# ---------------------------------------------------------------------------
def _discard_tags(game: DushanGame, p: int) -> list[str | None]:
    """把 game.ji_events 中该玩家的鸡事件按顺序对回其弃牌，返回逐张标记。

    2026-10-12：改用**展示视图** ``discard_view``（只增不减，含被碰/杠走的牌），
    与前端牌河（同样把被吃走的牌灰显留在原位）严格对齐。
    """
    view = getattr(game, "discard_view", None)
    seq = [int(t) for t in (view[p] if view and view[p] else game.discards[p])]
    tags: list[str | None] = [None] * len(seq)
    ji = [e for e in game.ji_events if int(e[0]) == p]
    k = 0
    for i, t in enumerate(seq):
        if t in game.ji_tiles and k < len(ji):
            tags[i] = str(ji[k][2])
            k += 1
    return tags


def _meld_ji_tag(game: DushanGame, p: int, tile: int) -> str | None:
    """该副露中被碰/杠走的牌若是鸡牌，返回类型名（冲锋鸡/横鸡/幺鸡）。

    前端据此把「被碰/杠的鸡牌」横置一张指示来源家：冲锋鸡/横鸡用专用牌面
    （冲.png/横.png），普通幺鸡与翻鸡种用原牌面。
    """
    if tile not in game.ji_tiles:
        return None
    # _ji_val_by_meld 存的是类型名（冲锋鸡/横鸡/幺鸡），分值改由配置换算
    return str(game._ji_val_by_meld.get((p, tile)) or "幺鸡")


# ---------------------------------------------------------------------------
# 对手工厂
# ---------------------------------------------------------------------------
def make_opponent(spec: dict, seed: int) -> tuple[Bot, str]:
    kind = spec.get("type", "random")
    if kind == "model":
        stem = resolve_model_stem(str(spec.get("model") or "master"))
        if stem is None:
            stem = "rl" if (MODEL_DIR / "rl.pt").exists() else None
        if stem is None:
            name = "随机(无模型)"
            return RandomBot(seed=seed), name
        entry = get_model(stem)
        temp = float(spec.get("temperature", 0.0))
        label = {"dushan_master": "大师", "dushan_s1": "大师S1", "dushan_s2": "大师S2",
                 "dushan_s3": "大师S3"}.get(stem, f"模型·{stem}")
        name = label if temp <= 1e-6 else f"{label}(T{temp:g})"
        return NetAgent(entry["net"], seed=seed, temperature=temp, name=name), name
    if kind == "teacher":
        style = spec.get("style") or "balanced"
        if style not in TEACHER_STYLES:
            style = "balanced"
        cn = {"balanced": "均衡", "aggressive": "激进", "defensive": "保守",
              "chicken_lover": "爱鸡", "tenpai_rush": "抢听", "gambler": "赌徒"}
        name = f"教师·{cn.get(style, style)}"
        return HeuristicBot(seed=seed, weights=TEACHER_STYLES[style]), name
    return RandomBot(seed=seed), "随机"


# ---------------------------------------------------------------------------
# 人类玩家：阻塞式 Bot
# ---------------------------------------------------------------------------
class HumanBot(Bot):
    name = HUMAN_SEAT_NAME

    def __init__(self, session: "Session"):
        self.session = session

    def choose(self, game) -> Action:
        return self.session.wait_human(game)


# ---------------------------------------------------------------------------
# 理由生成
# ---------------------------------------------------------------------------
def visible_count(game: DushanGame, tile: int) -> int:
    """某张牌在**公开信息**里已出现的张数（所有弃牌 + 我的副露）。"""
    n = 0
    for p in range(4):
        n += game.discards[p].count(tile)
    for mt, t, _s in game.melds[game.actor()] if False else []:
        pass
    me = game.actor()
    for mt, t, _s in game.melds[me]:
        if t == tile:
            n += 4 if mt in (MELD_KONG_EXPOSED, MELD_KONG_ADDED, MELD_KONG_CONCEALED) else 3
    return n


def _is_kong_meld(mt: int) -> bool:
    return mt in (MELD_KONG_EXPOSED, MELD_KONG_ADDED, MELD_KONG_CONCEALED)


def remaining_unseen(game: DushanGame, me: int, drop: int | None, tile: int) -> int:
    """某张牌「还剩几张」= 4 − 已见张数。

    已见 = 四家弃牌 + 四家副露（碰 3 / 杠 4）+ 我的暗手（``drop`` 若已打掉则少算一张）。
    对手暗手不可见，一律不计——这即通行做法的「剩余枚数」。
    """
    seen = 0
    for p in range(4):
        seen += int(game.discards[p].count(tile))
        for mt, t, _s in game.melds[p]:
            if int(t) == tile:
                seen += 4 if _is_kong_meld(mt) else 3
    seen += int(game.hands[me][tile]) - (1 if drop == tile else 0)
    return max(0, 4 - seen)


def explain_action(game: DushanGame, me: int, action: Action) -> str:
    """给推荐动作配一句中文理由。"""
    k = action.kind
    if k == HU:
        return "立即和牌得分" + ("（热炮/报叫等无需通行证）" if game.repao_discard or
                                 (game.last_discard and game.baojiao[game.last_discard[0]])
                                 else "")
    if k in (ANGANG, BUGANG, MINGGANG):
        return f"杠得杠分并获得点胡通行证（{MELD_LABEL.get({'angang': 4, 'bugang': 3, 'minggang': 2}.get(k, k), '杠')}）"
    if k == PENG:
        return "碰出后加快组牌，但注意保持听牌"
    if k == PASS:
        return "放弃此次响应，避免扩大副露暴露信息"
    if k != DISCARD:
        return ""
    t = action.tile
    h = list(game.hands[me])
    h[t] -= 1
    wins = winning_tiles(h, game.melds[me])
    msgs: list[str] = []
    if t in game.ji_tiles:
        msgs.append("打鸡收鸡分（若被碰要赔责任鸡）")
    if wins:
        names = "、".join(tile_cn(x) for x in wins[:3])
        more = f" 等{len(wins)}类" if len(wins) > 3 else ""
        msgs.append(f"打完仍听牌，叫 {names}{more}")
    else:
        msgs.append("打完暂不听牌，继续进张")
    vis = visible_count(game, t)
    if vis >= 3:
        msgs.append(f"已现 {vis}/4 张，几乎安全")
    elif vis >= 1:
        msgs.append(f"已现 {vis}/4 张，点炮风险中等")
    else:
        msgs.append("生张，点炮风险偏高")
    return "；".join(msgs)


# ---------------------------------------------------------------------------
# 会话
# ---------------------------------------------------------------------------
class Session:
    def __init__(self, sid: str, specs: list[dict], my_seat: int | None,
                 seed: int, speed: float, auto: bool = False,
                 rules: dict | None = None, names: list[str] | None = None,
                 hold: bool = False):
        self.sid = sid
        self.specs = specs
        self.seed = seed
        self.speed = speed
        self.auto = auto
        # ---- 客户端全屏特效（开局鸡牌）期间的「暂停推进」闸门（2026-10-13 用户口径：
        #      开局先播鸡牌特效，特效结束再开始牌局）----
        # 客户端检测到换局 → 播特效的同时 POST /api/hold {on:true}；播完 POST {on:false}。
        # 三重防呆，保证会话永远不会被卡死：
        #   ① hold_hand 只对本手生效（迟到的 on:true 不会误扣下一手）；
        #   ② hold_until 到点自动放行（客户端关页 / 断连 / 崩溃）；
        #   ③ auto 会话（服务端自检）根本不走这个闸门。
        self.hold = bool(hold)
        self.hold_hand = 1 if hold else 0      # 新会话：第一手（_play_hand 会自增到 1）
        self.hold_until = time.monotonic() + HOLD_MAX
        self.rules = rules or {}    # 规则开关：jiesuan(结算翻鸡) / kaiju(开局翻鸡) / mantiangji / 包赔开关
        # 自定义名字（按相对座位 0=我 1=下家 2=对家 3=上家；空串=跟随对手类型自动命名）
        self.custom_names = [str(x or "").strip()[:12] for x in (names or [])][:4]
        self.rng = random.Random(seed)

        self.lock = threading.RLock()
        self.act_event = threading.Event()
        self.next_event = threading.Event()
        self.chosen: Action | None = None
        self.legal_now: list[Action] = []
        self.game: DushanGame | None = None

        self.names = [f"座位{i}" for i in range(4)]
        self.kinds = ["random"] * 4
        self.cum = [0.0] * 4
        self.hand_index = 0
        # 本手是否调用过 /api/hint（模型大师推荐）。每手开头重置，终局写进记录：
        # 「本人胡牌 / 本人累计」两个统计量的分母 = 全程没看过推荐的局（真·自己打）。
        self.hint_used = False
        self.dealer = 0
        self.dealer_streak = 0
        self.my_seat = my_seat if my_seat is not None else self.rng.randrange(4)
        self.stop = False
        self.error: str | None = None
        # 「结束本局」：置位后当前这一手在下一个 step 边界提前终止（不计分不入历史）
        self.force_end = False

        # 统计：本会话（座位粒度）
        self.stats: dict[int, dict] = {
            p: {"games": 0, "wins": 0, "tsumo": 0, "ron": 0, "deal_in": 0,
                "huang": 0, "tenpai_huang": 0, "baoji_pay": 0.0,
                "chongfeng": 0, "hengji": 0, "baoji": 0,
                "zeren_pay": 0.0, "score": 0.0}
            for p in range(4)
        }

        self.events: deque = deque(maxlen=600)
        # ---- 回放（2026-10-12）：本手事件的完整副本 + 当前手牌张数快照 ----
        # self.events 是全局滑动窗口（maxlen=600），跨手会被冲掉；回放要的是
        # 「这一手从头到尾」的事件，所以单独留一份，每手开头清空。
        # _cur_game 只用于在事件里记 hc（各家手牌张数），手结束时置 None。
        self._hand_ev: list[dict] = []
        self._cur_game: DushanGame | None = None
        # 本手发牌（第 0 帧手牌；回放要能画出「还没出牌」那一帧）
        self._deal_packed: list[str] = []
        self._ev_draws = 0          # 本手累计摸牌数（回放里的「巡」近似，仅展示用）
        self._ev_discards = 0       # 本手累计打出数
        self.state: dict = {"ready": False}
        self.on_hand_end = None
        # 流水补齐游标（见 _emit_progress）：报叫只在引擎 game.log 里，
        # 需自己增量搬到事件流；听牌靠 ever_tenpai 标志跳变。每手一局 new game，故每手归零。
        self._log_seen = 0
        self._tenpai_seen = [False] * 4
        # 横鸡轮事件游标（引擎 game.hengji_events 增量搬运，同 game.log 机制）
        self._hj_seen = 0
        # 流水行编号：每手归零（前端显示「局号-序号」可区分不同局）
        self._ev_seq = 0

        self.thread = threading.Thread(target=self._run, daemon=True,
                                       name=f"game-{sid[:6]}")
        self.thread.start()

    # -- 座位 ---------------------------------------------------------------
    def _setup_seats(self, game: DushanGame) -> dict[int, Bot]:
        bots: dict[int, Bot] = {}
        for p in range(4):
            rel = (p - self.my_seat) % 4
            custom = self.custom_names[rel] if rel < len(self.custom_names) else ""
            if p == self.my_seat:
                bots[p] = HumanBot(self)
                self.names[p] = custom or HUMAN_SEAT_NAME
                self.kinds[p] = "human"
            else:
                spec = self.specs[(p - self.my_seat) % 4 - 1] if len(self.specs) >= 3 else {}
                seed = self.seed * 7919 + self.hand_index * 131 + p
                bot, name = make_opponent(spec if isinstance(spec, dict) else {}, seed)
                bots[p] = bot
                self.names[p] = custom or name
                self.kinds[p] = spec.get("type", "random") if isinstance(spec, dict) else "random"
        return bots

    # -- 主循环 -------------------------------------------------------------
    def _run(self) -> None:
        try:
            while not self.stop:
                self._play_hand()
                if self.stop:
                    break
                if self.auto:
                    continue
                self.next_event.clear()
                self.next_event.wait(timeout=3600)
        except Exception:
            self.error = traceback.format_exc()
            with self.lock:
                self.state = {"ready": True, "error": self.error}

    def _play_hand(self) -> None:
        cfg = config_from_rules(self.rules)
        game = DushanGame(cfg, seed=self.seed + self.hand_index * 7919,
                          dealer=self.dealer, dealer_streak=self.dealer_streak)
        self.hand_index += 1
        # 新的一手 = 新的 DushanGame（game.log / baojiao / ever_tenpai 全部重置），
        # 游标必须同步归零，否则会漏发或误读上一手的位置
        self._log_seen = 0
        self._tenpai_seen = [False] * 4
        self._hj_seen = 0
        self._ev_seq = 0
        self.force_end = False
        self.hint_used = False      # 新的一手：推荐使用标记归零
        # 回放：本手事件副本与巡刻度归零（_cur_game 供 _push 记各家手牌张数）
        self._hand_ev = []
        # 发牌快照：庄家 14 张、其余 13 张（第 0 帧要画出来）
        try:
            self._deal_packed = [pack_hand(game.hands[p]) for p in range(4)]
        except Exception:
            self._deal_packed = []
        # 2026-10-12 #85：开局翻鸡写进流水——翻到的指示牌 / 策略 / 计算后新增的鸡牌
        if game.kaiju_flip is not None:
            mode = str(getattr(cfg, "kaiju_fanji", "off"))
            flips = sorted(int(t) for t in fanji_tiles_from(game.kaiju_flip, mode))
            self._push("fanji", -1,
                       f"开局翻鸡（{FANJI_MODE_CN.get(mode, mode)}）：翻出 {tile_cn(game.kaiju_flip)}"
                       + ("，新增鸡牌 " + "、".join(tile_cn(t) for t in flips) if flips
                          else "，无新增鸡牌"))
        self._ev_draws = 0
        self._ev_discards = 0
        with self.lock:
            self.game = game
            self._cur_game = game
        bots = self._setup_seats(game)
        self._publish(game)
        steps = 0
        while game.phase != Phase.OVER and not self.stop and not self.force_end:
            # 客户端开局鸡牌特效「扣牌」：特效播放期间不推进牌局（发牌快照已经 publish
            # 出去了，客户端据此显示本局鸡牌；特效播完客户端放行 → 牌局才真正开始）。
            if self.hold:
                while (self.hold and not self.stop and self.hold_hand == self.hand_index
                       and time.monotonic() < self.hold_until):
                    time.sleep(0.03)
                self.hold = False
            actor = game.actor()
            action = bots[actor].choose(game)
            before = [sum(game.hands[p]) for p in range(4)]
            # 横鸡轮快照（step 前后对比 → 检测「开轮 / 收口」）：
            # _close_hengji_rounds 删键、开轮加键，键集合对比即可；seats 原地修改不影响
            rounds_before = dict(getattr(game, "_hengji_rounds", {}) or {})
            with self.lock:
                game.step(action)
                steps += 1
                if game.phase != Phase.OVER and steps > MAX_STEPS:
                    game.result = {"type": "timeout", "deltas": [0.0] * 4,
                                   "wall_left": game.wall_left}
                    game.score_delta = [0.0] * 4
                    game.next_dealer = game.dealer
                    game.phase = Phase.OVER
                if game.phase == Phase.OVER:
                    self.game = None
            # 横鸡轮收口发生在弃牌判定之前 → 收口行排在打牌行前面；
            # 开轮由本步打出的横鸡触发 → 开轮行排在打牌行后面
            rounds_after = getattr(game, "_hengji_rounds", {}) or {}
            for sp in rounds_before:
                if sp not in rounds_after:
                    self._emit_hengji(actor, sp, "close")
            self._emit_action(actor, action, game)
            for sp, rnd in rounds_after.items():
                if sp not in rounds_before:
                    self._emit_hengji(actor, sp, "open")
            self._emit_progress(game)       # 报叫 / 听牌（_emit_action 覆盖不到的）
            for p in range(4):
                # 终局结算会把点炮胡牌张并入赢家手牌（+1），不能误报成摸牌
                if sum(game.hands[p]) > before[p] and game.phase != Phase.OVER:
                    self._emit_draw(p, game)
            if game.phase == Phase.OVER:
                self._emit_result(game)
            self._publish(game)
            if not self.auto and self.speed > 0:
                time.sleep(self.speed)
        if self.force_end and game.phase != Phase.OVER:
            # 提前结束本局：不计分、不入历史（_settle 里对 abort 只做座位推进）
            with self.lock:
                game.result = {"type": "abort", "deltas": [0.0] * 4,
                               "wall_left": game.wall_left}
                game.score_delta = [0.0] * 4
                game.next_dealer = game.dealer
                game.phase = Phase.OVER
                self.game = None
            self._push("result", -1, "本局已提前结束（不计分）")
        self._settle(game)
        # 回放事件已随记录落盘，这里清掉当前 game 引用（后续事件不再需要 hc）
        with self.lock:
            self._cur_game = None

    def _settle(self, game: DushanGame) -> None:
        res = game.result or {}
        if not res.get("type"):
            # 2026-10-12 修：本局**未完成就中止**（会话被停止：/api/quit、脚本退出、
            # 进程收尾…）。此前这里会照常走 _make_record/_persist，落盘一条
            # type=None、deltas=[]、detail=[] 的空白记录，后果是：
            #   · /api/stats 的 games 被多算 → 所有「局占比」分母虚高、占比偏低；
            #   · 「对局记录」里多出一条点不动的空白局；
            #   · replays.jsonl 同步多一条空回放。
            # 与 abort 同义：不计分、不入统计、不写历史、不动连庄，只沿用庄家。
            with self.lock:
                self.dealer = int(game.dealer)
            self._publish(game)
            return
        if res.get("type") == "abort":
            # 提前结束：不计分、不更新统计、不入历史（庄家不变，连庄清零）
            with self.lock:
                self.dealer_streak = 0
                self.dealer = int(res.get("next_dealer", game.dealer))
            self._publish(game)
            return
        deltas = res.get("deltas") or [0.0] * 4
        record = self._make_record(game, res)
        self._update_stats(game, res)
        self._persist(record, replay=self._make_replay(game))
        with self.lock:
            for p in range(4):
                self.cum[p] += float(deltas[p])
            if res.get("type") == "win" and res.get("winner") == self.dealer:
                self.dealer_streak += 1
            else:
                self.dealer_streak = 0
            self.dealer = int(res.get("next_dealer", game.dealer))
        self._publish(game)
        if self.on_hand_end is not None:
            try:
                self.on_hand_end(self, game)
            except Exception:
                pass

    # -- 统计与历史 ---------------------------------------------------------
    def _make_record(self, game: DushanGame, res: dict) -> dict:
        me = self.my_seat
        rec: dict = {
            "time": time.strftime("%Y-%m-%d %H:%M:%S"),
            # 回放主键：毫秒时间戳 + 手号（同一秒内连打两手也不会撞）
            "rid": f"{int(time.time() * 1000)}-{self.hand_index}",
            "hand": self.hand_index,
            "my_seat": me,
            "names": list(self.names),
            "dealer": int(game.dealer),
            "type": res.get("type"),
            "deltas": [round(float(x), 2) for x in (res.get("deltas") or [])],
        }
        # 终局听牌记录（胡牌/黄庄统一：座位 + 名字 + 各自等待张，流水查验用）
        tp = [int(p) for p in (res.get("tenpai") or [])]
        rec["tenpai"] = tp
        rec["tenpai_names"] = [self.names[p] for p in tp]
        # 赢家的胡牌张结算时已并入手牌（14 张），算等待张前须去掉，
        # 否则走「打张后兜底」分支会给出误导性的假听张（如 333万+5万
        # 胡4万后显示成 听3万/6万）。一炮多响时每位赢家都去掉。
        win_seats: set[int] = set()
        if res.get("type") == "win":
            win_seats = {int(x) for x in (res.get("winners") or [res["winner"]])}
        win_tile = res.get("win_tile")
        drop_for = (lambda p: win_tile if p in win_seats and isinstance(win_tile, int)
                    and win_tile >= 0 else None)
        waits: dict[int, list[int]] = {}
        for p in tp:
            try:
                wt_tiles, _tt = game.ting_info(p, drop=drop_for(p))
            except Exception:
                wt_tiles = []
            if wt_tiles:
                waits[p] = [int(t) for t in wt_tiles]
        rec["tenpai_waits"] = {str(k): v for k, v in waits.items()}
        # 终局各听牌者的牌型（「非平胡·听牌」局占比用：有玩家听非平胡牌型）
        tenpai_types: dict[str, list[str]] = {}
        for p in tp:
            try:
                _tt, tps = game.ting_info(p, drop=drop_for(p))
            except Exception:
                tps = []
            big = [str(x) for x in tps if str(x) != "平胡"]
            if big:
                tenpai_types[str(p)] = big
        rec["tenpai_types"] = tenpai_types
        # 牌局级标志（历史累计的新口径聚合用；旧记录没有这些字段时按 detail 兜底）
        labels = [str(d[1]) for d in (res.get("detail") or [])]
        rec["flags"] = {
            "chongfeng": "冲锋鸡" in labels,
            "ji_discard": any(x in labels for x in ("冲锋鸡", "横鸡", "幺鸡")),
            "baoji": "包鸡" in labels,
            "tenpai_bigfan": bool(tenpai_types),
            # 2026-10-12：赢家报叫的平胡局 fan_cn 会显示成「报叫」（引擎口径），
            # 但它并不是「具名大牌型」——统计口径要排除，保持与原语义一致。
            "win_bigfan": is_win_bigfan(res.get("winners_fan")),
            "huang": res.get("type") != "win",
            "huang_noten": res.get("type") != "win" and len(tp) < 4,
            # 本手是否看过模型大师推荐（「本人胡牌 / 本人累计」的分母依据）
            "hint_used": bool(self.hint_used),
        }
        if res.get("type") == "win":
            w = int(res["winner"])
            winners = [int(x) for x in (res.get("winners") or [w])]
            rec.update({
                "winner": w, "winner_name": self.names[w],
                "winners": winners,
                "winner_names": [self.names[x] for x in winners],
                "loser": res.get("loser"),
                "loser_name": self.names[res["loser"]] if res.get("loser") is not None else None,
                "is_tsumo": bool(res.get("is_tsumo")),
                "rob_kong": bool(res.get("rob_kong")),
                "how": res.get("how"),
                "void_name": self.names[res["void_player"]]
                if res.get("void_player") is not None else None,
                "fan_cn": res.get("fan_cn"), "fan": res.get("fan"),
                "detail": [[self.names[d[0]] if isinstance(d[0], int) and 0 <= d[0] < 4
                            else str(d[0]), str(d[1]), str(d[2]) if len(d) > 2 else "",
                            float(d[3]) if len(d) > 3 else float(d[2])]
                           for d in (res.get("detail") or [])],
                "winner_hand": game.hand_tiles(w),
            })
        else:
            rec.update({
                "detail": [[self.names[d[0]] if isinstance(d[0], int) and 0 <= d[0] < 4
                            else str(d[0]), str(d[1]), str(d[2]) if len(d) > 2 else "",
                            float(d[3]) if len(d) > 3 else float(d[2])]
                           for d in (res.get("detail") or [])],
            })
        return rec

    def _update_stats(self, game: DushanGame, res: dict) -> None:
        with self.lock:
            for p in range(4):
                self.stats[p]["games"] += 1
            d = res.get("deltas") or [0.0] * 4
            for p in range(4):
                self.stats[p]["score"] += float(d[p])
            if res.get("type") == "win":
                winners = [int(x) for x in (res.get("winners") or [res["winner"]])]
                for w in winners:
                    self.stats[w]["wins"] += 1
                    if res.get("is_tsumo"):
                        self.stats[w]["tsumo"] += 1
                    elif res.get("loser") is not None:
                        self.stats[w]["ron"] += 1
                if not res.get("is_tsumo") and res.get("loser") is not None:
                    self.stats[int(res["loser"])]["deal_in"] += 1
                baoji_this = set()
                for det in (res.get("detail") or []):
                    who, label = det[0], str(det[1])
                    val = float(det[3]) if len(det) > 3 else 0.0
                    if isinstance(who, int) and 0 <= who < 4:
                        if label == "冲锋鸡":
                            self.stats[who]["chongfeng"] += 1
                        if label == "横鸡":
                            self.stats[who]["hengji"] += 1
                        if label == "包鸡" and who not in baoji_this:
                            baoji_this.add(who)     # 一局最多记一次（多张鸡只算一局）
                            self.stats[who]["baoji"] += 1
                        if label == "责任鸡" and val < 0:
                            pass  # 责任鸡记在 detail 的负方向
                # 责任鸡：从 detail 找 (p,'责任鸡',..,val>0) 的被碰者其实不在 detail——
                # 这里按"谁打出的鸡被碰"简化：detail 中责任鸡的受益者是 p，付出者是丢弃者
            else:
                for p in range(4):
                    self.stats[p]["huang"] += 1
                    if p in (res.get("tenpai") or []):
                        self.stats[p]["tenpai_huang"] += 1
                baoji_this = set()
                for det in (res.get("detail") or []):
                    who, label, val = det[0], str(det[1]), float(det[3]) if len(det) > 3 else 0.0
                    if label in ("包鸡", "包大牌面") and isinstance(who, int) and 0 <= who < 4:
                        self.stats[who]["baoji_pay"] += abs(val)
                    if label == "包鸡" and isinstance(who, int) and 0 <= who < 4 \
                            and who not in baoji_this:
                        baoji_this.add(who)
                        self.stats[who]["baoji"] += 1

    HISTORY_FILE = HISTORY_DIR / "records.jsonl"
    # 回放数据单独落一个文件（2026-10-12）：每手事件流带文本，比记录本身大一到两个
    # 数量级。放进 records.jsonl 会让 /api/stats 的逐行 json.loads 变成几百毫秒。
    REPLAY_FILE = HISTORY_DIR / "replays.jsonl"

    def _persist(self, record: dict, replay: dict | None = None) -> None:
        try:
            HISTORY_DIR.mkdir(parents=True, exist_ok=True)
            with open(self.HISTORY_FILE, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
            if replay is not None:
                rp = dict(replay)
                rp["rid"] = record.get("rid")
                with open(self.REPLAY_FILE, "a", encoding="utf-8") as f:
                    f.write(json.dumps(rp, ensure_ascii=False,
                                       separators=(",", ":")) + "\n")
                remember_replay(rp)
        except Exception:
            if VERBOSE:
                traceback.print_exc()

    def _make_replay(self, game: DushanGame) -> dict:
        """把本手事件压成回放数据。

        每条事件压成定长前缀的数组，前端按索引取，避免每帧 JSON 里重复键名：
          [seq, kind, rel, tile, ctag, 巡号 r, [四家手牌张数], text]
        tile = -1 表示「这张事件没有具体牌」（如摸牌但未知、报叫）。
        hc 是服务端权威的手牌张数，前端不必靠摸/打/碰/杠去推算（抢杠等
        中断路径推算会偏），只用来画背面张数。
        """
        evs = []
        for i, ev in enumerate(self._hand_ev, 1):
            t = ev.get("tile")
            evs.append([
                i, ev.get("kind") or "", int(ev.get("rel") or 0),
                -1 if t is None else int(t), ev.get("ctag") or "",
                int(ev.get("r") or 1), [int(x) for x in (ev.get("hc") or [0, 0, 0, 0])],
                ev.get("text") or "",
                # [8] = 四家手牌牌种（紧凑编码，见 pack_hand）；老记录没有 → 空串
                list(ev.get("hands") or ["", "", "", ""]),
            ])
        return {
            "hand": self.hand_index,
            "my_seat": self.my_seat,
            "names": list(self.names),
            "dealer": int(game.dealer),
            "type": (game.result or {}).get("type"),
            # 开局发牌（第 0 帧）：13 张 ×4 + 庄家多一张
            "deal": list(getattr(self, "_deal_packed", []) or []),
            # 本局鸡牌种（回放里给鸡牌打角标）
            "ji_tiles": sorted(int(t) for t in (getattr(game, "ji_tiles", None) or [])),
            "ev": evs,
            # 终局快照（2026-10-12）：回放结束时的**结算页**直接复用实盘那套渲染，
            # 所以这里存一份与 /api/state 同口径的终局 state（已剥掉只有实时对局
            # 才需要的大块字段）。没有它的老记录 → 前端提示「无结算数据」。
            "final": self._final_snapshot(game),
        }

    def _final_snapshot(self, game: DushanGame) -> dict:
        """回放用的终局快照：与 ``_snapshot`` 同口径，剥掉大块/实时字段。

        ``events``（最近 160 条）、``counter``（27 项记牌器）、``stats`` 这些
        对结算页没用，逐手存进 replays.jsonl 会让文件涨好几倍。
        """
        try:
            snap = self._snapshot(game)
        except Exception:
            if VERBOSE:
                traceback.print_exc()
            return {}
        for k in ("events", "counter", "stats", "hand", "last_draw", "pending",
                  "my_ting", "error", "sid"):
            snap.pop(k, None)
        return snap

    # -- 前端交互 -----------------------------------------------------------
    def wait_human(self, game: DushanGame) -> Action:
        acts = game.legal_actions()
        if not acts:
            return Action(PASS)
        if len(acts) == 1 and acts[0].kind == PASS:
            return acts[0]
        if self.auto:
            hu = [a for a in acts if a.kind == HU]
            return hu[0] if hu and self.rng.random() < 0.9 else self.rng.choice(acts)

        with self.lock:
            self.act_event.clear()
            self.legal_now = list(acts)
            self.chosen = None
            self.state["pending"] = self._pending_payload(game, acts)
            self.state["turn"] = self.my_seat
            self.game = game

        got = self.act_event.wait(timeout=1800)
        with self.lock:
            if got and self.chosen is not None and self.chosen in self.legal_now:
                act = self.chosen
            else:
                act = next((a for a in acts if a.kind == PASS), acts[0])
            self.chosen = None
            self.legal_now = []
            self.game = None
            self.state["pending"] = None
        return act

    def submit(self, kind: str, tile: int) -> dict:
        with self.lock:
            if not self.legal_now:
                return {"ok": False, "error": "当前不是你的决策点"}
            for a in self.legal_now:
                if a.kind == kind and int(a.tile) == int(tile):
                    self.chosen = a
                    self.act_event.set()
                    return {"ok": True, "action": a.label()}
            # 2026-10-12 诊断：用户报告「报叫后仍能碰杠捉炮」，但引擎锁经
            # 1500 局压力测试闭合。凡是走到这里的非法提交（尤其碰/杠）都留痕，
            # 便于下次复现时定位是哪条路径放行的。
            g = self.game
            bao = list(getattr(g, "baojiao", []) or []) if g is not None else None
            if kind in ("peng", "minggang", "angang", "bugang"):
                print(f"[submit-reject] 非法碰杠提交 kind={kind} tile={tile} "
                      f"legal={[a.label() for a in self.legal_now]} baojiao={bao} "
                      f"phase={getattr(g, 'phase', None)}", flush=True)
            return {"ok": False, "error": f"非法动作 {kind}:{tile}"}

    def hint(self, strategy: str = "model:master") -> dict:
        with self.lock:
            game = self.game
            if game is None or self.state.get("pending") is None:
                return {"ok": False, "error": "现在不是你的决策点"}
            # 决策点有效即记为「本手看过推荐」——不论最终用教师 / 随机 / 模型哪种策略，
            # 都是参考了外部建议而非纯粹自己判断（决定「本人胡牌 / 本人累计」的分母）。
            self.hint_used = True
            try:
                # 教师策略：启发式，返回单一推荐
                if strategy.startswith("teacher:"):
                    style = strategy.split(":", 1)[1]
                    if style not in TEACHER_STYLES:
                        style = "balanced"
                    bot = HeuristicBot(seed=0, weights=TEACHER_STYLES[style])
                    a = bot.choose(game)
                    return {"ok": True, "model": f"teacher:{style}", "head": "teacher", "value": 0.0,
                            "items": [self._hint_item(game, a, 1.0)],
                            "situation": coach.situation_summary(game, self.my_seat),
                            "tip": "",
                            "note": "教师策略为启发式，无概率分布"}
                # 随机策略
                if strategy == "random":
                    a = RandomBot(seed=0).choose(game)
                    return {"ok": True, "model": "random", "head": "random", "value": 0.0,
                            "items": [self._hint_item(game, a, 1.0)],
                            "situation": coach.situation_summary(game, self.my_seat),
                            "tip": "",
                            "note": "随机策略"}
                # 模型策略（默认）
                want = strategy.split(":", 1)[1] if ":" in strategy else "master"
                stem = resolve_model_stem(want)
                if stem is None:
                    return {"ok": False, "error": "没有可用的模型"}
                entry = get_model(stem)
                agent = NetAgent(entry["net"], seed=0, temperature=0.0, name="hint")
                head, mask, probs, value, _obs = agent.distribution(game)
                k = max(1, min(4, int(mask.sum())))
                top = torch.topk(probs, k)
                items = []
                for idx, pr in zip(top.indices.tolist(), top.values.tolist()):
                    a = index_to_action(head, int(idx), game)
                    items.append(self._hint_item(game, a, float(pr)))
                try:
                    situation = coach.situation_summary(game, self.my_seat)
                except Exception:
                    situation = {}
                try:
                    tip = coach.compare_tip(game, self.my_seat, items)
                except Exception:
                    tip = ""
                return {"ok": True, "model": stem, "head": head,
                        "value": round(float(value), 3), "items": items,
                        "situation": situation, "tip": tip,
                        "note": "value 是模型对该局面（你的视角）的估值，正数偏有利"}
            except Exception as exc:
                return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}

    def _hint_item(self, game: DushanGame, a: Action, prob: float) -> dict:
        try:
            ex = coach.explain_action_full(game, self.my_seat, a)
        except Exception:
            ex = {"tags": [], "lines": [], "reason": explain_action(game, self.my_seat, a)}
        return {"label": a.label(), "kind": a.kind, "tile": int(a.tile),
                "prob": round(float(prob), 4), "reason": ex.get("reason", ""),
                "tags": ex.get("tags", []), "lines": ex.get("lines", [])}

    def ting_hint(self) -> dict:
        """当前决策点：每张可打牌「打出后是否听牌 / 听什么牌 / 各还剩几张」。

        返回 ``{"ok": True, "map": {"<tile>": {"tiles": [{"t": 9, "left": 3}],
        "types": ["大对子"]}}}``。打完**不听牌**的候选牌不会出现在 map 里，
        前端据此决定点牌时是否弹提示（只有听牌才提示）。
        """
        with self.lock:
            game = self.game
            if game is None or self.state.get("pending") is None:
                return {"ok": False, "error": "现在不是你的决策点"}
            me = self.my_seat
            out: dict[str, dict] = {}
            for a in list(self.legal_now):
                if a.kind != DISCARD:
                    continue
                t = int(a.tile)
                try:
                    tiles, types = game.ting_info(me, drop=t)
                except Exception:
                    continue
                if not tiles:
                    continue
                out[str(t)] = {
                    "tiles": [{"t": int(w), "left": remaining_unseen(game, me, t, int(w))}
                              for w in tiles],
                    # 牌型名（平胡不列；「平胡」只在无其他牌型时由引擎兜底返回）
                    "types": [str(x) for x in types if str(x) != "平胡"],
                }
            return {"ok": True, "map": out}

    def update_rules(self, rules: dict) -> None:
        """「下一局生效」：用前端最新载荷替换规则（翻鸡策略 / 追加鸡牌 / 分值表 / 包赔）。

        _play_hand() 每手开头都会 config_from_rules(self.rules) 重读一次，
        所以只要在唤醒下一手**之前**替换掉，新规则下一局即刻生效。
        （dict 整体替换在 CPython 下是原子操作，_play_hand 不会读到半成品字典）
        """
        self.rules = dict(rules or {})

    def next_hand(self, rules: dict | None = None) -> dict:
        if self.state.get("over"):
            if isinstance(rules, dict):
                self.update_rules(rules)
            self.next_event.set()
            return {"ok": True}
        return {"ok": False, "error": "当前这局还没结束"}

    def set_hold(self, on: bool) -> dict:
        """客户端开局鸡牌特效：on=true 暂停推进（只对**当前这一手**生效），false 立即放行。

        客户端在「检测到换局」的那一刻发 on=true —— 此时服务端 hand_index 已经是新的一手，
        所以 hold_hand 记的就是新一手；迟到的 on=true（本手已经跑了几步）只会留下一个
        到点自动失效的标记，绝不会误扣下一手。
        """
        with self.lock:
            if on:
                self.hold = True
                self.hold_hand = self.hand_index
                self.hold_until = time.monotonic() + HOLD_MAX
            else:
                self.hold = False
        return {"ok": True, "hold": self.hold}

    def end_hand_now(self) -> dict:
        """「结束本局」：提前终止当前这一手（下一个 step 边界生效，不计分不入历史）。"""
        with self.lock:
            if self.state.get("over"):
                return {"ok": False, "error": "当前这局已经结束"}
            self.force_end = True
            # 人类可能正阻塞在决策点等待 → 唤醒（wait_human 会返回兜底动作，
            # 主循环 step 一次后在边界检测到 force_end 直接终止）
            self.act_event.set()
        return {"ok": True}

    # -- 快照 ---------------------------------------------------------------
    def _pending_payload(self, game: DushanGame, acts: list[Action]) -> dict:
        if game.phase == Phase.SELF_KONG:
            mode = "respond" if getattr(game, "_rob_kong", None) is not None else "self_kong"
        elif game.phase == Phase.DISCARD:
            mode = "discard"
        else:
            mode = "respond"
        buttons, discardables = [], []
        for a in acts:
            if a.kind == DISCARD:
                discardables.append(int(a.tile))
            else:
                buttons.append({"kind": a.kind, "tile": int(a.tile), "label": a.label()})
        buttons.sort(key=lambda b: BUTTON_ORDER.get(b["kind"], 9))
        return {
            "mode": mode,
            "buttons": buttons,
            "discardables": discardables,
            "last": list(game.last_discard) if game.last_discard else None,
            "can_hint": resolve_model_stem("master") is not None,
        }

    @staticmethod
    def _fresh_draw(game: DushanGame, me: int, over: bool) -> int | None:
        """本回合「刚摸进、尚未打出」的那张牌（前端单独摆在手牌右侧）。

        仅在「自己摸牌后的决策窗口」内有效：碰/杠进牌（``drawn`` 为 False）、
        已打出（进入响应阶段）都不算；手牌数须为 3n+2。
        """
        if over:
            return None
        ld = getattr(game, "last_draw", None)
        if ld is None or not bool(getattr(game, "drawn", False)):
            return None
        if game.phase not in (Phase.DISCARD, Phase.SELF_KONG):
            return None
        try:
            if int(game.actor()) != int(me):
                return None
        except Exception:
            return None
        t = int(ld)
        if not (0 <= t < NUM_TILE_TYPES):
            return None
        if int(game.hands[me][t]) <= 0:
            return None
        if sum(int(x) for x in game.hands[me]) % 3 != 2:
            return None
        return t

    def _snapshot(self, game: DushanGame) -> dict:
        me = self.my_seat
        hand: list[int] = []
        for t in range(NUM_TILE_TYPES):
            hand.extend([t] * game.hands[me][t])

        players = []
        over = game.phase == Phase.OVER
        for rel in range(4):
            p = (me + rel) % 4
            # 透视模式用：各玩家手牌明牌（本地单机对局，随快照下发）
            hand_tiles: list[int] = []
            for t in range(NUM_TILE_TYPES):
                hand_tiles.extend([t] * int(game.hands[p][t]))
            # 终局亮牌：附听牌张与听牌牌型（总览查验用）；
            # 赢家去掉胡牌张后还原真实等待张（见 _make_record 同款口径）
            ting_tiles: list[int] = []
            ting_types: list[str] = []
            if over:
                drop = None
                r0 = game.result or {}
                if r0.get("type") == "win":
                    wset = {int(x) for x in (r0.get("winners") or [r0["winner"]])}
                    wt0 = r0.get("win_tile")
                    if p in wset and isinstance(wt0, int) and wt0 >= 0:
                        drop = wt0
                ting_tiles, ting_types = game.ting_info(p, drop=drop)
            players.append({
                "rel": rel,
                "seat": p,
                "name": self.names[p],
                "kind": self.kinds[p],
                "hand_count": int(sum(game.hands[p])),
                "hand_tiles": hand_tiles,
                "ting_tiles": ting_tiles,
                "ting_types": ting_types,
                # 被碰/杠的鸡牌带类型标记（冲锋鸡/横鸡），前端用专用牌面并横置指向被碰玩家
                "melds": [{"type": int(mt), "label": MELD_LABEL.get(mt, "?"),
                           "tile": int(t), "src": int(s),
                           "ji": _meld_ji_tag(game, p, t)} for mt, t, s in game.melds[p]],
                # 牌河用**展示视图**（2026-10-12）：被碰/明杠吃走的牌不再从河里消失，
                # 而是原位灰显（discard_claimed 标记），一眼看得出哪张被谁吃走。
                "discards": [int(t) for t in (
                    game.discard_view[p] if getattr(game, "discard_view", None) and
                    game.discard_view[p] else game.discards[p])],
                "discard_claimed": [bool(b) for b in (
                    game.discard_claimed_view[p]
                    if getattr(game, "discard_claimed_view", None) else [])],
                # 每张弃牌的鸡类标记（冲锋鸡/横鸡/幺鸡），与 discards 一一对应
                "discard_tags": _discard_tags(game, p),
                "score": round(float(self.cum[p]), 2),
                "is_dealer": p == game.dealer,
                "baojiao": bool(game.baojiao[p]),
                "has_gang": bool(game.has_dou(p)),
                "chongfeng_done": game.first_discard_done[p],
            })

        # 记牌器（2026-10-11）：各牌种「还剩几张」= 4 − 公开已见
        # （四家弃牌 + 四家副露碰3/杠4 + 我的暗手；对手暗手不可见，与通行口径一致）
        seen_cnt = [0] * NUM_TILE_TYPES
        for p in range(4):
            for t in game.discards[p]:
                seen_cnt[int(t)] += 1
            for mt, t, _s in game.melds[p]:
                seen_cnt[int(t)] += 4 if _is_kong_meld(mt) else 3
        for t in range(NUM_TILE_TYPES):
            seen_cnt[t] += int(game.hands[me][t])
        counter = [max(0, 4 - seen_cnt[t]) for t in range(NUM_TILE_TYPES)]
        # 听牌提示条：我的出牌决策点给当前听什么（一直显示到我打出牌为止）
        my_ting: list[int] = []
        pend0 = self.state.get("pending")
        if not over and isinstance(pend0, dict) and pend0.get("mode") == "discard":
            try:
                tiles_t, _tps = game.ting_info(me)
                my_ting = [int(t) for t in tiles_t]
            except Exception:
                my_ting = []

        return {
            "ready": True,
            "sid": self.sid,
            "over": over,
            "hand_index": self.hand_index,
            "my_seat": me,
            "names": list(self.names),
            "dealer": int(game.dealer),
            "dealer_streak": int(self.dealer_streak),
            "wall_left": int(game.wall_left),
            "turn": int(game.actor()) if not over else -1,
            "phase": game.phase.value,
            "last": list(game.last_discard) if game.last_discard else None,
            "repao": bool(game.repao_discard),
            "hengji_active": bool(game.hengji_active),
            "hengji_opened": bool(game.hengji_opened),
            # 正处于横鸡轮内的鸡牌种（轮内跟打同种鸡也算横鸡，前端提示芯片用）
            "hengji_species": [int(t) for t in game.hengji_species],
            # 本局鸡牌种（幺鸡 + 开局翻鸡新增）与开局翻出的指示牌，前端显示与手牌标记
            "ji_tiles": sorted(int(t) for t in game.ji_tiles),
            "kaiju_flip": int(game.kaiju_flip) if game.kaiju_flip is not None else None,
            "hand": hand,
            "last_draw": self._fresh_draw(game, me, over),
            # 记牌器：34 种牌各自剩余张数（下标=牌 id）
            "counter": counter,
            # 我的听牌（出牌决策点；空=未听牌）
            "my_ting": my_ting,
            "players": players,
            "pending": self.state.get("pending"),
            "events": list(self.events)[-160:],
            "scores": [round(float(self.cum[p]), 2) for p in range(4)],
            "stats": {str(p): self.stats[p] for p in range(4)},
            "result": self._result_payload(game, players) if over else None,
            "error": self.error,
        }

    def _who(self, seat: object) -> str:
        try:
            s = int(seat)
        except (TypeError, ValueError):
            return "全场"
        return "全场" if s < 0 else self.names[s]

    def _result_payload(self, game: DushanGame, players: list[dict]) -> dict:
        res = game.result or {}
        me = self.my_seat
        out: dict = {
            "type": res.get("type"),
            "deltas": [round(float(x), 2) for x in (res.get("deltas") or [0.0] * 4)],
            "wall_left": res.get("wall_left"),
            "mine": round(float((res.get("deltas") or [0.0] * 4)[me]), 2),
        }
        detail = res.get("detail") or []
        out["detail"] = [{"who": self._who(d[0]), "seat": int(d[0]) if isinstance(d[0], int) else -1,
                          "label": str(d[1]), "info": str(d[2]) if len(d) > 2 else "",
                          "value": float(d[3]) if len(d) > 3 else 0.0} for d in detail]
        # 两两结算矩阵：pair[a][b] = b 付给 a；pair_detail 为分项
        if res.get("pair"):
            out["pair"] = [[round(float(x), 2) for x in row] for row in res["pair"]]
            out["pair_detail"] = res.get("pair_detail") or []
        # 翻鸡信息（win 时 detail 里有 (‑1,'翻鸡',...) 项）
        fanji = [d for d in detail if str(d[1]) == "翻鸡"]
        out["fanji"] = fanji[0][2] if fanji else None
        # 「手中鸡」「翻鸡」逐牌种张数（键=座位；结算页渲染 鸡牌图×数量）
        if res.get("ji_hold_detail") is not None:
            out["ji_hold_detail"] = res.get("ji_hold_detail") or {}
            out["fanji_detail"] = res.get("fanji_detail") or {}
        # 结算页鸡牌三分类（2026-10-13 用户口径）的逐张明细：
        #   ji_discard_detail[p] = 该家打出且未被碰/杠走的鸡  [{tile, tag, val}]
        #   ji_capture_detail[p] = 该家被捉走的点炮鸡        [{tile, tag, val}]
        #   ji_claim_detail      = 责任鸡逐条                [{claimer, discarder, tile, tag, premium, gold}]
        # 前端据此把「手中鸡牌 / 打出鸡牌 / 责任鸡」渲染成 牌图×张数（金鸡再标 ×倍率）。
        if res.get("ji_discard_detail") is not None:
            out["ji_discard_detail"] = res.get("ji_discard_detail") or {}
            out["ji_capture_detail"] = res.get("ji_capture_detail") or {}
        if res.get("ji_claim_detail") is not None:
            out["ji_claim_detail"] = res.get("ji_claim_detail") or []
        # 金鸡牌种（2026-10-12）：翻鸡牌 ∩ 开局定下的鸡牌。结算页据此给对应牌加
        # 「金鸡」角标、在「手中鸡」明细里标注 ×N（顶部不再写「金鸡×2」文字）。
        out["gold_tiles"] = sorted(int(t) for t in (res.get("gold_tiles") or []))
        # 金鸡倍率（结算页明细里标「X2」用；与结算那一局的 cfg 同源）
        out["gold_mult"] = float(getattr(game.cfg, "jin_ji_multiplier", 2.0))
        # 总览用：本局鸡牌种 / 翻鸡策略 / 结算翻出的指示牌与对应鸡牌 / 胡牌方式
        out["ji_tiles"] = sorted(int(t) for t in game.ji_tiles)
        out["kaiju_flip"] = int(game.kaiju_flip) if game.kaiju_flip is not None else None
        out["jiesuan_fanji"] = str(getattr(game.cfg, "jiesuan_fanji", "both"))
        out["kaiju_fanji"] = str(getattr(game.cfg, "kaiju_fanji", "off"))
        if res.get("type") == "win":
            out["fanji_flip"] = res.get("fanji_flip")
            out["fanji_tiles"] = res.get("fanji_tiles") or []
            out["fanji_gold"] = bool(res.get("fanji_gold"))
            out["how"] = res.get("how")
            out["how_value"] = res.get("how_value")
            out["type_value"] = res.get("type_value")
            out["win_tile"] = res.get("win_tile")
        if res.get("type") == "win":
            w = int(res["winner"])
            winners = [int(x) for x in (res.get("winners") or [w])]
            lose = res.get("loser")
            void = res.get("void_player")

            def _melds_of(seat: int) -> list[dict]:
                return [{"type": int(mt), "label": MELD_LABEL.get(mt, "?"),
                         "tile": int(t), "src": int(s),
                         "ji": _meld_ji_tag(game, seat, t)}
                        for mt, t, s in game.melds[seat]]

            out.update({
                "winner": w,
                "winner_rel": (w - me) % 4,
                "winner_name": self.names[w],
                "winners": winners,
                "winners_rel": [(x - me) % 4 for x in winners],
                "winner_names": [self.names[x] for x in winners],
                "winners_fan": [{"seat": int(x.get("seat")), "rel": (int(x["seat"]) - me) % 4,
                                 "name": self.names[int(x["seat"])],
                                 "fan_cn": x.get("fan_cn"), "fan": float(x.get("fan") or 0)}
                                for x in (res.get("winners_fan") or [])
                                if isinstance(x, dict) and x.get("seat") is not None],
                "loser": lose,
                "loser_name": self.names[lose] if lose is not None else None,
                "is_tsumo": bool(res.get("is_tsumo")),
                "rob_kong": bool(res.get("rob_kong")),
                "fan_cn": res.get("fan_cn"),
                "fan": res.get("fan"),
                "void_name": self.names[void] if void is not None else None,
                "winner_hand": game.hand_tiles(w),
                # 各赢家手牌（一炮多响时每位赢家都含点炮张，渲染用）
                "winner_hands": {str(x): game.hand_tiles(x) for x in winners},
                # 完整副露（含类型/来源/鸡类标记），结算页用统一渲染
                "winner_melds": _melds_of(w),
                "winner_melds_map": {str(x): _melds_of(x) for x in winners},
            })
        else:
            tp = [int(p) for p in (res.get("tenpai") or [])]
            out.update({
                "tenpai": tp,
                "tenpai_rel": [(p - me) % 4 for p in tp],
                "tenpai_names": [self.names[p] for p in tp],
                # 流局：各听牌者可达最大牌面与「每个未听者的赔付额」（自摸分+大牌面分）
                "baodapai": [{"seat": int(b["seat"]), "rel": (int(b["seat"]) - me) % 4,
                              "name": self.names[int(b["seat"])],
                              "type": b.get("type"), "type_value": float(b.get("type_value") or 0),
                              "pay": float(b.get("pay") or 0)}
                             for b in (res.get("baodapai") or [])],
                "baodapai_on": bool(res.get("baodapai_on")),
                "zi_mo": float(getattr(game.cfg, "zi_mo", 0.0)),
                # 流局不结算鸡：此处应恒为全 0（校验口径用）
                "chickens": [round(float(x), 2) for x in (res.get("chickens") or [0.0] * 4)],
            })
        return out

    def _publish(self, game: DushanGame) -> None:
        with self.lock:
            snap = self._snapshot(game)
            self.state = snap

    # -- 事件流 -------------------------------------------------------------
    def _push(self, kind: str, rel: int, text: str, tile: int | None = None,
              ctag: str = "") -> None:
        with self.lock:
            self._ev_seq += 1
            ev = {
                "t": round(time.time() - self._t0, 1),
                # 行编号 = 手内序号；hand = 局号（前端显示「局号-序号」区分不同局）
                "hand": self.hand_index, "seq": self._ev_seq,
                "kind": kind, "rel": rel, "text": text, "tile": tile, "ctag": ctag,
            }
            self.events.append(ev)
            self._append_replay_ev(dict(ev))

    def _append_replay_ev(self, ev: dict) -> None:
        """把事件补上回放需要的字段，存进本手副本（_hand_ev）。

        回放要能「调到第 N 巡」并画出每家还剩几张手牌，所以除事件本身还要记：
          · r  —— 巡号（按累计打出数 / 4 + 1 估算；碰杠会让某一巡少一次摸牌，
                  这里只用作出场顺序的粗刻度，不追求与牌理严格对应）
          · hc —— 事件发生**之后**各家手牌张数（服务端权威值，避免前端靠
                  摸/打/碰/杠去推算出偏差；抢杠之类的中断路径尤其容易算错）
        注意（不是 bug）：引擎的 `step(打牌)` 内部会顺带把「下家自动摸牌」也做完
        （见 rules.py `_step_discard` → `_begin_respond` → `_next_player`），
        所以打牌事件的 hc 里下家已经 +1，紧接着的摸牌事件 hc 不变 —— 回放里表现为
        两帧手牌张数相同。这是引擎状态的真实值，不去人为抹平。
        """
        if ev["kind"] == "draw":
            self._ev_draws += 1
        if ev["kind"] == "discard":
            self._ev_discards += 1
        ev["r"] = self._ev_discards // 4 + 1
        g = self._cur_game
        if g is not None:
            try:
                ev["hc"] = [int(sum(g.hands[p])) for p in range(4)]
                # 四家手牌的实际牌种（紧凑编码）：回放要把牌摊开画，
                # 只给张数（hc）画不出牌面。见 pack_hand。
                ev["hands"] = [pack_hand(g.hands[p]) for p in range(4)]
            except Exception:
                pass
        self._hand_ev.append(ev)

    _t0 = time.time()

    def _emit_action(self, seat: int, action: Action, game: DushanGame) -> None:
        rel = (seat - self.my_seat) % 4
        who = self.names[seat]
        k = action.kind
        if k == DISCARD:
            tag = "（热炮）" if game.repao_discard else ""
            # ctag：这张鸡牌按引擎口径的类型名（冲锋鸡/横鸡/幺鸡）；
            # 幺鸡在流水里显示为「普通鸡」（2026-10-11 用户口径：打出鸡牌要标明种类）
            if action.tile in game.ji_tiles:
                ctag = str(game._ji_tag_of_last[seat])
                ji = f"（{'普通鸡' if ctag == '幺鸡' else ctag}）"
            else:
                ctag, ji = "", ""
            self._push("discard", rel, f"{who} 打出 {tile_cn(action.tile)}{ji}{tag}",
                       action.tile, ctag)
        elif k == HU:
            self._push("hu", rel, f"{who} 胡牌！")
        elif k == PENG:
            self._push("peng", rel, f"{who} 碰 {tile_cn(action.tile)}", action.tile)
        elif k in (ANGANG, BUGANG, MINGGANG):
            self._push("kong", rel, f"{who} {action.label()}（得通行证）", action.tile)
        # 报叫不在这里发：它写在引擎 game.log 里，由 _emit_progress 增量取（见该函数）

    def _emit_hengji(self, seat: int, species: int, what: str) -> None:
        """横鸡轮开/收事件（2026-10-11：流水记录横鸡轮的开始与结束）。

        口径：打出横鸡的玩家**开始**横鸡轮；开轮者（或轮内已吃过横鸡者）
        **再次出牌**即收口。文本首 token 是名字，前端会替换成相对称谓。
        """
        cn = tile_cn(species)
        if what == "open":
            self._push("hengji", (seat - self.my_seat) % 4,
                       f"{self.names[seat]} 开出横鸡轮 · {cn}", species, "横鸡轮开始")
        else:
            self._push("hengji", (seat - self.my_seat) % 4,
                       f"{self.names[seat]} 再次出牌 · 横鸡轮结束（{cn}）", species,
                       "横鸡轮结束")

    def _emit_progress(self, game: DushanGame) -> None:
        """把引擎里「只记状态、不发事件」的进度补进流水：报叫 / 听牌。

        碰、杠、打、胡 由 _emit_action 带更完整文案发射，这里**只补它覆盖不到的**：
          · 报叫 —— 引擎在 _step_discard 里 `self.log.append((p, "报叫"))`，
            按 game.log 增量取，天然不重不漏（同一手内 log 只增不改）；
          · 听牌 —— `_update_tenpai` 只置 ever_tenpai 标志，按标志 0→1 跳变补一条。

        报叫本身已含听牌含义（且同一 step 内 ever_tenpai 也会被置位），
        所以该座位报叫时不再重复发「听牌」。
        特意**不带等待张**：非终局时等待张属私有信息，仅 over 后随快照公开
        （见 _snapshot 里 ting_tiles 只在 over 时填）。
        """
        if game is None:
            return

        log = getattr(game, "log", None) or []
        if self._log_seen > len(log):       # 防御：万一 log 被重置，游标也跟着回零
            self._log_seen = 0
        for seat, txt in log[self._log_seen:]:
            if txt == "报叫":
                self._push("baojiao", (seat - self.my_seat) % 4,
                           f"{self.names[seat]} 报叫！", None, "报叫")
        self._log_seen = len(log)

        ever = getattr(game, "ever_tenpai", None)
        if not ever:
            return
        bao = getattr(game, "baojiao", [False] * 4)
        for p in range(4):
            if self._tenpai_seen[p] or not ever[p]:
                continue
            self._tenpai_seen[p] = True     # 先置位：无论发不发事件都只处理一次
            if bao[p]:
                continue                    # 该座位已由「报叫」覆盖，不重复
            self._push("ting", (p - self.my_seat) % 4, f"{self.names[p]} 听牌")

    def _emit_draw(self, seat: int, game: DushanGame | None = None) -> None:
        rel = (seat - self.my_seat) % 4
        # 名字与头像绑定后，人类座位也有真实名字（如「云天明」），不再硬编码「你」
        who = self.names[seat]
        tile = getattr(game, "last_draw", None) if game is not None else None
        if tile is not None:
            self._push("draw", rel, f"{who} 摸进 {tile_cn(tile)}", tile)
        else:
            self._push("draw", rel, f"{who} 摸牌")

    def _emit_result(self, game: DushanGame) -> None:
        """终局结算流水行（``rel = -1``：全局行，不是某一家在动作）。

        2026-10-12 修两个结算口径错误（用户实机反馈）：
          ① 胡牌行原用 ``res['fan']`` 当分数显示 —— 那是**首位赢家的总进账**
             （两两矩阵 pair 按行求和），不是净分。自摸局赢家既收三家自摸分、
             又要向别人的鸡/豆付钱，总进账 9 而净分只有 6，流水就写成「+9」
             与结算页的 +6 打架。现在一律取 ``deltas[seat]``（引擎写入的净分）。
          ② 荒牌行原硬编码 ``rel = 0``，前端会加「你」前缀并吃掉首个词，
             把「黄庄 · 包大牌面（…）」渲染成「**你** · 包大牌面（…）」，
             读起来像「我包了大家的牌」。现在换成 ``rel = -1``，前端按
             「结算」行原样渲染（见 renderLog 的 result 分支）。
        """
        res = game.result or {}
        deltas = res.get("deltas") or [0.0] * 4

        def net(seat: int) -> str:
            try:
                v = float(deltas[int(seat)])
            except Exception:
                v = 0.0
            return f"{v:+.0f}" if abs(v - round(v)) < 1e-6 else f"{v:+.2f}"

        if res.get("type") == "win":
            ws = [int(x) for x in (res.get("winners") or [res["winner"]])]
            how = res.get("how") or ("自摸" if res.get("is_tsumo") else "点炮")
            loser = res.get("loser")
            if len(ws) > 1:
                body = "、".join(f"{self.names[w]} {net(w)}" for w in ws)
                loss = "" if res.get("is_tsumo") else (
                    f"（{self.names[int(loser)]} 放炮）" if loser is not None else "")
                self._push("result", -1, f"一炮多响 · {how}{loss}：{body}")
            else:
                w = ws[0]
                if res.get("is_tsumo"):
                    detail = ""
                elif loser is None:
                    detail = ""
                elif res.get("rob_kong"):
                    detail = f"（{self.names[int(loser)]} 被抢杠）"
                else:
                    detail = f"（{self.names[int(loser)]} 放炮）"
                self._push("result", -1,
                           f"{self.names[w]} 胡 {res.get('fan_cn')} · {how}{detail}"
                           f" · {net(w)} 分")
        else:
            tenpai = [int(x) for x in (res.get("tenpai") or [])]
            noten = [q for q in range(4) if q not in tenpai]
            if not tenpai:
                txt = "黄庄（四家未听牌）· 不结算"
            elif not noten:
                txt = "黄庄（四家听牌）· 不结算"
            else:
                head = "黄庄 · 未听者包大牌面"
                if res.get("baodapai_on"):
                    parts = "、".join(
                        f"{self.names[int(b['seat'])]} {b.get('type') or '平胡'}"
                        f" 每家赔 {b.get('pay'):g}" for b in (res.get("baodapai") or []))
                    head = f"{head}（{parts or '无人听牌'}）"
                else:
                    head = f"{head}（已关闭）"
                txt = (f"{head} · 未听 " + "、".join(self.names[q] for q in noten)
                       + " " + "、".join(net(q) for q in noten) + " 分")
            self._push("result", -1, txt)
        # 2026-10-12 #85：结算翻鸡写进流水——翻到的指示牌 / 策略 / 计算后的翻鸡牌。
        # 只有胡牌（type == "win"）才结算翻鸡；黄庄不翻，策略关闭也不翻。
        if res.get("type") == "win" and res.get("fanji_flip") is not None:
            mode = str(getattr(game.cfg, "jiesuan_fanji", "both"))
            tiles = [int(t) for t in (res.get("fanji_tiles") or [])]
            ftxt = (f"结算翻鸡（{FANJI_MODE_CN.get(mode, mode)}）："
                    f"翻出 {tile_cn(int(res['fanji_flip']))}")
            ftxt += ("，翻鸡牌 " + "、".join(tile_cn(t) for t in tiles)
                     + ("（金鸡）" if res.get("fanji_gold") else "")
                     if tiles else "，无翻鸡牌")
            self._push("fanji", -1, ftxt)


SESSIONS: dict[str, Session] = {}
SESS_LOCK = threading.Lock()


def new_session(specs: list[dict], my_seat: int | None, seed: int,
                speed: float, auto: bool = False, rules: dict | None = None,
                names: list[str] | None = None, hold: bool = False) -> Session:
    sid = uuid.uuid4().hex[:12]
    s = Session(sid, specs, my_seat, seed, speed, auto=auto, rules=rules, names=names,
                hold=hold)
    with SESS_LOCK:
        if len(SESSIONS) > 24:
            for old in list(SESSIONS)[:8]:
                SESSIONS.pop(old).stop = True
        SESSIONS[sid] = s
    return s


def get_session(sid: str) -> Session | None:
    with SESS_LOCK:
        return SESSIONS.get(sid)


# ---------------------------------------------------------------------------
# HTTP
# ---------------------------------------------------------------------------
CTYPES = {
    ".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
    ".mp3": "audio/mpeg", ".ttf": "font/ttf", ".ico": "image/x-icon",
    ".html": "text/html; charset=utf-8", ".js": "text/javascript",
    ".css": "text/css", ".json": "application/json",
}


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = "DushanWeb/1.0"

    def log_message(self, fmt, *args):
        if VERBOSE:
            sys.stderr.write("[web] " + (fmt % args) + "\n")

    def _send(self, code: int, body: bytes, ctype: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        if ctype.startswith(("image/", "audio/", "font/")):
            self.send_header("Cache-Control", "max-age=86400")
        else:
            self.send_header("Cache-Control", "no-store")
        self.end_headers()
        try:
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def _json(self, obj, code: int = 200) -> None:
        body = json.dumps(obj, ensure_ascii=False, default=float).encode("utf-8")
        self._send(code, body, "application/json; charset=utf-8")

    def _send_download(self, body: bytes, ctype: str, filename: str) -> None:
        """带 Content-Disposition 的下载响应（导出对局记录用）。"""
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Content-Disposition",
                         f'attachment; filename="{filename}"')
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        try:
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def _body(self) -> dict:
        n = int(self.headers.get("Content-Length") or 0)
        if n <= 0:
            return {}
        try:
            return json.loads(self.rfile.read(n).decode("utf-8") or "{}")
        except Exception:
            return {}

    def _serve_static(self, relpath: str, root: Path | None = None) -> bool:
        """服务静态文件（防目录穿越，支持中文文件名 percent-encoding）。"""
        relpath = unquote(relpath)
        base = (root or STATIC).resolve()
        f = (base / relpath).resolve()
        try:
            f.relative_to(base)
        except ValueError:
            return False
        if not f.is_file():
            return False
        ctype = CTYPES.get(f.suffix.lower(), "application/octet-stream")
        self._send(200, f.read_bytes(), ctype)
        return True

    def do_GET(self):                                  # noqa: N802
        path = self.path.split("?")[0]
        if path in ("/", "/index.html"):
            return self._serve_static("dushan.html") or \
                self._send(500, b"dushan.html missing", "text/plain")
        if path == "/favicon.ico":
            return self._send(204, b"", "image/x-icon")
        if path.startswith("/assets/"):
            if self._serve_static(path[len("/assets/"):], root=ASSETS):
                return
            return self._send(404, b"not found", "text/plain")
        if path == "/api/meta":
            return self._json({
                "page_ver": _page_ver(),
                "models": available_models(),
                "teachers": list(TEACHER_STYLES),
                "tiles": [{"id": t, "name": tile_name(t), "cn": tile_cn(t),
                           "suit": t // 9, "rank": t % 9 + 1}
                          for t in range(NUM_TILE_TYPES)],
                "yaoji": YAOJI,
                "rules": "独山麻将：108 张，不能吃；点胡需通行证（杠/报叫/胡牌张成非平胡）；"
                         "1条恒为鸡，手中/副露鸡每张+1（金鸡时幺鸡×2），打出的鸡也计分"
                         "（冲锋+3/横+2/普通+1，被碰杠走则转移给碰杠者、打出者另付溢价）；"
                         "未听牌者包「已亮明的鸡」（自己打出的+碰杠来的）；"
                         "点炮张归胡牌者（是鸡牌则鸡分一并转移）；一炮多响各家均得牌；"
                         "抢杠（仅补杠）/热炮无视通行证且放炮者鸡分全烧；"
                         "龙七对须由胡牌张凑成第四张；胡后翻鸡上下鸡，金鸡翻倍；"
                         "黄庄只赔自摸分+最大牌型分。",
                "master": resolve_model_stem("master"),
                "master_info": master_identity(),
            })
        if path == "/api/state":
            sid = self._query().get("sid", "")
            s = get_session(sid)
            if not s:
                return self._json({"ok": False, "error": "会话不存在"}, 404)
            with s.lock:
                st = dict(s.state)
                st["page_ver"] = _page_ver()
                return self._json(st)
        if path == "/api/hint":
            sid = self._query().get("sid", "")
            s = get_session(sid)
            if not s:
                return self._json({"ok": False, "error": "会话不存在"}, 404)
            return self._json(s.hint(self._query().get("strategy") or "model:master"))
        if path == "/api/ting":
            # 打牌听牌提示：每张可打牌打完是否听牌 / 听什么 / 各剩几张
            sid = self._query().get("sid", "")
            s = get_session(sid)
            if not s:
                return self._json({"ok": False, "error": "会话不存在"}, 404)
            return self._json(s.ting_hint())
        if path == "/api/history":
            n = int(self._query().get("limit", "30"))
            recs: list[dict] = []
            f = Session.HISTORY_FILE
            if f.exists():
                with open(f, encoding="utf-8") as fh:
                    for i, line in enumerate(fh):
                        line = line.strip()
                        if line:
                            try:
                                rec = json.loads(line)
                            except Exception:
                                continue
                            # 0 基行号：删除按钮据此精确定位（文件只在尾部追加，行号稳定）
                            rec["_ln"] = i
                            recs.append(rec)
            recs = recs[-n:][::-1]
            # 标记哪些局还能回放（旧记录没有 rid / 回放已滚出缓存 → 前端置灰不可点）
            _load_replays()
            for r in recs:
                r["has_replay"] = bool(r.get("rid")) and r.get("rid") in _REPLAY_CACHE
            return self._json({"ok": True, "records": recs})
        if path == "/api/history_export":
            # 导出对局记录（2026-10-12 用户需求）：?format=json|csv 直接下载。
            fmt = (self._query().get("format") or "json").strip().lower()
            if fmt not in ("json", "csv"):
                fmt = "json"
            body, ctype, fname = export_history(Session.HISTORY_FILE, fmt)
            return self._send_download(body, ctype, fname)
        if path == "/api/replay":
            # 单局回放数据：/api/replay?rid=<记录里的 rid>
            rp = load_replay((self._query().get("rid") or "").strip())
            if rp is None:
                return self._json({"ok": False, "error": "这一局的回放已不在缓存中"})
            return self._json({"ok": True, "replay": rp})
        if path == "/api/stats":
            # 历史累计（2026-10-12 口径）：
            #   对局=局数；自摸=我自摸胡的局占比；
            #   送胡=「整局视角」：该局只要有玩家不是自摸胡（点炮 / 抢杠 / 热炮…）就算，
            #        不带个人输赢视角（原「点胡」口径算的是相反的事：别人点炮我胡）；
            #   冲锋鸡/出鸡/包鸡/非平胡·听牌/非平胡·胡牌/黄庄·局占率/黄庄·未听牌=局占比；
            #   本人胡牌/本人累计=只在「全程没开启过模型大师推荐」的局里统计
            #        （分母 solo_games 是这些局数，不是我参与的全部局数）。
            # 旧记录没有 flags 时按 detail / how / tenpai 兜底。
            agg = {"games": 0, "zimo": 0, "songhu_games": 0, "score": 0.0,
                   "chongfeng_games": 0, "ji_discard_games": 0, "baoji_games": 0,
                   "tenpai_bigfan_games": 0, "win_bigfan_games": 0,
                   "huang_games": 0, "huang_noten_games": 0,
                   "solo_games": 0, "solo_win_games": 0, "solo_score": 0.0}
            f = Session.HISTORY_FILE
            if f.exists():
                with open(f, encoding="utf-8") as fh:
                    for line in fh:
                        try:
                            r = json.loads(line)
                        except Exception:
                            continue
                        if r.get("type") == "abort":
                            continue        # 提前结束的局不计入任何统计
                        me = r.get("my_seat", 0)
                        d = (r.get("deltas") or [0.0] * 4)
                        agg["games"] += 1
                        agg["score"] += float(d[me]) if me < len(d) else 0.0
                        fl = r.get("flags") or {}
                        labels = [str(x[1]) for x in (r.get("detail") or [])]
                        if fl.get("chongfeng", "冲锋鸡" in labels):
                            agg["chongfeng_games"] += 1
                        if fl.get("ji_discard",
                                  any(x in labels for x in ("冲锋鸡", "横鸡", "幺鸡"))):
                            agg["ji_discard_games"] += 1
                        if fl.get("baoji", "包鸡" in labels):
                            agg["baoji_games"] += 1
                        if fl.get("tenpai_bigfan", False):
                            agg["tenpai_bigfan_games"] += 1
                        # 「本人胡牌 / 本人累计」的分母：本局全程没请求过模型大师推荐。
                        # 旧记录缺该标志 → 按“用过推荐”保守处理，不计入（无法证明是纯自己打）。
                        solo = not fl.get("hint_used", True)
                        dme = float(d[me]) if me < len(d) else 0.0
                        if solo:
                            agg["solo_games"] += 1
                            agg["solo_score"] += dme
                        if r.get("type") == "win":
                            # 送胡（整局视角）：这局有没有人靠别人打出的牌胡
                            # —— 点炮 / 抢杠 / 热炮等一切非自摸都算
                            if not r.get("is_tsumo"):
                                agg["songhu_games"] += 1
                            winners = [int(x) for x in (r.get("winners") or
                                                        [r.get("winner")]) if x is not None]
                            big = fl.get("win_bigfan")
                            if big is None:
                                big = is_bigfan_name(r.get("fan_cn"))
                            if big:
                                agg["win_bigfan_games"] += 1
                            if me in winners:
                                if r.get("is_tsumo"):
                                    agg["zimo"] += 1
                                if solo:
                                    agg["solo_win_games"] += 1
                        else:
                            agg["huang_games"] += 1
                            if fl.get("huang_noten", len(r.get("tenpai") or []) < 4):
                                agg["huang_noten_games"] += 1
            return self._json({"ok": True, "all_time": agg})
        return self._send(404, b"not found", "text/plain")

    def _query(self) -> dict:
        if "?" not in self.path:
            return {}
        out = {}
        for kv in self.path.split("?", 1)[1].split("&"):
            if "=" in kv:
                k, v = kv.split("=", 1)
                out[k] = v.replace("%20", " ")
        return out

    def do_POST(self):                                 # noqa: N802
        path = self.path.split("?")[0]
        data = self._body()

        if path == "/api/new":
            specs = data.get("opponents")
            if not isinstance(specs, list) or len(specs) != 3:
                specs = [{"type": "model", "model": "master"},
                         {"type": "teacher", "style": "balanced"},
                         {"type": "random"}]
            specs = [s if isinstance(s, dict) else {} for s in specs][:3]
            my_seat = data.get("my_seat")
            my_seat = int(my_seat) if isinstance(my_seat, int) and 0 <= my_seat < 4 else None
            seed = int(data.get("seed") or random.randrange(10 ** 8))
            speed = float(data.get("speed", 0.45))
            speed = min(max(speed, 0.0), 5.0)
            rules = data.get("rules") if isinstance(data.get("rules"), dict) else {}
            names = data.get("names") if isinstance(data.get("names"), list) else None
            # 开局鸡牌特效「扣牌」：客户端要求第一手也先等特效播完（on=false 或不存在=不扣）
            hold = bool(data.get("hold"))
            s = new_session(specs, my_seat, seed, speed, rules=rules, names=names, hold=hold)
            return self._json({"ok": True, "sid": s.sid, "my_seat": s.my_seat})

        if path == "/api/hold":
            # 客户端全屏特效（开局鸡牌）期间的暂停/放行闸门，见 Session.set_hold
            s = get_session(str(data.get("sid", "")))
            if not s:
                return self._json({"ok": False, "error": "会话不存在"}, 404)
            return self._json(s.set_hold(bool(data.get("on"))))

        if path == "/api/speed":
            # 对手出牌节奏可在对局中修改（设置面板「出牌与推荐」里那一项）
            s = get_session(str(data.get("sid", "")))
            if not s:
                return self._json({"ok": False, "error": "会话不存在"}, 404)
            try:
                v = float(data.get("speed", s.speed))
            except (TypeError, ValueError):
                v = s.speed
            s.speed = min(max(v, 0.0), 5.0)
            return self._json({"ok": True, "speed": s.speed})

        if path == "/api/history_delete":
            # 删除对局记录（连同它的回放）。**支持单条与批量**：
            #   单条：{"ln": 12} 或 {"rid": "..."}；
            #   批量：{"lns": [1,2,3], "rids": ["a","b"]}（两者可并用，逐项尽力删）。
            # 删完统计自动同步 —— /api/stats 是逐行扫 records.jsonl 现算的，
            # 不另存账本，所以「数据统计」永远只统计对局记录里还在的局。
            def _as_int_list(v):
                if v is None:
                    return []
                if isinstance(v, (list, tuple)):
                    it = v
                else:
                    it = [v]
                out = []
                for x in it:
                    try:
                        out.append(int(x))
                    except (TypeError, ValueError):
                        continue
                return out

            def _as_str_list(v):
                if v is None:
                    return []
                it = v if isinstance(v, (list, tuple)) else [v]
                return [str(x).strip() for x in it if str(x).strip()]

            lns = _as_int_list(data.get("lns"))
            rids = _as_str_list(data.get("rids"))
            if not lns and not rids:
                lns = _as_int_list(data.get("ln"))
                rids = _as_str_list(data.get("rid"))
            items: list[tuple] = []
            for i in range(max(len(lns), len(rids))):
                items.append((lns[i] if i < len(lns) else None,
                              rids[i] if i < len(rids) else None))
            if not items:
                return self._json({"ok": False, "error": "缺少定位参数"}, 400)
            r = delete_history_records(Session.HISTORY_FILE, Session.REPLAY_FILE, items)
            if r.get("ok"):
                print(f"[history] 删除 {len(items)} 项 "
                      f"（记录 {r['removed_rec']} 行 / 回放 {r['removed_replay']} 行"
                      + (f" / 跳过 {len(r['skipped'])} 项" if r.get("skipped") else "")
                      + "）", flush=True)
            return self._json(r, 200 if r.get("ok") else 400)

        if path == "/api/act":
            s = get_session(str(data.get("sid", "")))
            if not s:
                return self._json({"ok": False, "error": "会话不存在"}, 404)
            return self._json(s.submit(str(data.get("kind", "")), int(data.get("tile", -1))))

        if path == "/api/next":
            s = get_session(str(data.get("sid", "")))
            if not s:
                return self._json({"ok": False, "error": "会话不存在"}, 404)
            # 前端把最新 rules 一并送来 → 下一局用新设置（否则冻结在开局那一刻）
            rules = data.get("rules") if isinstance(data.get("rules"), dict) else None
            return self._json(s.next_hand(rules))

        if path == "/api/end_hand":
            # 「结束本局」：提前终止当前这一手（不计分、不入历史）
            s = get_session(str(data.get("sid", "")))
            if not s:
                return self._json({"ok": False, "error": "会话不存在"}, 404)
            return self._json(s.end_hand_now())

        if path == "/api/quit":
            s = get_session(str(data.get("sid", "")))
            if s:
                s.stop = True
                s.act_event.set()
                s.next_event.set()
            return self._json({"ok": True})

        return self._send(404, b"not found", "text/plain")


# ---------------------------------------------------------------------------
# 自检
# ---------------------------------------------------------------------------
def selftest(n_hands: int, specs: list[dict]) -> int:
    print(f"[selftest] 自动对局 {n_hands} 局（独山规则），对手 {json.dumps(specs, ensure_ascii=False)}",
          flush=True)
    s = Session("selftest", specs, my_seat=0, seed=20261001, speed=0.0, auto=True)
    seen: list[dict] = []

    def on_end(sess, game):
        r = dict(game.result or {})
        r["_names"] = list(sess.names)
        seen.append(r)

    s.on_hand_end = on_end
    t0 = time.time()
    while len(seen) < n_hands and not s.error:
        time.sleep(0.05)
        if time.time() - t0 > 600:
            print(f"[selftest] 超时：只完成 {len(seen)}/{n_hands} 局", flush=True)
            s.stop = True
            return 1
    s.stop = True
    if s.error:
        print(s.error, flush=True)
        return 1

    n_win = n_tsumo = n_ron = n_huang = 0
    for i, r in enumerate(seen, 1):
        names = r.get("_names") or []
        t = r.get("type")
        d = r.get("deltas") or [0.0] * 4
        if t == "win":
            n_win += 1
            w = int(r["winner"])
            if r.get("is_tsumo"):
                n_tsumo += 1
                how = "自摸"
            else:
                n_ron += 1
                how = f"点胡(放炮:{names[r['loser']]})" if r.get("loser") is not None else "点胡"
            desc = f"{names[w]} 胡 {r.get('fan_cn')} +{r.get('fan')}分 {how}"
        elif t == "huangzhuang":
            n_huang += 1
            desc = f"黄庄 · 听牌 {[names[p] for p in r.get('tenpai', [])]}"
        else:
            desc = f"!! 异常终局 {t}"
        print(f"  第 {i:>3} 局: {desc:<52} 得分 {[round(x, 1) for x in d]}", flush=True)

    bad: list[str] = []
    for i, r in enumerate(seen, 1):
        t = r.get("type")
        d = r.get("deltas") or [0.0] * 4
        if t not in ("win", "huangzhuang"):
            bad.append(f"第{i}局 终局类型异常：{t}")
        if abs(sum(d)) > 1e-6:
            bad.append(f"第{i}局 非零和：Σ={sum(d)}")

    dt = time.time() - t0
    print(f"\n[selftest] {len(seen)} 局 / {dt:.1f}s（{dt / max(1, len(seen)) * 1000:.0f} ms/局）"
          f"  胡 {n_win}（自摸 {n_tsumo} / 点胡 {n_ron}）  黄庄 {n_huang}", flush=True)
    if bad:
        print("[selftest] FAIL\n  " + "\n  ".join(bad), flush=True)
        return 1
    print("[selftest] PASS — 全部终局且零和", flush=True)
    return 0


def _lan_urls(port: int) -> list:
    """枚举本机可用于「同一局域网内手机/平板访问」的 IPv4 地址。

    手机要能连上，必须用局域网地址（如 http://192.168.1.7:8770），
    127.0.0.1 只是手机自己。这里把网卡地址都列出来供用户照着输。
    """
    found = []          # [(网卡名, ip)]
    try:                # 优先 psutil：能拿到网卡名，且更容易过滤虚拟网卡
        import psutil  # type: ignore
        for name, addrs in psutil.net_if_addrs().items():
            for a in addrs:
                if a.family != socket.AF_INET or not a.address:
                    continue
                ip = a.address
                # 回环 / APIPA（169.254.x 是 DHCP 失败时的自分配地址，连不通）
                if ip.startswith("127.") or ip.startswith("169.254."):
                    continue
                found.append((name, ip))
    except Exception:
        pass
    if not found:       # 兜底：不需要 psutil 也能用
        try:
            for info in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET):
                ip = info[4][0]
                if not ip.startswith("127."):
                    found.append(("", ip))
        except Exception:
            pass

    def rank(item):     # 家庭/办公局域网常见网段排前面
        ip = item[1]
        return (0 if ip.startswith("192.168.") else
                1 if ip.startswith("10.") else
                2 if ip.startswith("172.") else 3)

    out, seen = [], set()
    for name, ip in sorted(found, key=rank):
        if ip in seen:
            continue
        seen.add(ip)
        out.append((name, f"http://{ip}:{port}"))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="0.0.0.0",
                    help="监听地址。0.0.0.0=同一局域网的手机/平板也能连（默认）；"
                         "127.0.0.1=只允许本机访问")
    ap.add_argument("--port", type=int, default=8770)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--selftest", type=int, default=0)
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    global VERBOSE
    VERBOSE = args.verbose
    torch.set_num_threads(max(1, args.threads))

    specs = [{"type": "model", "model": "master"},
             {"type": "teacher", "style": "balanced"},
             {"type": "random"}]
    if args.selftest:
        return selftest(args.selftest, specs)

    srv = ThreadingHTTPServer((args.host, args.port), Handler)
    srv.daemon_threads = True
    print(f"独山麻将 · 网页对局  →  本机  http://127.0.0.1:{args.port}", flush=True)
    if args.host in ("0.0.0.0", "::", ""):
        urls = _lan_urls(args.port)
        if urls:
            print("   手机/平板（连同一个 Wi-Fi）→", flush=True)
            for name, url in urls:
                print(f"      {url}" + (f"   [{name}]" if name else ""), flush=True)
            print("   ※ 手机打不开时：Windows 防火墙首次会弹窗，请勾选「专用网络」允许；"
                  "若已误点拒绝，见 docs/手机访问.md", flush=True)
        else:
            print("   （未探测到局域网 IPv4 地址，请确认已连上 Wi-Fi/网线）", flush=True)
    else:
        print(f"   --host {args.host}：仅该地址可访问（局域网设备连不上）", flush=True)
    print(f"大师模型: {resolve_model_stem('master')}", flush=True)
    print("Ctrl+C 退出", flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\nbye")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
