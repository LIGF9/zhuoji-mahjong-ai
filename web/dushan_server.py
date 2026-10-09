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
import pickle
import random
import re
import sys
import threading
import time
import traceback
import uuid
import zipfile
from collections import deque
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
from zhuoji.net import NetConfig, build_model  # noqa: E402
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
HISTORY_DIR = Path(__file__).resolve().parent / "history"

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
    return DushanConfig(
        jiesuan_fanji=norm_fanji(r.get("jiesuan", r.get("jiesuan_fanji", "both")),
                                 allow_off=False),
        kaiju_fanji=norm_fanji(r.get("kaiju", r.get("kaiju_fanji", "off")),
                               allow_off=True),
        mantiangji=bool(r.get("mantiangji", False)),
        # 可调分值表（网页设置）
        zi_mo=_f("zimo", 3.0), dian_pao=_f("dianpao", 3.0),
        gang_kai=_f("gangkai", 5.0), gang_pao=_f("gangpao", 5.0),
        da_dui_zi=_f("dadui", 5.0), qing_yi_se=_f("qing", 10.0),
        qi_dui=_f("qidui", 10.0), long_qi_dui=_f("long", 20.0),
        dan_diao=_f("dudiao", 10.0), baojiao_bonus=_f("baojiao", 13.0),
        # 鸡的分值表（教程 10.1，全部可在设置里调）
        kong_chickens=_f("kong", 3.0),
        jin_ji_multiplier=_f("jinji", 2.0),
        ji_hold=_f("jihold", 1.0),
        ji_discard_plain=_f("jiplain", 1.0),
        ji_chongfeng=_f("jicf", 3.0),
        ji_heng=_f("jiheng", 2.0),
        ji_fanji=_f("jifanji", 1.0),
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
    nc = ck.get("net") or {}
    cfg = NetConfig(int(nc.get("channels", 48)), int(nc.get("blocks", 2)),
                    int(nc.get("hidden", 256)))
    net = build_model(cfg)
    net.load_state_dict(ck["state_dict"])
    net.eval()
    params = sum(int(v.numel()) for v in ck["state_dict"].values() if hasattr(v, "numel"))
    entry = {
        "net": net,
        "mtime": mtime,
        "config": ck.get("config") or {},
        "params": params,
        "meta": {"epoch": ck.get("epoch"), "val_loss": ck.get("val_loss"),
                 "iter": ck.get("iter"), "eval": ck.get("eval"),
                 "net": {"channels": cfg.channels, "blocks": cfg.blocks,
                         "hidden": cfg.hidden},
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
    """把 game.ji_events 中该玩家的鸡事件按顺序对回其弃牌，返回逐张标记。"""
    tags: list[str | None] = [None] * len(game.discards[p])
    ji = [e for e in game.ji_events if int(e[0]) == p]
    k = 0
    for i, t in enumerate(game.discards[p]):
        if int(t) in game.ji_tiles and k < len(ji):
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
                 rules: dict | None = None, names: list[str] | None = None):
        self.sid = sid
        self.specs = specs
        self.seed = seed
        self.speed = speed
        self.auto = auto
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
        self.dealer = 0
        self.dealer_streak = 0
        self.my_seat = my_seat if my_seat is not None else self.rng.randrange(4)
        self.stop = False
        self.error: str | None = None

        # 统计：本会话（座位粒度）
        self.stats: dict[int, dict] = {
            p: {"games": 0, "wins": 0, "tsumo": 0, "ron": 0, "deal_in": 0,
                "huang": 0, "tenpai_huang": 0, "baoji_pay": 0.0,
                "chongfeng": 0, "zeren_pay": 0.0, "score": 0.0}
            for p in range(4)
        }

        self.events: deque = deque(maxlen=600)
        self.state: dict = {"ready": False}
        self.on_hand_end = None

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
        with self.lock:
            self.game = game
        bots = self._setup_seats(game)
        self._publish(game)
        steps = 0
        while game.phase != Phase.OVER and not self.stop:
            actor = game.actor()
            action = bots[actor].choose(game)
            before = [sum(game.hands[p]) for p in range(4)]
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
            self._emit_action(actor, action, game)
            for p in range(4):
                # 终局结算会把点炮胡牌张并入赢家手牌（+1），不能误报成摸牌
                if sum(game.hands[p]) > before[p] and game.phase != Phase.OVER:
                    self._emit_draw(p, game)
            if game.phase == Phase.OVER:
                self._emit_result(game)
            self._publish(game)
            if not self.auto and self.speed > 0:
                time.sleep(self.speed)
        self._settle(game)

    def _settle(self, game: DushanGame) -> None:
        res = game.result or {}
        deltas = res.get("deltas") or [0.0] * 4
        record = self._make_record(game, res)
        self._update_stats(game, res)
        self._persist(record)
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
                for det in (res.get("detail") or []):
                    who, label = det[0], str(det[1])
                    val = float(det[3]) if len(det) > 3 else 0.0
                    if isinstance(who, int) and 0 <= who < 4:
                        if label == "冲锋鸡":
                            self.stats[who]["chongfeng"] += 1
                        if label == "责任鸡" and val < 0:
                            pass  # 责任鸡记在 detail 的负方向
                # 责任鸡：从 detail 找 (p,'责任鸡',..,val>0) 的被碰者其实不在 detail——
                # 这里按"谁打出的鸡被碰"简化：detail 中责任鸡的受益者是 p，付出者是丢弃者
            else:
                for p in range(4):
                    self.stats[p]["huang"] += 1
                    if p in (res.get("tenpai") or []):
                        self.stats[p]["tenpai_huang"] += 1
                for det in (res.get("detail") or []):
                    who, label, val = det[0], str(det[1]), float(det[3]) if len(det) > 3 else 0.0
                    if label in ("包鸡", "包大牌面") and isinstance(who, int) and 0 <= who < 4:
                        self.stats[who]["baoji_pay"] += abs(val)

    HISTORY_FILE = HISTORY_DIR / "records.jsonl"

    def _persist(self, record: dict) -> None:
        try:
            HISTORY_DIR.mkdir(parents=True, exist_ok=True)
            with open(self.HISTORY_FILE, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        except Exception:
            if VERBOSE:
                traceback.print_exc()

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
        return {"ok": False, "error": f"非法动作 {kind}:{tile}"}

    def hint(self, strategy: str = "model:master") -> dict:
        with self.lock:
            game = self.game
            if game is None or self.state.get("pending") is None:
                return {"ok": False, "error": "现在不是你的决策点"}
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

    def next_hand(self) -> dict:
        if self.state.get("over"):
            self.next_event.set()
            return {"ok": True}
        return {"ok": False, "error": "当前这局还没结束"}

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
                "discards": [int(t) for t in game.discards[p]],
                # 每张弃牌的鸡类标记（冲锋鸡/横鸡/幺鸡），与 discards 一一对应
                "discard_tags": _discard_tags(game, p),
                "score": round(float(self.cum[p]), 2),
                "is_dealer": p == game.dealer,
                "baojiao": bool(game.baojiao[p]),
                "has_gang": bool(game.has_dou(p)),
                "chongfeng_done": game.first_discard_done[p],
            })

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
            self.events.append({
                "t": round(time.time() - self._t0, 1),
                "kind": kind, "rel": rel, "text": text, "tile": tile, "ctag": ctag,
            })

    _t0 = time.time()

    def _emit_action(self, seat: int, action: Action, game: DushanGame) -> None:
        rel = (seat - self.my_seat) % 4
        who = self.names[seat]
        k = action.kind
        if k == DISCARD:
            tag = "（热炮）" if game.repao_discard else ""
            ji = "（鸡）" if action.tile in game.ji_tiles else ""
            # ctag：这张鸡牌按引擎口径的类型名（冲锋鸡/横鸡/幺鸡），
            # 前端据此弹「XX 打出 横鸡」toast（状态芯片不再常驻提示横鸡轮）
            ctag = str(game._ji_tag_of_last[seat]) if action.tile in game.ji_tiles else ""
            self._push("discard", rel, f"{who} 打出 {tile_cn(action.tile)}{ji}{tag}",
                       action.tile, ctag)
        elif k == HU:
            self._push("hu", rel, f"{who} 胡牌！")
        elif k == PENG:
            self._push("peng", rel, f"{who} 碰 {tile_cn(action.tile)}", action.tile)
        elif k in (ANGANG, BUGANG, MINGGANG):
            self._push("kong", rel, f"{who} {action.label()}（得通行证）", action.tile)
        # 报叫在引擎里记录 log，这里从 game.log 增量不可行，改为快照徽标

    def _emit_draw(self, seat: int, game: DushanGame | None = None) -> None:
        rel = (seat - self.my_seat) % 4
        who = "你" if seat == self.my_seat else self.names[seat]
        tile = getattr(game, "last_draw", None) if game is not None else None
        if tile is not None:
            self._push("draw", rel, f"{who} 摸进 {tile_cn(tile)}", tile)
        else:
            self._push("draw", rel, f"{who} 摸牌")

    def _emit_result(self, game: DushanGame) -> None:
        res = game.result or {}
        if res.get("type") == "win":
            ws = [int(x) for x in (res.get("winners") or [res["winner"]])]
            if len(ws) > 1:
                who = "、".join(self.names[x] for x in ws)
                how = res.get("how") or "点炮"
                self._push("result", (ws[0] - self.my_seat) % 4,
                           f"{who} 同时胡牌 · {how}（一炮多响）")
            else:
                w = ws[0]
                self._push("result", (w - self.my_seat) % 4,
                           f"{self.names[w]} 胡牌 · {res.get('fan_cn')} · +{res.get('fan')} 分")
        else:
            bd = res.get("baodapai") or []
            tip = "、".join(f"{self.names[int(b['seat'])]} {b.get('type') or '平胡'}"
                            f" 每家赔 {b.get('pay'):g}" for b in bd) or "无人听牌"
            self._push("result", 0, f"黄庄 · 包大牌面（{tip}）")


SESSIONS: dict[str, Session] = {}
SESS_LOCK = threading.Lock()


def new_session(specs: list[dict], my_seat: int | None, seed: int,
                speed: float, auto: bool = False, rules: dict | None = None,
                names: list[str] | None = None) -> Session:
    sid = uuid.uuid4().hex[:12]
    s = Session(sid, specs, my_seat, seed, speed, auto=auto, rules=rules, names=names)
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
        if path == "/api/history":
            n = int(self._query().get("limit", "30"))
            recs: list[dict] = []
            f = Session.HISTORY_FILE
            if f.exists():
                with open(f, encoding="utf-8") as fh:
                    for line in fh:
                        line = line.strip()
                        if line:
                            try:
                                recs.append(json.loads(line))
                            except Exception:
                                pass
            recs = recs[-n:][::-1]
            return self._json({"ok": True, "records": recs})
        if path == "/api/stats":
            # 历史累计（records.jsonl 里 seat == my_seat 的那家 = 人类）
            agg = {"games": 0, "wins": 0, "tsumo": 0, "ron": 0, "deal_in": 0,
                   "huang": 0, "baoji_pay": 0.0, "chongfeng": 0, "score": 0.0}
            f = Session.HISTORY_FILE
            if f.exists():
                with open(f, encoding="utf-8") as fh:
                    for line in fh:
                        try:
                            r = json.loads(line)
                        except Exception:
                            continue
                        me = r.get("my_seat", 0)
                        d = (r.get("deltas") or [0.0] * 4)
                        agg["games"] += 1
                        agg["score"] += float(d[me]) if me < len(d) else 0.0
                        if r.get("type") == "win":
                            winner = r.get("winner")
                            loser = r.get("loser")
                            if winner is not None and int(winner) == me:
                                agg["wins"] += 1
                                if r.get("is_tsumo"):
                                    agg["tsumo"] += 1
                                else:
                                    agg["ron"] += 1
                            elif loser is not None and int(loser) == me:
                                agg["deal_in"] += 1
                        else:
                            agg["huang"] += 1
                        for det in (r.get("detail") or []):
                            label = str(det[1])
                            if label == "冲锋鸡" and det[0] == r.get("names", [""]*4)[me]:
                                agg["chongfeng"] += 1
                            if label == "包鸡" and det[0] == r.get("names", [""]*4)[me]:
                                agg["baoji_pay"] += abs(float(det[3]) if len(det) > 3 else 0.0)
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
            speed = min(max(speed, 0.0), 3.0)
            rules = data.get("rules") if isinstance(data.get("rules"), dict) else {}
            names = data.get("names") if isinstance(data.get("names"), list) else None
            s = new_session(specs, my_seat, seed, speed, rules=rules, names=names)
            return self._json({"ok": True, "sid": s.sid, "my_seat": s.my_seat})

        if path == "/api/speed":
            # 对手出牌节奏可在对局中修改（设置面板「出牌与推荐」里那一项）
            s = get_session(str(data.get("sid", "")))
            if not s:
                return self._json({"ok": False, "error": "会话不存在"}, 404)
            try:
                v = float(data.get("speed", s.speed))
            except (TypeError, ValueError):
                v = s.speed
            s.speed = min(max(v, 0.0), 3.0)
            return self._json({"ok": True, "speed": s.speed})

        if path == "/api/act":
            s = get_session(str(data.get("sid", "")))
            if not s:
                return self._json({"ok": False, "error": "会话不存在"}, 404)
            return self._json(s.submit(str(data.get("kind", "")), int(data.get("tile", -1))))

        if path == "/api/next":
            s = get_session(str(data.get("sid", "")))
            if not s:
                return self._json({"ok": False, "error": "会话不存在"}, 404)
            return self._json(s.next_hand())

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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
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
    print(f"独山麻将 · 网页对局  →  http://{args.host}:{args.port}", flush=True)
    print(f"大师模型: {resolve_model_stem('master')}", flush=True)
    print("Ctrl+C 退出", flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\nbye")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
