"""
网页版贵州捉鸡对局服务 —— 你 vs 神经网络 / 随机 / 启发式教师。

启动::

    python web/server.py --port 8765
    # 然后浏览器打开 http://127.0.0.1:8765

设计要点
--------
1. **只用标准库**（``http.server``）。不引入新依赖，也就不会碰到
   `pip` 那个批量删除守卫，已经装好的 torch 环境一点不动。
2. **不改规则引擎**。``ZhuojiGame.play()`` 是个整体循环，插不进人类玩家，
   所以这里自己写主循环：每轮只调一次 ``game.step()``，人类玩家用一个
   阻塞式 Bot（:class:`HumanBot`）实现——**游戏线程停在 ``choose()`` 里**，
   等前端提交动作才继续。规则引擎因此一行都不用改。
3. **不泄露隐藏信息**。前端快照里对手只有手牌**张数**，没有手牌内容；
   鸡牌也只给"幺鸡恒为鸡"这条公开信息（第二只鸡要胡牌后翻牌才知道）。
4. 每步之后重建快照并加个小延时，让人类看得清对手在干什么。
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import threading
import time
import traceback
import uuid
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from zhuoji.agent import NetAgent  # noqa: E402
from zhuoji.bots import TEACHER_STYLES, Bot, HeuristicBot, RandomBot  # noqa: E402
from zhuoji.encoder import index_to_action  # noqa: E402
from zhuoji.fan import (  # noqa: E402
    MELD_KONG_ADDED,
    MELD_KONG_CONCEALED,
    MELD_KONG_EXPOSED,
    MELD_PONG,
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
    RulesConfig,
    ZhuojiGame,
)
from zhuoji.tiles import (  # noqa: E402
    NUM_TILE_TYPES, YAOJI, parse_tile, tile_cn, tile_name,
)

STATIC = Path(__file__).resolve().parent / "static"
MODEL_DIR = ROOT / "models"

MELD_LABEL = {
    MELD_PONG: "碰",
    MELD_KONG_EXPOSED: "点豆",
    MELD_KONG_ADDED: "爬坡豆",
    MELD_KONG_CONCEALED: "闷豆",
}

#: 动作按钮的展示优先级（越小越靠前）
BUTTON_ORDER = {HU: 0, ANGANG: 1, BUGANG: 1, MINGGANG: 1, PENG: 2, PASS: 3}

HUMAN_SEAT_NAME = "你"

#: 单局步数上限（正常一局 60~120 步）。超了说明有 bug，强制收尾而不是挂死会话。
MAX_STEPS = 3000

VERBOSE = False


# ---------------------------------------------------------------------------
# 模型仓库
# ---------------------------------------------------------------------------
_MODELS: dict[str, dict] = {}
_MODELS_LOCK = threading.Lock()


def available_models() -> list[dict]:
    """列出 ``models/*.pt``。顺带把模型载进缓存——既让界面能显示训练轮次，
    也把首次开局要等的加载时间提前消化掉。"""
    out = []
    for p in sorted(MODEL_DIR.glob("*.pt")):
        info = {"stem": p.stem, "size_kb": round(p.stat().st_size / 1024, 1),
                "mtime": time.strftime("%Y-%m-%d %H:%M", time.localtime(p.stat().st_mtime))}
        try:
            with _MODELS_LOCK:
                loaded = p.stem in _MODELS
            if not loaded:
                get_model(p.stem)
        except Exception:                      # 坏文件不该让整个接口挂掉
            info["bad"] = True
        with _MODELS_LOCK:
            if p.stem in _MODELS:
                info.update(_MODELS[p.stem]["meta"])
        out.append(info)
    return out


def get_model(stem: str) -> dict:
    """按 stem 载入 ``models/<stem>.pt`` 并缓存（多个会话共用一份权重）。"""
    with _MODELS_LOCK:
        if stem in _MODELS:
            return _MODELS[stem]
        path = MODEL_DIR / f"{stem}.pt"
        if not path.exists():
            raise FileNotFoundError(f"找不到模型 {path}")
        ck = torch.load(path, map_location="cpu", weights_only=False)
        nc = ck.get("net") or {}
        cfg = NetConfig(int(nc.get("channels", 48)), int(nc.get("blocks", 2)),
                        int(nc.get("hidden", 256)))
        net = build_model(cfg)
        net.load_state_dict(ck["state_dict"])
        net.eval()
        entry = {
            "net": net,
            "meta": {"epoch": ck.get("epoch"), "val_loss": ck.get("val_loss"),
                     "iter": ck.get("iter"), "eval": ck.get("eval"),
                     "net": {"channels": cfg.channels, "blocks": cfg.blocks,
                             "hidden": cfg.hidden}},
            "path": str(path),
        }
        _MODELS[stem] = entry
        print(f"[model] {path.name} → channels={cfg.channels} blocks={cfg.blocks} "
              f"hidden={cfg.hidden} epoch={entry['meta'].get('epoch')} "
              f"iter={entry['meta'].get('iter')} val_loss={entry['meta'].get('val_loss')}",
              flush=True)
        return entry


# ---------------------------------------------------------------------------
# 对手工厂
# ---------------------------------------------------------------------------
OPPONENT_KINDS = ("model", "random", "teacher")


def make_opponent(spec: dict, seed: int) -> tuple[Bot, str]:
    """由一份前端配置造出一个对手 Bot，返回 (bot, 展示名)。"""
    kind = spec.get("type", "random")
    if kind == "model":
        stem = str(spec.get("model") or "rl")
        entry = get_model(stem)
        temp = float(spec.get("temperature", 0.0))
        name = f"模型·{stem}" if temp <= 1e-6 else f"模型·{stem}(T{temp:g})"
        return NetAgent(entry["net"], seed=seed, temperature=temp, name=name), name
    if kind == "teacher":
        style = spec.get("style") or "balanced"
        if style not in TEACHER_STYLES:
            style = "balanced"
        cn = {"balanced": "均衡", "aggressive": "激进", "defensive": "保守",
              "chicken_lover": "爱鸡", "tenpai_rush": "抢听", "gambler": "赌徒"}
        name = f"教师·{cn.get(style, style)}"
        return HeuristicBot(seed=seed, weights=TEACHER_STYLES[style]), name
    name = "随机"
    return RandomBot(seed=seed), name


# ---------------------------------------------------------------------------
# 人类玩家：阻塞式 Bot
# ---------------------------------------------------------------------------
class HumanBot(Bot):
    """把游戏线程**停住**，直到前端提交了动作。"""

    name = HUMAN_SEAT_NAME

    def __init__(self, session: "Session"):
        self.session = session

    def choose(self, game) -> Action:
        return self.session.wait_human(game)


# ---------------------------------------------------------------------------
# 会话
# ---------------------------------------------------------------------------
class Session:
    """一局接一局的会话：所有状态都在这里，HTTP 线程只读快照字典。"""

    def __init__(self, sid: str, specs: list[dict], my_seat: int | None,
                 seed: int, speed: float, auto: bool = False):
        self.sid = sid
        self.specs = specs
        self.seed = seed
        self.speed = speed
        self.auto = auto
        self.rng = random.Random(seed)

        self.lock = threading.RLock()
        self.act_event = threading.Event()
        self.next_event = threading.Event()
        self.chosen: Action | None = None
        self.legal_now: list[Action] = []
        self.game: ZhuojiGame | None = None

        self.names = [f"座位{i}" for i in range(4)]
        self.kinds = ["random"] * 4
        self.cum = [0.0] * 4
        self.hand_index = 0
        self.dealer = 0
        self.dealer_streak = 0
        self.my_seat = my_seat if my_seat is not None else self.rng.randrange(4)
        self.stop = False
        self.error: str | None = None

        self.events: deque = deque(maxlen=600)
        self.state: dict = {"ready": False}
        #: 每局结束时回调 ``(session, game)``；自检用它收集结果
        self.on_hand_end = None

        self.thread = threading.Thread(target=self._run, daemon=True,
                                       name=f"game-{sid[:6]}")
        self.thread.start()

    # -- 名字 / 座位 --------------------------------------------------------
    def _setup_seats(self, game: ZhuojiGame) -> dict:
        """给这一局的四个座位配好身份；人类固定坐在 self.my_seat。"""
        bots: dict[int, Bot] = {}
        for p in range(4):
            if p == self.my_seat:
                bots[p] = HumanBot(self)
                self.names[p] = HUMAN_SEAT_NAME
                self.kinds[p] = "human"
            else:
                spec = self.specs[(p - self.my_seat) % 4 - 1] if len(self.specs) >= 3 else {}
                seed = self.seed * 7919 + self.hand_index * 131 + p
                bot, name = make_opponent(spec if isinstance(spec, dict) else {}, seed)
                bots[p] = bot
                self.names[p] = name
                self.kinds[p] = spec.get("type", "random") if isinstance(spec, dict) else "random"
        return bots

    # -- 主循环 ------------------------------------------------------------
    def _run(self) -> None:
        try:
            while not self.stop:
                self._play_hand()
                if self.stop:
                    break
                if self.auto:          # 自检模式：一局接一局，不等"下一局"信号
                    continue
                # 等前端点「下一局」（1 小时后自动放行，避免线程永久挂住）
                self.next_event.clear()
                self.next_event.wait(timeout=3600)
        except Exception:                                  # pragma: no cover
            self.error = traceback.format_exc()
            with self.lock:
                self.state = {"ready": True, "error": self.error}

    def _play_hand(self) -> None:
        cfg = RulesConfig()
        game = ZhuojiGame(cfg, seed=self.seed + self.hand_index * 7919,
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
                # 兜底：正常一局 60~120 步，超过就强制收尾，别把会话挂死
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
                if sum(game.hands[p]) > before[p]:
                    self._emit_draw(p, game)
            if game.phase == Phase.OVER:
                self._emit_result(game)
            self._publish(game)
            if not self.auto and self.speed > 0:
                time.sleep(self.speed)

        self._settle(game)

    def _settle(self, game: ZhuojiGame) -> None:
        res = game.result or {}
        deltas = res.get("deltas") or [0.0] * 4
        with self.lock:
            for p in range(4):
                self.cum[p] += float(deltas[p])
            if res.get("type") == "win" and res.get("winner") == self.dealer:
                self.dealer_streak += 1
            else:
                self.dealer_streak = 0
            self.dealer = int(res.get("next_dealer", game.dealer))
        self._publish(game)
        if self.on_hand_end is not None:      # 供自检收集每局结果
            try:
                self.on_hand_end(self, game)
            except Exception:                 # pragma: no cover
                pass

    # -- 前端交互 ----------------------------------------------------------
    def wait_human(self, game: ZhuojiGame) -> Action:
        acts = game.legal_actions()
        if not acts:
            return Action(PASS)
        # 只有一个"过"可选时直接过：没必要让人点一下
        if len(acts) == 1 and acts[0].kind == PASS:
            return acts[0]
        if self.auto:                       # 自检模式：随机挑一个合法动作
            hu = [a for a in acts if a.kind == HU]
            return hu[0] if hu and self.rng.random() < 0.9 else self.rng.choice(acts)

        with self.lock:
            self.act_event.clear()          # 先清，再发布 pending，避免竞态
            self.legal_now = list(acts)
            self.chosen = None
            self.state["pending"] = self._pending_payload(game, acts)
            self.state["turn"] = self.my_seat
            self.game = game

        got = self.act_event.wait(timeout=900)
        with self.lock:
            if got and self.chosen is not None and self.chosen in self.legal_now:
                act = self.chosen
            else:                            # 超时或异常：退回"过"，别把牌局卡死
                act = next((a for a in acts if a.kind == PASS), acts[0])
            self.chosen = None
            self.legal_now = []
            self.game = None
            self.state["pending"] = None
        return act

    def submit(self, kind: str, tile: int) -> dict:
        """前端提交动作。只接受**当前合法动作列表里**的对象，避免构造非法动作。"""
        with self.lock:
            if not self.legal_now:
                return {"ok": False, "error": "当前不是你的决策点"}
            for a in self.legal_now:
                if a.kind == kind and int(a.tile) == int(tile):
                    self.chosen = a
                    self.act_event.set()
                    return {"ok": True, "action": a.label()}
        return {"ok": False, "error": f"非法动作 {kind}:{tile}"}

    def hint(self) -> dict:
        """让模型替你想一步：返回它在当前决策点的 Top-3 动作与概率。"""
        with self.lock:
            game = self.game
            if game is None or self.state.get("pending") is None:
                return {"ok": False, "error": "现在不是你的决策点"}
            if not (MODEL_DIR / "rl.pt").exists():
                return {"ok": False, "error": "没有可用的模型"}
            try:
                entry = get_model("rl")
                agent = NetAgent(entry["net"], seed=0, temperature=0.0, name="hint")
                head, mask, probs, value, _obs = agent.distribution(game)
                k = max(1, min(3, int(mask.sum())))
                top = torch.topk(probs, k)
                items = []
                for idx, pr in zip(top.indices.tolist(), top.values.tolist()):
                    a = index_to_action(head, int(idx), game)
                    items.append({"label": a.label(), "kind": a.kind, "tile": int(a.tile),
                                  "prob": round(float(pr), 4)})
                return {"ok": True, "model": "rl", "head": head,
                        "value": round(float(value), 3), "items": items,
                        "note": "value 是模型对该局面（你的视角）的估值，正数偏有利"}
            except Exception as exc:                       # pragma: no cover
                return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}

    def next_hand(self) -> dict:
        if self.state.get("over"):
            self.next_event.set()
            return {"ok": True}
        return {"ok": False, "error": "当前这局还没结束"}

    # -- 快照 --------------------------------------------------------------
    def _pending_payload(self, game: ZhuojiGame, acts: list[Action]) -> dict:
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
            "can_hint": (MODEL_DIR / "rl.pt").exists(),
        }

    def _snapshot(self, game: ZhuojiGame) -> dict:
        me = self.my_seat
        hand: list[int] = []
        for t in range(NUM_TILE_TYPES):
            hand.extend([t] * game.hands[me][t])

        players = []
        for rel in range(4):
            p = (me + rel) % 4
            players.append({
                "rel": rel,
                "seat": p,
                "name": self.names[p],
                "kind": self.kinds[p],
                "hand_count": int(sum(game.hands[p])),
                "melds": [{"type": int(mt), "label": MELD_LABEL.get(mt, "?"),
                           "tile": int(t), "src": int(s)} for mt, t, s in game.melds[p]],
                "discards": [int(t) for t in game.discards[p]],
                "score": round(float(self.cum[p]), 2),
                "is_dealer": p == game.dealer,
                "has_dou": bool(game.has_dou(p)),
            })

        over = game.phase == Phase.OVER
        return {
            "ready": True,
            "sid": self.sid,
            "over": over,
            "hand_index": self.hand_index,
            "my_seat": me,
            "dealer": int(game.dealer),
            "dealer_streak": int(self.dealer_streak),
            "wall_left": int(game.wall_left),
            "turn": int(game.actor()) if not over else -1,
            "phase": game.phase.value,
            "last": list(game.last_discard) if game.last_discard else None,
            "hand": hand,
            "players": players,
            "pending": self.state.get("pending"),
            "events": list(self.events)[-160:],
            "scores": [round(float(self.cum[p]), 2) for p in range(4)],
            "result": self._result_payload(game, players) if over else None,
            "error": self.error,
        }

    def _who(self, seat: object) -> str:
        try:
            s = int(seat)                       # type: ignore[arg-type]
        except (TypeError, ValueError):
            return "全场"
        return "全场" if s < 0 else self.names[s]

    def _result_payload(self, game: ZhuojiGame, players: list[dict]) -> dict:
        res = game.result or {}
        me = self.my_seat
        out: dict = {
            "type": res.get("type"),
            "deltas": [round(float(x), 2) for x in (res.get("deltas") or [0.0] * 4)],
            "wall_left": res.get("wall_left"),
            "mine": round(float((res.get("deltas") or [0.0] * 4)[me]), 2),
        }
        if res.get("type") == "win":
            w = int(res["winner"])
            lose = res.get("loser")
            void = res.get("void_player")
            out.update({
                "winner": w,
                "winner_rel": (w - me) % 4,
                "winner_name": self.names[w],
                "loser": lose,
                "loser_name": self.names[lose] if lose is not None else None,
                "is_tsumo": bool(res.get("is_tsumo")),
                "rob_kong": bool(res.get("rob_kong")),
                "fan_cn": res.get("fan_cn"),
                "fan": res.get("fan"),
                "total_fan": res.get("total_fan"),
                "dealer_bonus": res.get("dealer_bonus"),
                # 明细统一成"已解析好的展示项"，前端不用再猜数据结构
                "dou": [{"who": self._who(a), "label": str(b),
                         "tile": tile_cn(parse_tile(str(c))), "value": float(d)}
                        for a, b, c, d in (res.get("dou") or [])],
                "chicken": [{"who": self._who(a), "label": str(b),
                             "count": int(c), "value": float(d)}
                            for a, b, c, d in (res.get("chicken") or [])],
                "void_name": self.names[void] if void is not None else None,
                "winner_hand": game.hand_tiles(w),
                "winner_melds": [{"label": MELD_LABEL.get(mt, "?"), "tile": int(t)}
                                 for mt, t, _s in game.melds[w]],
            })
        else:
            tp = [int(p) for p in (res.get("tenpai") or [])]
            out.update({
                "tenpai": tp,
                "tenpai_rel": [(p - me) % 4 for p in tp],
                "tenpai_names": [self.names[p] for p in tp],
                "cha_jiao": [{"who": self._who(a), "label": str(b), "value": float(c)}
                             for a, b, c in (res.get("cha_jiao") or [])],
            })
        return out

    def _publish(self, game: ZhuojiGame) -> None:
        with self.lock:
            snap = self._snapshot(game)
            self.state = snap

    # -- 事件流 ------------------------------------------------------------
    def _push(self, kind: str, rel: int, text: str, tile: int | None = None) -> None:
        with self.lock:
            self.events.append({
                "t": round(time.time() - self._t0, 1),
                "kind": kind, "rel": rel, "text": text, "tile": tile,
            })

    _t0 = time.time()

    def _emit_action(self, seat: int, action: Action, game: ZhuojiGame) -> None:
        rel = (seat - self.my_seat) % 4
        who = self.names[seat]
        k = action.kind
        if k == DISCARD:
            self._push("discard", rel, f"{who} 打出 {tile_cn(action.tile)}", action.tile)
        elif k == HU:
            self._push("hu", rel, f"{who} 胡牌！")
        elif k == PENG:
            self._push("peng", rel, f"{who} 碰 {tile_cn(action.tile)}", action.tile)
        elif k in (ANGANG, BUGANG):
            self._push("kong", rel, f"{who} {action.label()}", action.tile)
        elif k == MINGGANG:
            self._push("kong", rel, f"{who} 点豆 {tile_cn(action.tile)}", action.tile)
        # PASS 不记录：响应阶段的"过"太吵

    def _emit_draw(self, seat: int, game: ZhuojiGame) -> None:
        rel = (seat - self.my_seat) % 4
        who = "你" if seat == self.my_seat else self.names[seat]
        self._push("draw", rel, f"{who} 摸牌")

    def _emit_result(self, game: ZhuojiGame) -> None:
        res = game.result or {}
        if res.get("type") == "win":
            w = int(res["winner"])
            extra = res.get("fan_cn") or ""
            self._push("result", (w - self.my_seat) % 4,
                       f"{self.names[w]} 胡牌 · {extra} · {res.get('total_fan')} 番")
        else:
            self._push("result", 0, "黄庄 · 查叫结算")


SESSIONS: dict[str, Session] = {}
SESS_LOCK = threading.Lock()


def new_session(specs: list[dict], my_seat: int | None, seed: int,
                speed: float, auto: bool = False) -> Session:
    sid = uuid.uuid4().hex[:12]
    s = Session(sid, specs, my_seat, seed, speed, auto=auto)
    with SESS_LOCK:
        if len(SESSIONS) > 24:                     # 简单限流：踢掉最早的一批
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
class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = "ZhuojiWeb/1.0"

    def log_message(self, fmt, *args):                # pragma: no cover
        if VERBOSE:
            sys.stderr.write("[web] " + (fmt % args) + "\n")

    # -- 工具 --------------------------------------------------------------
    def _send(self, code: int, body: bytes, ctype: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        try:
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):   # pragma: no cover
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

    # -- GET ---------------------------------------------------------------
    def do_GET(self):                                  # noqa: N802
        path = self.path.split("?")[0]
        if path in ("/", "/index.html"):
            f = STATIC / "index.html"
            if not f.exists():
                return self._send(500, b"index.html missing", "text/plain")
            return self._send(200, f.read_bytes(), "text/html; charset=utf-8")
        if path == "/favicon.ico":
            return self._send(204, b"", "image/x-icon")
        if path == "/api/meta":
            return self._json({
                "models": available_models(),
                "teachers": list(TEACHER_STYLES),
                "tiles": [{"id": t, "name": tile_name(t), "cn": tile_cn(t),
                           "suit": t // 9, "rank": t % 9 + 1}
                          for t in range(NUM_TILE_TYPES)],
                "yaoji": YAOJI,
                "rules": "贵阳捉鸡：108 张，不能吃，豆（杠）是点胡通行证，幺鸡恒为鸡",
            })
        if path == "/api/state":
            sid = self._query().get("sid", "")
            s = get_session(sid)
            if not s:
                return self._json({"ok": False, "error": "会话不存在"}, 404)
            with s.lock:
                return self._json(dict(s.state))
        if path == "/api/hint":
            sid = self._query().get("sid", "")
            s = get_session(sid)
            if not s:
                return self._json({"ok": False, "error": "会话不存在"}, 404)
            return self._json(s.hint())
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

    # -- POST --------------------------------------------------------------
    def do_POST(self):                                 # noqa: N802
        path = self.path.split("?")[0]
        data = self._body()

        if path == "/api/new":
            specs = data.get("opponents")
            if not isinstance(specs, list) or len(specs) != 3:
                specs = [{"type": "model", "model": "rl"},
                         {"type": "teacher", "style": "balanced"},
                         {"type": "random"}]
            specs = [s if isinstance(s, dict) else {} for s in specs][:3]
            my_seat = data.get("my_seat")
            my_seat = int(my_seat) if isinstance(my_seat, int) and 0 <= my_seat < 4 else None
            seed = int(data.get("seed") or random.randrange(10 ** 8))
            speed = float(data.get("speed", 0.45))
            speed = min(max(speed, 0.0), 3.0)
            s = new_session(specs, my_seat, seed, speed)
            # 只回 sid / my_seat —— 座位名要等发牌后 _setup_seats 才定，
            # 建会话时给的名字必然是占位值，返回出去只会误导前端
            return self._json({"ok": True, "sid": s.sid, "my_seat": s.my_seat})

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
# 自检：不开浏览器也能把主循环跑通
# ---------------------------------------------------------------------------
def selftest(n_hands: int, specs: list[dict]) -> int:
    """把主循环跑一遍并**做断言**：必须终局、必须零和。

    用 ``on_hand_end`` 回调收集结果，而不是去 poll 快照——自动模式下
    一局只要 0.1s，"结束"这个快照状态存活时间极短，采样必然漏掉。
    """
    print(f"[selftest] 自动对局 {n_hands} 局，对手 {json.dumps(specs, ensure_ascii=False)}",
          flush=True)
    s = Session("selftest", specs, my_seat=0, seed=20260930, speed=0.0, auto=True)
    seen: list[dict] = []

    def on_end(sess, game):
        r = dict(game.result or {})
        r["_names"] = list(sess.names)
        seen.append(r)

    s.on_hand_end = on_end
    t0 = time.time()
    while len(seen) < n_hands and not s.error:
        time.sleep(0.05)
        if time.time() - t0 > 300:
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
        if t == "win":
            n_win += 1
            w = int(r["winner"])
            if r.get("is_tsumo"):
                n_tsumo += 1
                how = "自摸"
            else:
                n_ron += 1
                how = (f"点胡(放炮:{names[r['loser']]})"
                       if r.get("loser") is not None else "点胡")
            desc = (f"{names[w]} 胡 {r.get('fan_cn')} {r.get('total_fan')}番 {how}"
                    f"  牌墙余 {r.get('wall_left')}")
        elif t == "huangzhuang":
            n_huang += 1
            desc = f"黄庄 · 听牌 {[names[p] for p in r.get('tenpai', [])]}"
        else:
            desc = f"!! 异常终局 {t}"
        print(f"  第 {i:>3} 局: {desc:<56} 得分 {[round(x, 1) for x in r['deltas']]}",
              flush=True)

    # ---- 断言 ----
    bad: list[str] = []
    for i, r in enumerate(seen, 1):
        t = r.get("type")
        d = r.get("deltas") or [0.0] * 4
        if t not in ("win", "huangzhuang"):
            bad.append(f"第{i}局 终局类型异常：{t}")
        if abs(sum(d)) > 1e-6:
            bad.append(f"第{i}局 非零和：Σ={sum(d)}")
        if t == "win" and not (0 <= int(r.get("winner", -1)) < 4):
            bad.append(f"第{i}局 胡牌者非法：{r.get('winner')}")

    dt = time.time() - t0
    print(f"\n[selftest] {len(seen)} 局 / {dt:.1f}s（{dt / max(1, len(seen)) * 1000:.0f} ms/局）"
          f"  胡 {n_win}（自摸 {n_tsumo} / 点胡 {n_ron}）  黄庄 {n_huang}", flush=True)
    print(f"[selftest] 最终累计 {s.state.get('scores')}", flush=True)
    if bad:
        print("[selftest] FAIL\n  " + "\n  ".join(bad), flush=True)
        return 1
    print("[selftest] PASS — 全部终局且零和", flush=True)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--threads", type=int, default=4, help="torch 推理线程数")
    ap.add_argument("--selftest", type=int, default=0, help="自动对局 N 局后退出")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    global VERBOSE
    VERBOSE = args.verbose
    torch.set_num_threads(max(1, args.threads))

    specs = [{"type": "model", "model": "rl"},
             {"type": "teacher", "style": "balanced"},
             {"type": "random"}]
    if args.selftest:
        return selftest(args.selftest, specs)

    srv = ThreadingHTTPServer((args.host, args.port), Handler)
    srv.daemon_threads = True
    print(f"贵州捉鸡 · 网页对局  →  http://{args.host}:{args.port}", flush=True)
    print(f"模型目录: {MODEL_DIR}", flush=True)
    for m in available_models():
        print(f"  · {m['stem']}.pt  {m['size_kb']}KB  {m['mtime']}", flush=True)
    print("Ctrl+C 退出", flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\nbye")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
