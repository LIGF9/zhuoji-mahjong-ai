"use strict";
// 独山麻将 · 本地引擎 API 层（离线单文件版）。
// 与 web/dushan_server.py 的 HTTP 端点同形状，但完全本地：
//   · 内部持一个 DushanGame 实例 + 4 个座位 agent（1 人类 + 3 模型）；
//   · 模型座位 = 编码(encodeV3) → 推理(setInferFn 注入) → mask → argmax → index_to_action；
//   · 人类座位在决策点挂起，等 /api/act；
//   · /api/state 直接返回当前快照（前端轮询模型与在线版一致）。
// CommonJS；浏览器端由打包脚本做 __engModule 包装内联（本文件不依赖 Node API）。
//
// 用法（Node 测试）：
//   const { createLocalApi } = require("./local_engine_server.js");
//   const api = createLocalApi();
//   api.setInferFn(obs => ({ [obs.head]: Float32Array }));   // mock 或真 ORT
//   const r = await api.handle("/api/new", { ... });

const { DushanGame, defaultConfig, FANJI_BOTH } = require("./engine/dushan.js");
const { encodeV3 } = require("./engine/encoder_v3.js");
const { NUM_TILE_TYPES, YAOJI, tileCn, tileName } = require("./engine/tiles.js");
const {
  MELD_PONG, MELD_KONG_EXPOSED, MELD_KONG_ADDED, MELD_KONG_CONCEALED,
  meldIsKong,
} = require("./engine/fan.js");

// ---------------------------------------------------------------------------
// 常量
// ---------------------------------------------------------------------------
const HUMAN_SEAT_NAME = "你";
const MODEL_NAME = "大师";
const MODEL_STEM = "c4_it2280";
const MAX_STEPS = 3000;
const HOLD_MAX_MS = 20000;
const BUTTON_ORDER = { hu: 0, angang: 1, bugang: 1, minggang: 1, peng: 2, pass: 3 };
const MELD_LABEL = {
  [MELD_PONG]: "碰", [MELD_KONG_EXPOSED]: "明杠",
  [MELD_KONG_ADDED]: "补杠", [MELD_KONG_CONCEALED]: "暗杠",
};
const FANJI_MODE_CN = { up: "上鸡", down: "下鸡", both: "上下鸡", off: "关闭" };
const RULE_DEFAULTS = {
  end_baoji: true, end_baogang: false,
  huang_baoji: false, huang_baogang: false, huang_baodapai: true,
};

// ---------------------------------------------------------------------------
// 小工具
// ---------------------------------------------------------------------------
function mulberry32(seed) {
  let a = seed >>> 0;
  return function () {
    a |= 0; a = (a + 0x6D2B79F5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
function shuffledDeck(rng) {
  const deck = [];
  for (let t = 0; t < NUM_TILE_TYPES; t++) for (let i = 0; i < 4; i++) deck.push(t);
  for (let i = deck.length - 1; i > 0; i--) {
    const j = Math.floor(rng() * (i + 1));
    const tmp = deck[i]; deck[i] = deck[j]; deck[j] = tmp;
  }
  return deck;
}
function delay(ms) { return new Promise(res => setTimeout(res, Math.max(0, ms))); }
// pyD {__dict__:[[k,v]...]} / setJson {__set__:[...]} → 普通 JS 对象/数组（递归）
function pyVal(x) {
  if (x === null || x === undefined) return x;
  if (Array.isArray(x)) return x.map(pyVal);
  if (typeof x === "object") {
    if (Array.isArray(x.__dict__)) {
      const o = {};
      for (const [k, v] of x.__dict__) o[String(k)] = pyVal(v);
      return o;
    }
    if (Array.isArray(x.__set__)) return x.__set__.slice().sort((a, b) => a - b);
  }
  return x;
}
function fanjiTilesFrom(flip, mode) {
  const s = Math.floor(flip / 9), r = flip % 9 + 1;
  const out = new Set();
  if (mode === "up" || mode === "both") out.add(s * 9 + (r % 9));
  if (mode === "down" || mode === "both") out.add(s * 9 + (((r - 2) % 9) + 9) % 9);
  out.delete(flip);
  return out;
}
function normFanjiCfg(v, allowOff) {
  const codes = ["up", "down", "both", "off"];
  let code = v == null ? "off" : String(v).trim().toLowerCase();
  const cnMap = { "上鸡": "up", "下鸡": "down", "上下鸡": "both", "关闭": "off" };
  if (cnMap[code]) code = cnMap[code];
  if (!codes.includes(code)) code = "off";
  if (code === "off" && !allowOff) code = FANJI_BOTH;
  return code;
}

// 前端 rules 载荷 → DushanConfig（对齐 dushan_server.config_from_rules）
function configFromRules(rules) {
  const r = rules && typeof rules === "object" ? rules : {};
  const sc = r.scores && typeof r.scores === "object" ? r.scores : {};
  const f = (key, def) => {
    const v = parseFloat(sc[key] !== undefined ? sc[key] : def);
    return Number.isFinite(v) && v >= 0 ? v : def;
  };
  const flags = {};
  for (const k of Object.keys(RULE_DEFAULTS))
    flags[k] = r[k] === undefined ? RULE_DEFAULTS[k] : !!r[k];
  const extraJi = [];
  if (Array.isArray(r.extra_ji)) {
    for (const x of r.extra_ji) {
      const t = parseInt(x, 10);
      if (Number.isInteger(t) && t >= 0 && t < NUM_TILE_TYPES && t !== YAOJI
          && !extraJi.includes(t)) extraJi.push(t);
    }
    extraJi.sort((a, b) => a - b);
  }
  const cfg = defaultConfig();
  Object.assign(cfg, {
    jiesuan_fanji: normFanjiCfg(r.jiesuan !== undefined ? r.jiesuan : r.jiesuan_fanji, true),
    kaiju_fanji: normFanjiCfg(r.kaiju !== undefined ? r.kaiju : r.kaiju_fanji, true),
    mantiangji: !!r.mantiangji,
    zi_mo: f("zimo", 3), dian_pao: f("dianpao", 3),
    gang_kai: f("gangkai", 5), gang_pao: f("gangpao", 5),
    da_dui_zi: f("dadui", 5), qing_yi_se: f("qing", 10),
    qi_dui: f("qidui", 10), long_qi_dui: f("long", 20),
    dan_diao: f("dudiao", 10), baojiao_bonus: f("baojiao", 10),
    kong_chickens: f("kong", 3), jin_ji_multiplier: f("jinji", 2),
    ji_hold: f("jihold", 1), ji_discard_plain: f("jiplain", 1),
    ji_chongfeng: f("jicf", 3), ji_heng: f("jiheng", 2), ji_fanji: f("jifanji", 1),
    extra_ji: extraJi,
  }, flags);
  return cfg;
}

// ---------------------------------------------------------------------------
// 模型 agent：编码 → 推理 → mask → argmax → index_to_action
// ---------------------------------------------------------------------------
// 推理契约：inferFn(obs) 返回 { discard: F32Arr, self_kong: F32Arr, respond: F32Arr }
// （obs = encodeV3(game, seat) 的 {tiles, glob, head, mask}；obs.tiles/glob 可被消费方转移）
function indexToAction(head, idx, game) {
  if (head === "respond") {
    // [pass, peng, minggang, hu]
    if (idx === 0) return { kind: "pass", tile: -1 };
    const tile = (game._rob_kong !== null && game._rob_kong !== undefined)
      ? game._rob_kong[1] : game.last_discard[1];
    if (idx === 1) return { kind: "peng", tile };
    if (idx === 2) return { kind: "minggang", tile };
    return { kind: "hu", tile: -1 };
  }
  if (head === "self_kong") {
    if (idx === 27) return { kind: "pass", tile: -1 };
    for (const [kind, t] of game._selfKongOptions(game.current))
      if (t === idx) return { kind, tile: t };
    return { kind: "pass", tile: -1 };
  }
  // discard：idx==27 → 自摸胡
  if (idx === 27) return { kind: "hu", tile: -1 };
  return { kind: "discard", tile: idx };
}

async function modelChoose(game, seat, inferFn) {
  const obs = encodeV3(game, seat);
  const logits = await inferFn({
    tiles: obs.tiles, glob: obs.glob, head: obs.head, mask: obs.mask, seat,
  });
  const l = logits[obs.head];
  if (!l || !l.length) throw new Error("inferFn 未返回 " + obs.head + " 头 logits");
  const n = Math.min(obs.mask.length, l.length);
  let best = -1, bestV = -Infinity;
  for (let i = 0; i < n; i++) {
    if (!obs.mask[i]) continue;
    if (l[i] > bestV) { bestV = l[i]; best = i; }
  }
  if (best < 0) {
    // mask 全空（理论不发生）→ 按合法动作兜底
    const acts = game.legalActions();
    return acts.length ? acts[0] : { kind: "pass", tile: -1 };
  }
  const act = indexToAction(obs.head, best, game);
  // 合法性兜底（模型与规则层有极小概率错位时防止 step 抛异常）
  const legal = game.legalActions();
  const ok = legal.some(a => a.kind === act.kind
    && (a.kind === "hu" || a.kind === "pass" ? true : a.tile === act.tile));
  if (!ok) return legal[0] || { kind: "pass", tile: -1 };
  return act;
}

// ---------------------------------------------------------------------------
// 会话
// ---------------------------------------------------------------------------
let _SID_SEQ = 0;

class LocalSession {
  constructor(api, body) {
    this.api = api;
    this.sid = "local" + String(Date.now() % 100000) + String(++_SID_SEQ);
    const specs = Array.isArray(body.opponents) ? body.opponents : [];
    // 离线版：对手一律是内嵌模型（specs 仅用于速度/兼容形状，模型写死 c4_it2280）
    this.seed = Number.isInteger(body.seed) ? body.seed
      : Math.floor(Math.random() * 1e8);
    this.rng = mulberry32(this.seed);
    this.speed = Math.min(Math.max(parseFloat(body.speed !== undefined ? body.speed : 0.45) || 0, 0), 5);
    this.rules = body.rules && typeof body.rules === "object"
      ? JSON.parse(JSON.stringify(body.rules)) : {};
    const ms = body.my_seat;
    this.my_seat = Number.isInteger(ms) && ms >= 0 && ms < 4 ? ms : Math.floor(this.rng() * 4);
    const names = Array.isArray(body.names) ? body.names : [];
    this.hold = !!body.hold;
    this.holdHand = this.hold ? 1 : 0;
    this.holdUntil = 0;

    this.names = [HUMAN_SEAT_NAME, MODEL_NAME, MODEL_NAME, MODEL_NAME];
    this.kinds = ["human", "model", "model", "model"];
    this.cum = [0, 0, 0, 0];
    this.hand_index = 0;
    this.hint_used = false;
    this.dealer = 0;
    this.dealer_streak = 0;
    this.stop = false;
    this.forceEnd = false;
    this.error = null;

    this.stats = {};
    for (let p = 0; p < 4; p++) {
      this.stats[p] = {
        games: 0, wins: 0, tsumo: 0, ron: 0, deal_in: 0, huang: 0,
        tenpai_huang: 0, baoji_pay: 0, chongfeng: 0, hengji: 0, baoji: 0,
        zeren_pay: 0, score: 0,
      };
    }

    this.events = [];
    this._evMax = 600;
    this._handEv = [];
    this._curGame = null;
    this._dealPacked = [];
    this._evDraws = 0;
    this._evDiscards = 0;
    this._evSeq = 0;
    this._t0 = Date.now();
    this.state = { ready: false };

    this._actResolve = null;
    this._legalNow = [];
    this._nextResolve = null;
    this._runDone = this.run();
  }

  // -- 主循环 --------------------------------------------------------------
  async run() {
    try {
      while (!this.stop) {
        await this.playHand();
        if (this.stop) break;
        await new Promise(res => { this._nextResolve = res; });
      }
    } catch (e) {
      this.error = String(e && e.stack || e);
      this.state = { ready: true, error: this.error };
    }
  }

  _cfg() { return configFromRules(this.rules); }

  async playHand() {
    const cfg = this._cfg();
    const game = new DushanGame({
      config: cfg, wall: shuffledDeck(this.rng),
      dealer: this.dealer, dealer_streak: this.dealer_streak,
    });
    this.hand_index += 1;
    this._evSeq = 0;
    this._handEv = [];
    this._evDraws = 0;
    this._evDiscards = 0;
    this.forceEnd = false;
    this.hint_used = false;
    this._logSeen = 0;
    this._tenpaiSeen = [false, false, false, false];
    this._hjSeen = 0;
    this._dealPacked = [0, 1, 2, 3].map(p => packHand(game.hands[p]));
    if (game.kaiju_flip !== null && game.kaiju_flip !== undefined) {
      const mode = String(cfg.kaiju_fanji || "off");
      const flips = [...fanjiTilesFrom(game.kaiju_flip, mode)].sort((a, b) => a - b);
      this._push("fanji", -1,
        `开局翻鸡（${FANJI_MODE_CN[mode] || mode}）：翻出 ${tileCn(game.kaiju_flip)}`
        + (flips.length ? "，新增鸡牌 " + flips.map(tileCn).join("、") : "，无新增鸡牌"));
    }
    this._curGame = game;
    this.game = game;
    this._publish(game);
    await this._waitHold(game);
    let steps = 0;
    while (game.phase !== "over" && !this.stop && !this.forceEnd) {
      if (this.hold) {
        await this._waitHold(game);
        if (this.stop || this.forceEnd) break;
      }
      const actor = game.actor();
      let action;
      if (actor === this.my_seat) {
        action = await this._waitHuman(game);
        if (this.forceEnd || this.stop) break;
      } else {
        if (this.speed > 0) await delay(this.speed * 1000);
        action = await modelChoose(game, actor, this.api.inferFn);
      }
      const before = [0, 1, 2, 3].map(p => game.hands[p].reduce((a, b) => a + b, 0));
      const roundsBefore = new Map(game._hengji_rounds);
      game.step(action);
      steps += 1;
      if (game.phase !== "over" && steps > MAX_STEPS) {
        game.result = { type: "timeout", deltas: [0, 0, 0, 0], wall_left: game.wallLeft() };
        game.score_delta = [0, 0, 0, 0];
        game.next_dealer = game.dealer;
        game.phase = "over";
      }
      for (const sp of roundsBefore.keys())
        if (!game._hengji_rounds.has(sp)) this._emitHengji(actor, sp, "close");
      this._emitAction(actor, action, game);
      for (const [sp] of game._hengji_rounds)
        if (!roundsBefore.has(sp)) this._emitHengji(actor, sp, "open");
      this._emitProgress(game);
      for (let p = 0; p < 4; p++) {
        const after = game.hands[p].reduce((a, b) => a + b, 0);
        if (after > before[p] && game.phase !== "over") this._emitDraw(p, game);
      }
      if (game.phase === "over") this._emitResult(game);
      this._publish(game);
      if (this.forceEnd && game.phase !== "over") break;
    }
    if (this.forceEnd && game.phase !== "over") {
      game.result = { type: "abort", deltas: [0, 0, 0, 0], wall_left: game.wallLeft() };
      game.score_delta = [0, 0, 0, 0];
      game.next_dealer = game.dealer;
      game.phase = "over";
      this._push("result", -1, "本局已提前结束（不计分）");
      this._publish(game);
    }
    this._settle(game);
    this._curGame = null;
  }

  _waitHold(game) {
    if (this.holdHand !== this.hand_index) this.hold = false;
    if (!this.hold) return Promise.resolve();
    if (!this.holdUntil) this.holdUntil = Date.now() + HOLD_MAX_MS;
    const deadline = this.holdUntil;
    return (async () => {
      while (this.hold && this.holdHand === this.hand_index
             && !this.stop && Date.now() < deadline) await delay(30);
      this.hold = false;
      this.holdUntil = 0;
    })();
  }

  _waitHuman(game) {
    const acts = game.legalActions();
    if (!acts.length) return Promise.resolve({ kind: "pass", tile: -1 });
    if (acts.length === 1 && acts[0].kind === "pass") return Promise.resolve(acts[0]);
    return new Promise(res => {
      this._legalNow = acts;
      this._actResolve = res;
      this.state.pending = this._pendingPayload(game, acts);
      this.state.turn = this.my_seat;
      this._publish(game);
    });
  }

  submit(kind, tile) {
    if (!this._legalNow.length) return { ok: false, error: "当前不是你的决策点" };
    const hit = this._legalNow.find(a => a.kind === kind && a.tile === tile);
    if (!hit) return { ok: false, error: `非法动作 ${kind}:${tile}` };
    const res = this._actResolve;
    this._actResolve = null;
    this._legalNow = [];
    this.state.pending = null;
    if (res) res(hit);
    return { ok: true, action: kind };
  }

  nextHand(rules) {
    if (this._isOver()) {
      if (rules && typeof rules === "object") this.rules = JSON.parse(JSON.stringify(rules));
      if (this._nextResolve) { const r = this._nextResolve; this._nextResolve = null; r(); }
      return { ok: true };
    }
    return { ok: false, error: "当前这局还没结束" };
  }

  setHold(on) {
    if (on) {
      this.hold = true;
      this.holdHand = this.hand_index;
      this.holdUntil = Date.now() + HOLD_MAX_MS;
    } else {
      this.hold = false;
    }
    return { ok: true, hold: this.hold };
  }

  endHandNow() {
    if (this._isOver()) return { ok: false, error: "当前这局已经结束" };
    this.forceEnd = true;
    if (this._actResolve) {
      // 人类正挂在决策点：喂一个兜底动作让主循环走到边界后终止
      const acts = this._legalNow;
      const act = (acts.find(a => a.kind === "pass")) || acts[0] || { kind: "pass", tile: -1 };
      const res = this._actResolve;
      this._actResolve = null;
      this._legalNow = [];
      this.state.pending = null;
      res(act);
    }
    return { ok: true };
  }

  _isOver() {
    return !this.game || this.game.phase === "over";
  }

  // -- 结算 / 历史 ----------------------------------------------------------
  _settle(game) {
    const res = game.result || {};
    if (!res.type) {
      this.dealer = game.dealer;
      this._publish(game);
      return;
    }
    if (res.type === "abort") {
      this.dealer_streak = 0;
      this.dealer = Number.isInteger(res.next_dealer) ? res.next_dealer : game.dealer;
      this._publish(game);
      return;
    }
    const deltas = res.deltas || [0, 0, 0, 0];
    const record = this._makeRecord(game, res);
    this._updateStats(game, res);
    this.api.addHistory(record, this._makeReplay(game));
    for (let p = 0; p < 4; p++) this.cum[p] += deltas[p];
    if (res.type === "win" && res.winner === this.dealer) this.dealer_streak += 1;
    else this.dealer_streak = 0;
    this.dealer = Number.isInteger(res.next_dealer) ? res.next_dealer : game.dealer;
    this._publish(game);
  }

  _makeRecord(game, res) {
    const deltas = res.deltas || [0, 0, 0, 0];
    const rec = {
      time: new Date().toISOString().replace("T", " ").slice(0, 19),
      rid: `${Date.now()}-${this.hand_index}`,
      hand: this.hand_index,
      my_seat: this.my_seat,
      names: this.names.slice(),
      dealer: game.dealer,
      type: res.type,
      deltas: deltas.map(x => Math.round(x * 100) / 100),
      tenpai: (res.tenpai || []).slice(),
      tenpai_names: (res.tenpai || []).map(p => this.names[p]),
      detail: (res.detail || []).map(d => [
        Number.isInteger(d[0]) && d[0] >= 0 && d[0] < 4 ? this.names[d[0]] : String(d[0]),
        String(d[1]),
        d.length > 2 ? String(d[2]) : "",
        d.length > 3 ? d[3] : d[2],
      ]),
      flags: {
        chongfeng: (res.detail || []).some(d => d[1] === "冲锋鸡"),
        ji_discard: (res.detail || []).some(d => ["冲锋鸡", "横鸡", "幺鸡"].includes(d[1])),
        baoji: (res.detail || []).some(d => d[1] === "包鸡"),
        tenpai_bigfan: false,
        win_bigfan: (res.winners_fan || []).some(x =>
          !["", "平胡", "报叫"].includes(String(pyVal(x).fan_cn || ""))),
        huang: res.type !== "win",
        huang_noten: res.type !== "win" && (res.tenpai || []).length < 4,
        hint_used: !!this.hint_used,
      },
    };
    // 终局听牌（流水查验用，简化：只记座位/等待张）
    const waits = {};
    for (const p of rec.tenpai) {
      try {
        const drop = (res.type === "win"
          && (res.winners || [res.winner]).includes(p)
          && Number.isInteger(res.win_tile) && res.win_tile >= 0) ? res.win_tile : null;
        const [tiles] = game.tingInfo(p, drop);
        if (tiles && tiles.length) waits[String(p)] = tiles;
      } catch (e) { /* 忽略 */ }
    }
    rec.tenpai_waits = waits;
    if (res.type === "win") {
      const w = res.winner;
      const winners = res.winners || [w];
      rec.winner = w;
      rec.winner_name = this.names[w];
      rec.winners = winners.slice();
      rec.winner_names = winners.map(x => this.names[x]);
      rec.loser = res.loser;
      rec.loser_name = res.loser !== null && res.loser !== undefined ? this.names[res.loser] : null;
      rec.is_tsumo = !!res.is_tsumo;
      rec.rob_kong = !!res.rob_kong;
      rec.how = res.how;
      rec.void_name = res.void_player !== null && res.void_player !== undefined
        ? this.names[res.void_player] : null;
      rec.fan_cn = res.fan_cn;
      rec.fan = res.fan;
      rec.winner_hand = game.handTiles(w);
    }
    return rec;
  }

  _updateStats(game, res) {
    for (let p = 0; p < 4; p++) this.stats[p].games += 1;
    const deltas = res.deltas || [0, 0, 0, 0];
    for (let p = 0; p < 4; p++) this.stats[p].score += deltas[p];
    if (res.type === "win") {
      const winners = res.winners || [res.winner];
      for (const w of winners) {
        this.stats[w].wins += 1;
        if (res.is_tsumo) this.stats[w].tsumo += 1;
        else this.stats[w].ron += 1;
      }
      if (!res.is_tsumo && res.loser !== null && res.loser !== undefined)
        this.stats[res.loser].deal_in += 1;
      const baojiThis = new Set();
      for (const det of (res.detail || [])) {
        const who = det[0], label = String(det[1]);
        if (!(who >= 0 && who < 4)) continue;
        if (label === "冲锋鸡") this.stats[who].chongfeng += 1;
        if (label === "横鸡") this.stats[who].hengji += 1;
        if (label === "包鸡" && !baojiThis.has(who)) {
          baojiThis.add(who);
          this.stats[who].baoji += 1;
        }
      }
    } else {
      for (let p = 0; p < 4; p++) {
        this.stats[p].huang += 1;
        if ((res.tenpai || []).includes(p)) this.stats[p].tenpai_huang += 1;
      }
      const baojiThis = new Set();
      for (const det of (res.detail || [])) {
        const who = det[0], label = String(det[1]), val = det.length > 3 ? det[3] : 0;
        if ((label === "包鸡" || label === "包大牌面") && who >= 0 && who < 4)
          this.stats[who].baoji_pay += Math.abs(val);
        if (label === "包鸡" && who >= 0 && who < 4 && !baojiThis.has(who)) {
          baojiThis.add(who);
          this.stats[who].baoji += 1;
        }
      }
    }
  }

  _makeReplay(game) {
    const evs = this._handEv.map((ev, i) => [
      i + 1, ev.kind || "", ev.rel | 0, ev.tile === null || ev.tile === undefined ? -1 : ev.tile,
      ev.ctag || "", ev.r | 0, ev.hc || [0, 0, 0, 0], ev.text || "",
      ev.hands || ["", "", "", ""],
    ]);
    const final = this._snapshot(game);
    for (const k of ["events", "counter", "stats", "hand", "last_draw", "pending",
      "my_ting", "error", "sid"]) delete final[k];
    return {
      hand: this.hand_index,
      my_seat: this.my_seat,
      names: this.names.slice(),
      dealer: game.dealer,
      type: (game.result || {}).type,
      deal: this._dealPacked.slice(),
      ji_tiles: [...game.ji_tiles].sort((a, b) => a - b),
      ev: evs,
      final,
    };
  }

  // -- 快照 -----------------------------------------------------------------
  _pendingPayload(game, acts) {
    let mode;
    if (game.phase === "self_kong")
      mode = (game._rob_kong !== null && game._rob_kong !== undefined) ? "respond" : "self_kong";
    else if (game.phase === "discard") mode = "discard";
    else mode = "respond";
    const buttons = [], discardables = [];
    for (const a of acts) {
      if (a.kind === "discard") discardables.push(a.tile);
      else buttons.push({ kind: a.kind, tile: a.tile, label: actionLabelCn(a) });
    }
    buttons.sort((x, y) => (BUTTON_ORDER[x.kind] || 9) - (BUTTON_ORDER[y.kind] || 9));
    return {
      mode, buttons, discardables,
      last: game.last_discard ? game.last_discard.slice() : null,
      can_hint: true,
    };
  }

  _freshDraw(game, me, over) {
    if (over) return null;
    const ld = game.last_draw;
    if (ld === null || ld === undefined || !game.drawn) return null;
    if (game.phase !== "discard" && game.phase !== "self_kong") return null;
    if (game.actor() !== me) return null;
    if (!(ld >= 0 && ld < NUM_TILE_TYPES)) return null;
    if (game.hands[me][ld] <= 0) return null;
    let s = 0;
    for (const x of game.hands[me]) s += x;
    if (s % 3 !== 2) return null;
    return ld;
  }

  _snapshot(game) {
    const me = this.my_seat;
    const hand = [];
    for (let t = 0; t < NUM_TILE_TYPES; t++)
      for (let i = 0; i < game.hands[me][t]; i++) hand.push(t);
    const over = game.phase === "over";
    const players = [];
    for (let rel = 0; rel < 4; rel++) {
      const p = (me + rel) % 4;
      const handTiles = [];
      for (let t = 0; t < NUM_TILE_TYPES; t++)
        for (let i = 0; i < game.hands[p][t]; i++) handTiles.push(t);
      const tingTiles = [], tingTypes = [];
      if (over) {
        try {
          const r0 = game.result || {};
          let drop = null;
          if (r0.type === "win") {
            const wset = r0.winners || [r0.winner];
            if (wset.includes(p) && Number.isInteger(r0.win_tile) && r0.win_tile >= 0)
              drop = r0.win_tile;
          }
          const [tt, tps] = game.tingInfo(p, drop);
          tingTiles.push(...tt);
          tingTypes.push(...tps);
        } catch (e) { /* 忽略 */ }
      }
      players.push({
        rel, seat: p, name: this.names[p], kind: this.kinds[p],
        hand_count: handTiles.length,
        hand_tiles: handTiles,
        ting_tiles: tingTiles, ting_types: tingTypes,
        melds: game.melds[p].map(([mt, t, s]) => ({
          type: mt, label: MELD_LABEL[mt] || "?", tile: t, src: s,
          ji: meldJiTag(game, p, t),
        })),
        discards: (game.discard_view && game.discard_view[p].length
          ? game.discard_view[p] : game.discards[p]).slice(),
        discard_claimed: (game.discard_claimed_view[p] || []).slice(),
        discard_tags: discardTags(game, p),
        score: Math.round(this.cum[p] * 100) / 100,
        is_dealer: p === game.dealer,
        baojiao: !!game.baojiao[p],
        has_gang: game.hasDou(p),
        chongfeng_done: !!game.first_discard_done[p],
      });
    }
    const seenCnt = new Array(NUM_TILE_TYPES).fill(0);
    for (let p = 0; p < 4; p++) {
      for (const t of game.discards[p]) seenCnt[t] += 1;
      for (const [mt, t] of game.melds[p]) seenCnt[t] += meldIsKong(mt) ? 4 : 3;
    }
    for (let t = 0; t < NUM_TILE_TYPES; t++) seenCnt[t] += game.hands[me][t];
    const counter = seenCnt.map(c => Math.max(0, 4 - c));
    let myTing = [];
    const pend0 = this.state.pending;
    if (!over && pend0 && pend0.mode === "discard") {
      try { myTing = game.tingInfo(me)[0].slice().sort((a, b) => a - b); } catch (e) { /* */ }
    }
    return {
      ready: true,
      sid: this.sid,
      over,
      hand_index: this.hand_index,
      my_seat: me,
      names: this.names.slice(),
      dealer: game.dealer,
      dealer_streak: this.dealer_streak,
      wall_left: game.wallLeft(),
      turn: over ? -1 : game.actor(),
      phase: game.phase,
      last: game.last_discard ? game.last_discard.slice() : null,
      repao: !!game.repao_discard,
      hengji_active: game.hengjiActive,
      hengji_opened: !!game.hengji_opened,
      hengji_species: game.hengjiSpecies(),
      ji_tiles: [...game.ji_tiles].sort((a, b) => a - b),
      kaiju_flip: game.kaiju_flip === undefined ? null : game.kaiju_flip,
      hand,
      last_draw: this._freshDraw(game, me, over),
      counter,
      my_ting: myTing,
      players,
      pending: this.state.pending || null,
      events: this.events.slice(-160),
      scores: this.cum.map(x => Math.round(x * 100) / 100),
      stats: Object.fromEntries([0, 1, 2, 3].map(p => [String(p), this.stats[p]])),
      result: over ? this._resultPayload(game, players) : null,
      error: this.error,
    };
  }

  _resultPayload(game, players) {
    const res = pyVal(game.result) || {};
    const me = this.my_seat;
    const out = {
      type: res.type,
      deltas: (res.deltas || [0, 0, 0, 0]).map(x => Math.round(x * 100) / 100),
      wall_left: res.wall_left,
      mine: Math.round((res.deltas || [0, 0, 0, 0])[me] * 100) / 100,
    };
    const detail = res.detail || [];
    out.detail = detail.map(d => ({
      who: whoName(d[0], this.names), seat: Number.isInteger(d[0]) ? d[0] : -1,
      label: String(d[1]), info: d.length > 2 ? String(d[2]) : "",
      value: d.length > 3 ? Number(d[3]) || 0 : 0,
    }));
    if (res.pair) {
      out.pair = res.pair.map(row => row.map(x => Math.round(x * 100) / 100));
      out.pair_detail = res.pair_detail || [];
    }
    const fanji = detail.find(d => String(d[1]) === "翻鸡");
    out.fanji = fanji ? fanji[2] : null;
    if (res.ji_hold_detail !== null && res.ji_hold_detail !== undefined) {
      out.ji_hold_detail = res.ji_hold_detail || {};
      out.fanji_detail = res.fanji_detail || {};
    }
    if (res.ji_discard_detail !== null && res.ji_discard_detail !== undefined) {
      out.ji_discard_detail = res.ji_discard_detail || {};
      out.ji_capture_detail = res.ji_capture_detail || {};
    }
    if (res.ji_claim_detail !== null && res.ji_claim_detail !== undefined)
      out.ji_claim_detail = res.ji_claim_detail || [];
    out.gold_tiles = (res.gold_tiles || []).slice().sort((a, b) => a - b);
    out.gold_mult = Number(game.cfg.jin_ji_multiplier) || 2;
    out.ji_tiles = [...game.ji_tiles].sort((a, b) => a - b);
    out.kaiju_flip = game.kaiju_flip === undefined ? null : game.kaiju_flip;
    out.jiesuan_fanji = String(game.cfg.jiesuan_fanji);
    out.kaiju_fanji = String(game.cfg.kaiju_fanji);
    if (res.type === "win") {
      const w = res.winner;
      const winners = res.winners || [w];
      const lose = res.loser;
      const voidP = res.void_player;
      const meldsOf = seat => game.melds[seat].map(([mt, t, s]) => ({
        type: mt, label: MELD_LABEL[mt] || "?", tile: t, src: s,
        ji: meldJiTag(game, seat, t),
      }));
      out.fanji_flip = res.fanji_flip;
      out.fanji_tiles = res.fanji_tiles || [];
      out.fanji_gold = !!res.fanji_gold;
      out.how = res.how;
      out.how_value = res.how_value;
      out.type_value = res.type_value;
      out.win_tile = res.win_tile;
      out.winner = w;
      out.winner_rel = (w - me) % 4;
      out.winner_name = this.names[w];
      out.winners = winners.slice();
      out.winners_rel = winners.map(x => (x - me) % 4);
      out.winner_names = winners.map(x => this.names[x]);
      out.winners_fan = (res.winners_fan || []).map(x => ({
        seat: x.seat, rel: (x.seat - me) % 4, name: this.names[x.seat],
        fan_cn: x.fan_cn, fan: Number(x.fan) || 0,
      }));
      out.loser = lose;
      out.loser_name = lose !== null && lose !== undefined ? this.names[lose] : null;
      out.is_tsumo = !!res.is_tsumo;
      out.rob_kong = !!res.rob_kong;
      out.fan_cn = res.fan_cn;
      out.fan = res.fan;
      out.void_name = voidP !== null && voidP !== undefined ? this.names[voidP] : null;
      out.winner_hand = game.handTiles(w);
      out.winner_hands = Object.fromEntries(winners.map(x => [String(x), game.handTiles(x)]));
      out.winner_melds = meldsOf(w);
      out.winner_melds_map = Object.fromEntries(winners.map(x => [String(x), meldsOf(x)]));
    } else {
      const tp = (res.tenpai || []).slice();
      out.tenpai = tp;
      out.tenpai_rel = tp.map(p => (p - me) % 4);
      out.tenpai_names = tp.map(p => this.names[p]);
      out.baodapai = (res.baodapai || []).map(b => ({
        seat: b.seat, rel: (b.seat - me) % 4, name: this.names[b.seat],
        type: b.type, type_value: Number(b.type_value) || 0, pay: Number(b.pay) || 0,
      }));
      out.baodapai_on = !!res.baodapai_on;
      out.zi_mo = Number(game.cfg.zi_mo) || 0;
      out.chickens = (res.chickens || [0, 0, 0, 0]).map(x => Math.round(x * 100) / 100);
    }
    return out;
  }

  _publish(game) {
    this.state = this._snapshot(game);
  }

  // -- 事件流 ---------------------------------------------------------------
  _push(kind, rel, text, tile, ctag) {
    this._evSeq += 1;
    const ev = {
      t: Math.round((Date.now() - this._t0) / 100) / 10,
      hand: this.hand_index, seq: this._evSeq,
      kind, rel, text, tile: tile === undefined ? null : tile,
      ctag: ctag || "",
    };
    this.events.push(ev);
    while (this.events.length > this._evMax) this.events.shift();
    // 回放副本
    if (kind === "draw") this._evDraws += 1;
    if (kind === "discard") this._evDiscards += 1;
    const rev = Object.assign({}, ev);
    rev.r = Math.floor(this._evDiscards / 4) + 1;
    const g = this._curGame;
    if (g) {
      rev.hc = [0, 1, 2, 3].map(p => g.hands[p].reduce((a, b) => a + b, 0));
      rev.hands = [0, 1, 2, 3].map(p => packHand(g.hands[p]));
    }
    this._handEv.push(rev);
  }

  _emitAction(seat, action, game) {
    const rel = (seat - this.my_seat) % 4;
    const who = this.names[seat];
    const k = action.kind;
    if (k === "discard") {
      const tag = game.repao_discard ? "（热炮）" : "";
      let ctag = "", ji = "";
      if (game.ji_tiles.has(action.tile)) {
        ctag = game._ji_tag_of_last[seat];
        ji = `（${ctag === "幺鸡" ? "普通鸡" : ctag}）`;
      }
      this._push("discard", rel, `${who} 打出 ${tileCn(action.tile)}${ji}${tag}`,
        action.tile, ctag);
    } else if (k === "hu") {
      this._push("hu", rel, `${who} 胡牌！`);
    } else if (k === "peng") {
      this._push("peng", rel, `${who} 碰 ${tileCn(action.tile)}`, action.tile);
    } else if (k === "angang" || k === "bugang" || k === "minggang") {
      this._push("kong", rel, `${who} ${actionLabelCn(action)}（得通行证）`, action.tile);
    }
  }

  _emitHengji(seat, species, what) {
    const cn = tileCn(species);
    if (what === "open")
      this._push("hengji", (seat - this.my_seat) % 4,
        `${this.names[seat]} 开出横鸡轮 · ${cn}`, species, "横鸡轮开始");
    else
      this._push("hengji", (seat - this.my_seat) % 4,
        `${this.names[seat]} 再次出牌 · 横鸡轮结束（${cn}）`, species, "横鸡轮结束");
  }

  _emitProgress(game) {
    const log = game.log || [];
    if (this._logSeen > log.length) this._logSeen = 0;
    for (let i = this._logSeen; i < log.length; i++) {
      const [seat, txt] = log[i];
      if (txt === "报叫")
        this._push("baojiao", (seat - this.my_seat) % 4,
          `${this.names[seat]} 报叫！`, null, "报叫");
    }
    this._logSeen = log.length;
    const ever = game.ever_tenpai;
    if (!ever) return;
    for (let p = 0; p < 4; p++) {
      if (this._tenpaiSeen[p] || !ever[p]) continue;
      this._tenpaiSeen[p] = true;
      if (game.baojiao[p]) continue;
      this._push("ting", (p - this.my_seat) % 4, `${this.names[p]} 听牌`);
    }
  }

  _emitDraw(seat, game) {
    const rel = (seat - this.my_seat) % 4;
    const who = this.names[seat];
    const tile = game ? game.last_draw : null;
    if (tile !== null && tile !== undefined)
      this._push("draw", rel, `${who} 摸进 ${tileCn(tile)}`, tile);
    else
      this._push("draw", rel, `${who} 摸牌`);
  }

  _emitResult(game) {
    const res = game.result || {};
    const deltas = res.deltas || [0, 0, 0, 0];
    const net = seat => {
      const v = Number(deltas[seat]) || 0;
      return Math.abs(v - Math.round(v)) < 1e-6
        ? (v >= 0 ? "+" : "") + Math.round(v)
        : (v >= 0 ? "+" : "") + v.toFixed(2);
    };
    if (res.type === "win") {
      const ws = res.winners || [res.winner];
      const how = res.how || (res.is_tsumo ? "自摸" : "点炮");
      const loser = res.loser;
      if (ws.length > 1) {
        const body = ws.map(w => `${this.names[w]} ${net(w)}`).join("、");
        const loss = res.is_tsumo ? "" :
          (loser !== null && loser !== undefined ? `（${this.names[loser]} 放炮）` : "");
        this._push("result", -1, `一炮多响 · ${how}${loss}：${body}`);
      } else {
        const w = ws[0];
        let detailTxt = "";
        if (!res.is_tsumo && loser !== null && loser !== undefined)
          detailTxt = res.rob_kong ? `（${this.names[loser]} 被抢杠）`
            : `（${this.names[loser]} 放炮）`;
        this._push("result", -1,
          `${this.names[w]} 胡 ${res.fan_cn} · ${how}${detailTxt} · ${net(w)} 分`);
      }
    } else {
      const tenpai = (res.tenpai || []).slice();
      const noten = [0, 1, 2, 3].filter(q => !tenpai.includes(q));
      let txt;
      if (!tenpai.length) txt = "黄庄（四家未听牌）· 不结算";
      else if (!noten.length) txt = "黄庄（四家听牌）· 不结算";
      else {
        let head = "黄庄 · 未听者包大牌面";
        if (res.baodapai_on) {
          const parts = (res.baodapai || []).map(b =>
            `${this.names[b.seat]} ${b.type || "平胡"} 每家赔 ${b.pay}`).join("、");
          head = `${head}（${parts || "无人听牌"}）`;
        } else head = `${head}（已关闭）`;
        txt = head + " · 未听 " + noten.map(q => this.names[q]).join("、")
          + " " + noten.map(net).join("、") + " 分";
      }
      this._push("result", -1, txt);
    }
    if (res.type === "win" && res.fanji_flip !== null && res.fanji_flip !== undefined) {
      const mode = String(game.cfg.jiesuan_fanji || "both");
      const tiles = res.fanji_tiles || [];
      let ftxt = `结算翻鸡（${FANJI_MODE_CN[mode] || mode}）：翻出 ${tileCn(res.fanji_flip)}`;
      ftxt += tiles.length
        ? ("，翻鸡牌 " + tiles.map(tileCn).join("、") + (res.fanji_gold ? "（金鸡）" : ""))
        : "，无翻鸡牌";
      this._push("fanji", -1, ftxt);
    }
  }
}

// ---------------------------------------------------------------------------
// 独立小函数（不依赖 session）
// ---------------------------------------------------------------------------
function actionLabelCn(a) {
  const cnMap = { discard: "打", hu: "胡", angang: "暗杠", bugang: "补杠",
    peng: "碰", minggang: "点杠", pass: "过" };
  const cn = cnMap[a.kind] || a.kind;
  return a.tile === undefined || a.tile === null || a.tile < 0 ? cn : cn + tileCn(a.tile);
}
function packHand(counts) {
  const chars = "0123456789abcdefghijklmnopqrstuvwxyz";
  let out = "";
  for (let t = 0; t < NUM_TILE_TYPES; t++)
    for (let i = 0; i < counts[t]; i++) out += chars[t];
  return out;
}
function whoName(seat, names) {
  const s = Number(seat);
  if (!Number.isInteger(s) || s < 0) return "全场";
  return names[s] || "全场";
}
function meldJiTag(game, p, tile) {
  if (!game.ji_tiles.has(tile)) return null;
  const v = game._ji_val_by_meld.get(`${p},${tile}`);
  return v || "幺鸡";
}
function discardTags(game, p) {
  const view = game.discard_view && game.discard_view[p].length
    ? game.discard_view[p] : game.discards[p];
  const tags = new Array(view.length).fill(null);
  const ji = game.ji_events.filter(e => e[0] === p);
  let k = 0;
  for (let i = 0; i < view.length; i++) {
    const t = view[i];
    if (game.ji_tiles.has(t) && k < ji.length) { tags[i] = String(ji[k][2]); k += 1; }
  }
  return tags;
}

// ---------------------------------------------------------------------------
// API 层
// ---------------------------------------------------------------------------
class LocalApi {
  constructor(opts) {
    opts = opts || {};
    this.inferFn = opts.inferFn || null;
    this.sessions = new Map();
    this.history = [];      // {record, has_replay}
    this.replays = new Map();
    this._histSeq = 0;
  }

  setInferFn(fn) { this.inferFn = fn; }

  addHistory(record, replay) {
    record._ln = this._histSeq++;
    this.history.push(record);
    while (this.history.length > 500) this.history.shift();
    if (replay) {
      this.replays.set(record.rid, replay);
      while (this.replays.size > 300) {
        const first = this.replays.keys().next().value;
        this.replays.delete(first);
      }
    }
  }

  _sess(body) {
    const sid = String((body && body.sid) || "");
    return this.sessions.get(sid) || null;
  }

  async handle(path, body) {
    const [p, qs] = String(path || "").split("?");
    const q = {};
    if (qs) for (const kv of qs.split("&")) {
      const i = kv.indexOf("=");
      if (i > 0) q[kv.slice(0, i)] = kv.slice(i + 1);
    }
    body = body || {};
    switch (p) {
      case "/api/meta": return this.hMeta();
      case "/api/new": return this.hNew(body);
      case "/api/state": {
        const s = this.sessions.get(String(q.sid || ""));
        if (!s) return { ok: false, error: "会话不存在" };
        return s.state;
      }
      case "/api/act": {
        const s = this._sess(body);
        if (!s) return { ok: false, error: "会话不存在" };
        return s.submit(String(body.kind || ""), parseInt(body.tile, 10) || 0);
      }
      case "/api/next": {
        const s = this._sess(body);
        if (!s) return { ok: false, error: "会话不存在" };
        return s.nextHand(body.rules);
      }
      case "/api/end_hand": {
        const s = this._sess(body);
        if (!s) return { ok: false, error: "会话不存在" };
        return s.endHandNow();
      }
      case "/api/hold": {
        const s = this._sess(body);
        if (!s) return { ok: false, error: "会话不存在" };
        return s.setHold(!!body.on);
      }
      case "/api/speed": {
        const s = this._sess(body);
        if (!s) return { ok: false, error: "会话不存在" };
        const v = parseFloat(body.speed);
        if (Number.isFinite(v)) s.speed = Math.min(Math.max(v, 0), 5);
        return { ok: true, speed: s.speed };
      }
      case "/api/hint": return this.hHint(this.sessions.get(String(q.sid || "")), q.strategy);
      case "/api/ting": return this.hTing(this.sessions.get(String(q.sid || "")));
      case "/api/history": {
        const n = Math.max(1, parseInt(q.limit, 10) || 30);
        const recs = this.history.slice(-n).reverse().map(r => Object.assign(
          { has_replay: !!(r.rid && this.replays.has(r.rid)) }, r));
        return { ok: true, records: recs };
      }
      case "/api/history_delete": return this.hHistoryDelete(body);
      case "/api/history_export": {
        const fmt = (q.format || "json").toLowerCase();
        return {
          ok: true, format: fmt === "csv" ? "csv" : "json",
          exported_at: new Date().toISOString().replace("T", " ").slice(0, 19),
          count: this.history.length,
          records: this.history.slice(),
        };
      }
      case "/api/replay": {
        const rid = String(q.rid || "");
        const rp = this.replays.get(rid);
        if (!rp) return { ok: false, error: "这一局的回放已不在缓存中" };
        return { ok: true, replay: rp };
      }
      case "/api/stats": return { ok: true, all_time: this.hStats() };
      case "/api/quit": {
        const s = this._sess(body);
        if (s) s.stop = true;
        return { ok: true };
      }
      default:
        return { ok: false, error: "not found: " + p };
    }
  }

  hMeta() {
    const tiles = [];
    for (let t = 0; t < NUM_TILE_TYPES; t++)
      tiles.push({ id: t, name: tileName(t), cn: tileCn(t), suit: Math.floor(t / 9), rank: t % 9 + 1 });
    return {
      page_ver: "",
      models: [],
      teachers: [],
      tiles,
      yaoji: YAOJI,
      rules: "独山麻将（离线单机版）：108 张，不能吃；点胡需通行证（杠/报叫/胡牌张成非平胡）；"
        + "1条恒为鸡，手中/副露鸡每张+1（金鸡时幺鸡×2），打出的鸡也计分"
        + "（冲锋+3/横+2/普通+1，被碰杠走则转移给碰杠者、打出者另付溢价）；"
        + "未听牌者包「已亮明的鸡」；点炮张归胡牌者；一炮多响各家均得牌；"
        + "抢杠（仅补杠）/热炮无视通行证且放炮者鸡分全烧；龙七对须由胡牌张凑成第四张；"
        + "胡后翻鸡上下鸡，金鸡翻倍；黄庄只赔自摸分+最大牌型分。",
      master: MODEL_STEM,
      master_info: {
        stem: MODEL_STEM, origin: MODEL_STEM, line: "", ver: "c4",
        abbr: "C4", size: "", label: `${MODEL_NAME} c4（内嵌 ONNX）`,
      },
    };
  }

  hNew(body) {
    const s = new LocalSession(this, body);
    this.sessions.set(s.sid, s);
    while (this.sessions.size > 8) {
      const first = this.sessions.keys().next().value;
      const old = this.sessions.get(first);
      old.stop = true;
      this.sessions.delete(first);
    }
    return { ok: true, sid: s.sid, my_seat: s.my_seat };
  }

  async hHint(s, strategy) {
    if (!s) return { ok: false, error: "会话不存在" };
    const game = s.game;
    if (!game || !s.state.pending || !s._legalNow.length)
      return { ok: false, error: "现在不是你的决策点" };
    s.hint_used = true;
    if (!this.inferFn) return { ok: false, error: "模型尚未就绪" };
    try {
      const me = s.my_seat;
      const obs = encodeV3(game, me);
      const logits = await this.inferFn(Object.assign({}, obs, { seat: me }));
      const l = logits[obs.head];
      const n = Math.min(obs.mask.length, l.length);
      // 温度=0 贪心；取 mask 内 logits top-3 做展示（softmax 归一仅用于显示）
      const idxs = [];
      for (let i = 0; i < n; i++) if (obs.mask[i]) idxs.push(i);
      if (!idxs.length) return { ok: false, error: "无合法动作" };
      idxs.sort((a, b) => l[b] - l[a]);
      const top = idxs.slice(0, 3);
      const mx = Math.max(...top.map(i => l[i]));
      let sum = 0;
      const exps = top.map(i => { const e = Math.exp(l[i] - mx); sum += e; return e; });
      const items = top.map((i, k) => {
        const a = indexToAction(obs.head, i, game);
        const legal = s._legalNow.find(x =>
          x.kind === a.kind && (x.kind === "hu" || x.kind === "pass" || x.tile === a.tile));
        const act = legal || a;
        return {
          label: actionLabelCn(act), kind: act.kind, tile: act.tile,
          prob: Math.round(exps[k] / sum * 10000) / 10000,
          reason: "", tags: [], lines: [],
        };
      });
      return {
        ok: true, model: MODEL_STEM, head: obs.head,
        value: 0, items,
        situation: {}, tip: "",
        note: "离线单机：value 不可用，prob 为贪心 softmax 显示值",
      };
    } catch (e) {
      return { ok: false, error: String(e && e.message || e) };
    }
  }

  hTing(s) {
    if (!s) return { ok: false, error: "会话不存在" };
    const game = s.game;
    if (!game || !s.state.pending) return { ok: false, error: "现在不是你的决策点" };
    const me = s.my_seat;
    const out = {};
    for (const a of s._legalNow) {
      if (a.kind !== "discard") continue;
      const t = a.tile;
      let tiles, types;
      try { [tiles, types] = game.tingInfo(me, t); } catch (e) { continue; }
      if (!tiles || !tiles.length) continue;
      out[String(t)] = {
        tiles: tiles.map(w => ({ t: w, left: remainingUnseen(game, me, t, w) })),
        types: (types || []).filter(x => x !== "平胡"),
      };
    }
    return { ok: true, map: out };
  }

  hHistoryDelete(body) {
    const asIntList = v => {
      if (v === null || v === undefined) return [];
      const it = Array.isArray(v) ? v : [v];
      return it.map(x => parseInt(x, 10)).filter(Number.isInteger);
    };
    const asStrList = v => {
      if (v === null || v === undefined) return [];
      const it = Array.isArray(v) ? v : [v];
      return it.map(x => String(x).trim()).filter(Boolean);
    };
    let lns = asIntList(body.lns), rids = asStrList(body.rids);
    if (!lns.length && !rids.length) {
      lns = asIntList(body.ln);
      rids = asStrList(body.rid);
    }
    if (!lns.length && !rids.length) return { ok: false, error: "缺少定位参数" };
    let removed = 0, removedReplay = 0;
    const skipped = [];
    const byLn = new Map(this.history.map(r => [r._ln, r]));
    const dropRids = new Set();
    for (let i = 0; i < Math.max(lns.length, rids.length); i++) {
      const ln = i < lns.length ? lns[i] : null;
      const rid = i < rids.length ? rids[i] : null;
      let rec = null;
      if (ln !== null && byLn.has(ln)) {
        rec = byLn.get(ln);
        if (rid && rec.rid && rec.rid !== rid) {
          skipped.push({ ln, rid, why: `行号与记录不匹配（第 ${ln} 行是 ${rec.rid}）` });
          continue;
        }
      } else if (rid) {
        rec = this.history.find(r => r.rid === rid) || null;
        if (!rec) { skipped.push({ ln, rid, why: `记录已不在缓存里：${rid}` }); continue; }
      } else {
        skipped.push({ ln, rid, why: "定位不到" });
        continue;
      }
      dropRids.add(rec.rid);
    }
    if (dropRids.size) {
      const before = this.history.length;
      this.history = this.history.filter(r => !dropRids.has(r.rid));
      removed = before - this.history.length;
      for (const rid of dropRids) if (this.replays.delete(rid)) removedReplay += 1;
    }
    const ok = removed > 0;
    return {
      ok, removed_rec: removed, removed_replay: removedReplay, skipped,
      error: ok ? "" :
        (skipped.map(x => x.why).join("；") || "记录已被删除"),
    };
  }

  hStats() {
    const agg = {
      games: 0, zimo: 0, songhu_games: 0, score: 0,
      chongfeng_games: 0, ji_discard_games: 0, baoji_games: 0,
      tenpai_bigfan_games: 0, win_bigfan_games: 0,
      huang_games: 0, huang_noten_games: 0,
      solo_games: 0, solo_win_games: 0, solo_score: 0,
    };
    for (const r of this.history) {
      if (r.type === "abort") continue;
      const me = r.my_seat || 0;
      const d = r.deltas || [0, 0, 0, 0];
      agg.games += 1;
      agg.score += d[me] || 0;
      const fl = r.flags || {};
      const labels = (r.detail || []).map(x => String(x[1]));
      if (fl.chongfeng || labels.includes("冲锋鸡")) agg.chongfeng_games += 1;
      if (fl.ji_discard || labels.some(x => ["冲锋鸡", "横鸡", "幺鸡"].includes(x)))
        agg.ji_discard_games += 1;
      if (fl.baoji || labels.includes("包鸡")) agg.baoji_games += 1;
      if (fl.tenpai_bigfan) agg.tenpai_bigfan_games += 1;
      const solo = !fl.hint_used;
      const dme = d[me] || 0;
      if (solo) { agg.solo_games += 1; agg.solo_score += dme; }
      if (r.type === "win") {
        if (!r.is_tsumo) agg.songhu_games += 1;
        if (fl.win_bigfan) agg.win_bigfan_games += 1;
        const winners = r.winners || [r.winner];
        if (winners.includes(me)) {
          if (r.is_tsumo) agg.zimo += 1;
          if (solo) agg.solo_win_games += 1;
        }
      } else {
        agg.huang_games += 1;
        if (fl.huang_noten || (r.tenpai || []).length < 4) agg.huang_noten_games += 1;
      }
    }
    return agg;
  }
}

function remainingUnseen(game, me, drop, tile) {
  let seen = 0;
  for (let p = 0; p < 4; p++) {
    for (const t of game.discards[p]) if (t === tile) seen += 1;
    for (const [mt, t] of game.melds[p]) if (t === tile) seen += meldIsKong(mt) ? 4 : 3;
  }
  seen += game.hands[me][tile] - (drop === tile ? 1 : 0);
  return Math.max(0, 4 - seen);
}

module.exports = {
  createLocalApi: opts => new LocalApi(opts),
  LocalApi, LocalSession,
  configFromRules, modelChoose, indexToAction,
  MODEL_STEM,
};
