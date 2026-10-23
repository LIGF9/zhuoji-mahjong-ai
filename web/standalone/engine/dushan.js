"use strict";
// 独山/贵州捉鸡麻将规则引擎（对应 zhuoji/dushan.py，计分按《贵州捉鸡麻将教程》口径）。
// 纯原生 JS、无依赖；内部数值全用 double，Python float 运算顺序照抄。

const {
  NUM_TILE_TYPES, YAOJI, tileCn,
} = require("./tiles");
const {
  PING_HU, DA_DUI_ZI, QI_DUI, LONG_QI_DUI, QING_YI_SE, FAN_CN,
  MELD_PONG, MELD_KONG_EXPOSED, MELD_KONG_ADDED, MELD_KONG_CONCEALED,
  meldIsKong, isSevenPairs, isTenpai, isWinningHand, winningTiles,
  _allTriplets, _isQingYiSe,
} = require("./fan");
const {
  DISCARD, HU, ANGANG, BUGANG, PENG, MINGGANG, PASS, PHASE,
  ZhuojiGame,
} = require("./rules");

// ---------------------------------------------------------------------------
// 独山分值表（单位：鸡）
// ---------------------------------------------------------------------------
const JI_CHONGFENG = 3.0;
const JI_HENG = 2.0;
const JI_YAO = 1.0;
const KONG_CHICKENS = 3.0;
const BAO_KONG = 3.0;
const ZI_MO_JI = 3.0;
const REPAO_JI = 3.0;

const JI_TAG_BY_VAL = { [JI_CHONGFENG]: "冲锋鸡", [JI_HENG]: "横鸡", [JI_YAO]: "幺鸡" };

const DUSHAN_HU_TYPE = {
  [PING_HU]: 0.0,
  [DA_DUI_ZI]: 8.0,
  [QI_DUI]: 13.0,
  [QING_YI_SE]: 13.0,
  dan_diao: 13.0,
  [LONG_QI_DUI]: 26.0,
};

const DUSHAN_FAN_CN = Object.assign({}, FAN_CN, { dan_diao: "单吊" });

const CHICKEN_LABELS = ["冲锋鸡", "横鸡", "幺鸡", "手中鸡", "捉炮鸡", "杠牌", "翻鸡",
  "责任鸡", "包鸡", "包杠", "包大牌面", "自摸", "热炮", "抢杠"];

// ---- 翻鸡模式 ----
const FANJI_UP = "up";
const FANJI_DOWN = "down";
const FANJI_BOTH = "both";
const FANJI_OFF = "off";
const FANJI_CN = { [FANJI_UP]: "上鸡", [FANJI_DOWN]: "下鸡", [FANJI_BOTH]: "上下鸡", [FANJI_OFF]: "关闭" };

function normFanji(v, { allowOff }) {
  let code;
  if (typeof v === "boolean") code = v ? FANJI_BOTH : FANJI_OFF;
  else if (typeof v === "string") {
    const s = v.trim();
    const cnMap = { "上鸡": FANJI_UP, "下鸡": FANJI_DOWN, "上下鸡": FANJI_BOTH, "关闭": FANJI_OFF };
    code = s in cnMap ? cnMap[s] : s.toLowerCase();
  } else if (v === null || v === undefined) code = FANJI_OFF;
  else code = String(v).toLowerCase();
  if (![FANJI_UP, FANJI_DOWN, FANJI_BOTH, FANJI_OFF].includes(code)) code = FANJI_OFF;
  if (code === FANJI_OFF && !allowOff) code = FANJI_BOTH;
  return code;
}

function fanjiTilesFrom(flip, mode) {
  // 按模式给出指示牌 flip 对应的翻鸡牌集合（不含指示牌自身）。
  const s = Math.floor(flip / 9), r = flip % 9 + 1;
  const out = new Set();
  if (mode === FANJI_UP || mode === FANJI_BOTH) out.add(s * 9 + (r % 9));
  if (mode === FANJI_DOWN || mode === FANJI_BOTH) out.add(s * 9 + (((r - 2) % 9) + 9) % 9);
  out.delete(flip);
  return out;
}

// ---------------------------------------------------------------------------
// Python 语义工具
// ---------------------------------------------------------------------------

// dataclasses.asdict(DushanConfig()) 全字段默认值
function defaultConfig() {
  return {
    base_score: 1.0,
    men_dou: 2.0, pa_po_dou: 3.0, dian_dou: 1.0, dian_dou_payer: "konger",
    dou_requires_tenpai: true,
    chicken_scope: "hand_meld", chongfeng_multiplier: 2.0, gold_chicken: 2.0,
    zeren_extra: 1.0, man_tang_chicken: false,
    ping_hu_needs_dou_for_ron: true,
    enable_qiang_gang: true, enable_gang_shang_pao: true, enable_baoting: false,
    enable_pass_lock: true,
    dealer_extra_fan: 1.0, dealer_streak_fan: 1.0, first_dealer: null,
    baojiao_auto: true, baojiao_bonus: 10.0,
    jiesuan_fanji: "both", kaiju_fanji: "off", mantiangji: false,
    jin_ji_multiplier: 2.0, gang_fee: 3.0,
    ji_hold: 1.0, ji_discard_plain: 1.0, ji_chongfeng: 3.0, ji_heng: 2.0,
    ji_fanji: 1.0,
    kong_chickens: 3.0, zi_mo: 3.0, dian_pao: 3.0, gang_kai: 5.0, gang_pao: 5.0,
    da_dui_zi: 5.0, qing_yi_se: 10.0, qi_dui: 10.0, long_qi_dui: 20.0,
    dan_diao: 10.0,
    end_baoji: true, end_baogang: false, huang_baoji: false,
    huang_baogang: false, huang_baodapai: true,
    extra_ji: [],
  };
}

// Python "%g" 格式（默认 6 位有效数字）
function pyG(x) {
  if (Object.is(x, -0)) x = 0;
  if (Number.isInteger(x) && Math.abs(x) < 1e6) return String(x);
  const exp = Math.floor(Math.log10(Math.abs(x)));
  if (exp < -4 || exp >= 6) {
    let s = x.toExponential(5);
    let [m, e] = s.split("e");
    if (m.includes(".")) m = m.replace(/0+$/, "").replace(/\.$/, "");
    const ei = parseInt(e, 10);
    return `${m}e${ei >= 0 ? "+" : "-"}${String(Math.abs(ei)).padStart(2, "0")}`;
  }
  let s = x.toPrecision(6);
  if (s.includes(".")) s = s.replace(/0+$/, "").replace(/\.$/, "");
  return s;
}

// Python round(x, nd)：对 double 的精确十进制值做 ties-to-even 舍入
function pyRound(x, nd) {
  if (!Number.isFinite(x)) return x;
  if (x === 0) return 0;
  const sign = x < 0 ? "-" : "";
  const ax = Math.abs(x);
  const f = new Float64Array(1); f[0] = ax;
  const u = new Uint32Array(f.buffer);
  const bits = (BigInt(u[1]) << 32n) | BigInt(u[0]);
  const rawExp = Number((bits >> 52n) & 0x7ffn);
  let mant = bits & ((1n << 52n) - 1n);
  let e2;
  if (rawExp === 0) e2 = -1074;
  else { mant |= (1n << 52n); e2 = rawExp - 1075; }
  let n10 = 1n;
  for (let i = 0; i < nd; i++) n10 *= 10n;
  let N;
  if (e2 >= 0) {
    N = (mant * n10) << BigInt(e2);
  } else {
    const den = 1n << BigInt(-e2);
    const num = mant * n10;
    const q = num / den, r = num % den;
    const twice = r << 1n;
    if (twice > den || (twice === den && (q & 1n) === 1n)) N = q + 1n;
    else N = q;
  }
  if (nd <= 0) return Number(sign + N.toString());
  const ds = N.toString().padStart(nd + 1, "0");
  return Number(sign + ds.slice(0, -nd) + "." + ds.slice(-nd));
}

function pyD(pairs) { return { __dict__: pairs }; }
function setJson(s) { return { __set__: [...s].sort((a, b) => a - b) }; }
function sortedNums(s) { return [...s].sort((a, b) => a - b); }
function sumArr(a) { let s = 0; for (const x of a) s += x; return s; }

// ---------------------------------------------------------------------------
// DushanGame
// ---------------------------------------------------------------------------
class DushanGame extends ZhuojiGame {
  constructor({ config, wall, dealer, dealer_streak = 0 } = {}) {
    super();
    const cfg = Object.assign(defaultConfig(), config || {});
    cfg.jiesuan_fanji = normFanji(cfg.jiesuan_fanji, { allowOff: true });
    cfg.kaiju_fanji = normFanji(cfg.kaiju_fanji, { allowOff: true });
    if (!Array.isArray(cfg.extra_ji)) cfg.extra_ji = cfg.extra_ji ? [...cfg.extra_ji] : [];
    this.cfg = cfg;
    this.dealer_streak = dealer_streak || 0;
    // 独山状态必须先于基类初始化：_enter_self_phase 会用到报叫锁定
    this._initDushanState();
    this._startBase(wall, dealer);
    // 开局翻鸡：从牌墙尾部翻一张指示牌，只翻开查看，不移出牌墙。
    if (this.cfg.kaiju_fanji !== FANJI_OFF && this.wall.length > 0) {
      const flip = this.wall[this.wall.length - 1];
      this.kaiju_flip = flip;
      for (const t of fanjiTilesFrom(flip, this.cfg.kaiju_fanji))
        this.ji_tiles.add(t);
    }
  }

  // -- 独山状态 -----------------------------------------------------------
  _initDushanState() {
    this.baojiao = [false, false, false, false];
    this.first_discard_done = [false, false, false, false];
    this.hengji_opened = false;
    this.hengji_starter = null;
    this.ji_events = [];                    // [打出者, 基础分, 类型名]
    this.ji_event_tiles = [];               // 与 ji_events 严格同步（金鸡判定用牌面）
    this._hengji_done = new Set();
    // 牌种 -> {start: 开轮者, seats: Set(轮内已吃横鸡的玩家)}
    this._hengji_rounds = new Map();
    this.hengji_events = [];                // [玩家座位, 牌种, "open"/"close"]
    this._ji_tag_of_last = ["幺鸡", "幺鸡", "幺鸡", "幺鸡"];
    // key: `${claimer},${tile}` -> 类型名（插入序与 Python dict 一致）
    this._ji_val_by_meld = new Map();
    this.claimed_ji = [];                   // [打出者, 碰/杠者, 类型名, 是否明杠, 牌]
    this.discard_view = [[], [], [], []];
    this.discard_claimed_view = [[], [], [], []];
    this.captured_ji = [];                  // [胡牌者, 分值, 类型名, 牌]
    this._respond_hu_votes = [];
    this._n_hu = 0;
    this.repao_discard = false;
    this._after_gang_draw = false;
    this._any_discard = false;
    this.ji_tiles = new Set([YAOJI]);
    for (const t0 of this.cfg.extra_ji) {
      const t = parseInt(t0, 10);
      if (t >= 0 && t < NUM_TILE_TYPES && t !== YAOJI) this.ji_tiles.add(t);
    }
    this.kaiju_flip = null;
  }

  // -- 横鸡轮（只读派生） --------------------------------------------------
  get hengjiActive() {
    return this._hengji_rounds.size > 0;
  }

  hengjiSpecies() {
    return [...this._hengji_rounds.keys()].sort((a, b) => a - b);
  }

  jiNameIfDiscarded(p, tile) {
    if (!this.ji_tiles.has(tile)) return "";
    if (!this.first_discard_done[p]) return "冲锋鸡";
    if (this._hengji_done.has(tile)) {
      const rnd = this._hengji_rounds.get(tile);
      return (rnd !== undefined && !rnd.seats.has(p)) ? "横鸡" : "幺鸡";
    }
    return "横鸡";
  }

  _closeHengjiRounds(p) {
    const toClose = [];
    for (const [sp, r] of this._hengji_rounds)
      if (r.seats.has(p)) toClose.push(sp);
    for (const sp of toClose) {
      this._hengji_rounds.delete(sp);
      this.hengji_events.push([p, sp, "close"]);
    }
  }

  // -- 鸡分值换算 ----------------------------------------------------------
  jiValue(tag) {
    const cfg = this.cfg;
    if (tag === "冲锋鸡") return parseFloat(cfg.ji_chongfeng);
    if (tag === "横鸡") return parseFloat(cfg.ji_heng);
    return parseFloat(cfg.ji_discard_plain);
  }

  // -- 报叫锁定：不可碰杠 ---------------------------------------------------
  _canPeng(p, tile) {
    if (this.baojiao[p]) return false;
    return super._canPeng(p, tile);
  }

  _canMinggang(p, tile) {
    if (this.baojiao[p]) return false;
    return super._canMinggang(p, tile);
  }

  _selfKongOptions(p) {
    if (this.baojiao[p]) return [];
    return super._selfKongOptions(p);
  }

  // -- 牌型判定（独山口径，叠加制） -----------------------------------------
  _winTypeSet(hand14, melds, winTile = null) {
    const types = new Set();
    const nMelds = melds.length;
    const seven = nMelds === 0 && isSevenPairs(hand14);
    const normal = isWinningHand(hand14, melds);
    if (!(seven || normal)) return types;
    if (seven) {
      const dragon = winTile === null
        ? hand14.some(x => x === 4)
        : hand14[winTile] === 4;
      types.add(dragon ? LONG_QI_DUI : QI_DUI);
    } else {
      const allTripletMelds = melds.every(m => m[0] !== 0);
      if (nMelds === 4 && allTripletMelds && sumArr(hand14) === 2) {
        types.add("dan_diao");
      } else if (_allTriplets(hand14, melds)) {
        types.add(DA_DUI_ZI);
      }
    }
    if (_isQingYiSe(hand14, melds)) types.add(QING_YI_SE);
    if (types.size === 0) types.add(PING_HU);
    return types;
  }

  _hasPassport(p, hand14 = null, winTile = null) {
    if (this.melds[p].some(m => meldIsKong(m[0]))) return true;
    if (this.baojiao[p]) return true;
    if (hand14 !== null) {
      const s = this._winTypeSet(hand14, this.melds[p], winTile);
      s.delete(PING_HU);
      return s.size > 0;
    }
    for (const t of winningTiles(this.hands[p], this.melds[p])) {
      const h = this.hands[p].slice();
      h[t] += 1;
      const s = this._winTypeSet(h, this.melds[p], t);
      s.delete(PING_HU);
      if (s.size > 0) return true;
    }
    return false;
  }

  canRon(p, tile) {
    if (this.cfg.enable_pass_lock && this.pass_lock[p].has(tile)
        && !this.repao_discard) return false;
    const h = this.hands[p].slice();
    h[tile] += 1;
    if (!isWinningHand(h, this.melds[p])) return false;
    // 抢杠胡：补杠被抢时无视通行证（听该张即可胡）
    const rob = this._rob_kong;
    if (rob !== null && rob !== undefined && rob[1] === tile) return true;
    if (this._hasPassport(p, h, tile)) return true;
    if (this.repao_discard) return true;
    if (this.last_discard && this.baojiao[this.last_discard[0]]) return true;
    return false;
  }

  _canRobKong(p, tile) {
    if (this.cfg.enable_pass_lock && this.pass_lock[p].has(tile)) return false;
    const h = this.hands[p].slice();
    h[tile] += 1;
    return isWinningHand(h, this.melds[p]);
  }

  // -- 抢杠：补杠可被抢，自杠/明杠不可抢 ------------------------------------
  _stepSelfKong(action) {
    if ((action.kind === ANGANG || action.kind === BUGANG)
        && this._rob_kong === null && this.baojiao[this.current]) {
      throw new Error(`非法动作（玩家 ${this.current}）：已报叫，严禁碰杠`);
    }
    if (this._rob_kong === null && action.kind === BUGANG) {
      const p = this.current;
      const t = action.tile;
      if (this.cfg.enable_qiang_gang) {
        const robbers = this._opponents(p).filter(q => this._canRobKong(q, t));
        if (robbers.length > 0) {
          this._rob_kong = [robbers[0], t];
          this._rob_kong_konger = p;
          this.phase = PHASE.SELF_KONG;
          return;
        }
      }
      this._doBugang(p, t);
      return;
    }
    super._stepSelfKong(action);
  }

  // -- 出牌阶段：冲锋鸡 / 横鸡轮 / 热炮 / 自动报叫 ---------------------------
  _stepDiscard(action) {
    const p = this.current;
    if (action.kind === HU) {
      if (!(this.drawn && this.canTsumo(p))) {
        throw new Error(`非法自摸胡（玩家 ${p}）：drawn=${this.drawn}，碰后未摸牌须先出牌`);
      }
      this._settleWin(p, -1, true);
      return;
    }
    const t = action.tile;
    // 热炮 = 杠后第一张弃牌
    this.repao_discard = this.gang_discard[p];
    this.hands[p][t] -= 1;
    this.discards[p].push(t);
    this.discard_view[p].push(t);
    this.discard_claimed_view[p].push(false);
    this._any_discard = true;

    // 横鸡轮收口：开轮者 / 轮内已吃横鸡者再次出牌 → 该种轮结束
    this._closeHengjiRounds(p);

    if (this.ji_tiles.has(t)) {
      let tag;
      if (!this.first_discard_done[p]) {
        tag = "冲锋鸡";
      } else if (this._hengji_done.has(t)) {
        tag = this._hengji_rounds.has(t) ? "横鸡" : "幺鸡";
      } else {
        this._hengji_done.add(t);
        this._hengji_rounds.set(t, { start: p, seats: new Set([p]) });
        this.hengji_events.push([p, t, "open"]);
        if (!this.hengji_opened) {
          this.hengji_opened = true;
          this.hengji_starter = p;
        }
        tag = "横鸡";
      }
      const rnd = this._hengji_rounds.get(t);
      if (rnd !== undefined) rnd.seats.add(p);
      const val = this.jiValue(tag);
      this.ji_events.push([p, val, tag]);
      this.ji_event_tiles.push(t);
      this._ji_tag_of_last[p] = tag;
    }

    // 自动报叫：第一张打出后已听牌
    if (!this.first_discard_done[p]) {
      this.first_discard_done[p] = true;
      if (this.cfg.baojiao_auto && isTenpai(this.hands[p], this.melds[p])) {
        this.baojiao[p] = true;
        this.log.push([p, "报叫"]);
      }
    }

    this._updateTenpai(p);
    this.last_discard_was_gang = this.gang_discard[p];
    this.gang_discard[p] = false;
    this.last_discard = [p, t];
    this._beginRespond(p, t);
  }

  _gangDraw(p) {
    super._gangDraw(p);
    this._after_gang_draw = true;
  }

  _nextPlayer(fromP) {
    this.repao_discard = false;
    this._after_gang_draw = false;
    super._nextPlayer(fromP);
  }

  // -- 响应阶段：一炮多响收集 / 记录被碰鸡牌类型 ------------------------------
  _beginRespond(discarder, tile) {
    const others = this._opponents(discarder);
    const huList = others.filter(q => this.canRon(q, tile));
    const claimList = others.filter(q => !huList.includes(q)
      && (this._canPeng(q, tile) || this._canMinggang(q, tile)));
    const order = huList.concat(claimList);
    this._respond_hu_votes = [];
    this._n_hu = huList.length;
    if (order.length > 0) {
      this._respond_order = order;
      this._respond_idx = 0;
      this.phase = PHASE.RESPOND;
    } else {
      this._nextPlayer(discarder);
    }
  }

  _advanceRespond(discarder, tile) {
    this._respond_idx += 1;
    if (this._respond_hu_votes.length > 0 && this._respond_idx >= this._n_hu) {
      const winners = this._respond_hu_votes.slice();
      this._respond_order = [];
      this._respond_idx = 0;
      this._respond_hu_votes = [];
      this._n_hu = 0;
      this._settleWinMulti(winners, tile, false, discarder, false);
      return;
    }
    if (this._respond_idx >= this._respond_order.length) {
      this._respond_order = [];
      this._respond_idx = 0;
      this._respond_hu_votes = [];
      this._n_hu = 0;
      this._nextPlayer(discarder);
    }
  }

  _stepRespond(action) {
    const p = this._currentResponder();
    const [discarder, tile] = this.last_discard ? this.last_discard : [null, -1];
    if (action.kind === HU) {
      if (!this._respond_hu_votes.includes(p)) this._respond_hu_votes.push(p);
      this._advanceRespond(discarder, tile);
      return;
    }
    if (action.kind === PASS) {
      this.pass_lock[p].add(tile);
      this._advanceRespond(discarder, tile);
      return;
    }
    if ((action.kind === PENG || action.kind === MINGGANG) && this.baojiao[p]) {
      throw new Error(`非法动作（玩家 ${p}）：已报叫，严禁碰杠`);
    }
    const claim = action.kind === PENG || action.kind === MINGGANG;
    const jiTag = discarder !== null ? this._ji_tag_of_last[discarder] : "幺鸡";
    super._stepRespond(action);
    this._respond_hu_votes = [];
    this._n_hu = 0;
    if (claim) {
      if (discarder !== null) {
        const cv = this.discard_claimed_view[discarder];
        if (cv.length > 0) cv[cv.length - 1] = true;
      }
      if (this.ji_tiles.has(tile)) {
        this._ji_val_by_meld.set(`${p},${tile}`, jiTag);
        this.claimed_ji.push([discarder, p, jiTag, action.kind === MINGGANG, tile]);
      }
      this.repao_discard = false;
    }
  }

  // -- 翻鸡（胡牌者翻牌墙顶） ------------------------------------------------
  _dushanChickenState() {
    if (this.cfg.jiesuan_fanji === FANJI_OFF) return [false, new Set()];
    if (this.wallLeft() <= 0) return [false, new Set()];
    const mode = this.cfg.jiesuan_fanji;
    const flip = this.wall[this.wall_pos];
    const fanji = fanjiTilesFrom(flip, mode);
    const gold = this._goldTilesOf(fanji);
    return [gold.size > 0, fanji];
  }

  _goldTilesOf(fanji) {
    if (!fanji || fanji.size === 0) return new Set();
    return new Set([...fanji].filter(t => this.ji_tiles.has(t)));
  }

  // -- 结算（教程口径） ------------------------------------------------------
  _ownJiAfterClaims(gold = null) {
    const goldSet = new Set(gold || []);
    const jin = goldSet.size > 0 ? parseFloat(this.cfg.jin_ji_multiplier) : 1.0;
    const alive = this.ji_events.map(() => true);
    for (const ev of this.claimed_ji) {
      const discarder = ev[0], tag = ev[2];
      for (let i = this.ji_events.length - 1; i >= 0; i--) {
        if (!alive[i] || this.ji_events[i][0] !== discarder
            || this.ji_events[i][2] !== tag) continue;
        alive[i] = false;
        break;
      }
    }
    const vals = [[], [], [], []];
    const tags = [[], [], [], []];
    const tiles = [[], [], [], []];
    this.ji_events.forEach((ev, i) => {
      if (!alive[i]) return;
      const t = i < this.ji_event_tiles.length ? this.ji_event_tiles[i] : -1;
      vals[ev[0]].push(ev[1] * (goldSet.has(t) ? jin : 1.0));
      tags[ev[0]].push(ev[2]);
      tiles[ev[0]].push(t);
    });
    return [vals, tags, tiles];
  }

  _chickenCounts(tenpai, fanji, jin, voidP = null) {
    const cfg = this.cfg;
    const c = [0.0, 0.0, 0.0, 0.0];
    const items = [[], [], [], []];
    const holdBd = [[], [], [], []];
    const fanjiBd = [[], [], [], []];

    const add = (p, val, label, info = "") => {
      if (val <= 0) return;
      c[p] += val;
      items[p].push([label, info, val]);
    };

    const gold = this._goldTilesOf(fanji);
    const fanjiOnly = new Set([...(fanji || [])].filter(t => !this.ji_tiles.has(t)));

    const [ownVals, ownTags] = this._ownJiAfterClaims(gold);

    // 捉炮鸡按牌种扣张数，避免与手中鸡重复
    const capBy = [new Map(), new Map(), new Map(), new Map()];
    for (const cap of this.captured_ji) {
      const w = cap[0], tile = cap[3];
      capBy[w].set(tile, (capBy[w].get(tile) || 0) + 1);
    }

    for (let p = 0; p < 4; p++) {
      if (!tenpai[p]) continue;
      // 1) 打出的鸡
      for (let i = 0; i < ownVals[p].length; i++)
        add(p, ownVals[p][i], ownTags[p][i], "打出");
      // 2) 手中/副露鸡牌
      const hold = parseFloat(cfg.ji_hold);
      let nHand = 0, nVal = 0.0, nGold = 0;
      for (const t of sortedNums(this.ji_tiles)) {
        let nT = this.hands[p][t] + this.meldedCount(p, t) - (capBy[p].get(t) || 0);
        nT = Math.max(0, nT);
        if (nT <= 0) continue;
        const mult = gold.has(t) ? jin : 1.0;
        nHand += nT;
        nVal += nT * hold * mult;
        if (gold.has(t)) nGold += nT;
        holdBd[p].push([t, nT]);
      }
      if (nVal > 0) {
        let info = `${pyG(nHand)} 只`;
        if (nGold) info += `（含金鸡 ${pyG(nGold)} 张 ×${pyG(jin)}）`;
        add(p, nVal, "手中鸡", info);
      }
      // 4) 翻鸡牌（只算非金鸡种）
      if (fanjiOnly.size > 0) {
        let nf = 0;
        for (const ft of fanjiOnly) nf += this.hands[p][ft] + this.meldedCount(p, ft);
        let extraInfo = "";
        if (cfg.mantiangji) {
          let nd = 0;
          for (const ft of fanjiOnly)
            nd += this.discards[p].filter(x => x === ft).length;
          if (nd) {
            nf += nd;
            extraInfo = `（含弃牌 ${pyG(nd)} 张）`;
          }
        }
        if (nf) {
          add(p, nf * parseFloat(cfg.ji_fanji), "翻鸡",
            `共持有 ${pyG(nf)} 张` + extraInfo);
          for (const ft of sortedNums(fanjiOnly)) {
            let nFt = this.hands[p][ft] + this.meldedCount(p, ft);
            if (cfg.mantiangji)
              nFt += this.discards[p].filter(x => x === ft).length;
            if (nFt > 0) fanjiBd[p].push([ft, nFt]);
          }
        }
      }
    }
    // 5) 捉炮鸡
    for (const cap of this.captured_ji) {
      const w = cap[0], val = cap[1], tag = cap[2], tile = cap[3];
      const v = val * (gold.has(tile) ? jin : 1.0);
      add(w, v, "捉炮鸡", `${tag}·${tileCn(tile)}`);
    }
    // 全烧
    if (voidP !== null && voidP !== undefined && voidP >= 0 && voidP < 4) {
      c[voidP] = 0.0;
      items[voidP] = [];
      holdBd[voidP] = [];
      fanjiBd[voidP] = [];
    }
    return [c, items, ownVals, ownTags, holdBd, fanjiBd];
  }

  // -- 牌型/听牌辅助 ---------------------------------------------------------
  _huTypeValue(tp) {
    const cfg = this.cfg;
    const table = {
      [PING_HU]: 0.0,
      [DA_DUI_ZI]: cfg.da_dui_zi,
      [QI_DUI]: cfg.qi_dui,
      [QING_YI_SE]: cfg.qing_yi_se,
      dan_diao: cfg.dan_diao,
      [LONG_QI_DUI]: cfg.long_qi_dui,
    };
    return tp in table ? table[tp] : 0.0;
  }

  _tenpaiFlag(p) {
    if (isTenpai(this.hands[p], this.melds[p])) return true;
    const n = this.melds[p].length;
    if (sumArr(this.hands[p]) === (4 - n) * 3 + 1) return false;
    const h = this.hands[p].map(x => x);
    for (let d = 0; d < NUM_TILE_TYPES; d++) {
      if (h[d] <= 0) continue;
      h[d] -= 1;
      const ok = winningTiles(h, this.melds[p]).length > 0;
      h[d] += 1;
      if (ok) return true;
    }
    return false;
  }

  tingInfo(p, drop = null) {
    const base = this.hands[p].map(x => x);
    if (drop !== null && drop >= 0 && drop < NUM_TILE_TYPES && base[drop] > 0)
      base[drop] -= 1;
    const melds = this.melds[p];
    let tiles = winningTiles(base, melds);
    if (tiles.length === 0 && sumArr(base) !== (4 - melds.length) * 3 + 1) {
      let found = false;
      for (let d = 0; d < NUM_TILE_TYPES; d++) {
        if (base[d] <= 0) continue;
        base[d] -= 1;
        const wt = winningTiles(base, melds);
        if (wt.length > 0) { tiles = wt; found = true; break; }
        base[d] += 1;
      }
      if (!found) return [[], []];
    }
    const types = new Set();
    for (const t of tiles) {
      const h = base.slice();
      h[t] += 1;
      for (const tp of this._winTypeSet(h, melds, t)) types.add(tp);
    }
    let named = [...types].sort()
      .filter(tp => tp !== PING_HU)
      .map(tp => DUSHAN_FAN_CN[tp] !== undefined ? DUSHAN_FAN_CN[tp] : tp);
    if (named.length === 0 && tiles.length > 0) named = ["平胡"];
    return [tiles.slice().sort((a, b) => a - b), named];
  }

  _maxHuType(p) {
    const base = this.hands[p].map(x => x);
    const melds = this.melds[p];
    let tiles = winningTiles(base, melds);
    if (tiles.length === 0 && sumArr(base) !== (4 - melds.length) * 3 + 1) {
      for (let d = 0; d < NUM_TILE_TYPES; d++) {
        if (base[d] <= 0) continue;
        base[d] -= 1;
        const wt = winningTiles(base, melds);
        if (wt.length > 0) { tiles = wt; break; }
        base[d] += 1;
      }
    }
    let bestV = 0.0, bestCn = "";
    for (const t of tiles) {
      const h = base.slice();
      h[t] += 1;
      const types = this._winTypeSet(h, melds, t);
      let v = 0.0;
      for (const tp of types) v += this._huTypeValue(tp);
      if (v > bestV) {
        bestV = v;
        bestCn = [...types].sort()
          .filter(tp => tp !== PING_HU)
          .map(tp => DUSHAN_FAN_CN[tp] !== undefined ? DUSHAN_FAN_CN[tp] : tp)
          .join("+");
      }
    }
    return [bestV, bestCn];
  }

  _claimedJiDetail(gold = null) {
    const cfg = this.cfg;
    const goldSet = new Set(gold || []);
    const jin = goldSet.size > 0 ? parseFloat(cfg.jin_ji_multiplier) : 1.0;
    const out = [];
    for (const ev of this.claimed_ji) {
      const discarder = ev[0], claimer = ev[1], tag = ev[2];
      const tile = ev.length > 4 ? ev[4] : -1;
      let premium = this.jiValue(tag) - parseFloat(cfg.ji_hold);
      const isGold = goldSet.has(tile);
      if (isGold) premium *= jin;
      if (premium > 0) {
        out.push({
          claimer, discarder, tile, tag,
          premium: pyRound(premium, 3), gold: isGold,
        });
      }
    }
    return out;
  }

  _claimedJiRows(gold = null) {
    return this._claimedJiDetail(gold)
      .map(d => [d.claimer, d.discarder, d.premium, d.tag]);
  }

  _transferDiscardedJi(discarder, winners, tile) {
    if (!this.ji_tiles.has(tile)) return;
    let idx = -1;
    for (let i = this.ji_events.length - 1; i >= 0; i--) {
      if (this.ji_events[i][0] === discarder) { idx = i; break; }
    }
    if (idx < 0) return;
    const val = this.ji_events[idx][1], tag = this.ji_events[idx][2];
    this.ji_events.splice(idx, 1);
    if (idx < this.ji_event_tiles.length) this.ji_event_tiles.splice(idx, 1);
    for (const w of winners) this.captured_ji.push([w, val, tag, tile]);
  }

  _settleWin(winner, winTile, isTsumo, loser = null, robKong = false) {
    this._settleWinMulti([winner], winTile, isTsumo, loser, robKong);
  }

  _settleWinMulti(winners0, winTile, isTsumo, loser = null, robKong = false) {
    const cfg = this.cfg;
    const winners = winners0.map(w => w);
    const primary = winners[0];

    // ---- 胡牌张归属 ----
    const wtI = (typeof winTile === "number" && Number.isFinite(winTile) && winTile >= 0)
      ? winTile : -1;
    if (wtI >= 0) {
      for (const w of winners) this.hands[w][wtI] += 1;
      if (!isTsumo && loser !== null && loser !== undefined) {
        if (robKong) {
          if (this.hands[loser][wtI] > 0) this.hands[loser][wtI] -= 1;
        } else {
          const ds = this.discards[loser];
          if (ds.length > 0 && ds[ds.length - 1] === wtI) ds.pop();
          this._transferDiscardedJi(loser, winners, wtI);
        }
      }
    }

    const typesBy = {};
    for (const w of winners)
      typesBy[w] = this._winTypeSet(this.hands[w], this.melds[w],
        wtI >= 0 ? wtI : null);

    const fanCnOf = (w) => {
      const named = [...typesBy[w]].sort()
        .filter(tp => tp !== PING_HU)
        .map(tp => DUSHAN_FAN_CN[tp] !== undefined ? DUSHAN_FAN_CN[tp] : tp)
        .join("+");
      if (named) return named;
      return this.baojiao[w] ? "报叫" : "平胡";
    };

    const [goldBool, fanji] = this._dushanChickenState();
    const goldTiles = this._goldTilesOf(fanji);
    const jin = goldTiles.size > 0 ? cfg.jin_ji_multiplier : 1.0;
    const detail = [];
    const fanjiFlip = this.wallLeft() > 0 ? this.wall[this.wall_pos] : null;
    if (fanji.size > 0) {
      detail.push([-1, "翻鸡",
        tileCn(fanjiFlip) + (goldTiles.size > 0 ? "（金鸡）" : ""), 0.0]);
    }

    const pair = [[0.0, 0.0, 0.0, 0.0],
                  [0.0, 0.0, 0.0, 0.0],
                  [0.0, 0.0, 0.0, 0.0],
                  [0.0, 0.0, 0.0, 0.0]];
    const pairDet = new Map(); // key: `${a},${b}` -> rows

    const pflow = (a, b, val, label, info = "", who = null) => {
      if (val <= 0) return;
      pair[a][b] += val;
      const key = `${a},${b}`;
      const row = [label, info, val, who === null ? a : who];
      const arr = pairDet.get(key);
      if (arr) arr.push(row);
      else pairDet.set(key, [row]);
    };

    // 终局听牌（胡牌者视为叫牌）
    const tenpai = [0, 1, 2, 3].map(p => this._tenpaiFlag(p));
    for (const w of winners) tenpai[w] = true;

    // ---- 全烧 ----
    let voidPlayer = null;
    if (!isTsumo && loser !== null && loser !== undefined
        && (robKong || this.repao_discard)) {
      voidPlayer = loser;
    }

    // ---- 鸡数统计 ----
    const [c, items, ownVals, ownTags, holdBd, fanjiBd] = this._chickenCounts(
      tenpai, fanji, jin, voidPlayer);
    for (let p = 0; p < 4; p++)
      for (const [label, info, val] of items[p])
        detail.push([p, label, info, val]);
    if (voidPlayer !== null)
      detail.push([voidPlayer, "全烧", "鸡分作废", 0.0]);

    // ---- 结算页鸡牌三分类的逐张明细 ----
    const [dvals, dtags, dtiles] = this._ownJiAfterClaims(goldTiles);
    const ddPairs = [[], [], [], []];
    for (let p = 0; p < 4; p++) {
      ddPairs[p] = dvals[p].map((v, i) =>
        pyD([["tile", dtiles[p][i]], ["tag", dtags[p][i]],
             ["val", pyRound(v, 3)]]));
    }
    const capPairs = [[], [], [], []];
    for (const cap of this.captured_ji) {
      const w = cap[0], val = cap[1], tag = cap[2], tile = cap[3];
      const v = val * (goldTiles.has(tile) ? jin : 1.0);
      capPairs[w].push(pyD([["tile", tile], ["tag", tag],
        ["val", pyRound(v, 3)]]));
    }
    if (voidPlayer !== null) {
      ddPairs[voidPlayer] = [];
      capPairs[voidPlayer] = [];
    }
    const discardDetail = pyD(ddPairs.map((arr, p) => [String(p), arr]));
    const captureDetail = pyD(capPairs.map((arr, p) => [String(p), arr]));
    let claimRaw = this._claimedJiDetail(goldTiles);
    if (voidPlayer !== null)
      claimRaw = claimRaw.filter(d => d.claimer !== voidPlayer);
    const claimDetail = claimRaw.map(d =>
      pyD([["claimer", d.claimer], ["discarder", d.discarder],
           ["tile", d.tile], ["tag", d.tag],
           ["premium", d.premium], ["gold", d.gold]]));

    // ---- 胡牌方式 ----
    let how, howVal;
    if (isTsumo) {
      [how, howVal] = this._after_gang_draw
        ? ["杠上开花", cfg.gang_kai] : ["自摸", cfg.zi_mo];
    } else if (robKong) {
      [how, howVal] = ["抢杠", cfg.dian_pao];
    } else if (this.repao_discard) {
      [how, howVal] = ["热炮", cfg.gang_pao];
    } else {
      [how, howVal] = ["点炮", cfg.dian_pao];
    }
    const payers = isTsumo
      ? [0, 1, 2, 3].filter(q => q !== primary)
      : (loser !== null && loser !== undefined ? [loser] : []);
    const howTag = how + (winners.length > 1 ? `（${winners.length}家）` : "");
    for (const w of winners) {
      for (const q of payers)
        if (q !== w) pflow(w, q, howVal, howTag);
      detail.push([w, how, "", howVal]);
    }

    // ---- 牌型奖励与报叫 ----
    const typeTotalBy = {};
    const typeCnBy = {};
    for (const w of winners) {
      const types = typesBy[w];
      let typeTotal = 0.0;
      for (const tp of types) typeTotal += this._huTypeValue(tp);
      const typeCn = [...types].sort()
        .filter(tp => tp !== PING_HU)
        .map(tp => DUSHAN_FAN_CN[tp] !== undefined ? DUSHAN_FAN_CN[tp] : tp)
        .join("+");
      typeTotalBy[w] = typeTotal;
      typeCnBy[w] = typeCn;
      if (typeTotal > 0 && payers.length > 0) {
        for (const q of payers)
          if (q !== w) pflow(w, q, typeTotal, typeCn || "平胡", "牌型");
        detail.push([w, typeCn || "平胡", "牌型", typeTotal]);
      }
      if (this.baojiao[w] && cfg.baojiao_bonus > 0 && payers.length > 0) {
        for (const q of payers)
          if (q !== w) pflow(w, q, cfg.baojiao_bonus, "报叫");
        detail.push([w, "报叫", "", cfg.baojiao_bonus]);
      }
    }

    // ---- 杠分 ----
    const KONG_WHO = { 1: "下家", 2: "对家", 3: "上家" };
    for (let p = 0; p < 4; p++) {
      if (!tenpai[p]) continue;
      if (voidPlayer !== null && p === voidPlayer) continue;
      for (const m of this.melds[p]) {
        const [mt, t, src] = m;
        if (!meldIsKong(mt)) continue;
        const rel = (((src - p) % 4) + 4) % 4;
        const who = KONG_WHO[rel] || "";
        const info = mt === MELD_KONG_CONCEALED ? `暗杠 ${tileCn(t)}`
          : mt === MELD_KONG_ADDED ? `补杠 ${tileCn(t)}`
          : `明杠 ${tileCn(t)}` + (who ? `（${who}点杠）` : "");
        const payersK = mt === MELD_KONG_EXPOSED
          ? [src] : [0, 1, 2, 3].filter(q => q !== p);
        for (const q of payersK)
          pflow(p, q, cfg.kong_chickens, "杠分", info);
        detail.push([p, "杠分", info, cfg.kong_chickens]);
      }
    }

    // ---- 责任鸡溢价 ----
    const unt = new Set([0, 1, 2, 3].filter(q => !tenpai[q]));
    for (const [claimer, discarder, premium, tag] of this._claimedJiRows(goldTiles)) {
      if (voidPlayer !== null && claimer === voidPlayer) continue;
      if (unt.has(claimer)) continue;
      pflow(claimer, discarder, premium, "责任鸡", tag);
      detail.push([claimer, "责任鸡", tag, premium]);
    }

    // ---- 包鸡/包杠 ----
    const premPairs = new Map(); // key: `${cl},${dsc}` -> [[tg, prem], ...]
    for (const [cl, dsc, prem, tg] of this._claimedJiRows(goldTiles)) {
      const key = `${cl},${dsc}`;
      const arr = premPairs.get(key);
      if (arr) arr.push([tg, prem]);
      else premPairs.set(key, [[tg, prem]]);
    }
    for (let p = 0; p < 4; p++) {
      if (tenpai[p]) continue;
      if (voidPlayer !== null && p === voidPlayer) continue;
      if (cfg.end_baoji) {
        const ming = ownTags[p].map((tg, i) => [tg, ownVals[p][i]]);
        let nM = 0, vM = 0.0;
        for (const t of sortedNums(this.ji_tiles)) {
          const k = this.meldedCount(p, t);
          if (k <= 0) continue;
          nM += k;
          vM += k * parseFloat(cfg.ji_hold) * (goldTiles.has(t) ? jin : 1.0);
        }
        if (vM > 0) ming.push([`副露鸡 ${pyG(nM)} 张`, vM]);
        for (const [tg, val] of ming) {
          if (val <= 0) continue;
          for (let q = 0; q < 4; q++) {
            if (q === p) continue;
            pflow(q, p, val, "包鸡", tg, p);
          }
          detail.push([p, "包鸡", tg, val]);
        }
        for (let q = 0; q < 4; q++) {
          if (q === p) continue;
          const prem = premPairs.get(`${p},${q}`);
          if (!prem) continue;
          let psum = 0;
          for (const [, v] of prem) psum += v;
          const etag = prem.length === 1
            ? prem[0][0] : prem.map(x => x[0]).join("、");
          pflow(q, p, psum, "包鸡", `${etag}溢价`, p);
          detail.push([p, "包鸡", `${etag}溢价`, psum]);
        }
      }
      if (cfg.end_baogang) {
        let kongs = 0;
        for (const m of this.melds[p]) if (meldIsKong(m[0])) kongs += 1;
        for (let i = 0; i < kongs; i++) {
          for (let q = 0; q < 4; q++) {
            if (q !== p) pflow(q, p, BAO_KONG, "包杠", "", p);
          }
          detail.push([p, "包杠", "", BAO_KONG]);
        }
      }
    }

    // ---- 两两互减 ----
    for (let a = 0; a < 4; a++) {
      for (let b = 0; b < 4; b++) {
        if (a === b) continue;
        const diff = c[a] - c[b];
        if (diff > 0) {
          pair[a][b] += diff;
          const rows = [];
          for (const [label, info, val] of items[a])
            rows.push([label, info, val, a]);
          for (const [label, info, val] of items[b])
            if (val) rows.push([label, info, -val, b]);
          const key = `${a},${b}`;
          const old = pairDet.get(key) || [];
          pairDet.set(key, rows.concat(old));
        }
      }
    }

    // ---- 汇总 delta ----
    const delta = [0.0, 0.0, 0.0, 0.0];
    for (let a = 0; a < 4; a++)
      for (let b = 0; b < 4; b++)
        delta[a] += pair[a][b] - pair[b][a];

    const pTypes = typesBy[primary];
    const fanCn = fanCnOf(primary);
    const fanTotal = sumArr(pair[primary]);
    let wt;
    if (wtI >= 0) wt = wtI;
    else if (isTsumo && this.last_draw !== null && this.last_draw !== undefined)
      wt = this.last_draw;
    else wt = null;

    const winnersFan = winners.map(w => pyD([
      ["seat", w],
      ["fan_cn", fanCnOf(w)],
      ["fan_type", [...typesBy[w]].sort().join("+") || PING_HU],
      ["fan", sumArr(pair[w])],
    ]));

    const pairDetail = [...pairDet.entries()].map(([key, rows]) => {
      const [a, b] = key.split(",").map(Number);
      return pyD([["a", a], ["b", b],
        ["items", rows.map(r => pyD([["label", r[0]], ["info", r[1]],
          ["value", r[2]], ["who", r.length > 3 ? r[3] : a]]))]]);
    });

    this.result = {
      type: "win",
      winner: primary,
      winners: winners.slice(),
      loser,
      is_tsumo: isTsumo,
      rob_kong: robKong,
      void_player: voidPlayer,
      win_tile: wt,
      fan_type: [...pTypes].sort().join("+") || PING_HU,
      fan_cn: fanCn,
      fan: fanTotal,
      total_fan: fanTotal,
      chickens: c,
      tenpai: [0, 1, 2, 3].filter(q => tenpai[q]),
      fanji_flip: (fanji.size > 0 && fanjiFlip !== null) ? fanjiFlip : null,
      fanji_tiles: sortedNums(fanji),
      fanji_gold: goldTiles.size > 0,
      gold_tiles: sortedNums(goldTiles),
      ji_hold_detail: pyD([0, 1, 2, 3].map(p => [String(p), holdBd[p]])),
      fanji_detail: pyD([0, 1, 2, 3].map(p => [String(p), fanjiBd[p]])),
      ji_discard_detail: discardDetail,
      ji_capture_detail: captureDetail,
      ji_claim_detail: claimDetail,
      jiesuan_fanji: cfg.jiesuan_fanji,
      kaiju_fanji: cfg.kaiju_fanji,
      how,
      how_value: howVal,
      type_value: typeTotalBy[primary],
      winners_fan: winnersFan,
      dealer_bonus: 0.0,
      dou: [],
      chicken: detail.filter(d => CHICKEN_LABELS.includes(d[1])),
      detail,
      pair,
      pair_detail: pairDetail,
      wall_left: this.wallLeft(),
    };
    this._finalize(delta, primary);
  }

  _settleHuangzhuang() {
    const cfg = this.cfg;
    const detail = [];
    const pair = [[0.0, 0.0, 0.0, 0.0],
                  [0.0, 0.0, 0.0, 0.0],
                  [0.0, 0.0, 0.0, 0.0],
                  [0.0, 0.0, 0.0, 0.0]];
    const pairDet = new Map();

    const pflow = (a, b, val, label, info = "", who = null) => {
      if (val <= 0) return;
      pair[a][b] += val;
      const key = `${a},${b}`;
      const row = [label, info, val, who === null ? a : who];
      const arr = pairDet.get(key);
      if (arr) arr.push(row);
      else pairDet.set(key, [row]);
    };

    const tenpai = [0, 1, 2, 3].map(p => this._tenpaiFlag(p));
    const baodapai = [];

    // ---- 包大牌面 ----
    for (let p = 0; p < 4; p++) {
      if (!tenpai[p]) continue;
      const [tval, tcn] = this._maxHuType(p);
      const pay = parseFloat(cfg.zi_mo) + tval;
      baodapai.push(pyD([["seat", p], ["type", tcn || "平胡"],
        ["type_value", tval], ["pay", pay]]));
      if (!cfg.huang_baodapai) continue;
      const payers = [0, 1, 2, 3].filter(q => !tenpai[q]);
      for (const q of payers)
        pflow(p, q, pay, "包大牌面", tcn || "平胡", q);
      if (payers.length > 0)
        detail.push([p, "包大牌面", tcn || "平胡（自摸鸡）", pay]);
    }

    // ---- 可选：包鸡 / 包杠 ----
    if (cfg.huang_baoji || cfg.huang_baogang) {
      const [ownVals, ownTags] = this._ownJiAfterClaims(null);
      const jinH = 1.0;
      for (let p = 0; p < 4; p++) {
        if (tenpai[p]) continue;
        if (cfg.huang_baoji) {
          const ming = ownTags[p].map((tg, i) => [tg, ownVals[p][i]]);
          const nYaoM = this.meldedCount(p, YAOJI);
          let nOthM = 0;
          for (const t of this.ji_tiles)
            if (t !== YAOJI) nOthM += this.meldedCount(p, t);
          const vM = (nYaoM * jinH + nOthM) * parseFloat(cfg.ji_hold);
          if (vM > 0) ming.push([`副露鸡 ${pyG(nYaoM + nOthM)} 张`, vM]);
          for (const [tg, v] of ming) {
            if (v <= 0) continue;
            for (let q = 0; q < 4; q++) {
              if (q !== p) pflow(q, p, v, "包鸡", tg, p);
            }
            detail.push([p, "包鸡", tg, v]);
          }
        }
        if (cfg.huang_baogang) {
          let kongs = 0;
          for (const m of this.melds[p]) if (meldIsKong(m[0])) kongs += 1;
          for (let i = 0; i < kongs; i++) {
            for (let q = 0; q < 4; q++) {
              if (q !== p) pflow(q, p, BAO_KONG, "包杠", "", p);
            }
            detail.push([p, "包杠", "", BAO_KONG]);
          }
        }
      }
    }

    const delta = [0.0, 0.0, 0.0, 0.0];
    for (let a = 0; a < 4; a++)
      for (let b = 0; b < 4; b++)
        delta[a] += pair[a][b] - pair[b][a];

    const pairDetail = [...pairDet.entries()].map(([key, rows]) => {
      const [a, b] = key.split(",").map(Number);
      return pyD([["a", a], ["b", b],
        ["items", rows.map(r => pyD([["label", r[0]], ["info", r[1]],
          ["value", r[2]], ["who", r.length > 3 ? r[3] : a]]))]]);
    });

    this.result = {
      type: "huangzhuang",
      tenpai: [0, 1, 2, 3].filter(q => tenpai[q]),
      chickens: [0.0, 0.0, 0.0, 0.0],
      baodapai,
      baodapai_on: !!cfg.huang_baodapai,
      cha_jiao: detail,
      detail,
      pair,
      pair_detail: pairDetail,
      wall_left: 0,
    };
    this._finalize(delta, this.dealer);
  }

  // -- 全量状态 dump（与 Python dump_state 同形状） --------------------------
  dumpState() {
    const s = {};
    s.baojiao = this.baojiao.slice();
    s.first_discard_done = this.first_discard_done.slice();
    s.hengji_opened = this.hengji_opened;
    s.hengji_starter = this.hengji_starter;
    s.ji_events = this.ji_events.map(e => e.slice());
    s.ji_event_tiles = this.ji_event_tiles.slice();
    s._hengji_done = setJson(this._hengji_done);
    s._hengji_rounds = pyD([...this._hengji_rounds.entries()].map(([t, r]) =>
      [t, pyD([["start", r.start], ["seats", setJson(r.seats)]])]));
    s.hengji_events = this.hengji_events.map(e => e.slice());
    s._ji_tag_of_last = this._ji_tag_of_last.slice();
    s._ji_val_by_meld = pyD([...this._ji_val_by_meld.entries()].map(([k, v]) => {
      const [a, b] = k.split(",").map(Number);
      return [[a, b], v];
    }));
    s.claimed_ji = this.claimed_ji.map(e => e.slice());
    s.discard_view = this.discard_view.map(a => a.slice());
    s.discard_claimed_view = this.discard_claimed_view.map(a => a.slice());
    s.captured_ji = this.captured_ji.map(e => e.slice());
    s._respond_hu_votes = this._respond_hu_votes.slice();
    s._n_hu = this._n_hu;
    s.repao_discard = this.repao_discard;
    s._after_gang_draw = this._after_gang_draw;
    s._any_discard = this._any_discard;
    s.ji_tiles = setJson(this.ji_tiles);
    s.kaiju_flip = this.kaiju_flip;
    s.dealer_streak = this.dealer_streak;
    s.wall = this.wall.slice();
    s.wall_pos = this.wall_pos;
    s.hands = this.hands.map(a => a.slice());
    s.melds = this.melds.map(ms => ms.map(m => m.slice()));
    s.discards = this.discards.map(a => a.slice());
    s.ever_tenpai = this.ever_tenpai.slice();
    s.pass_lock = this.pass_lock.map(setJson);
    s.gang_discard = this.gang_discard.slice();
    s.drawn = this.drawn;
    s.dealer = this.dealer;
    s.current = this.current;
    s.phase = this.phase;
    s.last_discard = this.last_discard ? this.last_discard.slice() : null;
    s.last_discard_was_gang = this.last_discard_was_gang;
    s._respond_order = this._respond_order.slice();
    s._respond_idx = this._respond_idx;
    s._rob_kong = this._rob_kong ? this._rob_kong.slice() : null;
    s.chongfeng_player = this.chongfeng_player;
    s.zeren_player = this.zeren_player;
    s.zeren_ponger = this.zeren_ponger;
    s._first_yaoji_discarded = this._first_yaoji_discarded;
    s.result = this.result === null
      ? null : pyD(Object.entries(this.result).map(([k, v]) => [k, v]));
    s.score_delta = this.score_delta.slice();
    s.next_dealer = this.next_dealer;
    s.log = this.log.map(e => e.slice());
    s.last_draw = this.last_draw;
    if (this._rob_kong_konger !== undefined)
      s._rob_kong_konger = this._rob_kong_konger;
    return s;
  }
}

module.exports = {
  DushanGame,
  // 辅助导出（供编码器移植复用）
  DUSHAN_HU_TYPE, DUSHAN_FAN_CN, CHICKEN_LABELS,
  JI_CHONGFENG, JI_HENG, JI_YAO, KONG_CHICKENS, BAO_KONG, ZI_MO_JI, REPAO_JI,
  JI_TAG_BY_VAL,
  FANJI_UP, FANJI_DOWN, FANJI_BOTH, FANJI_OFF, FANJI_CN,
  normFanji, fanjiTilesFrom, defaultConfig,
  pyG, pyRound,
};
