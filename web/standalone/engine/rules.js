"use strict";
// 贵州捉鸡麻将规则引擎基类（对应 zhuoji/rules.py 的 ZhuojiGame）。
// 独山引擎 DushanGame 继承并覆盖大部分行为；基类保留共用基础设施。

const { NUM_TILE_TYPES, YAOJI, tileCn } = require("./tiles");
const {
  MELD_PONG, MELD_KONG_EXPOSED, MELD_KONG_ADDED, MELD_KONG_CONCEALED,
  meldIsKong, isWinningHand, isTenpai, winningTiles, isHighValueHand,
} = require("./fan");

// ---- 阶段与动作 ----
const PHASE = {
  SELF_KONG: "self_kong",
  DISCARD: "discard",
  RESPOND: "respond",
  OVER: "over",
};

const DISCARD = "discard";
const HU = "hu";
const ANGANG = "angang";
const BUGANG = "bugang";
const PENG = "peng";
const MINGGANG = "minggang";
const PASS = "pass";

const ACTION_CN = {
  [DISCARD]: "打", [HU]: "胡", [ANGANG]: "暗杠", [BUGANG]: "补杠",
  [PENG]: "碰", [MINGGANG]: "点杠", [PASS]: "过",
};

function actionLabel(a) {
  const cn = ACTION_CN[a.kind];
  return a.tile < 0 ? cn : cn + tileCn(a.tile);
}

class ZhuojiGame {
  constructor() {
    // 由子类负责初始化字段（Python 里 _start 直接写 self.__dict__）
  }

  // -- 初始化（基类部分；wall 由外部给定，不做洗牌） -----------------------
  _startBase(wall, dealer) {
    const cfg = this.cfg;
    this.wall = wall.slice();
    this.wall_pos = 0;

    this.hands = [];
    for (let p = 0; p < 4; p++) this.hands.push(new Array(NUM_TILE_TYPES).fill(0));
    for (let i = 0; i < 13; i++)
      for (let p = 0; p < 4; p++)
        this.hands[p][this._drawFromWall()] += 1;

    this.melds = [[], [], [], []];
    this.discards = [[], [], [], []];
    this.ever_tenpai = [false, false, false, false];
    this.pass_lock = [new Set(), new Set(), new Set(), new Set()];
    this.gang_discard = [false, false, false, false];
    this.drawn = false;

    if (dealer !== null && dealer !== undefined) {
      this.dealer = dealer;
    } else if (cfg.first_dealer !== null && cfg.first_dealer !== undefined) {
      this.dealer = cfg.first_dealer;
    } else {
      this.dealer = 0; // 无 rng 场景下的确定性回退
    }

    this.current = this.dealer;
    this.hands[this.dealer][this._drawFromWall()] += 1;
    this.drawn = true;

    this.phase = PHASE.OVER;
    this.last_discard = null;
    this.last_discard_was_gang = false;
    this._respond_order = [];
    this._respond_idx = 0;
    this._rob_kong = null;

    this.chongfeng_player = null;
    this.zeren_player = null;
    this.zeren_ponger = null;
    this._first_yaoji_discarded = false;

    this.result = null;
    this.score_delta = [0.0, 0.0, 0.0, 0.0];
    this.next_dealer = this.dealer;
    this.log = [];

    this._enterSelfPhase(this.dealer);
  }

  // -- 基础设施 ------------------------------------------------------------
  _drawFromWall() {
    const t = this.wall[this.wall_pos];
    this.last_draw = t;
    this.wall_pos += 1;
    return t;
  }

  wallLeft() { return this.wall.length - this.wall_pos; }

  hand(p) { return this.hands[p]; }

  hasDou(p) { return this.melds[p].some(m => meldIsKong(m[0])); }

  meldedCount(p, tile) {
    let c = 0;
    for (const m of this.melds[p])
      if (m[1] === tile) c += meldIsKong(m[0]) ? 4 : 3;
    return c;
  }

  handTiles(p) {
    const out = [];
    this.hands[p].forEach((c, t) => {
      for (let i = 0; i < c; i++) out.push(t);
    });
    return out;
  }

  _opponents(p) { return [(p + 1) % 4, (p + 2) % 4, (p + 3) % 4]; }

  actor() {
    // 抢杠时 current 仍是补杠者，必须返回 _rob_kong[0]。
    if (this.phase === PHASE.RESPOND) return this._respond_order[this._respond_idx];
    if (this.phase === PHASE.SELF_KONG && this._rob_kong !== null) return this._rob_kong[0];
    return this.current;
  }

  // -- 胡牌判定 ------------------------------------------------------------
  canTsumo(p) { return isWinningHand(this.hands[p], this.melds[p]); }

  canRon(p, tile) {
    if (this.cfg.enable_pass_lock && this.pass_lock[p].has(tile)) return false;
    const h = this.hands[p].slice();
    h[tile] += 1;
    if (!isWinningHand(h, this.melds[p])) return false;
    if (this.cfg.ping_hu_needs_dou_for_ron) {
      if (!this.hasDou(p) && !isHighValueHand(h, this.melds[p])) return false;
    }
    return true;
  }

  _canPeng(p, tile) {
    return this.melds[p].length < 4 && this.hands[p][tile] >= 2;
  }

  _canMinggang(p, tile) {
    return this.melds[p].length < 4 && this.hands[p][tile] >= 3;
  }

  // -- 阶段切换 ------------------------------------------------------------
  _selfKongOptions(p) {
    const opts = [];
    const h = this.hands[p];
    for (let t = 0; t < NUM_TILE_TYPES; t++)
      if (h[t] === 4) opts.push([ANGANG, t]);
    for (const m of this.melds[p])
      if (m[0] === MELD_PONG && h[m[1]] >= 1) opts.push([BUGANG, m[1]]);
    return opts;
  }

  _enterSelfPhase(p) {
    this._updateTenpai(p);
    if (this._rob_kong !== null || this._selfKongOptions(p).length > 0)
      this.phase = PHASE.SELF_KONG;
    else
      this.phase = PHASE.DISCARD;
  }

  _updateTenpai(p) {
    if (!this.ever_tenpai[p] && isTenpai(this.hands[p], this.melds[p]))
      this.ever_tenpai[p] = true;
  }

  // -- 合法动作 ------------------------------------------------------------
  legalActions() {
    if (this.phase === PHASE.SELF_KONG) {
      if (this._rob_kong !== null) {
        const [p, t] = this._rob_kong;
        const acts = [{ kind: PASS, tile: -1 }];
        if (this.canRon(p, t)) acts.push({ kind: HU, tile: -1 });
        return acts;
      }
      const p = this.current;
      const acts = [{ kind: PASS, tile: -1 }];
      for (const [kind, t] of this._selfKongOptions(p))
        acts.push({ kind, tile: t });
      return acts;
    }

    if (this.phase === PHASE.DISCARD) {
      const p = this.current;
      const acts = [];
      for (let t = 0; t < NUM_TILE_TYPES; t++)
        if (this.hands[p][t] > 0) acts.push({ kind: DISCARD, tile: t });
      if (this.drawn && this.canTsumo(p)) acts.push({ kind: HU, tile: -1 });
      return acts;
    }

    if (this.phase === PHASE.RESPOND) {
      const p = this._currentResponder();
      const acts = [{ kind: PASS, tile: -1 }];
      if (this._rob_kong !== null) {
        if (this.canRon(p, this._rob_kong[1])) acts.push({ kind: HU, tile: -1 });
        return acts;
      }
      const tile = this.last_discard[1];
      if (this.canRon(p, tile)) acts.push({ kind: HU, tile: -1 });
      if (this._canMinggang(p, tile)) acts.push({ kind: MINGGANG, tile });
      if (this._canPeng(p, tile)) acts.push({ kind: PENG, tile });
      return acts;
    }

    return [];
  }

  _currentResponder() { return this._respond_order[this._respond_idx]; }

  // -- 主循环 ------------------------------------------------------------
  step(action) {
    if (this.phase === PHASE.OVER) return true;
    this.log.push([this.actor(), actionLabel(action)]);
    if (this.phase === PHASE.SELF_KONG) this._stepSelfKong(action);
    else if (this.phase === PHASE.DISCARD) this._stepDiscard(action);
    else this._stepRespond(action);
    return this.phase === PHASE.OVER;
  }

  // 阶段一：摸牌后的暗杠 / 补杠
  _stepSelfKong(action) {
    if (this._rob_kong !== null) {
      const [p, t] = this._rob_kong;
      const konger = this._rob_kong_konger;
      this._rob_kong = null;
      if (action.kind === HU)
        this._settleWin(p, t, false, konger, true);
      else
        this._doBugang(konger, t);
      return;
    }

    const p = this.current;
    if (action.kind === PASS) { this.phase = PHASE.DISCARD; return; }
    if (action.kind === ANGANG) {
      const t = action.tile;
      this.hands[p][t] -= 4;
      this.melds[p].push([MELD_KONG_CONCEALED, t, p]);
      this._gangDraw(p);
      return;
    }
    if (action.kind === BUGANG) {
      const t = action.tile;
      if (this.cfg.enable_qiang_gang) {
        const robbers = this._opponents(p).filter(q => this.canRon(q, t));
        if (robbers.length > 0) {
          this._rob_kong = [robbers[0], t];
          this._rob_kong_konger = p;
          this.phase = PHASE.SELF_KONG;
          return;
        }
      }
      this._doBugang(p, t);
    }
  }

  _doBugang(p, t) {
    this.hands[p][t] -= 1;
    for (let i = 0; i < this.melds[p].length; i++) {
      const m = this.melds[p][i];
      if (m[0] === MELD_PONG && m[1] === t) {
        this.melds[p][i] = [MELD_KONG_ADDED, t, m[2]];
        break;
      }
    }
    this._gangDraw(p);
  }

  _gangDraw(p) {
    if (this.wallLeft() <= 0) { this._settleHuangzhuang(); return; }
    this.hands[p][this._drawFromWall()] += 1;
    this.current = p;
    this.drawn = true;
    this.gang_discard[p] = true;
    this.pass_lock[p] = new Set();
    this._enterSelfPhase(p);
  }

  // 阶段二：出牌 / 自摸胡（独山引擎覆盖）
  _stepDiscard(action) {
    const p = this.current;
    if (action.kind === HU) { this._settleWin(p, -1, true); return; }
    const t = action.tile;
    this.hands[p][t] -= 1;
    this.discards[p].push(t);
    if (t === YAOJI && !this._first_yaoji_discarded) {
      this._first_yaoji_discarded = true;
      this.chongfeng_player = p;
    }
    this._updateTenpai(p);
    this.last_discard_was_gang = this.gang_discard[p];
    this.gang_discard[p] = false;
    this.last_discard = [p, t];
    this._beginRespond(p, t);
  }

  _beginRespond(discarder, tile) {
    const others = this._opponents(discarder);
    const huList = others.filter(q => this.canRon(q, tile));
    const claimList = others.filter(q => !huList.includes(q)
      && (this._canPeng(q, tile) || this._canMinggang(q, tile)));
    const order = huList.concat(claimList);
    if (order.length > 0) {
      this._respond_order = order;
      this._respond_idx = 0;
      this.phase = PHASE.RESPOND;
    } else {
      this._nextPlayer(discarder);
    }
  }

  _nextPlayer(fromP) {
    const nxt = (fromP + 1) % 4;
    if (this.wallLeft() <= 0) { this._settleHuangzhuang(); return; }
    this.hands[nxt][this._drawFromWall()] += 1;
    this.current = nxt;
    this.drawn = true;
    this.gang_discard[nxt] = false;
    this.pass_lock[nxt] = new Set();
    this._enterSelfPhase(nxt);
  }

  // 阶段三：响应
  _stepRespond(action) {
    const p = this._currentResponder();
    const discarder = this.last_discard[0], tile = this.last_discard[1];

    if (action.kind === PASS) {
      this.pass_lock[p].add(tile);
      this._respond_idx += 1;
      if (this._respond_idx >= this._respond_order.length) {
        this._respond_order = [];
        this._respond_idx = 0;
        this._nextPlayer(discarder);
      }
      return;
    }

    if (action.kind === HU) {
      this._respond_order = [];
      this._settleWin(p, tile, false, discarder, false);
      return;
    }

    const ds = this.discards[discarder];
    if (ds.length > 0 && ds[ds.length - 1] === tile) ds.pop();

    if (action.kind === PENG) {
      this.hands[p][tile] -= 2;
      this.melds[p].push([MELD_PONG, tile, discarder]);
      if (tile === YAOJI && this.zeren_player === null
          && this.chongfeng_player !== null) {
        this.zeren_player = discarder;
        this.zeren_ponger = p;
        this.chongfeng_player = null;
      }
      this.current = p;
      this.drawn = false;
      this.pass_lock[p] = new Set();
      this._respond_order = [];
      this._respond_idx = 0;
      this._enterSelfPhase(p);
    } else if (action.kind === MINGGANG) {
      this.hands[p][tile] -= 3;
      this.melds[p].push([MELD_KONG_EXPOSED, tile, discarder]);
      this.current = p;
      this.pass_lock[p] = new Set();
      this._respond_order = [];
      this._respond_idx = 0;
      this._gangDraw(p);
    }
  }

  // -- 结算（独山引擎覆盖；基类只保留 _finalize） ---------------------------
  _settleWin() { throw new Error("基类 _settleWin 应由子类实现"); }
  _settleHuangzhuang() { throw new Error("基类 _settleHuangzhuang 应由子类实现"); }

  _finalize(delta, nextDealer) {
    const base = this.cfg.base_score;
    this.score_delta = delta.map(d => d * base);
    this.result.deltas = this.score_delta.slice();
    this.next_dealer = nextDealer;
    this.phase = PHASE.OVER;
  }
}

module.exports = {
  PHASE, DISCARD, HU, ANGANG, BUGANG, PENG, MINGGANG, PASS,
  ACTION_CN, actionLabel, ZhuojiGame,
};
