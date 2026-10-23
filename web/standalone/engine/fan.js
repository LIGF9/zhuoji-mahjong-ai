"use strict";
// 胡牌判定与番型计算（对应 zhuoji/fan.py）。
// 预计算表 + 花色分解：单花色 9 维计数向量（每项 ≤4）按 5 进制编码进 MELD_SET。

const { NUM_TILE_TYPES, suitOf } = require("./tiles");

// ---- 番型常量 ----
const PING_HU = "ping_hu";
const DA_DUI_ZI = "da_dui_zi";
const QI_DUI = "qi_dui";
const LONG_QI_DUI = "long_qi_dui";
const QING_YI_SE = "qing_yi_se";
const QING_DA_DUI = "qing_da_dui";
const QING_QI_DUI = "qing_qi_dui";
const QING_LONG_BEI = "qing_long_bei";

const FAN_TABLE = {
  [PING_HU]: 1,
  [DA_DUI_ZI]: 5,
  [QI_DUI]: 7,
  [LONG_QI_DUI]: 10,
  [QING_YI_SE]: 10,
  [QING_DA_DUI]: 15,
  [QING_QI_DUI]: 17,
  [QING_LONG_BEI]: 20,
};

const FAN_CN = {
  [PING_HU]: "平胡",
  [DA_DUI_ZI]: "大对子",
  [QI_DUI]: "七对",
  [LONG_QI_DUI]: "龙七对",
  [QING_YI_SE]: "清一色",
  [QING_DA_DUI]: "清大对",
  [QING_QI_DUI]: "清七对",
  [QING_LONG_BEI]: "青龙背",
};

// ---- 副露类型 ----
const MELD_CHOW = 0;
const MELD_PONG = 1;
const MELD_KONG_EXPOSED = 2;
const MELD_KONG_ADDED = 3;
const MELD_KONG_CONCEALED = 4;

const MELD_CN = {
  [MELD_CHOW]: "吃",
  [MELD_PONG]: "碰",
  [MELD_KONG_EXPOSED]: "点杠",
  [MELD_KONG_ADDED]: "补杠",
  [MELD_KONG_CONCEALED]: "暗杠",
};

const KONG_TYPES = [MELD_KONG_EXPOSED, MELD_KONG_ADDED, MELD_KONG_CONCEALED];

function meldIsKong(mt) { return KONG_TYPES.includes(mt); }

// ---- 预计算：单花色「可完全拆成面子」的向量集合 ----
const _POW5 = [];
for (let i = 0; i < 9; i++) _POW5[i] = Math.pow(5, i);

function _encode9(vec) {
  let e = 0;
  for (let i = 0; i < 9; i++) e += vec[i] * _POW5[i];
  return e;
}

function _buildMeldSet() {
  const baseMelds = [];
  for (let r = 0; r < 9; r++) {          // 刻子
    const v = new Array(9).fill(0);
    v[r] = 3;
    baseMelds.push(v);
  }
  for (let r = 0; r < 7; r++) {          // 顺子
    const v = new Array(9).fill(0);
    v[r] = v[r + 1] = v[r + 2] = 1;
    baseMelds.push(v);
  }
  const out = new Set();
  function rec(start, chosen) {
    if (chosen.length <= 4) {
      const acc = new Array(9).fill(0);
      for (const m of chosen)
        for (let i = 0; i < 9; i++) acc[i] += m[i];
      if (Math.max(...acc) <= 4) out.add(_encode9(acc));
    }
    if (chosen.length === 4) return;
    for (let i = start; i < baseMelds.length; i++)
      rec(i, chosen.concat([baseMelds[i]]));
  }
  rec(0, []);
  return out;
}

const MELD_SET = _buildMeldSet();

function _meldsOk(counts27) {
  for (let s = 0; s < 3; s++) {
    let e = 0;
    for (let i = 0; i < 9; i++) {
      const c = counts27[s * 9 + i];
      if (c > 4) return false;
      e += c * _POW5[i];
    }
    if (!MELD_SET.has(e)) return false;
  }
  return true;
}

// ---- 胡牌判定 ----
function isSevenPairs(counts) {
  // 14 张是否七对结构（可含 4 张相同，即龙七对）。
  let total = 0;
  for (const c of counts) total += c;
  if (total !== 14) return false;
  let pairs = 0;
  for (const c of counts) {
    if (c === 0) continue;
    if (c === 2) pairs += 1;
    else if (c === 4) pairs += 2;
    else return false;
  }
  return pairs === 7;
}

// 胡牌/听牌判定缓存（纯加速，无行为差异）
const _WIN_CACHE = new Map();
const _TENPAI_CACHE = new Map();
const _CACHE_CAP = 300000;

function _handKey(handCounts, melds) {
  return handCounts.join(",") + "|"
    + melds.map(m => m.join(":")).join(";");
}

function isWinningHand(handCounts, melds) {
  const key = _handKey(handCounts, melds);
  let cached = _WIN_CACHE.get(key);
  if (cached === undefined) {
    cached = _isWinningHandImpl(handCounts, melds);
    if (_WIN_CACHE.size < _CACHE_CAP) _WIN_CACHE.set(key, cached);
  }
  return cached;
}

function _isWinningHandImpl(handCounts, melds) {
  const nMelds = melds.length;
  const need = 4 - nMelds;
  if (need < 0) return false;
  let total = 0;
  for (const c of handCounts) total += c;
  if (total !== need * 3 + 2) return false;
  if (nMelds === 0 && isSevenPairs(handCounts)) return true;
  const c = handCounts.slice();
  for (let t = 0; t < NUM_TILE_TYPES; t++) {
    if (c[t] >= 2) {
      c[t] -= 2;
      const ok = _meldsOk(c);
      c[t] += 2;
      if (ok) return true;
    }
  }
  return false;
}

function winningTiles(handCounts, melds) {
  // 13 张（或等价）暗手的听牌张列表。
  const key = _handKey(handCounts, melds);
  let cached = _TENPAI_CACHE.get(key);
  if (cached === undefined) {
    cached = _winningTilesImpl(handCounts, melds);
    if (_TENPAI_CACHE.size < _CACHE_CAP) _TENPAI_CACHE.set(key, cached);
  }
  return cached;
}

function _winningTilesImpl(handCounts, melds) {
  const nMelds = melds.length;
  if (nMelds > 4) return [];
  const expect = (4 - nMelds) * 3 + 1;
  let total = 0;
  for (const c of handCounts) total += c;
  if (total !== expect) return [];
  const c = handCounts.slice();
  const res = [];
  for (let t = 0; t < NUM_TILE_TYPES; t++) {
    if (c[t] >= 4) continue;
    c[t] += 1;
    if (isWinningHand(c, melds)) res.push(t);
    c[t] -= 1;
  }
  return res;
}

function isTenpai(handCounts, melds) {
  return winningTiles(handCounts, melds).length > 0;
}

// ---- 番型判定 ----
function _isQingYiSe(handCounts, melds) {
  const suits = new Set();
  handCounts.forEach((c, t) => {
    if (c) suits.add(suitOf(t));
  });
  if (suits.size > 1) return false;
  for (const m of melds) {
    suits.add(suitOf(m[1]));
    if (suits.size > 1) return false;
  }
  return suits.size === 1;
}

function _allTriplets(handCounts, melds) {
  for (const m of melds)
    if (m[0] === MELD_CHOW) return false;
  const c = handCounts.slice();
  const nMelds = melds.length;
  let sumAll = 0;
  for (const x of c) sumAll += x;
  for (let t = 0; t < NUM_TILE_TYPES; t++) {
    if (c[t] >= 2) {
      c[t] -= 2;
      const ok = _meldsOk(c)
        && c.every(x => x % 3 === 0)
        && c.reduce((s, x) => s + x, 0) === (4 - nMelds) * 3;
      c[t] += 2;
      if (ok) return true;
    }
  }
  return false;
}

function classifyFan(handCounts, melds, winTile = null) {
  // 返回 [牌型名, 番值]；不成胡时返回 ["", 0]。
  const c = handCounts.slice();
  const nMelds = melds.length;
  if (nMelds === 0 && isSevenPairs(c)) {
    const dragon = winTile === null ? c.some(x => x === 4) : c[winTile] === 4;
    const base = dragon ? LONG_QI_DUI : QI_DUI;
    if (_isQingYiSe(c, melds)) {
      const name = base === LONG_QI_DUI ? QING_LONG_BEI : QING_QI_DUI;
      return [name, FAN_TABLE[name]];
    }
    return [base, FAN_TABLE[base]];
  }
  if (!isWinningHand(c, melds)) return ["", 0];
  const qing = _isQingYiSe(c, melds);
  if (_allTriplets(c, melds)) {
    const name = qing ? QING_DA_DUI : DA_DUI_ZI;
    return [name, FAN_TABLE[name]];
  }
  if (qing) return [QING_YI_SE, FAN_TABLE[QING_YI_SE]];
  return [PING_HU, FAN_TABLE[PING_HU]];
}

function isHighValueHand(handCounts, melds) {
  // 是否"平胡以上"牌型（点胡通行证判定用）。
  const [, fan] = classifyFan(handCounts, melds);
  return fan > FAN_TABLE[PING_HU];
}

module.exports = {
  PING_HU, DA_DUI_ZI, QI_DUI, LONG_QI_DUI, QING_YI_SE,
  QING_DA_DUI, QING_QI_DUI, QING_LONG_BEI,
  FAN_TABLE, FAN_CN,
  MELD_CHOW, MELD_PONG, MELD_KONG_EXPOSED, MELD_KONG_ADDED, MELD_KONG_CONCEALED,
  MELD_CN, KONG_TYPES, meldIsKong,
  MELD_SET, _meldsOk,
  isSevenPairs, isWinningHand, winningTiles, isTenpai,
  _isQingYiSe, _allTriplets, classifyFan, isHighValueHand,
};
