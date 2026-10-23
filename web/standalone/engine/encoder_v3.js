"use strict";
// 状态编码 v3（对应 zhuoji/encoder_v3.py + encoder_v2.py + encoder.py 的 encode_v1）。
// 63 通道 / 130 全局维；v1 ⊂ v2 ⊂ v3 严格超集，运算顺序照抄 Python/numpy：
// 中间量用 double，落进 Float32Array 即 float32 落地；f32 标量运算用 Math.fround。

const { NUM_TILE_TYPES, DECK_SIZE, YAOJI, suitOf } = require("./tiles");
const { meldIsKong, winningTiles, FAN_TABLE } = require("./fan");
const { PHASE } = require("./rules");

const f32 = Math.fround;

const NUM_CHANNELS_V3 = 63;
const GLOBAL_DIM_V3 = 130;
const _V2_CHANNELS = 31;
const _V2_GLOBAL_DIM = 98;
const _OPENING_TILES = 52;

const HEAD_DISCARD = "discard";
const HEAD_SELF_KONG = "self_kong";
const HEAD_RESPOND = "respond";
const HEAD_DIMS = { [HEAD_DISCARD]: 28, [HEAD_SELF_KONG]: 28, [HEAD_RESPOND]: 4 };

const ALL_PACKS_V3 = ["a", "b", "c", "d", "e", "f", "g"];
const PACK_TILE_SLICES_V3 = { d: [31, 43], e: [43, 53], f: [53, 63] };
const PACK_GLOB_SLICES_V3 = { g: [98, 130] };
const PACK_TILE_SLICES_V2 = { a: [15, 27], b: [27, 31] };
const PACK_GLOB_SLICES_V2 = { b: [70, 82], c: [82, 98] };

// 番值表（encoder_v2._FAN_VALUE：fan.FAN_TABLE + dan_diao=1）
const _FAN_VALUE = Object.assign({}, FAN_TABLE, { dan_diao: 1 });

function rel(p, me) { return ((p - me) % 4 + 4) % 4; }
function seatOf(r, me) { return (me + r) % 4; }

// Python 负索引回绕（hand[-1] → hand[26]）
function pyAt(arr, i) { return arr[i < 0 ? arr.length + i : i]; }

function clip(x, lo, hi) { return Math.min(hi, Math.max(lo, x)); }

// tile_from(s, rk)（rk 为 1 基点数）——可能产生 -1（s=0, rk=0），调用方按 Python 语义处理
function tileFromPy(s, rk) { return s * 9 + rk - 1; }

function turnOf(game) {
  return clip((DECK_SIZE - game.wallLeft() - _OPENING_TILES) / 4.0, 0.0, 12.0);
}

function jiTilesOf(game) {
  return game.ji_tiles instanceof Set && game.ji_tiles.size > 0
    ? game.ji_tiles : new Set([YAOJI]);
}

// ---------------------------------------------------------------------------
// v1（encoder.py::encode_v1，15 通道 / 70 全局维，不可改动口径）
// ---------------------------------------------------------------------------

function waitsOf(game, p) {
  const h = game.hands[p];
  let s = 0;
  for (const x of h) s += x;
  if (s !== (4 - game.melds[p].length) * 3 + 1) return [];
  return winningTiles(h, game.melds[p]);
}

function respondMask(game, p) {
  const mask = new Array(4).fill(0);
  mask[0] = 1;                                              // 过
  if (game._rob_kong !== null && game._rob_kong !== undefined) {
    if (game.canRon(p, game._rob_kong[1])) mask[3] = 1;     // 抢杠胡
    return mask;
  }
  const tile = game.last_discard[1];
  if (game.canRon(p, tile)) mask[3] = 1;
  if (game._canMinggang(p, tile)) mask[2] = 1;
  if (game._canPeng(p, tile)) mask[1] = 1;
  return mask;
}

function actionHeadAndMask(game, me) {
  if (game.phase === PHASE.DISCARD) {
    const p = game.current;
    const mask = new Array(HEAD_DIMS[HEAD_DISCARD]).fill(0);
    for (let t = 0; t < NUM_TILE_TYPES; t++)
      if (game.hands[p][t] > 0) mask[t] = 1;
    if (game.drawn && game.canTsumo(p)) mask[27] = 1;
    return [HEAD_DISCARD, mask];
  }
  if (game.phase === PHASE.SELF_KONG) {
    const mask = new Array(HEAD_DIMS[HEAD_SELF_KONG]).fill(0);
    mask[27] = 1;                                           // 过
    if (game._rob_kong !== null && game._rob_kong !== undefined) {
      const p = game._rob_kong[0];
      if (game.canRon(p, game._rob_kong[1]))
        return [HEAD_RESPOND, respondMask(game, p)];
      return [HEAD_SELF_KONG, mask];
    }
    for (const opt of game._selfKongOptions(game.current))
      mask[opt[1]] = 1;
    return [HEAD_SELF_KONG, mask];
  }
  if (game.phase === PHASE.RESPOND) {
    const p = game.actor();
    return [HEAD_RESPOND, respondMask(game, p)];
  }
  return [HEAD_DISCARD, new Array(HEAD_DIMS[HEAD_DISCARD]).fill(0)];
}

// 返回 { tiles: Float32Array(15*27), glob: Float32Array(70), head, mask }
function encodeV1(game, me) {
  const tiles = new Float32Array(15 * NUM_TILE_TYPES);
  const hand = game.hands[me];

  // -- 自己的手牌（多层阈值特征） ----------------------------------------
  for (let t = 0; t < NUM_TILE_TYPES; t++) {
    const c = hand[t];
    if (c) {
      tiles[0 * 27 + t] = 1.0;
      tiles[4 * 27 + t] = c * 0.25;
      if (c >= 2) tiles[1 * 27 + t] = 1.0;
      if (c >= 3) tiles[2 * 27 + t] = 1.0;
      if (c >= 4) tiles[3 * 27 + t] = 1.0;
    }
  }

  // -- 四家牌河（相对座位） ----------------------------------------------
  for (let p = 0; p < 4; p++) {
    const r = rel(p, me);
    for (const t of game.discards[p]) tiles[(5 + r) * 27 + t] += 0.25;
  }

  // -- 四家副露 ----------------------------------------------------------
  for (let p = 0; p < 4; p++) {
    const r = rel(p, me);
    for (const m of game.melds[p])
      tiles[(9 + r) * 27 + m[1]] += meldIsKong(m[0]) ? 1.0 : 0.75;
  }

  // -- 可见度 / 剩余 ------------------------------------------------------
  const visible = new Float64Array(NUM_TILE_TYPES);
  for (let t = 0; t < NUM_TILE_TYPES; t++) {
    let v = tiles[4 * 27 + t] * 4.0;
    for (let r = 0; r < 4; r++) {
      v += tiles[(5 + r) * 27 + t] * 4.0;
      v += tiles[(9 + r) * 27 + t] * 4.0;
    }
    visible[t] = v;
    tiles[13 * 27 + t] = f32(clip(v, 0.0, 4.0) / 4.0);
    tiles[14 * 27 + t] = f32(clip(4.0 - v, 0.0, 4.0) / 4.0);
  }

  // -- 全局标量 ----------------------------------------------------------
  const g = new Float32Array(70);
  const wallLeft = game.wallLeft();
  g[0] = f32(wallLeft / 108.0);
  g[1] = f32((108.0 - wallLeft) / 108.0);
  g[2] = f32(rel(me, game.dealer) / 3.0);
  g[3] = me === game.dealer ? 1.0 : 0.0;
  g[4] = f32(game.dealer_streak / 5.0);
  g[5 + rel(game.dealer, me)] = 1.0;

  for (let p = 0; p < 4; p++) {
    const r = rel(p, me);
    let kongs = 0;
    for (const m of game.melds[p]) if (meldIsKong(m[0])) kongs += 1;
    g[9 + r] = f32(kongs / 4.0);
    g[13 + r] = f32(game.melds[p].length / 4.0);
    g[17 + r] = f32(Math.min(game.discards[p].length, 24) / 24.0);
    g[21 + r] = game.ever_tenpai[p] ? 1.0 : 0.0;
  }

  let handSum = 0;
  for (const x of hand) handSum += x;
  g[25] = f32(handSum / 14.0);
  g[26] = game.hasDou(me) ? 1.0 : 0.0;
  const wait = waitsOf(game, me);
  g[27] = wait.length ? 1.0 : 0.0;
  g[28] = f32(wait.length / 13.0);
  g[29] = f32((4 - visible[YAOJI]) / 4.0);
  g[30] = f32((hand[YAOJI] + game.meldedCount(me, YAOJI)) / 4.0);
  g[31] = game._first_yaoji_discarded ? 1.0 : 0.0;
  g[32] = game.chongfeng_player === me ? 1.0 : 0.0;
  g[33] = game.chongfeng_player !== null && game.chongfeng_player !== undefined ? 1.0 : 0.0;
  g[34] = game.zeren_player === me ? 1.0 : 0.0;
  g[35] = game.phase === PHASE.SELF_KONG ? 1.0 : 0.0;
  g[36] = game.phase === PHASE.DISCARD ? 1.0 : 0.0;
  g[37] = game.phase === PHASE.RESPOND ? 1.0 : 0.0;
  g[38] = game.actor() === me ? 1.0 : 0.0;

  const base = 39;
  if (game.last_discard !== null && game.last_discard !== undefined) {
    const dP = game.last_discard[0], dT = game.last_discard[1];
    g[base + dT] = 1.0;
    g[base + 27 + rel(dP, me)] = 1.0;
  }

  const [head, mask] = actionHeadAndMask(game, me);
  return { tiles, glob: g, head, mask };
}

// ---------------------------------------------------------------------------
// v2（encoder_v2.py，31 通道 / 98 全局维）
// ---------------------------------------------------------------------------

function encodeTilesV2(tiles, game, me) {
  // -- 15–26：四家牌河最近 1/2/3 张 ---------------------------------------
  for (let p = 0; p < 4; p++) {
    const r = rel(p, me);
    const river = game.discards[p];
    for (let k = 0; k < 3; k++)
      if (river.length >= k + 1)
        tiles[(15 + k * 4 + r) * 27 + river[river.length - 1 - k]] = 1.0;
  }

  // 可见张数（与 v1 同口径，从 tiles 通道重算）
  const visible = new Float64Array(NUM_TILE_TYPES);
  for (let t = 0; t < NUM_TILE_TYPES; t++) {
    let v = tiles[4 * 27 + t] * 4.0;
    for (let r = 0; r < 4; r++) {
      v += tiles[(5 + r) * 27 + t] * 4.0;
      v += tiles[(9 + r) * 27 + t] * 4.0;
    }
    visible[t] = v;
  }

  // -- 27：现物安全度 ------------------------------------------------------
  for (let t = 0; t < NUM_TILE_TYPES; t++) {
    let gb = 0;
    for (let r = 0; r < 4; r++) gb += tiles[(5 + r) * 27 + t] > 0 ? 1.0 : 0.0;
    tiles[27 * 27 + t] = f32(gb / 3.0);
  }

  // -- 28：筋牌安全度（同花色 ±3） -----------------------------------------
  for (let s = 0; s < 3; s++) {
    for (let rk = 0; rk < 9; rk++) {
      let acc = 0;
      if (rk < 6) acc += visible[s * 9 + rk + 3];   // acc[:, :-3] += vis3[:, 3:]
      if (rk >= 3) acc += visible[s * 9 + rk - 3];  // acc[:, 3:] += vis3[:, :-3]
      tiles[28 * 27 + s * 9 + rk] = f32(clip(acc, 0.0, 4.0) / 4.0);
    }
  }

  // -- 29：生牌危险度 ------------------------------------------------------
  for (let t = 0; t < NUM_TILE_TYPES; t++)
    tiles[29 * 27 + t] = f32(1.0 - visible[t] / 4.0);

  // -- 30：对手潜在需求 ----------------------------------------------------
  const meldSuit = [0.0, 0.0, 0.0];
  const riverSuit = [0.0, 0.0, 0.0];
  for (let p = 0; p < 4; p++) {
    if (p === me) continue;
    for (const m of game.melds[p]) meldSuit[suitOf(m[1])] += 1.0;
    for (const t of game.discards[p]) riverSuit[suitOf(t)] += 1.0;
  }
  const msSum = meldSuit[0] + meldSuit[1] + meldSuit[2];
  const rsSum = riverSuit[0] + riverSuit[1] + riverSuit[2];
  const d1 = Math.max(1.0, msSum);
  const d2 = Math.max(1.0, rsSum);
  const w = [0, 1, 2].map(s => f32(meldSuit[s] / d1));
  for (let s = 0; s < 3; s++)
    w[s] = f32(w[s] + f32(0.5 * f32(riverSuit[s] / d2)));
  const wMax = Math.max(w[0], w[1], w[2]);
  const d3 = f32(Math.max(1e-6, wMax));
  for (let s = 0; s < 3; s++) w[s] = f32(w[s] / d3);
  for (let t = 0; t < NUM_TILE_TYPES; t++) tiles[30 * 27 + t] = w[suitOf(t)];
}

function maxReachableFan(game, me) {
  const hand = game.hands[me].slice();
  const melds = game.melds[me];
  let s = 0;
  for (const x of hand) s += x;
  if (s !== (4 - melds.length) * 3 + 1) return 0.0;
  const waits = winningTiles(hand, melds);
  if (!waits.length) return 0.0;
  let best = 0.0;
  for (const t of waits) {
    const h = hand.slice();
    h[t] += 1;
    const types = game._winTypeSet(h, melds, t);
    let val = 0;
    for (const tp of types) if (tp !== "ping_hu") val += (_FAN_VALUE[tp] || 0);
    if (!val) val = _FAN_VALUE["ping_hu"];
    best = Math.max(best, val);
  }
  return best;
}

function encodeGlobV2(g, game, me) {
  const hand = game.hands[me];
  const nMeldsMe = game.melds[me].length;
  const wallLeft = game.wallLeft();
  const turn = clip((DECK_SIZE - wallLeft - _OPENING_TILES) / 4.0, 0.0, 12.0);

  // -- 70–73：各家估计向听数下界 / 6 ---------------------------------------
  for (let p = 0; p < 4; p++) {
    const r = rel(p, me);
    const river = game.discards[p];
    const counts = new Map();
    for (const t of river) counts.set(t, (counts.get(t) || 0) + 1);
    let pairKinds = 0;
    for (const v of counts.values()) if (v >= 2) pairKinds += 1;
    let adj = 0;
    for (const a of counts.keys()) {
      for (const dr of [1, -1]) {
        const rr = (a % 9 + 1) + dr;           // rank_of 1 基，照抄 Python
        if (rr >= 0 && rr < 9 && counts.has(tileFromPy(suitOf(a), rr))) adj += 1;
      }
    }
    adj = Math.floor(adj / 2);
    const est = Math.min(3, pairKinds + adj);
    const lb = clip(4 - game.melds[p].length - est, 0.0, 6.0);
    g[70 + r] = f32(lb / 6.0);
  }

  // -- 74–77：各家已亮明鸡数 / 4 -------------------------------------------
  const perPlayer = [0.0, 0.0, 0.0, 0.0];
  for (const ev of game.ji_events) perPlayer[ev[0] % 4] += 1.0;
  for (const ev of game.claimed_ji) perPlayer[ev[1] % 4] += 1.0;
  for (let p = 0; p < 4; p++)
    g[74 + rel(p, me)] = f32(Math.min(perPlayer[p], 4.0) / 4.0);

  // -- 78–81：各家当下听牌置信 ---------------------------------------------
  for (let p = 0; p < 4; p++) {
    const conf = 0.2 * game.melds[p].length + 0.05 * turn;
    g[78 + rel(p, me)] = f32(clip(conf, 0.0, 1.0));
  }

  // -- 82–85：巡目与阶段 ----------------------------------------------------
  g[82] = f32(turn / 12.0);
  if (turn <= 4) g[83] = 1.0;
  else if (turn <= 9) g[84] = 1.0;
  else g[85] = 1.0;

  // -- 86–93：八个番型可行性 ------------------------------------------------
  const allCounts = new Float32Array(NUM_TILE_TYPES);
  for (let t = 0; t < NUM_TILE_TYPES; t++) allCounts[t] = hand[t];
  for (const m of game.melds[me]) allCounts[m[1]] += 3.0;
  let total = 0;
  for (let t = 0; t < NUM_TILE_TYPES; t++) total += allCounts[t];
  if (!total) total = 1.0;

  let nKongs = 0;
  for (const m of game.melds[me]) if (meldIsKong(m[0])) nKongs += 1;
  let nTripletsHand = 0;
  for (let t = 0; t < NUM_TILE_TYPES; t++) if (hand[t] >= 3) nTripletsHand += 1;
  const suitTot = [0.0, 0.0, 0.0];
  for (let s = 0; s < 3; s++) {
    let st = 0;
    for (let i = 0; i < 9; i++) st += allCounts[s * 9 + i];
    suitTot[s] = st;
  }
  g[86] = f32(Math.min(1.0, (nKongs + nTripletsHand) / 2.0));           // 豆
  g[87] = f32(Math.max(suitTot[0], suitTot[1], suitTot[2]) / total);    // 清一色
  let nPairsHand = 0;
  for (let t = 0; t < NUM_TILE_TYPES; t++) if (hand[t] === 2) nPairsHand += 1;
  g[88] = nMeldsMe === 0 ? f32(nPairsHand / 7.0) : 0.0;                 // 七对
  g[89] = f32(Math.min(1.0, (nTripletsHand + nMeldsMe) / 4.0));         // 大对子
  g[90] = f32(1.0 / (1.0 + nTripletsHand + nKongs));                    // 平胡
  // 将对：照抄 Python 代码 `rank_of(t) in (1,4,7)`（rank_of 1 基 → 点数 1/4/7）
  let jiang = 0.0;
  for (let t = 0; t < NUM_TILE_TYPES; t++) {
    const rk = t % 9 + 1;
    if (rk === 1 || rk === 4 || rk === 7) jiang += allCounts[t];
  }
  g[91] = f32(jiang / total);
  g[92] = nMeldsMe === 0 ? 1.0 : 0.0;                                   // 门前清
  let bestRun = 0.0;
  for (let s = 0; s < 3; s++) {
    let seg = 0;
    for (let i = 0; i < 9; i++) if (allCounts[s * 9 + i] > 0) seg += 1;
    bestRun = Math.max(bestRun, seg / 9.0);
  }
  g[93] = f32(bestRun);                                                 // 青龙

  // -- 94–97 ---------------------------------------------------------------
  g[94] = f32(maxReachableFan(game, me) / 30.0);
  const tenpai = g[27] > 0.5;
  g[95] = f32((1.0 - wallLeft / DECK_SIZE) * (tenpai ? 0.0 : 1.0));
  const othersJi = perPlayer[0] + perPlayer[1] + perPlayer[2] + perPlayer[3]
    - perPlayer[me];
  g[96] = (tenpai ? 0.0 : 1.0) * (othersJi > 0 ? 1.0 : 0.0);
  const jiTiles = jiTilesOf(game);
  let jiVal = 0.0;
  for (const t of jiTiles) {
    jiVal += hand[t];
    if (hand[t] >= 3 || game.meldedCount(me, t) >= 3) jiVal += hand[t];
  }
  g[97] = f32(Math.min(1.0, jiVal / 8.0));
}

// ---------------------------------------------------------------------------
// v3（encoder_v3.py，63 通道 / 130 全局维）
// ---------------------------------------------------------------------------

function visibleCounts(game, me) {
  // _visible：自己手牌 + 四家牌河 + 四家副露（碰 3 / 杠 4）
  const vis = new Float64Array(NUM_TILE_TYPES);
  const hand = game.hands[me];
  for (let t = 0; t < NUM_TILE_TYPES; t++) vis[t] = hand[t];
  for (let p = 0; p < 4; p++) {
    for (const t of game.discards[p]) vis[t] += 1.0;
    for (const m of game.melds[p]) vis[m[1]] += meldIsKong(m[0]) ? 4.0 : 3.0;
  }
  return vis;
}

function handShape(hand) {
  // _hand_shape：(刻子数, 对子数, 搭子数, 孤张数)
  let triplets = 0.0, pairs = 0.0;
  for (let t = 0; t < NUM_TILE_TYPES; t++) {
    if (hand[t] >= 3) triplets += 1.0;
    if (hand[t] === 2) pairs += 1.0;
  }
  let partials = 0.0, lone = 0.0;
  for (let s = 0; s < 3; s++) {
    for (let rk = 0; rk < 9; rk++) {          // rk 0 基下标（照抄 Python）
      const t = s * 9 + rk;
      if (hand[t] <= 0) continue;
      const nb1 = [];
      for (const d of [-1, 1]) if (rk + d >= 0 && rk + d < 9) nb1.push(s * 9 + rk + d);
      const nb2 = [];
      for (const d of [-2, 2]) if (rk + d >= 0 && rk + d < 9) nb2.push(s * 9 + rk + d);
      const near = nb1.some(u => hand[u] > 0);
      const far = nb2.some(u => hand[u] > 0);
      if (near || far) {
        if (far && !near) partials += 0.5;
        else if (near && !nb1.some(u => u < t && hand[u] > 0)) partials += 1.0;
      } else if (hand[t] === 1) lone += 1.0;
    }
  }
  return [triplets, pairs, partials, lone];
}

function encodeTilesV3(tiles, game, me) {
  const hand = game.hands[me];

  // -- 31–42：四家牌河第 4/5/6 近张 ----------------------------------------
  for (let p = 0; p < 4; p++) {
    const r = rel(p, me);
    const river = game.discards[p];
    for (const k of [3, 4, 5])
      if (river.length >= k + 1)
        tiles[(31 + (k - 3) * 4 + r) * 27 + river[river.length - 1 - k]] = 1.0;
  }

  const jiTiles = jiTilesOf(game);

  // -- 43–46：四家开出横鸡轮次数 / 2（整通道广播） --------------------------
  const opened = [0.0, 0.0, 0.0, 0.0];
  for (const ev of game.hengji_events)
    if (String(ev[2]) === "open") opened[ev[0] % 4] += 1.0;
  for (let p = 0; p < 4; p++) {
    const val = f32(Math.min(opened[p] / 2.0, 1.0));
    for (let t = 0; t < NUM_TILE_TYPES; t++) tiles[(43 + rel(p, me)) * 27 + t] = val;
  }

  // -- 47：当前是否还有横鸡轮开放（整通道广播） -----------------------------
  if (game.hengjiActive)
    for (let t = 0; t < NUM_TILE_TYPES; t++) tiles[47 * 27 + t] = 1.0;

  // -- 48/49：我现在打这张牌会记成什么鸡 ------------------------------------
  for (let t = 0; t < NUM_TILE_TYPES; t++) {
    const nm = game.jiNameIfDiscarded(me, t);
    if (nm === "横鸡") tiles[48 * 27 + t] = 1.0;
    else if (nm === "冲锋鸡") tiles[49 * 27 + t] = 1.0;
  }

  // -- 50：本局鸡牌种 -------------------------------------------------------
  for (const t of jiTiles) tiles[50 * 27 + t] = 1.0;

  // -- 51：手牌里的鸡牌张数 / 4 ---------------------------------------------
  for (const t of jiTiles)
    tiles[51 * 27 + t] = f32(Math.min(hand[t] / 4.0, 1.0));

  // -- 52：被别家碰/杠走的鸡牌张数 / 4 --------------------------------------
  if (game._ji_val_by_meld instanceof Map) {
    for (const key of game._ji_val_by_meld.keys()) {
      const t = parseInt(String(key).split(",")[1], 10);
      tiles[52 * 27 + t] = f32(Math.min(tiles[52 * 27 + t] + 0.25, 1.0));
    }
  }

  const visible = visibleCounts(game, me);
  const remain = new Float64Array(NUM_TILE_TYPES);
  for (let t = 0; t < NUM_TILE_TYPES; t++) remain[t] = clip(4.0 - visible[t], 0.0, 4.0);
  const turn = turnOf(game);

  // -- 53：放铳风险 ---------------------------------------------------------
  let conf = 0.0;
  for (const r of [1, 2, 3]) {
    const q = seatOf(r, me);
    const c = clip(0.2 * game.melds[q].length + 0.05 * turn, 0.0, 1.0);
    if (c > conf) conf = c;
  }
  for (let t = 0; t < NUM_TILE_TYPES; t++)
    tiles[53 * 27 + t] = f32(f32(conf) * f32(1.0 - f32(visible[t] / 4.0)));

  // -- 54–57：手牌结构 ------------------------------------------------------
  // 注意：Python 用 1 基 rank_of + tile_from，rk+d 允许 0（→ tile_from(s,0)=s*9-1，
  // 跨花色邻居；s=0 时为 -1，Python 负索引回绕到 hand[26]），rank 8/9 无右邻。
  for (let t = 0; t < NUM_TILE_TYPES; t++) {
    const c = hand[t];
    if (c <= 0) continue;
    const s = Math.floor(t / 9), rk = t % 9 + 1;
    const nb1 = [];
    for (const d of [-1, 1]) if (rk + d >= 0 && rk + d < 9) nb1.push(tileFromPy(s, rk + d));
    const nb2 = [];
    for (const d of [-2, 2]) if (rk + d >= 0 && rk + d < 9) nb2.push(tileFromPy(s, rk + d));
    if (nb2.some(u => pyAt(hand, u) > 0)) tiles[54 * 27 + t] = 1.0;   // 隔张搭子
    if (nb1.some(u => pyAt(hand, u) > 0)) tiles[55 * 27 + t] = 1.0;   // 相邻搭子
    let run = 1;
    let rr = rk - 1;
    while (rr >= 0 && pyAt(hand, tileFromPy(s, rr)) > 0) { run += 1; rr -= 1; }
    rr = rk + 1;
    while (rr < 9 && pyAt(hand, tileFromPy(s, rr)) > 0) { run += 1; rr += 1; }
    tiles[56 * 27 + t] = f32(Math.min(run / 3.0, 1.0));               // 连续段 / 3
    if (c === 1 && !nb1.concat(nb2).some(u => pyAt(hand, u) > 0))
      tiles[57 * 27 + t] = 1.0;                                       // 孤张
  }

  // -- 58/59：最近 3 手内被谁打过（1, 2/3, 1/3 衰减） -----------------------
  for (let r = 0; r < 4; r++) {
    const river = game.discards[seatOf(r, me)];
    for (let k = 0; k < 3; k++) {
      if (river.length >= k + 1) {
        const w = 1.0 - k / 3.0;
        const t = river[river.length - 1 - k];
        if (w > tiles[58 * 27 + t]) tiles[58 * 27 + t] = f32(w);
        if (r === 3 && w > tiles[59 * 27 + t]) tiles[59 * 27 + t] = f32(w);
      }
    }
  }

  // -- 60：自己是否已报叫 ---------------------------------------------------
  if (game.baojiao && game.baojiao[me])
    for (let t = 0; t < NUM_TILE_TYPES; t++) tiles[60 * 27 + t] = 1.0;

  // -- 61：严格现物 ---------------------------------------------------------
  for (let t = 0; t < NUM_TILE_TYPES; t++) {
    let strict = 1.0;
    for (const r of [1, 2, 3]) strict *= tiles[(5 + r) * 27 + t] > 0 ? 1.0 : 0.0;
    tiles[61 * 27 + t] = strict;
  }

  // -- 62：放铳风险 × 还摸得到 ----------------------------------------------
  for (let t = 0; t < NUM_TILE_TYPES; t++)
    tiles[62 * 27 + t] = f32(tiles[53 * 27 + t] * f32(remain[t] / 4.0));
}

function encodeGlobV3(g, game, me) {
  const hand = game.hands[me];
  const turn = turnOf(game);
  const wallLeft = game.wallLeft();
  const meldsMe = game.melds[me];
  const nMeldsMe = meldsMe.length;

  // -- 98–101：四家因鸡被碰/杠次数 / 4 --------------------------------------
  const claim = [0.0, 0.0, 0.0, 0.0];
  for (const ev of game.claimed_ji) claim[ev[1] % 4] += 1.0;
  for (let p = 0; p < 4; p++)
    g[98 + rel(p, me)] = f32(Math.min(claim[p] / 4.0, 1.0));

  // -- 102–105：四家打出鸡牌次数 / 4 ----------------------------------------
  const evCnt = [0.0, 0.0, 0.0, 0.0];
  for (const ev of game.ji_events) evCnt[ev[0] % 4] += 1.0;
  for (let p = 0; p < 4; p++)
    g[102 + rel(p, me)] = f32(Math.min(evCnt[p] / 4.0, 1.0));

  // -- 106–108：横鸡轮总体状态 ----------------------------------------------
  const species = typeof game.hengjiSpecies === "function" ? game.hengjiSpecies() : [];
  g[106] = f32(Math.min(species.length / 2.0, 1.0));
  g[107] = game.hengji_opened ? 1.0 : 0.0;
  const starter = game.hengji_starter;
  g[108] = (starter !== null && starter !== undefined)
    ? f32(rel(starter, me) / 3.0) : 0.0;

  // -- 109–112：四家是否已报叫 ----------------------------------------------
  if (game.baojiao)
    for (let p = 0; p < 4; p++) g[109 + rel(p, me)] = game.baojiao[p] ? 1.0 : 0.0;

  // -- 113–117：手牌结构 ----------------------------------------------------
  const [triplets, pairs, partials, lone] = handShape(hand);
  g[113] = f32(clip((3 * triplets + 2 * pairs + partials) / 8.0, 0.0, 1.0));
  g[114] = f32(Math.min(partials / 6.0, 1.0));
  g[115] = f32(Math.min(pairs / 6.0, 1.0));
  g[116] = f32(Math.min(triplets / 4.0, 1.0));
  g[117] = f32(Math.min(lone / 6.0, 1.0));

  // -- 118–121：副露速度 ----------------------------------------------------
  for (let p = 0; p < 4; p++)
    g[118 + rel(p, me)] = f32(Math.min(1.0, game.melds[p].length / Math.max(1.0, turn)));

  // -- 122–123：听牌质量 ----------------------------------------------------
  let handSum = 0;
  for (const x of hand) handSum += x;
  let waits = [];
  if (handSum === (4 - nMeldsMe) * 3 + 1)
    waits = winningTiles(hand, meldsMe);
  const visible = visibleCounts(game, me);
  const remain = new Float64Array(NUM_TILE_TYPES);
  for (let t = 0; t < NUM_TILE_TYPES; t++) remain[t] = clip(4.0 - visible[t], 0.0, 4.0);
  if (waits.length) {
    let mass = 0.0;
    for (const t of waits) mass += remain[t];
    g[122] = f32(Math.min(mass / 13.0, 1.0));
    g[123] = f32(Math.min((mass / Math.max(1, waits.length)) / 4.0, 1.0));
  }

  // -- 124：全场已亮明张数 / 64 ---------------------------------------------
  let shown = 0.0;
  for (let p = 0; p < 4; p++) shown += game.discards[p].length;
  for (let p = 0; p < 4; p++)
    for (const m of game.melds[p]) shown += meldIsKong(m[0]) ? 4.0 : 3.0;
  g[124] = f32(Math.min(shown / 64.0, 1.0));

  // -- 125：场上最高听牌置信 ------------------------------------------------
  let maxConf = 0.0;
  for (const r of [1, 2, 3]) {
    const q = seatOf(r, me);
    const c = clip(0.2 * game.melds[q].length + 0.05 * turn, 0.0, 1.0);
    if (c > maxConf) maxConf = c;
  }
  g[125] = f32(maxConf);

  // -- 126：可达最大番值 × 是否已听 -----------------------------------------
  const tenpai = g[27] > 0.5;
  g[126] = f32(Math.min(maxReachableFan(game, me) / 30.0, 1.0) * (tenpai ? 1.0 : 0.0));

  // -- 127：距黄庄剩余空间 × 未听牌 -----------------------------------------
  g[127] = f32((wallLeft / DECK_SIZE) * (tenpai ? 0.0 : 1.0));

  // -- 128：鸡相关事件总数 / 8 ----------------------------------------------
  const nJi = game.ji_events.length + game.claimed_ji.length
    + game.hengji_events.length;
  g[128] = f32(Math.min(nJi / 8.0, 1.0));

  // -- 129：手牌 + 副露总张数 / 14 ------------------------------------------
  g[129] = f32(Math.min((handSum + 3.0 * nMeldsMe) / 14.0, 1.0));
}

// ---------------------------------------------------------------------------
// 包掩码与入口
// ---------------------------------------------------------------------------

function parsePacksV3(spec) {
  if (spec === null || spec === undefined) return ALL_PACKS_V3.slice();
  let chars;
  if (typeof spec === "string")
    chars = spec.toLowerCase().split("").filter(c => ALL_PACKS_V3.includes(c));
  else
    chars = spec.map(c => String(c).toLowerCase()).filter(c => ALL_PACKS_V3.includes(c));
  return ALL_PACKS_V3.filter(p => chars.includes(p));
}

function applyPackMask(tiles, g, tileSlices, globSlices, packs) {
  for (const name of Object.keys(tileSlices)) {
    if (!packs.includes(name)) {
      const [lo, hi] = tileSlices[name];
      for (let ch = lo; ch < hi; ch++)
        for (let t = 0; t < NUM_TILE_TYPES; t++) tiles[ch * 27 + t] = 0.0;
    }
  }
  for (const name of Object.keys(globSlices)) {
    if (!packs.includes(name)) {
      const [lo, hi] = globSlices[name];
      for (let i = lo; i < hi; i++) g[i] = 0.0;
    }
  }
}

// packs: 语料全开 → 留默认。返回 { tiles, glob, head, mask }
function encodeV3(game, player, packs) {
  const spec = parsePacksV3(packs);
  const v2Packs = ["a", "b", "c"].filter(p => spec.includes(p));

  // v1 底座（0–14 通道 / 0–69 维）+ head/mask
  const base = encodeV1(game, player);

  // v2（15–30 通道 / 70–97 维）
  const tiles = new Float32Array(NUM_CHANNELS_V3 * NUM_TILE_TYPES);
  tiles.set(base.tiles, 0);
  encodeTilesV2(tiles, game, player);
  const g = new Float32Array(GLOBAL_DIM_V3);
  g.set(base.glob, 0);
  encodeGlobV2(g, game, player);
  applyPackMask(tiles, g, PACK_TILE_SLICES_V2, PACK_GLOB_SLICES_V2, v2Packs);

  // v3（31–62 通道 / 98–129 维）
  encodeTilesV3(tiles, game, player);
  encodeGlobV3(g, game, player);
  applyPackMask(tiles, g, PACK_TILE_SLICES_V3, PACK_GLOB_SLICES_V3, spec);

  return { tiles, glob: g, head: base.head, mask: base.mask };
}

module.exports = { encodeV3 };
