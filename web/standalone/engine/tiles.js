"use strict";
// 牌张基础定义（对应 zhuoji/tiles.py）。
// 0..8 万 W1..W9；9..17 条 T1..T9；18..26 筒 B1..B9。

const NUM_TILE_TYPES = 27;
const TILES_PER_TYPE = 4;
const DECK_SIZE = NUM_TILE_TYPES * TILES_PER_TYPE;

const SUITS = ["W", "T", "B"];
const SUIT_CN = { W: "万", T: "条", B: "筒" };

// 幺鸡 —— 一条
const YAOJI = 9;

function suitOf(t) { return Math.floor(t / 9); }
function rankOf(t) { return t % 9 + 1; }
function tileFrom(suit, rank) { return suit * 9 + (rank - 1); }
function tileName(t) { return SUITS[suitOf(t)] + rankOf(t); }
function tileCn(t) {
  const cn = "一二三四五六七八九"[rankOf(t) - 1];
  return cn + SUIT_CN[SUITS[suitOf(t)]];
}
function fullDeck() {
  const deck = [];
  for (let t = 0; t < NUM_TILE_TYPES; t++)
    for (let i = 0; i < TILES_PER_TYPE; i++) deck.push(t);
  return deck;
}
function tilesToCounts(tiles) {
  const counts = new Array(NUM_TILE_TYPES).fill(0);
  for (const t of tiles) counts[t] += 1;
  return counts;
}
function countsToTiles(counts) {
  const out = [];
  counts.forEach((c, t) => { for (let i = 0; i < c; i++) out.push(t); });
  return out;
}

module.exports = {
  NUM_TILE_TYPES, TILES_PER_TYPE, DECK_SIZE, SUITS, SUIT_CN, YAOJI,
  suitOf, rankOf, tileFrom, tileName, tileCn, fullDeck,
  tilesToCounts, countsToTiles,
};
