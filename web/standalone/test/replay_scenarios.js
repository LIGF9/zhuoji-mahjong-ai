"use strict";
// 一炮多响合成场景对拍（JS 侧重放）。
const fs = require("fs");
const path = require("path");
const { DushanGame } = require("C:/Users/LIGF/WorkBuddy/2026-09-30-12-49-43/zhuoji-mahjong/web/standalone/engine/dushan.js");

const FLOAT_TOL = 1e-9;
let fail = 0;

function deepCmp(a, b, p, errs) {
  if (errs.length >= 8) return;
  if (typeof a === "number" && typeof b === "number") {
    if (!(Math.abs(a - b) <= FLOAT_TOL || Object.is(a, b))) errs.push(`${p}: ${a} vs ${b}`);
    return;
  }
  if (Array.isArray(a) && Array.isArray(b)) {
    if (a.length !== b.length) { errs.push(`${p}: len`); return; }
    for (let i = 0; i < a.length; i++) deepCmp(a[i], b[i], `${p}[${i}]`, errs);
    return;
  }
  if (a && b && typeof a === "object" && typeof b === "object") {
    const ka = Object.keys(a), kb = Object.keys(b);
    if (ka.length !== kb.length || ka.some(k => !(k in b))) {
      errs.push(`${p}: keys ${JSON.stringify(ka)} vs ${JSON.stringify(kb)}`);
      return;
    }
    for (const k of ka) deepCmp(a[k], b[k], `${p}.${k}`, errs);
    return;
  }
  if (a !== b) errs.push(`${p}: ${JSON.stringify(a)} vs ${JSON.stringify(b)}`);
}

const scenarios = JSON.parse(fs.readFileSync(
  path.join(__dirname, "scenarios_multi_hu.json"), "utf8"));

for (const [name, sc] of Object.entries(scenarios)) {
  const g = new DushanGame({ config: sc.config, wall: sc.wall, dealer: sc.dealer });
  // 应用与 Python 侧相同的状态覆盖
  const su = sc.setup;
  g.hands = su.hands.map(a => a.slice());
  g.melds = [[], [], [], []];
  g.discards = [[], [], [], []];
  g.ever_tenpai = [false, false, false, false];
  g.pass_lock = [new Set(), new Set(), new Set(), new Set()];
  g.gang_discard = [false, false, false, false];
  g.drawn = su.drawn;
  g.current = su.current;
  g.phase = su.phase;
  g.last_discard = null;
  g.last_discard_was_gang = false;
  g.baojiao = [false, false, false, false];
  g.first_discard_done = [false, false, false, false];
  g.wall_pos = su.wall_pos;

  let pass = true;
  sc.steps.forEach((st, si) => {
    const errs = [];
    deepCmp(g.legalActions().map(a => [a.kind, a.tile]), st.legal, "legal", errs);
    deepCmp(g.dumpState(), st.state, "state", errs);
    if (errs.length) {
      pass = false;
      console.log(`  ${name} s${si}:`, errs.slice(0, 4));
    }
    g.step({ kind: st.act[0], tile: st.act[1] });
  });
  const pyRes = sc.result && sc.result.__dict__
    ? Object.fromEntries(sc.result.__dict__) : sc.result;
  const resErrs = [];
  deepCmp(g.result, pyRes, "result", resErrs);
  if (resErrs.length) {
    pass = false;
    console.log(`  ${name} result:`, resErrs.slice(0, 4));
  }
  console.log(`${name}: ${pass ? "PASS" : "FAIL"}`);
  if (!pass) fail++;
}
console.log(fail === 0 ? "== 场景全部 PASS ==" : `== FAIL ${fail} ==`);
process.exit(fail === 0 ? 0 : 1);
