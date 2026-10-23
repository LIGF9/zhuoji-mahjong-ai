// A2/A3 差分对拍验收台（Node）。
// 契约：
//   - web/standalone/engine/dushan.js 必须导出 class DushanGame：
//       new DushanGame({ config, wall, dealer })
//         config: Python DushanConfig 的 dataclasses.asdict 全字段对象
//         wall:   初始牌墙（发牌顺序数组），引擎不做洗牌
//         dealer: 庄家座位
//       .actor()          -> 0..3
//       .legalActions()   -> [{kind, tile}]，与 Python legal_actions 顺序一致
//       .step({kind,tile})-> 推进一步
//       .dumpState()      -> 与 Python dump_state 相同形状的全量字段对象：
//            list/tuple -> 数组；set -> {__set__:[...]}; dict -> {__dict__:[[k,v]...]};
//            Enum -> str(value)；排除 rng/cfg；其余字段名与 Python __dict__ 完全一致
//       .result           -> 终局结算 dict（形状与 Python 相同）
//   - web/standalone/engine/encoder_v3.js 导出 encodeV3(game, player) ->
//       { tiles: Float32Array(63*27), glob: Float32Array(130), head, mask: number[] }
// 用法：node web/standalone/replay_harness.js [corpus.json]
"use strict";
const fs = require("fs");
const path = require("path");

const corpusPath = process.argv[2]
  || path.join(__dirname, "..", "..", "reports", "standalone_corpus.json");
const corpus = JSON.parse(fs.readFileSync(corpusPath, "utf8"));

const engineDir = path.join(__dirname, "engine");
let Engine = null, EncV3 = null;
try {
  Engine = require(path.join(engineDir, "dushan.js"));
} catch (e) {
  console.error("引擎模块加载失败：", e.message);
  console.error("（web/standalone/engine/dushan.js 尚未就位）");
  process.exit(2);
}
try {
  EncV3 = require(path.join(engineDir, "encoder_v3.js"));
} catch (_) { /* 编码器未就位时跳过 obs 对拍 */ }

const FLOAT_TOL = 1e-9;
let totalMismatch = 0;

function deepCmp(a, b, p, errs, depth) {
  if (errs.length >= 8) return;
  if (typeof a === "number" && typeof b === "number") {
    if (!(Math.abs(a - b) <= FLOAT_TOL
          || (a === 0 && b === 0) || Object.is(a, b))) {
      errs.push(`${p}: ${a} vs ${b}`);
    }
    return;
  }
  if (Array.isArray(a) && Array.isArray(b)) {
    if (a.length !== b.length) { errs.push(`${p}: len ${a.length} vs ${b.length}`); return; }
    for (let i = 0; i < a.length; i++) deepCmp(a[i], b[i], `${p}[${i}]`, errs, depth + 1);
    return;
  }
  if (a && b && typeof a === "object" && typeof b === "object") {
    const ka = Object.keys(a), kb = Object.keys(b);
    if (ka.length !== kb.length || ka.some(k => !(k in b))) {
      errs.push(`${p}: keys ${JSON.stringify(ka)} vs ${JSON.stringify(kb)}`);
      return;
    }
    for (const k of ka) deepCmp(a[k], b[k], `${p}.${k}`, errs, depth + 1);
    return;
  }
  if (a !== b) errs.push(`${p}: ${JSON.stringify(a)} vs ${JSON.stringify(b)}`);
}

function f32(b64) {
  return Buffer.from(b64, "base64");
}

let allPass = true;
for (let gi = 0; gi < corpus.games.length; gi++) {
  const rec = corpus.games[gi];
  const g = new Engine.DushanGame({
    config: rec.config, wall: rec.wall, dealer: rec.dealer,
  });
  let pass = true, stepErrs = 0, obsChecked = 0, logitsChecked = 0;
  const obsByStep = new Map(rec.obs.map(o => [o.step, o]));

  for (let si = 0; si < rec.steps.length; si++) {
    const st = rec.steps[si];
    // 1) 合法动作表（顺序敏感）
    const jsLegal = g.legalActions().map(a => [a.kind, a.tile]);
    const errs = [];
    deepCmp(jsLegal, st.legal, "legal", errs, 0);
    // 2) 状态全量 dump
    deepCmp(g.dumpState(), st.state, "state", errs, 0);
    if (errs.length) {
      pass = false; stepErrs++;
      totalMismatch++;
      if (stepErrs <= 2) {
        console.log(`  g${gi} s${si} 不一致（前 8 条）：`);
        for (const e of errs) console.log("    -", e);
      }
    }
    // 3) 编码器 + ORT logits 对拍（采样步）
    const ob = obsByStep.get(si);
    if (ob && EncV3) {
      const o = EncV3.encodeV3(g, g.actor());
      const e2 = [];
      const headOk = o.head === ob.head;
      if (!headOk) e2.push(`head ${o.head} vs ${ob.head}`);
      deepCmp(Array.from(o.mask), ob.mask, "mask", e2, 0);
      const tBuf = Buffer.from(o.tiles.buffer, o.tiles.byteOffset, o.tiles.byteLength);
      if (!tBuf.equals(f32(ob.tiles))) {
        const pv = new Float32Array(f32(ob.tiles));
        let md = 0;
        for (let i = 0; i < pv.length; i++) md = Math.max(md, Math.abs(pv[i] - o.tiles[i]));
        if (md > 1e-6) e2.push(`tiles maxdiff=${md.toExponential(2)}`);
      }
      const gBuf = Buffer.from(o.glob.buffer, o.glob.byteOffset, o.glob.byteLength);
      if (!gBuf.equals(f32(ob.glob))) {
        const pv = new Float32Array(f32(ob.glob));
        let md = 0;
        for (let i = 0; i < pv.length; i++) md = Math.max(md, Math.abs(pv[i] - o.glob[i]));
        if (md > 1e-6) e2.push(`glob maxdiff=${md.toExponential(2)}`);
      }
      obsChecked++;
      if (e2.length) {
        pass = false;
        if (obsChecked <= 3) {
          console.log(`  g${gi} s${si} 观测不一致：`, e2.slice(0, 4));
        }
      }
      // 4) logits 对拍（JS 编码 → 本地 ORT（可选挂载）→ 与 python 基准）
      if (Engine.ortInfer) {
        const lg = Engine.ortInfer(o);
        let md = 0;
        for (let h = 0; h < 4; h++) {
          const pv = new Float32Array(f32(ob.logits[h]));
          for (let i = 0; i < pv.length; i++) {
            md = Math.max(md, Math.abs(pv[i] - lg[h][i]));
          }
        }
        logitsChecked++;
        if (md > 1e-3) {
          pass = false;
          console.log(`  g${gi} s${si} logits maxdiff=${md.toExponential(2)}`);
        }
      }
    }
    g.step({ kind: st.act[0], tile: st.act[1] });
  }
  // 5) 终局结算
  const resErrs = [];
  const pyRes = rec.result && rec.result.__dict__
    ? Object.fromEntries(rec.result.__dict__) : rec.result;
  deepCmp(g.result, pyRes, "result", resErrs, 0);
  if (resErrs.length) {
    pass = false; totalMismatch++;
    console.log(`  g${gi} 结算不一致：`, resErrs.slice(0, 4));
  }
  console.log(`game ${gi} seed=${rec.seed}: ${pass ? "PASS" : "FAIL"} `
    + `(steps=${rec.steps.length}, obs=${obsChecked}, logits=${logitsChecked})`);
  allPass = allPass && pass;
}

console.log(allPass ? "== 全部 PASS ==" : `== FAIL（不一致点 ${totalMismatch} 处）==`);
process.exit(allPass ? 0 : 1);
