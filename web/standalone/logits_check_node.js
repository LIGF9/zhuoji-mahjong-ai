// 真模型 logits 对拍（Node，ort.min.js + ort-wasm-simd-threaded.wasm）。
// 语料 reports/standalone_corpus.json 的 obs 带 Python ONNX logits 基准（base64 f32）。
// 取前 20 条 obs：JS 编码 → 本地 ORT 推理 → 与基准逐位比（判据 1e-3）。
// 用法：node web/standalone/logits_check_node.js [maxObs]
"use strict";
const path = require("path");
const fs = require("fs");
const ort = require(path.join(__dirname, "lib", "ort.min.js"));
const Enc = require(path.join(__dirname, "engine", "encoder_v3.js"));
const Eng = require(path.join(__dirname, "engine", "dushan.js"));

const MAX_OBS = parseInt(process.argv[2], 10) || 20;
const TOL = 1e-3;
const HEADS = ["discard", "self_kong", "respond", "value"];

async function main() {
  ort.env.wasm.numThreads = 1;
  ort.env.wasm.wasmBinary = fs.readFileSync(
    path.join(__dirname, "lib", "ort-wasm-simd-threaded.wasm"));
  const model = new Uint8Array(fs.readFileSync(
    path.join(__dirname, "..", "..", "models", "standalone",
      "dushan_m2_c4_it2280.onnx")));
  const session = await ort.InferenceSession.create(model,
    { executionProviders: ["wasm"] });
  console.log("[logits] session ok  inputs=", session.inputNames.join(","),
    " outputs=", session.outputNames.join(","));

  const corpus = JSON.parse(fs.readFileSync(
    path.join(__dirname, "..", "..", "reports", "standalone_corpus.json"), "utf8"));
  let checked = 0, maxdiff = 0, fails = 0;
  outer:
  for (const rec of corpus.games) {
    const g = new Eng.DushanGame({ config: rec.config, wall: rec.wall, dealer: rec.dealer });
    const obsByStep = new Map(rec.obs.map(o => [o.step, o]));
    for (let si = 0; si < rec.steps.length; si++) {
      const ob = obsByStep.get(si);
      if (ob) {
        const o = Enc.encodeV3(g, g.actor());
        if (o.head !== ob.head) { console.log("head 不一致:", o.head, "vs", ob.head); fails++; }
        const out = await session.run({
          tiles: new ort.Tensor("float32", o.tiles, [1, 63, 3, 9]),
          glob: new ort.Tensor("float32", o.glob, [1, 130]),
        });
        for (let h = 0; h < 4; h++) {
          const raw = Buffer.from(ob.logits[h], "base64");
          const base = new Float32Array(raw.buffer, raw.byteOffset, raw.byteLength / 4);
          const got = out[HEADS[h]].data;
          for (let i = 0; i < base.length; i++) {
            const d = Math.abs(base[i] - got[i]);
            if (d > maxdiff) maxdiff = d;
            if (d > TOL) {
              fails++;
              if (fails <= 6)
                console.log(`  diff ${HEADS[h]}[${i}]: base=${base[i]} got=${got[i]} d=${d.toExponential(2)}`);
            }
          }
        }
        checked++;
        if (checked >= MAX_OBS) break outer;
      }
      g.step({ kind: rec.steps[si].act[0], tile: rec.steps[si].act[1] });
    }
  }
  console.log(`[logits] obs=${checked} maxdiff=${maxdiff.toExponential(3)}`
    + ` 判据 ${TOL}` + (maxdiff <= TOL ? "  == PASS ==" : `  == FAIL（${fails} 处超差）==`));
  process.exit(maxdiff <= TOL ? 0 : 1);
}

main().catch(e => { console.error("ERR", e && e.message || e); process.exit(1); });
