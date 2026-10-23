// 本地引擎 API 层冒烟测试（Node）：
//   开局 → 人类 pass/打牌 → 3 个模型座位连续行动 → 终局 → 下一局（共 4 局）。
// 推理用 mock（固定偏好：优先胡/杠/碰，否则打第一张合法牌），不依赖 ORT。
// 用法：node web/standalone/smoke_local.js
"use strict";
const path = require("path");
const { createLocalApi, indexToAction } = require(path.join(__dirname, "local_engine_server.js"));

let failures = 0;
function check(cond, msg) {
  if (cond) { console.log("  ok  -", msg); return; }
  failures += 1;
  console.log("  FAIL-", msg);
}

// ---- mock 推理：按 mask 选动作（偏好 hu > 杠 > 碰 > 打第一张） ----------------
function mockInfer(obs) {
  const n = obs.mask.length;
  const out = new Float32Array(n);
  // 打分：hu(27/3号位) 最高，其次杠/碰，其余 0；非法位由 server 端 mask 掉
  for (let i = 0; i < n; i++) {
    if (!obs.mask[i]) continue;
    if (obs.head === "discard") out[i] = i === 27 ? 100 : 1;
    else if (obs.head === "respond") out[i] = i === 3 ? 100 : i === 0 ? 1 : 2;
    else out[i] = i === 27 ? 1 : 5;   // self_kong：优先杠
  }
  return { [obs.head]: out };
}

async function main() {
  const api = createLocalApi();
  api.setInferFn(mockInfer);

  // 1) meta
  const meta = await api.handle("/api/meta");
  check(meta.ok !== false && Array.isArray(meta.tiles) && meta.tiles.length === 27,
    "/api/meta 返回 27 个牌种");
  check(meta.master === "c4_it2280", "/api/meta.master = c4_it2280");

  // 2) 开局
  const nr = await api.handle("/api/new", {
    opponents: [{ type: "model" }, { type: "model" }, { type: "model" }],
    my_seat: 0, speed: 0, rules: { jiesuan: "both", kaiju: "off" },
  });
  check(nr.ok && nr.sid, `/api/new ok sid=${nr.sid} my_seat=${nr.my_seat}`);
  const sid = nr.sid;

  // 3) 打 4 局
  for (let handNo = 1; handNo <= 4; handNo++) {
    const over = await playOneHand(api, sid, handNo);
    check(over, `第 ${handNo} 局打完且终局合法`);
    if (!over) break;
    if (handNo < 4) {
      const nx = await api.handle("/api/next", { sid, rules: { jiesuan: "both", kaiju: "off" } });
      check(nx.ok, `第 ${handNo} 局后 /api/next ok`);
      if (!nx.ok) break;
    }
  }

  // 4) 历史与统计
  const hist = await api.handle("/api/history?limit=30");
  check(hist.ok && hist.records.length === 4,
    `/api/history 返回 4 条记录（实际 ${hist.records ? hist.records.length : 0}）`);
  const stats = await api.handle("/api/stats?sid=" + sid);
  check(stats.ok && stats.all_time.games === 4,
    `/api/stats games=4（实际 ${stats.all_time.games}）`);
  const del = await api.handle("/api/history_delete", { ln: hist.records[0]._ln });
  check(del.ok && del.removed_rec === 1, "/api/history_delete 删除 1 条");
  const hist2 = await api.handle("/api/history?limit=30");
  check(hist2.records.length === 3, "删除后剩 3 条");

  console.log(failures ? `\n== SMOKE FAIL（${failures} 处）==` : "\n== SMOKE PASS ==");
  process.exit(failures ? 1 : 0);
}

async function playOneHand(api, sid, handNo) {
  let guard = 0;
  let sawHumanPending = false, sawModelMoves = 0;
  for (;;) {
    if (++guard > 20000) { console.log("  FAIL-步数超限（疑似卡死）"); return false; }
    const st = await api.handle("/api/state?sid=" + sid);
    if (!st || !st.ready) { await sleep(5); continue; }
    if (st.error) { console.log("  FAIL-会话错误:", st.error); return false; }
    if (st.over) {
      const res = st.result || {};
      const legal = res.type === "win" || res.type === "huangzhuang";
      console.log(`  第 ${handNo} 局 → ${res.type} winner=${res.winner ?? "-"} `
        + `deltas=[${(res.deltas || []).join(", ")}] wall_left=${res.wall_left}`);
      check(legal, `第 ${handNo} 局终局类型合法（${res.type}）`);
      const dsum = (res.deltas || []).reduce((a, b) => a + b, 0);
      check(Math.abs(dsum) < 1e-6, `第 ${handNo} 局零和 Σ=${dsum}`);
      check(Array.isArray(st.events) && st.events.length > 0, "事件流非空");
      check(st.players && st.players.length === 4, "快照 players=4");
      check(sawHumanPending, `第 ${handNo} 局出现过人类决策点`);
      const oppEv = (st.events || []).filter(e =>
        e.rel >= 1 && e.rel <= 3 && ["draw", "discard", "peng", "kong", "hu"].includes(e.kind));
      check(oppEv.length > 0, `第 ${handNo} 局模型座位行动过（对手事件 ${oppEv.length} 条）`);
      return legal;
    }
    // 人类决策点
    if (st.pending && st.turn === st.my_seat) {
      sawHumanPending = true;
      const pend = st.pending;
      let act;
      if (pend.mode === "discard" && pend.discardables.length)
        act = { kind: "discard", tile: pend.discardables[0] };
      else if (pend.buttons.length) {
        // 优先 hu > peng/minggang > pass
        const pref = ["hu", "angang", "bugang", "minggang", "peng", "pass"];
        let pick = null;
        for (const k of pref) {
          pick = pend.buttons.find(b => b.kind === k);
          if (pick) break;
        }
        act = pick ? { kind: pick.kind, tile: pick.tile } : null;
      }
      if (!act) { console.log("  FAIL-空决策点", JSON.stringify(pend)); return false; }
      const r = await api.handle("/api/act", { sid, kind: act.kind, tile: act.tile });
      if (!r.ok) { console.log("  FAIL-/api/act 被拒:", r.error, JSON.stringify(act)); return false; }
      continue;
    }
    // 模型座位在推进（轮询可见中间状态）
    if (st.turn >= 0 && st.turn !== st.my_seat) sawModelMoves += 1;
    await sleep(2);
  }
}

function sleep(ms) { return new Promise(r => setTimeout(r, ms)); }

main().catch(e => { console.error(e); process.exit(1); });
