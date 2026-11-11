//! O0：`Game` → `/api/state` JSON 视图。
//!
//! 字段名/语义逐项对齐 `web/dushan_server.py::Session._snapshot`（32 个顶层字段）
//! 与 `_pending_payload`。前端 `web/static/dushan.html` 一字不改，因此这里不能
//! 擅自「优化」字段名或省略前端会读的键。
//!
//! 口径要点（来自契约盘点）：
//!   * 牌是 `0..26` 整数，`255` 是 Rust 内部哨兵，出 JSON 时 `pass/hu` 用 `-1`；
//!   * `players[]` 按**相对座位** `rel=0..3`，`players[0]` 恒为「我」；
//!   * `turn/dealer/my_seat/scores/deltas` 用**绝对座位**；
//!   * `counter` 必须长度 27；`events` 的 `tile` 是 `null` 而非 `-1`。

use dushan_engine::actions::{self, Action, KIND_NAMES};
use dushan_engine::fan::{self, FanTable};
use dushan_engine::state::{self, Game};
use dushan_engine::tiles::{self, Counts};
use serde_json::{json, Value};

/// 前端只用 `!==` 比较 meta 与 state 的 page_ver，给同一个常量即可。
pub const PAGE_VER: &str = "offline-1.0.0";

pub struct SnapshotCtx<'a> {
    pub game: &'a Game,
    pub ft: &'a FanTable,
    pub me: u8,
    pub sid: &'a str,
    pub names: &'a [String; 4],
    pub kinds: &'a [&'static str; 4],
    pub hand_index: u32,
    pub cum: &'a [f64; 4],
    pub dealer_streak: u32,
    pub events: &'a [Value],
    pub pending: Option<Value>,
    pub result: Option<Value>,
    pub error: Option<&'a str>,
    pub stats: &'a Value,
    pub opp_think: &'a Value,
}

pub fn tile_name(t: u8) -> String {
    const S: [&str; 3] = ["W", "T", "B"];
    format!("{}{}", S[(t / 9) as usize], t % 9 + 1)
}

fn counts_tiles(c: &Counts) -> Vec<u8> {
    let mut v = Vec::with_capacity(14);
    for t in 0..27u8 {
        for _ in 0..c[t as usize] {
            v.push(t);
        }
    }
    v
}

fn mask_tiles(mask: u32) -> Vec<u8> {
    (0..27u8).filter(|&t| mask & (1u32 << t) != 0).collect()
}

fn meld_label(mt: u8) -> &'static str {
    match mt {
        fan::MELD_PONG => "碰",
        fan::MELD_KONG_EXPOSED => "明杠",
        fan::MELD_KONG_ADDED => "补杠",
        fan::MELD_KONG_CONCEALED => "暗杠",
        _ => "?",
    }
}

fn phase_str(p: u8) -> &'static str {
    match p {
        actions::PHASE_SELF_KONG => "self_kong",
        actions::PHASE_DISCARD => "discard",
        actions::PHASE_RESPOND => "respond",
        _ => "over",
    }
}

fn button_order(kind: u8) -> u8 {
    match kind {
        actions::HU => 0,
        actions::ANGANG | actions::BUGANG | actions::MINGGANG => 1,
        actions::PENG => 2,
        actions::PASS => 3,
        _ => 9,
    }
}

fn json_tile(a: &Action) -> i64 {
    if a.tile == 255 { -1 } else { a.tile as i64 }
}

/// `/api/act` 的字符串动作 → 引擎 Action。返回 None 表示非法 kind。
pub fn parse_action(kind: &str, tile: i64) -> Option<Action> {
    let k = KIND_NAMES.iter().position(|&s| s == kind)? as u8;
    let t = if tile < 0 { 255u8 } else { tile as u8 };
    Some(Action::new(k, t))
}

/// 决策点载荷：mode / buttons / discardables / last / can_hint。
pub fn pending_payload(g: &Game, ft: &FanTable) -> Value {
    let mode = match g.phase {
        actions::PHASE_SELF_KONG => {
            if g.rob_kong.is_some() { "respond" } else { "self_kong" }
        }
        actions::PHASE_DISCARD => "discard",
        _ => "respond",
    };
    let acts = g.legal_actions(ft);
    let mut buttons: Vec<Value> = Vec::new();
    let mut discardables: Vec<i64> = Vec::new();
    let mut order: Vec<(u8, Value)> = Vec::new();
    for a in &acts {
        if a.kind == actions::DISCARD {
            discardables.push(a.tile as i64);
            continue;
        }
        let mut b = json!({
            "kind": KIND_NAMES[a.kind as usize],
            "tile": json_tile(a),
            "label": a.label(),
        });
        if a.kind == actions::HU {
            // 前端只在胡按钮下方显示「凭什么能胡」；离线版暂留空串，
            // 不伪造规则依据（涉及 ron_reasons 的跨实现移植，属二期）。
            b["why"] = json!("");
        }
        order.push((button_order(a.kind), b));
    }
    order.sort_by_key(|x| x.0);
    buttons.extend(order.into_iter().map(|x| x.1));

    let last = g.last_discard.map(|(p, t)| json!([p, t]));
    json!({
        "mode": mode,
        "buttons": buttons,
        "discardables": discardables,
        "last": last,
        "can_hint": true,
    })
}

/// 与 Python `_snapshot` 同构的 32 字段视图。
pub fn snapshot(ctx: &SnapshotCtx) -> Value {
    let g = ctx.game;
    let me = ctx.me as usize;
    let over = g.phase == actions::PHASE_OVER;

    let hand = counts_tiles(&g.hands[me]);

    let mut players: Vec<Value> = Vec::with_capacity(4);
    for rel in 0..4usize {
        let p = (me + rel) % 4;
        let hand_tiles = counts_tiles(&g.hands[p]);
        let (ting_tiles, ting_types): (Vec<u8>, Vec<String>) = if over {
            (g.winning_tiles(p, ctx.ft), Vec::new())
        } else {
            (Vec::new(), Vec::new())
        };
        let melds: Vec<Value> = g.melds[p]
            .iter()
            .map(|&(mt, t, s)| json!({
                "type": mt,
                "label": meld_label(mt),
                "tile": t,
                "src": s,
                "ji": Value::Null,
            }))
            .collect();
        let discards: Vec<u8> = g.discards[p].iter().copied().collect();
        let claimed: Vec<bool> = vec![false; discards.len()];
        let tags: Vec<Value> = vec![Value::Null; discards.len()];
        players.push(json!({
            "rel": rel,
            "seat": p,
            "name": ctx.names[p],
            "kind": ctx.kinds[p],
            "hand_count": tiles::counts_sum(&g.hands[p]),
            "hand_tiles": hand_tiles,
            "ting_tiles": ting_tiles,
            "ting_types": ting_types,
            "melds": melds,
            "discards": discards,
            "discard_claimed": claimed,
            "discard_tags": tags,
            "score": round2(ctx.cum[p]),
            "is_dealer": p == g.dealer as usize,
            "baojiao": g.baojiao[p],
            "has_gang": g.has_dou(p),
            "chongfeng_done": g.first_discard_done[p],
        }));
    }

    // 记牌器：4 − 公开已见（四家牌河 + 副露 + 我的暗手）
    let mut seen = [0i32; 27];
    for p in 0..4 {
        for &t in &g.discards[p] {
            seen[t as usize] += 1;
        }
        for &(mt, t, _) in &g.melds[p] {
            seen[t as usize] += if fan::meld_is_kong(mt) { 4 } else { 3 };
        }
    }
    for t in 0..27 {
        seen[t] += g.hands[me][t] as i32;
    }
    let counter: Vec<i32> = seen.iter().map(|&n| (4 - n).max(0)).collect();

    let mode_discard = ctx
        .pending
        .as_ref()
        .and_then(|p| p.get("mode"))
        .and_then(|v| v.as_str())
        == Some("discard");
    let my_ting: Vec<u8> = if !over && mode_discard {
        g.winning_tiles(me, ctx.ft)
    } else {
        Vec::new()
    };

    let gold_known: Vec<u8> = if let Some(flip) = g.kaiju_flip {
        let mode = g.cfg.kaiju_fanji;
        if mode != 0 && state::fanji_mask(flip, mode) & (1u32 << tiles::YAOJI) != 0 {
            vec![tiles::YAOJI]
        } else {
            Vec::new()
        }
    } else {
        Vec::new()
    };

    let last_draw = if g.drawn && g.actor() == ctx.me && g.last_draw != 255 {
        json!(g.last_draw)
    } else {
        Value::Null
    };

    let events: Vec<Value> = ctx
        .events
        .iter()
        .rev()
        .take(160)
        .rev()
        .cloned()
        .collect();

    json!({
        "ready": true,
        "sid": ctx.sid,
        "over": over,
        "hand_index": ctx.hand_index,
        "my_seat": ctx.me,
        "names": ctx.names,
        "opp_think": ctx.opp_think,
        "dealer": g.dealer,
        "dealer_streak": ctx.dealer_streak,
        "wall_left": g.wall_left(),
        "turn": if over { -1 } else { g.actor() as i64 },
        "phase": phase_str(g.phase),
        "last": g.last_discard.map(|(p, t)| json!([p, t])),
        "repao": g.repao_discard,
        "hengji_active": g.hengji_open != 0,
        "hengji_opened": g.hengji_open != 0,
        "hengji_species": mask_tiles(g.hengji_open),
        "ji_tiles": mask_tiles(g.ji_tiles),
        "kaiju_flip": g.kaiju_flip,
        "gold_known": gold_known,
        "hand": hand,
        "last_draw": last_draw,
        "counter": counter,
        "my_ting": my_ting,
        "players": players,
        "pending": ctx.pending,
        "events": events,
        "scores": ctx.cum.iter().map(|&x| round2(x)).collect::<Vec<_>>(),
        "stats": ctx.stats,
        "result": ctx.result,
        "error": ctx.error,
        "page_ver": PAGE_VER,
    })
}

fn round2(x: f64) -> f64 { (x * 100.0).round() / 100.0 }

/// `/api/meta`：只给前端真正读取的字段（见契约盘点 §2.2 / §4）。
/// 离线版只注册两个贪心模型，不声明 dets —— 设备端搜索档在 O2 另行处理。
pub fn meta() -> Value {
    let tiles: Vec<Value> = (0..27u8)
        .map(|t| json!({
            "id": t,
            "name": tile_name(t),
            "cn": tiles::tile_cn(t),
            "suit": t / 9,
            "rank": t % 9 + 1,
        }))
        .collect();
    let ident = |stem: &str, line: &str, ver: &str, size: &str| json!({
        "stem": stem, "origin": "offline", "line": line,
        "ver": ver, "abbr": ver, "size": size,
    });
    // `champion` 一律 false：离线四档由 `window.__DUSHAN_TIERS` 直接给出，
    // 若这里标 true，前端 registerChampionOptions() 会把同一张网再追加一遍，
    // 下拉里出现「传奇 · 学生」两条一模一样的选项。
    json!({
        "page_ver": PAGE_VER,
        "models": [
            {
                "stem": "champ_slim_s",
                "size_kb": 3160,
                "mtime": 0,
                "iter": 0,
                "net": "s",
                "params": 816300,
                "champion": false,
                "label": "传奇 · 学生（离线）",
                "latency": {"model": 0.02},
                "ident": ident("champ_slim_s", "E-SLIM", "S1", "0.82M"),
            },
            {
                "stem": "dushan_master",
                "size_kb": 2960,
                "mtime": 0,
                "iter": 0,
                "net": "m1",
                "params": 769000,
                "champion": false,
                "label": "大师 · 主力（离线）",
                "latency": {"model": 0.03},
                "ident": ident("dushan_master", "MASTER", "M", "0.77M"),
            },
            {
                "stem": "expert_s7",
                "size_kb": 470,
                "mtime": 0,
                "iter": 1,
                "net": "s",
                "params": 119000,
                "champion": false,
                "label": "进阶 · 小网（离线）",
                "latency": {"model": 0.01},
                "ident": ident("expert_s7", "S7R", "S7", "0.12M"),
            }
        ],
        "tiles": tiles,
        "yaoji": tiles::YAOJI,
        "rules": "",
        "master": "dushan_master",
        "master_info": {"label": "大师 · 离线（769K）", "abbr": "M"},
    })
}
