//! O0 对拍器：读 `runs/offline_golden.json`，用同一副牌墙重放同一串动作，
//! 逐个快照与 Python `Session._snapshot` 做 JSON 逐字段 diff。
//!
//! 忽略清单（IGNORE）只放「两边环境必然不同」或「离线版按 §7.2 明确裁剪」的
//! 字段；任何其它差异都算失败，必须修 `view.rs` 而不是放宽白名单。

use std::fs;

use dushan_engine::actions::Action;
use dushan_engine::fan::FanTable;
use dushan_engine::state::{Config, Game};
use offline_core::view::{self, SnapshotCtx};
use serde_json::{json, Value};

/// 允许不一致的路径（点号分隔，`[]` 表示数组任意元素）。
///   * page_ver：Python 由 Handler 注入，Rust 是固定常量；
///   * pending.can_hint：Python 探测磁盘上的 .pt，离线版恒 true；
///   * buttons[].why / melds[].ji / discard_tags / discard_claimed / ting_types：
///     离线版按 §7.2 裁剪为占位值（前端有兜底）；
///   * events / stats / opp_think / result：会话层元数据，O1 才完整。
const IGNORE: &[&str] = &[
    "page_ver",
    "events",
    "stats",
    "opp_think",
    "result",
    "pending.can_hint",
    "pending.buttons[].why",
    "players[].discard_tags",
    "players[].discard_claimed",
    "players[].melds[].ji",
    "players[].ting_types",
];

/// 把 `players[0].melds[1].ji` 归一化成 `players[].melds[].ji`。
fn norm(path: &str) -> String {
    let mut out = String::new();
    let mut it = path.chars().peekable();
    while let Some(c) = it.next() {
        if c == '[' {
            out.push_str("[]");
            while let Some(&n) = it.peek() {
                it.next();
                if n == ']' { break; }
            }
        } else {
            out.push(c);
        }
    }
    out
}

fn ignored(path: &str) -> bool {
    let n = norm(path);
    IGNORE.iter().any(|pat| {
        let p = norm(pat);
        n == p || n.starts_with(&format!("{p}."))
    })
}

fn diff(a: &Value, b: &Value, path: &str, out: &mut Vec<String>) {
    if ignored(path) {
        return;
    }
    match (a, b) {
        (Value::Object(ma), Value::Object(mb)) => {
            let mut keys: Vec<&String> = ma.keys().chain(mb.keys()).collect();
            keys.sort();
            keys.dedup();
            for k in keys {
                let p = if path.is_empty() { k.clone() } else { format!("{path}.{k}") };
                match (ma.get(k), mb.get(k)) {
                    (Some(x), Some(y)) => diff(x, y, &p, out),
                    (None, Some(y)) => {
                        if !ignored(&p) {
                            out.push(format!("+ {p} = {y}"));
                        }
                    }
                    (Some(x), None) => {
                        if !ignored(&p) {
                            out.push(format!("- {p} = {x}"));
                        }
                    }
                    _ => {}
                }
            }
        }
        (Value::Array(va), Value::Array(vb)) => {
            if va.len() != vb.len() {
                out.push(format!("! {path} 长度 {} vs {}", va.len(), vb.len()));
            }
            for (i, (x, y)) in va.iter().zip(vb.iter()).enumerate() {
                diff(x, y, &format!("{path}[{i}]"), out);
            }
        }
        (Value::Number(x), Value::Number(y)) => {
            let fx = x.as_f64().unwrap_or(f64::NAN);
            let fy = y.as_f64().unwrap_or(f64::NAN);
            if (fx - fy).abs() > 1e-6 {
                out.push(format!("! {path}: {fx} vs {fy}"));
            }
        }
        (x, y) if x != y => out.push(format!("! {path}: {x} vs {y}")),
        _ => {}
    }
}

fn rust_snapshot(g: &Game, ft: &FanTable) -> Value {
    let names = [
        "我".to_string(), "下家".to_string(), "对家".to_string(), "上家".to_string(),
    ];
    let kinds = ["human", "model", "model", "model"];
    let cum = [0.0f64; 4];
    let stats = json!({"0": {}, "1": {}, "2": {}, "3": {}});
    let think = json!({});
    let events: Vec<Value> = Vec::new();
    let pending = Some(view::pending_payload(g, ft));
    view::snapshot(&SnapshotCtx {
        game: g,
        ft,
        me: 0,
        sid: "golden",
        names: &names,
        kinds: &kinds,
        hand_index: 1,
        cum: &cum,
        dealer_streak: 0,
        events: &events,
        pending,
        result: None,
        error: None,
        stats: &stats,
        opp_think: &think,
    })
}

fn main() {
    let path = std::env::args().nth(1).unwrap_or_else(|| "runs/offline_golden.json".into());
    let raw = match fs::read_to_string(&path) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("读不到 {path}: {e}（先跑 tools/offline_golden.py）");
            std::process::exit(2);
        }
    };
    let doc: Value = serde_json::from_str(&raw).expect("golden JSON 解析失败");
    let ft = FanTable::build();
    let cases = doc["cases"].as_array().expect("cases");
    let mut total = 0;
    let mut bad = 0;
    for case in cases {
        let seed = case["seed"].as_i64().unwrap_or(0);
        let wall: Vec<u8> = case["wall"]
            .as_array().unwrap().iter().map(|v| v.as_u64().unwrap() as u8).collect();
        let dealer = case["dealer"].as_u64().unwrap() as u8;
        let mut g = Game::new_with_wall(wall, dealer, Config::default(), &ft);
        let actions: &Vec<Value> = case["actions"].as_array().unwrap();
        let snaps: &Vec<Value> = case["snapshots"].as_array().unwrap();
        for (i, gold) in snaps.iter().enumerate() {
            let got = rust_snapshot(&g, &ft);
            let mut d = Vec::new();
            diff(gold, &got, "", &mut d);
            total += 1;
            if d.is_empty() {
                println!("  seed={seed} step={i} PASS");
            } else {
                bad += 1;
                println!("  seed={seed} step={i} FAIL ({} 处)", d.len());
                for line in d.iter().take(12) {
                    println!("      {line}");
                }
            }
            if let Some(a) = actions.get(i) {
                let kind = a[0].as_str().unwrap();
                let tile = a[1].as_i64().unwrap();
                let act = view::parse_action(kind, tile).expect("未知动作");
                g.step(act, &ft);
            }
        }
    }
    println!("\nO0 对拍：{total} 个快照，失败 {bad} 个");
    if bad > 0 {
        std::process::exit(1);
    }
}
