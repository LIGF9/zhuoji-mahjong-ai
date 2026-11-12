//! 对局记录 / 数据统计 / 回放（2026-11-12 移植自 `web/dushan_server.py`）。
//!
//! 离线版此前把 `/api/history`、`/api/stats`、`/api/replay` 全写成空实现，
//! 界面上「对局记录」「数据统计」两个面板永远是空的 —— 这是用户 2026-11-12
//! 反馈「都没生效了」的原因。这里把线上那套口径完整搬过来：
//!
//! * `records.jsonl` —— 每手一行，字段与线上 `_make_record` 同义；
//!   `/api/stats` 是**逐行扫这个文件现算**的，没有独立账本，
//!   所以删掉一条记录，统计会自动跟着少一局（用户口径）。
//! * `replays.jsonl` —— 每手一行事件流 + 终局快照，比记录大一到两个数量级，
//!   单独一个文件，否则每次 `/api/stats` 的逐行解析会变慢。
//! * 回放按 rid 随机取，文件是追加写的 → 只读**尾部 8MB** 建「最近 N 局」
//!   的内存字典（与线上 `_load_replays` 一致）。
//!
//! 目录：优先 `DUSHAN_HISTORY_DIR` 环境变量，其次外壳通过 [`set_dir`] 指定的
//! 应用私有目录（安卓/Tauri 用 app_data_dir），再退回 `$HOME/.dushan/history`。
//! 安卓上没有 HOME，**必须**由外壳 set_dir，否则会写到不可写目录。

use serde_json::{Map, Value};
use std::collections::VecDeque;
use std::fs;
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Mutex, OnceLock};

static DIR: OnceLock<Mutex<Option<PathBuf>>> = OnceLock::new();
static REPLAY_CACHE: OnceLock<Mutex<VecDeque<(String, Value)>>> = OnceLock::new();
static REPLAY_READY: AtomicBool = AtomicBool::new(false);

const REPLAY_CACHE_MAX: usize = 300;
const REPLAY_TAIL_BYTES: u64 = 8 * 1024 * 1024;

fn dir_cell() -> &'static Mutex<Option<PathBuf>> {
    DIR.get_or_init(|| Mutex::new(None))
}

fn cache_cell() -> &'static Mutex<VecDeque<(String, Value)>> {
    REPLAY_CACHE.get_or_init(|| Mutex::new(VecDeque::new()))
}

/// 外壳（安卓 / 桌面 Tauri）启动服务前指定可写目录。
pub fn set_dir(p: impl Into<PathBuf>) {
    *dir_cell().lock().unwrap() = Some(p.into());
}

/// 当前历史目录。
pub fn data_dir() -> PathBuf {
    if let Some(p) = dir_cell().lock().unwrap().clone() {
        return p;
    }
    if let Ok(p) = std::env::var("DUSHAN_HISTORY_DIR") {
        if !p.trim().is_empty() {
            return PathBuf::from(p);
        }
    }
    for key in ["HOME", "USERPROFILE"] {
        if let Ok(h) = std::env::var(key) {
            if !h.trim().is_empty() {
                return PathBuf::from(h).join(".dushan").join("history");
            }
        }
    }
    PathBuf::from("history")
}

fn records_path() -> PathBuf {
    data_dir().join("records.jsonl")
}

fn replays_path() -> PathBuf {
    data_dir().join("replays.jsonl")
}

fn read_lines(p: &PathBuf) -> Vec<String> {
    match fs::read_to_string(p) {
        Ok(s) => s.lines().map(|x| x.to_string()).collect(),
        Err(_) => Vec::new(),
    }
}

fn parse_lines(p: &PathBuf) -> Vec<Value> {
    read_lines(p)
        .iter()
        .filter_map(|s| {
            let s = s.trim();
            if s.is_empty() {
                None
            } else {
                serde_json::from_str::<Value>(s).ok()
            }
        })
        .collect()
}

// ---------------------------------------------------------------------------
// 写入
// ---------------------------------------------------------------------------
fn append_line(p: &PathBuf, line: &str) {
    if let Some(dir) = p.parent() {
        let _ = fs::create_dir_all(dir);
    }
    if let Ok(mut f) = fs::OpenOptions::new().create(true).append(true).open(p) {
        let _ = writeln!(f, "{line}");
    }
}

fn cache_put(rp: Value) {
    let Some(rid) = rp.get("rid").and_then(|x| x.as_str()).map(str::to_string) else {
        return;
    };
    let mut c = cache_cell().lock().unwrap();
    if let Some(pos) = c.iter().position(|(k, _)| *k == rid) {
        c.remove(pos);
    }
    c.push_back((rid, rp));
    while c.len() > REPLAY_CACHE_MAX {
        c.pop_front();
    }
}

/// 追加一条记录（可带回放）。写失败只打日志，绝不影响对局。
pub fn append(rec: &Value, replay: Option<Value>) {
    let line = match serde_json::to_string(rec) {
        Ok(s) => s,
        Err(_) => return,
    };
    append_line(&records_path(), &line);
    if let Some(mut rp) = replay {
        if let Some(o) = rp.as_object_mut() {
            o.insert(
                "rid".into(),
                rec.get("rid").cloned().unwrap_or(Value::Null),
            );
        }
        if let Ok(s) = serde_json::to_string(&rp) {
            append_line(&replays_path(), &s);
        }
        cache_put(rp);
    }
}

// ---------------------------------------------------------------------------
// 回放缓存
// ---------------------------------------------------------------------------
fn load_replays_once() {
    if REPLAY_READY.swap(true, Ordering::SeqCst) {
        return;
    }
    let p = replays_path();
    let Ok(mut f) = fs::File::open(&p) else {
        return;
    };
    let size = f.metadata().map(|m| m.len()).unwrap_or(0);
    if size > REPLAY_TAIL_BYTES {
        let _ = f.seek(SeekFrom::Start(size - REPLAY_TAIL_BYTES));
        // 丢掉可能被截断的首行
        let mut skip = Vec::new();
        let mut b = [0u8; 1];
        while let Ok(1) = f.read(&mut b) {
            if b[0] == b'\n' {
                break;
            }
            skip.push(b[0]);
            if skip.len() > 1 << 20 {
                break;
            }
        }
    }
    let mut buf = String::new();
    let _ = f.read_to_string(&mut buf);
    for line in buf.lines() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        if let Ok(v) = serde_json::from_str::<Value>(line) {
            cache_put(v);
        }
    }
}

/// 按 rid 取回放；不在缓存时懒加载一次文件尾部。
pub fn replay(rid: &str) -> Option<Value> {
    if rid.is_empty() {
        return None;
    }
    let hit = cache_cell()
        .lock()
        .unwrap()
        .iter()
        .find(|(k, _)| k == rid)
        .map(|(_, v)| v.clone());
    if hit.is_some() {
        return hit;
    }
    load_replays_once();
    cache_cell()
        .lock()
        .unwrap()
        .iter()
        .find(|(k, _)| k == rid)
        .map(|(_, v)| v.clone())
}

fn has_replay(rid: &str) -> bool {
    if rid.is_empty() {
        return false;
    }
    load_replays_once();
    cache_cell().lock().unwrap().iter().any(|(k, _)| k == rid)
}

// ---------------------------------------------------------------------------
// 读取
// ---------------------------------------------------------------------------
/// 最近 `limit` 条记录（**新的在前**），每条带 `_ln`（0 基行号，删除按钮用）
/// 与 `has_replay`。
pub fn load(limit: usize) -> Vec<Value> {
    let mut out: Vec<Value> = parse_lines(&records_path())
        .into_iter()
        .enumerate()
        .map(|(i, mut v)| {
            if let Some(o) = v.as_object_mut() {
                o.insert("_ln".into(), Value::from(i as i64));
            }
            v
        })
        .collect();
    let n = limit.max(1);
    if out.len() > n {
        out = out.split_off(out.len() - n);
    }
    out.reverse();
    for v in out.iter_mut() {
        let rid = v.get("rid").and_then(|x| x.as_str()).unwrap_or("");
        let can = has_replay(rid);
        if let Some(o) = v.as_object_mut() {
            o.insert("has_replay".into(), Value::Bool(can));
        }
    }
    out
}

fn all_records() -> Vec<Value> {
    parse_lines(&records_path())
}

// ---------------------------------------------------------------------------
// 统计（/api/stats 的历史累计）
// ---------------------------------------------------------------------------
const BIGFAN_EXCLUDE: [&str; 3] = ["", "平胡", "报叫"];

fn is_bigfan_name(name: &str) -> bool {
    !BIGFAN_EXCLUDE.contains(&name)
}

fn labels_of(r: &Value) -> Vec<String> {
    r.get("detail")
        .and_then(|x| x.as_array())
        .map(|a| {
            a.iter()
                .map(|d| {
                    d.as_array()
                        .and_then(|x| x.get(1))
                        .and_then(|x| x.as_str())
                        .unwrap_or("")
                        .to_string()
                })
                .collect()
        })
        .unwrap_or_default()
}

fn flag_bool(fl: &Map<String, Value>, key: &str) -> Option<bool> {
    fl.get(key).and_then(|x| x.as_bool())
}

/// 历史累计。口径与线上 `dushan_server.py::/api/stats` 逐项对齐：
/// 前面 10 项分母是全部对局，最后两项（本人胡牌 / 本人累计）分母是
/// 「全程没开过大师推荐」的局（`solo_games`）。
pub fn stats() -> Value {
    let mut games = 0i64;
    let mut zimo = 0i64;
    let mut songhu = 0i64;
    let mut score = 0.0f64;
    let mut chongfeng = 0i64;
    let mut ji_discard = 0i64;
    let mut baoji = 0i64;
    let mut tenpai_bigfan = 0i64;
    let mut win_bigfan = 0i64;
    let mut huang = 0i64;
    let mut huang_noten = 0i64;
    let mut solo_games = 0i64;
    let mut solo_win = 0i64;
    let mut solo_score = 0.0f64;

    for r in all_records() {
        if r.get("type").and_then(|x| x.as_str()) == Some("abort") {
            continue;
        }
        if r.get("type").and_then(|x| x.as_str()).is_none() {
            continue; // 半截记录（老版本中止写入）不计
        }
        let me = r.get("my_seat").and_then(|x| x.as_i64()).unwrap_or(0).max(0) as usize;
        let deltas: Vec<f64> = r
            .get("deltas")
            .and_then(|x| x.as_array())
            .map(|a| a.iter().map(|x| x.as_f64().unwrap_or(0.0)).collect())
            .unwrap_or_default();
        let dme = deltas.get(me).copied().unwrap_or(0.0);
        games += 1;
        score += dme;

        let fl = r.get("flags").and_then(|x| x.as_object()).cloned().unwrap_or_default();
        let labels = labels_of(&r);
        let has = |k: &str| labels.iter().any(|x| x == k);

        if flag_bool(&fl, "chongfeng").unwrap_or_else(|| has("冲锋鸡")) {
            chongfeng += 1;
        }
        if flag_bool(&fl, "ji_discard")
            .unwrap_or_else(|| has("冲锋鸡") || has("横鸡") || has("幺鸡"))
        {
            ji_discard += 1;
        }
        if flag_bool(&fl, "baoji").unwrap_or_else(|| has("包鸡")) {
            baoji += 1;
        }
        if flag_bool(&fl, "tenpai_bigfan").unwrap_or(false) {
            tenpai_bigfan += 1;
        }
        // 旧记录没有 hint_used → 按「用过推荐」保守处理（与线上一致）
        let solo = !flag_bool(&fl, "hint_used").unwrap_or(true);
        if solo {
            solo_games += 1;
            solo_score += dme;
        }

        if r.get("type").and_then(|x| x.as_str()) == Some("win") {
            if !r.get("is_tsumo").and_then(|x| x.as_bool()).unwrap_or(false) {
                songhu += 1;
            }
            let big = flag_bool(&fl, "win_bigfan")
                .unwrap_or_else(|| is_bigfan_name(r.get("fan_cn").and_then(|x| x.as_str()).unwrap_or("")));
            if big {
                win_bigfan += 1;
            }
            let winners: Vec<i64> = r
                .get("winners")
                .and_then(|x| x.as_array())
                .map(|a| a.iter().filter_map(|x| x.as_i64()).collect())
                .unwrap_or_else(|| {
                    r.get("winner").and_then(|x| x.as_i64()).into_iter().collect()
                });
            if winners.iter().any(|&w| w == me as i64) {
                if r.get("is_tsumo").and_then(|x| x.as_bool()).unwrap_or(false) {
                    zimo += 1;
                }
                if solo {
                    solo_win += 1;
                }
            }
        } else {
            huang += 1;
            let n_tenpai = r.get("tenpai").and_then(|x| x.as_array()).map(|a| a.len()).unwrap_or(0);
            if flag_bool(&fl, "huang_noten").unwrap_or(n_tenpai < 4) {
                huang_noten += 1;
            }
        }
    }

    serde_json::json!({"ok": true, "all_time": {
        "games": games, "zimo": zimo, "songhu_games": songhu,
        "score": (score * 100.0).round() / 100.0,
        "chongfeng_games": chongfeng, "ji_discard_games": ji_discard,
        "baoji_games": baoji, "tenpai_bigfan_games": tenpai_bigfan,
        "win_bigfan_games": win_bigfan, "huang_games": huang,
        "huang_noten_games": huang_noten, "solo_games": solo_games,
        "solo_win_games": solo_win,
        "solo_score": (solo_score * 100.0).round() / 100.0
    }})
}

// ---------------------------------------------------------------------------
// 导出
// ---------------------------------------------------------------------------
const EXPORT_FIELDS: [(&str, &str); 20] = [
    ("time", "时间"),
    ("rid", "回放ID"),
    ("hand", "局号"),
    ("type", "结果"),
    ("my_seat", "我的座位"),
    ("dealer", "庄家"),
    ("names", "四家"),
    ("winner_names", "赢家"),
    ("loser_name", "点炮者"),
    ("how", "胡牌方式"),
    ("fan_cn", "牌型"),
    ("fan", "番数"),
    ("is_tsumo", "自摸"),
    ("rob_kong", "抢杠"),
    ("tenpai_names", "听牌者"),
    ("deltas", "四家分"),
    ("ji_tiles", "本局鸡牌"),
    ("fanji", "翻鸡"),
    ("void_name", "烧鸡者"),
    ("detail", "结算明细"),
];

fn csv_escape(s: &str) -> String {
    if s.contains(',') || s.contains('"') || s.contains('\n') || s.contains('\r') {
        format!("\"{}\"", s.replace('"', "\"\""))
    } else {
        s.to_string()
    }
}

fn stamp() -> String {
    // 与线上 `strftime("%Y%m%d-%H%M%S")` 同形，文件名一眼能看出导出时间
    let s = now_str(); // "YYYY-MM-DD HH:MM:SS"
    let digits: String = s.chars().filter(|c| c.is_ascii_digit()).collect();
    if digits.len() >= 14 {
        format!("{}-{}", &digits[0..8], &digits[8..14])
    } else {
        format!("{}", now_ms())
    }
}

/// 返回 `(body, content_type, filename)`；`fmt` = `json` | `csv`。
pub fn export(fmt: &str) -> (Vec<u8>, &'static str, String) {
    let rows = all_records();
    let st = stamp();
    if fmt == "csv" {
        let mut out = String::from("\u{feff}");
        out.push_str(&EXPORT_FIELDS.map(|(_k, cn)| cn).join(","));
        out.push('\n');
        for r in &rows {
            let mut cells = Vec::with_capacity(EXPORT_FIELDS.len());
            for (k, _cn) in EXPORT_FIELDS {
                let v = r.get(k);
                let s = match v {
                    None | Some(Value::Null) => String::new(),
                    Some(Value::String(s)) => s.clone(),
                    Some(Value::Array(_)) | Some(Value::Object(_)) => {
                        serde_json::to_string(v.unwrap()).unwrap_or_default()
                    }
                    Some(x) => x.to_string(),
                };
                cells.push(csv_escape(&s));
            }
            out.push_str(&cells.join(","));
            out.push('\n');
        }
        return (
            out.into_bytes(),
            "text/csv; charset=utf-8",
            format!("dushan_records_{st}.csv"),
        );
    }
    let body = serde_json::json!({
        "exported_at": "", "count": rows.len(), "records": rows
    });
    (
        serde_json::to_vec_pretty(&body).unwrap_or_default(),
        "application/json; charset=utf-8",
        format!("dushan_records_{st}.json"),
    )
}

// ---------------------------------------------------------------------------
// 删除 / 清空
// ---------------------------------------------------------------------------
/// 一条删除请求的定位参数：行号（0 基）或 rid，二者至少给一个。
pub type Key = (Option<i64>, Option<String>);

fn rewrite(drop_lines: Option<&[usize]>, drop_rids: Option<&[String]>) -> i64 {
    let p = records_path();
    let lines = read_lines(&p);
    let mut kept = Vec::with_capacity(lines.len());
    let mut removed = 0i64;
    for (i, line) in lines.iter().enumerate() {
        let drop_by_ln = drop_lines.map(|d| d.contains(&i)).unwrap_or(false);
        let drop_by_rid = drop_rids
            .map(|rids| {
                serde_json::from_str::<Value>(line.trim())
                    .ok()
                    .and_then(|v| v.get("rid").and_then(|x| x.as_str()).map(str::to_string))
                    .map(|r| rids.iter().any(|x| *x == r))
                    .unwrap_or(false)
            })
            .unwrap_or(false);
        if drop_by_ln || drop_by_rid {
            removed += 1;
        } else {
            kept.push(line.clone());
        }
    }
    if removed > 0 {
        let mut body = kept.join("\n");
        if !body.is_empty() {
            body.push('\n');
        }
        // 写临时文件再 rename：中途崩溃不会把记录文件写坏
        let tmp = p.with_extension("jsonl.tmp");
        if fs::write(&tmp, body).is_ok() {
            let _ = fs::rename(&tmp, &p);
        }
    }
    removed
}

fn rewrite_replays(drop_rids: &[String]) -> i64 {
    let p = replays_path();
    let lines = read_lines(&p);
    let mut kept = Vec::with_capacity(lines.len());
    let mut removed = 0i64;
    for line in lines.iter() {
        let rid = serde_json::from_str::<Value>(line.trim())
            .ok()
            .and_then(|v| v.get("rid").and_then(|x| x.as_str()).map(str::to_string))
            .unwrap_or_default();
        if !rid.is_empty() && drop_rids.iter().any(|x| *x == rid) {
            removed += 1;
        } else {
            kept.push(line.clone());
        }
    }
    if removed > 0 {
        let mut body = kept.join("\n");
        if !body.is_empty() {
            body.push('\n');
        }
        let tmp = p.with_extension("jsonl.tmp");
        if fs::write(&tmp, body).is_ok() {
            let _ = fs::rename(&tmp, &p);
        }
    }
    removed
}

/// 删除若干条记录（连同回放）。返回 `{ok, removed_rec, removed_replay, skipped}`。
pub fn delete(keys: &[Key]) -> Value {
    let p = records_path();
    let lines = read_lines(&p);
    let mut by_rid: std::collections::HashMap<String, usize> = std::collections::HashMap::new();
    for (i, s) in lines.iter().enumerate() {
        if let Ok(v) = serde_json::from_str::<Value>(s.trim()) {
            if let Some(r) = v.get("rid").and_then(|x| x.as_str()) {
                if !r.is_empty() {
                    by_rid.entry(r.to_string()).or_insert(i);
                }
            }
        }
    }

    let mut drop_lines: Vec<usize> = Vec::new();
    let mut drop_rids: Vec<String> = Vec::new();
    let mut skipped: Vec<Value> = Vec::new();
    for (ln, rid) in keys.iter() {
        let rid_o = rid.clone().filter(|s| !s.trim().is_empty());
        let idx = match ln {
            Some(i) if *i >= 0 && (*i as usize) < lines.len() => Some(*i as usize),
            Some(i) => {
                skipped.push(serde_json::json!({"ln": i, "rid": rid_o, "why": format!("行号越界：{i}")}));
                None
            }
            None => match &rid_o {
                None => {
                    skipped.push(serde_json::json!({"ln": Value::Null, "rid": Value::Null, "why": "缺少定位参数"}));
                    None
                }
                Some(r) => match by_rid.get(r) {
                    Some(i) => Some(*i),
                    None => {
                        skipped.push(serde_json::json!({"ln": Value::Null, "rid": r, "why": format!("记录已不在文件里：{r}")}));
                        None
                    }
                },
            },
        };
        let Some(i) = idx else { continue };
        let rec: Value = serde_json::from_str(lines[i].trim()).unwrap_or(Value::Null);
        let rec_rid = rec
            .get("rid")
            .and_then(|x| x.as_str())
            .unwrap_or("")
            .to_string();
        if let Some(want) = &rid_o {
            if !rec_rid.is_empty() && rec_rid != *want {
                skipped.push(serde_json::json!({"ln": ln, "rid": want,
                    "why": format!("行号与记录不匹配（第 {i} 行是 {rec_rid}）")}));
                continue;
            }
        }
        if !drop_lines.contains(&i) {
            drop_lines.push(i);
        }
        let r = if !rec_rid.is_empty() {
            rec_rid
        } else {
            rid_o.unwrap_or_default()
        };
        if !r.is_empty() && !drop_rids.contains(&r) {
            drop_rids.push(r);
        }
    }

    let removed_rec = rewrite(Some(&drop_lines), None);
    let mut removed_replay = 0i64;
    if !drop_rids.is_empty() {
        removed_replay = rewrite_replays(&drop_rids);
        let mut c = cache_cell().lock().unwrap();
        c.retain(|(k, _)| !drop_rids.iter().any(|x| x == k));
    }
    let ok = removed_rec > 0;
    let mut out = serde_json::json!({
        "ok": ok, "removed_rec": removed_rec, "removed_replay": removed_replay,
        "skipped": skipped,
    });
    if !ok {
        let mut whys: Vec<String> = Vec::new();
        if let Some(arr) = out.get("skipped").and_then(|x| x.as_array()) {
            for s in arr {
                let w = s.get("why").and_then(|x| x.as_str()).unwrap_or("定位不到");
                if !whys.iter().any(|x| x == w) {
                    whys.push(w.to_string());
                }
            }
        }
        if let Some(o) = out.as_object_mut() {
            o.insert(
                "error".into(),
                Value::String(if whys.is_empty() {
                    "记录已被删除".into()
                } else {
                    whys.join("；")
                }),
            );
        }
    }
    out
}

/// 清空全部记录与回放（统计随之归零）。
pub fn clear() -> Value {
    let p = records_path();
    let removed = parse_lines(&p).len() as i64;
    let _ = fs::write(&p, "");
    let _ = fs::write(replays_path(), "");
    cache_cell().lock().unwrap().clear();
    REPLAY_READY.store(true, Ordering::SeqCst); // 文件已空，别再回头读旧内容
    serde_json::json!({"ok": true, "removed_rec": removed})
}

// ---------------------------------------------------------------------------
// 时间（记录里的 `time` 字段 / rid 主键）
// ---------------------------------------------------------------------------
// std 只有 UTC，没有本地时区。前端 `/api/new` 会把 `new Date().getTimezoneOffset()`
// 一起传上来（分钟，东八区 = -480），这里换算成本地时间；没传就按东八区兜底
// （本产品只有中文用户，宁可显示北京时间也不显示 UTC）。
static TZ_MIN: OnceLock<Mutex<i32>> = OnceLock::new();

fn tz_cell() -> &'static Mutex<i32> {
    TZ_MIN.get_or_init(|| Mutex::new(-480))
}

/// 设置本地时区偏移（分钟，与 JS `getTimezoneOffset()` 同号：东八区 = -480）。
pub fn set_tz_minutes(m: i32) {
    if (-14 * 60..=14 * 60).contains(&m) {
        *tz_cell().lock().unwrap() = m;
    }
}

fn civil_from_days(z: i64) -> (i64, u32, u32) {
    let z = z + 719_468;
    let era = if z >= 0 { z } else { z - 146_096 } / 146_097;
    let doe = (z - era * 146_097) as i64;
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = (doy - (153 * mp + 2) / 5 + 1) as u32;
    let m = if mp < 10 { mp + 3 } else { mp - 9 } as u32;
    (y + if m <= 2 { 1 } else { 0 }, m, d)
}

/// 当前时刻的本地时间串 `YYYY-MM-DD HH:MM:SS`（记录面板第一列）。
pub fn now_str() -> String {
    let off = *tz_cell().lock().unwrap() as i64;
    let secs = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0)
        - off * 60;
    let days = secs.div_euclid(86_400);
    let rem = secs.rem_euclid(86_400);
    let (y, m, d) = civil_from_days(days);
    format!(
        "{:04}-{:02}-{:02} {:02}:{:02}:{:02}",
        y,
        m,
        d,
        rem / 3600,
        (rem % 3600) / 60,
        rem % 60
    )
}

/// 毫秒时间戳（回放主键 `rid` 的前半段；同一秒连打两手也不会撞）。
pub fn now_ms() -> u128 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis())
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 历史目录是**进程级单例**（`DIR`），三个用例并行跑会互相把目录改走 ——
    /// 所以这里串行化，每个用例进临界区后再 set_dir 自己的临时目录。
    static TEST_LOCK: Mutex<()> = Mutex::new(());

    fn tmp_dir(tag: &str) -> PathBuf {
        let p = std::env::temp_dir().join(format!("dushan_rec_test_{tag}_{}", std::process::id()));
        let _ = fs::remove_dir_all(&p);
        fs::create_dir_all(&p).unwrap();
        p
    }

    #[test]
    fn append_and_load_newest_first() {
        let _g = TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let d = tmp_dir("load");
        set_dir(d.clone());
        for i in 0..3 {
            append(
                &serde_json::json!({"time": format!("t{i}"), "rid": format!("r{i}"),
                                    "type": "win", "my_seat": 0, "deltas": [1.0, -1.0, 0.0, 0.0]}),
                Some(serde_json::json!({"ev": []})),
            );
        }
        let rows = load(2);
        assert_eq!(rows.len(), 2);
        assert_eq!(rows[0]["rid"], "r2");
        assert_eq!(rows[0]["_ln"], 2);
        assert_eq!(rows[0]["has_replay"], true);
        assert!(replay("r1").is_some());
        let _ = fs::remove_dir_all(&d);
    }

    #[test]
    fn stats_counts_only_finished_hands() {
        let _g = TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let d = tmp_dir("stats");
        set_dir(d.clone());
        append(
            &serde_json::json!({"rid": "a", "type": "win", "my_seat": 0, "is_tsumo": true,
                "winners": [0], "fan_cn": "清一色", "deltas": [8.0, -3.0, -3.0, -2.0],
                "detail": [["a", "冲锋鸡", "1条", 2.0]],
                "flags": {"chongfeng": true, "ji_discard": true, "win_bigfan": true,
                          "hint_used": false}}),
            None,
        );
        append(
            &serde_json::json!({"rid": "b", "type": "huang", "my_seat": 0,
                "deltas": [0.0, 0.0, 0.0, 0.0], "tenpai": [0, 1],
                "flags": {"huang_noten": true, "hint_used": true}}),
            None,
        );
        append(&serde_json::json!({"rid": "c", "type": "abort"}), None);
        let s = stats();
        let a = &s["all_time"];
        assert_eq!(a["games"], 2);      // abort 不计
        assert_eq!(a["zimo"], 1);
        assert_eq!(a["songhu_games"], 0);
        assert_eq!(a["huang_games"], 1);
        assert_eq!(a["huang_noten_games"], 1);
        assert_eq!(a["chongfeng_games"], 1);
        assert_eq!(a["win_bigfan_games"], 1);
        assert_eq!(a["solo_games"], 1);          // 只有 a 局没用推荐
        assert_eq!(a["solo_win_games"], 1);
        assert!((a["score"].as_f64().unwrap() - 8.0).abs() < 1e-9);
        let _ = fs::remove_dir_all(&d);
    }

    #[test]
    fn delete_by_ln_and_clear() {
        let _g = TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let d = tmp_dir("del");
        set_dir(d.clone());
        for i in 0..4 {
            append(
                &serde_json::json!({"rid": format!("r{i}"), "type": "win", "my_seat": 0,
                                    "deltas": [0.0, 0.0, 0.0, 0.0]}),
                Some(serde_json::json!({"kind": "x"})),
            );
        }
        let r = delete(&[(Some(1), Some("r1".to_string()))]);
        assert_eq!(r["ok"], true);
        assert_eq!(r["removed_rec"], 1);
        assert_eq!(r["removed_replay"], 1);
        assert!(replay("r1").is_none());
        assert!(replay("r2").is_some());
        let rows = load(10);
        assert_eq!(rows.len(), 3);
        assert!(rows.iter().all(|x| x["rid"] != "r1"));

        let c = clear();
        assert_eq!(c["removed_rec"], 3);
        assert_eq!(stats()["all_time"]["games"], 0);
        let _ = fs::remove_dir_all(&d);
    }
}
