//! O1：会话循环 / 三家 AI / 累计分 / 结算载荷。
//!
//! 与 Python `Session` 的行为对齐，但按 §7.2 做单机简化：
//!   * 不启后台线程——`/api/state` 每次轮询时按需推进 AI；**节奏由单调时钟
//!     按秒把关**（2026-11-12），speed 是「相邻两个 AI 动作的最小间隔」，
//!     0 = 瞬发。旧实现只拿它分桶决定每轮询推进几步，手填秒数完全不生效；
//!   * 历史 / 回放 / 统计不落盘（接口返回合法空 JSON）；
//!   * 结算直接复用 `rust_engine::settle`（已与 Python 逐位对拍），再转成
//!     `_result_payload` 的字段形状给前端结算页。

use std::time::{Duration, Instant};

use dushan_engine::actions::{self, Action};
use dushan_engine::fan::FanTable;
use dushan_engine::rng::Rng;
use dushan_engine::settle;
use dushan_engine::state::{Config, Game};
use dushan_engine::tiles;
use serde_json::{json, Map, Value};

use crate::policy::{self, Net};
use crate::search::{self, Cfg};
use crate::view::{self, SnapshotCtx, PAGE_VER};

/// 三张网的注册表（键 = ONNX 的 stem）。
///
/// 2026-11-11 四档化之前这里是 `slim`/`master` 两个字段、按编码版本选网；
/// 现在多了 0.12M 的 expert 网，且同一编码（v3）下有两张网，按 enc 选已经
/// 不够用 —— 必须按**策略串里的 stem** 精确取，否则「选进阶却跑传奇」。
pub struct Nets {
    pub map: Vec<(&'static str, Net)>,
}

impl Nets {
    pub fn get(&self, stem: &str) -> Option<&Net> {
        self.map.iter().find(|(s, _)| *s == stem).map(|(_, n)| n)
    }
    pub fn any(&self) -> bool { !self.map.is_empty() }
}

/// 把前端传来的 stem 收敛到三张真实权重之一。
///
/// 认别名是因为历史配置里 master 有 `master`/`dushan_master` 两种写法；
/// 不认识的 stem 一律回落到主力学生（导不了权重时也要有牌可打）。
pub fn canon_stem(raw: &str) -> &'static str {
    match raw {
        "master" | "dushan_master" => "dushan_master",
        "expert_s7" | "s7" => "expert_s7",
        "slim_s" | "champ_slim_s" => "champ_slim_s",
        _ => "champ_slim_s",
    }
}

pub struct Session {
    pub game: Game,
    pub ft: FanTable,
    /// 本局的规则配置。前端 `/api/new` / `/api/next` 的 rules 载荷解析结果，
    /// 下一局沿用（对齐线上 Session 的 self.rules）。
    pub cfg: Config,
    pub me: u8,
    pub names: [String; 4],
    pub kinds: [&'static str; 4],
    /// 每个座位对手的 determinization 数：0 = 贪心，32/128 = 真搜索。
    /// 四档全部是贪心，这里只保留给「老配置里还存着搜索档」的兼容路径。
    pub opp_det: [u32; 4],
    /// 每个座位的策略：`ting` 前缀走速听规则，其余按 `opp_stem` 取网。
    pub opp_ting: [bool; 4],
    pub opp_stem: [&'static str; 4],
    pub cum: [f64; 4],
    pub hand_index: u32,
    pub dealer: u8,
    pub dealer_streak: u32,
    pub events: Vec<Value>,
    pub seq: u32,
    pub pending: Option<Value>,
    pub result: Option<Value>,
    pub stats: Value,
    pub opp_think: Value,
    pub error: Option<String>,
    /// 对手出牌节奏（秒）：相邻两个 AI 动作之间的最小间隔，0 = 瞬发。
    pub speed: f64,
    /// 下一次允许推进 AI 的单调时刻。`/api/speed` 改节奏时立即重算，
    /// 避免「把 3 秒改成 0.5 秒还要再等 3 秒」。
    next_ai_at: Instant,
    pub force_end: bool,
    pub settled: bool,
    pub seed: u64,
}

fn round2(x: f64) -> f64 { (x * 100.0).round() / 100.0 }
fn round3(x: f64) -> f64 { (x * 1000.0).round() / 1000.0 }

fn meld_label(mt: u8) -> &'static str {
    use dushan_engine::fan;
    match mt {
        fan::MELD_PONG => "碰",
        fan::MELD_KONG_EXPOSED => "明杠",
        fan::MELD_KONG_ADDED => "补杠",
        fan::MELD_KONG_CONCEALED => "暗杠",
        _ => "?",
    }
}

fn f(v: &Value, k: &str, d: f64) -> f64 {
    v.get(k).and_then(|x| x.as_f64()).unwrap_or(d)
}

fn arr_f64(v: &Value, k: &str) -> Vec<f64> {
    v.get(k)
        .and_then(|x| x.as_array())
        .map(|a| a.iter().map(|x| x.as_f64().unwrap_or(0.0)).collect())
        .unwrap_or_else(|| vec![0.0; 4])
}

fn arr_i64(v: &Value, k: &str) -> Vec<i64> {
    v.get(k)
        .and_then(|x| x.as_array())
        .map(|a| a.iter().filter_map(|x| x.as_i64()).collect())
        .unwrap_or_default()
}

impl Session {
    /// `names_rel[rel]` 是前端传来的相对座位名；`strats_rel` 是三家对手的
    /// 策略串（`ting:fast` / `model:dushan_master` / `model:expert_s7` / ...）。
    /// 转成绝对座位后存入。
    pub fn new(
        ft: FanTable,
        seed: u64,
        me: u8,
        names_rel: &[String; 4],
        strats_rel: &[String; 3],
        cfg: Config,
    ) -> Self {
        let mut names = ["我", "下家", "对家", "上家"].map(str::to_string);
        for rel in 0..4usize {
            names[(me as usize + rel) % 4] = names_rel[rel].clone();
        }
        let mut kinds = ["model"; 4];
        kinds[me as usize] = "human";
        let mut opp_stem = ["champ_slim_s"; 4];
        let mut opp_ting = [false; 4];
        let mut opp_det = [0u32; 4];
        for rel in 1..4usize {
            let seat = (me as usize + rel) % 4;
            let (kind, stem) = strats_rel[rel - 1]
                .split_once(':')
                .unwrap_or(("model", strats_rel[rel - 1].as_str()));
            opp_ting[seat] = kind == "ting";
            opp_stem[seat] = canon_stem(stem);
            opp_det[seat] = match kind {
                "search32" => 32,
                "search128" => 128,
                _ => 0,
            };
            kinds[seat] = if opp_ting[seat] {
                "ting"
            } else if opp_det[seat] == 32 {
                "search32"
            } else if opp_det[seat] == 128 {
                "search128"
            } else {
                "model"
            };
        }
        let mut wall = tiles::full_deck();
        Rng::new(seed).shuffle(&mut wall);
        let dealer = (seed % 4) as u8;
        let game = Game::new_with_wall(wall, dealer, cfg, &ft);
        Self {
            game,
            cfg,
            ft,
            me,
            names,
            kinds,
            opp_det,
            opp_ting,
            opp_stem,
            cum: [0.0; 4],
            hand_index: 1,
            dealer,
            dealer_streak: 0,
            events: Vec::new(),
            seq: 0,
            pending: None,
            result: None,
            stats: json!({"0": {}, "1": {}, "2": {}, "3": {}}),
            opp_think: json!({}),
            error: None,
            speed: 0.5,
            next_ai_at: Instant::now(),
            force_end: false,
            settled: false,
            seed,
        }
    }

    /// 设置对手节奏（0~5 秒）。改完立刻放行下一次 AI 步，用户不用等旧节奏走完。
    pub fn set_speed(&mut self, v: f64) {
        self.speed = v.clamp(0.0, 5.0);
        self.next_ai_at = Instant::now();
    }

    fn pace_after_human(&mut self) {
        // 自己出完牌之后，对手也要隔 speed 秒才动（与线上 Session 的
        // `time.sleep(self.speed)` 同口径），不能因为计时器还停在过去而瞬发。
        self.next_ai_at = Instant::now() + Duration::from_secs_f64(self.speed.max(0.0));
    }

    fn rel_of(&self, seat: i64) -> i64 {
        if seat < 0 { -1 } else { (seat - self.me as i64 + 4) % 4 }
    }

    fn push_event(&mut self, kind: &str, seat: i64, tile: Option<u8>, text: String) {
        self.seq += 1;
        self.events.push(json!({
            "t": 0.0,
            "hand": self.hand_index,
            "seq": self.seq,
            "kind": kind,
            "rel": self.rel_of(seat),
            "text": text,
            "tile": tile,
            "ctag": "",
        }));
        if self.events.len() > 600 {
            self.events.remove(0);
        }
    }

    /// 推进最多 `max_steps` 个 AI 动作；遇到「我的决策点」或终局即停。
    /// 返回真正执行了几步（供 snapshot 决定下一次放行时刻）。
    pub fn advance(&mut self, nets: &Nets, max_steps: usize) -> usize {
        let mut done = 0usize;
        for _ in 0..max_steps {
            if self.game.phase == actions::PHASE_OVER {
                self.finalize();
                break;
            }
            if self.force_end {
                // 与服务端 end_hand_now 同义：直接收掉本手。
                self.game.phase = actions::PHASE_OVER;
                self.finalize();
                break;
            }
            let actor = self.game.actor();
            if actor == self.me {
                self.pending = Some(view::pending_payload(&self.game, &self.ft));
                break;
            }
            let a = self.decide(nets, actor);
            self.apply(a);
            done += 1;
        }
        if self.game.phase == actions::PHASE_OVER {
            self.finalize();
        }
        done
    }

    /// 某个 AI 座位做一次决策：速听（规则）、贪心或真 PIMC 搜索（det=32/128）。
    fn decide(&self, nets: &Nets, actor: u8) -> Action {
        let heuristic = || policy::heuristic_choose(&self.game, actor, &self.ft);
        if self.opp_ting[actor as usize] {
            return crate::ting::choose(&self.game, actor, &self.ft);
        }
        let Some(net) = nets.get(self.opp_stem[actor as usize]) else {
            return heuristic();
        };
        let det = self.opp_det[actor as usize];
        if det == 0 {
            return net.choose(&self.game, actor, &self.ft).unwrap_or_else(heuristic);
        }
        let legal = self.game.legal_actions(&self.ft);
        if legal.len() <= 1 {
            return legal.first().copied().unwrap_or_else(Action::pass);
        }
        let cfg = Cfg { n_det: det as usize, ff_cap: 24, topk: 3, ply: 1 };
        let seed = self.seed
            ^ (self.hand_index as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15)
            ^ (self.seq as u64).wrapping_mul(0x2545_F491_4F6C_DD1D);
        match search::deep_q(net, &self.game, actor, &legal, &cfg, seed, &self.ft) {
            Ok(q) if !q.is_empty() => {
                let mut best = 0usize;
                for (i, v) in q.iter().enumerate() {
                    if *v > q[best] {
                        best = i;
                    }
                }
                legal.get(best).copied().unwrap_or_else(Action::pass)
            }
            _ => net.choose(&self.game, actor, &self.ft).unwrap_or_else(heuristic),
        }
    }

    fn apply(&mut self, a: Action) {
        let seat = self.game.actor() as i64;
        let kind = match a.kind {
            actions::DISCARD => Some("discard"),
            actions::HU => Some("hu"),
            actions::PENG => Some("peng"),
            actions::ANGANG | actions::BUGANG | actions::MINGGANG => Some("kong"),
            _ => None,
        };
        if let Some(k) = kind {
            let tile = if a.tile == 255 { None } else { Some(a.tile) };
            let who = if seat >= 0 { self.names[seat as usize].clone() } else { "全场".into() };
            let text = format!("{} {}", who, a.label());
            self.push_event(k, seat, tile, text);
        }
        self.game.step(a, &self.ft);
        self.pending = None;
        if self.game.phase == actions::PHASE_OVER {
            self.finalize();
        }
    }

    /// 人类动作（`/api/act`）。返回 `(ok, action_label_or_error)`。
    pub fn submit(&mut self, kind: &str, tile: i64) -> (bool, String) {
        if self.game.phase == actions::PHASE_OVER {
            return (false, "当前这局已结束".into());
        }
        if self.game.actor() != self.me {
            return (false, "还没轮到你".into());
        }
        let Some(a) = view::parse_action(kind, tile) else {
            return (false, format!("非法动作 {kind}"));
        };
        let legal = self.game.legal_actions(&self.ft);
        if !legal.iter().any(|x| x.kind == a.kind && x.tile == a.tile) {
            return (false, format!("非法动作 {}", a.label()));
        }
        let label = a.label();
        self.apply(a);
        self.pace_after_human();
        (true, label)
    }

    fn finalize(&mut self) {
        if self.settled {
            return;
        }
        self.settled = true;
        self.pending = None;
        let res = match self.game.outcome {
            Some(o) if o.n_winners > 0 => settle::settle_win(&mut self.game, &self.ft),
            _ => settle::settle_huangzhuang(&self.game, &self.ft),
        };
        let payload = self.result_payload(&res);
        let deltas = arr_f64(&res, "deltas");
        for p in 0..4 {
            self.cum[p] = round2(self.cum[p] + deltas.get(p).copied().unwrap_or(0.0));
        }
        self.result = Some(payload);
        let how = res.get("type").and_then(|x| x.as_str()).unwrap_or("流局");
        self.push_event("result", -1, None, format!("本局结算：{how}"));
    }

    /// 覆盖规则配置：`/api/next` 携带新 rules 时调用，**下一局**生效
    /// （与线上 Session.self.rules 的语义一致：设置面板标了「下一局生效」）。
    pub fn set_rules(&mut self, rules: Option<&Value>) {
        self.cfg = crate::rules_cfg::config_from_rules(rules);
    }

    /// 下一手：轮庄、清事件、重建牌墙。
    pub fn next_hand(&mut self) -> bool {
        if self.game.phase != actions::PHASE_OVER {
            return false;
        }
        self.seed = self.seed.wrapping_add(7919);
        self.hand_index += 1;
        // 简化轮庄：非连庄时顺延；连庄口径不影响规则内核。
        self.dealer = (self.dealer + 1) % 4;
        self.dealer_streak = 0;
        let mut wall = tiles::full_deck();
        Rng::new(self.seed).shuffle(&mut wall);
        self.game = Game::new_with_wall(wall, self.dealer, self.cfg, &self.ft);
        self.events.clear();
        self.seq = 0;
        self.pending = None;
        self.result = None;
        self.settled = false;
        self.force_end = false;
        self.next_ai_at = Instant::now();
        true
    }

    pub fn end_hand_now(&mut self) {
        self.force_end = true;
    }

    /// `/api/state`：按真实秒数放行 AI，再出快照。
    ///
    /// 前端 450 ms 轮询只是「来问一次」，不是节奏本身：旧实现把 speed 当成
    /// 「每轮询走几步」的分桶（≥2 秒反而一次走 3 步），用户手填 2.5 秒与
    /// 默认 0.45 秒完全无差别。现在用单调时钟记录下一次放行时刻，
    /// speed=0 时一次把 AI 推到我的决策点（瞬发）。
    pub fn snapshot(&mut self, nets: &Nets) -> Value {
        let due = self.speed <= 0.0 || Instant::now() >= self.next_ai_at;
        if due {
            let cap = if self.speed <= 0.0 { 64 } else { 1 };
            let n = self.advance(nets, cap);
            if n > 0 {
                self.next_ai_at = Instant::now() + Duration::from_secs_f64(self.speed.max(0.0));
            }
        }
        let pending = if self.game.phase == actions::PHASE_OVER {
            None
        } else if self.game.actor() == self.me {
            Some(view::pending_payload(&self.game, &self.ft))
        } else {
            self.pending.clone()
        };
        let err = self.error.as_deref();
        view::snapshot(&SnapshotCtx {
            game: &self.game,
            ft: &self.ft,
            me: self.me,
            sid: "offline",
            names: &self.names,
            kinds: &self.kinds,
            hand_index: self.hand_index,
            cum: &self.cum,
            dealer_streak: self.dealer_streak,
            events: &self.events,
            pending,
            result: self.result.clone(),
            error: err,
            stats: &self.stats,
            opp_think: &self.opp_think,
        })
    }

    /// `/api/hint`：按策略给出推荐动作。
    ///
    /// 策略串形如 `ting:fast` / `model:dushan_master` / `model:expert_s7` /
    /// `model:champ_slim_s`：
    ///   * `ting:` 前缀 → 速听规则（adept），不加载任何模型；
    ///   * `model:` 前缀 → 对应网的贪心（expert / master / legend）；
    ///   * `search32/128:` 只作老配置兼容，四档本身不搜索。
    /// 逐级降级：搜索 → 贪心/速听 → 纯规则，保证永远有推荐。
    pub fn hint(&mut self, nets: &Nets, strategy: &str) -> Value {
        self.advance(nets, 0);
        if self.game.phase == actions::PHASE_OVER || self.game.actor() != self.me {
            return json!({"ok": false, "error": "当前不是你的决策点"});
        }
        let (kind, raw_stem) = strategy.split_once(':').unwrap_or(("model", strategy));
        let stem = canon_stem(raw_stem);
        let is_ting = kind == "ting";
        let net = nets.get(stem);
        let enc = net.map(|n| n.enc()).unwrap_or(3);
        let legal = self.game.legal_actions(&self.ft);
        let t0 = std::time::Instant::now();

        // 1) 真搜索（仅老配置；四档都是贪心）。
        let mut scored: Vec<(f64, Action)> = Vec::new();
        let mut engine = "tract";
        if let (Some(cfg), Some(n)) = (Cfg::from_kind(kind), net) {
            let seed = self.seed
                ^ (self.hand_index as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15)
                ^ (self.seq as u64).wrapping_mul(0x2545_F491_4F6C_DD1D);
            if let Ok(q) = search::deep_q(n, &self.game, self.me, &legal, &cfg, seed, &self.ft) {
                for (i, a) in legal.iter().enumerate() {
                    scored.push((q.get(i).copied().unwrap_or(0.0), *a));
                }
                engine = "tract-pimc";
            }
        }
        // 2) 速听 / 贪心。
        if scored.is_empty() {
            if is_ting {
                scored = crate::ting::rank(&self.game, self.me, &self.ft);
                engine = "rule-ting";
            } else if let Some(v) = net.and_then(|n| n.rank(&self.game, self.me, &self.ft)) {
                scored = v.into_iter().map(|(s, a)| (s as f64, a)).collect();
            }
        }
        scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

        // 3) 纯规则兜底。
        if scored.is_empty() {
            let mut l = legal.clone();
            l.sort_by_key(|a| match a.kind {
                actions::HU => 0,
                actions::ANGANG | actions::BUGANG | actions::MINGGANG => 1,
                actions::PENG => 2,
                _ => 3,
            });
            scored = l.into_iter().take(3).map(|a| (0.0, a)).collect();
            engine = "rule-fallback";
        }

        let top: Vec<(f64, Action)> = scored.into_iter().take(3).collect();
        // prob 用 top 内 softmax（q 是分差量纲，温度 1 即可给出相对强弱）。
        let max_q = top.first().map(|x| x.0).unwrap_or(0.0);
        let exps: Vec<f64> = top.iter().map(|x| (x.0 - max_q).exp()).collect();
        let z: f64 = exps.iter().sum::<f64>().max(1e-9);
        let reason = match engine {
            "tract-pimc" => format!("离线 PIMC {} 路推演推荐", kind.trim_start_matches("search")),
            "rule-ting" => "速听：向听数与进张优先".to_string(),
            "rule-fallback" => "规则兜底推荐".to_string(),
            _ => "离线版模型贪心推荐".to_string(),
        };
        let items: Vec<Value> = top
            .iter()
            .enumerate()
            .map(|(i, (q, a))| json!({
                "label": a.label(),
                "kind": actions::KIND_NAMES[a.kind as usize],
                "tile": if a.tile == 255 { -1 } else { a.tile as i64 },
                "prob": round3(exps[i] / z),
                "q": round2(*q),
                "reason": reason,
                "tags": [],
                "lines": [],
            }))
            .collect();
        let model_name = if is_ting { "ting:fast" } else { stem };

        json!({
            "ok": true,
            "model": model_name,
            "head": if engine == "tract-pimc" { kind } else { "model" },
            "value": round2(top.first().map(|x| x.0).unwrap_or(0.0)),
            "items": items,
            "situation": {"tenpai": false, "calls": [], "shanten": 0,
                          "ji_held": 0, "baojiao": false, "has_kong": false, "note": ""},
            "tip": match engine {
                "tract-pimc" => "离线单机：设备端 PIMC 推演给出推荐".to_string(),
                "rule-ting" => "离线单机：速听规则给出推荐".to_string(),
                _ => "离线单机：已用设备端模型给出推荐".to_string(),
            },
            "note": "",
            "elapsed_ms": t0.elapsed().as_millis() as u64,
            "src": {"stem": if is_ting { "ting" } else { stem },
                    "engine": engine, "enc": enc, "enc_expected": enc},
            "src_ok": true,
        })
    }

    /// `/api/ting`：当前听牌张（map 结构前端兼容）。
    pub fn ting(&mut self, mode_now: bool) -> Value {
        let tiles: Vec<u8> = self.game.winning_tiles(self.me as usize, &self.ft);
        if mode_now {
            let cur = if tiles.is_empty() {
                Value::Null
            } else {
                json!({"tiles": tiles.iter().map(|&t| json!({"t": t, "left": 4})).collect::<Vec<_>>(),
                       "types": []})
            };
            json!({"ok": true, "cur": cur})
        } else {
            let mut map = Map::new();
            for t in &tiles {
                map.insert(t.to_string(), json!({
                    "tiles": [{"t": t, "left": 4}], "types": []
                }));
            }
            json!({"ok": true, "map": map})
        }
    }

    /// 把 `settle.rs` 的 JSON 转成 Python `_result_payload` 的字段形状。
    fn result_payload(&self, res: &Value) -> Value {
        let me = self.me as usize;
        let g = &self.game;
        let deltas = arr_f64(res, "deltas");
        let mut o = Map::new();
        let typ = res.get("type").and_then(|x| x.as_str()).unwrap_or("huang");
        o.insert("type".into(), json!(if typ == "win" { "win" } else { "huang" }));
        o.insert("deltas".into(), json!(deltas.iter().map(|&x| round2(x)).collect::<Vec<_>>()));
        o.insert("wall_left".into(), json!(g.wall_left()));
        o.insert("mine".into(), json!(round2(*deltas.get(me).unwrap_or(&0.0))));
        for k in ["follow_dealer", "follow_tile", "follow_pay"] {
            o.insert(k.into(), res.get(k).cloned().unwrap_or(Value::Null));
        }
        let ft_cn = res.get("follow_tile")
            .and_then(|x| x.as_i64())
            .filter(|&t| (0..27).contains(&t))
            .map(|t| tiles::tile_cn(t as u8)).unwrap_or_default();
        o.insert("follow_tile_cn".into(), json!(ft_cn));

        // detail: [p,label,info,value] → {who,seat,label,info,value}
        let detail: Vec<Value> = res.get("detail").and_then(|x| x.as_array())
            .map(|a| a.iter().map(|d| {
                let arr = d.as_array();
                let seat = arr.and_then(|x| x.first()).and_then(|x| x.as_i64()).unwrap_or(-1);
                let label = arr.and_then(|x| x.get(1)).and_then(|x| x.as_str()).unwrap_or("");
                let info = arr.and_then(|x| x.get(2)).and_then(|x| x.as_str()).unwrap_or("");
                let val = arr.and_then(|x| x.get(3)).and_then(|x| x.as_f64()).unwrap_or(0.0);
                json!({"who": if seat < 0 { "全场".into() } else { self.names[seat as usize].clone() },
                       "seat": seat, "label": label, "info": info, "value": val})
            }).collect()).unwrap_or_default();
        let fanji = detail.iter().find(|d| d["label"] == "翻鸡")
            .and_then(|d| d["info"].as_str().map(str::to_string));
        o.insert("detail".into(), json!(detail));
        o.insert("fanji".into(), json!(fanji));

        let pair = res.get("pair").cloned()
            .unwrap_or_else(|| json!([[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0]]));
        o.insert("pair".into(), pair);
        o.insert("pair_detail".into(), res.get("pair_detail").cloned().unwrap_or(json!([])));

        for k in ["ji_hold_detail", "fanji_detail", "ji_discard_detail",
                  "ji_capture_detail", "ji_claim_detail", "gold_tiles",
                  "quad_gold_tiles", "fanji_tiles", "fanji_flip", "fanji_gold",
                  "jiesuan_fanji", "kaiju_fanji", "how", "how_value",
                  "type_value", "win_tile", "winners_fan", "chickens",
                  "baodapai", "baodapai_on", "fan", "fan_cn"] {
            if let Some(v) = res.get(k) {
                o.insert(k.into(), v.clone());
            }
        }
        o.insert("gold_mult".into(), json!(g.cfg.jin_mult));
        let ji_tiles: Vec<u8> = (0..27u8).filter(|&t| g.ji_tiles & (1u32 << t) != 0).collect();
        o.insert("ji_tiles".into(), json!(ji_tiles));
        o.insert("kaiju_flip".into(), json!(g.kaiju_flip));
        o.insert("zi_mo".into(), json!(g.cfg.zi_mo));

        if typ == "win" {
            let winners = arr_i64(res, "winners");
            let w = res.get("winner").and_then(|x| x.as_i64()).unwrap_or(0);
            let winners = if winners.is_empty() { vec![w] } else { winners };
            let loser = res.get("loser").and_then(|x| x.as_i64());
            let void = res.get("void_player").and_then(|x| x.as_i64());
            o.insert("winner".into(), json!(w));
            o.insert("winner_rel".into(), json!(self.rel_of(w)));
            o.insert("winner_name".into(), json!(self.names.get(w as usize).cloned().unwrap_or_default()));
            o.insert("winners".into(), json!(winners));
            o.insert("winners_rel".into(), json!(winners.iter().map(|&x| self.rel_of(x)).collect::<Vec<_>>()));
            o.insert("winner_names".into(), json!(winners.iter().map(|&x| self.names.get(x as usize).cloned().unwrap_or_default()).collect::<Vec<_>>()));
            o.insert("loser".into(), json!(loser));
            o.insert("loser_name".into(), json!(loser.map(|x| self.names.get(x as usize).cloned().unwrap_or_default())));
            o.insert("void_name".into(), json!(void.map(|x| self.names.get(x as usize).cloned().unwrap_or_default())));
            o.insert("void_seat".into(), json!(void));
            o.insert("is_tsumo".into(), res.get("is_tsumo").cloned().unwrap_or(json!(false)));
            o.insert("rob_kong".into(), res.get("rob_kong").cloned().unwrap_or(json!(false)));
            let hand_of = |p: usize| -> Vec<u8> {
                let mut v = Vec::new();
                for t in 0..27u8 { for _ in 0..g.hands[p][t as usize] { v.push(t); } }
                v
            };
            let melds_of = |p: usize| -> Vec<Value> {
                g.melds[p].iter().map(|&(mt, t, s)| json!({
                    "type": mt, "label": meld_label(mt), "tile": t, "src": s, "ji": Value::Null
                })).collect()
            };
            o.insert("winner_hand".into(), json!(hand_of(w as usize)));
            let mut hands = Map::new();
            let mut melds_map = Map::new();
            for &x in &winners {
                hands.insert(x.to_string(), json!(hand_of(x as usize)));
                melds_map.insert(x.to_string(), json!(melds_of(x as usize)));
            }
            o.insert("winner_hands".into(), Value::Object(hands));
            o.insert("winner_melds".into(), json!(melds_of(w as usize)));
            o.insert("winner_melds_map".into(), Value::Object(melds_map));
        } else {
            let tp = arr_i64(res, "tenpai");
            o.insert("tenpai".into(), json!(tp));
            o.insert("tenpai_rel".into(), json!(tp.iter().map(|&x| self.rel_of(x)).collect::<Vec<_>>()));
            o.insert("tenpai_names".into(), json!(tp.iter().map(|&x| self.names.get(x as usize).cloned().unwrap_or_default()).collect::<Vec<_>>()));
        }
        o.insert("page_ver".into(), json!(PAGE_VER));
        Value::Object(o)
    }
}
