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
use crate::coach;

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
    /// 本手是否请求过大师推荐（`/api/hint`）。「本人胡牌 / 本人累计」两个统计
    /// 项的分母是「全程没开过推荐的局」（`solo_games`），与线上同口径。
    pub hint_used: bool,
    /// 发牌快照（27 牌种紧凑编码，每家一个串）：回放的**第 0 帧**。
    pub deal_packed: [String; 4],
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

/// 27 个牌种各映射 1 个字符（与线上 `pack_hand` 同一张表）。
/// 一手 13 张 = 13 字符，四家 ≈ 55 字节 —— 回放逐帧带手牌也不会把文件撑爆。
pub fn pack_hand(c: &[u8; 27]) -> String {
    const CH: &[u8] = b"0123456789abcdefghijklmnopqrstuvwxyz";
    let mut out = String::with_capacity(14);
    for (t, &n) in c.iter().enumerate() {
        for _ in 0..n {
            out.push(CH[t] as char);
        }
    }
    out
}

/// 本会话四家的计数桶（`/api/state` 的 `stats`，统计面板「本会话四家」用）。
fn new_stats() -> Value {
    let mut o = Map::new();
    for p in 0..4 {
        o.insert(p.to_string(), json!({
            "games": 0, "wins": 0, "tsumo": 0, "ron": 0, "deal_in": 0,
            "chongfeng": 0, "hengji": 0, "baoji": 0, "baoji_pay": 0.0,
            "huang": 0, "tenpai_huang": 0, "score": 0.0
        }));
    }
    Value::Object(o)
}

fn bump(stats: &mut Value, p: usize, key: &str, by: f64) {
    let Some(o) = stats.get_mut(p.to_string()).and_then(|x| x.as_object_mut()) else {
        return;
    };
    let cur = o.get(key).and_then(|x| x.as_f64()).unwrap_or(0.0);
    o.insert(key.to_string(), json!(cur + by));
}

/// 某个座位（可先打掉 `drop` 那张）的等待张与**非平胡**牌型集合。
///
/// 与线上 `Session._make_record` 里的 `game.ting_info(p, drop=...)` 同义，但
/// **两边的终局手牌口径不同**：Python 结算后赢家手牌是 14 张（胡牌张并进去了），
/// 需要先扣；Rust `settle_win` 末尾会把胡牌张「复原」回原位，手牌本来就是
/// 13 张（`(4-副露)*3+1`）。所以这里按**张数**判断：多一张才扣，不多不扣 ——
/// 只按 drop 硬扣会把赢家算成 12 张、等待张直接算空（记录里看不出「听什么」）。
fn waits_of(g: &Game, p: usize, drop: Option<u8>, ft: &FanTable) -> (Vec<u8>, Vec<String>) {
    let n_melds = g.melds[p].len();
    if n_melds > 4 {
        return (Vec::new(), Vec::new());
    }
    let expect = (4 - n_melds) * 3 + 1;
    let mut h = g.hands[p];
    let n = tiles::counts_sum(&h) as usize;
    if n == expect + 1 {
        if let Some(d) = drop {
            if (d as usize) < 27 && h[d as usize] > 0 {
                h[d as usize] -= 1;
            }
        }
    }
    if tiles::counts_sum(&h) as usize != expect {
        return (Vec::new(), Vec::new());
    }
    let mut waits = Vec::new();
    let mut types: Vec<String> = Vec::new();
    for t in 0..27u8 {
        if h[t as usize] >= 4 {
            continue;
        }
        let mut c = h;
        c[t as usize] += 1;
        if dushan_engine::fan::is_winning_hand(&c, &g.melds[p], ft) {
            waits.push(t);
            for ty in dushan_engine::fan::win_type_set(&c, &g.melds[p], Some(t), ft) {
                if ty != dushan_engine::fan::PING_HU {
                    // 记录里存中文名（线上 ting_info 给的就是「大对子 / 清一色」），
                    // 英文常量只在引擎内部用；`tenpai_types` 会进导出/统计口径。
                    let cn = settle::fan_cn_of(ty);
                    if !types.iter().any(|x| *x == cn) {
                        types.push(cn);
                    }
                }
            }
        }
    }
    (waits, types)
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
        let deal_packed = [
            pack_hand(&game.hands[0]),
            pack_hand(&game.hands[1]),
            pack_hand(&game.hands[2]),
            pack_hand(&game.hands[3]),
        ];
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
            stats: new_stats(),
            opp_think: json!({}),
            error: None,
            hint_used: false,
            deal_packed,
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

    /// 当前牌面的「逐帧快照」三元组：`(四家手牌张数, 四家手牌紧凑编码, 巡号)`。
    /// 回放器按这三项画别家的牌背 / 我方的牌面，并给「跳到第 N 巡」做刻度。
    fn frame_fields(&self) -> (Value, Value, i64) {
        let hc: Vec<u32> = (0..4)
            .map(|p| tiles::counts_sum(&self.game.hands[p]))
            .collect();
        let hands: Vec<String> = (0..4).map(|p| pack_hand(&self.game.hands[p])).collect();
        let discarded: usize = self.game.discards.iter().map(|d| d.len()).sum();
        (json!(hc), json!(hands), (discarded / 4) as i64 + 1)
    }

    fn push_event(&mut self, kind: &str, seat: i64, tile: Option<u8>, text: String) {
        self.seq += 1;
        let (hc, hands, r) = self.frame_fields();
        self.events.push(json!({
            "t": 0.0,
            "hand": self.hand_index,
            "seq": self.seq,
            "kind": kind,
            "rel": self.rel_of(seat),
            "text": text,
            "tile": tile,
            "ctag": "",
            "hc": hc,
            "hands": hands,
            "r": r,
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
        let mut ev_idx = None;
        if let Some(k) = kind {
            let tile = if a.tile == 255 { None } else { Some(a.tile) };
            let who = if seat >= 0 { self.names[seat as usize].clone() } else { "全场".into() };
            let text = format!("{} {}", who, a.label());
            self.push_event(k, seat, tile, text);
            ev_idx = Some(self.events.len() - 1);
        }
        // 打出的是否鸡牌（决定要不要给这条 discard 事件补 ctag）。
        let is_ji_discard = a.kind == actions::DISCARD && a.tile < 27
            && (self.game.ji_tiles & (1u32 << a.tile)) != 0;
        let hj_before = self.game.hengji_open;
        self.game.step(a, &self.ft);
        // 1) events[].ctag：前端「冲锋鸡/横鸡」全屏特效与 toast 的唯一开关。
        //    必须 step 之后再读 `ji_tag_of_last` —— 不能用 ji_name_if_discarded
        //    预测：该种第一次开横鸡轮时它会判成普通鸡，与 step_discard 实际
        //    记下的横鸡不一致（线上用的是引擎里同名字段的真实值）。
        if let (Some(i), true) = (ev_idx, is_ji_discard) {
            self.events[i]["ctag"] =
                json!(view::ji_tag_cn(self.game.ji_tag_of_last[seat as usize]));
        }
        // 1b) 这条事件是**动作前**推的（要先用旧牌面判鸡牌），回放的逐帧快照
        //     必须换成动作生效后的状态，否则回放里每次出牌都少一张、越放越歪。
        if let Some(i) = ev_idx {
            let (hc, hands, r) = self.frame_fields();
            self.events[i]["hc"] = hc;
            self.events[i]["hands"] = hands;
            self.events[i]["r"] = json!(r);
        }
        // 2) 横鸡轮开/收事件：与线上 `_emit_hengji` 同口径，流水/toast 才有
        //    「开出横鸡轮 · X」提示。掩码变化的每一位对应一个独立鸡种轮。
        let changed = hj_before ^ self.game.hengji_open;
        if changed != 0 {
            for t in 0..27u8 {
                let b = 1u32 << t;
                if changed & b == 0 { continue; }
                let open = self.game.hengji_open & b != 0;
                let cn = tiles::tile_cn(t);
                let who = self.names[seat as usize].clone();
                let text = if open {
                    format!("{who} 开出横鸡轮 · {cn}")
                } else {
                    format!("{who} 再次出牌 · 横鸡轮结束（{cn}）")
                };
                self.push_event("hengji", seat, Some(t), text);
                let last = self.events.len() - 1;
                self.events[last]["ctag"] =
                    json!(if open { "横鸡轮开始" } else { "横鸡轮结束" });
            }
        }
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
        self.update_stats(&res);
        // 2026-11-12：对局记录 / 数据统计 / 回放落盘（此前离线版是空实现，
        // 面板永远空白）。「结束本局」提前终止与线上 abort 同义 —— 不计分、
        // 不入历史，所以 force_end 的局只更新画面，不写 records.jsonl。
        if !self.force_end {
            self.persist_hand(&res);
        }
        let how = res.get("type").and_then(|x| x.as_str()).unwrap_or("流局");
        self.push_event("result", -1, None, format!("本局结算：{how}"));
    }

    // -- 历史 / 统计 / 回放（移植自 web/dushan_server.py）--------------------

    /// 本会話四家的计数（`/api/state` 的 `stats`，统计面板「本会话四家」）。
    /// 与线上 `_update_stats` 同口径：**按绝对座位累加**，输出到视图时再转
    /// 相对座位（`players[]` 是相对座位序）。
    fn update_stats(&mut self, res: &Value) {
        let deltas = arr_f64(res, "deltas");
        for p in 0..4 {
            bump(&mut self.stats, p, "games", 1.0);
            bump(&mut self.stats, p, "score", round2(deltas.get(p).copied().unwrap_or(0.0)));
        }
        let detail: Vec<Value> = res
            .get("detail")
            .and_then(|x| x.as_array())
            .cloned()
            .unwrap_or_default();
        let seat_label = |d: &Value| -> (i64, String, f64) {
            let a = d.as_array();
            let seat = a.and_then(|x| x.first()).and_then(|x| x.as_i64()).unwrap_or(-1);
            let label = a
                .and_then(|x| x.get(1))
                .and_then(|x| x.as_str())
                .unwrap_or("")
                .to_string();
            let val = a.and_then(|x| x.get(3)).and_then(|x| x.as_f64()).unwrap_or(0.0);
            (seat, label, val)
        };

        if res.get("type").and_then(|x| x.as_str()) == Some("win") {
            let winners: Vec<i64> = {
                let w = arr_i64(res, "winners");
                if w.is_empty() {
                    res.get("winner").and_then(|x| x.as_i64()).into_iter().collect()
                } else {
                    w
                }
            };
            let tsumo = res.get("is_tsumo").and_then(|x| x.as_bool()).unwrap_or(false);
            let loser = res.get("loser").and_then(|x| x.as_i64());
            for w in &winners {
                if (0..4).contains(w) {
                    let p = *w as usize;
                    bump(&mut self.stats, p, "wins", 1.0);
                    if tsumo {
                        bump(&mut self.stats, p, "tsumo", 1.0);
                    } else if loser.is_some() {
                        bump(&mut self.stats, p, "ron", 1.0);
                    }
                }
            }
            if !tsumo {
                if let Some(l) = loser.filter(|x| (0..4).contains(x)) {
                    bump(&mut self.stats, l as usize, "deal_in", 1.0);
                }
            }
            let mut baoji_seen: Vec<usize> = Vec::new();
            for d in &detail {
                let (seat, label, _v) = seat_label(d);
                if !(0..4).contains(&seat) {
                    continue;
                }
                let p = seat as usize;
                match label.as_str() {
                    "冲锋鸡" => bump(&mut self.stats, p, "chongfeng", 1.0),
                    "横鸡" => bump(&mut self.stats, p, "hengji", 1.0),
                    "包鸡" if !baoji_seen.contains(&p) => {
                        baoji_seen.push(p);
                        bump(&mut self.stats, p, "baoji", 1.0);
                    }
                    _ => {}
                }
            }
        } else {
            let tenpai: Vec<i64> = arr_i64(res, "tenpai");
            for p in 0..4 {
                bump(&mut self.stats, p, "huang", 1.0);
                if tenpai.contains(&(p as i64)) {
                    bump(&mut self.stats, p, "tenpai_huang", 1.0);
                }
            }
            let mut baoji_seen: Vec<usize> = Vec::new();
            for d in &detail {
                let (seat, label, val) = seat_label(d);
                if !(0..4).contains(&seat) {
                    continue;
                }
                let p = seat as usize;
                if label == "包鸡" || label == "包大牌面" {
                    bump(&mut self.stats, p, "baoji_pay", val.abs());
                }
                if label == "包鸡" && !baoji_seen.contains(&p) {
                    baoji_seen.push(p);
                    bump(&mut self.stats, p, "baoji", 1.0);
                }
            }
        }
    }

    /// 一手记录（字段与线上 `_make_record` 同名同义）。
    fn make_record(&self, res: &Value) -> Value {
        let me = self.me as usize;
        let g = &self.game;
        let deltas: Vec<Value> = arr_f64(res, "deltas")
            .iter()
            .map(|&x| json!(round2(x)))
            .collect();
        let mut o = Map::new();
        o.insert("time".into(), json!(crate::records::now_str()));
        o.insert(
            "rid".into(),
            json!(format!("{}-{}", crate::records::now_ms(), self.hand_index)),
        );
        o.insert("hand".into(), json!(self.hand_index));
        o.insert("my_seat".into(), json!(me));
        o.insert("names".into(), json!(self.names));
        o.insert("dealer".into(), json!(self.dealer));
        o.insert(
            "type".into(),
            json!(if res.get("type").and_then(|x| x.as_str()) == Some("win") {
                "win"
            } else {
                "huang"
            }),
        );
        o.insert("deltas".into(), json!(deltas));
        // 2026-11-07：第一巡同轮跟打
        for k in ["follow_dealer", "follow_tile"] {
            o.insert(k.into(), res.get(k).cloned().unwrap_or(Value::Null));
        }
        o.insert(
            "follow_pay".into(),
            json!(res.get("follow_pay").and_then(|x| x.as_f64()).unwrap_or(0.0)),
        );

        // 终局听牌：座位 + 名字 + 各自等待张 + 非平胡牌型
        let tp: Vec<i64> = arr_i64(res, "tenpai");
        o.insert("tenpai".into(), json!(tp));
        o.insert(
            "tenpai_names".into(),
            json!(tp
                .iter()
                .map(|&p| self.names.get(p as usize).cloned().unwrap_or_default())
                .collect::<Vec<_>>()),
        );
        let winners: Vec<i64> = {
            let w = arr_i64(res, "winners");
            if w.is_empty() {
                res.get("winner").and_then(|x| x.as_i64()).into_iter().collect()
            } else {
                w
            }
        };
        let win_tile = res.get("win_tile").and_then(|x| x.as_i64());
        let drop_for = |p: i64| -> Option<u8> {
            match win_tile {
                Some(t) if winners.contains(&p) && (0..27).contains(&t) => Some(t as u8),
                _ => None,
            }
        };
        let mut waits = Map::new();
        let mut types = Map::new();
        for &p in &tp {
            if !(0..4).contains(&p) {
                continue;
            }
            let (w, tps) = waits_of(g, p as usize, drop_for(p), &self.ft);
            if !w.is_empty() {
                waits.insert(p.to_string(), json!(w));
            }
            if !tps.is_empty() {
                types.insert(p.to_string(), json!(tps));
            }
        }
        o.insert("tenpai_waits".into(), Value::Object(waits));
        o.insert("tenpai_types".into(), Value::Object(types.clone()));

        let labels: Vec<String> = res
            .get("detail")
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
            .unwrap_or_default();
        let has = |k: &str| labels.iter().any(|x| x == k);
        let is_win = res.get("type").and_then(|x| x.as_str()) == Some("win");
        let win_bigfan = res
            .get("winners_fan")
            .and_then(|x| x.as_array())
            .map(|a| {
                a.iter().any(|x| {
                    let fc = x.get("fan_cn").and_then(|c| c.as_str()).unwrap_or("");
                    !matches!(fc, "" | "平胡" | "报叫")
                })
            })
            .unwrap_or(false);
        o.insert(
            "flags".into(),
            json!({
                "chongfeng": has("冲锋鸡"),
                "ji_discard": has("冲锋鸡") || has("横鸡") || has("幺鸡"),
                "baoji": has("包鸡"),
                "tenpai_bigfan": !types.is_empty(),
                "win_bigfan": win_bigfan,
                "huang": !is_win,
                "huang_noten": !is_win && tp.len() < 4,
                "hint_used": self.hint_used,
            }),
        );

        // detail：与线上导出同形 [名字, 标签, 说明, 数值]
        let detail_export: Vec<Value> = res
            .get("detail")
            .and_then(|x| x.as_array())
            .map(|a| {
                a.iter()
                    .map(|d| {
                        let arr = d.as_array();
                        let seat = arr.and_then(|x| x.first()).and_then(|x| x.as_i64()).unwrap_or(-1);
                        let who = if (0..4).contains(&seat) {
                            self.names[seat as usize].clone()
                        } else {
                            "全场".to_string()
                        };
                        json!([
                            who,
                            arr.and_then(|x| x.get(1)).and_then(|x| x.as_str()).unwrap_or(""),
                            arr.and_then(|x| x.get(2)).and_then(|x| x.as_str()).unwrap_or(""),
                            arr.and_then(|x| x.get(3)).and_then(|x| x.as_f64()).unwrap_or(0.0)
                        ])
                    })
                    .collect()
            })
            .unwrap_or_default();
        o.insert("detail".into(), json!(detail_export));

        if is_win {
            let w = res.get("winner").and_then(|x| x.as_i64()).unwrap_or(0);
            let loser = res.get("loser").and_then(|x| x.as_i64());
            let void = res.get("void_player").and_then(|x| x.as_i64());
            o.insert("winner".into(), json!(w));
            o.insert(
                "winner_name".into(),
                json!(self.names.get(w as usize).cloned().unwrap_or_default()),
            );
            o.insert("winners".into(), json!(winners));
            o.insert(
                "winner_names".into(),
                json!(winners
                    .iter()
                    .map(|&x| self.names.get(x as usize).cloned().unwrap_or_default())
                    .collect::<Vec<_>>()),
            );
            o.insert("loser".into(), json!(loser));
            o.insert(
                "loser_name".into(),
                json!(loser.map(|x| self.names.get(x as usize).cloned().unwrap_or_default())),
            );
            o.insert(
                "void_name".into(),
                json!(void.map(|x| self.names.get(x as usize).cloned().unwrap_or_default())),
            );
            o.insert("void_seat".into(), json!(void));
            for k in ["is_tsumo", "rob_kong"] {
                o.insert(
                    k.into(),
                    json!(res.get(k).and_then(|x| x.as_bool()).unwrap_or(false)),
                );
            }
            for k in ["how", "fan_cn", "fan"] {
                o.insert(k.into(), res.get(k).cloned().unwrap_or(Value::Null));
            }
            let mut hand = Vec::new();
            if (0..4).contains(&w) {
                for t in 0..27u8 {
                    for _ in 0..g.hands[w as usize][t as usize] {
                        hand.push(t);
                    }
                }
            }
            o.insert("winner_hand".into(), json!(hand));
        }
        o.insert("ji_tiles".into(), json!((0..27u8)
            .filter(|&t| g.ji_tiles & (1u32 << t) != 0)
            .collect::<Vec<_>>()));
        o.insert("fanji".into(), {
            let d = res.get("detail").and_then(|x| x.as_array());
            let f = d.and_then(|a| {
                a.iter().find(|v| {
                    v.as_array().and_then(|x| x.get(1)).and_then(|x| x.as_str()) == Some("翻鸡")
                })
            });
            f.and_then(|v| v.as_array().and_then(|x| x.get(2)).cloned())
                .unwrap_or(Value::Null)
        });
        Value::Object(o)
    }

    /// 回放（事件流 + 发牌快照 + 终局快照），字段与线上 `_make_replay` 对齐。
    fn make_replay(&self, res: &Value, fin: Value) -> Value {
        let mut evs: Vec<Value> = Vec::with_capacity(self.events.len());
        for ev in &self.events {
            // 「本局结算」事件是纯提示，回放里不需要（终局快照已经能画结算页）
            if ev.get("kind").and_then(|x| x.as_str()) == Some("result") {
                continue;
            }
            evs.push(json!([
                ev.get("seq").and_then(|x| x.as_u64()).unwrap_or(0),
                ev.get("kind").and_then(|x| x.as_str()).unwrap_or(""),
                ev.get("rel").and_then(|x| x.as_i64()).unwrap_or(0),
                ev.get("tile").and_then(|x| x.as_i64()).unwrap_or(-1),
                ev.get("ctag").and_then(|x| x.as_str()).unwrap_or(""),
                ev.get("r").and_then(|x| x.as_i64()).unwrap_or(1),
                ev.get("hc").cloned().unwrap_or(json!([13, 13, 13, 13])),
                ev.get("text").and_then(|x| x.as_str()).unwrap_or(""),
                ev.get("hands").cloned().unwrap_or(json!(["", "", "", ""])),
            ]));
        }
        json!({
            "hand": self.hand_index,
            "my_seat": self.me,
            "names": self.names,
            "dealer": self.dealer,
            "type": res.get("type").cloned().unwrap_or(Value::Null),
            "deal": self.deal_packed,
            "ji_tiles": (0..27u8)
                .filter(|&t| self.game.ji_tiles & (1u32 << t) != 0)
                .collect::<Vec<_>>(),
            "ev": evs,
            "final": fin,
        })
    }

    /// 终局快照：与 `/api/state` 同口径，剥掉只有实时对局才需要的大块字段
    /// （事件流 / 记牌器 / 统计 / pending 等），否则 replays.jsonl 会翻倍。
    fn final_snapshot(&self) -> Value {
        let mut snap = view::snapshot(&SnapshotCtx {
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
            pending: None,
            result: self.result.clone(),
            error: None,
            stats: &self.stats,
            opp_think: &self.opp_think,
        });
        if let Some(o) = snap.as_object_mut() {
            for k in [
                "events", "counter", "stats", "hand", "last_draw", "pending", "my_ting",
                "error", "sid",
            ] {
                o.remove(k);
            }
        }
        snap
    }

    fn persist_hand(&mut self, res: &Value) {
        let fin = self.final_snapshot();
        let rec = self.make_record(res);
        let rp = self.make_replay(res, fin);
        crate::records::append(&rec, Some(rp));
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
        self.deal_packed = [
            pack_hand(&self.game.hands[0]),
            pack_hand(&self.game.hands[1]),
            pack_hand(&self.game.hands[2]),
            pack_hand(&self.game.hands[3]),
        ];
        self.events.clear();
        self.seq = 0;
        self.pending = None;
        self.result = None;
        self.settled = false;
        self.force_end = false;
        self.hint_used = false;
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
        // `stats` 内部按**绝对座位**累加（跨手稳定），但 `players[]` 是相对座位序，
        // 统计面板「本会话四家」也是按 rel 取的 —— 这里转一次，避免 my_seat≠0 时
        // 四家的数字整表错位。
        let mut stats_rel = Map::new();
        for rel in 0..4usize {
            let seat = (self.me as usize + rel) % 4;
            stats_rel.insert(
                rel.to_string(),
                self.stats.get(seat.to_string()).cloned().unwrap_or(Value::Null),
            );
        }
        let stats_rel = Value::Object(stats_rel);
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
            stats: &stats_rel,
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
        // 看过推荐就记一笔：「本人胡牌 / 本人累计」两项统计只算全程没看推荐的局
        self.hint_used = true;
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
        // 2026-11-12 用户口径：桌面版/单机版的推荐要与网页版一致 —— 每个候选都带
        // 完整教学理由（tags 彩色标签 + lines 多行讲解 + reason 摘要），再加面板
        // 顶部的 situation 与底部的一句话 tip。生成器见 `crate::coach`（移植自
        // web/coach.py + zhuoji/reason.py），只用公开信息，不读牌墙与别家暗手。
        let me = self.me as usize;
        let items: Vec<Value> = top
            .iter()
            .enumerate()
            .map(|(i, (_q, a))| {
                let ex = coach::explain_action_full(&self.game, me, a, &self.ft);
                json!({
                    "label": a.label(),
                    "kind": actions::KIND_NAMES[a.kind as usize],
                    "tile": if a.tile == 255 { -1 } else { a.tile as i64 },
                    "prob": round3(exps[i] / z),
                    "reason": ex.reason,
                    "tags": ex.tags,
                    "lines": ex.lines,
                })
            })
            .collect();
        let situation = coach::situation_summary(&self.game, me, &self.ft);
        let tip = coach::one_line(&items, &self.game, me, &self.ft);
        let model_name = if is_ting { "ting:fast" } else { stem };

        json!({
            "ok": true,
            "model": model_name,
            "head": if engine == "tract-pimc" { kind } else { "model" },
            "value": round2(top.first().map(|x| x.0).unwrap_or(0.0)),
            "items": items,
            "situation": situation,
            "tip": tip,
            "note": "",
            "elapsed_ms": t0.elapsed().as_millis() as u64,
            "src": {"stem": if is_ting { "ting" } else { stem },
                    "engine": engine, "enc": enc, "enc_expected": enc},
            "src_ok": true,
        })
    }

    /// `/api/ting`：局中任意时刻都能判定听牌（2026-11-12 用户反馈）。
    pub fn ting(&mut self, mode_now: bool) -> Value {
        let _ = mode_now;   // 两种模式共用同一份载荷（见 ting_payload 注释）
        ting_payload(&mut self.game, self.me as usize, &self.ft)
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
                    "type": mt, "label": meld_label(mt), "tile": t, "src": s,
                    "ji": view::meld_ji_tag(g, p, t)
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

// ---------------------------------------------------------------------------
// 听牌载荷（2026-11-12）
// ---------------------------------------------------------------------------
/// `/api/ting` 的统一载荷。抽成自由函数，便于用构造手牌做确定性单测。
///
/// 旧实现只调 `Game::winning_tiles()`，而它要求手牌正好是「听牌张数」
/// `(4-副露数)*3+1`（13/10/7/4/1）；而玩家真正的出牌决策点是 **14 张**，
/// 所以按钮在自己回合永远答「未听牌」，只有等别人出牌的 13 张状态才碰巧有效。
///
/// 现在按张数分两路，**两个字段始终同时下发**：
///   * `cur`  —— `n == expect`（13 张等，等别人出牌 / 别人回合）→ 直接给等待张，
///     空表示未听牌。右下角听牌按钮读这个。
///   * `map`  —— `n == expect + 1`（14 张等，自己的出牌决策点）→ 逐张试打，
///     给「打这张后听什么」，与线上 Python `ting_hint()` 同口径。
///     选牌上浮提示读这个。
/// 两者都不改对局状态（试打后立刻把手牌还原）。
pub fn ting_payload(g: &mut Game, me: usize, ft: &FanTable) -> Value {
    let n = tiles::counts_sum(&g.hands[me]) as i32;
    let expect = (4 - g.melds[me].len() as i32) * 3 + 1;
    // 2026-11-12 用户反馈：「听牌计算器没扣我自己手上的牌」——
    // 例：2345 筒听 2/5 筒，手里各有一张，应显示各剩 3 张，旧实现硬编码 4。
    // 剩余张数 = 4 − 已见，已见 = 四家弃牌 + 四家副露（碰 3 / 杠 4）+ 我的暗手
    // （`drop` 那张若已打掉则少算一张）。与 Python `remaining_unseen` 同式同口径。
    let mut pub_used = [0i32; 27];
    for p in 0..4 {
        for &t in g.discards[p].iter() {
            if (t as usize) < 27 {
                pub_used[t as usize] += 1;
            }
        }
        for m in g.melds[p].iter() {
            if (m.1 as usize) < 27 {
                pub_used[m.1 as usize] += if dushan_engine::fan::meld_is_kong(m.0) { 4 } else { 3 };
            }
        }
    }
    let hand_now = g.hands[me];
    let left_of = |w: u8, drop: Option<u8>| -> i64 {
        let mut seen = pub_used[w as usize] + hand_now[w as usize] as i32;
        if drop == Some(w) {
            seen -= 1;
        }
        (4 - seen).max(0) as i64
    };
    // 听的是不是「值钱牌型」：与推荐理由同源（平胡不列，只在没有其他牌型时兜底）
    let types_of = |waits: &[u8], drop: Option<u8>| -> Vec<String> {
        let mut names: Vec<String> = Vec::new();
        for &w in waits {
            let mut h = hand_now;
            if let Some(d) = drop {
                // 这一张已经被打出去，不在手上了
                h[d as usize] = h[d as usize].saturating_sub(1);
            }
            h[w as usize] = h[w as usize].saturating_add(1);
            for tp in dushan_engine::fan::win_type_set(&h, &g.melds[me], Some(w), ft) {
                if tp == dushan_engine::fan::PING_HU {
                    continue;
                }
                let cn = match tp {
                    dushan_engine::fan::DA_DUI_ZI => "大对子",
                    dushan_engine::fan::QI_DUI => "小七对",
                    dushan_engine::fan::LONG_QI_DUI => "龙七对",
                    dushan_engine::fan::QING_YI_SE => "清一色",
                    dushan_engine::fan::DAN_DIAO => "单吊",
                    _ => tp,
                }
                .to_string();
                if !names.contains(&cn) {
                    names.push(cn);
                }
            }
        }
        names
    };
    let pack = |waits: &[u8], drop: Option<u8>| -> Value {
        json!({
            "tiles": waits.iter()
                .map(|&t| json!({"t": t, "left": left_of(t, drop)}))
                .collect::<Vec<_>>(),
            "types": types_of(waits, drop),
        })
    };
    let mut cur = Value::Null;
    if n == expect {
        let waits = g.winning_tiles(me, ft);
        if !waits.is_empty() {
            cur = pack(&waits, None);
        }
    }
    let mut map = Map::new();
    if n == expect + 1 {
        for t in 0..27u8 {
            if g.hands[me][t as usize] == 0 {
                continue;
            }
            g.hands[me][t as usize] -= 1;
            let waits = g.winning_tiles(me, ft);
            g.hands[me][t as usize] += 1;
            if !waits.is_empty() {
                map.insert(t.to_string(), pack(&waits, Some(t)));
            }
        }
    }
    json!({"ok": true, "cur": cur, "map": map})
}

#[cfg(test)]
mod tests {
    use super::*;

    fn hand_of(tiles: &[u8]) -> dushan_engine::state::Game {
        let mut g = Game::blank();
        for &t in tiles {
            g.hands[0][t as usize] += 1;
        }
        g
    }

    /// 门清 13 张听牌：111 222 333 44 55 → 至少听 4 / 5（索引 3、4）。
    /// 这正是「13 张手牌也应该能判定听牌」的场景。
    /// 注：引擎对这类「对子待补」牌型的可胡张比直觉更宽（还会接受 1/2/3/6），
    /// 这是结算同源逻辑，测试只校验「确实判出听牌且包含 4/5」。
    #[test]
    fn ting_thirteen_tiles_reports_waits() {
        let ft = FanTable::build();
        let mut g = hand_of(&[0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 4, 4]);
        let v = ting_payload(&mut g, 0, &ft);
        assert!(!v["cur"].is_null(), "13 张听牌必须给出 cur");
        let tiles: Vec<u8> = v["cur"]["tiles"].as_array().unwrap()
            .iter().map(|x| x["t"].as_u64().unwrap() as u8).collect();
        assert!(!tiles.is_empty());
        assert!(tiles.contains(&3) && tiles.contains(&4), "应包含 4/5 两门: {tiles:?}");
        assert!(v["map"].as_object().unwrap().is_empty(), "13 张不该有逐张候选");
    }

    /// 同一手牌多摸一张（14 张）→ 打出多余的那张仍在听，`map` 要给出候选。
    #[test]
    fn ting_fourteen_tiles_reports_discard_map() {
        let ft = FanTable::build();
        let mut g = hand_of(&[0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 4, 4, 5]);
        let v = ting_payload(&mut g, 0, &ft);
        assert!(v["cur"].is_null(), "14 张不是等待态，cur 应为空");
        let mp = v["map"].as_object().unwrap();
        let hit = mp.get("5").expect("打出 6 之后应当听牌");
        let tiles: Vec<u8> = hit["tiles"].as_array().unwrap()
            .iter().map(|x| x["t"].as_u64().unwrap() as u8).collect();
        assert!(tiles.contains(&3) && tiles.contains(&4), "应包含 4/5 两门: {tiles:?}");
    }

    /// 散牌 13 张未听 → 两个字段都应为空（不能误报「听」）。
    #[test]
    fn ting_not_ready_reports_empty() {
        let ft = FanTable::build();
        let mut g = hand_of(&[0, 3, 6, 9, 12, 15, 18, 21, 24, 1, 5, 10, 20]);
        let v = ting_payload(&mut g, 0, &ft);
        assert!(v["cur"].is_null());
        assert!(v["map"].as_object().unwrap().is_empty());
    }

    /// 2026-11-12 用户反馈：「听牌计算器没扣我手上的牌」。
    /// 111 222 333 万 + 44 条 + 55 筒 → 听 4 条 / 5 筒，两门手里各有 2 张 ⇒ 各剩 2。
    #[test]
    fn ting_left_subtracts_own_hand() {
        let ft = FanTable::build();
        let mut g = hand_of(&[0, 0, 0, 1, 1, 1, 2, 2, 2, 12, 12, 22, 22]);
        let v = ting_payload(&mut g, 0, &ft);
        let arr = v["cur"]["tiles"].as_array().expect("13 张听牌要有 cur");
        let left = |t: u64| arr.iter()
            .find(|x| x["t"].as_u64() == Some(t))
            .map(|x| x["left"].as_u64().unwrap());
        assert_eq!(left(12), Some(2), "4 条手里 2 张 → 剩 2：{arr:?}");
        assert_eq!(left(22), Some(2), "5 筒手里 2 张 → 剩 2：{arr:?}");
    }

    /// 牌河/副露也要扣：牌河已各现 1 张 → 4 条/5 筒各剩 1；再被碰走 3 张则归零。
    #[test]
    fn ting_left_subtracts_discards_and_melds() {
        let ft = FanTable::build();
        let mut g = hand_of(&[0, 0, 0, 1, 1, 1, 2, 2, 2, 12, 12, 22, 22]);
        g.discards[1].push(12);
        g.discards[2].push(22);
        g.melds[3].push((dushan_engine::fan::MELD_PONG, 12, 0));
        let v = ting_payload(&mut g, 0, &ft);
        let arr = v["cur"]["tiles"].as_array().unwrap();
        let left = |t: u64| arr.iter()
            .find(|x| x["t"].as_u64() == Some(t))
            .map(|x| x["left"].as_u64().unwrap());
        assert_eq!(left(12), Some(0), "手里 2 + 牌河 1 + 副露 3 = 6 已见（封顶 0）：{arr:?}");
        assert_eq!(left(22), Some(1), "手里 2 + 牌河 1 = 3 已见 → 剩 1：{arr:?}");
    }
}
