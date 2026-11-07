//! 牌局状态机 —— 与 `zhuoji/dushan.py` 的 `DushanGame` 逐字段对齐。
//!
//! 只覆盖**影响决策**的部分（合法动作、状态推进、听牌/胡牌判定）。
//! 终局分差结算暂不实现，只记录 `Outcome`，由上层（Python 或后续 Rust 模块）处理。
//!
//! 设计要点
//! --------
//! * 计数向量用定长数组 `[u8; 27]`，整个 `Game` 是 `Copy + Clone` 的（除 Vec 字段），
//!   克隆成本远低于 Python 的 pickle 往返。
//! * `pass_lock` 用 27 位掩码，避免 `HashSet`。
//! * 横鸡轮用「牌种 → 座位掩码」两张定长表表示，与 Python 的 dict 等价。

use crate::actions::*;
use crate::fan::{self, FanTable, Meld, MELD_KONG_ADDED, MELD_KONG_CONCEALED,
                 MELD_KONG_EXPOSED, MELD_PONG};
use crate::tiles::{full_deck, Counts, DECK_SIZE, NUM_TILES, YAOJI};

/// 弃牌区鸡类型
pub const JI_PLAIN: u8 = 0;
pub const JI_CHONGFENG: u8 = 1;
pub const JI_HENG: u8 = 2;

/// 一局的最终结果（只记归属，不算分）
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Outcome {
    pub winners: [u8; 3],
    pub n_winners: u8,
    pub win_tile: u8,
    pub is_tsumo: bool,
    pub loser: u8,     // 255 = 自摸
    pub rob_kong: bool,
}

#[derive(Clone, Copy, Debug)]
pub struct Config {
    pub enable_pass_lock: bool,
    pub enable_qiang_gang: bool,
    pub baojiao_auto: bool,
    pub ping_hu_needs_dou_for_ron: bool,
    pub ji_chongfeng: f32,
    pub ji_heng: f32,
    pub ji_discard_plain: f32,
    // ---- 结算口径（与 zhuoji/dushan.py DushanConfig 默认值一致） ----
    pub baojiao_bonus: f32,      // 报叫胡牌 +10
    pub jiesuan_fanji: u8,       // 结算翻鸡 0=off 1=up 2=down 3=both（默认 both）
    pub kaiju_fanji: u8,         // 开局翻鸡 0=off（默认）1=up 2=down 3=both
    pub mantiangji: bool,        // 满堂鸡
    pub jin_mult: f32,           // 金鸡倍率（默认 2）
    pub ji_hold: f32,            // 手中/副露鸡每张
    pub ji_fanji: f32,           // 翻鸡每张
    pub kong_chickens: f32,      // 杠分
    pub zi_mo: f32,
    pub dian_pao: f32,
    pub gang_kai: f32,
    pub gang_pao: f32,
    pub da_dui_zi: f32,
    pub qi_dui: f32,
    pub long_qi_dui: f32,
    pub dan_diao: f32,
    pub qing_yi_se: f32,
    pub end_baoji: bool,         // 胡牌局：未听牌者包鸡（默认开）
    pub end_baogang: bool,
    pub huang_baoji: bool,
    pub huang_baogang: bool,
    pub huang_baodapai: bool,    // 流局：未听牌者包大牌面（默认开）
    // ---- 2026-11-07：第一巡同轮跟打（与 DushanConfig 默认一致）----
    pub first_round_follow: bool, // 规则开关（默认开）
    pub follow_pay: f32,          // 庄家赔每家的鸡数（默认 3）
    pub follow_hold: u8,          // 被跟打的 4 张牌算庄家手中几只鸡（默认 4）
}

impl Default for Config {
    fn default() -> Self {
        Self {
            enable_pass_lock: true,
            enable_qiang_gang: true,
            baojiao_auto: true,
            ping_hu_needs_dou_for_ron: true,
            ji_chongfeng: 3.0,
            ji_heng: 2.0,
            ji_discard_plain: 1.0,
            baojiao_bonus: 10.0,
            jiesuan_fanji: 3,
            kaiju_fanji: 0,
            mantiangji: false,
            jin_mult: 2.0,
            ji_hold: 1.0,
            ji_fanji: 1.0,
            kong_chickens: 3.0,
            zi_mo: 3.0,
            dian_pao: 3.0,
            gang_kai: 5.0,
            gang_pao: 5.0,
            da_dui_zi: 5.0,
            qi_dui: 10.0,
            long_qi_dui: 20.0,
            dan_diao: 10.0,
            qing_yi_se: 10.0,
            end_baoji: true,
            end_baogang: false,
            huang_baoji: false,
            huang_baogang: false,
            huang_baodapai: true,
            first_round_follow: true,
            follow_pay: 3.0,
            follow_hold: 4,
        }
    }
}

/// 一局牌。
#[derive(Clone)]
pub struct Game {
    pub cfg: Config,
    pub hands: [Counts; 4],
    pub melds: [Vec<Meld>; 4],       // (kind, tile, src)
    pub discards: [Vec<u8>; 4],
    pub wall: Vec<u8>,
    pub wall_pos: usize,

    pub dealer: u8,
    pub dealer_streak: u32,
    pub current: u8,
    pub phase: u8,
    pub drawn: bool,

    pub gang_discard: [bool; 4],
    pub pass_lock: [u32; 4],
    pub ever_tenpai: [bool; 4],
    pub baojiao: [bool; 4],
    pub first_discard_done: [bool; 4],

    pub last_discard: Option<(u8, u8)>,
    pub last_discard_was_gang: bool,
    pub repao_discard: bool,
    pub after_gang_draw: bool,
    pub any_discard: bool,

    pub respond_order: Vec<u8>,
    pub respond_idx: usize,
    pub respond_hu_votes: Vec<u8>,
    pub n_hu: usize,

    pub rob_kong: Option<(u8, u8)>,
    pub rob_kong_konger: u8,

    pub chongfeng_player: i8,
    pub zeren_player: i8,
    pub zeren_ponger: i8,
    pub first_yaoji_discarded: bool,

    pub ji_tiles: u32,               // 鸡牌种掩码
    pub hengji_done: u32,            // 该种横鸡已出现的掩码
    pub hengji_open: u32,            // 横向轮开放中的牌种掩码
    pub hengji_seats: [u8; 27],      // 每种鸡的轮内座位掩码
    pub ji_events: Vec<(u8, f32, u8)>,   // (打出者, 分值, 类型)
    pub ji_event_tiles: Vec<u8>,
    // ---- 结算用记录（对齐 DushanGame._init_dushan_state） ----
    pub kaiju_flip: Option<u8>,          // 开局翻出的指示牌
    // (打出者, 碰/杠者, 类型, 是否明杠, 牌)
    pub claimed_ji: Vec<(u8, u8, u8, bool, u8)>,
    // (胡牌者, 分值, 类型, 牌) —— 捉炮鸡
    pub captured_ji: Vec<(u8, f32, u8, u8)>,
    pub ji_tag_of_last: [u8; 4],         // 各家最近一次打出的鸡类型（JI_*）
    pub last_draw: u8,                   // 最近一次摸进的牌（255=无）
    // ---- 2026-11-05：v3 编码器所需的横鸡事件流（encoder_v3 ch43-46 / glob108）----
    // 语义与 DushanGame.hengji_events / hengji_starter 同源：
    // 每个「新横鸡轮开出」事件 +1（一轮收口后同种再开算第二次）。
    pub hengji_opens: [u8; 4],           // 各家本局开出横鸡轮的次数
    pub hengji_starter: i8,              // 全场第一个开出横鸡轮的玩家（-1=无）
    pub hengji_event_n: u32,             // hengji_events 总数（open+close，v3 glob128）

    // ---- 2026-11-07：第一巡同轮跟打 ----
    // 判定条件与 Python DushanGame._check_first_round_follow 严格同源
    // （全场前 4 次弃牌恰为四家首张、首张由庄家打出、四张牌种相同）。
    // 快照过桥可直接灌结果字段；重放路径由 step_discard 自行判定。
    pub follow_dealer: bool,             // 已触发（庄家获通行证 + 4 张手中鸡）
    pub follow_tile: u8,                 // 被跟打的牌种（255=无）
    pub first_discard_tiles: [u8; 4],    // 各家第一张弃牌（255=未打）
    pub first_discard_order: [u8; 4],    // 首弃牌的打出顺序（座位）
    pub first_discard_n: u8,             // 已记录的首弃牌数
    pub total_discards: u16,             // 全场弃牌次数（「同轮」判据）

    pub outcome: Option<Outcome>,
    pub steps: u32,
}

#[inline]
pub fn bit(t: u8) -> u32 {
    1u32 << (t as u32)
}

/// 翻鸡牌种掩码：模式 1=up(+1, 9回1) 2=down(-1, 1回9) 3=both；不含指示牌自身。
pub fn fanji_mask(flip: u8, mode: u8) -> u32 {
    let (s, r1) = ((flip / 9) as u32, (flip % 9 + 1) as u32); // r1 = 1..9
    let mut m = 0u32;
    if mode == 1 || mode == 3 {
        m |= bit((s * 9 + r1 % 9) as u8);        // 上鸡
    }
    if mode == 2 || mode == 3 {
        m |= bit((s * 9 + (r1 + 7) % 9) as u8);  // 下鸡（r1-2 mod 9）
    }
    m & !bit(flip)
}

impl Game {
    // ------------------------------------------------------------------
    // 构造
    // ------------------------------------------------------------------
    /// 用给定牌墙起局（对拍用：牌墙由 Python 侧生成，保证两边完全一致）。
    pub fn new_with_wall(wall: Vec<u8>, dealer: u8, cfg: Config, ft: &FanTable) -> Self {
        assert_eq!(wall.len(), DECK_SIZE);
        let mut g = Game {
            cfg,
            hands: [[0u8; NUM_TILES]; 4],
            melds: [Vec::new(), Vec::new(), Vec::new(), Vec::new()],
            discards: [Vec::new(), Vec::new(), Vec::new(), Vec::new()],
            wall,
            wall_pos: 0,
            dealer,
            dealer_streak: 0,
            current: dealer,
            phase: PHASE_OVER,
            drawn: false,
            gang_discard: [false; 4],
            pass_lock: [0; 4],
            ever_tenpai: [false; 4],
            baojiao: [false; 4],
            first_discard_done: [false; 4],
            last_discard: None,
            last_discard_was_gang: false,
            repao_discard: false,
            after_gang_draw: false,
            any_discard: false,
            respond_order: Vec::new(),
            respond_idx: 0,
            respond_hu_votes: Vec::new(),
            n_hu: 0,
            rob_kong: None,
            rob_kong_konger: 0,
            chongfeng_player: -1,
            zeren_player: -1,
            zeren_ponger: -1,
            first_yaoji_discarded: false,
            ji_tiles: bit(YAOJI),
            hengji_done: 0,
            hengji_open: 0,
            hengji_seats: [0; 27],
            ji_events: Vec::new(),
            ji_event_tiles: Vec::new(),
            kaiju_flip: None,
            claimed_ji: Vec::new(),
            captured_ji: Vec::new(),
            ji_tag_of_last: [JI_PLAIN; 4],
            last_draw: 255,
            hengji_opens: [0; 4],
            hengji_starter: -1,
            hengji_event_n: 0,
            follow_dealer: false,
            follow_tile: 255,
            first_discard_tiles: [255; 4],
            first_discard_order: [0; 4],
            first_discard_n: 0,
            total_discards: 0,
            outcome: None,
            steps: 0,
        };
        for _ in 0..13 {
            for p in 0..4usize {
                let t = g.draw_from_wall();
                g.hands[p][t as usize] += 1;
            }
        }
        let t = g.draw_from_wall();
        g.hands[dealer as usize][t as usize] += 1;
        g.drawn = true;
        g.last_draw = t;
        // 开局翻鸡：牌墙尾翻一张指示牌（只查看不移出，对齐 Python _start）
        if g.cfg.kaiju_fanji != 0 {
            let flip = g.wall[g.wall.len() - 1];
            g.kaiju_flip = Some(flip);
            g.ji_tiles |= fanji_mask(flip, g.cfg.kaiju_fanji);
        }
        g.enter_self_phase(dealer, ft);
        g
    }

    /// 用默认牌墙（未洗）起局，仅用于单元测试。
    pub fn new_unshuffled(dealer: u8, ft: &FanTable) -> Self {
        Self::new_with_wall(full_deck(), dealer, Config::default(), ft)
    }

    #[inline]
    fn draw_from_wall(&mut self) -> u8 {
        let t = self.wall[self.wall_pos];
        self.wall_pos += 1;
        t
    }

    #[inline]
    pub fn wall_left(&self) -> usize {
        self.wall.len() - self.wall_pos
    }

    #[inline]
    pub fn has_dou(&self, p: usize) -> bool {
        self.melds[p].iter().any(|m| fan::meld_is_kong(m.0))
    }

    #[inline]
    pub fn melded_count(&self, p: usize, tile: u8) -> u32 {
        let mut c = 0;
        for m in &self.melds[p] {
            if m.1 == tile {
                c += if fan::meld_is_kong(m.0) { 4 } else { 3 };
            }
        }
        c
    }

    /// 当前该行动的人（抢杠时是抢杠者）。
    #[inline]
    pub fn actor(&self) -> u8 {
        if self.phase == PHASE_RESPOND {
            return self.respond_order[self.respond_idx];
        }
        if self.phase == PHASE_SELF_KONG {
            if let Some((p, _)) = self.rob_kong {
                return p;
            }
        }
        self.current
    }

    // ------------------------------------------------------------------
    // 判定
    // ------------------------------------------------------------------
    pub fn is_tenpai(&self, p: usize, t: &FanTable) -> bool {
        !self.winning_tiles(p, t).is_empty()
    }

    pub fn winning_tiles(&self, p: usize, t: &FanTable) -> Vec<u8> {
        let n_melds = self.melds[p].len();
        if n_melds > 4 {
            return Vec::new();
        }
        let expect = (4 - n_melds) * 3 + 1;
        if crate::tiles::counts_sum(&self.hands[p]) as usize != expect {
            return Vec::new();
        }
        let mut c = self.hands[p];
        let mut res = Vec::new();
        for tile in 0..NUM_TILES as u8 {
            if c[tile as usize] >= 4 {
                continue;
            }
            c[tile as usize] += 1;
            if fan::is_winning_hand(&c, &self.melds[p], t) {
                res.push(tile);
            }
            c[tile as usize] -= 1;
        }
        res
    }

    pub fn can_tsumo(&self, p: usize, t: &FanTable) -> bool {
        fan::is_winning_hand(&self.hands[p], &self.melds[p], t)
    }

    /// 独山通行证：杠 / 报叫 / 胡牌张成非平胡牌型。
    #[inline]
    fn has_passport14(&self, p: usize, hand14: &Counts, win_tile: u8, t: &FanTable) -> bool {
        // 第一巡同轮跟打：被跟打的 4 张牌视为庄家的通行证
        if self.follow_dealer && p == self.dealer as usize {
            return true;
        }
        if self.has_dou(p) || self.baojiao[p] {
            return true;
        }
        let types = fan::win_type_set(hand14, &self.melds[p], Some(win_tile), t);
        types.iter().any(|x| *x != fan::PING_HU)
    }

    fn has_passport_loose(&self, p: usize, t: &FanTable) -> bool {
        // 第一巡同轮跟打：被跟打的 4 张牌视为庄家的通行证
        if self.follow_dealer && p == self.dealer as usize {
            return true;
        }
        if self.has_dou(p) || self.baojiao[p] {
            return true;
        }
        for tile in self.winning_tiles(p, t) {
            let mut h = self.hands[p];
            h[tile as usize] += 1;
            let types = fan::win_type_set(&h, &self.melds[p], Some(tile), t);
            if types.iter().any(|x| *x != fan::PING_HU) {
                return true;
            }
        }
        false
    }

    pub fn can_ron(&self, p: usize, tile: u8, t: &FanTable) -> bool {
        if self.cfg.enable_pass_lock
            && self.pass_lock[p] & bit(tile) != 0
            && !self.repao_discard
        {
            return false;
        }
        let mut h = self.hands[p];
        h[tile as usize] += 1;
        if !fan::is_winning_hand(&h, &self.melds[p], t) {
            return false;
        }
        if let Some((_, rt)) = self.rob_kong {
            if rt == tile {
                return true;
            }
        }
        if self.has_passport14(p, &h, tile, t) {
            return true;
        }
        if self.repao_discard {
            return true;
        }
        if let Some((disc, _)) = self.last_discard {
            if self.baojiao[disc as usize] {
                return true;
            }
        }
        false
    }

    /// 抢杠胡资格：不得过水 + 听该张。
    fn can_rob_kong(&self, p: usize, tile: u8, t: &FanTable) -> bool {
        if self.cfg.enable_pass_lock && self.pass_lock[p] & bit(tile) != 0 {
            return false;
        }
        let mut h = self.hands[p];
        h[tile as usize] += 1;
        fan::is_winning_hand(&h, &self.melds[p], t)
    }

    #[inline]
    fn can_peng(&self, p: usize, tile: u8) -> bool {
        !self.baojiao[p] && self.melds[p].len() < 4 && self.hands[p][tile as usize] >= 2
    }

    #[inline]
    fn can_minggang(&self, p: usize, tile: u8) -> bool {
        !self.baojiao[p] && self.melds[p].len() < 4 && self.hands[p][tile as usize] >= 3
    }

    pub fn self_kong_options(&self, p: usize) -> Vec<(u8, u8)> {
        let mut opts = Vec::new();
        if self.baojiao[p] {
            return opts;
        }
        for tile in 0..NUM_TILES as u8 {
            if self.hands[p][tile as usize] == 4 {
                opts.push((ANGANG, tile));
            }
        }
        let h = &self.hands[p];
        for m in &self.melds[p] {
            if m.0 == MELD_PONG && h[m.1 as usize] >= 1 {
                opts.push((BUGANG, m.1));
            }
        }
        opts
    }

    // ------------------------------------------------------------------
    // 合法动作
    // ------------------------------------------------------------------
    pub fn legal_actions(&self, t: &FanTable) -> Vec<Action> {
        let mut acts = Vec::with_capacity(34);
        match self.phase {
            PHASE_SELF_KONG => {
                if let Some((p, rt)) = self.rob_kong {
                    acts.push(Action::pass());
                    if self.can_ron(p as usize, rt, t) {
                        acts.push(Action::hu());
                    }
                    return acts;
                }
                acts.push(Action::pass());
                for (kind, tile) in self.self_kong_options(self.current as usize) {
                    acts.push(Action::new(kind, tile));
                }
                acts
            }
            PHASE_DISCARD => {
                let p = self.current as usize;
                for tile in 0..NUM_TILES as u8 {
                    if self.hands[p][tile as usize] > 0 {
                        acts.push(Action::discard(tile));
                    }
                }
                if self.drawn && self.can_tsumo(p, t) {
                    acts.push(Action::hu());
                }
                acts
            }
            PHASE_RESPOND => {
                let p = self.respond_order[self.respond_idx] as usize;
                acts.push(Action::pass());
                if let Some((_, rt)) = self.rob_kong {
                    if self.can_ron(p, rt, t) {
                        acts.push(Action::hu());
                    }
                    return acts;
                }
                let tile = self.last_discard.unwrap().1;
                if self.can_ron(p, tile, t) {
                    acts.push(Action::hu());
                }
                if self.can_minggang(p, tile) {
                    acts.push(Action::new(MINGGANG, tile));
                }
                if self.can_peng(p, tile) {
                    acts.push(Action::new(PENG, tile));
                }
                acts
            }
            _ => acts,
        }
    }

    // ------------------------------------------------------------------
    // 推进
    // ------------------------------------------------------------------
    pub fn step(&mut self, a: Action, t: &FanTable) -> bool {
        if self.phase == PHASE_OVER {
            return true;
        }
        self.steps += 1;
        match self.phase {
            PHASE_SELF_KONG => self.step_self_kong(a, t),
            PHASE_DISCARD => self.step_discard(a, t),
            PHASE_RESPOND => self.step_respond(a, t),
            _ => {}
        }
        self.phase == PHASE_OVER
    }

    fn step_self_kong(&mut self, a: Action, t: &FanTable) {
        if let Some((p, rt)) = self.rob_kong {
            let konger = self.rob_kong_konger;
            self.rob_kong = None;
            if a.kind == HU {
                self.settle_win(p, rt, false, konger, true);
            } else {
                self.do_bugang(konger as usize, rt, t);
            }
            return;
        }
        let p = self.current as usize;
        match a.kind {
            PASS => self.phase = PHASE_DISCARD,
            ANGANG => {
                let tile = a.tile;
                self.hands[p][tile as usize] -= 4;
                self.melds[p].push((MELD_KONG_CONCEALED, tile, p as u8));
                self.gang_draw(p, t);
            }
            BUGANG => {
                let tile = a.tile;
                if self.cfg.enable_qiang_gang {
                    let mut robbers: Vec<u8> = Vec::new();
                    for i in 1..4 {
                        let q = (p + i) % 4;
                        if self.can_rob_kong(q, tile, t) {
                            robbers.push(q as u8);
                        }
                    }
                    if let Some(&r) = robbers.first() {
                        self.rob_kong = Some((r, tile));
                        self.rob_kong_konger = p as u8;
                        self.phase = PHASE_SELF_KONG;
                        return;
                    }
                }
                self.do_bugang(p, tile, t);
            }
            _ => {}
        }
    }

    fn do_bugang(&mut self, p: usize, tile: u8, t: &FanTable) {
        self.hands[p][tile as usize] -= 1;
        for i in 0..self.melds[p].len() {
            let m = self.melds[p][i];
            if m.0 == MELD_PONG && m.1 == tile {
                self.melds[p][i] = (MELD_KONG_ADDED, tile, m.2);
                break;
            }
        }
        self.gang_draw(p, t);
    }

    fn gang_draw(&mut self, p: usize, t: &FanTable) {
        if self.wall_left() == 0 {
            self.settle_huangzhuang();
            return;
        }
        let tile = self.draw_from_wall();
        self.hands[p][tile as usize] += 1;
        self.current = p as u8;
        self.drawn = true;
        self.gang_discard[p] = true;
        self.pass_lock[p] = 0;
        self.after_gang_draw = true;
        self.last_draw = tile;
        self.enter_self_phase(p as u8, t);
    }

    fn step_discard(&mut self, a: Action, t: &FanTable) {
        let p = self.current as usize;
        if a.kind == HU {
            self.settle_win(p as u8, 255, true, 255, false);
            return;
        }
        let tile = a.tile;
        self.total_discards = self.total_discards.saturating_add(1);
        self.repao_discard = self.gang_discard[p];
        self.hands[p][tile as usize] -= 1;
        self.discards[p].push(tile);
        self.any_discard = true;

        self.close_hengji_rounds(p);
        if self.ji_tiles & bit(tile) != 0 {
            let tag = if !self.first_discard_done[p] {
                JI_CHONGFENG
            } else if self.hengji_done & bit(tile) != 0 {
                if self.hengji_open & bit(tile) != 0 {
                    JI_HENG
                } else {
                    JI_PLAIN
                }
            } else {
                self.hengji_done |= bit(tile);
                self.hengji_open |= bit(tile);
                self.hengji_seats[tile as usize] = 1u8 << p;
                self.hengji_opens[p] = self.hengji_opens[p].saturating_add(1);
                if self.hengji_starter < 0 {
                    self.hengji_starter = p as i8;
                }
                self.hengji_event_n += 1; // open 事件
                JI_HENG
            };
            if self.hengji_open & bit(tile) != 0 {
                self.hengji_seats[tile as usize] |= 1u8 << p;
            }
            let val = match tag {
                JI_CHONGFENG => self.cfg.ji_chongfeng,
                JI_HENG => self.cfg.ji_heng,
                _ => self.cfg.ji_discard_plain,
            };
            self.ji_events.push((p as u8, val, tag));
            self.ji_event_tiles.push(tile);
            self.ji_tag_of_last[p] = tag;
        }

        if !self.first_discard_done[p] {
            self.first_discard_done[p] = true;
            if self.first_discard_n < 4 {
                self.first_discard_tiles[p] = tile;
                self.first_discard_order[self.first_discard_n as usize] = p as u8;
                self.first_discard_n += 1;
            }
            self.check_first_round_follow();
            if self.cfg.baojiao_auto && self.is_tenpai(p, t) {
                self.baojiao[p] = true;
            }
        }

        self.update_tenpai(p, t);
        self.last_discard_was_gang = self.gang_discard[p];
        self.gang_discard[p] = false;
        self.last_discard = Some((p as u8, tile));
        self.begin_respond(p as u8, tile, t);
    }

    /// 第一巡同轮跟打：全场前 4 次弃牌恰是四家各第一张、第一张由庄家打出、
    /// 且四张牌种相同 → 庄家获通行证 + 4 张手中鸡 + 赔每家 follow_pay 只鸡。
    /// 条件与 Python ``DushanGame._check_first_round_follow`` 严格同源。
    fn check_first_round_follow(&mut self) -> bool {
        if self.follow_dealer || !self.cfg.first_round_follow {
            return false;
        }
        if self.total_discards != 4 || self.first_discard_n != 4 {
            return false;
        }
        if self.first_discard_order[0] != self.dealer {
            return false;
        }
        let mut seen = 0u8;
        for &p in &self.first_discard_order {
            seen |= 1u8 << p;
        }
        if seen != 0x0f {
            return false;
        }
        let t0 = self.first_discard_tiles[self.dealer as usize];
        if t0 >= 27 || self.first_discard_tiles.iter().any(|&x| x != t0) {
            return false;
        }
        self.follow_dealer = true;
        self.follow_tile = t0;
        true
    }

    /// 玩家 p 现在打出 tile 会记成什么鸡：0=非鸡(空) 1=冲锋鸡 2=横鸡 3=普通鸡(幺鸡)。
    /// 与 DushanGame.ji_name_if_discarded / _step_discard 判定同源（不改状态）。
    pub fn ji_name_if_discarded(&self, p: usize, tile: u8) -> u8 {
        if self.ji_tiles & bit(tile) == 0 {
            return 0;
        }
        if !self.first_discard_done[p] {
            return JI_CHONGFENG;
        }
        if self.hengji_done & bit(tile) != 0 {
            let open = self.hengji_open & bit(tile) != 0;
            let joined = self.hengji_seats[tile as usize] & (1u8 << p) != 0;
            return if open && !joined { JI_HENG } else { JI_PLAIN };
        }
        JI_HENG
    }

    fn close_hengji_rounds(&mut self, p: usize) {
        let mut to_close = 0u32;
        for sp in 0..27usize {
            if self.hengji_seats[sp] & (1u8 << p) != 0 {
                to_close |= 1u32 << sp;
            }
        }
        self.hengji_event_n += to_close.count_ones(); // close 事件（每轮一个）
        self.hengji_open &= !to_close;
    }

    fn begin_respond(&mut self, discarder: u8, tile: u8, t: &FanTable) {
        let mut hu_list: Vec<u8> = Vec::new();
        let mut claim_list: Vec<u8> = Vec::new();
        for i in 1..4 {
            let q = ((discarder as usize) + i) % 4;
            if self.can_ron(q, tile, t) {
                hu_list.push(q as u8);
            } else if self.can_peng(q, tile) || self.can_minggang(q, tile) {
                claim_list.push(q as u8);
            }
        }
        self.respond_hu_votes.clear();
        self.n_hu = hu_list.len();
        if hu_list.is_empty() && claim_list.is_empty() {
            self.next_player(discarder, t);
            return;
        }
        let mut order = hu_list;
        order.extend(claim_list);
        self.respond_order = order;
        self.respond_idx = 0;
        self.phase = PHASE_RESPOND;
    }

    fn next_player(&mut self, from_p: u8, t: &FanTable) {
        self.repao_discard = false;
        self.after_gang_draw = false;
        let nxt = ((from_p as usize) + 1) % 4;
        if self.wall_left() == 0 {
            self.settle_huangzhuang();
            return;
        }
        let tile = self.draw_from_wall();
        self.hands[nxt][tile as usize] += 1;
        self.current = nxt as u8;
        self.drawn = true;
        self.gang_discard[nxt] = false;
        self.pass_lock[nxt] = 0;
        self.last_draw = tile;
        self.enter_self_phase(nxt as u8, t);
    }

    fn advance_respond(&mut self, discarder: u8, tile: u8, t: &FanTable) {
        self.respond_idx += 1;
        if !self.respond_hu_votes.is_empty() && self.respond_idx >= self.n_hu {
            let winners = self.respond_hu_votes.clone();
            self.respond_order.clear();
            self.respond_idx = 0;
            self.respond_hu_votes.clear();
            self.n_hu = 0;
            self.settle_win_multi(&winners, tile, discarder);
            return;
        }
        if self.respond_idx >= self.respond_order.len() {
            self.respond_order.clear();
            self.respond_idx = 0;
            self.respond_hu_votes.clear();
            self.n_hu = 0;
            self.next_player(discarder, t);
        }
    }

    fn step_respond(&mut self, a: Action, t: &FanTable) {
        let p = self.respond_order[self.respond_idx] as usize;
        let (discarder, tile) = self.last_discard.unwrap_or((255, 255));

        if a.kind == HU {
            if !self.respond_hu_votes.contains(&(p as u8)) {
                self.respond_hu_votes.push(p as u8);
            }
            self.advance_respond(discarder, tile, t);
            return;
        }
        if a.kind == PASS {
            self.pass_lock[p] |= bit(tile);
            self.advance_respond(discarder, tile, t);
            return;
        }

        // 碰 / 明杠：从打出者牌河取走这张
        if let Some(d) = self.discards[discarder as usize].last() {
            if *d == tile {
                self.discards[discarder as usize].pop();
            }
        }
        match a.kind {
            PENG => {
                self.hands[p][tile as usize] -= 2;
                self.melds[p].push((MELD_PONG, tile, discarder));
                if self.ji_tiles & bit(tile) != 0 {
                    self.claimed_ji.push((
                        discarder, p as u8, self.ji_tag_of_last[discarder as usize],
                        false, tile));
                }
                if tile == YAOJI && self.zeren_player < 0 && self.chongfeng_player >= 0 {
                    self.zeren_player = discarder as i8;
                    self.zeren_ponger = p as i8;
                    self.chongfeng_player = -1;
                }
                self.current = p as u8;
                self.drawn = false;
                self.pass_lock[p] = 0;
                self.respond_order.clear();
                self.respond_idx = 0;
                self.update_tenpai(p, t);
                self.phase = PHASE_DISCARD;
            }
            MINGGANG => {
                self.hands[p][tile as usize] -= 3;
                self.melds[p].push((MELD_KONG_EXPOSED, tile, discarder));
                if self.ji_tiles & bit(tile) != 0 {
                    self.claimed_ji.push((
                        discarder, p as u8, self.ji_tag_of_last[discarder as usize],
                        true, tile));
                }
                self.current = p as u8;
                self.pass_lock[p] = 0;
                self.respond_order.clear();
                self.respond_idx = 0;
                self.gang_draw(p, t);
            }
            _ => {}
        }
        self.respond_hu_votes.clear();
        self.n_hu = 0;
        self.repao_discard = false;
    }

    fn enter_self_phase(&mut self, p: u8, t: &FanTable) {
        self.update_tenpai(p as usize, t);
        if self.rob_kong.is_some() || !self.self_kong_options(p as usize).is_empty() {
            self.phase = PHASE_SELF_KONG;
        } else {
            self.phase = PHASE_DISCARD;
        }
    }

    fn update_tenpai(&mut self, p: usize, t: &FanTable) {
        if !self.ever_tenpai[p] && self.is_tenpai(p, t) {
            self.ever_tenpai[p] = true;
        }
    }

    // ------------------------------------------------------------------
    // 终局（只记归属）
    // ------------------------------------------------------------------
    fn settle_win(&mut self, winner: u8, win_tile: u8, is_tsumo: bool, loser: u8, rob: bool) {
        let mut winners = [0u8; 3];
        winners[0] = winner;
        self.outcome = Some(Outcome {
            winners,
            n_winners: 1,
            win_tile,
            is_tsumo,
            loser,
            rob_kong: rob,
        });
        self.phase = PHASE_OVER;
    }

    fn settle_win_multi(&mut self, winners: &[u8], tile: u8, loser: u8) {
        // 捉炮鸡转移（对齐 Python _transfer_discarded_ji）：点炮张若是鸡牌，
        // 该张的鸡事件从打出者名下移除，分值按牌型全额记给各胡牌者。
        if self.ji_tiles & bit(tile) != 0 {
            if let Some(idx) = self.ji_events.iter().rposition(|ev| ev.0 == loser) {
                let (_, val, tag) = self.ji_events.remove(idx);
                if idx < self.ji_event_tiles.len() {
                    self.ji_event_tiles.remove(idx);
                }
                for &w in winners {
                    self.captured_ji.push((w, val, tag, tile));
                }
            }
        }
        let mut w = [0u8; 3];
        for (i, &x) in winners.iter().take(3).enumerate() {
            w[i] = x;
        }
        self.outcome = Some(Outcome {
            winners: w,
            n_winners: winners.len() as u8,
            win_tile: tile,
            is_tsumo: false,
            loser,
            rob_kong: false,
        });
        self.phase = PHASE_OVER;
    }

    fn settle_huangzhuang(&mut self) {
        self.outcome = Some(Outcome {
            winners: [0; 3],
            n_winners: 0,
            win_tile: 255,
            is_tsumo: false,
            loser: 255,
            rob_kong: false,
        });
        self.phase = PHASE_OVER;
    }
}

// ---------------------------------------------------------------------------
// 快照构造（对拍用）：从任意局面直接建 Game，绕过发牌与洗牌
// ---------------------------------------------------------------------------
impl Game {
    pub fn blank() -> Self {
        Game {
            cfg: Config::default(),
            hands: [[0u8; NUM_TILES]; 4],
            melds: [Vec::new(), Vec::new(), Vec::new(), Vec::new()],
            discards: [Vec::new(), Vec::new(), Vec::new(), Vec::new()],
            wall: full_deck(),
            wall_pos: 0,
            dealer: 0,
            dealer_streak: 0,
            current: 0,
            phase: PHASE_OVER,
            drawn: false,
            gang_discard: [false; 4],
            pass_lock: [0; 4],
            ever_tenpai: [false; 4],
            baojiao: [false; 4],
            first_discard_done: [false; 4],
            last_discard: None,
            last_discard_was_gang: false,
            repao_discard: false,
            after_gang_draw: false,
            any_discard: false,
            respond_order: Vec::new(),
            respond_idx: 0,
            respond_hu_votes: Vec::new(),
            n_hu: 0,
            rob_kong: None,
            rob_kong_konger: 0,
            chongfeng_player: -1,
            zeren_player: -1,
            zeren_ponger: -1,
            first_yaoji_discarded: false,
            ji_tiles: bit(YAOJI),
            hengji_done: 0,
            hengji_open: 0,
            hengji_seats: [0; 27],
            ji_events: Vec::new(),
            ji_event_tiles: Vec::new(),
            kaiju_flip: None,
            claimed_ji: Vec::new(),
            captured_ji: Vec::new(),
            ji_tag_of_last: [JI_PLAIN; 4],
            last_draw: 255,
            hengji_opens: [0; 4],
            hengji_starter: -1,
            hengji_event_n: 0,
            follow_dealer: false,
            follow_tile: 255,
            first_discard_tiles: [255; 4],
            first_discard_order: [0; 4],
            first_discard_n: 0,
            total_discards: 0,
            outcome: None,
            steps: 0,
        }
    }
}
