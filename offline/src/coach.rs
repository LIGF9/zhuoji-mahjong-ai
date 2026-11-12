//! 推荐理由引擎（Rust 版）：把候选动作翻成可读、可教学的中文解释。
//!
//! 2026-11-12 用户口径：「桌面版的 AI 模型推荐和网页版保持一致，保留完整推荐理由」。
//! 网页版（`web/coach.py` + `zhuoji/reason.py`）本来就有三层理由：每个候选的
//! `lines`（多行教学）、`tags`（彩色小标签）、`reason`（摘要），面板顶部的
//! `situation`（向听/听牌/持鸡/报叫/通行证）与底部的 `tip`（一句话推荐）。
//! 离线/桌面版此前只给「动作 + 概率 + 条形」——不是不想给，是 Rust 侧没有这套
//! 生成器。本模块就是那份生成器的移植。
//!
//! 口径
//! ----
//! * **只用公开信息**：自己手牌、四家弃牌、副露、鸡种、已现张数、公开计分常量。
//!   不读牌墙、不读别家暗手 —— 与网页版 `PublicView` 隔离层同一条红线；
//! * 分值一律读 `Config`（设置里改了自动同步），不写死数字；
//! * 逐段对齐 `coach.explain_action_full` / `coach.situation_summary` /
//!   `reason.one_line`（模型档分支），文案保持一致，便于两端对照排错。

use dushan_engine::actions::{self, Action};
use dushan_engine::fan::{self, FanTable};
use dushan_engine::state::{bit, Config, Game, JI_CHONGFENG, JI_HENG};
use dushan_engine::tiles::{self, Counts, NUM_TILES};
use serde_json::{json, Value};

use crate::ting;

const REL_CN: [&str; 4] = ["自己", "下家", "对家", "上家"];

/// 番型中文名（与 `coach.py` 的 `names` / `type_names` 同一份）。
fn fan_cn(tp: &str) -> &'static str {
    match tp {
        fan::DA_DUI_ZI => "大对子",
        fan::QI_DUI => "小七对",
        fan::LONG_QI_DUI => "龙七对",
        fan::QING_YI_SE => "清一色",
        fan::DAN_DIAO => "单吊",
        fan::PING_HU => "平胡",
        _ => "其他",
    }
}

/// 番型分值（等价于 `DushanGame._hu_type_value`）。
fn fan_value(tp: &str, cfg: &Config) -> f64 {
    match tp {
        fan::DA_DUI_ZI => cfg.da_dui_zi as f64,
        fan::QI_DUI => cfg.qi_dui as f64,
        fan::LONG_QI_DUI => cfg.long_qi_dui as f64,
        fan::QING_YI_SE => cfg.qing_yi_se as f64,
        fan::DAN_DIAO => cfg.dan_diao as f64,
        _ => 0.0,
    }
}

/// Python `{x:g}` 的等价写法：整数不带小数点，小数最多两位且去掉尾零。
fn gnum(x: f64) -> String {
    if (x - x.round()).abs() < 1e-9 {
        return format!("{}", x.round() as i64);
    }
    let s = format!("{:.2}", x);
    s.trim_end_matches('0').trim_end_matches('.').to_string()
}

fn cn(t: i32) -> String {
    if (0..NUM_TILES as i32).contains(&t) {
        tiles::tile_cn(t as u8)
    } else {
        "?".to_string()
    }
}

/// 裸计数版可胡张（与 `settle::winning_tiles_raw` 同式，那边是私有）。
pub fn winning_calls(c: &Counts, melds: &[fan::Meld], ft: &FanTable) -> Vec<u8> {
    let n_melds = melds.len();
    if n_melds > 4 {
        return Vec::new();
    }
    let expect = (4 - n_melds) * 3 + 1;
    if tiles::counts_sum(c) as usize != expect {
        return Vec::new();
    }
    let mut res = Vec::new();
    for t in 0..NUM_TILES as u8 {
        if c[t as usize] >= 4 {
            continue;
        }
        let mut h = *c;
        h[t as usize] += 1;
        if fan::is_winning_hand(&h, melds, ft) {
            res.push(t);
        }
    }
    res
}

/// 牌 t 在公开信息之外还剩几张（可供摸/胡）。
fn live_count(g: &Game, _me: usize, t: u8, hand: &Counts) -> i32 {
    let mut used = 0i32;
    for p in 0..4 {
        used += g.discards[p].iter().filter(|&&x| x == t).count() as i32;
    }
    for p in 0..4 {
        for m in g.melds[p].iter() {
            if m.1 == t {
                used += if fan::meld_is_kong(m.0) { 4 } else { 3 };
            }
        }
    }
    used += hand[t as usize] as i32;
    (4 - used).max(0)
}

/// 我手中（暗手）持有的鸡牌张数。
fn hand_ji(g: &Game, me: usize) -> i32 {
    let mut n = 0i32;
    for t in 0..NUM_TILES as u8 {
        if g.ji_tiles & bit(t) != 0 {
            n += g.hands[me][t as usize] as i32;
        }
    }
    n
}

/// 这张鸡牌打出后会记成什么（与引擎同源）。
fn ji_name_of_discard(g: &Game, me: usize, t: u8) -> &'static str {
    match g.ji_name_if_discarded(me, t) {
        JI_CHONGFENG => "冲锋鸡",
        JI_HENG => "横鸡",
        _ => "幺鸡",
    }
}

/// 听牌描述：叫 X 剩 n、Y 剩 m（含牌型提示）。
fn winning_line(g: &Game, me: usize, hand13: &Counts, ft: &FanTable) -> String {
    let calls = winning_calls(hand13, &g.melds[me], ft);
    if calls.is_empty() {
        return String::new();
    }
    let mut parts: Vec<String> = Vec::new();
    for &t in calls.iter().take(4) {
        parts.push(format!("{}(剩{}张)", tiles::tile_cn(t), live_count(g, me, t, hand13)));
    }
    if calls.len() > 4 {
        parts.push("…".to_string());
    }
    let mut premium: Vec<String> = Vec::new();
    for &t in calls.iter() {
        let mut h = *hand13;
        h[t as usize] += 1;
        for tp in fan::win_type_set(&h, &g.melds[me], Some(t), ft) {
            let name = fan_cn(tp);
            let v = fan_value(tp, &g.cfg);
            if name != "平胡" && v > 0.0 {
                let s = format!("{}(+{})", name, gnum(v));
                if !premium.contains(&s) {
                    premium.push(s);
                }
            }
        }
    }
    premium.sort();
    let mut line = format!("打出后即听牌：叫 {}", parts.join("、"));
    if !premium.is_empty() {
        line.push_str(&format!("，可听 {}", premium.join("、")));
    }
    line
}

/// 未听牌时的进张分析：向听数 + 有效进张。
fn ukeire_line(g: &Game, me: usize, hand: &Counts, n_melds: usize, ft: &FanTable) -> String {
    let base = ting::shanten(hand, n_melds);
    if base < 0 {
        return "已成胡".to_string();
    }
    if base == 0 {
        return winning_line(g, me, hand, ft);
    }
    let (mut kinds, mut total) = (0i32, 0i32);
    let mut names: Vec<String> = Vec::new();
    for t in 0..NUM_TILES as u8 {
        if hand[t as usize] >= 4 {
            continue;
        }
        let mut h = *hand;
        h[t as usize] += 1;
        let better = ting::shanten(&h, n_melds) < base;
        if !better {
            continue;
        }
        let live = live_count(g, me, t, hand);
        if live > 0 {
            kinds += 1;
            total += live;
            if names.len() < 5 {
                names.push(tiles::tile_cn(t));
            }
        }
    }
    let _ = ft;
    if kinds == 0 {
        return format!("打出后向听 {base} 进，暂无有效进张（牌形僵）");
    }
    let more = if kinds > 5 { format!(" 等{kinds}种") } else { String::new() };
    format!(
        "打出后向听 {base} 进，进张 {kinds} 种共 {total} 张：{}{}",
        names.join("、"),
        more
    )
}

/// 生熟张与点炮敞口。
fn danger_line(g: &Game, me: usize, t: u8) -> String {
    let zero = [0u8; NUM_TILES];
    let vis = 4 - live_count(g, me, t, &zero);
    let cfg = &g.cfg;
    let cap = cfg.dian_pao
        + cfg
            .qing_yi_se
            .max(cfg.qi_dui)
            .max(cfg.da_dui_zi)
            .max(cfg.long_qi_dui)
            .max(cfg.dan_diao);
    if vis >= 3 {
        format!("【风险】已现 {vis}/4 张，几乎安全（点炮最多赔 {} 分）", gnum(cap as f64))
    } else if vis >= 1 {
        format!("【风险】已现 {vis}/4 张，点炮风险中等")
    } else {
        format!(
            "【风险】生张，点炮需赔方式分+牌型分（最高 {}）",
            gnum(cap as f64)
        )
    }
}

/// 手牌结构提示（对子数、清一色潜质）。
fn potential_line(hand: &Counts) -> Option<String> {
    let total: i32 = hand.iter().map(|&x| x as i32).sum();
    if total < 10 {
        return None;
    }
    let mut suit_n = [0i32; 3];
    let mut pairs = 0;
    let mut tri = 0;
    for t in 0..NUM_TILES as u8 {
        let n = hand[t as usize];
        if n == 0 {
            continue;
        }
        suit_n[tiles::suit_of(t) as usize] += n as i32;
        if n == 2 {
            pairs += 1;
        }
        if n >= 3 {
            tri += 1;
        }
    }
    let mut bits: Vec<String> = Vec::new();
    let mx = *suit_n.iter().max().unwrap_or(&0);
    if mx >= 10 && total - mx <= 3 {
        bits.push("清一色潜质（单花色已占大半）".to_string());
    }
    if tri > 0 {
        bits.push(format!("{tri} 个刻子坯"));
    }
    if pairs > 0 {
        bits.push(format!("{pairs} 个对子"));
    }
    if bits.is_empty() {
        return None;
    }
    Some(format!("【牌形】{}", bits.join("，")))
}

/// 一次解释的结构（对齐 Python `_pack` 的返回）。
pub struct Explain {
    pub tags: Vec<String>,
    pub lines: Vec<String>,
    pub reason: String,
}

fn pack(tags: Vec<String>, lines: Vec<String>) -> Explain {
    let mut uniq: Vec<String> = Vec::new();
    for t in tags {
        if !uniq.contains(&t) {
            uniq.push(t);
        }
    }
    let reason = lines
        .iter()
        .take(2)
        .map(|l| match l.split_once('】') {
            Some((_, rest)) => rest.to_string(),
            None => l.clone(),
        })
        .collect::<Vec<_>>()
        .join("；");
    Explain { tags: uniq, lines, reason }
}

/// 给某个动作生成完整解释（`coach.explain_action_full` 的移植）。
pub fn explain_action_full(g: &Game, me: usize, a: &Action, ft: &FanTable) -> Explain {
    let cfg = &g.cfg;
    let mut tags: Vec<String> = Vec::new();
    let mut lines: Vec<String> = Vec::new();
    let hand = g.hands[me];
    let n_melds = g.melds[me].len();
    let k = a.kind;

    if k == actions::HU {
        let mut h = hand;
        let mut wt: i32 = if a.tile != 255 { a.tile as i32 } else { -1 };
        if wt < 0 {
            if let Some((_, t)) = g.last_discard {
                wt = t as i32;
            }
        }
        if wt >= 0 {
            h[wt as usize] = h[wt as usize].saturating_add(1);
        }
        let types = fan::win_type_set(&h, &g.melds[me], if wt >= 0 { Some(wt as u8) } else { None }, ft);
        let tv: f64 = types.iter().map(|tp| fan_value(tp, cfg)).sum();
        let mut cn_parts: Vec<String> = Vec::new();
        let mut sorted = types.clone();
        sorted.sort_unstable();
        for tp in sorted {
            if tp == fan::PING_HU {
                continue;
            }
            cn_parts.push(fan_cn(tp).to_string());
        }
        let combo = cn_parts.join("+");
        let (how, val, payer);
        if g.current as usize == me
            && (g.phase == actions::PHASE_DISCARD || g.phase == actions::PHASE_SELF_KONG)
        {
            if g.after_gang_draw {
                how = "杠上开花";
                val = cfg.gang_kai as f64;
            } else {
                how = "自摸";
                val = cfg.zi_mo as f64;
            }
            payer = "其他每家各付".to_string();
        } else {
            let src = g.last_discard.map(|(p, _)| p as i32).unwrap_or(-1);
            if g.repao_discard {
                how = "热炮";
                val = cfg.gang_pao as f64;
            } else {
                how = "点炮";
                val = cfg.dian_pao as f64;
            }
            payer = if src >= 0 {
                format!("{}独付", REL_CN[((src - me as i32).rem_euclid(4)) as usize])
            } else {
                "点炮者独付".to_string()
            };
        }
        tags.push("得分".to_string());
        lines.push(format!("【方式】{how}：{payer} {} 分", gnum(val)));
        if !combo.is_empty() {
            lines.push(format!("【牌型】{combo}（+{}）", gnum(tv)));
        }
        lines.push(format!(
            "【鸡分】你已持有 {} 只鸡（金鸡时计分×2）",
            hand_ji(g, me)
        ));
        return pack(tags, lines);
    }

    if k == actions::ANGANG || k == actions::BUGANG || k == actions::MINGGANG {
        let pay = if k == actions::MINGGANG { "点杠者独付" } else { "其他每家各付" };
        tags.push("杠分".to_string());
        lines.push(format!(
            "【杠分】+{}（{pay}），并获点胡通行证",
            gnum(cfg.kong_chickens as f64)
        ));
        lines.push(format!(
            "【后续】杠后再摸一张，有机会杠上开花（每家 +{}）",
            gnum(cfg.gang_kai as f64)
        ));
        if k == actions::BUGANG {
            lines.push(format!(
                "【注意】补杠可能被抢杠——被抢则按抢杠赔 {}+牌型，且自己鸡分全烧",
                gnum(cfg.dian_pao as f64)
            ));
        }
        tags.push("通行证".to_string());
        return pack(tags, lines);
    }

    if k == actions::PENG {
        let t = a.tile;
        let mut h2 = hand;
        if t != 255 {
            h2[t as usize] = h2[t as usize].saturating_sub(2);
        }
        let s0 = ting::shanten(&hand, n_melds);
        let s1 = ting::shanten(&h2, n_melds + 1);
        tags.push("组牌".to_string());
        if s1 < s0 {
            lines.push(format!("【进度】碰后向听 {s0}→{s1}，显著加速"));
        } else if s1 == 0 {
            let wl = winning_line(g, me, &h2, ft);
            if wl.is_empty() {
                lines.push("【进度】碰后即听牌".to_string());
            } else {
                lines.push(format!("【进度】碰后即听牌。{wl}"));
            }
        } else {
            lines.push(format!("【进度】碰后向听 {s0}→{s1}（副露固定，灵活度下降）"));
        }
        if t != 255 && g.ji_tiles & bit(t) != 0 {
            tags.push("鸡分".to_string());
            lines.push("【鸡分】碰鸡牌：对方赔溢价（冲锋+2/横+1），且 3 只鸡面计入你手中鸡、对每家生效".to_string());
        }
        lines.push("【注意】副露会暴露牌型信息；报叫玩家不能碰杠".to_string());
        return pack(tags, lines);
    }

    if k == actions::PASS {
        let (src, t) = g.last_discard.map(|(p, t)| (p as i32, t as i32)).unwrap_or((-1, -1));
        let who = if src >= 0 {
            REL_CN[((src - me as i32).rem_euclid(4)) as usize]
        } else {
            ""
        };
        tags.push("放弃".to_string());
        lines.push(format!(
            "【放弃】不响应 {who} 的 {}，保持暗手不暴露",
            cn(t)
        ));
        return pack(tags, lines);
    }

    if k != actions::DISCARD {
        return pack(tags, lines);
    }

    // ---- 打牌（核心教学场景） ----
    let t = a.tile;
    let mut h2 = hand;
    h2[t as usize] = h2[t as usize].saturating_sub(1);
    let mut wl = winning_line(g, me, &h2, ft);
    if wl.is_empty() {
        wl = ukeire_line(g, me, &h2, n_melds, ft);
    }
    let is_tenpai = ting::shanten(&h2, n_melds) == 0;
    tags.push(if is_tenpai { "听牌" } else { "进张" }.to_string());
    lines.push(format!("【进度】{wl}"));

    if g.ji_tiles & bit(t) != 0 {
        tags.push("鸡分".to_string());
        match ji_name_of_discard(g, me, t) {
            "冲锋鸡" => lines.push(format!(
                "【鸡分】首张打鸡＝冲锋鸡：终局对其他每家 +{}；若被碰/杠走，那张牌归对方（面 3/4 张），你另赔溢价 {}",
                gnum(cfg.ji_chongfeng as f64),
                gnum((cfg.ji_chongfeng - cfg.ji_discard_plain) as f64)
            )),
            "横鸡" => lines.push(format!(
                "【鸡分】打鸡＝横鸡：终局对其他每家 +{}；若被碰/杠走，那张牌归对方，你另赔溢价 {}",
                gnum(cfg.ji_heng as f64),
                gnum((cfg.ji_heng - cfg.ji_discard_plain) as f64)
            )),
            _ => {
                let left = hand_ji(g, me) - 1;
                if left > 0 {
                    lines.push(format!(
                        "【鸡分】普通鸡打出仍算 1 只（归你）；若被人碰/杠走则转移给对方。手中仍持 {left} 只鸡"
                    ));
                } else {
                    lines.push("【鸡分】普通鸡打出仍算 1 只（归你）；若被人碰/杠走则转移给对方。打完手中无鸡".to_string());
                }
            }
        }
    } else {
        let keep = hand_ji(g, me);
        if keep > 0 {
            lines.push(format!("【鸡分】保留手中 {keep} 只鸡（每只终局对三家各 +{}）", gnum(cfg.ji_hold as f64)));
        } else {
            tags.push("鸡分".to_string());
            let mut kinds: Vec<u8> = (0..NUM_TILES as u8)
                .filter(|&x| g.ji_tiles & bit(x) != 0)
                .collect();
            kinds.sort_unstable();
            let names: Vec<String> = kinds.iter().map(|&x| tiles::tile_cn(x)).collect();
            lines.push(format!("【鸡分】手中无鸡，注意鸡牌种：{}", names.join("、")));
        }
    }

    // 未听牌时打冲锋/横鸡会被包鸡
    if g.ji_tiles & bit(t) != 0 && winning_calls(&h2, &g.melds[me], ft).is_empty() {
        let name = ji_name_of_discard(g, me, t);
        if name == "冲锋鸡" || name == "横鸡" {
            lines.push(format!("【警告】你未听牌，打出的{name}终局要包鸡（赔其他每家）"));
        }
    }

    lines.push(danger_line(g, me, t));
    if let Some(pot) = potential_line(&h2) {
        lines.push(pot);
    }
    pack(tags, lines)
}

/// 局面速览（`coach.situation_summary` 的移植）。
pub fn situation_summary(g: &Game, me: usize, ft: &FanTable) -> Value {
    let mut hand = g.hands[me];
    let n_melds = g.melds[me].len();
    let sum: u32 = tiles::counts_sum(&hand);
    let (s, calls): (i32, Vec<u8>) = if sum % 3 == 2 {
        let mut best = 8;
        let mut best_t: i32 = -1;
        for t in 0..NUM_TILES as u8 {
            if hand[t as usize] == 0 {
                continue;
            }
            hand[t as usize] -= 1;
            let v = ting::shanten(&hand, n_melds);
            if v < best {
                best = v;
                best_t = t as i32;
            }
            hand[t as usize] += 1;
        }
        if best <= 0 && best_t >= 0 {
            hand[best_t as usize] -= 1;
            let c = winning_calls(&hand, &g.melds[me], ft);
            hand[best_t as usize] += 1;
            (best, c)
        } else {
            (best, Vec::new())
        }
    } else {
        (
            ting::shanten(&hand, n_melds),
            winning_calls(&hand, &g.melds[me], ft),
        )
    };

    let ji_n = hand_ji(g, me);
    let has_kong = g.melds[me].iter().any(|m| fan::meld_is_kong(m.0));
    let passport = !calls.is_empty() || has_kong || g.baojiao[me];
    let mut premium_pass = false;
    for &t in calls.iter() {
        let mut h = hand;
        h[t as usize] += 1;
        let types = fan::win_type_set(&h, &g.melds[me], Some(t), ft);
        if types.iter().any(|tp| *tp != fan::PING_HU) {
            premium_pass = true;
            break;
        }
    }

    let calls_cn: Vec<String> = calls.iter().take(6).map(|&t| tiles::tile_cn(t)).collect();
    let mut ji_kinds: Vec<u8> = (0..NUM_TILES as u8)
        .filter(|&x| g.ji_tiles & bit(x) != 0)
        .collect();
    ji_kinds.sort_unstable();
    let ji_cn: Vec<String> = ji_kinds.iter().map(|&x| tiles::tile_cn(x)).collect();

    let note = if !calls.is_empty() {
        if ji_n > 0 {
            "已听牌，可安心打鸡收鸡分"
        } else {
            "已听牌"
        }
    } else if g.baojiao[me] {
        "报叫状态：已有点胡通行证，但不能再碰杠"
    } else if has_kong {
        "有杠＝通行证，可打冲锋鸡不惧点胡限制"
    } else if s <= 1 {
        return json!({
            "shanten": s, "tenpai": false, "calls": calls_cn, "ji_held": ji_n,
            "ji_tiles": ji_cn, "baojiao": g.baojiao[me], "has_kong": has_kong,
            "passport": passport, "passport_by_premium": premium_pass,
            "note": format!("向听 {s} 进，先组牌；无通行证时避免乱打鸡被责任鸡"),
        });
    } else {
        return json!({
            "shanten": s, "tenpai": false, "calls": calls_cn, "ji_held": ji_n,
            "ji_tiles": ji_cn, "baojiao": g.baojiao[me], "has_kong": has_kong,
            "passport": passport, "passport_by_premium": premium_pass,
            "note": format!("向听 {s} 进，距离听牌尚远，优先保留搭子与鸡牌"),
        });
    };

    json!({
        "shanten": s, "tenpai": !calls.is_empty(), "calls": calls_cn, "ji_held": ji_n,
        "ji_tiles": ji_cn, "baojiao": g.baojiao[me], "has_kong": has_kong,
        "passport": passport, "passport_by_premium": premium_pass, "note": note,
    })
}

// ---------------------------------------------------------------------------
// 一句话推荐（`reason.one_line` 的模型档分支）
// ---------------------------------------------------------------------------

/// 有效进张（种数, 张数）——只统计公开可见的剩余张数。
fn ukeire(g: &Game, me: usize, hand: &Counts, n_melds: usize) -> (i32, i32) {
    let base = ting::shanten(hand, n_melds);
    let (mut kinds, mut total) = (0i32, 0i32);
    for t in 0..NUM_TILES as u8 {
        if hand[t as usize] >= 4 {
            continue;
        }
        let mut h = *hand;
        h[t as usize] += 1;
        if ting::shanten(&h, n_melds) >= base {
            continue;
        }
        let live = live_count(g, me, t, hand);
        if live > 0 {
            kinds += 1;
            total += live;
        }
    }
    (kinds, total)
}

/// 打出 ta 与打出 tb 的公开指标差异 → (优势句, 劣势句)。
fn diff(g: &Game, me: usize, ta: u8, tb: u8, ft: &FanTable) -> (Vec<String>, Vec<String>) {
    let mut pros: Vec<String> = Vec::new();
    let mut cons: Vec<String> = Vec::new();
    let hand = g.hands[me];
    let n_melds = g.melds[me].len();
    let mut ha = hand;
    ha[ta as usize] = ha[ta as usize].saturating_sub(1);
    let mut hb = hand;
    hb[tb as usize] = hb[tb as usize].saturating_sub(1);

    let ca = tiles::tile_cn(ta);
    let cb = tiles::tile_cn(tb);
    let sa = ting::shanten(&ha, n_melds);
    let sb = ting::shanten(&hb, n_melds);
    let tings_a = winning_calls(&ha, &g.melds[me], ft);
    let tings_b = winning_calls(&hb, &g.melds[me], ft);
    let va = 4 - live_count(g, me, ta, &[0u8; NUM_TILES]);
    let vb = 4 - live_count(g, me, tb, &[0u8; NUM_TILES]);

    if !tings_a.is_empty() && tings_b.is_empty() {
        pros.push(format!(
            "打{ca}即听牌（叫 {}），打{cb}仍是 {sb} 进",
            tings_a.iter().map(|&t| tiles::tile_cn(t)).collect::<Vec<_>>().join("、")
        ));
    } else if !tings_b.is_empty() && tings_a.is_empty() {
        cons.push(format!("打{cb}才听牌，打{ca}要退回 {sa} 进"));
    } else if sa < sb {
        pros.push(format!("向听更近（打{ca} {sa} 进 vs 打{cb} {sb} 进）"));
    } else if sa > sb {
        cons.push(format!("向听更远（打{ca} {sa} 进 vs 打{cb} {sb} 进）"));
    }

    // 有效进张：同向听、都未听时才可比（与 reason.py::_diff 同序同措辞）
    if sa == sb && tings_a.is_empty() && tings_b.is_empty() {
        let (ka, na) = ukeire(g, me, &ha, n_melds);
        let (kb, nb) = ukeire(g, me, &hb, n_melds);
        if na > nb {
            pros.push(format!("有效进张更多（{na} 张/{ka} 种 vs 打{cb} {nb} 张/{kb} 种）"));
        } else if na < nb {
            cons.push(format!("有效进张更少（{na} 张 vs 打{cb} {nb} 张）"));
        }
    }

    if va > vb {
        pros.push(format!("更安全（{ca}已现 {va}/4 张，{cb} {vb}/4 张）"));
    } else if vb > va {
        cons.push(format!("点炮敞口更大（{ca}已现 {va}/4 张，{cb} {vb}/4 张）"));
    }
    (pros, cons)
}

fn action_brief(g: &Game, me: usize, a: &Action, ft: &FanTable) -> String {
    let hand = g.hands[me];
    let n_melds = g.melds[me].len();
    match a.kind {
        actions::HU => "直接胡牌、立即结算得分，落袋为安".to_string(),
        actions::PASS => "过牌保持暗手不暴露、维持牌型灵活".to_string(),
        actions::PENG => {
            let mut h2 = hand;
            if a.tile != 255 && h2[a.tile as usize] >= 2 {
                h2[a.tile as usize] -= 2;
            }
            let s0 = ting::shanten(&hand, n_melds);
            let s1 = ting::shanten(&h2, n_melds + 1);
            format!("碰后向听 {s0}→{s1}，副露固定、会暴露牌型")
        }
        actions::ANGANG | actions::BUGANG | actions::MINGGANG => format!(
            "杠分 +{} 并获点胡通行证，杠后再摸一张",
            gnum(g.cfg.kong_chickens as f64)
        ),
        actions::DISCARD => {
            let mut h2 = hand;
            h2[a.tile as usize] = h2[a.tile as usize].saturating_sub(1);
            let wl = winning_line(g, me, &h2, ft);
            if !wl.is_empty() {
                return wl;
            }
            ukeire_line(g, me, &h2, n_melds, ft)
        }
        _ => String::new(),
    }
}

/// 一句话推荐：围绕「为什么是它（对比次选）」，不编造分数。
///
/// 离线四档都是贪心（没有 Q 差值），所以这里的 `gap` 恒为空 —— 与网页版
/// `model:` 策略一致：只说名次与公开牌面差异，不说「期望高 X 分」。
pub fn one_line(items: &[Value], g: &Game, me: usize, ft: &FanTable) -> String {
    if items.len() < 2 {
        return String::new();
    }
    let top = &items[0];
    let second = &items[1];
    // 前端（以及 Python 服务端）的 kind 是动作名字符串（"discard"/"peng"/…）。
    let ka = top.get("kind").and_then(|v| v.as_str()).unwrap_or("");
    let kb = second.get("kind").and_then(|v| v.as_str()).unwrap_or("");
    let la = top.get("label").and_then(|v| v.as_str()).unwrap_or("");
    let lb = second.get("label").and_then(|v| v.as_str()).unwrap_or("");

    if ka == actions::KIND_NAMES[actions::DISCARD as usize]
        && kb == actions::KIND_NAMES[actions::DISCARD as usize]
    {
        let ta = top.get("tile").and_then(|v| v.as_i64()).unwrap_or(-1);
        let tb = second.get("tile").and_then(|v| v.as_i64()).unwrap_or(-1);
        if ta < 0 || tb < 0 {
            return String::new();
        }
        let (pros, cons) = diff(g, me, ta as u8, tb as u8, ft);
        let mut bits: Vec<String> = Vec::new();
        if !pros.is_empty() {
            bits.push(pros.iter().take(2).cloned().collect::<Vec<_>>().join("；"));
            if !cons.is_empty() {
                bits.push(format!("代价是{}", cons[0]));
            }
        } else if !cons.is_empty() {
            bits.push(format!(
                "牌面不占优：{}，但仍在推荐排序中领先",
                cons[0]
            ));
        } else {
            bits.push(format!("公开牌面与{lb}接近，按推荐指数排序"));
        }
        let ja = g.ji_tiles & bit(ta as u8) != 0;
        let jb = g.ji_tiles & bit(tb as u8) != 0;
        if ja != jb {
            let ji = if ja { ta as u8 } else { tb as u8 };
            let nm = ji_name_of_discard(g, me, ji);
            bits.push(format!("{}是{nm}（鸡分归属不同）", tiles::tile_cn(ji)));
        }
        return format!("建议{la}比{lb}更优：{}。", bits.join("；"));
    }

    if ka == actions::KIND_NAMES[actions::HU as usize] {
        return format!("建议{la}：能胡就胡、落袋为安，别为做大牌错过。");
    }
    if ka == actions::KIND_NAMES[actions::PENG as usize]
        && kb == actions::KIND_NAMES[actions::PASS as usize]
    {
        return format!("建议{la}：碰后向听加速，代价是副露固定、暴露牌型。");
    }
    if ka == actions::KIND_NAMES[actions::PASS as usize]
        && kb == actions::KIND_NAMES[actions::PENG as usize]
    {
        return format!("建议{la}：不碰保持暗手灵活，不暴露牌型。");
    }
    let tile = top.get("tile").and_then(|v| v.as_i64()).unwrap_or(-1);
    let kind_id = (0..7u8)
        .find(|&k| actions::KIND_NAMES[k as usize] == ka)
        .unwrap_or(actions::PASS);
    let pseudo = Action::new(kind_id, if tile >= 0 { tile as u8 } else { 255 });
    let ba = action_brief(g, me, &pseudo, ft);
    let ba = if ba.is_empty() { la.to_string() } else { ba };
    format!("建议{la}：{ba}。")
}
