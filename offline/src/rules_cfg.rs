//! 前端 `rules` 载荷 → 引擎 `Config`。
//!
//! 与 `web/dushan_server.py::config_from_rules` **逐字段对应**：键名、默认值、
//! 容错口径（非法值回落默认、追加鸡去重/越界丢弃）都照抄，保证「同一份设置，
//! 线上与单机跑出同一套规则」。
//!
//! 为什么必须有这一层：离线版以前一律 `Config::default()`，前端设置里的
//! 冲锋鸡分值、追加鸡牌种、开局翻鸡、满堂鸡、包赔开关在单机版**全部失效**，
//! 表现为「规则和线上不同步」——尤其是冲锋鸡：追加种（如八筒）在离线里
//! 根本不算鸡，自然也不会记冲锋鸡/横鸡。

use dushan_engine::state::{bit, Config};
use dushan_engine::tiles::{NUM_TILES, YAOJI};
use serde_json::Value;

/// 翻鸡模式：0=off 1=up 2=down 3=both。对齐 `norm_fanji(v, allow_off)`。
fn norm_fanji(v: Option<&Value>, allow_off: bool) -> u8 {
    let code: u8 = match v {
        Some(Value::Bool(true)) => 3,
        Some(Value::Bool(false)) => 0,
        Some(Value::String(s)) => {
            let s = s.trim();
            match s {
                "上鸡" => 1,
                "下鸡" => 2,
                "上下鸡" => 3,
                "关闭" => 0,
                _ => match s.to_ascii_lowercase().as_str() {
                    "up" => 1,
                    "down" => 2,
                    "both" => 3,
                    _ => 0,
                },
            }
        }
        // Python: v is None → off；其它类型 str(v).lower() 不在枚举内 → off
        _ => 0,
    };
    if code == 0 && !allow_off { 3 } else { code }
}

/// 数值项：非负有限数才认，否则回默认（Python `_f` 的 `v if v >= 0 else default`）。
fn num_f(scores: Option<&Value>, key: &str, default: f32) -> f32 {
    let raw = scores.and_then(|s| s.get(key));
    let v = match raw {
        Some(Value::Number(n)) => n.as_f64(),
        Some(Value::String(s)) => s.trim().parse::<f64>().ok(),
        _ => None,
    };
    match v {
        Some(x) if x.is_finite() && x >= 0.0 => x as f32,
        _ => default,
    }
}

/// Python `bool(x)` 的 JSON 等价（"0"/非空串为 True，0/空串/null 为 False）。
fn truthy(v: Option<&Value>, default: bool) -> bool {
    match v {
        Some(Value::Bool(b)) => *b,
        Some(Value::Number(n)) => n.as_f64().map(|x| x != 0.0).unwrap_or(default),
        Some(Value::String(s)) => !s.is_empty(),
        Some(Value::Null) | None => default,
        Some(_) => true,
    }
}

/// `rules.extra_ji` → 位掩码（幺鸡恒为鸡，出现在这里忽略；越界/非整数丢弃）。
fn extra_ji_mask(rules: &Value) -> u32 {
    let mut m = 0u32;
    if let Some(arr) = rules.get("extra_ji").and_then(|v| v.as_array()) {
        for x in arr {
            let t = match x {
                Value::Number(n) => n.as_i64(),
                Value::String(s) => s.trim().parse::<i64>().ok(),
                _ => None,
            };
            if let Some(t) = t {
                if (0..NUM_TILES as i64).contains(&t) && t as u8 != YAOJI {
                    m |= bit(t as u8);
                }
            }
        }
    }
    m
}

/// 前端 `rules` 字典 → `Config`。`None` / 非对象等价于「全部默认」。
pub fn config_from_rules(rules: Option<&Value>) -> Config {
    let empty = Value::Object(serde_json::Map::new());
    let r = match rules {
        Some(v) if v.is_object() => v,
        _ => &empty,
    };
    let sc = r.get("scores");
    let mut c = Config::default();

    // ---- 翻鸡策略（结算翻鸡允许 off；开局翻鸡允许 off）----
    // 注意区分「key 缺失」（用线上默认 both/off）与「显式 null」（norm_fanji → off）：
    // Python 是 `r.get("jiesuan", "both")`，缺 key 才走默认值。
    c.jiesuan_fanji = match r.get("jiesuan").or_else(|| r.get("jiesuan_fanji")) {
        Some(v) => norm_fanji(Some(v), true),
        None => 3,
    };
    c.kaiju_fanji = match r.get("kaiju").or_else(|| r.get("kaiju_fanji")) {
        Some(v) => norm_fanji(Some(v), true),
        None => 0,
    };
    c.mantiangji = truthy(r.get("mantiangji"), false);

    // ---- 分值表（键名与前端的 SCORE_DEFAULT 一致）----
    c.zi_mo = num_f(sc, "zimo", 3.0);
    c.dian_pao = num_f(sc, "dianpao", 3.0);
    c.gang_kai = num_f(sc, "gangkai", 5.0);
    c.gang_pao = num_f(sc, "gangpao", 5.0);
    c.da_dui_zi = num_f(sc, "dadui", 5.0);
    c.qing_yi_se = num_f(sc, "qing", 10.0);
    c.qi_dui = num_f(sc, "qidui", 10.0);
    c.long_qi_dui = num_f(sc, "long", 20.0);
    c.dan_diao = num_f(sc, "dudiao", 10.0);
    c.baojiao_bonus = num_f(sc, "baojiao", 10.0);
    c.kong_chickens = num_f(sc, "kong", 3.0);
    c.jin_mult = num_f(sc, "jinji", 2.0);
    c.ji_hold = num_f(sc, "jihold", 1.0);
    c.ji_discard_plain = num_f(sc, "jiplain", 1.0);
    c.ji_chongfeng = num_f(sc, "jicf", 3.0);
    c.ji_heng = num_f(sc, "jiheng", 2.0);
    c.ji_fanji = num_f(sc, "jifanji", 1.0);

    // ---- 未听牌包赔开关（RULE_DEFAULTS）----
    c.end_baoji = truthy(r.get("end_baoji"), true);
    c.end_baogang = truthy(r.get("end_baogang"), false);
    c.huang_baoji = truthy(r.get("huang_baoji"), false);
    c.huang_baogang = truthy(r.get("huang_baogang"), false);
    c.huang_baodapai = truthy(r.get("huang_baodapai"), true);
    c.first_round_follow = truthy(r.get("first_round_follow"), true);

    // ---- 追加鸡牌种 ----
    c.extra_ji = extra_ji_mask(r);
    c
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn defaults_when_absent() {
        let c = config_from_rules(None);
        assert_eq!(c.ji_chongfeng, 3.0);
        assert_eq!(c.ji_heng, 2.0);
        assert_eq!(c.jiesuan_fanji, 3);
        assert_eq!(c.kaiju_fanji, 0);
        assert_eq!(c.extra_ji, 0);
        assert!(c.end_baoji);
    }

    #[test]
    fn reads_scores_and_extra_ji() {
        let r = json!({
            "jiesuan": "up", "kaiju": "both", "mantiangji": true,
            "scores": {"jicf": 5, "jiheng": 2.5, "jicf_typo": 99, "zimo": -1},
            "extra_ji": [7, 7, 9, 27, -1, "3"],
            "huang_baoji": true
        });
        let c = config_from_rules(Some(&r));
        assert_eq!(c.ji_chongfeng, 5.0);
        assert_eq!(c.ji_heng, 2.5);
        assert_eq!(c.zi_mo, 3.0, "负值回落默认");
        assert_eq!(c.jiesuan_fanji, 1);
        assert_eq!(c.kaiju_fanji, 3);
        assert!(c.mantiangji);
        assert!(c.huang_baoji);
        // 7、3 有效；9=幺鸡忽略；27/-1 越界丢弃
        assert_eq!(c.extra_ji, bit(7) | bit(3));
    }

    #[test]
    fn fanji_aliases() {
        assert_eq!(norm_fanji(Some(&json!(true)), false), 3);
        assert_eq!(norm_fanji(Some(&json!("上下鸡")), true), 3);
        assert_eq!(norm_fanji(Some(&json!("off")), false), 3, "结算翻鸡不允许 off");
        assert_eq!(norm_fanji(Some(&json!("off")), true), 0);
        assert_eq!(norm_fanji(Some(&Value::Null), false), 3);
        assert_eq!(config_from_rules(None).jiesuan_fanji, 3, "缺 key 走线上默认 both");
    }
}
