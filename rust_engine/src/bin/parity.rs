//! 对拍入口：读行式用例，输出与 Python 侧同样格式的判定结果。
//!
//! 输入行格式（`|` 分隔，两侧空白可有可无）：
//! ```text
//! <27 个计数>|<kind:tile;kind:tile>|<-1 或 0..26>
//! ```
//! 输出行格式：
//! ```text
//! <win 0/1>|<types 逗号分隔>|<qing 0/1>|<seven 0/1>|<da_dui 0/1>
//! ```

use std::io::{self, BufRead, Write};

use dushan_engine::fan::{self, FanTable, Meld};
use dushan_engine::tiles::Counts;

fn parse_counts(s: &str) -> Result<Counts, String> {
    let mut c = [0u8; 27];
    let items: Vec<&str> = s.split_whitespace().collect();
    if items.len() != 27 {
        return Err(format!("需要 27 个计数，实际 {}", items.len()));
    }
    for (i, it) in items.iter().enumerate() {
        c[i] = it.parse::<u8>().map_err(|e| format!("计数解析失败 {it}: {e}"))?;
    }
    Ok(c)
}

fn parse_melds(s: &str) -> Result<Vec<Meld>, String> {
    let s = s.trim();
    if s.is_empty() {
        return Ok(Vec::new());
    }
    let mut out = Vec::new();
    for part in s.split(';') {
        let part = part.trim();
        if part.is_empty() {
            continue;
        }
        let mut it = part.split(':');
        let k = it.next().ok_or("缺 meld kind")?;
        let t = it.next().ok_or("缺 meld tile")?;
        out.push((
            k.trim().parse::<u8>().map_err(|e| format!("meld kind: {e}"))?,
            t.trim().parse::<u8>().map_err(|e| format!("meld tile: {e}"))?,
            0,
        ));
    }
    Ok(out)
}

fn main() {
    let table = FanTable::build();
    let stdin = io::stdin();
    let mut out = io::BufWriter::new(io::stdout().lock());

    for (lineno, line) in stdin.lock().lines().enumerate() {
        let line = match line {
            Ok(l) => l,
            Err(e) => {
                eprintln!("读第 {} 行失败: {e}", lineno + 1);
                std::process::exit(2);
            }
        };
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let parts: Vec<&str> = line.split('|').collect();
        if parts.len() != 3 {
            eprintln!("第 {} 行字段数 {} != 3: {line}", lineno + 1, parts.len());
            std::process::exit(2);
        }
        let counts = match parse_counts(parts[0]) {
            Ok(v) => v,
            Err(e) => {
                eprintln!("第 {} 行: {e}", lineno + 1);
                std::process::exit(2);
            }
        };
        let melds = match parse_melds(parts[1]) {
            Ok(v) => v,
            Err(e) => {
                eprintln!("第 {} 行: {e}", lineno + 1);
                std::process::exit(2);
            }
        };
        let wt = parts[2].trim().parse::<i32>().unwrap_or(-1);
        let win_tile = if (0..27).contains(&wt) { Some(wt as u8) } else { None };

        let win = fan::is_winning_hand(&counts, &melds, &table) as u8;
        let types = fan::win_type_set(&counts, &melds, win_tile, &table);
        let qing = fan::is_qing_yi_se(&counts, &melds) as u8;
        let seven = fan::is_seven_pairs(&counts) as u8;
        let da_dui = (win == 1 && fan::all_triplets(&counts, &melds, &table)) as u8;

        writeln!(
            out,
            "{}|{}|{}|{}|{}",
            win,
            types.join(","),
            qing,
            seven,
            da_dui
        )
        .unwrap();
    }
    out.flush().unwrap();
}
