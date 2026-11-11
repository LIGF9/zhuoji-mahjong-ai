// Tauri 2 mobile 入口由 lib.rs 的 `#[cfg_attr(mobile, tauri::mobile_entry_point)]` 提供。
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

fn main() {
    tauri_app_lib::run()
}
