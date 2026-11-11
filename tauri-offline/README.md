# 独山麻将 · 离线单机版（Tauri 2 / Android）

同一个 APK 提供**两种模式**，启动页显式二选一（2026-11-12 起）：

| 模式 | 进入 | 依赖 |
|---|---|---|
| 在线模式 | `enter_online` → 局域网服务端的移动网页版（自动补 `?touch=1`） | 电脑端 `web/dushan_server.py` 已启动，手机与电脑同一 Wi-Fi |
| 离线模式 | `enter_game` → `http://127.0.0.1:8791/` 本地 Rust 服务 | 无网络；规则、结算与 AI 推理全在设备内 |

离线页面与联网版共用**同一份** `web/static/dushan.html`：模型与静态资源在编译期
由 `offline/src/assets.rs` 嵌进 `.so`，本地服务只实现 16 个 `/api/*`，不复制、
不改写任何游戏 HTML/CSS/JS。规则/结算走 `rust_engine`（与 Python 逐位对拍），
推理走纯 Rust `tract`，因此没有 ORT `.so`、没有 JNI AssetManager。

## 使用（玩家）

1. 安装 `dist/独山麻将-单机-v1.0.0-arm64.apk`（arm64 真机 / 模拟器）。
2. 打开 APP → 首屏「选择模式」。本地引擎在后台提前加载，切到离线模式通常秒进。
3. 想连电脑上的服务端：选「在线模式」→ 填启动日志里的局域网地址
   （默认 `http://172.26.68.203:8770`）→「测试连接」→「进入网页版」。
   地址会记住，IP 变了直接改。
4. 硬件返回键从游戏页回到启动页，可以再换一种模式。

## 关键实现

- `src/index.html` / `src/main.js` / `src/style.css`：模式选择 + 在线地址表单 + 离线加载进度。
  没有 Tauri IPC 时（浏览器预览）在线直接跳转、离线只给提示，便于视觉验收。
- `src-tauri/src/lib.rs`：`engine_status`（本地引擎就绪）、`enter_game`（离线）、
  `check_online`（明文 TCP 探 `GET /api/meta`，不读对局数据）、`enter_online`（在线）、
  `app_info`。地址归一化只允许 http(s)，`?touch=1` 只加一次。
- `src-tauri/gen/android/app/build.gradle.kts`：`applicationId=com.zhuoji.mahjong.offline`，
  `usesCleartextTraffic=true`（局域网是 http），横屏 + 沉浸式 + 小窗配置沿用 v1.1.0。

## 构建（开发者）

依赖：Rust（`aarch64-linux-android` target）、Android SDK（platform 36）、NDK、
JDK 17、Node（`@tauri-apps/cli`）。

```bash
cd tauri-offline
export JAVA_HOME=<jdk17>
export ANDROID_HOME=~/Library/Android/sdk
export NDK_HOME=$ANDROID_HOME/ndk/<版本>
node src-tauri/tauri android build --apk --target aarch64
```

签名配置在 `src-tauri/gen/android/keystore.properties`（不入库）。
`src-tauri/tauri` 是 Gradle rust 插件调用 CLI 的转发 shim。

## 测试

```bash
cargo test --manifest-path tauri-offline/src-tauri/Cargo.toml --lib
cargo test --manifest-path offline/Cargo.toml --lib
```

覆盖：地址归一化 / `?touch=1` 只追加一次 / 在线探测拒绝畸形地址 /
离线 `?touch=1` 幂等。游戏 API 端到端与结算对拍见
`docs/离线版-可行性与实施方案.md` §10–§11。

> 注意：**改了启动页、前端或 Rust 内核后必须重新打包 APK** 才能在设备上看到；
> 本机可先用 `cargo run --manifest-path offline/Cargo.toml --bin offline_server`
> 起 127.0.0.1:8771 做同源预演（不经过 Tauri）。
