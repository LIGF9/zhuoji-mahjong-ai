# 独山麻将 · 安卓客户端（Tauri 2）

局域网直连的全屏安卓客户端。**APP 不包含、不修改任何游戏代码**——
游戏页面、规则引擎、AI 推理全部由你电脑上已运行的 Python 服务端原样提供，
因此客户端升级或改规则都不需要重新打包 APK。

## 使用（玩家）

1. 电脑上启动服务端（仓库根目录）：
   ```bash
   .venv/bin/python web/dushan_server.py --port 8770 --host 0.0.0.0 --threads 8
   ```
   启动日志会打印 `手机/平板（连同一个 Wi-Fi）→ http://<局域网IP>:8770`。
2. 手机与电脑连**同一个 Wi-Fi**，安装 `dist/独山麻将-v1.0.0-arm64.apk`（arm64 真机）。
3. 打开 APP：默认地址已填当前电脑的局域网地址，点「测试连接」确认，
   再点「进入游戏」即全屏进入。地址修改后会记住，下次自动使用。
4. 局域网 IP 变了怎么办：打开 APP → 连接测试会失败 → 改成启动日志里的新地址 → 进入。

## 构建（开发者）

依赖：Rust（含 `aarch64-linux-android` target）、Android SDK（platform 36 + build-tools）、
NDK、JDK 17、Node（全局或本地 `@tauri-apps/cli`）。

```bash
cd tauri-app
export JAVA_HOME=<jdk17>
export ANDROID_HOME=~/Library/Android/sdk
export NDK_HOME=$ANDROID_HOME/ndk/<版本>
node src-tauri/tauri android build --apk --target aarch64
# 产物：src-tauri/gen/android/app/build/outputs/apk/universal/release/app-universal-release.apk
```

签名：`gen/android/keystore.properties`（storeFile/storePassword/keyAlias/keyPassword）
指向 `gen/android/app/zhuoji-release.jks`。两个文件均不入库；没有它们也能出包，
只是 release APK 未签名（不可直接安装）。

`src-tauri/tauri` 是 CLI 转发 shim：Tauri 的 Gradle rust 插件用 `node tauri ...`
从 `src-tauri/` 目录调用 CLI，本机 CLI 是全局安装，shim 负责转发。

## 测试

```bash
cd tauri-app/src-tauri && cargo test   # 7 项：地址归一化 + touch 参数 + 连接探测（需本机 8770 服务在跑）
```

已验证（2026-11-05）：
- Rust 后端 7/7 通过：`normalize_adds_scheme_and_trims_slash`、`normalize_rejects_bad_input`、
  `normalize_rejects_non_http_scheme`、`touch_param_is_appended_once`、
  `check_server_rejects_garbage`、`check_server_unreachable_port_fails_fast`、
  `check_server_hits_real_backend`（最后一项打真实服务端）。
- 启动页 6 种安卓横屏视口（800×360@3x … 960×540）无溢出、按钮全部可见可点。
- 游戏 API 端到端 13/13（meta/new/state/hint/act/speed/history/stats，不结束一手、历史零污染）。
- 视觉一致性：`http://127.0.0.1:8770` 与 `http://<LAN-IP>:8770` 开局静态页像素差 **0.0000%**
  （同 URL 两次加载的动画噪声基线 0.51%）。
- APK 校验：`com.zhuoji.mahjong`，minSdk 24 / targetSdk 36，INTERNET + 横屏 +
  release usesCleartextTraffic=true，arm64-v8a 原生库，V2 签名验证通过。
- **安卓模拟器实机流程**（Pixel 4 / API 34 / arm64，HVF 加速）：安装 → 启动页渲染正确 →
  自动探测 `GET /api/meta` 200 OK 433ms → 进入真实牌局（开局卡片、桌面、大师建议面板、
  推荐牌头顶 1/2/3 徽章均正常）→ 硬件返回键回启动页且地址保留。
  期间发现并修复一个真实集成 bug：Tauri Android WebView 不命中
  `(hover:none)+(pointer:coarse)`，游戏误判为窄桌面窗口而弹「窗口宽度不足」门禁；
  修复方式是 `enter_game` 导航时自动补 `?touch=1`（游戏自带口径，未改游戏一行代码，
  保存的地址保持干净）。
- 未覆盖：实体安卓真机（本机无设备），模拟器已验证同一份 APK（sha256 `1558f4aa…`）。

## 架构说明（为什么不做"离线版"）

服务端是 Python + PyTorch/ONNX + macOS 原生扩展（约 3000 行），
Rust 搜索内核（`rust_engine/`）的网络前向明确"攒批回传 Python"。
完全离线需要把整个服务端与神经网络推理移植到 Android，工作量以周计且无法在本机验证。
局域网直连方案下，游戏体验与网页版逐字节一致——这正是"视觉和逻辑零 bug"的保证方式。
