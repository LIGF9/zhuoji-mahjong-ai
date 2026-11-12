# 独山麻将 · 桌面版（Tauri 2）

> 双击即玩：规则、结算、AI 推理、前端页面与模型全部编译进应用，**不需要 Python、
> 不需要模型目录、不需要网络**；同时它还是一个局域网服务端 —— 同一 Wi-Fi 下的手机
> 打开窗口标题里的地址就能玩移动版。

## 它是什么

| | 手机版 `tauri-offline/` | 桌面版 `desktop/` |
|---|---|---|
| 内核 | `offline-core`（Rust） | 同左，同一份代码 |
| 监听 | `127.0.0.1:8791`（仅本机） | **`0.0.0.0:8771`**（本机 + 局域网） |
| 首屏 | 双模式选择页（在线/离线） | 直接进宽屏对局页 |
| 产物 | `.apk`（arm64-v8a） | `.app`（macOS，可双击） |

## 构建

```bash
cd desktop
tauri build --bundles app
# → src-tauri/target/release/bundle/macos/独山麻将.app
```

需要 Rust 1.77+ 与 Tauri 2 CLI（本机全局安装）。macOS 打包不需要 JDK。

## 运行

双击 `独山麻将.app`。窗口标题会变成：

```
独山麻将 · 手机访问 http://192.168.x.x:8771/?touch=1
```

手机与电脑连同一 Wi-Fi，浏览器打开这个地址即可（横屏体验最佳）。

## 只想跑服务端（不开窗口）

```bash
cargo build --release --manifest-path ../offline/Cargo.toml --bin offline_server
../offline/target/release/offline_server 8771 0.0.0.0
```

启动日志会打印本机与局域网两个地址。这个二进制**单文件可拷贝**，无任何运行时依赖。

## 排查

- `DUSHAN_HTTP_LOG=1` 启动：打印 `[http] GET /api/state` 请求流水。
- 端口被占用时窗口会显示「本地服务未就绪」兜底页（`web/index.html`），
  不会无声闪退。
