# 独山麻将 AI

贵州独山麻将的本地对局与 AI 教练：网页版、Android 离线版、macOS / Windows 桌面版。
规则、结算与 AI 推理全部在本机完成，不需要联网、不需要账号。

## 下载（推荐）

到 [GitHub Releases](https://github.com/LIGF9/zhuoji-mahjong-ai/releases/latest) 直接下载：

| 平台 | 文件 | 说明 |
| --- | --- | --- |
| Android | `dushan-mahjong-android-arm64.apk` | 断网可玩，规则 / 模型 / 素材全部编进 APK |
| macOS | `dushan-mahjong-macos-arm64.zip` | Apple Silicon；解压后把 `独山麻将.app` 拖进「应用程序」 |
| Windows | `dushan-mahjong-windows-x64-setup.exe` | NSIS 安装包，首次运行会引导安装 WebView2 Runtime |

> 桌面版的形态是「电脑上跑引擎，窗口当屏幕」：启动后在本机 **8771** 端口起服务，
> 同一 Wi-Fi 下手机浏览器打开 `http://<电脑IP>:8771/?touch=1` 也能连上来当远程屏幕。
> 所有连到这台电脑的设备看到的是同一局（刻意设计，不是多人独立对局）。
> Windows 首次启动若弹防火墙提示，允许「专用网络」即可。
>
> Release 资产用 ASCII 文件名（GitHub 会把非 ASCII 资产名折叠成 `-`），
> 说明表格与 App 内部标题仍是中文。
>
> Release 里的包**没有代码签名**：macOS 首次打开若提示「无法验证开发者」，
> 右键 →「打开」，或执行 `xattr -dr com.apple.quarantine /Applications/独山麻将.app`；
> Windows SmartScreen 提示时点「更多信息 → 仍要运行」。

## 从源码跑网页版

```bash
git clone https://github.com/LIGF9/zhuoji-mahjong-ai.git
cd zhuoji-mahjong-ai
python3 -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python web/dushan_server.py --port 8770
```

浏览器打开 `http://127.0.0.1:8770`。常用参数：`--host 0.0.0.0`（局域网可访问）、
`--threads 8`、`--verbose`。

Rust 加速内核是可选的：先 `cd rust_engine && ./build.sh`，
生成的 `dushan_engine.so` 不入库；不编译也能玩，服务端会自动回落到纯 Python 搜索，
只是同档位下思考慢一些。

## 从源码构建各端

| 端 | 命令 | 产物 |
| --- | --- | --- |
| Android（需 Android SDK/NDK + JDK 17） | `cd tauri-offline && npm run android:build` | `gen/android/app/build/outputs/apk/universal/release/app-universal-release.apk` |
| macOS 桌面 | `cd desktop && tauri build --bundles app` | `src-tauri/target/release/bundle/macos/独山麻将.app` |
| Windows 桌面 | 在 Windows 上 `cd desktop && tauri build --config src-tauri/tauri.windows.conf.json` | `bundle/nsis/*.exe` |
| 本机一键出包（macOS） | `tools/pack_all.sh all` | `tauri-offline/dist/` 与 `desktop/dist/`，文件名带打包时间 |

三端共用同一份前端页面 `web/static/dushan.html`；离线端把它和 3 个 ONNX 模型
在**编译期**嵌进二进制，所以改完前端必须重编对应端，光换磁盘文件不会生效。
四档难度（全部贪心、无搜索）：🐣 入门·速听 / 🙂 进阶·小网 / 🤖 大师·主力 / 👑 传奇。

## 发布新版本

```bash
git tag v1.2.0 && git push origin v1.2.0
```

推送 `v*` 标签会触发 `.github/workflows/release.yml`：Windows / macOS / Android
三个 runner 各自原生构建，最后自动创建 Release 并上传三端安装包。
全程不需要本机 GitHub Token，也可以在 Actions 页面手动 `workflow_dispatch` 触发。

## 目录结构

```
web/             网页版服务端 + 前端（dushan_server.py、static/）
zhuoji/          Python 规则引擎、编码器、网络、搜索
rust_engine/     Rust 搜索内核源码（可选加速）
offline/         离线版内核：规则 + tract 推理 + 极简 HTTP（编译期嵌资源）
tauri-offline/   离线版 Android 外壳
desktop/         macOS / Windows 桌面外壳
models/          AI 权重；onnx/ 下 3 个文件是离线端构建必需
tools/           训练、评测、出包脚本
docs/ reports/   方案、历次修复与评测报告
```

## 规则与文档

- **玩法规则**：任一版本「设置 → 玩法规则」（最后一个板块），只讲本玩法特有规则。
- 实现方案与历次修复：`docs/离线版-可行性与实施方案.md`
- 四档难度强度：`reports/离线版-四档强度矩阵.md`
- 仓库分层说明：`docs/仓库结构-2026-11-12.md`

## 常见问题

- **端口被占用**：网页版换 `--port 8771`；桌面版固定 8771，先关掉已在跑的实例。
- **AI 思考慢**：把 AI 档位调低；或在 Apple Silicon 上装 `onnxruntime` 后以
  `DUSHAN_SEARCH_DEVICE=ort` 启动服务端。
- **想清空历史对局**：删除 `web/history/`，服务端会自动重建。
