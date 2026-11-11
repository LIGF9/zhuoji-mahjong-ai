# 独山麻将 AI

一个能陪你打贵州独山麻将的本地 AI：网页对局 + AI 教练提示。
全部在本机运行，无需联网、无需注册。

## 环境要求

- Python **3.10+**（开发环境为 3.13）
- macOS / Linux / Windows 均可
- Apple Silicon 或独立显卡会自动启用加速；纯 CPU 也能玩

## 安装

```bash
git clone https://github.com/LIGF9/zhuoji-mahjong-ai.git
cd zhuoji-mahjong-ai

python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## 启动

```bash
python web/dushan_server.py --port 8770
```

然后浏览器打开 **http://127.0.0.1:8770** 即可开始对局。

常用参数：

| 参数 | 说明 |
|---|---|
| `--port 8770` | 监听端口（默认 8770） |
| `--host 0.0.0.0` | 监听地址；只想本机访问可改 `127.0.0.1` |
| `--threads 4` | 服务端线程数 |
| `--verbose` | 打印请求日志 |

## 关于 Rust 加速内核

仓库已附带 macOS（Apple Silicon）预编译的 `dushan_engine.so`，开箱即用。

其他平台（Intel Mac / Linux / Windows）需要自己编译一次：

```bash
cd rust_engine
./build.sh
cd ..
```

产物会被复制为仓库根目录的 `dushan_engine.so`。需要 Rust 工具链（https://rustup.rs）。
即使不编译也能玩：服务端在 Rust 内核不可用时会自动回落到纯 Python 搜索，只是同档位下 AI 思考慢一些。

## 目录结构

```
web/                 联机版服务端与前端（dushan_server.py、coach.py、static/）
zhuoji/              Python 规则引擎、编码器、神经网络、搜索与 AI 智能体
rust_engine/         Rust 搜索内核源码（编译后可大幅提速）
offline/             离线版内核（Rust）：规则 + tract 推理 + 极简 HTTP
tauri-offline/       离线版 Android 外壳（APK，断网可玩）
tauri-app/           联机版 Android 外壳（局域网直连，v1.1.0）
dushan_engine.so     预编译内核（macOS arm64）
models/              对局 AI 权重 + 冠军注册表
tools/               训练、评测、闸门、出包等脚本
docs/                方案与协议文档（结构总览见 docs/仓库结构-2026-11-12.md）
```

> 详细的分层、数据流与「看着没人用但删不得」清单，见
> [`docs/仓库结构-2026-11-12.md`](docs/仓库结构-2026-11-12.md)。

## 离线单机版（Android）

不打服务端、不联网，模型与素材全部编进 APK。

```bash
cd tauri-offline
npm run android:build
# 产物：src-tauri/gen/android/app/build/outputs/apk/universal/release/app-universal-release.apk
```

四档难度（全部贪心、无搜索）：🐣 入门·速听 / 🙂 进阶·小网 / 🤖 大师·主力 / 👑 传奇·学生。
强度差与实测耗时见 `reports/离线版-四档强度矩阵.md`。

## 常见问题

**端口被占用**：换一个端口，例如 `--port 8771`。

**AI 思考慢**：把页面右上角的 AI 档位从「极算」调到「简算」；或在 Apple Silicon 上安装
`onnxruntime` 后以 `DUSHAN_SEARCH_DEVICE=ort` 启动服务端。

**想清空历史对局**：删除 `web/history/` 目录即可，服务端会自动重建。
