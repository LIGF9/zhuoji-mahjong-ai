# 移植到 Mac —— AI Agent 执行手册（2026-10-29）

> **本文写给在新 Mac 上执行移植的 AI Agent，自包含、按顺序执行即可。**
> 目标：从 GitHub 克隆 `zhuoji-mahjong-ai` 仓库，在 Mac 上复现全部能力——
> ① 单 HTML 离线游戏（双击即玩）② 在线人机对局服务（端口 8770）③ （可选）训练与评测链路。
> **不携带任何离线文件**，仓库即全部；被排除的大件都是可再生成物，本文给出再生成命令。

---

## 项目背景速览（Agent 必读）

- 贵州独山/捉鸡麻将 AI：Python 规则引擎 + RL 自博弈训练 + Web 前端。
- **最强模型 = `models/_ckpts/dushan_m2_c4_it2280.pt`**（4.07M 参数，判读口径 5.434±0.244，
  2400 副配对净分差，seed 31337）。单 HTML 与在线版默认均用它。
- 其他入库权重：`dushan_master.pt`（教师/上一代大师）、`dushan_big_s11.pt`、`dushan_big_s7.pt`、`bc.pt`。
- 推理契约：`encodeV3` → tiles f32[63×27] + glob f32[130]；ONNX 输入
  tiles(B,63,3,9)/glob(B,130)，输出 discard/self_kong/respond/value；mask 0 位置取 −1e9 后 argmax；温度 0 贪心。
- 仓库根目录 = 本文所有命令的执行目录。

---

## 第 0 步：SSH key（如 Mac 尚未配置 GitHub）

仓库走 SSH（当年 443 被屏蔽）。若 `ssh -T git@github.com` 不通：

```bash
ssh-keygen -t ed25519 -C "mac-migration"          # 一路回车
cat ~/.ssh/id_ed25519.pub                          # 复制输出
# GitHub → Settings → SSH and GPG keys → New SSH key → 粘贴
ssh -T git@github.com                              # 应出现 "Hi LIGF9!"
```

## 第 1 步：克隆

```bash
git clone git@github.com:LIGF9/zhuoji-mahjong-ai.git zhuoji-mahjong
cd zhuoji-mahjong
git log --oneline -1   # 应为 0a1e0e2 或更新（"单HTML修复2: ... localStorage SecurityError shim"）
```

## 第 2 步：Python 环境（3.13）

```bash
python3 --version                      # 需 3.13.x；没有则 brew install python@3.13
python3 -m venv .venv
source .venv/bin/activate
pip install numpy>=2.0 matplotlib>=3.8 "torch==2.6.0"
```

⚠️ **与 Windows 的关键差异**：`requirements.txt` 里写的是 `torch==2.6.0+cpu`
（Windows CPU 源专用写法），**Mac 上必须去掉 `+cpu`**，直接从 PyPI 装。
不要执行 requirements.txt 文件头注释里那条 Windows 安装命令。

## 第 3 步：环境自检

```bash
python scripts/migration_bootstrap.py --check
```

看到各检查项 OK 即可。若报脚本内绝对路径问题，可执行
`python scripts/migration_bootstrap.py --fix-paths`（用当前 venv 的 python 运行）。

## 第 4 步：恢复 ORT 运行时（`web/standalone/lib/` 不入库）

```bash
python scripts/fetch_ort_runtime.py
```

下载 onnxruntime-web **1.20.1** 四件套到 `web/standalone/lib/`（约 26MB）。
**四个文件必须同版本配套**（ort.min.js 非 jsep 构建 + 非 jsep wasm + jsep wasm + .mjs 胶水），
脚本已保证；若手动下载不可混用版本。
下载走 jsdelivr CDN，若网络不通换代理重试。

## 第 5 步：导出 ONNX 模型（`models/standalone/` 不入库）

```bash
python scripts/export_onnx.py \
  --ckpt models/_ckpts/dushan_m2_c4_it2280.pt \
  --out models/standalone/dushan_m2_c4_it2280.onnx \
  --opset 17
```

- 预期输出文件 **≈15.5 MiB**；脚本自带数值校验（PyTorch vs ONNX 输出 max abs diff，
  历史 PASS 值 ≈4.6e-5，同数量级即可）。
- 该 ckpt 是 4.07M 的 m0 结构（128ch×12 blocks×512hid，无空间头无 aux），
  检查点含优化器状态，脚本会自动处理。

## 第 6 步：构建单 HTML（离线游戏成品）

```bash
python scripts/build_standalone_html.py
```

产出两个文件：
- `web/standalone/dushan_standalone.html`（开发版 ≈0.5MB，需 lib/ 在场）
- `web/standalone/dushan_standalone_single.html`（**≈44.5MB 严格单文件**，
  base64 内嵌 ort js/mjs/wasm/onnx/牌面音效，**双击即玩、完全离线**）

构建日志应显示「localStorage 访问点替换为安全 shim：87 处」——这是内嵌
webview 沙箱兼容（commit 0a1e0e2）的关键标志，少了说明改坏了源文件。

## 第 7 步：验证（按顺序，全过才算移植成功）

```bash
# 7a. 单文件全链路验证（Node ≥18，无需浏览器）：
#     解析 HTML → 内嵌 ORT 会话 → 本地引擎 API → 真模型推理打完一整局
node web/standalone/check_single_html.js        # 必须 PASS

# 7b. 起在线版
python web/dushan_server.py --port 8770
# 浏览器打开 http://127.0.0.1:8770 ，点"开始对局"能进牌局、能胡牌结算即 PASS
```

7a 可选增强：`node web/standalone/smoke_local.js`（本地引擎 API 冒烟）。
差分对拍台 `replay_harness.js` 需要语料 json（语料不入库），需要时用
`scripts/export_standalone_corpus.py` / `web/standalone/test/export_stress_corpus.py`
再生成后运行：`node web/standalone/replay_harness.js <corpus.json>`。

## 第 8 步（可选）：训练与评测链路

- 训练：`python scripts/selfplay.py --threads <物理核数> --workers <物理核数的一半> ...`
  ⚠️ 原 Windows 命令的 `--threads 24 --workers 12` 是按 i7-14700K（20 物理核）调的，
  **Mac 上按实际物理核数改**，比例不变。
- 判读（唯一可信口径）：`scripts/score_checkpoints.py` 共用牌池配对净分差，
  1200 副 CI±0.3 / 2400 副 ±0.27；**门槛 Δ≥+0.40 且 CI 下界>0；选点只用 2400 副**。
- 训练/判读脚本里的 Windows 绝对路径（venv、python.exe）需用
  `migration_bootstrap.py --fix-paths` 或手改。
- `scripts/keep_awake.py` 是 Windows 防睡眠实现，Mac 上改用系统设置或
  `caffeinate` 命令替代，**不要直接运行**。

---

## Mac 平台已知差异与坑

| 项 | 说明 |
|---|---|
| torch 安装 | 去掉 `+cpu` 后缀；Apple Silicon 自动用 MPS/优化 BLAS，CPU 推理反而更快 |
| curl 代理 | Windows 上调本地 API 需 `--noproxy '*'`；Mac 若无系统代理则不需要 |
| CDP 测试脚本 | `web/standalone/cdp_*.js` 写死了 Windows headless Edge 路径与端口 9223/9227，Mac 上需改可执行文件路径 |
| 字体 | UI 兜底字体 "Microsoft YaHei" Mac 上无，自动回退 PingFang SC，纯单 HTML 版不受影响（视觉自包含） |
| 沙箱 shim | 单 HTML 已内嵌 localStorage 安全 shim，任何浏览器/内嵌 webview 直接可用，无需再改 |
| 中文路径 | 仓库路径避免含空格/特殊字符即可，中文文件名在 macOS APFS 上无问题 |

## 红线（改代码前必读）

1. **在线版** `web/dushan_server.py` + `web/static/dushan.html` 与单 HTML 是两套并行产物，
   改单 HTML 相关文件（`web/standalone/`）不要顺手改在线版，反之亦然。
2. 逐轮训练日志 `ec= kc= lr=` 的字段顺序是判读接口，**新字段只能追加行尾**。
3. 报叫 UI 显示名是「报叫」（不改 fan_type）；`baojiao_bonus` 为 10。
4. 训练超参改动必须留痕 + A/B（决策铁律 R7）；单旋钮、同起点（R1）。
5. `models/_ckpts/` 其余历史检查点已删（−25.35GiB），manifest 在
   `reports/_ckpt_slim_manifest_20261023.csv`，勿尝试恢复不存在的文件。

## 验收清单（全部 ✓ 才算完成）

- [ ] `git log --oneline -1` ≥ `0a1e0e2`
- [ ] `python scripts/migration_bootstrap.py --check` 全 OK
- [ ] `web/standalone/lib/` 四件套齐（ort.min.js / .mjs / jsep.wasm / 非 jsep.wasm）
- [ ] `models/standalone/dushan_m2_c4_it2280.onnx` ≈15.5MB，导出校验 diff ~1e-5 量级
- [ ] `dushan_standalone_single.html` ≈44.5MB，构建日志含「87 处」shim
- [ ] `node web/standalone/check_single_html.js` PASS
- [ ] `python web/dushan_server.py --port 8770` 后浏览器对局可开局、可结算
- [ ] 单 HTML 双击可玩：开局 → 跟打 → 结算面板弹出 → 下一局正常
