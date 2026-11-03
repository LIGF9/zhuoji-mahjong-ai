#!/usr/bin/env bash
# 构建 Rust 引擎的两个产物，顺序不能颠倒：
#   1) replay 二进制（**不带** python feature，否则会链上 pyo3 的
#      extension-module，运行时报 `_PyBaseObject_Type` symbol not found）
#   2) libdushan_engine.dylib（带 python feature）→ 复制成 ../dushan_engine.so
#
# 坑：`cargo build --release --features python` 会顺带重编所有 bin（含 replay）
#     并且因为 .cargo/config.toml 里的 dynamic_lookup，链接不报错、运行时才炸。
#     所以必须用 `--lib` 限定目标，且 replay 放在前面编。
set -euo pipefail
cd "$(dirname "$0")"

export PATH="$HOME/.cargo/bin:$PATH"

cargo build --release --bin replay
cargo build --release --features python --lib
# 部署必须用 mv（原子换目录项），不能用 cp —— cp 原地截断会让**正在映射旧 .so
# 的进程**（过夜训练 / 千局对战）SIGBUS 崩溃；mv 换 inode，旧映射继续有效。
mv -f target/release/libdushan_engine.dylib ../dushan_engine.so
echo "OK  replay=$(ls -l target/release/replay | awk '{print $5}')B  so=$(ls -l ../dushan_engine.so | awk '{print $5}')B"
