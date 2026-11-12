#!/bin/bash
# 独山麻将窗口模拟器启动器（双击即可）
#
# 为什么不用 `emulator ... &`：把模拟器挂在 AI 会话的后台 shell 上，shell 在
# 会话空闲/结束被系统回收时会连带把 emulator 一起 SIGTERM，窗口就突然消失
# （日志末尾 "Wait for emulator ... to shutdown gracefully before kill"，
#  退出码 137 或 0）。本脚本把自己重新投到一个独立的 AppleScript applet
# 进程里运行，父进程可以立刻退出，模拟器不再依附任何会话 shell。
set -eu

LOG="$HOME/zhuoji-mahjong-ai/runs/emu_window_applet.log"
mkdir -p "$(dirname "$LOG")"

if [ "${1:-}" = "--child" ]; then
  export ANDROID_HOME="$HOME/Library/Android/sdk"
  export ANDROID_SDK_ROOT="$ANDROID_HOME"
  export PATH="$ANDROID_HOME/platform-tools:$PATH"
  exec "$ANDROID_HOME/emulator/emulator" \
    -avd zhuoji_test \
    -no-snapshot \
    -no-audio \
    -no-boot-anim \
    -gpu swiftshader_indirect \
    -memory 1536 \
    -cores 4 \
    > "$LOG" 2>&1
fi

HERE="$(cd "$(dirname "$0")" && pwd)"
APP_DIR="$(mktemp -d /tmp/StartZhuojiEmu.XXXXXX)"
osacompile -o "$APP_DIR/StartZhuojiEmu.app" \
  -e 'with timeout of 31536000 seconds' \
  -e "do shell script \"bash '$HERE/start_emulator_window.command' --child\"" \
  -e 'end timeout' > /dev/null
open "$APP_DIR/StartZhuojiEmu.app"
echo "已在独立进程中启动 zhuoji_test 窗口模拟器，日志：$LOG"
