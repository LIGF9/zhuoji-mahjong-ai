#!/usr/bin/env bash
# 一键出包（安卓 APK + 桌面 .app）。
#
# 为什么要先清理（2026-11-12 两次踩坑，见 docs/离线版-可行性与实施方案.md §14.7）：
#   1) Gradle 的 packageUniversalRelease 会把「源码没变」判成 UP-TO-DATE，
#      直接复用 outputs/apk 里的旧包 —— 时间戳不动、内容还是上一版；
#   2) cargo 的 .fingerprint 对 include_dir!/include_bytes! 的目录变化不敏感，
#      改完 web/static/dushan.html 不删指纹可能不重编，.so 里仍是旧页面。
# 所以每次出包都强制清：Gradle 的 outputs/intermediates、三组 .fingerprint、
# 旧的 .so，再让 cargo 重编。
#
# 产物文件名带**打包时间**（包名里能看到是哪一次打的）：
#   tauri-offline/dist/独山麻将-v<版本>-<YYYYMMDD-HHMM>-arm64.apk
#   desktop/dist/独山麻将-v<版本>-<YYYYMMDD-HHMM>.app
#
# 用法：
#   tools/pack_all.sh            # APK + 桌面
#   tools/pack_all.sh android    # 只出 APK
#   tools/pack_all.sh desktop    # 只出桌面
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
TS="$(date +%Y%m%d-%H%M)"
TARGET="${1:-all}"

ver_of() {  # $1 = tauri.conf.json 路径
  python3 - "$1" <<'PY'
import json, sys
print(json.load(open(sys.argv[1], encoding="utf-8"))["version"])
PY
}

APK_VER="$(ver_of "$ROOT/tauri-offline/src-tauri/tauri.conf.json")"
DESK_VER="$(ver_of "$ROOT/desktop/src-tauri/tauri.conf.json")"
echo "== 打包时间 $TS | APK v$APK_VER | 桌面 v$DESK_VER"

# ---------------------------------------------------------------- 安卓 APK
if [[ "$TARGET" == "all" || "$TARGET" == "android" ]]; then
  AND="$ROOT/tauri-offline/src-tauri"
  echo "-- 清理安卓旧产物"
  rm -rf "$AND/gen/android/app/build/outputs/apk"
  rm -rf "$AND/gen/android/app/build/intermediates/merged_native_libs"
  rm -rf "$AND/gen/android/app/build/intermediates/stripped_native_libs"
  # 只删这三组的指纹：全删 .fingerprint 会让整棵依赖树（几百个 crate）重编
  rm -rf "$AND"/target/aarch64-linux-android/release/.fingerprint/tauri-app-* \
         "$AND"/target/aarch64-linux-android/release/.fingerprint/offline-core-* \
         "$AND"/target/aarch64-linux-android/release/.fingerprint/dushan-engine-*
  rm -f "$AND/target/aarch64-linux-android/release/libtauri_app_lib.so" \
        "$AND/target/aarch64-linux-android/release/deps/libdushan_engine"*.so
  # 旧的同名产物也清掉，避免把上一版误当成这次的结果
  mkdir -p "$ROOT/tauri-offline/dist"

  echo "-- 构建 APK（tauri android build --apk --target aarch64）"
  if [[ -d "$HOME/jdks/jdk-21.0.2.jdk/Contents/Home" ]]; then
    export JAVA_HOME="$HOME/jdks/jdk-21.0.2.jdk/Contents/Home"
  fi
  export PATH="$HOME/.cargo/bin:$PATH"
  # tauri CLI 由 WorkBuddy 的 node 环境提供，版本目录会变 → 按需探测
  if ! command -v tauri >/dev/null 2>&1; then
    for d in "$HOME"/.workbuddy/binaries/node/versions/*/bin; do
      if [[ -x "$d/tauri" ]]; then export PATH="$d:$PATH"; break; fi
    done
  fi
  command -v tauri >/dev/null 2>&1 || { echo "!! 找不到 tauri CLI" >&2; exit 1; }
  ( cd "$ROOT/tauri-offline" && npm run android:build )

  SRC="$AND/gen/android/app/build/outputs/apk/universal/release/app-universal-release.apk"
  OUT="$ROOT/tauri-offline/dist/独山麻将-v$APK_VER-$TS-arm64.apk"
  [[ -f "$SRC" ]] || { echo "!! 没找到 APK: $SRC" >&2; exit 1; }
  cp "$SRC" "$OUT"
  # 校验：APK 里的 .so 必须比源码新，否则就是 Gradle 复用了旧包
  echo "-- APK: $OUT"
  ls -l "$OUT" | awk '{print "   size", $5, "bytes"}'
  shasum -a 256 "$OUT" | awk '{print "   sha256", $1}'
  stat -f "   mtime  %Sm" -t "%Y-%m-%d %H:%M:%S" "$OUT"
fi

# ---------------------------------------------------------------- 桌面 .app
if [[ "$TARGET" == "all" || "$TARGET" == "desktop" ]]; then
  echo "-- 构建桌面版（tauri build --bundles app）"
  export PATH="$HOME/.cargo/bin:$PATH"
  ( cd "$ROOT/desktop" && tauri build --bundles app )

  APPSRC="$ROOT/desktop/src-tauri/target/release/bundle/macos/独山麻将.app"
  mkdir -p "$ROOT/desktop/dist"
  OUTAPP="$ROOT/desktop/dist/独山麻将-v$DESK_VER-$TS.app"
  rm -rf "$OUTAPP"
  ditto "$APPSRC" "$OUTAPP"          # ditto 保留签名/扩展属性
  echo "-- APP: $OUTAPP"
  du -sh "$OUTAPP" | awk '{print "   size", $1}'
  stat -f "   mtime  %Sm" -t "%Y-%m-%d %H:%M:%S" "$OUTAPP"
fi

echo "== 完成 $TS"
