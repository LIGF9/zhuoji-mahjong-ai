# -*- coding: utf-8 -*-
"""下载 onnxruntime-web 1.20.1 运行时到 web/standalone/lib/（单 HTML 打包前置）。

lib/ 不入 git（26MB 二进制，可由此脚本一键恢复）。
注意 ort.min.js 是**非 jsep** 构建，运行时动态 import 配套 .mjs 胶水并加载
非 jsep 的 ort-wasm-simd-threaded.wasm——四个文件必须同版本配套。
"""
import os
import urllib.request

VER = "1.20.1"
BASE = f"https://cdn.jsdelivr.net/npm/onnxruntime-web@{VER}/dist"
LIB = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "web", "standalone", "lib")
FILES = [
    "ort.min.js",
    "ort-wasm-simd-threaded.mjs",
    "ort-wasm-simd-threaded.wasm",
]

def main():
    os.makedirs(LIB, exist_ok=True)
    for fn in FILES:
        dst = os.path.join(LIB, fn)
        if os.path.exists(dst) and os.path.getsize(dst) > 0:
            print(f"[skip] {fn} 已存在")
            continue
        url = f"{BASE}/{fn}"
        print(f"[get ] {url}")
        urllib.request.urlretrieve(url, dst)
        print(f"[ ok ] {fn} {os.path.getsize(dst) / 2**20:.1f} MiB")

if __name__ == "__main__":
    main()
