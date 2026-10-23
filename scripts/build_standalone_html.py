#!/usr/bin/env python3
"""独山麻将 · 离线单文件打包器。

输入（全部只读，不改任何被引用文件）：
  web/static/dushan.html                          —— 在线版前端（对它做文本级补丁）
  web/standalone/engine/{tiles,fan,rules,dushan,encoder_v3}.js
  web/standalone/local_engine_server.js           —— 本地引擎 API 层
  web/standalone/lib/ort.min.js + ort-wasm-simd-threaded.jsep.wasm
  models/standalone/dushan_m2_c4_it2280.onnx
  web/static/assets/**                            —— 牌面/桌布/头像/音效/字体

输出：
  web/standalone/dushan_standalone.html           —— 开发版（外部 lib/、模型走 fetch 相对路径）
  web/standalone/dushan_standalone_single.html    —— 严格单文件（ort/wasm/onnx/资产全内嵌）

对 dushan.html 的补丁点（在线版更新后重跑本脚本即可重新打包）：
  P1  api() → window.__LocalAPI（本地引擎直调，无 HTTP）
  P2  引擎装配脚本 + ORT 会话引导（注入 </body> 前）
  P3  隐藏「选择模型」下拉（离线只有内嵌模型 c4_it2280）
  P4  五处模板字面量资产路径（tileImg/sndPath/bgm/indSrc/avatar）改为 __asset() 查表
  P5  剩余静态 /assets/ 字面量（CSS url、<img src>、JS 字符串）直接替换
  P6  字体：ttf 分支重定向到 woff2 子集（省 9.9MB）
"""
from __future__ import annotations

import base64
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC_HTML = ROOT / "web" / "static" / "dushan.html"
ENGINE_DIR = ROOT / "web" / "standalone" / "engine"
LOCAL_SERVER = ROOT / "web" / "standalone" / "local_engine_server.js"
LIB_DIR = ROOT / "web" / "standalone" / "lib"
MODEL = ROOT / "models" / "standalone" / "dushan_m2_c4_it2280.onnx"
ASSETS = ROOT / "web" / "static" / "assets"

OUT_DEV = ROOT / "web" / "standalone" / "dushan_standalone.html"
OUT_SINGLE = ROOT / "web" / "standalone" / "dushan_standalone_single.html"

DEV_ASSET_BASE = "../static/assets/"
DEV_MODEL_URL = "../../models/standalone/dushan_m2_c4_it2280.onnx"

# 引擎模块依赖序（tiles → fan → rules → dushan → encoder_v3 → local_engine_server）
ENGINE_FILES = ["tiles", "fan", "rules", "dushan", "encoder_v3"]

# 资产收集：整个目录收录 / 排除规则
ASSET_INCLUDE_DIRS = ["tiles", "table", "sprites", "avatar", "sound", "font"]
ASSET_EXCLUDE = {
    "font/zhunyuan.ttf",          # 9.9MB：字体分支已重定向到 woff2 子集
    "table/continue.png",         # 3.2MB：dushan.html 未引用
    "icon.png",                   # favicon，页面未引用
    "table/number",               # 38K，页面未引用（目录）
}


def die(msg: str) -> None:
    print(f"[build] FAIL: {msg}", file=sys.stderr)
    sys.exit(1)


def patch(html: str, old: str, new: str, tag: str, count: int = 1) -> str:
    n = html.count(old)
    if n < count:
        die(f"补丁 {tag} 未命中（期望 ≥{count} 处，实际 {n} 处）——"
            f"web/static/dushan.html 可能已更新，请核对补丁点")
    return html.replace(old, new, count)


# ---------------------------------------------------------------------------
# 资产收集（单文件模式用）
# ---------------------------------------------------------------------------
def collect_assets() -> dict[str, str]:
    out: dict[str, str] = {}
    for d in ASSET_INCLUDE_DIRS:
        base = ASSETS / d
        if not base.is_dir():
            die(f"资产目录缺失：{base}")
        for f in sorted(base.rglob("*")):
            if not f.is_file():
                continue
            rel = f.relative_to(ASSETS).as_posix()
            if any(rel == e or rel.startswith(e + "/") for e in ASSET_EXCLUDE):
                continue
            mime = {
                ".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
                ".mp3": "audio/mpeg", ".woff2": "font/woff2", ".ttf": "font/ttf",
            }.get(f.suffix.lower())
            if mime is None:
                continue
            out[rel] = f"data:{mime};base64," + base64.b64encode(f.read_bytes()).decode()
    return out


# ---------------------------------------------------------------------------
# 引擎模块包装（去 CommonJS 化：__engModule 工厂 + 本地 require 解析）
# ---------------------------------------------------------------------------
def wrap_module(name: str, src: str) -> str:
    body = src.rstrip("\n")
    return f'__engModule("{name}", function (require, module, exports) {{\n{body}\n}});\n'


GLUE_TEMPLATE = """(function () {
/* ==== 离线单机版：本地引擎装配 ==== */
const __MODS = {};
function __engModule(name, factory) {
  const module = { exports: {} };
  const require = (p) => {
    let key = String(p).replace(/^\\.\\//, "").replace(/\\.js$/, "").replace(/^engine\\//, "");
    if (!__MODS[key]) throw new Error("[standalone] module not loaded: " + p);
    return __MODS[key];
  };
  factory(require, module, module.exports);
  __MODS[name] = module.exports;
}
/* -- 引擎模块（依赖序：tiles → fan → rules → dushan → encoder_v3）-- */
%%ENGINE_MODULES%%
/* -- 本地引擎 API 层 -- */
%%LOCAL_SERVER%%
window.__DushanEngine = __MODS;   /* 供 logits 对拍页 / 调试取用 */

/* -- ORT 会话引导（模型内嵌 / 开发版 fetch）-- */
const WASM_BYTES = %%WASM_EXPR%%;            /* Uint8Array 或 null */
const MODEL_BYTES = %%MODEL_EXPR%%;          /* Promise<Uint8Array> */

window.__DushanReady = (async () => {
  let session = null;
  try {
    ort.env.wasm.numThreads = 1;   // 无跨域隔离（file:// 打开）→ 单线程
    %%ORT_WASM_SETUP%%
    const mb = await MODEL_BYTES;
    session = await ort.InferenceSession.create(mb, { executionProviders: ["wasm"] });
    console.log("[standalone] ONNX 会话就绪：inputs=" + session.inputNames.join(",")
      + " outputs=" + session.outputNames.join(","));
  } catch (e) {
    console.error("[standalone] ONNX 会话创建失败：", e);
  }
  const api = __MODS.local_engine_server.createLocalApi();
  api.setInferFn(async (obs) => {
    if (!session) throw new Error("模型未就绪");
    const inNames = session.inputNames;
    const tilesName = inNames.indexOf("tiles") >= 0 ? "tiles" : inNames[0];
    const globName = inNames.indexOf("glob") >= 0 ? "glob" : inNames[1];
    const out = await session.run({
      [tilesName]: new ort.Tensor("float32", obs.tiles.slice(), [1, 63, 3, 9]),
      [globName]: new ort.Tensor("float32", obs.glob.slice(), [1, 130]),
    });
    const names = session.outputNames;
    const pick = (key) => {
      let i = names.findIndex(n => n.indexOf(key) >= 0);
      if (i < 0) i = names.indexOf(key);
      return out[i >= 0 ? names[i] : key].data;
    };
    return {
      discard: pick("discard"), self_kong: pick("self_kong"), respond: pick("respond"),
    };
  });
  return api;
})();

/* 前端 api() 的后端：所有端点等模型就绪后进入本地引擎 */
window.__LocalAPI = function (path, body) {
  return window.__DushanReady.then(api => api.handle(path, body));
};
})();
"""

PRELUDE_TEMPLATE = """<script>
/* ==== 离线单机版：资产解析器（%%MODE%%） ==== */
%%ASSETS_DECL%%
function __asset(path) {
  if (typeof __ASSETS !== "undefined") {
    const hit = __ASSETS[path];
    if (hit) return hit;
  }
  return __ASSET_BASE + path;
}
</script>
<style>/* 离线单机版：内嵌唯一模型（c4_it2280），隐藏对手模型下拉 */
#boot-players .pickmodel, #set-players .pickmodel { display: none !important; }
</style>
"""


def build(mode: str) -> tuple[str, str]:
    html = SRC_HTML.read_text(encoding="utf-8")
    single = mode == "single"

    # ---- P6 字体：ttf 分支重定向到 woff2 子集（两处）----
    html = patch(html,
                 'zhunyuan.ttf") format("truetype")',
                 'zhunyuan_subset.woff2") format("woff2")',
                 "P6 字体重定向", count=2)

    # ---- P4 模板字面量资产路径 → __asset() ----
    html = patch(html,
                 'const indSrc = color => `/assets/table/tile_indicator_${color}.png`;',
                 'const indSrc = color => __asset("table/tile_indicator_" + color + ".png");',
                 "P4 indSrc")
    html = patch(html,
                 '  const x = TILES[t]; return `/assets/tiles/${x.rank}${SUIT_FILE[x.suit] || "万"}.png`;',
                 '  const x = TILES[t]; return __asset("tiles/" + x.rank + (SUIT_FILE[x.suit] || "万") + ".png");',
                 "P4 tileImg")
    html = patch(html,
                 '  if (ROOT_SND.includes(name)) return "/assets/sound/" + f;\n'
                 '  return `/assets/sound/${gender === "girl" ? "sound_girl" : "sound_boy"}/${f}`;',
                 '  if (ROOT_SND.includes(name)) return __asset("sound/" + f);\n'
                 '  return __asset("sound/" + (gender === "girl" ? "sound_girl" : "sound_boy") + "/" + f);',
                 "P4 sndPath")
    html = patch(html,
                 '  if (!v || v === "random") return `/assets/sound/bgmusic/bg${Math.floor(Math.random() * 5)}.mp3`;\n'
                 '  return `/assets/sound/bgmusic/${/^bg[0-4]$/.test(v) ? v : "bg0"}.mp3`;',
                 '  if (!v || v === "random") return __asset("sound/bgmusic/bg" + Math.floor(Math.random() * 5) + ".mp3");\n'
                 '  return __asset("sound/bgmusic/" + (/^bg[0-4]$/.test(v) ? v : "bg0") + ".mp3");',
                 "P4 bgm")
    html = patch(html,
                 '`/assets/avatar/${AVATAR_DIR[nm] || "boy"}/${nm}.jpg`',
                 '__asset("avatar/" + (AVATAR_DIR[nm] || "boy") + "/" + nm + ".jpg")',
                 "P4 avatarSrc", count=2)
    html = patch(html,
                 'function avaSrc(nm) { return `/assets/avatar/${AVATAR_DIR[nm]}/${nm}.jpg`; }',
                 'function avaSrc(nm) { return __asset("avatar/" + AVATAR_DIR[nm] + "/" + nm + ".jpg"); }',
                 "P4 avaSrc")
    html = patch(html,
                 '<img src="/assets/avatar/${dir}/${nm}.jpg" alt="${nm}">',
                 '<img src="${__asset("avatar/" + dir + "/" + nm + ".jpg")}" alt="${nm}">',
                 "P4 avaPicker")
    html = patch(html,
                 '<img src="/assets/sprites/${sprite[b.kind] || "peng"}.png" alt="${b.label}">',
                 '<img src="${__asset("sprites/" + (sprite[b.kind] || "peng") + ".png")}" alt="${b.label}">',
                 "P4 actSprites")
    html = patch(html,
                 'return "/assets/sprites/" + ({ 1: "next_win", 2: "opposite_win", 3: "before_win" }[rel] || "self_win") + ".png";',
                 'return __asset("sprites/" + ({ 1: "next_win", 2: "opposite_win", 3: "before_win" }[rel] || "self_win") + ".png");',
                 "P4 resIconSrc")

    # ---- P1 api() shim ----
    html = patch(html,
                 'async function api(path, body) {\n'
                 '  const opt = body ? { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) } : {};\n'
                 '  const r = await fetch(path, opt);\n'
                 '  return r.json();\n'
                 '}',
                 'async function api(path, body) {\n'
                 '  /* 离线单机版：直接调用内嵌本地引擎（无 HTTP） */\n'
                 '  return window.__LocalAPI(path, body);\n'
                 '}',
                 "P1 api shim")

    # ---- P5 剩余静态 /assets/ 字面量 ----
    assets_map = collect_assets() if single else {}
    static_refs = sorted(set(re.findall(r'["\'(]/assets/([^"\')$`]+)["\')]', html)))

    def sub(m: re.Match) -> str:
        quote, path, close = m.group(1), m.group(2), m.group(3)
        if single:
            if path not in assets_map:
                print(f"[build] 警告：静态引用未在资产表：{path}（保留原样）")
                return m.group(0)
            return quote + assets_map[path] + close
        return quote + DEV_ASSET_BASE + path + close

    html = re.sub(r'(["\'(])/assets/([^"\')$`]+)(["\')])', sub, html)
    print(f"[build] 静态资产引用替换：{len(static_refs)} 种路径")

    # ---- P2 注入：资产解析器 + 引擎装配 + ORT ----
    engine_modules = "".join(
        wrap_module(name, (ENGINE_DIR / f"{name}.js").read_text(encoding="utf-8"))
        for name in ENGINE_FILES)
    local_server = wrap_module(
        "local_engine_server", LOCAL_SERVER.read_text(encoding="utf-8"))

    if single:
        ort_src = (LIB_DIR / "ort.min.js").read_text(encoding="utf-8")
        mjs_src = (LIB_DIR / "ort-wasm-simd-threaded.mjs").read_text(encoding="utf-8")
        wasm_b64 = base64.b64encode((LIB_DIR / "ort-wasm-simd-threaded.wasm")
                                    .read_bytes()).decode()
        model_b64 = base64.b64encode(MODEL.read_bytes()).decode()
        prelude = PRELUDE_TEMPLATE.replace("%%MODE%%", "单文件内嵌") \
            .replace("%%ASSETS_DECL%%", "const __ASSETS = " + json.dumps(assets_map) + ";\n"
                     "const __ASSET_BASE = \"/assets/\";")
        assets_block = ""       # 资产表已并入 prelude
        ort_block = f"<script>\n{ort_src}\n</script>\n"
        b64_block = (
            f'<script type="application/octet-stream" id="__b64_wasm">{wasm_b64}</script>\n'
            f'<script type="application/octet-stream" id="__b64_model">{model_b64}</script>\n'
            f'<script type="text/plain" id="__mjs_text">{mjs_src}</script>\n'
        )
        b64_to_u8 = ('(function(){const b=document.getElementById("__b64_%s")'
                     '.textContent.trim();const bin=atob(b);'
                     'const u=new Uint8Array(bin.length);'
                     'for(let i=0;i<bin.length;i++)u[i]=bin.charCodeAt(i);return u;})()')
        glue = GLUE_TEMPLATE
        glue = glue.replace("%%WASM_EXPR%%", b64_to_u8 % "wasm")
        glue = glue.replace("%%MODEL_EXPR%%", "Promise.resolve(" + b64_to_u8 % "model" + ")")
        # ort-wasm glue：wasm 二进制内嵌 + .mjs 胶水以 Blob URL 供动态 import
        #（浏览器标准做法；Node 冒烟走 check_single_html.js 的磁盘路径补丁）
        glue = glue.replace(
            "%%ORT_WASM_SETUP%%",
            'ort.env.wasm.wasmBinary = WASM_BYTES;\n'
            '    ort.env.wasm.wasmPaths = { mjs: URL.createObjectURL(new Blob('
            '[document.getElementById("__mjs_text").textContent],'
            ' { type: "text/javascript" })) };')
    else:
        prelude = PRELUDE_TEMPLATE.replace("%%MODE%%", "开发版") \
            .replace("%%ASSETS_DECL%%",
                     f'const __ASSET_BASE = "{DEV_ASSET_BASE}";')
        assets_block = ""
        ort_block = '<script src="lib/ort.min.js"></script>\n'
        b64_block = ""
        glue = GLUE_TEMPLATE
        glue = glue.replace("%%WASM_EXPR%%", "null /* 开发版：按 wasmPaths 相对路径找 */")
        glue = glue.replace("%%MODEL_EXPR%%",
                            f'fetch("{DEV_MODEL_URL}").then(r => r.arrayBuffer())'
                            '.then(b => new Uint8Array(b))')
        glue = glue.replace("%%ORT_WASM_SETUP%%",
                            'ort.env.wasm.wasmPaths = "lib/";')

    glue = glue.replace("%%ENGINE_MODULES%%", engine_modules)
    glue = glue.replace("%%LOCAL_SERVER%%", local_server)
    glue_block = f'<script>\n{glue}</script>\n'

    # 注入位置：prelude 进 </head> 前；引擎脚本进 </body> 前
    if "</head>" not in html:
        die("找不到 </head>")
    html = html.replace("</head>", prelude + "</head>", 1)
    if "</body>" not in html:
        die("找不到 </body>")
    html = html.replace("</body>",
                        assets_block + ort_block + b64_block + glue_block + "</body>", 1)
    return html, prelude


def main() -> int:
    for p in (SRC_HTML, LOCAL_SERVER, MODEL, LIB_DIR / "ort.min.js"):
        if not p.exists():
            die(f"输入缺失：{p}")
    print(f"[build] 源：{SRC_HTML.relative_to(ROOT)}")
    dev, _ = build("dev")
    OUT_DEV.write_text(dev, encoding="utf-8")
    print(f"[build] 开发版 → {OUT_DEV.relative_to(ROOT)}  "
          f"({OUT_DEV.stat().st_size / 1e6:.1f} MB)")
    single, _ = build("single")
    OUT_SINGLE.write_text(single, encoding="utf-8")
    print(f"[build] 单文件版 → {OUT_SINGLE.relative_to(ROOT)}  "
          f"({OUT_SINGLE.stat().st_size / 1e6:.1f} MB)")
    print("[build] OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
