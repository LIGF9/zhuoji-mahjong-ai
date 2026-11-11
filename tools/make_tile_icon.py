#!/usr/bin/env python3
"""离线版应用图标：直接用**牌墙里的那张幺鸡**（web/static/assets/tiles/1条.png）。

为什么不是另画一只鸟
--------------------
用户要求「logo 用游戏中牌墙中的幺鸡图案」。牌面素材只有 76×118 一张位图，
但它是牌桌上真实渲染的那张 —— 复用它可以保证「桌面图标 / 启动图 / 牌桌里的
幺鸡」三处完全同源，不会出现「图标是孔雀、牌桌是另一只鸟」的割裂。

放大策略
--------
原图 5× LANCZOS + 轻锐化。扁平线稿放大后边缘会略微发虚，所以再叠一层
1.15 的 USM（半径 1.2px）把绿色线条拉实；不引入任何新颜色，避免和牌桌
里的同一张牌看起来不像。

产出
----
  * mipmap-*/ic_launcher.png            传统方图（含圆角与深绿毡底）
  * mipmap-*/ic_launcher_round.png      圆形遮罩版
  * mipmap-*/ic_launcher_foreground.png 自适应图标前景（透明底，牌居中于安全区）
  * drawable/splash_tile.png            启动页用的透明底大图
  * src-tauri/icons/*                   桌面 / 其它平台
"""
from __future__ import annotations

import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter

ROOT = Path(__file__).resolve().parent.parent
TILE = ROOT / "web/static/assets/tiles/1条.png"
RES = ROOT / "tauri-offline/src-tauri/gen/android/app/src/main/res"
SRC_ICONS = ROOT / "tauri-offline/src-tauri/icons"
FE_SRC = ROOT / "tauri-offline/src"

# 与 res/values/ic_launcher_background.xml、启动页毡面同色
BG_TOP = (12, 56, 43)
BG_BOTTOM = (3, 22, 16)
BG_FLAT = (4, 55, 43)

DENSITIES = {"mdpi": 48, "hdpi": 72, "xhdpi": 96, "xxhdpi": 144, "xxxhdpi": 192}


def load_tile(height: int) -> Image.Image:
    im = Image.open(TILE).convert("RGBA")
    w = round(im.width * height / im.height)
    im = im.resize((w, height), Image.LANCZOS)
    # 轻锐化：线稿放大后边缘发虚，USM 把绿线拉实（不新增颜色）
    im = im.filter(ImageFilter.UnsharpMask(radius=1.2, percent=115, threshold=2))
    return im


def rounded_mask(size: tuple[int, int], radius: int) -> Image.Image:
    m = Image.new("L", size, 0)
    ImageDraw.Draw(m).rounded_rectangle((0, 0, size[0] - 1, size[1] - 1), radius, fill=255)
    return m


def gradient_bg(size: int) -> Image.Image:
    g = Image.new("RGB", (1, size))
    for y in range(size):
        t = y / max(size - 1, 1)
        g.putpixel((0, y), tuple(round(a + (b - a) * t) for a, b in zip(BG_TOP, BG_BOTTOM)))
    return g.resize((size, size), Image.BILINEAR).convert("RGBA")


def square_icon(size: int, crop_round: bool = False) -> Image.Image:
    """方图：渐变毡底 + 幺鸡牌（带投影）。"""
    S = 1024
    bg = gradient_bg(S)
    # 牌面高度占 68%：48dp 小尺寸下也能看出「一张牌」，再大就会顶到圆角
    tile = load_tile(round(S * 0.68))
    x = (S - tile.width) // 2
    y = round(S * 0.150)

    shadow = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    sh = Image.new("RGBA", tile.size, (0, 0, 0, 150))
    sh.putalpha(tile.getchannel("A").point(lambda v: int(v * 0.58)))
    shadow.paste(sh, (x, y + round(S * 0.022)), sh.getchannel("A"))
    shadow = shadow.filter(ImageFilter.GaussianBlur(round(S * 0.018)))
    bg.alpha_composite(shadow)
    bg.alpha_composite(tile, (x, y))

    mask = rounded_mask((S, S), round(S * 0.225))
    if crop_round:
        mask = Image.new("L", (S, S), 0)
        ImageDraw.Draw(mask).ellipse((-S * 0.02, -S * 0.02, S * 1.02, S * 1.02), fill=255)
    out = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    out.paste(bg, (0, 0), mask)
    return out.resize((size, size), Image.LANCZOS)


def foreground(size: int) -> Image.Image:
    """自适应图标前景：透明底，牌只占中间 ~58%（108dp 画布的 66% 安全区内）。"""
    S = 1024
    out = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    tile = load_tile(round(S * 0.56))
    x = (S - tile.width) // 2
    y = (S - tile.height) // 2
    sh = Image.new("RGBA", tile.size, (0, 0, 0, 120))
    sh.putalpha(tile.getchannel("A").point(lambda v: int(v * 0.5)))
    shadow = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    shadow.paste(sh, (x, y + round(S * 0.012)), sh.getchannel("A"))
    shadow = shadow.filter(ImageFilter.GaussianBlur(round(S * 0.012)))
    out.alpha_composite(shadow)
    out.alpha_composite(tile, (x, y))
    return out.resize((size, size), Image.LANCZOS)


def main() -> int:
    if not TILE.exists():
        print(f"缺少牌面素材: {TILE}", file=sys.stderr)
        return 1
    for name, px in DENSITIES.items():
        d = RES / f"mipmap-{name}"
        d.mkdir(parents=True, exist_ok=True)
        square_icon(px).save(d / "ic_launcher.png")
        square_icon(px, crop_round=True).save(d / "ic_launcher_round.png")
        fg_px = round(px * 108 / 48)          # 108dp 画布
        foreground(fg_px).save(d / "ic_launcher_foreground.png")
        print(f"  mipmap-{name}: {px}px 方图/圆图 + {fg_px}px 前景")

    # 启动页（WebView 里那张卡片）用的透明底大图
    drawable = RES / "drawable"
    drawable.mkdir(parents=True, exist_ok=True)
    load_tile(512).save(drawable / "splash_tile.png")
    load_tile(256).save(FE_SRC / "splash_tile.png")
    print("  drawable/splash_tile.png + src/splash_tile.png")

    # Tauri 桌面图标（保持与安卓一致）
    if SRC_ICONS.exists():
        square_icon(1024).save(SRC_ICONS / "icon.png")
        for px, fn in ((32, "32x32.png"), (64, "64x64.png"),
                       (128, "128x128.png"), (256, "128x128@2x.png")):
            square_icon(px).save(SRC_ICONS / fn)
        print("  src-tauri/icons/* 已刷新")

    # 背景色常量与图标保持一致
    bgx = RES / "values/ic_launcher_background.xml"
    if bgx.exists():
        bgx.write_text(
            '<?xml version="1.0" encoding="utf-8"?>\n<resources>\n'
            '  <!-- 与牌桌毡面同色（tools/make_tile_icon.py 生成图标时的底色） -->\n'
            f'  <color name="ic_launcher_background">#FF{BG_FLAT[0]:02X}{BG_FLAT[1]:02X}{BG_FLAT[2]:02X}</color>\n'
            '</resources>\n', encoding="utf-8")
        print("  ic_launcher_background.xml 同步为 #04372B")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
