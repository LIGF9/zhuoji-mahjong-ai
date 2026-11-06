#!/usr/bin/env python3
"""生成 web/static/assets/sound/ 下的音效资源（2026-11-06 补齐丢失音源）。

背景：原工程依赖一批 .mp3（27 牌名 × 男女声 + 碰杠胡自摸 + BGM），但这批
二进制从未进过 git，迁移到新机后整个目录缺失 → 所有设备声音全哑。
本脚本用 macOS 中文 TTS（say + afconvert）读牌名/动作词，用 numpy 合成
操作音与 5 段背景音乐，统一输出 16kHz/16bit 单声道 WAV（浏览器与
Android WebView 原生支持，无需解码器、无需额外依赖）。

用法：.venv/bin/python tools/gen_sound_assets.py
产物：web/static/assets/sound/**.wav（可重复生成，覆盖同名文件）
"""
from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import wave
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "web" / "static" / "assets" / "sound"
SR = 16000
CN_NUM = {"一": "1", "二": "2", "三": "3", "四": "4", "五": "5",
          "六": "6", "七": "7", "八": "8", "九": "9"}
SUITS = ("万", "条", "筒")
ACTIONS = ("碰", "杠", "胡", "自摸")
VOICES = {"boy": ["Eddy", "Reed", "Tingting"], "girl": ["Tingting", "Sandy", "Eddy"]}


def write_wav(path: Path, data: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    x = np.clip(data, -1.0, 1.0)
    pcm = (x * 32767).astype("<i2")
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(pcm.tobytes())


def env(n: int, attack=0.005, release=0.12) -> np.ndarray:
    t = np.arange(n, dtype=np.float64) / SR
    a = np.clip(t / max(attack, 1e-6), 0, 1)
    r = np.clip((t[-1] - t) / max(release, 1e-6), 0, 1) if n > 1 else np.ones(n)
    return a * r


def tone(freq: float, dur: float, amp=0.4, kind="sine", decay=4.0) -> np.ndarray:
    n = int(SR * dur)
    t = np.arange(n, dtype=np.float64) / SR
    if kind == "noise":
        x = np.random.default_rng(int(freq) + n).normal(size=n)
    else:
        x = np.sin(2 * np.pi * freq * t) + 0.3 * np.sin(4 * np.pi * freq * t)
    return amp * x * np.exp(-decay * t) * env(n, release=min(0.08, dur * 0.4))


def mix(*arrs: np.ndarray) -> np.ndarray:
    """把几段不同长度的音叠加到统一长度（避免 numpy 广播报错）。"""
    n = max(len(a) for a in arrs)
    out = np.zeros(n, dtype=np.float64)
    for a in arrs:
        out[:len(a)] += a
    return out


def synth(name: str) -> np.ndarray:
    if name == "点击":
        return tone(880, 0.07, 0.35, decay=18)
    if name == "摸牌":
        return mix(tone(560, 0.10, 0.45, "noise", decay=35), tone(320, 0.09, 0.22))
    if name == "打牌":
        return mix(tone(220, 0.20, 0.5, "noise", decay=22), tone(140, 0.18, 0.35, decay=16))
    if name == "流局":                       # 两声下行，收局感
        a, b = tone(440, 0.22, 0.32, decay=9), tone(294, 0.38, 0.32, decay=6)
        return np.concatenate([a, np.zeros(int(SR * 0.06)), b])
    raise ValueError(name)


def bgm(seed: int) -> np.ndarray:
    """8 秒五声音阶琶音，首尾做循环淡入淡出，音量偏轻。"""
    rng = np.random.default_rng(seed)
    scale = [261.63, 293.66, 329.63, 392.00, 440.00, 523.25, 587.33]
    dur, step = 8.0, 0.5
    n = int(SR * dur)
    out = np.zeros(n, dtype=np.float64)
    for k in range(int(dur / step)):
        f = scale[int(rng.integers(0, len(scale)))]
        seg = tone(f, 0.46, 0.16, decay=2.2)
        i = int(SR * k * step)
        out[i:i + len(seg)] += seg[:max(0, min(len(seg), n - i))]
        # 低八度衬底
        pad = tone(f / 2, 0.48, 0.07, decay=1.2)
        out[i:i + len(pad)] += pad[:max(0, min(len(pad), n - i))]
    fade = int(SR * 0.25)
    out[:fade] *= np.linspace(0, 1, fade)
    out[-fade:] *= np.linspace(1, 0, fade)
    return 0.22 * out


def tts(text: str, dest: Path, voice: str) -> bool:
    dest.parent.mkdir(parents=True, exist_ok=True)   # afconvert 不会自己建父目录
    with tempfile.TemporaryDirectory() as td:
        aiff = Path(td) / "v.aiff"
        r = subprocess.run(["say", "-v", voice, "-r", "175", "-o", str(aiff), text],
                           capture_output=True, text=True)
        if r.returncode != 0 or not aiff.exists():
            return False
        c = subprocess.run(["afconvert", "-f", "WAVE", "-d", f"LEI16@{SR}",
                            "-c", "1", str(aiff), str(dest)],
                           capture_output=True, text=True)
        return c.returncode == 0 and dest.exists()


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    made, failed = 0, []

    for n in ("打牌", "摸牌", "流局", "点击"):
        write_wav(OUT / f"{n}.wav", synth(n))
        made += 1

    for i in range(5):
        write_wav(OUT / "bgmusic" / f"bg{i}.wav", bgm(100 + i))
        made += 1

    for gender, candidates in VOICES.items():
        # 依次试候选语音，取第一个本机已下载的
        voice = next((v for v in candidates if tts("一", Path(tempfile.gettempdir()) / "_probe.wav", v)),
                     None)
        probe = Path(tempfile.gettempdir()) / "_probe.wav"
        if probe.exists():
            probe.unlink()
        if not voice:
            print(f"[warn] {gender}: 无可用中文语音，跳过该通道", file=sys.stderr)
            continue
        print(f"[tts] {gender} → {voice}")
        texts: list[tuple[str, str]] = []
        for s in SUITS:
            for cn, ar in CN_NUM.items():
                texts.append((f"{ar}{s}", cn + s))
        for a in ACTIONS:
            texts.append((a, a))
        for fname, spoken in texts:
            dest = OUT / f"sound_{gender}" / f"{fname}.wav"
            if not tts(spoken, dest, voice):
                failed.append(str(dest))
            else:
                made += 1

    total = sum(p.stat().st_size for p in OUT.rglob("*.wav"))
    print(f"[done] {made} 个 wav，合计 {total / 1024 / 1024:.2f} MiB → {OUT}")
    if failed:
        print(f"[warn] {len(failed)} 个生成失败：{failed[:5]}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
