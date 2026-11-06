"""把上游真人配音 mp3 短音效转成 16kHz / 16bit 单声道 PCM WAV。

背景（2026-11-06 音效七修）：安卓 WebView 里 `<audio>` 元素播放 mp3 短音效
会在长局后被静音（playing 事件照常触发、实际无输出），而 Web Audio 的
decodeAudioData 对 32kbps mp3 支持不稳定。PCM WAV 是**原始采样**，前端可以用
`AudioContext.createBuffer()` 自己填 Float32Array，完全不依赖浏览器解码器，
也不依赖任何 HTMLMediaElement —— 从机制上消掉这两类不确定性。

BGM（bgmusic/bg0..bg4.mp3，52s × 64kbps）保持 mp3 流式播放，不转 PCM：
转完单首约 1.7MB、全部 8MB，而且 BGM 是常驻单元素，故障不在这条路径上。

用法：.venv/bin/python tools/gen_pcm_sfx.py
产物：与 mp3 同目录、同名 .wav（可重复生成，覆盖同名文件）
依赖：macOS /usr/bin/afconvert；其它平台可换 ffmpeg -ac 1 -ar 16000 -c:a pcm_s16le
"""

from __future__ import annotations

import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "web" / "static" / "assets" / "sound"
AFCONVERT = "/usr/bin/afconvert"
# BGM 是长音频，保持 mp3；根目录「胡牌/自摸」是上游遗留、前端未引用，一并跳过。
SKIP_STEMS = {"胡牌", "自摸"}


def convert(src: Path) -> tuple[bool, str]:
    dst = src.with_suffix(".wav")
    cmd = [AFCONVERT, "-f", "WAVE", "-d", "LEI16@16000", "-c", "1", str(src), str(dst)]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        if r.returncode != 0:
            return False, (r.stderr or r.stdout).strip()[:120]
        # 校验 RIFF/WAVE + fmt PCM16 mono 16000 + data 长度
        b = dst.read_bytes()
        if b[:4] != b"RIFF" or b[8:12] != b"WAVE":
            return False, "not RIFF/WAVE"
        if int.from_bytes(b[22:24], "little") != 1 or int.from_bytes(b[34:36], "little") != 16:
            return False, "not PCM16"
        if int.from_bytes(b[24:28], "little") != 16000:
            return False, "sample rate != 16000"
        return True, f"{len(b)}B"
    except Exception as e:  # noqa: BLE001
        return False, str(e)[:120]


def main() -> None:
    mp3s = sorted(
        p for p in ROOT.rglob("*.mp3")
        if "bgmusic" not in p.parts
        # 只跳过「根目录」的遗留文件；sound_boy/sound_girl 下的「自摸」是必需音效
        and not (p.parent == ROOT and p.stem in SKIP_STEMS)
    )
    ok, bad, total = 0, [], 0
    for p in mp3s:
        good, msg = convert(p)
        if good:
            ok += 1
            total += p.with_suffix(".wav").stat().st_size
        else:
            bad.append((str(p.relative_to(ROOT)), msg))
    print(f"mp3={len(mp3s)} wav_ok={ok} wav_bad={len(bad)} bytes={total:,}")
    for name, msg in bad:
        print("  FAIL", name, msg)
    if bad:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
