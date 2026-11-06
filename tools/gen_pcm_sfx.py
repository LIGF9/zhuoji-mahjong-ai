"""把上游真人配音 mp3 短音效转成 16kHz / 16bit 单声道 PCM WAV，并做响度归一化。

背景（2026-11-06 音效七修）：安卓 WebView 里 `<audio>` 元素播放 mp3 短音效
会在长局后被静音（playing 事件照常触发、实际无输出），而 Web Audio 的
decodeAudioData 对 32kbps mp3 支持不稳定。PCM WAV 是**原始采样**，前端可以用
`AudioContext.createBuffer()` 自己填 Float32Array，完全不依赖浏览器解码器，
也不依赖任何 HTMLMediaElement —— 从机制上消掉这两类不确定性。

响度归一化（2026-11-06 音效八修）：上游各条录音响度参差（实测 sound_girl
中位峰值 0.46 vs sound_boy 0.73，最安静的只有 0.36），叠加音量滑杆后安静的
牌名几乎听不见。转码后统一做 **RMS 归一化 + 峰值限幅**：目标 RMS=0.18
（≈-15dBFS，游戏短音效常用响度），若提升后峰值超过 0.95 则退化为峰值限幅，
避免削波失真。所有 66 条输出响度一致，前端增益调节才有意义。

BGM（bgmusic/bg0..bg4.mp3，52s × 64kbps）保持 mp3 流式播放，不转 PCM：
转完单首约 1.7MB、全部 8MB，而且 BGM 是常驻单元素，故障不在这条路径上。

用法：.venv/bin/python tools/gen_pcm_sfx.py
产物：与 mp3 同目录、同名 .wav（可重复生成，覆盖同名文件）
依赖：macOS /usr/bin/afconvert；其它平台可换 ffmpeg -ac 1 -ar 16000 -c:a pcm_s16le
"""

from __future__ import annotations

import struct
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "web" / "static" / "assets" / "sound"
AFCONVERT = "/usr/bin/afconvert"
# BGM 是长音频，保持 mp3；根目录「胡牌/自摸」是上游遗留、前端未引用，一并跳过。
SKIP_STEMS = {"胡牌", "自摸"}
TARGET_RMS = 0.18   # 目标响度 ≈ -15dBFS
PEAK_CAP = 0.95     # 峰值上限，防削波


def normalize_wav(path: Path) -> tuple[float, float]:
    """RMS 归一化 + 峰值限幅，就地重写 16bit PCM data。返回 (gain, peak_after)。

    只处理 afconvert 生成的「PCM16 / 单声道 / data 紧随 fmt」布局；遍历 chunk
    定位 data 以兼容 JUNK/fact 扩展块。样本按 f32 计算增益后写回 int16，
    写回前 clamp 到 [-1, 1) 防止 int16 溢出。
    """
    b = bytearray(path.read_bytes())
    if b[:4] != b"RIFF" or b[8:12] != b"WAVE":
        raise ValueError("not RIFF/WAVE")
    off, data_off, data_len = 12, None, None
    while off + 8 <= len(b):
        cid = bytes(b[off:off + 4])
        sz = int.from_bytes(b[off + 4:off + 8], "little")
        body = off + 8
        if cid == b"data":
            data_off, data_len = body, sz
            break
        off = body + sz + (sz & 1)
    if data_off is None or data_len is None:
        raise ValueError("no data chunk")
    n = data_len // 2
    if n <= 0:
        raise ValueError("empty data")

    # 两遍：先算 RMS 与峰值，再按增益写回
    ssum = 0.0
    peak = 0.0
    for i in range(n):
        v = struct.unpack_from("<h", b, data_off + i * 2)[0] / 32768.0
        ssum += v * v
        a = v if v >= 0 else -v
        if a > peak:
            peak = a
    rms = (ssum / n) ** 0.5
    if rms <= 1e-9 or peak <= 1e-9:
        return 1.0, peak
    gain = TARGET_RMS / rms
    if peak * gain > PEAK_CAP:
        gain = PEAK_CAP / peak          # 峰值限幅优先，避免削波
    if abs(gain - 1.0) < 1e-3:
        return gain, peak
    for i in range(n):
        v = struct.unpack_from("<h", b, data_off + i * 2)[0] / 32768.0 * gain
        if v > 0.999969:
            v = 0.999969
        elif v < -1.0:
            v = -1.0
        struct.pack_into("<h", b, data_off + i * 2, int(v * 32767.0))
    path.write_bytes(bytes(b))
    return gain, min(peak * gain, PEAK_CAP)


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
        gain, peak = normalize_wav(dst)
        return True, f"{dst.stat().st_size}B gain={gain:.2f} peak={peak:.3f}"
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
