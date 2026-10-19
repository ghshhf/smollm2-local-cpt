# -*- coding: utf-8 -*-
"""说话模块：文字 → 语音。

两条引擎，按可用性自动选：
  natural —— edge-tts 神经音色（联网；音质接近真人；出 mp3）
  local   —— Windows SAPI Huihui（离线；音质机械；出 wav）

本模块不碰北极星任何参数（见 interface.json，kind=tool）。
它提供的是「外设」，不是「行为」——所以不受恒定维接口契约约束。

为什么要有两条：神经音色依赖微软服务，断网就哑；SAPI 永远在。
工业级封装的第一原则是「不能因为网断了就变成哑巴」。
"""
import asyncio
import os
import subprocess
import tempfile
import time

SAPI_VOICE = "Microsoft Huihui Desktop"
EDGE_VOICE = "zh-CN-XiaoxiaoNeural"

EDGE_VOICES = {
    "xiaoxiao": "zh-CN-XiaoxiaoNeural",   # 女，自然，通用
    "xiaoyi": "zh-CN-XiaoyiNeural",       # 女，活泼
    "yunxi": "zh-CN-YunxiNeural",         # 男，年轻
    "yunjian": "zh-CN-YunjianNeural",     # 男，解说腔
    "yunyang": "zh-CN-YunyangNeural",     # 男，新闻腔
}


def _edge_mp3(text, out, voice, rate_pct=0):
    import edge_tts
    rate = "%+d%%" % rate_pct

    async def _go():
        c = edge_tts.Communicate(text, voice, rate=rate)
        await c.save(out)

    asyncio.run(_go())
    return out


def _sapi_wav(text, out, voice=SAPI_VOICE, rate=0):
    """SAPI 走 PowerShell。中文靠 UTF-8 临时文件传递，避开命令行编码坑。"""
    txt = out + ".src.txt"
    with open(txt, "w", encoding="utf-8") as f:
        f.write(text)
    ps = (
        "Add-Type -AssemblyName System.Speech;"
        "$t=[IO.File]::ReadAllText('%s',[Text.Encoding]::UTF8);"
        "$s=New-Object System.Speech.Synthesis.SpeechSynthesizer;"
        "try{$s.SelectVoice('%s')}catch{};"
        "$s.Rate=%d;"
        "$s.SetOutputToWaveFile('%s');"
        "$s.Speak($t);$s.Dispose()"
    ) % (txt.replace("\\", "/"), voice, int(rate), out.replace("\\", "/"))
    try:
        r = subprocess.run(
            ["powershell", "-NoProfile", "-NonInteractive", "-Command", ps],
            capture_output=True, timeout=120,
        )
        if r.returncode != 0:
            raise RuntimeError(r.stderr.decode("utf-8", "replace")[:300])
        if not os.path.exists(out) or os.path.getsize(out) < 200:
            raise RuntimeError("SAPI 没产出音频")
    finally:
        try:
            os.remove(txt)
        except OSError:
            pass
    return out


def to_wav(path, wav=None):
    """任意格式 → 播放器能吃的 wav（soundfile 解码，纯 Python 不依赖 ffmpeg）。"""
    if path.lower().endswith(".wav"):
        return path
    import soundfile as sf
    data, sr = sf.read(path, dtype="float32", always_2d=False)
    if data.ndim > 1:
        data = data.mean(axis=1)
    if wav is None:
        wav = os.path.join(tempfile.gettempdir(), "polaris_tts_%d.wav" % int(time.time() * 1000))
    sf.write(wav, data, sr)
    return wav


def play(path):
    """同步播放。wav 直接给 winsound，mp3 先转 wav。"""
    wav = to_wav(path)
    import winsound
    winsound.PlaySound(wav, winsound.SND_FILENAME)
    return wav


def speak(text, out=None, engine="auto", voice=None, rate=0, playback=False):
    """文字 → 语音文件。

    engine: auto | natural | local
    out:    目标路径；None 时落到临时目录
    rate:   -10..10（local）或 -50..+50%（natural，此处按同比例放大）
    返回 dict(engine, path, seconds, bytes)
    """
    text = (text or "").strip()
    if not text:
        raise ValueError("空文本")

    if out is None:
        out = os.path.join(tempfile.gettempdir(),
                           "polaris_tts_%d" % int(time.time() * 1000))

    used, path = None, None
    order = {"auto": ["natural", "local"], "natural": ["natural"], "local": ["local"]}[engine]

    for eng in order:
        try:
            if eng == "natural":
                v = EDGE_VOICES.get(voice or "", voice) or EDGE_VOICE
                p = out if out.lower().endswith(".mp3") else out + ".mp3"
                _edge_mp3(text, p, v, rate_pct=rate * 5)
            else:
                p = out if out.lower().endswith(".wav") else out + ".wav"
                _sapi_wav(text, p, voice or SAPI_VOICE, rate=rate)
            used, path = eng, p
            break
        except Exception as e:
            if eng == order[-1]:
                raise RuntimeError("两条引擎都失败，最后一条：%s" % str(e)[:200])

    info = {"engine": used, "path": os.path.abspath(path), "bytes": os.path.getsize(path)}
    try:
        import soundfile as sf
        info["seconds"] = round(sf.info(path).duration, 2)
    except Exception:
        info["seconds"] = None

    if playback:
        play(path)
    return info


if __name__ == "__main__":
    import json
    import sys
    t = sys.argv[1] if len(sys.argv) > 1 else "你好，主人。我是北极星。"
    print(json.dumps(speak(t), ensure_ascii=False))
