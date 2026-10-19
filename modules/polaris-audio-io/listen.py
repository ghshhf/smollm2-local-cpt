# -*- coding: utf-8 -*-
"""听觉模块：语音 → 文字。

用的是工业级 Whisper 的**完整**权重（encoder + decoder 都在），
不是我们 audio_attach.py 里那个只留 encoder、把 decoder 扔掉的实验版。

区别就在这一步：
  完整版 —— decoder 是训练过的音素→文字解码器，CER 个位数
  实验版 —— 只有 encoder 提特征，认字交给一个没听过声音的中文 LM

本模块不碰北极星任何参数（kind=tool）。它先保证「听得清」，
之后要不要内化进主干，是另一条独立路线（见 docs/audio-align-diagnosis-2026-10-19.md）。
"""
import os
import time

_MODEL = {}
_SR = 16000


def load_audio(path, sr=_SR):
    """任意音频 → (float32 mono numpy, 采样率)。不依赖 ffmpeg。"""
    import numpy as np
    import soundfile as sf

    data, orig = sf.read(path, dtype="float32", always_2d=False)
    if data.ndim > 1:
        data = data.mean(axis=1)
    if orig != sr:
        from math import gcd
        from scipy.signal import resample_poly
        g = gcd(int(orig), int(sr))
        data = resample_poly(data, sr // g, int(orig) // g).astype(np.float32)
    return data, sr


def load(model_dir=None):
    """模型单例。CPU 上加载约 5~15 秒，别在循环里反复调。"""
    if _MODEL:
        return _MODEL["model"], _MODEL["proc"]
    from transformers import WhisperForConditionalGeneration, WhisperProcessor

    d = model_dir or os.environ.get("POLARIS_WHISPER") or \
        os.path.join(os.path.dirname(__file__), "..", "..", "models", "whisper-small")
    d = os.path.abspath(d)
    if not os.path.exists(os.path.join(d, "model.safetensors")):
        raise FileNotFoundError("缺 Whisper 权重：%s" % d)

    proc = WhisperProcessor.from_pretrained(d)
    model = WhisperForConditionalGeneration.from_pretrained(d)
    model.eval()
    _MODEL.update(model=model, proc=proc, dir=d)
    return model, proc


PROMPT_ZH = "以下为普通话。"


def hear(path, language="zh", task="transcribe", model_dir=None, beam=1):
    """语音 → 文字。返回 dict(text, chars, seconds, audio_sec, rtf)。

    🔴 必须给简体 prompt：Whisper 中文默认吐繁体（"離線引擎測試"），
    这不是识别错，是书写系统选错。prompt 一给就回简体，是官方推荐做法。

    🔴 beam 默认 1：实测 beam=1 与 beam=3 的 CER 完全相同（0.120 vs 0.120），
    但 4.57s vs 8.74s，快一倍。这台机器上多花的那 4 秒买不到任何精度。
    """
    import torch

    model, proc = load(model_dir)
    audio, _ = load_audio(path)
    feat = proc(audio, sampling_rate=_SR, return_tensors="pt").input_features
    prompt_ids = proc.get_prompt_ids(PROMPT_ZH, return_tensors="pt") \
        if language == "zh" else None

    t0 = time.time()
    with torch.no_grad():
        kw = dict(num_beams=beam)
        if prompt_ids is not None:
            kw["prompt_ids"] = prompt_ids
            kw["max_new_tokens"] = 448 - int(prompt_ids.shape[-1]) - 8
        else:
            kw["max_new_tokens"] = 440
        try:
            ids = model.generate(feat, language=language, task=task, **kw)
        except (TypeError, ValueError):
            gc = model.generation_config
            gc.language = language
            gc.task = task
            gc.forced_decoder_ids = None
            ids = model.generate(feat, **kw)
    el = time.time() - t0

    text = proc.batch_decode(ids, skip_special_tokens=True)[0].strip()
    audio_sec = len(audio) / float(_SR)
    return {
        "text": text,
        "chars": len(text.replace(" ", "")),
        "audio_sec": round(audio_sec, 2),
        "seconds": round(el, 2),
        "rtf": round(el / max(audio_sec, 0.01), 2),
    }


def cer(ref, hyp):
    """字符级编辑距离 / 参考长度。中文不加空格，直接按字符算。"""
    a = [c for c in (ref or "") if not c.isspace()]
    b = [c for c in (hyp or "") if not c.isspace()]
    if not a:
        return 0.0 if not b else 1.0
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1] / float(len(a))


if __name__ == "__main__":
    import json
    import sys
    p = sys.argv[1] if len(sys.argv) > 1 else None
    if not p:
        print("用法: listen.py <音频文件>")
        raise SystemExit(2)
    print(json.dumps(hear(p), ensure_ascii=False))
