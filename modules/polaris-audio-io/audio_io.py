# -*- coding: utf-8 -*-
"""北极星音频外设统一入口。

  say   文字 → 语音（说话）
  hear  语音 → 文字（听觉）
  loop  自己说、自己听、自己打分（闭环自测，不需要麦克风）

用法：
  python audio_io.py say "你好，主人"  --play
  python audio_io.py hear data/a.wav
  python audio_io.py loop "今天下午要传权重，注意别把信箱挤死"  --rounds 3
"""
import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import listen
import speak


def cmd_say(a):
    r = speak.speak(a.text, out=a.out, engine=a.engine, voice=a.voice,
                    rate=a.rate, playback=not a.no_play)
    print(json.dumps(r, ensure_ascii=False))
    return 0


def cmd_hear(a):
    r = listen.hear(a.path, language=a.lang, model_dir=a.model)
    print(json.dumps(r, ensure_ascii=False))
    return 0


def cmd_loop(a):
    """闭环自测：TTS 生成 → Whisper 听回 → 算 CER。

    这是判断「外设是否真的可用」的唯一干净方式：
    参考答案由我们自己给定，音频由我们自己合成，不存在标注错误。
    """
    rows, total = [], 0.0
    for i in range(a.rounds):
        info = speak.speak(a.text, engine=a.engine, voice=a.voice, playback=False)
        heard = listen.hear(info["path"], language=a.lang, model_dir=a.model)
        c = listen.cer(a.text, heard["text"])
        total += c
        rows.append({
            "round": i + 1,
            "tts_engine": info["engine"],
            "audio_sec": info["seconds"],
            "asr_sec": heard["seconds"],
            "rtf": heard["rtf"],
            "cer": round(c, 4),
            "ref": a.text,
            "hyp": heard["text"],
        })
        print("[%d] CER=%.3f  %.1fs音频/%.1fs识别  %s"
              % (i + 1, c, info["seconds"] or 0, heard["seconds"], heard["text"]), flush=True)

    mean = total / max(a.rounds, 1)
    print("\n均值 CER = %.4f（%d 轮）" % (mean, a.rounds))
    if a.json:
        with open(a.json, "w", encoding="utf-8") as f:
            json.dump({"mean_cer": mean, "rows": rows}, f, ensure_ascii=False, indent=2)
        print("落盘：%s" % a.json)
    return 0 if mean < 0.5 else 1


def main():
    p = argparse.ArgumentParser(prog="audio_io", description="北极星音频外设")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("say", help="文字 → 语音")
    s.add_argument("text")
    s.add_argument("-o", "--out", default=None)
    s.add_argument("-e", "--engine", default="auto", choices=["auto", "natural", "local"])
    s.add_argument("-v", "--voice", default=None)
    s.add_argument("-r", "--rate", type=int, default=0)
    s.add_argument("--no-play", action="store_true")
    s.set_defaults(fn=cmd_say)

    h = sub.add_parser("hear", help="语音 → 文字")
    h.add_argument("path")
    h.add_argument("--lang", default="zh")
    h.add_argument("--model", default=None)
    h.set_defaults(fn=cmd_hear)

    l = sub.add_parser("loop", help="闭环自测")
    l.add_argument("text")
    l.add_argument("-n", "--rounds", type=int, default=1)
    l.add_argument("-e", "--engine", default="auto", choices=["auto", "natural", "local"])
    l.add_argument("-v", "--voice", default=None)
    l.add_argument("--lang", default="zh")
    l.add_argument("--model", default=None)
    l.add_argument("--json", default=None)
    l.set_defaults(fn=cmd_loop)

    a = p.parse_args()
    return a.fn(a)


if __name__ == "__main__":
    raise SystemExit(main())
