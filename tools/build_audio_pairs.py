# -*- coding: utf-8 -*-
"""build_audio_pairs.py — 把 AISHELL-1 子集切成 (音频, 文本) 对齐对。

输入：data/audio/raw/ 下的说话人 tar.gz 解包结果 + aishell_transcript_v0.8.txt
输出：data/audio/aishell_pairs.jsonl，每行 {"utt","audio","text","dur","sr"}
过滤：时长 1~15s、文本 2~60 字、去重、去空。stdlib wave 读元信息，零依赖。
"""
import json
import os
import sys
import tarfile
import wave

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(ROOT, "data", "audio", "raw")
WAV_DIR = os.path.join(ROOT, "data", "audio", "wav")
OUT = os.path.join(ROOT, "data", "audio", "aishell_pairs.jsonl")
TRANSCRIPT = "aishell_transcript_v0.8.txt"

MIN_DUR, MAX_DUR = 1.0, 15.0
MIN_CHARS, MAX_CHARS = 2, 60


def extract_tars():
    os.makedirs(WAV_DIR, exist_ok=True)
    for f in sorted(os.listdir(RAW)):
        if not f.endswith(".tar.gz"):
            continue
        dest = os.path.join(WAV_DIR, f[:-7])
        marker = os.path.join(dest, ".done")
        if os.path.exists(marker):
            continue
        try:
            with tarfile.open(os.path.join(RAW, f), "r:gz") as tf:
                tf.extractall(dest)
            open(marker, "w").close()
            print("[extract] %s -> %s" % (f, dest), flush=True)
        except Exception as e:
            print("[extract FAIL] %s %s" % (f, e), flush=True)


def load_transcript():
    m = {}
    p = os.path.join(RAW, TRANSCRIPT)
    if not os.path.exists(p):
        print("[transcript MISSING] %s" % p, flush=True)
        return m
    with open(p, encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split(maxsplit=1)
            if len(parts) == 2:
                m[parts[0]] = parts[1].strip()
    print("[transcript] %d utts" % len(m), flush=True)
    return m


def dur_of(path):
    try:
        with wave.open(path, "rb") as w:
            return w.getnframes() / float(w.getframerate()), w.getframerate()
    except Exception:
        return 0.0, 0


def main():
    extract_tars()
    tr = load_transcript()
    seen_text = set()
    rows = []
    for dirpath, _, files in os.walk(WAV_DIR):
        for fn in files:
            if not fn.endswith(".wav"):
                continue
            utt = fn[:-4]
            if utt not in tr:
                continue
            text = tr[utt]
            if not (MIN_CHARS <= len(text) <= MAX_CHARS):
                continue
            if text in seen_text:
                continue
            ap = os.path.join(dirpath, fn)
            d, sr = dur_of(ap)
            if not (MIN_DUR <= d <= MAX_DUR):
                continue
            seen_text.add(text)
            rows.append({"utt": utt, "audio": ap, "text": text,
                         "dur": round(d, 2), "sr": sr})
    rows.sort(key=lambda r: r["utt"])
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    tot = sum(r["dur"] for r in rows)
    print("[pairs] %d 对 -> %s" % (len(rows), OUT), flush=True)
    print("[pairs] 总时长 %.1f 小时" % (tot / 3600), flush=True)
    spk = sorted(set(r["utt"].split("/")[0] for r in rows))
    print("[pairs] 说话人数 %d" % len(spk), flush=True)


if __name__ == "__main__":
    sys.exit(main())
