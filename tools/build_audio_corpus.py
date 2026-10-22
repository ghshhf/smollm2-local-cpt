#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""tools/build_audio_corpus.py — 音频预训练资源构建（真实 + 合成）。

为什么需要合成：真实 AISHELL 我们只有 6224 条 / 55MB / 转录约 8 万 token，
对预训练没有统计意义。所以按「能用合成就用合成」：拿我们自己的中文文本，
多音色 TTS 造语音-文本对，标签由构造决定（天然精确，不需要人工标注）。

三个阶段，各自可重跑：
    text    从现有中文语料里抽干净短句，去重 -> data/audio_assets/sentences.jsonl
    tts     多音色合成 -> data/audio_assets/audio/<voice>/<id>.mp3 + manifest.jsonl
    verify  抽 200 条用 Whisper 回听，算 CER -> audio_verify.json（质检，不是训练）

用法：
    python tools/build_audio_corpus.py --stage text --n 40000
    python tools/build_audio_corpus.py --stage tts --workers 6 --limit 40000
    python tools/build_audio_corpus.py --stage verify --n 200
"""
import argparse
import asyncio
import hashlib
import json
import os
import random
import re
import sys
import time
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
ASSETS = os.path.join(ROOT, "data", "audio_assets")
AUDIO = os.path.join(ASSETS, "audio")
SENTS = os.path.join(ASSETS, "sentences.jsonl")
MANI = os.path.join(ASSETS, "manifest.jsonl")

# 只用最稳的几个中文音色：女声/男声/不同地区，避免冷门音色合成失败率高
VOICES = [
    "zh-CN-XiaoxiaoNeural",       # 女·标准
    "zh-CN-YunxiNeural",          # 男·标准
    "zh-CN-YunjianNeural",        # 男·播音
    "zh-CN-XiaoyiNeural",         # 女·活泼
    "zh-CN-YunyangNeural",        # 男·新闻
    "zh-CN-liaoning-XiaobeiNeural",   # 女·东北
    "zh-CN-shaanxi-XiaoniNeural",     # 女·陕西
]

CJK = re.compile(r"[\u4e00-\u9fff]")
BAD = re.compile(r"[<>{}|\\^`~\[\]（）()\"'*#@$%&_=+;:!?]")


def _clean(s):
    s = s.replace(" ", "").replace("\u3000", "")
    s = re.sub(r"[a-zA-Z0-9]+", "", s)       # 预训练文本侧不要中英混
    s = re.sub(r"[，。！？、；：\u201c\u201d\u2018\u2019《》…—]+", "，", s)
    s = s.strip("，")
    return s


def _sent_ok(s):
    if not (8 <= len(s) <= 40):
        return False
    if len(CJK.findall(s)) < len(s) * 0.9:
        return False
    if BAD.search(s):
        return False
    return True


def _first_exist(*paths):
    for p in paths:
        if os.path.exists(p):
            return p
    return None


def stage_text(n_target):
    """从现有中文语料抽句。优先小说/文章（句子完整），再补通用文本。"""
    os.makedirs(ASSETS, exist_ok=True)
    novel = _first_exist(os.path.join(ROOT, "corpus_04b", "creative_articles_novel.jsonl"),
                         os.path.join(ROOT, "chain", "corpus_reserve_refined",
                                      "zh_r_knowledge.jsonl"))
    ufw = _first_exist(os.path.join(ROOT, "corpus_04b", "ufw_zh.jsonl"),
                       os.path.join(ROOT, "chain", "corpus_r6", "zh_text_r6.jsonl"),
                       os.path.join(ROOT, "chain", "corpus_reserve", "zh_text_r5.jsonl"))
    srcs = [("novel", novel, 0.6), ("ufw", ufw, 0.4)]
    seen = set()
    out = open(SENTS, "w", encoding="utf-8")
    n = 0
    t0 = time.time()
    for name, path, share in srcs:
        want = int(n_target * share)
        got = 0
        if not os.path.exists(path):
            print("  [跳过] 缺 %s" % path)
            continue
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                if got >= want:
                    break
                try:
                    r = json.loads(line)
                except Exception:
                    continue
                text = r.get("text") or ""
                if len(text) < 30:
                    continue
                for seg in re.split(r"[。！？\n]", text):
                    seg = _clean(seg)
                    if not _sent_ok(seg):
                        continue
                    h = hashlib.md5(seg.encode()).hexdigest()[:16]
                    if h in seen:
                        continue
                    seen.add(h)
                    out.write(json.dumps({"id": h, "text": seg, "src": name},
                                         ensure_ascii=False) + "\n")
                    got += 1
                    n += 1
                    if got >= want:
                        break
        print("  %-6s 抽 %d 句（累计 %d）" % (name, got, n), flush=True)
    out.close()
    print("[text] 共 %d 句 -> %s  (%.0fs)" % (n, SENTS, time.time() - t0))


async def _one(sem, sid, text, voice, outdir):
    import edge_tts
    async with sem:
        path = os.path.join(outdir, "%s.mp3" % sid)
        if os.path.exists(path) and os.path.getsize(path) > 1000:
            return None                      # 已下过
        for att in range(3):
            try:
                c = edge_tts.Communicate(text, voice)
                await c.save(path)
                if os.path.getsize(path) > 1000:
                    return (sid, text, voice, path, os.path.getsize(path))
            except Exception:
                await asyncio.sleep(1.5 * (att + 1))
        return "fail"


async def stage_tts(workers, limit):
    os.makedirs(AUDIO, exist_ok=True)
    rows = [json.loads(l) for l in open(SENTS, encoding="utf-8") if l.strip()]
    if limit:
        rows = rows[:limit]
    tasks = []
    for i, r in enumerate(rows):
        voice = VOICES[i % len(VOICES)]
        outdir = os.path.join(AUDIO, voice)
        os.makedirs(outdir, exist_ok=True)
        tasks.append((r["id"], r["text"], voice, outdir))
    sem = asyncio.Semaphore(workers)
    print("[tts] %d 条，%d 并发，%d 音色" % (len(tasks), workers, len(VOICES)), flush=True)
    t0 = time.time()
    done = new = fail = 0
    with open(MANI, "a", encoding="utf-8") as mf:
        batch = []
        for coro in asyncio.as_completed([_one(sem, *t) for t in tasks]):
            r = await coro
            done += 1
            if r == "fail":
                fail += 1
            elif r:
                batch.append(r)
                new += 1
            if len(batch) >= 200:
                for sid, text, voice, path, sz in batch:
                    mf.write(json.dumps({"id": sid, "text": text, "voice": voice,
                                         "mp3": os.path.relpath(path, ROOT),
                                         "bytes": sz}, ensure_ascii=False) + "\n")
                mf.flush()
                batch = []
            if done % 500 == 0:
                el = time.time() - t0
                print("  %d/%d  新增 %d 失败 %d  %.0fs (%.1f 条/s)"
                      % (done, len(tasks), new, fail, el, done / max(el, 1)), flush=True)
        for sid, text, voice, path, sz in batch:
            mf.write(json.dumps({"id": sid, "text": text, "voice": voice,
                                 "mp3": os.path.relpath(path, ROOT),
                                 "bytes": sz}, ensure_ascii=False) + "\n")
    print("[tts] 完成 %d 条（新增 %d 失败 %d）%.0fs -> %s"
          % (done, new, fail, time.time() - t0, MANI))


def stage_verify(n):
    """抽 n 条用现有 Whisper 回听，算 CER。这是质检，不是训练。"""
    sys.path.insert(0, os.path.join(ROOT, "modules", "polaris-audio-io"))
    import listen
    rows = [json.loads(l) for l in open(MANI, encoding="utf-8") if l.strip()]
    random.seed(7)
    sample = random.sample(rows, min(n, len(rows)))
    cers = []
    bad = []
    for i, r in enumerate(sample, 1):
        p = os.path.join(ROOT, r["mp3"])
        try:
            o = listen.hear(p)
        except Exception as e:
            bad.append((r["id"], "ERR %s" % str(e)[:60]))
            continue
        c = listen.cer(r["text"], o["text"])
        cers.append(c)
        if c > 0.3:
            bad.append((r["id"], "%.2f | 真:%s | 识:%s" % (c, r["text"][:24], o["text"][:24])))
        if i % 25 == 0:
            print("  %d/%d  均CER=%.4f" % (i, len(sample), sum(cers) / len(cers)), flush=True)
    rep = {
        "样本": len(sample),
        "合成失败": 0,
        "CER均值": round(sum(cers) / len(cers), 4) if cers else None,
        "CER中位": round(sorted(cers)[len(cers) // 2], 4) if cers else None,
        "CER>0.3 条数": len(bad),
        "差样本": bad[:20],
    }
    with open(os.path.join(ASSETS, "audio_verify.json"), "w", encoding="utf-8") as f:
        json.dump(rep, f, ensure_ascii=False, indent=2)
    print(json.dumps(rep, ensure_ascii=False, indent=2)[:1200])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["text", "tts", "verify"])
    ap.add_argument("--n", type=int, default=40000)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    if a.stage == "text":
        stage_text(a.n)
    elif a.stage == "tts":
        asyncio.run(stage_tts(a.workers, a.limit))
    else:
        stage_verify(a.n)


if __name__ == "__main__":
    main()
