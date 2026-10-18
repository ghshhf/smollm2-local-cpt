#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
build_reserve.py — 在 node1(魔搭 DSW, 中国网络)上构建 0.3B 知识储备的【分类语料库】。
分类对齐过去 R 级:
  zh_r_knowledge.jsonl— 知识骨干 (复制现有 R6: wiki+edu)
  zh_r_ancient.jsonl  — 古籍/史/诸子/诗词 (flowerone CC0 + NiuTrans 古文原文 777书)
  zh_r_novels.jsonl   — 小说/叙事 (wdndev/webnovel 9000本, 按卷头切片)
  zh_r_modern.jsonl   — 公众号/社科/科普/当代 (nonamestreet/weixin_public_corpus)
每个源独立 try/except; 已存在且非空则跳过(幂等, 支持断点续跑)。
结果上报 ECS /file/status_reserve.txt

环境变量:
  NOVEL_MB    小说目标字节上限 MB (默认 400)
  MODERN_MB   公众号目标字节上限 MB (默认 400)
  DO_WEIXIN   1=包含公众号(需 git clone 3GB + unzip, 默认 0)
"""
import os, json, sys, subprocess, time

W = "/mnt/workspace/node1/chain"
RESERVE = os.path.join(W, "corpus_reserve")
os.makedirs(RESERVE, exist_ok=True)
TMP = "/tmp/reserve"
os.makedirs(TMP, exist_ok=True)

NOVEL_MB = float(os.environ.get("NOVEL_MB", "400"))
MODERN_MB = float(os.environ.get("MODERN_MB", "400"))
DO_WEIXIN = os.environ.get("DO_WEIXIN", "0") == "1"

def L(m):
    print(m, flush=True)

def sh(cmd, timeout=900):
    return subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout)

def wget(url, out, timeout=600):
    r = sh(f"wget -c -q -L --timeout=90 -O '{out}' '{url}'", timeout=timeout)
    return r.returncode == 0 and os.path.exists(out) and os.path.getsize(out) > 100

def ensure(mod, pipname=None):
    try:
        return __import__(mod)
    except Exception:
        sh(f"pip install -q {pipname or mod}")
        return __import__(mod)

def hf_files(repo):
    r = sh(f"curl -s -m30 'https://hf-mirror.com/api/datasets/{repo}'")
    try:
        return [s["rfilename"] for s in json.loads(r.stdout).get("siblings", [])]
    except Exception:
        return []

def write_jsonl(path, records):
    n = 0
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            t = (rec.get("text") or "").strip()
            if len(t) < 30:
                continue
            f.write(json.dumps({"text": t, "src": rec.get("src", "?"), "cat": rec.get("cat", "?")},
                               ensure_ascii=False) + "\n")
            n += 1
    return n

def done(path):  # 幂等: 存在且非空才跳过
    return os.path.exists(path) and os.path.getsize(path) > 0

# ---------- 知识骨干 (knowledge) ----------
def src_knowledge():
    out = os.path.join(RESERVE, "zh_r_knowledge.jsonl")
    if done(out):
        L(f"[knowledge] skip (exists {os.path.getsize(out)/1e6:.1f}MB)"); return
    src = os.path.join(W, "corpus_r6", "zh_text_r6.jsonl")
    if os.path.exists(src):
        sh(f"cp '{src}' '{out}'")
        L(f"[knowledge] copied R6 -> {out}  {os.path.getsize(out)/1e6:.1f}MB")
    else:
        L("[knowledge] R6 source not found on NAS")

# ---------- 古籍/史/诸子 (ancient) ----------
def src_ancient():
    out = os.path.join(RESERVE, "zh_r_ancient.jsonl")
    if done(out):
        L(f"[ancient] skip (exists {os.path.getsize(out)/1e6:.1f}MB)"); return
    recs = []
    try:
        pq = ensure("pyarrow.parquet", "pyarrow")
        p = os.path.join(TMP, "classical.parquet")
        cands = [f for f in hf_files("flowerone/chinese-classical-corpus")
                 if "corpus" in f and f.endswith(".parquet")]
        if cands and wget(f"https://hf-mirror.com/datasets/flowerone/chinese-classical-corpus/resolve/main/{cands[0]}", p):
            t = pq.read_table(p).to_pylist()
            for r in t:
                c = (r.get("content") or "").strip()
                if len(c) >= 30:
                    recs.append({"text": c, "src": "classical:" + str(r.get("source", "?")), "cat": "ancient"})
            L(f"[ancient] flowerone corpus: +{len(recs)} recs")
    except Exception as e:
        L(f"[ancient] flowerone failed: {e}")
    try:
        d = os.path.join(TMP, "Classical-Modern")
        if not os.path.isdir(d):
            sh(f"git clone --depth 1 https://github.com/NiuTrans/Classical-Modern '{d}'")
        gudir = os.path.join(d, "古文原文")
        if os.path.isdir(gudir):
            book = {}
            for root, _, files in os.walk(gudir):
                if "text.txt" in files:
                    try:
                        txt = open(os.path.join(root, "text.txt"), encoding="utf-8", errors="ignore").read().strip()
                    except Exception:
                        continue
                    if not txt:
                        continue
                    bookname = os.path.basename(os.path.dirname(root)) or os.path.basename(root)
                    book[bookname] = book.get(bookname, "") + "\n" + txt
            for name, txt in book.items():
                if len(txt.strip()) >= 30:
                    recs.append({"text": txt.strip(), "src": "niutrans:" + name, "cat": "ancient"})
            L(f"[ancient] NiuTrans 古文原文: +{len(book)} books")
    except Exception as e:
        L(f"[ancient] NiuTrans failed: {e}")
    n = write_jsonl(out, recs)
    L(f"[ancient] -> {out}  {n} lines / {os.path.getsize(out)/1e6:.1f}MB")

# ---------- 小说/叙事 (novels) — JSONL 分卷头切片 ----------
def src_novels():
    out = os.path.join(RESERVE, "zh_r_novels.jsonl")
    if done(out):
        L(f"[novels] skip (exists {os.path.getsize(out)/1e6:.1f}MB)"); return
    cap = NOVEL_MB * 1e6
    remaining = cap
    n = 0
    shards = [f"data/webnovel_{i}.jsonl" for i in range(10)]
    with open(out, "w", encoding="utf-8") as f:
        for s in shards:
            if remaining <= 0:
                break
            rng = int(min(remaining + 2_000_000, 60_000_000))
            url = f"https://hf-mirror.com/datasets/wdndev/webnovel-chinese/resolve/main/{s}"
            p = os.path.join(TMP, os.path.basename(s))
            r = sh(f"curl -sL -m300 -r 0-{rng} -o '{p}' '{url}'", timeout=360)
            if r.returncode != 0 or not os.path.exists(p) or os.path.getsize(p) < 100:
                L(f"[novels] shard {s} dl failed"); 
                if os.path.exists(p): os.remove(p)
                continue
            data = open(p, "rb").read()
            last = data.rfind(b"\n")
            if last >= 0:
                data = data[:last]
            cnt = 0
            for line in data.decode("utf-8", errors="ignore").split("\n"):
                line = line.strip()
                if not line:
                    continue
                try:
                    o = json.loads(line)
                except Exception:
                    continue
                txt = (o.get("text") or "").strip()
                if len(txt) < 50:
                    continue
                f.write(json.dumps({"text": txt, "src": "webnovel:" + str(o.get("title", "?")), "cat": "novel"},
                                   ensure_ascii=False) + "\n")
                remaining -= len(txt.encode("utf-8"))
                n += 1; cnt += 1
            L(f"[novels] shard {s}: +{cnt} lines, remaining {remaining/1e6:.1f}MB")
            os.remove(p)
    L(f"[novels] -> {out}  {n} lines / {os.path.getsize(out)/1e6:.1f}MB")

# ---------- 公众号/社科/科普 (modern) ----------
def src_modern():
    out = os.path.join(RESERVE, "zh_r_modern.jsonl")
    if done(out):
        L(f"[modern] skip (exists {os.path.getsize(out)/1e6:.1f}MB)"); return
    if not DO_WEIXIN:
        L("[modern] DO_WEIXIN=0, skip (run separately)"); return
    d = os.path.join(TMP, "weixin_public_corpus")
    if not os.path.isdir(d):
        sh(f"git clone --depth 1 https://github.com/nonamestreet/weixin_public_corpus '{d}'")
    # 装 unzip(若缺)
    if sh("which unzip").returncode != 0:
        sh("apt-get install -y -q unzip 2>/dev/null || yum install -y -q unzip 2>/dev/null")
    zpath = os.path.join(d, "articles.zip")
    if os.path.exists(zpath) and sh("which unzip").returncode == 0:
        sh(f"cd '{d}' && unzip -o -q articles.zip -d ext")
        # 按字节上限增量写出(避免一次性入内存 + 精确截断)
        cap = MODERN_MB * 1e6
        got = 0
        n = 0
        with open(out, "w", encoding="utf-8") as f:
            for cand in ["ext/articles.jsonl", "articles.jsonl", "ext/articles.txt"]:
                fp = os.path.join(d, cand)
                if not os.path.exists(fp):
                    continue
                with open(fp, encoding="utf-8", errors="ignore") as fh:
                    for line in fh:
                        if got >= cap:
                            break
                        try:
                            o = json.loads(line)
                        except Exception:
                            continue
                        content = (o.get("content") or "")
                        title = o.get("title") or ""
                        name = o.get("name") or "?"
                        if len(content) < 80:
                            continue
                        t = (title + "\n" + content).strip()
                        f.write(json.dumps({"text": t, "src": "weixin:" + str(name), "cat": "modern"},
                                           ensure_ascii=False) + "\n")
                        got += len(t.encode("utf-8"))
                        n += 1
                break
        L(f"[modern] weixin -> {out}  {n} lines / {os.path.getsize(out)/1e6:.1f}MB")
    else:
        L("[modern] unzip missing or zip absent")

if __name__ == "__main__":
    L("===== build_reserve start @ " + time.strftime("%Y-%m-%d %H:%M")
      + f" (NOVEL_MB={NOVEL_MB}, MODERN_MB={MODERN_MB}, DO_WEIXIN={DO_WEIXIN}) =====")
    src_knowledge()
    src_ancient()
    src_novels()
    src_modern()
    L("===== reserve dir =====")
    for fn in sorted(os.listdir(RESERVE)):
        fp = os.path.join(RESERVE, fn)
        L(f"  {fn}: {os.path.getsize(fp)/1e6:.1f}MB")
    L("===== build_reserve done =====")
