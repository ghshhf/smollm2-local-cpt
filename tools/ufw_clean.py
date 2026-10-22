#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ufw_clean.py — ufw_zh 网页壳噪声剥离 + 去重 + 剔繁体，输出分片

壳噪声结论(2026-10-19 audit): 13.67% 文档含壳话术, 但绝大多数只是首尾几行,
不能整篇删。这里做行级剥离 + 字符级清 URL/图片, 保留正文。

用法:
  python tools/ufw_clean.py --file corpus_04b/ufw_zh.jsonl \
      --out-dir corpus_04b/ufw_clean_parts --part-mb 1000
"""
import argparse
import hashlib
import json
import os
import re
import sys
import time

# 行级壳: 命中且该行长度 <= SHELL_MAXLEN 才整行剥掉(避免误伤正文长段落)
# ANY: 这些词组出现在行内任何位置都是壳话术
SHELL_ANY = re.compile(
    r"(点击上方|扫码关注|长按识别|长按二维码|识别二维码|扫描二维码|关注我们|关注公众号"
    r"|微信公众号|公众号\s*[:：]|本文来自|阅读原文|版权归|如有侵权|更多精彩"
    r"|上一篇|下一篇|相关阅读|投稿邮箱|回复关键词|拉你进|小编微信|加微信|微信号\s*[:：]"
    r"|转载请|商务合作|设为星标|点个.{0,4}在看)"
)
# HEAD: 必须行首(避免"数据来源:XX"这类正文被误删)
SHELL_HEAD = re.compile(
    r"^(来源|文章来源|内容来源|责任编辑|原标题|免责声明|广告|投稿|编辑\s*[:：])"
)
SHELL_MAXLEN = 60

# 字符级清: markdown 图片 / 裸 URL
MD_IMG = re.compile(r"!\[[^\]]{0,60}\]\([^)]{0,300}\)")
URL = re.compile(r"https?://[^\s，。；）)】\]]{4,300}")

PUNCT_ONLY = re.compile(r"^[\s\W_]+$", re.UNICODE)

PAIRS = [("说", "說"), ("为", "為"), ("来", "來"), ("与", "與"), ("无", "無"),
         ("见", "見"), ("个", "個"), ("这", "這"), ("时", "時"), ("国", "國")]


def clean_text(t):
    """返回清洗后文本(可能为空)"""
    # 字符级
    t = MD_IMG.sub("", t)
    t = URL.sub("", t)
    lines = t.split("\n")
    keep = []
    for ln in lines:
        s = ln.strip()
        if not s:
            keep.append("")
            continue
        # 纯符号/装饰行("▼" "——" "***")
        if len(s) <= 4 and PUNCT_ONLY.match(s):
            continue
        # 壳行
        if len(s) <= SHELL_MAXLEN and (SHELL_ANY.search(s) or SHELL_HEAD.match(s)):
            continue
        keep.append(s)
    out = "\n".join(keep)
    # 折叠 >=3 连续换行
    out = re.sub(r"\n{3,}", "\n\n", out).strip()
    return out


def is_trad(t):
    s = sum(t.count(x) for x, _ in PAIRS)
    tr = sum(t.count(y) for _, y in PAIRS)
    return tr > s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", default="corpus_04b/ufw_zh.jsonl")
    ap.add_argument("--out-dir", default="corpus_04b/ufw_clean_parts")
    ap.add_argument("--part-mb", type=int, default=1000)
    ap.add_argument("--min-chars", type=int, default=120, help="清洗后短于此丢")
    a = ap.parse_args()

    os.makedirs(a.out_dir, exist_ok=True)
    part_bytes = a.part_mb * 1024 * 1024

    n = kept = drop_short = drop_trad = drop_dup = 0
    chars_in = chars_out = 0
    lines_stripped = 0
    seen = set()
    idx = 0
    fout = None
    written = 0
    paths = []

    def open_part():
        nonlocal fout, idx, written
        if fout:
            fout.close()
        idx += 1
        p = os.path.join(a.out_dir, "ufw_zh_clean.part%02d.jsonl" % idx)
        paths.append(p)
        fout = open(p, "w", encoding="utf-8", newline="\n")
        written = 0
        return p

    open_part()
    t0 = time.time()
    total_bytes = os.path.getsize(a.file)
    with open(a.file, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            n += 1
            try:
                o = json.loads(line)
            except Exception:
                continue
            t = o.get("text") or o.get("content") or ""
            if not t:
                continue
            chars_in += len(t)
            c = clean_text(t)
            if len(c) < a.min_chars:
                drop_short += 1
                continue
            if is_trad(c):
                drop_trad += 1
                continue
            h = hashlib.md5(re.sub(r"\s+", "", c).encode("utf-8", "ignore")).hexdigest()
            if h in seen:
                drop_dup += 1
                continue
            seen.add(h)
            o["text"] = c
            o["clean"] = 1
            s = json.dumps(o, ensure_ascii=False) + "\n"
            b = len(s.encode("utf-8"))
            if written + b > part_bytes and written > 0:
                open_part()
            fout.write(s)
            written += b
            kept += 1
            chars_out += len(c)
            if n % 100000 == 0:
                print("  %d docs, kept %d, %.1fMB out, %.0fs"
                      % (n, kept, written / 1048576.0, time.time() - t0), flush=True)
    fout.close()

    st = {
        "in_docs": n, "kept": kept,
        "drop_short": drop_short, "drop_trad": drop_trad, "drop_dup": drop_dup,
        "chars_in": chars_in, "chars_out": chars_out,
        "char_keep_rate": round(chars_out / max(1, chars_in), 4),
        "parts": paths, "elapsed_s": round(time.time() - t0, 1),
    }
    with open(os.path.join(a.out_dir, "_clean_stats.json"), "w", encoding="utf-8") as g:
        json.dump(st, g, ensure_ascii=False, indent=1)
    print(json.dumps(st, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
