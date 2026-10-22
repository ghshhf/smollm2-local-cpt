#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""0.4B 词表清洗：只做一件事——剔除繁体/异体/非中文。

判据：单字必须在《通用规范汉字表》8105 字内（简体规范集）；
      多字词必须由 8105 内汉字组成。
用法：python tools/vocab_clean.py [--src data/vocab_append_0.4B_calibrated.json] [--out data/vocab_append_0.4B_final.json]
"""
import argparse
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DICT8105 = os.path.join(ROOT, "lexicon", "rime_frost", "8105.dict.yaml")


def load_8105():
    chars = set()
    started = False
    with open(DICT8105, encoding="utf-8", errors="ignore") as f:
        for ln in f:
            if not started:
                if ln.strip() == "...":
                    started = True
                continue
            ln = ln.rstrip("\n")
            if not ln or ln.startswith("#"):
                continue
            c = ln.split("\t")[0].strip()
            if len(c) == 1:
                chars.add(c)
    return chars


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=os.path.join(ROOT, "data", "vocab_append_0.4B_calibrated.json"))
    ap.add_argument("--out", default=os.path.join(ROOT, "data", "vocab_append_0.4B_final.json"))
    a = ap.parse_args()

    std = load_8105()
    if len(std) < 8000:
        print("[fatal] 8105 字表加载异常，只有 %d 字" % len(std))
        return 1

    d = json.load(open(a.src, encoding="utf-8"))
    src = d["append"]

    keep, drop_nonhan, drop_trad, drop_word_trad = [], [], [], []
    for t in src:
        if not t:
            continue
        # 非中文（含英文/数字/标点）
        if not all("\u4e00" <= c <= "\u9fff" for c in t):
            drop_nonhan.append(t)
            continue
        if len(t) == 1:
            if t in std:
                keep.append(t)
            else:
                drop_trad.append(t)
        else:
            bad = [c for c in t if c not in std]
            if bad:
                drop_word_trad.append((t, "".join(bad)))
            else:
                keep.append(t)

    # 保序去重
    seen = set()
    final = []
    for t in keep:
        if t in seen:
            continue
        seen.add(t)
        final.append(t)

    n_char = sum(1 for t in final if len(t) == 1)
    n_word = len(final) - n_char
    ratio = n_word / max(1, len(final))

    # 估算词节省（按原报告的每条均省 token 比例线性折算）
    def scaled(old):
        if not old:
            return 0
        return int(old * (len(final) / max(1, len(src))))

    out = dict(d)
    out.update({
        "append": final,
        "append_chars": n_char,
        "append_words": n_word,
        "append_total": len(final),
        "new_vocab_size": d["base_vocab"] + len(final),
        "param_delta_embed": (d["base_vocab"] + len(final) - d["base_vocab"]) * d["hidden_size"],
        "corpus_token_saving_chars": scaled(d.get("corpus_token_saving_chars")),
        "corpus_token_saving_words": scaled(d.get("corpus_token_saving_words")),
        "clean_rule": "single char in 8105; words composed only of 8105 chars; drop non-CJK",
        "clean_stats": {
            "src_total": len(src),
            "kept": len(final),
            "dropped_non_cjk": len(drop_nonhan),
            "dropped_trad_char": len(drop_trad),
            "dropped_word_with_trad": len(drop_word_trad),
            "word_ratio": round(ratio, 4),
        },
        "trad_char_sample": "".join(drop_trad[:200]),
        "word_trad_sample": ["%s(%s)" % (w, b) for w, b in drop_word_trad[:60]],
    })
    json.dump(out, open(a.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)

    print("源词表 %d 条  →  清洗后 %d 条" % (len(src), len(final)))
    print("  单字 %d（原 %d）  词 %d（原 %d）" % (n_char, d["append_chars"], n_word, d["append_words"]))
    print("  剔除：非中文 %d / 非8105单字 %d / 含非8105字的词 %d"
          % (len(drop_nonhan), len(drop_trad), len(drop_word_trad)))
    print("  新词表大小 %d（base %d）" % (out["new_vocab_size"], d["base_vocab"]))
    print("  非8105单字样例: %s" % "".join(drop_trad[:80]))
    print("  含非8105字的词样例: %s" % "、".join(w for w, _ in drop_word_trad[:20]))
    print("  → %s" % a.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
