# -*- coding: utf-8 -*-
"""vocab_extend.py — 为 0.4B 生成「尾部 append」的中文词表候选

路线（docs/data-plan-0.4B.md 定稿口径）：
  旧 49,152 token 一行不动、id 全不变；只在尾部 append 新中文 token。
  加什么 = 8105 常用字 + 词频/收益 top N 词。
  新行 embedding = 该词在旧词表下各子词 embedding 的均值（另由 init 脚本做）。

本脚本只产出「候选清单 + 收益报告」，不动任何权重。
排序用真实收益分，不是裸词频：
    收益 = 词频 × (旧词表下该词的子词数 - 1)
因为一个高频但 1 token 就能拼出来的词，加进词表毫无意义。

用法:
  python tools/vocab_extend.py                 # 默认 N=30000
  python tools/vocab_extend.py --n 30000 --out data/vocab_append_0.4B.json
"""
import argparse
import json
import os
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
LEX = ROOT / "lexicon"
TOK_JSON = ROOT / "backups" / "0.2B" / "r6_final_20261016" / "tokenizer.json"
OUT_DEFAULT = ROOT / "data" / "vocab_append_0.4B.json"

MAXLEN = 8


def bytes_to_unicode():
    bs = list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, [chr(c) for c in cs]))


B2U = bytes_to_unicode()
U2B = {v: k for k, v in B2U.items()}


def decode_tok(s):
    try:
        return bytes([U2B[c] for c in s]).decode("utf-8", errors="replace")
    except Exception:
        return None


def is_cjk(s):
    return bool(s) and all('\u4e00' <= c <= '\u9fff' for c in s)


def load_old_vocab():
    d = json.loads(TOK_JSON.read_text(encoding="utf-8"))
    v = d["model"]["vocab"]
    out = {}
    for t, tid in v.items():
        r = decode_tok(t)
        if r and r not in out:
            out[r] = tid
    return out, len(v)


def load_chars():
    """8105 常用字表 -> {字: 字频}（只用 8105，不加生僻字）"""
    out = {}
    p = LEX / "rime_frost" / "8105.dict.yaml"
    started = False
    for ln in p.read_text(encoding="utf-8", errors="ignore").splitlines():
        if not started:
            if ln.strip() == "...":
                started = True
            continue
        if not ln or ln.startswith("#"):
            continue
        parts = ln.split("\t")
        if len(parts) >= 2 and parts[0]:
            f = int(parts[2]) if len(parts) >= 3 and parts[2].strip().isdigit() else 1
            if parts[0] not in out or f > out[parts[0]]:
                out[parts[0]] = f
    return out


def load_words():
    """合并 白霜 base/ext/tencent + THUOCL + jieba，取最大词频"""
    w = {}

    def add(k, f):
        if not k or len(k) > MAXLEN or not is_cjk(k) or len(k) < 2:
            return
        if k not in w or f > w[k]:
            w[k] = f

    def parse_rime(p):
        started = False
        for ln in p.read_text(encoding="utf-8", errors="ignore").splitlines():
            if not started:
                if ln.strip() == "...":
                    started = True
                continue
            if not ln or ln.startswith("#"):
                continue
            parts = ln.split("\t")
            if len(parts) >= 2 and parts[0]:
                f = int(parts[2]) if len(parts) >= 3 and parts[2].strip().isdigit() else 0
                add(parts[0], f)

    for f in ["base.dict.yaml", "ext.dict.yaml", "tencent.dict.yaml"]:
        p = LEX / "rime_frost" / f
        if p.exists():
            parse_rime(p)

    for f in sorted((LEX / "thuocl").glob("*.txt")):
        for ln in f.read_text(encoding="utf-8", errors="ignore").splitlines():
            parts = ln.split("\t")
            if parts and parts[0].strip():
                df = parts[1].strip() if len(parts) > 1 else "0"
                add(parts[0].strip(), int(df) if df.isdigit() else 0)

    jp = LEX / "jieba" / "dict.txt"
    if jp.exists():
        for ln in jp.read_text(encoding="utf-8", errors="ignore").splitlines():
            p = ln.split(" ")
            if len(p) >= 2 and p[1].isdigit():
                add(p[0], int(p[1]))
    return w


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=30000)
    ap.add_argument("--out", default=str(OUT_DEFAULT))
    a = ap.parse_args()

    from tokenizers import Tokenizer
    tk = Tokenizer.from_file(str(TOK_JSON))

    old, old_raw = load_old_vocab()
    print("旧词表: raw=%d  可解码=%d" % (old_raw, len(old)), flush=True)

    chars = load_chars()
    words = load_words()
    print("8105 字表: %d   合并词库(纯汉字2-8字): %d" % (len(chars), len(words)), flush=True)

    # ---------- 1) 单字 ----------
    char_cands = []
    n_char_dup = 0
    for c, f in sorted(chars.items(), key=lambda x: -x[1]):
        if c in old:
            n_char_dup += 1
            continue
        n_sub = len(tk.encode(c, add_special_tokens=False).ids)
        if n_sub <= 1:
            continue
        char_cands.append({"tok": c, "freq": f, "sub": n_sub, "gain": f * (n_sub - 1)})
    print("单字候选: %d  (已在旧词表 %d)" % (len(char_cands), n_char_dup), flush=True)

    # ---------- 2) 词 ----------
    pool = [(w, f) for w, f in words.items() if w not in old]
    pool.sort(key=lambda x: -x[1])
    pool = pool[:200000]
    print("待评词条(取词频前 20 万): %d" % len(pool), flush=True)

    word_cands = []
    for i, (w, f) in enumerate(pool):
        n_sub = len(tk.encode(w, add_special_tokens=False).ids)
        if n_sub <= 1:
            continue
        word_cands.append({"tok": w, "freq": f, "sub": n_sub, "gain": f * (n_sub - 1)})
        if (i + 1) % 50000 == 0:
            print("  ...%d" % (i + 1), flush=True)

    word_cands.sort(key=lambda x: -x["gain"])
    top_words = word_cands[: a.n]
    print("词候选: 可省token=%d  取 top %d" % (len(word_cands), len(top_words)), flush=True)

    # ---------- 3) 汇总 ----------
    append = [c["tok"] for c in char_cands] + [w["tok"] for w in top_words]
    seen, uniq = set(), []
    for t in append:
        if t not in seen:
            seen.add(t)
            uniq.append(t)

    new_size = old_raw + len(uniq)
    report = {
        "base_vocab": old_raw,
        "append_chars": len(char_cands),
        "append_words": len(top_words),
        "append_total": len(uniq),
        "new_vocab_size": new_size,
        "hidden_size": 576,
        "param_delta_embed": 576 * len(uniq),
        "param_delta_lmhead": 576 * len(uniq),
        "init_rule": "mean of old subword embeddings",
        "char_sample": [c["tok"] for c in char_cands[:20]],
        "word_sample": [w["tok"] for w in top_words[:30]],
        "append": uniq,
    }
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False)

    print("\n=== 结果 ===")
    print("旧 %d -> 新 %d  (+%d)" % (old_raw, new_size, len(uniq)))
    print("  单字 +%d   词 +%d" % (len(char_cands), len(top_words)))
    print("  embedding 增量 %.1fM  (lm_head 若 untied 再加 %.1fM)" % (
        576 * len(uniq) / 1e6, 576 * len(uniq) / 1e6))
    print("  字样例:", "".join(report["char_sample"]))
    print("  词样例:", " / ".join(report["word_sample"]))
    print("  写出:", a.out)


if __name__ == "__main__":
    main()
