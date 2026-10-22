# -*- coding: utf-8 -*-
"""vocab_calibrate.py — 用「咱们自己的语料」重新统计候选词频，替掉新闻腔先验

为什么必须校准(2026-10-19):
  vocab_extend.py 的词频先验来自白霜词库(新闻/政务语料统计)，
  排出来 top 全是「版权/房地产/证券/监督/改革/有关部门」——
  而我们的语料是小说+公众号+百科+教育，分布完全不同。
  直接照搬会让词表 80% 的槽位浪费在我永远不写的词上。

做法:
  抽样去重后的真实语料 -> 用合并词库做正向最大匹配分词 -> 统计实际词频
  收益 = 语料实际频次 × (旧词表下子词数 - 1)   ← 这才是真实省下的 token 数

用法:
  python tools/vocab_calibrate.py --chars 30000000 --n 30000
"""
import argparse
import json
import os
import pathlib
import random
import sys
import time
from collections import Counter

ROOT = pathlib.Path(__file__).resolve().parent.parent
LEX = ROOT / "lexicon"
TOK_JSON = ROOT / "backups" / "0.2B" / "r6_final_20261016" / "tokenizer.json"
OUT_DEFAULT = ROOT / "data" / "vocab_append_0.4B_calibrated.json"
MAXLEN = 8

# 唯一来源文件（排除 r2-r6 累积快照，避免重复计数）
CORPUS_FILES = [
    "zh_text_know_edu.jsonl",
    "zh_r_novels.jsonl",
    "zh_r_weixin.jsonl",
    "zh_text_lang.jsonl",
    "zh_text.jsonl",
    "zh_text_v2_ai.jsonl",
]


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
    w = {}

    def add(k, f):
        if not k or len(k) > MAXLEN or not is_cjk(k) or len(k) < 2:
            return
        if f <= 0:
            f = 1
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


def sample_corpus(target_chars, seed=42):
    """从唯一来源文件里分行抽样，凑够 target_chars 个汉字"""
    random.seed(seed)
    lines_by_file = []
    for name in CORPUS_FILES:
        p = ROOT / "pretrain_corpus" / name
        if not p.exists():
            continue
        with open(p, encoding="utf-8", errors="replace") as f:
            n = sum(1 for _ in f)
        lines_by_file.append((p, n))
        print("  %-26s %8d 行" % (name, n), flush=True)
    if not lines_by_file:
        sys.exit("[err] 没找到语料")

    total_lines = sum(n for _, n in lines_by_file)
    want = max(1, int(total_lines * target_chars / 3.0e8))
    per_file = max(1, want // len(lines_by_file))

    buf = []
    got = 0
    for p, n in lines_by_file:
        idx = set(random.sample(range(n), min(per_file, n)))
        with open(p, encoding="utf-8", errors="replace") as f:
            for i, ln in enumerate(f):
                if i not in idx:
                    continue
                try:
                    t = json.loads(ln).get("text") or json.loads(ln).get("content") or ""
                except Exception:
                    continue
                if t:
                    buf.append(t)
                    got += sum(1 for c in t if '\u4e00' <= c <= '\u9fff')
    return "".join(buf), got


def segment_count(text, dic, maxlen):
    """正向最大匹配，返回 Counter(词/字)"""
    cnt = Counter()
    i, L = 0, len(text)
    while i < L:
        hit = None
        for k in range(min(maxlen, L - i), 0, -1):
            cand = text[i:i + k]
            if cand in dic:
                hit = cand
                break
        if hit:
            cnt[hit] += 1
            i += len(hit)
        else:
            cnt[text[i]] += 1
            i += 1
    return cnt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chars", type=int, default=30000000)
    ap.add_argument("--n", type=int, default=30000)
    ap.add_argument("--out", default=str(OUT_DEFAULT))
    a = ap.parse_args()

    from tokenizers import Tokenizer
    tk = Tokenizer.from_file(str(TOK_JSON))

    old, old_raw = load_old_vocab()
    chars = load_chars()
    t0 = time.time()
    words = load_words()
    print("词库 %d 条  (%.1fs)" % (len(words), time.time() - t0), flush=True)

    print("抽样语料:", flush=True)
    text, n_cjk = sample_corpus(a.chars)
    print("抽样完成: 字符 %d  汉字 %d  (%.1fs)" % (len(text), n_cjk, time.time() - t0), flush=True)

    t1 = time.time()
    cnt = segment_count(text, words, MAXLEN)
    print("分词完成: 不同片段 %d  (%.1fs)" % (len(cnt), time.time() - t1), flush=True)

    # 单字候选（语料实际频次）
    char_cands = []
    for c, f in cnt.items():
        if len(c) != 1 or not is_cjk(c) or c in old:
            continue
        n_sub = len(tk.encode(c, add_special_tokens=False).ids)
        if n_sub <= 1:
            continue
        char_cands.append({"tok": c, "freq": f, "sub": n_sub, "gain": f * (n_sub - 1)})
    # 语料没覆盖但 8105 里的字（低频字）兜底追加，保证覆盖面
    covered = {c["tok"] for c in char_cands}
    for c in chars:
        if c in old or c in covered:
            continue
        n_sub = len(tk.encode(c, add_special_tokens=False).ids)
        if n_sub <= 1:
            continue
        char_cands.append({"tok": c, "freq": 0, "sub": n_sub, "gain": 0})
    char_cands.sort(key=lambda x: -x["gain"])

    # 词候选
    word_cands = []
    for w, f in cnt.items():
        if len(w) < 2 or not is_cjk(w) or w in old:
            continue
        n_sub = len(tk.encode(w, add_special_tokens=False).ids)
        if n_sub <= 1:
            continue
        word_cands.append({"tok": w, "freq": f, "sub": n_sub, "gain": f * (n_sub - 1)})
    word_cands.sort(key=lambda x: -x["gain"])
    top_words = word_cands[: a.n]

    # 预算实际节省
    save_chars = sum(c["gain"] for c in char_cands)
    save_words = sum(w["gain"] for w in top_words)
    base_tokens = len(tk.encode(text[:200000], add_special_tokens=False).ids)

    append, seen = [], set()
    for t in [c["tok"] for c in char_cands] + [w["tok"] for w in top_words]:
        if t not in seen:
            seen.add(t)
            append.append(t)

    report = {
        "sampled_cjk": n_cjk,
        "base_vocab": old_raw,
        "append_chars": len(char_cands),
        "append_words": len(top_words),
        "append_total": len(append),
        "new_vocab_size": old_raw + len(append),
        "hidden_size": 576,
        "param_delta_embed": 576 * len(append),
        "init_rule": "mean of old subword embeddings",
        "corpus_token_saving_chars": save_chars,
        "corpus_token_saving_words": save_words,
        "char_sample": [c["tok"] for c in char_cands[:30]],
        "word_sample": [w["tok"] for w in top_words[:40]],
        "top_words": [{"w": w["tok"], "f": w["freq"], "sub": w["sub"], "gain": w["gain"]}
                      for w in top_words[:200]],
        "append": append,
    }
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False)

    print("\n=== 结果（语料校准版）===")
    print("旧 %d -> 新 %d  (+%d)" % (old_raw, old_raw + len(append), len(append)))
    print("  单字 +%d   词 +%d" % (len(char_cands), len(top_words)))
    print("  抽样语料 %d 汉字: 字表可省 %d token, 词表可省 %d token" % (
        n_cjk, save_chars, save_words))
    print("  合计省 %.1f%% (估算)" % ((save_chars + save_words) / max(1, n_cjk * 1.55) * 100))
    print("  字样例:", "".join(report["char_sample"]))
    print("  词样例:", " / ".join(report["word_sample"]))
    print("  写出:", a.out)


if __name__ == "__main__":
    main()
