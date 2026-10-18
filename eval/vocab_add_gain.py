# -*- coding: utf-8 -*-
"""
量化「加词表」的收益：
  1) 现有 SmolLM2(49k, 字节级 BPE) 词表里到底有多少单个汉字已是独立 token
  2) 追加 N 个中文词后，同一批中文文本的 token/字 能降多少

只做测量，不改任何权重、不动任何训练。
用法: venv/Scripts/python.exe eval/vocab_add_gain.py [N]
"""
import json, sys, pathlib

ROOT = pathlib.Path(__file__).resolve().parent.parent
TOK  = ROOT / "backups" / "R4_weights" / "tokenizer.json"
LEX  = ROOT / "lexicon"
SAMPLE_CHARS = 300_000


def bytes_to_unicode():
    """GPT-2 字节级 BPE 的 byte<->unicode 映射（SmolLM2 同款）"""
    bs = list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b); cs.append(256 + n); n += 1
    return dict(zip(bs, [chr(c) for c in cs]))


B2U = bytes_to_unicode()
U2B = {v: k for k, v in B2U.items()}


def decode_tok(s):
    """把词表里的字节映射串还原成真实文本"""
    try:
        return bytes([U2B[c] for c in s]).decode("utf-8", errors="replace")
    except Exception:
        return None


def is_cjk(s):
    return s and all('\u4e00' <= c <= '\u9fff' for c in s)


def load_vocab():
    d = json.loads(TOK.read_text(encoding="utf-8"))
    v = d["model"]["vocab"] if "model" in d else d["vocab"]
    return {decode_tok(t): tid for t, tid in v.items()}


def load_chars():
    """白霜 8105 常用字表 / 41448 大字表 -> {字: 字频}"""
    out = {}
    def parse_rime(p):
        started = False
        for ln in p.read_text(encoding="utf-8", errors="ignore").splitlines():
            if not started:
                if ln.strip() == "...": started = True
                continue
            if not ln or ln.startswith("#"): continue
            parts = ln.split("\t")
            if len(parts) >= 2 and parts[0]:
                f = int(parts[2]) if len(parts) >= 3 and parts[2].strip().isdigit() else 1
                if parts[0] not in out or f > out[parts[0]]: out[parts[0]] = f
    for f in ["8105.dict.yaml", "41448.dict.yaml", "GB18030-2022.dict.yaml"]:
        p = LEX / "rime_frost" / f
        if p.exists(): parse_rime(p)
    return out


def load_lexicon():
    w = {}
    def parse_rime(p):
        started = False
        for ln in p.read_text(encoding="utf-8", errors="ignore").splitlines():
            if not started:
                if ln.strip() == "...": started = True
                continue
            if not ln or ln.startswith("#"): continue
            parts = ln.split("\t")
            if len(parts) >= 2 and parts[0]:
                f = int(parts[2]) if len(parts) >= 3 and parts[2].strip().isdigit() else 1
                if parts[0] not in w or f > w[parts[0]]: w[parts[0]] = f
    for f in ["base.dict.yaml", "ext.dict.yaml", "tencent.dict.yaml"]:
        p = LEX / "rime_frost" / f
        if p.exists(): parse_rime(p)
    for f in sorted((LEX / "thuocl").glob("*.txt")):
        for ln in f.read_text(encoding="utf-8", errors="ignore").splitlines():
            parts = ln.split("\t")
            if parts and parts[0].strip():
                k = parts[0].strip()
                if k not in w: w[k] = 1
    jp = LEX / "jieba" / "dict.txt"
    if jp.exists():
        for ln in jp.read_text(encoding="utf-8", errors="ignore").splitlines():
            p = ln.split(" ")
            if p and p[0] and p[0] not in w: w[p[0]] = 1
    return w


def sample_text():
    for name in ["zh_text_know_edu.jsonl", "zh_text_know_full.jsonl", "zh_text_lang.jsonl"]:
        p = ROOT / "pretrain_corpus" / name
        if not p.exists(): continue
        buf, n = [], 0
        with p.open(encoding="utf-8", errors="ignore") as fh:
            for ln in fh:
                try: t = json.loads(ln).get("text", "")
                except Exception: t = ""
                if t: buf.append(t); n += len(t)
                if n >= SAMPLE_CHARS: break
        if buf: return "".join(buf)
    return ""


def segment(text, dic, maxlen):
    """贪心最长匹配分词，返回 [(片段, 是否命中词表)]"""
    out, i, L = [], 0, len(text)
    while i < L:
        hit = None
        for k in range(min(maxlen, L - i), 0, -1):
            if text[i:i + k] in dic:
                hit = text[i:i + k]; break
        if hit:
            out.append((hit, True)); i += len(hit)
        else:
            out.append((text[i], False)); i += 1
    return out


def main():
    vocab = load_vocab()          # 真实文本 -> token id
    vocab = {k: v for k, v in vocab.items() if k}
    print("=== 现有词表 (SmolLM2 49k, 字节级 BPE 已解码) ===")
    print("词表大小:", len(vocab))
    cjk1 = [t for t in vocab if len(t) == 1 and is_cjk(t)]
    print(f"已是独立 token 的单汉字: {len(cjk1)}")
    print(f"已是独立 token 的汉字词(>=2字): {len([t for t in vocab if len(t) > 1 and is_cjk(t)])}")

    from tokenizers import Tokenizer
    tk = Tokenizer.from_file(str(TOK))

    text = sample_text()
    if not text:
        print("!! 没取到语料样本"); return
    n_cjk = sum(1 for c in text if '\u4e00' <= c <= '\u9fff')
    print(f"\n=== 语料样本 ===\n总字符 {len(text):,}  其中汉字 {n_cjk:,}")

    base = len(tk.encode(text, add_special_tokens=False).ids)
    print(f"\n=== 基线（真实 tokenizer）===")
    print(f"token 数 {base:,}  →  token/汉字 = {base/n_cjk:.4f}")

    lex = load_lexicon()
    chars = load_chars()
    print(f"\n=== 候选词库 ===\n词条 {len(lex):,}   字表 {len(chars):,}")
    N = int(sys.argv[1]) if len(sys.argv) > 1 else 30000
    top_words = [k for k, _ in sorted(lex.items(), key=lambda x: -x[1])
                 if is_cjk(k) and len(k) >= 2][:N]

    def evaluate(name, cands):
        added = [k for k in cands if k not in vocab]      # ← 天然去重
        dic = set(added)
        maxlen = max((len(w) for w in added), default=2)
        cache = {}
        new = 0
        for s, hit in segment(text, dic, maxlen):
            if hit:
                new += 1
            else:
                if s not in cache:
                    cache[s] = len(tk.encode(s, add_special_tokens=False).ids)
                new += cache[s]
        gain = (1 - new / base) * 100
        print(f"{name:26s} 加 {len(added):>6,}  |  token/汉字 {new/n_cjk:6.4f}  |  "
              f"省 {gain:5.1f}%  |  1024 可多装 {(base/new-1)*100:5.0f}%  |  "
              f"+{576*len(added)/1e6:5.1f}M 参数")
        return len(added), new / n_cjk, gain

    print(f"\n=== 三种加法对比（基线 token/汉字 = {base/n_cjk:.4f}）===")
    evaluate("A 只加 8105 常用字", [c for c in sorted(chars, key=lambda x: -chars[x])[:8105]])
    evaluate("B 只加词频 top %d 词" % N, top_words)
    ch = [c for c in sorted(chars, key=lambda x: -chars[x])[:8105]]
    evaluate("C 字表 + 词 top %d" % N, ch + top_words)
    ch2 = [c for c in sorted(chars, key=lambda x: -chars[x])[:8105]]
    evaluate("D 字表 + 词 top 60000",
             ch2 + [k for k, _ in sorted(lex.items(), key=lambda x: -x[1])
                    if is_cjk(k) and len(k) >= 2][:60000])


if __name__ == "__main__":
    main()
