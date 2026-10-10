# -*- coding: utf-8 -*-
"""构建下一轮"基础教育"语料 v3：公版经典文学 + 中文维基百科。
原则(用户 2026-10-09 裁定)：只喂文字，不教编程；全部外部人写/编审来源。
输出: pretrain_corpus/zh_text_v3_edu.jsonl（仅新教材，与 v2 主池在训练时合并）
"""
import os, re, json, glob, hashlib, sys

ROOT = "D:/work/_local_ai/train_15b"
RAW = os.path.join(ROOT, "next_corpus_raw")
OUT = os.path.join(ROOT, "pretrain_corpus", "zh_text_v3_edu.jsonl")
V2 = os.path.join(ROOT, "pretrain_corpus", "zh_text_v2.jsonl")

cn = re.compile(r'[\u4e00-\u9fff]')
ws_re = re.compile(r'\s+')
sent_cut = re.compile(r'(?<=[。！？；\n])')

def norm(t):
    t = re.sub(r'\r', '', t)
    t = re.sub(r'[ \t\u3000]+', ' ', t)
    t = re.sub(r' {3,}\n', '\n', t)
    t = re.sub(r'\n{3,}', '\n\n', t)
    return t.strip()

def strip_gutenberg(t):
    # 去头尾 License/Release 块
    m = re.search(r'\*\*\*\s*START OF (?:THE|THIS) PROJECT GUTENBERG EBOOK[^*]*\*\*\*(.*)', t, re.S)
    if m:
        t = m.group(1)
    m2 = re.search(r'^(.*?)\*\*\*\s*END OF (?:THE|THIS) PROJECT GUTENBERG EBOOK', t, re.S)
    if m2:
        t = m2.group(1)
    return t

def is_codey(d):
    """编程/数据行占比过高则剔除(基础教育不含编程, 用户裁定)"""
    lines = [l for l in d.split('\n') if l.strip()]
    if not lines:
        return True
    code_ch = sum(len(re.findall(r'[{};<>=|&\\/\[\]`#$_]|def |import |return |http', l)) for l in lines)
    total = sum(len(l) for l in lines)
    return code_ch / max(1, total) > 0.10

def split_docs(text, min_len=60, target=500, cap=1100):
    paras = [p.strip() for p in re.split(r'\n\s*\n', text) if p.strip()]
    if not paras:
        paras = [text]
    docs, buf = [], ""
    for p in paras:
        if len(p) > cap:
            for s in sent_cut.split(p):
                s = s.strip()
                if not s:
                    continue
                if len(buf) + len(s) > target and buf:
                    docs.append(buf); buf = ""
                buf = (buf + s) if not buf else (buf + s)
                if len(buf) >= cap:
                    docs.append(buf); buf = ""
        else:
            if len(buf) + len(p) > cap and buf:
                docs.append(buf); buf = ""
            buf = (buf + "\n" + p) if buf else p
    if buf:
        docs.append(buf)
    out = []
    for d in docs:
        d = norm(d)
        if len(d) < min_len:
            continue
        c = len(cn.findall(d))
        if c / max(1, len(d)) < 0.12:   # 与 v2 一致, 保留低中文比(题名等)但滤英文
            continue
        if is_codey(d):
            continue
        out.append(d)
    return out

def load_v2_hashes():
    seen = set()
    if os.path.isfile(V2):
        for line in open(V2, encoding="utf-8"):
            try:
                txt = json.loads(line).get("text", "")
            except Exception:
                continue
            seen.add(hashlib.md5(ws_re.sub('', txt).encode()).hexdigest())
    return seen

def gutenberg_docs():
    out = []
    for fp in sorted(glob.glob(os.path.join(RAW, "gutenberg", "pg*.txt"))):
        t = open(fp, encoding="utf-8", errors="ignore").read()
        t = strip_gutenberg(t)
        docs = split_docs(t, min_len=80, target=600)
        out.append(("gutenberg:" + os.path.basename(fp), docs))
    return out

def wiki_docs(max_articles=120000):
    import pyarrow.parquet as pq
    fp = os.path.join(RAW, "wiki_zh_shard2.parquet")
    pf = pq.ParquetFile(fp)
    n = 0
    for batch in pf.iter_batches(batch_size=2000, columns=["title", "text"]):
        titles = batch.column("title").to_pylist()
        texts = batch.column("text").to_pylist()
        for ti, te in zip(titles, texts):
            n += 1
            if n > max_articles:
                return [("wiki", acc)]
            te = te or ""
            # 清掉维基内部残留
            te = re.sub(r'\n=[^=\n]+=\\n', '\n', te)
            yield_docs = split_docs(f"{ti}\n{te}" if ti else te, min_len=80, target=600)
            yield ("wiki", yield_docs)
    return

def main():
    seen = load_v2_hashes()
    print(f"v2 去重基: {len(seen):,} 条")
    total_new, total_cn = 0, 0
    with open(OUT, "w", encoding="utf-8") as fout:
        def emit(label, docs):
            nonlocal total_new, total_cn
            kept = 0
            for d in docs:
                h = hashlib.md5(ws_re.sub('', d).encode()).hexdigest()
                if h in seen:
                    continue
                seen.add(h)
                fout.write(json.dumps({"text": d}, ensure_ascii=False) + "\n")
                kept += 1; total_new += 1; total_cn += len(cn.findall(d))
            print(f"  [{label}] 新增 {kept:,} 篇")
        for label, docs in gutenberg_docs():
            emit(label, docs)
        # 维基用生成器逐批处理, 防内存爆
        for label, docs in wiki_docs():
            emit(label, docs)
    n = sum(1 for _ in open(OUT, encoding="utf-8"))
    print(f"\nv3 教育池: {n:,} 篇 / 新增中文 {total_cn:,} 字 / 估算 token {total_cn/1.6:,.0f}")
    print(f"输出: {OUT}")

if __name__ == "__main__":
    main()
