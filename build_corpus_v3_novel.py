# -*- coding: utf-8 -*-
"""构建下一轮小说语料池 v3-novel：古腾堡中文公版小说(46本) + vivym 网文数据集。
按用户 2026-10-09 裁定：小说为主，百科太笼统已弃用；只文字不编程。
输出: pretrain_corpus/zh_text_v3_novel.jsonl
"""
import os, re, json, glob, hashlib

ROOT = "D:/work/_local_ai/train_15b"
RAW = os.path.join(ROOT, "next_corpus_raw")
OUT = os.path.join(ROOT, "pretrain_corpus", "zh_text_v3_novel.jsonl")
V2 = os.path.join(ROOT, "pretrain_corpus", "zh_text_v2.jsonl")

cn = re.compile(r'[\u4e00-\u9fff]')
ws_re = re.compile(r'\s+')
sent_cut = re.compile(r'(?<=[。！？；\n])')
code_pat = re.compile(r'[{};<>=|&\\/\[\]`#$_]|def |import |return |http')

def norm(t):
    t = re.sub(r'\r', '', t)
    t = re.sub(r'[ \t\u3000]+', ' ', t)
    t = re.sub(r' {3,}\n', '\n', t)
    t = re.sub(r'\n{3,}', '\n\n', t)
    return t.strip()

def strip_gutenberg(t):
    m = re.search(r'\*\*\*\s*START OF (?:THE|THIS) PROJECT GUTENBERG EBOOK[^*]*\*\*\*(.*)', t, re.S)
    if m:
        t = m.group(1)
    m2 = re.search(r'^(.*?)\*\*\*\s*END OF (?:THE|THIS) PROJECT GUTENBERG EBOOK', t, re.S)
    if m2:
        t = m2.group(1)
    # 去 "Produced by" 行
    t = re.sub(r'Produced by [^\n]*\n?', '', t, count=3)
    return t

def is_codey(d):
    lines = [l for l in d.split('\n') if l.strip()]
    if not lines:
        return True
    cc = sum(len(code_pat.findall(l)) for l in lines)
    return cc / max(1, sum(len(l) for l in lines)) > 0.10

def split_docs(text, min_len=80, target=600, cap=1100):
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
                buf = buf + s if not buf else buf + s
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
        if len(cn.findall(d)) / max(1, len(d)) < 0.12:
            continue
        if is_codey(d):
            continue
        out.append(d)
    return out

def load_v2_hashes():
    seen = set()
    for line in open(V2, encoding="utf-8"):
        try:
            txt = json.loads(line).get("text", "")
        except Exception:
            continue
        seen.add(hashlib.md5(ws_re.sub('', txt).encode()).hexdigest())
    return seen

def emit(fout, seen, label, docs):
    kept = 0
    for d in docs:
        h = hashlib.md5(ws_re.sub('', d).encode()).hexdigest()
        if h in seen:
            continue
        seen.add(h)
        fout.write(json.dumps({"text": d}, ensure_ascii=False) + "\n")
        kept += 1
    print(f"  [{label}] +{kept:,}")
    return kept

def gutenberg_all(fout, seen):
    total = 0
    for fp in sorted(glob.glob(os.path.join(RAW, "gutenberg", "pg*.txt"))):
        t = open(fp, encoding="utf-8", errors="ignore").read()
        t = strip_gutenberg(t)
        total += emit(fout, seen, "g:" + os.path.basename(fp), split_docs(t))
    return total

def vivym_all(fout, seen, max_files=None):
    """vivym/fictions-chinese parquet -> 文本行"""
    import pyarrow.parquet as pq
    files = sorted(glob.glob(os.path.join(RAW, "vivym_*.parquet")))
    if max_files:
        files = files[:max_files]
    total = 0
    for fp in files:
        try:
            pf = pq.ParquetFile(fp)
        except Exception as e:
            print(f"  [skip corrupt] {os.path.basename(fp)}: {e}")
            continue
        cols = pf.schema_arrow.names
        # 自适应列: 找最像正文的列
        text_col = next((c for c in cols if c.lower() in ("text","content","body","novel","chapter_text","longtext")), cols[-1])
        n_batch = 0
        for b in pf.iter_batches(batch_size=500):
            vals = b.column(text_col).to_pylist()
            docs = []
            for v in vals:
                if v and len(v) > 80:
                    docs += split_docs(str(v))
            total += emit(fout, seen, f"vivym:{os.path.basename(fp)}#{n_batch}", docs)
            n_batch += 1
            if n_batch % 20 == 0:
                print(f"    ...{os.path.basename(fp)} batch {n_batch}, rows so far {pf.metadata.num_rows}")
    return total

def main():
    seen = load_v2_hashes()
    print(f"v2 去重基: {len(seen):,}")
    total_new, total_cn = 0, 0
    with open(OUT, "w", encoding="utf-8") as fout:
        total_new += gutenberg_all(fout, seen)
        total_new += vivym_all(fout, seen)
    n = sum(1 for _ in open(OUT, encoding="utf-8"))
    for line in open(OUT, encoding="utf-8"):
        total_cn += len(cn.findall(json.loads(line).get("text", "")))
    print(f"\nv3 小说池: {n:,} 篇 / 中文 {total_cn:,} 字 / token ≈ {total_cn/1.6:,.0f}")

if __name__ == "__main__":
    main()
