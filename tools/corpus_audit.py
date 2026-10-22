# -*- coding: utf-8 -*-
"""0.4B 语料盘点：字节 / 行数 / 样例，按板块归类，识别递进版本重叠。"""
import os, json, glob, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def count_lines(path):
    n = 0
    with open(path, 'rb') as f:
        while True:
            b = f.read(1 << 22)
            if not b:
                break
            n += b.count(b'\n')
    return n

def sample(path, k=2):
    out = []
    try:
        with open(path, 'r', encoding='utf-8', errors='replace') as f:
            for i, line in enumerate(f):
                if i >= k:
                    break
                try:
                    o = json.loads(line)
                except Exception:
                    out.append(line[:120])
                    continue
                if isinstance(o, dict):
                    keys = list(o.keys())
                    txt = ''
                    for kk in ('text', 'content', 'code', 'completion', 'messages'):
                        if kk in o:
                            v = o[kk]
                            txt = v if isinstance(v, str) else json.dumps(v, ensure_ascii=False)
                            break
                    out.append({'keys': keys, 'len': len(txt), 'head': txt[:90].replace('\n', ' ')})
                else:
                    out.append(str(o)[:100])
    except Exception as e:
        out.append('ERR ' + str(e))
    return out

TARGETS = [
    ('pretrain_corpus', os.path.join(ROOT, 'pretrain_corpus')),
    ('corpus_04b', os.path.join(ROOT, 'corpus_04b')),
    ('next_corpus_raw', os.path.join(ROOT, 'next_corpus_raw')),
    ('data/l3_ms', os.path.join(ROOT, 'data', 'l3_ms')),
    ('node/l3_local', os.path.join(ROOT, 'node', 'l3_local')),
]

rows = []
for name, d in TARGETS:
    if not os.path.isdir(d):
        continue
    for p in sorted(glob.glob(os.path.join(d, '**', '*'), recursive=True)):
        if not os.path.isfile(p):
            continue
        ext = os.path.splitext(p)[1].lower()
        if ext not in ('.jsonl', '.parquet', '.json', '.txt'):
            continue
        sz = os.path.getsize(p)
        if sz < 1 << 20:
            continue
        rel = os.path.relpath(p, ROOT)
        n = None
        samp = None
        if ext == '.jsonl':
            n = count_lines(p)
            samp = sample(p)
        rows.append({'grp': name, 'file': rel, 'mb': round(sz / 1048576, 1), 'lines': n, 'sample': samp})
        print('AUDIT %-52s %8.1fMB %10s' % (rel, sz / 1048576, n), flush=True)

with open(os.path.join(ROOT, 'data', 'corpus_audit_04b.json'), 'w', encoding='utf-8') as f:
    json.dump(rows, f, ensure_ascii=False, indent=1)
print('DONE rows=%d total=%.2fGB' % (len(rows), sum(r['mb'] for r in rows) / 1024), flush=True)
