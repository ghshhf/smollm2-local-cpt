#!/usr/bin/env python3
# build_owner_corpus.py — 从本地记忆文件抽取「主人风格语料」(owner enrollment corpus)
# 纯标准库, 不依赖任何第三方包。
# 用途: 为身份嵌入训练(Phase 0)准备注册语料 —— 抽用户原话/「」引用/用户轮次, 去重, 产 jsonl。
import os, re, json, hashlib
from collections import Counter

USER_MEM = "C:/Users/admin/.workbuddy/user-efd4ce18-4e0b-4417-99bd-5f49bb0def18-personal/MEMORY.md"
PROJ_MEM_DIR = "D:/.workbuddy/memory"
OUT_DIR = "D:/work/_local_ai/train_15b/owner_corpus"
OUT = os.path.join(OUT_DIR, "owner_corpus.jsonl")

# 匹配: 用户原话(带引号) / 「」引用 / daily log 用户轮次
re_yuanhua = re.compile(r'用户原话[：:]\s*[“"](.+?)[”"]')
re_corner = re.compile(r'[「『]([^」』]{4,})[」』]')
re_user_turn = re.compile(r'用户[：:]\s*["“]([^"”]{4,})["”]')
MIN_LEN = 6

def clean(s):
    return s.strip().strip('"“”‘’').strip()

def extract_from_file(path, src):
    out = []
    try:
        with open(path, encoding='utf-8') as f:
            lines = f.readlines()
    except FileNotFoundError:
        return out
    for ln in lines:
        m = re_yuanhua.search(ln)
        if m:
            t = clean(m.group(1))
            if len(t) >= MIN_LEN:
                out.append((t, src, "user_quote"))
            continue
        for mm in re_corner.finditer(ln):
            t = clean(mm.group(1))
            if len(t) >= MIN_LEN:
                out.append((t, src, "corner_quote"))
        for mm in re_user_turn.finditer(ln):
            t = clean(mm.group(1))
            if len(t) >= MIN_LEN:
                out.append((t, src, "user_turn"))
    return out

def main():
    items = []
    items += extract_from_file(USER_MEM, "user_MEMORY.md")
    if os.path.isdir(PROJ_MEM_DIR):
        for fn in sorted(os.listdir(PROJ_MEM_DIR)):
            if fn.endswith('.md') and fn != 'MEMORY.md':  # 项目 daily log 为主; 索引文件跳过(已下沉指针)
                items += extract_from_file(os.path.join(PROJ_MEM_DIR, fn), "proj_" + fn)
    seen, uniq = set(), []
    for t, src, kind in items:
        h = hashlib.sha1(t.encode('utf-8')).hexdigest()
        if h in seen:
            continue
        seen.add(h)
        uniq.append({"text": t, "src": src, "kind": kind})
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(OUT, 'w', encoding='utf-8') as f:
        for r in uniq:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"提取主人语料 {len(uniq)} 条 -> {OUT}")
    print("按类型:", dict(Counter(r['kind'] for r in uniq)))
    print("按源:", dict(Counter(r['src'] for r in uniq)))

if __name__ == '__main__':
    main()
