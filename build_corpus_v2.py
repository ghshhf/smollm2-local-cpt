#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
扩充中文持续预训练语料 (zh_text_v2.jsonl)
- 抓取所有本地中文文本源: 现有语料 / 小说库(未解压md + zip内md,txt) / D:/work 报告库 / _ref 知识库
- 去 HTML 标签 / 空白归一 / 切块 / 精确去重
- 支持增量: 若 v2 已存在, 以其哈希为去重基, 新源只追加不重复 (实现"不断补资料")
用法:
  python build_corpus_v2.py                # 全量构建(含现有语料)
  python build_corpus_v2.py --append       # 仅把 SOURCES 里"尚未入 v2"的部分追加
  python build_corpus_v2.py --only <glob>  # 只加某一类(调试)
"""
import argparse, os, re, sys, glob, zipfile, hashlib, json

ROOT = "D:/work/_local_ai/train_15b"
EXISTING = os.path.join(ROOT, "pretrain_corpus", "zh_text.jsonl")
V2 = os.path.join(ROOT, "pretrain_corpus", "zh_text_v2.jsonl")          # 主池(人写/外部)
V2_AI = os.path.join(ROOT, "pretrain_corpus", "zh_text_v2_ai.jsonl")    # AI 生成池(隔离, 不默认进训练)

NOVEL = "D:/work/novel"
WORK = "D:/work"
REF = "D:/work/_ref"

# D:/work 下排除(代码/二进制/模型/大资源)
EXCLUDE_DIRS = {"_scripts", "_local_ai", "_market_data", "_assets", ".git", "venv",
                "__pycache__", "node_modules", "_kb_raw", "tmp", "_handoff"}

cn = re.compile(r'[\u4e00-\u9fff]')
tag_re = re.compile(r'<[^>]+>')
script_re = re.compile(r'<(script|style)[^>]*>.*?</\1>', re.I | re.S)
ent_re = re.compile(r'&(?:#\d+|#x[0-9a-fA-F]+|[a-zA-Z]+);')
ws_re = re.compile(r'\s+')
sent_cut = re.compile(r'(?<=[。！？；\n])')

def strip_html(t):
    t = script_re.sub(' ', t)
    t = tag_re.sub(' ', t)
    t = ent_re.sub(' ', t)
    return t

def norm(t):
    t = ws_re.sub(' ', t).strip()
    return t

def split_docs(text, min_len=60, target=500, cap=1100, min_cn_ratio=0.12):
    """把一段文本切成 60~1100 字的文档, 中文占比需>min_cn_ratio(默认0.12, 保留代码/报告)"""
    # 先按段落
    paras = [p.strip() for p in re.split(r'\n\s*\n', text) if p.strip()]
    if not paras:
        paras = [text]
    docs = []
    buf = ""
    for p in paras:
        # 超长单段按句切
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
        if c / max(1, len(d)) < min_cn_ratio:
            continue
        out.append(d)
    return out

def read_text_file(fp):
    try:
        t = open(fp, encoding="utf-8", errors="ignore").read()
    except Exception:
        return ""
    if fp.lower().endswith((".html", ".htm")):
        t = strip_html(t)
    return t

def harvest_file(fp):
    t = read_text_file(fp)
    return split_docs(t)

def harvest_zip(zp):
    docs = []
    try:
        zf = zipfile.ZipFile(zp)
    except Exception:
        return docs
    for name in zf.namelist():
        if name.lower().endswith((".md", ".txt")):
            try:
                t = zf.read(name).decode("utf-8", "ignore")
            except Exception:
                continue
            docs += split_docs(t)
        elif name.lower().endswith((".html", ".htm")):
            try:
                t = zf.read(name).decode("utf-8", "ignore")
            except Exception:
                continue
            docs += split_docs(strip_html(t))
    return docs

def collect_sources(only=None):
    srcs = []  # (label, doc_list)
    # 1) 现有语料: 原样保留(已是人审训练数据, 不过滤/不重切, 避免丢掉代码卡等)
    if not only or only == "existing":
        if os.path.isfile(EXISTING):
            docs = []
            for line in open(EXISTING, encoding="utf-8"):
                line = line.strip()
                if not line:
                    continue
                try:
                    d = json.loads(line)
                    txt = d.get("text") or d.get("output") or d.get("content") or ""
                except Exception:
                    txt = line
                if txt and len(txt) >= 20:
                    docs.append(txt)
            srcs.append(("existing", docs))
    # 1.5) 外部人写原始小说 txt(网盘搬运的书, 非本机 AI 产物) -> 主池
    if not only or only == "extnovel":
        ext = []
        chap_re = re.compile(r'第[0-9一二三四五六七八九十百千]+章')
        for fp in (os.path.join(ROOT, "shennulv_raw.txt"),
                   os.path.join(NOVEL, "novel_tokyo_412.txt")):
            if os.path.isfile(fp):
                t = read_text_file(fp)
                m = chap_re.search(t)
                if m:  # 丢掉网盘制作说明等元数据前言, 从第一章开始
                    t = t[m.start():]
                ext += split_docs(t, min_len=100, target=900)
        srcs.append(("ext_novel", ext))
    # 2) 小说库 未解压 md
    if not only or only == "novel":
        docs = []
        for fp in glob.glob(os.path.join(NOVEL, "**", "*.md"), recursive=True):
            docs += harvest_file(fp)
        srcs.append(("novel_md", docs))
        # zip 内
        zdocs = []
        for zp in glob.glob(os.path.join(NOVEL, "**", "*.zip"), recursive=True):
            zdocs += harvest_zip(zp)
        srcs.append(("novel_zip", zdocs))
    # 3) D:/work 报告库
    if not only or only == "work":
        docs = []
        for ext in ("*.md", "*.txt", "*.html"):
            for fp in glob.glob(os.path.join(WORK, "**", ext), recursive=True):
                # 排除目录
                parts = set(os.path.normpath(fp).split(os.sep))
                if parts & EXCLUDE_DIRS:
                    continue
                if fp.lower().endswith(".zip"):
                    continue
                docs += harvest_file(fp)
        srcs.append(("work_reports", docs))
    # 4) _ref
    if not only or only == "ref":
        docs = []
        for ext in ("*.md", "*.txt", "*.html"):
            for fp in glob.glob(os.path.join(REF, "**", ext), recursive=True):
                docs += harvest_file(fp)
        srcs.append(("ref", docs))
    return srcs

# 来源归属(2026-10-09 用户提醒: 本地很多书是 AI 生成的, 不能混灌):
#   人写/外部池(主): existing=已训过的既有语料(含外部网文影川)
#   AI 池(隔离): novel_*=D:/work/novel 本机 AI 流水线产物(gen_novel/llama_novel 等)
#                work_reports=D:/work 报告库(AI 撰写)  ref=AI 写的扫描摘要
AI_SOURCES = {"novel_md", "novel_zip", "work_reports", "ref"}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--append", action="store_true", help="仅追加尚未入 v2 的部分")
    ap.add_argument("--only", default="", help="只加某类: existing/novel/work/ref")
    args = ap.parse_args()

    seen = set()
    for vp in (V2, V2_AI):
        if not os.path.isfile(vp):
            continue
        for line in open(vp, encoding="utf-8"):
            line = line.strip()
            if not line:
                continue
            try:
                txt = json.loads(line).get("text", "")
            except Exception:
                txt = line
            seen.add(hashlib.md5(ws_re.sub('', txt).encode("utf-8")).hexdigest())

    srcs = collect_sources(only=args.only or None)
    # 全量构建时重写(带池归属); append 时追加
    base_exists = os.path.isfile(V2) or os.path.isfile(V2_AI)
    mode = "a" if (args.append or (args.only and base_exists)) else "w"
    n_main = 0; n_ai = 0; chars_main = 0; chars_ai = 0
    with open(V2, mode, encoding="utf-8") as fm, open(V2_AI, mode, encoding="utf-8") as fa:
        for label, docs in srcs:
            fout = fa if label in AI_SOURCES else fm
            kept = 0
            for d in docs:
                h = hashlib.md5(ws_re.sub('', d).encode("utf-8")).hexdigest()
                if h in seen:
                    continue
                seen.add(h)
                fout.write(json.dumps({"text": d, "src": label}, ensure_ascii=False) + "\n")
                kept += 1
                if label in AI_SOURCES:
                    n_ai += 1; chars_ai += len(d)
                else:
                    n_main += 1; chars_main += len(d)
            print(f"  [{label}] {'AI池' if label in AI_SOURCES else '主池'} 产出 {kept:,} 篇")

    def stat(fp):
        if not os.path.isfile(fp):
            return 0, 0
        n = 0; cn_c = 0
        for line in open(fp, encoding="utf-8"):
            try:
                t = json.loads(line).get("text", "")
            except Exception:
                continue
            n += 1; cn_c += len(cn.findall(t))
        return n, cn_c
    for name, fp in (("主池", V2), ("AI池", V2_AI)):
        n, cn_c = stat(fp)
        print(f"{name}: 文档 {n:,}  中文字 {cn_c:,}  估算token {cn_c/1.6:,.0f}")

if __name__ == "__main__":
    main()
