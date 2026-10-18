#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
refine_novel.py — 网文语料精炼（填 0.4B「创意·文章」缺口）
设计原则：规则必须对应体检实测发现的问题，不做无依据的主观打分（避免空砍）

体检依据 (webnovel_9, 6万行样本):
  重复 0.65% / HTML 0% / URL 0.03% / 广告 0.01% / 成人征兆 4.50%
  短于100字 1.23% / 结尾截断 0.71% / 长度 med=3046 p99=7420 max=96507
  唯一书目 93 本 / 6万行 ⇒ 单书章节极多，必须做书目配额防霸占

规则:
  1. 硬过滤: 太短(<300字) / 异常长(>20000字) / 成人色情 / 广告推广 / HTML / URL
  2. 整篇 sha1 去重 (0.65%)
  3. 书目配额: 单书 token 上限 = 预算 × BOOK_CAP_FRAC，防单书灌满
  4. 流式累积到 BUDGET 即停（语料过剩，绝不全量处理）
输出: corpus_04b/creative_articles_novel.jsonl  (每行 {"text": ...})
"""
import os, sys, json, re, hashlib, collections, time

SRC = sys.argv[1] if len(sys.argv) > 1 else "data/novel/wn9.jsonl"
OUTDIR = "corpus_04b"
OUT = os.path.join(OUTDIR, "creative_articles_novel.jsonl")
BUDGET = int(os.environ.get("BUDGET", 240_000_000))   # 2.4亿 token
TOK_PER_HAN = 2.4589                                   # 经验值（与既有管线一致）
BOOK_CAP_FRAC = float(os.environ.get("BOOK_CAP_FRAC", 0.04))  # 单书最多占总预算 4%
MIN_CHARS = 300
MAX_CHARS = 20000

CJK = re.compile(r"[\u4e00-\u9fff]")
RE_HTML = re.compile(r"<[a-zA-Z/][^>]{0,60}>")
RE_URL = re.compile(r"https?://|www\.")
RE_AD = re.compile(r"(欢迎收藏|求推荐票|求月票|求订阅|本书首发|请关注微信|手机阅读|"
                   r"起点中文网|笔趣阁|小说免费阅读|请大家关注|投票支持|打赏)")
RE_ADULT = re.compile(r"(娇喘|呻吟|内裤|胸部|乳房|肉棒|阴道|做爱|淫水|淫荡|"
                      r"一丝不挂|赤裸|脱光|下体|私处|奸|嫖|色情)")

os.makedirs(OUTDIR, exist_ok=True)

t0 = time.time()
src_size = os.path.getsize(SRC)
book_cap = int(BUDGET * BOOK_CAP_FRAC)

stat = collections.Counter()
book_tok = collections.Counter()
seen = set()
kept_rows = 0
kept_tok = 0

print("源: %s (%.1f MB)" % (SRC, src_size / 1e6))
print("预算: %d token, 单书上限 %d token (%.0f%%), 长度窗口 %d-%d 字"
      % (BUDGET, book_cap, BOOK_CAP_FRAC * 100, MIN_CHARS, MAX_CHARS))
print("-" * 70)

fin = open(SRC, "r", encoding="utf-8", errors="replace")
fout = open(OUT, "w", encoding="utf-8")

for line in fin:
    stat["read"] += 1
    if kept_tok >= BUDGET:
        stat["stop_budget"] += 1
        break
    line = line.strip()
    if not line:
        stat["empty"] += 1
        continue
    try:
        r = json.loads(line)
    except Exception:
        stat["badjson"] += 1
        continue
    t = r.get("text") or ""
    if not isinstance(t, str):
        stat["notext"] += 1
        continue
    n = len(t)
    if n < MIN_CHARS:
        stat["too_short"] += 1
        continue
    if n > MAX_CHARS:
        stat["too_long"] += 1
        continue
    if RE_HTML.search(t):
        stat["html"] += 1
        continue
    if RE_URL.search(t):
        stat["url"] += 1
        continue
    if RE_AD.search(t):
        stat["ad"] += 1
        continue
    if RE_ADULT.search(t):
        stat["adult"] += 1
        continue
    h = hashlib.sha1(t[:400].encode("utf-8", "replace")).hexdigest()
    if h in seen:
        stat["dup"] += 1
        continue
    seen.add(h)
    # 书目配额
    title = str(r.get("title") or "_unknown")
    if book_tok[title] >= book_cap:
        stat["book_cap"] += 1
        continue
    tok = int(len(CJK.findall(t)) * TOK_PER_HAN)
    if tok <= 0:
        stat["nohan"] += 1
        continue
    book_tok[title] += tok
    fout.write(json.dumps({"text": t}, ensure_ascii=False) + "\n")
    kept_rows += 1
    kept_tok += tok
    if kept_rows % 10000 == 0:
        el = time.time() - t0
        print("  已选 %d 章 / %.2f 亿 token  (%.0fs, 读 %d 行)"
              % (kept_rows, kept_tok / 1e8, el, stat["read"]), flush=True)

fin.close()
fout.close()

print("-" * 70)
print("完成 %.1fs" % (time.time() - t0))
print("读入 %d 行，保留 %d 章 / %.3f 亿 token" % (stat["read"], kept_rows, kept_tok / 1e8))
print("剔除明细:")
for k, v in stat.most_common():
    if k not in ("read",):
        print("   %-12s %8d" % (k, v))
print("书目数 %d，top10 占比:" % len(book_tok))
for k, v in book_tok.most_common(10):
    print("   %-38s %10.0f token (%.1f%%)" % (k[:38], v, v * 100.0 / max(1, kept_tok)))
print("输出: %s (%.1f MB)" % (OUT, os.path.getsize(OUT) / 1e6))
