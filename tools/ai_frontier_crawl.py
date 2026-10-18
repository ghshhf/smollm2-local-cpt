#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ai_frontier_crawl.py — AI 前沿文章爬虫（每日增量）
源（全部官方 API，无需登录）:
  1. arXiv API    cs.AI / cs.LG / cs.CL / cs.CV 各 30 篇最新
  2. HF Papers    huggingface.co/api/papers 每日热榜 30 篇
输出:
  data/frontier/YYYY-MM-DD.jsonl   当日全量（含 id/标题/摘要/链接/来源/分类）
  data/frontier/index.md           累计索引（按日期倒序，附标题）
用法:
  python tools/ai_frontier_crawl.py            # 抓今天
  python tools/ai_frontier_crawl.py --days 3   # 补抓近3天
"""
import os, sys, json, time, re, argparse
import urllib.request
import ssl
import xml.etree.ElementTree as ET

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(ROOT, "data", "frontier")
PROXY = "http://127.0.0.1:3067"   # Karing 规则代理（国外走代理）
CTX = ssl.create_default_context()
CTX.check_hostname = False
CTX.verify_mode = ssl.CERT_NONE   # 代理 HTTPS 证书链不完整，关校验

ARXIV_CATS = ["cs.AI", "cs.LG", "cs.CL", "cs.CV"]
SEEN_FILE = os.path.join(OUT_DIR, "_seen_ids.json")  # 跨日去重账本：历史已入库的规范 id


def norm_id(aid):
    """arXiv id 规范化：2610.22364v1 -> 2610.22364（版本号不算新论文）"""
    return re.sub(r"v\d+$", "", str(aid or "").strip())


def load_seen():
    if os.path.exists(SEEN_FILE):
        try:
            with open(SEEN_FILE, encoding="utf-8") as f:
                return set(json.load(f))
        except Exception:
            pass
    # 冷启动：从历史 jsonl 建账本（老数据不算新）
    seen = set()
    for fn in os.listdir(OUT_DIR):
        if not fn.endswith(".jsonl"):
            continue
        with open(os.path.join(OUT_DIR, fn), encoding="utf-8") as f:
            for l in f:
                l = l.strip()
                if not l:
                    continue
                try:
                    seen.add(norm_id(json.loads(l).get("id")))
                except Exception:
                    pass
    return seen


def save_seen(seen):
    with open(SEEN_FILE, "w", encoding="utf-8") as f:
        json.dump(sorted(seen), f)


def fetch(url, timeout=40, use_proxy=True):
    opener = urllib.request.build_opener(
        urllib.request.ProxyHandler({"http": PROXY, "https": PROXY}) if use_proxy else
        urllib.request.ProxyHandler({}),
        urllib.request.HTTPSHandler(context=CTX))
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 frontier-crawl"})
    return opener.open(req, timeout=timeout).read()


def crawl_arxiv(cat, n=30):
    """arXiv 官方 API"""
    url = ("http://export.arxiv.org/api/query?search_query=cat:%s"
           "&sortBy=submittedDate&sortOrder=descending&max_results=%d" % (cat, n))
    try:
        xml = fetch(url)
    except Exception as e:
        print("[arxiv:%s] 失败 %s" % (cat, e)); return []
    ns = {"a": "http://www.w3.org/2005/Atom"}
    out = []
    try:
        root = ET.fromstring(xml)
        for e in root.findall("a:entry", ns):
            title = re.sub(r"\s+", " ", e.find("a:title", ns).text).strip()
            summ = re.sub(r"\s+", " ", e.find("a:summary", ns).text).strip()
            link = e.find("a:id", ns).text.strip()
            pub = e.find("a:published", ns).text[:10]
            out.append({"src": "arxiv", "cat": cat, "id": link.split("/abs/")[-1],
                        "title": title, "abstract": summ[:1200],
                        "url": link, "date": pub})
    except Exception as e:
        print("[arxiv:%s] 解析失败 %s" % (cat, e))
    print("[arxiv:%s] %d 篇" % (cat, len(out)))
    return out


def crawl_hf_papers(n=30):
    """HF Daily Papers"""
    try:
        raw = fetch("https://huggingface.co/api/papers?limit=%d" % n)
        d = json.loads(raw)
    except Exception as e:
        print("[hf-papers] 失败 %s" % e); return []
    out = []
    for r in d:
        p = r.get("paper", r)
        out.append({"src": "hf-papers", "cat": "daily", "id": p.get("id", ""),
                    "title": (p.get("title") or "").strip(),
                    "abstract": (p.get("summary") or "")[:1200],
                    "url": "https://huggingface.co/papers/%s" % p.get("id", ""),
                    "date": (p.get("publishedAt") or "")[:10],
                    "upvotes": r.get("upvotes", 0)})
    print("[hf-papers] %d 篇" % len(out))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=1)
    args = ap.parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)

    all_items = []
    for cat in ARXIV_CATS:
        all_items += crawl_arxiv(cat, 30)
        time.sleep(1)
    all_items += crawl_hf_papers(30)

    # 按 arXiv id / hf id 去重（本轮内）
    seen, uniq = set(), []
    for it in all_items:
        k = (it["src"], norm_id(it["id"]))
        if k in seen:
            continue
        seen.add(k); uniq.append(it)

    # 跨日去重：历史账本里已有的论文不再入库（核心修复——
    # 旧版只做同日合并，新的一天会把 arXiv"最新30篇"里的老论文整批重收，
    # 实测 10-15 与 10-17 重复 70 篇）
    ledger = load_seen()
    fresh = [it for it in uniq if norm_id(it["id"]) not in ledger]
    for it in uniq:
        ledger.add(norm_id(it["id"]))
    save_seen(ledger)
    print("[dedup] 本轮抓取 %d 篇，跨日去重后新增 %d 篇" % (len(uniq), len(fresh)))
    uniq = fresh

    if args.days <= 1:
        # 日常增量：按抓取日期归档，与同日已有文件合并去重（一天可跑多次）
        # 修 bug：曾按论文 published 日期归档，周末 arXiv 公布周期导致全部写进
        # 旧日期文件并覆盖历史数据（2026-10-17 实测：107 篇覆盖了 10-15 的文件）
        today = time.strftime("%Y-%m-%d")
        out = os.path.join(OUT_DIR, "%s.jsonl" % today)
        old = []
        if os.path.exists(out):
            with open(out, encoding="utf-8") as f:
                for l in f:
                    l = l.strip()
                    if l:
                        try:
                            old.append(json.loads(l))
                        except Exception:
                            pass
        seen2, merged = set(), []
        for it in old + uniq:
            k = (it["src"], norm_id(it["id"]))
            if k in seen2:
                continue
            seen2.add(k); merged.append(it)
        with open(out, "w", encoding="utf-8") as f:
            for it in merged:
                f.write(json.dumps(it, ensure_ascii=False) + "\n")
        print("[save] %s  %d 篇（新增 %d，合并同日旧档）" % (today, len(merged), len(merged) - len(old)))
    else:
        # 补抓历史（--days N）：按论文原始日期分组归档，原逻辑保留
        by_date = {}
        for it in uniq:
            d = it.get("date") or time.strftime("%Y-%m-%d")
            by_date.setdefault(d, []).append(it)
        for d, items in sorted(by_date.items(), reverse=True)[:args.days]:
            out = os.path.join(OUT_DIR, "%s.jsonl" % d)
            with open(out, "w", encoding="utf-8") as f:
                for it in items:
                    f.write(json.dumps(it, ensure_ascii=False) + "\n")
            print("[save] %s  %d 篇" % (d, len(items)))

    # 刷新索引
    idx = os.path.join(OUT_DIR, "index.md")
    lines = ["# AI 前沿文章索引", ""]
    for fn in sorted(os.listdir(OUT_DIR), reverse=True):
        if not fn.endswith(".jsonl"):
            continue
        d = fn[:-6]
        with open(os.path.join(OUT_DIR, fn), encoding="utf-8") as f:
            items = [json.loads(l) for l in f if l.strip()]
        lines.append("## %s（%d 篇）" % (d, len(items)))
        for it in items[:50]:
            star = " ★%d" % it["upvotes"] if it.get("upvotes") else ""
            lines.append("- [%s] %s（%s）%s" % (it["cat"], it["title"], it["url"], star))
        lines.append("")
    with open(idx, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print("[index] %s" % idx)


if __name__ == "__main__":
    main()
