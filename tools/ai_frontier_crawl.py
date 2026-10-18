#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ai_frontier_crawl.py — AI 前沿文章爬虫（每日增量，中英双爬）
源（全部官方 API，无需登录）:
  1. arXiv API    12 个 CS 分类各 25 篇最新（cs.AI/LG/CL/CV/MA/RO/NE/SE/CR/IR + stat.ML + eess.AS）
  2. HF Papers    huggingface.co/api/papers 每日热榜 30 篇
  3. OpenAlex 英  title_and_abstract 搜 LLM/RL 等核心词，近 5 天 article（补非 arXiv 的期刊/会议）
  4. OpenAlex 中  title 搜 CS 中文核心词（补中文期刊：计算机学报/自动化学报等 DOAJ 收录源）
去重（双轨，跨源+跨日）:
  - id 轨：arXiv id / HF id / OpenAlex W-id / DOI
  - 标题指纹轨：norm_title -> 8 位 hash（小写、去标点、去空白）——跨源同一篇（arXiv vs OpenAlex
    同一论文不同 id 体系）也能抓住；HF 日榜与 arXiv 重复、OpenAlex 与 arXiv 重复均在此层拦掉
输出:
  data/frontier/YYYY-MM-DD.jsonl   当日全量（含 id/标题/摘要/链接/来源/分类）
  data/frontier/index.md           累计索引（按日期倒序，附标题）
用法:
  python tools/ai_frontier_crawl.py            # 抓今天
  python tools/ai_frontier_crawl.py --days 3   # 补抓近3天
"""
import os, sys, json, time, re, hashlib, argparse
import urllib.request
import ssl
import xml.etree.ElementTree as ET

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(ROOT, "data", "frontier")
PROXY = "http://127.0.0.1:3067"   # Karing 规则代理（国外走代理）
CTX = ssl.create_default_context()
CTX.check_hostname = False
CTX.verify_mode = ssl.CERT_NONE   # 代理 HTTPS 证书链不完整，关校验

ARXIV_CATS = ["cs.AI", "cs.LG", "cs.CL", "cs.CV", "cs.MA", "cs.RO",
              "cs.NE", "cs.SE", "cs.CR", "cs.IR", "stat.ML", "eess.AS"]
ARXIV_N = 25
SEEN_FILE = os.path.join(OUT_DIR, "_seen_ids.json")  # 跨日去重账本：{"ids":[...], "titles":[...]}

# OpenAlex 过滤：自出版仓库/无来源的 junk（Zenodo 人文哲学科普、PhilPapers 等）
JUNK_SRC = ("zenodo", "philpapers", "(none)", "proquest")
# OpenAlex 中文检索词（title.search，CS 向，排掉"人工智能赋能英语教学"靠 abstract 命中的噪音）
ZH_TERMS = ["大语言模型", "深度学习", "神经网络", "自然语言处理", "计算机视觉",
            "强化学习", "知识图谱", "机器学习", "多模态", "扩散模型", "智能体", " Transformer"]
EN_TERMS = ['"large language model"', '"reinforcement learning"', '"self-improving"',
            '"test-time training"', '"mixture of experts"', '"world model"',
            '"continual learning"', '"distillation"', '"AI agent"']


def norm_id(aid):
    """arXiv id 规范化：2610.22364v1 -> 2610.22364（版本号不算新论文）"""
    return re.sub(r"v\d+$", "", str(aid or "").strip())


def title_hash(title):
    """标题指纹：小写、去一切非字母数字汉字字符 —— 跨源同一论文的归一化键。
    'GPT-4 Technical Report' vs 'GPT 4 technical report' -> 同键。"""
    t = re.sub(r"[^0-9a-z\u4e00-\u9fff]", "", str(title or "").lower())
    return hashlib.md5(t.encode("utf-8")).hexdigest()[:8] if t else ""


def load_seen():
    """账本兼容新旧格式：旧=扁平 id 列表；新={"ids":[...], "titles":[...]}"""
    if os.path.exists(SEEN_FILE):
        try:
            with open(SEEN_FILE, encoding="utf-8") as f:
                d = json.load(f)
            if isinstance(d, list):
                return {"ids": set(d), "titles": set()}
            return {"ids": set(d.get("ids", [])), "titles": set(d.get("titles", []))}
        except Exception:
            pass
    # 冷启动：从历史 jsonl 建账本（老数据不算新）
    ids, titles = set(), set()
    for fn in os.listdir(OUT_DIR):
        if not fn.endswith(".jsonl"):
            continue
        with open(os.path.join(OUT_DIR, fn), encoding="utf-8") as f:
            for l in f:
                l = l.strip()
                if not l:
                    continue
                try:
                    it = json.loads(l)
                    ids.add(norm_id(it.get("id")))
                    th = title_hash(it.get("title"))
                    if th:
                        titles.add(th)
                except Exception:
                    pass
    return {"ids": ids, "titles": titles}


def save_seen(seen):
    with open(SEEN_FILE, "w", encoding="utf-8") as f:
        json.dump({"ids": sorted(seen["ids"]), "titles": sorted(seen["titles"])}, f)


def fetch(url, timeout=40, use_proxy=True):
    opener = urllib.request.build_opener(
        urllib.request.ProxyHandler({"http": PROXY, "https": PROXY}) if use_proxy else
        urllib.request.ProxyHandler({}),
        urllib.request.HTTPSHandler(context=CTX))
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 frontier-crawl"})
    return opener.open(req, timeout=timeout).read()


def crawl_arxiv(cat, n=ARXIV_N):
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


def _oa_abstract(inv):
    """OpenAlex abstract_inverted_index -> 普通文本"""
    if not inv:
        return ""
    pos = {}
    for w, idxs in inv.items():
        for i in idxs:
            pos[i] = w
    return " ".join(pos[i] for i in sorted(pos))


def _oa_junk(w):
    """过滤自出版 junk：来源是 Zenodo/PhilPapers 或无来源"""
    src = (w.get("primary_location") or {}).get("source") or {}
    sname = (src.get("display_name") or "(none)").lower()
    return any(j in sname for j in JUNK_SRC)


def crawl_openalex(lang, terms, since, n=50, title_only=False):
    """OpenAlex 官方 API（免 key）。lang='zh' 中文期刊 / 'en' 英文期刊会议（补 arXiv 之外的）"""
    field = "title.search" if title_only else "title_and_abstract.search"
    q = " OR ".join('"%s"' % t.strip('"') for t in terms)
    url = ("https://api.openalex.org/works?filter=language:%s,%s:%s,"
           "from_publication_date:%s&sort=publication_date:desc&per-page=%d&mailto=kou@local"
           % (lang, field, urllib.request.quote(q), since, n))
    try:
        d = json.loads(fetch(url, timeout=45))
    except Exception as e:
        print("[openalex-%s] 失败 %s" % (lang, e)); return []
    out = []
    today = time.strftime("%Y-%m-%d")
    for w in d.get("results", []):
        if _oa_junk(w):
            continue
        pub = w.get("publication_date") or ""
        if not pub or pub > today:      # 未来日期=脏数据
            continue
        src = (w.get("primary_location") or {}).get("source") or {}
        doi = w.get("doi") or ""
        landing = (w.get("primary_location") or {}).get("landing_page_url") or doi or w.get("id")
        out.append({"src": "openalex", "cat": "oa-" + lang,
                    "id": doi.replace("https://doi.org/", "") or w["id"].split("/")[-1],
                    "title": (w.get("title") or "").strip(),
                    "abstract": _oa_abstract(w.get("abstract_inverted_index"))[:1200],
                    "url": landing, "date": pub,
                    "venue": src.get("display_name") or ""})
    print("[openalex-%s] %d 篇（原始 %d）" % (lang, len(out), len(d.get("results", []))))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=1)
    args = ap.parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)

    all_items = []
    # 1) arXiv 12 类
    for cat in ARXIV_CATS:
        all_items += crawl_arxiv(cat)
        time.sleep(1)
    # 2) HF 日榜
    all_items += crawl_hf_papers(30)
    # 3) OpenAlex 英文（近 5 天，补期刊/会议）
    since_en = time.strftime("%Y-%m-%d", time.localtime(time.time() - 5 * 86400))
    all_items += crawl_openalex("en", EN_TERMS, since_en)
    # 4) OpenAlex 中文（窗口放宽到 4 个月，中文期刊出版慢）
    all_items += crawl_openalex("zh", ZH_TERMS, "2026-06-01", title_only=True)

    # 本轮内去重（id + 标题指纹 双轨）
    seen_id, seen_th, uniq = set(), set(), []
    for it in all_items:
        k = norm_id(it["id"])
        th = title_hash(it["title"])
        if k in seen_id or (th and th in seen_th):
            continue
        seen_id.add(k)
        if th:
            seen_th.add(th)
        uniq.append(it)

    # 跨日去重：账本（id + 标题指纹）里有的不再入库
    ledger = load_seen()
    fresh = [it for it in uniq
             if norm_id(it["id"]) not in ledger["ids"] and title_hash(it["title"]) not in ledger["titles"]]
    for it in uniq:
        ledger["ids"].add(norm_id(it["id"]))
        th = title_hash(it["title"])
        if th:
            ledger["titles"].add(th)
    save_seen(ledger)
    n_cn = sum(1 for it in fresh if it["cat"] == "oa-zh")
    print("[dedup] 本轮抓取 %d 篇，去重后新增 %d 篇（中文 %d）" % (len(uniq), len(fresh), n_cn))
    uniq = fresh

    if args.days <= 1:
        # 日常增量：按抓取日期归档，与同日已有文件合并去重（一天可跑多次）
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
        seen2, seen2th, merged = set(), set(), []
        for it in old + uniq:
            k = norm_id(it["id"])
            th = title_hash(it["title"])
            if k in seen2 or (th and th in seen2th):
                continue
            seen2.add(k)
            if th:
                seen2th.add(th)
            merged.append(it)
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
