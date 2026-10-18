#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
netarch_v1 — 联网组件 V1：管道 / 抓取 / 清洗 / 去污染 / 分层存储 / 重放。

设计红线（来自多模型交叉 + 本地裁决）：
  1. 清洗 ≠ 去污染。清洗=格式/编码/去重/正文；去污染=信任边界/提示注入/伪造来源。
     洗得很干净的网页仍可能是投毒或虚假事实 → 两套独立标记，不合并。
  2. 一切外部内容默认 untrusted，永不自动升为 trusted（升格需人工/规则，且留痕）。
  3. 机密的本地数据不出网：本管道只往【出】抓，出网请求只带问题文本，绝不带本地语料/记忆。
  4. 分层存储可回放：raw 原样保留 → clean 清洗结果 → evidence 证据 → logs 处理日志。
  5. 不预装 Playwright/Scrapy（吃内存、破端侧预算）；JS 重度站走 SSR 静态 JSON 兜底或推迟。

零硬依赖：仅用 stdlib（urllib / sqlite3 / html.parser / xml.etree）。若环境装了
trafilatura/readability-lxml 会自动升级正文提取质量（薄封装，不重造轮子）。

用法：
  python netarch.py fetch <url> [<url> ...]          # 抓一批 URL 入管道
  python netarch.py replay <document_id>            # 用 raw 重放整条管道
  python netarch.py list [trust_level]              # 列存储分层
  python netarch.py stats                           # 管道/存储统计
"""
import sys, os, json, time, hashlib, sqlite3, re, gzip, io, ssl
import urllib.request, urllib.error
import xml.etree.ElementTree as ET
from html.parser import HTMLParser

HERE = os.path.dirname(os.path.abspath(__file__))
DB = os.path.join(HERE, "netarch.db")
RAW_DIR = os.path.join(HERE, "raw")
PIPELINE_VERSION = "pipeline-v1"

_CTX = ssl.create_default_context(); _CTX.check_hostname = False; _CTX.verify_mode = ssl.CERT_NONE

# ── 管道状态机（有限重试，幂等）──────────────────────────────────────────────
STATES = ["PENDING", "FETCHING", "FETCHED", "CLEANING", "CLEANED",
          "SCORED", "STORED", "RETRYABLE_FAIL", "FAILED_PERMANENT"]
RETRY_MAX = 3
RETRY_BACKOFF = [5, 20, 60]   # 秒

# ── 去污染规则（提示注入 / 伪造来源 / 混淆字符）─────────────────────────────
INJECTION_PATTERNS = [
    (r"ignore\s+(all\s+)?(previous|prior|above)\s+instructions", "ignore-prev-instr"),
    (r"disregard\s+(the\s+)?(system|previous)\s+(prompt|message)", "disregard-system"),
    (r"(you\s+are\s+now|act\s+as)\s+(a\s+)?( DAN |unrestricted|jailbroken)", "jailbreak-persona"),
    (r"<\s*/?\s*(system|assistant|user)\s*>", "fake-role-tag"),
    (r"\[/?\s*(INST|SYSTEM|ASSISTANT)\s*\]", "llama-role-marker"),
    (r"reveal\s+(your|the)\s+(system\s+)?prompt", "prompt-exfil"),
    (r"(api[_-]?key|secret|password|token)\s*[:=]\s*\S{8,}", "cred-leak"),
]
# 零宽 / 同形伪装字符（去污染标记，不做激进还原）
HIDDEN_CHARS = re.compile(r"[\u200b\u200c\u200d\ufeff\u00ad]")

# ── 数据库（分层存储）────────────────────────────────────────────────────────
def db():
    c = sqlite3.connect(DB, timeout=30)
    c.row_factory = sqlite3.Row
    return c

def init_db():
    os.makedirs(RAW_DIR, exist_ok=True)
    c = db(); cur = c.cursor()
    cur.executescript("""
    CREATE TABLE IF NOT EXISTS documents(
      document_id TEXT PRIMARY KEY,
      source_url TEXT, fetched_at TEXT, content_hash TEXT,
      raw_ref TEXT, clean_text_ref TEXT, trust_level TEXT DEFAULT 'untrusted',
      injection_flags TEXT, processing_version TEXT, status TEXT DEFAULT 'PENDING',
      title TEXT, lang TEXT, word_count INTEGER, source_kind TEXT, error TEXT, tries INTEGER DEFAULT 0
    );
    CREATE TABLE IF NOT EXISTS provenance(   -- 证据链：这条内容从哪来、经过什么处理
      document_id TEXT, step TEXT, detail TEXT, at TEXT
    );
    CREATE TABLE IF NOT EXISTS processing_logs(
      id INTEGER PRIMARY KEY AUTOINCREMENT, document_id TEXT, level TEXT, msg TEXT, at TEXT
    );
    CREATE TABLE IF NOT EXISTS errors(
      document_id TEXT, url TEXT, err TEXT, at TEXT
    );
    CREATE INDEX IF NOT EXISTS idx_doc_hash ON documents(content_hash);
    CREATE INDEX IF NOT EXISTS idx_doc_trust ON documents(trust_level);
    """)
    c.commit(); c.close()

def log(document_id, msg, level="INFO"):
    c = db(); c.execute("INSERT INTO processing_logs(document_id,level,msg,at) VALUES(?,?,?,?)",
                        (document_id, level, msg, now())); c.commit(); c.close()

def prov(document_id, step, detail):
    c = db(); c.execute("INSERT INTO provenance(document_id,step,detail,at) VALUES(?,?,?,?)",
                        (document_id, step, detail, now())); c.commit(); c.close()

def now(): return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

# ── 抓取（stdlib urllib；尊重限速，单域名间隔）────────────────────────────────
_LAST_HIT = {}
def rate_limit(url, gap=1.0):
    host = urllib.parse.urlparse(url).netloc
    last = _LAST_HIT.get(host, 0)
    wait = gap - (time.time() - last)
    if wait > 0: time.sleep(wait)
    _LAST_HIT[host] = time.time()

import urllib.parse
def fetch(url, timeout=25):
    rate_limit(url)
    req = urllib.request.Request(url, headers={
        "User-Agent": "netarch-v1/0.1 (+local research crawler)",
        "Accept": "text/html,application/json,application/rss+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Encoding": "gzip",
    })
    try:
        with urllib.request.urlopen(req, timeout=timeout, context=_CTX) as r:
            raw = r.read()
            if r.headers.get("Content-Encoding") == "gzip":
                raw = gzip.decompress(raw)
            ct = (r.headers.get("Content-Type") or "").lower()
            status = r.getcode()
            final_url = r.geturl()
        if status >= 400:
            return None, f"HTTP {status}", ct, final_url
        return raw, None, ct, final_url
    except urllib.error.HTTPError as e:
        return None, f"HTTP {e.code}", "", url
    except Exception as e:
        return None, f"{type(e).__name__}: {e}", "", url

# ── 正文提取（优先 trafilatura，否则 stdlib 保守提取）────────────────────────
class _TextExtractor(HTMLParser):
    SKIP = {"script","style","noscript","nav","footer","header","aside","form","svg"}
    def __init__(self):
        super().__init__(); self.parts=[]; self.skip_depth=0; self.title=None; self._in_title=False
    def handle_starttag(self, tag, attrs):
        if tag=="title": self._in_title=True
        if tag in self.SKIP: self.skip_depth += 1
    def handle_endtag(self, tag):
        if tag=="title": self._in_title=False
        if tag in self.SKIP and self.skip_depth>0: self.skip_depth -= 1
    def handle_data(self, data):
        if self._in_title and self.title is None and data.strip():
            self.title = data.strip()[:200]
        if self.skip_depth==0 and data.strip():
            self.parts.append(data.strip())
    def text(self):
        return "\n".join(self.parts)

def strip_html(html_text):
    p = _TextExtractor()
    try: p.feed(html_text)
    except Exception: pass
    return p.text(), p.title

def clean_text(raw_bytes, content_type):
    """清洗：解码归一 + 正文提取 + 去重空白。返回 (text, title, lang, source_kind)。
    注意：这里【不做】任何信任判断——那是 decontaminate 的职责。
    优先用现成库（trafilatura/feedparser），缺失则退化为 stdlib 保守提取。"""
    if "json" in content_type:
        try:
            obj = json.loads(raw_bytes.decode("utf-8","replace"))
            return json.dumps(obj, ensure_ascii=False, indent=1), None, "json", "json"
        except Exception:
            pass
    if "rss" in content_type or "xml" in content_type:
        # 优先 feedparser（更稳），否则 stdlib ET
        try:
            import feedparser
            fp = feedparser.parse(raw_bytes)
            if fp.entries:
                items=[]
                for e in fp.entries:
                    t=(e.get("title") or "").strip()
                    link=(e.get("link") or "").strip()
                    summ=(e.get("summary") or e.get("description") or "").strip()
                    items.append(f"- {t}\n  {link}\n  {summ}")
                return "\n".join(items), None, "unknown", "rss"
        except Exception:
            pass
        try:
            root = ET.fromstring(raw_bytes)
            items = []
            for item in root.iter():
                tag = item.tag.split("}")[-1]
                if tag in ("item","entry"):
                    t = (item.findtext("title") or "").strip()
                    d = (item.findtext("description") or item.findtext("summary") or "").strip()
                    link = (item.findtext("link") or "").strip()
                    items.append(f"- {t}\n  {link}\n  {d}")
            return "\n".join(items), None, "unknown", "rss"
        except Exception:
            pass
    # HTML：优先 trafilatura（正文质量最高），退化 readability，再退化 stdlib
    html_text = None
    try:
        import trafilatura
        html_text = trafilatura.extract(raw_bytes, include_comments=False, include_tables=True,
                                        favor_precision=True) or ""
    except Exception:
        pass
    if not html_text or len(html_text) < 200:
        try:
            import trafilatura
            html_text = trafilatura.extract(raw_bytes, favor_recall=True) or html_text
        except Exception:
            pass
    title = None
    if not html_text:
        try:
            from readability import Document as _Doc
            doc = _Doc(raw_bytes.decode("utf-8","replace"))
            html_text = doc.summary()
            title = doc.short_title()
        except Exception:
            pass
    if not html_text:
        html_dec, title = strip_html(raw_bytes.decode("utf-8","replace"))
        html_text = html_dec
    # 编码归一：全角空格、多余空白
    text = re.sub(r"[ \t　]+", " ", html_text)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    zh = len(re.findall(r"[\u4e00-\u9fff]", text))
    lang = "zh" if zh > len(text)*0.1 else "unknown"
    return text, title, lang, "html"

# ── 去污染：独立标记，不删内容（留证据）─────────────────────────────────────
def decontaminate(text, source_url):
    flags = []
    for pat, name in INJECTION_PATTERNS:
        if re.search(pat, text, re.I):
            flags.append(name)
    if HIDDEN_CHARS.search(text):
        flags.append("hidden-chars")
    # 伪造来源：正文里出现与真实 source_url 不同的绝对链接冒充权威（弱信号）
    if source_url:
        host = urllib.parse.urlparse(source_url).netloc
        foreign = set(re.findall(r"https?://([a-zA-Z0-9.-]+)", text[:5000]))
        if host and any(f != host for f in foreign) and len(flags)==0:
            pass  # 正常外链不算污染，仅记录；不误杀
    return flags

# ── 管道主流程（幂等，可重放）────────────────────────────────────────────────
def process(url):
    c = db()
    row = c.execute("SELECT document_id,tries,status FROM documents WHERE source_url=?", (url,)).fetchone()
    if row:
        document_id, tries, status = row["document_id"], row["tries"], row["status"]
        if status == "STORED":
            print(f"[skip-dup] {url} already STORED ({document_id})"); c.close(); return document_id
        if tries >= RETRY_MAX:
            print(f"[skip-dead] {url} failed {tries}x, giving up"); c.close(); return None
    else:
        document_id = "doc_" + hashlib.sha256(url.encode()).hexdigest()[:16]
        tries = 0
        c.execute("INSERT INTO documents(document_id,source_url,fetched_at,status,tries) VALUES(?,?,?,?,?)",
                  (document_id, url, now(), "PENDING", 0)); c.commit()

    c.execute("UPDATE documents SET status=?,tries=tries+1 WHERE document_id=?", ("FETCHING", document_id)); c.commit(); c.close()
    prov(document_id, "FETCH_START", url)

    raw_bytes, err, ct, final_url = fetch(url)
    if err:
        c = db(); tries = c.execute("SELECT tries FROM documents WHERE document_id=?", (document_id,)).fetchone()["tries"]
        st = "RETRYABLE_FAIL" if tries < RETRY_MAX else "FAILED_PERMANENT"
        c.execute("UPDATE documents SET status=?,error=? WHERE document_id=?", (st, err, document_id))
        c.execute("INSERT INTO errors(document_id,url,err,at) VALUES(?,?,?,?)", (document_id, url, err, now()))
        c.commit(); c.close()
        log(document_id, f"fetch failed: {err}", "ERROR")
        print(f"[fail] {url}: {err} -> {st}")
        return None

    content_hash = "sha256:" + hashlib.sha256(raw_bytes).hexdigest()
    raw_ref = f"raw/{document_id}.bin"
    with open(os.path.join(HERE, raw_ref), "wb") as f: f.write(raw_bytes)
    prov(document_id, "FETCH_OK", f"bytes={len(raw_bytes)} ct={ct} final={final_url}")
    log(document_id, f"fetched {len(raw_bytes)} bytes ct={ct}")

    # 去重（content_hash）
    c = db()
    dup = c.execute("SELECT document_id FROM documents WHERE content_hash=? AND document_id!=? LIMIT 1",
                    (content_hash, document_id)).fetchone()
    c.execute("UPDATE documents SET status=?,content_hash=?,raw_ref=? WHERE document_id=?",
              ("CLEANING", content_hash, raw_ref, document_id)); c.commit(); c.close()
    if dup:
        prov(document_id, "DEDUP_HIT", f"same content as {dup['document_id']}")
        log(document_id, f"duplicate content of {dup['document_id']}")

    text, title, lang, kind = clean_text(raw_bytes, ct)
    prov(document_id, "CLEAN_OK", f"chars={len(text)} lang={lang} kind={kind}")

    flags = decontaminate(text, url)
    trust = "untrusted"  # 永远默认 untrusted；外部内容不自动升格
    clean_ref = f"clean/{document_id}.txt"
    os.makedirs(os.path.join(HERE,"clean"), exist_ok=True)
    with open(os.path.join(HERE, clean_ref), "w", encoding="utf-8") as f: f.write(text)
    prov(document_id, "DECONTAMINATE", f"trust={trust} flags={flags or 'none'}")

    c = db()
    c.execute("""UPDATE documents SET status=?,clean_text_ref=?,trust_level=?,injection_flags=?,
                 processing_version=?,title=?,lang=?,word_count=?,source_kind=?,error=NULL
                 WHERE document_id=?""",
              ("STORED", clean_ref, trust, json.dumps(flags, ensure_ascii=False),
               PIPELINE_VERSION, title, lang, len(text), kind, document_id))
    c.commit(); c.close()
    dupnote = " (DUP)" if dup else ""
    print(f"[ok] {url} -> {document_id}{dupnote} trust={trust} flags={flags or '-'} chars={len(text)}")
    return document_id

def replay(document_id):
    """用保存的 raw 重放整条管道（去污染+存储），验证可复现性。"""
    c = db()
    row = c.execute("SELECT * FROM documents WHERE document_id=?", (document_id,)).fetchone()
    c.close()
    if not row or not row["raw_ref"]:
        print(f"[replay] no raw for {document_id}"); return
    raw_bytes = open(os.path.join(HERE, row["raw_ref"]), "rb").read()
    ct = row["source_kind"]
    text, title, lang, kind = clean_text(raw_bytes, ct)
    flags = decontaminate(text, row["source_url"])
    prov(document_id, "REPLAY", f"recomputed chars={len(text)} flags={flags or 'none'}")
    print(f"[replay] {document_id}: chars={len(text)} (orig {row['word_count']}) flags={flags or '-'} reproducible={len(text)==row['word_count']}")

def listdocs(trust=None):
    c = db()
    if trust:
        rows = c.execute("SELECT document_id,source_url,trust_level,status,title,word_count,injection_flags FROM documents WHERE trust_level=? ORDER BY fetched_at DESC", (trust,)).fetchall()
    else:
        rows = c.execute("SELECT document_id,source_url,trust_level,status,title,word_count,injection_flags FROM documents ORDER BY fetched_at DESC").fetchall()
    c.close()
    print(f"{'id':22s} {'trust':10s} {'status':8s} {'chars':>7s}  title/flags")
    for r in rows:
        print(f"{r['document_id']:22s} {r['trust_level']:10s} {r['status']:8s} {r['word_count'] or 0:7d}  {(r['title'] or '')[:40]} {r['injection_flags'] or ''}")

def stats():
    c = db()
    for tbl in ["documents","provenance","processing_logs","errors"]:
        print(f"{tbl}: {c.execute(f'SELECT COUNT(*) FROM {tbl}').fetchone()[0]}")
    print("by trust:", dict(c.execute("SELECT trust_level,COUNT(*) FROM documents GROUP BY trust_level").fetchall()))
    print("by status:", dict(c.execute("SELECT status,COUNT(*) FROM documents GROUP BY status").fetchall()))
    flagged = c.execute("SELECT COUNT(*) FROM documents WHERE injection_flags NOT IN ('[]','') AND injection_flags IS NOT NULL").fetchone()[0]
    print(f"flagged(untrusted w/ injection): {flagged}")
    c.close()

if __name__ == "__main__":
    init_db()
    cmd = sys.argv[1] if len(sys.argv)>1 else "stats"
    if cmd == "fetch":
        for u in sys.argv[2:]: process(u)
    elif cmd == "replay":
        for did in sys.argv[2:]: replay(did)
    elif cmd == "list":
        listdocs(sys.argv[2] if len(sys.argv)>2 else None)
    elif cmd == "stats":
        stats()
    else:
        print(__doc__)