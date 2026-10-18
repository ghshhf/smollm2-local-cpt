# qwen (served: Qwen3.8-27B)

我们需要回答中文用户。任务：本地 0.5B / CPU优先 / 三值 BitNet 1.58 个人模型要做联网自学，已生成匿名查询词，聚焦具体爬取与搜索工程，不是整体架构。必须覆盖5点并明确中文站点适用性。需要具体 Python 工具链（包名+用途）、每步 CPU 耗时估算、最小可行爬取栈 MVP 清单，最后指出最大2个风险点。

我们需要给出可落地方案并交叉比对。需要谨慎：反爬/robots，礼貌，不鼓励绕过付费墙/登录墙/法律限制。可以提及代理/IP轮换一般不推荐，尤其个人本地小预算，合规风险。需要估计：日≤50查询/≤10MB流量。0.5B CPU inference irrelevant maybe but CPU extraction overhead. Need specific numbers approximate. Need discuss search entry: DuckDuckGo HTML, Bing/Google, SearXNG, whitelist direct HTTPS. Coverage/stability/CPU overhead under budget. For Chinese sites applicability. Need likely recommend SearXNG local? But SearXNG requires network to engines, may be unstable in China? If local in China? User says local 0.5B/CPU, maybe Chinese sites. Need mention network environment (GFW, domestic engines). SearXNG can use Bing, Baidu, Sogou, 360, DuckDuckGo, Google; but engines may block. Direct whitelist is most stable and low cost but coverage limited. Public search direct: DuckDuckGo HTML endpoint (html.duckduckgo.com/html/?q=) simple, no JS, but Chinese coverage poor, may CAPTCHA. Bing HTML? bing.com/search?q= may have anti-bot, but often parseable; Google hard. Need recommend hybrid: whitelist + SearXNG (or DuckDuckGo HTML as fallback) not raw Google. For 50 queries/day, search results pages maybe 1-2 MB each, 50*~0.3-1MB = 15-50MB if raw HTML; but budget 10MB total. Need optimize: search only for unknown topics, use whitelist, compress, cache. 50 queries/day with 10MB means average 200KB per query including pages? Very tight. If search result page ~50-150KB compressed, plus 2-3 snippets? Need maybe 50 queries * 50KB = 2.5MB, then fetch 1-3 pages * 150KB = 5-7MB. But if full pages, 10MB only ~30-50 text pages. Need recommend budget allocation: 10 queries search, 40 direct whitelist? But user says 50 queries/day. Could interpret 50 search queries, not fetch pages? Need estimate. We can say if 50 full search queries + 2-3 fetch each, 10MB insufficient; must limit to 50 total HTTP objects or 10 search + 40 whitelist. Need provide formula.

Web extraction: trafilatura vs readability-lxml vs headless. CPU time: trafilatura fast ~0.05-0.5s per 100KB HTML, best for articles, docs, wiki; readability-lxml 0.05-0.3s, less robust; Playwright headless 2-15s per page, 100-500MB RAM, CPU high, only for JS heavy (Zhihu, WeChat? WeChat often static HTML? mp.weixin.qq.com may be static but has anti? Zhihu needs JS/anti, but also API? WeChat articles can be fetched with normal HTML if not blocked, but may require referer? Need mention). Judgment: MVP no headless; use trafilatura + fallback readability; only optional Playwright for few high-value JS pages, with budget.

Anti-crawl/rate/robots: local single crawler polite: check robots.txt, cache, respect Crawl-delay, identify UA? UA randomization? For personal crawler, better stable descriptive UA with contact? But privacy: minimize footprint, maybe generic browser UA. Need not randomize too much (can look bot). Use per-domain rate limit e.g. 1 req/2-5s, concurrency 1, timeout, retries exponential backoff, max pages, compress, ETag/Last-Modified, local cache. Proxy/IP rotation: not needed for 50/day; increases risk and privacy; only if blocked, but consider switching source not proxy. Chinese sites special: Zhihu (anti-bot, JS, login wall, API cookies, captcha; use public search? maybe not; avoid scraping; use RSS? not available; use whitelist of official docs, not Zhihu), Douban (rate, captcha, login; avoid or low frequency), WeChat public (mp.weixin.qq.com articles often fetchable but token/anti, no index, need URL; use RSS bridges? not recommended; direct if have URL, verify content; not crawl), news portals (CDN, anti-bot, paywall, ads; use RSS/official APIs if available, whitelist, avoid aggressive). Baidu? Baidu Baike may have anti, but can fetch with normal? Baidu search blocked. Sogou WeChat search? Not stable. Need mention legal: only public non-login, no circumvent technical measures, respect copyright, personal use.

Cost and quality: 10MB/day, 50 queries/day. Estimate tokens. Chinese char ~1-1.5 bytes? UTF-8 Chinese 3 bytes/char; tokenization for 0.5B maybe BPE: Chinese ~1.5-3 chars/token? Actually for many tokenizers, Chinese char often 1 token or 1-2? Need estimate conservatively: 1MB UTF-8 Chinese ~333k chars, if 1.5 chars/token => ~220k tokens; if 1 char/token => 333k tokens. But net effective after extraction/dedup maybe 30-50%. 10MB raw HTML yields maybe 1-3MB text, 300k-1M chars, 200k-500k tokens. But high noise. Dedup net new maybe 50k-200k tokens/day. Need present range: if 10MB is raw compressed? If 10MB downloaded, after extraction maybe 2-4MB text, 0.5-1.5M chars, 0.3-0.8M tokens; after dedup/noise 30-60% => 100k-500k tokens. If strict 50 queries and only snippets, less. Need be realistic: 10MB/day can fetch 30-80 article pages (100-300KB HTML each) or 100-300 text pages (30-100KB). Net new high-quality maybe 50k-200k tokens/day. For 0.5B model, too much? Maybe for self-learning need curate. Sources priority: official docs/API references, standards, textbooks, Wikipedia/Baidu Baike (with caution), arXiv/Papers with code, government/regulatory, stable blogs (engineering), then news, social. Academic/docs highest; encyclopedias medium; social low. For Chinese: arXiv, GitHub docs, MDN? Chinese docs: Python docs, PyTorch? But Chinese sites: 百度百科 (coverage but quality variable), 维基百科 (Chinese) good but may be blocked? Need mention network. CSDN/博客园/知乎 low SNR, anti; official product docs high. News low. Priority: 1) domain whitelist docs/standards/official; 2) arXiv/open access; 3) Chinese/English Wikipedia; 4) official blogs/changelogs; 5) curated news RSS; 6) forums/social only as leads not training.

Privacy footprint: Search queries leak to engine; URL params, referrer, cookies, TLS metadata (SNI, IP, timing) visible. Minimize: local query anonymization/generalization, remove personal identifiers, use search engine with no tracking? DuckDuckGo HTML still reveals query to DDG; SearXNG local proxy but upstream sees query unless use Tor? But local SearXNG can randomize? Actually SearXNG forwards to engines; can set botdetection, no tracking? It may add? Need mention: if privacy critical, use direct whitelist and avoid search; if must search, use local SearXNG with minimal engines, disable cookies, no referrer, strip query params, use HTTPS, maybe local DNS, avoid web search for sensitive. Use `urllib`/`httpx` with `headers` no referer, `Accept-Language` maybe zh-CN, no cookies (or session per domain but not persistent), no third-party tracking by not loading JS/images. For direct fetch, set `Referer` empty, only fetch canonical URL, remove tracking params (utm_*, fbclid, gclid, spm, scene, from, etc). Use URL normalization. Use local cache to avoid repeated. Use `httpx` with `follow_redirects=False`? Or limited redirects to avoid leaking? Redirects can leak? If follow, server sees. Maybe disable or allow one. Use `Accept-Encoding: gzip, br` to reduce bandwidth but CPU decompress. Disable JS, images, CSS. Use `max_redirects=2`. Use `User-Agent` stable but not identifying? For privacy, UA reveals software; but needed. Could use generic browser UA. Do not send cookies. Use ephemeral TLS? Hard. Use local DNS to avoid DNS leaks? Maybe use DoH? But DoH reveals to provider. Could use system. Need mention SNI/IP unavoidable. Query terms: local generalize: remove entities, use categories, use local taxonomy, send only broad terms; if need specific, use whitelist direct.

Need Python toolchain: packages:
- httpx or requests: HTTP client. `httpx` better async? For CPU single, `requests` simpler. Use `httpx` with HTTP/2? maybe `httpx[http2]` but CPU overhead. For MVP: `requests` or `httpx`.
- `trafilatura` main extraction; deps `lxml`, `html5lib`? `trafilatura` includes.
- `readability-lxml` fallback.
- `lxml` parsing, CSS.
- `beautifulsoup4` optional for fallback/tables? maybe not MVP.
- `cssselect`? lxml has cssselect? `lxml` uses `cssselect`? Need package maybe `cssselect`.
- `chardet`/`charset-normalizer` for encoding; `requests` handles.
- `robotsparser` for robots.txt.
- `urlexp` or `urllib.parse` for URL normalization; `tldextract` maybe for domain.
- `pyahocorasick` or `rapidfuzz` for dedup? For small scale `hashlib`, `simhash`? `datasketch` for MinHash? Could be heavy. MVP: SHA-256 of normalized text + SimHash optional.
- `py7zr`? no.
- `playwright` optional, not MVP.
- `searxng` not Python package? Self-host via Docker, or use `searxng` as service. For MVP maybe no SearXNG, use DuckDuckGo HTML via `requests` or `ddgs` package? There is `duckduckgo_search` (ddgs) but may break. For stable: parse HTML with `requests` + `lxml`. Could mention `ddgs` optional.
- `newspaper3k`? Not needed.
- `langdetect`? maybe no.
- `jieba` for Chinese tokenization? For dedup/quality maybe optional. But 0.5B tokenizer maybe use `tokenizers` if model tokenizer available. For MVP maybe not.
- `diskcache` or SQLite for cache. `sqlite3` stdlib.
- `tenacity` for retries? optional.
- `pydantic` for config? optional.
- `loguru`? optional.
- `aiohttp` if async, but CPU bound extraction not benefit. Use single-threaded.
- `playwright` optional.
- `html2text`? trafilatura outputs markdown/html/text. Use `trafilatura.extract` with `output_format='txt'` or 'md'.
- `pandas`? no.
- `scrapy`? Overkill; but could use? For MVP no, because CPU and complexity. But for robust crawling maybe `scrapy` with `scrapy-robots`, `scrapy-playwright`. However user wants Python toolchain; we can say avoid Scrapy for MVP, use httpx + trafilatura.

Need step CPU耗时估算. Need maybe table:
- Query normalization/local generalization: <5ms.
- robots fetch/cache: first 50-300ms network + 1ms local; cached <1ms.
- Search request: network 100-800ms; parse result HTML: 5-30ms for 100KB (lxml) maybe 20-100ms; trafilatura not needed for search snippets; CPU total 50-200ms.
- Fetch article: network 200-2000ms; decompress 1-5ms/MB; HTML parse/trafilatura 50-500ms for 200KB; readability 30-200ms; normalization/dedup 10-50ms; total CPU 0.1-1s per page (excluding network). Headless: 3-20s CPU, RAM 150-500MB.
- 50 queries/day: if 10 searches + 40 fetches: CPU extraction ~10*0.1 + 40*0.3 = 13s/day, negligible; headless 10 pages = 60-200s. Need present.

Need "最小可行爬取栈" list: install packages: `httpx` (or requests), `trafilatura`, `lxml`, `robotsparser`, `charset-normalizer`, `sqlite3` stdlib, `hashlib`, `re`, maybe `tldextract`, `tenacity`. Optional `readability-lxml`. Not Playwright. Steps:
1. config: whitelist domains, budget, rate limits.
2. local query planner: generate 1-3 broad queries, filter.
3. fetch robots for domains, cache.
4. choose search: if query not in whitelist, use DuckDuckGo HTML or SearXNG; parse top 5; filter whitelist/allowed domains; dedup.
5. fetch up to N pages (e.g. 2 per query) with rate limit, max 512KB, gzip, no JS/images.
6. extract with trafilatura; fallback readability; clean; estimate quality (length, domain, title, language).
7. dedup: URL hash, text SHA, n-gram/simhash; store in SQLite.
8. output chunked text to model training pipeline.
MVP run: `pip install httpx trafilatura lxml robotsparser charset-normalizer tldextract tenacity` maybe `readability-lxml`. Use single script.

Need cross-comparison. Let's structure final answer:
- Assumptions and budget: 50 queries/day, 10MB/day; define query = one search or one target URL? We'll state if 50 search queries plus fetches impossible; recommend 50 total "lookup operations" or 10 search + 40 direct. Use compressed bytes. If 10MB is egress raw after compression. Need maybe table:
  - Search result page: 30-150KB compressed; CPU parse 20-100ms.
  -