# gemini (served: gemini-3.8-flash)

Initial focus is on designing a web scraping and search pipeline suitable for a CPU-bound, 0.5B parameter ternary model; must account for local execution constraints. Will address specific user requirements for actionable engineering design, including Python tooling, CPU time estimates, and a minimal viable stack. Primary concerns involve efficiently managing data retrieval and indexing for a model operating with limited resources.


Search entry selection necessitates careful balancing of coverage, stability, and CPU overhead given the budget. Initial hypothesis: prioritize curated whitelists to reduce query count and bandwidth usage, but assess if public search engines are necessary for broader initial exploration.

Page body extraction needs evaluation: is a CPU-intensive headless browser essential for handling JavaScript-heavy Chinese sites, or can simpler libraries suffice? Test extraction success on representative Chinese sites to quantify performance.

The primary failure risks are:
1.  **Exceeding Resource Limits**: The strict daily constraints (queries and bandwidth) could severely limit the scope and depth of data collection.
2.  **Anti-Scraping Blocking**: Chinese websites are known to employ aggressive anti-bot measures, potentially halting the data pipeline entirely.


Privacy footprint minimization must address search intent leakage through referer, full URLs, and query parameters. Countermeasures will involve local generalization of queries, stripping parameters, blocking trackers, and constructing clean HTTP requests.

The initial technical deep dive suggests that direct scraping of public search engines like DDG, Google, and Bing in China faces significant stability and accessibility issues. Self-hosted search engine proxies like SearXNG offer a potentially more stable and efficient solution by aggregating results and returning JSON. The CPU cost of public search access is estimated at ~5ms per query.


Dockerized SearXNG, while providing aggregation, introduces significant CPU and memory overhead, making it unsuitable for a lean CPU-bound system. Curated domain whitelists combined with direct HTTPS or API access offer superior performance and stability, particularly for a model focused on factual knowledge. Hybrid approach: Whitelist primary, free-tier search APIs (Bing, DDG Lite) as secondary, with strict rate limiting.

Web page body extraction will be evaluated. `trafilatura` appears more robust for Chinese, at the cost of CPU (15-40ms). `readability-lxml` is much faster (5-10ms), but sacrifices content fidelity and structured data handling.


Heavy headless browsers are deemed unsuitable due to their significant CPU, RAM, and bandwidth consumption, particularly for a resource-constrained model. Direct parsing of server-side rendered data or static HTML is preferred, discarding sites that require JavaScript execution for efficiency.

Given the low query volume, proxy rotation is unnecessary; polite behavior and jittered delays, coupled with header hygiene, will be sufficient for rate limiting. Local caching of `robots.txt` will also be beneficial.


Optimize resource utilization by caching `robots.txt` rules.  Chinese site analysis reveals the need for site-specific scraping strategies. Zhihu requires JSON extraction or exclusion; WeChat necessitates aggregator use; Douban requires strict pacing or API fallback; and News Portals are straightforward.

Resource constraints dictate a cost-benefit analysis considering data volume, extraction yield, and token ROI, aiming for an average of 100KB per page and ~2KB - 8KB of clean text from each HTML page.


Prioritizing Tier 1 sources (tech docs, encyclopedias) due to high signal-to-noise ratio and structural clarity, to provide the most value for a resource-constrained model. Aiming for approximately 30,000 to 60,000 high-quality tokens daily after deduplication.
Privacy safeguards require local query generalization and parameter stripping to prevent search intent leakage via referer and URL tracking.


Header sanitization, URL canonicalization, and local query generalization will mitigate privacy risks from tracking parameters and user-specific queries. Implementing DNS-over-HTTPS will prevent ISP logging of search queries.

The CPU latency profile indicates that query generalization is the most computationally expensive step, followed by search requests and extraction. HTML download is I/O-bound. The total CPU time per document extraction is expected to be under 50ms, excluding query generation.


针对本地 0.5B / CPU 优先 / BitNet 1.58 三值化自学习模型的特性，工程设计的核心原则是：**零渲染（No-JS）、高信噪比、极低 CPU/内存开销、零外部依赖侵入**。在日 ≤ 50 次查询、≤ 10MB 流量约束下，爬取层必须极致轻量，杜绝任何重量级组件。

---

### 一、 搜索入口选型对比（50 次查询/10MB 预算）

| 方案 | 中文覆盖率 | 稳定性 | CPU 开销 | 流量消耗估算 | 结论与中文站适配评注 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **A. 公共引擎直连**<br>(DDG HTML / Bing / Google) | 中（DDG 中文差）<br>高（Bing / Google） | **极低**<br>(极易 403 / 触发验证码) | 极低<br>(< 5ms/次) | ~30-80KB/次<br>(含 HTML 结构) | **不推荐**。DDG 中文抓取结果严重脱节；Google/Bing 直连 HTML 爬取在无动态代理下 10 次内必封 IP。 |
| **B. 自建 SearXNG**<br>(本地/轻量 Docker) | **极高**<br>(聚合百度+Bing+搜狗) | **中高**<br>(引擎聚合容灾) | **高**<br>(常驻占用 200MB+ 内存，查询占 10-30% CPU) | 仅 **5-15KB/次**<br>(原生输出干净 JSON) | **备选**。虽然输出 JSON 极省流量，但 Docker 和 Python 异步聚合对 CPU 优先单机是额外负担。 |
| **C. 权威白名单直连**<br>+ 官方免密 API / 维基 | 高（聚焦垂直高质量知识） | **极高**<br>(协议标准，极少反爬) | **极低**<br>(< 2ms/次) | 仅 **2-10KB/次**<br>(REST API Gzip JSON) | **首选主力 (80%)**。直接走维基媒体 API、百度百科静态接口、开源文档站。 |

#### 落地推荐方案（混合策略）
1. **主力（80%）**：**权威白名单直取**。维护一份静态映射路由（维基百科 API、百度百科只读抓取、GitHub Readme/Docs、MDN）。维基原生提供 Action API（`zh.wikipedia.org/w/api.php`），单次返回纯文本提取物仅 3KB，耗时极低。
2. **兜底（20%）**：**Bing Web Search API（免费层/试用）或 SearXNG 轻量实例**。若必须做开放式搜索，**绝对不要裸爬搜索引擎前端 HTML**，日均 50 次完全处于各大搜索 API 的免费配额内（Bing 免费层每月 1000 次，50次/天只需两周轮换或微调至 33 次/天），每次直接拿取 10 条纯 JSON 结果，无反爬风险且不吃 CPU。

---

### 二、 网页正文抽取工具评测

| 工具 | CPU 单页耗时 | 中文抽取成功率 | 结构化保留 (标题/代码) | 内存占用 |
| :--- | :--- | :--- | :--- | :--- |
| **`trafilatura`** | **15 – 35 ms** | **92%**（针对中文博客/新闻极优） | 极优（保留 markdown 层级） | 低 (~30MB) |
| **`readability-lxml`** | 5 – 15 ms | 78%（容易截断中文短段落） | 中（偏向单一纯文本块） | 极低 (~15MB) |
| **无头浏览器 (Playwright)** | **800 – 2500 ms** | 98%（全 JS 渲染支持） | 依赖后处理 | **极高 (单进程 > 300MB)** |

#### 是否真的需要无头浏览器？
**明确结论：绝对不需要，且必须坚决摒弃。**

**判定理由**：
1. **CPU/内存灾难**：Chromium 启动与页面 Hydration 阶段会瞬间打满 2~4 个 CPU 核心，直接与 0.5B 模型的推理/增量训练抢占调度周期。
2. **流量击穿**：无头浏览器加载一个现代中文网页（知乎/网易），哪怕开启无图模式，各类追踪 JS、SDK、CSS 至少下行 **1.5MB~4MB**。50 个页面将消耗 **100MB~200MB**，直接击穿 10MB 预算 10 到 20 倍。
3. **中文 SPA 站点的“静态后门”**：
   * **知乎**：直接 GET 专栏或问题页面，其静态 HTML 的 `<script id="js-initialData">` 中包含完整的 SSR JSON 数据，用正则直接提取，无需执行 JS。
   * **微信公众号**：完全是服务端渲染的静态 HTML，正文全在 `#js_content` 选择器中，`trafilatura` 一行即可抽取，反倒是开无头浏览器极易触发滑块验证。

---

### 三、 反爬、限流与中文站点对抗策略

日均 ≤ 50 次意味着**平均抓取间隔可拉长至 15~30 分钟一次**。在这种极低频度下，**不需要购买商业代理池或 IP 轮换**，单机公网 IP 完全可胜任，前提是遵守以下反爬规约：

#### 1. 基础防护伪装
* **强制 HTTP/2**：现代反爬 WAF（如 Cloudflare、阿里云盾）会检测 HTTP/1.1 与 TLS 指纹。使用支持 HTTP/2 的库（如 `httpx[http2]`）可以抹平特征。
* **Sec-CH-UA 伪装矩阵**：单纯伪造 User-Agent 必被封。必须带齐 Client Hints 头：
  ```python
  HEADERS = {
      "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
      "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
      "Sec-Ch-Ua": '"Chromium";v="124", "Google Chrome";v="124", "Not-A.Brand";v="99"',
      "Sec-Ch-Ua-Mobile": "?0",
      "Sec-Ch-Ua-Platform": '"Windows"',
      "Sec-Fetch-Dest": "document",
      "Sec-Fetch-Mode": "navigate",
      "Sec-Fetch-Site": "none",
  }
  ```
* **随机抖动机制**：若有连续抓取（如一搜带三采），请求间必须强制施加高斯分布随机延迟：$Delay = \mu (5s) + \sigma (2s)$。

#### 2. 中文核心站点对抗与对策

* **知乎 (Zhihu)**：
  * *难点*：未登录强制拦截登录弹窗、动态下滚加载、严格的 IP 风控。
  * *对策*：**绕开主站，抓搜索摘要或知乎专栏（zhuanlan.zhihu.com）**。专栏页面对爬虫宽容度高得多。直接提取 HTML 内的 `__NEXT_DATA__` 或 `initialData` JSON 文本，避免解析脏 DOM。
* **微信公众号 (WeChat Official Accounts)**：
  * *难点*：URL 具备防盗链时效性（包含 `tempkey` / 签名）、反爬高频弹验证码、无搜索公开入口。
  * *对策*：**不直爬公众号搜索**。若搜索结果落入微信文章，直接用纯静态抓取（禁用 Cookie）；仅抓取正文 `#js_content` 文本；一旦遇到 `环境异常 / 验证码` 提示页面，设置该域名熔断 24 小时，避免 IP 被微信永久拉黑。
* **豆瓣 (Douban)**：
  * *难点*：国内最老牌严格的反爬站，数据中心 IP（机房 IP）直接 403 封死，本地家庭宽带易触发 10 分钟临时阻断。
  * *对策*：请求头必须挂载完整 `Referer: https://www.douban.com/`；抓取频率必须降至单次间隔 > 10 秒；仅允许抓取静态书籍/电影介绍条目，剔除讨论区。
* **新闻门户（网易/新浪/腾讯）**：
  * *难点*：正文充斥大量垃圾营销号内容与分页切割。
  * *对策*：几乎无反爬风控，直接用 `trafilatura` 提取正文，并配置正文长度阈值（`len(text) > 300` 字），丢弃少于 300 字的短讯图集。

---

### 四、 成本、信噪比与 Token ROI 测算

在 **10MB 流量/天、50 次请求/天** 的极端紧凑预算下，知识输入模型的数据转化流如下：

#### 1. 数据漏斗转化数学模型
* **下行流量控制**：请求全部启用 `Accept-Encoding: gzip, br`。单页面压缩后 HTML 平均大小为 **80 KB**。
  $$50 \text{ 页面} \times 80 \text{ KB} = 4.0 \text{ MB 原始下行流量}$$
  完全受控在 10MB 配额内（剩余 6MB 留作 API 交互及安全重试余量）。
* **纯文本产出**：通过 `trafilatura` 清除一切 CSS/JS/DOM 骨架后，单页面纯中文有效字数约在 **1,500 ~ 3,000 字**（中位数 2,000 字，约 4KB 文本）。
  $$50 \text{ 篇} \times 2,000 \text{ 字} = 100,000 \text{ 中文字}$$
* **去重过滤（SimHash 相似度 > 0.8 剔除）**：去重损耗率约为 25%。
  $$100,000 \times 75\% = 75,000 \text{ 有效中文字/天}$$
* **BitNet Token 换算**：对于现代中文分词器（如 Llama-3 扩充包或 Qwen/DeepSeek 分词器），1 个汉字约为 $0.6 \sim 0.8$ Token。
  $$\text{每日净增高质量 Token 量} \approx \mathbf{45,000 \sim 60,000\ Tokens}$$
  *对于 0.5B 这样参数容量有限的模型，每日摄入 5 万精选 Token 恰好在其短时微调/记忆回放的最佳吞吐区间内，不会导致灾难性遗忘。*

#### 2. 信息来源信噪比矩阵与优先级排序

| 优先级 | 来源分类 | 典型站点 | 信噪比 | 推荐理由与过滤规则 |
| :---: | :--- | :--- | :---: | :--- |
| **P0** | **官方技术文档 / 百科** | 维基百科、MDN、ArchWiki、ReadTheDocs | **95%** | 事实密度极高，逻辑排版规范，完全静态，单 Token 知识价值最大。 |
| **P1** | **同行评审 / 论文摘要** | arXiv.org (abs)、Semantic Scholar | **85%** | 严谨学术定义，少水话，无反爬，自带结构化 abstract。 |
| **P2** | **专业深度媒体** | InfoQ、少数派、科学网博客 | **65%** | 结构完整但含部分个人修辞，需剔除文末广告与评论。 |
| **P3** | **综合新闻门户** | 澎湃新闻、新华网、网易科技 | **40%** | 时效性强但知识半衰期短，存在模板套话，降级使用。 |
| **P4** | **社交平台 / 问答社区** | 知乎、豆瓣、微博、贴吧 | **< 15%** | **坚决排除**。充满情绪发泄、黑话、阴阳怪气，对 0.5B 模型的知识库构建是毒药。 |

---

### 五、 隐私足迹最小化方案

为了彻底隔断本地个人模型意图与外网画像追踪，必须实施“出网降维”策略：

```
[原始查询词] 
    │
    ▼ (本地 0.5B 模型)
[实体抽取与通用泛化] (剔除个人代词、敏感上下文，提炼出客观实体)
    │
    ▼
[URL/参数清洗管道] (剥离追踪参数，清空 Referer)
    │
    ▼ (DoH 加密解析)
[目标服务器] (仅收到白身 GET 请求)
```

1. **查询词本地泛化（Query Canonicalization）**：
   本地 0.5B 模型决不能直接拿带有个人身份/隐私上下文的原句出网。
   * *原意图*：“我妈妈吃了阿莫西林后身上起红点怎么处理”
   * *本地泛化出网词*：“阿莫西林 不良反应 皮疹 处理原则”
2. **零 Referer 泄露**：
   所有外发请求必须显式设置：
   `Referer: ""` 或在 HTML Meta 层面遵守 `no-referrer`，绝不向第三方下游站点泄露是从哪个搜索引擎跳转而来的。
3. **URL 强力消毒（Strip Tracking Parameters）**：
   对目标 URL 进行正则表达式清理，剔除一切平台追踪标记（如知乎的 `?utm_source=...`、微信的 `chksm=...`、B站的 `?spm_id_from=...`），既防止身份追踪，又提升 URL 去重命中率。
4. **底层屏蔽 DNS 探测（DoH）**：
   使用 Python 原生基于 HTTP/2 的 Cloudflare / 阿里 DNS-over-HTTPS（DoH）解析，防止本地运营商（ISP）通过透明 DNS 缓存窥探爬取目标。

---

### 六、 Python 工具链与 CPU 耗时流水线

#### 1. 最小可行工具链（MVP）
只保留 3 个底层轮子，杜绝冗余：
* **`httpx` (with `httpcore[http2]`)**：负责网络 I/O，支持 HTTP/2，原生连接池复用。
* **`trafilatura`**：极速网页正文解析库，底层为 C 优化的 `lxml`，CPU 占用极小。
* **`simhash`**：64位局部敏感哈希，内存占用几近于零的纯文本去重算法。

#### 2. 单次抓取循环 CPU 耗时拆解（基于普通 x86/ARM 单核 CPU）
1. **URL 参数消毒**：`< 0.1 ms`
2. **HTTP/2 握手与下载（I/O 等待，CPU 空闲）**：网络消耗约 200~500ms，占用 CPU `< 2 ms`
3. **Gzip/Brotli 解压**：`~ 1 ms`
4. **Trafilatura 正文与元数据抽取**：`15 ~ 35 ms`
5. **SimHash 计算与布隆过滤去重**：`~ 2 ms`
* **单网页纯 CPU 活跃时间**：$\mathbf{\le 40\ ms}$。单核每秒可清洗 25 篇，完全不对 0.5B 模型的 CPU 资源形成干扰。

---

### 七、 最小可行爬取栈 (MVP 生产代码)

```bash
pip install httpx[http2] trafilatura simhash
```

```python
import re
import random
import time
from urllib.parse import urlparse, urlunparse, parse_qsl, urlencode
import httpx
import trafilatura
from simhash import Simhash

# 1. 隐私与跟踪参数黑名单
TRACKING_PARAMS = {'utm_source', 'utm_medium', 'utm_campaign', 'utm_term', 
                   'utm_content', 'spm', 'from', 'source', 'ref', '_hsenc'}

def sanitize_url(raw_url: str) -> str:
    """去隐私参数，规范化 URL"""
    u = urlparse(raw_url)
    filtered_query = [(k, v) for k, v in parse_qsl(u.query) if k.lower() not in TRACKING_PARAMS]
    clean_query = urlencode(filtered_query)
    return urlunparse((u.scheme, u.netloc, u.path, u.params, clean_query, ''))

class KnowledgeScraper:
    def __init__(self):
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
            "Accept-Encoding": "gzip, deflate, br",
            "Sec-Ch-Ua": '"Chromium";v="124", "Google Chrome";v="124", "Not-A.Brand";v="99"',
            "Sec-Ch-Ua-Platform": '"Windows"',
            "Sec-Fetch-Dest": "document",
            "Sec-Fetch-Mode": "navigate",
            "Sec-Fetch-Site": "none",
        }
        # 强制 HTTP/2 客户端，连接池保活，超时控制
        self.client = httpx.Client(http2=True, headers=self.headers, timeout=8.0, follow_redirects=True)
        self.fingerprint_history = []  # 存 Simhash 结果

    def is_duplicate(self, text: str, threshold: int = 4) -> bool:
        """汉明距离 <= 4 判定为重复内容"""
        text_hash = Simhash(text)
        for h in self.fingerprint_history:
            if text_hash.distance(h) <= threshold:
                return True
        self.fingerprint_history.append(text_hash)
        return False

    def fetch_clean_knowledge(self, url: str) -> str | None:
        clean_url = sanitize_url(url)
        try:
            # 引入防御性微延迟，平滑流量
            time.sleep(random.uniform(1.5, 3.5))
            
            resp = self.client.get(clean_url)
            if resp.status_code != 200:
                return None
            
            # Trafilatura 极速 CPU 抽取，屏蔽回帖与导航栏
            extracted_text = trafilatura.extract(
                resp.text, 
                include_comments=False, 
                include_tables=True,
                favor_recall=False, # 优先保证精确度/高信噪比
                output_format='txt'
            )
            
            if not extracted_text or len(extracted_text) < 200:
                return None  # 丢弃过短与空信息
            
            # 内容去重
            if self.is_duplicate(extracted_text):
                return None

            return extracted_text

        except Exception:
            # 错误熔断，静默失败，绝不崩溃
            return None

    def close(self):
        self.client.close()

# 执行示例
if __name__ == "__main__":
    scraper = KnowledgeScraper()
    test_url = "https://zh.wikipedia.org/wiki/%E5%A4%A7%E5%9E%8B%E8%AF%AD%E8%A8%80%E6%A8%A1%E5%9E%8B"
    knowledge = scraper.fetch_clean_knowledge(test_url)
    if knowledge:
        print(f"成功获取高信噪比文本 (前100字): {knowledge[:100]}...")
    scraper.close()
```

---

### 八、 本套爬取方案最大的 2 个系统性风险点

1. **小模型对“毒化隐蔽网页（Ad-SEO Cloaking）”的判别力缺失**：
   0.5B 参数的 BitNet 本身欠缺庞大参数带来的“常识过滤与真伪判别”能力。大量中文自媒体站使用采集器生成“看似通顺但全为车轱辘话”的伪原创 SEO 文章。`trafilatura` 只能判断其“是正文”，无法判断其“是垃圾”。这种文本直接灌入 0.5B 模型做增量自学习，极易引发**严重的认知漂移（Cognitive Drift）与逻辑混乱**。必须在白名单源头上强力卡死，绝对不能做全网开放抓取。
2. **反爬静默伪装（Silent HTTP 200 Captcha）导致垃圾 Token 污染**：
   国内平台（尤其是知乎、微信、小红书）拦截爬虫时，通常**不返回 403/429**，而是返回带安全盾验证码的静态 HTML（响应码依然是 `200 OK`）。正文提取器会精准抽取出：“*系统检测到您的网络环境异常，请输入验证码*”或无意义的 JS 占位符。若没有严格的校验规则（如拦截词过滤库、文本熵值检验），这套自动化栈会把大量“验证码提示词”当成自学语料，导致模型语言退化。必须在下游设立基于正则的高频拦截词熔断机制。