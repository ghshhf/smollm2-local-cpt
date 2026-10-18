# deepseek (served: deepseek-v4-pro)

我们需要回答用户中文问题。需要聚焦具体爬取与搜索工程。用户要求覆盖5点，交叉比对，明确标注中文站点适用性。需要具体Python工具链包名+用途、每步CPU耗时估算、最小可行爬取栈清单、最大2风险点。需要基于本地0.5B/CPU优先/三值BitNet 1.58个人模型做联网自学，抓取决策层已生成匿名查询词。日≤50查询/≤10MB流量。需要给出可落地爬取工程方案，不是整体架构。

我们需要思考推断。要实在，不编造。涉及具体包、耗时需要合理估算，可能需说明硬件假设。要注明中文站点适用性。我们要比较三种搜索入口。是否真的需要无头浏览器？明确判断。反爬/robots。成本质量。隐私足迹最小化。

需要给出具体工具链。可能包：httpx, selectolax, trafilatura, readability-lxml, lxml, beautifulsoup4? jieba? 可能不需要。SearXNG via docker? DuckDuckGo HTML via ddgs? Bing? Google custom search API? 公共搜索引擎直连可能用 `ddgs` 或 `duckduckgo_search`，Bing 网页搜索无官方爬虫 API? Google Custom Search JSON API 每天100次免费? 对中文站点覆盖率。需要比较。

我们需考虑实际：日≤50查询，10MB流量。搜索API返回HTML或JSON，再抓取top URLs。公共直连搜索引擎有反爬、HTML结构变动。DuckDuckGo HTML 版 (`html.duckduckgo.com/html/`) 较稳定，可解析。Bing 可能有 bot 检测。Google 直连被墙/验证码，不适合大陆中文站点？中文站点对于大陆用户可能需要国内搜索（Bing国内可用，百度？）但问题只提到DDG/Bing/Google/SearXNG/白名单。可以提百度？用户没有列出百度，但中文站点适用性，可能提百度/搜狗/必应国内版。但用户说“必须覆盖以下5点” 搜索入口选型中列了公共搜索引擎直连（DuckDuckGo HTML / Bing / Google）、自建 SearXNG 元搜索、还是直接维护权威域名白名单。我可以主要比较这三种，但针对中文站点说明：Google 直连不可靠；Bing 国内可达但反爬；DDG HTML 英文/中文一般；百度/搜狗可能可作为补充但未列，是否引入？用户没有要求不能提，但要聚焦。也许可以指出中文站点优先：SearXNG 配合 bing、百度、搜狗、duckduckgo；或白名单直接抓。但预算 50 queries/day，10MB，CPU 低。SearXNG 自建需要维护实例，CPU 开销主要是容器运行，不是查询本身；元搜索自身会并发请求多个上游出流量。如果本地部署 SearXNG，查询一次返回聚合结果，出站流量可能超过直接？它请求多个引擎，每个查询约几 KB? 10MB预算可能不够? 需要估算。可能SearXNG 在本地 CPU 上轻量但持续运行消耗内存? 50 query/day 可以用按需启动容器。公共直连 DDG HTML 无需额外服务，CPU开销低。白名单直接抓取 CPU 最低、稳定性最高，但覆盖率最低。需要给权衡。

关于网页正文抽取：trafilatura vs readability-lxml vs headless。CPU耗时估算：trafilatura 每个 HTML 10-100ms？readability-lxml 更快？无头浏览器每秒可能500ms-2s per page on CPU，开销大。中文页面 JS 重度需要无头？但预算 10MB/天不需要大量，偶发 JS 需要可用 pyppeteer/playwright 按需。明确判断：日≤50/10MB 且 CPU 优先，默认不需要无头浏览器。使用 trafilatura 作为主抽取，readability-lxml 作为备选。对 JS 重度页面，先检查是否是静态HTML中已含内容，或用移动端API/原始接口，只有极少数高价值来源才按需无头。需标注知乎/公众号/新闻动态加载：知乎有反爬/登录墙，trafilatura可能取到部分；公众号文章页面很多是服务端渲染的图文，trafilatura可用但图片/某些动态模块缺失；新闻门户多是静态或轻JS，trafilatura成功率较高。无头浏览器对知乎/公众号不是万能，会触发更严格风控，且CPU开销大。可建议优先选择非JS源：百科、文档、论文摘要、静态新闻。

反爬/限流/robots：本地单机礼貌。UA随机化作用有限且可能增加风险？通常固定真实UA或明确bot UA，遵守robots。请求限速：每域名 1 request/2-5s，日50次完全不需要IP代理。代理/IP轮换对隐私和反封可能有用但流量小，不必要；代理会导致隐私足迹更差/额外成本。中文站点难点：知乎强反爬、登录墙；豆瓣 API 已关闭/网页有验证码；微信公众号需微信客户端环境，直接抓取受限；新闻门户多数可抓，但边界。对策：能白名单直接抓的，用官方API/RSS/开放接口；搜索引擎只获取链接，不抓需要登录的；对高风险站点设置更慢限速或放弃。

成本与质量估算：按10MB/天、50次查询/天。假设每次查询返回 10 个结果，共 500 个候选，按唯一 URL 去重后平均 30-50 个新页面？10MB 流量如果只抓正文，假设每页 HTML ~200KB，正文提取后 ~5-15KB，10MB 可能抓 50-200 个 HTML? 10MB = 10,000KB，如果每页平均 200KB，约50页；如果 gzip 压缩，实际传输更小? Need clarify. 每页正文有效 token 估算：中文 1 token≈0.7-1.5字? But model tokenizer? 假设提取后约 3-8KB 文本，中文约1500-4000字，token约1000-3000? 日净增有效 token：去重后高质量页面 20-50 页，每页 500-1500 有效 token，合计 10k-75k token。但流量预算如果抓大HTML可能下载少。应该按压缩后响应计算，HTML gzip压缩率70-80%。10MB可下载约 50-100 个典型页面。去重后净增有效 token 也许 20k-60k。信噪比：学术/文档/百科 > 技术博客/官方文档 > 新闻 > 问答/论坛 > 社交媒体。优先级：百科/官方文档/学术摘要/RSS 高；技术博客与教程中；论坛/知乎低；公众号需高成本低成功率，除非白名单特定。应优先通过 RSS/API/white list 获取高信噪比。

隐私足迹最小化：抓取会泄漏：

- 搜索引擎：查询词直接给搜索引擎；即使匿名查询词，搜索引擎根据查询词+IP可关联。完整 URL 可能包括跟踪参数（utm_*）、查询字符串，第三方资源（统计、字体、CDN）会接到Referer。需要：只发必要参数，剥离 utm、去除 fragment、设置 `Referer` 不发送或发送同源。使用禁用第三方域名：在抓正文时只请求主文档，不加载图片/CSS/JS（trafilatura/readability 只取 HTML 不加载子资源），避免第三方跟踪。若用无头浏览器必须阻止第三方请求（Playwright route abort）。
- 查询词本地泛化：将具体查询转换为类别词或模糊化；比如“张三论文题目” 泛化为 “论文标题”或模板。但用户明确抓取决策层已生成匿名查询词，需要再提示域名/URL中存在查询词。搜索引擎参数仅保留 q，不发送高级参数、地理位置、语言无关。禁用 cookie，不持久化? 无头浏览器隔离临时 profile。
- 抓取主文档时，发出的请求会带 IP、UA、Accept-Language。不要发送 Referer（默认 httpx 不跨域?）若从搜索引擎结果页点击会有 Referer，但我们直接请求目标URL，Referer可设为 None。阻止 DNS 泄露? DNS 查询本地运营商可见，但隐私威胁模型不同。
- 建议：通过本地代理按域名过滤，白名单仅允许目标域名和必要 CDN；关闭 cookies，禁用 HTTP/2 连接复用? Not necessary.
- 具体工具：httpx 设置 headers 无 Referer；使用 `PrivateControl`? maybe.

需要给出具体Python工具链及CPU耗时估算。列出包名用途：

- `httpx` 异步/同步 HTTP，支持 HTTP/2 (可选), 限速和重试
- `selectolax` 或 `lxml.html` 快速解析
- `trafilatura` 正文抽取（主）
- `readability-lxml` 正文抽取（备）
- `lxml` 依存或替代解析
- `ddgs`/`duckduckgo_search` 用于 DDG 搜索
- `beautifulsoup4`? 可能不需要
- `fake-useragent`? 不建议，随机化 UA 意义小，可用固定合理UA；可列出但不推荐。
- `tenacity`/`backoff` 重试
- `urllib.robotparser` 标准库 robots.txt
- `polite`? no.
- `playwright` 可选按需，但不出现在MVP最小栈。
- `jieba`? 若做中文去重/关键词，可能需要但非必需。
- `scrapy`? 太重，不推荐MVP。
- `sqlite-utils`? 存储。
- `waybackmachine?` no.

每步 CPU 耗时估算需说明参考硬件：4核低功耗 x86（如 N100/J4125）或树莓派？0.5B on CPU means maybe low-power. We need give assumptions. Could say 4核 x86 或 2-4核 ARM，仅供参考。耗时估算：

1. 搜索查询：DDG HTML 请求网络耗时主要，CPU解析 HTML ~10-30ms。
2. 下载主文档：网络为主，CPU解压 gzip ~1-5ms/KB? Actually.
3. HTML解析：selectolax解析 200KB ~5-15ms。
4. trafilatura 抽取：中文页面 200KB ~30-150ms（取决于长度和复杂度），readability-lxml ~50-200ms? Need maybe estimate.
5. 去重/清洗/分词 token: 每页几KB 文本 ~5-20ms.
6. 无头浏览器（如需要）: 每个页面 1-3s CPU，启动浏览器额外 500ms-1s。对≤50 query/day可接受 but for JS pages.
Total per page: 纯粹搜索+HTTP+抽取：约 200-500ms CPU excluding network; 低功耗可能更慢1-2s wall time. 50 pages/day within minutes CPU.

需要给具体 MVP 清单：MVP该装哪几个包、跑哪几步。可以给 `pip install httpx selectolax trafilatura readability-lxml lxml duckduckgo-search tenacity`; 不用无头。步骤：

1. 从决策层接收洁净查询词/URL。
2. 用 DDG HTML 或白名单域名合成候选 URL。
3. 检查 robots + 域名频控。
4. httpx 显式 GET 主文档，不加载子资源，headers 固定 UA、无 Referer、Accept-Language 中文。
5. 解析响应，trafilatura 抽取；失败回退 readability-lxml；仍然失败则丢弃或标记需要无头。
6. 后处理：文本清洗、去重（SimHash/MinHash / URL+内容指纹）、语言过滤。
7. 入库供 0.5B 本地消化。

最大 2 个风险点：What are biggest risks? Need think. 1）搜索引擎入口稳定性/反爬：DDG HTML 可能限流/结构变动，Bing/Google 会挑战；中文站点尤其。SearXNG 维护复杂。导致候选来源供给不稳定。2）网页正文质量/JS与反爬导致“有效 token 远低于流量预算”：很多高价值中文内容被 JS 动态加载或登录墙（知乎/公众号），静态抽取失败，需要无头但无头又风控/CPU，导致捕获取向低质量源（百科/新闻）漂移，知识增量少。 Or maybe 隐私与合规风险：抓取公开站点违反 robots/用户协议导致法律风险，但问题要求最大2个风险点，可以选工程风险。Maybe user expects "反爬/站点封禁" and "数据质量/噪声污染"。需要明确。

我们需要交叉比较表。三个入口覆盖率、稳定性、CPU开销。需要按日≤50查询/≤10MB。可做表格。注明中文站点适用性。表格内容：

- DDG HTML 直连：覆盖率中（英文好，中文一般，对百度索引内容少；但可以搜索中文），稳定性中（HTML接口可能限流，但无需API key），CPU开销低（解析结果页 ~20ms）。中文站点：可用，但中文内容不如百度/必应国内，对知乎/新闻返回有限；作为默认国际源可行。
- Bing 直连：覆盖率中高，国内可达，中文覆盖好，稳定性中低（HTML有bot检测，可能需要 cookies，易跳转验证），CPU开销低中（页面更复杂，解析30-60ms）。中文站点：较好，优先候选。
- Google 直连：覆盖率高，中文好，稳定性低（国内不可直连、bot验证严格，基本不适合大陆本地），CPU开销低中，但需要代理则增加隐私成本。中文站点：如果用户网络可直接访问，可选，但不推荐MVP。
- SearXNG 自建：覆盖率中高（聚合多个引擎，包括 bing/google/baidu? 取决于配置），稳定性中（需要自己维护容器和上游引擎反爬），CPU开销中（常驻内存200-500MB，每次查询等待上游；按需运行可降低），中文站点：通过配置百度/必应/搜狗较好，但需要额外调校。
- 权威域名白名单直抓：覆盖率低（仅预定义域名，适合百科/文档/论文），稳定性高（直连目标站点，无搜索引擎依赖），CPU开销最低（跳过搜索），中文站点：对中文百科、官方文档、开源文档很适合，对新闻/新页面没有发现能力。

我们还需要说“直接维护白名单”不是二选一，建议混合：白名单负责已知高值域名，DDG/Bing/SearXNG负责发现新 URL。每日50 queries 小，可以先用白名单满足 30-50% 高信噪比，剩余查询用于搜索发现。这样降低对搜索引擎依赖。

关于“是否真的需要无头浏览器？” 给出明确判断：不需要作为默认环节。在预算下，无头浏览器每页 CPU 高、增加被风控和流量（子资源）风险，且静态抽取能覆盖百科、文档、多数新闻、RSS/API。若只针对个别高价值 JS 页面，可以后续按域名加白名单按需启用，且必须屏蔽第三方请求。给出触发条件：trafilatura+readability 都抽不到正文，且该 URL 被标记为高价值，才用无头。估算：无头每页 1.5-3s CPU vs trafilatura 0.05-0.15s，相差10-30倍，对CPU优先不可接受。所以默认不装 Playwright。

需要讨论中文站点特殊性：知乎、公众号、豆瓣、新闻门户。可以表格：

- 知乎：强反爬，