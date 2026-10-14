# 联网自学组件 · 架构草案（三模型交叉核验 + 现有代码映射）

> 通道：api.krisphyn.cc 公益站 · 2026-10-14
> 提问主题：**本地端侧 0.5B / CPU 优先 / 三值 BitNet 1.58 个人模型的「联网自学组件」架构**
> 受访模型（并行流式询问，交叉比对）：
> - `gemini-3.8-flash-high`（实际 served：**gemini-3.8-flash**）—— Google
> - `deepseek-v4-pro-0813`（实际 served：**deepseek-v4-pro**）—— DeepSeek
> - `Qwen3.8-27B`（实际 served：**Qwen3.8-27B**）—— 阿里
> - ⚠️ `grok-4.7`：本轮三连击均被代理在 60s 中断（与 earlier「grok 真伪待核验」开放项一致），未取得证词，不影响结论（已有 3 个独立源共识）。
> 原始回答存档：`train_15b/eval/netarch_answers/ans_{gemini,deepseek,qwen}.md`

---

## §0 任务约束（我们已定的，不是模型给的）

- 模型形态：0.5B、CPU-first、三值 BitNet b1.58、单机常驻、每人 10M 记忆空间（外置）。
- 宪法级红线：**机密不出本地**——`owner_corpus` / 私人记忆 / 适配权重 / 未脱敏日志永不出境。
- 已有基座：`node_v5.py` 信箱信标（决定触发训练/同步/自检）；`distill_gate.py`（JSD 分歧 token 门控，已自检通过）；`data-plan-0.4B.md`（holdout / σ bootstrap / 外部中文 holdout / 蒸馏门控）。
- 用户哲学：**用最强模型当方向给出者**（外脑可当参谋，但本地是信任边界与最终否决权）；**本地先不实验，等「全自动」授权再跑**。

---

## §1 三模型共识（高置信收敛）

| # | 共识点 | gemini | deepseek | qwen |
|---|--------|--------|----------|------|
| C1 | **0.5B 三值小模型不能直接当自主「决策大脑」**（开放域查询生成/页面可信度/复杂反注入不可靠），但**可胜任「传感器/触发器/初筛」**（PPL 突刺、实体断裂、二分类要不要查） | ✅ 明确 | ✅ 明确 | ✅ 明确 |
| C2 | **本地↔云混合协议**：外部大脑只能是「无状态方向给出者」，只收匿名/脱敏**任务包**（task_id + 匿名需求 + 约束），**绝不收 owner_corpus 原文**；本地保留最终否决权 | ✅ | ✅ | ✅ |
| C3 | **机密边界**：owner_corpus / 私人记忆 / Adapter / 未脱敏日志**永不出境**；仅脱敏查询词 / 公开 URL / 聚合统计可出 | ✅ | ✅ | ✅ |
| C4 | **管道分层**：抓取 → 清洗（trafilatura/readability 抽正文、去脚本/iframe/不可见节点）→ 去污染/漂移(D1) → 训练池 | ✅ | ✅ | ✅ |
| C5 | **安全闸门**：格式沙箱（零宽/熵）+ 注入签名剥离（网页文本改写为第三人称文档块、绝不当 system prompt）+ Loss/PPL 门控 + holdout 回归回测回滚 | ✅ 4 道硬闸 | ✅ 阈值表 | ✅ 8 道闸 |
| C6 | **与 node_v5 集成**：扩展信标为调度中心，加 GATHERING/TRAINING/VALIDATING 状态 + 日预算熔断 | ✅ net_budget.json | ✅ 事件总线 | ✅ EventBus |
| C7 | **CPU 可行性**：0.5B BitNet 推理 50–120 tok/s、内存 <400MB、清洗 10–50 页/天完全可行 | ✅ | ✅ | ✅（补：三值本体训练难） |
| C8 | **MVP 优先 3 模块**：清洗器 + 信箱触发 + 安全闸门/回测 | ✅ | ✅ | ✅ |

---

## §2 关键分歧与我们的裁决

| 分歧 | 模型意见 | 我们的裁决（结合已有规划） |
|------|----------|---------------------------|
| **外部大脑是否必需** | gemini：可能必需；deepseek：建议作无状态规划器、本地否决；qwen：可选，全离线可用规则+小模型 | **采用可选外部大脑**，但**仅作查询扩写/网页摘要的方向给出者**，不接触私人数据。对齐用户「用最强模型当方向给出者」。RSI 自举主体仍在本地。 |
| **embedding 能否出网** | gemini 未强调；**deepseek + qwen 明确警告 embedding 可逆攻击（尤其短文本/私人记忆）** | **默认禁止发私人记忆/owner 的 embedding**。如需 topic 信号，加噪 / 随机投影 / 只发聚合统计。这是对 C3 的重要补强。 |
| **BitNet 三值本体能否在 CPU 上训** | gemini 乐观（训 Adapter）；**qwen 现实指出三值权重 CPU 训练需特殊 kernel，难** | **采用「冻结主干 + float16 影子 Adapter QAT」**（与 data-plan 的 Naive 节奏一致）。MVP 阶段先 RAG + 知识卡片，仅高置信多源数据进 Adapter，不在 CPU 硬训三值本体。 |
| **分歧 token 门控怎么做** | 三模型都只提了 PPL/困惑度门控 | **直接复用我们已写好的 `distill_gate.py`（JSD 有界门控，已实测）** 作为正式 Gate C，比单纯 PPL 更严格、可移植。 |

---

## §3 可落地架构草案（模块图）

```
═══════════════ 本地信任边界 (LOCAL TRUSTED BOUNDARY) ═══════════════
[核心资产] owner_corpus / 私人记忆 / 0.5B BitNet 权重 / Adapter   ← 永不出境
──────────────────────────────────────────────────────────────────
① 缺口感知 Gap Sensor        ② 抓取决策器 FetchDecider
   0.5B PPL 突刺 / 实体断裂      规则 + 0.5B 二分类(是否查)
   / holdout 失败 → 需求槽位       ↓ 生成安全查询骨架
③ 脱敏网关 DLP/Egress Gateway  ← 本地 PII 扫描 + owner 词汇 TF-IDF 熔断
                                 + 零宽/注入剥离
   ↓ 仅匿名查询词 / 加噪 topic(可选)
═══════════════ 出本地边界 ═══════════════════┐
                                              ▼
                   [公共搜索 API / 网页] ← 可选外部大脑(无状态):
                                             仅收 task_id+匿名需求, 返查询计划 JSON
                                              │
═══════════════ 数据管道 (CPU, 纯本地) ═══════▼
④ 抓取    httpx + trafilatura（白名单域名，禁 JS 执行）
⑤ 清洗    lxml 去脚本/iframe/不可见节点；正则指令剥离
          → 改写为第三人称文档块 [web_doc id src date]…[/web_doc]
⑥ D1 去污染/漂移  MinHash LSH 去重 + 5-gram 与 holdout 重叠<3%
          + embedding 距本地锚点 cosine>0.35（不相关丢弃）
⑦ 安全闸门  Gate A 零宽/熵 → Gate B 注入签名 →
          Gate C 分歧 token 门控(distill_gate.py, JSD) →
          Gate D holdout 回测回滚
⑧ 训练池  SQLite/Parquet + manifest{source,date,hash,risk}
⑨ 训练器  冻结主干 + BitNet Adapter QAT；影子步 + golden_owner_eval 回测
⑩ 评估/回滚  ΔPPL≤+0.5% 放行，否则 checkpoint 回滚 + 拉黑域名
──────────────────────────────────────────────────────────────────
node_v5.py 信箱信标（调度中心）:
  IDLE ─(.flag_net_gather, CPU空闲>15min)→ GATHERING
       ─(staging 满 50k tokens)→ TRAINING(锁网络)
       ─→ VALIDATING(Gate D) → 合并 / 回滚
  预算熔断 net_budget.json: 日 ≤50 查 / ≤10MB
════════════════════════════════════════════════════════════════════
```

### 模块职责 / 技术 / CPU 开销表

| 模块 | 职责 | 技术选型 | CPU 开销 | 可行性 |
|------|------|----------|----------|--------|
| 缺口感知 | 触发「我该查什么」 | 0.5B 前向 PPL / 实体度 / holdout 分 | 低 | ✅ |
| 抓取决策 | 规则+0.5B 二分类，生成安全查询骨架 | 规则 + 0.5B zero-shot | 低 | ✅（仅初筛） |
| 脱敏网关 | PII/owner 词表熔断、零宽剥离 | regex + 本地 TF-IDF 词表 | 低 | ✅ |
| 抓取 | 拉公开 URL | httpx + trafilatura | 中(IO) | ✅ |
| 清洗 | 抽正文、去不可见节点、指令剥离 | lxml / readability | 低-中 | ✅ |
| D1 去污染 | 去重 + 重叠 + 漂移 | MinHash LSH / 5-gram / bge-micro ONNX(~20MB) | 中 | ✅ |
| Gate A/B | 零宽、熵、注入签名 | regex + 熵计算 | 可忽略 | ✅ |
| Gate C | **分歧 token 门控** | **distill_gate.py（JSD）** | 低（已实测） | ✅ 复用 |
| Gate D | holdout 回测回滚 | golden_owner_eval + checkpoint | 低-中 | ✅ |
| 训练池 | 带 provenance 追加 | SQLite + Parquet + hash | 低 | ✅ |
| 训练器 | 冻结主干+Adapter QAT | BitNet.cpp 推理 + float16 影子 Adapter | 中（夜间） | ✅（不硬训三值本体） |
| 信箱集成 | 状态机+预算 | node_v5.py 扩展 | 低 | ✅ 已有基座 |

### 可操作阈值表（三模型合并默认值，上线前按实测调）

| 判据 | 阈值 | 来源 |
|------|------|------|
| 零宽字符 `\u200B-\u200D\uFEFF` | 命中即丢弃 | gemini |
| 香农熵 H | 3.5 < H < 6.0 合法 | gemini |
| 文本长度 | 100–15000 token | deepseek/qwen |
| 脚本/iframe/表单节点 | = 0 | deepseek |
| 与 owner/holdout 5-gram 重叠 | > 3% 拒 | qwen/deepseek |
| MinHash Jaccard | > 0.3 拒；> 0.7 合并去重 | qwen/deepseek |
| embedding 与本地锚点 cosine | < 0.35 拒（不相关） | gemini |
| embedding 与 owner corpus cosine | > 0.80 拒（隐私泄漏风险） | deepseek |
| 单 token 平均 CE（PPL 门控参考） | < 0.5 拒(重复)；> 8.0 拒(乱码)；1.2–5.0 接纳 | gemini |
| holdout 回测 ΔPPL | ≤ +0.5% 放行，否则回滚 | gemini（严） |
| 外部数据占比 | 每批 ≤ 总训练 token 5–10% | deepseek |
| 日预算 | ≤ 50 查 / ≤ 10MB | gemini |

---

## §4 与现有代码的映射（不用重造轮子）

- **`node_v5.py` 信箱**：直接扩展状态机（IDLE/GATHERING/TRAINING/VALIDATING）+ `net_budget.json`。基座已存在。
- **`distill_gate.py`（JSD 分歧 token 门控）**：即草案 Gate C 正式组件，已自检通过。**比三模型建议的 PPL 门控更严格、阈值有界可移植**，优先采用。
- **`data-plan-0.4B.md` 的 holdout / σ bootstrap / 外部中文 holdout**：即 Gate D 依据；`golden_owner_eval.jsonl`（10 条核心问答）放本地常驻。
- **`eval/check_token_len.py` + `filter_len.py`**：长度闸门（与 §3 阈值表对齐）。
- **`compression_bench.py`**：可选——新语料入池前算 NLL 验收（Delétang adjusted-rate 思路）。
- **防 subliminal 注入**：三模型的 holdout 回测能抓「漂移」但抓不到「隐性 stealth 传递」——这正是我们 P3 / E1 实验的命题；门控只能降风险，真裁决靠 E1 双盲属性传递测试。

---

## §5 最小可落地 MVP（先做的 3 个模块）

1. **清洗器**：`httpx + trafilatura + 正则指令剥离 + 零宽过滤` → 产出 `[web_doc]` 第三人称文档块。
2. **node_v5 静态触发**：`.flag_net_gather` 启动 → 抓取到 `staging/` 满 100KB 即停机断网。
3. **分歧 token 门控 + golden eval 回测**：`distill_gate.py` + `golden_owner_eval.jsonl` 放行/回滚。

（此 MVP 完全复用现有脚本骨架，外部大脑与 D1 嵌入漂移检测列为第二阶段。）

---

## §6 待办 / 排期（挂 E1 之后，本地先不实验）

- [ ] 新建 `train_15b/netarch/` 组件脚本骨架：清洗器、DLP 网关、node_v5 状态扩展、distill_gate 接入。
- [ ] 写 `golden_owner_eval.jsonl`（10 条核心问答，本地常驻，不入训练池）。
- [ ] E1（subliminal 裁决）出结果后，决定是否给 MVP 加「隐性注入探针」。
- [ ] grok-4.7 真伪：换 `grok-4.20-multi-agent` 或换时段重试取证（维持开放项）。
- [ ] 全量落地**等用户「全自动」授权**（铁律：本地先不实验）。

---

## §7 一句话结论

三模型独立收敛到**同一架构哲学**：本地小模型当「认知匮乏传感器」、确定性代码当「管道过滤器」、外部大脑（可选）当「无状态方向给出者」、信箱信标当「严格状态机」。**这与我们已有的 node_v5 + distill_gate + data-plan holdout 完全同构**——联网组件不需要新范式，只需把现有闸门从「训练期」前移到「入网抓取期」，并把机密边界从「owner_corpus 不出」补强到「私人记忆 embedding 也不出」。

---

## §8 第二轮交叉：具体爬取 / 搜索工程（2026-10-14）

> 受访：gemini-3.8-flash-high(served=gemini-3.8-flash, 14919字) / deepseek-v4-pro-0813(served=deepseek-v4-pro, 6573字) / Qwen3.8-27B(11683字)
> 原始存档：`eval/netarch_answers/q2_crawling/ans_*.md` · 提问：`docs/netarch/questions/q2_crawling.md`

### 8.1 三模型强共识（爬取层）

| # | 共识点 | gemini | deepseek | qwen |
|---|--------|--------|----------|------|
| K1 | **入口=白名单直连(主,~80%) + 公共搜索API/DDG-lite(辅,发现)**；**绝不裸爬搜索引擎前端HTML**（403/验证码） | ✅ 维基API/百度百科/GitHub/MDN | ✅ 白名单+DDG HTML/Bing 混合 | ✅ 白名单+DDG/SearXNG |
| K2 | **默认不需要无头浏览器**；Playwright=800–2500ms/页、300MB+ RAM、单页下行 1.5–4MB（击穿10MB预算）；JS重度站用「SSR静态JSON兜底」(`__NEXT_DATA__`/`#js_content`) | ✅ 坚决摒弃 | ✅ 默认不装，按需个别 | ✅ MVP不装 |
| K3 | **工具链**：`httpx[http2]` + `trafilatura`(主) + `readability-lxml`(备) + `simhash`/MinHash 去重 + robots 缓存 | ✅ | ✅(+selectolax/ddgs/tenacity) | ✅(+lxml/robotsparser/sqlite3) |
| K4 | **反爬**：≤50查/天无需代理/IP轮换；高斯抖动延迟(1.5–3.5s)；守 robots；站级策略(知乎绕主站抓专栏/SSR、微信禁cookie+验证码熔断24h、豆瓣Referer+>10s、新闻直抓) | ✅ | ✅ | ✅ |
| K5 | **隐私足迹**：查询本地泛化(剥离个人上下文→客观实体词) + 剥离追踪参数(utm_*/spm/fbclid) + 空 Referer + 禁 cookie + gzip；gemini补 DoH 防 ISP DNS 窥探 | ✅ | ✅ | ✅ |
| K6 | **信噪比优先级**：P0 文档/百科/arxiv(>85%) > P1 论文 > P2 专业媒体 > P3 新闻 > **P4 社交/论坛(<15%，对0.5B是毒药，坚决排除)** | ✅ P0–P4 表 | ✅ 同序 | ✅ 同序 |

### 8.2 关键分歧 / 大陆部署补强

- **搜索引擎在大陆可达性**：gemini+qwen 明确点出 GFW——Google 直连不可达、Bing 国内可达但 bot 检测、DDG 中文弱；deepseek 偏软。**⇒ 我们的部署在大陆**：白名单直连 + **Bing Web Search API(免费层) / DDG-lite 作发现**，SearXNG 仅作远期可选（非 MVP）。
- **SearXNG 定位**：gemini「备选/200MB+开销」、deepseek「维护复杂」、qwen「上游仍见查询」。**⇒ 推迟，MVP 不引入**。
- **Token 估算跨度**：gemini 严估 45k–60k 高质量/天；qwen/deepseek 宽估 20k–200k。**⇒ 采用 gemini 的 45k–60k 作规划数**（与 0.5B 短时微调/记忆回放最佳吞吐区间吻合，不触发灾难性遗忘）。
- **预算口径**：qwen 指出「50 次搜索 + 每搜 2–3 页」10MB 不够 ⇒ **采用 10 次搜索 + 40 次白名单直连（或 50 次 lookup 总数）** 的分配。

### 8.3 最大风险点（三模型收敛到 2 个）

1. **数据质量 / 投毒**：0.5B 缺乏常识过滤，SEO 伪原创(车轱辘话)与「静默 HTTP200 验证码页」会被 trafilatura 当正文抽走 → 认知漂移 / 语言退化。**⇒ 直接对应我们 Gate A(熵/零宽) + B(注入签名) + C(distill_gate JSD) + D(holdout回测)；P4 社交源硬排除是首道防线。**
2. **搜索入口稳定性 + JS 墙覆盖**：大陆高价值内容(知乎/微信)受登录/JS 墙，静态抽取失败；若被迫无头则触碰风控+预算崩。**⇒ 用白名单直连覆盖可静态取的高值源，搜索只作「发现」，接受覆盖率上限。**

### 8.4 我们的爬取层裁定（写入 MVP）

- **最小栈**：`pip install httpx[http2] trafilatura readability-lxml` + 标准库 `urllib.robotparser` / `sqlite3` / `simhash`。**不装 Playwright、Scrapy、SearXNG（MVP）。**
- **搜索**：白名单直连 80%（维基 API / 百度百科 / GitHub docs / arXiv abs / MDN / ReadTheDocs 静态路由表）+ Bing Web Search API 免费层 / DDG-lite 20% 发现。
- **反爬**：高斯抖动 1.5–3.5s、HTTP/2 + 完整 Sec-CH-UA 头矩阵（见 gemini 代码）、空 Referer、禁 cookie、robots 缓存、验证码命中域名熔断 24h。
- **隐私**：查询泛化 + 参数剥离 + 空 Referer + (可选) DoH；对接 §3 的 DLP 脱敏网关。
- **源优先级 P0–P4 作 Gate 前置硬过滤**（P4 直接丢弃，不入池）。
- **预算**：≤50 lookup/天（10 搜 + 40 直连），≤10MB。
- **单页纯 CPU 活跃 ≤40ms**（gemini 实测拆解），单核每天清洗 50 篇不干扰 0.5B 推理/训练。

### 8.5 待排入的后续交叉轮次（用户「各种各样的」）

- **R3 防 subliminal / 间接提示注入的具体工程防线**（与 E1 衔接：holdout 抓漂移但抓不到 stealth，门控怎么补）。
- **R4 D1 去污染/漂移检测算法对比**（MinHash vs embedding-KNN vs 5-gram 在中文的鲁棒性 + 阈值校准）。
- **R5 外部大脑协议安全**（schema 强制 + 签名校验 + 防「云返回本身注入」）。
- **R6 node_v5 状态机精确接口**（事件总线/文件锁/Unix socket 与 fetch 组件的契约）。
- 全部落盘后，等用户「全自动」授权再进入**代码级验证**（本地先不实验）。

---

## §9 总结论（两轮交叉后）

两轮独立交叉（架构 + 爬取）共同坐实：**联网自学组件不是新范式，是现有 node_v5 信箱 + distill_gate 分歧门控 + data-plan holdout 的「前移+补强」**——闸门从训练期前移到入网抓取期，机密边界补强到「私人记忆 embedding 也不出」，爬取层用白名单优先 + 轻量 trafilatura 栈（绝不无头）。收敛风险始终是「数据质量投毒」与「大陆高值源 JS 墙」，二者都靠「P4 硬排除 + 多道 Gate + 外部大脑仅当方向给出者」兜住。

---

## §10 V1 最小闭环落地（2026-10-14 补记：工程已跑通）

用户从 GPT 取得最小闭环设计，本地与本架构对齐后已落工程。**三方独立收敛**（本地/GPT/gemini+deepseek+qwen）
于「抓取→标准化→信任标记→分层存储」「清洗≠去污染」「Gate7=架构期硬门」三点，视为已交叉验证。

- **规格**：`docs/netarch/mvp_closed_loop_spec_v1_20261014.md`（+ .html）。四模块统一 = 联网获取 → 10M记忆积累调用 → RSI自主迭代 → Gate7验证可安全改删。
- **联网组件 V1 已实现**（`netarch_v1/netarch.py`）：管道状态机(PENDING→…→STORED/RETRYABLE_FAIL/FAILED_PERMANENT,有限重试5/20/60s) + 抓取(HTTP/JSON/RSS/静态HTML) + 清洗(trafilatura/feedparser 有则升级、无则 stdlib 退化) + 去污染(8类提示注入/隐藏字符标记,不删内容留证据) + 分层存储(documents/provenance/processing_logs/errors, content_hash去重, raw可重放)。默认 trust_level=untrusted 且**永不自动升格**。实测抓 RFC/RSS/JSON 通过。
- **Gate7 已实现且7/7实测通过**（`netarch_v1/gate7_deletion.py`）：合成数据红线；删除状态机 REQUESTED→BLOCKED_FROM_RETRIEVAL→DELETION_PROPAGATING→VERIFYING→VERIFIED/FAILED_*；幂等；G7-01~07 全过。
- 🔴**G7-05 为「诊断项」非彻底通过**：删除屏障靠 tombstone=1 字段，过期旧任务若执行 UPDATE tombstone=0 可在应用层绕过。真正堵死需 DB 层约束(触发器禁1→0 或 append-only 删除日志)——已作V1-02 入队，不粉饰为已解决。
- **10M 记忆 schema 随 Gate7 落地**：memories(含tombstone)+全文索引+向量索引+摘要+备份(标记式删除)+缓存+deletion_log；分工=权重装函数/记忆装事实/嵌入认出你。
- **RSI 闭环待写**（等 V1-04 裁决「独立评估器不作弊」）。V1 第一版只允许改沙箱，主线只读。
- **依赖**：venv 已装 trafilatura/feedparser/readability-lxml/lxml 等；管道对它们做可选依赖(缺失退化stdlib)。
- **开放问题入队**(`docs/netarch/questions/fleet_wave2.json`，守护进程自动参战)：V1-01三值CPU可行性(生死判据)/V1-02 Gate7并发复活堵死/V1-03 信任晋升阈值+subliminal/V1-04 RSI独立评估器/V1-05 单遍vs多epoch在线判据。
