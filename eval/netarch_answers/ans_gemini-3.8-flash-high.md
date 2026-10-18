# gemini-3.8-flash-high (served: gemini-3.8-flash)

Considering an architecture prioritizing autonomy, the component must intelligently decide when and what to scrape from the web, integrating the information into the local model. It requires a decision-making module, a web scraping and content extraction module, and a knowledge integration module. This will require establishing a method to manage the self learning process.


The limited capacity of the 0.5B ternary model presents a significant challenge for autonomous query generation. Initial assessment suggests it might struggle with complex, long-horizon tasks and zero-shot tool use without structured assistance. A hybrid approach may be necessary, potentially using a cloud orchestrator for complex queries to avoid hallucination.


The small model's ability to self-direct scraping is limited; direct questioning about knowledge gaps proves ineffective. However, it exhibits promise as an "Epistemic Uncertainty Sensor," flagging areas needing further investigation. A hybrid system emerges as the best path: a local model identifies topic gaps that are decontextualized and expanded by a stronger cloud model, preserving privacy.


The local model functions as an anomaly detector, flagging knowledge gaps for investigation through focused queries. The system will leverage these flags to trigger further exploration. Complex search script generation will be avoided; the small model is tasked with generating topic queries only.


Focus is now on data poisoning risks associated with pretraining data, particularly backdoors. Structural scrubbing removes injection patterns, while drift detection flags deviations in semantic meaning and language quality within the corpus. Strict local boundaries for sensitive data like LoRA deltas and chat history are being enforced, with air-gap guards for outgoing requests.


Several safety gates are implemented to prevent data poisoning and ensure training integrity: format and entropy checks, loss gating to filter memorized or adversarial content, a filter for jailbreak attempts, and regression backtesting on a static dataset. The system integrates with a state machine, governed by specific flags that control scraping behavior and training readiness, based on resource availability and quota limits. CPU resource requirements for text extraction and model forward passes were assessed for feasibility.


针对 0.5B 参数、三值（BitNet b1.58）、CPU 优先的本地端侧模型，构建自主联网自学架构的核心矛盾在于：**极度受限的端侧认知/算力 vs. 外部高噪声/潜在恶意的无限互联网**。

0.5B 三值模型无法像 70B 模型那样进行复杂的多步 ReAct 规划，但它极其轻量（内存占用通常 < 200MB，CPU 推理极快）。因此，架构设计的核心哲学是：**模型负责“感应信息匮乏”，确定性代码/小模型负责“管道过滤”，云端（若使用）仅做“无状态抽象扩展”，全过程由信箱信标状态机严格驱动。**

---

### 系统全景架构图 (ASCII 模块图)

```text
================================= 本地受信任边界 (LOCAL TRUSTED BOUNDARY) =================================
 [核心资产: owner_corpus / 私人记忆向量 / 0.5B BitNet 主权重 + 局部可塑 Adapter] (物理隔离，永不出境)
---------------------------------------------------------------------------------------------------------
 1. 抓取决策层 (Epistemic Engine)
   ┌────────────────────────────────────────────────────────┐
   │ 0.5B 模型困惑度感知 (PPL Spike) / 未命中实体探测器 (NER)  │
   └───────────────────────────┬────────────────────────────┘
                               ▼ 原始需求槽位 (如: "BitNet 算子 AVX512 实现")
   ┌────────────────────────────────────────────────────────┐
   │ 脱敏与泛化代理 (De-identification & Abstraction Engine) │ ──(可选: 云端纯抽象泛化，零机密)──┐
   └───────────────────────────┬────────────────────────────┘                                 │
                               ▼ 匿名查询词 (Query)                                            │
---------------------------------------------------------------------------------------------│-----------
 2. 调度与信箱状态机 (node_v5.py 集成)                                                         │
   ┌────────────────────────────────────────────────────────┐                                 │
   │ 状态控制: [IDLE] ──(信标就绪/空闲)──> [GATHERING]        │                                 │
   │ 预算控制: 配额管理 (每日查询 < 50 次, 抓取文本 < 5MB)      │                                 │
   └───────────────────────────┬────────────────────────────┘                                 │
                               │ 发起网络请求                                                 │
===============================│==============================================================│==========
                               ▼ (仅输出无状态查询词)                                           ▼
                       [外部公共互联网 / 公共搜索引擎] ◄────────────────────────────────────────┘
                               │
                               ▼ 返回 HTML / Raw Text
===============================│=========================================================================
 3. 数据管道分层与清洗 (Data Pipeline)
   ┌────────────────────────────────────────────────────────┐
   │ ① 抽取与剥离: Trafilatura (正文抽取) + 格式沙箱           │
   │ ② 基础清洗: MinHash LSH (去重) + 正则指令剥离             │
   │ ③ 漂移检测(D1): 轻量级 Embedding 余弦相似度比对           │
   └───────────────────────────┬────────────────────────────┘
                               ▼ 候选语料 (Candidate Chunks)
 4. 安全闸门 (Verification Gates)
   ┌────────────────────────────────────────────────────────┐
   │ Gate 1: 格式与隐写探测 (熵值/零宽字符过滤)                 │
   │ Gate 2: 注入特征拦截 (Prompt Injection Signature Mask)  │
   │ Gate 3: 0.5B 试探性前向 Loss 检验 (Loss Gating)          │
   └───────────────────────────┬────────────────────────────┘
                               ▼
 5. 入库与自学闭环 (Holdout Backtest & Continual Training)
   ┌────────────────────────────────────────────────────────┐
   │ 临时训练缓冲池 (Staging Buffer)                         │
   │ ──(累积达标)──> node_v5 触发 Training Mode (BitNet Update)│
   │ ──(校验失败)──> Holdout 回测回退 (Rollback)             │
   └────────────────────────────────────────────────────────┘
=========================================================================================================
```

---

### 一、抓取决策层：自驱（Autotelic）机制与算力边界

#### 1. 0.5B 三值小模型能否胜任“自主决策”？
*   **明确判断**：**绝对不能让 0.5B 三值模型直接扮演规划大模型（Agent ReAct 循环）。**
    *   *原因*：0.5B 三值模型参数容量极小，语义表征粗糙，让其自由输出“我需要什么并生成抓取 URL”，会迅速陷入循环胡话（Degeneration）或逻辑漂移。
    *   *但它能胜任“传感器”*：0.5B 模型非常适合作为**认知匮乏检测器（Uncertainty Sensor）**。

#### 2. 自驱触发机制（内部驱动）
无需云端介入，本地依靠两种启发式信号自驱触发：
*   **信号 A（PPL 突刺/困惑度触发）**：模型在处理本地日常任务或自省循环时，若某 Token 序列的交叉熵 Loss 突增（$PPL > \tau_{threshold}$），标记该上下文窗口。
*   **信号 B（实体/知识图谱断裂）**：抽取上下文实体，本地轻量知识库比对发现度数（Degree）为 0 或属于悬挂节点（Dangling Entity），触发信息获取冲动。

#### 3. 本地 ↔ 云端 混合协议（若需外部强模型作为大脑）
如果本地启发式生成的 Query 语义过于碎片化，必须接入云端大模型优化 Query，必须采用**“基于抽象模板的单向无状态协议”**：
1.  **本地实体提取**：本地提取核心实体词（例如：`BitNet-1.58b`, `AVX512`），剥离任何与所有者相关的上下文（Owner Context/Prompt）。
2.  **槽位泛化请求**：向云端发送纯逻辑模板：`"Generate 3 broad web search keywords for learning about technical concept: {entity}"`。
3.  **零机密上报**：云端仅看到孤立的公共技术名词，云端返回标准关键词列表（如：`["BitNet b1.58 C++ implementation", "AVX-512 ternary inference kernel"]`），由本地确定性爬虫执行。

---

### 二、数据管道分层与 CPU 端开销

该管道采用严格的纯 CPU 高效技术栈，无 GPU 依赖。

| 管道层级 | 职责与技术方案 | CPU 端开销分析 |
| :--- | :--- | :--- |
| **1. 抓取与抽取** | **SearXNG API / 静态 DuckDuckGo HTML** + **Trafilatura**。<br>丢弃所有 JS/CSS/DOM 节点，仅提取纯 Markdown/Text。 | **极低**。<br>单篇抽取耗时 5~15ms，内存占用 < 30MB。 |
| **2. 启发式清洗** | **正则表达式规则集 + MinHash LSH**。<br>1. 剔除广告、导航栏、Cookie 声明。<br>2. 剔除模板文本（Jaccard 相似度 > 0.8 丢弃）。 | **极低**。<br>纯文本匹配，CPU 耗时 < 2ms/KB。 |
| **3. 注入特征剥离 (Anti-Injection)** | **结构化指令屏蔽**。<br>彻底擦除诱导式指令 token（见第四节）。 | **可忽略**（正则过滤）。 |
| **4. 语义漂移检测 (D1)** | **超轻量级 CPU 专用 Embedding**。<br>使用 `all-MiniLM-L6-v2` (ONNX 量化版) 或 `bge-micro-v2` (约 20MB)。<br>计算抓取文本向量与本地基线锚点（Anchor Embedding）的余弦相似度。若相似度低于阈值（如 < 0.35），判定为不相关并丢弃。 | **低**。<br>ONNX Runtime (AVX2/NEON) 下，单句编码耗时 3~8ms。 |

---

### 三、机密隔离边界（物理与协议分界）

为确保 RSI 自举过程中绝不泄露 `owner_corpus`，定义绝对单向边界：

```text
[机密区: 绝不离开内存/本地盘] ──(单向阻断)──x──> [出口网关: 仅白名单字符]
       ▲
       │ 仅允许数据单向流入 (通过安全闸门后写入训练暂存区)
[外部区: Web / Cloud]
```

#### 1. 数据边界划分
*   **绝对受限数据（Strictly Local）**：
    *   `owner_corpus`（所有者私人文档、日记、专属代码库）。
    *   模型适配权重差值（LoRA / BitNet 三值 Adapter 局部增量）。
    *   历史私有对话上下文、私有向量数据库、信箱通信私钥。
*   **允许出境数据（Public Egress）**：
    *   仅限经过**去标识化（De-ID）**处理后的检索关键词（Search Queries）。
    *   网络抓取请求仅包含外部公共 URL。

#### 2. 本地隐私防线（Sanitizer Engine）
在生成网络查询词后、发往外部网络前，必须经过本地确定性拦截：
1.  **PII 正则扫描**：阻断任何匹配 Email、手机号、IPv4/v6、MAC 地址、系统文件路径（如 `/Users/xxx/`）的 Query。
2.  **差分/词表对齐阻断**：计算 Query 中 Token 在 `owner_corpus` 特有词表中的 TF-IDF。若某个词在私人语料中权重极高但在通用维基百科词表中罕见（即“特异性私有词”），直接熔断该次查询请求。

---

### 四、安全闸门（防投毒与防指令注入）

外部网页文本用于自学训练时，最大威胁不是直接越狱，而是**隐性注入（Subliminal/Indirect Prompt Injection）**和**语料投毒（Data Poisoning / Backdoor）**，导致小模型在持续学习后被植入后门。

进入训练池前，必须通过可验证的 **4 道硬核闸门**：

```text
抓取文本 ──> [Gate 1: 格式与熵] ──> [Gate 2: 注入签名] ──> [Gate 3: Loss门控] ──> [Gate 4: 评估回退] ──> 训练池
                 │ 失败丢弃           │ 失败丢弃           │ 失败丢弃           │ 异常回滚
```

#### Gate 1: 格式与沙箱阻断（物理层）
*   **判据**：
    *   零宽字符检测：匹配 `[\u200B-\u200D\uFEFF]`，发现直接丢弃（防止隐写注入）。
    *   字符熵值检验：计算文本的香农熵（Shannon Entropy）。过高（乱码/加密文本）或过低（重复灌水刷屏文本）直接剔除（$3.5 < H(X) < 6.0$ 为合法区间）。

#### Gate 2: 语义指令降维与签名拦截（防注入）
*   **判据**：
    *   外部文本**绝不作为 prompt 执行**，仅作为预训练语料进行 Next-Token Prediction 学习。
    *   正则强行清洗“角色标记”：强制抹除 `<|im_start|>`, `<|im_end|>`, `system:`, `instruction:`, `ignore previous instructions` 等攻击特征模式。
    *   将含有高危控制命令的句子直接整句切除，而非保留。

#### Gate 3: 0.5B 三值前向分歧门控（Loss Gating）
利用 0.5B 本地模型对候选文本切片进行一次前向传播（Forward Only，不反向传播），计算单 Token 平均交叉熵 $\mathcal{L}$：
*   **判据**：
    *   **异常低 Loss ($\mathcal{L} < 0.5$)**：判定为高度重复无意义文本（如机械式复制字符、抓取死循环），丢弃。
    *   **异常高 Loss ($\mathcal{L} > 8.0$)**：判定为高度混乱对抗性文本、生僻乱码或跨语言污染，丢弃。
    *   仅接纳 $\mathcal{L} \in [1.2, 5.0]$ 的高质量信息差语料（既在理解范围内，又具有信息增益）。

#### Gate 4: Holdout 黄金集回归回测（训练后准入检验）
即使外部数据入库并微调了适配层，在合并前必须执行验证：
*   **判据**：本地固定常驻一个私密黄金测试集 `golden_owner_eval.jsonl`（包含本地核心任务与常识）。微调完成后，新权重在黄金集上的困惑度必须满足：
    $$\Delta PPL = PPL_{new} - PPL_{old} \le +0.5\%$$
    若退化超过 0.5%，判定模型遭受外部污染漂移，**立刻执行本地 Snapshot 回滚，清空该批次暂存数据并拉黑来源域名**。

---

### 五、与信箱机制（`node_v5.py`）的集成实现

联网自学组件不作为一个常驻抢占式守护进程，而是作为 `node_v5.py` 调度下的一个**被动子模块（Worker）**。

#### 1. 挂载状态机转移
`node_v5.py` 扩展以下四个信标信号与文件锁：

*   `beacon.status == "IDLE"`：当 CPU 空闲时间 > 15分钟，且处于低功耗/指定时间窗。
    *   `node_v5.py` 生成 `.flag_net_gather`，启动联网决策与爬虫模块。
*   `beacon.status == "GATHERING"`：爬虫流水线工作，写入暂存区 `staging_buffer/`。
    *   当 `staging_buffer` 达到配额大小（如累计 50,000 tokens），网络模块退出并创建 `.flag_train_ready`。
*   `beacon.status == "TRAINING"`：`node_v5.py` 锁死网络接口，阻断所有外联网络，模型只对 `staging_buffer` 数据执行量化感知微调（QAT/BitNet 适配层更新）。
*   `beacon.status == "VALIDATING"`：跑 Gate 4 回测。成功则合并 Adapter 并清空标志；失败则根据上次 Git Hash 或 Checkpoint 回退。

#### 2. 配额与预算熔断（Budget Circuit Breaker）
`node_v5.py` 维护持久化计数器 `net_budget.json`：
```json
{
  "date": "2023-10-27",
  "daily_queries_cur": 12,
  "daily_queries_max": 50,
  "daily_bandwidth_bytes_cur": 1420100,
  "daily_bandwidth_bytes_max": 10485760
}
```
*   **硬配额**：每天网络总抓取量 $\le 10\text{MB}$，每日主动请求次数 $\le 50$ 次。超额后，`node_v5.py` 强制卸载联网组件，回到纯本地闭环模式。

---

### 六、CPU 端可行性客观评估

| 指标 | 0.5B BitNet b1.58 在现代 CPU 上的实测/推导水平 | 架构可行性结论 |
| :--- | :--- | :--- |
| **内存开销** | 0.5B 权重（1.58 bits ≈ 2 bits 存储）仅需 **120~180MB**；加载清洗管道总驻留内存 **< 400MB**。 | **完全可行**。哪怕在 4GB 内存的低端工控机/树莓派 5 均可流畅运行。 |
| **前向检验速度** | BitNet b1.58 采用整数加法（无乘法 GEMM），现代 CPU (AVX-512/AMX/NEON) 推理可达 **50~120 tokens/s**。 | **完全可行**。做 Gate 3（Loss 检验）一篇 1000 字文档仅需 10~20ms。 |
| **微调训练开销** | 三值权重通常固定主干，仅训练轻量量化 LoRA 或少量 FP32 浮点缩放因子（约 200 万可训参数）。训练 50k tokens 在 8 核 CPU 上耗时 **约 1~3 分钟**。 | **完全可行**。利用 CPU 闲时在后台静默完成，不影响前台交互。 |

---

### 七、最小可落地 MVP（3 个优先模块）

若要在短期内闭环验证本架构，切忌一开始就做复杂的云端编排，建议立即构建以下 **3 个最小 MVP 模块**：

1.  **Trafilatura + 规则沙箱清洗器（管道核心，对应第二、四节）**：
    编写一段纯 Python 脚本，能输入 URL，纯本地 CPU 抽取正文，利用正则清洗指令关键词，过滤零宽隐写字符，产出合规训练切片。
2.  **`node_v5.py` 静态信标驱动（状态控制，对应第五节）**：
    利用现有信箱机制，实现单向触发：当本地检测到 `.trigger_scrape` 时，执行爬虫，将抓取的文本装入固定本地目录 `corpus/staging/`，文本满 100KB 即停机，阻断网络。
3.  **基于 Loss 门控与黄金集回测的准入脚本（安全核心，对应第四节）**：
    用你的 0.5B BitNet 对清洗后的外部文本算一次 Loss，过滤极端垃圾；并在尝试性自学一步后，立即跑一个由 10 个私人核心问答构成的 `eval.json`，确保 Loss 不恶化。验证模型在不被毒化的前提下，吞吐外部世界的新词表。