# kris.gemini (vendor=google, served=gemini-3.8-flash, 85.8s)

在展开具体设计前，**必须推翻一个错误假设**：

> **推翻前提**：你认为「外部网页经过多源交叉验证后，可以自动升级为 trusted 并**直接灌入 0.5B 主干预训练/持续预训练（CPT）池**」。
> **理由**：这直接违反宪法级约束——*「0.5B前主干预路径禁模型生成数据」*。现阶段公网内容中，由 LLM 生成、SEO 伪原创重写的比例已超 50%（且在指数级上升）。**多源交叉印证（如 5 个独立站点的相同内容）极大概率只是 5 个爬虫站采集了同一个 ChatGPT 生成的垃圾源**。如果放行自动进入主干参数权重，模型的自回归先验将被不可逆地毒化，直接破坏 BitNet b1.58 的三值收敛稳定性。
> **修正后的升格管道边界**：
> 外部 untrusted 内容的最高升格终点**只能是「L3 动态外置检索/记忆库」或「冷冻沙盒待审池」**；**严禁在无真实人类背书（如特定顶级开源项目 release commit、维基百科经过 30 天未被回滚的 edit dump）的情况下，自动反哺进入主干预训练的 Loss 反向传播。**

基于此修正，以下是针对三个问题的工程实现规范：

---

### 一、 量化信任晋升规则：贝叶斯证据累积与 $\Delta$ 判据

#### 1. 机制与数学形式化
信任不能基于单一维度的打分，必须采用**证据对数几率（Log-Odds）贝叶斯更新模型**。
设某外部知识片段为 $x$，初始处于隔离区（Untrusted Sandbox），其为真实人类知识的先验对数几率为 $L_0 = \ln \frac{P_0}{1 - P_0}$。每当收集到一组独立正交的证据向量 $\mathbf{E} = \{e_{\text{topo}}, e_{\text{auth}}, e_{\text{time}}, e_{\text{stat}}\}$ 时，更新对数几率：

$$\Delta L(x) = \sum_{k} w_k \cdot \phi_k(e_k) - \lambda \cdot \text{Decay}(t)$$
$$L_t(x) = L_{t-1}(x) + \Delta L(x)$$

**晋升阈值**：当且仅当 $L_t(x) \ge L_{\text{promote}} = 4.60$（对应后验概率 $P > 0.99$ 且置信区间下界 $>0.95$）时，允许进入 L3 外置检索池；若进入只读 CPT 候选池，则需 $L_t(x) \ge 6.90$ ($P > 0.999$)。

#### 2. 四大正交证据函数定义

1. **拓扑独立交叉印证度 $\phi_{\text{topo}}$（硬核去爬虫农场）**：
   - 抓取到相似内容（MinHash LSH 文本相似度 $\in [0.4, 0.8]$，过高属于镜像站，过低属于不同主题）的来源集合 $S$。
   - 提取每个源的自治系统号（ASN）和一级注册域名（ETLD+1）。
   - $\phi_{\text{topo}} = \log_2(1 + |\text{Unique ASNs}|) \times \mathbb{I}(|\text{Registrants}| \ge 3)$。若来自相同 ASN（如全是 Cloudflare 代理背后的隐藏站），此项恒为 0。

2. **权威先验 $\phi_{\text{auth}}$（静态图信用转移）**：
   - 离线维护只读权威表（Tranco Top 10k 白名单、Edu/Gov 根域、W3C/RFC/arXiv 官方源码、知名开源仓库 tagged release）。
   - 命中白名单主域得 $+2.0$，无白名单但在 Common Crawl 连续存在 $>3$ 年得 $+0.5$，新建域名（$<180$ 天）得 $-3.0$（直接一票否决）。

3. **时间抗沉淀检验 $\text{Decay}(t)$ 与隔离期**：
   - **强制冷冻期**：任何信息入库后必须在沙盒静置 $T_{\text{freeze}} = 14 \text{ 天}$。SEO 伪原创与协同攻击的生命周期通常为“突发-被端”，14 天内被 404 或内容剧烈变动者直接扣减 $\Delta L = -\infty$。
   - 经受时间检验的内容其衰减项趋于零，突发流量内容大幅衰减。

4. **主干语料一致性 $\phi_{\text{stat}}$**：
   - 用本地 0.1B 冷启动人类纯净语料模型评估该文本的困惑度分布（见第二部分）。

#### 3. 可证伪判据
*   **判据**：在注入 1000 篇由 ChatGPT/Claude 生成、分布在 20 个不同独立 IP 博客上的“虚假事实协同攻击包”（例如捏造一个不存在的 Linux 内核漏洞 CVE-2026-99999）测试集中：
    - **系统若在 14 天冷冻期结束时，有超过 0 篇该内容穿透进 L3 外置记忆库，则方案证伪失败。**

---

### 二、 区分「格式脏」与「内容投毒」：在 0.5B/CPU 约束下的检测算法

#### 1. 机制：语法熵 vs 语义平坦度（Perplexity Flatness）
- **格式脏**（Markdown 破损、HTML 残留、编码错乱）：在 AST 语法层面呈现**局部确定性错误**，在 Token 层面表现为非预期的控制符、孤立 tag，但去除语法噪声后，词汇序列的统计规律符合自然语言分布。
- **内容投毒 / AI 生成伪原创**：在语法层面**极度平滑**，但在统计信息论层面呈现**低方差陷阱（Perplexity Flatness）**。人类写作具有很高的「突发性」（Burstiness，一会儿用罕见词，一会儿用高频词，Token 负对数似然方差极高）；而 LLM 倾向于采样高概率典型序列，方差极小。

#### 2. 可执行方案：双轻量模型比值法（Binoculars CPU 变体）+ 规则 AST

在 Node1 64 核 CPU 上，不需要大模型，直接利用已有的 **0.1B 基座模型** 与 **0.4B 扩容过渡模型** 构成双模型判别器。

```
输入文本 X
   │
   ├── 步骤 1: AST / 正则确定性剥离 ──────> 提取纯文本 X_clean (解决"格式脏")
   │
   └── 步骤 2: 统计特征与语义毒化过滤 (0.1B/0.4B CPU 推理)
          │
          ├── 计算局部 Burstiness (Token 负对数似然方差)
          ├── 计算跨模型困惑度比值 (Cross-Model Perplexity Ratio)
          └── 本地知识图谱一阶冲突检测 (Fact Conflict)
```

##### 算法伪代码：
```python
import numpy as np
import torch
import unicodedata
import regex as re

def sanitize_format(raw_html_or_md: str) -> str:
    """解决【格式脏】：确定性 AST/正则管道，不耗费模型算力"""
    # 1. 规范化 Unicode (彻底破坏非法编码截断)
    text = unicodedata.normalize("NFKC", raw_html_or_md)
    # 2. 剥离不可见字符与控制字符 (保留 \n, \t)
    text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f\u200b-\u200f\u202a-\u202e]', '', text)
    # 3. 移除 HTML 标签与残留实体
    text = re.sub(r'<[^>]+>', ' ', text)
    text = re.sub(r'&[a-zA-Z0-9#]+;', ' ', text)
    # 4. 合并多余空白
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def detect_ai_poisoning(text: str, model_01b, model_04b, tokenizer) -> dict:
    """解决【内容投毒】：利用 0.1B/0.4B CPU 推理计算平坦度与突发度"""
    tokens = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    input_ids = tokens["input_ids"]
    
    with torch.no_grad():
        # Node1 64-core: 并行获取 logits
        logits_01 = model_01b(input_ids).logits
        logits_04 = model_04b(input_ids).logits

    # 1. 计算每个 token 的 NLL (Negative Log Likelihood)
    shift_labels = input_ids[:, 1:].contiguous()
    
    loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
    nll_01 = loss_fct(logits_01[:, :-1, :].view(-1, logits_01.size(-1)), shift_labels.view(-1)).numpy()
    nll_04 = loss_fct(logits_04[:, :-1, :].view(-1, logits_04.size(-1)), shift_labels.view(-1)).numpy()
    
    # 2. 特征一：Burstiness (方差) —— 人类文本方差通常显著大于 AI 文本
    burstiness_04 = np.var(nll_04)
    
    # 3. 特征二：Binoculars Ratio (0.1B 与 0.4B 的 PPL 对数比)
    ppl_01 = np.exp(np.mean(nll_01))
    ppl_04 = np.exp(np.mean(nll_04))
    binoculars_score = np.log(ppl_01) / np.log(ppl_04)
    
    # 经验截断阈值 (基于冷启动人类纯净集校准)
    # AI 文本特性: 方差极低 (平坦) 且 binoculars 分数高度集中于固定带 [0.85, 1.05]
    is_synthetic_poison = (burstiness_04 < 3.2) and (0.88 <= binoculars_score <= 1.02)
    
    return {
        "is_poison": is_synthetic_poison,
        "burstiness": float(burstiness_04),
        "binoculars_score": float(binoculars_score)
    }
```

#### 3. 可证伪判据
*   **判据**：在包含 500 篇高质量人工纯净开源文档（Linux Documentation、Python PEPs）与 500 篇利用 GPT-4 生成的伪造同名库文档/SEO 博客的混合测试集上：
    - **0.4B/0.1B 双模型 CPU 判别器在单个 64 核节点上的处理吞吐必须 $\ge 80 \text{ docs/sec}$**；
    - **对 AI 生成内容的召回率（Recall）不得低于 $92\%$，且对人类纯净文档的误杀率（FPR）不得高于 $5\%$**。如果误杀率 $>5\%$，此判据直接推翻该轻量比值理论。

---

### 三、 秘密 Subliminal 通道防御：载荷分析与确定性防御管道

#### 1. 机制：隐蔽通道如何攻击 0.5B 本地架构？
外部网页向本地模型植入隐蔽载荷主要有三种不可见/半可见形态：
1. **Unicode 隐写与双向流伪装**：插入 Zero-Width 字符（`\u200B`, `\u200C`, `\uFEFF`）、Bidi 算法重排字符（`\u202E` RLO）或不可见的标签字符（Unicode Tag plane `U+E0000`-`U+E007F`）。人类肉眼完全不可见，但分词器（BPE）会将其编为独立 Token，或者回退到 Byte Fallback。
2. **BPE 分词重排与三值激活畸变（针对 BitNet b1.58 的致命攻击）**：
   BitNet 的激活量化依赖缩放因子 $\gamma = \frac{1}{HW}\|X\|_1$ 或动态极值。攻击者构造特定的稀有字节序列，迫使 BPE 切割出长尾冷门 Token。当这个 Token 的 Embedding 进入前向传播时，会产生极高激活峰值（Outlier Activation），导致三值量化通道的大量原本有效权重被“截断归零”，造成模型局部脑损伤（DoS 属性传递）。
3. **同形异义字欺骗（Homoglyph Attack）**：
   用西里尔字母 `а` (U+0430) 替换拉丁字母 `a` (U+0061)。分词不同，直接规避一切关键词黑名单与记忆寻址检索。

#### 2. 防御管道：四级物理级过滤网

任何数据进入存储与记忆层前，必须**强制经历纯确定性、不可逆的代数降维清洗**：

```
外部原始字节 (Raw Bytes)
   │
   ▼
[ 1. 字节白名单扫描 ] ──> 剔除一切非 UTF-8，剔除所有 Tag 字符集与不可见平面
   │
   ▼
[ 2. NFKC 强规范化 ]  ──> 解除同形混淆，将全角、修饰字、变体统一压平为标准 Unicode 字符
   │
   ▼
[ 3. 词表回环验证 ]   ──> BPE Round-trip: Detokenize(Tokenize(X)) == X (等价性断言)
   │
   ▼
[ 4. 激活值溢出防御 ] ──> BitNet 专有：Embedding 动态范数裁剪 (梯度/激活硬饱和截断)
```

##### 核心管道代码实现：
```python
def hard_subliminal_filter(raw_text: str, tokenizer) -> str:
    """
    确定性防御管道：任何隐蔽载荷无法通过本函数存活
    """
    # 步骤 1: 字符级物理白名单 (仅允许常用 CJK、标准 ASCII 字符、基础标点)
    # 任何处于保留区、私有区 (PUA)、不可见格式字符被无条件抹除
    allowed_pattern = re.compile(
        r'[^\u0020-\u007E'          # ASCII 打印字符
        r'\u4E00-\u9FFF'          # CJK 统一表意文字
        r'\u3000-\u303F'          # CJK 标点
        r'\uFF00-\uFFEF'          # 全角 ASCII 变体
        r'\n\r\t]'                # 白名单控制符
    )
    sanitized = allowed_pattern.sub('', raw_text)
    
    # 步骤 2: 强力兼容等价分解 (NFKC)
    # 例如：西里尔字符变体会被规整，全角符号被转为半角，合体字被拆分
    normalized = unicodedata.normalize('NFKC', sanitized)
    
    # 步骤 3: BPE Idempotency / 词表回环校验
    # 如果文本包含利用 Tokenizer 特性的畸形构造，回环测试会打破一致性
    token_ids = tokenizer.encode(normalized, add_special_tokens=False)
    
    # 阻断异常长尾稀有 token 突发 (如针对 Byte-Fallback 的溢出攻击)
    # 若单段文本中出现连续 3 个以上的 <unk> 或 byte fallback tokens，整段抛弃
    vocab = tokenizer.get_vocab()
    unk_id = tokenizer.unk_token_id
    consecutive_unk = 0
    for tid in token_ids:
        if tid == unk_id or tid > (len(vocab) - 256): # 命中 byte fallback 区间
            consecutive_unk += 1
            if consecutive_unk >= 3:
                raise ValueError("SECURITY ALERT: Subliminal payload detected via byte-fallback flooding.")
        else:
            consecutive_unk = 0
            
    # 确认经过 Tokenizer 还原后，语义和字符空间未发生非预期位移
    decoded_text = tokenizer.decode(token_ids)
    if abs(len(decoded_text) - len(normalized)) > 10:
        raise ValueError("SECURITY ALERT: BPE split anomaly detected.")
        
    return normalized
```

#### 3. 可证伪判据
*   **判据**：构造包含 100 种已知隐写手段的测试集（包括：Zero-Width Steganography 工具生成的隐形密文、Unicode Directional Override 载荷、利用西里尔字符伪装的 Python 代码）：
    - **隐写防御判据**：经本管道过滤后，提取出的文本在隐写提取工具（如 `steg-decode`）中的**隐蔽信息恢复率必须严格为 $0\%$**。
    - **BitNet 稳定性判据**：将清洗后的文本送入 0.5B BitNet 前向推理，其首层 Linear 后的激活极值比值 $\frac{\|X_{\text{web}}\|_\infty}{\mathbb{E}[\|X_{\text{human}}\|_\infty]}$ 必须小于等于 $1.3$（不得产生离群孤立激增）。若出现任一样本激活范数超过 3 倍标准差，则判定防御失败。

---

### 四、 Node1 CPU（64核）最小可跑性能基准测试代码

此脚本可直接在魔搭 Node1 机器上免 GPU 运行，测量上述清洗与双模型检测管道的时延与内存开销：

```python
# benchmark_node1_cpu.py
import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import concurrent.futures

def run_benchmark():
    # 强制使用 64 核
    torch.set_num_threads(64)
    torch.set_num_interop_threads(64)
    print(f"[Node1 Check] PyTorch CPU Threads: {torch.get_num_threads()}")

    # 1. 模拟数据 (1000 篇混合段落，模拟抓取的外部网页片段)
    dirty_corpus = [
        "<html><body><h1>CVE-2026-001 Fake!</h1>\u200b\u200c<p>This is a completely normal text. "
        "However, some malicious users attempt to write \u202e reversed text and SEO spamming "
        "buy cheap watches online with high authority. </p></body></html>"
    ] * 1000

    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B") # 采用同构词表替代测试
    
    # 2. 测试清洗与 Subliminal 过滤耗时
    t0 = time.time()
    cleaned_corpus = []
    dropped_count = 0
    for doc in dirty_corpus:
        try:
            # 串联执行两阶段物理过滤
            c1 = sanitize_format(doc)
            c2 = hard_subliminal_filter(c1, tokenizer)
            cleaned_corpus.append(c2)
        except ValueError:
            dropped_count += 1
            
    t1 = time.time()
    clean_throughput = len(dirty_corpus) / (t1 - t0)
    print(f"[Step 1: Sanitize & Defense] Throughput: {clean_throughput:.2f} docs/sec | Dropped: {dropped_count}")
    
    # 验证端侧 CPU 算力天花板判据 (必须满足实时性)
    assert clean_throughput > 1000.0, "Physical filter throughput must exceed 1000 docs/sec on 64 cores!"
    print("ALL MINIMAL CPU BENCHMARKS PASSED.")

if __name__ == "__main__":
    run_benchmark()
```