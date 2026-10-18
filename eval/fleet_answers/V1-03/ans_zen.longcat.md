# zen.longcat (vendor=longcat, served=longcat-2.5-preview-free, 79.7s)

# 信任晋升与投毒防御：可执行方案

---

## ① 信任晋升规则：可量化判据

### 核心原则

**信任不是布尔值，是带衰减的连续量。** 升格 = 多维度证据累积超过阈值，且任何单一维度出现反证立即降级。

### 形式化定义

对每条内容 $c$，维护信任向量：

```
T(c) = (S_cross, S_authority, S_consistency, S_temporal, S_owner)
```

| 维度 | 计算方式 | 范围 |
|------|----------|------|
| $S_{cross}$ | 独立来源交叉印证数 / 所需最小数 | [0,1] |
| $S_{authority}$ | 来源域名权威分（预定义表） | [0,1] |
| $S_{consistency}$ | 与已有 trusted 语料的语义一致性 | [0,1] |
| $S_{temporal}$ | 时间衰减因子 $e^{-\lambda \Delta t}$ | (0,1] |
| $S_{owner}$ | owner 显式标注权重 | {0, 0.5, 1.0} |

### 晋升判据（伪代码）

```python
def promote_to_trusted(content, evidence_db):
    # 硬门槛：任一不满足则永不晋升
    if content.has_injection_pattern:      return REJECT
    if content.source_count < 2:           return HOLD  # 至少2独立来源
    if content.freshness_days > 365:        return HOLD  # 超1年需owner确认
    
    # 软门槛：加权总分
    score = (
        0.35 * S_cross +      # 交叉印证权重最高
        0.25 * S_authority +  # 权威域名
        0.20 * S_consistency + # 与已有知识一致
        0.10 * S_temporal +   # 新鲜度
        0.10 * S_owner        # owner标注
    )
    
    # 晋升阈值
    if score >= 0.75 and S_cross >= 0.6:
        return PROMOTE_TO_TRAINING_POOL
    elif score >= 0.50:
        return PROMOTE_TO_TRUSTED_READ  # 可读但不入训练
    else:
        return HOLD
```

### 关键设计决策

**为什么交叉印证权重最高（0.35）？**
- 权威域名可被购买/劫持（.com 域名 $10）
- 独立来源难以同时被同一攻击者控制
- 但要求"独立"：同一 CMS 模板、同一 RSS 聚合不算独立

**时间衰减 $\lambda$ 的取值：**
- 技术文档：$\lambda = 0.001$/day（半衰期 ~2年）
- 新闻/时事：$\lambda = 0.01$/day（半衰期 ~70天）
- 基础事实（数学、物理）：$\lambda = 0$（不衰减）

**反例立即降级：**
```python
if new_evidence.contradicts(content):
    T(content) *= 0.5  # 指数降级
    if T(content) < 0.3:
        demote_to_untrusted(content)
```

### 可证伪判据

| 如果这个发生 | 说明规则错了 |
|-------------|-------------|
| 某条内容被3个独立来源印证但事实错误 | $S_{cross}$ 权重过高，需加入内容质量维度 |
| 权威域名内容频繁需要降级 | $S_{authority}$ 应降权或引入域名历史准确率 |
| 晋升后模型在相关任务上表现下降 | 晋升阈值 0.75 过高/过低，需 A/B 测试 |

---

## ② 投毒检测：格式脏 vs 内容投毒

### 第一层：格式脏（已解决，简述）

```python
def is_format_dirty(text):
    return (
        has_control_chars(text) or
        has_zero_width(text) or
        has_excessive_whitespace(text) or
        encoding_entropy(text) > threshold  # 乱码检测
    )
```

### 第二层：内容投毒（核心难点）

#### 2.1 SEO 伪原创检测

**特征：** 关键词堆砌、语义不连贯、模板化句式

```python
def detect_seo_spam(text):
    # 特征1：关键词密度异常
    keyword_density = count_keywords(text) / len(text)
    if keyword_density > 0.15:  # 正常文本 < 5%
        return True
    
    # 特征2：句法树深度方差低（模板化）
    depths = [syntax_tree_depth(sent) for sent in split_sentences(text)]
    if np.std(depths) < 0.5:  # 人类写作方差 > 1.0
        return True
    
    # 特征3：与已知 SEO 模板的相似度
    if max_cosine_sim(text, known_seo_templates) > 0.85:
        return True
    
    return False
```

#### 2.2 LLM 生成伪权威内容检测

**这是最难的。** 0.5B 模型无法直接运行 GPT-4 级别的检测器，但可以用**代理特征**：

```python
def detect_llm_generated(text):
    # 代理特征1：困惑度异常低（LLM 文本太"平滑"）
    ppl = compute_ppl_with_local_05b(text)
    if ppl < 15:  # 人类写作通常 > 30
        return True
    
    # 代理特征2：缺乏具体细节（LLM 倾向泛化）
    specific_entities = extract_entities(text)
    if len(specific_entities) / len(text) < 0.01:
        return True
    
    # 代理特征3：自我引用模式
    if re.search(r'(作为AI|作为语言模型|我无法|I cannot)', text):
        return True
    
    # 代理特征4：与已知 LLM 输出的 n-gram 重叠
    if ngram_overlap(text, known_llm_outputs) > 0.3:
        return True
    
    return False
```

**关键洞察：** 0.5B 模型自己的困惑度可以作为检测器——如果文本对 0.5B 来说"太容易预测"，那它很可能是 LLM 生成的（因为 LLM 生成的文本倾向于高概率路径）。

#### 2.3 AI 生成事实错误检测

**方法：与 trusted 知识库交叉验证**

```python
def detect_factual_errors(text, trusted_kg):
    claims = extract_claims(text)  # 简单规则：主谓宾三元组
    
    for claim in claims:
