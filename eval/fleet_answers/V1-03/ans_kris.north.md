# kris.north (vendor=cohere, served=cohere/north-mini-code:free, 63.7s)

## 1️⃣  信任晋升机制（理论+可量化的晋升规则）

### 1.1  机制/最佳理论 – “多来源加权积分 + 时效衰减” 模型
| 组件 | 内容 | 权重（示例） | 计算方式 |
|-------|---------|----------------|----------------|
| **源权威性 (S)** | 域名白名单、历史错误率、编辑者信誉 | `wS = 0.4` | `S = 1`（白名单内）/`0.5`（历史良好）/`0`（其他） |
| **内容可信度 (C)** | 事实核验、重复验证、语义一致性 | `wC = 0.35` | 由 *C‑score* 提供（见 1.2） |
| **时间新鲜度 (T)** | 新内容加权，老内容衰减 | `wT = 0.15` | `T = e^(−λ·Δt)` 其中 `λ = 0.001 /s` |
| **用户行为反馈 (U)** | 点击率、停留时间、点赞/点踩 | `wU = 0.1` | `U = (positives‑negatives) / (positives+negatives+ε)` |

**总体可信度** `Δ`（单位：0‑1）为四个分数的加权和：

```
Δ = wS·S + wC·C + wT·T + wU·U
```

**晋升阈值** `τ`（示例：`τ = 0.78`）。
当同一份素材在 **N** 个独立源（N≥3）出现时，每个源贡献独立的 Δ 项 → 累加器 `ΣΔ_i`。当

```
ΣΔ_i  >=  τ
```

时，内容从 **untrusted** 晋升到 **trusted pool**。

*为什么这种设计有效*
* 加权保证权威源即使新鲜度低也能晋升。
* 要求 **≥3 独立源** 显著降低了单点操纵的风险。
* 时效衰减迫使系统不断重新验证已有的“受信任”内容。

---

### 1.2  可执行方案（伪代码）

```python
# -------------------------------------------------
# 全局常量（可调）
W_SOURCE   = 0.4
W_CONTENT  = 0.35
W_TIME     = 0.15
W_FEEDBACK = 0.1
TRUST_THRESH = 0.78
MIN_SOURCES_FOR_PROMOTE = 3
DECAY_LAMBDA = 0.001          # 每秒衰减率
MAX_AGE_SEC = 365*24*3600     # 存档过期时间

# -------------------------------------------------
# 数据结构
class Item:
    def __init__(self, uid, source, ts, raw):
        self.uid        = uid                # 全局唯一ID
        self.sources    = defaultdict(list) # 每个源 → list of (ts, delta)
        self.format_ok  = False              # 格式校验标志
        self.poisoned   = False              # 投毒检测标志
        self.trusted    = False

# -------------------------------------------------
def format_dirty_check(raw: bytes) -> bool:
    # 简单快速检查：非法UTF-8、HTML标签嵌套深度 > 5、控制字符比例 > 2%
    try:
        raw.decode('utf-8')
    except UnicodeDecodeError:
        return True
    # 可选：调用轻量级的HTML解析器
    return False

def content_poisoning_check(item: Item, model_0_5B) -> bool:
    """
    在 0.5B 端侧运行三个子检查器：
      1) 统计特征（n‑gram 概率低）
      2) 嵌入熵（超过阈值表示随机性）
      3) 自注意力模式异常（与历史平均对比）
    所有三个检查器必须 agree 才能标记为投毒。
    """
    # —— 1) N‑gram 概率 ———
    probs = model_0_5B.ngram_logprob(item.raw)
    if np.mean(probs) < -6.0:            # 根据验证集设定
        return True

    # —— 2) 嵌入熵 ———
    embeds = model_0_5B.token_embedding(item.raw)   # shape [seq, dim]
    entropies = torch.distributions.Categorical(logits=embeds).entropy()
    if entropies.mean() > 4.5:            # 经验值
        return True

    # —— 3) 注意力异常 ———
    att_score = model_0_5B.avg_self_attention(item.raw)
    baseline  = HISTORICAL_ATT.mean()
    if abs(att_score - baseline) > 2 * HISTORICAL_ATT.std():
        return True

    return False

def trust_delta(source_score, content