# kris.nemotron (vendor=nvidia, served=nvidia/nemotron-3-ultra-550b-a55b, 54.0s)

### ① 可量化的信任晋升规则：从「Untrusted」到「Trusted」的形式化判据

#### 机制/最佳理论
**核心原则：Trust = f(Evidence, Consistency, Provenance, Cost)**。拒绝单一维度（如域名白名单），采用**贝叶斯信任积分制**，引入「证据链完整性」「跨源一致性熵」「时效衰减半衰期」「人工确认成本」四个可微分变量。  
**形式化定义**：  
设文档 $d$ 的信任分 $T_d(t) \in [0,1]$，晋升阈值 $\tau_{\text{promote}} = 0.85$（可调）。  
$$T_d(t) = \sigma\Big( \underbrace{w_1 \cdot \text{CrossVal}(d)}_{\text{交叉印证}} + \underbrace{w_2 \cdot \text{AuthScore}(\text{src}(d))}_{\text{来源权威}} - \underbrace{w_3 \cdot \text{Decay}(t - t_{\text{pub}})}_{\text{时间衰减}} - \underbrace{w_4 \cdot \text{InjectRisk}(d)}_{\text{注入风险}} \Big)$$  
其中 $\sigma$ 为 sigmoid，权重 $w_i$ 满足 $\sum w_i = 1$，默认 $w=[0.4, 0.3, 0.2, 0.1]$。**晋升条件**：$\Delta T_d = T_d(t_{\text{now}}) - T_d(t_{\text{ingest}}) \ge \Delta_{\text{min}} = 0.3$ **且** $T_d(t_{\text{now}}) \ge \tau_{\text{promote}}$。

**关键子函数可量化定义**：
1. **CrossVal(d)**：多源交叉印证分。  
   - 检索 Top-$k$（$k=5$）独立源（不同域名、不同IP段、不同ASN）对同一实体/声明的覆盖率。  
   - $\text{CrossVal}(d) = \frac{1}{k}\sum_{i=1}^k \mathbb{1}[\text{ClaimMatch}(d, d_i) \land \text{SourceIndependent}(d, d_i)]$。  
   - **硬规则**：单一来源（含镜像站）$\text{CrossVal} \le 0.2$。
2. **AuthScore(src)**：来源权威分（静态预计算，每周更新）。  
   - 基于 **域名图 PageRank + 证书透明度日志年限 + 人工白名单加权**。  
   - 白名单（如 `.gov.cn`, `.edu`, `arxiv.org`, `github.com` 官方仓库）直接赋值 0.9；其余由图算法输出 $[0, 0.7]$。
3. **Decay(Δt)**：指数衰减，半衰期 $t_{1/2} = 180$ 天。$\text{Decay} = 1 - 2^{-Δt / 180}$。
4. **InjectRisk(d)**：提示注入/对抗模式匹配分（见第3节），归一化到 $[0,1]$。

#### 可执行方案/伪代码
```python
# trust_promotion.py -- 单文档晋升判定，纯 CPU、无外部依赖
import math, time, hashlib
from collections import defaultdict

# 静态权威表（离线生成，含种子白名单）
AUTH_DB = load_auth_db()  # {domain: score ∈ [0,1]}

def claim_fingerprint(text: str) -> str:
    """抽取核心声明指纹：实体+关系+数值，去停用词、归一化数字"""
    # 轻量级：正则+NER-onnx(0.02B) → 返回排序后的 tuple hash
    return hashlib.md5(normalize_claims(extract_claims(text)).encode()).hexdigest()[:16]

def cross_val(fp: str, src_domain: str, idx: InvertedIndex) -> float:
    """倒排索引查同指纹、不同源文档数"""
    hits = idx.get(fp, [])
    indep = sum(1 for d in hits if d.domain != src_domain and d.asn != idx.get_asn(src_domain))
    return min(1.0, indep / 5.0)  # k=5

def trust_score(doc: Doc, idx: InvertedIndex, now_ts: int) -> float:
    fp = claim_fingerprint(doc.text)
    cv   = cross_val(fp, doc.domain, idx)
    auth = AUTH_DB.get(doc.domain, 0.1)
    decay = 1 - 2 ** (-(now_ts - doc.pub_ts) / (180*86400))
   