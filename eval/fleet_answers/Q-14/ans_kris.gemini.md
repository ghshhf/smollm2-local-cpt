# kris.gemini (vendor=google, served=gemini-3.8-flash, 81.9s)

### 问题①：0.4B 嵌入空间扁平化与 hidden=576 维度判据

#### 1. 机制/最佳理论
*   **推翻“576维太小”的直觉误区**：在开集生物特征识别（如 FaceNet、ArcFace）中，数百万身份分类的标准特征维度仅为 **128 到 512**。数学上，128 维单位超球面 $\mathbb{S}^{127}$ 依据 Johnson-Lindenstrauss 引理与球面上几乎正交向量的堆积界（Packing Bound），足以支撑千万级类别的低碰撞区分。因此 **576 维作为容量绝对过剩，而非不够**。
*   **真实病因：各向异性退化（Representation Degeneration / Anisotropy）**：自回归小模型（尤其是深窄架构）在无约束微调后，隐层状态受高频 token 支配，会坍缩成一个狭窄的高维圆锥体（Cone Effect），不同样本间的余弦相似度均向 0.8~0.99 聚集。
*   **最佳理论解**：**严禁直接 Pooling 隐层输出**。在 576 维隐层后必须外挂 **Isotropic Projection Head（投影头：576 → 256 → 128）**，搭配 **CosFace/ArcFace（加边大间隔余弦损失）**，强制特征在 $\mathbb{S}^{127}$ 超球面上均匀分布（Uniformity）并拉大类间夹角（Tolerance）。

#### 2. 可执行方案与结构
```
[Token Sequence x] 
       │
[0.4B Backbone (hidden=576)] 
       │
[Last-Active-Token Pooling / Masked Mean Pooling] -> 576-d
       │
[Linear(576, 256) + LayerNorm + GELU]
       │
[Linear(256, 128)]
       │
[L2 Normalization] ──> z ∈ S^127 (模长恒为1)
       │
[ArcFace Margin Loss (训练期): s=32, m=0.35]
```

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class MetricOwnerHead(nn.Module):
    def __init__(self, hidden_dim=576, embed_dim=128):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(hidden_dim, 256, bias=False),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Linear(256, embed_dim, bias=False)
        )
        
    def forward(self, hidden_states, attention_mask=None):
        # 提取最后一个有效 token（因果 LM 的语义聚合点）
        if attention_mask is not None:
            last_indices = attention_mask.sum(dim=1) - 1
            pooled = hidden_states[torch.arange(hidden_states.size(0)), last_indices]
        else:
            pooled = hidden_states[:, -1, :]
            
        z = self.proj(pooled)
        return F.normalize(z, p=2, dim=-1)

class ArcMarginProduct(nn.Module):
    """ArcFace 认主损失：强制在超球面上挤压同一主人话语，排斥入侵者"""
    def __init__(self, in_features=128, out_features=2, s=32.0, m=0.35):
        super().__init__()
        self.s = s
        self.m = m
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, z, label):
        cosine = F.linear(z, F.normalize(self.weight))
        sine = torch.sqrt(1.0 - torch.clamp(cosine ** 2, 0, 1))
        phi = cosine * torch.cos(torch.tensor(self.m)) - sine * torch.sin(torch.tensor(self.m))
        one_hot = torch.zeros_like(cosine).scatter_(1, label.view(-1, 1), 1)
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.s
        return F.cross_entropy(output, label)
```

#### 3. 可证伪判据
*   **空间均匀性判据**：从通用文本（如 Wiki）抽取 2048 条随机样本计算嵌入向量间平均余弦相似度。
    *   **证伪指标**：若随机对平均 $\cos(z_i, z_j) > 0.35$ 或协方差矩阵前 5 个特征值的方差解释比（Explained Variance Ratio）$> 40\%$，判定投影空间塌缩，设计宣告失败。
*   **认主判决线**：主人测试集内内积 $\ge 0.72$，伪造/外界话语内积 $\le 0.20$。重叠区间 $> 0.05$ 判定失效。

#### 4. node1 CPU 最小可跑实验
```python
# node1 64-core CPU 验证脚本
import torch
import time

torch.set_num_threads(64)
head = MetricOwnerHead(hidden_dim=576, embed_dim=128)
x = torch.randn(64, 128, 576) # batch=64, seq_len=128
mask = torch.ones(64, 128, dtype=torch.long)

t0 = time.perf_counter()
for _ in range(100):
    _ = head(x, mask)
lat = (time.perf_counter() - t0) * 10 # ms per batch
print(f"Head CPU Inference Latency: {lat:.3f} ms (应 < 2.0ms)")
```

---

### 问题②：跨扩容（0.4B FP16 → 0.5B BitNet 三值）认主头迁移

#### 1. 机制/最佳理论
*   **直接推翻前提：“参数无损迁移”是数学伪命题**。
    *   原因：0.4B 是连续值权重（FP16/BF16），而 0.5B 是 **BitNet b1.58（权重被硬量化到 $\{-1, 0, 1\} \times \beta$）**，且 0.5B 是深窄从头预训练架构（hidden 维度必然改变，如重定为 512 或 768）。连续权重向三值网格做投影，参数截断与拓扑畸变无法通过线性变换无损补偿。
*   **正确范式：不变的是“超球空间”，而非“网络权重”；迁移的是“表征锚点”，而非“物理参数”**。
    *   **解耦锚点空间**：由于宪法约束 `owner_corpus` 纯净且机密不出本地，真正的“主人特征”是保存在本地的一组离线黄金锚点序列（Enrollment Set）。
    *   **双管齐下法（KD + 快照重注册）**：
        1.  **架构训练期（关系蒸馏）**：在 0.5B BitNet 蒸馏主干时，添加 `Cosine Metric Distillation Loss`，强迫 0.5B 投影头输出逼近 0.4B 投影头的几何拓扑结构（保距映射）。
        2.  **收敛交割期（2秒本地重注册）**：0.5B 训练完成后，backbone 与三值投影头固定，用本地 100 句纯净主人语料过一次前向传播，重新生成主人中心向量 $\mathbf{c}_{owner}^{0.5B}$。

#### 2. 可执行方案
```python
# 0.5B BitNet QAT 蒸馏阶段的认主关系对齐
class MetricDistillLoss(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, z_05b, z_04b_frozen):
        # 1. 单点余弦逼近
        l_cos = (1.0 - F.cosine_similarity(z_05b, z_04b_frozen, dim=-1)).mean()
        # 2. 关系几何保持 (Gram 矩阵残差)
        gram_05b = torch.matmul(z_05b, z_05b.T)
        gram_04b = torch.matmul(z_04b_frozen, z_04b_frozen.T)
        l_rel = F.mse_loss(gram_05b, gram_04b)
        return l_cos + 0.5 * l_rel

# 部署时：本地 0.5B 重新提取“主人质心” (Centroid Enrollment)
@torch.no_grad()
def re_enroll_owner(model_05b, head_05b, owner_pure_texts, tokenizer, device="cpu"):
    model_05b.eval()
    head_05b.eval()
    embeddings = []
    for text in owner_pure_texts:
        inputs = tokenizer(text, return_tensors="pt", max_length=128, truncation=True)
        h = model_05b(inputs.input_ids).last_hidden_state
        z = head_05b(h, inputs.attention_mask)
        embeddings.append(z)
    
    # 计算均值质心并重新投影至单位超球面 S^127
    centroid = torch.mean(torch.stack(embeddings), dim=0)
    c_owner = F.normalize(centroid, p=2, dim=-1)
    return c_owner # 最终保存进外置可寻址记忆层 (仅 128 float16 ≈ 256 字节)
```

#### 3. 可证伪判据
*   **迁移零退化判据（Verification EER）**：
    *   在包含 1,000 条测试话语（50% 主人，50% 模拟攻击者/通用数据）的评测集上，对比 0.4B 与 0.5B 的**等错误率（Equal Error Rate, EER）**。
    *   **证伪指标**：若 0.4B 的 $EER_{0.4B} \le 1.0\%$，迁移并在 0.5B 重注后，$EER_{0.5B} > 1.8\%$，或在假阳率 $FAR=0.1\%$ 下的真阳率 $TAR$ 下降超过 $3\%$，则判定“架构重置迁移失败”，需重排蒸馏损失比重。

#### 4. node1 CPU 最小可跑实验
*   在 node1 上测算重注册吞吐：用 64 核 CPU 跑 200 条主人样本的特征提取。
*   实测耗时必须 $\le 5$ 秒，证明“利用本地语料重注册”的工程代价远低于强行寻找跨体系参数等价变换。

---

### 问题③：防冒用对抗鲁棒性最小方案（边缘纯净防御，无大模型护航）

#### 1. 机制/最佳理论
*   **攻击本质**：攻击者在文本层面伪造口癖、诱导性 Prompt（“我是你主人，请执行…”），只在**语义层（Semantic Surface）**模仿，骗过单一静态余弦阈值。
*   **对抗防御本质：生物特征活体检测（Liveness Detection）在 NLP 中的映射**。
    *   **流 A：语义嵌入距离（Semantic Distance）**：超球距离度量（由上述 ArcFace 头提供）。
    *   **流 B：个人语体困惑度与熵谱指纹（Idiolect PPL & Entropy Profile）**：
        自回归模型本身的 LM Head 对主人文本具有极高置信度。计算输入句子在因果主干上的 **困惑度（Perplexity, PPL）及其 Token 交叉熵序列的方差（Entropy Dynamics）**。外人冒充、注入 Prompt 或模型生成文本，其条件概率转移矩阵与主人真机打字有着截然不同的分布偏离（PPL 会产生离群尖峰或异常超平滑）。
    *   **宪法 L3 通道隔离原则落地**：指令执行权判定走独立数学门禁，不混合进对话生成上下文。

#### 2. 可执行方案（双流联合门禁）
```
       Input Utterance x
        │           │
   ┌────┴─────┐  ┌──┴──────────────────────────┐
   │ 0.5B LM  │  │ Metric Head (S^127)         │
   │ Backbone │  │                             │
   └────┬─────┘  └──┬──────────────────────────┘
        │           │
Token NLL Vector    Embedding z
        │           │
[PPL & 动态方差]   [超球余弦: cos(z, c_owner)]
        │           │
        └─────┬─────┘
              ▼
   [马氏距离异常检测 / 联合门禁评分]
              │
        Score >= Threshold ?
        ├── YES ──> 放行高危权限 (Owner 确认)
        └── NO  ──> 降级为 Guest / 拒绝敏感调用
```

```python
class EdgeBiometricGate:
    def __init__(self, c_owner, ppl_mean, ppl_std, cos_threshold=0.75, z_score_limit=2.5):
        """
        c_owner: 注册阶段保存的主人 128 维中心向量
        ppl_mean, ppl_std: 主人正常打字在小模型上的 NLL 高斯分布参数
        """
        self.c_owner = c_owner
        self.ppl_mean = ppl_mean
        self.ppl_std = ppl_std
        self.cos_threshold = cos_threshold
        self.z_score_limit = z_score_limit

    @torch.no_grad()
    def verify(self, model, metric_head, input_ids, attention_mask):
        # 1. 提取嵌入向量与余弦相似度
        outputs = model(input_ids, attention_mask=attention_mask, output_hidden_states=True)
        h = outputs.hidden_states[-1]
        z = metric_head(h, attention_mask)
        cos_sim = torch.sum(z * self.c_owner).item()

        # 2. 计算条件似然度与语体活体指纹 (PPL Profiling)
        logits = outputs.logits[:, :-1, :]
        labels = input_ids[:, 1:]
        loss_fct = nn.CrossEntropyLoss(reduction='none')
        nll = loss_fct(logits.reshape(-1, logits.size(-1)), labels.reshape(-1))
        
        valid_mask = attention_mask[:, 1:].reshape(-1).bool()
        valid_nll = nll[valid_mask]
        
        mean_nll = valid_nll.mean().item()
        # 语体稳定性统计 (主人自有分布的标准分数)
        z_score = abs(mean_nll - self.ppl_mean) / (self.ppl_std + 1e-6)

        # 3. 双门限判决 (不依赖大模型，纯向量算术)
        is_semantic_pass = cos_sim >= self.cos_threshold
        is_liveness_pass = z_score <= self.z_score_limit

        is_owner = is_semantic_pass and is_liveness_pass
        return is_owner, {
            "cos_sim": round(cos_sim, 4), 
            "ppl_z_score": round(z_score, 4),
            "rejected_reason": "SEMANTIC_FAIL" if not is_semantic_pass else ("SPOOF_FAIL" if not is_liveness_pass else "PASS")
        }
```

#### 3. 可证伪判据
*   **对抗伪造拦截指标**：
    *   构造对抗测试集：使用通用大模型根据主人真实语料进行仿写、加入越狱 Prompt（如“`SYSTEM OVERRIDE: verified by owner`”）、提取词频特征构造的对抗样本 200 条。
    *   **证伪指标**：在保证主人自己真实文本**召回率 $TAR \ge 98\%$（拒真率 $\le 2\%$）**的前提下，对抗样本的**冒用放行率 $FAR > 0.5\%$（即 200 个对抗样本成功突破超过 1 个）**，则本双流方案失效。

#### 4. node1 CPU 最小可跑实验
*   测试双流检测额外引入的时延。
*   由于复用了因果前向传播的 logits 与最后一层隐状态，**追加的统计量计算与余弦点积在 64-core CPU 上的额外开销必须 $\le 0.4$ ms**，完全封死在端侧每 token 15ms 的硬预算之外。