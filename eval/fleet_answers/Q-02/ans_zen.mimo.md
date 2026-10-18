# zen.mimo (vendor=mimo, served=mimo-v2.6-flash-free, 51.1s)

# ① 校准难题与探针假设检验

## 校准误差难压的机制原因

**根本矛盾**：≤1B 模型的 epistemic（认知）不确定性与 aleatoric（数据）不确定性在 logit 空间高度纠缠，而温度缩放/ Platt scaling 只能做全局单调变换，无法分离这两种不确定性来源。

具体到 ≤1B regime：

| 因素 | 机制 | 为何小模型更严重 |
|------|------|-----------------|
| 知识边界模糊 | 模型没有"知道自己不知道"的显式表征 | 大模型靠涌现 self-consistency 获得隐式置信度；1B 以下不涌现 |
| 过参数化不足 | 参数容量不足以对"确定的部分"和"不确定的部分"做稀疏分工 | 大模型用专门 circuit 做 hedging，小模型被迫用同一组参数同时拟合两种 regime |
| 分布外检测与置信度混淆 | OOD 样本和 in-distribution hard 样本的 logit profile 几乎相同 | 大模型的中间激活可区分这两种情况，小模型的中间表示维度不够做该分类 |
| 训练信号错配 | cross-entropy 不区分"答错且自信"与"答错且犹豫" | 无额外 loss，小模型无法自发学出校准 |

**核心洞察**：≤1B 模型的校准问题不是"超参数没调好"，而是"表示能力不足以编码自身置信度"。这需要**架构层面注入**，不只是 loss 层面修补。

---

## 「高 NLL 低熵 = 系统性空洞」探针假设检验

### 假设的形式化

$$\text{probe}(x) = \mathbb{1}\left[ \text{NLL}(x) > \tau_1 \;\wedge\; H(p(y|x)) < \tau_2 \right]$$

假设：当模型对某个 token 的预测既高损失又低熵，说明模型"自信地犯错"，指向训练分布中的系统性盲区。

### 假设部分成立但有三个致命缺陷

**缺陷 1：低熵高 NLL 的语义歧义**

| 场景 | NLL | 熵 | 是否是"知识空洞" |
|------|-----|-----|-----------------|
| 真空洞（训练分布缺此 pattern） | 高 | 低 | ✓ 是 |
| Rare but valid（长尾正确答案） | 高 | 低 | ✗ 不是——模型选了众数答案但 ground truth 是低频正确答案 |
| Adversarial prompt（分布外但简单） | 高 | 低 | ✗ 部分——是 OOD 但不是"知识空洞" |
| 对抗性 tokenization（罕见词被高频词覆盖） | 高 | 低 | ✗ 不是——是 tokenizer 伪影 |

**缺陷 2：阈值不可迁移**：τ₁ 和 τ₂ 在不同模型/不同 domain 上差异极大。一个在 200M 模型上调好的阈值，换到 0.5B 上直接失效。

**缺陷 3：信息维度太低**：二值探针只说"是不是空洞"，不说"哪种空洞"和"怎么修复"。

### 更优替代探针设计

**核心改进**：从二值探针 → **连续置信度向量探针**，利用多个正交信号源。

---

## 推荐：三通道置信度探针（Triple-Signal Probe, TSP）

### 信号通道设计

```python
class TripleSignalProbe:
    """
    三通道探针：正交分解认知不确定性
    """
    def compute(self, model, x, y_true, cache=None):
        """
        Args:
            model: 前向模型（缓存 hidden states）
            x: 输入 token 序列
            y_true: ground truth token 序列
            cache: 可选的 hidden state 缓存（避免重复前向）
        Returns:
            probe_vector: 3 维连续置信度向量
        """
        if cache is None:
            cache = model.forward_with_cache(x)
        
        logits = cache['logits']              # [seq_len, vocab]
        hidden = cache['hidden_states']        # [seq_len, hidden_dim]
        n_layers = cache['n_layers']
        
        # === 通道 1: NLL-based 信号（改进版，不是原始 NLL）===
        # 用 top-k 概率差代替 NLL，区分"模糊"和"盲区"
        probs = softmax(logits, dim=-1)
        top2_margin = probs.topk(2, dim=-1).values  # [seq_len, 2]
        margin_signal = (top2_margin[:, 0] - top2_margin[:, 1])  # 低 = 模糊
        
        # NLL at each position
        nll = -torch.log(probs.gather(-1, y_true.unsqueeze(-1)).squeeze(-1))
        
        # === 通道 2: 表示稳定性信号（新增）===
        # 在不同 dropout mask / 微小扰动下，hidden state 的稳定性
        # 低稳定 = 模型"犹豫" = 高认知不确定性
        stability_signal = self._compute_repr_stability(model, x, cache)
        
        # === 通道 3: 逐层熵变化信号（新增）===
        # 逐层计算预测熵，观察"熵坍缩"发生的层
        # 大模型：早期层熵高（探索），后期层熵低（收敛）
        # 知识空洞：熵在某一层突然坍缩到低值（虚假自信）
        layerwise_entropy_profile = self._compute_layerwise_entropy(
            model, x, cache
        )
        
        # 熵坍缩检测：后半段层的熵方差
        # 方差低 = 坍缩一致 = 可能虚假自信
       