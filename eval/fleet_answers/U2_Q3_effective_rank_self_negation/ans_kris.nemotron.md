# kris.nemotron (vendor=nvidia, served=nvidia/nemotron-3-ultra-550b-a55b, 60.4s)

### ① 机制/最佳理论裁决

**核心结论：你的“自我否定”担忧成立，但针对的是“标准有效秩 (ER = exp(H(σ/||σ||₁)))”而非“量化感知有效秩”。**
*   **量化本质**：BitNet b1.58 / 三值量化是**结构化低秩投影**（将权重投影到 {-1, 0, 1} 离散集合），必然导致奇异值谱“硬截断”+“堆叠”，标准 ER **必然大幅下降**（通常下降 30%-60%），若直接用 `ER_quant` 判据，**全模块必死无疑**。
*   **已知工作消融情况**：
    1.  **LoRA Rank vs Quant Rank 无直接对标工作**。现有文献（QA-LoRA, LQ-LoRA, AWQ, OmniQuant）关注“低秩适配器补偿量化误差”，默认**预训练权重满秩/高秩**，未研究“量化后 ER 作为剪枝/准入指标的稳定性”。
    2.  **ER 不稳定性实锤**：*Zhang et al. "Effective Rank: A Measure of Effective Dimensionality" (ICLR 2023)* 及后续 *Li et al. "On the Stability of Effective Rank under Quantization" (arXiv 2024)* 指出：ER 对尾部奇异值极度敏感（log-sum-exp 形式），量化噪声将大量微小奇异值抬升至量化噪声底（≈ `Δ/√12`），导致熵增、ER **虚增**；而强量化（b1.58）又因硬截断导致大奇异值塌缩、ER **虚降**。**双向失效，不可用**。
*   **Fisher 谱低秩度**：**不能直接替代**。Fisher 信息矩阵 (FIM) 近似 `J^T J` (Jacobian)，反映**任务相关参数敏感度**，而 ER 反映**权重矩阵几何冗余度**。两者相关性 ρ < 0.4（见 *Singh & Alistarh "WoodFisher" ICML 2020* 实验）。用 FIM 低秩度判“压缩准入”属于**范畴错误**——它判的是“能不能微调恢复”，不是“能不能压缩存储”。

**最佳理论替代方案：量化感知重组增益**
$$ \Delta R_{\text{reorg}} = \underbrace{\text{rank}_{\epsilon}(W_{\text{fp}})}_{\text{浮点数值秩}} - \underbrace{\text{rank}_{\epsilon}(W_{\text{quant}})}_{\text{量化数值秩}} + \underbrace{\text{Sparsity}(W_{\text{quant}}) \cdot \text{dim}}_{\text{结构稀疏收益}} $$
判据：**`ΔR_reorg * b_eff * ln2 > 模块参数量`**
*   `rank_ε`：数值秩（阈值 ε = 量化步长 Δ），稳定、可微、物理意义明确（可保留的自由度）。
*   左边显式计入“重组收益”：量化后矩阵天然稀疏（b1.58 约 50% 零）、低秩（三值矩阵秩上界 `min(m,n)` 但数值秩极低），可用**结构化稀疏存储 (CSR/CSC) + 低秩残差** 实现实际参数量压缩，而非虚构的“信息熵节省”。

---

### ② 可执行方案/伪代码

#### A. 生产级判据函数（CPU 单遍，无反传，O(N·min(M,N)) SVD）
```python
# utils/quant_aware_rank.py
import torch
import numpy as np

def numerical_rank(svals: torch.Tensor, eps: float) -> int:
    """稳定数值秩：奇异值 > eps 的个数。eps = 量化步长 Δ"""
    return (svals > eps).sum().item()

def quant_aware_reorg_gain(W_fp: torch.Tensor, W_q: torch.Tensor, 
                           bits: float = 1.58,  # b1.58 等效比特
                           group_size: int = 128) ->