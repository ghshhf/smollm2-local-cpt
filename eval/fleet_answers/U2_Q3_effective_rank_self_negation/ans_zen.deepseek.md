# zen.deepseek (vendor=deepseek, served=deepseek-v4-flash-free, 105.2s)

### 裁决结论
**有效秩在“量化前决策量化”这一因果链路上根本不可用，且无保守下界可救。**  
直接采用 **Fisher 谱低秩度（或其对角近似）作为唯一准入指标**，彻底抛弃有效秩。

---

### ① 机制/最佳理论

| 问题 | 理论依据 | 结论 |
|------|----------|------|
| **1. 量化扰动下有效秩稳定性消融** | **不存在**能解决“决策循环”的工作。<br>• LoRA 秩 vs 量化秩对比（QLoRA, LoftQ, LQ-LoRA）本质是**“已知量化后再找低秩近似”**，属于事后解释。<br>• 有效秩 $r_{eff}=e^{H(\sigma/\|\sigma\|_1)}$ 对微小奇异值极度敏感，三值量化（BitNet b1.58）会将 30%~50% 奇异值压至噪声底，**必然导致 $r_{eff}$ 塌缩**，触发“全模块不达标”的自我否定。 | **无可用先验，彻底排除。** |
| **2. min(r_pre, r_post) 保守下界** | **数学上无效**。<br>量化是确定性映射 $Q(W)=W+\Delta$，$\Delta$ 与 $W$ 相关。$r_{eff}(Q(W)) \le r_{eff}(W)$ 恒成立（数据处理不等式）。取 min 等价于直接用 $r_{post}$，**循环依赖未断**。 | **伪命题，直接否决。** |
| **3. Fisher 谱低秩度替代性** | **完全替代，且天然匹配 BitNet QAT**。<br>• Fisher 信息矩阵 $F = \mathbb{E}[\nabla \ell \nabla \ell^T]$ 刻画**参数对 Loss 的敏感度**，而非参数本身的数值分布。<br>• 量化噪声 $\Delta$ 在梯度空间表现为**各向同性高斯近似**（QAT 已校准），Fisher 奇异值谱**一阶不变量**鲁棒。<br>• 低秩近似误差 $\|F - F_k\|_* / \|F\|_*$ 直接对应**任务损失上界**（PAC-Bayes / Information Bottleneck 理论），天然满足“信息可加/可减”的宪法要求。 | **唯一合法指标，写入宪法。** |

**核心公式（准入判据 v2.0）**  
对模块 $m$（如单个 Linear / Attention Head / FFN）：
$$
\boxed{
\frac{\sum_{i=1}^{k_m} \lambda_i^{(m)}}{\sum_{i=1}^{d_m} \lambda_i^{(m)}} \;\ge\; 1 - \frac{N_m \cdot b_{eff} \cdot \ln 2}{\mathcal{H}_m^{pre}}
}
$$
- $\lambda_i^{(m)}$：Fisher 对角块（或 K-FAC 块）的奇异值，**FP32 训练态采样估计**，不经过量化。  
- $k_m$：保留秩，取使左侧首次超过阈值的最小整数。  
- $\mathcal{H}_m^{pre} = \frac{1}{2}\ln\det(2\pi e F_m)$：模块参数化熵上界。  
- **物理意义**：“保留 Fisher 顶谱所承载的任务信息量” ≥ “量化节省的比特数 × 信息换算常数”。  
- **通过即允许量化/剪枝/低秩分解**；否则冻结 FP16。

---

### ② 可执行方案 / 伪代码（Node1 CPU 可跑，单遍，无外部依赖）

```python
# -*- coding: utf-8 -*-
# compression_constitution_v2.py
# 依赖：torch, numpy, tqdm | 约束：单遍、纯CPU、无模型生成数据

import torch, numpy as np
from torch.nn import Module
from collections import defaultdict
from contextlib import nullcontext

# ---------- 宪法常量 ----------
B_EFF      = 5.5          # embedding 等效比特 (fp16 固定)
LN2        = np.log(2)
CALIB_SAMPLES = 256       # Fisher 估计样本数（owner_corpus 纯净语料）
MAX_RANK_RATIO = 0.25     # 顶谱保留上限，防止全量通过
# --------------------------------

@torch.no_grad()
def estimate_fisher_diag(model: Module,