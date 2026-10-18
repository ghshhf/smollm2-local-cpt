# kris.gemini (vendor=google, served=gemini-3.8-flash, 83.7s)

### 结论先行与裁决总览

**推翻“三值模型可做严格保函数 Net2Net 扩容”的技术妄想。**

三值（BitNet b1.58）与全精度（FP32/FP16）的代数拓扑完全不同：全精度是**连续紧致空间**，对标量除法封闭；三值是**离散离散点集 $\{-1, 0, 1\}$**，对除法不封闭，且强绑定了全局耦合标量 $\gamma = \text{mean}(|W|)$。

针对三问的正式裁决：
1. **文献现状**：**学术界与工业界均不存在极端低比特（1.58-bit）下的严格保函数 Net2Net 方案。** 现有 QAT 扩容文献全部停留在 INT8（可做可除的缩放因子补偿），三值量化只能走“破坏性初始化 + 蒸馏对齐”或“连续空间扩容后重新离散化”。
2. **Buffer 显式保存 $\gamma$**：**治标不治本，直接破坏 QAT 收敛动力学。** 强行 Buffer 冻结 $\gamma$ 虽能避免初始化瞬间的全局重算，但无法解决出射权重 $W_2$ 必须除以 $2$ 的离散不可表示性；且在后续反向传播中，潜权重（Latent Weight）尺度与 $\gamma$ 脱钩，导致 Straight-Through Estimator (STE) 梯度剧烈振荡。
3. **复制式初始化能否精确保函数**：**绝对不可行（数学上严格证明恒等崩溃与雪崩翻转）。**

---

### ① 机制 / 最佳理论：三值 Net2Net 保函数破产的数学证明

#### 破产点 1：加深（Net2DeeperNet）——恒等映射的“维度衰减死穴”
在 Net2Net 中，插入新层要求 $f(X) = X$（恒等映射）。
设 BitNet b1.58 线性层计算公式为：
$$\tilde{W} = \text{RoundClip}\left(\frac{W}{\gamma + \epsilon}, -1, 1\right), \quad \gamma = \frac{1}{d_{\text{out}} d_{\text{in}}} \sum_{i,j} |W_{ij}|$$
反量化输出为：
$$Y = \tilde{X} \tilde{W}^T \times \gamma$$
若要实现恒等映射 $Y = \tilde{X}$，必须令反量化后的有效矩阵 $M = \tilde{W}^T \times \gamma = I_d$。
1. 若 $\tilde{W} = I_d$（对角矩阵）：
   $$\gamma = \text{mean}(|I_d|) = \frac{1}{d \times d} \sum_{i=1}^d 1 = \frac{d}{d^2} = \frac{1}{d}$$
2. 计算反量化输出：
   $$Y = \tilde{X} \cdot I_d \cdot \gamma = \tilde{X} \cdot \frac{1}{d}$$
**数学死结**：在 $d=1024$ 时，输出信号强度被机械性强制衰减 **1024 倍**。若要在三值下保持 $Y = \tilde{X}$，必须要求 $\gamma = 1$，这意味着 $\sum |W_{ij}| = d^2$，即矩阵内**所有元素必须全为 $\pm 1$**，根本不存在稀疏对角恒等矩阵的三值表达。

#### 破产点 2：加宽（Net2WiderNet）——“除法不可表示”与“雪崩式跳变”
设将隐藏层维度从 $d$ 扩至 $d+1$，复制第 $k$ 个神经元：
1. **出射矩阵 $W_2$ 的不可分性**：
   为保持 $h_k W_{2,k} = h_k' W_{2,k}^{(1)} + h_k' W_{2,k}^{(2)}$，全精度下令 $W_{2,k}^{(1)} = W_{2,k}^{(2)} = \frac{1}{2} W_{2,k}$。
   但在三值空间中，若原权重 $W_{2,k} = 1$，则 $\frac{1}{2} \notin \{-1, 0, 1\}$。
   - 若离散化为 $0$：信号丢失 $100\%$；
   - 若离散化为 $1$：激活值翻倍（膨胀 $100\%$）。
2. **全局标量耦合导致的雪崩式翻转（Avalanche Quantization Flip）**：
   设行复制导致均值变动：
   $$\gamma_{\text{new}} = \gamma_{\text{old}} + \Delta \gamma$$
   即使 $\Delta \gamma$ 仅为 $0.01\%$，由于三值量化采用硬截断函数 $\text{Round}(W / \gamma)$，所有处于量化边界临界区（即 $|W_{ij}| \approx 0.5 \gamma$）的潜权重，其量化值将发生大规模批量翻转（$0 \to 1$ 或 $1 \to 0$）。一个局部神经元的扩展，必然通过标量 $\gamma$ 触发全矩阵权重的离散相变。

---

### ② 针对三问的裁决与工程执行方案

#### 裁决 1：文献方案审判
- **文献真实情况**：从 Net2Net (Chen et al., 2016)、NetMorph 到 FPGM，所有声称“保函数”的论文均默认权重处于 $\mathbb{R}$（连续实数域）。
- **极低比特现状**：BitNet b1.58 (Microsoft, 2024) 原生架构直接放弃任何 Net2Net 扩容路线。学术界对于 1-bit/1.58-bit 网络的扩展，**唯一被验证可行的方案是：宽模型蒸馏窄模型，或者在全精度空间完成 Net2Net 训练后再做 QAT。**

#### 裁决 2：Buffer 显式保存 $\gamma$ 的可行性裁决
- **结论**：**严禁在架构中引入此 Hack。**
- **理由**：
  1. 若将 $\gamma$ 作为 buffer 固定，推导公式变为 $Y = \tilde{X} \tilde{W}^T \times \gamma_{\text{buf}}$。但在随后的 QAT 训练中，潜权重 $W$ 会更新，导致 $\text{mean}(|W|) \neq \gamma_{\text{buf}}$。
  2. 此时 STE 估计器的前向缩放与反向梯度完全脱节，潜权重将快速漂移至饱和区（$|W| \gg \gamma$），导致全矩阵退化为 $-1$ 和 $+1$，彻底丧失三值网络的稀疏 $0$ 表达能力。

#### 裁决 3：复制式初始化的输出等价性裁决
- **结论**：**严格不等价，不可作为进化主干手段。**

#### RSI 闭环路线强制纠偏方案（严格对齐宪法）
1. **0.1B $\to$ 0.4B 阶段**：**必须全程在 FP16/BF16 精度下执行 Net2Net 保函数扩容**。连续空间下标量除以 $c$ 严格成立，恒等映射通过全零偏置或残差直通保证，函数误差为 $0$。
2. **0.4B $\to$ 0.5B 阶段**：**执行宪法既定路线——彻底抛弃 Net2Net 继承，采用“深窄原生 BitNet 架构 + 0.4B 蒸馏”**：
   - 0.5B 原生 BitNet 初始化采用常规的潜权重随机初始化（如 He 正态分布，保证 $\gamma = \sqrt{2/\pi}\sigma$）。
   - 采用 0.4B FP16 模型作为 Teacher，在 Node1 上进行 Task-Agnostic Logits 蒸馏 + 隐层特征点互信息蒸馏。
   - **严禁试图从 0.4B 直接切片/拼装成 0.5B 三值模型。**

---

### ③ 可证伪判据

若任何人声称找到了“三值量化下严格保函数的 Net2Net 初始化方案”，执行以下判据，出现任一情况即判定该方案**破产**：

1. **恒等性崩塌判据**：在未引入非标准逐层自适应系数（打破标准 BitLinear 算子定义）的前提下，向现有 BitNet 模型插入一层该方案初始化的“恒等层”，输入随机正态分布张量 $X$，若变换后的输出 $Y$ 满足：
   $$\frac{\|Y - X\|_F}{\|X\|_F} > 1e-4$$
   则证明保函数性在数值上不成立（实际 BitNet 恒等初始化误差通常 $> 90\%$）。
2. **雪崩翻转判据**：在加宽单层神经元时，统计该层原有位置的三值矩阵 $\tilde{W}_{\text{old}}$ 与扩容后截取对应位置 $\tilde{W}_{\text{new}}[0:d_{\text{out}}, 0:d_{\text{in}}]$。若非零元素翻转率（Hamming Distance）：
   $$\mathcal{H}(\tilde{W}_{\text{old}}, \tilde{W}_{\text{new\_crop}}) > 0$$
   即判定该机制无法隔离扩展扰动，保函数等价性破产。

---

### ④ Node1 CPU (64核) 最小可跑实测代码

本脚本直接在 Node1 环境验证：
1. 插入恒等层后信号衰减 $1/d$ 的灾难；
2. 加宽时复制操作导致全矩阵 $\gamma$ 偏移并引发三值雪崩翻转。

```python
import torch
import torch.nn as nn

def quantize_ternary(W):
    """标准 BitNet b1.58 三值量化逻辑"""
    gamma = W.abs().mean()
    if gamma < 1e-5:
        return torch.zeros_like(W), gamma
    W_scaled = W / gamma
    W_quant = torch.clamp(torch.round(W_scaled), -1, 1)
    return W_quant, gamma

def run_falsification_test():
    torch.manual_seed(42)
    d = 1024
    batch_size = 4
    X = torch.randn(batch_size, d)

    print("===== 判据 1: Net2DeeperNet 恒等映射验证 =====")
    # 尝试构造三值恒等矩阵
    W_identity = torch.eye(d)
    W_q_id, gamma_id = quantize_ternary(W_identity)
    
    # 模拟 BitLinear 前向: Y = (X @ W_q^T) * gamma
    Y_id = (X @ W_q_id.T) * gamma_id
    
    ratio = torch.norm(Y_id) / torch.norm(X)
    print(f"理论衰减倍数: 1/d = {1/d:.6f}")
    print(f"实际输出范数 / 输入范数: {ratio.item():.6f}")
    print(f"恒等崩塌判定: {'破产 (衰减近千倍)' if ratio < 0.01 else '通过'}")

    print("\n===== 判据 2: Net2WiderNet 标量耦合与雪崩翻转验证 =====")
    # 初始化一个随机潜权重层
    W = torch.randn(d, d) * 0.02
    W_q_orig, gamma_orig = quantize_ternary(W)
    
    # 模拟复制第 0 行扩容 (d -> d+1)
    row_to_copy = W[0:1, :]
    W_expanded = torch.cat([W, row_to_copy], dim=0)
    
    W_q_expanded, gamma_expanded = quantize_ternary(W_expanded)
    
    # 检查原有矩阵截取部分是否保持一致
    W_q_crop = W_q_expanded[:d, :]
    flipped_elements = (W_q_orig != W_q_crop).sum().item()
    flip_rate = flipped_elements / (d * d)
    
    print(f"原始 gamma: {gamma_orig.item():.6f}, 扩容后 gamma: {gamma_expanded.item():.6f}")
    print(f"受到牵连翻转的权重个数: {flipped_elements} / {d*d}")
    print(f"全局权重翻转率: {flip_rate * 100:.4f}%")
    print(f"严格保函数判定: {'破产 (全局耦合导致雪崩翻转)' if flipped_elements > 0 else '通过'}")

if __name__ == "__main__":
    # 在 64 核 CPU 上单线程/多线程极速秒级运行
    run_falsification_test()
```

### 最终路线锁定
保持宪法铁律：**0.1B $\to$ 0.4B 在连续空间（FP16/BF16）做 Net2Net 保函数扩容；0.4B 落地后，0.5B 三值直接从零深窄 QAT 并通过蒸馏吸纳 0.4B 资产，严禁跨越精度鸿沟强行对三值做 Net2Net。**