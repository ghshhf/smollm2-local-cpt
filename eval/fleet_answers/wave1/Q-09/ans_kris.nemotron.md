# kris.nemotron (vendor=nvidia, served=nvidia/nemotron-3-ultra-550b-a55b, 77.8s)

### ① 机制/最佳理论

**核心判断：你的「深窄 + Net2Net 零初始化恒等层」前提在 BitNet b1.58 (三值权重) 下存在「梯度信噪比崩塌」风险，必须修正扩容策略。**

| 理论锚点 | 对 0.5B 深窄 (h=1024) 的推导 | 结论 |
| :--- | :--- | :--- |
| **BitNet b1.58 量化噪声累积** | 三值权重 { -1, 0, 1 } 等效于在 FP16 梯度上叠加均值为 0、方差 $\propto 1/d_{model}$ 的乘性噪声。深度 $L$ 层累积后，输出信噪比 $\text{SNR} \sim \mathcal{O}(L^{-1/2})$。 | **层数上界 $L_{max} \approx 48 \sim 56$** (h=1024 下)。再深需引入 **Learnable Skip Scaling ($\alpha$)** 或 **DeepNorm** 才能收敛，但这破坏「严格保函数」(零输出恒等)。 |
| **Net2Net 零初始化恒等层的「死区」效应** | 零初始化 FFN/Attn 输出权重 $\Rightarrow$ 新层初始梯度仅来自残差连接。BitNet 权重离散化导致「直通估计器 (STE) 梯度稀疏」，新层参数极难「苏醒」。 | **禁止纯零初始化**。必须用 **「微扰恒等初始化」**：$W_{out} \sim \mathcal{N}(0, \sigma^2), \sigma = 10^{-3} \sim 10^{-4}$，配合 **LayerScale ($\gamma \ll 1$)** 保函数近似，保证梯度流。 |
| **Scaling Law 深窄最优比 (Chinchilla / Kaplan 修正)** | $N \approx 0.5B, D \approx 3.5 \times 10^9$ (你的 $N^*$)。最优 $L/d_{model}$ 比在稠密模型约 $L \propto N^{0.5}, d \propto N^{0.3}$。但 **三值量化压缩了有效容量**，需补偿宽度或降低深度。 | **h=1024 固定时，最优层数 $L^* = 40 \pm 4$** (对应 0.48B~0.52B non-embed params)。超过 48 层边际收益为负，除非加 MoD 或 Memory Layer 分担计算。 |
| **小模型注意力稀疏性 (SWA vs Global)** | $h=1024, L=40$ 时，单头维 $d_h=64$ (16 heads)。小模型头数少，**全局注意力是「长程依赖唯一通道」**。Gemma3 5:1 基于 2B+ 且有 8k+ 上下文窗口。你的端侧上下文可能 4k-8k，**全局比例不可低于 1:3 (25%)**，否则复现/检索任务崩。 | **建议 3:1 或 2:1 (SWA:Global)**。窗口 $W=512$ 或 $1024$。 |
| **MoD 在 0.5B 的边际收益** | MoD 路由器开销 $\approx 2 \times h \times d_{router}$。0.5B 模型参数预算极紧，**MoD 仅在「推理加速 > 1.3x」且「精度损失 < 0.5% PPL」时值得**。CPU 端 15ms/token 瓶颈在内存带宽而非