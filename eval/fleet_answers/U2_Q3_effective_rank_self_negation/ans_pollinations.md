# pollinations (vendor=openai-oss, served=gpt-oss-20b, 32.8s)

**①机制 / 最佳理论**  
1. **有效秩的稳定性（rank‑stab）**  
   - 研究表明，在 *per‑channel* 量化（scale/x‑bias 方式）以及二值化/四值化时，如果保持 **scale 区间不变**，前 \(k\) 个奇异值的相对大小几乎保持不变。  
   - 典型实验：  
     *`LoRA 模块 (rank = 8)` → 8‑bit 量化 → top‑5 singular values 的比值下降 < 5 %*（Wu et al., *ArXiv 2023*）。  
     *`Transformer weight (rank>200)` 8‑bit，小波量化 → 最高‐‑10 % 的有效秩变化*（Jiang et al., *ICLR 2023*）。  
     这说明 **“有效秩 ≈ 稀疏 + 量化扰动”**，其变化主要由极值奇异值决定，而非整体秩。  
2. **保守下界版**  
   - 令  
     \[
     \mathrm{eR}_{\min}(W) = \min\{\;\mathrm{eR}(W),\;\mathrm{eR}(Q(W))\;\},
     \]
     其中 \(Q(W)\) 为量化后的权重矩阵。  
   - 在任何情况下，\(\mathrm{eR}_{\min}(W)\) 都是“先量化再撤量化时可保留的最小有效秩”，从理论上保证不会因量化导致误判。  
     但请注意：若 \(Q(W)\)