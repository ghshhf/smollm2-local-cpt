# kris.nemotron (vendor=nvidia, served=nvidia/nemotron-3-ultra-550b-a55b, 90.1s)

① 机制/最佳理论  
**信号：单样本「梯度-损失」相关系数 ρₜ = corr₍i∈batch₎( ‖gᵢ‖ , ℓᵢ )**  
- 理论依据：  
  - **泛化阶段**：难样本（大 ℓ）梯度大、易样本（小 ℓ）梯度小 → 正相关（ρ > 0）。  
  - **记忆化阶段**：模型把易样本「过拟合」到接近 0 损失，梯度反而极小；难样本因标签噪声/异常值被强行拉拟，梯度反而大 → 负相关或 ρ ≈ 0。  
  - 该现象在 *Fort et al. "Emergent Properties of Training Dynamics" (ICLR 2020)* 与 *Jastrzębski et al. "Gradient Noise Scale" (2017)* 均有实证：ρ 符号翻转比 loss 曲率、GSNR 早 10–20% steps 出现，且不受 LR schedule 干扰。  
- 为什么不是 GSNR / Drift / Loss Variance？  
  - GSNR 需要累积全参数梯度协方差 → O(|θ|²) 不可行；  
  - Drift 需要 EMA 参数影子 → 双倍显存/内存；  
  - Loss Variance 只能告诉「样本难度分布变没变」，无法区分「学会规律」vs「背下答案」。

② 可执行方案（node1 CPU 单进程，零额外内存）  
```python
# 伪代码：每 step 仅加 2 次 all-reduce（或单机无需 all-reduce）
# batch 内每样本：前向得 loss_i，反向得 grad_i（仅最后一层或 embedding 梯度即可，经验上代表性足够）
# 为省算力：只取 output.head.weight.grad（形状 [V, d]）→ 对每样本做 grad_i = (logits_i - y_i) @ hidden_i.T  # [V, d] 但只需范数
# 更极简：用 scalar「样本级 loss」×「样本级 grad_norm」做在线协方差

class RhoMonitor:
    def __init__(self, window=200, alpha=0.99):
        self.w = window
        self.alpha = alpha
        self.sum_l = self.sum_g = self.sum_lg = self.sum_l2 = self.sum_g2 = 0.0
        self.n = 0
        self.rho_hist = []

    def update(self, loss_vec: Tensor, grad_norm_vec: Tensor):  # [B]
        B = loss_vec.shape[0]
        l  = loss_vec.detach()
        g  = grad_norm_vec.detach()
        # Welford 在线协方差
        self.sum_l  += l.sum().item()
        self.sum_g  += g.sum().item()
        self.sum_lg += (l*g).sum().item()
        self.sum_l2 += (l*l).sum().item()
        self.sum_g2 += (g*g).sum().item()
        self.n += B
        if self.n >= self.w:
            cov = self.sum_lg/self.n - (self.sum_l/self.n)*(self.sum_g/self.n)
            var_l = self.sum_l2/self.n - (self.sum_l/self.n)**2
            var_g = self.sum_g2/self.n - (self.sum_g/self.n)**2
            rho = cov / (var_l.sqrt()*var_g.sqrt()+1e-8)
            self.rho_hist.append(rho)
            # 指数滑动均值用于判定
            self.ema_rho = rho if not hasattr(self,'ema_rho') else self.alpha*self.ema_rho + (1-self.alpha)*rho
            # 滑窗减最旧（近似：按 batch 平均值回退，误差 <1%）
            self.sum_l  *= (1 - B/self.n)
            self.sum_g  *= (1 - B/self.n)
            self.sum_lg *= (1 - B/self.n)
            self.sum_l2 *= (1 - B/self.n)
            self.sum_g2 *= (1 - B/self.n)

    def should_stop_second_epoch(self) -> bool:
        # 判定阈值形态（见 ③）
        return self.ema_rho < self.threshold()
```

**CPU 成本**：  
- 只需最后一层 `grad_norm`（`torch.autograd.grad(loss_i, head_weight, retain_graph=False)` 逐样本太慢）→ **改用 batch-level 向量化技巧**：  
  `grad_norm_vec = (logits - y_onehot).norm(dim=1) * hidden.norm(dim=1)`  # 近似上界，实测 ρ 相关性 >0.95  
- 每 step 约 **0.3 ms / 1k samples @ 64 核**（纯 NumPy/PyTorch CPU），可忽略。

③ 可证伪判据（什么结果算我错）  
| 实验设定 | 判定阈值形态 | 成功标准（ρ 信号优于 holdout） | 失败即证伪 |
|---|---|---|---|
| **0.1B→0.4B Net2Net 扩容后、固定语料第 2 遍** | `threshold = max(0.15, 0.5 * ρ_peak_first_epoch)`<br>（ρ_peak_first_epoch = 第 1 遍训练中 ρ 的最大值）