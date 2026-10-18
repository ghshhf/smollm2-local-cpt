# ext.pollinations (vendor=openai-oss, served=gpt-oss-20b, 31.8s)

**① 机制 / 最佳理论**

| 目标 | 在线信号 | 理由 |
|------|-----------|------|
| **判断「正在泛化」 vs 「正在死记」** | **Per‑Sample Loss‑Variance (PS‑LVar)**  <br>**\( \sigma_{\ell}^2 \) = 1 ⁄ N Σ (ℓᵢ – μℓ )²**   <br>**(跟梯度范数的相关性 ρ(σ²,‖∇θ‖))** | <br>1. **记忆化** ：模型在单次曝光后对极少数样本产生过大梯度，导致该样本的 loss 极端偏离整体均值，出现高 **σ²** 并且 **ρ(σ²,‖∇‖) ≈ 1**（梯度与 loss 对应正向）。 <br>2. **泛化** ：loss 分布相对平滑，σ² 低；即使梯度较大，σ² 与 ∥∇∥ 相关性弱（ρ ≈ 0）。<br>3. 通过 **滑动窗口**（timestep w = 1000）实时更新 μℓ 与 σ²，保持在线可计算。<br>**理论根源**：在一遍训练下，梯度噪声（sampling noise）是高频随机波动；若某样本的 loss 维持远高于均值，说明它对模型的更新占主导 → 记忆化。|
| **补充：对记忆化的二次判别** | **Gradient Norm Flow (GNF)**  <br>**Δg = ∥∇θᵢ∥ – ∥∇θᵢ₋₁∥**  <br>**Rescale‑Invariant Indicator (RII)** = Δg / (σ²+ϵ) | <br>记忆化样本通常导致 ∥∇∥ 突变，而可解释的学习曲线应保持减小。RII 低值 → 记忆化。 |

---

**② 可执行方案 / 伪代码**

```python
# 伪代码：单遍在线监控（可直接嵌入数据管道或训练 loop）

# Hyper‑parameters
EXP = 0.99          # exponential smoothing for mean/var
ALPHA_G = 0.99      # smoothing for gradient norm
WINDOW = 1000       # sliding window size (tokens)
EPS = 1e‑6

state = {  # maintained on CPU
    'mu_l' : 0.0,      # running mean loss
    'sigma2_l' : 0.0,  # running variance loss
    'mu_g'  : 0.0,     # running mean grad‑norm
    'cnt'   : 0,
    'queue' : deque(maxlen=WINDOW)
}

def update(sample, loss, grad):
    # 1. Δ = loss – μ
    delta_l = loss - state['mu_l']
    state['mu_l'] += delta_l / (state['cnt']+1)
    # 2. variance update (online
    delta2_l = loss - state['mu_l']
    state['sigma2_l'] += (delta_l * delta2_l - state['sigma2_l']) / (state['cnt']+1)

    # 3. grad‑norm
    gn = np.linalg.norm(grad)
    state['mu_g'] += (gn - state['mu_g']) * (1 - ALPHA_G)
    state['cnt'] += 1

    # 4. sliding‑window queue for high‑variance detection
    state['queue'].append(loss)
    if len(state['queue']) == WINDOW:
        # compute windowed variance as an additional sanity check
        win_var = np.var(list(state['queue']))
        # flag high‑variance window
        if win_var > THRESH_WV:
            warn("High variance window detected")

    # 5. Compute local indicators
    RII = (gn - state['mu_g']) / (state['sigma2_l'] + EPS)  # or inverse
    return {'mu_l':state['mu_l'],
            'sigma2_l':state['sigma2_l'],
            'mu_g':state['mu_g'],
            'RII':RII}
```

* **CPU requirement**: All arithmetic uses 32‑bit floats.  
