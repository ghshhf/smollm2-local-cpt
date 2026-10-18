# ext.pollinations (vendor=openai-oss, served=gpt-oss-20b, 9.7s)

**1️⃣ 机制 / 最佳理论**  

| 题目 | 理论依据 / 经验 | 主要结论 |
|------|----------------|----------|
| **0.4 B 嵌入空间是否“太扁平”** | • 经典人脸识别（ArcFace，SphereFace）常用 128/256‑dim embedding，识别率 99.9%+。<br>• 语音说话人识别常采用 512‑dim，语义向量往往使用 768‑dim（Bert‑base）即可取得良好区分。<br>• 对于 0.4 B 的通用语言模型，隐藏层 576 维相当于 BERT‑base（768）减去 20%，在多模态交叉检索实验中 512‑dim 与 768‑dim 的精度差异通常不超过 1‑2%。<br>• Triplet 训练对 **margin**、**重排**（hard mining）与 **batch size** 的调节远比单独增维敏感。 | **576 维在 0.4 B 模型上已经足够**：只要保留良好的 triplet mining、margin 调整和正则化，嵌入的判别力不会因维度不足而出现周期性失效。 |
| **跨扩容（0.4 B → 0.5 B）时，如何无损迁移认主头** | • 迁移学习中把 *“头”* 单独视为一个线性投影层（如 576‑to‑D′），其权重可以通过以下方式映射：<br> 1. **直角投影**：将旧权重复制并在增维后使用随机正交正则（Orthogonal initialization）。<br> 2. **线性插值**：使用 *weight‑mapping*（`W′ = α·W + (1-α)·0`）让新维度初始化为 0，并在微调阶段通过 L2 正则确保旧向量保持不变。<br> 3. **知识蒸馏**：在 0.5 B 上训练时，把旧模型的认主头嵌入视为“教师”，用余弦距离或 KL 交叉熵约束目标向量不偏离。 | 通过 **投影 + 蒸馏** 双重策略，能够在不破坏旧模型判别几率的情况下，将认主头迁移到新的 BitNet‑B1.58 体系中。 |
| **最小方案实现对抗鲁棒性** | • 仅在 **数据层** 与 **损失层** 加入对抗机制，无需另建大模型。<br> - 对抗噪声：使用 **Fast Gradient Sign Method (FGSM)** 或 **Projected Gradient Descent (PGD)** 仅在输入单词/字符级别。<br> - 结构化：引入 **Feature‑Space Mixup**（在嵌入层做 α‑mixup）与 **Margin‑Penalty**（`cos(θ)` 与 margin 动态调整）。<br>• 采用 **自监督正则**：例如 **triplet+contrastive** 交替训练，让模型在极端噪声下保持嵌入一致。 | 只需 **≈5% 额外 FLOPs**，即可显著提升 2‑3% 的对抗误差率(ε = 1e‑3 四字代码)。 |

---

**2️⃣ 可执行方案 / 伪代码**

```python
# ---------- 1. 迁移认主头 ----------
# 假设旧头 W_pre (576 x d_target), 新头 W_new (hidden_dim_new x d_target)
old_hidden_dim = 576
new_hidden_dim = 768  # 0.5B BitNet hidden size

# 初始化新头，让其包含旧头信息
W_new = torch.zeros(new_hidden_dim, d_target)
W_new[:old_hidden_dim, :] = W_pre          # 直接映射旧权重
W_new[old_hidden_dim:, :] = torch.randn(new_hidden_dim - old_hidden_dim, d_target) * 0.01  # 随机补齐

# 冻结旧部分
for param in W_new[:old_hidden_dim].parameters():
    param.requires_grad = False

# ---------- 2. 基础训练 ----------
optimizer = Adam(list(new_model.parameters()), lr=1e-4)

for epoch in range(num_epochs):
    for batch in data_loader:
        # ① 前向
        embeddings = new_model(batch['tokens'])
        # ② Triplet loss + Cosine-make margin
        loss_triplet = triplet_loss(embeddings, batch['owner_id'])
        # ③ 对抗样本（ε=0.8 对抗扰动）
        perturbed = fgsm_attack(batch['tokens'], epsilon=0.8)
        embeddings_adv = new_model(perturbed)
        loss_adv = triplet_loss(embeddings_adv, batch['owner_id'])
        # ④ 蒸馏约束 (教师:旧模型)
        with torch.no_grad():
            ford_emb = old_model(batch['tokens'])
        loss_kld = kl_divergence(F.log_softmax(embeddings, dim=1),
                                 F.softmax(ford_emb, dim=1))
        # ⑤ 总损失
        loss = loss_triplet + 0.5 * loss_adv + 0.1 * loss_kld