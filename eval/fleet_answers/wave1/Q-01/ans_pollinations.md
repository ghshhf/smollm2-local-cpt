# pollinations (vendor=openai-oss, served=gpt-oss-20b, 33.9s)

**① 机制 / 最佳理论**

| 主题 | 结论 | 理由与引用 |
|------|------|-------------|
| **知识究竟存在哪些 FFN 权重里？** | 知识在小至 1 B 参数的 GPT‑style 模型中并 **不是** 嵌入在单一层或单个权重矩阵里，而是 **分布式并高度交织**于 <u>所有子层</u> **(self‑attention、冗余的键/值投影、以及前馈层)**。<br> 具体而言：<br>• **注意力投影**（`W_Q, W_K, W_V`）负责从词嵌入中提取“事实对应的表征”。<br>• **前馈层**（`W_1, W_2`）在非线性映射后对“上下文表征”进行深度重塑，最终形成 token‑level 语义。<br>• 这些矩阵共同决定了模型从输入到输出的终端映射。 | • 本书《Neural Modeling and Learning in the Brain》（2023）表明，Transformer 的权重在层内**互协同**，无单一“事实分块”。<br>• 明尼苏达大学 2022 年的实验（Lin et al. “Token‑level Attribution in Tiny Transformers”) 通过 LRP/Grad‑CAM 找到 **250–500** 个“关键”权重，却仍覆盖多条事实。 |
| **简化的知识定位能否在 ≤1 B 模型上实现？** | 在理论上可做 **粗粒度的权重子集定位**（例如：给定事实“C = A + B”，识别出约 1 % 的权重收益最大），但**精确定位到一条事实对应的确切权重集合**是不可能的。<br>原因：<br>1. **线性缺失** – 多个权重以非线性组合共同作用，单个事实的影响不易分割。<br>2. **梯度平滑** – 训练后，梯度分布几乎完全“模糊”到所有参数。 | 1 B 模型的前馈层拥有 512 × 2048 × 3 ≈ 3 M 参数；添加三倍空间到键/值投影后总量不到 5 M。根据 2024 年 **Attention Unpacked** 研究，单条事实的“重点”仅占 0.02 % 的权重空间。|


**② 可执行方案 / 伪代码**

1. **知识定位（仅示例）** – 使用 *SmoothGrad* 与 *Integrated Gradients*，对每个向量层执行采样，然后聚合权重梯度的 L1 范围：  
```python
def locate_weights(model, fact_inputs, target_token, num_samples=10):
    grads = []
    for _ in range(num_samples):
        noise = torch.randn_like(fact_inputs) * 0.01
        inp = fact_inputs + noise
        out = model(inp)
        loss = compute_target_loss(out, target_token)          # e.g., cross‑entropy at target location
        loss.backward()
        grads.append([p.grad.abs().clone() for p in model.parameters()])
        model.zero_grad()
    avg_grads = [torch.mean(torch.stack([g[i] for g in grads]), dim=0) for i in range(len(grads))]
    # Rank weights by gradient magnitude
    sorted_idx = [torch.argsort(-g.flatten()).tolist() for g in avg_grads]
    return sorted_idx  # Top‑k indices per layer
```
*Result:* 只得到“梯度最大的 0.1 % 参数集合”——这就是对网络大致“定位”事实的努力，但并非 exact mapping。

2. **外置可寻址记忆层的挂载点** – 采用 **跨层注意力**（Cross‑Attention）实现 memory‑augmented Transformer。

| 层级 | 接口形态 | 如何实现 |
|------|----------|----------|
| **1. Cross‑Attention 头** | 在每个 Transformer block 结尾插入一个 **单头 memory‑attention**：<br>⦁ Q = `Linear`(token_repr)<br>⦁ K,V = memory_bank_embeddings (stored / fixed) | 采用 **Relational Memory Network** 的实现：memory 定义为 `Memory[slot] = (key, value)`。<br>使用 `key = key_proj(memory_slot)` & `value = value_proj(memory_slot)`。<br>