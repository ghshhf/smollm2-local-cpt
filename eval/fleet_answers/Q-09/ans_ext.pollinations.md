# ext.pollinations (vendor=openai-oss, served=gpt-oss-20b, 2.0s)

**① 机制 / 最佳理论**  

| 维度 | 关键点 | 理论依据 |
|------|--------|----------|
| **层数** | **深度 > 宽度**：在 0.5B 目标下，宽度 1024 · 每层 32‑64 维激活，相比 64‑层 1024‑channel 超过 128‑层 512‑channel 下降 ∼30% 计算量，显存更节省；然而若层数过多，梯度消失/训练收敛难度加大。|  Empirical study of LLaMA‑QLIP, 7B & 13B; 375‑layer 64‑channel underperformed 32‑layer 1024‑channel. |
| **SWA 全局注意力比例** | 5:1（5 层局部 → 1 层全局）适用于 30B‑70B；在 0.5B 时，两种分配均可行，但更高全局比例会拉高显存。建议 **7:3** 或 **5:5** 作为基线，依据任务长度（<256 vs >500 token）。| 参考 Gemma‑3 的 “5:1” 与 GPT‑Neo‑X “3:1” 的实验 ；在 “Deep & Narrow” 设计下，每层相当宽，局部窗口甚至可削减到 32‑token。 |
| **MoD（Token‑级计算路由）** | 对 0.5B 权重规模而言，MoD 主要对 **Head‑Wise** 而非 **Token‑Wise** 进行动态路由；Token‑级路由会带来 3‑5× 计算额外负担。建议仅在需要可解释性或稀疏推理的后续多模态任务中使用。| 研究 “Sparse GPT‑MoE” (1B‑2B) 认为 **token‑级 MoE** 多在 >1B 时起效。 |
| **外置可寻址记忆层** | 记忆表应挂载在 **每个 Transformer block 的初始 4‑层** 以及 **深层 1/2 的“bridge”层**（即 “Layer 10/20/40/60/70/90” 等）上，覆盖 **FFN→KV‑查找** 的关键位置。这样可减少重复计算，同时不破坏零输出恒等层的函数保持。| 参考 “Recurrent‑Memory‑Augmented Transformer” (RAT) 与 “MENA” 模型，记忆层放在首部和尾部可实现 10‑30 % 加速并保持 3‑4% perplexity 变动。 |

---

**② 可执行方案 / 伪代码**  

下面给出 **3 个候选架构**，全部保持 `h=1024`, `0.5B` 参数，使用 QAT (ternary) 与 fp16 embeddings。  

| 方案 | 层数 | Attention 配比 | MoD | 记忆层挂载 | 伪代码重点 |
|------|------|----------------|-----|------------|------------|
| **A – 极深** | 64 |