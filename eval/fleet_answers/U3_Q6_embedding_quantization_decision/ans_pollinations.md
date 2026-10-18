# pollinations (vendor=openai-oss, served=gpt-oss-20b, 31.0s)

**① 机制/最佳理论**

- **比例与成本**  
  - 0.2 B 参数模型（hidden = 576）中，embedding 占比 ≈ 17 %（≈28 M 字节）。  
  - 若用 fp16：占用 56 M 字节；若用 ternary（3 bit）：占用 10.5 M 字节（≈ 5.3 倍压缩）。  
  - 但对准频率低的词进行 3‑bit 量化，会使这些词的查询表查到的向量精度大幅下降，导致查询时直接影响多分类/对齐，同时在反向传播中梯度会被“截断”，高频词的梯度也会被不恰当地拉高，造成“梯度漂移”。  
- **梯度与自适应**  
  - 针对词频的“混合精度”方案，即 **base_fp16 + delta_ternary**（LoRA‑style）。trainable 只更新 delta，保持基线权重不变，降低梯度“漂移”风险。  
  - 该方案的核心理论是：  
    - **参数稀疏化**：Δ 只对低频词使用，保证在高频词上保持 fp16 精度。  
    - **梯度聚居**：因为 Δ 与 base 在同一日语向量空间累加，梯度在全局不会出现梯度爆炸/消失。  
- **硬件吞吐与带宽**  
  - GPL CPU → 64核 node1：fp16 内存访问通常在 16 GB/s，它已满负荷；3‑bit 访问则需要极致压缩、解码开销，CPU 上的解码会成为瓶颈（尤其在 15 ms/Token 的目标下）。  
  - GPU 后续可使用 GPU 的 *int8* / *int3* 支持，但现阶段 CPU 仅支持 *fp16/float32*，若使用 ternary 必须手写 SIMD 解码路径，带来额外开发与维系成本。  
- **贴合演化路线**  
  - **0.1 B → 0.4 B** 主轴是 **Net‑2‑Net** 扩容； embeddings 看重点是保持 **fp16** 以便后续 0.5 B 的蒸馏继承。  
  - 0.2 B 正是两侧的 “翻转点”，建议 **保持 fp16**，除非在评估阶段发现 **实际性能损失 > 5 %** 或 **能耗 / 带宽** 超标，否则不主动尝试 ternary 量化。

**结论**  
- **0.2 B** 规模最好**不三值化 embeddings**。  
- 可以做**少量实验（≈ 1 % 低频词混合 3‑bit）**验证后决定；如效果不明显或梯度异常，直接回退到 fp16。

---

**② 可执行方案或伪代码**

```python
# 伪代码：Hybrid embedding – base_fp16 + delta_ternary (LoRA-style)

class HybridEmbedding(nn.Module):
    def __init__(self, vocab_size, d_model, freq_threshold=500):
        super().__init__()
        # base weights: fp16
        self.base = nn.Embedding(vocab_size, d_model).half()

        # delta weights: ternary (Binarized & scale)
        # delta is zero for tokens >= freq_threshold
        self.delta = nn.Parameter(torch.zeros(vocab_size, d_model))
        self.delta.register_hook(lambda grad: grad * 0)  # Freeze initially

        # mask: 1 for low‑freq, 0 for high‑freq
        mask = torch.arange(vocab_size) < freq_threshold
        self.register_buffer("mask", mask.to(torch.bool))

        # scale factor for ternary: [-s, 0, +s]
        self.scale = nn.Parameter(torch.full((1,), 1.0))

        # Optimizer will only update var with mask==1
        self.delta.data = self.delta.data * self.mask.float().unsqueeze(-1)

    def forward(self, token_ids):
        # base lookup
        base_vec = self.base(token_ids).half()

        # ternary delta lookup
