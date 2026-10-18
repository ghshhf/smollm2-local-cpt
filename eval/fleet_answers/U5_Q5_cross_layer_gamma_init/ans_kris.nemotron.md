# kris.nemotron (vendor=nvidia, served=nvidia/nemotron-3-ultra-550b-a55b, 53.2s)

### 裁决结论先行
**复制式初始化在 BitNet b1.58 下能「严格保函数」（bit-exact），前提是：把 `gamma` 当作 **buffer 同步复制**，而非运行时重算 `mean(|W|)`。**  
文献中无现成「量化感知 Net2Net」，但可把 **LSQ/STE 直通估计器的「冻结 scale」范式** 直接迁移过来 —— 这就是最佳理论依据。

---

### ① 机制/最佳理论
| 问题 | 理论锚点 | 为什么能工作 |
|------|----------|----------------|
| **1. 文献支撑** | **LSQ (Learned Step Size Quantization, CVPR 2020)** + **Net2Net (ICML 2016)** 的组合推论 | LSQ 核心结论：**量化 scale 只要在前向传播固定，反向只需直通梯度**。Net2Net 要求 `f_new(x) ≡ f_old(x)`。若 `gamma` 视为 **非参数 buffer**（同步复制），前向数学表达式 `gamma * sign(W) @ x` 与源层逐元素完全一致 → **函数保持证明成立**。 |
| **2. 显式复制 gamma** | **Buffer 而非 Parameter** 机制 | `gamma = mean(|W|)` 是 **确定性函数**。把它从「运行时统计量」改为「随权重一起 `register_buffer` 复制的常量」，消除了「重新计算带来的数值漂移」。这等价于把量化算子退化为 **固定缩放的线性算子**，完美符合 Net2Net 函数保持定义。 |
| **3. 三值化输出等价性** | **全局标量的传递性** | 源层：`y = γ_s * sign(W_s) @ x`<br>新层（复制权重+复制 gamma）：`y' = γ_s * sign(W_s) @ x`（`sign` 逐元素确定，`γ_s` 标量完全相同）<br>**∴ y' ≡ y**，无任何近似。**量化步长退化/Clamp 问题不存在**，因为根本不重新统计。 |

> **关键洞察**：BitNet b1.58 的 `gamma = mean(|W|)` **仅在训练期每步重算**；**推理/部署/结构变更时必须冻结为常量**。Net2Net 结构变更本质上是「部署态拓扑手术」，理应沿用部署态规则 —— **冻结 gamma 同步复制**。

---

### ② 可执行方案 / 伪代码
```python
# ---------- 通用工具 ----------
def freeze_gamma(module: nn.Module):
    """把所有 QuantLinear 的 gamma 从『运行时统计』改为『buffer 常量』"""
    for m in module.modules():
        if isinstance(m, BitLinear):          # 你的三值化线性层
            if not hasattr(m, 'gamma_frozen'):
                m.register_buffer('gamma_frozen', m.gamma.detach().clone())
                m.gamma = m.gamma_frozen      # 覆盖 property / attribute
                del m.gamma_frozen            # 只留 gamma (buffer)

# ---------- Net2Net 加宽 ----------
def widen_layer(src: BitLinear, tgt_width: int) -> BitLinear:
    """
    src: 原层 (in_dim, out_dim)
    返回: 新层 (in_dim, tgt_width)  且 f_new ≡ f_src (在 src.out_dim 维度上)
    """
    assert tgt_width >= src.out_features
    new = BitLinear(src.in_features, tgt_width, bias=src.bias is not None)
    with torch.no_grad():
        # 1. 权重复制 + 零填充
        new.weight[:src.out_features] = src.weight
        new.weight[src.out_features:] = 0.0
        # 2. gamma 显式复制 (buffer 级别，不重算 mean|W|)
        new.gamma.copy_(src.gamma)            # 关键：buffer 同步
        # 3. bias 同理
        if src.bias is not None:
            new.bias[:src.out_features] = src.bias
            new.bias[src.out_features:] = 0.0
    return new

# ---------- Net2Net 加深 (插层: Identity 初始化) ----------
def insert_identity_layer(prev: BitLinear, nxt: BitLinear) -> BitLinear:
    """
    在 prev -> nxt 之间插入一层 Identity 映射 (宽度不变)
    返回新层 new, 使得 nxt(new(prev(x))) ≡ nxt(prev(x))
    """
    d = prev.out_features
    new = BitLinear(d, d, bias=False)
    with torch.no_grad():
        new.weight.copy_(torch.eye(d))        # 单位矩阵
        new.gamma.copy_(torch.ones_like(new.gamma))  # gamma=1.0 (buffer)
    return new
```

**部署前必须调用一次**：`freeze_gamma(model)` —— 之后所有结构变更、推理、ONNX 导出全部走 `gamma (buffer)`。

---

### ③ 可证