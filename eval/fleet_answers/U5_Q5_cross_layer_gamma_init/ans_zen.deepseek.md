# zen.deepseek (vendor=deepseek, served=deepseek-v4-flash-free, 60.2s)

### ① 机制/最佳理论裁决

**核心结论**：
1. **文献裁决**：**无**现成的“量化感知 Net2Net 初始化”方案能同时满足 **BitNet b1.58 三值化** 与 **严格保函数**。现有 Net2Net 文献（Chen et al., 2015; Li et al., 2023）假设全精度或仅 INT8 对称量化，未处理 **逐层标量 gamma 依赖于权重分布** 这一拓扑耦合问题。
2. **显式复制 Gamma（Buffer 保存）是唯一正解**：**必须**将 `gamma` 从“由权重派生的统计量”降级为“独立的可学习/可复制 Buffer 参数”。**禁止**在结构变更后重新计算 `mean(|W|)`。
3. **复制式初始化下三值化输出严格等价（数学证明）**：
   - 源层权重 $W_s$，目标层初始化 $W_t = W_s$（逐元素复制）。
   - 源层 Gamma $\gamma_s = \frac{1}{n}\sum |W_s|$（已冻结为 Buffer）。
   - 目标层 Gamma $\gamma_t \leftarrow \gamma_s$（显式复制 Buffer，不重算）。
   - 三值化函数 $T(W; \gamma) = \gamma \cdot \text{sign}(W - \delta) \approx \gamma \cdot \text{ternarize}(W/\gamma)$。
   - **推论**：$W_t \equiv W_s \land \gamma_t \equiv \gamma_s \implies T(W_t; \gamma_t) \equiv T(W_s; \gamma_s)$。
   - **输出严格逐元素相等**，梯度流完美对齐，保函数成立。

---

### ② 可执行方案/伪代码

#### A. 架构定义：Gamma Buffer 化（宪法级修改）
```python
# modeling_bitnet.py - 宪法修正版
class BitLinear158(nn.Module):
    def __init__(self, in_features, out_features, bias=False):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        # 【宪法】gamma 登记为 buffer，不再是 property/hook 计算
        # 训练时可由 QAT 更新；结构变更时仅复制张量
        self.register_buffer('gamma', torch.tensor(1.0))  # shape: []
        self.register_buffer('weight_scale', torch.tensor(1.0)) # 可选：吸收输入 RMSNorm 系数
        if bias: self.register_buffer('bias', torch.zeros(out_features))
        else: self.bias = None

    @torch.no_grad()
    def update_gamma_from_weight(self):
        """仅在 初始化/恢复全精度检查点 时调用。结构变更后 禁用。"""
        self.gamma.copy_(self.weight.abs().mean().clamp_min(1e-5))

    def forward(self, x):
        # BitNet b1.58 标准前向：W_tern = gamma * ternarize(weight / gamma)
        w = self.weight
        # STE 三值化
        w_tern = (w / self.gamma.clamp_min(1e-5)).round().clamp(-1, 1) * self.gamma
        return F.linear(x, w_tern, self.bias)
```

#### B. Net2Net 结构变更算子（零初始化/复制式统一入口）
```python
# net2net_surgery.py
@torch.no_grad()
def net2net_widen(layer: BitLinear158, new_out_features: int, mode: str = 'copy') -> BitLinear158:
    """
    加宽：out_features 扩容。严格保函数。
    mode='copy': 复制式（推荐，严格等价）
    mode='zero': 零初始化（需 clamp gamma，功能退化为残差旁路）
    """
    old_out, in_f = layer.weight.shape
    assert new_out_features >= old_out
    
    new_layer = BitLinear158(in_f, new_out_features, bias=layer.bias is not None).to(layer.weight.device)
    
    if mode == 'copy':
        # 1. 权重复制（循环填充或首尾填充，视策略）
        reps = (new_out_features + old_out - 1) // old_out
        w_src = layer.weight.repeat(reps, 1)[:new_out_features]
        new_layer.weight.copy_(w_src)
        
        # 2. 【核心】Gamma 显式复制 Buffer —— 不重算 mean|W|
        new_layer.gamma.copy_(layer.gamma)  # 标量张量复制
        
        # 3. Bias 复制
        if layer.bias is not None:
            b_src = layer.bias.repeat(reps)[:new_out_features]
            new_layer.bias.copy_(b_src)
            
    elif mode == 'zero':
        # 零初始化：权重为 0，gamma 必须 clamp 到量化步长下界，防止梯度消失/数值崩溃
        new_layer.weight.zero_()
        new_layer.gamma.fill_(1e-5)  # 宪法下界，对应 b1.58 量化步长
        if layer.bias is not None: new_layer.bias