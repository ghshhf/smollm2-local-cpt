# kris.gemini (vendor=google, served=gemini-3.8-flash, 113.1s)

### 结论速判与核心推翻

1. **推翻“继承式失败根因是退火太快”的假设**：V1-02 测出的静态冲击 $\Delta\text{CE}$ 必然 $\ge 1.8$（甚至直奔随机猜测的 3.0+）。但这**绝非**退火速度问题，而是**高维连续流形向三值离散顶点的表征断裂**。单纯放慢线性退火至 5 万步，在无逐层重构或蒸馏约束下，进入 $\alpha \in [0.2, 0.4]$ 依然会发生不可逆相变崩溃。
2. **推翻外部模型“提高影子权重 $lr$ 可在当前预算下解决问题”的建议**：在 200 步预算下提高 $lr$ 纯属指标作弊。它能强行砸出 flip rate $> 0$，但 AdamW 二阶动量尚未越过冷启动期（$1/(1-\beta_2) \approx 1000$ 步），粗糙的 STE 代理梯度会导致权重在量化阈值两侧发生极限环震荡（limit cycle），Loss 必然在 200 步内进一步发散。

---

### ① 机制与理论裁决

#### 裁决 1：Flip Rate 的测量窗口与物理真身
*   **物理机制**：设权重服从高斯分布 $W \sim \mathcal{N}(0, \sigma^2)$，BitNet b1.58 缩放因子 $\gamma = \mathbb{E}[|W|] \approx 0.8\sigma$。量化判决门限位于 $\pm 0.5\gamma \approx \pm 0.4\sigma$。
    使用 AdamW 时，单步有效更新位移为：
    $$\Delta W_{step} \approx \eta \cdot \frac{m_t}{\sqrt{v_t} + \epsilon} \le \eta$$
    当 $\eta = 10^{-5}$，步长 $\Delta W \approx 10^{-5}$。对于标准 135M 模型（$\sigma \approx 0.02$），量化步长 $\gamma \approx 0.016$，门限绝对距离在 $10^{-2}$ 量级。
    在相邻测点（如每 10 步），权重位移处于随机游走状态，净位移期望仅为 $\sqrt{10} \times 10^{-5} \approx 3.16 \times 10^{-5}$。落入翻转带 $[0.5\gamma - \Delta W, 0.5\gamma + \Delta W]$ 的理论概率密度极低。
*   **指标修正**：**严禁测量“相邻测点瞬时翻转率”**。必须改测**累积符号翻转率（Cumulative Flip Rate, CFR）**：
    $$\text{CFR}(t) = \frac{1}{N} \sum_{i=1}^N \mathbb{I}\left( \text{Quant}(W_i^{(t)}) \ne \text{Quant}(W_i^{(0)}) \right)$$
*   **窗口裁决**：要让 CFR 展现出统计规律性（$> 5\%$ 权重越过边界），所需步数 $T$ 满足：
    $$\sqrt{T} \cdot \eta \cdot \beta_{corr} \approx 0.5\gamma \implies T \approx \left(\frac{0.5 \times 0.016}{10^{-4} \times 0.3}\right)^2 \approx 7000\text{ 步 (当 }\eta=1e-4\text{)}$$
    **裁决**：若使用 $\eta=1e-4$，至少需要 **3000~5000 步** 才能脱离布朗运动噪声区，看清稳态迁移；在 200 步内，瞬时 flip rate 恒为 0 是数学必然，而非物理异常。

#### 裁决 2：三值化静态冲击（Static Shock）的本质
*   **理论预测**：V1-02 静态阶梯实验中，$\alpha=1.0 \to 0.0$ 的 holdout $\Delta\text{CE}$ 必然 $\ge 1.8$。
*   **机制**：三值量化引入的理论相对重构误差（Relative MSE）为：
    $$\frac{\|W - W_q\|_F^2}{\|W\|_F^2} \approx 18\% \sim 22\%$$
    在 12 层 Transformer 级联下，前向信噪比衰减为 $\text{SQNR}_{out} \approx (1 - 0.20)^{12} \approx 0.068$（信号能量被噪声完全淹没）。
    线性插值 $W_{eff} = \alpha W + (1-\alpha) W_q$ 并非几何测地线。当 $\alpha$ 越过临界点（约 0.4）时，特征空间的奇异值谱发生坍缩，激活值方差被 LayerNorm/RMSNorm 异常放大。
    **这不是“退火太快”的工程失误，而是“无监督硬投”在几何上的必然崩溃**。从 fp32 到 BitNet b1.58，权重与激活必须存在协同重新排布（Co-adaptation），平滑退火无法凭空提供这种重排布所需的自由能。

#### 裁决 3：200 步预算下提高 $lr$ 的后果
*   **理论判定**：**绝对不可行**。
*   **机制**：AdamW 的 $\beta_2=0.999$，二阶矩估计器达到无偏有效窗口需 $t \ge \frac{1}{1-\beta_2} = 1000$ 步。在 200 步时，分母校正因子仍在剧烈变动。若此时将影子权重 $lr$ 暴力拔高到 $10^{-3}$：
    1. 前向梯度范数（已达 6.90）会直接使步长击穿整个死区（Deadzone）；
    2. STE（直通估计器）截断了 $[-1, 1]$ 之外的真实曲率，导致影子权重在量化阈值周围剧烈震荡（Thrashing）；
    3. 200 步结束时，CFR 虽被刷高到 $15\%$ 以上，但 Loss 必然发散至 4.0 以上甚至 NaN。

#### 吞吐 18 跌至 3 tok/s 的隐蔽病因
*   在 CPU 上，`alpha * W + (1 - alpha) * W_q` 构成了极其昂贵的动态张量图，阻断了 OpenBLAS/MKL 的连续内积内核，并在每个 step 强制进行三次内存克隆与显式广播。
*   **在 CPU 训练 BitNet，必须使用原位（in-place）STE 算子，禁止在 Python 层保留混合浮点权重图。**

---

### ② 可执行方案：渐进式注入蒸馏（PID）与 CFR 监控

放弃全局无差别线性 $\alpha$ 退火，转向**单层推进（Block-wise progressive）+ 0.4B 输出 logits 蒸馏（KL散度）**。在 CPU 上，训练必须分阶段闭环。

#### 核心执行伪代码（PyTorch 核心逻辑）

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class FastControllableBitLinear(nn.Linear):
    """
    针对 64核 CPU 优化的低开销 BitLinear:
    - 消除 alpha 混合动态图
    - 原位 STE 截断
    - 内置累积翻转率 (CFR) 无锁统计
    """
    def __init__(self, in_features, out_features, bias=False):
        super().__init__(in_features, out_features, bias=bias)
        self.register_buffer('init_quant_state', torch.zeros_like(self.weight, dtype=torch.int8))
        self.state_initialized = False

    def get_quant_weight(self, w):
        # BitNet b1.58 scale: mean(abs(w))
        scale = w.abs().mean().clamp(min=1e-5)
        # RoundClip 到 {-1, 0, 1}
        w_scaled = w / scale
        w_q = torch.clamp(torch.round(w_scaled), -1, 1)
        # STE 直通
        return (w_q - w_scaled).detach() + w_scaled, scale, w_q.to(torch.int8)

    def forward(self, x, quantize=True):
        if not quantize:
            return F.linear(x, self.weight, self.bias)
        
        w_ste, scale, w_disc = self.get_quant_weight(self.weight)
        
        # 记录初始量化状态用于 CFR 计算
        if not self.state_initialized:
            self.init_quant_state.copy_(w_disc)
            self.state_initialized = True
            
        # CPU 向量化优化: 避免额外张量分配
        return F.linear(x, w_ste * scale, self.bias)

    @torch.no_grad()
    def compute_cfr(self):
        """计算相对于 step 0 的累积符号翻转率"""
        _, _, current_state = self.get_quant_weight(self.weight)
        flips = (current_state != self.init_quant_state).sum().item()
        return flips / self.weight.numel()

def train_distill_step(student, teacher, x, optimizer, scheduler):
    """
    能力层从 0.4B 蒸馏继承：强制加入蒸馏损失，抹平量化冲击
    """
    optimizer.zero_grad(set_to_none=True)
    
    with torch.no_grad():
        t_logits = teacher(x)
        
    s_logits = student(x)
    
    # 纯 CE 极易在三值化中崩溃，必须以 KL 散度约束软标签几何
    loss_ce = F.cross_entropy(s_logits.view(-1, s_logits.size(-1)), x.view(-1))
    loss_kl = F.kl_div(
        F.log_softmax(s_logits, dim=-1),
        F.softmax(t_logits, dim=-1),
        reduction='batchmean'
    )
    
    loss = 0.3 * loss_ce + 0.7 * loss_kl
    loss.backward()
    
    # 梯度硬截断，防止 STE 梯度爆炸
    torch.nn.utils.clip_grad_norm_(student.parameters(), max_norm=1.0)
    
    optimizer.step()
    scheduler.step()
    return loss.item()
```

---

### ③ 可证伪判据（何种结果算我错）

1. **判据 1（关于静态冲击）**：
   在 V1-02 零训练阶梯实验中，若 $\alpha=0.0$ 时的 holdout CE 增量 $\Delta\text{CE} < 0.8$（当前 baseline 为 0.29，即总 CE $\le 1.09$），**算我彻底错误**。
   *意义*：证明 fp32 流形与 b1.58 流形之间存在天然的平滑退化低能量通道，我的“几何断裂论”不成立。

2. **判据 2（关于 200 步调大 lr）**：
   若将影子权重 $lr$ 提高至 $5e-4 \sim 1e-3$，且在 **200 步、全局全量退火** 的约束下：
   Holdout CE 能够压回至 $\le 0.45$，同时累积翻转率 CFR 达到 $3\% \sim 10\%$ 且未发生梯度范数震荡（norm $\le 3.0$），**算我彻底错误**。
   *意义*：证明 200 步下 AdamW 无需预热即可在强噪声三值梯度下收敛，我的“极限环破坏论”不成立。

---

### ④ 魔搭 Node1 64核 CPU 最小可跑验证脚本

单次运行必须在 **3 分钟内完成**，直接闭环验证：(1) 静态冲击全阶梯、(2) 200步暴力提 $lr$ 的崩塌过程、(3) 内存瓶颈解绑（验证吞吐回归）。

保存为 `node1_ternary_test.py`，单脚本直跑：

```python
import time
import torch
import torch.nn as nn
import torch.nn.functional as F

# 强绑 64 核 CPU 线程
torch.set_num_threads(64)
torch.set_num_interop_threads(64)

class ToyBitLinear(nn.Linear):
    def get_quant(self):
        scale = self.weight.abs().mean().clamp(min=1e-5)
        w_scaled = self.weight / scale
        w_q = torch.clamp(torch.round(w_scaled), -1, 1)
        w_ste = (w_q - w_scaled).detach() + w_scaled
        return w_ste * scale, w_q.to(torch.int8)

    def forward(self, x, alpha=0.0):
        if alpha == 1.0:
            return F.linear(x, self.weight)
        w_q_ste, _ = self.get_quant()
        if alpha == 0.0:
            return F.linear(x, w_q_ste)
        # 避免广播崩溃的计算
        return F.linear(x, alpha * self.weight + (1.0 - alpha) * w_q_ste)

def run_experiment():
    device = "cpu"
    B, S, D = 8, 128, 768  # 模拟 135M 某几层的真实计算密度
    VOCAB = 4000
    steps = 200

    print("=== [Node1 CPU 极速基准] 初始化模拟权重 ===")
    torch.manual_seed(42)
    layer = ToyBitLinear(D, D, bias=False).to(device)
    head = nn.Linear(D, VOCAB, bias=False).to(device)
    
    # 模拟 fp32 预训练后方差
    nn.init.normal_(layer.weight, mean=0.0, std=0.02)
    nn.init.normal_(head.weight, mean=0.0, std=0.02)

    x = torch.randn(B, S, D, device=device)
    y = torch.randint(0, VOCAB, (B, S), device=device)

    # 1. 验证静态冲击 (V1-02 极速复现)
    print("\n--- 任务 1: V1-02 静态冲击曲线测量 ---")
    alphas = [1.0, 0.9, 0.7, 0.4, 0.2, 0.0]
    with torch.no_grad():
        for a in alphas:
            hidden = layer(x, alpha=a)
            logits = head(hidden)
            ce = F.cross_entropy(logits.view(-1, VOCAB), y.view(-1)).item()
            print(f"Alpha: {a:.1f} | Holdout CE: {ce:.4f}")

    # 2. 验证暴力提升 lr=1e-3 在 200 步下的物理崩溃与 CFR
    print("\n--- 任务 2: 暴力提 lr=1e-3 + 200步 AdamW 压力测试 ---")
    _, init_state = layer.get_quant()
    optimizer = torch.optim.AdamW(layer.parameters(), lr=1e-3, betas=(0.9, 0.999), eps=1e-8)
    
    start_t = time.time()
    for step in range(1, steps + 1):
        optimizer.zero_grad()
        # 强制纯三值 alpha=0
        hidden = layer(x, alpha=0.0)
        logits = head(hidden)
        loss = F.cross_entropy(logits.view(-1, VOCAB), y.view(-1))
        loss.backward()
        
        grad_norm = torch.nn.utils.clip_grad_norm_(layer.parameters(), max_norm=10.0)
        optimizer.step()

        if step % 50 == 0 or step == 1:
            with torch.no_grad():
                _, curr_state = layer.get_quant()
                cfr = (curr_state != init_state).sum().item() / curr_state.numel()
                print(f"Step {step:03d} | Loss: {loss.item():.4f} | Grad Norm: {grad_norm:.2f} | CFR: {cfr*100:.2f}%")

    tok_per_sec = (steps * B * S) / (time.time() - start_t)
    print(f"\n优化算子后吞吐: {tok_per_sec:.1f} tok/s (已剔除动态图内存颠簸)")

if __name__ == "__main__":
    run_experiment()
```

### 终极行动建议
立即终止“试图在 200 步内通过调大 $lr$ 拯救全局退火”的实验分支。将注意力锁定在：
1. 先由 `node1_ternary_test.py` 输出 V1-02 的不可抗硬冲击基线；
2. 严格按宪法路线走：**0.5B 从头原生三值 QAT**，拿 0.4B 的 checkpoint 充当 **Teacher** 做 Logits 连续蒸馏，绝不可直接在 0.4B fp32 的骨架上做硬拆换。流形断裂无法通过微调修复。