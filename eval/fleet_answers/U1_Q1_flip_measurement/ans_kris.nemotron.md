# kris.nemotron (vendor=nvidia, served=nvidia/nemotron-3-ultra-550b-a55b, 54.7s)

### � 裁决

#### 1. Flip Rate 测量窗口：`steps=5000, lr=1e-4` **勉强够“看见”，不够“看清”** —— 必须改为 **`lr=1e-3, steps=2000`** 或 **`lr=1e-4, steps=20000`**

**① 机制/最佳理论**  
三值量化步长 $\Delta \approx 2\langle|W|\rangle$。影子权重 $W$ 需移动 $\sim 0.5\Delta$ 才能触发符号翻转。  
物理位移上界：$\delta W_{\max} \approx \eta \cdot \|\nabla W\| \cdot T$。  
你的实测：$\|\nabla W\| \sim 7$（后期），$\langle|W|\rangle \sim 0.05$（典型 135M 模型），$\Delta \sim 0.1$。  
- `lr=1e-4, T=5000` → $\delta W_{\max} \approx 3.5 \times 10^{-3}$ ≈ **$0.035\Delta$** → 期望翻转概率 $< 10^{-3}$，**仍看不见**。  
- `lr=1e-3, T=2000` → $\delta W_{\max} \approx 1.4 \times 10^{-2}$ ≈ **$0.14\Delta$** → 边界带 34~46% 权重中约 5~7% 会翻转，**flip rate 可测**。  
- `lr=1e-4, T=20000` → 同数量级，但算力 10 倍，**不划算**。

**② 可执行方案（伪代码）**  
```python
# V1-03_flip_window_probe.yaml
lr: 1e-3                 # 关键：提高 10 倍
steps: 2000              # 200 步 * 10 = 2000 步，总 token 不变
alpha_schedule: "linear" # 保持原退火，验证“退火太快”假说
log_every: 50
metrics:
  - flip_rate_per_layer  # 每层单独记录
  - grad_norm_shadow
  - holdout_ce
```
**若算力极其受限**，可只跑 **前 2 层 + 后 2 层**（代表浅/深分布），batch size 不变，显存/算力几乎不增。

**③ 可证伪判据**  
- ❌ `lr=1e-4, steps=5000` 仍得 `flip_rate < 1e-4` → **窗口仍不足，裁决错**。  
- ✅ `lr=1e-3, steps=2000` 出现 `flip_rate ∈ [1e-3, 1e-2]` 且呈现 **先升后降**（边界带清空→稳定）→ **窗口合格，裁决对**。

**④ node1 CPU 最小可跑版**  
`batch=4, seq=256, 2000 steps` ≈ **1.2 万 token** ≈ **40 分钟**（3 tok/s × 2000 × 1.3 倍开销）。可跑。

---

#### 2. 三值化静态冲击阈值：**`delta_CE < 0.3` = 可承受 / `≥ 0.8` = 静态冲击大、继承式失败根因是退火太快**  
（你给的 `1.2` 过松，`0.3` 过紧，**取中间值 `0.8` 为硬分界**）

**① 机制/最佳理论**  
- BitNet b1.58 论文 & 复现（1.3B/3B）显示：**从头训练**三值模型，训练初期 CE 仅比 fp16 基线 **高 0.15~0.25 nats**（≈0.1~0.15 bits/token）。  
- **继承式**（fp32→三值）属于 **“模型编辑”**，信息瓶颈在 **权重重投影**。若 `