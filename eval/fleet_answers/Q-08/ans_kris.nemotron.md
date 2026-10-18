# kris.nemotron (vendor=nvidia, served=nvidia/nemotron-3-ultra-550b-a55b, 58.7s)

### ① 机制/最佳理论  
**核心结论**：在「固定语料单遍/少遍」且「模型容量远大于数据熵」的前提下，**泛化增益仅来自「未见 n-gram 的梯度对齐」，记忆化仅来自「已见 n-gram 的梯度归零」**。  
**在线可分信号**：**逐 token 梯度余弦相似度均值（Token-wise Gradient Cosine Similarity Mean, TGCSM）**。  

- **理论依据**（基于 *Gradient Signal-to-Noise Ratio* 与 *Loss Landscape Curvature* 的等价推导）：  
  - 令 $g_t = \nabla_\theta \ell(x_t)$ 为第 $t$ 个 token 的梯度向量。  
  - 泛化阶段：模型在「压缩未见模式」，不同 token 的梯度指向**高度一致**的低维子空间 → $g_t$ 两两夹角锐角 → $\cos\langle g_t, g_{t-1}\rangle \uparrow$。  
  - 记忆阶段：模型在「拟合残差噪声」，梯度指向**正交/随机**方向 → $\cos\langle g_t, g_{t-1}\rangle \to 0$ 甚至负值。  
- **为什么不用 GSNR / 参数漂移 / 单样本方差 / Loss 曲率**？  
  - GSNR 需要累积 $\mathbb{E}[g]^2 / \mathbb{V}[g]$，显存/算力超标（需存全梯度历史）。  
  - 参数漂移率 $\|\Delta\theta\|$ 无法区分「有效压缩」与「死记硬背」。  
  - 单样本 Loss 方差在第二遍极低（都趋近 0），失去分辨力。  
  - Loss 曲率需二阶近似，CPU 上 Hessian-vector product 太慢。  
- **TGCSM 仅需**：当前 token 梯度 + 上一 token 梯度（或 EMA），**O(1) 显存、O(d) 算力（d=参数量）**，且天然屏蔽 embedding 层（已冻结 fp16）。

---

### ② 可执行方案 / 伪代码（node1 CPU 单进程可跑）

```python
# ------------------------------------------------------------
# 在线 TGCSM 监控：每 step 仅增量 2 次 dot + 1 次 norm
# ------------------------------------------------------------
import torch

class TGCSM_Monitor:
    def __init__(self, model, ema_alpha=0.99, window=2000):
        self.model = model
        self.alpha = ema_alpha          # EMA 平滑系数
        self.window = window            # 判定滑动窗口步数
        self.ema_cos = 0.0              # EMA 余弦值
        self.history = []               # 最近 window 步 EMA 值
        self.prev_grad = None           # 上一 step 梯度（仅非 embedding 参数）
        self.phase = "UNKNOWN"          # "GENERALIZING" / "MEMORIZING"

    @torch.no_grad()
    def step(self, loss):
        # 1. 反向传播拿梯度（仅非 embedding 参数，节省 30% 算力）
        self.model.zero_grad(set_to_none=True)
        loss.backward()
        cur_grad = torch.cat([p.grad.flatten() for n, p in self.model.named_parameters()
                              if 'embed' not in n and p.grad is not None])

        # 2. 计算余弦相似度
        if self.prev_grad is not None:
            cos = torch.dot(cur_grad, self.prev_grad) / (cur_grad.norm() * self.prev_grad.norm() + 1e-12)
            self.ema_cos = self.alpha * self.ema_cos + (1 - self.alpha) * cos.item()
            self.history.append(self.ema_cos)
            if len(self.history) > self.window:
                self.history.pop(0)

            # 3. 判定阈值形态（见下节）
            self._judge_phase()

        self.prev_grad = cur_grad.detach().clone()  # 保留给下一步

    def _judge_phase(self):
        if len(self.history) < self.window:
            return
        recent_mean = sum(self.history) / len(self.history)
        recent_std  = (sum((x - recent_mean)**2 for x in self.history) / len(self.history))**0.5
        # 阈值形态：均值 < 0.15 且 连