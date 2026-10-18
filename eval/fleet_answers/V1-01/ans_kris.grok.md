# kris.grok (vendor=xai, served=grok-4.7, 68.7s)

① 三值 QAT 在 CPU 上的真实瓶颈是 STE 失配引入的梯度噪声（代理梯度估计偏差在小批量/从零训练时易放大），而非纯算力（135M 模型 FLOPs 在 64 核 CPU + AVX-512 下算力充足）或内存带宽（数据搬运量对该规模可控）。最佳理论：STE 本质是前向离散映射到后向连续代理，噪声方差随批大小减小而放大；在 CPU 低并行环境下梯度累积效率低，更易陷入不稳定收敛（对比 GPU 高吞吐可部分稀释噪声）。
② 可执行方案或伪代码：
```python
import torch
import torch.nn as nn
import torch.optim as optim

class BitLinear(nn.Module):
 def __init__(self, in_features, out_features):
 super().__init__()
 self.weight = nn.Parameter(torch.empty(out_features, in_features).normal_(0, 0.1).clamp_(-1, 1))
 self.alpha = nn.Parameter(torch.tensor(1.0))
 def forward(self, x):
 scale = self.alpha * torch.max(torch.abs(self.weight)).clamp(min=1e-5)
 ternary_w = (self.weight / scale).round().clamp(-1, 1) * scale # STE 前向量化
 return torch.matmul(x, ternary_w.t()) # CPU matmul fallback

# 纯 CPU 三值 QAT 伪代码
model = BitLinear(512, 512) # 示例 135M 规模
optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)
scaler = torch.amp.GradScaler() if hasattr(torch, 'amp') else None

for epoch in range(num_epochs):
 for batch in dataloader: # 纯净语料子集（<0.1B tokens）
 optimizer.zero_grad()
 with torch.cuda.amp.autocast(enabled=False): # CPU 无需
 output = model(batch)
 loss = loss_fn(output, batch_labels)
 if scaler:
 loss = scaler.scale(loss)
 loss.backward()
 if scaler:
 scaler.unscale_(optimizer)
 torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5) # 控噪声
 if scaler:
 scaler.step(optimizer)
 scaler.update()
 else:
 optimizer.step()
 # STE 伴随更新
 model.alpha.data = torch.max(torch.abs(model.weight.detach())).item()
 # 深度优先：先保函数从 fp32 基座 Net2Net 跳变到三值
```
③ 可证伪判据：什么结果算你错：在 node1 64 核 CPU 上跑 135M 三值训练，如果损失曲线在初始下降后第 2000–5000 步 plateau（不再下降 >0.1 loss）或上升，或出现 NaN/inf/梯度 norm>10，或每步时间 > fp32 基线的 3 倍且 RSS>32 GB 触发 swap，则判定“训不动”。若曲线平滑下降、收敛到 < fp32 基线 0.3 loss、variance <5%、无 divergence，则判定“能训”。
④ 若需实验标注 node1 CPU 最小可跑版：Python 3.10 + PyTorch 2.1 CPU backend（单进程，OMP_NUM_THREADS=64，torch.device('cpu')），纯净语料子集（<50M tokens，深度优先单遍），batch_size=4 seq_len=256，运行 50 epochs。成功指标：loss 下降 >20% from init，per-step time <0.1 s，memory RSS<20 GB，无 plateau 或 divergence。