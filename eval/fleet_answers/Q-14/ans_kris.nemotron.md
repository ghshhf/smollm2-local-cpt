# kris.nemotron (vendor=nvidia, served=nvidia/nemotron-3-ultra-550b-a55b, 56.2s)

### ① 0.4B 嵌入空间扁平度与 576 维是否够用

**机制/最佳理论**  
- **Johnson-Lindenstrauss (JL) 引理 + 流形假设**：若 owner/user 总数 $N_{id} \le 10^3$，576 维足以将 pairwise 距离保真到 $\epsilon < 0.1$；但**类内方差**（同一人不同光照/角度/噪声）才是瓶颈。  
- **有效秩** $r_{eff} = \frac{(\sum \sigma_i)^2}{\sum \sigma_i^2}$：在 0.4B 最后一层 hidden state 上实测，若 $r_{eff} < 120$，说明空间已退化到“扁平胶囊”，Triplet 无法拉开 margin。  
- **结论**：576 维**理论够、工程险**——必须显式正则化秩与各向异性。

**可执行方案（伪代码级）**  
```python
# 1. 训练前探针：算有效秩
def effective_rank(hidden_states):  # [B, T, 576]
    U, S, _ = torch.linalg.svd(hidden_states.flatten(0,1).float(), full_matrices=False)
    return (S.sum()**2 / (S**2).sum()).item()

# 2. 三重损失 + 秩/各向异性正则
class TripletRankLoss(nn.Module):
    def __init__(self, margin=0.3, lambda_rank=1e-3, lambda_iso=1e-3):
        super().__init__()
        self.margin = margin
        self.lambda_rank = lambda_rank
        self.lambda_iso = lambda_iso

    def forward(self, anchor, pos, neg):  # [B, 576]
        # 标准 triplet
        d_ap = (anchor - pos).pow(2).sum(-1)
        d_an = (anchor - neg).pow(2).sum(-1)
        loss_trip = F.relu(d_ap - d_an + self.margin).mean()

        # 秩正则：推大有效秩
        X = torch.cat([anchor, pos, neg], 0)          # [3B, 576]
        S = torch.linalg.svdvals(X.float())
        r_eff = (S.sum()**2 / (S**2).sum())
        loss_rank = -torch.log(r_eff / 576)           # 越接近 576 越小

        # 各向异性正则：压扁条件数
        cond = S[0] / (S[-1] + 1e-6)
        loss_iso = torch.log(cond)

        return loss_trip + self.lambda_rank * loss_rank + self.lambda_iso * loss_iso

# 3. 推理时：L2 归一化 + 温度缩放
def get_embedding(model, input_ids):
    h = model(input_ids).last_hidden_state[:, 0]      # [B, 576]
    h = F.normalize(h, dim=-1) * 10.0                 # temperature=10
    return h
```

**可证伪判据**  
| 指标 | 通过阈值 | 失败即判定“空间太扁平” |
|------|----------|------------------------|
| 有效秩 $r_{eff}$ (eval set) | $\ge 180$ | $< 120$ |
| EER (Equal Error Rate) @ 1:1 验证 | $\le 2.5\%$ | $> 5\%$ |
| FAR @ FRR=1% (1:1000 检索) | $\le 0.5\%$ | $> 2\%$ |

**node1 CPU 最小可跑实验**  
- 数据：自采 50 人 × 20 条语音/文本（共 1k 样本）  
- 训练：batch=64, 2 epoch, AdamW 2e-4, **单张 64 核 CPU 约 12 分钟**  
- 评估：同上指标，脚本 `eval_triplet.py` 单进程跑完 < 30 秒。

---

### ② 0.4B → 0.5B（BitNet b1.58，hidden 变化）认主头无损迁移

**机制/最佳理论**  
- **Net2Net 函数保持**要求：$f_{0.5B}(x) \equiv f_{0.4B}(x)$ 对认主头输入分布。  
- BitNet b1.58 **权重三值化** $\{-1,0,1\}$ 且 **无 bias**，输出分布会变窄、量化噪声大；直接复用 0.4B 头会导致分布偏移。  
- **解法**：冻结 0.4B 头，仅在其前插 **可学习的 LayerNorm + 低秩 Adapter (LoRA-r=8)**，用 **KD + 对