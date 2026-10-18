# kris.nemotron (vendor=nvidia, served=nvidia/nemotron-3-ultra-550b-a55b, 53.8s)

**① 机制/最佳理论裁决**  
- **0.2B（hidden=576）不值得把 embedding 三值化**——哪怕接受低频 token 精度损失。  
  - **量化收益不对等**：embedding 查找表是**稀疏读、密集写**；三值化只省 2/3 显存带宽，但推理时仍需 `gather → dequant → fp16 add`，CPU 上分支错配与去量化开销抵消带宽收益。  
  - **低频 token 梯度放大效应**：Zipf 分布下低频行更新极少，三值量化噪声会被高频行梯度“平均掉”，导致低频行表示坍缩为同一簇点，破坏尾部知识——这在“禁模型生成数据、人写纯净语料”前提下不可逆。  
  - **翻转点规律**：embedding 占比随规模单调**下降**；0.2B 是峰值（17.4%），0.4B 降至 ~10%，0.5B 仅 ~7%。**死守 fp16 仅多 28 MB**，在 64 核 CPU 上 L3 命中率已 >95%，不构成 15 ms/token 瓶颈。  

- **部分量化（高频 fp16 + 低频三值）无成熟方案**——LoRA-style `base_fp16 + delta_ternary` **不解决推理开销**：  
  - 推理仍需两次查表 + 一次 fp16 加法，指令数反增；  
  - 训练期 `delta` 梯度仍会通过 `base` 反传，除非手写 `stop_gradient`，否则低频噪声依然污染高频行；  
  - 词表动态分桶（高频/低频）需离线统计，联网自学习场景下词频漂移会导致分桶失效。  

**结论**：**0.2B embedding 全量 fp16 固化**，导出端侧时整体 **int4 非对称量化**（per-token scale + zero-point）即可；省下的工程预算投给**主干三值化 QAT**与**记忆层索引结构**。

---

**② 可执行方案（伪代码级）**  

```python
# 1. 训练期：embedding 纯 fp16，参与 QAT 但不量化
class BitNetEmbedding(nn.Module):
    def __init__(self