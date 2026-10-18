# kris.nemotron (vendor=nvidia, served=nvidia/nemotron-3-ultra-550b-a55b, 71.0s)

### ① 校准误差难压根因 & 探针假设裁决 / 更优替代设计

**核心机制（为什么压不下去）：**
≤1B 模型的 **参数量不足以支撑「预测分布」与「校准分布」的解耦**。
- **Logit 坍缩**：小模型最后一层投影矩阵 $W_{out} \in \mathbb{R}^{V \times d}$ 秩不足，$d \ll V$。正确 token 与错误 token 的 logit 差值被压缩进同一低维锥体，**Softmax 温度 $T$ 失效**（缩放只能整体平移锥体，无法展开角度）。
- **熵-NLL 耦合退化**：训练目标只最小化 $H(p_{data}, p_{model})$，未显式约束 $H(p_{model})$。小模型倾向「高频 token 低熵高自信、低频 token 高熵低自信」，**熵成为词频的代理变量，而非不确定性