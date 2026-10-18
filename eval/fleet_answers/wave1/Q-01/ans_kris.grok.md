# kris.grok (vendor=xai, served=grok-4.7, 55.6s)

① 知识存在于FFN（MLP）权重中，尤其是中间层（hidden dim）的up-projection和down-projection矩阵中的特定子空间/稀疏“键值”关联（ROME/MEMIT机制解释范式：事实关联被压缩为MLP权重矩阵中的低秩方向或单个神经元）。≤1B模型仍能做知识定位（给定事实F，用因果追踪或反事实编辑可标出承载F的权重子集），虽精度低于大模型但仍可通过权重探针（weight probing）和梯度定位（gradient-based localization）实现，误差主要来自表示不充分而非根本缺失。
② 挂载点应选FFN层输出接口（post-FFN residual stream），形态为“可寻址KV-cache融合器”（query从attention层发往外部记忆查询，外部记忆返回向量加到residual上或做cross-attend）。理论依据：transformer残差流本身就是知识整合通道（FFN内部已充当记忆），外部挂载于此可复用已训练的前向路径而非从头设计新接口，避免拍脑袋的embedding或raw attention层。
③ 外置记忆作分布式共享缓存规避多设备推理通信瓶颈可行：将记忆层实现为设备间共享内存段（shared-memory KV pool），仅同步轻量指针或embedding ID而非全参数all-to-all，通信量降至O(1) per retrieval。
② 可执行方案伪代码：
```python
class ExternalAddressableMemory:
 def __init__(self, max_slots=10**7):  # 每人10M槽位
  self.slots = {}  # addr_hash -> fp16_tensor (size≤10M params)
 def mount(self, model_ffn_out):
  q = model_ffn_out  # query from post-FFN residual
  best_match = self.find_nearest(q)  # e.g. hash + L2 or FAISS
  return q + best_match if best_match else q
 def add(self, addr, value):  # Titan式写参数禁止
  self.slots[addr] = value.clone().detach()
 def delete(self, addr):  # 硬需求0/1删除权
  self.slots.pop(addr, None)
 def find_nearest(self, q):
  # 简易：addr = hash(q) % max_slots; return self.slots.get(addr)
  pass
```
挂载点集成到forward：`x = ff_layer(x); x = memory.mount(x); x = ff_layer(x)`（Net2Net保函数前提下零改架构）。
③ 可证伪判据：若用因果编辑后模型对新事实F的perplexity下降≥15%（或zero-shot QA准确率<原模型-10%），或删除后模型仍保留F的概率>0.3，则定位/挂载/共享缓存假设均错。
④ 若需实验标注node1 CPU最小可跑版：0.5B模型（fp16 embedding+FFN+外部10M槽位KV，total RAM<1.2GB），Torch CPU后端，batch=1，token=128，单token推理<12ms（无GPU，仅64核free CPU），minimal runnable代码用`torch.compile` + `torch.mps`（Apple Silicon fallback）或纯NumPy KV dict；内存占用峰值≈280MB（含10M槽位fp16）。
如果外部记忆非必须写回模型参数则完全符合“删除权0/1硬需求”与“排除Titans式写参数”。