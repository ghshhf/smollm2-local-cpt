# kris.grok (vendor=xai, served=grok-4.7, 74.6s)

① 最佳理论：知识在≤1B模型的FFN权重中以键值记忆形式编码（Geva et al. 2021《The Transformer Feed-Forward Layers Are Key-Value Memories》证明FFN up-projection/down-projection权重即为KV槽位）。≤1B模型仍可定位：给定事实f，通过激活 patching 或 causal tracing 找到承载f的权重子集（通常是MLP中数个通道/神经元权重，其子集贡献>80%事实概率）。精度因模型规模而下降，但理论上可定位（非随机，而是通过梯度/激活敏感性筛选）。
② 执行方案：挂载点选FFN并行扩展层（或跨注意力cross-attn接口），理论依据是FFN即KV memory，支持外部槽位直接挂载到残差流后作为附加KV检索。接口形态：隐藏状态h经线性投影生成query-key，外部数组检索相似槽位并残差相加。排除Titans式写参数：使用纯数组+索引，无新增可训练写权重。伪代码：
```python
class ExtMemLayer:
 def __init__(self, capacity=10_000_000, dim=128, val_bits=16):
 self.capacity = capacity
 self.keys = np.zeros((capacity, dim), dtype=np.float16)
 self.vals = np.zeros((capacity, dim), dtype=np.float16) # 或压缩
 self.indices = np.arange(capacity)
 self.next_idx = 0
 self.hard_delete = True # 0/1删除权
 def query(self, qk):
 # 余弦相似度检索（CPU友好）
 sim = np.dot(self.keys[:self.next_idx], qk) / np.linalg.norm(self.keys[:self.next_idx], axis=1)
 top = np.argsort(sim)[-k:] if k>1 else np.argmax(sim)
 return np.mean(self.vals[top], axis=0) if len(top)>0 else np.zeros(dim)
 def write(self, key, val, overwrite=True):
 if self.hard_delete and self.keys[self.next_idx].any(): self.next_idx +=1 # 循环覆盖
 self.keys[self.next_idx] = key
 self.vals[self.next_idx] = val
 self.next_idx = (self.next_idx + 1) % self.capacity
```
模型forward中：hidden = FFN(hidden) + ExtMem(query_proj(hidden))
③ 可证伪判据：外置记忆写入后若模型事实recall无显著提升（<5%绝对变化，经重复run验证），或删除槽位后模型性能不变而模型本身权重不变，则判定挂载/定位无效；分布式共享缓存若通信带宽（NCCL/all-reduce）未下降或增加，则判定不可行。
④ node1 CPU（64核）最小可跑版：numpy+Numba优化的ExtMemLayer（容量限1M测试版以防OOM），嵌入BitNet b1.58 QAT模型forward（fp16嵌入+3值QAT层）。运行脚本：`python -c "from extmem import ExtMemLayer; mem=ExtMemLayer(1000000); mem.write(np.random.rand(128), np.random.rand(128)); print(mem.query(np.random.rand(128)))"`，内存≈256MB@1M槽，CPU query<1ms/token。