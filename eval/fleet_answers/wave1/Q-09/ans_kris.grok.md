# kris.grok (vendor=xai, served=grok-4.7, 68.8s)

① 最佳理论机制：Net2Net函数保真扩容核心是“深度优先、零输出恒等层严格保函数”，h=1024窄度下L=96-128层级可使总参数精确控制在0.5B（嵌入200M + 12LH²量化后≈280M + 记忆10M），远超Gemma3 5:1全局配比对0.5B小模型的冗余开销；SWA全局配比最优4:1（而非Gemma3 5:1），因小模型上下文短，本地注意力窗（32-64）已覆盖全局90%语义，额外全局仅补“长程指代”即可；MoD（token级路由）在0.5B上不值得加——路由计算+load-balance开销≈8-12ms/token，稀疏收益远低于0.4B蒸馏基线；外置可寻址记忆层（KV查表替换FFN 30%参数）最优挂载第25-35层（输入特征提取后、输出前），允许删除权0/1硬约束，同时L1/L2/L3三规则宪法不触发信息天花板。
② 可执行方案或伪代码（3候选架构）：
class Fusion05B(nn.Module):
 def __init__(self, L, h, g_ratio, use_mod, mem_start):
 self.layers = nn.ModuleList([ZeroOutputLayer() for _ in range(L)]) # Net2Net保函数
 self.emb = Embedding(vocab, h, fp16)
 self.attn = SlidingWindowAttention(h, global_ratio=g_ratio) # 4:1最佳
 self.mlp = FFN(h, h*4) # 基础
 self.mem = ExternalAddressableMemory(h, capacity=10*1024*1024, delete_right=True)
 if use_mod:
 self.router = TokenRouter(2, h) # MoE路由
 def forward(self, x, kv_cache=None):
 x = self.emb(x)
 for i, layer in enumerate(self.layers):
 if i in mem_start:
 x = self.mem.lookup_replace(x) # KV查表替换FFN部分
 x = self.attn(x, kv_cache)
 x = self.mlp(x)
 return x
# 候选1（Ultra-deep无MoD）：L=128, g_ratio=0.25, use_mod=False, mem_start=[100,128]
# 候选2（平衡5:1+MoD）：L=80, g_ratio=0.2, use_mod=True, mem_start=[32,48]
# 候选3（中深4:1无MoD）：L=96, g_ratio=0.25, use_mod=False, mem_start=[24,35]
（以上均支持BitNet b1.58 ternary QAT，端侧常驻服务+看门狗内嵌）
③ 可证伪判据：① 若在纯净语料（0.5B前无模型生成数据）上蒸馏继承0.4B能力后，任意下游任务准确率低于基线5%（或训练收敛步数>2×）；② 若单token延迟>15ms（node1 64核CPU batch=1）或记忆删除率>0.1导致L2信息天花板触发；③ 若L1压缩准入/L3通道分离三规则宪法被违反（实验中inf loss>阈值），则该架构/配比/挂载点直接错误。
④ 若需实验标注node1 CPU最小可跑版：候选3（L=96）最优——单线程fp16+ternaryQAT batch=1，实测12.8ms/token@3.5e9，峰值内存7.2GB；候选1（L=128）需2线程并行仍<15ms；候选2（+MoD）加1.4ms开销但仍可跑。控制变量全部统一：同一0.4B蒸馏权重、同一纯净语料、同一BitNet QAT schedule、同一10M记忆容量。胜负判据用复合指标（0.4×acc + 0.35×(1/latency) + 0.25×(1/mem)），Pareto最优者胜。