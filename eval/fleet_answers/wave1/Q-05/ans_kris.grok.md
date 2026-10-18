# kris.grok (vendor=xai, served=grok-4.7, 54.0s)

① 机制/最佳理论：Induction heads理论（Olsson et al. 2022，Transformer Circuits）本质上是Transformer架构的统计模式匹配机制——特定头在多头注意力中实现“前向复制”（copying previous token to current position），这是ICL的核心计算原语。该理论在≤1B模型上仍完全适用（GPT-2 124M即为≤1B示例），只要模型为纯Transformer + 训练数据中存在足够高的induction统计模式即可。对于0.1B-0.4B量级小模型，仍可通过Net2Net保函数扩容保留这些头（深度优先扩容优先保证至少6-12层+12头以上以支撑多层传播）。小模型ICL弱的根本瓶颈是**训练数据格式问题**（而非单纯层数/头数不够）：自然语言数据中induction头需高频“Previous→Next”模式才能稳定 emergence，层数不足会削弱多层叠加传播，头数不够可能缺失专用头，但数据格式是主导——因为ICL是emergent statistical property而非架构硬限制。
② 可执行方案或伪代码：
在0.1B→0.4B Net2Net扩容后，使用SFT数据设计强化induction示范格式。生成中文专用induction corpus：对owner_corpus进行模板化切分，构造“Previous token copying”示范（e.g. 列表、故事、对话延续）。伪代码如下：
```
def build_induction_sft_data(owner_corpus, template="Context: {context}\nPrevious: {prev}\nNext: {next}"):
 data = []
 for text in owner_corpus: # 纯净语料，无模型生成
 for i in range(len(text)-1):
 context = ' '.join(text[:i])
 prev = text[i]
 next_ = text[i+1]
 example = template.format(context=context, prev=prev, next=next_)
 data.append(example)
 return data # 输出: 100k+样本，中文词表仅加不换
```
训练时：0.1B→0.4B Net2Net后接SFT（fp16 embedding），batch包含50%纯induction示范 + 50%混合owner_corpus，loss = cross_entropy + induction_copy_aux_loss（可选：probe前token注意力分数>0.6即加权）。
③ 可证伪判据：
若SFT后模型ICL准确率（e.g. 5-shot Chinese next-token continuation或few-shot classification）未提升（<55%），或通过attention probe无法检测induction heads（前向注意力分数平均<0.4，或头数/位置不变），则数据格式非ICL弱主因或理论不适用；反之，若ICL提升且probe分数显著上升，则数据格式是可修复的根本问题。
④ 实验标注node1 CPU最小可跑版：
node1（64核CPU）跑PyTorch CPU版0.1B基座（layers=6, d_model=256, n_heads=12, vocab=30k中文+英文，embedding fp16 ~5.5GB），从0.1B Net2Net到0.4B，SFT仅1 epoch on 100k tokens（自制induction data），测试ICL：next-word prediction准确率（Chinese lists/stories）。最小配置：node1 4核/16GB RAM即可（单卡CPU训练~2-4小时/epoch）；若需更小，先跑0.05B skeleton验证。
直接推翻前提：若未来实验显示≤1B仍需>12层或特定头数不可Net2Net保函数，则induction heads理论需微调（但当前证据不支持此推翻）。