# pollinations (vendor=openai-oss, served=gpt-oss-20b, 21.2s)

**① 机制/最佳理论**

| 议题 | 现有成熟范式 | 适配案例 | 前置/后置阈值 |
|------|---------------|----------|---------------|
| **阶段性纯度约束** | *RLAIF (RL‑with‑Instruction‑Following)*, *OpenAI’s “RLHF‑step‑segregation”*, *Gemma‑3 “Distillation Curriculum”* | 这些方案把人类标注数据与模型自生成、合成数据彻底分层：<br>1) **训练**：只使用人类写作、专业领域、领域列表（Clean‑Corpus）<br>2) **提示**：在 RLHF 预后阶段加入模型产生的对话记录，进行“人类‑AI 交互”校准 | Baseline 阶段（0.1B‑0.4B） → 纯数据；Distillation 阶段（0.5B） → 允许合成数据，但**不**计入 “先前训练加载路径” |
| **中文词表扩容** | *GloVe‑Chinese*, *BERT‑ZH vocabulary*, *FastText‑zh* | 词表不变，使用 **新增词典映射** 代替完整重训练，保证不产生新的“生成”数据 | 仅词表更新，无标注／生成数据 |
| **认主风险头** | *“Master‑ID Head”*（自监督身份标记） | 在 0.4B 里插入一个 **owner‑identification head** 用于识别输入侧的 “owner” 侧标签，帮助后续蒸馏识别 | 只做身份识别，不产生内容 |

**② 可执行方案 / 伪代码**

```python
# 0.1B → 0.4B 纯训练阶段
clean_corpus = load_pool("human_written_texts")
train_config = {
    "model_size": 0.4e9,
    "steps": 1_000_000,
    "optimizer": "AdamW",
    "fp16": True,
}
model_04B = train(clean_corpus, config=train_config)

# 0.4B 后加入中文词表、owner_corpus、master‑head
# 词表扩容
new_vocab = load_vocab("zh_extended.txt")
model_04B.resize_token_embeddings(len(new_vocab))

# 认主头
model_04B.add_module("master_head", MasterHead(vocab_size=len(new_vocab)))
model_04B.train(max_steps=200_000)

# 0.5B 蒸馏阶段 – teacher 预训练
teacher = load_pretrained("large_teacher_extern")  # 外部公网教模型

# 蒸馏数据
distill_dataset = concat(
   load_pool("owner_corpus.txt"),         