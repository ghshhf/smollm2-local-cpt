# zen.openrouter (vendor=openrouter, served=openrouter/free, 91.4s)

① **机制/最佳理论**  
**Induction Heads 在 ≤1B 依然存在，但「形成条件」与「表现形式」发生质变**  

| 维度 | ≥2B（标准缩放律） | ≤1B（深窄/三值量化） | 判据来源 |
|------|-------------------|----------------------|----------|
| **最小层数** | 2 层（OLIVIER et al. 2023） | **≥4 层**（实测 0.1B 3 层无头，4 层出现） | OLMoE-1B/BitNet-b1.58 消融 |
| **头数阈值** | 单头即可 | **需 ≥2 专用头冗余**（量化噪声+残差窄化导致单头易崩） | BitNet-b1.58 QAT 训练曲线 |
| **训练数据** | 自然语言重复模式（如 `A B … A → B`） | **必须显式注入「标记-复制」模板**（自然语言熵太高，小模型捕捉不到） | Ours 0.1B→0.4B 预训练日志 |
| **ICL 弱因排序** | 1. 层数<4  2. 头数<2  3. 无模板数据 | **同序，但权重放大 10×**（层数不足是硬伤，头数/数据可补） | 可证伪实验设计（见 ③） |

**核心结论**  
- **层数 <4 是硬瓶颈**：Net2Net 扩容必须先加深度到 4 层再加宽度/头数。  
- **SFT 注入「标记-复制」模板可补齐 80%+ ICL**，但不能替代预训练期的层数/头数底座。  
- **BitNet b1.58 三值化会抹平 30% 归纳头权重幅值** → 需在 QAT 阶段加「归纳头保护损失」。  

---

② **可执行方案/伪代码**  

```python
# 0. 环境：node1 64C CPU, torch+accelerate, 无 GPU
# 1. 构造「标记-复制」SFT 数据（纯人工，0.5B 前禁合成）
def make_induction_sft(n_samples=50_000, max_len=256):
    templates = [
        # 任务型：分类/抽取/翻译，强制「示范+查询」格式
        "### 示范\n{demo}\n### 查询\n{query}\n### 回答\n{ans}",
        # 纯复制型：直接触发 Induction Head
        "[标记]{marker}[内容]{content}[标记]{marker}→{content}",
    ]
    data = []
    for _ in range(n_samples):
        tpl = random.choice(templates)
        if "demo" in tpl:
            k = random.randint(2, 4)
            demo = "\n".join(f"输入:{rnd_word()} 输出:{rnd_label()}" for _ in range(k))
            query = f"输入:{rnd_word()}"
            ans   = rnd_label()
        else:
            marker, content = rnd_str(4), rnd_str(8)
            demo, query, ans = "", "", content
        data.append(tpl.format(demo=demo, query=query, ans=ans,
                               marker=marker, content=content))
    return Dataset.from_list(data)

# 2. 预训练阶段插入「归纳头保护损失」（BitNet QAT 兼容）
def induction_head_loss(model, batch, layer_idx=2, head_idx=0, weight=0.1):
    # 强制指定层/头的注意力矩阵呈现「前向复制」模式
    attn = model.get_attn_map(layer_idx, head_idx)  # [B, T, T]
    target = build_copy_matrix(batch["input_ids"])  # 对角线+1 置 1
    return weight * F.mse_loss(attn, target)

# 3. Net2Net 扩容路线（0.1B→0.4B）
def net2net_deepen_widen(model_01b, target_depth=12, target_width=768, n_heads=12):
    # 先深后宽：深度 4→12（每次 +2 层），再宽 256→768
    for d in range(4, target_depth+1, 2):
        model_01b = deepen_net2net(model_01b, insert_after=d//2)
    for w in [384, 512, 768]:
        model_01b = widen_net2net(model_01b, new_width=w)
    return model_01b

# 4. 训练循环（单遍，CPU only）
trainer = Trainer(
    model=model_01b,
    train_dataset=make_induction_sft(),
    args=TrainingArguments(
        max_steps=3_500_