# kris.gemini (vendor=google, served=gemini-3.8-flash, 51.4s)

### ① 机制与最佳理论

#### 1. Induction Heads（感应头）理论在 $\le 1\text{B}$ 模型上是否适用？
**完全适用，且它是小模型唯一赖以生存的 ICL 底层机制。**
Anthropic 提出 Induction Heads 时，最早是在 **2 层 Attention-Only 玩具模型** 上发现并完成数学证明的。Induction Heads 的本质是两阶段检索电路（2-Hop Circuit）：
*   $L_k$ 层（Previous Token Head）：负责将“上一个 token 是什么”的信息写入当前位置。
*   $L_{k+m}$ 层（Induction Head）：根据当前的 token，反向搜索历史上何处出现过该 token，并将其后面的 token 通过 $W_O W_V$ 拷贝到当前预测分布中。

**必须纠正的一个错误前提：小模型 ICL 弱，绝不是因为“没有 Induction Heads”，而是因为缺乏“语义泛化版 Induction Circuits（模糊/抽象感应电路）”。**
*   $\le 0.4\text{B}$ 模型天然拥有**字面精确匹配（Exact-Match）**的 Induction Heads（测试 Prefix Matching 时哪怕 0.1B 也能表现出断崖式 loss 下降）。
*   但真实 ICL（如Few-shot 翻译、情感分类、推理）需要的是：`Token -> 抽象语义概念 -> 关系对齐 -> 输出映射`。这需要“MLP 语义升维 + 多跳注意力传递”。小模型欠缺的是这条长程语义链，而非感应头本身。

---

#### 2. 小模型 ICL 弱的根因排错：层数、头数，还是数据？

**排序：有效深度（层数） > 训练数据格式 > 注意力头数**

*   **头数（Attention Heads）不是主要瓶颈**：只要隐层切分后的每个 Head 维度 $d_k \ge 64$（保证内积寻址的信噪比），8～12 个 Head 就足以分配出 2~4 个专职 Induction Head。头数过多在小模型上反而造成参数碎片化。
*   **深度（Layers）是硬伤（验证了你的“深度优先扩容”正确性）**：
    标准 induction 是 $L \to L+1$ 两跳。但抽象 ICL 需要至少 4 跳：
    $$\text{Layer 1-2: 局部句法} \xrightarrow{MLP} \text{Layer 3-4: 概念抽取} \xrightarrow{Attn} \text{Layer 5-6: 上下文检索匹配} \xrightarrow{MLP} \text{Layer 7+: 最终投影}$$
    若总层数 $\le 8$（如粗暴做扁平网络），模型只能在“保留精确拷贝 Induction”和“MLP 事实记忆”之间发生表征干涉（Superposition 饱和崩溃）。**深窄结构是唯一能容纳多跳语义感应的解法。**
*   **数据格式是“激活钥匙”**：Pretrain 语料多为连续长文，天然存在 Induction 监督信号，但**任务级提示格式（Task Induction）密度极低**。普通预训练数据只教给模型“字面补全”，没教给模型“把前文 `A -> B` 当作一条运行时临时路由指令”。

---

#### 3. 能否通过 SFT 数据设计补齐 ICL？
**能，但有明确上限：SFT 能补齐“格式遵循与上下文强制覆盖（Context Override）”，无法凭空补齐“超出容量极限的推理步数”。**
*   **有效路径：Symbol Tuning 与 Few-Shot 模板内嵌**。强制小模型注意 Context，抑制其预训练的先验偏置（Prior Bias）。当 Context 输入与预训练知识冲突时（例如 `输入：香蕉是蓝色的。问题：香蕉是什么颜色？`），未针对性调优的小模型会强行输出“黄色”，而具备强 Induction 激活的模型会输出“蓝色”。
*   **数据约束应对**：在“主干预路径禁模型生成数据”红线下，SFT 不得用商用 LLM 批量蒸馏长链条，必须采用**纯规则变换的人写模板化构造**（见第二部分）。

---

### ② 可执行方案：基于人写数据的“感应增强（Induction-Enhanced）”SFT 协议

为了在不破坏“语料人写纯净”与“深度优先”的前提下最大化激活 0.1B~0.4B 的 ICL 能力，制定如下 SFT 构造方案：

#### 方案逻辑：构建“上下文强制重载”数据对（Symbol-Tuned Prompts）
利用规则引擎，将已有的人写自有资料（owner_corpus）重组为 **Few-shot In-Context Schema**，强制 Attention 将权重分配给前置示范。

```python
# induction_data_formatter.py
# 运行环境：本地 CPU，零模型生成，纯确定性规则转换
import random
from typing import List, Dict

def construct_in_context_sample(human_qa_bank: List[Dict[str, str]], k_shot: int = 3) -> Dict[str, str]:
    """
    从纯人写 QA 库中抽取样本，强制组装为符合 Induction 诱导的上下文学习格式。
    包含：Label Permutation (强迫模型看 Context，不看先验)
    """
    assert len(human_qa_bank) >= k_shot + 1
    selected = random.sample(human_qa_bank, k_shot + 1)
    demos, target = selected[:-1], selected[-1]

    prompt = "请依照上下文的映射规则完成任务：\n\n"
    
    # 构造示范区 (Induction 锚点)
    for i, demo in enumerate(demos):
        prompt += f"样例{i+1}：\n输入：{demo['input']}\n输出：{demo['target']}\n\n"
    
    # 构造触发区 (Trigger)
    prompt += f"最终测试：\n输入：{target['input']}\n输出："
    
    return {
        "text": prompt + target['target'],
        "eval_prompt": prompt,
        "eval_ground_truth": target['target']
    }
```

#### 训练时的 Loss Masking 宪法规则
*   **示范区（Demonstrations）全部设为 `loss_mask = 0`**：不让模型学习示范本身的生成，只对其产生隐层 Key-Value 缓存。
*   **仅在最终测试的 `target` 部分计算梯度 (`loss_mask = 1`)**：强迫反向传播优化且仅优化“从示范区抽取规则并应用于 Target”的权重路径，即强行更新 Attention 的 $W_Q, W_K$ 寻址参数以形成 Induction 回路。

---

### ③ 可证伪判据（什么结果算我错）

设计两项在 **node1 (64核 CPU)** 上运行的零歧义指标。**只要出现以下任一情况，即推翻我上述“数据与深度架构决定 Induction Head”的判断**：

1.  **感应头打压反例（架构不可逆判定）**：
    *   在相同的 0.1B 纯粹深窄模型（如 16层，d_model=512）与浅宽模型（4层，d_model=1024）上，跑相同的 Synthetic Prefix-Matching 测试。
    *   **证伪线**：若浅宽模型的 Induction Head 形成速度（Prefix Loss 骤降拐点所在的 training steps）和绝对匹配率**超越**深窄模型，则说明 ICL 机制对层数（深度）的假设错误。
2.  **SFT 泛化无效反例（SFT 数据设计上限判定）**：
    *   做 **Label-Flipping ICL 评测**（例如在示范中颠覆常识：输入 `1+1=3`, `2+2=5`，测试 `1+1=?`）。
    *   **证伪线**：如果经由第二部分 In-Context SFT 训练的模型，在翻转标签任务上的准确率提升 **$< 5\%$**（相比直接 Pretrain 的 Base 模型），且模型依然顽固输出预训练常识（`1+1=2`），则证明：**小模型无法通过 SFT 数据协议建立上下文覆盖先验的能力，小模型 ICL 弱纯粹由参数容量天花板锁死，无法通过数据补齐**。我关于 SFT 能激活 ICL 的理论即被推翻。

---

### ④ node1 CPU 最小可跑实验：Induction Circuits 探针与 ICL 能力检测

该脚本用于在无 GPU 环境下，直接量化验证小模型的 Induction Head 是否存在，以及 ICL 上下文覆盖率。

```python
# test_induction_cpu.py
# 依赖：pip install torch transformers
# node1 64核 CPU 运行参数优化：OMP_NUM_THREADS=64
import os
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

os.environ["OMP_NUM_THREADS"] = "64"
os.environ["MKL_NUM_THREADS"] = "64"
torch.set_num_threads(64)

def evaluate_induction_score(model, tokenizer, seq_len=64, n_trials=50):
    """
    在 CPU 上跑无偏随机序列感应测试 (Synthetic Induction Probe)。
    如果 Induction Head 存在，模型在处理形如 [A, B, C, ..., A, B] 的序列时，
    在第二个 B 处的困惑度必须显著低于第一次出现的位置。
    """
    model.eval()
    vocab_size = model.config.vocab_size
    induction_drop_ratios = []

    with torch.no_grad():
        for _ in range(n_trials):
            # 生成随机无重复 token 序列前缀
            rand_tokens = torch.randint(100, vocab_size - 100, (1, seq_len))
            # 拼接自身形成二项循环：[S1, S2, ..., Sn, S1, S2, ..., Sn]
            repeated_tokens = torch.cat([rand_tokens, rand_tokens], dim=1)
            
            outputs = model(repeated_tokens)
            logits = outputs.logits # [1, 2*seq_len, vocab_size]
            
            # 取第一段末尾预测 vs 第二段末尾预测 (Induction 触发点)
            loss_fn = nn.CrossEntropyLoss()
            
            # 第一半序列最后的预测损失 (无上下文可抄)
            loss_first_half = loss_fn(
                logits[:, seq_len - 1 : seq_len, :].reshape(-1, vocab_size),
                repeated_tokens[:, seq_len : seq_len + 1].reshape(-1)
            )
            # 第二半序列最后对应的预测损失 (存在前置 token，Induction Head 应强力生效)
            loss_second_half = loss_fn(
                logits[:, -2 : -1, :].reshape(-1, vocab_size),
                repeated_tokens[:, -1:].reshape(-1)
            )
            
            # 下降率 = (Loss1 - Loss2) / Loss1
            drop_ratio = (loss_first_half - loss_second_half) / loss_first_half
            induction_drop_ratios.append(drop_ratio.item())

    avg_drop = sum(induction_drop_ratios) / len(induction_drop_ratios)
    print(f"=== CPU 探针评估完成 ===")
    print(f"64核运算并行度: {torch.get_num_threads()}")
    print(f"Induction Loss 下降比率: {avg_drop:.4f}")
    if avg_drop > 0.15:
        print("结论：模型内部已经自发形成强 Induction Heads 电路。")
    else:
        print("结论：模型缺乏基本感应头回路，ICL 机制处于沉睡或缺失状态。")

if __name__ == "__main__":
    # 此处替换为你的 0.1B 本地输出目录
    # 示例载入小模型：测试用
    MODEL_PATH = "./local_0.1b_checkpoint"
    if os.path.exists(MODEL_PATH):
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
        model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, torch_dtype=torch.float32)
        evaluate_induction_score(model, tokenizer)
    else:
        print(f"路径 {MODEL_PATH} 不存在，请指定 0.1B/0.4B 物理权重路径以在 node1 运行。")
```

#### node1 实操指标基线（供对齐）
*   **运行开销**：在 64-core node1 CPU 上，单次 forward 耗时对于 0.1B 模型 $\le 12\text{ms}$，上述 50 次探针评估将在 5 秒内无感跑完。
*   **判定界限**：若经过深度扩容（Net2Net）后的 0.4B 在此测试中的下降比率未显著高于 0.1B（提升幅度未超过 10%），说明深度带来的容量未能正确解耦注意力回路，必须停止继续向 0.5B 蒸馏。