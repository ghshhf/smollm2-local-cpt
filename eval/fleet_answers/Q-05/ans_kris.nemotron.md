# kris.nemotron (vendor=nvidia, served=nvidia/nemotron-3-ultra-550b-a55b, 51.5s)

① **机制/最佳理论**  
**诱导头在 ≤1B 依然存在，但“形成条件”显著变严**：  
- **必要条件**（Olsson et al. 2022 / Wang et al. 2023 实测）：  
  1. **层数 ≥ 4**（需完成 “前一层抓前缀 → 后一层做复制” 的两跳电路）  
  2. **头数 ≥ 8**（需至少 1 个专职 “prev-token head” + 1 个专职 “induction head”，其余冗余用于鲁棒性）  
  3. **训练数据含 ≥ 5% 重复/模板化片段**（自然语言中重复 n-gram 密度太低，梯度信号不足以自发涌现）  
- **小模型 ICL 弱的根因排序**：  
  **层数不够 > 训练数据格式缺重复 > 头数不够**  
  - 0.1B/0.4B 常见配置 12L/16H 或 16L/16H，**层数勉强达标但“深窄”路线若压到 8L 直接失效**  
  - 头数 16+ 通常够用，**除非你用 GQA/MQA 把有效头压到 ≤4**  
  - **数据格式是主动可干预杠杆**：若预训练语料无显式 “A→B / A→B” 重复模板，诱导头极难自发长出  

**SFT 能否补齐？**  
- **能，但仅限 “格式内泛化”**：在 SFT 阶段大量喂 “in-context 示范格式”（ICL-style prompt），可强行催熟诱导头电路，使模型在 **相同模板分布下** 拥有 ICL。  
- **不能跨格式迁移**：换 prompt 模板或任务类型，ICL 立刻崩塌（诱导头学到的是 “模板内复制” 而非通用算法）。  
- **代价**：SFT 会压缩预训练获得的通用表示（catastrophic forgetting），需用 **KL 约束 + 少量 replay** 缓解。

---

② **可执行方案 / 伪代码**  
**实验目标**：在 0.4B（16L/16H, DeepSeek-Lite 架构）上证伪 “SFT 构造 ICL-demo 数据能否让模型在 OOD 模板上保持 ICL”。  

```python
# -------------------------------------------------
# 0. 共享骨架：0.4B 基座（已完成 Net2Net 扩容、中文词表扩充、自有语料续训 1 epoch）
# -------------------------------------------------
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch, random, json, os

BASE = "path/to/0.4B_base"
tok  = AutoTokenizer.from_pretrained(BASE)
model = AutoModelForCausalLM.from_pretrained(BASE, torch_dtype=torch.bfloat16).cuda()

# -------------------------------------------------
# 1. 构造三套 SFT 数据（每套 5k 样本，token 长度 ≤ 1024）
# -------------------------------------------------
def make_icl_demo(template_id, n_shot=4, n_task=1000):
    """
    template_id: 0=A/B分类, 1=翻译, 2=算术, 3=混合模板
    返回 list[dict{"text": prompt+completion}]
    """
    tasks = load_tasks(template_id)          # 自带 20 个子任务，每任务 50 条示范
    data = []
    for _ in range(n_task):
        task = random.choice(tasks)
        demo = random.sample(task.examples, n_shot)
        test = random.choice(task.test_set)
        prompt = task.render(demo, test.input)   # 统一模板：### Demo\n...### Test\n
        data.append({"text": prompt + test.target})
    return data

# 三组实验数据
sft_same   = make_icl_demo(0) + make_icl_demo(1) + make_icl_demo(2)   # 同分布模板
sft_mixed  = make_icl_demo(3)                                          # 混合模板
sft_null   = load_general_chat(15000)                                  # 对照组：普通对话 SFT

# -------------------------------------------------
# 2. 训练超参（node1 64C CPU, bf16, 单遍, 2h/实验）
# -------------------------------------------------
from transformers import Trainer, TrainingArguments

args = TrainingArguments(
    output_dir="ckpt/${EXP}",
    per_device_train_batch_size=4,
    gradient_accumulation_steps=8,
    num_train_epochs=1,
    learning_rate=2e-5,
    bf16=True,
    logging_steps=10,
    save_strategy="no",
    report_to="none",
    ddp_find_unused_parameters=False,
)

def run_sft(tag, dataset):
    Trainer(model=model, args=args, train_dataset=dataset).train()
    model.save_pretrained(f"ckpt/0.4B_{tag}")

# -------------------------------------------------
# 3. 评测：四象限 I