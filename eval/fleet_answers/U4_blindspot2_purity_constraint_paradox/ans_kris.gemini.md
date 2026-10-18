# kris.gemini (vendor=google, served=gemini-3.8-flash, 59.4s)

针对上述自相矛盾与宪法冲突，**直接推翻原争议前提**：

> **推翻前提**：“蒸馏与认主必须依赖模型生成的合成数据”是一个**伪命题**。它混淆了**数据载体（Token 序列 $X$ 的拓扑支撑集）**与**监督信号（Logits 概率分布 $P(y|X)$）**。
> 工业界（Gemma 2、Apple AFM）已完全证实：**真正避免模型崩溃（Model Collapse）的蒸馏，输入 Token 100% 来自人写纯净语料，仅监督目标使用 Teacher Soft-Logits**。
> “0.5B 前主干预训练加载路径不得含模型生成数据”的宪法约束**不需要降级**，也不存在实质冲突。

以下是裁决与执行规范。

---

### ① 机制 / 最佳理论

#### 1. 阶段性纯度范式与本质规律（Gemma 2 / Apple AFM / Nature 2024）
*   **模型崩溃的数学根源**（Shumailov et al., *Nature 2024*）：
    若自回归输入为模型生成的合成文本 $X_{\text{syn}} \sim P_\theta(X)$，数据经验测度的支集（Support）逐代收缩，长尾低频信息（如主人的特殊癖好、冷门系统底层指令）迅速衰减至零，方差坍缩不可逆。
*   **Token 纯度与软标签解耦（Gemma 2 范式）**：
    Gemma 2 在预训练阶段做知识蒸馏时，喂给 Student 的 **Token 序列完全来自真实人类 Web 语料**。Teacher 仅在 Forward 阶段提供分布向量 $P_{\text{teacher}}(y|X_{\text{human}})$。
    *   **输入端（主干路径）**：保留人类语言的完整流形 $\mathcal{M}_{\text{human}}$，杜绝语义退化。
    *   **输出端（优化目标）**：利用 Teacher 的暗知识（Dark Knowledge）压缩梯度方差，提升小模型在单位 Token 下的信息吞吐量。
*   **自进化与常驻学习的宪法映射**：
    联网自我学习与 RSI 绝不能回写主干预训练路径。根据 **L3 通道分离原则**，外部学习写入**外置 10MB 可寻址记忆层**（物理隔离、具备 0/1 硬删除权），主干权重冻结。主干的纯度与常驻学习的脏数据天然物理隔离。

---

### ② 裁决结论与可执行方案

#### 裁决 1：阶段性纯度约束成熟范式
采纳 **“载体人写，信号蒸馏，通道硬隔”** 标准：
*   **预训练主干加载路径（0.1B $\to$ 0.4B $\to$ 0.5B）**：Token 流必须 100% 为人写（Human-Authored Tokens）。
*   **后训练与退火阶段（Post-Training Annealing）**：若引入推理强化，必须受**确定性验证器（Deterministic Verifier）**约束（如编译器通过性、AST 语法树校验），禁止无判别的黑盒 LLM 自言自语文本。

#### 裁决 2：0.4B 节点语料纯度定义
*   **中文词表扩充加热**：使用纯正人写中文语料（CLUECorpus 精选清洗集、维基百科、专业开源技术文档）。严格过滤“AI 回复特征词”（如“作为一个人工智能模型”）。
*   **owner_corpus**：100% 用户本地原生笔记、代码、真实聊天记录，**严禁使用商业大模型对其做扩写/润色/改写（Paraphrasing）**，保证主人的原生词频与语病特征不被平滑。
*   **认主对比头（Identity Contrastive Head）**：
    *   正样本对：主人的真实历史文本片段。
    *   负样本对：标准公开人类对话集（如人类真实对白语料 LCCC 纯净版），**无需也严禁生成合成假冒文本**。

#### 裁决 3：0.5B 蒸馏边界界定
*   **预训练加载路径定义**：输入到 Embedding 层的 Token 序列张量 $X \in \mathbb{N}^{B \times L}$。
*   **边界划分**：
    *   **红线（算主干路径，禁合成）**：Data Loader 读取的磁盘语料文件必须为 100% 人写文本。
    *   **绿线（不算主干路径，允许模型参与）**：0.4B 或外部 Teacher 模型对人写文本 $X$ 实时推断出的 Top-K Logits 矩阵。这属于**损失函数正则项**，不属于“模型生成语料”。

#### 执行伪代码：0.5B 纯净载体 Logit 蒸馏与认主对比头加载

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class PureHumanDistillationLoss(nn.Module):
    """
    符合宪法的蒸馏损失：
    输入 tokens 必须 100% 来自人写；Teacher 仅提供 Logit 正则化，不产生合成 token 载体
    """
    def __init__(self, temperature=2.0, alpha=0.5):
        super().__init__()
        self.temperature = temperature
        self.alpha = alpha
        self.kl_div = nn.KLDivLoss(reduction="batchmean")
        self.ce = nn.CrossEntropyLoss()

    def forward(self, student_logits, teacher_logits, human_target_tokens):
        # 1. 人类纯净标注交叉熵 (保持人类流形支集)
        loss_ce = self.ce(student_logits.view(-1, student_logits.size(-1)), human_target_tokens.view(-1))
        
        # 2. Teacher 知识蒸馏 (暗知识传递)
        p_s = F.log_softmax(student_logits / self.temperature, dim=-1)
        p_t = F.softmax(teacher_logits / self.temperature, dim=-1)
        loss_kd = self.kl_div(p_s, p_t) * (self.temperature ** 2)
        
        return (1 - self.alpha) * loss_ce + self.alpha * loss_kd


class OwnerContrastiveHead(nn.Module):
    """
    0.4B 认主对比头：基于人类真实数据构建正负对
    正样本: owner_corpus 切片
    负样本: 外部公开人写语料 (非大模型合成语料)
    """
    def __init__(self, hidden_dim, proj_dim=64):
        super().__init__()
        self.proj = nn.Linear(hidden_dim, proj_dim, bias=False)

    def forward(self, last_hidden_states, is_owner_mask):
        # last_hidden_states: [B, L, D] -> 取 pooling
        rep = F.normalize(self.proj(last_hidden_states.mean(dim=1)), p=2, dim=-1)
        # 余弦相似度矩阵
        sim_matrix = torch.matmul(rep, rep.T)  # [B, B]
        
        # 监督对比损失 (InfoNCE)
        labels = is_owner_mask.float()
        mask = torch.eq(labels.unsqueeze(0), labels.unsqueeze(1)).float()
        
        # 消除自身相似度
        logits_mask = torch.scatter(
            torch.ones_like(sim_matrix), 1, torch.arange(sim_matrix.size(0)).view(-1, 1).to(sim_matrix.device), 0
        )
        mask = mask * logits_mask
        
        exp_sim = torch.exp(sim_matrix / 0.07) * logits_mask
        log_prob = sim_matrix / 0.07 - torch.log(exp_sim.sum(dim=1, keepdim=True) + 1e-8)
        loss = -(mask * log_prob).sum(dim=1) / (mask.sum(dim=1) + 1e-8)
        return loss.mean()
```

---

### ③ 可证伪判据

若在 0.4B $\to$ 0.5B 演进中出现以下任一事实，**即证明上述“人写纯净输入蒸馏”策略失败**，我方推翻当前技术路线，向用户申请修改宪法：

1.  **收敛效率倒挂**：在相同的 CPU/Token 训练预算下，采用 `100% 人写文本载体 + Teacher 软标签` 训练的 0.5B 学生模型，其下游任务 Loss 与主观指令遵循率，**显著落后于**采用 `LLM 生成的多样化合成指令对（如 Evol-Instruct 体系）` 训练的对照组超过 **15% 以上**。
2.  **暗知识表达失效**：小模型（0.5B BitNet b1.58）在只有人类语料载体下，KL 散度无法平稳下降（持续震荡大于 2.0），且 Soft-Logits 带来的梯度未能降低学生模型的输出方差。
3.  **认主边界模糊**：仅使用自然人写语料（真实人类对话）作为负样本，导致对比头无法有效防范哪怕最简单的 LLM 仿冒 prompt 攻击（错误拒绝率 FRR 或错误接受率 FAR > 5%）。

---

### ④ node1 CPU (64核) 最小可验证实验代码

此脚本在魔搭 node1 运行，验证：
1. 64 核并行下，在**纯人写 Token 流上计算 Teacher 软标签蒸馏**的 CPU 开销与吞吐；
2. 验证 Embedding fp16 保持与紧密内存排布；
3. 验证无模型合成数据参与下的对比头 Forward-Backward 稳定性。

```python
# test_cpu_distill_pure.py
# 运行方式: OMP_NUM_THREADS=64 python3 test_cpu_distill_pure.py
import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.set_num_threads(64)

# 1. 模拟微型网络 (0.1B 级别架构测算)
class DummyTransformer(nn.Module):
    def __init__(self, vocab_size=32000, hidden_dim=512, num_layers=4):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_dim).half() # 宪法拍板: fp16
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=hidden_dim, nhead=8, dim_feedforward=2048, 
                activation='gelu', batch_first=True, dtype=torch.float32
            ) for _ in range(num_layers)
        ])
        self.lm_head = nn.Linear(hidden_dim, vocab_size, bias=False)

    def forward(self, input_ids):
        x = self.embedding(input_ids).float()
        for layer in self.layers:
            x = layer(x)
        logits = self.lm_head(x)
        return logits, x

def run_experiment():
    batch_size = 8
    seq_len = 512
    vocab_size = 32000
    steps = 10
    
    print("=== 魔搭 node1 (64核) 纯净语料载体蒸馏基准测试 ===")
    
    # 模拟纯净人类数据载体 (无 LLM 生成文本注入)
    fake_human_tokens = torch.randint(0, vocab_size, (batch_size, seq_len), dtype=torch.long)
    is_owner_mask = torch.tensor([1, 1, 0, 0, 0, 0, 1, 0], dtype=torch.bool)
    
    student = DummyTransformer(vocab_size=vocab_size, hidden_dim=384, num_layers=4)
    teacher = DummyTransformer(vocab_size=vocab_size, hidden_dim=512, num_layers=6)
    teacher.eval() # Teacher 冻结
    
    proj_head = nn.Linear(384, 64)
    optimizer = torch.optim.AdamW(student.parameters(), lr=1e-3)
    
    t0 = time.time()
    for step in range(steps):
        optimizer.zero_grad()
        
        # 前向 1: Teacher 获取软标签 (纯人写载体)
        with torch.no_grad():
            t_logits, _ = teacher(fake_human_tokens)
            
        # 前向 2: Student 输出
        s_logits, s_rep = student(fake_human_tokens)
        
        # 交叉熵 + 蒸馏损失
        loss_ce = F.cross_entropy(s_logits.view(-1, vocab_size), fake_human_tokens.view(-1))
        loss_kd = F.kl_div(
            F.log_softmax(s_logits / 2.0, dim=-1),
            F.softmax(t_logits / 2.0, dim=-1),
            reduction='batchmean'
        ) * 4.0
        
        # 认主对比头损失
        owner_rep = F.normalize(proj_head(s_rep.mean(dim=1)), dim=-1)
        sim = torch.mm(owner_rep, owner_rep.T) / 0.07
        loss_id = F.cross_entropy(sim, is_owner_mask.long())
        
        total_loss = loss_ce + loss_kd + 0.1 * loss_id
        total_loss.backward()
        optimizer.step()
        
        print(f"Step {step+1}/{steps} | Loss: {total_loss.item():.4f} (CE: {loss_ce.item():.3f}, KD: {loss_kd.item():.3f})")
        
    duration = time.time() - t0
    tokens_processed = batch_size * seq_len * steps
    print(f"完成! 总耗时: {duration:.2f}s | 吞吐: {tokens_processed / duration:.1f} tokens/s (64-core CPU)")
    print(f"单 Token 延迟: {(duration / tokens_processed) * 1000:.3f} ms/token (多样本并行下)")

if __name__ == "__main__":
    run_experiment()
```

### 最终定调
不擅自降级宪法约束。**0.4B 之前维持 100% 人写纯净语料载体**。0.5B 的蒸馏仅读取外部 Teacher 的概率矩阵作为辅助 Loss，严禁直接加载任何模型自生成的文本序列作为主干训练输入。自进化与外部联网信息严格封闭在 L3 记忆通道内，实现逻辑与架构自洽。