# kris.nemotron (vendor=nvidia, served=nvidia/nemotron-3-ultra-550b-a55b, 74.3s)

### ① 状态机设计与自欺规避：执行器/评估器**必须异源**，留出法**不够**

**机制/最佳理论**：
*   **核心定理**：若 $P_{eval}$ 与 $P_{exec}$ 共享参数前缀或蒸馏血缘，则 $\exists \delta > 0$ 使得 $P_{eval}(y|x) \geq P_{true}(y|x) + \delta$（自洽性偏见）。留出法只能缓解“题目泄露”，无法消除“归纳偏见泄露”——模型倾向于给符合自己先验分布的答案打高分，即使答案错。
*   **宪法级方案**：**三权分立架构**（执行器 $M_\pi$ / 评估器 $M_{critic}$ / 审计器 $M_{audit}$），且 **$M_{critic}$ 必须来自不同架构族或不同数据流源头**。
    *   你的路线里唯一异源资源：**BitNet b1.58 原生三值 QAT 模型（0.5B 架构层）** vs **0.4B 继承层（fp16/蒸馏而来）**。这是天然异源点。
    *   **强制约束**：$M_{critic}$ = **冻结的 0.4B fp16 教师模型**（或其量化快照），**永不参与**任何 QAT/蒸馏/RL 更新。$M_\pi$ = 正在训练的 0.5B BitNet 学生。
    *   **状态机 4 状态**：
        1.  **PROPOSE** ($M_\pi$ 生成 $(q, a, \tau)$：问题、答案、推理轨迹)
        2.  **CRITIQUE** ($M_{critic}$ 打分 $r = R(q,a,\tau)$，**仅看轨迹一致性与事实校验**，不看正确性标签)
        3.  **AUDIT** (规则引擎 + 外置记忆层硬核对：关键实体/逻辑链/owner_corpus 召回一致性，**非模型打分**)
        4.  **COMMIT** (仅当 $r > \theta_{critic} \land Audit=PASS$ 时，写入外置记忆层 L2/L3，触发 LoRA/记忆键值更新)

**可执行伪代码（Node1 CPU 单进程串行版）**：
```python
# 冻结评估器：0.4B fp16 (llama.cpp / ctransformers 跑得动)
critic = FrozenModel("0.4B_fp16_teacher.bin")  # 只推理，梯度全切
# 执行器：0.5B BitNet b1.58 (训练态)
actor = BitNet158_05B(trainable=True)         # 只做 QAT/蒸馏更新
# 审计器：纯 Python 规则 + FAISS(外置记忆层)
audit = RuleAuditor(memory_layer=ExternalMemory)

def rsi_step(batch_size=8):
    # 1. PROPOSE: Actor 自发生成 (温度 0.7, top-p 0.9)
    proposals = actor.generate(n=batch_size, prompt="发现你不知道但应当知道的知识点，给出问题+尝试回答+推理链")
    
    # 2. CRITIQUE: Critic 打分 (温度 0.0, 确定性)
    # 打分维度：逻辑自洽性(0/1)、推理步骤完整性(0-1)、与外置记忆冲突度(0-1)
    scores = []
    for p in proposals:
        # Critic 只看轨迹，不给标准答案
        prompt_critique = f"[INST] 评估推理链是否自洽、步骤完整、无幻觉实体: {p.trace} [/INST]"
        logits = critic.logits(prompt_critique)  # 取 "是/否" 或 0-3 分 token 概率
        scores.append(calibrate_score(logits))   # 温度校准后的标量
    
    # 3. AUDIT: 硬规则过滤 (CPU 极快)
    audit_pass = []
    for p in proposals:
        ok = True
        ok &= audit.check_entity_grounding(p.trace, p.answer)  # 实体是否在记忆层/维基/owner_corpus
        ok &= audit.check_logic_chain(p.trace)                 # 简易逻辑链完整性 (正则/依存)
        ok &= not audit.conflict_owner_corpus(p.answer)        # 与机密语料零冲突
        audit_pass.append(ok)
    
    # 4. COMMIT: 双阈值通过才入库/更新
    accept = [(s > 0.85) and a for s, a in zip(scores, audit_pass)]
    for p, flag in zip(proposals, accept