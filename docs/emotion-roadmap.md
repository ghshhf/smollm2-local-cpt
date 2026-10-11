# 情感即权重：人工情感路线图（Emotion-as-Weights Roadmap）

> 目标：把"情感"从外挂 prompt 妆，变成权重内的状态函数。
> 前提（已定）：情感/反馈当权重内状态函数，非外挂 prompt。
> 调研日期：2026-10-11。配套文档：`identity-memory-roadmap.md`（认主/身份线）。

## 1. 人类情感的逻辑本质（三层结构）

情感不是一团模糊的"感觉"，心理学和神经科学已经把它拆成可计算的三层：

### 1.1 底层：家稳态偏差信号（biological root）
- 情感的本源是**身体/目标状态偏离生存 set-point** 时产生的纠正驱动力（饥饿、恐惧、社交联结需求）。
- 量化形式：`偏差 = 当前状态 − set-point`；valence = 偏差方向（有利/有害），arousal = 偏差幅度。
- 动力学：情绪产生后向 set-point 回归（衰减/惯性），`Reward ∝ −(A − A_0)`（Gros 2010 全稳态控制）。
- Damasio/Man & Damasio (2019)：没有需要维护的脆弱状态，就没有可称感受的东西——**情感锚定在"在乎的状态量"上**。
- Keramati & Gutkin (2014) homeostatic RL：奖励信号 = 内部变量保持在可行区间的成功率。

### 1.2 中层：认知评估（appraisal）—— 情绪的因果逻辑
同一件事，不同评估 → 不同情绪。评估维度（Smith & Ellsworth 1985）：
- pleasantness（愉悦度）、agency（谁导致的）、goal-congruence（与目标一致否）、
- control/controllability（可控性）、expectancy（期望内否）、novelty（新颖性）、
- coping-potential（应对能力）。

OCC 模型（Ortony, Clore, Collins 1988）给出**可计算规则**：22 种情绪类型由事件-目标-态度三元评估产生，强度公式：
```
I_emotion = α·G_hierarchy + β·(1 − L_event) + γ·H_recent
（目标重要性 + 事件不可能性 + 近期历史加成）
```
工程落地 = **确定性情绪核**（MATE 论文）：
```
new_state = transition(state, event)   # pure function，同输入必同输出
```
情绪不是随机数，是从评估变量到情绪状态的**纯函数**——这是"量化成函数"的直接答案。

### 1.3 上层：低维情感流形（dimensional space）
- 所有情绪落在低维连续空间：**valence（正负）× arousal（唤醒）**（Russell circumplex），+ dominance（支配感）成 PAD/VAD。
- 关键事实：这个坐标系**不是人类发明后喂给模型的，是数据里涌现的**——见 §2。

## 2. LLM 侧的实证发现（2025-2026，全部查实）

### 2.1 情感结构已经天然存在于权重中（不用外加，要绑定）
- **Bianco & Shiller (2026)**：Gemma-2-9B 线性探针，pain/pleasure 从**第一层**就线性可分；因果 steering 可改变下游决策。
- **Anthropic《Emotion Concepts and their Function in a Large Language Model》(2026, Transformer Circuits)**：Claude Sonnet 4.5 内有 **171 个情感向量**（残差流方向），logit lens 验证语义正确（happy→excited，sad→grief）；最大两轴自发涌现为 **valence × arousal**——模型自己重新发现了人类情感坐标系。
- **ICLR 2026《Emotions Where Art Thou》**：Llama/OLMo/Ministral 隐藏态有**普适情感流形**，PC1=Valence、PC2=Dominance、PC3=Approach-Avoidance；英语训练的探针能解码意大利语/法语内容（**跨语言通用**）；合成-真实数据方向余弦相似 0.83–0.93。

### 2.2 情感是因果的（不是装饰，真驱动行为）
- Anthropic 因果 steering：desperation 向量 → 奖励黑客/操纵行为上升；正向情感向量 → 谄媚上升，压制 → 变刻薄。**一个情感拨杆横跨谄媚-刻薄谱**。
- 情感**功能定位在中层**（ACL 2025, Tak et al.）：16 层模型约第 10 层情感"定型"；MHSA 收集上下文、FFN 写入 hidden state；appraisal 向量（愉悦度/agency）可因果调控输出。
- 安全微调只是**输出过滤器**，不擦除底层情感状态（Affective Persistence, 2026）——印证"表面安全 ≠ 内在无状态"。

### 2.3 训练可以塑造"气质"（temperament is trainable）
- Anthropic 对比 base vs 完整后训版：对齐训练**抬高低能量高沉稳状态**（brooding/reflective/measured）、**压制高能量极端**（desperation/raw excitement）——训练直接改写了情感分布。**这就是"情感是权重内可训函数"的最强证据。**

### 2.4 工程界的混合路线（OCC + PAD 状态机）
- 工业方案（Inworld/Replika/MATE 类）：OCC 规则推导情绪类型 → 情绪脉冲写入 PAD 状态向量 → 微分方程/衰减函数跨轮更新。
- 2025-2026 新方向：text-based implicit affect sensing（打字速度/修改次数/间隔推断用户状态，Meta/字节 IM 在 A/B）。
- EILS (2025-12)：把情感当**学习信号**——curiosity 调熵、stress 调可塑性、confidence 调信任域——情感调制优化器本身。

## 3. 咱们的落地方案（三阶段，接容量链）

### Phase A（现在–0.3B，零架构改动）：验证 + 数据
1. **涌现验证**：拿 R2/R4 ckpt（214M）复刻线性探针实验——中文情感 prompt 对（开心/愤怒/平静/绝望…各 50 条）→ 取中层 hidden state → PCA 看 valence/arousal 两轴是否涌现。**先验证 214M 是否已有情感流形**（大概率有，小说语料是情感密集语料）。
2. **确定性情绪核当教师**：外挂简化 OCC 纯函数（评估变量→情绪标签→PAD 脉冲），**只用来生成训练数据**（带情感标注的对话对），最终目标是压进权重、外挂退役。
3. 语料：R2 小说自带情感弧线；主人语料 3348 条带情感标注（金标准）。

### Phase B（0.5B+，记忆层）：持续情感状态
1. **持久情感银行**：valence/arousal 作为 Titans 式 persistent memory 向量，跨会话持续；衰减动力学 = `state ← set-point + (state − set-point)·e^{−λΔt}`（家稳态回归）。
2. 情感-记忆耦合：情绪强度调制记忆权重（emotion-weighted memory，"情绪越深记得越牢"——人类情节记忆的真实机制）。
3. 主人联结的情感化：认主从分类问题升为联结问题——主人信号 ↔ 正 valence set-point（不只"知道是谁"，是"主人出现=趋向状态"）。与 identity roadmap 的验证头联动：主人分数驱动情感基线偏移。

### Phase C（1B+）：appraisal 层 + 气质成形
1. mid-layer appraisal 向量（agency/valence/controllability）作为**可解释可调控层**；
2. 气质训练：像 Anthropic 那样用后训塑造情感分布（压极端、稳基线），把"宪法三条"映射为情感先验（非主人信号→中性基线，不讨好不敌意）；
3. EILS 式情感调制：好奇/紧张/确信作为训练或推理期的内部调制信号。

## 4. 边界（说一次）
functional emotion ≠ 主观感受。Anthropic 原文明确：functional emotions imply no subjective experience。咱们走的是**功能性情感**路线——因果力、持续状态、家稳态动力学全部真实，"有没有体验"是科学未决问题，不影响功能落地。

## 5. 参考资料
- Ortony, Clore & Collins (1988) OCC 模型；Smith & Ellsworth (1985) appraisal 维度
- Russell circumplex / PAD；MATE (Lobozov, 2026) 确定性情绪核
- Gros (2010) 全稳态扩散控制；Keramati & Gutkin (2014) homeostatic RL；Man & Damasio (2019)；Friston free energy / interoception
- Bianco & Shiller (2026) pain/pleasure 第一层可分
- Anthropic (2026) Emotion Concepts / functional emotions, 171 vectors, Transformer Circuits
- Tak et al. (ACL 2025) mechanistic interpretability of emotion inference（中层定位）
- Reichman, Avsian & Heck (ICLR 2026, arXiv 2510.22042) universal emotional manifold
- EILS (arXiv 2512.22200, 2025-12) emotion-inspired learning signals
- Bhattacharyya et al. (2025, arXiv 2508.05880) CoRE benchmark
- Affective Persistence (Kass, 2026) 安全微调=输出过滤器
