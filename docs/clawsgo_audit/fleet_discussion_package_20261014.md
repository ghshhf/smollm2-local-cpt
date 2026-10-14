# 舰队交叉裁决 · 后续讨论包
> 生成 2026-10-14 · 用途：把**未决/有争议**的题目打包丢给 fleet/外部模型再跑一轮；已决题目仅作背景附录

---

## 第一部分：未决 / 我有异议 / 需要再轮交叉的（5 项）

### 🔴 U1. Q1 —— V1-01 三值探针 flip=0 的真实含义（核心分歧）

**外部回答**：flip=0 说明"权重在错误基底里走"，归因是 **α 退火与 lr 没联合调度**（lr 太小没跟上三值化梯度尺度变化）。建议：提高影子权重 lr、慢退火。

**我实测反驳**：
- 实测边界带（离量化边界 |0.5γ, 1.5γ| 内）权重占 **34~46%**（翻转物理上完全可能）
- 但 lr=1e-5 在 200 步内位移仅量化步长的 **0.004%~0.031%**
- **需 3000~25000 步才可能观测到一次符号翻转**
- ⇒ **flip=0 的真身是：200 步预算 + lr=1e-5 在物理上不可能观测到翻转。是测量窗口不足，非模型行为。**
- lr 这么小，调度再精细也推不出 loss 7 倍暴涨（0.29→2.29）。

**真因（我判）**：α 线性退火把前向推向纯三值 W_q，而权重完全没动 = **未补偿的三值化冲击**。flip=0 与 loss 暴涨是因果链：权重不动 + α 退火推进 ⇒ 未补偿冲击。

**需要外部裁决的点**：
1. 在继承式三值化训练中，**权重符号翻转率（flip rate）该用什么窗口测**？是否应该用 `steps=5000, lr=1e-4` 才能看清？
2. **三值化静态冲击（α=1→0 时 holdout CE 增量）有多大**？——我已派活 V1-02 α阶梯实验（零训练纯前向），结果待收。请外部预判：delta<0.3=模型能承受三值化 / >=1.2=静态冲击大。
3. "提高影子权重 lr"在当前 200 步预算下**能否测出区别**？我的计算说不能（位移仍太小）。

---

### 🟡 U2. Q3 —— 压缩准入规则的有效秩判据（自我否定风险）

**外部回答**：把 `ΔH_saved > N_m·b_eff·ln2` 换成 **有效秩判据**：`Δ(有效秩)·b_eff·ln2 > 模块参数量`。左边计入"重组收益"，不误杀小模块。

**我挂起**：有效秩**对量化扰动本身不稳定**（量化正是降秩操作）。可能出现"量化后有效秩必然下降 ⇒ 所有模块都不达标"的**自我否定**。需先在已知好/坏模块上标定判别力，不能直接立为规则。

**需要外部裁决的点**：
1. 有没有已知工作在"量化扰动下有效秩的稳定性"上做过消融？（如用 LoRA 秩 vs 量化秩对比）
2. 是否存在**保守下界**版本：用 `min(有效秩_量化前, 有效秩_量化后)` 或只算"非量化层的有效秩增益"？
3. 我们能否用 **Fisher 谱低秩度** 替代有效秩？（外部在 Q3 回答里也提了 Fisher + 梯度协方差互信息下界，这更稳健）

---

### 🔴 U3. Q6 —— embedding 占比随规模的单调性（我承认我错了，但有后续决策）

**外部回答/实测确认**：`emb=V·d` 线性、`非emb≈12·L·d²` 二次 ⇒ **非 embedding 占比随规模单调上升**；小模型 embedding 占比反而最高。我此前说"0.2B 占比最高"**方向说反**。

**实测**：135M ckpt: 总 162.8M / emb 28.3M(**17.4%**) / 非 emb 134.5M(82.6%) / 主干线性 106.2M(65.2%)。

**实际后果（需决策）**：
- 0.2B (hidden=576) 恰在翻转点附近（emb 占比从 >50% 降到 <50% 的交叉点）
- **"embedding 保 fp16"在 0.2B 档相对代价被推到最高**，不能默认沿用 BitNet 配方
- 外部建议：**部分量化 embedding**（低频三值、高频 fp16），但它自己点出坑：查找表两套精度、梯度要分开处理，否则低频行梯度拉偏高频行。

**需要外部裁决的点**：
1. 0.2B 这个特定规模，**是否值得把 embedding 也三值化**（接受低频 token 精度损失换带宽）？还是死守 fp16？
2. **部分量化 embedding 的实现陷阱**有没有成熟方案？（LoRA-style: embedding = base_fp16 + delta_ternary，只训 delta）

---

### 🟡 U4. 盲区2 —— 语料纯度约束的自相矛盾（宪法级，不擅自降级）

**外部回答**："0.5B 前主干预训练不得含模型生成数据"的真实代价 = 0.1B→0.5B 能力上限被人写语料稀缺性卡死。而"识别主人 + 自进化 + 联网自我学习"恰恰大量依赖合成数据。**默认了"成为自己">"学到最好"，但这与"联网自我学习"目标自相矛盾**（联网会引入海量非人写数据）。

建议：纯度约束从"硬约束"降为"阶段性可审计约束"——0.3B 前保人写纯净，0.3B 后引入合成数据但打标、隔离、可回滚。

**我裁决：挂起，不擅自降级**——这是宪法硬约束（三条宪法之一），降级需用户拍板。但自相矛盾成立，**必须在 0.4B 之前决**。

**需要外部裁决的点**：
1. 有没有已知的"阶段性纯度约束"成熟范式？（如 RLAIF 的合成数据隔离、Gemma3 的 distillation curriculum）
2. **0.4B 这个节点**（我们要加中文词表、灌入 owner_corpus、训认主头），**语料纯度策略怎么定**？是保持纯净、还是开始引入合成？
3. 如果 0.5B 蒸馏教师用 Naive-N0.5-Flash（模型生成数据），主干预训练路径在 0.5B 前不得含模型生成数据 —— 这个边界怎么画？（蒸馏数据算不算"主干预训练加载路径"？）

---

### 🔴 U5. Q5 补陷阱 —— 跨层加宽时新层 γ 从哪来？（外部漏了）

**背景**：外部回答采纳"per-layer γ + 接受可预算跳变"。我补：**每层单独 γ 后，跨层加宽/插层时新层的 γ 从哪来**？

- 零初始化层：mean|W|=0 → γ=0 需 clamp 到 1e-5 ⇒ **量化步长退化为 1e-5，权重几乎全零**
- 复制式初始化：复制相邻层权重，γ=mean|W_copy| ≈ 源层 γ，**这个方案能工作**
- 但**复制式初始化在三值化下能否精确保函数**？（源层 γ 相同，目标层初始化为源层副本 ⇒ 三值后输出一致？）

**需要外部裁决的点**：
1. Net2Net 加宽/加深文献里，**量化感知的初始化**有没有现成方案？
2. 我们是否应该在加宽时**显式复制源层的 γ**（作为 buffer 保存）而非重新计算 mean|W|？

---

## 第二部分：已决 / 采纳落地的（7 题 + 3 盲区，仅作背景）

| 题 | 裁决 | 落地状态 |
|---|---|---|
| **Q2** | 联网=行动能力开关（非信息天花板）；0.5B 做"工具调用+检索"不喂语料 | ✅ 已改 CONSTITUTION A6、0.5B 方向 |
| **Q4** | 保函数只能加深、不能加宽（构造性事实非定理，GELU 下例外无效） | ✅ 已锁定深度扩容路线 |
| **Q5** | per-layer γ + 接受可预算跳变 | ✅ 已采纳（我补：每层单独统计） |
| **Q7** | 纯权重遗忘可验证性低（只能"已知攻击下不可恢复"），外置记忆库护城河更硬 | ✅ 加强原决策 |
| **Q8** | 预言机防御：①版本冻结+哈希 ②通过附 trace(proof-carrying) ③双预言机(挂起) | ✅ ①②可立即做 |
| **Q9** | 认主走双因子（行为轨迹主 + 生物高价值确认），生物不可降级（删除权载体），加密域挂起 | ✅ 采纳方向 |
| **Q10** | capability-based security + 持久层/易失层分层；梯度上传不安全；本地训练+只读联网 | ✅ 全部采纳，解掉旧冲突 P-10.3 |
| **Q11** | 行为连续性度量 = 探针输入量 KL(旧‖新)/logits 差；逐层跳变预算分配 | ✅ 采纳，与 Q1 合并指标 |
| **盲区1** | 保函数 vs 三值化互斥，需优先级律 | 🔴 **已立 CONSTITUTION A5** |
| **盲区3** | 旧档权重存为血缘档案（save_total_limit 升格血缘链） | ✅ 采纳，成本与 B9 合并核算 |
| **盲区1(子)** | Q1 原判据 Δℒ≥+1.2 判失败被推翻 | ⚠️ 已在 A5 明确废弃，改为"跳变预算+恢复能力" |

---

## 附录：已收到的完整舰队答案清单（13 题 × 6 模型）

| 题目 | 覆盖模型（6 个） |
|---|---|
| Q-01 | ext.pollinations / kris.gemini / kris.grok / kris.nemotron / kris.north / zen.openrouter |
| Q-02 | ext.pollinations / kris.gemini / kris.nemotron / kris.north / zen.mimo / zen.openrouter |
| Q-04 | kris.gemini / kris.grok / kris.longcat / kris.nemotron / kris.north / zen.openrouter |
| Q-05 | ext.pollinations / kris.gemini / kris.grok / kris.nemotron / kris.north / zen.openrouter |
| Q-08 | ext.pollinations / kris.gemini / kris.glm / kris.nemotron / kris.north / zen.openrouter |
| Q-09 | ext.pollinations / kris.gemini / kris.longcat / kris.nemotron / kris.north / zen.openrouter |
| Q-11 | kris.gemini / kris.grok / kris.nemotron / kris.north / zen.longcat / zen.openrouter |
| Q-14 | ext.pollinations / kris.gemini / kris.grok / kris.nemotron / kris.north / zen.openrouter |
| V1-01 | ext.pollinations / kris.gemini / kris.glm / kris.grok / kris.north / zen.openrouter |
| V1-02 | ext.pollinations / kris.gemini / kris.nemotron / kris.north / zen.longcat / zen.openrouter |
| V1-03 | ext.pollinations / kris.gemini / kris.nemotron / kris.north / zen.longcat / zen.openrouter |
| V1-04 | ext.pollinations / kris.gemini / kris.grok / kris.nemotron / kris.north / zen.openrouter |
| V1-05 | ext.pollinations / kris.gemini / kris.grok / kris.nemotron / kris.north / zen.openrouter |

---

## 投喂建议（给 fleet/外部模型）

请把 **U1-U5** 作为独立问题再跑一轮交叉（建议用 gemini+nemotron+grok 三模型），重点关注：
1. **U1 的物理约束**（200 步看不出 flip 是否为根本限制 vs 测量窗口问题）
2. **U3 的 0.2B embedding 量化决策**（部分量化 embedding 的实现可行性）
3. **U4 的宪法级纯度约束决策**（必须在 0.4B 前定，用户需拍板）
4. **U5 的跨层加宽 γ 初始化**（量化感知 Net2Net 的实现细节）

已决部分不必再跑，仅作上下文背景。

---

## 相关文件（全量上下文）

- 逐条裁决：`docs/clawsgo_audit/external_answers_verdict_20261014.md`
- 宪法新增：`CONSTITUTION.md`（附录 A5/A6）
- V1-02 α阶梯实验（零训练，派活 node1 跑中）：`eval/v1_02_alpha_ladder.py`
- 外部原始回答：`docs/clawsgo_audit/external_questions_20261014.md`（我们发给外部的 11 问）
- 舰队原始答案：`eval/fleet_answers/*/`