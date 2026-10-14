# V1-04 交叉裁决：RSI 闭环第一版状态机与「独立评估器不作弊」

> 2026-10-14 · gemini-3.8-flash / nemotron-3-ultra / openrouter 等独立交叉 · 原始答案 `fleet_answers/V1-04/`
> **强趋同**：gemini 与 nemotron 独立收敛到同一核心结论。

## 一、🔴 核心裁决（两家强趋同）

**「留出法让同源评估器在没见过的问题上打分」被一致判为"绝对不够、必然自欺崩溃"。**

| 维度 | gemini | nemotron | 裁决 |
|---|---|---|---|
| 留出法够吗 | **绝对不够**：同源模型共享隐空间拓扑与归纳偏置，执行器的"幻觉盲区"与评估器的"评分盲区"高度重叠（共谋） | **不够**：留出只缓解"题目泄露"，无法消除"归纳偏见泄露"——倾向给符合自己先验的答案打高分 | **一致判否** |
| 第一原则 | 评估器**不能是纯生成式语言模型**，必须降维到"确定性预言机(Deterministic Oracle)"或"非对称信息博弈" | 评估器/执行器**必须异源**；异源还不够，要"三权分立"（执行器/评估器/审计器），评估器必须来自不同架构族或数据流源头 | **一致：异源 + 降维到确定性** |
| 裁决交给谁 | 无偏计算环境：AST 检查器 / Python REPL / 单元测试断言 / 物理执行结果 | 独立 Critic（不同架构族）打分 | **一致：最终胜负交确定性环境，不交 LLM 自评** |

## 二、采纳的 RSI 状态机（S0-S7，gemini 最完整）

```text
S0 认知空洞扫描（Δ-NLL/高困惑触发）
   ↓
S1 试题构造器（Actor 生成 输入-预期输出-执行断言）
   ↓
S2a 求解执行器(Prover)      ┊  S2b 证伪反例器(Falsifier，注入对抗边界)
   ↓                                ↓
S3 确定性沙箱裁判（Python AST / pytest / 时延探针）
   ↓ 判定失败                    ↓ 全通
S4 拒绝+沉淀负样本            S5 拟合收益评估（测 Golden 集有无退化）
（写入不可变死信队列）           ↓ 负迁移        ↓ 无衰退且收益>0
                              S6 触发硬回滚     S7 原子提交
```

## 三、三权分立的具体配置（nemotron 独家，可直接落地）

- **执行器 M_π**：0.5B BitNet b1.58（训练态）
- **评估器 M_critic**：0.4B fp16（llama.cpp / ctransformers 跑得动，**异源**）
- **审计器 M_audit**：独立留痕（版本/权限/回滚路径）
- Critic 打分固定 **温度 0.0（确定性）**

## 四、「自知是模型」的宪法级训练目标（gemini 独家，可直接落地）

引入**选择性预测损失（Selective Prediction / Abstention Loss）**：
```
L_epistemic = (1-r)·L_task(y, ŷ) + r·c_abstain + β·KL(Confidence ‖ EmpiricalAccuracy)
```
- 模型可预测特殊 token `[I_DONT_KNOW]`（触发拒答 r=1），只付极小固定代价 c_abstain；
- 强行作答且被 Oracle 判错 → 极重惩罚；答对 → 正常奖励；
- ⇒ 迫使自我校准，不敢在不确定区给自己放水。

## 五、区分知识空洞 vs 噪声（epistemic vs aleatoric）
高 NLL 可能是"知识空洞"也可能是"语料噪声/歧义"。区分手段（结合 autotelic）：
- 真空洞：换问法/换上下文 NLL 持续高；噪声：特定表述才高、换表述即降。
- 裁决仍交给确定性 oracle（若能构造出可执行断言且失败=真空洞）。

## 六、待实测（⏳）
- 0.4B critic 能否有效检出 0.5B 三值执行器的系统性错误（不共谋）；
- CMA/Δ-NLL 空洞信号与 S0 触发的误报率。

## 七、原始答案
- `fleet_answers/V1-04/ans_kris.gemini.md`（67.3s，S0-S7 状态机+abstention 损失）
- `fleet_answers/V1-04/ans_kris.nemotron.md`（三权分立+异源 critic+具体配置）
- 其余 openrouter/north 等