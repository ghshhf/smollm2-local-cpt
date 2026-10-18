# 北极星 · 本地知识地图 v1（2026-10-17）

> 用途：北极星对话收集信息的**归类骨架**。对话暴露缺口 → 新知识按域归位 → 巩固轮挂卡。
> 配套原则（前沿论文已验证）：on-policy 收集（Hy-Memory）→ 按边际贡献归因（Mosaic）→ 增量避开主奇异方向（SST）→ 不靠疯狂预训练。

## 一、LoRA 卡片层（可直接对话/挂卡，instruction→output）

### 1. 对话域 ★当前主线
| 文件 | 条数 | 内容 | 状态 |
|---|---|---|---|
| data/chat_sft_v1.jsonl | 306 | 对话格式打底（56 基础对话+250 卡片采样） | 训练中 out_lora_chat_v1 |
| data/chat_sft_v2.jsonl | 310 | +北极星身份/主人称呼 | 待 v1 验收后续训 |

### 2. 代码域（成熟）
| 文件 | 条数 | 类型分布 |
|---|---|---|
| card_code.jsonl（总卡） | 1463 | fn 774 / distill 521 / code 148 / paradigm 20 |
| card_code_lang | 748 | 语言与库问答（含 521 distill） |
| card_code_quant | 384 | 量化/交易/回测/因子 |
| card_code_data | 187 | 数据获取/清洗/存储 |
| card_code_engine | 144 | 并发/测试/安全/构建 |
| role_data_v6 | 1958 | code_fn 1338 / distill 600 / paradigm 20 |

### 3. 推理域（已去虚胖，2026-10-17）
| 文件 | 条数 | 真伪 |
|---|---|---|
| **card_reason_real.jsonl** | **123** | **唯一可训练真链**（种子 20 + extra1/2/3 103） |
| card_reason_quarantine.jsonl | 1911 | 启发式占位，**冻结不进训练**（split_reason_real.py 拆出） |
| card_reason.jsonl | 1471 | reason_cot（大部分 heuristic 标注，挂卡前需过同款占位过滤） |
| role_data_v8 / v8_long | 1958+76 | 待核来源（用 split_reason_real.py 的占位判定核） |

> 去虚胖规则（长期）：占位只隔离不删除——它是老师回填的题目清单；真链没回填前，任何训练批次禁止混入占位。

### 4. 文风域（大）
card_prose 470 + adult 141 + adult_yc 6914 + classic 131 + suspense 156 + urban 6956 ≈ 10.4k 条（prose_cont 为主）

### 5. 小说设定域
card_novel 630 + delta 185（concept 517/char 91/rel 13/outline/rules/seam）

### 6. 角色域（演进版本，取最新为准）
role_data 144 → clean 78 → v2 249 → v3 624 → v4 1232 → v5 1696（叠加累积）

### 7. 前沿域 ★新增（2026-10-17，外部知识源卡片化，v2 细分版）
| 文件 | 条数 | 说明 |
|---|---|---|
| data/card_frontier.jsonl | 190（95 篇 × 机制卡+定位卡） | v2：跨列表去重（id 规范化+标题双键）+ 细分主题 `_topic` + 相关度 `_rel`（core 19 篇） |
| data/frontier/topics.md | 主题索引 | 11 子主题：测试时优化(4★)/持续学习(2★)/记忆经验(5★)/蒸馏效率(8★)/多智能体(18)/扩散(8)/VLM(10)/评测(9)/机器人(5)/其他(26) |

> 细分规则（build_frontier_cards.py 内置，每日增量自动归类）：
> - **core（★）**：直接服务北极星进化回路（测试时优化/防遗忘/记忆/蒸馏），巩固轮配比加权多采；
> - **related（·）**：领域相关（agent/扩散/VLM/评测），正常采样；
> - **low（×）**：与本地主线无关（水印/气象/眼底等 8 篇），默认不采。
> 前沿文章=第七知识域：frontier/*.jsonl 每天新增 → 跑 build_frontier_cards.py → 增量卡进下一轮巩固。

## 二、预训练语料层（不可对话检索，仅 CPT 底料）

pretrain_corpus/：zh_text_lang（47M tok）、know_edu、know_full、r2/r3/r4/r5/r6 小说与知识、weixin。corpus_04b/creative_articles_novel。**按用户定调：不再预训练，此层冻结为底座素材。**

## 三、外部知识源（原始层，未卡片化）
- data/frontier/：每日 arxiv cs.AI/LG/CL/CV + hf-papers 各 30 篇（10-15 已 107 篇，标题+摘要）——**卡片化后进第七域**
- D:/work/_kb/：crypto 深度研究 HTML（Filecoin 等）+ ima 索引——**待问答化进主人档案/知识域**

## 四、缺口清单（对话查漏补缺的起点）

1. **对话域最薄**：基础对话仅 60 条，无多轮对话样本（当前全单轮）——北极星还不会"接话"。
2. **主人档案域 = 0**：北极星不知道主人是谁、偏好、项目、习惯——对话收集的第一优先类。
3. ~~前沿论文未卡片化~~ **已完成**：card_frontier.jsonl 214 条（2026-10-17）。
4. **知识问答域缺失**：know_* 是 CPT 语料，没有对应的可对话知识卡（常识/科学/历史 QA）。
5. ~~推理域真链占比 <2%~~ **已隔离**：真链 123 条单独成卡（card_reason_real），占位 1911 条冻结待老师回填（card_reason_quarantine）。
6. **个人语料未对话化**：weixin 语料、_kb 研究都只是原文，没有转成问答。
7. **推理占位回填待做**：1911 条占位 = 老师的题目清单，回填节奏与经验池预算合并规划。

## 五、演进规则（对话→进化的归位流程）

1. 北极星每轮对话后，judge（WorkBuddy）判定：答对 / 答错但可修 / 超纲（缺知识）。
2. 超纲 → 记入对应域的**缺口池**（按上面五域+主人档案）；答错可修 → 错因→修正对入经验池。
3. 经验池攒批（≥100 条或 24h）→ 睡眠巩固：`train_lora.py --init_adapter` 上一张卡续训。
4. 验收闸：probe 六问 + 域抽查不回退才合入，否则丢弃该批。
5. 每周日盘点本地图，更新版本号。
