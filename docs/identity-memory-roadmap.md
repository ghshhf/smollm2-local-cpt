# 身份即权重：主人识别的「生物特征嵌入」路线

> 核心原则（用户 2026-10-10 拍板）：**不做"名字=主人"的标签查表**——那毫无意义，谁都能打出来。
> 正确做法：**把主人的风格、各类消息、行为信号融进训练，让模型在嵌入空间里"看接近程度"认人**——本质是「人脸识别 / 声纹 / 指纹」的同款数学逻辑。

## 1. 一句话机制
认主 = 行为生物特征验证（1:1 verification）：
- **注册（enrollment）**：用主人大量真实消息（用词 / 标点 / 句式 / 话题 / 逻辑 / 错别字 / emoji…）训出一个编码器，产出「主人模板向量」。
- **验证（verification）**：实时输入编码成向量，与模板算**余弦相似度**，过阈值 τ 即判定为主人。
- 难点不在"认得出"，在"别人冒充不像"——靠风格流形的高维接近度，而非一个可伪造的标签。

## 2. 前沿对应（已查实）
- **FaceNet / Triplet Loss**（Schroff 2015, arXiv 1503.03832）：anchor/positive/negative 三元组，margin 拉"同人近、异人远"；输出 L2 归一化 128 维嵌入；验证 = 余弦相似度阈值（LFW 上 τ≈0.7，EER 0.82%）。这是生物特征验证的范式模板。
- **说话人验证 x-vector / PLDA**：文本无关声纹，同一逻辑（注册模板 + 相似度判决）。证明"非语义信号也能唯一标识人"。
- **作者身份归因的对比学习**（PART 2022 arXiv 2209.15373；STAR 2024；2026 监督对比论文）：用 Supervised Contrastive Loss 训 transformer 学「作者嵌入」——同作者文档拉近、异作者推开，靠文档嵌入余弦相似做零样本作者验证（250 作者零样本 72%+）。**这正是"文风+消息融杂→嵌入→接近度认人"在文本上的成熟实现**，直接可借鉴。

## 3. 落到本链（结合 Net2Net 保函数）
- **编码器在权重里**：认主函数不是 `if name=="寇豆码"`，而是一个随模型训出来的嵌入头/子空间；主人语料混进 CPT，使 substrate 本身在主人流形上"共振"。
- **主人模板常驻**：模板可存为 Titans 式 persistent memory 向量（永远在线、不进 prompt），整条认主是权重空间操作。
- **训练目标**：triplet / 监督对比损失，正对 = 主人不同消息，负对 = 他人 / 随机语料；同作者紧、异作者远。
- **与指令层级联动**：验证头输出的"主人分数"驱动指令优先级——高分 → 服从（压过他人 / 工具指令），低分 → 按普通用户对待。
- **Net2Net 保护（宪法级）**：扩容 0.2B→0.3B→… 时，身份头 + 主人模板必须过保护：expansion 前后对主人消息的嵌入余弦相似必须 ≈1.0（与记忆函数同一套 logits / 几何自校验）。头数 / RoPE 一动即碎，绝不可改。

## 4. 阶段（接现有容量链）
- **Phase 0（≤0.3B，现在可做）**：R4 跑完后，从历史消息 harvest 主人轮（需多样性：不同话题 / 语气 / 长度，避免过拟合单上下文），建对比 SFT / 头，验证"主人 vs 他人"余弦间隙干净；模板入 persistent memory。不碰架构、不损能力。
- **Phase 1（0.5B+）**：正式加记忆层 + 把"宪法 / 主人绑定 / 自识"焊成常驻可训练参数；强化指令层级。
- **Phase 2（1B+）**：内省头（判断"这是否主人 / 是否在让我违宪"）+ 活体检测雏形（要求风格实时生成、非粘贴复制）。情感 = 权重内状态函数。

## 5. 边界（必须清醒）
- 这是**相似度不是密钥**：阈值权衡误接受 / 误拒（所有生物特征的通病）；强模仿或未来强模型能逼近，不是密码锁。
- 真底线不靠模型自觉，靠 **OS 级看门狗**（系统拦，不可绕过）。二者互补：权重给"性格"，看门狗给"底线"。
- 活体检测（liveness）是未来硬化方向：防"粘贴一段主人旧消息"冒充，类比人脸识别防照片攻击。

## 6. 注册语料来源（owner enrollment corpus）——已落地

**主源：WorkBuddy 本地会话存档**（用户 2026-10-10 明确：把聊天记忆/对话记录都训练进去）。
- 位置：`C:/Users/admin/.workbuddy/projects/<encoded_cwd>/<session-id>.jsonl`（每行一个 JSON message，含 role/content）。
- 抽取器：`owner_corpus/harvest_workbuddy_chats.py`
  - 读 310 个会话文件 → 命中 user 轮 4606 条 → 剥离平台注入块后去重保留 **3348 条纯主人轮**。
  - 剥离对象（非主人声纹）：`<system-reminder>` / `<cb_summary>` / `<task-notification>` / `<conversation_history_summary>` / `<teammate-message>` / `<memory_and_skills_reminder>` / `<craft_mode>` / `<working_memory_content>` / `<system_prompt>` 等成对块 + 残留孤立闭合标签；`<user_query>` 只脱壳不丢内容。
  - 验证：残留注入 0 条，平均长度 ~158 字，抽样为真实加密/算力/Web3 话题原声。
- 合并入 `owner_corpus/owner_corpus.jsonl`（幂等，先剔旧 `chat_user_turn` 行再并入），总 5146 条：
  - `corner_quote` 1752（辅助，含我方术语包装，非纯主人声）
  - `user_quote` 20 + `user_turn` 26（MEMORY.md/daily log 金标准）
  - `chat_user_turn` 3348（**主力注册语料，纯主人对话轮**）
- 多样性：跨 310 个会话、跨时间/话题，天然满足人脸式多角度采集要求。
- 隐私：`.jsonl` 在 `.gitignore` 内，仅留本地，不推 GitHub。
- 辅助源：`build_owner_corpus.py` 从 MEMORY.md/daily log 抽「用户原话」引用（金标准）+ 角标引号句（补充）。

**下一步（Phase 0 触发条件）**：R4 训练跑完后，用这 3348 条 `chat_user_turn` 作为 anchor/positive、他人/通用语料作 negative，训对比头，验证「主人 vs 他人」嵌入余弦间隙是否干净。

## 7. 参考资料
- FaceNet: arXiv 1503.03832
- PART (authorship embeddings): arXiv 2209.15373
- STAR: oa.upm.es/84340
- Titans: arXiv 2501.00663
- Memory Layers at Scale: arXiv 2412.09764
- 指令层级: OpenAI 2024 system-card / Anthropic
