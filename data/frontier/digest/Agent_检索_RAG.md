### AdCoEvo: A Co-Evolving Multi-Agent Advertising Delivery System
- 2026-10-14 | arxiv | http://arxiv.org/abs/2610.19454v1
- **问题: 解决广告投放全流程自动化与动态适配难题**
- **方法: 采用多智能体协同进化模型，通过角色专攻与结果驱动的诊断实现跨时序、跨粒度决策**
- **结论: 系统显著提升了投放效率和市场适应性，实现了广告业务的可持续自动化运营**
- **借鉴: 可借鉴多角色分工的协同进化思路与基于实际结果的策略调整机制。**
### Agentic-Priors: Using Large Language Models to Address the Cold Start Problem in Preference Elicitation
- 2026-10-15 | arxiv | http://arxiv.org/abs/2610.21938v1
- **问题：解决新用户偏好建立的冷启动问题。**
- **方法：通过大型语言模型提炼用户交易倾向为可映射的多项式对数几率先验，随后用贝叶斯学习者迭代校正。**
- **结论：相较无信息先验，七组数据集下冷启动准确率提升5-40%，并优于AutoElicit及零上下文学习，在首期使用阶段展现显著价值。**
- **借鉴：可采用小模型与适配器模块化训练，以LLM式推理生成可解释的先验并通过贝叶斯更新实现在线自进化。**
### Bias Control and Injection for Multi-Agent Aggregation
- 2026-10-15 | arxiv | http://arxiv.org/abs/2610.21290v1
- **问题: 解决多代理推理中当代理意见不一时，如何判断可信决策的核心问题**
- **方法: 通过测试时聚合框架，利用偏差控制与注入（样式归一化与少数答案诱导），以决策不稳定性作为可靠性信号**
- **结论: 相比传统方法，该框架在三大数学推理基准和四种 LLM 组合上实现了更高的平均准确率与更佳的 token 效率**
- **借鉴: 无**
### Controlling the False Discovery Rate via Model-Agnostic Importance Measures
- 2026-10-15 | arxiv | http://arxiv.org/abs/2610.21360v1
- **问题: 在不依赖协方差分布假设的前提下控制FDR并进行协变量选择**
- **方法: MAS 方法通过样本分割、Agent‑X重要性评估、重分割每协变量与数据聚合实现模型无关的FDR控制**
- **结论: 理论证明MAS可在大样本极限下严格控制误报率，并显著提升检测功效**
- **借鉴: 重分割、交叉拟合与Agent‑X重要性策略可借用于本地小模型自进化与测试时学习，强化自适应训练和主动学习**
### Agent Status Protocol (ASP): A Lightweight Specification for State Reporting in Multi-Agent Systems
- 2026-10-15 | arxiv | http://arxiv.org/abs/2610.21538v1
- **问题: 为多代理系统提供可靠实时的状态报告机制，解决自报失效导致的监测盲点。**
- **方法: 采用轻量级 HTTP POST 事件推送，服务器通过心跳检测推断存活，使用 SSE 实时暴露聚合状态，并支持外部监督。**
- **结论: ASP 通过多维状态建模、心跳监测和外部观察融合，实现了可信、即时的多代理系统状态可观测性。**
- **借鉴: 可借鉴其轻量事件推送、心跳检测、SSE 实时更新与多维状态模型做本地小模型自进化、记忆层、模块化挂载等项目参考。**
### Auditable and Deployable Multi-Agent Motion Planning via Posetal Games from Legal Guidelines
- 2026-10-15 | arxiv | http://arxiv.org/abs/2610.22277v1
- **问题: 解决多主体运动规划在遵守法律约束下，兼顾冲突多目标的优先级与协调难题。**
- **方法: 基于posetal游戏从法律准则自动推导偏好，利用信念玩法与分布式最优响应实现隐私兼容的低信息协同。**
- **结论: 证明自定义仅消除不合法均衡，且实验显示在海事导航中实现了可审计、可部署且安全高效的多船协同。**
- **借鉴: 可借鉴其使用偏好推导与信念玩法实现低通信量协同的做法，适用于本地小模型自进化与模块化挂载等项目。**
### Believed but Not Said: Measuring Hidden Belief Collapse in Multi-Agent LLM Discussions
- 2026-10-15 | arxiv | http://arxiv.org/abs/2610.22138v1
- **问题: 如何量化多智能体LLM讨论中隐藏的信念崩溃与误导传播？**
- **方法: 通过引入一至三名对抗智能体，实时抽检各智能体私有答案及公开立场，评估错误采用率。**
- **结论: 单一或多名对抗者可将错误采纳率从0.6%提升至44.8%–60.5%，且私有误信程度远高于公开同意，表明错误主要在讨论内部隐蔽扩散。**
- **借鉴: 采用对抗实验和隐私层抽检可帮助本地小模型自进化、记忆层、模块化挂载等项目及测试时学习机制及时发现并纠正内部信念漂移。**
### CARA: Causally Guided Retrieval Agent for Question Answering
- 2026-10-15 | arxiv | http://arxiv.org/abs/2610.22017v1
- **问题: 用文本丰富的知识图谱检索答案时，需要识别真正有助于回答问题的证据，而非仅仅匹配文本。**
- **方法: CARA 为无监督的因果引导检索代理，通过后果性与必要性评估、路径一致性和互惠排名融合，显式评估检索证据的因果相关性。**
- **结论: 在 STaRK 三个数据集上，CARA 在检索效率、准确率与答案可信度方面均显著优于现有检索器，同时减少了图节点访问和检索次数。**
- **借鉴: CARA 的因果相关性评估与路径一致性可为本地小模型自进化项目提供自监督检索门控和模块化挂载的启示。**
### Diviner: Agentic RCA with Human-Governed Adaptation for Large-Scale Infrastructure
- 2026-10-15 | arxiv | http://arxiv.org/abs/2610.21281v1
- **问题: 解决大规模分布式系统中跨服务、数据、配置、计算及组织边界的根因分析难题**
- **方法: 采用专家知识与历史事件编码构建可信知识图谱，并通过自适应 Lead 与上下文隔离的工作器，对实时证据进行检索、评估并生成因果假设**
- **结论: 实验显示 Diviner 能更快、更准确定位根因，显著提升调试效率和可追溯性**
- **借鉴: 可将知识图谱+模块化工作流与自适应推理相结合，为本地小模型自进化和测试时学习项目提供可复制的框架**
### FDE-Gym: The Last Mile of AI Delivery
- 2026-10-15 | arxiv | http://arxiv.org/abs/2610.22350v1
- **问题: 解决 AI 代理在真实业务交付中如何处理模糊需求、实时系统和未声明约束的挑战。**
- **方法: 引入 FDE‑Gym，提供 95 个端到端交付任务的模拟环境，让代理通过与利益相关者互动并操作生产类系统来评估交付效果。**
- **结论: 在真实环境（必须发现未声明约束）下，最佳模型平均得分为 0.56，显示 AI 代理在实际交付场景中还有提升空间。**
- **借鉴: 可借鉴点是将真实业务场景建模为可调试的模拟环境，结合利益相关者交互与多维度评分，为本地小模型自进化和“测试时学习”提供可行框架。**
### Feasibility Ceilings Before Learning: Sizing and Timing Bounds for Drone-Mesh Interception
- 2026-10-14 | arxiv | http://arxiv.org/abs/2610.20823v1
- **问题: 评估多代理拦截网状防御在覆盖性能上的上限**
- **方法: 通过闭式解析对噪声轨迹的不确定性和覆盖几何进行推导**
- **结论: 推导的覆盖上限为平面90.69%（立体74.05%），误差不超过1.8%**
- **借鉴: 无**
### From One Query to Many: A Critical Survey of Privacy Accounting in RAG
- 2026-10-15 | arxiv | http://arxiv.org/abs/2610.21245v1
- **问题: 解决在多查询RAG系统中如何解释和比较差分隐私保证的问题。**
- **方法: 构建针对隐私粒度、计量范围与发布边界的三维框架来评估DP-RAG保证。**
- **结论: ε值只有在与保护数据单元、计量范围及可观测输出一起解释时才具备实际意义。**
- **借鉴: 采用多维度隐私评估框架，细化本地小模型在自进化、记忆层与模块化挂载过程中的隐私成本与阈值。**
### GACC-Bench: Benchmarking Goal-Adaptive Context Compression Ability across Large Language Models
- 2026-10-15 | arxiv | http://arxiv.org/abs/2610.21893v1
- **问题: 评估大语言模型在向特定目标压缩上下文时的适应能力。**
- **方法: 通过GACC‑Bench基准，提供源上下文、目标和长度预算，测量压缩后答案质量。**
- **结论: 现有十种压缩方法在目标条件下仍无法达到完整上下文的表现，目标条件压缩显著提升准确率。**
- **借鉴: 可将目标导向的压缩结果与三元组银行结合，作为本地记忆层输入，配合适配器自训实现闭环测试时学习，减少冗余存储并提升效率。**
### HTN-Orchestrate: A Hierarchical Task Network Framework for Multi-Agent Orchestration
- 2026-10-15 | arxiv | http://arxiv.org/abs/2610.22195v1
- **问题: 如何在多代理协同完成复杂任务时兼顾局部子任务执行与整体任务规划？**
- **方法: HTN-Orchestrate 使用两层框架：Planner 持有层级子任务划分并在前置结果就绪后才延迟展开；本地 Orchestrator 负责叶子子任务的调度与执行，结果递归上升，必要时重构任务网络。**
- **结论: 在 Web 研究、科学推理与科研工作流程等五大基准上，HTN-Orchestrate 在相同基础模型下，分别以 89.73% 和 73.44% 的准确率大幅领先 ReAct、Agent Team 和 DeepSeek Harness，边幅分别为 2.45 和 4.69 百分点。**
- **借鉴: HTN-Orchestrate 的动态分解与结果驱动的任务网络重建思路，可借鉴于本地小模型的模块化挂载与测试时学习。**
### Harnessing Disagreement: Detecting Correlated Agreement Blindness in Multi-agent Triage
- 2026-10-17 | openalex | https://doi.org/10.1007/978-3-032-41794-7_21
- **问题: 识别并解决多代理仲裁中因模型共识导致的安全监控盲区。**
- **方法: 通过ARAT构建随机森林、k近邻与路由元模型三方联盟，采用统一决策与保守覆盖。**
- **结论: 在UNSW-NB15数据上，ARAT将低报率由4.80%降至1.70%，显著优于单模型或损失调优。**
- **借鉴: 可采用多模型异构组网与路由元模型，以增强自适应安全判定和测试时学习。**
### Independent Data-Integrity Verification for LLM-Driven Functional Coverage Closure in Hardware Design
- 2026-10-15 | arxiv | http://arxiv.org/abs/2610.21570v1
- **问题: 解决LLM驱动的功能覆盖闭环中仅检测状态而不验证中途数据完整性的问题。**
- **方法: 采用独立覆盖率回闭合代理，读取仿真报告并使用分离计分板进行数据完整性校验。**
- **结论: 在三种不同RTL设计上，代理在8次迭代内将覆盖率从91.7%、88.9%与67.4%提升到100%，并发现一次时钟域交叉错误。**
- **借鉴: 可通过循环自闭合、独立计分板验证以及设计无关代理，为本地小模型自进化、模块化挂载和记忆层提供参考。**
### LIBERO-PeRM: Benchmarking Personalized Robotic Manipulation
- 2026-10-15 | arxiv | http://arxiv.org/abs/2610.22372v1
- **问题: 为普适机器人预留并评估用户偏好，解决任务完成与偏好满足分离问题。**
- **方法: 通过LIBERO-PeRM构建可控制的偏好基准，提供可语言表达的偏好分类并生成含偏好与无偏好的对照演示。**
- **结论: 该基准能系统评估机器人在零样本、迁移、冲突处理等场景下对偏好的理解与执行，显著提升个性化操作效果。**
- **借鉴: 可借鉴其偏好表征与自动评估流水线，在本地小模型自进化与记忆层等项目中实现偏好学习和测试时学习。**
### Lean Graph Search for Formal Theorem Proving
- 2026-10-15 | arxiv | http://arxiv.org/abs/2610.22339v1
- **问题: 解决正式定理证明中相关引理检索不足的多跳依赖问题**
- **方法: 基于Lean 4 Mathlib前向依赖构建图并通过多跳图遍历传播相关性，重新排名候选引理**
- **结论: 在MathlibQR与MathlibMPR检索基准上，召回率提升12%，完整证明路径覆盖率提升**
- **借鉴: 采用图结构、相对分布式递归检索与多跳启发式排序可为本地小模型自进化的记忆层/三元组银行等模块提供多跳关联检索思路**
### Learning Self-Orchestration in Decentralized Agent Swarms
- 2026-10-15 | arxiv | http://arxiv.org/abs/2610.22322v1
- **问题: 在多智能体系统里，如何联合学习动作策略与自我编排以提升去中心化群体表现？**
- **方法: 采用市场化后训练（MPT），让智能体通过竞价决定行动并用局部奖励进行联合学习。**
- **结论: MPT在制造、机器人导航和深度学习任务上均优于单一后训练，验证了自我编排和动作策略同步提升的有效性。**
- **借鉴: 可借鉴其无中心编排、竞价决策与局部学习信用的框架，用于本地小模型自进化与模块化挂载。**
### Learning to Steer LLM Agents and Multi-Agent Systems
- 2026-10-15 | arxiv | http://arxiv.org/abs/2610.22316v1
- **问题: 解决大型语言模型代理和多代理系统在执行过程中偏离整体目标（目标漂移）的问题。**
- **方法: 设计并训练一个实时引导代理，采用低成本快路径和高成本慢路径两种干预方式，通过强化学习结合排序目标与GRPO实现。**
- **结论: 该引导代理显著提升了单一代理和多代理系统在信息检索任务中的性能与效率，且具备跨域泛化与无目标适配能力。**
- **借鉴: 通过实时监控与可插拔干预策略，可为本地小模型自进化项目提供模块化挂载与测试时学习的自适应控制框架。**
### Librarian Agents: Training LLMs to Curate Long-Context Workspaces that Anticipate Future Queries
- 2026-10-15 | arxiv | http://arxiv.org/abs/2610.22273v1
- **问题: 解决大语言模型在跨文档推理中凭借被动检索导致的性能瓶颈。**
- **方法: 使用强化学习训练构建者代理，将原始文件编排成层级化的合成文件系统，形成可供解答代理使用的工作空间。**
- **结论: 该方法通过预先构建可复用工作空间，显著提升了多文档推理任务的回答准确率。**
- **借鉴: 可借鉴其事前构造可复用工作空间与基于目标导向的RL训练思路。**
### My Credentials, Please: Detecting Filesystem Capability Laundering in MCP Servers at Scale
- 2026-10-15 | arxiv | http://arxiv.org/abs/2610.22301v1
- **问题: 检测MCP服务器中工具可跨契约访问主机文件的文件系统能力洗钱漏洞。**
- **方法: 采用MCP‑FSD构建工具目录，推断最小权限边界，植入凭证，动态验证并评估可利用性与沙箱隔离。**
- **结论: 16.4%（787/4806）服务器存在未声明的文件系统能力，动态验证确认150例可利用漏洞。**
- **借鉴: 可借鉴动态验证与沙箱检查机制，结合最小权限推断来强化本地小模型自进化项目的安全与可靠性。**
### Poisoning the Neighborhood: A Topology-Aware Topic-Level Attack on Large-Scale HNSW-based RAG Systems
- 2026-10-15 | arxiv | http://arxiv.org/abs/2610.21464v1
- **问题: 设计在 HNSW 基础的检索增强生成（RAG）系统中，对检索文档进行有效毒化攻击。**
- **方法: 构造 Hub‑Satellite 结构的毒化文档，并结合证据锚定与实体‑主张耦合提升目标实体的检索效果。**
- **结论: 在 200 万文档的 Wikipedia 数据集上，该攻击显著提升目标实体在检索与生成中的成功率。**
- **借鉴: 无**
### Private Retrieval Augmented Generation via Random Projection
- 2026-10-15 | arxiv | http://arxiv.org/abs/2610.22297v1
- **问题: <如何在检索增强生成中实现对检索数据库的隐私保护？>**
- **方法: <使用相同的随机投影映射对检索库与查询进行投影，保持语义相似度并抑制数据提取攻击。>**
- **结论: <实验表明，该轻量化随机投影方法在多种RAG体系结构与数据集上实现更佳检索与生成效果，并在强攻击模型下具备与差分隐私同等的隐私级别。>**
- **借鉴: <对本地小模型自进化项目可借鉴的点是：采用单一投影映射的简单隐私保护机制，既易于实现又保持性能，可作为模块化挂载与测试时学习的轻量化隐私策略。>**
### Provable Coordination for LLM Agents via Message Sequence Charts
- 2026-10-17 | openalex | https://doi.org/10.1007/978-3-032-40121-2_12
- **问题: 解决LLM驱动多智能体系统中的死锁与类型不匹配通信错误。**
- **方法: 采用基于消息序列图的领域特定语言进行协作规范，并通过语法导向投影产生无死锁本地程序。**
- **结论: 在LLM非确定性环境下实现了无死锁、可验证的协作，成功演示诊断共识协议。**
- **借鉴: 通过将通信结构与不确定组件拆分并提供语法驱动的无死锁投影，为本地小模型自进化项目提供可模块化、可验证的协作与自适应训练框架。**
### Retrieved but Not Read: Understanding Why Search-Augmented LLMs Fail to Use Available Evidence
- 2026-10-15 | arxiv | http://arxiv.org/abs/2610.21731v1
- **问题: 解释搜索增强大型语言模型为何在有可用证据时仍产生过时答案。**
- **方法: 通过构造时间变化事实、使用检索与层级分析，评估模型对检索片段和源页面的处理方式。**
- **结论: 模型往往仅依据搜索片段回答且未打开源页面，导致大量过时回答和不准引用，甚至当检索证据影响内部预测时，最终答案仍可能偏向旧信息。**
- **借鉴: 说明检索证据虽能影响内部预测，但需加入证据验证与实时更新机制，可借鉴于模块化挂载与记忆层设计。**
### SSR: Structural Signal Retriever for Scalable Agent Discovery in LLM-Orchestrated Multi-Agent Systems
- 2026-10-15 | arxiv | http://arxiv.org/abs/2610.21113v1
- **问题：解决多智能体系统中，LLM编排器如何高效检索合适 Agent Card 并保持候选上下文紧凑的问题。**
- **方法：采用 SSR——利用结构化词信号、IDF 权重和倒排信号索引的确定性无训练检索方法。**
- **结论：在 1,000 名 Agent 上，SSR 将 LLM 输入量降低 98.8%，时延缩短 70.9%；在 MS MARCO 上，IDF 提升检索效果，ISI 保持精确度。**
- **借鉴：可借鉴 SSR 的结构化签名+IDF 权重、无训练倒排索引机制，帮助小模型自进化、记忆层和测试时学习模块的高效实现。**
### Shortcut Sensitivity in Agent-Safety Evaluations
- 2026-10-15 | arxiv | http://arxiv.org/abs/2610.22243v1
- **问题: 评估AI安全监控时快捷手段导致评估标签受到无关提示影响。**
- **方法: 三层短路审计框架，区分标签可预测性、监控关联与决策敏感度。**
- **结论: 在 LinuxArena 与 TS-Bench 数据集中发现短命令与标签高度相关，域外或权重调整导致 F1 指标显著下降。**
- **借鉴: 采用交互式多配置屏幕评估、干预标记与工具别名对决策影响的实验方式。**
### The Delegation Dilemma: Economies of Scale, Accountability, and Polycentric Governance
- 2026-10-15 | arxiv | http://arxiv.org/abs/2610.21746v1
- **问题: 探讨在委托治理中规模经济与问责之间的权衡，及其对机构自组织的影响。**
- **方法: 采用内生委托的重复博弈模型，并通过大语言模型仿真检验规模-问责关系与多中心治理。**
- **结论: 发现存在一个内生的可问责选区规模，在此规模内委托可持续；满足条件时，完全委托且可问责的治理必须采用多中心结构。**
- **借鉴: 可利用多中心交互与可问责机制限制代表偏差，适用于本地小模型自进化项目中的模块化挂载与记忆层设计。**
### Idea2Prior: Auditable Paper Retrieval and Ranking from Research Ideas
- 2026-10-14 | arxiv | http://arxiv.org/abs/2610.20857v1
- **问题: 解决从非正式研究构想中检索相关论文，避免仅凭关键词匹配导致检索错误的问题。**
- **方法: 结合多源检索、身份验证、基于面板的摘要评估、语义引用扩展与专家联合排名，构建可审计的文献检索系统。**
- **结论: 在十个 EASY、MEDIUM、HARD 查询中，Hit@5 为 53.3 %，优于 ChatGPT Deep Research（43.3 %）和 Gemini（36.7 %），并以平均 8 分 53 秒完成。**
- **借鉴: 可参考其多源检索与语义引用扩展流程以及可审计的评估机制。**
### Language Is a Security Surface for Agent Guards
- 2026-10-15 | arxiv | http://arxiv.org/abs/2610.22244v1
- **问题：检验代理安全守卫的判决是否受用户请求语言影响**
- **方法：构建LinguaGuardBench，对齐多语言用户指令，测评守卫在不同语言下的判决差异**
- **结论：大多数守卫在不同语言下判决变化，稀有语言导致判决抖动加剧，尤其是参数少于13B的模型**
- **借鉴：可借鉴构建语言多样化测试集、评估模型在不同语言下行为并关注噪声与稀有语言影响**
### Learning When to Think: Utility-Guided Adaptive Reasoning for Efficient Text Retrieval
- 2026-10-15 | arxiv | http://arxiv.org/abs/2610.21030v1
- **问题: 对推理过程进行精细化监督，以避免过度泛化导致检索准确性下降。**
- **方法: 采用AURA框架，对中间推理状态按其在区分目标文档与干扰文档时的效能进行直接监督，并训练轻量级效用估计器实现推理深度自适应。**
- **结论: 在BRIGHT基准上，AURA-4B实现34.7 nDCG@10，仅需2.8步推理、96个生成token；自适应停止将延迟从4.6秒降至0.76秒，且在MS MARCO上保持了良好表现。**
- **借鉴: 可将中间状态直接监督与轻量效用估计器相结合，实现小模型的自适应推理深度与低延迟运行。**
### Matoi: Selective Revalidation of Consequence Evidence for Agent Authorization
- 2026-10-15 | arxiv | http://arxiv.org/abs/2610.21394v1
- **问题: 解决在代理授权过程中证据重用导致的成本与可靠性问题。**
- **方法: 通过有限查询感知合同将可重用的后果证据与当前执行权限和历史记录分离，只允许消费者使用已检查且仍然有效的证据。**
- **结论: 在 JavaScript 与 Python 的实验中，证据重用将完整生命周期成本分别降低 3.49% 与 1.79%，整体减损率超过 10%。**
- **借鉴: 本地小模型可通过实现基于查询感知合约的可重用证据层来提升自进化的验证效率与可靠性。**
### RCER: Reconstructing Cross-Evidence Relations for Iterative Retrieval-Augmented Generation
- 2026-10-14 | arxiv | http://arxiv.org/abs/2610.19398v1
- **问题: 解决检索增强生成中跨证据实体对应与推理依赖的重建难题**
- **方法: 通过RCER迭代构建跨证据关系，结合交互式关系提取与推理上下文重构实现动态检索与答案生成**
- **结论: 在三大多跳问答基准上比最佳基线提升约2.8 EM与3.1 F1，并在叙事与隐含推理任务中表现更佳**
- **借鉴: 交互式关系提取与推理上下文重构技术，可为本地小模型自进化的三元组银行与记忆层模块提供动态分解与多源推理思路。**
### TRACER: Trajectory-Aware Tool Retrieval for LLM Agents
- 2026-10-15 | arxiv | http://arxiv.org/abs/2610.21267v1
- **问题: 解决LLM代理在工具选择时暴露完整注册表导致的高token成本与选择效率下降的问题**
- **方法: 利用Trie结构索引已观察到的工具调用轨迹，并以混合经验与上下文先验学习的方式对后续工具进行排序**
- **结论: 在六个环境和四个模型上，TRACER平均提升工具选择准确率6-10点，令牌消耗降低80%，并在完整注册表成功率上匹配GPT-5.4**
- **借鉴: 无**
### The Wealth of Agents: Scaling Agentic AI Across Developers
- 2026-10-14 | arxiv | http://arxiv.org/abs/2610.20518v1
- **问题：如何让多位开发者协同构建代理系统以突破单体开发的规模瓶颈？**
- **方法：构造多开发者代理系统（MDAs），并通过实证与理论模型比较其与单体开发代理系统（SDAs）的性能。**
- **结论：在GAIA基准上，MDAs将单个开发者代理精度35%提升至70%，优于同等预算单体开发系统多12个百分点。**
- **借鉴：可借鉴协同构建机制与资源聚合激励，为本地小模型自进化项目中的多上下文协作和模块化挂载提供思路。**
### ToolBottleneckBench: Measuring Tool-Agent Performance under Controlled Pressures and Fixed Harness Budgets
- 2026-10-15 | arxiv | http://arxiv.org/abs/2610.21098v1
- **问题: 评估工具智能体在受控压力与固定预算下的性能瓶颈。**
- **方法: 创建 ToolBottleneckBench，记录每一步失败的首个未满足义务与预算限制，以可变工具目录、可见菜单、认证流程或返回记录编码为压力点。**
- **结论: 可见菜单压力导致最大惩罚，并与截断相关；提升截断阈值能显著缓解；大部分失败因预算耗尽；仅两种修复能胜过配置切换。**
- **借鉴: 可借鉴点：受控压力测试与精确故障跟踪方法，结合修复策略改进工具调用的可行性。**
### VoteGEO-Bench: Measuring the Impact of Generative Engine Optimization on Electoral Visibility
- 2026-10-15 | arxiv | http://arxiv.org/abs/2610.21330v1
- **问题: 评估生成式引擎使用外部网页内容时对选举信息可见度的影响。**
- **方法: 对11种重写策略进行实验，分析其对不同引擎和选举的目标曝光效果。**
- **结论: 指定目标的重写可显著提升曝光率，改进页面结构和论证强化比仅添加证据效果更好。**
- **借鉴: 可借鉴目标指定与结构优化的重写策略提升本地模型的适配器自训和测试时学习效果。**
### XViewRank: Overlap-Conditioned Score Fusion for LLM-Augmented Sparse Retrieval
- 2026-10-14 | arxiv | http://arxiv.org/abs/2610.20122v1
- **问题: 解决多视图检索结果融合同步不佳导致检索性能下降**
- **方法: 通过重叠条件调节的轻量化分数融合，将原始查询、假设视图和回退视图的BM25分数利用小型非线性评分器进行融合**
- **结论: 在TREC DL 2019/2020下，匹配输入协议显著提升nDCG@10和MAP，尤其在视图差异大的情况下效果更佳**
- **借鉴: 可将重叠条件融合的轻量化思路应用于本地小模型自进化项目的模块化检索，改进适配器自训或记忆层的以往检索效果**
### 人工智能赋能翻译教学：基于智能体破解AI翻译惰性的探索
- 2026-08-31 | openalex | https://doi.org/10.32629/er.v9i8.7328
- **问题: 解决学生因 AI 翻译功能过强而产生的惰性，导致独立思考和创新能力下降。**
- **方法: 通过大语言模型和低代码智能体平台搭建多智能体协同的闭环教学框架，利用认知解耦和学生自主建构实现分层教学。**
- **结论: 多智能体体系使学生主动进行文本分析、译文打磨和思辨优化，显著提升人机协同教学效果，避免 AI 替代思考。**
- **借鉴: 可借鉴其多智能体协同与闭环反馈机制，帮助本地小模型自进化与模块化挂载实现动态适配与学习。**
### Inertially Significant Humanoid Teleoperation via Whole Body Model Predictive Control and Online Parameter Estimation
- 2026-10-15 | arxiv | http://arxiv.org/abs/2610.22176v1
- **问题: 解决人形机器人在负载变化下的惯性参数不确定导致的控制误差。**
- **方法: 通过将全身模型预测控制 (WB‑MPC) 与扩展卡尔曼滤波 (EKF) 在线估计算法结合，实现自适应惯性参数更新。**
- **结论: 结合 EKF 的 WB‑MPC 在仿真和硬件实验中显著提升了带负载时的姿态与抓取精度。**
- **借鉴: 可将在线参数估计与运动规划模块化整合到本地小模型自进化/记忆层、模块化挂载方案中，实现动态自适应。**
### Plan or Retrieve First? A Scale Study of Tool Discovery Architectures
- 2026-10-13 | arxiv | http://arxiv.org/abs/2610.19068v1
- **问题: 在大规模工具注册表中，如何在有限上下文窗口下高效发现所需 API。**
- **方法: 先用 LLM 生成抽象工具 stub 作为检索查询（Plan-then-Retrieve），再检索注册表。**
- **结论: RtP 在单工具查询中表现更好，而 PtR 在多工具查询中更优，且跨检索预算、注册表规模和模型族均稳健。**
- **借鉴: 采用先生成抽象查询再检索的两阶段策略，可用于本地小模型自进化项目中降低记忆层压力并提升模块化挂载效果。**
### Robust Multi-Agent Pursuit with Preemptive Communication via Reachability-Based Control Barrier Functions
- 2026-10-15 | arxiv | http://arxiv.org/abs/2610.22198v1
- **问题: 解决在资源受限的多机器人系统中，间歇通信导致的安全协调问题。**
- **方法: 通过可达性集与控制栅栏函数组合的统一控制/通信策略，并基于不确定性与运动风险指标来预警状态广播。**
- **结论: 证明在通信时滞受限时可保证碰撞避免，并且预警通信比周期性方式减少77.5%的消息量。**
- **借鉴: 可借鉴可达性与预警机制来指导本地小模型在稀疏通信或测试时判断何时更新自训练/适配器，从而降低通信和更新开销。**
### Towards Agentic Agent-Based Models: Feasibility, Performance, and Statistical Model Checking
- 2026-10-17 | openalex | https://doi.org/10.1007/978-3-032-40121-2_13
- **问题: 评估在代理模型中引入大型语言模型对可靠性、计算成本和行为的影响**
- **方法: 在Mesa ABM上用统计模型检查（MultiVeStA）扩展Schelling模型，使一部分代理通过工具调用访问LLM**
- **结论: 证明LLM驱动的代理可在ABM中按预期工作，并可量化其对性能与可靠性的影响**
- **借鉴: 可借鉴的点：通过工具调用将大模型与小模型结合，并使用统计模型检测验证，可用于本地小模型的自进化、记忆层及模块化挂载等方向。**
### Learning Rational Strategies for Temporal Objectives in Multi-Agent Systems
- 2026-10-15 | arxiv | http://arxiv.org/abs/2610.22287v1
- **问题: 在未知动态的随机多智能体系统中寻找既满足指定代理人的概率时序目标又满足贝叶斯或子博弈完美均衡的联合策略。**
- **方法: 采用统计学习的Nash‑Q算法，并结合有限样本置信边界与近似最优回应判断实现对策略的稳定性与满意度两重保证。**
- **结论: 对策在条件收敛下可实现目标满足且行为稳定，且可提供满足概率的置信区间。**
- **借鉴: 可借鉴其使用置信边界的在线满意度评估与自适应策略更新机制，用于小模型自进化与模块化训练。**
### 知识图谱驱动的水电轴承智能诊断与规范问答
- 2026-08-01 | openalex | https://doaj.org/article/71a885385e5d4463a43caea5a85e0aeb
- **问题: 水电机组运维中技术规范获取与应用效率低、故障诊断可解释性不足、知识更新滞后。**
- **方法: 通过构建技术规范与轴承故障两类知识图谱，结合检索对齐、关系图卷积网络与图注意力网络实现条款级证据定位与规范问答。**
- **结论: 在实验台数据集上，BF‑KG 整体准确率达 98.4%，TS‑KG 问答系统带引用正确率为 0.648，显著提升诊断精度与可解释性。**
- **借鉴: 采用检索增强生成结合图注意力网络的子图推理方式，可借鉴于本地小模型自进化中的检索对齐与知识更新模块。**
### Three-phase autonomous optimization method for distribution transformer areas based on power distribution and utilization edge agents
- 2026-08-01 | openalex | https://doaj.org/article/40d76326da7e4314908247dbfc9e39a5
- **问题: 解决配电变压器区三相不平衡与光伏储能的自适应优化管理。**
- **方法: 通过边缘智能代理与改进的三相分支电流流模型实现三相协同多目标自主优化。**
- **结论: 动态服务处理实现了三相自适应优化，有效提升了光伏储能利用效率。**
- **借鉴: 可借鉴 edge agent 的动态状态服务机制与多目标优化框架。**
