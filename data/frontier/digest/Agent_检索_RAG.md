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
