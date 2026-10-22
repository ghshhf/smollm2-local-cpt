### In-Context Learning Before Specialisation: Theory of Learning in a Solvable Topic Model
- 2026-10-15 | arxiv | http://arxiv.org/abs/2610.21901v1
- **问题: 研究有限主题模型中前置训练与上下文学习泛化策略的竞争，确定各自的适用时机。**
- **方法: 采用可解析主题模型，推导样本复杂度与预测器出现的阈值并进行实验验证。**
- **结论: 证明上下文预测器在样本量较小时就能出现，而专用预测器仅在训练样本数超过模型参数量后才表现出来。**
- **借鉴: 可将样本阈值理论用于本地小模型自进化或记忆层，判断何时启动适配器自训或测试时学习。**
### Two Sample Sizes in Test-Time LLM Ensembling: Buffered ERM and Minimax Theory
- 2026-10-15 | arxiv | http://arxiv.org/abs/2610.21363v1
- **问题: 解决了在测试时通过加权集群提升大语言模型性能时，如何有效利用问题数量 Q 与每问题生成数 N 两种样本规模来学习模型权重。**
- **方法: 采用带边界缓冲的经验风险最小化 (ERM) 方法，在已标记问题与生成数据的两种样本基础上学习加权集群权重，并通过最陡拟合评估其风险。**
- **结论: 证明了在满足 margin 条件下，经验风险最小化能实现 O(Q^{-1/2}+N^{-β/2}) 的 excess risk，并进一步通过简化二模型二答案模型证明其在不同 regime 下的最优性。**
- **借鉴: 可借鉴点：采用两样本尺寸的加权集群学习策略，结合 margin 缓冲机制，可用于本地小模型自进化、三元组银行或测试时学习项目中对模型权重的动态调节与模块化挂载。**
### T2-z: Test-Time Optimization of Environment Latents for Robotic World Models
- 2026-10-15 | arxiv | http://arxiv.org/abs/2610.22364v1
- **问题: 在未见环境中，预测机器人行为所需的环境因子缺失。**
- **方法: T2-Z通过可学习环境潜在向量和交替训练实现测试时环境学习。**
- **结论: 通过对潜变量的即时优化，模型在七个仿真环境中显著提升预测精度。**
- **借鉴: 可利用潜变量的测试时优化机制来实现模型的在线自适应。**
### naspaclust: A hybridisation of spatial fuzzy clustering with nature-inspired optimisation
- 2026-10-12 | arxiv | http://arxiv.org/abs/2610.16356v1
- **问题: 解决空间模糊聚类效果不佳的问题**
- **方法: 将模糊地理加权聚类与多种自然启发优化算法相结合**
- **结论: 多种优化算法的结合显著提升了空间模糊聚类的性能**
- **借鉴: 将混合优化思想应用于本地小模型自进化和模块化挂载，可提升模型自适应性**
### EMO: A Closed-Loop Test-Time Computation Framework via Alternating Context Optimization and Logits Steering
- 2026-10-15 | arxiv | http://arxiv.org/abs/2610.22160v1
- **问题: 在测试时无训练、无金标数据的情况下提升冻结语言模型的自适应性能。**
- **方法: 通过交替的上下文估计（E步）和 logits 纠正（M步）构建闭环无训练框架，利用外部评判者文本梯度更新上下文，并在共享前缀下构造置信度约束的 logits 校正。**
- **结论: 在理论假设下EMO可在单轮内显著降低目标 KL 散度，并在实验中持续改善生成响应质量。**
- **借鉴: 其无训练闭环上下文与 logits 双环调节思路，可借用于本地小模型自进化中的“做梦闭环/记忆层”更新，提供无监督的高效适配与置信度控制。**
