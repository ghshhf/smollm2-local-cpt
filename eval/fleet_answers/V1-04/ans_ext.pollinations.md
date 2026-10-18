# ext.pollinations (vendor=openai-oss, served=gpt-oss-20b, 25.8s)

**① 机制 / 最佳理论**  
> **安全的 RS‑loop 状态机**  
1. **三阶段分离**（而不是单一模型）  
   - **生成器（Executor）** 负责把 “问题 + 先前规则” 变为 “答案”  
   - **评估器（Critic）** 只看生成器的 *外部输出*，不再接触内部生成过程  
   - **修改器（Regulator）** 把评估结果写入沙箱规则/代码  

   通过 *模型隔离*（保持权重不共享）实现 “自知是模型” 的逻辑要求；评估器无法在执行时直接“偷听”生成器任何中间激活，从而避免评估时给自己放水。  
2. **留出法 + 元一次近似**  
   - 对每一次自我提问，**将问题随机划分为两组**：  
     * **先行集**（“先前数据”）在训练期间已见过，用来微调 Executor；  
     * **评估集**（“全新问题”）仅用于 Critic 的评分。  
   - Critic 被训练时只出现 **never‑seen** 内容，保持客观性。  

   仅留出法并非万无一失——如果 Evaluator 本身充当了生成器的“暗箱”，它仍会趋于偏好可解释性好的答案而不论真伪。解决方案是把 **Critic 设计成多模型投票**：  
   - ① 采用两层评估器（DevFake + DevReal）  
   - ② 在同一轮次中，Evaluator 输出 *confidence score* 𝐶，Executor 在 𝐶≥τ 时才继续，低于阈值则触发 “再思考” （重问）。  

   这样产生的 𝐶 形成了一个 **“自我监督检验”**，即使 Evaluator 本身是生成器的子网，也因为它在评分时不使用自身已见过的答案而保持客观。

**② 可训练形式 – Autotelic NLL**  

> **高 NLL > 知识空洞** 的核心问题在于区分 **噪声/歧义** 与 **缺陷**。可行的多层信号分离方法：  

```
for each generated question Q:
    # Step 1 – Primary NLL
    nll_main = -log P(Q | context)
    # Step 2 – Homogenisation NLL (cross‑model agreement)
    nll_cross = mean( -log P_i(Q | context) )    # i=1..k models
    # Step 3 – Language‑model perplexity on Q
    ppl = perplexity(Q)
    # Step 4 – Semantic coherence (BERT‑score of paraphrases)
    coh = BERTscore(Q, paraphrase_of_Q)

    # Composite defect signal
    defect = sigmoid( α*nll_main + β*nll_cross - γ*coh + δ*ppl )

    # Autotelic loss ← defect · learner_loss
    learner_loss += defect * ce_loss( target, pred )
```

**解释**  
- `nll_cross`：不同模型（或同一模型的“snapshots”）的预测一致性越差，说明答案不稳，提示“知识空洞”。  
- `coh` 与 `ppl`：对于纯噪声或歧义，模型往往给出非常低的 perplexity 与高度相似的同义句（coherence≈1），从而 **抑制** defect。  
- 通过 λ 参数（α,β,γ,δ）在验证集上调优，保证 `defect≈1` 只在真正的盲点出现。  

**③ 沙箱边界与污染防护**  

| 策略 | 目的 | 操作细节 | Node1 CPU 约束 |
|------|------|-----------|--------------|
| 1. 只写 *配置文件*（JSON/TOML） | 运行时规则不改代码 | 依赖字典 → 官方 Rust/Go 解析库 | < 200 ms  |
| 2. 只在沙箱内写脚本（Python 3.10） | 执行器允许调用 | 限制 sys.path → 沙箱目录 | < 1 s  |
| 3. 哈希签名校验 | 防止篡改 | 预发布时签名 → 运行前校验 | < 10 ms  |
| 4. 只允许 *Append‑only* 记录 | 版本可追溯 | 采用 append‑only log | < 10 ms  |

> **如何保证修改不污染主线**  
> * 通过 **双机/双进程隔离**：主进程永远只读取沙箱配置文件的 **snapshot**，写入时生成新的 snapshot 并送到回滚队列；主线在回滚后重新加载；任何一次修改都可通过 `git diff` 自动回滚。  
> * 采用 **资源访问粒度**：沙箱只给模型可读、可写的 **非系统目录**，不允许访问 `/usr/lib`、`/var` 等，避免通过钩子模块污染运行时。  

**④ Node1 CPU 上的最小可跑 RS‑loop 实验设计**

> 目标：在 **Node1 的 64 核 CPU** 上验证「不自欺」的 RS‑loop 是否能完成 *一次完整的自评*。  

| 步骤 | 说明 | 代码框架（伪） | 运行时间 | 证伪判据 |
|------|------