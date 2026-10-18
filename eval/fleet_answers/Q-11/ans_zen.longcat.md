# zen.longcat (vendor=longcat, served=longcat-2.5-preview-free, 41.5s)

以下按你要求的格式分层回答。先给结论，再给机制、伪代码、证伪判据。

---

## ① 抓取决策权：固定源白名单 vs autotelic 的混合递进

### 机制 / 最佳理论

直接上 autotelic 在 0.5B 体量下**必失败**，原因有三，且都不是工程细节而是结构性的：

1. **知识空洞识别的元认知要求 ≈ 能力上限以上**。判断"我哪里不知道"需要对自身参数化知识做全局不确定性估计——这在 0.5B 的激活规模下没有足够的内部状态空间来做可靠的 metacognition。Autotelic 抓取的文献（Dreamer、Go-Explore 系）都建立在**大模型（>10B）** 上，其成功率的前提是模型已有足够的 world model 来生成有信息量的探索方向。

2. **词表未对齐前，autotelic 的探索空间是未定义的**。0.5B 融合版才加中文词表，词表注入后的 embedding 层处于高漂移状态（你已定 embedding 保 fp16），此时让模型自主决定"去抓什么"等于用一个未稳定的接口来定义探索边界。

3. **白名单的隐含价值不是"安全"，而是"可证伪性"**。固定源白名单的本质是给后续 D1 漂移检测提供**干净的基线分布**。没有基线，你后面所有阈值都是空中楼阁。

**正确做法：三阶段递进，不是二选一。**

```
阶段 0（0.5B 初期）：硬白名单，纯注入，零自主
阶段 1（蒸馏稳定后 1-2 周）：半自主——白名单内源按困惑度排序，抓"模型最不确定"的白名单子域
阶段 2（owner_corpus 覆盖足够后）：autotelic 探索 + 白名单约束探索方向（不是约束目标域）
```

阶段 1 是关键过渡：让模型在白名单内做"已知未知"的探索，这既训练了元认知，又把风险限制在已知干净的语料空间内。

### 可执行方案 / 伪代码

```python
class FetchPolicy:
    def __init__(self, stage, whitelist, owner_corpus_stats):
        self.stage = stage
        self.whitelist = whitelist
        self.owner_stats = owner_corpus_stats  # 每源的历史困惑度/信息增益

    def select_source(self, budget):
        if self.stage == 0:
            # 硬白名单，均匀轮询
            return self._round_robin(budget)

        elif self.stage == 1:
            # 半自主：按 owner_corpus 困惑度加权
            # 高困惑度 = 模型在此域信息不足 = 优先抓取
            scores = {src: self._perplexity_score(src) for src in self.whitelist}
            return self._weighted_sample(scores, budget)

        elif self.stage == 2:
            # autotelic 提案 + 白名单过滤
            proposals = self._autotelic_propose(budget * 3)  # 多提 3x 供过滤
            filtered = [p for p in proposals if self._in_whitelist_scope(p)]
            return self._dedup(filtered)[:budget]

    def _perplexity_score(self, src):
        # 用 0.5B 模型对 src 现有样本算困惑度
        # 归一化到 [0,1]，高值优先
        return self.owner_stats[src].recent_perplexity_norm

    def _in_whitelist_scope(self, proposal):
        # 不是 URL 匹配，而是语义域匹配
        # 用 owner_corpus 的 domain embedding 做最近邻
        return self._domain_sim(proposal) > 0.7
```

### 可证伪判据（什么结果算你错）

- **若直接 autotelic 可行**：0.5B 模型在无白名单约束下，连续 7 天抓取的内容 owner 审核通过率 >90%，且无一次注入事件 → 我的判断错，autotelic 在 0.5B 可行。
- **若阶段 1 无意义**：半自主策略的抓取内容在后续训练中，信息增益（L2 上限提升量）与阶段 0 无显著差异（<5%）→ 元认知排序无效，应直接跳到阶段 2 或退回阶段 0。

**最小可跑版（node1 CPU）**：阶段 0 只需一个 cron + curl 列表，零模型推理。阶段 1 需要在 node1 上跑 0.5B 推理算困惑度，64 核 CPU 跑 0.5B fp16 大约 30-60 秒/源/天，可接受。

---

## ② 防污染管线最小配置：D1 漂移检测阈值

### 机制 / 最佳理论

D1 漂移检测的核心矛盾：**阈值太高 → 污染进来了；阈值太低 → 正常的新信息被挡在门外**。

关键区分两种"漂移"：

| 类型 | 表现 | 处理 |
|---|---|---|
| **良性漂移** | 新源引入新实体/新风格，但语义结构与 owner_corpus 兼容 | 放行 |
| **恶性漂移** | 语义空间整体偏移，通常来自注入或低质源 | 拦截 |

最优阈值不是固定值，而是**自适应的**：

- **