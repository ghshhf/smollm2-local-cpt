# kris.nemotron (vendor=nvidia, served=nvidia/nemotron-3-ultra-550b-a55b, 60.9s)

### ① 抓取决策权：白名单→自主 的「两阶段课程」是唯一可行路径，直接 Autotelic 等于自杀

**机制/最佳理论**  
- **知识空洞识别 ≠ 信息价值判断**。小模型（≤0.5B）的 `p(know|x)` 校准误差极大，**熵高 ≠ 真空洞**，更多是「没学会表示」。直接 Autotelic 会把「自己不懂的垃圾」当宝抓回来，触发 **L2 信息天花板的反向突破：把天花板砸成地板**。  
- **课程学习理论（Bengio 2009）+ 信息增益上界**：  
  `IG_max = H(y|x) - H(y|x, D_clean)`。白名单阶段先把 `H(y|x, D_clean)` 压到模型容量极限，再开放 Autotelic 时，模型才有「判断力剩余」去评估 `ΔIG`。  
- **宪法约束 L3 通道分离**：抓取通道（网络 IO）与推理通道（CPU 算力）必须物理隔离，**白名单阶段不占推理预算**，Autotelic 阶段才引入「抓取预算 token」。

---

### ② 可执行方案：两阶段课程 + 硬性熔断指标

```python
# fetch_policy.py  ——  放在看门狗进程（独立于推理进程），单线程跑
WHITELIST = [
    "https://arxiv.org/rss/cs.CL", "https://arxiv.org/rss/cs.AI",
    "https://github.com/trending?since=daily",   # 只抓 README.md
    "https://huggingface.co/api/daily_papers",  # 官方精选
    "file:///mnt/owner_corpus/incremental/",     # 本地增量投喂
]
AUTOTELIC_BUDGET_TOKENS = 2048   # 每轮推理最多让出给抓取的 token 额度
MIN_IG_THRESHOLD = 0.02          # nats/token，低于即丢弃
MAX_DRIFT_KL = 0.15              # D1 漂移阈值（见下文）

class FetchPolicy:
    def __init__(self, model, tokenizer, mem_layer):
        self.model = model          # 0.5B BitNet 推理实例（INT8/1.58 量化）
        self.tok = tokenizer
        self.mem = mem_layer        # 外置可寻址记忆（10M/人）
        self.phase = "WHITELIST"    # WHITELIST | AUTOTELIC
        self.whitelist_done = 0     # 已完成白名单轮次
        self.IG_ema = 0.0           # 信息增益指数移动平均

    def step(self, trigger_context: str) -> list[str]:
        """
        trigger_context: 当前对话/任务最后 512 token，作为「好奇心种子」
        返回：本轮允许抓取的 URL 列表（空则不抓）
        """
        if self.phase == "WHITELIST":
            urls = self._whitelist_round()
            if self.whitelist_done >= 3:   # 3 轮全源覆盖即可
                self.phase = "AUTOTELIC"
                log("🎓 白名单毕业 → Autotelic")
            return urls

        # ---- AUTOTELIC 阶段 ----
        # 1. 用模型自己打分：这段上下文里「我最不确定且最想知道」的 span
        gaps = self._detect_knowledge_gaps(trigger_context)
        if not gaps:
            return []

        # 2. 用搜索 API（DuckDuckGo html scrape / searxng 本地实例）把 gap 变查询
        queries = [f"{g} site:github.com OR site:arxiv.org" for g in gaps[:3]]
        cand_urls = self._search(queries, topk=10)

        # 3. 轻量预打分：用 0.1B 小模型跑一遍摘要概率，估算 IG
        scored = []
        for u in cand_urls:
            summary = self._fetch_summary(u, max_tokens=128)
            if not summary: continue
            ig = self._estimate_ig(summary)
            if ig > MIN_IG_THRESHOLD:
                scored.append((ig, u))
        scored.sort(reverse=True)

        # 4. 预算裁剪
        budget = AUTOTELIC_BUDGET_TOKENS
        picked = []
        for ig, u in scored:
            est_tokens = len(self.tok.encode(u)) + 256  # 粗估
            if budget >= est_tokens:
                picked.append