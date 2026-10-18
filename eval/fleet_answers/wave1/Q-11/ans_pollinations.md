# pollinations (vendor=openai-oss, served=gpt-oss-20b, 24.0s)

## ① 抓取决策权（固定白名单 vs autotelic）

| ① 机制/最佳理论 | ② 可执行方案 / 伪代码 | ③ 可证伪判据 | ④ Node1 最低可跑规格 |
|-----------------|----------------------|--------------|-------------------------|
| **混合策略**：先维持一个“白名单‑白名单+边界”，再逐步允许模型在检测到 *知识缺口* 时进行 *autotelic* 抓取。<br>- **白名单**：区域、域、可控来源（公司内网、可信外部API、标注好的知识库）。<br>- **autotelic**：基于 *自我检测*（缺口评分 + 异常检测）触发抓取；抓取时再通过“二次过滤”确认质量，确保不偏离白名单。<br>- **好处**：初期安全可控；长期细化可实现自学习。 | **伪代码** | ```python
# 出发点：white_list = {...}
# 缺口检测模型：gap_model
# 抓取器：fetcher(src, result_cb)
# 过滤器：filter_fn(data) -> bool

def decide_and_fetch(query):
    # 1. 先用 gap_model 判断是否需要抓取
    if gap_model.predict(query) < GAP_THRESH:
        return  # 就地回答，无抓取
    # 2. 先尝试白名单来源
    for src in gap_model.suggest_sources(query):
        if src in white_list:
            data = fetcher(src, query)
            if filter_fn(data):
                store(data)
                return data
    # 3. 如果白名单不足，则开启 autotelic
    for src in external_candidates(query):
        if src not in white_list:
            data = fetcher(src, query)
            if filter_fn(data):
                # 只临时保存，未加入白名单
                store_temp(data)
                return data
```

- **白名单极简化**：只保留已审核多维度(x, y, z)的域。<br>- **autotelic触发阈值**：`GAP_THRESH = 0.8`（gap_model 输出 0~1）。<br>- **二次过滤**：`filter_fn = lambda d: (len(d.split())>200) and not contains_banned_terms(d)` | 只要模型在 96 % 的测试查询上产生的抓取最终标记为“已通过过滤”即视为正确。<br>错误判据：如果在 *同一日* 产生 `n` 次未通过过滤的抓取，其中 `≥5` 次导致错误信息或安全漏洞则判定为实施失败。 | **节点配置**：CPU 64核 3.2 GHz，IO × 4 HBM2L 内存（64 GB），预装 Docker + Ray 5.0，GPU 暂不必。|

> **对前提的质疑**：若 `gap_model` 的准确率低于 60 %，混合策略会引入误抓；因此必须先在本地使用已有“0.4B”模型进行统一的知识稀疏性校准，确认阈值再上线。

## ② D1 漂移检测阈值与过滤层级

| ② 机制/最佳理论 | ③ 可执行方案 / 伪代码 | ④ 可证伪判据 | ⑤ Node1 最低可跑规格 |
|-------------------|-----------------------|--------------|-----------------------|
| **双阈值漂移检测**：<br>1. **热度漂移**（EMA of query frequency）<br>2. **语义漂移**（embedding 距离）<br>结合两者给出 *D1* 前后 `drift_score`；阈值设为 `THRESH = 0.12`（标准化后）。|
| **伪代码** | ```python
# 预置 EMA_Bucket 记录最近 N 条查询
EMA_Bucket.update(query_embed)
ema_mean, ema_std = EMA_Bucket.mean_std()

# 统计与历史对齐
hist_embeds = load_hist_embeddings()  # 历史 n-last
dist = cosine_similarity(query_embed, hist_embeds.mean(axis=0))

#漂移得分
d1_score = (ema_mean - EMA_OLD_MEAN) / ema_std + dist
# 归一至 [0,1]
d1_norm = min(max((d1_score - MIN) / (MAX-MIN), 0), 1)

if d1_norm > THRESH:
    # 开始拦截
    apply_filter(query)
    log("漂移触发: %.3f" % d1_norm)
```

- **过滤层级**：<br>**层Ⅰ**：技术性错误（无效语法、空格等）<br>**层Ⅱ**：潜在敏感词汇（Banned-list）<br>**层Ⅲ**：高风险意图（如生成不实信息/指令恶意性）<br>**层Ⅳ**：灰度审计（可人工介入）| 若在 `阈值+5%` 内检测到漂移，却未触发任何过滤导致反语言模型错误输出，计为错误。<br>若没检测到漂移却出现错误信息，也为错误。| CPU 64核 3.2 GHz，4 GPU（RTX 3090 24GB，未来GPU可选）|
> **过滤极端化测试**：格式化一批“高信息量但不敏感”内容，若过滤率低于 51 % 则认为 “信息增益已被杀死”。可用精度-召回曲线进行评估。

## 防间接 Prompt 注入 / Subliminal 静默传递：最小落地防线

| ① 机制/最佳理论 | ② 可执行方案 / 伪代码 | ③ 可证伪判据 | ④ Node1 最低可跑规格 |
|-------------------|-----------------------|--------------|-----------------------|
| **双通道隐藏词表 + 上下文实时监