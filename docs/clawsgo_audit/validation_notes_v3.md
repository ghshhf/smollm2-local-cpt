# 校验记录（未来问题全集版，2026-10-14）

本文件记录本次交付中**工具校验的实际结果**、**引文核验的实际深度**、以及**我判断不应当迁就校验器的部分**。

## 1. 结构化产物的校验结果

| 产物 | 校验器 | 结果 | 错误 | 警告 |
|---|---|---|---|---|
| `evidence_ledger_v4.csv` + `search_boundary_v3.json` | `audit_evidence_ledger.py` | **VALID_FOR_HUMAN_SOURCE_REVIEW** | 0 | 0 |
| `future_predictions_v1.csv` | 未跑校验器（见 §4） | — | — | — |

来源总数 36。其中**原文级核验 15 条**、**二手 21 条**（含 4 条 Nature 系付费墙、20 余条仅有检索摘要）。逐条的核验状态写在台账的 `notes` 字段里，格式统一为「原文级 / 二手 / 未读 + 具体核到什么」。

**校验过程中修正的四处（均为格式问题，非内容问题）：**
1. `relation` 字段用了枚举外的值（`theory`/`mechanism`/`review`/`analysis`）→ 映射到 `method`/`challenging`/`contextual`。
2. `source_type` 同样超出枚举 → 映射到 `primary_method`/`preprint`/`primary_research`/`systematic_review`。
3. `study_design_or_document_type` 有 2 处长度不足 3 字符（中文两字词）→ 补全。
4. `publication_date` 有 2 处写成 `2020/2022`、`2017/2026` → 改为首年份。

**这四处都不涉及证据内容，只涉及枚举与格式。** 记录在此，是为了让你知道台账的其余部分没有被工具改写。

## 2. 本次的引文核验深度（按你的硬要求执行）

**方法**：`curl` 抓取 arXiv 全文 HTML → 本地 HTML→文本剥离 → 句级正则检索。**不使用检索摘要作为原文级证据。**

**下载并逐句核验的 5 篇：**

| 论文 | 核到的原文段落 |
|---|---|
| Mixture-of-Depths (2404.02258) | 摘要中 `capping the number of tokens (k)`、`top-k routing mechanism`、`static computation graph with known tensor sizes` |
| Model soups (2203.05482) | `fine-tuned models often appear to lie in a single low error basin`；ViT-G `90.94%`；greedy soup 的入汤条件 |
| Grokking (2201.02177) | 摘要中 `improving generalization ... can happen well past the point of overfitting` |
| YaRN (2309.00071) | 摘要中 `requiring 10x less tokens and 2.5x less training steps than previous methods` |
| HippoRAG (2405.14831) | 摘要中 `hippocampal indexing theory of human long-term memory` |

**未核到原文、一律标注为「二手」的，共 21 条**，其中影响最大的四条：
- **Shumailov (Nature 2024)**：付费墙。
- **subliminal learning (Nature 2026)**：付费墙。**本报告 §3.8 与 §12 ★1 的核心依赖它**，因此我在报告里同时给了一个**实验（F3-8）作为补核手段**——这是我能给的最强处理方式：结论标二手，同时给出用你自己的数据核它的方案。
- **emergent misalignment (Nature 2025)**：付费墙。同样处理（F3-1、F11-2）。
- **bitnet.cpp 的 2.37–6.17× 加速比**：来源为你方自述 + 检索摘要，**本次未核**，且报告 §10.2 明确指出该数字是**纯三值**设定下的，**不能套到你们的混合精度上**。

## 3. 我明确没有引用、并说明理由的数字

- **AI 生成内容占新网页文本的比例**：本次检索返回的估计从 **10% 到 >50%** 不等，区间过宽且来源质量参差，**报告中没有引用任何单一数值**，只保留了"污染具有自增强性"这一定性结论。
- **Landauer 相关的绝对数值**（例如"训练的理论下界约 20 nJ"、"实际比 Landauer 极限低 10^21 倍"）：**未引用**。只保留了方向性结论（训练的能量价格由"写参数"而非"做算术"设定），因为它对你们的三值路线有直接推论，而绝对数值本身的置信区间太宽。
- **2026 年 arXiv 编号的若干条目**（2601–2609 段）：检索返回了它们，但我未逐一核验其存在性与内容，**台账中一律标为二手，且未用于任何定量或结构性结论**。

## 4. `future_predictions_v1.csv` 未跑 `validate_prediction_matrix.py` 的原因

该校验器要求预测的 `hypothesis_id` 能解析到一份符合 schema 2.0 的假设记录（`validate_hypothesis_schema.py` 的输出）。本轮的交付物形态是**报告 + 盲区 + 创新组合 + 最小实验**，不是一份完整的假设记录，因此没有对应的记录文件可供交叉校验。

**我没有为了让文件通过校验而伪造一份假设记录**，也没有把预测表改写成校验器期望的列结构。预测表是**自包含**的：每条预测都带 `falsifier_condition`、`indeterminate_band`、`negative_control`、`source_ids`，可以脱离任何记录单独使用。

上一轮矩阵 v3 中 `FOCAL_AND_RIVAL_EXPECTATIONS_IDENTICAL` 的假阳性问题（校验器的归一化 `re.sub(r"[^a-z0-9]+"," ",s)` 会删除全部 CJK 字符）在本次没有复现，因为本表不含"焦/竞两侧期望"这一对列。

## 5. 本报告与上一轮交付的关系

本次是**扩展**而非**修订**。上一轮（逐 0.1B 执行手册）的结论全部保留，本次新增的是：
- §1–§13 共 70 余条的逐条梳理（上一轮只覆盖了其中三个模块插入点）；
- **三条统摄规则 L1/L2/L3**（上一轮没有）；
- **§4.2 对 P1 的修正**：宽度扩容在共享/分组 α 下**不可能**严格保函数（上一轮只说"零填充稀释"，机制不完整）；
- **§3.8 蒸馏隐写通道**：可能推翻 P3，是本次最重要的发现；
- **§12 的 10 类盲区与 7 个创新组合点**。

两份定量核算脚本（`analysis/ladder_compute.py` 与 `analysis/make_fig_nstar.py`）只依赖 numpy 与 matplotlib，常数写在文件头部并标注来源，可直接复现本报告的全部数字。

---

ClawsGO Science Agent
