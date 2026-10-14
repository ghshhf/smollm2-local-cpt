# 校验记录（执行手册版，2026-10-14）

本文件记录本次交付中**工具校验的实际结果**，包括通过项、未通过项、以及我判断为工具缺陷而不修改内容的部分。凡未通过的，均说明为什么保留原样而不迁就校验器。

## 1. 五份结构化产物的校验结果

| 产物 | 校验器 | 结果 | 错误 | 警告 |
|---|---|---|---|---|
| `execution_manual_record.json` | `validate_hypothesis_schema.py` | **VALID_BLOCKED_BY_GATES** | 0 | 0 |
| `execution_manual_falsification_controls.json` | `check_falsification_controls.py` | INVALID_CHECKLIST | 16 | 16 |
| `execution_manual_operationalization.json` | `check_operationalization.py` | **VALID_WITH_MEASUREMENT_GAPS** | 0 | 1 |
| `prediction_rival_matrix_v3.csv` | `validate_prediction_matrix.py` | INVALID_MATRIX | 9 | 0 |
| `evidence_ledger_v3.csv` + `search_boundary_v2.json` | `audit_evidence_ledger.py` | **VALID_FOR_HUMAN_SOURCE_REVIEW** | 0 | 0 |

记录规模：假设 16、预测 24（含 8 条竞争假设的对侧预测）、测量 14、阴性对照 6、零假设 16、分析 6、因果估计量 2、声明来源 11。

## 2. 未通过项一：`FOCAL_AND_RIVAL_EXPECTATIONS_IDENTICAL`（16 + 9 次）

**这是校验器的缺陷，不是内容缺陷。** 已第二次核实：

```python
def _normalize(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", value.casefold()).strip()
```

该归一化**删除所有非 ASCII 字符**。纯中文的 `expected_if_focal` 与 `expected_if_rivals` 被同时归一化为空串，于是恒等比较成立。

**我的独立复核**（用保留汉字的比较）：

```
矩阵 v3：总行数 24 | 汉字感知比较下 focal==rival 的行：无
对照清单：汉字感知比较下 focal==rival 的判别测试：无
```

被标记的 9 行（P8、P9、P10、P11、P13、P16、P-R5、P-R6、P-R8）经逐行验证，`norm(focal)` 与 `norm(rival)` **均为空串**——即纯中文行；含拉丁字符（NLL、FLOPs、2σ、epoch）的行全部通过。

**处理决定：保留中文原文，不迁就校验器。** 理由：① 用户的工作语言是中文，交付物必须可读；② 改写为英文会牺牲可读性而不增加任何科学信息；③ 添加无意义的 ASCII 标签是欺骗校验器。本缺陷已第二次出现（上一版审计中为 `prediction_rival_matrix_v2.csv` 的 13/14 行），若后续仍要使用这套工具，建议向工具维护方报告该归一化实现。

## 3. 未通过项二：`HUMAN_REVIEW_INCOMPLETE`（16 次，警告）

对照清单中每个假设的 `human_review_status` 为 `pending`。**这是如实状态**：本清单由 AI 生成，尚未经人类研究者复核。不改为 `approved`。

## 4. 工具报出的登记状态

- `execution_manual_record.json` 的登记状态为 **VALID_BLOCKED_BY_GATES**，因为记录中 `data_governance_gate = requires_review`（记忆层数据政策缺失，且个人数据删除权未定）。**这一阻断是真实的**，不是格式问题：Gate 6/7/8 在该项补齐前无法执行。
- `execution_manual_operationalization.json` 为 **VALID_WITH_MEASUREMENT_GAPS**（1 条警告），对应 M-LATENCY-MEM 的阈值依赖未提供的端侧硬件预算。

## 5. 本次对上一版交付的三处纠正

| 项 | 上一版 | 本次核验结果 |
|---|---|---|
| bert2BERT 的 FLOPs 节省 | 64.4% | **45%（BERT BASE）/ 47%（GPT BASE）**——原文摘要 |
| LiGO 的会议归属与数字 | NeurIPS 2023；GPT2-Medium 22.5% | **ICLR 2023**（arXiv Comments 字段）；原文可比数字为 GPT2-1.5B **约 39%**，22.5% 未在原文找到 |
| 「Memory Layers 不增 FLOPs」 | 标注为「本次检索边界内未找到支撑材料」 | **原文 §1 与 §5.4 明确写了该表述**；上一版是检索失败，本版更正 |

上述纠正来自把论文 HTML 全文下载后做句级正则检索，而不是读检索摘要。原始 HTML 与检索脚本保留在 `analysis/` 目录。

## 6. 仍标注为「未读全文」的来源

- **Shumailov et al.**：Nature 正文 `s41586-024-07566-y` 返回 303 跳转到 `idp.nature.com` 登录页，**仅读到 arXiv 预印本摘要**。本交付中凡涉及该来源的表述一律限于摘要内容，不使用其正文的量化结论。
- **Titans 深层记忆的吞吐代价**：原文以图 8 呈现，数值未提取。**上一版曾引用「15–20% 吞吐换 0.5–1.0 困惑度」这一数字，本次无法证实也无法证伪，已从手册中删除。**

## 7. 数值可复现性

§2 与 §4.1 的全部数字由 `analysis/ladder_compute.py` 生成，中间量写在 `analysis/ladder_numbers.json`；三联图由 `analysis/make_figs.py` 生成。两个脚本只依赖 numpy 与 matplotlib，常数写在文件头部并标注来源。

---
ClawsGO Science Agent
