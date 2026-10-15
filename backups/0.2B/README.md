# 0.2B 档备份（SmolLM2-214M 系）

Net2Net 保函数扩容产物：hidden 576 / FFN 3072 / 30 层 / vocab 49152 / tied embeddings（= 214.1M 参数）。
本档经历 R4（语言基础）→ R5（知识·教育）→ R6（全量知识，01:47 封顶 done=7100）。

## 两份本地权重快照

| 子目录 | 拉取日 | 是什么 | 格式 / 分片 | 拼回 |
|---|---|---|---|---|
| [`base_20261010/`](base_20261010/) | 10-10 | Net2Net 扩容后底座（R4 step 0 重启时点） | safetensors 856,595,368 B / 9 片 | `reassemble_02b.py`（大小+sha256 自校验） |
| [`r4_final_20261011/`](r4_final_20261011/) | 10-11 | **R4 语言档训练后权重**（本库最新 0.2B 成品） | pytorch_model.bin 856,674,211 B / 9 片 | `reassemble.py`（大小断言） |

## 语料谱系（仅本地，不入 GitHub）

[`r4_corpus/`](r4_corpus/)：R2–R5 全部分级语料 + 维基tmp（约 1.6GB，`manifest.txt` 带 sha256）。
*.jsonl 被 .gitignore 排除，只存在于本地/NAS；本目录 README 说明用途。

## ⚠️ 缺口：R5 / R6 正式版未拉回

- R6 已于 2026-10-16 01:47 封顶（done=7100，loss 1.74→1.41），856MB 级权重在 node1 NAS。
- node1 02:01 后失联（疑似实例回收）。恢复后按 backups/README 规约 §5 拉回，落 `0.2B/r6_final_<日期>/`。
- 拉回前，GitHub 上最新的 0.2B 成品是 `r4_final_20261011/`。

## 加载方式

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
m = AutoModelForCausalLM.from_pretrained("<拼回目录>")   # 用同目录 config.json + tokenizer
t = AutoTokenizer.from_pretrained("<拼回目录>")
```
