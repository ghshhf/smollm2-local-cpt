# archive/2026-10-19 — 归档说明（不删，只归类）

> 2026-10-19 12:5x 归档。总量 **15G**。
> 原则：**这里的东西不删**。要回滚就把目录 `mv` 回原位，路径写在每行"恢复"里。

## 归档了什么

| 类目 | 大小 | 内容 | 为什么放这儿 | 恢复 |
|---|---|---|---|---|
| `legacy_qwen/` | 3.9G | `Qwen2.5-0.5B-Instruct`、`Qwen2.5-Coder-1.5B-Instruct`、`code_lora_qwen` | 上一轮"代码生成视频 + LoRA"路线的底座与适配器。现在统一走自家北极星 0.3B，千问只作历史对照 | `mv archive/2026-10-19/legacy_qwen/* models/` |
| `legacy_chain/` | 6.9G | `chain/ckpt`、`ckpt_old`、`ckpt_gpu`（含 checkpoint-03b / checkpoint-latest / run_live） | 0.2B 时代续训链的四代检查点。当前代码线以 `models/sft_code_03b` 为起点，旧代只剩考古价值 | `mv archive/2026-10-19/legacy_chain/* chain/` |
| `legacy_cache/` | 2.2G | `chain/cache_R5.npy`、`cache_R6.npy` | R5/R6 轮次的 tokenize 缓存，可由语料重算 | `mv archive/2026-10-19/legacy_cache/* chain/` |
| `legacy_base/` | 1.3G | `sftboot/base`（model.safetensors，与 `legacy_chain/ckpt_gpu/checkpoint-03b` **抽样 md5 相同** `1849f7fc6332`） | 引导包里的 0.3B 裸底座副本，`models/sft_code_03b` 已在用，这份是二份 | `mv archive/2026-10-19/legacy_base/base sftboot/` |
| `legacy_smollm135/` | 260M | SmolLM2-**135M** 两份（`AI-ModelScope--` / `HuggingFaceTB--`） | 10 月 11 日的老基线，现在最低起步就是 0.3B | `mv archive/2026-10-19/legacy_smollm135/models /mnt/workspace/models/` |

## 现在 `models/` 只剩这些（都是当前在用的）

| 目录 | 用途 |
|---|---|
| `sft_code_03b` | 代码线续训起点（step 94/94，1475 条，lr 2e-4） |
| `sft_seed200_remote` | 远程恒维增量产物（200 种子，q/k/v/o 9.2%） |
| `siglip2-base` | 图片识别线视觉编码器（1.5G，408 张量，header 已校验） |
| `whisper-small` | 听觉线 ASR |
| `data` | 历史遗留小目录（441K） |

## 规矩

1. 归档区**只增不删**。新的历史代进来时按日期开新桶 `archive/YYYY-MM-DD/`。
2. 要删归档区里任何东西，先在 `docs/remote-inventory-2026-10-19.md` 记账，再动手。
3. 目录索引由 `python node/rc.py sync` 生成，本机 `data/remote_index.json` 是它的镜像，**远程关机也能查**。
