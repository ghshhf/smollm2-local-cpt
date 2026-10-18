# 0.1B 时代归档清单（2026-10-17 清理）

> 用户裁定：现在是 0.2B 时代，0.1B 时代产物不再使用；云端（GitHub `ghshhf/smollm2-local-cpt`）已有备份。
> 本清单即归档台账——物理目录已删除，需要回溯时从云端备份恢复。

## 已删除（约 12.3 GB）

| 目录/文件 | 大小 | 是什么 | 时代 |
|---|---|---|---|
| `out_smollm2_135m_pt/` | 1.6G | 135M（0.1B）预训练输出，R2 小说续训 era | 10-09~10 |
| `out_smollm2_135m_pt_val/` | 0 | 上者的验证目录 | 10-09 |
| `out_smollm2_200m_pt/` | 2.4G | 135M→200M 扩容时代的中间预训练输出（成品快照 r4/r6 在 `backups/0.2B`，云端有） | 10-10 |
| `out_smollm2_bench/` | <1K | 旧基准输出 | 10-09 |
| `backups/0.1B/` | 712M | 0.1B 分片备份（8 zip + MANIFEST + 拼回器）；云端已核对 8 分片大小与本地一致 | 10-10 |
| `out_lora/` | 164M | Qwen2.5-1.5B 时代 LoRA 实验 | 10-07 |
| `out_lora_v2/` | 3.0G | Qwen2.5-1.5B 时代 LoRA v2 实验 | 10-07 |
| `out_lora_role/` | 1.2G | 角色卡 LoRA 实验 | 10-07 |
| `out_lora_v3_test/` | 305M | v3 测试 | 10-07 |
| `out_lora_novel/` | 1.2G | 小说 LoRA 实验 | 10-08 |
| `out_lora_prose/` | 13K | 文风 LoRA（空壳） | 10-08 |
| `out_lora_prose_classic/` | 972M | 文风·经典 LoRA | 10-08 |
| `out_lora_prose_suspense/` | 1.2G | 文风·悬疑 LoRA | 10-08 |
| `out_lora_prose_urban/` | 9K | 文风·都市 LoRA（空壳） | 10-09 |
| `out_lora_v3/` | 527M | v3 LoRA 实验 | 10-08 |
| `0.2B`（文件） | 0 字节 | 误重定向产生的空文件 | - |

## 保留（现役）

- `backups/0.2B/` — 现役基底（r4_final / **r6_final_20261016**，R6 封顶版），编程/对话/识图全部从它续训
- `out_lora_chat_v1/` — 对话 LoRA（checkpoint-63 自带 tokenizer，v2 续训挂载点）
- `models/code_02b_v1/` — 编程续训（step15 落盘，暂停可续）
- `models/vlm_02b_align/` `models/vlm_test/` — 识图线
- `D:/work/_local_ai/models/SmolLM2-135M/` — 官方 135M 完整底座（10 文件与官方逐字节一致，safetensors 272 张量自洽；**唯一种子底座，勿删**）
- `gpu_ckpt/` — BitNet 试跑 checkpoint（0.5B 融合线在用）
- `netarch_v1/` — 记忆层/RSI 实验（B 线）
- `pretrain_corpus/` `data/` `corpus_04b/` `node/` `b4/` `venv_ttt/` — 语料与在用工具

## 云端备份核对（2026-10-17）

- `backups/0.1B`：GitHub API 列出 8 分片（7×99,614,720 + 48,618,221 字节）+ MANIFEST + README + 拼回器，与本地分片大小完全一致 ✅
- 组合 sha256（MANIFEST）：`8fc40113...464dd564c9`，恢复时用 `reassemble.py` 拼回自校验
