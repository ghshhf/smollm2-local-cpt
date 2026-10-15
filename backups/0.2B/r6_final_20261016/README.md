# r6_final_20261016 — 0.2B R6 知识档封顶正式版

## 身份
- SmolLM2-200M（Net2Net 扩容版：hidden 576 / FFN 3072 / 30 层 / vocab 49152 / tied embeddings）
- **R6 知识教育档封顶权重**：done=7100，2026-10-16 00:16 由 node1 NAS `chain/ckpt_old/` 回捞
- 注意: ckpt_old/step.txt 内容为 `400`（链上局部步数计数），全局进度以 chain.log `R6 complete @ 7100` 为准
- sha256: `4c91e3502f54afba5733b3417f0ca2378bb0e27d5838f9a69b507b69b29d067a`（856,595,368 字节）
- 与 `base_20261010`（sha256 `0ee38bf6...`，R4 前底座）**是两份不同权重**——本份多训了 R4-R6 共 3000 步

## 文件
- `model.safetensors` — 权重本体（回捞时已拼合并校验）
- `config.json` / `generation_config.json` / `step.txt`（=7100）— ckpt_old 原件
- `tokenizer.json` / `tokenizer_config.json` / `merges.txt` — node1 base_tok 原件
- `reassemble.py` — 拼回器（幂等 + sha256 自校验，已本地验证）

## 备注
- 回捞路径：node1 split 95MiB×9 → ECS /upload → 本地下载拼合 → sha256 比对 NAS 原值
- node1 当前已进入 0.3B 阶段（R7 知识档，ckpt/ 目录 1.29GB 是在训权重，非本份）
