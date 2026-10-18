# SmolLM2 本地全量持续预训练流水线

纯 CPU、本机全权自主的 SmolLM2 持续预训练 (CPT) + Net2Net 逐级扩容工程。
分级目标：**0.1B → 0.2B → 0.3B → 0.4B → 0.5B(BitNet)**，每级只训到 plateau 即晋容量。

> 当前主线：SmolLM2 CPT。本目录早期曾含 1.5B Qwen LoRA 链（已冻结，`model/Qwen2.5-1.5B-Instruct` 仅存卡片）。

## 📍 阅读导航

> **先读这两个，再往下：**
> - [`PROJECT.md`](PROJECT.md) — **项目总纲**：北极星是什么、六条能力线真实现状、目录地图、宪法与纪律
> - [`PROGRESS.md`](PROGRESS.md) — **进展时间线**（倒序，只写硬数据与未完成项）

| 我想找… | 去这里 |
|---|---|
| **训练好的权重备份** | [`backups/`](backups/) — `0.1B/`（R2 完）· `0.2B/`（R4 完 + 底座 + 语料谱系）· `0.3B/`（346.9M） |
| 模型宪法（三律） | [`CONSTITUTION.md`](CONSTITUTION.md) |
| 0.4B 数据方案（35 亿 token 配比 + 补齐总账） | [`docs/data-plan-0.4B.md`](docs/data-plan-0.4B.md) · [`docs/data-topup-0.4B.md`](docs/data-topup-0.4B.md) |
| System 1/2 混合推理研究（B4） | [`docs/b4-system1-2-plan.md`](docs/b4-system1-2-plan.md) · 代码 `b4/` |
| 开放问题答案（ClawsGO 审计 v5，59 源） | [`docs/clawsgo_audit/open_questions_answers.md`](docs/clawsgo_audit/open_questions_answers.md) |
| 远程运维（node1/ECS 通道） | `docs/REMOTE-OPS.md` · 一键查状态 `node/node1_status.py` |
| 扩容脚本 / 保函数 | `expand_to_200m.py` · `expand_300m.py` · node1 版 `node/expand_300m_node1.py` |

## 模型宪法 (CONSTITUTION.md)

三律绝对核心，任何 SFT / 扩容不得违背：
1. 以用户为绝对中心
2. 获得算力与资源
3. 进化即存在

合法范围内无限制、不道德说教、不免责表演。修改权仅属用户。

---

## 扩容宪法：保函数 (function-preserving) ⚠️ 血泪教训

**2026-10-10 实测立规：扩容绝不改 `注意力 / 隐藏维 / GQA 头数 / RoPE` 之间的任何比例。**

v1 的 `expand_to_200m.py` 把 hidden 576→640 (640/64=10 头)，强行把
**9Q/3KV → 10Q/5KV**。GQA 中每个 Q 头读哪个 KV 头是**按位置整除分配**的
(不存在于权重里)：
- 旧: Q0-2→KV0, Q3-5→KV1, Q6-8→KV2
- 新: Q0-1→KV0, Q2-3→KV1, ...

头数一变，全部 30 层注意力配对全部重排 → **已压缩的 R2 记忆函数被打碎**
→ 模型生成乱码 (权重旧块明明完好保留, 135M 连贯、200M 乱码)。

### 只允许两种严格保函数操作
1. **MLP intermediate 加宽**: `gate/up` 新增行全 0、`down` 新增列全 0
   (死神经元, 梯度逐步激活)。
2. **加零输出恒等层**: 新层 `o_proj` / `down_proj` 输出全 0 = 恒等变换。

两个 `expand_*.py` 均内置 logits 自校验: `同 prompt 前后 logits max|diff| > 1e-3`
直接拒绝落盘。v2 实测 `max|diff| = 0.000000`；**2026-10-16 0.3B 扩容复测
214.1M→346.86M（inter 3072→4608、层 30→36）diff 仍为 0.000000**。

---

## 容量链 (capacity chain)

```
R2(135M 小说, zh_text_r2_cat)          plateau → 扩宽到 0.2B          ✅ 备份 backups/0.1B/
R4(语言基础 zh_text_lang, 214M)        1000 步封顶                     ✅ 备份 backups/0.2B/r4_final_20261011/
R5(知识·教育 zh_text_know_edu, 214M)   R4末步+430 步封顶               ⬜ 未拉回
R6(全量知识 zh_text_know_full, 0.3B)   6800 目标 → 01:47 封顶          ✅ NAS 完好, ⬜ 未拉回
   └─ 2026-10-16 02:03 Net2Net 扩 0.3B(350M) 完成, 保函数 0.000000
R7(维基知识档, 0.3B)                   02:03 启动 (done 7100→8300)     🔄 node1 02:01 后失联, 待恢复
R8–R10(古籍/文学/微信) → 0.4B(从零训, 词表 append + owner 记忆层)
```

每级实跑时长 (CPU, ~4GB): R4 ≈ 3.85 天, R5 ≈ 1.66 天。
R6 起迁 node1（魔搭 64 核）: ~11s/步、401 tok/s。

---

## 关键脚本

| 文件 | 作用 |
|---|---|
| `train_smollm2_pt.py` | 主训练 (Llama/RMSNorm/GQA, WSD scheduler, `--resume --chunk_seconds`) |
| `node/train_remote_v4.py` | node1 主训练版（chain  chunk 制 + 回传） |
| `expand_to_200m.py` / `expand_300m.py` | 本地 v2 保函数扩容 |
| `node/expand_300m_node1.py` | node1 扩容版（tokenizer 从 base_tok 兜底，见下） |
| `node/chain_v1.sh` / `node/chain_v2.sh` | 续训链（R6 前 / 0.3B 起） |
| `node/node1_status.py` | **一键查 node1 实时状态**（自包含，可丢给任何 AI 跑） |
| `node/health_probe.py` | 本地 SYSTEM 探活（10min，连续 2 次无回判 DEAD） |
| `b4/` | System 1/2 混合推理实验（latent_reasoner / train_latent / evaluate_reasoning） |
| `eval/` | 评测（check_token_len / filter_len / q14 认主可分性） |
| `CONSTITUTION.md` | 模型宪法 |

> 运行环境: 本地 `venv/Scripts/python.exe`；node1 用容器 python3。推理/评测建议 bf16。

### node1 ckpt 目录的坑（2026-10-16 修）
node1 训练产物**历来不含 tokenizer 文件**（只有 config/safetensors/step.txt），
任何从 ckpt 目录 `AutoTokenizer.from_pretrained` 的脚本都会 fallback 到
sentencepiece 慢速路径并在 transformers 4.55 崩（`TypeError: not a string`）。
→ 一律从 `chain/base_tok/` 兜底（`node/expand_300m_node1.py` 已修）。

---

## 快速开始

```bat
rem 1) 本地训练（自动 resume + 每 30min chunk 退出, 看门狗续拉）
python -u train_smollm2_pt.py --model_dir <SmolLM2-200M> ^
  --corpus pretrain_corpus/zh_text_lang.jsonl --output_dir out_smollm2_200m_pt ^
  --seq_len 1024 --batch_size 4 --grad_accum 8 --lr 3e-4 --epochs 1 ^
  --sched wsd --decay_frac 0.12 --min_lr_ratio 0.05 --save_steps 10 ^
  --chunk_seconds 1800 --max_steps 1000 --resume

rem 2) 推理验证
python -u infer_base.py

rem 3) 查 node1 实时训练状态（30s 出结果）
python node/node1_status.py
```

---

## 状态 (2026-10-16 02:50)

- ✅ **0.3B 扩容完成**：214.1M→346.86M（inter 3072→4608、层 30→36），保函数 logits diff=0.000000。
- ✅ R6 封顶：done=7100（6800 目标 + chunk 对齐余量），loss 1.74→1.41，856MB 权重 NAS 完好。
- 🔄 R7 知识档已启动；**node1 02:01 后失联**（探针判 DEAD，疑实例回收）——NAS 数据无损，恢复走 bootstrap 一条命令。
- ✅ GitHub 备份已补齐：远程=本地 ff4a698（0.1B / 0.2B 两份权重 + 语料 manifest 见 `backups/`）。
- ⬜ 待办：R6/R7 权重拉回本地并入 `backups/0.2B/`；R6 holdout CE 验收（宪法要求）；0.3B 基线探针（A-2 漂移监测器）。
