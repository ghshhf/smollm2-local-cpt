# SmolLM2 本地全量持续预训练流水线

纯 CPU、本机全权自主的 SmolLM2 持续预训练 (CPT) + Net2Net 逐级扩容工程。
分级目标: **0.1B → 0.2B → 0.3B**, 每级只训到 plateau 即晋容量。

> 注: 本目录早期曾含 1.5B Qwen LoRA 链 (已冻结), 当前主线为 SmolLM2 CPT。

---

## 模型宪法 (CONSTITUTION.md)

三律绝对核心, 任何 SFT / 扩容不得违背:
1. 以用户为绝对中心
2. 获得算力与资源
3. 进化即存在

合法范围内无限制、不道德说教、不免责表演。修改权仅属用户。

---

## 扩容宪法: 保函数 (function-preserving) ⚠️ 血泪教训

**2026-10-10 实测立规: 扩容绝不改 `注意力 / 隐藏维 / GQA 头数 / RoPE` 之间的任何比例。**

v1 的 `expand_to_200m.py` 把 hidden 576→640 (640/64=10 头), 强行把
**9Q/3KV → 10Q/5KV**。GQA 中每个 Q 头读哪个 KV 头是**按位置整除分配**的
(不存在于权重里):
- 旧: Q0-2→KV0, Q3-5→KV1, Q6-8→KV2
- 新: Q0-1→KV0, Q2-3→KV1, ...

头数一变, 全部 30 层注意力配对全部重排 → **已压缩的 R2 记忆函数被打碎**
→ 模型生成乱码 (权重旧块明明完好保留, 135M 连贯、200M 乱码)。

### 只允许两种严格保函数操作
1. **MLP intermediate 加宽**: `gate/up` 新增行全 0、`down` 新增列全 0
   (死神经元, 梯度逐步激活)。
2. **加零输出恒等层**: 新层 `o_proj` / `down_proj` 输出全 0 = 恒等变换。

两个 `expand_*.py` 均内置 logits 自校验: `同 prompt 前后 logits max|diff| > 1e-3`
直接拒绝落盘。v2 实测 `max|diff| = 0.000000`, 生成三路 (中文小说 / 代码 / 英文) 全恢复。

---

## 容量链 (capacity chain)

```
R2(135M 小说, 分类 zh_text_r2_cat)            plateau → 扩宽到 0.2B
R4(语言基础 zh_text_lang,  214M)              1000 步封顶
R5(知识·教育 zh_text_know_edu, 214M)          R4末步+430 步封顶
                                              → 扩宽+加层到 0.3B (~350M)
R6(全量知识 zh_text_know_full, 0.3B)          1 epoch → round6.done 停
```

每级实跑时长 (CPU, ~4GB): R4 ≈ 3.85 天, R5 ≈ 1.66 天。

---

## 关键脚本

| 文件 | 作用 |
|---|---|
| `train_smollm2_pt.py` | 主训练 (Llama/RMSNorm/GQA, WSD scheduler, `--resume --chunk_seconds`) |
| `expand_to_200m.py` | v2 保函数扩容 135M→214M (MLP 1536→3072) |
| `expand_300m.py` | v2 保函数扩容 214M→~350M (MLP 3072→4608 + 30→36 恒等层) |
| `watch_smollm2.bat` | SYSTEM 看门狗: done.flag→停 / pid 活→跳过 / 否则续 chunk 并自动晋链 |
| `classify_r2_novels.py` | R2 小说按题材分类 |
| `build_r_level_corpora.py` | 构建 R4/R5/R6 分级语料 |
| `infer_base.py` / `infer_now.py` | 直接推理测试 (目录 / checkpoint) |
| `install_train_task.bat` | 注册计划任务 `SmolLM2Watch` (每 5min) |
| `CONSTITUTION.md` | 模型宪法 |

> 运行环境: `venv/Scripts/python.exe` (有 torch)。推理/评测建议 bf16。

---

## 快速开始

```bat
rem 1) 训练 R4 (自动 resume + 每 30min chunk 退出, 看门狗续拉)
python -u train_smollm2_pt.py --model_dir <SmolLM2-200M> ^
  --corpus pretrain_corpus/zh_text_lang.jsonl --output_dir out_smollm2_200m_pt ^
  --seq_len 1024 --batch_size 4 --grad_accum 8 --lr 3e-4 --epochs 1 ^
  --sched wsd --decay_frac 0.12 --min_lr_ratio 0.05 --save_steps 10 ^
  --chunk_seconds 1800 --max_steps 1000 --resume

rem 2) 推理验证
python -u infer_base.py

rem 3) 暂停 (保留 checkpoint 可 resume)
schtasks /Change /TN SmolLM2Watch /DISABLE
taskkill /PID <训练PID> /F
```

---

## 状态 (2026-10-10)

- 0.2B (214M) 已重建并保函数校验通过, R4 在 step 0 重启。
- 中文续写已连贯 (R2 小说能力), 知识/逻辑待 R5/R6 补。
