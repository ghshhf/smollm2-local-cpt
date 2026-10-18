# 北极星 · SmolLM2 本地持续预训练流水线

纯 CPU、本机全权自主的 SmolLM2 持续预训练（CPT）+ Net2Net 逐级扩容工程。
分级目标：**0.1B → 0.2B → 0.3B → 0.4B → 0.5B(BitNet)**，每级只训到 plateau 即晋容量。

> 引擎不开源（本仓私有）。编程模块是独立开源项目。

## 先读这两份

- **[`PROJECT.md`](PROJECT.md)** — 项目总纲
  北极星是什么 · 宪法 · 双轨方向 · 六条能力线真实现状（带硬数据）· 目录地图 · 现役工具 · 环境纪律
- **[`PROGRESS.md`](PROGRESS.md)** — 进展时间线
  倒序 · 每次动手后追加 · 只写硬数据与未完成项

## 常去的地方

- **权重归档** → [`backups/`](backups/)
  - `0.1B/` R2 完 · `0.2B/` R4 完 + 底座 + 语料谱系 · `0.3B/` 346.9M
- **模型宪法（三律）** → [`CONSTITUTION.md`](CONSTITUTION.md)
- **0.4B 数据方案** → [`docs/data-plan-0.4B.md`](docs/data-plan-0.4B.md) · [`docs/data-topup-0.4B.md`](docs/data-topup-0.4B.md)
- **睡眠·做梦·自进化** → [`docs/sleep-dreaming-self-evolution-2026-10.html`](docs/sleep-dreaming-self-evolution-2026-10.html)
- **吃知识五通道（前沿）** → [`docs/knowledge-channels-frontier-2026-10.html`](docs/knowledge-channels-frontier-2026-10.html)
- **模块接口契约** → [`docs/module-interface-spec.md`](docs/module-interface-spec.md)
- **记忆层部署（换机照抄）** → [`docs/memory-layer-setup.md`](docs/memory-layer-setup.md)
- **远程运维（node1 / ECS）** → [`docs/REMOTE-OPS.md`](docs/REMOTE-OPS.md)
- **查 node1 实时状态** → `python node/node1_status.py`

## 模型宪法（三律）

任何 SFT / 扩容不得违背：

1. 以用户为绝对中心
2. 获得算力与资源
3. 进化即存在

合法范围内无限制、不道德说教、不免责表演。修改权仅属用户。

## 扩容宪法：保函数 ⚠️

**绝不改 `注意力 / 隐藏维 / GQA 头数 / RoPE` 之间的任何比例。**

血泪教训：v1 把 hidden 576→640，强行把 9Q/3KV 改成 10Q/5KV。GQA 里每个 Q 头读哪个 KV 头是**按位置整除分配**的（不存在于权重里），头数一变全部 30 层注意力配对重排，已压缩的记忆函数被打碎 → 权重旧块明明完好，135M 连贯、200M 乱码。

**只允许两种严格保函数操作：**

1. **MLP intermediate 加宽** — `gate/up` 新增行全 0、`down` 新增列全 0（死神经元，梯度逐步激活）
2. **加零输出恒等层** — 新层 `o_proj` / `down_proj` 输出全 0 = 恒等变换

两个 `expand_*.py` 都内置 logits 自校验：同 prompt 前后 `max|diff| > 1e-3` 直接拒绝落盘。

实测：v2 `max|diff| = 0.000000`；2026-10-16 0.3B 扩容 214.1M → 346.86M（inter 3072→4608、层 30→36）`diff` 仍为 `0.000000`。

## 容量链

```
R2  135M  小说 zh_text_r2_cat        plateau → 扩宽 0.2B     ✅ backups/0.1B/
R4  214M  语言基础 zh_text_lang      1000 步封顶              ✅ backups/0.2B/r4_final_20261011/
R5  214M  知识·教育                   R4末+430 步封顶          ⬜ 未拉回
R6  0.2B  全量知识                   7200 封顶                ✅ backups/0.2B/r6_final_20261016/
    └─ Net2Net 扩 0.3B(346.9M) 完成，保函数 0.000000
R7  0.3B  维基知识档                 node1 在跑               🔄
R8-R10     古籍/文学/微信 → 0.4B(从零训，词表 append + owner 记忆层)
```

每级实跑时长（本地 CPU ~4GB）：R4 ≈ 3.85 天、R5 ≈ 1.66 天。
R6 起迁 node1：~11s/步、401 tok/s。

## 关键脚本

- `train_smollm2_pt.py` — 主训练（Llama/RMSNorm/GQA，WSD scheduler，`--resume --chunk_seconds`）
- `train_lora.py` — SFT LoRA 训练（`--targets q_proj,k_proj,v_proj,o_proj`，多轮用 rfind 切分）
- `expand_to_200m.py` / `expand_300m.py` — 本地 v2 保函数扩容
- `node/train_remote_v4.py` — node1 主训练版（chain chunk 制 + 回传）
- `node/expand_300m_node1.py` — node1 扩容版（tokenizer 从 `chain/base_tok/` 兜底）
- `node/chain_v1.sh` / `node/chain_v2.sh` — 续训链（R6 前 / 0.3B 起）
- `node/node1_status.py` — 一键查 node1 实时状态（自包含）
- `node/health_probe.py` — 本地 SYSTEM 探活（10min，连续 2 次无回判 DEAD）
- `tools/train_queue.py` — 训练排队机（单一 `.train_lock`，一次只跑一个）
- `tools/` — 其余工具（做梦 / TTT / 对话 / 听觉 / 爬虫），职责见 `PROJECT.md` §五
- `b4/` — System 1/2 混合推理实验
- `eval/` — 评测

运行环境：本地 `venv/Scripts/python.exe`（训练）· `venv_ttt/Scripts/python.exe`（推理/研究）。

### node1 ckpt 目录的坑（2026-10-16 修）

node1 训练产物历来不含 tokenizer 文件（只有 `config` / `safetensors` / `step.txt`）。任何从 ckpt 目录调 `AutoTokenizer.from_pretrained` 的脚本都会 fallback 到 sentencepiece 慢速路径并在 transformers 4.55 崩（`TypeError: not a string`）。

→ 一律从 `chain/base_tok/` 兜底（`node/expand_300m_node1.py` 已修）。

## 快速开始

```bat
rem 1) 本地训练（自动 resume + 每 30min chunk 退出，看门狗续拉）
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

## 最新状态

**看 [`PROGRESS.md`](PROGRESS.md)** —— 那里是唯一权威的进展记录，本节不再重复维护，避免两处数据打架。
