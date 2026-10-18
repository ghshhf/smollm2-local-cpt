# 进展时间线（Progress Log）

> 规则：**倒序，最新在最上**。每次动手后追加一段，只写**硬数据**与**未完成项**，不写感想。
> 交付物写清路径，让下一个人（或未来的我）能直接接着干。
> 配套：项目总纲见 `PROJECT.md`。

---

## 2026-10-18 12:00-12:40 · 做梦线：精英档案多样性修复（塌桶 → 6 桶）

### 病灶（实测数据，非推测）

精英档案设计成 MAP-Elites，实际只产出 **2 个桶 / 2 条精英**，而银行里有 **9 条**过闸三元组：

```
UI可视化|func+scipy    8 条 -> [3,4,5,6,7,8,9,10]   ← 8 条全塌进 1 个桶
其他|bare              1 条 -> [2]
```

根因两条：
1. `task_domain()` 的域正则只认到「动画」关键词，12 个模板全被归成「UI可视化」；
2. `code_sigs()` 是布尔粗签名（有没有 `def`/有没有 `matplotlib`），8 份 matplotlib 代码完全相同。
   → 第 1 条以 q=0.80 进桶后，其余 7 条同桶同签名且质量不更高，全被 `skip`。

### 修了什么

**1) `tools/dream_elite.py` 桶划分重构：单一任务域 → 技能轴 × 技术轴**
- 技能轴 `skill` = 这个任务在练什么能力（三角波/衰减振荡/参数曲线/分形/场热力图…共 19 类正则）
- 技术轴 `tech` = 代码怎么画出来的（画法 `viz_kind` 9 类 × 驱动 `anim_driver` 4 类）
- 两轴正交，才能区分「同一能力不同实现」与「不同能力同样实现」
- 保留 `infer_skill()` 正则兜底，供外部/临时任务用；生成端知道答案就显式传 `--skill`

**2) 三重准入闸门**（旧版只有「同桶同签名 + 桶容量」两道）
- 近重复闸门：同桶内代码 3-gram Jaccard ≥ `DUP_JACCARD=0.82` 判为重复解，只有质量更高才替换
  （复用 `triple_bank.ngrams/jaccard`，不另造轮子）
- 桶容量闸门：每桶 top-3，超额且不比桶内最差高 → 丢弃
- 新桶直通：落进未见过的桶直接收（新能力，质量低也有研究价值）

**3) 修 `dream_loop.retrieve_examples()` 潜伏 bug（RSI 燃料链）**
- 旧实现只 grep `code=<路径>` 再 `open()` 磁盘：`data/dreams` 一被清理，few-shot 静默失效；
  而且读到的是可能被改过的文件，违背银行「存快照不存路径」的设计原则
- 改为直接解析库里 `code_text` 快照文本（`---` 块），实测检索到 2 条 / 531+504 字 ✅
- 当前 9 条 `code_path` 都还在，所以是**潜伏** bug，不是已爆的 bug

**4) 单一写入路径（去掉重复归档）**
- 精英档案本来就由 `triple_bank add` 顺带写入；新增的 `dream_loop` 归档会双写
- 改为：`dream_loop.accept(skill=)` → `triple_bank add --skill X` → 精英档案按显式技能分桶
- `dream_loop.smoke()` 里手工归档的老代码一并清掉

**5) `tools/dream_autorun.py` 模板从 12 个扩到 15 个 / 14 类技能**
- 每个模板显式标注 skill（生成端不该让正则去猜）
- 新增跨域任务：冒泡排序可视化（排序搜索）、康威生命游戏（生命游戏）、抛体反弹（刚体动画）
- 原模板全是数学曲线 → 技能轴太单一，是塌桶的上游原因

### 实测结果

| 项 | 结果 |
|---|---|
| 桶数（银行 9 条回算） | **2 → 6**；收入精英 7 条（踢掉 2 条逐字节重复 #7/#8） |
| 自检 `tools/dream_elite_selftest.py` | **24/24 通过**（8 闸门用例 + 桶分离度 + 15 技能标签一致性），改桶规则后必跑 |
| 技能标签一致性 | **15/15**（显式 skill 与正则推断零偏差；技能轴覆盖 14 类） |
| 近重复阈值校准 | #4 vs #7 J=**1.000**、#6 vs #8 J=**1.000**（都被拦）；#3 sin vs #5 cos J=**0.802**（刚好放过）→ 0.82 合理 |

**顺带发现（值得记）**：#6「正弦+余弦叠加」与 #8「多曲线对比」**任务文本不同但生成代码逐字节相同**
（J=1.000）—— 说明当时那版 coder 根本没读懂任务差异，只是在复读同一份模板。

### 本轮其他状态
- **p2 编程 LoRA 训练收尾**：`out_lora_code_qwen`，732/732 步 / 1 epoch / **18150s（5.04h）**，
  末步 loss **1.462**、entropy 1.516、mean_token_accuracy 0.699；checkpoint 100~700 每 100 步一存
- 做梦守护实测有效：日志两次 `[skip] 检测到训练在跑 (pid=8820 job=p2)`，第一定律没破

### 未完成 / 待拍板
- ⏳ **DreamLoop 计划任务仍指向 `out_lora_code_qwen/checkpoint-300`**，而 p2 已训到 732。
  用旧 checkpoint 等于白训 400 步；但 6ep 级训练有「指令能力被冲成复读」的前科（R7-R10 教训），
  换哪个 checkpoint 必须先用 `data/code_holdout_v1.jsonl` 20 题做 code_eval 对照，**不能拍脑袋换**
- ⏳ 精英档案目前**只写不读**：还没参与 `retrieve_examples` 的 few-shot 组装。
  MAP-Elites 的价值在「回放保多样」，下一步应让检索同时从精英档案抽 1 条「不同桶」示例
- ⏳ 修复后的桶规则尚未经过真实做梦一轮验证（等 CodeLoraP2 后置评测释放训练锁）

---


## 2026-10-18 · 做梦闭环跑通 + 对话语料合成 + 听觉线起步

### 1) 做梦（RSI）闭环：从设计态 → 可跑态 🟢

**新建**
- `tools/dream_loop.py` — 做梦编排器（v2）：记忆检索(few-shot) → 生成 → 沙箱执行 → `vid_verify` 闸门 → 入库 + rejects 回灌
- `tools/dream_elite.py` — MAP-Elites 精英档案（quality-diversity 去重）
- `tools/dream_autorun.py` — 周期自转调度器（原子锁 + 自管日志）
- `tools/weak_to_strong.py` — 弱到强升级骨架（LEACH 思路，教师接口可插拔）

**改造**
- `tools/vid_verify.py` → v2：每项 0~1 partial credit + `--min-quality`；**修致命误杀**——"有音轨"原为硬检查，导致所有无声的可视化 mp4 全部判死，改软检查 + `--require-audio`
- `tools/ttt_infer.py` — 底座参数化（默认 0.2B，可传 0.3B）；修 `tie_word_embeddings=True` 时 `lm_head.weight` 缺失的误报
- `tools/triple_bank.py` — 入库后顺带写精英档案

**实测结果**
| 项 | 结果 |
|---|---|
| 0.3B 上 latent 适配（ttt_latent） | 干净加载 346.9M / unexpected=0，4 步末损 2.9052，生成正常 ✅ |
| 冒烟（无模型） | exec → 视频验收 ACCEPT → 入库 → 精英去重 skip 正确 ✅ |
| 播种 | 手放正弦波动画示例入银行 #2，作 few-shot 自举起点 |
| **0.3B 做梦** | ❌ 全败：生成 `代码块` 三个字 / 中文废话。根因=底座无代码预训练 |
| **0.3B + 代码 LoRA（官方 code_eval）** | 代码块 33%、正常收尾 0%、**strict 0/3**、宽松 0/3 |
| **coder 底座做梦**（Qwen2.5-0.5B + `out_lora_code_qwen/checkpoint-300`） | 首轮 **1/3** → 提示加固后 **6/6 全过**，银行 #3→#9 |

**结论（重要）**：做梦*管线*完全成立；**0.3B 北极星无法自己写代码**（底座能力边界，非 bug）。
RSI = 达标后的加速器，不是从 0 到达标的捷径。

### 2) 中文对话：语料 404 → 3000 条 🟡

**病灶（2026-10-18 实测，非推测）**
- in-set completion CE = **1.578**；达标(<0.60) 仅 **14/439 = 3.2%** ⇒ 只学会抄写/套模板
- 过拟合探针 **PASS**（最短样本近 40 步均 0.2845）⇒ 管道无 bug
- 根因：数据量差一个数量级（404 条×4ep，每条答案只看 4 次）；词表成本 2.48 token/汉字放大 2.5×；EOS 弱
- 另：`models/chat_lora_v4_port03b` 的 target 仍是旧 7 模块（含 gate/up/down），跨代不可靠，**弃用，须在 0.3B 上重训 q/k/v/o**

**产出**
- `data/chat_seeds/seeds_v7{,b,c,d}.py` — 教师手写种子 **333 组 / 13 域**（不是模板套壳）
- `tools/synth_chat_data_v7.py` — 组合扩展器
- `data/chat_sft_v7.jsonl` — **3000 条，完整 prompt 唯一 3000/3000，同问多答冲突 0**
  - 单轮 692 / 多轮 2308；唯一问题 692
  - 域分布：tech 367、identity 317、greeting 313、emotion 265、work 254、life 233、knowledge 228、study 216、ability 214、owner 200、opinion 198、clarify 155、boundary 40
  - token：min 18 / p50 105 / p90 133 / p99 158 / max 182 / 总 296,154（均值 98.7）
- `data/chat_holdout_v7.jsonl` — 真盲测 30 题（训练集完全没有的新问法 + 关键点）
- `tools/chat_blind_eval.py` — 盲测执行器

**过程中修掉的三个真问题**
1. 答非问错配（把"话题相近"当"同义问法"共用答案池）→ 拆成独立组
2. 同问多答冲突（首版 1012 条单轮全是一问多答，违反项目零冲突规范）→ 改索引配对一问一答
3. **多轮 loss 污染**：`train_lora.py` 用 `find`（第一个 assistant）切 prompt，多轮样本会把第二轮 user 提问算进 loss = 教模型续写用户输入 → 改 `rfind`，实测 2308 条多轮**零污染**

**训练**：已提交 `chat_v7_03b` 进 `train_queue`（0.3B + q/k/v/o LoRA，4 ep，`--max_seq_length 384`）

### 3) 听觉线：0.3B 起步 🟡

| 环节 | 状态 |
|---|---|
| Whisper-small encoder | ✅ 已下 922MB（88.2M 参数，d_model=768） |
| AISHELL-1 | ✅ 24 说话人解包，**8197 对 / 10.6 小时**（中位 4.31s、22 字） |
| `tools/audio_attach.py` | ✅ 新写：Whisper → pool → MLPProjector(768→576) → **0.3B**，early fusion，三阶段 |
| 真盲测切分 | ✅ 按说话人留出 S0024/S0025（训练 7507 / 验证 690，人声完全没见过） |
| 特征预计算 | 🔄 后台跑（1500+150 条，单条约 4s，慢在 transformers 5.19 强制 30s padding） |

已对齐 VLM 踩过的坑：`fc2` 必须 `std=0.01` 微小随机初始化（零初始化会让 RMSNorm 反向炸 nan）。

### 4) 识图线：冻结（按用户指示）
- v5 盲测灾难幻觉；v6 val_loss 1.5552 但输出仍是弱视觉条件化的短语堆砌
- v6 实为 **0.2B(214M)** 血统，未升 0.3B
- 已知部署坑：infer 须显式 `--init <snapshot>/projector.pt`（否则默认加载 align 的 projector）；v6 projector 实为 `ps=1`

### 5) 情报线
- 论文爬虫 122 篇去重 / 14 桶分类 → `docs/frontier-crawl-report-2026-10-18.html`
- 物理拆分 `data/frontier/by_category/`（13 个 jsonl + `_index.json`）

### 6) 远程算力恢复 + 音频线外迁（11:00-12:00）

**背景**：用户重开魔搭免费 CPU 实例（**纯 CPU，无 GPU**）。实例重启后训练链未自动恢复。

**诊断到的事实**
- node1 cgroup **8 核**、内存 30GB、NAS 977T；load 0.10（完全空闲）
- `done_steps=7200`（0.2B 封顶值）、heartbeat 停在 02:03、**无任何训练进程**
- 0.4B 的真身在 `chain/ckpt_gpu/run_live`（03:52 最后写，实例重启后停摆），`ladder_budget=04B=3B`
- `chain.log` 报 `base_tok/special_tokens_map.json` 缺失

**已修**
- 补 `base_tok/special_tokens_map.json`（86B，bos/eos/unk = `<|endoftext|>`，与 tokenizer_config 一致）
- `tools/audio_attach.py`：预计算**每 500 条原子落盘**（`torch.save` 到 `.tmp` 再 `os.replace`），中断可续，不再白跑

**🔴 两条硬教训**
1. **node1 只执行单行命令** —— 多行 cmd 静默不执行（这正是前面多次"超时"的真因）。自定义 cmd 必须 `{ ...; } 2>&1 | curl ...` 分组，否则只有最后一条进管道。
2. **本机 CPU 被抢会让训练掉速 10 倍** —— 音频预计算与 p2 同跑时，p2 从 23s/it 掉到 **240s/it**；停掉后立刻回到 21.4s/it。**一次只跑一个重活**。

**音频线外迁（node1 辅助，本机不占 CPU）**

| 环节 | 实况 |
|---|---|
| whisper-small | node1 自己从 hf-mirror 下载（内网，不占本机带宽） |
| 数据打包 | wav 1.2GB + pairs → tar+gzip -1 → **883MB / 45 片**（split 20MB） |
| 上传 | ECS 信箱实测 **5.0MB/s 上行 / 1.73MB/s 下行**，45 片全部 code=201 |
| 传输工具 | 新增 `node/push_aud.py`（分片带重试）、`node/recv_aud.sh`（拉取+解包+路径重写，幂等）|
| node1 接收 | 拉满 45 片 → 解包到 `data/audio/` → 重写 pairs 的 Windows 路径 → 自动接 `--precompute` |

**本机同时**：p2 恢复到 21.4s/it（672/732），队列 `chat_of03b` → `chat_v7_03b` 待接力。

### 未完成
- [ ] `chat_v7_03b` 训练 + in-set CE 验收（<0.60 才算学会）
- [ ] 0.3B 中位长度过拟合探针
- [ ] 听觉特征预计算 → align 训练 → CER 盲测
- [ ] `dream_autorun.py` 注册 schtasks 常驻（注册时 `RepetitionDuration MaxValue` 报错待修）
- [ ] p2 管线（Qwen0.5B 代码 LoRA）step2 remap 1.5B + step3 四向评测收尾

---

## 2026-10-17 · 0.3B 权重归档 + 模块接口契约

- 0.3B 权重（`polaris-03b_20261018`，1.39GB fp32，sha256 `90f2f64b…30b70`）分片 14 份入私有仓
- 提交 `a5c85a4`：`backups/0.3B/` 含 14 分片 + `reassemble.py`（幂等 + sha256 自校验）+ provenance
- 坑：ModelScope 遗留 `.gitattributes` 指向 git-lfs，本机无 lfs 导致 commit 失败 → 删除
- 定模块接口契约（`docs/module-interface-spec.md`）：禁止硬编码主干维度，只许接恒定维 q/k/v/o + 残差流

---

## 2026-10-16 · 0.3B 扩容 + R6 封顶 + 记忆层选型

- 0.3B 扩容完成：214.1M → **346.86M**（inter 3072→4608、层 30→36），保函数 logits `diff = 0.000000`
- R6 封顶 done=7100，loss 1.74→1.41
- 0.2B R6 权重归档 `backups/0.2B/r6_final_20261016/`（9 分片 + 拼回器 + provenance），提交 `c3be988`
- 记忆层选型：**TinyContext**（PyPI `tinysuite-context` 0.5.1，MIT，SQLite+FTS5+sqlite-vec，天然满足 E-7 逐条可删）为主力；`mem0` 仅对照
- E1/P3 终止裁决（不重跑）：必须继承的系统里，控制点从输入端移到输出端
