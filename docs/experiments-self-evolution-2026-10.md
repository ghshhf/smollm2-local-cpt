# 自进化三件套：实验启动书（2026-10-18）

> 立书依据：宪法附录B（B4/B5/B6）。本文只定义**三个可跑的实验**，不写愿景。
> 一句话总纲：**自进化 ≡ 推理时进化**——不靠预训练堆容量，靠"更精细到极致的数据"在推理期装进可丢的地方。
> 适用边界：**只在本地跑**。远程 node1 全程预训练，不联网、不接自进化。

---

## 现状（实测，不是估计）

| 件 | 代码量 | 状态 |
|---|---|---|
| In-Place TTT | `tools/ttt_infer.py` 163 行 | 只跑过 0.2B 的"装知识前后"演示；`venv_ttt` 独立环境（transformers 4.57.3） |
| 三元组银行 | **0 行** | 有设计、无实现 |
| 睡眠/做梦 | **0 行** | 纯框架 |
| 上下文记忆层 | **0 行** | 纯框架；A1 原写"记忆层(0.5B)"，B5 已把口径改为"上下文+落盘+检索" |

---

## E-A 三元组银行（唯一今天就能开、不依赖训练栈）

**假设**：把"任务→代码→vid_verify 8 项验收"的通过样本存成可检索库，能让同类任务的首次通过率随库增长而上升。

**最小实现**（一个文件，SQLite，不引框架）：
```
tools/triple_bank.py
  init                     建表
  add --task T --code F --video V     入库前强制跑 vid_verify 8 项，不过则拒收并记失败原因
  query --task "..." --k 3            按任务相似度取 k 条已通过样本（首版用字符 3-gram Jaccard，不上 embedding）
  stats                               入库数 / 通过率 / 各失败项分布
```
表结构：`id, task, code, video_path, checks_json, created_at, task_ngrams`（ngram 存字符串，`LIKE` 即可）。

**判据（falsifier）**：连续 20 个新任务，前 10 个不看库、后 10 个检索 top-3 后生成，
若**首次通过率提升 < 10 个百分点** ⇒ 该库对生成端无用，降级为纯归档（只当 SFT 语料，不当检索源）。

**为什么先做它**：B4 要求"每轮掺外部活信号"，`vid_verify` 是唯一现成的客观判分器；银行不建起来，睡眠/做梦的产物无处沉淀，闭环没有记忆。

---

## E-B 上下文记忆层（用现成 0.2B，不训任何东西）

**假设**：B5 的口径成立——**上下文 + 落盘 + 检索回填**能替代"独立记忆模块"，且效果可测。

**最小实现**（2026-10-18 修订：不再手搓切块与索引，底层直接用 §E-B.0 的 TinyContext）：
```
tools/ctx_memory.py          ← 只是薄封装，索引/检索/embedding 全调 tinycontext
  ingest --dir <对话/笔记目录>        save_memories(kind='episodic')
  ask --q "..." [--memory on|off]     on=recall_memories(top-k) 回填；off=裸问
```
模型用 `backups/0.2B/r6_final_20261016`，推理走训练 venv（`venv`），**检索走 `venv_ttt`，两个 venv 用子进程隔开，绝不互相装包**。

**判据**：取 30 条**答案只存在于我们自己资料里**的问题（owner_corpus 里的内容，模型预训练不可能见过）。
- `memory=off` 准确率基线应接近 0（证明它确实不知道）；
- `memory=on` 若准确率 **< 40%** ⇒ 检索或切块有问题，先修检索，不上 embedding；
- `memory=on` 若 **> 80%** ⇒ B5 口径成立，记忆层不需要新模块，**A1 的"记忆层(0.5B)"直接降级为这件事**。

**为什么值得做**：这是唯一能在不花钱、不动远程的情况下，验证"外置记忆"这条路是否成立。而且它**天然合规 E-7**（全在权重外，可逐条删）。

### E-B.0 现成轮子：本地已装、已跑通的两个开源项目

> 🔴 **换机器装什么，看 [`docs/memory-layer-setup.md`](memory-layer-setup.md)。**
> 一句话：`pip install "tinysuite-context==0.5.1"`，然后用 `BAAI/bge-small-zh-v1.5` 做 embedding。
> 那份手册的每条命令都在干净目录从零跑过，含三个官方文档没写、必踩的坑
> （**Python API 不读环境变量** / `profile` 不走检索 / hf-mirror 直连已失效）。

**不在这一轮手搓 `ctx_memory.py` 之前，先把已有的东西用起来。** 2026-10-17 用户令「手搓加载器全归档，直接用成熟项目」之后，记忆层用的就是下面两个开源项目，全部装在 `venv_ttt`（**绝不进训练 venv**）。

| 项 | TinyContext（主力） | mem0（备选） |
|---|---|---|
| PyPI 包名 | `tinysuite-context` | `mem0ai` |
| 版本 | **0.5.1** | **2.2.1** |
| 上游 | `github.com/TinySuiteHQ/TinyContext` | `github.com/mem0ai/mem0` |
| 作者 | MarcellM01 | Mem0 |
| 许可证 | **MIT** | **Apache-2.0** |
| 一句话 | Token-light local memory with SQLite hybrid retrieval for AI agents | Long-term memory for AI Agents |
| 存储 | 单文件 SQLite（`memories` 表 + `memories_fts` 虚表） | 需向量库 + LLM 抽取，重 |
| 检索 | **FTS5 全文 + sqlite-vec 向量**混合 | 向量为主，靠 LLM 做记忆抽取 |
| 依赖 | `huggingface_hub, numpy, onnxruntime, openai, opentelemetry-api, platformdirs, **sqlite-vec**, tiktoken, tokenizers, urllib3` | 全家桶（含 qdrant-client 等） |
| 安装位置 | `D:/work/_local_ai/train_15b/venv_ttt/Lib/site-packages` | 同左 |
| 状态 | **已跑通，7 条 profile 在库** | 已装，未启用 |

**为什么选 TinyContext 当主力**：

1. **单文件 SQLite**，`memories.db` 一个文件就是全部记忆层——逐条 `DELETE` 即可删除，**天然满足宪法 §4 E-7 的"个人数据可寻址、可删除"**（这是选型的硬门槛，Titans 式写参数方案在这一条上直接出局）。
2. **中英混合检索够用**，且 `sqlite-vec` 与本仓库 venv 里装的 0.1.9 是同一个包，不引入第二套向量栈。
3. **依赖极轻**（10 个），不把训练环境拖下水。
4. mem0 需要 LLM 参与记忆抽取、还要独立向量库，对 16GB 本机是过度工程 —— 留作对照组，不做主线。

**本地落地路径（已存在，可直接查）**：

```
C:/Users/admin/AppData/Local/tinycontext/
├── memories.db                              ← 记忆本体（SQLite，65 KB）
└── models/
    ├── baai-bge-small-zh-v1.5-onnx/         ← 中文 embedding（手工拼装，见下方坑）
    └── bge-small-en-v1.5-onnx/              ← 默认英文，中文弱，不用
```

`memories` 表结构（**这就是 B5「上下文即记忆层」的物理实现**）：

```
id, session_id, content, created_at,
embedding, embedding_model, embedding_dimensions,
superseded_by, superseded_at,        ← 软更新：旧条目可被覆盖且留痕
last_recalled_at, recall_count,      ← 召回统计，可做衰减/淘汰
kind                                 ← 'profile'（每轮常驻） / 'episodic'（按需召回）
```

当前 7 条 profile 全部为 `kind='profile'`，即**每轮无条件注入**（不是检索命中才给）：模型名叫北极星 / 称呼用户为主人 / 0.2B 一手训出 / 绝口不提助手二字 / 持仓 TRX·GLM·RNDR·ZEC / 正在训对话能力 / 喜欢折腾本地小模型和加密货币。

**调用方式**：`tinycontext.save_memories(...)` / `tinycontext.recall_memories(...)`，入参 `MemoryInput(content, kind='profile'|'episodic')`；CLI 有 `mcp / serve / doctor` 三个子命令。

**踩过的坑（照抄会死）**：

| 坑 | 现象 | 解法 |
|---|---|---|
| 默认 embedding 是 `bge-small-en` | 中文 dense 检索错配（"主人叫什么名字"召不回） | 换 `BAAI/bge-small-zh-v1.5` |
| preset 全是英文 | 没有中文可选预设 | 手工拼本地目录 `baai-bge-small-zh-v1.5-onnx/`：`model.onnx`(95.9MB，取 zc277584121 仓库) + tokenizer 4 件(取 BAAI/bge-small-zh-v1.5) |
| `onnx-community` 同名仓库 | 下到 41KB 空壳 | 不用它；custom 模式 patterns 不含 onnx_data 外挂权重 |
| hf-mirror 挂代理 | 308 卡死 | **hf-mirror 必须直连**；出国外才走 3067 |
| `tinysuite-context` 死钉 `huggingface_hub==0.36.0` | 与训练 venv 的 transformers 5.19（需 ≥1.31）**互斥**，手滑装进训练 venv 直接 ImportError | 铁律：**训练 venv 禁装任何非训练包，记忆/推理库一律进 `venv_ttt`** |

**与 E-B 主线的关系**：TinyContext 提供的是**「存 + 检索 + 常驻注入」这一层现成能力**；E-B 要验证的判据（`--memory on/off` 30 题对照）仍然要做，只是**实现从"手搓 FTS5 切块"改成"复用 TinyContext + 换中文 embedding"**，工作量从两天降到半天。

---

## E-C 睡眠/做梦（依赖 A、B 都跑通才能开）

**假设**：把 E-A 库里的三元组，用 In-Place TTT 多遍前向装进 fast weights（睡眠态巩固），能把"查库才能答对"变成"不查库也能答对"，且不污染主干。

**最小实现**（顺序不可颠倒）：
1. 用 E-B 的 30 条题跑 TTT 关 / TTT 开 的对照（复用 `ttt_infer.py --demo`）；
2. 睡眠态：对 E-A 里已通过的样本做 N 遍前向（N=1/3/5 扫），每遍只更新 MLP 末端 `down_proj` 的 fast weights；
3. 做梦态：模型自产任务 → 写代码 → `vid_verify` 判分 → 通过才入 E-A 库。**这一环必须有活外部判分器（B4），否则就是自评自过。**

**判据**：
- 巩固后**主干权重 sha256 不变**（只动 fast weights）——不动即判失败；
- 巩固后不查库准确率提升，且**原始能力不回退**（拿中文 holdout CE 对打，退化 > 2% 判失败）。

**为什么最后做**：它同时依赖 E-A（有料可装）和 E-B（有基线可比），且要等 0.4B 权重回来才有意义——0.2B 上做只是方法学验证。

---

## 排期（按依赖，不按日期）

```
今天    E-A 建库 + 20 题对照（不依赖任何训练）
今天    E-B 索引 + 30 题对照（依赖现成 0.2B 权重）
待定    E-C（等 0.4B 权重落地 + A/B 均过判据）
```

**一条纪律**：三项都过判据之前，不许拿它们的结论去改宪法、改扩容计划。
