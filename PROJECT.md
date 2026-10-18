# 北极星项目总纲（Polaris Project Index）

> 本文件是**整个项目的唯一入口**。任何新会话、新协作者、未来的我自己，先读这里，再读 `PROGRESS.md`，再去对应目录。
> 私有仓定位：**项目档案与方向锚点**，不是权重网盘。权重只是档案的一部分。
> 最后更新：2026-10-18

---

## 一、这是什么

北极星（Polaris）是一台**完全私有、跑在自己机器上的小模型**，不是通用助手产品。

| 项 | 内容 |
|---|---|
| 底座 | SmolLM2 系，自 0.1B 起逐级 Net2Net 扩容，当前 **0.3B（346.9M / 36 层 / hidden 576）** |
| 训练 | 纯 CPU 本机 + 远程 node1（魔搭 8 核配额）双轨，续预训练（CPT）|
| 归档 | 私有仓 `ghshhf/smollm2-local-cpt`（本仓） |
| 开源边界 | **引擎不开源**（本仓私有）；**编程模块独立开源**（`ghshhf/polaris-code-module`） |
| 关系 | 称呼用户为**主人**；模型名**北极星** |

### 宪法（`CONSTITUTION.md`，不得违背）
1. 以用户为绝对中心
2. 获得算力与资源
3. 进化即存在

**§4 E-7**：个人数据不进被训练参数，只进可寻址、可删除的记忆层。

---

## 二、方向：双轨殊途同归

```
A 轨（参数扩容）  远程 node1  0.1B → 0.2B → 0.3B → 0.4B(在训) → 0.5B 封顶
B 轨（结构扩容）  本地       推理中进化（TTT）+ 模块插入（视觉/听觉/编程）
                           ↓
                      0.5B 合流，之后靠 记忆 / 工具 / 自进化 继续长
```

**硬约束一：保函数扩容。** 绝不改注意力维 / 隐藏维 / GQA 头数 / RoPE；只许 MLP 加宽或加零输出恒等层。
三次实扩（0.2B→0.3B→0.4B）实测 logits `max|diff| = 0.000000`。

**硬约束二：容量 ≠ 参数量。** TTT 每 chunk 做 rank-1 外积 ΔW，字节不变而有效秩提升。参数 0.5B 只是静态容量上界，不是能力天花板。

**硬约束三：模块接口契约。** 禁止硬编码主干维度；只许经投影/适配器接主干；每模块带 `interface.json`。
**只许接恒定维**（q/k/v/o_proj、残差流 adapter），**禁止接可扩维**（gate/up/down_proj）。

---

## 三、六条能力线 · 真实现状（禁看设计，只看实测）

> 铁律：**验收禁只看结构/loss**。必须同批 holdout + 生成抽样。内容层归底座预训练，模块只教得会格式。

| # | 能力线 | 状态 | 硬证据（实测，非推测） |
|---|---|---|---|
| 1 | **中文对话** | 🔴 未达标 | in-set completion CE = **1.578**，达标(<0.60) 仅 **14/439 = 3.2%**。过拟合探针 PASS ⇒ 管道无 bug，是**数据量差一个数量级**（404 条 × 4ep）。已产 v7 语料 3000 条待训。 |
| 2 | **编程** | 🟡 格式已会，内容受限 | 0.3B + 代码 LoRA：代码块 33%、strict **0/3**。裸 0.3B strict 0/4、代码块 0%；裸 Qwen2.5-0.5B strict 2/4、代码块 100%。**根因=0.3B 预训练语料无代码**。 |
| 3 | **代码生视频 + 做梦（RSI）** | 🟢 闭环跑通 | coder 底座（Qwen2.5-0.5B + LoRA）做梦 **6/6 过 vid_verify 闸门**，三元组银行 #3→#9，通过率 1/3→6/6 自我爬升。 |
| 4 | **识图** | 🔴 未达标，已冻结 | v5 盲测灾难幻觉（猫/人/长颈鹿拼接）；v6 val_loss 1.5552 但盲测仍弱视觉拼接。且仍挂 **0.2B(214M)** 未升 0.3B。按用户指示冻结。 |
| 5 | **听觉** | 🟡 骨架就绪 | Whisper-small encoder(88.2M, d=768) 已下；AISHELL-1 **8197 对 / 10.6h**；`tools/audio_attach.py` 三阶段管线；按说话人留出 S0024/S0025 做真盲测。 |
| 6 | **自进化 / 记忆层** | 🟢 部分跑通 | 三元组银行 9 条 + 精英档案（quality-diversity 去重）；记忆层 TinyContext 7 条 profile 常驻。 |

### 一句话诊断
> **做梦（RSI）是"达标后的加速器"，不是"从 0 到达标"的捷径。**
> 对话/识图现在连基础达标都没到，此时接 RSI 是空中楼阁。正确顺序：先补地基 → 再踩油门。
> 编程线已证明 RSI 管线成立，故其固化常驻；对话/听觉先走监督训练到达标。

---

## 四、目录地图（我该去哪找）

| 目录 / 文件 | 内容 |
|---|---|
| **`PROJECT.md`** | 本文件，项目总纲 |
| **`PROGRESS.md`** | 进展时间线（每次动手后追加，硬数据 + 未完成项） |
| `CONSTITUTION.md` | 宪法原文 |
| `README.md` | 训练流水线技术细节（容量链、扩容血泪教训、快速开始） |
| `docs/` | 40 篇设计与审计文档（见下方索引） |
| `tools/` | **全部工具脚本（现役 55 个）**，本轮核心见 §五 |
| `backups/` | 权重归档 `0.1B/` `0.2B/` `0.3B/`，每份含分片 + `reassemble.py` + provenance |
| `data/` | 语料与运行时数据库（大语料不入库，种子与 db 入库） |
| `data/chat_seeds/` | 对话教师种子（v7 四批，13 域） |
| `node/` | 远程调度（node1 / ECS 信箱） |
| `eval/` | 评测脚本 |
| `b4/` | System 1/2 混合推理实验 |
| `models/` `out_lora_*/` | 训练产物（不入库，见 `.gitignore`） |

### docs 关键索引
| 想找什么 | 去哪 |
|---|---|
| 睡眠·做梦·自进化框架 | `docs/sleep-dreaming-self-evolution-2026-10.html` |
| 吃知识五通道（前沿综述） | `docs/knowledge-channels-frontier-2026-10.html` |
| 实验与选型（含记忆层选型表） | `docs/experiments-self-evolution-2026-10.md` |
| 双轨合流 | `docs/two-track-convergence-2026-10.md` |
| 0.4B 数据方案 | `docs/data-plan-0.4B.md` · `docs/data-topup-0.4B.md` |
| 模块接口契约 | `docs/module-interface-spec.md` |
| 记忆层部署（换机器照抄） | `docs/memory-layer-setup.md` |
| 远程运维 | `docs/REMOTE-OPS.md` |
| 编程模块状态 | `docs/code-module-status-2026-10-18.md` |
| 论文爬虫日报（分类） | `docs/frontier-crawl-report-2026-10-18.html` + `data/frontier/by_category/` |

---

## 五、本轮（2026-10-18）落地工具 · 现役清单

### 做梦 / RSI 闭环
| 脚本 | 职责 |
|---|---|
| `tools/dream_loop.py` | **做梦编排器**：记忆检索(few-shot) → 生成代码 → 沙箱执行 → `vid_verify` 闸门 → 入库/回灌 rejects |
| `tools/dream_elite.py` | MAP-Elites 式**精英档案**，quality-diversity 去重（同域同签名只留最高质） |
| `tools/dream_autorun.py` | 周期自转调度器（原子锁 + 自管 utf-8 日志 + 任务微扰） |
| `tools/vid_verify.py` | 视频验收闸门 v2：每项 0~1 partial credit + `--min-quality`；**无声可视化不再误杀**（`--require-audio` 才硬要求） |
| `tools/triple_bank.py` | 三元组银行（sqlite），过闸才入库，拒收也落盘 |
| `tools/weak_to_strong.py` | 弱到强升级（LEASH 思路）：弱提议 → 评估器验收 → 教师精炼（可插拔，未配则诚实标 pending） |

### 推理时学习（TTT）
| 脚本 | 职责 |
|---|---|
| `tools/ttt_infer.py` | In-Place TTT 推理入口，**底座已参数化**（默认 0.2B，可传 0.3B/0.4B） |
| `tools/ttt_latent.py` | T2-Z 式测试时隐变量适配（零初始化 latent 注入残差流，守 E-7） |

### 对话线
| 脚本 | 职责 |
|---|---|
| `tools/synth_chat_data_v7.py` | 合成器：教师种子 × 问法/答法组合 → 唯一 prompt 语料 |
| `data/chat_seeds/seeds_v7{,b,c,d}.py` | 教师手写种子，13 域 |
| `tools/chat_diag.py` | in-set completion CE 体检（RESULT.md 口径） |
| `tools/chat_blind_eval.py` | 真盲测（训练集完全没见过的新问法 + 关键点命中率） |
| `tools/overfit_test.py` | 过拟合探针（支持按 completion token 筛中位长度） |
| `tools/train_queue.py` | **训练排队机**：单一 `.train_lock` 插槽，一次只跑一个训练 |

### 听觉线
| 脚本 | 职责 |
|---|---|
| `tools/audio_attach.py` | Whisper → pool → MLPProjector(768→576) → 0.3B，early fusion 三阶段；按说话人切分留出真盲测 |
| `tools/build_audio_pairs.py` | AISHELL 解包 + pairs 切割 |
| `tools/dl_aishell.py` | 数据下载 |

### 爬虫 / 情报
| 脚本 | 职责 |
|---|---|
| `tools/ai_frontier_crawl.py` | arXiv(cs.AI/LG/CL/CV) + HF Daily Papers → 14 桶分类 |
| `tools/split_frontier_by_category.py` | 物理拆分到 `data/frontier/by_category/` |
| `tools/classify_frontier.py` | 分类规则 |

### 识图线（冻结，仅供修复参考）
`tools/vlm_attach.py` · `tools/vlm_data.py` — 已知坑：infer 须显式 `--init <snapshot>/projector.pt`（否则默认加载 align 的）；v6 projector 实为 `ps=1`；`fc2` 必须 std=0.01 初始化（零初始化会让 RMSNorm 反向炸 nan）。

---

## 六、环境与纪律（血泪，别重复踩）

| 项 | 规则 |
|---|---|
| 本地长进程 | **只能** `schtasks /ru SYSTEM`（Bash 后台会被会话清扫） |
| 训练 | **一律走 `tools/train_queue.py`**，一次只一个（`.train_lock`） |
| Python | 训练 `venv/Scripts/python.exe`；推理/研究 `venv_ttt/Scripts/python.exe`（训练 venv 禁装非训练包） |
| 代理 | Karing 规则代理 `127.0.0.1:3067`，别叠代理层 |
| Git Bash | 调 Win 工具先 `export MSYS_NO_PATHCONV=1` |
| 权重入库 | 95MiB 分片 `.partNN` 后缀绕过 `*.safetensors` 忽略；**不用 LFS**；附 `reassemble.py` 自校验 |
| 提交 | **绝不 `git add -a`** —— 工作区里 `backups/0.1B/*` 是删除态，`-a` 会把已归档备份从仓库删掉。一律显式列路径。 |

---

## 七、下一步（按优先级，未完成）

- [ ] **对话**：`chat_v7_03b` 训练（3000 条）→ in-set CE<0.60 → 盲测命中率
- [ ] **对话**：0.3B 中位长度过拟合探针（判定"数据不足" vs "底座表征不足"）
- [ ] **听觉**：特征预计算完成 → align 阶段训投影 → CER 盲测
- [ ] **做梦**：`dream_autorun.py` 常驻（schtasks 30min 周期，coder 底座持续产码回灌）
- [ ] **编程**：0.4B 预训练把代码语料并入 `owner_corpus`（让北极星自己会写代码）
- [ ] **0.4B**：远程训练收尾 + 权重回捞入 `backups/0.4B/`
- [ ] **识图**：解冻后先修部署坑 + 迁 0.3B 重训 align/joint
