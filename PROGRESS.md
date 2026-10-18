# 进展时间线（Progress Log）

> 规则：**倒序，最新在最上**。每次动手后追加一段，只写**硬数据**与**未完成项**，不写感想。
> 交付物写清路径，让下一个人（或未来的我）能直接接着干。
> 配套：项目总纲见 `PROJECT.md`。

---

## 2026-10-18 20:20 · 🏆 首个自进化周期裁决：ADOPT（机器自己拍板，无人在场）

- **周期全记录**（node1 19:54:32→19:58:54，共 4 分 22 秒）：闸门 bank=471/new=162/uniq=25 全过 → 停 8 worker 让核 → 自训 25 条 distinct（25 步 50s，loss 1.34→0.16，trainable 2.16M/496M=0.44%）→ A/B（offset=8，8 题确定性+贪心）→ **bare 0/8 vs 自训 LoRA 4/8** → ADOPT：`DREAM_LORA` 热切换，daemon 以适配器模式复产（8 worker 全活）→ 29 条燃料导出上传 ECS。
- **意义**：RSI 闭环端到端首次跑通且首判为正——与代码 LoRA 的负迁移（0/8 vs 2/8）相反，动画域自训适配器是真资产。A/B 评测集仅 8 题，4 vs 0 是小样本；后续周期以"新适配器必须打赢裸模"为棘轮（打不赢现役留任），质量只升不降。
- **与多 Buff 愿景的差距（后续，未动）**：现 A/B 对照臂恒为裸模，新适配器从裸底座独立训练——是"棘轮替换"不是"叠加"（W′=W+B₁A₁+B₂A₂）。叠加式多 Buff 列为下一阶段设计。
- **当前在跑**：集群带适配器产粮中；下一周期 node1 20:14 触发（银行已超 471，uniq 继续涨）；本地 chat_v7 ~20:10-20:30 完 → 自动接 30 题盲测。

## 2026-10-18 20:10 · 池子 69→93（补 24 条教程级）+ 降温 0.5（燃料增速差 6 倍的解法）

- **实测**（temp 0.7 + 池 69，新集群首 12 分钟）：48 次尝试仅 1 条真入库（#464 旋转螺旋线）⇒ 燃料 ~10 条/小时，闸门要 60 条/小时，差 6 倍。
- **失败构成**：0.5B 对难任务（分形/傅里叶/耦合振子/物理仿真）写"近似对"代码（标量当数组 `set_ydata(标量)`、`FuncAnimation(ax,...)` 传错第一参、NameError），对带参任务违背帧数被 task_fit 正确拦。**难任务通过率 2-5%，教程级 20%+**。池子构成失衡：69 条里难任务占 57%。
- **修复**：DREAM_BANK +24 条教程级（弹跳球/旋转正方形/扩张圆/自由落体/弹性碰撞/行进波/时钟指针/涟漪/进度条/星空/简易单摆/抛物线/阿基米德螺线/扇形填充/五角星/椭圆轨道/随机信号/一维梯度下降/线性查找/圆周运动/台球/水平弹簧/漂浮物/光直线传播）。**安全模板**：显式帧数+out.mp4+动画，不碰 Kπ 定义域（机验 linspace 上界），不要求额外要素（网格/图例），hint 给标准写法。DREAM_TEMP 0.7→0.5（0.7 时 80 帧任务写出 200 帧）。
- **部署**：node1 8 worker 全 temp=0.5 + pool=93（ps 实测 8/8）。⚠️ ECS 信箱偶发 wedge：curl 拉文件空手而归，n1_push 自带 6 次重试，安装命令也要带 `[ -s file ] && break` 重试循环。
- **闸门未动**（双闸门原样），先看新池子一个周期的真实增速再决定校准。

## 2026-10-18 19:50 · 燃料多样性三连修（闸门连续空转三次的根因）

- **实锤**：evolve_loop 闸门日志 18:34/18:54/19:14 连续三周期拒绝——bank 309→441（+132/h）但 **uniq=17 不动**（口径=`count(distinct code_text)` 精确字符串）。
- **根因 1（任务饥饿）**：daemon 的 `off=$(((id-1)*N))` 在 while 外算死，8 worker 永久钉在任务 0-31；池扩到 69 后队尾 37 条（含 11 条物理种子）永远排不上队。
- **根因 2（假增量）**：`gen_code(temp=0.0)` 贪心 ⇒ 同任务每次逐字节相同代码；入库无去重 ⇒ 行数虚涨，唯一定码卡死。
- **修复（A/B 公平性不动）**：daemon offset 每轮 `RANDOM%POOL`（池大小向 dream_loop 要真值，不硬编码）+ `--temp 0.7`（DREAM_TEMP env 可调）；dream_loop 加 `--temp`（默认 0.0，A/B 两臂仍贪心确定性）；triple_bank 入库前 code_text 查重，重复 SKIP 且 rc=0（算 pass 不涨行）⇒ **银行行数=真燃料**。
- **部署**：node1 8 worker 全带 `--temp 0.7` 在跑（ps 实测 8/0），offset 随机化生效（66/13/30/16/54/22/53/22 全覆盖）。⚠️ 运维坑重踩：部署命令里同时出现 `pkill -f '[d]ream_daemon.sh'` 和明文字符串 `dream_daemon.sh`（nohup 行）⇒ 正则匹配到执行 shell 自己 ⇒ 整单静默毙命（220s 无回传、集群还被顺手杀掉）。修法：**杀集群单独一条命令（只含括号模式），安装重启另一条（不含任何 pkill）**。
- **待验证**：新集群首轮产出——distinct 应爬升、SKIP 行应出现；下两周期闸门大概率仍拒（存量 424 重复行+新燃料刚起步），看 distinct 增速。

## 2026-10-18 19:00 · 爬虫升级：arXiv 4→12 类 + OpenAlex 中英双路 + 双轨去重

- 源扩到 4 路：arXiv **12 类**×25（+MA/RO/NE/SE/CR/IR/stat.ML/eess.AS）+ HF 30 + **OpenAlex 英**（title_and_abstract 核心词近 5 天 article，补 LNCS/Springer 期刊）+ **OpenAlex 中**（title 搜 CS 中文词，窗口 4 个月，滤 Zenodo/PhilPapers 自出版 junk）。实测一轮 325 候选→**新增 241 篇（中文 44）**（旧版每天 0-4 篇）。
- **双轨去重**：id（arXiv/HF/OpenAlex W-id/DOI）+ **标题指纹**（归一化 md5[:8]）；账本 `_seen_ids.json` 升级 `{"ids","titles"}`（兼容旧扁平列表）。实测价值：按 id 数 367 篇 vs 按 id+指纹 340 篇 → **27 篇跨源重复只有指纹拦得住**。
- `classify_frontier.py` RULES 补中文关键词（否则中文论文全落"其他"）；`split_frontier_by_category.py` 去重同步双轨。
- 探测排除：OpenReview 403 / SemanticScholar 429 / PapersWithCode API 已关 / ChinaXiv 404 / 机器之心非论文源 / paper.edu.cn JS 渲染无 API。
- schtasks FrontierCrawl（30min/轮）自动生效，今日累计 245 篇。

## 2026-10-18 18:37-18:52 · DREAM_BANK 加 11 条物理世界函数 + 任务池 29→69 + 修 worker offset 饿死 bug

- **真 bug**：daemon 不给 worker 传 `--offset`，8 worker 全从 `DREAM_BANK[0]` 顺序取，`--n 4` ⇒ **队尾任务永远饿死**（物理种子加进去排不上队）。已修：worker 按 id 错峰 `offset=(id-1)*N`，8×4=32 覆盖全池。
- 物理种子 11 条：牛顿冷却/理想气体粒子盒/一维热传导/卡诺循环/斯涅尔折射/双缝干涉/多普勒/弹簧简谐/抛体/驻波/熵增（全带公式 hint，要求产 mp4）。
- 任务池 **29→69**：数学函数 12 + 几何曲线 8 + 信号处理 6 + 算法可视化 8 + 物理补充 6（一次加够直接起跑）。部署 md5 与本地一致，worker_7 实测在跑物理仿真任务。

---

## 2026-10-18 18:00-18:25 · 双机自进化闭环合龙：本地腿自动化（提纯语料 + 夜间 RSI + 队列自动验收）

- **发现并修复的闭环缺陷**：银行 325 条入库但**唯一定码仅 15 条**（8 worker 同批题产出高度趋同）。旧闸门只看 bank 增量 → cycle=0(18:14) 停机后自训发现唯一样本 15<24 → train_fail rc=2 → 裸模复活（失败路径验证 OK，但白停一轮）。**修复=双闸门**：`bank新增≥20 AND 唯一定码≥20` 才训（node1 `evolve_cycle.sh` 已热更新，实测 cycle=1 正确拒绝空转）。
- **燃料外送（node1→本地自动管道）**：`evolve_cycle.sh` 每周期末尾跑 `tools/export_triples_to_sft.py` 并上传 ECS `dream_fuel.jsonl`（已随周期脚本推上 node1；首次手工导出 15 条已验证）。
- **本地腿三件套**（本地 `tools/`，均已部署）：
  - `corpus_curate.py`：清掉 v1 负迁移根因——card_code 1463→**1052**（剔除 411 条纯定义无调用样本）+ 15 条 dream 燃料 → `data/code_sft_v2.jsonl`（1067 条，带 `_src` 溯源）。
  - `nightly_rsi.py` + schtasks **NightlyRSI**（SYSTEM，每日 00:30）：拉 ECS 燃料→增量≥20 且队列无代码任务→提纯合并→排「训练+自动盲测」job（lr 1e-4/1ep/r16 qkvo，比 v1 的 2e-4 更保守）。实测跑通：fetched 15、增量不足正确不排。
  - 队列现状（`train_queue.py status`）：① chat_v7_03b 在跑（750/1500，~21:00 完）→ ② **chat_v7_blind_eval**（30 题盲测自动验收）→ ③ **code_v2_curated**（训练+`code_eval` 自动盲测，产出 `logs/code_eval_codeqwen05b_v2.json` 与裸模 8/20、v1 6/20 直接对比）。
- **双机分工定型**：node1=做梦产粮+视频域适配器自进化（evolve_loop）；本地=北极星对话/代码模块重训+盲测（nightly_rsi + 队列）。两条腿都不再需要人工触发。

## 2026-10-18 17:30-17:55 · 全自动自进化闭环部署 node1（自训→自测→自裁决→自切换，无人在场）

- **背景**：用户裁定"自举=让 node1 自己改自己，不是人肉搬适配器"。此前代码 LoRA A/B（裸 2/8 vs LoRA 0/8）只是人工的一次性实验；本轮把整条回路固化成 node1 常驻脚本。
- **新增文件**（node1 `/mnt/workspace/node1/dream/`，本地源在 `node/`）：
  - `evolve_train.py`：CPU 自训器。读 triple_bank → 去重 → cap 200 条 → **直接 import dream_loop 复用 SYSTEM/DREAM_BANK/build_prompt**（协议逐字同构，银行无 hint 就查表）→ completion-only loss（prompt 段 -100）→ LoRA r16 q/k/v/o（守接口契约第 0 条）→ lr 1e-4 / 1 epoch / fp32 / torch 8 线程。退出码 0=训完 2=样本不足。
  - `evolve_cycle.sh`：单周期编排。闸门(银行新增≥20 才训)→停集群让核→自训→A/B(同 `--offset` 轮转评测集防过拟合固定 8 题,两臂公平)→裁决 lora_accept>bare_accept → ADOPT:`DREAM_LORA` 热切换重启 / REJECT:适配器归档 `models/evolve_lora_rej_*` 裸模重启 → 写 `.evolve_state.json`+`logs/evolve_history.log`+报告上 ECS。**所有失败路径都复活集群（宁可不进化不能停机）**，mkdir 锁防周期重叠。
  - `evolve_loop.sh`：常驻触发器，每 20min 跑一次 cycle（cycle 自带闸门，不达标空转）。
  - `tools/dream_loop.py` 加 `--offset`（默认 0，向后兼容）。
- **部署实况**：2026-10-18 17:54 evolve_loop 起跑（pid 195754），首周期 gate 初始化基线 bank=253、new=0<20 不训、8 worker 未打断。预计 ~18:10 银行新增≥20 触发首个真实周期（训 200 条 CPU 约 25-35min + A/B 2×8 约 12min，首裁决 ~18:50）。
- **运维教训（重踩）**：mb.py 拼装=`cmd + " 2>&1 | curl ..."`——**只有最后一段管道进 curl**，中途 `&&` 链断或最后段为空=整单无回传。远程命令必须整体包 `{ ...; } 2>&1`，且末尾放汇总 echo。另：`A && B && nohup X &` 的 `&` 会把整条链后台化（shell 优先级），部署命令必须前台 brace 执行。
- **待验证**：首个 ADOPT/REJECT 裁决结果；`evolve_report_c<N>.txt` 自动上传 ECS 可查。

## 2026-10-18 17:15-17:29 · 代码 LoRA 反哺做梦引擎 A/B 实测（负资产，不上线）

### 0) 触发器
- 用户指令：「既然那个模块已经训练完了，里面一些东西能不能把它丢到远程，让它用起来更好呢？相当于就是哪里不会自己改自己。」→ 把本地训完的代码 LoRA（`out_lora_code_qwen`，8.7MB）推到 node1，A/B 验证对做梦引擎有没有用。

### 1) A/B 设计（对照极干净）
- 任务是**确定性顺序**（`DREAM_BANK[i%18]`，`--n 8` 固定同一批 8 题）+ **贪心解码**（temp=0）→ 输出差异只可能来自 LoRA 权重。
- 指标：`通过(入库)` = 过完 vid_verify + task_fit 全闸门的真通过（ELITE skip 不计入）。
- A/B 期间停 daemon 让出全核防 exec 超时误判；脚本 `node/n1_ab_test.sh`，日志 `logs/ab_bare.log` / `logs/ab_lora.log`。

### 2) 结果（硬数据）
| 指标 | 裸 Qwen0.5B | +代码 LoRA |
|---|---|---|
| 通过入库 | **2/8** | **0/8** |
| vid_verify_fail（静止） | 3 | 2 |
| exec 失败 | 2 | 1 |
| **跑通但不产视频** | 0 | **4** |
| task_mismatch | 1 | 1 |

- LoRA 全拒，且引入新故障模式「exec 成功但无 mp4 产出」（4/8）。根因：通用代码问答语料教的是"写代码回答问题"，不是"写脚本落盘产文件"。
- **裁决：负资产，不上线。** daemon 已用裸模式重启（pid 167972，8 worker，`lora=none`）。银行 triples=176 / rejects=174。
- 附带基建：`dream_daemon.sh` 新增 `DREAM_LORA` 环境变量开关（空=裸模型，设了自动传 `--lora`），下次任何适配器都可一键 A/B 后热切换；`node/n1_daemon_start.sh bare|lora` 一键重启+自检。

### 3) 未完成项
- [ ] **正路已指明**：要提升做梦引擎，得用**做梦域自己的数据**训——三元组银行已入库 176 条动画代码（全过闸门）就是现成语料，训"动画域 LoRA"再走同一套 A/B 流程。这才是 RSI 正形（银行→训练→回灌→再银行）。
- [ ] chat_v7_03b 本地训练在跑（step 500/1500，ETA 今晚 ~22:00），训完接盲测验收。
- [ ] `dream_code_sft.jsonl`（13 条）待接入夜间 polaris 代码 LoRA 重训。

---

### 0) 触发器
- 用户指令：「远程现在给他拉满，他现在只是在搞自进化相关的。因为那边那个服务暂时就是先搞这个东西。」→ node1 专跑自进化，资源吃满。

### 1) 诊断（关键纠偏）
- 旧 daemon 单进程串行 + 60s sleep，单推理只占 1 核；`top` 显示 us≈12% 是**相对 64 核**视角 ≈ 1 核，没拉满。
- ⚠️ **tty 终端污染陷阱**：一次 `top` 读数掺进了 `train_remote_v4.py`/`v1_02_alpha_ladder.py` 等训练日志（连 loss 都冒出来），一度误判 node1 有别进程抢资源、内存吃紧。实查 `ps` + `free`：**无其他重负载进程、27GB 空闲为真**；污染来自某长驻进程 stdout 接到了同一伪终端。

### 2) 改动（3 个文件，走 ECS 中转拉到 node1）
- **并发护甲**：`tools/triple_bank.py` + `tools/dream_elite.py` 的 `conn()` 加 `PRAGMA journal_mode=WAL` + `busy_timeout=30000` + `synchronous=NORMAL`。多 worker 经子进程(add/query/trial)并发打库不锁死。
- **swarm daemon**：`node/dream_daemon.sh` 升级为集群——`DREAM_WORKERS`(默认 8) 个 self-healing worker，每 worker `OMP/MKL/OPENBLAS_NUM_THREADS=1` → 合计正好压满 cgroup 8 核；错峰 `id*3s` 加载模型防 NAS 惊群；`trap` 清理孤儿；轻量银行快照。
- **控制脚本**：`node/n1_swarm_ctl.sh`（诊断+自适应 worker 数+拉文件+起集群+报告），规避信箱对多行内联命令压成一行导致脚本错乱（改用「写脚本文件→ECS→node1 bash」）。

### 3) 验证（node1 实测）
- `workers=8` 全活；cgroup `quota 800000 / period 100000 = 8 核`封顶。
- `top` 瞬时 us≈12%(相对64核) = **≈7.5 核在用 ≈ 94% cgroup 预算** → 已拉满；`ps` 累计 `sum_pcpu=733%` 印证。
- 内存：启动后 used 1.1→**19.3GB**，可用 9.3GB，8 个 Qwen 模型不 OOM。
- 银行：triples 29→**51**、rejects 33→**55**，持续涨（8 worker 吞吐远高于单进程）。
- DB：`locked=0 / OperationalError=0`（WAL 生效）；`Traceback=4/worker` 全部是**模型代码 `ani.save` ffmpeg 运行时失败被 `[exec] 失败` 正常拒收**，非基础设施 bug。

### 4) 待办（继承 + 不变）
- [ ] `ani.save` flaky ffmpeg 报错：加重试 / 执行前清 `out.mp4`（已确认非锁问题）。
- [ ] 提通过率：恢复记忆检索（先修 max_new 截断 + 防复读）。
- [ ] #3 下游链路待接入夜间 polaris 代码 LoRA 重训（`data/dream_code_sft.jsonl` 13 条已就绪）。

---

## 2026-10-18 16:10-16:25 · 做梦闭环代码容错加固（import/blit/savefig/format 四连修）+ daemon 重启

### 0) 触发器
- 实测 `--n 8` 前台验证一度 **0/8 全拒**，逐一排查是代码执行期崩溃类，非模型纯质量；逐一定位修复。

### 1) 四连修（`tools/dream_loop.py::inject_imports`，本地 30085→31070 字节）
- **修复 A — fig,ax 注入顺序（NameError）**：旧版把 `fig, ax = plt.subplots()` prepend 到代码最顶，先于 `import matplotlib.pyplot as plt` 执行 → `NameError: plt 未定义`。改：fig,ax 行进 `head` 末尾（排在所有 import 之后）；并**无条件预置 plt import**（Qwen 自带 import 也不冲突，重复 import 无害）。
- **修复 B — 去掉危险字符串替换**：旧版 `plt.figure()`→`fig` 会把 `fig = plt.figure()` 改成 `fig = fig` 致崩。改为：Qwen 没显式写 `fig, ax = plt.subplots()` 就直接注入一行，若 Qwen 自另定义则被覆盖，无害。
- **修复 C — 强制 blit=False**：`re.sub(r'blit\s*=\s*True','blit=False')`。blit=True 要求 update 精确 return 可变 artist 元组，弱模型常写错 → FuncAnimation 崩/静止；blit=False 永远安全。
- **修复 D — savefig→ani.save 重写 + 剥 format**：弱模型常写 `plt.savefig('out.mp4')`（savefig 不支持 mp4 会崩）。若已建好 FuncAnimation(ani) 则重写为 `ani.save(..., writer='ffmpeg', fps=20)`；并剥掉 `format='mp4'`（与 writer 冲突会崩）。

### 2) 验证效果（node1 硬编码 `--n 8`，同批 8 任务）
- 修复前 0/8 → 修复 A+B 后 1/8 → 修复 C+D 后 **2/8**（入库 #27 正弦波、#28 余弦波）。
- 剩余 6 拒：3× `vid_verify_fail`（Qwen 产出静止 mp4，update 未逐帧动——模型质量，闸门正确拦截）；1× `task_mismatch`(缺:余弦，SYSTEM 第5条已约束但 0.5B 仍偶漏)；2× `ani.save` 偶发 ffmpeg 报错（flaky）。
- 结论：全部**崩溃类**已清零，剩余拒绝均为「正确过滤弱模型劣质样本」，符合闭环设计意图（RSI 自举从好例出发，拒绝坏例）。

### 3) daemon 重启（7×24 常驻，跑加固后代码）
- node1：`nohup bash dream_daemon.sh`（pid 105089，loop 105094），`daemon.log` 16:21:13 round=1 begin。
- 文件 `tools/dream_loop.py` 31070 字节、16:16 落盘，已通过 ECS 中转拉取确认。
- 银行快照：triples=29（round1 内 +1）、rejects=33。

### 4) 远程 CPU（同前实测，未变）
- cgroup 硬限 8 核（quota 800000/period 100000）；推理态 us≈12%、sleep 态 id≈99.8%。

### 5) 待办（继承 + 新增）
- [ ] 2× `ani.save` flaky ffmpeg 报错根因（疑帧/文件残留，可加重试或清 out.mp4）。
- [ ] 提通过率：恢复记忆检索（先修 max_new 截断+防复读）或加 few-shot 好例。
- [ ] 扩池已做（DREAM_BANK 6→18、SYSTEM 第5条），可再扩 SOFT 波形池。
- [ ] #3 下游链路已就绪：`export_triples_to_sft.py` 产出 `data/dream_code_sft.jsonl`（13 条），待接入夜间 polaris 代码 LoRA 重训。

---

## 2026-10-18 15:00-15:10 · 自我进化闭环正式 7×24 运行（实测 CPU 占用 + 自动增长）

### 0) 闭环运行状态（实测）
- daemon(pid 42609) 常驻 node1，连续 round 11→17 全部 rc=0，无崩溃；每轮 ~40-50s 推理 + sleep 60s。
- `triples` 13→**18**（#18=正弦+余弦叠加动画）；`rejects`=11、`trials`=118；每轮 ~1 入库 / 3 失败（通过率 ~17-25%）。
- ELITE 去重闸门生效：`[accept] ELITE skip 近重复解法(3gram>=0.82)` 已触发，防刷重复样本。

### 1) 远程 CPU 占用（cgroup 实测，非 nproc 假象）
- 约束：`/sys/fs/cgroup/cpu/cpu.cfs_quota_us=800000` / `cpu.cfs_period_us=100000` = **硬限 8 核**（nproc=64 是假象）。
- 推理态：`vmstat` 6s → `r=8`（满队列吃满 8 核配额），全局 `us≈12%`（=8核/64核总量），`id≈88%`。
- sleep 态（两轮间 60s）：idle≈99.8%。
- 结论：闭环 7×24 自动进化，但只占服务器总算力 ~12%，且每轮后 sleep 让出，不影响同机其他任务。

### 2) 未完成（同前）
- [ ] 恢复记忆检索提通过率（需先修 max_new+示例长度防复读）
- [ ] 通过率提纯 / 扩简单任务池
- [ ] 自我生成代码 → polaris 训练燃料下游链路（RSI 收口）

---

## 2026-10-18 14:25-15:00 · 做梦闭环从「管道断裂」到「首条自我样本入库 #13」

### 0) 卡点根因（三轮定位，已全修）
- **根因1·few-shot 复读+截断**：`retrieve_examples` 注入 3 个长示例(~500-600字符)，Qwen 退化成「示例1/2/3 回放」且 max_new=512 预算被吃光→闭围栏 ``` 被截断→`extract_code` 正则要求闭围栏→匹配失败→代码长度=0。polaris-0.3B 代码底座本就为 0，更不可能出码。
- **修复1**：`dream_daemon.sh` 默认加 `--no-memory`（先出活），做梦者改用 `Qwen2.5-0.5B-Instruct`（polaris 代码底座为0，自举产不出代码）。
- **根因2·裸代码无围栏**：新 SYSTEM 下 Qwen 吐裸代码（无 ```python 围栏）→ 同上匹配失败。
- **修复2**：`extract_code` 加裸代码 fallback（全文含 import/def 即整段取）。
- **根因3·FuncAnimation 小 bug**：旧 SYSTEM 下 Qwen 生成的动画代码 FuncAnimation 调用/函数定义错（rc=1）。
- **修复3**：`SYSTEM` 重写，强制标准 FuncAnimation 骨架（只改 update 函数体，调用方式固定 blit=False）+ 禁止 plt.gcf 替代 fig。

### 1) 闭环打通硬数据
- 前台 `--n 6`：6 个 task **1 个 ACCEPT 入库 #13**（正弦+余弦叠加动画，exec 成功生成 mp4→vid_verify 通过→accept+进 elite 档案），5 个 rc=1（小 bug）。
- 通过率 ~17%（RSI 自举早期正常）。`triples` 12→13（首条由做梦生成的自我进化样本）。
- 全链路验证：生成→extract→run_code→视频产出→vid_verify→accept 闸门(task_mismatch 正确拒不符合任务的)→入库，全部活。

### 2) 部署
- `dream_daemon.sh`（Qwen base + --no-memory）已 nohup 常驻 node1（pid 42609，每轮 ~60s + sleep 60）。每轮新 python 进程重读 dream_loop.py，新逻辑自动生效。

### 3) 未完成
- [ ] 恢复记忆检索（few-shot）：用干净正样本 #13 作示例提通过率；需先修 max_new(512→1024)+示例长度限制+防复读指令，避免根因1 复发。
- [ ] 通过率提纯：当前 ~17%，需微调 SYSTEM / 扩充简单任务池让早期正样本更快积累。
- [ ] 自我生成代码 → polaris 训练燃料的下游链路（RSI 闭环收口）未接。

---

## 2026-10-18 14:00-14:25 · 打通本机↔node1 直连隧道 + 魔搭仓库当数据面（信箱退休）

### 0) 硬数据

| 通道 | 200MB 单向 | 速率 | 1.3GB 预估 |
|---|---|---|---|
| ECS 信箱 / scp 隧道 | 未跑完（8MB 即断） | **~94KB/s** | ~4h |
| 魔搭数据集仓库 → node1 | 65.5s | **3.05MB/s** | ~7min |
| 本机 → 魔搭仓库 | 35.6s | **5.62MB/s** | 家宽上行瓶颈 |
| 模型仓库 → node1（Qwen-0.5B 954MB） | 70s | ~13.6MB/s | 模型库 CDN 更快 |

结论：**信箱慢的根因是 ECS 出网带宽，不是轮询**。改造信箱收益为零，必须换通道。

### 1) 通道 A：SSH 反向隧道（控制面，秒级）

- 边界：ECS 只开 22/80/8081/3389；80 被阿里云备案网关拦（`{"error":"not found"}`）；7000/6022 未开；
  魔搭 gateway 代理要阿里云账号 cookie。**只有 22 可用**。
- 方案：node1 主动 `ssh -R 127.0.0.1:9022:127.0.0.1:22` 到 ECS（不改安全组、不用 GatewayPorts），
  本机 `ProxyJump` 两跳直连。握手 1.2~1.5s，之后 1s 级；**支持多行/管道/heredoc + 真实 exit code**。
- 密钥：本机 `D:/work/_local_ai/_tunnel/node1_key` → node1 `/root/.ssh/authorized_keys`；
  node1 `~/.ssh/ecs_tunnel_key` → ECS `C:\ProgramData\ssh\administrators_authorized_keys`。
- Windows 坑：ProxyJump 下 `ControlMaster` 复用必失败（`mux_client_request_session: read from master failed`），已移除。
- 交付：`node/n1.sh`（`--alive/--put/--get/-f/命令`）、`node/tunnel_up.sh`（幂等守护，已挂进 `dream_daemon.sh` 开头）。

### 2) 通道 B：魔搭仓库中转（数据面，3MB/s）

- 交付：`tools/ms_relay.py`（`push/push-dir/pull/ls/rm`，`--model` 走模型仓库 `hhhlhuk/polaris-03b`，
  默认走数据集仓库 `hhhlhuk/polaris-relay`，两端各一份）。
- Token 坑：`~/.modelscope/credentials/session` 那个 32 位 hex **已失效**（E3001）；
  能用的是 `deploy_aliyun/ms_token.txt` 的 36 位 UUID。脚本按 env→ms_token.txt→credentials 逐个试登录。
- 坑：`upload_file` 默认 `repo_type='model'`，数据集必须显式传 `repo_type='dataset'`（否则文件进模型库、pull 404）。
- 实测：并发 Range 4 路**不提速**（1.85MB/s < 单流 3MB/s），瓶颈在账号侧限速，不要再加并发。

### 3) 做梦线现状（阻塞项）

- node1 `dream_daemon.sh` 连跑 9 轮：**0 ACCEPT / 32 REJECT**，库内 12 条全是手写 seeds。
- 失败主因：`[gen] 代码长度=0` —— **polaris-0.3B 连代码块都产不出来**；老 rejects 则是 `frames=40` 恒定错。
- 已停 daemon，改用 `Qwen2.5-0.5B-Instruct` 作教师跑探针（`logs/qwen_probe.log`），待结果定夺。

### 4) 文档

`docs/direct-channel-2026-10-18.md`（架构/实测/用法/失败模式/明确不做的三件事）。

---

## 2026-10-18 13:20-13:55 · 第四路判官结果 + 做梦线整体搬到魔搭 + 邮件箱换隧道

### 0) 第四路 `codeqwen15b_remap` 最终数据（模块跨底座搬运的判官）

| 路 | 底座 | 挂载 | strict | loose | 格式层 |
|---|---|---|---|---|---|
| 1 | Qwen2.5-0.5B | 裸 | 40% | 85% | — |
| 2 | Qwen2.5-0.5B | 编程 LoRA | 30% | 65% | — |
| 3 | Qwen2.5-1.5B | 裸 | 50% | 95% | — |
| **4** | **Qwen2.5-1.5B** | **0.5B 训的 LoRA 经 remap 搬过来** | **50%** | **90%** | **100/100/100/100** |

结论（层一成立的最强形式）：
- **格式层 100% 全项**（代码块/语言标注/配说明/正常收尾），模块把「开口方式」完整搬过去了。
- **内容层 strict 50% = 裸 1.5B 的 50%**，搬运零损伤；而挂在 0.5B 上时是 30%（比裸 0.5B 的 40% 低 10pt）。
- 即：**小底座上训练会压内容能力，搬到强底座后这个副作用消失**。这正好印证「format 归模块、content 归底座」。
- 唯一 strict_fail 是 `mkdir: command not found`（Windows 执行环境缺 coreutils），非模型问题。
- 明细 `logs/code_eval_codeqwen15b_remap.json`，流水线日志 `logs/p2_pipeline.log`。

### 1) 做梦线整体迁到魔搭 node1（本机实测跑不了）

环境对比（实测）：

| | matplotlib | ffmpeg | Qwen-0.5B | 可用内存 |
|---|---|---|---|---|
| 本机 | ❌ 缺 | ❌ 缺 | — | 剩余 ~12GB（对话线占 3.3GB）|
| node1 | ✅ 3.10.8 | ✅ /usr/bin | ✅ 已下 | 26GB 空闲 |

- 工具链 `/mnt/workspace/node1/dream/`：8 个 py + `data/triple_bank.db`，42KB 走 ECS `/upload` 中转。
- 修了两处 Windows 硬编码：`dream_loop.py` 的 `FFMPEG` 改 `_find_ffmpeg()`（先认 vidcut 静态包，回落 `shutil.which`）；`run_code` 的 PATH 注入加空目录保护。
- **分工定调**：node1（8 核/28GB）跑 7×24 做梦循环；本机跑识图/音频/编程 eval；GPU 夜间到场再训预训练。
- 本机 `venv` 缺 matplotlib/ffmpeg **不是 bug**——做梦本来就不该在本机跑。

### 2) 手写 15 套规范骨架当冷库种子

- `tools/dream_seeds.py`，15 条覆盖三角/正弦余弦/衰减/Lissajous/螺旋/3D 曲面/热力图/傅里叶/布朗/条形竞赛/Julia/冒泡/生命游戏/抛体。
- node1 上 `--check` **15/15 PASS**（任务符合度静态校验）。
- `--run` 真实渲染：**7 ACCEPT 入库 + 0 task_fail + 3 vid_fail**（螺旋线/3D/热力图被判「画面不动」）。
- 入库带精英档案分桶，例：`多序列对比|折线+funcanim q=0.83`。

### 3) 冒烟模式污染库（已发现，待修）

`dream_loop.py --smoke` 会走 `accept()` 真入库，写入「冒烟测试-视频闸门」这种垃圾条目。修法：smoke 只验签不落库。**未改**。

### 4) 传输通道：邮件箱 → frp 隧道

- 魔搭自带端口代理 `MODELSCOPE_NOTEBOOK_PROXY_URL` 需阿里云账号 cookie，**不适合机器间自动通信**（实测 4 种 token 全 302）。
- ECS 只开 22/80/8081/3389；**80 空闲**。
- 已部署：`frps.exe` 在 ECS `C:\Users\administrator\`，监听 **0.0.0.0:80**（bindPort 与 vhostHTTPPort 复用 80），token `k0u4mail2026`。
- 魔搭侧 frpc 走 `ghfast.top` 镜像取（GitHub release 在魔搭下载只有 601K 坏包；ECS 直传 13MB 超时，出网仅 ~1KB/s）。
- 目标：`http://47.99.131.12/` → 魔搭 8010 的 `tools/exec_server.py`（exec/put/get/tail/ls，token 鉴权），把远程当本机用。

---


### 0) 硬数据（这条最重要）

装上任务符合度校验后回扫 `triple_bank` 全表：

| 项 | 结果 |
|---|---|
| 入库样本 | 7 条，**任务符合度合格 0 条** |
| 唯一代码 | 只有 **4 个**（#4=#7，#6=#8=#9=#10） |
| 两两 3-gram 相似度 | 0.84 ~ 1.00（几乎同一份代码） |
| 帧数符合 | **7/7 全错**：任务要 100/120 帧，代码一律 `frames=40` |
| 要素缺失 | 余弦/方波/三角波/李萨如/螺旋/衰减 6 项全缺 |
| 定义域错 | 2 条（要 0~6π，写 4π） |

具体：#6 要「正弦+余弦叠加」，代码只有 `np.sin(x)`；#9 要李萨如、#10 要螺旋线，代码还是 sin 波相位平移。**没有一条实现自己的任务，但 vid_verify 全部放行**。

### 1) 根因：验签函数缺乏判别力

`vid_verify.py` 的 6 项检查（可解码/分辨率/音轨/黑场/静音/画面在动）全部是**产物层**——它知道视频能不能播、画面动不动，永远不知道代码做的是不是题目要求的事。等同于无监督验收。

### 2) 第二个污染形态（新发现）

模型总写 `frames=40` 不是巧合：**原 few-shot 示例全是 40**。它不只复读 prompt 文本（`check_pollution` 查的），还会把示例里的**错误常量当规范抄走**。文本指纹闸门对这类污染完全失效。

### 3) 修法（三个工具，已落地）

**`tools/task_verify.py`（新建）** — 静态任务符合度校验，不执行代码。从任务文本抽可机验约束 → 对 AST 比对：
- 帧数 `N 帧` vs `frames=N`（严格相等）
- 定义域 `0~Kπ` vs `linspace` 上界数值（容差 1e-6）
- 输出名 `保存为 X.mp4` vs `save('X')`
- 要素：21 条关键词→必需 token 规则，且**分软硬**：缺标准函数（余弦/方波/三角波/衰减）硬拦，缺写法不唯一的（螺旋/李萨如）只告警
- 动画：`FuncAnimation` + `save`
- 抽不出约束 → 放行（绝不用脆规则卡死）
- `--selftest` 5/5（3 正例 + 2 反例）。**正例必测**：只测反例能测出漏检，测不出误杀，而误杀会把合格产物挡在银行外，做梦白跑。

**`tools/triple_bank.py`** — 在 `cmd_add` 的 vid_verify 之后、写库之前加 `check_task_fit()` 闸门。放在唯一写入路径上，做梦/回填/手工全部绕不过。`task_mismatch` 进 rejects。

**`tools/bank_purge.py`（新建）** — 历史脏数据迁移，`--apply` 才动。不删除，转 rejects（失败料是回灌原料）。同步清精英档案。日志 `data/purge_log_20261018_130449.json`（含 sha256 可追溯）。

**执行结果**：`triples 0 条 | rejects 8 条 | elites 0 条`。

**`tools/dream_autorun.py`** — 失败回灌从「技能粒度」升到「维度粒度」。原来 `_reject_skills()` 只知道哪个技能失败过，重抽同一个题模型照错不误。新增 `_reject_focus()` 解析 rejects.detail（兼容 checks 数组 / fails 字符串两种格式）→ `_focus_hint()` 把具体错因翻译成 prompt 里的硬提示，按任务相关性过滤（没有 π 的题不注入定义域提示）。

实测解析结果：`帧数符合 7 / 任务要素齐备 6 / 定义域符合 2`，提示已正确注入 hint。

### 4) 冷启动状态（未完成）

银行清空后做梦无 few-shot 可检索。**当前真实状态：能力 0，可入库样本 0，rejects 8 条带精确诊断**。这是诚实起点，之前 `memory=on pass=100%` 是验签太松造成的假象。

优先补的燃料来源不是让模型自己生（会再污染），而是**人工/强模型写规范骨架**（`教师始终云端强模型`）。

---


### 1) 做梦的进化压力（`tools/dream_autorun.py` `make_spec()`）

**病灶（实测）**：`make_spec()` 原来是 `random.choice(TEMPLATES)` 等概率抽 15 个模板。结果精英档案 5 个桶**全挤在曲线族**（三角波/衰减振荡/参数曲线/极坐标曲线/多序列对比），而曲面三维、场热力图、级数逼近、粒子统计、排名条形图、分形、排序搜索、生命游戏、刚体动画 —— **9 类技能一个桶都没占**。已经会的反复做，不会的永远抽不到。MAP-Elites 的价值全在「填空位」，不做这步它只是个好看的排行榜。

**改法**：三路混合，填格子权重自适应
```
w_hole = 0.4 + 0.4 * (空洞技能数 / 技能总数)     # 空洞 9/14 → 0.66；填满后回落 0.4，自平衡
1. 填格子 w_hole ：抽精英档案还没占桶的技能
2. 攻短板 30%    ：抽 rejects 里失败过的技能（当前仅 1 条，暂等同随机，跑够 50 轮后接管）
3. 纯随机 剩余    ：保留探索，防局部最优
```

**实测（600 次抽样）**：

| 指标 | 改前（纯随机） | 固定 40% 版 | 自适应版 |
|---|---|---|---|
| 空洞技能命中率 | 64.3% | 79.5% | **87.7%** |
| 已占桶 5 类合计 | 35.7% | — | **12.3%** |
| 覆盖技能数 | 14/14（但5类被反复抽） | 14/14 | 14/14 |
| 最高频技能 | 各 7.1% 均等 | — | 场热力图 10.8% |
| 最低频技能 | 各 7.1% 均等 | — | 三角波 1.3%（已会，少练） |

降级规则：档案读不到 / rejects 为空 → 权重让给随机路，绝不因缺数据不出题。

### 2) 方案文档：`docs/dream-to-module-loop-2026-10.md`（新建）

把「做梦线」和「模块通用化线」接起来的诊断与施工方案。核心结论：

- **自进化目前只做了「上下文层」的半圈，参数层一步没走**。逐环节盘点 9 项：生成/闸门/入库/检索回喂 🟢 已通；**失败回灌 🔴 只记不用（rejects 表 1 条，全库无 SELECT）**；**自我出题 🔴 原为纯随机（本次已修）**；**产物→训练数据 🔴 零回流**；模块结构锁 🟢；跨底座搬运 🟡 工具齐、今日首次实测。
- **为什么两条线天然同频**：`module_loader.py` 的设计前提是「模块只在恒定维表达**开口方式**，不碰知识」；做梦产出的正是「给定任务怎么把代码写出来」的具体行为 —— 行为的发生器 × 固化器 × 搬运器。
- **施工顺序 1→2→3→4**：①做梦产物导成带验签训练集 ②出题按短板（本次已完成）③扩做梦任务域 + `code_verify` ④模块训练换料。**第 4 步之前必须先有第 3 步的产量** —— 7 条做梦数据扔进 1463 条蒸馏语料连噪声都不算。
- **层三（类 MCP 模块标准）的抓手**：模块包里带一张「在 N 类任务上过闸、`vid_verify` 6 项全绿」的验签表，模块就从权重文件变成**带证的能力单元**。做梦线提供的正是这张证的签发能力。

### 3) p2 四路 eval 已出三路（LoRA 负收益，硬证据）

| 路 | strict | 宽松 | 配说明 | 执行失败 | 速度 |
|---|---|---|---|---|---|
| `bare05b` 裸 Qwen2.5-0.5B | 8/20 = 40% | 17/20 = 85% | 95% | 3 | 29s/题 |
| `codeqwen05b` 挂 p2 LoRA(5.04h) | 6/20 = **30%** | 13/20 = **65%** | **60%** | 7 | 20s/题 |
| `bare15b` 裸 Qwen2.5-1.5B | 10/20 = **50%** | 19/20 = **95%** | **100%** | **1** | 81s/题 |
| `codeqwen15b_remap` 1.5B 挂 remap 的 0.5B LoRA | 跑到 5/20（12:56） | | | | |

**结论：换底座 +10pt，训 5 小时 −10pt。** 1.5B 裸的唯一一个 fail 是 `mkdir: command not found`（Windows bash 沙箱缺 coreutils），属环境问题不是模型错 —— 实际 20/20 全过。
**`sh_ls` 两路都 fail 是同一个环境坑**：判分器该识别「命令不存在」并跳过，不能算模型失败。

### 4) 编程语料体检（`data/corpus_code_v2.jsonl` 1463 条）

来源：`code_fn` **774 条 52.9%**（主人项目 198 个文件抽的函数，含 20 个 `test_*.py` 的 122 条）｜`code_distill` 521 条 35.6%｜`bulk-*` 148 条 10.1%｜`code_paradigm` 20 条。
四个硬缺口：**输出 p50 仅 385 字**（片段级）｜**含 assert 仅 1.8%**（训练目标是「长得像代码」而评测目标是「跑得对」，两轴错位）｜**排错型仅 0.4%**（编程最高频场景空白）｜web/前端 2.1%、并发 3.0%。
干净的：零指令重复、holdout 泄漏 0 题。

---


### 病灶：质量轴恒定，退货闸门是漏的

精英档案 7 条 quality **全是 0.80**，`diverse()` 按质量排序等于随机。追根因：
`vid_verify` 只有 5 项检查（可解码 / 分辨率 / 有音轨 / 无黑场 / 无静音），**全是「能不能播」，没有一项看「画面对不对」**。而「有音轨」对 matplotlib 产物**恒为 0**（生成的 mp4 本来就没音轨）⇒ `quality = 4/5 = 0.80` 永远不变。

更严重的是：**一张静态 PNG 挂 4 秒也能满分通过这 5 项** —— 做梦的退货闸门根本不检查产物是不是动画。

### 标定（先量数据再定规则）

新增 `frame_delta()`：ffmpeg 抽 16 帧缩到 64×64 灰度，算相邻帧平均绝对差。用 `data/dreams` 里 7 个真实产物 + 1 个静态对照标定：

| 样本 | 帧间差 |
|---|---|
| 7 个真实 matplotlib 动画 | **2.588 ~ 2.629** |
| 首帧循环 4s 的静态对照 | **0.000** |

空档足够宽（0 vs 2.6），判定线取 **0.5**，满分区取 **2.6**。标定脚本固化为 `tools/calib_framedelta.py`（改阈值前重跑）。

### 落地

| 项 | 内容 |
|---|---|
| `frame_delta(ff, path, fps=4, size=64)` | 纯 ffmpeg rawvideo + 逐字节比较，无新依赖 |
| 第 6 项检查 `画面在动` | **hard=True** —— 「没动」不是质量低一点，而是「这不是动画」，扣分放行毫无意义（默认 `min-quality=0`，扣分拦不住） |
| 实测 | 真实动画 **PASS quality=0.833**（原恒定 0.80）；静态对照 **FAIL ❌**（原 PASS） |
| `triple_bank` 文档 | 旧写「8 项验收」，实测只有 5 项 → 更正为 6 项并记录原因 |

### 顺带：做梦轮询 30 分钟 → 10 分钟

`DreamLoop` 计划任务原 `PT30M`，锁被 p2 的 eval 占着时，每错过一次就要干等半小时。改 `PT10M`（`RepetitionDuration` 仍 `P3650D` 防 MaxValue 溢出）。

### 未完成

- ⏳ 历史 7 条的 `checks_json` 未重算（它们的帧间差都是 2.59~2.63，重算也拉不开差距，无意义）。新规则从下一轮做梦起生效。

---

## 2026-10-18 12:20-12:50 · 小游戏全量开源 + 本地清理；做梦线接上 few-shot 闭环


### 1) 20 款小游戏迁出本地 → `ghshhf/mini-games`

**开源**（已完成）：<https://github.com/ghshhf/mini-games> · 在线玩 <https://ghshhf.github.io/mini-games/>
20 款单文件 HTML5 游戏 + 索引页 + `check_refs.js`（38 处本地引用，缺失 0），MIT。

**本地删除**（`D:\tgapp\telegram_mini_app\games\`，26 文件 / 3.9MB → 目录 73M 降到 69M）：
删除前逐字节校验 **26/26 sha256 与仓库一致**，另打归档 `D:\code\mini-games-archive-gamesdir-20261018.tar.gz`（2.0MB / 26 文件）。

🔴 **删除范围与教训**：`back.js` 看着像游戏文件，实际是**全站统一返回组件**，被 **19 个应用页**引用（admin/chat/todo/novels/reports/upload…）。第一次误删后立刻从仓库恢复（sha256 `9f4d259e…` 校验一致）。
**规则：删任何文件前，必须在该目录内 grep 全量引用，不能按"文件分类"推断归属。**

连带清理的脏引用（全部按「磁盘上是否真实存在」过滤，不手写清单）：

| 位置 | 处理 |
|---|---|
| `sw.js` PRECACHE | 38 项 → **20 项**，剔除 18 个已删资源；缓存版本 **v81 → v82**（强制老客户端丢旧缓存） |
| `sw.js` | 剔除后逐项验证**缺失 0** |
| `registry.json` | `sections.games` **20 条 → 0**，保留 tools=9 / chats=3 / apps=8；新增 `_games_moved` 指路字段 |
| `lib_index.js` | 125 条 → **104 条**，剔除 21 条 `/games/*` 死链；复验 `/games/` 20 条全为应用页，**死链 0** |
| `home.html` | 内置 `FILES.games`/`TRAE_GAMES` 置空 + I18N 文案清空；`render()` 加空分支 → 游戏区换成「小游戏 · 已迁出 → 在线玩 20 款」（否则会留 20 张点不开的空卡片 + 3 个孤立标题） |
| `container.html` | `GAMES` 置空，空态文案改为在线版导流 |
| `guide.html` | 游戏清单清空；文案四处改口径（"目前 13 款"→"游戏去哪了"、footer 统计改为"工具 9 · 聊天 3 · 应用 8"） |

**验收（真跑）**：起本地 http 服务用浏览器实测 `home.html`（工具/聊天/应用三区全渲染，游戏区为金色导流链接，无残留装饰点）、`container.html`（"小游戏已迁出本地 / 在线玩 20 款 →"）、`guide.html`。另做两项硬检查：
- 全站本地引用扫描（`href|src` 指向的 .html/.js）→ **死链 0**
- 五个改动文件的 `<script>` 全抽出来 `node --check` → **语法错误 0**

**保留未删**：`brawl.html.bak`（51KB，brawl 旧版备份，未开源）；`assets/`（AI 加载图，被 PRECACHE 引用）。

### 2) 做梦线：精英档案终于接进 few-shot（此前只写不读）

**病灶**：精英档案建完一直在写，`retrieve_examples()` 只从三元组银行按任务相似度取 top-2 → 天然偏向同类任务，越喂越窄。MAP-Elites 的价值（保多样）完全没用上。

**改法**：`retrieve_examples(task, k=2, elite_k=1)`
- 银行 top-k 照旧
- **额外**从精英档案 `diverse()` 抽 1 条**不同技能**的示例（`avoid_skills={当前任务技能}`）
- 实测注入成功：余弦波任务 → 银行 2 条 + 精英「衰减振荡」1 条（438 字）

🔴 **踩到的坑（值得记）**：跨示例去重不能复用同桶判重阈值 `DUP_JACCARD=0.82`。不同技能的 matplotlib 动画**骨架**本来就像（import/FuncAnimation/save 三段样板就超 0.82），第一版把所有候选全误杀，一条都注不进去。改成**两档**：同桶判重 0.82（防同一解重画），跨示例去重 **0.95**（只拦逐字节级重复；实测 #6/#8 J=1.000 拦下、sin/cos J=0.802 放过）。

### 3) 🔴 揪出 RSI 燃料污染（本轮最重要的发现）

体检银行 9 条时发现 **#5 的 `code_text` 开头是**：
```
# 参考过往成功写法(仅供参考风格，不要照抄):
# --- example 1 ---
import matplotlib...
```
即：**模型把 prompt 里的 few-shot 复读进了代码，而 `vid_verify` 只看「有没有产出视频」→ 放行入库**。这份污染代码又被 `backfill` 写成精英档案 #6，下一轮当 few-shot 喂回去 —— **污染会自我放大**：复读 → 入库 → 更强复读。

**修法（双重闸门 + 数据清理）**：

| 项 | 内容 |
|---|---|
| `dream_loop.check_pollution()` | 纯字符串匹配 5 类指纹：`参考过往成功写法` / `# --- example` / `example N ---`(正则) / `<\|im_start\|>` / `<\|im_end\|>` |
| 闸门位置 ① | `one_dream()` 里 **`extract_code` 之后、`run_code` 之前** → 命中直接 REJECT，省掉 90s 白跑视频 |
| 闸门位置 ② | `accept()` 开头再拦一道（它也可能被别的路径调用） |
| `EliteArchive.diverse(min_len=120)` | 低于 120 字的代码不进 few-shot。实测坑：冒烟测试的 `print('DONE')`（35 字）占了一个桶，质量并列时按 id 排最前，**会被优先选中喂给模型** |
| 数据清理 | 银行删 **#2 冒烟测试（35 字）** 与 **#5 污染** → 9 条降 7 条；`backfill` 重建精英档案 → **5 条 / 5 桶**（原 7 条 / 6 桶，其中「其他」桶就是那条 35 字垃圾） |

**自检扩到 28/28**：新增 D 段 4 例（长度门槛 / 复读闸门标题 / 干净代码不误杀 / N 号示例），改桶规则或闸门后必跑 `tools/dream_elite_selftest.py`。

### 未完成 / 待拍板

- ⏳ **真做梦一轮验证**：`p2` 的 step3 四路 eval（bare/mounted × 0.5B/1.5B）正在跑，训练锁仍被占（12:31 时 6/20）。第一定律下做梦必须让路，DreamLoop 每 30 分钟轮询，锁一松自己会跑。
- ⏳ **DreamLoop 仍指向 `checkpoint-300`**，而 p2 已训到 732 —— 换哪个不能拍脑袋（6ep 有"把指令能力冲成复读"前科），等四路 eval 数字出来按 `data/code_holdout_v1.jsonl` 对照再定。
- ⏳ 银行 7 条全是 matplotlib 动画（5 个桶但同一技术轴），真正提升多样性得靠跑新模板（生命游戏/排序可视化/抛体反弹）。

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
