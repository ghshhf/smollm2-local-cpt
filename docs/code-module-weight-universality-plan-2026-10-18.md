# 编程模块·权重通用 更新计划

日期：2026-10-18
状态：**计划已定稿，待用户一声"开始"执行 P2**

## 〇、愿景路线图（三层，用户 06:32 明确）

本计划（P1–P5）只是 **第一层** 的落地验证。用户的完整愿望是三层递进：

1. **层一 · 权重通用（当前在做）**：同一份训出的模块权重，挂到不同底座都能工作。本计划 P2 即证明编程模块（浅格式行为）的权重能否跨底座迁移。
2. **层二 · 底座通用（下一步）**：模块对底座架构无关——跨家族/跨架构底座无需逐底座重训即可挂载，逼近"插上就用"。机制方向：规范残差空间行为向量 + 薄投影（H→d）/ 多底座联合训练，使行为本身与 hidden/架构解耦。
3. **层三 · 类 MCP / SK 技能 的通用标准（终极愿望）**：本开源项目成为通用模块化标准生态——`interface.json` 即"挂载协议"（类比 MCP 的协议、SK 的 skill 包），任何兼容底座可即插即用模块。已具备种子：`interface.json` 的 `attach_points`(只恒维) + `forbidden_points`(禁扩维) + `module_loader.py` 的零填充/截断/丢弃 remap，已是"协议"雏形。

> 一句话：权重通用是地基，底座通用是 generalization，类 MCP 是生态形态。当前 P2 不动层二/层三，先把层一证出来。

## 一、目标钉死（相对上一轮的两处纠正）

1. **"通用"= 权重通用，不是包通用。**
   - 上一轮误理解成"接口/数据/评测/配方复用，权重按底座各自训"——错。
   - 真目标：**同一份训出的模块权重，挂到不同底座上都工作**。这才是模块化那条路的兑现点。
2. **底座不能用咱们的 0.3B，必须换开源代码模型。**
   - 0.3B 预训练语料 = Ultra-FineWeb-zh 中文网页，**无代码**；基线实测裸 0.3B 代码块 0% / 内容 0%。
   - 在它上面训模块，内容层必为 0，等于白训。
   - 模块化兑现点 = "挂到 Qwen 这类有代码的底座，代码知识全是底座的，模块只补浅格式行为"。
3. **"编辑模块" = 编程模块**（笔误）。

## 二、作用域（已拍板）

- **B：跨家族含 Qwen（hidden=896）** —— 匹配"权重通用"真目标，含你点名的千问。
- **训练底座：Qwen2.5-0.5B-Instruct**（本机已有 `D:/work/_local_ai/models/models/Qwen--Qwen2.5-0.5B-Instruct/snapshots/master`，hidden 896 / 24层 / kv128 / inter4864 / vocab151936，零下载）。
- **remap 目标（同族异尺寸）：Qwen2.5-1.5B-Instruct**（本机已有 `D:/work/_local_ai/train_15b/model/Qwen2.5-1.5B-Instruct`，hidden 1536 / 28层 / kv256 / qwen2，零下载）。即 P2 要证"跨尺寸权重通用"的那块试金石，已就位。
- **开源边界（发动机 / 手机壳 铁律，用户 06:53 钉死）：**
  - **北极星（底座引擎）= 不开源，是咱们自己的模型。** 这是「发动机」，绝不外传权重。
  - **通用模块（手机壳 / 屏幕 / 软件）= 开源。** 模块包含：LoRA 行为权重 + `interface.json` + tools + eval + 脱敏数据。**模块的 LoRA 权重随包一起开源**（它是「软件」不是「发动机」）；不开源的只有底座引擎（北极星）本身。
  - 机制自洽：模块只在恒定维（q/k/v/o）表达「开口方式」，知识在底座，所以开源模块不泄露引擎能力，别人拿自己的底座（如 Qwen）即可即插即用。

## 三、为什么编程权重比对话权重更可能"通用"（机制）

| | 要迁移的行为 | 深浅 | 跨家族直搬 |
|---|---|---|---|
| 对话模块 | 说话风格/认主/知识深度 | 深，绑表征 | ❌ FAIL（已证） |
| 编程模块 | 按协议吐代码块 + 中文说明 + 规矩 + 认主 | 浅，表层格式 | 🟡 待测（大概率可行） |

裸 Qwen 格式已 100%（代码块/语言/收尾），格式是浅行为，**可能跨家族也能搬**。这是本计划要攻的点，不是死路。

## 四、执行步骤

### P1 模块锁恒定维 —— ✅ 已完成（结构前提）
- `modules/polaris-code-zh/interface.json` 的 `attach_points` 仅 q/k/v/o，全标 `"class":"constant"`；`forbidden_points` 禁挂 gate/up/down/embed/lm_head。
- `tools/module_loader.py` 已具备跨 hidden 零填充(kept/padded) / SVD 截断(truncated) / 丢弃(dropped) 全套 remap（576→896 已实测能跑）。
- 待修订：`interface.json` 里 `portability.L2_cross_lineage` 仍写"对话模块 FAIL"——编程模块要推翻需实测（在 P2 后改）。

### P2 判官实验（核心，权重通用成不成立全看它）
1. 在 **Qwen2.5-0.5B-Instruct** 上训满一份编程 LoRA（数据=`data/code_sft_v1.jsonl` 全量 1463 条，只挂 q/k/v/o，1~2 epoch）。
2. 用 `tools/module_loader.py` 零填充 remap 到**同族不同尺寸代码底座**（hidden 896↔1536，如 0.5B↔1.5B）。
3. 跑 `tools/code_eval.py`：裸 vs 挂载，看 **format 层**（代码块/语言/收尾）+ **content 层**（真执行比对）是否保住。
4. 判定：format 保住 → **权重通用成立**；不过 → 进 P3。
- 注：你刚叫停训练，P2 的 Qwen 训练需你批准再开。

### P3 若 P2 失败：攻"真·通用权重"（跨 hidden 也通用）
- a：行为编码进**规范残差空间**（小 d 维行为向量），每底座一层薄投影 H→d 接入；行为向量本身与 hidden 无关 = 真通用，薄投影算底座适配器。
- b：多底座联合训练（一份权重在 Qwen 多尺寸共优化行为）→ 向底座无关行为收敛。
- 为什么 P3 是后备：P2 大概率能成（格式浅），先证 P2 再谈 P3。

### P4 验收门（诚实，不套旧门槛）
- 每个目标底座：裸 vs 挂载，`code_eval` 20 题，**format + content 双指标**。
- **权重通用 = 至少 2 个异构底座上 format 保住且 content 不退化。**

### P5 更新模块包 + 开源（开源「模块」含其 LoRA 权重；底座引擎 北极星 永不开源）
- 把编程专属"权重通用"实测结论写进 `interface.json`（推翻旧 L2 FAIL 写法），标注已验证通用底座清单。
- 开源到 `ghshhf/polaris-code-module`（或新建 `polaris-code-module`），含 interface.json + tools + eval + 脱敏数据 **+ 训练出的 `adapter/`（模块 LoRA 即「软件」，随包开源）**。
- **铁律：北极星底座引擎权重绝不推、绝不外传**；开源出去的只有模块（行为层），别人用自己的底座挂载即用。

## 五、当前真实状态（已核实）

- 本地训练已全停（你叫停的 4 个 `train_lora.py` 已杀，仅剩 Windows-MCP 基础设施进程）。
- 启动前提已核：venv python、`train_lora.py`、Qwen2.5-0.5B-Instruct 快照、`code_sft_v1.jsonl` 1463 条 —— 均在。
- `modules/polaris-code-zh/` 包已齐：interface.json + tools + eval + `code_sft_public.jsonl`(1407 条脱敏)。
- **remap 目标 Qwen2.5-1.5B-Instruct 已在本机**（`train_15b/model/Qwen2.5-1.5B-Instruct`，hidden 1536 / 28层 / qwen2）→ P2 全程**零外部下载**，不需要拉 1.5B；两个 Qwen 底座（0.5B hidden 896 + 1.5B hidden 1536）均就位。
- **未启动任何新训练**。

## 六、训练纪律（用户 06:53 钉死：一次只能跑一个训练）

🔴 **铁律：本机同一时刻只允许一个训练进程在跑。** 过去一次拉起多个 `train_lora.py` 被用户叫停，不能重犯。
- 落实机制：`tools/train_queue.py`（单一插槽锁 `.train_lock` + 队列 `.train_queue.json`）。所有训练必须经由 `schtasks /ru SYSTEM` 调起的 dispatcher 发起，dispatcher 跑前原子抢锁、跑完释放；锁被别的进程占着就排队等，从根上杜绝并发训练。
- P2 实操：把 `run_p2_pipeline.bat`（train→remap→eval 一条龙）作为**一个**队列任务提交，dispatcher 顺序执行；整个管线里只有 step1 是训练，故天然满足「一次一个」。未来若有多条训练，全部 submit 进队列，绝不直接裸跑 `train_lora.py`。

## 七、下一步（已获「全自动搞」授权，按排队机启动 P2）

- 提交 P2 管线到排队机 → schtasks `CodeLoraP2`（SYSTEM 常驻）拉起 dispatcher；本机长进程不能跑 Bash 后台，会被会话清扫杀。
- 跑完 → 看 `logs/code_eval_codeqwen15b_remap.json` 的 format 层是否保住（=权重通用成立）。
- 结论 → 更新 `interface.json` + 开源模块包（含 LoRA 权重，北极星引擎不开源）。
