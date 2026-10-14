# 远程算力运维手册（node1 / ECS / GPU）
> 从 MEMORY.md 下沉而来（2026-10-14）。操作细节、坑、脚本名都在这里。

## 一、算力盘点
| 资源 | 规格 | 状态 |
|---|---|---|
| 本机 | 12 核 CPU / 16.9GB / torch 2.14.1+cpu | 可用，但算力仅 node1 的 ~1/13 |
| **node1（魔搭 DSW 方式一）** | 免费 64 核 CPU / 长期 | **主力**，实测 18.7~29.1 s/步（214M） |
| 魔搭方式二 | GPU 24G | 剩 35h44m |
| **魔搭方式三（AMD GPU）** | 200G 显存 / 192G 内存，rocm7.3+py312+torch2.12 | **剩 100h，战略储备** |
| 阿里云试用 ECS | 杭州 47.99.131.12，2C4G，Windows Server 2022 | 控制面，≈烧到 11 月中 |

**用户决策**：CPU 免费时段继续跑，GPU 100h 作战略储备（扩容 0.3B/0.5B 重训 / CPU 撑不住时接力），不烧在 CPU 能干的活上。

## 二、控制面与数据面
- ECS `47.99.131.12:8081` = `mailserver.ps1`（信箱）；`/file/` = `C:\llama\files\` 动态目录（云助手放文件即上线）。
- 控制面：ECS `/task` 派活 → 节点轮询执行 → `/upload` 回传 ECS `/file/`。**localhost 隧道已失效（返 404），直连 8081。**
- 数据面：原 OSS 签名 URL（凭证不出本机）；OSS 欠费后改 ECS `/upload` 分块中继。
- node2 走**信箱寻址**（`n2:` 前缀），避开 node1 的 `/task` 共享队列。
- 云助手远程执行：`ecs_cmd.py`（PowerShell 类型，CommandContent 要 base64）。

## 三、🔴 坑清单（每条都是实机踩出来的）
1. **ECS 同名文件锁**：上传被中断会留孤儿进程持锁，同名再传报「文件正由另一进程使用」；复用旧名还会**读到 20 分钟前的旧快照（一字不差，极易误判）**。**输出文件名必须持续轮换**，且勿中断上传（单分片 95MB≈110s，用 `-m300`）。
2. **远程 torch<2.6 + 新版 transformers 拒绝 `torch.load`**（pickle CVE-2025-32434）→ 用 `node/convert_st.py` 转 `model.safetensors`；**tied weights（embed/lm_head 共享内存）会让 safetensors 拒存，须每个张量 `.clone()` 打破共享**。另：新版 transformers 把 `extra_special_tokens`(list) 当 dict 崩 → `_safe_load_tokenizer()` 归一化成 `{}`。
3. **CRLF**：本地写的 `.sh` 是 CRLF，node1 的 `/bin/sh` 是 dash，把行尾 `\r` 当普通字符 → `Syntax error: newline unexpected`。本机 Git Bash 的 `sh -n`（bash）放行，本地校验不出来 → `n1_push.py` 已加「文本文件（无 NUL）上传前 CRLF→LF 归一化」。
4. **dash 子 shell 继承父进程 cmdline**：`$(...)`/管道的子 shell 会被守卫当成「外部链」而自我死等 4h；且事后读 `/proc/<pid>/status` 判祖先会因子进程被 init 收养（ppid=1）误判 → 必须用**同一次 `ps -eo pid,ppid,cmd` 快照**，在同一份 pid→ppid 映射里回溯，全交给 awk 一次算完。另：POSIX sh 无 `local`，辅助函数循环变量会覆盖调用方同名变量。
5. **重启清空 /root**：NAS（`/mnt/workspace/...`）才持久，`/mnt/data` 与 `/mnt/workspace` 同盘（同 inode）。
6. **node2（AI Studio 天数智芯 BI-V150S）**：① upload 参数不能带 `/upload` 路径（脚本自己拼，带了→404）；② `save_ckpt` 缩进 bug 曾导致只存 lm_head（已修 v2）。

## 四、训练链（chain_v1.sh / chain_v2.sh）六道防线
针对「双链互删 run_tmp 白跑 2h」事故加固：
1. **guard 单实例**——已有链存活则**排队等待接管**，非报错退出
2. **trainer_alive 死等**——绝不删正在写的 run_tmp
3. **salvage**——先接管未 promote 的残留 ckpt
4. **ckpt_ok**——权重 <100MB 拒绝 promote
5. **maybe_upgrade**——每 chunk 边界拉新版、`sh -n` 校验后 `exec` 热升级（**PID 不变，改脚本永不需重启链**）
6. 失败退避 600s
另 `chain_watch.sh` 看门狗（node1 无守护者：链父进程=PID1，崩了没人拉）。

**当前链**：R4 → R5（done 2800）→ R6（done 6800，lr 2e-4）全自动，之后自动进 0.3B。
标准 200M 基座分片 `b200_00..09` 已在 ECS `/file/`；`bootstrap_v1.sh` 拉回 NAS `$W/r4base_std` 再 RESET 重训。

## 五、loss 口径（血泪）
- **「R4 无效」已结案（2026-10-13）：是度量口径 bug，不是模型/数据问题。**
  ① **标签双重 shift**：`input_ids=batch[:,:-1]` + `labels=batch[:,1:]`，HF CausalLM 内部已 shift 再手动移一次 → 目标错位一格。正确 = `x=batch[:,:-1]; model(input_ids=x, labels=x)`。
  ② **loss 打印被放大 accum(=8) 倍**：`loss_acc` 累计 80 个微批却除以 10，正确 `/(10*accum)`。
  修正后 node1 step10 打印 16.6961 → 真值 **2.087**，本地 train_log step25 **2.0887**，完全吻合。
- 🔴 **铁律：远程 loss 异常，先拿本地同一权重跑「同口径 holdout CE + 贪心生成」对照，别先怀疑权重。**
  本地验收脚本 `quick_ce.py`（口径已修正）；远程同款 `node/base_check.py`。
- GPU 侧曾出 `ckpt_100.npz`：273 键全对、与本地余弦 0.994，但 CE 6.52 vs 本地 1.57、生成全是字节碎片 → **判废**。根因是 GPU 训练起步 loss 34.8（应为 ~2.1）。

## 六、权重分发与备份
- GitHub 私有 `ghshhf/smollm2-local-cpt`：代码 131 件已推；0.1B 权重分片（`backups/0.1B/`，8 片 ≤99.6MB + MANIFEST + `reassemble.py`）已 commit（a189f33）待 push。权重/语料入 `.gitignore`；未来自动备份改 SSH deploy key 免明文 PAT。
- 0.2B 标准基座：本地 `out_smollm2_200m_pt/checkpoint-latest/pytorch_model.bin`（856,674,211 B）。
