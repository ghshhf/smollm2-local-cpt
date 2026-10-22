# 魔搭（ModelScope）在本项目里的完整用法

> 2026-10-19 立。起因：反复把"平台内已登录"当成"站外匿名机器"，导致建公开仓、找 token、四次重打包、
> 让用户在 Notebook 里反复试命令。本文件是唯一口径，与它冲突的临时判断一律作废。

## 零、第 0 条原则：先看本地有什么，再谈传输（违反次数最多的一条）

用户反复说过：**"服务器就相当于硬盘"、"东西都在本地"**。事实也确实如此——
`/mnt/workspace/model` 下 `模型`（21 天前）、`节点1`（8 天前）、`PKG` 早就存在，
0.2B/0.3B 权重、语料、tokenizer 本来就在 NAS 上。

所以任何"下载 / 上传 / 打包 / 中转"动作之前，**必须先做只读盘点**：

1. `ls /mnt/workspace/`、`ls /mnt/workspace/model/`、`ls /mnt/workspace/node1/`
2. `find /mnt/workspace -name "*.safetensors" -size +100M`（权重）
3. `find /mnt/workspace -name "*.jsonl" -size +10M`（语料）
4. `ps -eo pid,cmd | grep -E "node_v6|dream|train"`（进程）

**本地有 → 直接用，一步都不许传。** 本地没有 → 才谈仓库拉取。
"我以为要传"不是理由；没盘点就动手，等于把用户说过 N 遍的话当没听见。

## 一、魔搭在这个项目里是四个东西，不是一个

| 角色 | 具体是什么 | 关键事实 |
|---|---|---|
| **硬盘**（真相源） | 私有数据集 `hhhlhuk/polaris-relay`、模型仓 `hhhlhuk/polaris-03b` | OSS 底座。**实例被回收也不丢**，这是唯一可信的持久存储 |
| **算力** | node1 = 免费 CPU 实例（cgroup **8 核**，`nproc` 报 64 是假象） | 另有 GPU 24G（剩 ~35h）、AMD 200G 显存（100h，战略储备，不动） |
| **持久盘** | `/mnt/workspace`（NAS） | 实例重启不丢；**`/root` 重启即清空**，凭证/脚本都别放那儿 |
| **平台内登录态** | 实例内的 modelscope SDK 环境 | 平台已注入登录态，**实例内拉私有仓不用带 token** |

## 二、最重要的一条：登录态分两种，别搞混

| 执行位置 | 要不要 token | 怎么写 |
|---|---|---|
| **魔搭实例内部**（node1 / Notebook 终端） | **不要** | `snapshot_download('hhhlhuk/polaris-03b', local_dir='./base')` 直接就能下 |
| **本机 Windows** | 要 | `tools/ms_relay.py` 自己按 `env → deploy_aliyun/ms_token.txt（36位UUID）→ ~/.modelscope/credentials/{token,access_token,session}` 逐试 |

- `~/.modelscope/credentials/session` 里那个 **32 位 hex 已失效**（E3001）；只有 36 位 UUID 有效。
- 🔴 **禁止**为了绕鉴权去新建公开仓库。站内不存在鉴权问题；站外问题用 `ms_token.txt`。
- 🔴 **禁止**让用户在 Notebook 里手动登录或粘 token。他是平台用户，登录态是平台给的。

## 三、仓库清单（2026-10-19 实查）

| 仓库 | 类型 | 内容 |
|---|---|---|
| `hhhlhuk/polaris-relay` | 数据集（私有） | `corpus/ufw_zh/`(4片 3.7GB)、`inbox/`(清洗分片、启动包)、`outbox/`(poetry_zh.jsonl 53MB) |
| `hhhlhuk/polaris-03b` | 模型（私有） | 0.3B 全参 fp32 1.39GB + `step.txt` |
| `hhhlhuk/polaris-03b-sft1` | 模型（私有） | 0.3B fp16 693MB，36层/576hid/vocab49152，无 README 无 step（来源待确认） |
| `hhhlhuk/qwen3-4b-novel-lora` | 模型 | 小说线 LoRA |
| `hhhlhuk/polaris-boot` | 数据集（**公开，我误建的**） | `sft_boot.tgz`，API 删不掉（E3021），需网页删 |

## 四、三条通道（本机 ↔ node1），按用途选，别混用

| 通道 | 入口 | 用途 | 速度 |
|---|---|---|---|
| **隧道**（控制面） | `node/n1.sh` | 多行命令、`-f` 推脚本、`--put/--get` 小文件（<2MB） | 秒级 |
| **信箱**（兜底） | `node/mb.py` | 隧道全挂时的单行命令；`/task` 派活 | 4~15s，含 ECS 带宽瓶颈 94KB/s |
| **魔搭仓库**（数据面） | `tools/ms_relay.py` | 大文件（>5MB）**一律走这条** | 本机上传 5.62MB/s、node1 拉 3.05MB/s、模型仓 13.6MB/s |

- 大文件**禁止**走 scp/信箱：ECS 只有 94KB/s，1.3GB 要 4 小时。
- 并发 Range 分片**不提速**（实测 1.85MB/s < 单流 3MB/s），瓶颈在账号/IP 限速，别加并发。
- node1 侧拉取器是 `node/node_v6.py`（1s 轮询 + long-poll），它活着信箱才有回传。

## 五、实例被回收后的恢复顺序（这是常态，不是故障）

魔搭免费实例会被回收，容器内进程全消失，但 `/mnt/workspace` 还在。恢复只有一步：

```bash
curl -s -o /mnt/workspace/node1/bootstrap_all.py "http://47.99.131.12:8081/file/bootstrap_all.py?key=k0u4mail" \
  && cd /mnt/workspace/node1 && python3 bootstrap_all.py
```

`bootstrap_all.py` 幂等，负责拉起：`node_v6`（信箱拉取器）、训练链看门狗、做梦集群、自进化闭环；
并会顺手 `tunnel_up.sh` 把 SSH 反向隧道挂回来（所以恢复后 `n1.sh` 才能用）。

- 隧道单测：`bash node/n1.sh --alive`
- 隧道状态（在 node1 上）：`bash /mnt/workspace/node1/dream/tunnel_up.sh --status`
- 判断"进程在不在"看 `ps -eo pid,cmd | grep node_v6`，**不要看心跳**。

## 六、踩过的坑（每条都实机付过代价）

1. **`nproc` 骗人**：报 64，cgroup 实际 8 核。`THREADS=$(nproc)` 会 8 倍超订。要读 `/sys/fs/cgroup/cpu.max`。
2. **Notebook 单元格不是终端**：写 `python3 -c "..."` 会 `SyntaxError`。单元格里要么纯 Python，要么加 `!` 前缀。
3. **一次只给一个单元格**：拆成"先跑这段再跑那段"，用户只会跑最后那段 → `FileNotFoundError`。
4. **站内跑脚本用 `sys.executable`**：系统 `python3` 可能没装 torch。
5. **别反复打包重推**：先看清仓库里已有什么、NAS 上已有什么，能复用就不传。
6. **CRLF**：本机写的 `.sh` 是 CRLF，node1 的 `/bin/sh` 是 dash → `Syntax error: newline unexpected`。`n1_push.py` 已自动归一化。
7. **pkill 自杀**：同一条命令里出现 pkill 模式串和目标串，会杀掉自己。用变量拼接。
8. **node1 只吃单行命令**（信箱路径）：多行会静默截断，要 `{ ...; } 2>&1 | curl ...` 分组。
