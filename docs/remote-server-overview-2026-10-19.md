# 远程 node1 全面改造记录（2026-10-19）

> 本文是这一轮对远程服务器的**总账**：它到底是什么、能干什么、我们改了什么、还剩什么。
> 细节分册见 `docs/remote-server-plan-2026-10-19.md`（通道/性能）、`docs/remote-inventory-2026-10-19.md`（仓库盘点）、`docs/remote-index.md`（目录索引）。

---

## 一、一句话

远程从"一台只能靠贴命令盲摸的免费容器"，变成了**可直连、有索引、开机自起、线程自适应**的常备算力节点；
但它的 **8 核是上限不是保底**，实际稳定算力约 **4 核**（忙时 3.2 核），这是本文最需要记住的一条。

---

## 二、系统画像（实测）

| 项 | 值 |
|---|---|
| 系统 | **Ubuntu 22.04.5 LTS**，kernel `5.10.134-18.0.12.lifsea8.x86_64` |
| 形态 | **K8s Pod**（`kubepods/burstable/...`），魔搭 DSW，**不是虚拟机、没有 systemd** |
| 身份 | root |
| 磁盘 | `/mnt/workspace` = 阿里云 NAS 挂载，1.0P，已用 49T（5%），可用 976T |
| 存储性能 | stat **1.0~1.3µs**、打开+读 64B **4.1µs**（本机 Windows 分别是 4.8µs / 111.8µs） |
| GPU | 无（CPU 实例） |
| Python | torch 2.3.1+cpu、transformers 4.55.4 |
| sshd | OpenSSH 8.9p1，`0.0.0.0:22` |
| 开机行为 | 容器重建，`/mnt/workspace` 保留；**任何进程都要自己拉起** |

---

## 三、免费 CPU 情况（重点更新）

### 3.1 标称 vs 实际

| 口径 | 数值 | 说明 |
|---|---|---|
| `nproc`（早期记录） | 64 | **假象**，读到宿主机全部核 |
| `os.cpu_count()` / `sched_getaffinity` / `/proc/cpuinfo` | 8 / 8 / 8 | 容器视角，三处一致 |
| cgroup `cpu.cfs_quota_us / period_us` | 800000 / 100000 = **8 核** | 配额上限 |
| `cpu.stat` nr_throttled | 整轮测试仅 +1 次、1.0ms | **不是 cgroup 在限流** |
| `loadavg`（测试时） | 0.04 | 不是自己抢自己 |
| **并行加速比实测** | 1 进程 1.00 核、4 进程 **3.86~4.00** 核、8 进程 **3.11~3.20** 核 | **真瓶颈在宿主机超卖** |

**结论：配额 8 核，稳定可用 ≈ 4 核（忙时 3.2 核）。规划容量按 4 核算，不要按 8 核算。**

### 3.2 为什么是 burstable

`/proc/1/cgroup` 显示 QoS 类为 `burstable` —— 8 核是"最多能用多少"，不是"保证给多少"。
宿主机空闲时能摸到 4 核以上，被人占用时掉到 3.2 核。**所以线程数必须现场标定，不能写死。**

### 3.3 现场 A/B（同模型同数据，只改线程）

| `--threads` | 步速 | 吞吐 |
|---|---|---|
| 7 | 8.23~8.72 s/步 | 110~114 tok/s |
| 3 | 8.06~8.54 s/步 | **132~133 tok/s** |

**线程开多反而慢 16%** —— 这就是"按配额开线程"的直接代价。

### 3.4 已改为自适应

`tools/cpu_probe.py`：每个进程做等量算术，看多进程总耗时是否恒定，
`有效核数 = n × t(1) / t(n)`，取各档最大值四舍五入。

- `sft_full.py --threads auto`（现为默认）：开跑前标定一次，约 **0.5 秒**；
- `boot_resume.py`：优先读环境变量 `POLARIS_THREADS`，否则用标定值；
- `OMP_NUM_THREADS` / `MKL_NUM_THREADS` 同步跟随。

实测输出：`{"quota": 8, "effective": 4, "detail": [1→1.00, 4→3.86, 8→3.11]}`

### 3.5 与本机对比（同为 3~4 核算力）

| | 吞吐 |
|---|---|
| 远程（4 线程） | **132 tok/s** |
| 本机 i5-12450H（6 线程，8C12T） | 106 tok/s |

**远程快约 25%**：核数打折仍然赢，赢在 Linux（文件打开快 27 倍）+ 无 Windows 进程税 + 无移动端功耗墙。

---

## 四、通道：从 2.2 秒到 0.36 秒

| 路径 | 单命令 |
|---|---|
| 信箱 mailbox（原唯一路径） | 2.2~3.5s |
| ssh 单跳到 ECS（一台"正常服务器"） | 760~810ms |
| ssh 两跳到 node1（ProxyJump） | 1190~1430ms |
| **`rc.py`（常驻守护）** | **360ms** |
| **`rc.py shell`（常驻会话）** | **140ms/条** |
| `rc.py idx`（离线索引） | ~0ms |

链路：`本机 rc.py →(TCP 19122)→ chd.py(SYSTEM 计划任务) →SSH→ ECS(Windows) →direct-tcpip→ 9022 →反向隧道→ node1:22`
物理下限 45ms（本机→ECS 35ms + ECS→node1 6.4ms）。

关键坑（都已修，写进代码注释）：`TCP_NODELAY` 省掉 200ms Nagle 延迟、多行命令必须折成单行、`pkill -f` 会杀到自己、`mb.py` 只把最后一条命令接进管道。

### 4.1 传文件为什么也快（`rc.py put/get`）

不走魔搭/ECS 中转，直接在**常驻 SSH 通道**上做**分块 base64**：

```
本机文件 → 切 256KB 块 → base64 → 每块一条命令写进 /tmp/_rcput_xxx/p0001
        → 全部写完 → cat p* > 目标 → md5sum 校验 → 删临时目录
```

实测 **737KB/s**（1.8MB 的 `aishell_pairs.jsonl` 用了 2.4 秒）。
对照：ECS 中转那条路只有 947KB/s 且会断流，所以脚本级文件一律走 `put`。

**为什么音频不需要传**：911M 的 `data/audio/wav` 本来就在 NAS 上（`/mnt/workspace` 是挂载盘），
远程直接读，零传输。真正要补的只有 1.8MB 的 pairs 表——传输 2.4 秒，
后面 100 分钟全是**远程本地算力**在跑特征提取，跟网速无关。

---

## 五、自启动（开机不用管）

| 端 | 机制 | 文件 |
|---|---|---|
| 远程 | 隧道看门狗，断了 5 秒重连 | `node/tunnel.sh`（密钥 `/root/.ssh/n1t2`） |
| 远程 | 每次开机自检确保隧道在跑、按标定值起训练 | `node/boot_resume.py` |
| 远程 | 信箱消费者 `node_v6.py` | `node/bootstrap_all.py` |
| 本地 | 计划任务 `polaris-chd`（`/sc onstart /ru SYSTEM`） | `node/chd.py` |

---

## 六、仓库整理

| 动作 | 结果 |
|---|---|
| 清理纯垃圾/搬运副本 | **-2.91G**（`_pack_siglip` 2.9G 是同一份 siglip 的 tar+38 分片） |
| 历史代资产归档（不删） | `archive/2026-10-19/` 共 **15G**：legacy_qwen 3.9G、legacy_chain 6.9G、legacy_cache 2.2G、legacy_base 1.3G、legacy_smollm135 260M |
| 当前 `models/` | 只剩 `sft_code_03b`、`sft_seed200_remote`、`siglip2-base`、`whisper-small` |
| 索引 | `rc.py sync` → 719 条记录，本地 `data/remote_index.json` + `docs/remote-index.md`，**远程关机也能查** |

`rc.py` 命令：`ls / tree / du / cat / stat / find / ps / svc / sh / sync / idx / md / shell`。

---

## 七、训练与产粮

| 项 | 状态 |
|---|---|
| 代码线远程训练 | `models/sft_seed200_remote` @ step 100（200 种子 × 2 epoch），15.3 分钟 |
| 本机对照 | `models/sft_seed200_v4` @ step 100，33.6 分钟 |
| 训练线程 | 已改自适应（实测 4 线程） |
| 做梦产粮 | `dream_code.py` 批次 1 已在跑：45 道衍生题 × 2 候选，产物 `data/dream_code_batch1.jsonl` |
| 银行 | `triples` 201 条 / `rejects` 26 条（产粮会继续累积，满 200 新条触发下一轮训练） |

**实测接受率（已修正）**：批次 1 跑到 34/45 题（14:08 手动停，剩余 11 题未跑），
**ACCEPT 9 / REJECT 25 ≈ 26%**，明细 **8×T0（可运行、结果未核对）+ 1×T2（精确命中 oracle）**，
银行 `triples` 201 → **210**、`rejects` 26 → **49**。

**自我修正**：前 5 题恰好连续全灭（`def j_g bt_g(p, p, p, p, T)` 这类语法乱码），
我当时据此判成"0 接受率、代码线不能自产自销"，**这个结论已收回**。
0.3B 能出能跑的码，只是内容层薄——绝大部分只到 T0（跑得通、答案没核对）。
正确表述是：**代码线低速自给可行（≈26%），深度仍得靠检索/更大底座**，不是完全走不通。

---

## 七之二、抗回收：开机自己起，只有手动能关

node1 是 K8s Pod，**没有 systemd、没有 cron、没有 at**（PID1 是 `tini → entrypoint-main`）。
所以"自启动"只能靠两层自己的进程管理 + 一个官方钩子。

| 层 | 文件 | 作用 |
|---|---|---|
| 总管 | `node/supervisor.py` | 每 30s 扫 `/proc/*/cmdline`（不用 pgrep，避免引号/自杀坑），任务不在就拉起，失败指数退避（15s→300s 封顶） |
| 外壳 | `node/sup.sh` | `start / stop / status / guard`，幂等；`guard` 是外层环，每 60s 确保总管活着 |
| 开机钩子 | `/etc/dsw/bin/prestart/99_polaris_start.sh` | `entrypoint-main` 会按文件名顺序 `source` `prestart/*_*` —— 官方留的钩子；先等 NAS 最多 180s，再拉起 `sup.sh guard` |
| 兜底 | `/root/.bashrc` | 手动开终端也顺手确保总管在跑 |

**唯一停机开关是 `touch /mnt/workspace/node1/STOP`**（等价 `bash node/sup.sh stop`）：
总管看到 STOP 就把作业全停并自己退出，删掉 STOP 再 `sup.sh start` 就恢复。

🔴 一个设计细节：隧道任务带 `protect: true`，**STOP 时不剪**。
它是控制通道（氧气管），剪了就再也够不着机器，只能人工贴命令——正是要消灭的那件事。

实测三件事（都过了）：

| 测什么 | 做法 | 结果 |
|---|---|---|
| 自愈 | kill 掉一个守护任务 | 30~40s 内自动拉起（新 PID） |
| 手动停 | `touch STOP` | 总管退出、作业停、**隧道仍活**（通道没断，能远程恢复） |
| 开机钩子 | `source /etc/dsw/bin/prestart/99_polaris_start.sh` | 总管被拉起 |

**诚实的边界**：平台调度器**硬回收**（配额到期/被删 pod）无法从容器内部抵抗，
root 也不行——那需要平台 API 权限。能控制的是两件事：
① 不因"空闲"被回收（心跳 + 打 Jupyter `127.0.0.1:8088` + CPU 常年不闲）；
② 真被回收后**零手工恢复**（rootfs 在 cloudDisk `/mnt/systemDisk`、数据在 NAS，
`/etc` 钩子跟着实例回来）。下一次重启可验证 `/etc` 是否真的保留。

---

## 七之三、设备自适应：一份代码跑遍所有机器

用户要求"以后所有训练都自适应，就一行代码的事"。已落成 `tools/dev_pick.py`：

```python
from dev_pick import setup
device, n_threads = setup()      # cuda > mps > cpu，CPU 线程走现场标定
```

| 脚本 | 改动 |
|---|---|
| `tools/sft_full.py` | 本来就有 `cuda if available`，线程已 `auto` |
| `tools/dream_code.py` | 接 `dev_pick`，模型/输入张量 `.to(device)`，`--threads` 默认 auto |
| `tools/audio_attach.py` | Whisper encoder、LLM、projector、输入张量全部跟 device；缓存落盘统一 `.cpu()` |

远程实测打印 `[dev] CPU 4 线程(现场标定)`。
**当前实例确实没有 GPU**（torch 是 cpu 版），免费 GPU 要另开 GPU 规格实例；
换过去不用改任何参数，`dev_pick` 会自己认。

---

## 七之四、产粮转音频线（进行中）

代码线先收工，CPU 让给音频。远程家底：`data/audio/wav` **6224 条**（S0002~S0019，911M）、
`models/whisper-small`。缺的是 pairs 和特征缓存，已补齐：

| 步骤 | 状态 |
|---|---|
| 推 `audio_attach.py` / `dev_pick.py` / `aishell_pairs.jsonl` | ✅ |
| pairs 路径兼容（存的是生成机绝对路径 `D:\...`） | ✅ 按 utt 重建索引，剔除 2148 条（S0020~S0025 无 wav），剩 **6049**（5464 train + 585 val） |
| Whisper 加载兼容 | ✅ 不传 `dtype=`，加载后判一次再 `.float()`（远程 transformers 4.55 会报 unexpected keyword） |
| 特征预计算 | 🔄 后台跑，实测 **1.04s/条** ⇒ 5969 条约 **100 分钟** |
| 盲测说话人 | 改成 **S0018,S0019**（缓存内、训练集没见过），不再用默认 S0024/S0025（那两个没有 wav） |
| align 训练 | ⏳ 等预计算完接（一条命令：`--stage align --use-cache --steps 800 --val-speakers S0018,S0019`） |

预计算已挂进 `tasks.json`（带 `done_file`，跑完不会被总管反复重启）。

---

## 七之五、作业队列：全都挂起来，还能插队

常驻服务和一次性作业分开管：

| 文件 | 管什么 | 谁消费 |
|---|---|---|
| `node/tasks.json` | **常驻服务**（隧道这种必须一直在的），死了拉起 | supervisor 每轮 |
| `node/queue.json` | **一次性作业**（训练/预计算/评测），按优先级排队 | supervisor 每轮 |
| `node/q.py` | 本地命令行，改队列（拉→改→推） | 人 |

`q.py` 常用命令：

```
python node/q.py add "python3 tools/xxx.py --a 1" -n 名字 -p 1
python node/q.py ls              # 状态/优先级/已经跑了多久
python node/q.py go 3            # 🔴 插队：提到 P0
python node/q.py stop 3 | cont 3 # 手动冻结 / 继续
python node/q.py retry 3 | rm 3 | clear | slots 2
```

**插队不是"插到队首等前面跑完"，是真抢占**：
P0 进来时，正在跑的低优先级作业整组 `SIGSTOP` **冻住**（不是杀掉，内存和进度都留着），
P0 跑完再 `SIGCONT` 接着跑。CPU 作业动辄一两小时，FIFO 的插队等于没有。

优先级数字越小越优先（0 最高，9 最低），`slots` 控并发（4 核建议 1）。

实测三件事（都过了）：

| 测什么 | 做法 | 结果 |
|---|---|---|
| 排队 | 加 P9 作业，只有一个坑 | 老实等到，不抢 |
| 插队 | 音频(P1) 在跑时加 P0 `jump` | 音频立刻 `SIGSTOP` 冻住，`jump` 上台 |
| 解冻 | `jump` 结束 | 音频自动 `SIGCONT` 继续，日志从 580 条接着算 |

🔴 这轮踩到两个真坑，都写进注释了：
① **僵尸进程**：`Popen` 不 reap，`os.kill(pid,0)` 照样成功，作业永远显示"在跑" →
改成读 `/proc/<pid>/stat` 判 `Z` 状态 + 每轮 `waitpid(WNOHANG)` 收尸；
② **恢复顺序**：把"在跑/冻住/等待"三轮揉成一轮按 id 走，
导致插队作业结束后被 P9 抢走唯一的坑、P1 一直冻着 →
改成三轮严格分开，**先摸清在跑的、再解冻被插队的、最后才起新的**。

---

## 八、镜像（顺手确认的事实）

- 创建实例时**可以选镜像**，但候选只有官方列表；我们一直用的是 CPU 默认 `ubuntu22.04-py312-torch2.3.1-1.40.1`。
- 想换高版本 torch 可以换，但 Python/torch/transformers 会整体变，tokenizer 断言与权重加载都要重验。
- "镜像套镜像"（DinD）需要 privileged，K8s Pod 不给，而且对我们要的能力没有意义。

---

## 九、待办

1. **音频线**：等特征预计算跑完（≈100 分钟）→ 起 `--stage align`，先小步数（800）看损失曲线，别再像 M1 那样跑到 630 步才发现震荡。
2. **代码线**：批次 1 还剩 11 题没跑，需要时 `dream_code.py` 续跑（缓存/银行都在，不重复）。
3. 银行 `triples` 已 210；满 200 新条触发下一轮训练（现在会自动用标定线程与 device）。
4. `chain/` 还剩 6.9G 未归类，需先和 0.4B 语料比对再决定。
5. 图片识别线：`siglip2-base` 已就位，等音频线跑通再开。
6. 重启一次实例，验证 `/etc/dsw/bin/prestart/99_polaris_start.sh` 是否保留（决定开机钩子是不是真·永久）。
7. 本地 `python` 在 Git Bash 里指向 WindowsApps 存根（返回 49、无输出），一律用 `./venv/Scripts/python.exe`。

---

## 十、文件清单

| 文件 | 作用 |
|---|---|
| `node/rc.py` | 远程命令行 + 索引（在线/离线双档）+ `put/get` 传文件 |
| `node/chd.py` | 直连守护（paramiko 长连接，TCP 19122） |
| `node/supervisor.py` | 总管：自愈 + 作业队列(可插队) + 心跳防空闲回收 + STOP 停机开关 |
| `node/sup.sh` | 总管启停外壳（start/stop/status/guard） |
| `node/tasks.json` | 常驻服务表（pattern / cmd / log / protect / done_file） |
| `node/queue.json` | 作业队列（优先级 / 状态 / pid / done_file） |
| `node/q.py` | 队列命令行（add/ls/go/stop/cont/retry/rm/clear/slots/log） |
| `node/boot_hook.sh` | 开机钩子源文件（装到 `/etc/dsw/bin/prestart/99_polaris_start.sh`） |
| `node/tunnel.sh` | 反向隧道看门狗 |
| `node/boot_resume.py` | 开机自检 + 起通道/隧道 + 续训 |
| `tools/cpu_probe.py` | 有效核数现场标定 |
| `tools/dev_pick.py` | 设备/线程自适应唯一入口（cuda>mps>cpu） |
| `tools/sft_full.py` | 恒维增量训练（`--threads auto`） |
| `tools/dream_code.py` | 做梦产粮 + 三档门 |
| `tools/audio_attach.py` | 听觉投影对齐（Whisper→0.3B，align/joint/full） |
| `docs/remote-server-plan-2026-10-19.md` | 通道/性能分册 |
| `docs/remote-inventory-2026-10-19.md` | 仓库盘点分册 |
| `docs/remote-index.md` | 远程目录索引（离线可读） |

---

## 十一、node2 入网 + 角色智能体远程化（15:30 后追加）

### 11.1 第二台机器：node2

用户新接入的免费云主机，与 node1 完全不同的一台。

| 项 | node1（魔搭 DSW） | node2（免费云主机） |
|---|---|---|
| 系统 | Ubuntu 22.04 / K8s Pod | `Linux 4.19.0-gvisor`（gvisor 沙箱） |
| 算力 | 8 核配额、实测约 4 核 | 2 核 / 4096MB |
| 磁盘 | 平台 NAS | overlay 50G（容器重调度会丢） |
| 出网 | 需隧道 | **直连，无环境代理，pollinations 200** |
| Python | 3.10（venv） | 3.12.3（系统） |

派活实测 **0.1～1.3s** 回传，比 node1 不慢。角色智能体已整体落在它上面。

### 11.2 角色智能体搬到 node2

详见 `docs/agent-roles-remote-2026-10-19.md`。三条必改（自环成员 / setsid / base64）不做到就必死。
入口只有一条命令：`python node/ag.py ask coder "..."`。

### 11.3 音频线：预计算收官

node1 上 `audio_precompute` 于 **15:44:53 完成**：

```
[pre] 完成 6048 条 -> data/audio/aud_feats.pt (5258s, 1395.3MB)
[data] 训练 5464 条 | 验证(没见过的说话人) 584 条 | val=S0018,S0019
```

0.97s/条，87 分钟，验证集是**没见过的说话人**（真盲测）。

40 步冒烟结论：链路全对（cache 加载、pairs 剔除 2148 条无 wav、projector 仅 0.78M 可训练），
但**模型没学会** —— loss 18.76→11.87→15.28→22.42 震荡，盲测输出全是复读
「请把这段语音转成文字。,,,,,,,,,,,,assistant 英雄化」，CER 4.7-5.2。

⇒ 待验证的假设是「只是步数不够」，用 0.2B 底座跑 800 步做对照，再用 0.3B 跑正式版。

### 11.4 0.3B 主干上机

node1 上原本只有 0.2B（`chain/r4base_std`，30 层 / hidden 576 / inter 3072），
**没有 0.3B**（36 层 / hidden 576 / inter 4608）。align 阶段主干全冻结、只训 projector，
但 projector 对齐的是 0.3B 的语义空间，所以必须把 0.3B 传上去。

切片规则：**95MB 一片，共 15 片**（ECS 单文件 >100MB 实测掉包）。
实测上传 **5.6MB/s，17 件 4.3 分钟**；node1 侧下载 **1.66MB/s，约 17 分钟**，拼回后校验 sha256。

### 11.5 🔴 两条新踩的坑（都跟 ECS 单线程有关）

**坑一：大文件下载会把广播打瘫。**
node1 从 ECS 拉 15×95MB 期间，`board.py send` 卡了整整 2 分钟。
原因是 PowerShell HttpListener 单线程，一个 95MB 的 GET 在服务期间**整个 listener 被占用**，
控制面的轮询请求只能排队。

实测代价：单次 HTTP 探测 **66.5 秒**；node2 的心跳 18 分钟传不上去，**被判掉线**。

⇒ 三条对策（按优先级）：
1. **片从 95MB 降到 10MB**（`push_03b.py` 已改）。单片只占 ~6 秒，请求之间留出空隙，
   控制面就能插进去 —— 这是最小改动、立竿见影的一招。
2. 大文件传输窗口内**不要依赖广播做交互**，纯等待。
3. 真要根治，得换通道：frp 隧道（`C:\llama\frps.exe` 已在跑，bindPort=80，stcp 免开新端口，
   预期 0.1-0.3s）或魔搭私有数据集（同平台内网，不占 ECS）。**均未做。**

**坑二：board 命令里内联 `setsid nohup ... &` 会堵死 polaris_node 的命令循环。**
现象：该条命令的 bash 迟迟不退出 → polaris_node 单线程卡在 `p.wait()` →
后面排队的命令全部「超时未领取」（seq 已前进，**不会重放**，等于丢了）。
⇒ **后台起长任务一律走脚本**：`node/start_align.sh`、`node/pull_03b.sh` 这类，
脚本自己 `exit 0`，只把 `setsid` 的子进程留在外面。

### 11.6 新增文件

| 文件 | 作用 |
|---|---|
| `node/ag.py` | 一条命令调远程角色智能体（list/ask/pipe/up） |
| `node/push_03b.py` | 0.3B 切 95MB 片推 ECS（用 `/upload` 返回的 `saved` 校验，ECS 不支持 HEAD） |
| `node/pull_03b.sh` | node1 侧按序拼回 + sha256 校验 |
| `node/start_align.sh` | 后台起 align 训练，自带 pidfile 幂等，立即返回 |
| `_mk_ng2.py` | 打 node2 移植包（剔自环成员 + 生成 boot） |
| `docs/agent-roles-remote-2026-10-19.md` | 角色智能体远程化全记录 |

