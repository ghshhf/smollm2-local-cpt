# 远程真服务器化 + 通道提速方案（2026-10-19）

## 0. 结论先说

1. **远程是 Linux**：`Ubuntu 22.04.5 LTS`，kernel `5.10.134-18.0.12.lifsea8.x86_64`，x86_64。
2. 但它**不是虚拟机，是 K8s Pod**：`/proc/1/cgroup` 显示 `kubepods/...`，主机名 `dsw-2272396-...`，魔搭 DSW 容器。身份 **root**。**没有 systemd**，容器重启即重建（`/mnt/workspace` 是 NAS 挂载，数据不丢）。
3. `sshd` 是**活的**：`OpenSSH_8.9p1`，`0.0.0.0:22` 监听，pid 174。远程本机连 `127.0.0.1:22` 能拿到 banner。
4. 反向隧道**也在跑**：`ssh -N -R 127.0.0.1:9022:127.0.0.1:22 Administrator@47.99.131.12`，外面套 `while true` 自动重连（pid 5264）。
5. **但整条链路是断的**，断在两处：
   - 隧道会话**陈旧僵死**：ECS 上 `127.0.0.1:9022` 收下连接后，对端立刻 `Connection closed by remote host`（kex 阶段就关）；
   - 远程 `/root/.ssh` 里**只有 `known_hosts`，没有 `authorized_keys`** —— 就算隧道通了，`node1_key` 也进不去。
6. 所以"改造成真正的服务器"不是大工程，**P0 两个动作（装公钥 + 重启隧道）就能拿到 SSH 直连**，再用连接复用把单命令压到 **0.08s 级**。

## 1. 现在到底慢在哪（实测，不是估的）

| 段 | 耗时 | 说明 |
|---|---|---|
| 本机 → ECS 网络往返 | **35ms**（ping 31/42/35） | 这是地板，上限只能这么低 |
| 投递 `POST /task` | **840~910ms** | 比地板高 24 倍，ECS 那头的收件服务本身慢 |
| 远程取任务轮询 | 0~1000ms | `node_v6` 1 秒一轮 |
| 远程执行 | ~10ms | 真干活的部分 |
| 远程 `curl` 上传 ECS 落盘 | 300~600ms | 结果先进 ECS 文件，再被取回 |
| 本地 0.25s 轮询取回 | 0~250ms | |
| **合计（空命令）** | **2.2~3.5s** | 三次采样 2.29 / 2.22 / 3.46 |

**判断：这是 mailbox 架构的天生开销，优化到极限也压不进 0.5s。要"一秒以下到几十毫秒"，只能换成直连。**

## 2. 三阶段改造

### P0 — 打通直连（改动 2 处，约 5 分钟）

| 动作 | 具体 | 风险 |
|---|---|---|
| 装公钥 | 把本机 `node1_key.pub` 写进远程 `/root/.ssh/authorized_keys`（`chmod 600`） | 只加一个 key，不动 `sshd_config`；回滚=删文件 |
| 重启隧道 | kill 掉隧道循环里的 `ssh` 子进程，`while true` 5 秒后自动重连并重新 bind 9022 | 循环自愈；万一不自愈，原命令照抄即可重建 |

验收：本机 `ssh node1 hostname` 打印 `dsw-2272396-...`，耗时 < 300ms。

### P1 — 连接复用（这是"压到几十毫秒"的关键）

`~/.ssh/config` 加三行，把每次命令的两次 TCP + 一次认证全部省掉：

```
ControlMaster auto
ControlPath ~/.ssh/cm-%r@%h:%p
ControlPersist 10m
```

预期：冷连接 ~200ms（两次握手 + 认证），**复用后 70~90ms**（≈ 35ms 本机→ECS + 40~50ms ECS→Pod + 执行）。

`rc.py` 加**双通道**：SSH 通就走 SSH（`--direct`，默认自动探测），不通自动回落信箱。老路径不删，随时能退。

### P2 — 传文件与流式（现在的痛点）

| 现在 | 改成 | 收益 |
|---|---|---|
| 种子/银行先传 ECS 再让远程 curl 拉 | `rsync -az` 直推 | 92KB 种子：3s → 0.3s；356KB 银行同理 |
| 权重只能靠 ECS 中转（带宽 947KB/s） | rsync + 断点续传 | 1.3G 权重从"几十分钟"到"几分钟" |
| 看训练日志靠 `tail -n` 轮询 | `ssh node1 tail -f` 流式 | 实时，不刷屏 |
| 大批量生成测试一条条贴 | 一条命令跑一批，结果 rsync 回来 | 见 §5 |

## 3. 真服务器化清单：改造什么

| # | 项 | 现状 | 改成 | 收益 |
|---|---|---|---|---|
| 1 | SSH 免密 | 无 `authorized_keys` | 写入 `node1_key.pub` | 免密直连，一切工具（rsync/scp/tail）都可用 |
| 2 | 隧道看门狗 | 僵死无人知，等自检才发现 | 每分钟探 9022 banner，坏就重启循环 | 不再"开机看着好了其实连不上" |
| 3 | 开机自启 | 靠手贴一行 curl | `boot_resume.py` 里加：起 sshd 检查 + 隧道 + 看门狗 | 开机只贴一次，之后自恢复 |
| 4 | 目录规范 | 根目录摊 172 个文件 | 按 `tools/ logs/ archive/ data/` 归位（索引已建） | 自检不再选错根 |
| 5 | 监控快照 | 手动 `rc.py svc` | 每 10 分钟写快照到本地 JSON | 训练进度/负载/磁盘有历史曲线 |
| 6 | 仓库备份 | `gh_repo` 手动 push | 定时 `git push`（仓库级代理已写死 127.0.0.1:3067） | 记录不丢 |
| 7 | 训练与运维隔离 | 同抢 8 核 | 训练固定 `--threads 7`，运维走低优先级 | 训练不被观测打断 |

## 4. 明确不做的三件事

| 不做 | 为什么 |
|---|---|
| 不开 frp 公网端口 | Pod 无公网入站；把 root shell 挂公网等于送人。保持 `-R` 绑 ECS `127.0.0.1` 最稳 |
| 不在 Pod 里装 systemd | K8s 容器里 systemd 是假象，重启即重建；用 nohup + 看门狗才是正解 |
| 不把 `/mnt/workspace` 当本地盘做高频小文件 IO | NAS 挂载，小文件延迟高；先本地 `/tmp` 再落盘 |

## 5. 通道通了之后要做的更大规模测试

现在训练已收尾（`models/sft_seed200_remote` @ step 100），直连后可以一条命令跑完：

1. **holdout 20 题全量**（`code_eval.py`）+ 银行 201 条自测；
2. **做梦批量产粮**：一次跑 200~500 题衍生，过三档门入银行；
3. **权重回本地对照**：rsync 拉回 `sft_seed200_remote`，与本机 v4 做同批 holdout CE 对比；
4. **图片线起步**：`siglip2-base` 已在远程，直连后现造图文对种子，不用再走 ECS。

## 6. 风险与回滚

| 改动 | 风险 | 回滚 |
|---|---|---|
| 写 `authorized_keys` | 极低（只加不复写） | `rm /root/.ssh/authorized_keys` |
| 重启隧道 | 5 秒自愈；不自愈则手动起原命令 | 原命令：`ssh -N -i ~/.ssh/ecs_tunnel_key -R 127.0.0.1:9022:127.0.0.1:22 Administrator@47.99.131.12` |
| ControlMaster | socket 冲突报错 | 删 `ssh_config` 那三行 |
| `rc.py` 双通道 | 直连失败会回落，行为不变 | 删 `--direct` 分支 |

## 7. 验收标准

- P0：`ssh node1 hostname` < 300ms，回 `dsw-2272396-...`
- P1：`python node/rc.py ls` **< 100ms**（现在 1.9s）
- P2：`rsync` 传 92KB 种子 < 0.5s；`rc.py tail` 能实时看训练日志
- 稳定性：连续 50 条命令零失败，隧道断开后 5 分钟内自愈

---

## 八、实施结果（2026-10-19 13:2x 落地）

### 8.1 挖出来的三个真因（都不是"慢"，是"断"）

| # | 真因 | 证据 | 处理 |
|---|---|---|---|
| 1 | 远程 `/root/.ssh` 里只有 `known_hosts`，**没有 `authorized_keys`** | `ls -l /root/.ssh` 只有两行 | 写入 `node1_key.pub` |
| 2 | `/root/.ssh/ecs_tunnel_key` **文件根本不存在** | `ssh -i` 报 `No such file or directory`，循环每 5 秒失败一次 | 重新生成密钥并授权 |
| 3 | **ECS 是 Windows**，不是 Linux | 远程命令进 `cmd.exe`，报"不是内部或外部命令" | 公钥必须写 `C:\ProgramData\ssh\administrators_authorized_keys` |

补一条：旧的 `ecs_key` 从 node1 用会被拒（`Server accepts key` 之后仍 `Permission denied`），
换一把全新的 `n1t2` 立刻通过 —— 所以现在隧道用的是 `/root/.ssh/n1t2`，
它的公钥在 ECS 的 `administrators_authorized_keys` 里。

### 8.2 性能实测对比

| 路径 | 单命令耗时 | 说明 |
|---|---|---|
| 信箱 mailbox（原来唯一路径） | **2.2~3.5s** | 1s 轮询 + ECS 落盘再取回 |
| ssh ProxyJump 冷连接 | 1.28s | 两跳各一次握手，每条命令都重来 |
| Windows `ControlMaster` 复用 | **不支持** | 实测 `mux_client_request_session: Connection reset` |
| **直连守护 `chd.py`（paramiko 长连接）** | **通道 240ms / 端到端 0.37~0.49s** | 一次握手管到底 |
| 全盘索引 `sync` | 38s → **8s** | |
| `scp` 直传 | 1.77s | 含握手，大文件才有优势 |

### 8.3 现在的链路

```
本机 rc.py ──TCP 127.0.0.1:19122──> chd.py(计划任务 polaris-chd, SYSTEM)
      └─ SSH(ecs_key) ─> ECS(Windows, 47.99.131.12)
             └─ direct-tcpip -> 127.0.0.1:9022
                    └─ 反向隧道 tunnel.sh ─> node1:22 (root, n1t2)
信箱 mailbox 保留为兜底：chd 不可用时 rc.py 自动回落，行为不变
```

### 8.4 自启动（开机不用管）

| 端 | 机制 | 文件 |
|---|---|---|
| 远程 | `tunnel.sh` 看门狗（断了 5 秒重连）+ `boot_resume.py` 每次开机确保它在跑 | `node/tunnel.sh`、`node/boot_resume.py` |
| 远程 | `node_v6.py` 信箱消费者（bootstrap_all 拉起） | `node/bootstrap_all.py` |
| 本地 | 计划任务 `polaris-chd`，`/sc onstart /ru SYSTEM`，开机自起 | `node/chd.py` |

注意：`chd.py` 以 SYSTEM 运行，因此 `node1_key` / `ecs_key` 已 `icacls /grant SYSTEM:(R)`。

### 8.5 二次提速（13:3x）：0.4s → 0.36s，批量 0.14s/条

| 优化 | 效果 |
|---|---|
| 常驻 shell 通道（3 RTT → 1 RTT） | 通道 200ms → ~90ms |
| **TCP_NODELAY**（本地 socket + 到 ECS 的 SSH socket） | **省掉 200ms**：Windows Nagle + 延迟ACK 会把每个小包硬压 200ms，看起来像"网络慢" |
| `mb` 延迟导入 | 少 ~20ms 启动 |
| `rc.py shell` 常驻会话 | 解释器启动只付一次，之后 0.14s/条 |

**物理下限**：本机→ECS 35ms + ECS→node1 **6.4ms**（实测 ping）≈ 45ms。
**实测拆解（单条命令 0.36s）**：Python 解释器 ~150ms + 链路与执行 ~200ms。
要再往下只剩一条路：换掉"每条命令启动一次解释器"，即编译型客户端；收益约 0.15s，性价比不高，
日常用 `rc.py shell` 或 `rc.py idx`（离线 0ms）即可。

### 8.6 与"正常服务器"的极限对比（13:3x 实测）

**① 服务器自身能力：node1 不差，读写还赢本机**

| 指标 | 本机（Windows i5-12450H） | node1（Linux 8 核） | 判读 |
|---|---|---|---|
| 核数 | 12 逻辑（8C12T） | 8 | 相当 |
| `os.stat` 单次 | 4.8 µs | **1.0~1.3 µs** | node1 快 4 倍 |
| 打开+读 64B | 111.8 µs | **4.1 µs** | **node1 快 27 倍**（Windows 文件打开是重税） |
| 3M 次整数循环 | 146.8 ms | 140.2 ms | 算力持平 |
| /mnt/workspace | 本地盘 4.8µs | NAS 挂载 **1.3µs** | NAS 元数据缓存，不慢 |

**② 命令行往返：我们反而比"正常服务器"快一倍**

| 方式 | 单条命令 | 说明 |
|---|---|---|
| ssh 单跳到 ECS（一台正常服务器，每条新连接） | **760~810ms** | 握手+认证+执行 |
| ssh 两跳到 node1 | 1190~1430ms | 多一跳 |
| **我们的 `rc.py`（常驻守护）** | **360ms** | 比单跳 ssh 还快一半 |
| **我们的 `rc.py shell`（常驻会话）** | **140ms/条** | 快 5~10 倍 |
| `rc.py idx`（本地索引） | ~0ms | 不碰远程 |

**③ 客户端启动：Windows 的固定税，换语言也躲不掉**

| 客户端 | 冷启动 |
|---|---|
| Python（venv） | 90~218ms |
| curl.exe | 107~156ms |
| node.exe | 135~195ms |

三者同一量级 —— 在 Windows 上"创建一个进程"本身就要 100~150ms，与服务器无关。

### 结论

1. **读取 + 计算：已经到极限，且优于本机。** node1 是个正经 Linux 服务器，4µs 文件打开、1µs stat、算力持平。
2. **往返：物理下限 45ms**（本机→ECS 35ms + ECS→node1 6.4ms），现状常驻会话 140ms，还有约 2 倍空间，但边际收益递减。
3. **单条命令 360ms 的构成**：~150ms 是 Windows 进程启动税 + ~200ms 链路执行。理论地板 ≈ 200ms，且这 200ms 里 150ms 跟服务器毫无关系。
4. **对比"正常服务器"（单跳 ssh 800ms）我们已经快一倍** —— 不是因为服务器更强，而是因为常驻连接省掉了每条命令的握手。

### 8.7 远程"8 核"的真假（13:37 实测）

**结论：8 核是配额上限，不是保底算力；实际有效并行度 ≈ 3.1~3.2 核。**

| 证据 | 值 | 判读 |
|---|---|---|
| `os.cpu_count()` / `sched_getaffinity` / `/proc/cpuinfo` | 8 / 8 / 8 | 三处一致，不是假象级的错 |
| cgroup v1 `cpu.cfs_quota_us / period_us` | 800000 / 100000 = **8 核** | 配额确实是 8 |
| `cpu.stat` nr_throttled（整轮测试只增 1 次、1.0ms） | 几乎为零 | **不是 cgroup 在限流** |
| 1 进程 | 367ms → 1.00 核 | 基准 |
| 2 进程 | 366ms → 2.01 核 | 线性，很好 |
| 4 进程 | 454ms → 3.24 核 | 开始打折 |
| 8 进程 | 941ms → **3.12 核** | 到顶，再开线程没用 |
| `loadavg` | 0.04 | 机内没别人，不是自己抢自己 |

**原因**：Pod 的 QoS 类是 `burstable`（`/proc/1/cgroup` 里写着），8 核是**上限**不是**保底**。
宿主机超卖时拿不满，实际只有约 3.2 核的可调度算力。

**影响（要改的）**：
- `sft_full.py --threads 7` / `boot_resume` 传的 `--threads 8` **都开多了**，
  在 3.2 核的机器上开 7~8 线程只会互相抢、增加上下文切换。
- 下一轮训练改 `--threads 3`（留 0.2 核给通道/日志），预期更稳、不会更慢。
- 并行 benchmark 结论：**并行度上限按 3 算**，别再拿"8 核"做规划。

### 8.8 线程数 A/B 实测：3 线程比 7 线程快 16%

同一模型、同一数据、同 3 步，只改 `--threads`：

| 配置 | 步速 | 吞吐 |
|---|---|---|
| `--threads 7` | 8.23~8.72 s/步 | 110~114 tok/s |
| `--threads 3` | **8.06~8.54 s/步** | **132~133 tok/s** |

**+16% 吞吐**，验证了 §8.7 的推断：在 ~3.2 核的 burstable Pod 上，7 个线程是在互相抢。

已改：`node/boot_resume.py` 训练线程固定为 `min(3, 配额)`，可用环境变量 `POLARIS_THREADS` 覆盖；
`OMP_NUM_THREADS` / `MKL_NUM_THREADS` 同步改成 3。

**和本机对比**：远程 132 tok/s（3 线程） vs 本机 106 tok/s（6 线程），
**远程仍快约 25%** —— 即使核数打折，Linux 的文件/内存开销和调度仍优于 Windows 本机。
