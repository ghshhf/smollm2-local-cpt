# 本机 ↔ 魔搭 node1 直连通道（2026-10-18 建立）

> 一句话：**把「发电报」换成「打电话」**——控制流走 SSH 反向隧道（秒级、多行、有 exit code），
> 数据流走魔搭仓库（OSS 底座，3MB/s，比走 ECS 快 32 倍）。

## 一、为什么原来的信箱不够用

| 项 | 信箱 `mb.py`（ECS 8081） | 实测 |
|---|---|---|
| 交互模型 | HTTP 轮询，0.35~1s 一个来回 | 一条命令最少 4~15s |
| 命令形态 | **只能单行**，多行会被静默截断 | 反复踩坑 |
| 回传 | stdout 塞进 `curl` 管道，被截断/串行化 | 大输出要分片 |
| 带宽 | ECS 是 2C4G 窄带 | **8MB 传 85s 后断链 ≈ 94KB/s** |
| 失败感知 | 无 exit code | 只能看日志猜 |

关键结论：**信箱慢的根因不是轮询，是 ECS 的出网带宽 ~94KB/s**。所以「优化信箱」没有意义，
必须换通道。1.3GB checkpoint 走 ECS 要 4 小时。

## 二、现在两条通道

### 通道 A：SSH 反向隧道（控制面，秒级）

```
本机 --ssh--> ECS:22 --direct-tcpip--> 127.0.0.1:9022 (ECS 回环)
                                          ^
                                          | 由 node1 主动建立的反向转发
node1:22 <--------------------------------+
```

- node1 在 k8s 内网，**没有入向公网**；但出向自由。所以由它主动连 ECS 做 `-R`。
- ECS 只开 22/80/8081/3389；80 被阿里云备案网关拦（返回 `{"error":"not found"}`），
  7000/6022 未开，`MODELSCOPE_NOTEBOOK_PROXY_URL` 网关要阿里云账号 cookie。
  **22 是唯一可用的洞**，反向隧道就架在它上面。
- 绑定在 ECS 的 `127.0.0.1:9022`（不需要改安全组、不需要 GatewayPorts）。
- 本机用 `ProxyJump` 两跳过去，实测握手 1.2~1.5s，之后一条命令 1s 级。
- Windows OpenSSH 在 ProxyJump 下 `ControlMaster` 复用不可用（`mux_client_request_session: read from master failed`），
  已从配置里去掉，不要加回来。

**入口脚本**：`node/n1.sh`

```bash
./n1.sh --alive                  # 探活
./n1.sh "cd /mnt/workspace/node1/dream && python3 -c '...'"   # 多行/管道/heredoc 都行
./n1.sh -f local_script.sh       # 推脚本过去执行
./n1.sh --put <本地> <远端>       # scp（<2MB 合适）
./n1.sh --get <远端> <本地>
```

配置：`D:/work/_local_ai/_tunnel/ssh_config`（`Host node1` + `Host ecs-jump`）。

**守护**：`node/tunnel_up.sh`，幂等；已挂进 `dream_daemon.sh` 开头，
所以 node1 容器重建后重启 daemon 会自动把隧道拉起来。

```bash
bash /mnt/workspace/node1/dream/tunnel_up.sh --status   # tunnel UP/DOWN
```

### 通道 B：魔搭仓库中转（数据面，3MB/s）

大文件一律走这条，**不要用 scp**（隧道的数据面还是过 ECS，同样 94KB/s）。

```
本机 --上传--> hhhlhuk/polaris-relay (魔搭数据集仓库, OSS) --下载--> node1
```

**入口脚本**：`tools/ms_relay.py`（两端各一份）

```bash
# 本机推
python tools/ms_relay.py push <文件> --as inbox/名字
python tools/ms_relay.py push-dir <目录> --as inbox/目录名
python tools/ms_relay.py ls
python tools/ms_relay.py rm inbox/名字

# node1 拉（在这条命令里就地执行）
./n1.sh "cd /mnt/workspace/node1/dream && python3 tools/ms_relay.py pull inbox/名字 --out /mnt/workspace/node1/inbox"

# 权重/模型仓库（走 polaris-03b）
python tools/ms_relay.py push <文件> --as 路径 --model --repo hhhlhuk/polaris-03b
```

## 三、实测带宽（2026-10-18）

| 通道 | 200MB 单向 | 速率 | 备注 |
|---|---|---|---|
| ECS 信箱 / scp 隧道 | 未跑完（8MB 即断） | **~94KB/s** | 1.3GB ≈ 4h |
| 魔搭数据集仓库 → node1 | 65.5s | **3.05MB/s** | 1.3GB ≈ 7min |
| 本机 → 魔搭仓库 | 35.6s | **5.62MB/s** | 家宽上行瓶颈 |
| 模型仓库 → node1（Qwen2.5-0.5B 954MB） | 70s | ~13.6MB/s | 模型库 CDN 更快 |

并发 Range 4 路实测并不提速（聚合 1.85MB/s < 单流 3MB/s）——**瓶颈在账号/IP 侧限速，不在连接数**，
不要再去加并发。

## 四、Token 坑（踩过的）

- `~/.modelscope/credentials/session` 里那个 **32 位 hex 已失效**（`E3001 AccessToken错误`）。
- 真正能用的是 `D:/work/_local_ai/deploy_aliyun/ms_token.txt` 里那个 **36 位 UUID**。
- `ms_relay.py` 会按 `env → ms_token.txt → ~/.modelscope/credentials/{token,access_token,session}` 逐个试登录，
  第一个成功就用；日志里能看到 `login ok <- 路径`。
- node1 容器重建会清 `/root`，但要重新注入：
  `printf '%s' '<uuid>' > ~/.modelscope/credentials/token`（或设 `MODELSCOPE_ACCESS_TOKEN`）。

## 五、失败模式与恢复

| 现象 | 原因 | 处置 |
|---|---|---|
| `n1.sh` 报 Connection reset / 超时 | node1 的 `-R` 掉了，或 node1 容器重建 | `bash tunnel_up.sh`；仍不行看 `logs/tunnel.log` |
| ECS 重启 | node1 的 while 循环 5s 后自动重连 | 一般自愈，无需操作 |
| `login fail` 全部候选 | token 过期 | 换 `ms_token.txt` |
| `pull` 报 404 | 上传用了默认 repo_type=model（写成模型库了） | 数据集必须带 `--repo-type dataset`（脚本已处理） |
| push 报 `create_dataset` 需要 namespace | repo 必须 `owner/name` | 用 `hhhlhuk/polaris-relay` |

## 六、明确不做的

- **不改造信箱成长轮询**：带宽瓶颈在 ECS，改造只省轮询不省字节，收益为零。信箱保留为**兜底**
  （隧道全挂、只能走 HTTP 时用 `mb.py`）。
- **不搞 frp / cloudflared / 魔搭 gateway 代理**：80 被阿里云拦、gateway 要账号 cookie、GitHub release
  在魔搭只能下到坏包。22 端口的反向隧道已经够用，不要再折腾。
- **不并发分片下载魔搭**：实测无效。

## 七、后续可做（未做）

1. `n1.sh --push` 一把梭：>5MB 自动切 `ms_relay.py push` + 远端 `pull` + 校验 sha256。
2. 隧道数据面提速：目前还是过 ECS，若哪天 ECS 升带宽，scp 才值得重新评估。
3. 端口转发常驻：`ssh -N -L 18010:127.0.0.1:8010` 把 node1 的 `exec_server.py` 映射到本机，
   HTTP 调用的延迟可降到毫秒（现在用 `n1.sh` 已经够，暂不需要）。
