# 远程连接方式排查：邮件箱 → 隧道（2026-10-18）

## 目标

用户诉求原话：「（远程）搞一个隧道相关的，然后就能成为秒级甚至毫秒级，就类似于当成一个完整的服务器了。
咱们现在本地机上用的是那个信箱机制，还是不够完美，速度还不够快。」

要解决的是**本机 ↔ 魔搭 node1 的命令往返延迟**：现在信箱是轮询制，一次往返 0.35s~2.5s。

## 结论先行

**四条路都试过了，ECS 中转这条路结构性地快不了，不是配置问题。**

真正的毫秒级只有一条路：**不经 ECS，本机直连魔搭**。这需要用户在魔搭控制台点一下「SSH 直连」，或开一个阿里云安全组端口。

## 逐条实测

### 1) 魔搭自带端口代理网关 —— 需要阿里云账号登录，机器间不可用

环境变量里给了：

```
MODELSCOPE_NOTEBOOK_PROXY_URL=https://2269442-proxy-{port}.dsw-gateway-cn-hangzhou.data.aliyun.com
```

把 80 10 端口起服务后实测：
- 匿名访问 → `302` 跳 `account.aliyun.com/login/login.htm?oauth_callback=...&appId=MAAS`
- 试了 `DLC_USER_TOKEN` 的 4 种带法（Bearer header / cookie / 自定义 header / query）**全部 302**

→ 结论：这是给人用浏览器点的通道，不能做机器间自动通信。

### 2) frp + ECS 的 80 端口 —— 被阿里云备案网关挡在门外

- ECS 实际开放端口（实测 `/dev/tcp` 探测）：**22、80、8081、3389**；7000/6022/2222/10022 全关。
- 80 空闲 → 上 `frps.exe`（19.8MB，走 scp 上传成功），配 `bindPort=80` + `vhostHTTPPort=80`。
- 启动后 netstat 确实看到 `0.0.0.0:80 LISTENING`，但**那是 svchost（HTTP.sys），不是 frps**；frps 绑定失败后自行退出。
- `netsh http show servicestate | findstr ":80/"` 查不到注册 → 80 的响应不是本机任何服务发的。
- 本机绕代理直连 `http://47.99.131.12/` 返回：

```
HTTP/1.1 404 Not Found
Server: Microsoft-HTTPAPI/2.0
{"error":"not found"}
```

→ **大陆 ECS 的 80 端口未备案被阿里云网关拦截**，任何非 HTTP 的协议（frp 控制帧）都会被它回 404。
- 魔搭侧 frpc 因此报 `tls: first record does not look like a TLS handshake`，关掉 TLS 后变成 `login to the server failed: EOF`。

### 3) ECS 中转的带宽 —— 这才是根本瓶颈

实测魔搭从 ECS `/file/` 拉 13MB 的 frp 包：

| 尝试 | 结果 |
|---|---|
| 180s | 只收到 **158 字节** |
| 400s | 只收到 3.88MB（约 10KB/s） |
| 同一包走 `ghfast.top` 镜像 | **一次下完 13MB** |

→ ECS 出方向带宽极低（约 10KB/s）。**即使隧道打通，所有流量还要过 ECS，一样慢**。这条链路不可能做到毫秒级。

### 4) 魔搭直连 GitHub —— release 下不动

- `github.com` 头部探测 200，但 `objects.githubusercontent.com` 404
- cloudflared（39MB）只拿到 **601KB 坏包**
- frp 走 GitHub release 同样失败
- **可用替代：`ghfast.top` 镜像**（实测能下完，但速度不稳定，需 `-C -` 断点续传兜底）

## 可行的三条路（按推荐度）

| 方案 | 延迟 | 需要用户做什么 | 备注 |
|---|---|---|---|
| **A. 魔搭「SSH 直连」** | ~30-80ms | 在魔搭控制台开启 SSH，把 `ssh -p xxxxx root@...` 命令给我 | 官方原生，不经 ECS，最接近「完整服务器」 |
| **B. 阿里云安全组开 7000 + 6022** | ~80-150ms | 控制台加两条入方向规则 | 走 frp，ECS 已有 frps 就绪，但受 ECS 出网带宽限制 |
| **C. 维持信箱 + 缩短轮询** | 200-500ms | 无 | 现在就是这个，`mb.py` 0.35s 轮询；改 `node_v6` 到 0.2s 可再快一点 |

## 已落地的准备工作（不浪费）

- ECS 上 `frps.exe` + `frps.toml` 已就位（`C:\Users\administrator\`），等端口开放即可用。
- 魔搭上 `frpc` 二进制、`frpc.toml` 已就位（`/mnt/workspace/node1/`）。
- 魔搭上 `tools/exec_server.py` 已跑通（`0.0.0.0:8010`，token `k0u4mail2026`），提供：
  - `GET /health` — 主机名 + load
  - `POST /exec {cmd,cwd,timeout,bg}` — 执行命令，`bg=true` 返回 pid
  - `POST /put?path=` / `GET /get?path=` — 读写文件
  - `GET /tail?path=&n=` — 看日志尾部
  - `GET /ls?path=` — 列目录
- 只要通道打通，改一行 serverAddr 就能接上，不用重写。

## 注意

- ECS 上 `8081` 被信箱服务（PowerShell HttpListener，`mailserver_v4.ps1`）占用，PID 4（System/HTTP.sys），不能复用做隧道。
- 魔搭容器重启会清空 `/root`；**所有脚本必须放 `/mnt/workspace`（NAS 持久）**，这也是做梦工具链放在 `/mnt/workspace/node1/dream/` 的原因。
