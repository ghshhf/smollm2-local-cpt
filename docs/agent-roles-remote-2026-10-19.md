# 角色智能体搬到远程节点（2026-10-19）

> 用户诉求原话：「把每个都改造成跟那个服务器一样，不用我一直填命令」「远程一键调用其他智能体」。
> 落地结果：**本机一条命令，调远程那台免费云主机上的 5 个角色智能体**，不用 SSH、不用登录、不用填任何东西。

---

## 1. 结论先说

| 项 | 状态 | 实测 |
|---|---|---|
| node2 上角色服务常驻 | ✅ | pid 232，`127.0.0.1:18092`，5 角色 |
| 本机一条命令调用 | ✅ | `python node/ag.py ask coder "..."` |
| 中文 prompt 端到端 | ✅ | 5.0s，正常出中文 |
| 单角色连续 3 次 | ✅ | 16.4s / 4.2s / 5.4s 全成功 |
| 三角色流水线 `pipe` | 🟡 | 走通，但中间角色会超时（pollinations 偶发 >90s） |
| 节点重启自愈 | 🟡 | `ag.py up` 幂等拉起；开机钩子未做 |

---

## 2. 为什么能成：node2 是干净的出网环境

| 项 | 本机 | node2 |
|---|---|---|
| 出网 | 必须走代理 `127.0.0.1:3067` | **直连，无环境代理** |
| Python | venv 3.12 | 3.12.3（系统） |
| 资源 | 16G / i5-12450H | 4G / 2 核（gvisor overlay 50G） |
| pollinations 直连 | 200（经代理） | **200（直连）** |

⇒ 免 key 云脑那一层代码（pollinations / llm7 / kilo / unclose / ovh）全是纯 `urllib`，
   搬到没有代理的机器上反而更简单，一行环境变量都不用配。

---

## 3. 三个必须改造的地方（不改就必死）

### 3.1 剔除自环成员 —— 否则无限递归

`agents.json` 里 5 条 role 成员的 endpoint 是 `http://127.0.0.1:18091/role/ask`，那是**本机**的回环。
原样搬过去就是：`agent_roles` → `agent_team.ask_team` → 路由到 http 成员 → 又打到自己的 `agent_roles` → 死循环。

同时剔除：
- `http-echo`（指向本机 relay 8090）
- `g4f` / `g4f2`（依赖 `G4F_PY = C:/Users/admin/.workbuddy/.../python.exe`，Windows 绝对路径）
- `compute-pool`（要 `gh` CLI + token）

保留 10 条，实际可用 5 条纯 urllib 云脑。生成逻辑在 `_mk_ng2.py`，可重跑。

### 3.2 必须 `setsid` 起看护循环 —— 否则被 killpg 带走

`polaris_node.run_cmd` 用 `start_new_session=True`，命令超时会 `os.killpg` 整组。
第一版用 `nohup ... &`，现象是：**上一秒 `list` 还能用，20 秒后 `connect refused`** ——
派活的那条命令一结束，刚拉起的服务就被同一个进程组收走了。

改法：把看护循环写成 `_loop.sh`，用 `setsid` 脱离会话再起。
（node2 上有 `/usr/bin/setsid`，已确认。）

### 3.3 prompt 走 base64 —— 避开 shell 转义地狱

中文 + 引号 + 换行直接塞进单行 `sh -c` 会被吃掉。
`printf %s '<b64>' | base64 -d > /tmp/ag_q.json` 再 `curl --data-binary @文件`，
base64 字符集只有 `[A-Za-z0-9+/=]`，零转义风险。

---

## 4. 用法

```bash
# 看远程有哪些角色
python node/ag.py list

# 问一个角色
python node/ag.py ask coder "写个 LRU 缓存，只要代码"
python node/ag.py ask critic "这个方案有什么漏洞" -n node2 -p 18092

# 多角色串流水线（探索 -> 挑刺 -> 统稿）
python node/ag.py pipe "0.3B 自产代码 ACCEPT 率 26%，怎么提到 50%" -r researcher,critic,editor

# 服务掉了自动拉起（幂等，跑着就不动）
python node/ag.py up
```

底层还是广播协议：`board.json` → node2 领取 → 本地 `curl 127.0.0.1:18092` → `out_node2_<seq>.txt`。
`ag.py` 只是把这套包了一层，**没有新增服务端、没有开新端口、没有任何需要手填的东西**。

---

## 5. 实测数据

```
$ python node/ag.py ask coder "Write a python function that reverses a linked list, code only"
[ag] node2/coder via=team:pollinations  → 11.9s，出完整可跑链表反转

$ python node/ag.py ask coder "用一句话说明 Python 里 list 和 tuple 的核心区别"
→ 5.0s，中文回答正确

$ 连续三次 editor（SSR vs CSR）
→ 16.4s / 4.2s / 5.4s，三次内容一致且正确
```

---

## 6. 已知局限（别指望它解决硬问题）

1. **云脑会编来源**。三角色流水线跑「怎么把 ACCEPT 率提到 50%」，researcher 给出的
   `human-eval.org`、`ICFP 那个链接`、`Microsoft Edge/OpenAI AutoGPT 代码评测` 全是幻觉。
   它给的是通用建议，不是针对我们 0.3B 实况的判断。
2. **单轮可能超时**。pollinations 在长 prompt 上实测 90s+ 不返回。已把默认超时提到 180s，
   并让 `pipe` 在单轮失败时用前序产出兜底继续，不再整条链崩掉。
3. **ECS 单线程是真瓶颈**。node1 从那头下 15×95MB 权重时，广播轮询被长时间占用，
   表现为 `board.py send` 卡 2 分钟。**大文件传输期间不要依赖广播做交互。**
4. **重启不持久**。`~/agents` 在 gvisor overlay 上，容器重调度会丢。目前靠 `ag.py up` 手动幂等拉起；
   要真正开机自愈，得把 boot 挂在 `polaris_node` 的启动路径上（未做）。

---

## 7. 文件清单

| 文件 | 作用 |
|---|---|
| `node/ag.py` | 本机入口：list / ask / pipe / up |
| `_mk_ng2.py` | 打 node2 移植包（剔自环、生成 node2 版 agents.json 和 boot 脚本） |
| `ng2.tgz` | 移植包（25.7KB，含 agent_roles/agent_team/ai_hub/两份配置/boot） |
| `agent_roles.py` | 加了 `ROLES_HOST` 环境变量（默认仍 127.0.0.1，向后兼容） |
| node2 `~/agents/ng2_boot.sh` | setsid 看护，`_loop.sh` 死 5s 重起 |
