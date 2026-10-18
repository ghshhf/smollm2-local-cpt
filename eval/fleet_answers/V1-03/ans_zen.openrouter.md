# zen.openrouter (vendor=openrouter, served=openrouter/free, 18.3s)

**一、可量化的信任晋升规则（“Δ 多少就升格”）**  

| 步骤 | 形式化指标 | 计算方式 | 阈值（Δ） | 说明 |
|------|------------|----------|----------|------|
| 1️⃣  | **来源权重 wₛ** | 预置白名单域名 → wₛ=1.0；灰名单（知名但非官方） → wₛ=0.5；其余 → wₛ=0.0 | — | 只在白名单或灰名单上才能产生正增量。 |
| 2️⃣  | **交叉印证度 c** | 对同一片段，统计在 **N** 个独立来源（不同 IP/域）中出现的次数 k；c = k / N | Δc ≥ 0.4（即至少 40% 的独立来源出现） | 防止单一来源“刷榜”。 |
| 3️⃣  | **时间衰减因子 τ(t)** | τ(t) = e^{‑λ·Δt}，Δt 为内容首次出现到现在的天数；λ = ln(2)/T₁/₂，取 T₁/₂ = 30 天（半衰期） | — | 越老内容权重越低。 |
| 4️⃣  | **格式清洗分 f** | 经过 **格式脏** 检测（见第 2节）后得到的分数 f∈[0,1]，1 表示完全干净。 | Δf ≥ 0.8 | 只允许极高格式质量的内容进入信任累计。 |
| 5️⃣  | **信任增量 ΔT** | ΔT = wₛ · c · τ(t) · f | **当累计信任 T ≥ T₀ 时升格**<br>其中 T₀ = 0.6（经验值，可在验证集上调节） | 只要单次增量 ΔT ≥ 0.12（即 T₀/5），就可视为“有效晋升步骤”；累计五次以上即达升格门槛。 |

**伪代码（可直接落地到 0.5B 端侧服务）**

```python
# 预置参数
WHITELIST = {"edu.cn", "gov.cn", "wikipedia.org"}   # 权威域名
GREYLIST  = {"zhihu.com", "csdn.net", "jianshu.com"} # 半权威
LAMBDA    = math.log(2) / 30.0                     # 半衰期 30 天
T0        = 0.6                                    # 升格总阈值
STEP_MIN  = 0.12                                   # 最小有效增量

def trust_score(fragment, src_domain, first_seen_ts, now_ts):
    # 1. 来源权重
    if src_domain in WHITELIST:   w = 1.0
    elif src_domain in GREYLIST:  w = 0.5
    else:                         w = 0.0
    if w == 0.0: return 0.0

    # 2. 交叉印证度（需全局计数表，这里假设已有）
    k = citation_count[fragment]          # 已出现的独立来源数
    N = total_sources_seen[fragment]      # 曾经检查过的来源总数
    c = k / N if N > 0 else 0.0

    # 3. 时间衰减
    delta_t = now_ts - first_seen_ts      # 天数
    tau = math.exp(-LAMBDA * delta_t)

    # 4. 格式清洗分（见第2节）
    f = format_clean_score(fragment)      # 0~1

    # 5. 增量
    return w * c * tau * f

# 主循环（伪）
trust_pool = defaultdict(float)   # fragment → 累计信任
for frag, meta in stream_of_untrusted():
    dT = trust_score(frag,
                     meta["domain"],
                     meta["first_seen"],
                     now=time.time())
    if dT >= STEP_MIN:               # 只接受有效增量
        trust_pool[frag] += dT
        if trust_pool[frag] >= T0:
            promote_to_trusted(frag) # 进入训练池
```

**可证伪判据**：  
- 若在持续运行≥ 48 h（node1 CPU）后，出现 **任意** 已升格为 trusted 的片段随后被人工标注为**明显错误**（事实错误、恶意宣传或显式毒害），且该片段的 **ΔT** 累计值 **未超过** 0.6（即未触发