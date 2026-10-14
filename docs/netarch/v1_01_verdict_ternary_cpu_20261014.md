# V1-01 交叉裁决：三值 QAT 在纯 CPU 上训不训得出来（0.5B 生死判据）

> 2026-10-14 · 三模型独立交叉（gemini-3.8-flash / grok-4.7 / cohere-north-mini-code）· 原始答案存`fleet_answers/V1-01/`

## 一、三方一致与分歧

| 维度 | gemini-3.8-flash | grok-4.7 | cohere/north-mini-code | 裁决 |
|---|---|---|---|---|
| 真实瓶颈 | 硬件指令集负优化（访存风暴）+ STE 病态，**双杀** | STE 失配的梯度噪声（**非**算力、**非**带宽） | STE 估计偏差 + 内存带宽，**非** FLOP 不足 | **一致：STE 梯度噪声是首要瓶颈，不是算力** |
| 从零训规模上限 | ≤15M 参数 / ≤50M tokens（超则PPL 30~50 死平线） | 未给硬上限 | ~200M 参数可行上限 | gemini 最具体；north 上限偏乐观，以 gemini 为准 |
| 带初始化 QAT 上限 | 0.5B / ≤100M tokens 可行 | — | — | **采用：继承扩容后 0.5B QAT 可行** |
| 替代方案 | 残差退火 QAT（alpha→0）+ 0.4B 扩容继承 | Net2Net 从 fp32 基座跳变到三值 | fp32 主干 + 影子 Adapter QAT | **三方殊途同归：必须继承 fp16/fp32 权重，不能从零三值化** |

## 二、🔴 被推翻的两条原方案（重要）

### 推翻1：「CPU 从零预训练 0.5B 三值」
gemini 定量论证：0.5B 按 Chinchilla 需~3.5B tokens ≈ 1.05e19 FLOPs，即使 CPU 满载能效也需**持续 120 天以上**。物理不可行。

### 推翻 2：「冻结 fp32 主干 + float16 影子 Adapter 做 QAT」（我们原定方案）
gemini 指出**致命矛盾**：三值 {−1,0,+1} 在数学上**无法无损吸收浮点 Adapter**。若推理时保留浮点 Adapter，端侧内存读取量与 FP16 相同 ⇒ **CPU ≤15ms/token 硬指标必然击穿**。
⇒ 浮点 Adapter 只能作**训练期残差**，且必须**退火到 0**，不能留到推理。

## 三、裁决结论（采纳方案）

**路线**：`0.4B FP16 → Net2Net 深度优先扩容至 0.5B 深窄形 → 作为 Master Weights 继承 → CPU 跑小样本（50M~100M tokens）退火蒸馏 QAT`

**训练期机制 = 渐进式残差退火**（采纳 gemini 的 BitLinearAnnealed，修正 grok/north 的静态做法）：
- 前向：`W_eff = W_q + α(W_latent − W_q).detach()`，α 从 1.0 线性退火至 0；
- 推理期：α=0，**纯三值**，无任何浮点残留 ⇒ 不击穿 15ms/token；
- 激活：8-bit dynamic absmax 量化。

**为什么这个方案能同时满足两端**：训练期用浮点残差吸收 STE 噪声（好训），推理期残差归零只留三值（够小）。这正是三方都指向但表述不同的东西。

## 四、可证伪判据（合并三方，供node1 135M 探针用）

| 指标 | 能训 | 训不动 |
|---|---|---|
| 潜变量翻转率 Δ_flip | 前期 3~8% → 退火后降至 0.1~0.5% | <0.01%（锁死）或全程 >15%（震荡横跳） |
| 零权占比 R₀ | 稳定 35~55% | <10%（退化为二值）或 >85%（全零坍缩） |
| 相对发散度 Δℒ = ℒ_BitLinear − ℒ_FP16 | 1000 step 时 ≤ +0.35 且斜率平行 | ≥ +1.2 或 loss 完全平直/NaN |
| CPU 吞吐 | ≥800~1200 tok/s（64核 AVX-512） | <300 tok/s，或预估 5M tokens >8 小时 |
| plateau | 无 | 第 2000~5000 步 plateau（降<0.1）或 grad norm>10 |

**判决规则**：任一"训不动"列命中 ⇒ 立即终止从零三值化，强制走「继承扩容 + 退火QAT」。

## 五、待实测项（⏳ 需用户「全自动」授权）

1. **gemini 的 15M 参数上限**未独立验证；north 的 200M 上限偏乐观。需实测 135M 上翻��率与 R₀ 曲线。
2. **α 退火速率**（多少 step 退到 0）无共识，需实验扫。
3. 三方都未验证「继承 fp16 权重后，三值化的初始 loss 冲击有多大」——这是 Net2Net 探针（旧）之外的新问题。

## 六、原始答案

- `eval/fleet_answers/V1-01/ans_kris.gemini.md`（89.2s，含完整 BitLinearAnnealed 伪代码 + 探针脚本）
- `eval/fleet_answers/V1-01/ans_kris.grok.md`（68.7s）
- `eval/fleet_answers/V1-01/ans_kris.north.md`（58.5s）