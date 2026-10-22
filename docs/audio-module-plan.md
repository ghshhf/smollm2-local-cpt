# 音频模块计划书（听 + 说）—— 北极星私有语音能力

> 状态：草案 v1（2026-10-18）
> 定位：北极星「六能力线」之一，按模块契约接入（投影/适配器接恒定维，**不进主干参数**）。
> 核心结论：音频 ≠ 只识别。本模块 = **ASR（听）+ TTS（说）+ 语音对话闭环**，让北极星能"用嘴回答、用耳听"。

---

## 0. 为什么现在做

- 原音频线只规划了 ASR（Whisper 编码器 + 投影，align 训练中）。用户明确要"能发出声音"——即 TTS 与端到端语音对话。
- 北极星是私人模型，愿景是"永远私人"。语音对话闭环全部本地、零云依赖，正好契合。
- 2026-10-18 22:40 已按"9 点停训"约定手动停掉本地 `train_queue`（audio_precompute + align_train + chat_memory_ab 未跑），进程已清、锁已删。ASR align 训练可随时接回队列续跑。

---

## 1. 总体架构：语音对话闭环

```
麦克风 → VAD(人声检测) → ASR(听) → 北极星(0.5B 大脑) → TTS(说) → 扬声器
                ↑                                              ↓
                └─────────────── 多轮上下文（记忆层 / 对话历史）──────────────┘
```

- **ASR（听）**：把声音变成文字，喂给北极星。已起步。
- **北极星（大脑）**：0.5B 主干 + 对话 LoRA，产出文字回复。已在跑。
- **TTS（说）**：把北极星文字变成声音播出去。**本次新增的重点**。
- 三者都是独立"模块"，经投影/适配器与主干常驻连接，符合模块接口契约（第 0 条：只接恒定维 q/k/v/o_proj、残差流 adapter）。

参考的成熟本地开源管线（全是本地、可离线）：
- `On-Device Speech-to-Speech`（CPU，Ryzen 5600G 无 GPU，~2s 延迟）：VAD(pyannote) → Whisper → qwen2.5:0.5b → Kokoro-82M。
- `Faster-Local-Voice-AI`（8GB 无 GPU 笔记本，<1s STT→TTS）：Vosk + Ollama + Piper。
- `Astronix-Local-RAG`：Faster-Whisper + Llama3.2-1B + pyttsx3。
- Jetson 版：Whisper(CTranslate2) + llama.cpp + Piper。
- 共性：**ASR 用 Whisper 系，TTS 用轻量本地引擎，LLM 是独立模块**——北极星直接替换 LLM 那一格即可。

---

## 2. 子模块一：ASR（听）—— 续做，不重来

**现状**：已下 AISHELL-1；`tools/audio_attach.py --precompute` 已产出 `data/audio/aud_feats.pt`（346MB，约 9% 语料）；下一步 `audio_align_train` 训练 Whisper 编码器 → 北极星 hidden 的投影。盲测验收标准 = CER（中文识别错误率），**不许只看 loss**。

**选型（已定）**：Whisper 编码器 + 线性投影。推理端用 `faster-whisper` 或 `whisper.cpp`（Int8 量化，CPU 亚秒级）。
- 理由：Whisper 小模型中文识别已够用（上述多个项目实证）；我们要的是"接主干的投影"，不是重训 ASR。

**验收里程碑**：
- ASR align 训练续跑 → AISHELL-1 盲测 CER（目标 < 8%）；与裸 Whisper 对比。
- 接 VAD 做成"按住说话 / 自动断句"的实时听写。

---

## 3. 子模块二：TTS（说）—— 新增重点

### 3.1 开源方案对比（2025–2026 实测口碑 + 评测）

| 方案 | 规模 | 中文 | 克隆 | 许可 | 备注 |
|---|---|---|---|---|---|
| **F5-TTS** | 0.3B | 强（中英混） | 零样本 ~10s | **MIT** | flow-matching DiT，无时长/音素对齐，易训练，RTF≈0.15 |
| **GPT-SoVITS** | 小 | **最强**（中文） | 5s 零样本 / 1min 微调 | **MIT** | 自带数据预处理（分段+中文 ASR），WebUI，V3 支持情绪 |
| CosyVoice2 / 3 | 0.5B | 强+流式 | 零样本 | 阿里（需看协议） | 流式首包 150ms，但 Linux+GPU 重，本地化门槛高 |
| MaskGCT | — | 强 | 零样本 | 学术 | ICLR2025，无时长模型 |
| Kokoro | **82M** | 英文强 | 无 | 轻量 | CPU 极快，作"默认非克隆嗓"兜底 |
| Piper | 极小 | 一般 | 无 | 轻量 | 极快但自然度低 |

### 3.2 推荐落地

- **主选：F5-TTS（0.3B, MIT）**。
  - 架构干净（基于 E2-TTS，ConvNeXt 精炼文本表征 + flow matching DiT），没有时长模型/音素对齐等复杂部件，**训练和改代码成本最低**。
  - 零样本克隆：给 ~10s 参考音频即可复刻音色；想固定"北极星嗓"可少样本微调。
  - 有 **Fast F5-TTS（EPSS，arXiv:2505.19931）**：免训练加速，7 步生成、RTF 0.03（3090），CPU 也能用，直接 plug-in。
- **备选 / 中文强化：GPT-SoVITS**。若要最自然中文 + 情绪，用它替换 F5 做音色克隆那一格。自带中文数据预处理管线，省事。
- **兜底轻量：Kokoro-82M**。当不想克隆、只想快速出声时用，CPU 几乎零延迟。

### 3.3 关键论文（写进知识库，供 TTT/自进化参考）

- F5-TTS — *A Fairytaler that Fakes Fluent and Faithful Speech with Flow Matching*, arXiv:2410.06885 (2024)
- E2-TTS — *Embarrassingly Easy Fully Non-Autoregressive Zero-Shot TTS*, IEEE SLT 2024
- CosyVoice 2 — *Scalable Streaming Speech Synthesis with LLMs*, arXiv:2412.10117
- NaturalSpeech 3 — *Zero-Shot Speech Synthesis with Factorized Codec and Diffusion*, ICML 2024
- MaskGCT — *Zero-Shot TTS with Masked Generative Codec Transformer*, ICLR 2025, arXiv:2409.00750
- Flow Matching — Lipman et al., ICLR 2023（F5/CosyVoice 的方法论基座）
- Fast F5-TTS（EPSS）— arXiv:2505.19931（推理加速，必看）

### 3.4 TTS 训练 / 数据

- **零样本克隆（首选路径，无需训练 TTS 基座）**：录一段北极星固定 persona 嗓音参考音频（10s~1min）→ F5-TTS / GPT-SoVITS 直接克隆。
- **如需本地训基座**：用公开中文多说话人语料（AISHELL-3、DataBaker 等）训/微调 F5-TTS；目标小、CPU 可扛。
- **认主结合**：参考音频即"主人声纹"，与 `docs/identity-memory-roadmap.md` 的认主接口打通——只对主人声纹响应/切换 persona。

---

## 4. 集成方式（遵守模块契约）

- TTS 是**独立进程/模型**，北极星只产出文本，经一个 `audio_speak` 模块调 F5-TTS 合成并播放。
- 不把 TTS 塞进 0.5B 主干；主干只负责"想说什么"，声学生成外包给 TTS 模块（与 ASR 投影同构：主干常驻，模块即插即用）。
- 接口：北极星文本输出 → `queue` → TTS 模块（带参考音色）→ wav → 播放。流式可用 CosyVoice2/F5 的流式模式压首包延迟。

---

## 5. 里程碑（按优先级）

- **M1（接回 ASR）**：`train_queue` 续跑 `audio_align_train` → AISHELL-1 盲测 CER 验收。*（之前按 9 点约定暂停，随时可恢复）*
- **M2（TTS 落地）**：本地跑通 F5-TTS 推理（venv_ttt 或独立 venv），中文音色克隆 demo 出声。
- **M3（闭环）**：VAD → ASR → 北极星 → TTS 端到端实时 loop，延迟验收（CPU 目标 < 2s 端到端）。
- **M4（认主音色）**：固定北极星 persona 嗓 + 主人声纹切换，接 identity-memory-roadmap。

---

## 6. 风险与对策

- **CPU 推理慢**：F5-TTS/Kokoro 可 CPU 跑但 RTF 偏高；对策 = Fast F5-TTS 加速 + 必要时 INT8/ONNX；非实时场景可接受。
- **中文自然度**：F5 中英混好，纯中文最自然选 GPT-SoVITS 兜底。
- **克隆需参考音频**：提前录好 persona/主人参考音，存记忆层（可删、E-7 合规）。
- **风格一致性**：TTS 音色与北极星"人设"解耦，需显式绑定参考音，避免每次随机音色。

---

## 7. 与现有未完项的关系

- 原"听觉"能力线（进度 50%）扩为"音频（听+说）"能力线，进度重估。
- ASR align 训练接回 `train_queue` 即自动续；TTS 为全新 venv + 权重（不入库，按私有仓纪律只在档案存小 db/种子）。
- 语音对话闭环是"六能力组合"的终局拼图：爬虫供粮→编程→视频→**语音对话**→三元组入银行→TTT 现场装用。
