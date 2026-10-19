# polaris-audio-io —— 北极星音频外设

耳朵和嘴。工业级组件直接封装，**不训练**。

```bash
cd D:/work/_local_ai/train_15b

# 说
./venv/Scripts/python.exe modules/polaris-audio-io/audio_io.py say "你好，主人" --play

# 听
./venv/Scripts/python.exe modules/polaris-audio-io/audio_io.py hear data/a.wav

# 闭环自测（自己说、自己听、自己打分，不需要麦克风）
./venv/Scripts/python.exe modules/polaris-audio-io/audio_io.py loop "测试文本" -n 3
```

## 实测（2026-10-19，本机 i5-12450H CPU）

| 项 | 结果 |
|---|---|
| 说话（edge-tts 神经音色） | 4.78s 音频 / 28KB mp3 / 一次调用 ~2s |
| 说话（SAPI 离线回落） | 2.32s 音频 / 102KB wav / 断网可用 |
| 听（whisper-small，beam=1） | 5.4s 音频 → 4.57s，**RTF 0.78** |
| 闭环 CER（3 轮一致） | **0.125**，错在「权重→权众」这类同音字 |
| 对比：我们训了 20 小时的 0.3B 音频模块 | CER **7.42**，0/10 正确 |

**差 59 倍。** 这就是「工业级」和「自己从零训」的区别。

## 两个必须记住的坑

**1. Whisper 中文默认吐繁体。** 不加 prompt 时输出「離線引擎測試」。解法是给一段简体 prompt（`listen.py: PROMPT_ZH`），这是官方推荐做法，不是 hack。

**2. prompt 会吃 token 预算。** `max_new_tokens` 必须减去 `prompt_ids` 长度**再减 8**（还要留 `<|startoftranscript|>` 等 4 个特殊位），否则报 `combined length exceeds max_target_positions 448`。

## 设计原则

**双引擎，不能因为断网就变哑巴。** `auto` 模式先试 edge-tts（音质接近真人），失败落 SAPI（机械但永远在）。工业级封装的第一原则是可用性，不是音质。

**这是外设，不是能力。** 它零参数、不挂主干、不产生 LoRA，所以 `interface.json` 里 `kind=tool`、`attach_points=[]`，不受「只接恒定维」契约约束。那份契约管的是会改主干计算图的模块。

**外设与主干无参数共享，是刻意的。** 它不会随模型进化，但也不该随模型退化。耳朵坏了，不能因为模型训练出问题。

## 边界

- 解决了「听得到、说得出」，**没有解决「听得懂」**。语义理解仍归主干。
- 闭环 CER 不是真实听写精度：自家 TTS 音色单一、无噪声无口音，真实麦克风会更差。别拿 0.125 对外宣称。
- Whisper small 是 244M，换 medium/large 能压到 CER 5% 以内，代价是 CPU 上 RTF 破 3。

## 文件

| 文件 | 作用 |
|---|---|
| `audio_io.py` | 统一 CLI（say / hear / loop） |
| `speak.py` | TTS 双引擎 + 格式转换 + 播放 |
| `listen.py` | ASR（完整 Whisper，encoder+decoder）+ CER |
| `interface.json` | 模块契约（kind=tool） |

权重在 `models/whisper-small/`（967MB，含 decoder，与 node1 上那份 `audio_attach.py` 用的 **只保留 encoder** 的实验版不是一回事）。
