# 公益 API 资源（免费中转）

> ⚠️ 性质：社区公益站，**稳定性/寿命无保障**，只作弹性补充，不当依赖。
> ✅ 密钥来源：公开发布在评论区，任何人可注册获取——**本文件可随仓库开源，无泄密问题**。
> 🔴 红线不变：**机密数据（owner_corpus / 持仓 / 密钥 / 私人文档）绝不发往任何中转站**——公益站可见全部流量。

## krisphyn 公益站（2026-10-14 已实测可用）

- **Base URL**: `https://api.krisphyn.cc/v1`（OpenAI 兼容）
- **API Key**: `sk-go28BcSeT4Ig14N30NVLfDq6fZKquKRLRW0S3zdhosXGwa9U`
- **注册**: 好用可以自行注册（用户 2026-10-14 指示）

### 实测记录（2026-10-14）
- `deepseek-v4-pro-0813`：通，带 reasoning，后端有 KV cache（真通道直连）✅
- `glm-5.2`：通，**但 served 的实际是 `z-ai/glm-5.3-flash`** —— 站方做模型名映射，关键任务需验证实际模型 ⚠️

### 模型池（38 个，2026-10-14 快照）
DeepSeek: v4-pro / v4-flash / v4.1-flash ｜ Google: gemini-3.8-flash(low/medium/high/tiered) / 3.1-lite ｜ xAI: grok-4.7 / grok-4.20-multi-agent / grok-composer ｜ 国产: glm-5.2 / glm-5.3-flash / Qwen3.8-27B / Qwen3.8-Flash-Next / MiniMax-M3 / MiMo-V2.6 / MiniCPM-V-4.7(多模态) / step-5-preview / longcat-2.5 / sensenova-6.8 ｜ 其他: nemotron-3-super/ultra / north-mini-code / dots-3-note / hy4-preview ｜ Embedding: Qwen3-Embedding-8B

### 咱们的用途
1. **ClawsGO 算力耗尽时的第二意见来源**（gemini-3.8-flash-high / grok-4.7 / deepseek-v4-pro 交叉核验引文与方案）；
2. 0.5B 蒸馏教师的**候选生成通道之一**（远期，需先过数据治理评估——中转站可能记录生成内容，教师生成数据进训练池前须评估此风险）；
3. 快速推理兜底。

### 快速调用
```bash
curl https://api.krisphyn.cc/v1/chat/completions \
  -H "Authorization: Bearer sk-go28BcSeT4Ig14N30NVLfDq6fZKquKRLRW0S3zdhosXGwa9U" \
  -H "Content-Type: application/json" \
  -d '{"model":"gemini-3.8-flash-high","messages":[{"role":"user","content":"..."}]}'
```

### 待核验清单
- [ ] gemini-3.8-flash-high 实际 served 模型验证（防映射降级）
- [ ] 速率限制 / 日配额探测
- [ ] grok-4.7 真伪（是否真 grok 通道）
