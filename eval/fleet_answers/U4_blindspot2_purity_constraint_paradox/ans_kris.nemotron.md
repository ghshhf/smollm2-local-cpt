# kris.nemotron (vendor=nvidia, served=nvidia/nemotron-3-ultra-550b-a55b, 74.9s)

### � 裁决结果：维持宪法硬约束，**三条红线全守住**，给出可落地的“双轨制”工程路径

---

#### ① 机制/最佳理论
**核心判词**  
- **“主干预训练加载路径” ≡ 权重更新轨迹的 *优化器状态* 所见 token 流**。  
  凡进入 `optimizer.step()` 且产生 `grad → weight delta` 的 token，均属该路径；蒸馏数据、合成数据、联网数据若 **不触发主干反向传播**（只走 LoRA/Adapter/外置记忆/蒸馏头），即 **不计入** 该路径。  
- **成熟范式确认**：  
  1. **Gemma-3 / LLaMA-3 官方蒸馏流程**：教师生成 → 离线清洗 → **仅作 SFT/蒸馏损失**，预训练阶段冻结主干，不污染预训练轨迹。  
  2. **RLAIF 隔离范式 (Bai et al., 2022; OpenAI RLAIF blog)**：合成数据仅进入 **Reward Model 训练** 与 **PPO/Online DPO 的 Policy 头**，主干参数若冻结或仅走 LoRA，即满足“预训练路径纯净”。  
  3. **Net2Net 扩容不改变上述定义**：扩容瞬间复制权重，后续继续预训练的 token 流仍受宪法约束。  

**结论**：阶段性降级**无成熟范式可引用**——业界做法均为“物理隔离优化器状态”，而非“时间切片放宽纯度”。

---

#### ② 可执行方案（伪代码级）

```python
# ======== 宪法合规的三阶段数据管线 ========
# 阶段 0: 0.1B→0.4B Net2Net 扩容 + 中文词表扩展 + owner_corpus 注入
#   - 仅加载 human_corpus.jsonl (SHA256 审计通过)
#   - 优化器见到的 token 100% 人写
#   - 认主头 (OwnerHead) 随机初始化，冻结主干，仅训练 Head + 新增 Embedding

# 阶段 1: 0.4B 认主对比微调 (≤ 1 epoch)
#   - 数据：owner_corpus + 少量人工标注 (pos/neg pair)
#   - Loss = InfoNCE(owner_emb, pos_emb) - InfoNCE(owner_emb, neg_emb)
#   - 主干冻结，仅更新 OwnerHead + 新增 token Embedding
#   - 产物：0.4B_owner.pt  (主干权重哈希不变)

# 阶段 2: 0.5B 原生 BitNet b1.58 QAT 从头训练 (主干预训练)
#   - 架构：深窄 (L=32, H=1024, FFN=4096) + 三值权重 QAT
#   - 数据：human_corpus_v2.jsonl (新增人工清洗，仍 0 合成)
#   - 优化器见到的 token 依然 100% 人写 → 宪法通过
#   - 产物：0.5B_bitnet_pretrain.pt

# 阶段 3: 能力层蒸馏继承 (离线，主干冻结)
#   教师 = 0.4B_owner.pt (fp16) + 外部强教师 (可选)
#   学生 = 0.5B_bitnet_pretrain.pt (冻结主干, 仅解冻 LoRA-A/B + 蒸馏头)
#   数据：教师生成的合成数据 synth_distill.jsonl
#   关键：optimizer.step() 仅更新 LoRA + 蒸馏头，**主干梯度置零**
#   产物：0.5B_bitnet_distilled.pt (主干哈希仍等于阶段2)

# 阶段 4: 外置可寻址记忆层 + 联网自学习 (全生命周期)
#   - 记忆层：Key-Value 稀疏矩阵 (10M/用户)，删除权 0/1 硬编码
#   - 联网抓取 → 清洗 → 写入记忆层，**不触发主干优化器**
#   - 看门狗服务：定期 checkpoint 记忆层，主干权重只读

# ======== 审计钩子 (CI 集成) ========
def audit_purity(ckpt_path, stage):
    if stage <= "0.4B_pretrain":
        assert get_optimizer_token_source(ckpt_path) == "human_only"
    elif stage == "0.5B_pretrain":
        assert get_optimizer_token_source(ckpt_path) == "human_only"
    elif stage == "distill":
        assert get_main_grad_norm(ckpt_path