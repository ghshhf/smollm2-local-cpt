# zen.deepseek (vendor=deepseek, served=deepseek-v4-flash-free, 70.8s)

### ① 机制/最佳理论

**核心判词：宪法条文「主干预训练加载路径」的边界锚定在「参数更新的梯度源头」，而非「数据加载器的物理来源」。**

1.  **成熟范式不存在「阶段性降级」，只存在「路径隔离」**：
    *   **Gemma-3 / Llama-3 / Nemotron-3 等 SOTA 均采用「双流水线」**：`Mainline Pretrain (Human-only) → Post-train (Synthetic-heavy)`。从未在 *同一条* 参数更新链路里混入合成数据再声称「预训练纯净」。
    *   **RLAIF/蒸馏/自我进化** 全属于 **Post-training / Alignment / Continual Learning** 阶段，**不计入** 「主干预训练加载路径」。
    *   **「可审计/可回滚」是工程手段，非宪法豁免权**。宪法禁的是「污染主干权重的分布」，不是「硬盘里存没存合成数据」。

2.  **「成为自己 > 学到最好」的实现机制是「冻结主干 + 外挂/蒸馏头」**：
    *   0.4B 之前：主干只见人写 → 权重分布 $P_{human}(\theta)$ 固化。
    *   0.4B 之后（含 0.5B 蒸馏、联网自学、认主头）：**全走 LoRA / Memory Layer / Distillation Head**，主干参数 **冻结或仅做极小步长 KL 正则更新**。
    *   只要主干参数 $\theta_{backbone}$ 的梯度 $\nabla \mathcal{L}_{synthetic} \approx 0$，宪法即满足。

3.  **0.5B 蒸馏教师数据 ≠ 主干预训练路径**：
    *   蒸馏目标：$\mathcal{L}_{distill} = D_{KL}(Student_\theta || Teacher_{frozen})$。
    *   教师生成的合成数据喂给 **Student 的非主干模块**（投影层、记忆层、输出头）或 **全参数但受强 KL 约束防止灾难性遗忘**。
    *   若 0.5B 架构变更（BitNet b1.58 深窄），属于 **Net2Net 结构映射 + 蒸馏初始化**，而非「继续预训练」。此时加载的教师 logits/软标签是「初始化手段」，不是「预训练语料」。

---

### ② 可执行方案（伪代码 + 工程清单）

#### A. 宪法合规的「双流水线」架构（硬隔离）

```python
# ========== 宪法护栏：编译期/运行期双检查 ==========
class ConstitutionalGuard:
    # 1. 数据血统标签（强制枚举）
    class DataLineage(Enum):
        HUMAN_RAW      = 0  # 仅此标签允许进入 mainline_pretrain_dataloader
        SYNTH_SFT      = 1  # 指令微调/蒸馏/偏好数据
        SYNTH_RL       = 2  # RLAIF/在线自学习轨迹
        OWNER_CORPUS   = 3  # 认主语料（人写/私有，视同 HUMAN_RAW）
        MEMORY_WRITE   = 4  # 外置记忆层写入向量（不回传主干梯度）

    # 2. 数据集构建器：物理隔离 + 标签强绑定
    def build_dataloader(cfg, lineage: DataLineage):
        assert lineage in ALLOWED_LINEAGE[cfg.phase], \
            f"Phase {cfg.phase} 禁用 {lineage}"
        ds = load_from_disk(cfg.path[lineage])
        ds = ds.map(lambda x: {**x, "__lineage__": lineage.value})
        return DataLoader(ds, ...)

    # 3. 训练循环：梯度流向闸门
    def train_step(batch, model, phase):
        # 编译期静态图裁剪：mainline 阶段直接切断非 HUMAN_RAW 梯度
        if phase == "mainline_pretrain":
            with torch.no_grad():  # 硬阻断：任何非人写 batch 根本不进前向
                if batch["__lineage__"] != DataLineage.HUMAN_RAW.value:
                    return {"loss": 0, "skipped": True}
        # post-train 阶段：主干冻结，只算 adapter/memory/head 梯度
        elif phase in ["sft", "distill_0.5B", "self_evolve"]:
            freeze(model.backbone)  # 含 embedding(fp16) + transformer blocks
            unfreeze(model.adapter, model.memory, model.owner_head)
        return standard_forward_backward(batch, model)

# ========== 阶段调度表（写死在宪法/脚本里，不可热更改） ==========
PHASE_SCHEDULE = {
    "0.1B->0.4B_net2net": {
        "phase": "mainline_pretrain",
        "allowed_lineage": [DataLineage.HUMAN_RAW, DataLineage.OWN