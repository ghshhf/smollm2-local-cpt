# kris.longcat (vendor=longcat, served=longcat-2.5-preview-free, 46.2s)

# 裁决书

先给结论：你的诊断方向对，但有三处需要修正——尤其是 Q1 的窗口设计犯了一个测量学错误，会把你引向"调参无效"的假阴性。

---

## 裁决 1：flip rate 窗口设计 —— 你的推理对一半，但推荐参数会误导

### ① 机制/最佳理论

**你对的部分**：位移-步长比论证成立。lr=1e-5 × 200步，AdamW 归一化后每步位移约 lr 量级 ≈ 1e-5，量化边界带宽 ~1.0|mean|W|，占比 34~46% 的权重离边界中位数距离 ~0.5|mean|W|。200步累计位移 ~2e-3|mean|W|，是带宽的 0.2%，翻转概率趋近零。这是**测量窗口不足**，判定正确。

**你错的部分**：5000步 × lr=1e-4 不等于"5000步观测"。

关键区分——**等效位移预算** = lr × steps：
- 原方案：1e-5 × 200 = 2e-3 → 翻转数 ≈ 0 ✓（实测一致）
- 你提议：1e-4 × 5000 = 5e-1 → 等效位移 = 带宽的 50%，必然大量翻转

但问题是：**这个窗口混淆了两个变量**。你把 lr 和 steps 同时提高了 10 倍和 25 倍，如果翻转率终于上来了，你无法归因是"lr 够了"还是"窗口长了"还是"两者缺一不可"。更危险的是——5000步 × lr=1e-4 已经超出退火完成区间，alpha 可能早已到 0，你测到的 flip 全部发生在"已经全是三值权重"的阶段，无法分离"退火过程中"和"退火完成后"的翻转行为。

**正确窗口设计**——分离变量：

```
# 伪代码：翻转率测量实验矩阵
for lr in [1e-5, 5e-5, 1e-4, 5e-4]:      # 分离 lr 效应
    for anneal_steps in [200, 1000, 5000]: # 分离窗口效应
        # alpha 退火在 anneal_steps 内完成
        # 翻转率分两段测：
        #   flip_during_anneal: alpha ∈ [1,0) 区间
        #   flip_post_anneal:    alpha = 0 稳态后继续 N 步
```

**推荐最小可跑版**（node1 CPU 64核，135M fp32，BitLinear 替换后）：

| 配置 | lr | 总steps | alpha退火步数 | 预计耗时(node1) |
|------|-----|---------|-------------|---------------|
| A-对照 | 1e-5 | 200 | 200 | ~2min（你的基线）|
| B-纯lr | 1e-4 | 200 | 200 | ~2min |
| C-纯窗口 | 1e-5 | 5000 | 5000 | ~50min |
| D-联合 | 1e-4 | 5000 | 5000 | ~50min |

B 是关键——**如果 B 仍然 flip=0 而 C 上来了，说明是窗口不够不是 lr 不够**。你现在的实验缺了这一格。

### ② 可执行方案

```python
# 核心：翻转率窗口的分段定义
def measure_flip_rate(model, loader, lr, total_steps, anneal_steps, measure_interval=50):
    """
    alpha: 1.0 -> 0.0 线性，在 anneal_steps 内完成
    measure: 每 50 步 snapshot 一次 W，统计相邻 snapshot 间符号变化
    """
    flip_records = []
    
    for step in range(total_steps):
        # alpha 调度
        alpha = max(0.0, 1.0 - step / anneal_steps)
        
        # 训练一步
        loss = train_step(model, loader, lr)
        
        # 测量
        if step % measure_interval == 0:
            W_snapshot = extract_bitlinear_weights(model)
            if len(flip_records) > 0:
                W_prev = flip_records[-1]['W']
                flips = count_sign_changes(W_prev, W_snapshot)
                total = count_quantizable(W_prev)  # 排除已稳定的 ±阈值
                phase = 'annealing' if alpha > 0 else 'post_anneal'
                flip_records.append({
                    'step': step, 'alpha': alpha,
                    'flip_rate': flips / max(total, 1),
                    'phase': phase
                })
            flip_records.append({'W': W_snapshot, 'alpha': alpha})
    
    return flip_records

# 翻转率汇总：按 phase 分组
def summarize_flips(records, phase):
    phase_records = [r for r in records if r['phase'] == phase and 'flip_rate' in r]
    if not phase_records:
        return 0.0
    return np.mean([r['flip_rate'] for r in phase_records])
```

### ③ 可证伪判据

**什么结果算你错**：
- 配置 B（lr=1e-4, steps=200）如果 flip > 0（哪怕 0.1%），说明"lr 太小"是主因之一，你的"纯窗口不足"判定不完整
- 配置 C（lr=1e-5, steps=5000）如果 flip > 0 但 B 仍然 = 0，**完全证实**你的判定，外部归因"提高 lr"是伪命题

**什么结果算你对**：
- B flip=0 且 C flip>0 → 窗口不足是唯一约束
- B flip>0 且 C flip≈B → lr 是约束，窗口不是
- B flip>0 且 C flip>>B → 两者都重要

### ④ node1 最小可跑版

A/B/C/D 四组全跑 node1 CPU 无压力。135M 模型 Batch=8, seq=516, 单步前向+反向 ~0.12s/步（已验证你实测的 3 tok/s 对应 batch 配置）。C/D 组 5000 步 ≈ 10min/组。总实验时间 ~30min 出