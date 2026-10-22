# 音频对齐（align）训练失败诊断报告

日期：2026-10-19 17:40 ｜ 节点：node1（魔搭 8 核，实按 3~4 线程）｜ 主干：polaris-03b（346.9M / 36 层 / hidden 576 / vocab 49152）

## 一、现象

0.3B 音频对齐训练跑满 800 步（3574s，4.5s/步），训练后盲测：

```
[EVAL] 平均 CER = 7.422  中位 = 4.000  完全正确 = 0/10
```

- 参考「家中接连有了四位重病患者」→ 输出「与其他高校的学生之间的联系起来」
- **10 条盲测里有 6 条输出完全相同的句子**
- 训练日志 loss 在 8.9 ~ 17.9 之间来回震荡，800 步后仍是 14.49，从无下降趋势

表面上像是"训练崩了 / 梯度爆炸 / 步数不够"三选一，实际全不是。

## 二、五个诊断实验（逐个排除）

| # | 实验 | 结果 | 排除的猜测 |
|---|---|---|---|
| 1 | `tools/grad_probe.py` 量梯度 | total_grad = 85 / 12.7 / 37（不是 1e-8），4 个参数块都有梯度 | ❌ 梯度消失 |
| 2 | 同上，量 projector 输出尺度 | `audtok_norm = 39.93`，LLM embedding 行 norm 仅 2.978（13.4 倍） | ⚠️ 尺度异常但**非致命** |
| 3 | `tools/text_ppl.py` 测底座纯文本 NLL | **2.505 ~ 2.664**（PPL 12~14），chat 格式 3.735 | ❌ 底座 / 词表 / checkpoint 有问题 |
| 4 | `tools/feat_diag.py` 测特征区分度 | 不同音频两两余弦 mean=0.864（min 0.80 / max 0.94）；缓存 key=utt，剔除缺失 wav 后命中≈100% | ❌ 特征塌缩 / 缓存不命中 |
| 5 | `tools/prefix_scan.py` 扫 prefix 长度 | 1 token: 3.802 → 150 token: 3.839，**几乎无差别** | ❌ prefix 太长是毒源 |

特别说明第 2 项：projector 输出 norm 39.93 看似吓人，但 Llama RMSNorm 是**逐 token 归一化**，每个 token 归一化后 norm 恒为 `sqrt(d)·w`，与输入尺度无关。所以尺度不是病因，这条一度被误判为根因，已收回。

## 三、真因：一个显示 bug 误导了整次判断

`tools/audio_attach.py` 训练循环里：

```python
loss_acc += float(loss)
if (i + 1) % a.accum == 0 or i == order[-1]:
    ...
    step += 1
    if step % 10 == 0:
        print("loss=%.4f" % (loss_acc / 10))   # ← 这里
        loss_acc = 0.0
```

`loss_acc` 的累积窗口是 **accum × 10 = 40 条样本**（每 10 个 optimizer step 才清零一次），却硬除以 10。

⇒ 显示值 = 真实 loss × 4。

**真实 loss 一直就是 3.80，800 步后 3.62。** 也就是说：

- 模型没有崩，梯度没爆，底座没坏
- 它只是**几乎没学到东西**（800 步只降 0.18）

## 四、真正的瓶颈：训练量只有 0.6 个 epoch

- 数据：5464 条训练样本，`accum=4` ⇒ 每个 optimizer step 吃 4 条
- 800 step × 4 = **3200 条样本 = 0.59 个 epoch**

可训练参数只有 projector 的 **0.78M**，主干 36 层全冻结。要从随机初始化学会「把 whisper 帧映射进 0.3B 的语义空间，再让它转录」，0.6 个 epoch 在原理上不可能。

对比参照：LLaVA 类结构同样只训 projector，但打底是几万到几十万 step、几十万条数据。

## 五、修复方案

| 项 | 原值 | 新值 | 理由 |
|---|---|---|---|
| loss 显示 | `loss_acc / 10` | `loss_acc / acc_n` | 修掉 4 倍放大，避免再次误判 |
| `aud-tokens` | 150 | **32** | prefix_scan 实测 loss 无损失（3.801 vs 3.839），序列 219→101，速度约 2× |
| `accum` | 4 | **8** | 有效 batch 8，同样的数据量下步数减半即可达到同等样本覆盖 |
| `steps` | 800 | **6000** | 6000×8 = 48000 样本 ≈ 8.8 epoch，才够 projector 收敛 |
| 评估 | 只在末尾 | **每 500 步存 checkpoint + 抽样评** | 便于中途判定是否真的在学 |

预计耗时：`aud-tokens=32` 后单步约 2.5s ⇒ 6000 步 ≈ **4.2 小时**。

## 六、验收标准（禁信 loss）

1. 训练中每 500 步，用 3 条**训练集内**样本做 in-set completion，loss 需 < 0.6 才算学会
2. 结束后跑 `--eval-n 20` 真盲测（S0018/S0019 未见说话人）
3. 目标：CER < 0.5（当前 7.42，等于完全不认识）
4. 若 6000 步后 CER 仍 > 2.0 ⇒ 该方法对 0.3B 不成立，改走 LoRA（挂 q/k/v/o 恒定维，符合宪法），或退化为「Whisper 出文本 + LLM 后处理」

## 七、可复用工具（本次新增）

| 文件 | 用途 |
|---|---|
| `tools/grad_probe.py` | 量投影层梯度与输出尺度，判梯度消失 |
| `tools/feat_diag.py` | 特征区分度（中心化余弦 + 互信息代理） |
| `tools/text_ppl.py` | 底座纯文本 NLL，判定 checkpoint / 词表是否正常 |
| `tools/prefix_scan.py` | 扫 prefix 长度对 loss 的影响 |
| `tools/emb_probe.py` | 量 embedding / RMSNorm 的真实尺度 |

这五个脚本都是独立的，不改主流程，换任何模态（视觉、音频）都能直接复用。
