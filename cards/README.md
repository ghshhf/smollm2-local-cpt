# 知识卡（Knowledge Cards）使用说明

训练产出 4 张 LoRA 卡（V3→V6），其中真正正交的只有两张：

| 卡 | 目录 | 内容 |
|---|---|---|
| 小说/世界观卡 | `out_lora_v5` | V3+V4+V5 全部（含两本同人、守则宇宙）。V5 已包含 V3/V4，无需叠加。 |
| 代码卡 | `out_lora_v6` | 1958 条纯代码（游戏 JS 函数、量化/链上脚本、20 条高难范式）。 |
| （未来）推理卡 | `out_lora_v7` | V7 蒸馏推理链，教"先想后做"。 |

## 两种使用模式

### 模式 A：按卡切换（推荐）
不同领域互不干扰，随时换卡。
```
python cards/mount.py --adapter out_lora_v5      # 聊小说设定
python cards/mount.py --adapter out_lora_v6      # 写/读代码
python cards/mount.py --list                     # 看当前有哪些卡
python cards/mount.py                            # 仅基座
```

### 模式 B：合并成单 blob
把卡烘焙进基座，产出一个独立模型目录，之后直接加载、不用挂适配器。
```
python cards/merge_card.py --adapter out_lora_v5 --out model_merged_novel
python cards/merge_card.py --adapter out_lora_v6 --out model_merged_code
# 全功能（小说+代码）：先合 V5 再合 V6
python cards/merge_card.py --adapter out_lora_v5 --out _m1
python cards/merge_card.py --base _m1 --adapter out_lora_v6 --out model_merged_full
```

## 推理增益（V7 推理卡）
1.5B 容量有限，堆数据/步数收益小；真正有效的是**蒸馏强模型的思考链(CoT)**。
```
python build_dataset_v7_reason.py                          # 离线模板版（立即可训）
python build_dataset_v7_reason.py --teacher <端点> --model <老师id>   # 老师蒸馏版（真实增益）
```
然后在 `train_chain.py` 的 JOBS 末尾加 `("v7","data/role_data_v7.jsonl","out_lora_v7",760)` 重跑链。

> fp32 1.5B 在 CPU 上加载约 30-60s，生成偏慢，属正常。
