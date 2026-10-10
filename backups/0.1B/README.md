# 0.1B 里程碑备份 (SmolLM2-135M + R2 小说续训)

本目录是 **0.1B 这一站** 的离线快照，用于防止本地盘损坏 / 实验改崩后丢失已训成果。
推到 GitHub 私有库 `ghshhf/smollm2-local-cpt` 的 `backups/0.1B/`。

## 包含内容
| 文件(解压后) | 说明 | 约体积 |
|---|---|---|
| `base/` | SmolLM2-135M 底座 (config + tokenizer + safetensors) | 260MB |
| `model/pytorch_model.bin` | **R2 训完的 0.1B 权重** (全量 CPT, 小说分类语料 37721 篇) | 538MB |
| `model/step.txt` | 落盘步数 | 3B |
| `corpus/zh_text_r2_cat.jsonl` | R2 训练语料 (带 cat 题材标签) | 76MB |

## 不含
- `optimizer.pt` / `scheduler.pt`（续训态, 1GB+, 里程碑快照不需要；从权重续训时训练脚本会重建 optimizer）

## 分片与还原
- 压缩包被切成 8 片 `0.1B_part_00.zip` … `0.1B_part_07.zip`（每片 <100MB，绕过 GitHub 单文件硬限）。
- `MANIFEST.json`：分片清单 + 合并后 sha256（还原时逐片 + 整体校验）。
- `reassemble.py`：纯标准库重组脚本（跨平台）。

```bash
cd backups/0.1B
python reassemble.py        # 校验 sha256 -> 还原 0.1B.zip -> 解压到 ./0.1B_extracted/
```
解压后：`base/` 可直接 `from_pretrained`；`model/pytorch_model.bin` 是全模型权重（配 `base/` 的 config+tokenizer 加载）。

## 重建本备份
`make_0.1B_backup.py` 是生成脚本（重新跑会按当前本地 0.1B 资产再打包一次）。

## 升级链上下文
0.1B(R2) → Net2Net 保函数扩到 0.2B(214M, R4/R5) → 0.3B(350M, R6)。
扩容宪法：绝不改 注意力/隐藏维/GQA头数/RoPE，只加宽 MLP + 零输出恒等层。
