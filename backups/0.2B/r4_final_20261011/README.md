# r4_final_20261011 — R4 语言档训练后权重（0.2B）

**本库最新的 0.2B 成品权重**：2026-10-11 从 node1 NAS 拉取，R4（zh_text_lang 语言基础）训练后的 214M 权重。

## 内容
| 文件 | 说明 |
|---|---|
| `pytorch_model.bin.part00` … `part08` | 权重 9 片（8 × 95MiB + 56.98MiB，合计 856,674,211 B） |
| `config.json` | 576/3072/30 层/vocab 49152/tied（配权重加载） |
| `generation_config.json` / `tokenizer.json` / `tokenizer_config.json` | 四件套 |

## 拼回
```bash
cd backups/0.2B/r4_final_20261011
python reassemble.py        # 大小断言 -> 还原到 ./restored/
```

## provenance
- 拉取时间 2026-10-11 21:37（git 提交 5f439ea，"feat(R4): GitHub-only 取数路径"）。
- R4 于 2026-10-10 从 step 0 重启（0.2B 重建+保函数校验后），本权重为其训练产物。
- 后续 R5（知识·教育）/ R6（全量知识，01:47 封顶 done=7100）权重未拉回——见 [../README.md](../README.md) 缺口说明。
