# r4_corpus — R2–R5 分级语料谱系（仅本地 / NAS，不入 GitHub）

2026-10-11 从 node1 NAS 拉取的**全部分级训练语料**（约 1.6GB）。

`*.jsonl` 被仓库 .gitignore 排除（语料不进 GitHub），本目录只存在于本地与 NAS；
`manifest.txt` 为拉取时生成的清单（原始字节数 + sha256 + 分片对应关系）。

## 文件对照（manifest 摘要）
| 语料 | 用途 | 大小 |
|---|---|---|
| `zh_text_r2*.jsonl` | 0.1B 档 R2 小说（含 _cat 分类版） | ~75MB ×3 |
| `zh_text_r3.jsonl.part0*` | 0.1B 档 R3 | 305MB |
| `zh_text_lang.jsonl` / `zh_text_r4.jsonl` | 0.2B 档 R4 语言基础 | 77MB |
| `zh_text_r5.jsonl.part0*` | 0.2B 档 R5 知识·教育 | 231MB |
| `zh_text_know_edu/know_full.jsonl.part0*` | 知识档（R5/R6 用） | 133MB / 256MB |
| `zh_text_next*/_wiki_tmp.jsonl` | 维基与下一代语料中间产物 | 75–123MB |

> 注：R6 实际用的是 `chain/corpus_reserve/` 四档蒸馏精选（75MB），不是这里的全量原始语料。
