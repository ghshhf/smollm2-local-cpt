# 远程仓库盘点与整理（node1 / 2026-10-19 12:5x）

## 结论

1. 远程**不是缺空间**：`/mnt/workspace` 是阿里云 NAS 挂载，1.0P、已用 49T（5%）、可用 976T。问题是**结构和重复**，不是容量。
2. `node1` 实占 **49G**，其中 `chain` 16G + `corpus_04b` 14G 占六成，模型权重 7.5G 只是零头。
3. **已清 2.91G 纯垃圾/纯搬运副本**（见 A 级，证据在下面），零风险，可重建。
4. 还有 **约 15.1G 是"历史代遗留"**（B 级），需要点头才动，因为删了不能回滚。
5. 真正的病灶是**根目录摊了 172 个文件**：53 个 `.out`、38 个 `.log`、31 个 `.py`、22 个 `.txt`、19 个 `.sh`。这就是今天自检脚本"根选偏、模型认错"的土壤。

## 一、现状分布（清理前 49G）

| 目录 | 大小 | 性质 |
|---|---|---|
| `node1/chain` | 16G | 0.2B 时代续训链：checkpoint 四代 + 分词缓存 + 语料备份 |
| `node1/corpus_04b` | 14G | 0.4B 语料（l3_ms 7.4G、ufw_zh 3.9G、ufw 2.4G…） |
| `node1/models` | 7.5G | 权重：自家 0.3B 1.3G、siglip2 1.5G、whisper 927M、千问两代 3.9G |
| `node1/gh_repo` | 3.0G | 档案仓 |
| `node1/_pack_siglip` | 2.9G | **已删**（siglip 搬运中转） |
| `node1/sftboot` | 1.4G | 引导包（base + tok + data + tools） |
| `node1/smoke_ckpt` | 1.1G | 10 月 11 日冒烟产物（step30/final） |
| `node1/corpus_wiki` | 933M | 维基 parquet 3 片 |
| `node1/data` | 911M | 种子 92713B + 银行 356352B + 其它历史语料 |
| `node1/audio_parts` | 883M | 听觉线分片 |
| `node1/bench_dl`、`cli_dl` | 各 201M | 两个同期的下载基准目录 |

远程硬件：**8 核实核（cgroup），无 GPU**，CPU 纯跑。

## 二、重复实证（抽样首 8MB 的 md5 前 12 位）

| 权重文件 | 抽样 md5 | 判定 |
|---|---|---|
| `sftboot/base` | `1849f7fc6332` | **与下一行完全相同** |
| `chain/ckpt_gpu/checkpoint-03b` | `1849f7fc6332` | 同上 → 二者留一 |
| `models/sft_code_03b` | `27795d3ebc59` | 训练后权重，唯一，**必留** |
| `chain/ckpt` | `6618ac13cf1c` | 不同代 |
| `chain/ckpt_old` | `2cab84497331` | 不同代 |
| `chain/ckpt_gpu/checkpoint-latest` | `b341c108d84c` | 不同代 |

注：0.3B 的每个 checkpoint 文件大小都恰好 1,387,470,856 字节（同架构同精度），**大小相同不代表内容相同**，所以这里用抽样 md5 判，不靠字节数。

## 三、A 级：已清（2.91G，零风险）

| 对象 | 大小 | 判据 |
|---|---|---|
| `node1/_pack_siglip` | 2.9G | 里面是 `siglip2base.tar` 1.53G + 38 个 40MB 分片 —— **同一份权重的两种搬运中间格式**；落地文件 `models/siglip2-base/model.safetensors` header 校验通过（408 个张量），原文可随时重下 |
| `node1/relay_dl2` | 7.7M | 与 `relay_dl` 同名同大小，重复下载 |
| `/mnt/workspace/https:` | 5.5K | curl 把 URL 当路径建出的空壳（`modelscope.cn` 空目录） |
| `/mnt/workspace/!pip install -q transformers peft accele` | 4K | 命令被当文件名建的空目录 |
| `models/resume_20261019_123046`、`_123625` | 各 512B | 今天两次崩溃留下的空目录 |

## 四、B 级：历史代遗留（约 15G，**已分类归档，不删**）

| # | 对象 | 大小 | 是什么 | 建议 |
|---|---|---|---|---|
| 1 | `models/Qwen2.5-0.5B-Instruct` 954M + `Qwen2.5-Coder-1.5B-Instruct` 2.9G + `code_lora_qwen` 8.3M | 3.9G | **上一轮"视频 + LoRA"路线的底座与适配器** | 归档或删；现在统一走自家 0.3B，只做历史对照就打包压起来 |
| 2 | `chain/ckpt` 1.3G、`ckpt_old` 1.3G、`ckpt_gpu/checkpoint-latest` 1.6G、`ckpt_gpu/run_live` 1.5G | 5.7G | 0.2B chain 四代检查点 | 只留 `checkpoint-latest` 一代，其余删（旧步无法回滚） |
| 3 | `chain/cache_R5.npy` 1.1G、`cache_R6.npy` 1.2G | 2.3G | R5/R6 时代分词缓存 | 可重算，但费时；不急可留 |
| 4 | `corpus_04b/ufw_zh.jsonl` 3.9G vs `corpus_04b/ufw/*.parquet` 2.4G；`chain/corpus_reserve/zh_wiki_all.jsonl` 1.5G vs `corpus_wiki/*.parquet` 933M | 6.3G（含保留部分） | 同一批数据的 **jsonl 与 parquet 双份** | 每类只留一种格式（训练读哪个留哪个） |
| 5 | `sftboot/base` 1.3G | 1.3G | 与 `chain/ckpt_gpu/checkpoint-03b` **md5 相同** | 二者留一 |
| 6 | `/mnt/workspace/models/models` | 260M | SmolLM2-**135M** 两份（10 月 11 日老基线） | 删；现在最低是 0.3B |

## 五、结构方案（未执行）

现状：`node1` 根目录 **172 个文件横着**（53 `.out` / 38 `.log` / 31 `.py` / 22 `.txt` / 19 `.sh` / 1 `.bak`），脚本、日志、命令回显、备份全平铺在同一层。

建议收敛为：

```
node1/
  models/     权重：每个角色只留一代 + latest 指针
  data/       种子 / 银行 / holdout
  corpus/     语料：每种只留一种格式
  tools/      在用的脚本（sft_full.py、dream_code.py、seed_to_bank.py…）
  logs/       只留最近 7 天，旧的进 archive
  chain/      只留当前一代
  archive/    .out、.txt、.bak、旧代检查点
```

为什么值得做：自检脚本 `boot_resume.py` 靠"浅扫 models 取 mtime 最新 + 排除 LoRA"来自己认续训点。**副本越多，它越容易认错**——今天就是因为根选偏（选到 `/mnt/workspace/models`）白跑了两轮。目录收敛之后，自适应才真的可靠。

## 六、与"过去是什么"的对账

- 过去远程那轮 = **代码生成视频**，底座是**千问**（`Qwen2.5-0.5B` / `Qwen2.5-Coder-1.5B`）+ `code_lora_qwen` 适配器，训练方式是 **LoRA**。
- 现在 = **自家北极星 0.3B**（SmolLM2 血统），训练方式是**恒维 q/k/v/o 原地参数增量**（31.9M/346.9M = 9.2%）。
- 本轮选的**图片识别**同样用自家 0.3B：`siglip2-base`（已落地，1.5G，408 张量）出视觉编码 → 投影 → 北极星 0.3B 解码，训的也是恒维增量，不碰千问。

## 七、执行结果（2026-10-19 12:5x 落地）

### 7.1 归档（不删，只归类）：`archive/2026-10-19/`，合计 **15G**

| 桶 | 大小 | 内容 |
|---|---|---|
| `legacy_qwen/` | 3.9G | Qwen2.5-0.5B、Qwen2.5-Coder-1.5B、code_lora_qwen |
| `legacy_chain/` | 6.9G | chain/ckpt、ckpt_old、ckpt_gpu |
| `legacy_cache/` | 2.2G | cache_R5.npy、cache_R6.npy |
| `legacy_base/` | 1.3G | sftboot/base（与 ckpt_gpu/checkpoint-03b 抽样 md5 相同） |
| `legacy_smollm135/` | 260M | SmolLM2-135M 两份 |

归档区自带 `README.md`（每桶内容 + 为什么 + `mv` 回去的恢复命令）。规矩：**归档区只增不删**。

归档后 `models/` 只剩当前在用的：`sft_code_03b`、`sft_seed200_remote`、`siglip2-base`、`whisper-small`、`data`。

### 7.2 延迟到明天再动（今天没碰）

`chain/` 还剩 6.9G：`corpus_reserve` 2.7G、`r4base_std` 1.8G、`backup_corpus` 1.6G 及零散脚本/日志。这些和 0.4B 语料可能同源，要先比对再决定，不动。

### 7.3 远程命令行 `node/rc.py`

```bash
python node/rc.py ls models -l      # 列目录（在线, 1~3s）
python node/rc.py tree -d 2         # 目录树
python node/rc.py du                # 占用排序
python node/rc.py cat data/seed_code_200.jsonl -n 3
python node/rc.py find '*.safetensors'
python node/rc.py ps                # 进程
python node/rc.py svc               # 服务面板: 通道/隧道/训练/jupyter/端口/负载
python node/rc.py sh 'df -h /mnt/workspace'
python node/rc.py sync              # 全量扫描 → 本地索引
python node/rc.py idx               # 目录一览（离线, 0ms）
python node/rc.py idx qwen          # 关键词搜（离线）
python node/rc.py md                # 只重建 markdown 索引（离线）
```

本地索引：`data/remote_index.json`（717 条，机器读）+ `docs/remote-index.md`（人读，含每层目录用途、仓库地址、磁盘、torch 版本）。**远程关机也能查。**

踩过的两个坑（已修，写进代码注释）：
- `mb.py` 在命令末尾拼 `| curl`，裸写 `a; b; c` 只有最后一条进管道 → 统一包 `{ ...; } 2>&1`；
- 远程只吃单行命令，多行会被截断且表现为"(无回传)" → 传输层统一把换行折成 `; `。
