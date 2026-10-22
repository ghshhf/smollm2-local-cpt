# 远程 node1 索引（离线可查）

> 由 `python node/rc.py sync` 生成，本地镜像。远程关机也能看。

| 项 | 值 |
|---|---|
| 生成时间 | 2026-10-19T13:22:39+0800 |
| 主机 | node1（阿里云 NAS 挂载，8 核，无 GPU） |
| 根目录 | `/mnt/workspace/node1` |
| 磁盘 | 000104872f-caf15.cn-hangzhou.nas.aliyuncs.com:/7cfedfb0-8ff0-4a8a-a337-c316b2f9d8d8/  1.0P   49T  976T    5% /mnt/workspace |
| 仓库 | git@github.com:ghshhf/smollm2-local-cpt.git |
| 分支/HEAD | main / f465dfb 2026-10-11 情感路线: 需求即函数schema + 双版本行为档(主人分数开关/增长飞轮) |
| Python | torch 2.3.1+cpu tf 4.55.4 |

## 一、一级目录：叫什么、是什么

| 目录 | 大小 | 用途 |
|---|---|---|
| `.ipynb_checkpoints/` |  | Jupyter 检查点 |
| `__pycache__/` | 33K | Python 字节码缓存 |
| `archive/` | 15G | 归档区：历史代资产，只归类不删除 |
| `audio_parts/` | 883M | 听觉线音频分片（Whisper 训练素材） |
| `bench_dl/` | 201M | 下载基准对照目录（旧） |
| `chain/` | 6.9G | 旧续训链（0.2B 时代）剩余件 |
| `cli_dl/` | 201M | 命令行下载测试目录（旧） |
| `corpus_04b/` | 14G | 预训练/续训语料 |
| `corpus_wiki/` | 933M | 预训练/续训语料 |
| `data/` | 911M | 种子、银行、holdout、训练用 jsonl |
| `dream/` | 127M | 做梦器产物：衍生题、生成数据、旧 evolve 适配器 |
| `frp_0.61.1_linux_amd64/` | 34M | frp 内网穿透二进制 |
| `gh_repo/` | 3.0G | 档案仓 ghshhf/smollm2-local-cpt |
| `logs/` | 549K | 运行日志 |
| `mdl/` | 21K | 模型小工具/中间件 |
| `models/` | 5.0G | 权重：北极星 0.3B(sft_code_03b / sft_seed200_remote) / siglip2 视觉 / whisper 听觉 |
| `node/` |  | 信箱/自检/隧道/命令行客户端 |
| `relay_dl/` | 7.7M | relay 传输下载缓存 |
| `sftboot/` | 6.0M | 引导包：boot.sh + 数据 + tools（base 已归档） |
| `sftrun/` | 4.7M | 远程训练运行目录 |
| `smoke_ckpt/` | 1.1G | 10 月 11 日冒烟产物（step30 / final） |
| `tools/` |  | 在用的脚本（sft_full / dream_code / seed_to_bank…） |

## 二、二级目录（前 80 条）

| 路径 | 类型 | 大小 | 时间 |
|---|---|---|---|
| `.ipynb_checkpoints/node-checkpoint.log` | 文件 | 92B | 2026-10-11 14:00 |
| `.ipynb_checkpoints/node2-checkpoint.log` | 文件 | 0B | 2026-10-11 14:00 |
| `.ipynb_checkpoints/train-checkpoint.log` | 文件 | 8.5K | 2026-10-11 13:48 |
| `__pycache__/refine_reserve.cpython-311.pyc` | 文件 | 28.2K | 2026-10-15 06:00 |
| `archive/2026-10-19` | 目录 | - | 2026-10-19 12:55 |
| `audio_parts/ap_aa` | 文件 | 20.0M | 2026-10-18 11:41 |
| `audio_parts/ap_ab` | 文件 | 20.0M | 2026-10-18 11:42 |
| `audio_parts/ap_ac` | 文件 | 20.0M | 2026-10-18 11:42 |
| `audio_parts/ap_ad` | 文件 | 20.0M | 2026-10-18 11:42 |
| `audio_parts/ap_ae` | 文件 | 20.0M | 2026-10-18 11:42 |
| `audio_parts/ap_af` | 文件 | 20.0M | 2026-10-18 11:43 |
| `audio_parts/ap_ag` | 文件 | 20.0M | 2026-10-18 11:43 |
| `audio_parts/ap_ah` | 文件 | 20.0M | 2026-10-18 11:43 |
| `audio_parts/ap_ai` | 文件 | 20.0M | 2026-10-18 11:44 |
| `audio_parts/ap_aj` | 文件 | 20.0M | 2026-10-18 11:44 |
| `audio_parts/ap_ak` | 文件 | 20.0M | 2026-10-18 11:44 |
| `audio_parts/ap_al` | 文件 | 20.0M | 2026-10-18 11:44 |
| `audio_parts/ap_am` | 文件 | 20.0M | 2026-10-18 11:45 |
| `audio_parts/ap_an` | 文件 | 20.0M | 2026-10-18 11:45 |
| `audio_parts/ap_ao` | 文件 | 20.0M | 2026-10-18 11:45 |
| `audio_parts/ap_ap` | 文件 | 20.0M | 2026-10-18 11:46 |
| `audio_parts/ap_aq` | 文件 | 20.0M | 2026-10-18 11:46 |
| `audio_parts/ap_ar` | 文件 | 20.0M | 2026-10-18 11:46 |
| `audio_parts/ap_as` | 文件 | 20.0M | 2026-10-18 11:46 |
| `audio_parts/ap_at` | 文件 | 20.0M | 2026-10-18 11:47 |
| `audio_parts/ap_au` | 文件 | 20.0M | 2026-10-18 11:47 |
| `audio_parts/ap_av` | 文件 | 20.0M | 2026-10-18 11:47 |
| `audio_parts/ap_aw` | 文件 | 20.0M | 2026-10-18 11:48 |
| `audio_parts/ap_ax` | 文件 | 20.0M | 2026-10-18 11:48 |
| `audio_parts/ap_ay` | 文件 | 20.0M | 2026-10-18 11:48 |
| `audio_parts/ap_az` | 文件 | 20.0M | 2026-10-18 11:48 |
| `audio_parts/ap_ba` | 文件 | 20.0M | 2026-10-18 11:49 |
| `audio_parts/ap_bb` | 文件 | 20.0M | 2026-10-18 11:49 |
| `audio_parts/ap_bc` | 文件 | 20.0M | 2026-10-18 11:49 |
| `audio_parts/ap_bd` | 文件 | 20.0M | 2026-10-18 11:50 |
| `audio_parts/ap_be` | 文件 | 20.0M | 2026-10-18 11:50 |
| `audio_parts/ap_bf` | 文件 | 20.0M | 2026-10-18 11:50 |
| `audio_parts/ap_bg` | 文件 | 20.0M | 2026-10-18 11:50 |
| `audio_parts/ap_bh` | 文件 | 20.0M | 2026-10-18 11:51 |
| `audio_parts/ap_bi` | 文件 | 20.0M | 2026-10-18 11:51 |
| `audio_parts/ap_bj` | 文件 | 20.0M | 2026-10-18 11:51 |
| `audio_parts/ap_bk` | 文件 | 20.0M | 2026-10-18 11:52 |
| `audio_parts/ap_bl` | 文件 | 20.0M | 2026-10-18 11:52 |
| `audio_parts/ap_bm` | 文件 | 20.0M | 2026-10-18 11:52 |
| `audio_parts/ap_bn` | 文件 | 20.0M | 2026-10-18 11:52 |
| `audio_parts/ap_bo` | 文件 | 20.0M | 2026-10-18 11:53 |
| `audio_parts/ap_bp` | 文件 | 20.0M | 2026-10-18 11:53 |
| `audio_parts/ap_bq` | 文件 | 20.0M | 2026-10-18 11:53 |
| `audio_parts/ap_br` | 文件 | 20.0M | 2026-10-18 11:54 |
| `audio_parts/ap_bs` | 文件 | 2.6M | 2026-10-18 11:54 |
| `bench_dl/bench` | 目录 | - | 2026-10-18 14:13 |
| `chain/__pycache__` | 目录 | - | 2026-10-14 07:35 |
| `chain/_legacy_R7` | 目录 | - | 2026-10-19 03:17 |
| `chain/backup_corpus` | 目录 | - | 2026-10-14 02:25 |
| `chain/base_check.py` | 文件 | 3.1K | 2026-10-13 04:16 |
| `chain/base_tok` | 目录 | - | 2026-10-18 11:21 |
| `chain/bootstrap.log` | 文件 | 550B | 2026-10-12 23:31 |
| `chain/cache_R4.npy` | 文件 | 358.2M | 2026-10-13 03:01 |
| `chain/chain_fixrun.log` | 文件 | 15.8K | 2026-10-13 00:17 |
| `chain/chain_fixrun2.log` | 文件 | 236B | 2026-10-13 03:01 |
| `chain/chain_reset.log` | 文件 | 11.3K | 2026-10-13 00:05 |
| `chain/chain_resume_025419.log` | 文件 | 84B | 2026-10-17 02:54 |
| `chain/chain_resume_105335.log` | 文件 | 84B | 2026-10-18 10:53 |
| `chain/chain_resume_131322.log` | 文件 | 84B | 2026-10-18 13:13 |
| `chain/chain_resume_192322.log` | 文件 | 84B | 2026-10-16 19:23 |
| `chain/clean_corpus.py` | 文件 | 4.3K | 2026-10-14 02:21 |
| `chain/convert_st.py` | 文件 | 2.3K | 2026-10-13 00:49 |
| `chain/corpus_R4` | 文件 | 36B | 2026-10-12 23:52 |
| `chain/corpus_R5` | 文件 | 36B | 2026-10-14 04:51 |
| `chain/corpus_R6` | 文件 | 36B | 2026-10-15 00:57 |
| `chain/corpus_r4` | 目录 | - | 2026-10-12 23:52 |
| `chain/corpus_r5` | 目录 | - | 2026-10-14 04:51 |
| `chain/corpus_r6` | 目录 | - | 2026-10-15 00:57 |
| `chain/corpus_reserve` | 目录 | - | 2026-10-15 06:43 |
| `chain/corpus_reserve_refined` | 目录 | - | 2026-10-15 06:40 |
| `chain/done_steps.txt` | 文件 | 5B | 2026-10-16 19:23 |
| `chain/expand_300m_node1.py` | 文件 | 6.1K | 2026-10-16 01:59 |
| `chain/expand_run.log` | 文件 | 386B | 2026-10-16 01:59 |
| `chain/heartbeat` | 文件 | 45B | 2026-10-19 03:06 |
| `chain/ladder_budget.txt` | 文件 | 7B | 2026-10-18 02:33 |
| … | | | |

## 三、怎么用

```bash
python node/rc.py idx            # 目录树(离线, 0ms)
python node/rc.py idx qwen       # 关键词搜(离线)
python node/rc.py ls models -l   # 在线看(1~3s)
python node/rc.py svc            # 服务面板
python node/rc.py sync           # 重新扫描刷新本索引
```