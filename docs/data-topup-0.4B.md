# 0.4B 数据补齐总账（2026-10-16 启动，持续更新）

> 配套 `docs/data-plan-0.4B.md`。本文件记录补齐执行的实数与通道经验。
> token 口径：加词表后 1GB 文本 ≈ 3.3 亿汉字 ≈ 4.6 亿 token。

## 一、总盘（目标 35 亿 token）

| 板块 | 占比 | 目标 token | 现有实数 | 来源 | 状态 |
|---|---|---|---|---|---|
| 通用高质量中文 | 45% | 15.75 亿 | **≈17 亿** | `corpus_04b/ufw_zh.jsonl` 3.7GB / 972,767 行 / 15 亿汉字（Ultra-FineWeb-zh part 001-002） | ✅ 超配 |
| 知识与教育 | 25% | 8.75 亿 | ≈6.9 亿 | `chain/zh_wiki_all.jsonl` 1.5GB 全量维基 + R5/R6 知识精选 | ✅ 79%（可选补 Fineweb-Edu） |
| 创意·文章 | 8% | 2.8 亿 | ≈0.4 亿 | poetry 387,259 首/1430 万字 + gutenberg 46 本/44MB | ⬜ **缺口 ~2.4 亿** |
| 创意·文案 | 7% | 2.45 亿 | ≈1.4 亿 | weixin raw 300MB（76,839 条公众号）+ L3 multi_style（本地下载中） | 🔄 L3 到位后补齐 |
| 逻辑/推理 | 5% | 1.75 亿 | ≈0 | L3 qa（本地下载中） | 🔄 L3 到位后补齐 |
| 咱们的资料 | 2% | 0.7 亿 | ≈0.05 亿 | owner_corpus 5,146 条 + 小说设定 | ✅（upsample 4× 补足） |
| 缓冲/去重损耗 | 8% | 2.8 亿 | — | — | — |

**不含 L3 合计 ≈ 26 亿（74%）**；L3 两片精选配额（约 2.7 亿）到位后 ≈ 82%。
**剩余缺口集中在创意文章档 ~2.2 亿 token**（下一批：中文公版书扩充 / 民间文学 / 成语故事）。

## 二、NAS 落位（`/mnt/workspace/node1/corpus_04b/`）

| 文件 | 大小 | 内容 |
|---|---|---|
| `ufw_zh.jsonl` | 3.7GB | 97.3 万行通用网页（已过 200 字下限，未精选） |
| `poetry_zh.jsonl` | 55.8MB | 387,259 首古诗词（train.csv 一行一首，BOM 已处理） |
| `l3_ms/`、`l3_qa/` | 下载中 | L3 蒸馏（本地代理 245KB/s，预计今夜到位） |
| `poetry/` | 65MB | chinese-poetry-collection.git clone（原始 JSON） |

## 三、通道经验（重要，以后少踩坑）

1. **ModelScope CDN 是 node1 的高速通道**：`https://modelscope.cn/api/v1/datasets/{ns}/{name}/repo?Revision=master&FilePath={path}` → 302 国内 CDN，实测 **~45MB/s**。非 xet 数据一律先找 MS 镜像。
2. **openbmb/Ultra-FineWeb 与 Ultra-FineWeb-L3 用 xet 存储**：hf-mirror 对它们 308 跳官方 xet CDN（us.aws.cdn.hf.co），**node1 30 秒 0 字节**；本地直连 HF 被墙；本地代理仅 245KB/s；datasets-server.huggingface.co node1 不通。⇒ xet 数据集只能本地慢下中转。
3. **ECS 派活必须自带回传**：CMD 末尾 `... | curl -s -X POST 'http://47.99.131.12:8081/upload?name=X.txt&key=k0u4mail' --data-binary @-`，否则输出全丢（踩了 4 次）。
4. **ECS /file 同名文件锁**：每次查询必须轮换文件名（`_001` → `_002`），否则读到缓存旧回传（踩了 2 次）。
5. **本地长任务禁 `run_in_background`**（被系统回收，实测 2-11 分钟即杀）；用 `(nohup curl ... > log 2>&1 </dev/null &)` 子 shell（本地铁律，本次又踩一次）。
6. 自包含脚本派活模式：`echo '<base64>' | base64 -d > f.sh && (nohup bash f.sh > log 2>&1 &)`。

## 四、待办

| 项 | 状态 |
|---|---|
| L3 qa 片本地下载（475MB/1074MB @ 01:10，~440KB/s） | 🔄 nohup 进行中，约 01:35 完 |
| L3 ms 片本地下载 | ⬜ qa 完后串行 |
| L3 到位后 → ECS 上传 → node1 转 jsonl + 精选配额 | ⬜ |
| 创意文章缺口 2.2 亿：中文公版书扩充（gutenberg 全量中文 / 民间文学 / 成语故事） | ⬜ 下一批 |
| 全部到位后过 `refine_reserve.py`（硬过滤+块级去重+密度打分+多样性配额） | ⬜ |
| 0.4B 加词表后统一分词（cache .npy 与新语料一次做） | ⬜ 等扩容 |

## 五、主链状态（不受本任务影响）

R6：done_steps 6700/6800（chunk 制，~11s/步），预计 01:10 前后封顶 → 自动 Net2Net 扩 0.3B（350M）→ R7-R10。
