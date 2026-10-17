# polaris-03b_20261018 — 北极星 0.3B 底座权重（GitHub 备份）

## 身份
- 北极星 0.3B（Net2Net 保函数扩容链 0.1B→0.2B→0.3B，346.9M 参数 / 36 层 / hidden 768）
- 2026-10-18 回传本地（MD5 与 node1 NAS 原件一致），私有仓备份，**不开源**
- sha256: `90f2f64bd01f64bd4690414d709f1291d98bf855a11958a4269dffcb33f30b70`（1,387,470,856 字节）

## 文件
- `model.safetensors.part00 ~ part13` — 权重分片（95MiB×13 + 88.2MiB，共 14 片）
- `reassemble.py` — 拼回器（幂等 + sha256 自校验）
- `config.json` / `generation_config.json` / `step.txt` / `tokenizer.json` / `tokenizer_config.json`
- `.gitattributes` — ModelScope 镜像（hhhlhuk/polaris-03b）元信息

## 拼回
```bash
python reassemble.py .
# OK 1387470856 sha256 verified
```

## 备注
- 权重以 `.partNN` 分片入库：`.gitignore` 挡 `*.safetensors` 原文件，分片后缀不匹配规则，与 0.2B 同模式
- 全局训练进度以 chain.log 为准
