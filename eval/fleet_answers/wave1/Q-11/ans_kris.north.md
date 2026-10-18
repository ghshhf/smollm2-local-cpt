# kris.north (vendor=cohere, served=cohere/north-mini-code:free, 14.6s)

**⚠️ 核心修正 – “直接 Autotelic” 并不安全**
假设模型能够*自主*发现和获取任何资源（原始、自有或网络）是不可靠的。首先引入一个**固定源白名单**是有充分理由的——它提供了*初始*的防污染保障。白名单可以是JSON文件列表，包含已知安全的URL模式、正则表达式或本机数据集标签。白名单必须是