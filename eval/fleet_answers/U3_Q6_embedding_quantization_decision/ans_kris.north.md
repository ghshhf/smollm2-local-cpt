# kris.north (vendor=cohere, served=cohere/north-mini-code:free, 55.3s)

**① 机制/最佳理论**

| 现象 | 理论解释 | 设计启示 |
|------------|-------------------|------------------|
| **Embedding 占参数比例随模型规模单调上升** | 嵌入矩阵是全连接 `V × D` 层，参数量为 `O(V·D)`。核心（Transformer）模块的参数量为 `O(L·D²)`。随着 `L` 和 `D` 增加