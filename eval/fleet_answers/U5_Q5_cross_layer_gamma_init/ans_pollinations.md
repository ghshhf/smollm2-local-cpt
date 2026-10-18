# pollinations (vendor=openai-oss, served=gpt-oss-20b, 23.3s)

### ① 机制 / 最佳理论  
| 关键点 | 说明 |
|-------|------|
| **Net2Net 经典方案** | `Net2WiderNet` 通过复制已有神经元的权重与 bias 并用单位矩阵 \(I\) 与均匀映射 \(W\) 重新分配权重，从而保证前向输出不变；`Net2DeeperNet` 在已有层后插入 “恒等映射” 包含 \(0\) 或 “加法残差” 以保持功能。 