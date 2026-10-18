# kris.nemotron (vendor=nvidia, served=nvidia/nemotron-3-ultra-550b-a55b, 55.7s)

### ① 知识到底存在 FFN 哪些权重里？≤1B 模型能否做知识定位？

**机制/最佳理论：**  
**知识以「稀疏分布式表征」存储在 FFN 的「值投影矩阵 $W_{up}$（或 $W_{out}$）的行向量」中**，即每个神经元充当一条「键-值」规则：$W_{down}$ 的列向量近似为「键」（模式匹配），$W_{up}$ 的行向量近似为「值」（知识载荷）。这在 *Geva et al., "Transformer Feed-Forward Layers Are Key-Value Memories" (2021)* 及后续因果追踪工作中被反复验证。  

**≤1B 模型的定位结论：能做「层级/神经元级」功能定位，不能做「单条事实→精确权重子集」的外科手术式定位。**  
- **叠加假设**：小模型神经元高度多义，一条事实分散在数百个神经元的线性组合中（非 one-hot）。  
- **因果追踪/激活补丁**只能锁定「关键层段（通常在中后层 60%-80% 深度）」，再细粒度会因分布式表征失效。  
- **实操结论**：别试图「给定事实，输出权重索引集合」。可行的是「给定事实，输出关键层段 + Top-k 激活神经元索引」，误差率 >40%，不可用于精准写入/删除。

---

### ② 外置记忆层的「挂载点」选在哪里？（理论依据 + 接口形态）

**推翻拍脑袋前提**：既然无法精准定位权重，**挂载点不能选在「FFN 内部权重」**，而必须选在「残差流」——Transformer 的唯一全局总线。  

**最佳理论依据**：  
1. **残差流 = 读写总线**（*Elhage et al., "A Mathematical Framework for Transformer Circuits" 2021*）：所有层、注意力头、FFN 只通过残差流交换信息。  
2. **Memory-as-a-Layer 范式**（*Memorizing Transformers, kNN-LM, Transformer-XL*）：在固定深度插入「交叉注意力层」，Query 来自残差流，Key/Value 来自外置索引。  
3. **深窄架构（BitNet b1.58）特性**：层数多（如 48-60 层），中后层语义抽象度高、冗余低，**挂载点锁定在「第 65% 深度层的残差流输出 → 第 66% 层的注意力输入」之间**（单点插入，零侵入原权重）。

**接口形态（零拷贝、可热插拔）**：
```python
# 伪代码：外置记忆层作为 nn.Module 插入残差流
class ExternalMemoryLayer(nn.Module):
    def __init__(self, d_model, mem_index: "Annoy/HNSW