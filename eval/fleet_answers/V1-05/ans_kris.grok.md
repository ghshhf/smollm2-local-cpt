# kris.grok (vendor=xai, served=grok-4.7, 58.1s)

① 最佳理论是损失轨迹曲率（loss trajectory curvature）。机制：泛化阶段模型在单遍下学习通用模式，损失函数呈现平滑二次下降，曲率（二阶导）接近0且缓慢变化；死记阶段模型针对特定样本急剧拟合，损失会出现正尖峰（曲率 spikes），导致整体轨迹曲率显著增大。优于梯度噪声谱，因为曲率可通过损失序列的滑动二阶差分实时近似，无需高维谱估计。优于参数漂移率（漂移率高时可能因优化震荡而非死记），优于单样本损失方差（方差仅反映离散度，未区分尖峰 vs 均匀难样本）。
② GSNR 在单遍下退化至近乎无信息量。理由：Feldman 长尾理论依赖重复观测（多轮数据洗牌）来稳定谱密度估计；单遍下数据顺序固定、每样本仅见一次，导致梯度噪声非平稳，谱估计方差爆炸，信息量趋近于0。最小可行替代判据是参数漂移率（parameter drift rate）。可执行方案伪代码：
```
params_old = copy(params)
for step in total_steps:
 compute_grad(batch)
 update(params, optimizer)
 if step % checkpoint_interval == 0:
 drift = mean(abs(params - params_old)) / mean(abs(params_old))
 params_old = copy(params)
 # 结合 loss_decreasing flag 判断
 if drift < epsilon and loss_trend_stable_down:  # 记忆化信号
 trigger_second_pass()
 else:
 # 泛化信号，继续单遍
```
CPU 低成本：仅需额外 2× 参数内存（fp16 下 0.4B 模型 ≈1.6 GB），每 checkpoint_interval（建议 1000 步）做一次快照，无需额外梯度或 Hessian 计算。
③ 单样本损失方差高 = 难样本（hard samples），而非必然记忆化。判别式：若损失分布存在“低值模式”（多个样本 loss≈0 或极低），则记忆化；若损失分布为“中值均匀扩散”（所有样本 loss 处于同一量级 0.1–1.0 且方差高但无尖峰），则为难样本。记忆化特征还伴随“曲率 spikes”与“参数漂移局部化”（少数参数剧变）；难样本特征为“全局损失缓慢下降 + 方差均匀”。可证伪判据：若高方差样本在全训练后 holdout loss 仍保持高，则确为难样本；若这些样本在后续二刷时 loss 急剧下降则确为记忆化样本。
④ node1 CPU 最小可跑实验设计：使用 owner_corpus 子集 5 万条纯净样本（已去重、无模型生成数据）、0.1B 模型 fp16 嵌入（b_eff≈5.5）、batch size=4、单遍 10 轮（总步≈1250k）、CPU only（无 CUDA）。追踪：每 200 步记录全局 loss、per-batch loss 方差、轨迹曲率（滑动窗口二阶差分）、参数漂移率。目标判据验证：若方差>0.5 且伴随曲率 spikes>阈值，则标记为记忆化风险；若方差>0.5 但无 spikes 且 loss 持续下降，则标记为难样本。总内存峰值≈3 GB（参数+激活+追踪 buffer），可稳定跑完全程并输出 confusion matrix 与 holdout 相关性。