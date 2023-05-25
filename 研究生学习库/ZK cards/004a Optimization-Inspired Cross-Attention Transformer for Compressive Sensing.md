# 基本信息：

[[2023_Song et al_Optimization-Inspired Cross-Attention Transformer for Compressive Sensing.pdf]]

来源：  CVPR 2023

作者：Jiechong Song, Chong Mou, Shiqi Wang, Siwei Ma, Jian Zhang

机构：Peking University Shenzhen Graduate School, City University of Hong Kong, 

年份：2023

分区：-

标签： [[DUN]], [[CS]]

代码：[Codes](https://github.com/songjiechong/OCTUF)

# 摘要：

>By integrating certain optimization solvers with deep neural networks, deep unfolding network (DUN) with good interpretability and high performance has attracted growing attention in compressive sensing (CS). However, existing DUNs often improve the visual quality at the price of a large number of parameters and have the problem of feature information loss during iteration. In this paper, we propose an Optimization-inspired Cross-attention Transformer (OCT) module as an iterative process, leading to a lightweight OCT-based Unfolding Framework (OCTUF) for image CS. Specifically, we design a novel Dual Cross Attention (Dual-CA) sub-module, which consists of an InertiaSupplied Cross Attention (ISCA) block and a ProjectionGuided Cross Attention (PGCA) block. ISCA block introduces multi-channel inertia forces and increases the memory effect by a cross attention mechanism between adjacent iterations. And, PGCA block achieves an enhanced information interaction, which introduces the inertia force into the gradient descent step through a cross attention block. Extensive CS experiments manifest that our OCTUF achieves superior performance compared to state-of-the-art methods while training lower complexity.

背景：通过将优化求解器与深度神经网络相结合，具有良好可解释性和高性能的深度展开网络（DUN）在压缩传感（CS）中引起了越来越多的关注。

存在的问题：
1. 往往以大量参数为代价来提高视觉质量
2. 在迭代过程中存在特征信息丢失的问题

提出：交叉注意力Transformer（==Cross-attention Transformer，OCT==）
1. 设计双交叉注意力（Dual Cross Attention，Dual-CA）子模块
	1. InertiaSupplied Cross Attention （ISCA）块
	2. ProjectionGuided Cross Attention （PGCA）块

# 引言:

## 信号模型

>[!Model]
>$$\mathbf{y}=\mathbf{\Phi}\mathbf{x}$$
>$$\hat{\boldsymbol{x}}=\min_{\mathbf{x}}\frac{1}{2}\|\mathbf{\Phi}\mathbf{x}-\mathbf{y}\|_2^2+\lambda\mathcal{R}(\mathbf{x})$$

## 提出方法：

提出了一种有效的优化启发式交叉注意转换器（**O**ptimization-inspired **C**ross-attention **T**ransformer，OCT）模块作为迭代过程，并为图像压缩感知建立了一个基于 OCT 的轻量级展开框架（**OCT**-based **U**nfolding **F**ramework，OCTUF）
1. 参数量小
2. 重构性能强

![[Pasted image 20230525132733.png|500]]

 OCT 模块在特征空间中保持最大的信息流，它由一个双交叉注意力 (Dual Cross Attention，Dual-CA) 子模块和一个前馈网络 (Feed-Forward Network，FFN) 子模块组成，以形成每个迭代过程。Dual-CA 子模块包含一个惯性交叉注意 (Inertia-Supplied Cross Attention，ISCA) 块和一个投影交叉注意 (Projection-Guided Cross Attention，PGCA) 块。 ISCA 块计算相邻迭代信息的交叉注意力，并为优化算法添加惯性/记忆效应。PGCA 块使用梯度下降步骤和惯性项作为交叉注意 (CA) 块的输入来指导通道特征的精细融合。

# 正文：

## 深度展开网络（DUN）

>[!DUNs-Optimization]
>$$
>\begin{aligned}
&\operatorname*{min}_{\Theta}\sum_{j=1}{\mathcal{L}}(\mathbf{\hat{x}}_{j},\mathbf{x}_{j}), \\
&\mathbf{s.t.}\quad\hat{\mathbf{x}}_j=\arg\min\frac{1}{2}\|\mathbf{\Phi x}-\mathbf{y}_j\|_2^2+\lambda\mathcal{R}(\mathbf{x})
\end{aligned}$$

在大多数 DUN 中，==每次迭代的输入和输出本质上都是图像，这严重阻碍了信息传输，导致表示能力有限==。最近，一些方法提出了==将信息流结合到特征空间中的每个迭代过程中以增强信息传输==的想法。然而，现有的解决方案通常在处理通道信息方面缺乏灵活性，并且受到高模型复杂性的困扰。在本文中，我们提出了一个有效的解决方案。

## 视觉Transformer

# 实验/仿真验证




# 结论