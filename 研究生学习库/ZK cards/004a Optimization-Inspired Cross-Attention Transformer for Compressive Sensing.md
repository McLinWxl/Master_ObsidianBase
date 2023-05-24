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
>



# 模型推导：



# 实验/仿真验证



# 结论