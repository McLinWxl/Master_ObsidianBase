# 基本信息：

[[2023_Song et al_Optimization-Inspired Cross-Attention Transformer for Compressive Sensing.pdf]]

来源：  CVPR 2023

作者：Jiechong Song, Chong Mou, Shiqi Wang, Siwei Ma, Jian Zhang

机构：Peking University Shenzhen Graduate School, City University of Hong Kong, 

年份：2023

分区：-

标签： [[DUN]], [[CS]], [[Transformer]], [[ISTA-based]]

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

# 引言：

## 优化模型

>[!Model]
>$$\mathbf{y}=\mathbf{\Phi}\mathbf{x}$$
>$$\hat{\boldsymbol{x}}=\min_{\mathbf{x}}\frac{1}{2}\|\mathbf{\Phi}\mathbf{x}-\mathbf{y}\|_2^2+\lambda\mathcal{R}(\mathbf{x})$$

## 内容概要：

提出了一种有效的优化启发式交叉注意转换器（**O**ptimization-inspired **C**ross-attention **T**ransformer，OCT）模块作为迭代过程，并为图像压缩感知建立了一个基于 OCT 的轻量级展开框架（**OCT**-based **U**nfolding **F**ramework，OCTUF）
1. 参数量小
2. 重构性能强

![[Pasted image 20230525132733.png|500]]

 OCT 模块在特征空间中保持最大的信息流，它由一个双交叉注意力 (Dual Cross Attention，Dual-CA) 子模块和一个前馈网络 (Feed-Forward Network，FFN) 子模块组成，以形成每个迭代过程。Dual-CA 子模块包含一个惯性交叉注意 (Inertia-Supplied Cross Attention，ISCA) 块和一个投影交叉注意 (Projection-Guided Cross Attention，PGCA) 块。 ISCA 块计算相邻迭代信息的交叉注意力，并为优化算法添加惯性/记忆效应。PGCA 块使用梯度下降步骤和惯性项作为交叉注意 (CA) 块的输入来指导通道特征的精细融合。

# 相关工作：

## 深度展开网络（DUN）

>[!DUNs-Optimization]
>$$
>\begin{aligned}
&\operatorname*{min}_{\Theta}\sum_{j=1}{\mathcal{L}}(\mathbf{\hat{x}}_{j},\mathbf{x}_{j}), \\
&\mathbf{s.t.}\quad\hat{\mathbf{x}}_j=\arg\min\frac{1}{2}\|\mathbf{\Phi x}-\mathbf{y}_j\|_2^2+\lambda\mathcal{R}(\mathbf{x})
\end{aligned}$$

在大多数 DUN 中，==每次迭代的输入和输出本质上都是图像，这严重阻碍了信息传输，导致表示能力有限==。最近，一些方法提出了==将信息流结合到特征空间中的每个迭代过程中以增强信息传输==的想法。然而，现有的解决方案通常在处理通道信息方面缺乏灵活性，并且受到高模型复杂性的困扰。在本文中，我们提出了一个有效的解决方案。

# 提出方法

## 整体框架

近端梯度下降（PGD）算法 + ==惯性项==

>[!PGD+Inertia]
>$$
>\mathbf{s}^{(k)}= \mathbf{x}^{(k-1)}-\rho^{(k)}\mathbf{\Phi}^{\top}(\mathbf{\Phi}\mathbf{x}^{(k-1)}-\mathbf{y}) 
+\alpha^{(k)}(\mathbf{x}^{(k-1)}-\mathbf{x}^{(k-2)})
>$$
>$$
>\mathbf{x}^{(k)}=\arg\min\frac{1}{2}\left\|\mathbf{x}-\mathbf{s}^{(k)}\right\|_2^2+\lambda\mathcal{R}(\mathbf{x})
>$$

存在缺陷：缺乏适应性，并且由于图像级的级间传输而导致信息丢失
提出：Dual Cross Attention (Dual-CA)子模块，通过增加==多通道惯性力==，增强投影步骤中的信息交互，实现特征级传输。

>[!OCT-module]
>$$
>\mathbf{S}^{(k)}=\mathcal{H}_{\mathrm{Dual-CA}}(\mathbf{X}^{(k-1)},\mathbf{Z}^{(k-2)})
>$$
>$$
>\mathbf{X}^{(k)}=\mathcal{H}_{\mathrm{FFN}}(\mathbf{S}^{(k)})
>$$

$\mathbf{X}^{(0)}$ 由一个3×3的卷积 (Conv0(·)) 对 $\mathbf{x}^{0}$ 初始化

![[Pasted image 20230530142910.png]]

因此，提出的 OCTUF 可以巧妙地整合阶段间==特征级信息==，并实现与优化步骤的完美结合。

## Dual Cross Attention

![[Pasted image 20230530144003.png]]

为了使网络更紧凑并保持潜在的数学解释，将输入特征 $\mathbf{X}^{(k-1)}\in\mathbb{R}^{H\times W\times C}$ 分为两个部分：
	1. $\mathbf{r}^{(k-1)}\in\mathbb{R}^{H\times W\times1}$ （$\mathbf{X}^{(k-1)}$ 第一个通道）：输入 PGCA
	2. $\mathbf{Z}^{(k-1)}\in\mathbb{R}^{H\times W\times(C-1)}$  （$\mathbf{X}^{(k-1)}$ 后 C-1 通道）：输入 ISCA

### 交叉注意力

![[Pasted image 20230530145327.png]]

输入Q、K和V
1. 输入经过 $1\times 1$ 卷积提取输入特征
2. 再经过 $3\times 3$ 深度卷积对通道空间上下文编码
3. 由编码转置二维化后的K与编码二维化后的Q相乘，得到维度为 $\mathbb{R}^{(C-1)\times (C-1)}$ 的权重矩阵A
4. 编码二维化后的V作为基矩阵
5. 权重矩阵与基矩阵相乘再返回三维张量
6. 最后经过 $1\times 1$ 卷积提取特征后输出

### ISCA

![[Pasted image 20230530150731.png]]

 一般惯性项通常采用==直接减去相邻迭代输出的简单运算，在表4的消融中被证明是无效的。==

为了丰富惯性项中的信息交互，引入了多通道惯性项模块（ISCA），由交叉注意力模块充分考虑当前输入与前一次输入之间的关系。

### PGCA

![[Pasted image 20230530151205.png]]

为了自适应地结合梯度下降项和惯性项，引入投影交叉注意力模块（PGCA），由交叉注意力模块充分考虑梯度下降项与惯性项之间的关系。

GDB梯度下降模块公式为：
$$
\mathbf{\hat{r}}^{(k-1)}=\mathbf{r}^{(k-1)}-\rho^{(k)}\mathbf{\Phi}^{\top}(\mathbf{\Phi}\mathbf{r}^{(k-1)}-\mathbf{y}).
$$

注意：$\mathbf{\hat{r}}$ 与 $\hat{\bf{Z}}$ 的通道匹配时，在交叉注意力模块中，$\mathbf{\hat{r}}$ 经过 $1\times 1$ 卷积后将通道数扩大为 C-1

## 损失函数设计

重构图像与标签的 MSE 损失：

$$
\mathcal{L}(\mathbf{\Theta})=\frac{1}{NN_a}\sum_{j=1}^{N_a}\|\mathbf{x}_j-\mathbf{\hat{x}}_j\|_2^2
$$

# 实验/仿真验证

## 实验设置

训练数据集设置：BSD500 中的 400 张图片裁剪为 89600 个 $96\times96$ 像素大小的块，并进行图像增强

根据CS率 $\frac{M}{N}$ ，构造测量矩阵 $\bf{\Phi}$ 为大小为 $M\times1\times\sqrt{N}\times\sqrt{N}$ 的卷积核对原始图像进行采样

网络参数设置：
1. batch size=16
2. 特征通道数C=32
3. Adam优化器，学习率采用余弦退火策略（100 epochs后降低至5e-5）和warm-up（3 epochs）

测试数据集：Set11、Urban100，彩色图像在YCbCr空间处理，在Y通道上评价

评价指标：
1. 峰值信噪比（PSNR）
2. 结构相似性（SSIM）
3. 每秒浮点运算（FLOPs）

## 对比实验

OCTUF：迭代次数10，初始学习率5e-4
OCTUF+：迭代次数16，初始学习率2e-4

![[Pasted image 20230531125915.png]]

![[Pasted image 20230531130004.png]]

结论：提出方法OCTUF+的性能最好。

![[Pasted image 20230531125749.png]]

![[Pasted image 20230531130016.png]]

结论：提出方法OCTUF+能够恢复更清晰的边缘信息。

## 消融实验

### 整体消融

![[Pasted image 20230531130353.png]]

结论：提出的 OCTUF 具有稳定的训练过程和较大的学习率，同时，我们在训练时使用“warm-up”策略来提高其随时性能。

### 双注意力消融

![[Pasted image 20230531130511.png]]

“IF”表示通过 $\mathbf{s}^{(k)}= \mathbf{x}^{(k-1)}-\rho^{(k)}\mathbf{\Phi}^{\top}(\mathbf{\Phi}\mathbf{x}^{(k-1)}-\mathbf{y}) +\alpha^{(k)}(\mathbf{x}^{(k-1)}-\mathbf{x}^{(k-2)})$ 获得惯性项；

“FD”表示整个迭代过程是在特征域实现的

结论：ISCA 和 PGCA 块都更加关注细节内容和结构纹理

### 前馈网络消融

![[Pasted image 20230531131151.png]]

Baseline指没有LayerNorm（LN）
提出的：$2\times(LN+FFB)$

结论：提出方法实现了最好的性能

## 复杂度分析

![[Pasted image 20230531131423.png]]

结论：提出方法采样过程的参数与MADUN 和 FSOINet 相同，但使用更少的参数和更少的计算负担来产生更清晰的恢复图像。

## 噪声敏感性

![[Pasted image 20230531131526.png]]

结论：提出方法 OCTUF 对噪声具有很强的鲁棒性。

# 结论

>In this paper, we propose a novel optimization-inspired cross-attention Transformer (OCT) module as an iteration, leading to a lightweight OCT-based unfolding framework (OCTUF) for CS. Specifically, we present a Dual Cross Attention (Dual-CA) sub-module, which contains an inertiasupplied cross attention (ISCA) block and a projectionguided cross attention (PGCA) block as the projection step in iterative optimization. ISCA block precisely helps to achieve a good convergence by introducing a feature-level inertial term. PGCA block utilizes a cross attention mechanism to fuse the gradient descent term and the inertial term while ensuring the maximum information flow. Extensive experiments show that our OCTUF achieves superior performance compared to state-of-the-art methods with lower complexity. In the future, we will extend our OCTUF to other image inverse problems and video applications.

本文提出了一种新的受优化启发的交叉注意Transformer（OCT）模块作为迭代，从而为 CS 提供了一个基于 OCT 的轻量级展开框架（OCTUF）。具体来说，提出了一个双交叉注意 (Dual-CA) 子模块，其中包含一个惯性交叉注意 (ISCA) 块和一个投影交叉注意 (PGCA) 块作为迭代优化中的投影步骤。 ISCA 块通过引入==特征级惯性项==恰好有助于实现良好的收敛。 PGCA block==利用交叉注意力机制融合梯度下降项和惯性项==，同时保证最大的信息流。大量实验表明，与具有较低复杂性的最先进方法相比，我们的 OCTUF 实现了卓越的性能。