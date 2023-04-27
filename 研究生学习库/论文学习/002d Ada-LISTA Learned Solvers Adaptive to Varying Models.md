# 基本信息：

[链接：](https://arxiv.org/abs/2001.08456)

[[Ada-LISTA.pdf]]

作者：Aviad Aberdam，Alona Golts，Michael Elad

单位：University of Haifa

期刊：IEEE Transactions on Pattern Analysis and Machine Intelligence

年份：2021

中科院分区/影响因子：一区 / 24.314

标签： #ISTA_based 

# 摘要：

基于迭代求解器展开的神经网络，例如 LISTA（学习迭代软阈值算法），由于其加速性能而被广泛使用。然而，与非学习求解器相反，==这些网络是在某个字典上训练的，因此它们不适用于不同的模型场景==。这项工作引入了一种自适应学习求解器，称为 Ada-LISTA，它==接收成对的信号及其相应的字典作为输入，并学习一种通用架构==来为它们提供服务。我们证明该方案可以保证解决各种模型线性速率的稀疏编码，包括字典扰动和排列。我们还提供了广泛的数值研究，展示了它的实际适应能力。最后，我们将 Ada-LISTA 部署到自然图像修复中，其中补丁掩码在空间上变化，因此需要这样的适应。

# 引言:

传统 ISTA 迭代方法：[[002a ISTA 算法推导]]

$$\mathbf{x}_{k+1}=\mathcal{S}_{\frac{\lambda}{L}}\left(\mathbf{x}_k+\frac{1}{L} \mathbf{D}^T\left(\mathbf{y}-\mathbf{D} \mathbf{x}_k\right)\right), k=0,1, \ldots$$

$$\underset{\Theta}{\operatorname{minimize}} \sum_{i=1}^N\left\|\mathcal{F}_K\left(\mathbf{y}_i ; \Theta\right)-\mathbf{x}_{\mathbf{i}}\right\|_2^2$$

LISTA 网络展开方法：[[002b Algorithm Unrolling： Interpretable, Efficient Deep Learning for Signal and Image Processing#ISTA：Iterative Shrinkage and Thresholding Algorithm]]

$$\mathbf{x}_{k+1}=\mathcal{S}_\theta\left(\mathbf{W}_1 \mathbf{y}+\mathbf{W}_2 \mathbf{x}_k\right), \quad k=0,1, \ldots, K-1,$$
>$\boldsymbol{W}_1=\frac{1}{L}\boldsymbol{D}^H$， $\boldsymbol{W_2}=\boldsymbol{I}-\frac{1}{L}\boldsymbol{D}^H\boldsymbol{D}$

$$
\mathbf{x}_{k+1}=\mathcal{S}_\theta ((\boldsymbol{I}-\frac{1}{L}\boldsymbol{D}^H\boldsymbol{D}) \mathbf{x}_k+\frac{1}{L}\boldsymbol{D}^H \mathbf{y}), \quad k=0,1, \ldots, K-1,
$$

提出算法：Ada-LISTA：

将 LISTA 的适用性扩展到模型扰动和不同信号分布的场景
	1. 训练基于成对的信号及其对应的字典
	2. 应对字典扰动的鲁棒性：置换列、加性高斯噪声和完全更新的随机字典
	3. 证明 Ada-LISTA 在常量字典下实现了线性收敛率

# 模型推导：

将 ISTA 算法改写为：
$$\begin{aligned}
& \mathbf{x}_{k+1}=\mathcal{S}_{\theta_{k+1}}\left(\left(\mathbf{I}-\gamma_{k+1} \mathbf{D}^T \mathbf{W}_1^T \mathbf{W}_1 \mathbf{D}\right) \mathbf{x}_k+\gamma_{k+1} \mathbf{D}^T \mathbf{W}_2^T \mathbf{y}\right)
\end{aligned}$$

$$\underset{\Theta}{\operatorname{minimize}} \sum_{i=1}^N\left\|\mathcal{F}_K\left(\mathbf{y}_i, \mathbf{D}_i ; \Theta\right)-\mathbf{x}_{\mathbf{i}}\right\|_2^2$$

![[Ada-LISTA.png]]

# 实验/仿真验证

## 随机排列字典

字典的所有列随机排列

## 噪声扰动字典

实验了字典矩阵信噪比在 25dB，20dB，15dB 下的性能

## 随机字典

创建一个不同的高斯归一化字典 $D_i$，以及一个相应的具有递增基数的表示向量：$s = 4, 8, 12$

# 结论

本文引入了 LISTA 的扩展形式 —— Ada-LISTA。
它==接收信号及其字典作为输入==，并学习可以应对不同模型的通用架构。
这种修改在==处理不断变化的字典==方面产生了极大的灵活性，与非学习求解器（例如 ISTA 和 FISTA，它们对整个信号分布不可知）平衡了竞争环境，同时享受学习求解器的==加速和收敛优势==。
通过全面的理论研究以及广泛的合成和现实世界实验证实了我们方法的有效性。