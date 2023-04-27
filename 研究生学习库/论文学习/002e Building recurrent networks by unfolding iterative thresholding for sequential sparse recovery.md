# 基本信息：

[Arxiv.](https://arxiv.org/abs/1611.07252)

[Code](https://github.com/stwisdom/sista-rnn)

[[2017_Wisdom et al_Building recurrent networks by unfolding iterative thresholding for sequential.pdf]]

来源： 2017 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)

作者：Scott Wisdom, Thomas Powers, James Pitton, Les Atlas

机构：University of Washington

年份：2017

分区：-

标签： #ISTA_based #深度展开 

# 关键词

稀疏恢复、时序数据、压缩感知、深度展开、循环神经网络

# 摘要：

从历史上看，稀疏方法和神经网络，尤其是现代深度学习方法，一直是相对不同的领域。稀疏方法通常用于信号增强、压缩和恢复，通常在无监督框架中，而神经网络通常依赖于监督训练集。在本文中，我们使用序列稀疏恢复的具体问题，它使用==一系列稀疏系数对随时间变化的观察序列==进行建模，以展示稀疏建模算法如何与监督深度学习相结合以改进稀疏恢复。具体来说，我们展示了==用于顺序稀疏恢复的迭代软阈值算法 (ISTA) 对应于特定架构和参数约束下的循环神经网络 (RNN)==。然后，我们展示了使用监督数据通过反向传播训练此 RNN 以完成图像的列压缩感知任务的好处。这种训练对应于原始迭代阈值算法及其参数的适应。因此，我们通过示例展示了稀疏建模可以提供丰富的原则性和结构化深度网络架构资源，可以对其进行训练以提高特定任务的性能。

# 引言:

许多信号处理应用程序需要==从嘈杂和可能压缩的观察中恢复动态变化的信号==。顺序稀疏恢复处理一次短间隔的观察，并使用一系列关于静态字典的稀疏系数对信号的动态进行建模。通过利用这些系数的稀疏性和连续信号间隔之间的相关性，可以从嘈杂和压缩的观察中恢复和去噪信号。在测试时，这些无监督方法依赖于解决优化问题，通常使用迭代算法。

# 模型推导：

## 数学模型

>[!稀疏重构模型]
>$$
>\min\limits_{\mathbf{h}}\quad\dfrac{1}{2}\|\mathbf{x}-\mathbf{ADh}\|_2^2+\lambda\|\mathbf{h}\|_1
>$$
>其中，观测信号 $\mathbf{x}=\mathbf{As}+\epsilon$ 是对信号 $\mathbf{s=Dh}$ 的有噪观测，观测矩阵为：$\mathbf{A}\in\mathbb{F}^M\times N$ ； 字典矩阵为： $\mathbf{D}\in\mathbb{R}^{N\times N}$

![[SISTA Alg1.png]]

假设我们有一系列的观测信号 $\mathbf{x}_t$ ，并且这些信号时序相关：
$$
\mathbf{x}_t,\quad t=1,...,T
$$

为了模拟信号的时序相关，假设时间间隔信号的稀疏向量是前后相关的，因此，信号 $\mathbf{s}_t$ 可由 $\mathbf{s}_{t-1}$ 线性拟合而来：

$$
\mathbf{s}_t=\mathbf{Fs}_{t-1}+\mathbf{v}_t
$$

因此，时序信号的稀疏重构模型可以表示为：

>[!时序稀疏重构模型]
>$$
>\begin{array}{ll}\min\limits_{\mathbf{h}_{1:T}}\quad\sum_{t=1}^T\Big(\frac{1}{2}\big\|\mathbf{x}_t-\mathbf{A}\mathbf{D}\mathbf{h}_t\big\|_2^2+\lambda_1\big\|\mathbf{h}_t\big\|_1+\frac{\lambda_2}{2}\big\|{\mathbf{D}}\mathbf{h}_t-{\mathbf{F}\mathbf{D}}{\mathbf{h}}_{t-1}\big\|^2_2\Big).\\ \text{with}\quad \lambda_1=2\sigma^2\nu_1\quad \text{and}\quad \lambda_2=2\sigma^2{\nu_2}.\end{array}
>$$

![[SISTA Alg2.png]]

## 网络构建

### RNN

>[!单层]
>$$
>\mathbf{h}_t=\sigma_{\mathbf{b}}\left(\mathbf{W}\mathbf{h}_{t-1}+\mathbf{V}\mathbf{x}_t\right)
>$$
>$$ \hat{\mathbf{y}}_t=\mathbf{U}\mathbf{h}_t+\mathbf{c}
>$$

>[!多层级联]
>$$
>\mathbf{h}_{t}^{(k)}=\begin{cases}\sigma_{\mathbf{b}}\left(\mathbf{W}^{(1)}\mathbf{h}_{t-1}^{(1)}+\mathbf{V}\mathbf{x}_{t}\right),&k=1,\\ \sigma_\mathbf{b}\left(\mathbf{W}^{\left(k\right)}\mathbf{h}^{(k)}_{t-1}+\mathbf{S}^{(k)}\mathbf{h}_t^{(k-1)}\right),&k=2..K,\end{cases}
>$$
>$$ \hat{\mathbf{y}}_t=\mathbf{U}\mathbf{h}_t^{(K)}+\mathbf{c}
>$$

多层 RNN 网络中训练的参数有：
$$
\boldsymbol{\theta}_{\mathrm{RNN}}=\{\hat{\mathbf{h}}_0,\mathbf{b}^{(1:K)},\mathbf{W}^{\left(1:K\right)},\mathbf{V}^{\left(1:K\right)},\mathbf{S}^{\left(1;K\right)},\boldsymbol{U},\mathbf{c}\}
$$

![[SISTA Fig1.png]]

本文展开的 RNN 网络与传统 RNN 网络有以下区别
	1. 输入信号 $\mathbf{x}_t$ 与该组的每个隐藏节点都相连；
	2. 之前的状态输入由 $\mathbf{\hat{h}}_{t-1}=\mathbf{h}_{t-1}^{(k)}$ 修改为 $\mathbf{\hat{h}}_{t-1}=\mathbf{h}_{t-1}^{(K)}$  ；
	3. 令 $\mathbf{P}=\mathbf{D}^T\mathbf{F}\mathbf{D}$ 网络中的参数可以表示为：
		1. $\mathbf{V}^{(k)}=\dfrac{1}{\alpha}\mathbf{D}^T\mathbf{A}^T,\forall k$
		2. $\mathbf{S}^{(k)}=\mathbf{I}-\dfrac{1}{\alpha}\mathbf{D}^T(\mathbf{A}^T\mathbf{A}+\lambda_2\mathbf{I})\mathbf{D},k>1$
		3. $\mathbf{W}^{(1)}=\dfrac{\alpha+\lambda_2}{\alpha}\mathbf{P}-\dfrac{1}{\alpha}\mathbf{D}^T(\mathbf{A}^T\mathbf{A}+\lambda_2\mathbf{I})\mathbf{DP}$
		4. $\mathbf{W}^{(k)}=\dfrac{\lambda_2}{\alpha}\mathbf{P},k>1$
		5. $\mathbf{U=D}$
		6. $\mathbf{c=0}$

网络中训练的参数有：
$$
\boldsymbol{\theta}_{\mathrm{S1STA}}=\{\hat{\mathbf{h}}_0,\mathbf{A}^{(1:K)},\mathbf{D}^{\left(1:K\right)},\mathbf{F}^{\left(\mathrm{1:}K\right)},\alpha^{\left({1:K}\right)},\lambda_1^{\left({1;}K\right)},\lambda_2^{\left({1:}K\right)}\}
$$

损失函数定义为：本文定义 $f$  为 MSE

$$
\begin{array}{ll}\min\limits_{\boldsymbol{\theta}}&\sum\limits_{i=1}^{I}f(\hat{\mathbf{h}}_{1:T;i},\mathbf{y}_{1:T;i})\\ \text{subject to}&\hat{\mathbf{h}_{1:T;i}}=g\boldsymbol{\theta}(\mathbf{x}_{1:T;i}),i=1,\ldots,I\end{array}
$$

# 结论

在本文中，我们展示了如何将用于==时序稀疏恢复==的顺序迭代软阈值算法 (SISTA) 视为具有软阈值非线性和特定架构的多层循环神经网络 (RNN)。使用反向传播训练生成的 SISTA-RNN 对应于训练原始 SISTA 算法的泛化，该算法是结构化深度网络，执行模型参数的自动调整。