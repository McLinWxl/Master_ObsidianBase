# 基本信息：

[链接：](https://arxiv.org/abs/2103.12828)

[[Learning to Optimize： A Primer and A Benchmark.pdf]]

作者：Tianlong Chen，Xiaohan Chen， Wuyang Chen，Zhangyang Wang

单位：The University of Texas

期刊：Journal of Machine Learning Research

年份：2022

中科院分区 / 影响因子：三区 / 5.177

代码：[Open-L2O](https://github.com/VITA-Group/Open-L2O)

文章类型：[[综述]]

# 摘要：

学习优化 (L2O) 是一种利用机器学习开发优化方法的新兴方法，旨在减少手工工程的费力迭代。它根据其在一组训练问题上的表现自动设计优化方法。这个数据驱动的过程生成的方法可以有效地解决与训练中类似的问题。与之形成鲜明对比的是，典型和传统的优化方法设计是理论驱动的，因此它们在理论指定的问题类别上获得性能保证。这种差异使得 L2O 适用于针对特定数据分布重复解决特定优化问题，而它通常无法解决分布外问题。 L2O 的实用性取决于目标优化的类型、所选择的学习方法架构以及训练过程。这种新范式激发了一群研究人员探索 L2O 并报告他们的发现。

这篇文章有望成为==L2O持续优化的第一个全面调查和基准==。我们建立分类法，对现有工作和研究方向进行分类，提出见解，并确定开放的挑战。我们在几个有代表性的优化问题上对许多现有的 L2O 方法进行了基准测试。

# 引言:

## 背景和动机

**经典优化方法**以理论上合理的方式建立在基本方法的组件之上，例如梯度下降、共轭梯度、牛顿法、单纯形基更新（Simplex basis update）和随机抽样。

**L2O** 是另一种范式，它==通过训练开发优化方法==，即从其在样本问题上的表现中学习。该方法可能==缺乏坚实的理论基础，但在训练过程中提高了其性能==。培训过程经常离线进行，非常耗时。但是，该方法的在线应用（旨在）节省时间。当涉及到难以获得目标解的问题时，例如非凸优化和反问题应用，经过良好训练的 L2O 方法的解可以比经典方法具有更好的质量。

![[L2O fig1.png]]

在许多优化程序中，任务是对特定的数据分布重复执行某种类型的优化。

传统的优化器可以针对任务分配进行调整，但底层方法是为理论指定的优化问题类别设计的，而==不是任务的数据分布==。

在 L2O 中，训练优化器的过程根据==公式和任务分布==来塑造优化器。当分布集中时，学习到的优化器可能会“过度拟合”任务，并可能发现经典优化器不采用的“捷径”。

L2O 旨在生成具有以下优势的优化器：
	1. L2O 学习的优化器有望比经典方法==更快地==完成来自相同任务分布的一组优化。
	2. 在计算预算相似的情况下，学习到的优化器还可以为困难任务返回比经典方法==质量更高==的解决方案。



# 模型推导：

## 优化公式：



> [!L2O optimizer]
> $$
>x_{t+1}=x_t-m(\mathbf{z}_t;\phi)
>$$
>寻找最佳的优化规则可以用以下公式表示（在 $m$ 的参数空间上搜索一个最优的 $\phi$ ，使得损失函数最小（这里定义为目标函数 $f(\mathbf{x}_t)$ 在时间跨度内的加权和））：
>$$
>\phi = \min _\phi \mathbb{E}_{f \in \mathcal{T}}\left[\sum_{t=1}^T w_t f\left(\mathbf{x}_t\right)\right], \quad \text { with } \quad \mathbf{x}_{t+1}=\mathbf{x}_t-m\left(\mathbf{z}_t, \phi\right), t=1, \ldots, T-1
>$$
>其中，$m(\mathbf{z}_t;\phi):\mathcal{Z}\rightarrow \mathbb{R}^d$ 是一个映射函数，在给定输入特征 $\mathbf{z}_t\in\mathcal{Z}$ 时生成对 $x$ 的更新

>注意到：
>当 $m(\mathbf{z}_t;\phi) = \alpha \nabla f(x_t)$ 时，优化方法为一阶梯度下降法。

更新规则 $m$ 通常由==多层神经网络或递归神经网络==参数化。理论上，神经网络是通用逼近器，因此 L2O 可以发现全新的最优更新规则，而无需参考经典优化器采用的任何现有更新规则。因此，我们将其称为==无模型 L2O（model-free L2O）==。在这里，术语“模型”代表优化问题中的先验知识，专家通常利用这些知识来设计经典优化器。

==无模型 L2O的缺点包括：缺乏收敛保证和需要大量的训练样本==。在经典操作对良好性能至关重要的任务上，例如投影、归一化和衰减步长，无模型 L2O 要么无法实现良好性能，要么需要大量训练问题才能从头开始发现经典操作。为了避免这些缺点，我们考虑将==现有方法合并为学习的基础或起点==，从而将搜索减少到更少的参数和更小的算法空间。我们将这种替代方法称为==基于模型的 L2O（model-based L2O）==。

## Model-free L2O

### LSTM optimizer

通常，无模型 L2O 方法旨在学习优化的参数化更新规则，而不采用任何分析更新的形式（迭代更新除外）。最近的主流工作利用循环神经网络 (RNN)，其中大部分使用长短期记忆 (LSTM) 架构。展开 LSTM 以执行迭代更新并训练以找到较短的优化轨迹。一组参数在所有展开的步骤中共享。在每一步，LSTM 输入优化器的局部状态（例如零阶和一阶信息）并返回下一个迭代。

>[!LSTM optimizer]
>$$
>\mathcal{L}(\phi)=\mathbb{E}_{\left(\theta_0, f\right) \in \mathcal{T}}\left[\sum_{t=1}^{\mathrm{T}} w_t f\left(\theta_t\right)\right] \quad \text { where } \quad\left[\begin{array}{l}
g_t \\
h_{t+1}
\end{array}\right]=m\left(\nabla_t, h_{t+1}, \phi\right),\quad \theta_{t+1}=\theta_t+g_t
>$$

（具体内容由于没学习过 LSTM 网络所以没细看）

### Black-box Optimization

将 LSTM L2O 框架扩展到无导数或黑盒框架

### Particle swarm optimization

一组 LSTM 共同学习了一组样本（或一群粒子）的迭代更新公式。该模型可以将基于点的输入特征（例如梯度动量）和基于群体的特征（例如粒子的速度和群体算法的吸引力）作为输入。

### Minimax optimization

无模型 L2O 的一个更具挑战性的测试平台是解决连续极小极大优化问题。试图将 L2O 插入到极小极大优化的特定应用中，称为==对抗训练==。

$$
\min _\theta \mathbb{E}_{(\boldsymbol{x}, \boldsymbol{y}) \sim D}\left\{\max _{\boldsymbol{x}^{\prime} \in \mathbb{B}(\boldsymbol{x}, \epsilon)} \mathcal{L}\left(f\left(\boldsymbol{x}^{\prime}\right), \boldsymbol{y}\right)\right\}
$$

### Game theory

基于 RL 的无模型 L2O 最近引起了博弈论领域的兴趣。提出训练多代理系统 (MAS) 以实现对称的纯纳什均衡。

### Few-shot learning

LSTM L2O 框架 [13] 的另一个应用探索了无模型 L2O 在小数据体系中的应用。

## Model-based L2O

基于模型的 L2O 可以被视为一种“半参数化”选项，它利用了基于模型的结构/先验和数据驱动的学习能力。

### Plug and play(PnP)

第一种方法称为即插即用 (PnP)。这里的关键思想是将先前训练的神经网络 (NN) 插入到优化算法的更新部分（即代替解析表达式），然后立即将修改后的算法应用于优化从相同的样本中采样任务分配（无需额外培训）。

**Alternating direction method of multipliers (ADMM)**

原始优化算法：

$$
\min_{x,y}f(y)+g(x),\quad \text{subject to  \quad x=y}
$$

ADMM 算法：

$$
\begin{aligned}
x^{k+1} & =\operatorname{prox}_{\beta g}\left(y^k-u^k\right) \\
y^{k+1} & =\operatorname{prox}_{\alpha f}\left(x^{k+1}+u^k\right) \\
u^{k+1} & =u^k+x^{k+1}-y^{k+1}
\end{aligned}
$$

也就是说，训练参数的更新过程被替换为了：

$$
x^{k+1}=H_\theta(y^k-u^k)
$$

其中，$H_\theta$ 中的参数 $\theta$ 可以通过以下学习得到：

$$
\theta \in \underset{\tilde{\theta}}{\arg \min } \mathcal{L}(\tilde{\theta})
$$

### Algorithm unrolling

原始优化算法为迭代算法：

$$
x^{k+1}=T(x^k;d),\quad k=0,1,2,...,\infty
$$

展开算法：

$$
x^{k+1}=T_{\theta^k}(x^k;d),\quad k=0,1,2,...,K-1
$$

其中，$K$ 是展开的网络层数，$\theta^k$ 是第 $k$ 层展开网络的参数，可以通过下式得到：

$$
\min_{\{\theta^k\}_{k=0}^{K-1}}\mathcal{L}(x^K(\{\theta^k\}_{k=0}^{K-1}))
$$

![[L2O fig4.png]]

算法展开主要解决的问题：
1. Sparse and low rank regression：稀疏问题和低秩回归
2. Probabilistic graphical model：概率图模型
3. Differential equations：从数据中发现偏微分方程（PDE）
4. Quadratic optimization：二次优化

迭代优化算法可以分为：
1. Forward-backward splitting(FBS):
	1. ISTA:  solve ${\ell}_1\text{-minimization}$
	2. IHT: solve ${\ell}_0\text{-minimization}$
	3. AMP: approximate message passing
2. Primal-dual methods
	1. PDHG: primal-dual hybrid gradient
	2. ADMM: alternating direction method of multipliers
3. Others

根据优化目标，算法展开又可以分为： 
1. Objective-Based：最小化训练损失函数 ：$\min\limits_{\Theta}\ell(\Theta)$
2. Inverse Problems：从测量信号恢复真实信号 ：$\min\limits_{\Theta}\mathbb{E}_{d\sim\mathcal{D}}\left[\|x^K(\Theta,d)-x_d^\star\|^2\right]$

### 理论研究

>[!Capacity (有效性)]
>针对L2O的模型，是否存在可证明的模型参数，能够确保L2O在任务分布（即“在分布”）上优于传统的优化算法？是否有任何“保护机制”，以确保L2O在分布外（OoD）任务上表现不劣于传统算法？

>[!Trainability (可训练性)]
>如果这样的参数存在，我们应该使用什么训练方法来获取这些参数？是否存在保证训练方法收敛于理想参数的保证？

>[!Generalization (泛化能力)]
>经过训练的模型是否具有泛化性能，例如，能否泛化到来自相同训练实例源的测试实例（即“插值”）？经过训练的模型能否“外推”，例如在测试实例更加复杂的情况下表现，超过任何训练实例的情况？

>[!Interpretability (可解释性)]
>我们如何解释L2O模型学到了什么？

# Open-L2O 基准

## 稀疏凸优化（稀疏反问题、LASSO 最小化）

>[!稀疏反问题：稀疏重构]
>从嘈杂的线性测量中恢复稀疏向量$x_q^*$
>$$
>b_q=Ax_q^*+\varepsilon_q
>$$
>评价指标：NMSE
>$$
>\operatorname{NMSE}_{\operatorname{dB}}(\hat{x}_q,x_q^*)=10\log_{10}\left(\left\|\hat{x}_q-x_q^*\right\|^2/\|x_q^*\|^2\right)
>$$

![[L2O fig5.png]]

>[!LASSO 最小化]
>优化目标：
>$$
>x_q^{\mathrm{Lasso}}=\arg\min\limits_x f_q(x),\quad\mathrm{where}f_q(x)=\dfrac12\|Ax-b_q\|_2^2+\lambda\|x\|_1
>$$
>评价指标：Relative Loss
>$$
>R_{f,\mathcal{Q}}(x)=\dfrac{\mathbb{E}_{q\sim\mathcal{Q}}[f_{q}(x)-f_{q}^{*}]}{\mathbb{E}_{q\sim\mathbb{Q}}[f_{q}^{*}]}
>$$

![[L2O fig6.png]]

## 最小化非凸 Rastrigin 函数

>[!优化目标]
>$$
>x_q=\arg\min\limits_x f_q(x), \quad \text{where}f_q(\boldsymbol{x})=\dfrac{1}{2}\|\boldsymbol{A}_q\boldsymbol{x}-\boldsymbol{b}_q\|_2^2-\alpha\boldsymbol{c}_q\cos(2\pi\boldsymbol{x})+\alpha n
>$$

![[L2O fig7.png]]


## 训练神经网络

![[L2O fig8.png]]

## 实验结论

如果存在适当的模型或问题结构可以利用，==基于模型的L2O方法优于无模型的方法和分析优化器==。在从相同任务分布中采样的优化问题上，基于模型的L2O优化器表现出稳健、一致且可靠的性能。

# 结论

本文提供了新兴的L2O领域的首次“全景式”综述，同时伴随着首个基准测试。文章揭示了尽管L2O领域有着巨大潜力，但仍处于起步阶段，面临着从实践到理论的诸多挑战和研究机会。

在理论方面，第3.4节列出了许多开放的理论问题，帮助我们理解为什么以及如何基于模型的 L2O 方法能够胜过传统的优化器。此外，对于模型无关的L2O方法，其理论基础几乎不存在。例如，尽管L2O优化器的训练通常在实践中取得了成功，但几乎没有建立这种L2O训练过程收敛性性能的理论结果，这使得获得L2O模型的普遍可行性受到质疑。此外，对于基于模型的和模型无关的L2O方法，对于将训练的L2O推广或适应于任务分布之外的优化问题的保证还未受到广泛的研究，但这一点非常需要。

在实践方面，模型无关的L2O方法的可扩展性可能是成为更实用的最大障碍。这包括扩展到更大更复杂的模型和更多的L2O测试迭代。对于特定模型的L2O，当前的经验成功仅限于逆问题和稀疏优化的一些特殊实例，依赖于逐案例建模。探索更广泛的应用，甚至建立更通用的框架是有需求的。

此外，基于模型和基于模型无关的L2O方法之间不存在绝对的边界，两者之间的谱系可能会提示许多新的研究机会。[159]提出了一个很好的观点，即展开网络（作为基于模型的L2O的一个例子）是通用网络和解析算法之间的中间状态，可能更具数据效率进行学习。这一观点得到了[147]的支持，后者进一步主张，展开模型可以被视为后续数据驱动模型搜索的坚实起点。我们也相信，当前的连续优化算法可以改进端到端学习方法，以从已有的理论保证和最先进的算法中受益。

因此，为了总结本文，让我们引用温斯顿·丘吉尔的话：“现在这并不是结束，甚至不是结束的开始。然而，也许这是开始的结束。”尽管我们在本文中讨论的大多数方法仍处于探索性部署阶段，尚未准备好作为通用或商业求解器，但我们坚信机器学习刚刚开始为经典优化领域提供帮助。L2O研究进展的爆炸尚未开始。