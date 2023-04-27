# 基本信息：

[[2020_Monga et al_Algorithm Unrolling.pdf]]

作者：Vishal Monga，Yue-long Li，Yonina C. Eldar

单位：Pennsylvania State University

期刊：IEEE Signal Processing Magazine

年份：2021

中科院分区/影响因子：一区 / 15.204

标签： #ISTA_based #深度展开 

# 引言:

### 神经网络vs算法展开

深度神经网络优点：

1.  算法性能显著提升
2.  计算速度快

深度神经网络缺点：

1.  缺乏可解释性：纯粹的数据驱动方法
2.  计算资源需求大：多层网络带来了大量的参数
3.  训练数据集的数量、质量直接影响结果
4.  容易出现过拟合

算法展开：联系迭代算法与深度神经网络。兼具模型驱动和数据驱动的优点：

1.  可解释性强：通过先验知识物理建模
2.  更好的泛化性能：网络参数效率高，具有先验信息
3.  优秀的性能
4.  高效的计算速度
5.  合理的数据集大小

### 深度展开现状

2010，K. Gregor and Y. LeCun，Learning Fast Approximations of Sparse Coding 提出算法展开，将迭代算法连接到神经网络架构中

出现了针对信号和图像处理中许多重要问题迭代展开算法：压缩感知、反卷机、变分技术（variational techniques）

# 模型推导：

### 深度展开结构

![](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/7jP2lRj7REG6l8g5/img/e0a15265-effc-45bf-bb62-2e2f061b34da.png)

1.  给定迭代算法通过级联多层隐藏层产生深度网络
2.  迭代过程中的参数转移为神经网络中每一层的参数
3.  通过学习而非交叉验证或分析推导得到迭代中的参数
	1.  可以获得比原始迭代算法更好的性能
	2.  自然地继承了迭代过程中的可解释性

![](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/7jP2lRj7REG6l8g5/img/40ddef3e-d899-4a02-9938-04d737d0c31f.png)

## 通过算法展开生成可解释的网络

### ISTA：Iterative Shrinkage and Thresholding Algorithm

稀疏编码问题：给定一个输入向量 $\vec{y}\in \mathbb{R}^n$ ，一个完备字典 $\vec{W}\in\mathbb{R}^{n\times m}$，并且 $m>n$。期望实现用 $\vec{W}$ 实现 $\vec{y}$ 的稀疏表示。

$$\min_{x\in\mathbb{R}^m}{\cfrac{1}{2}||\vec{y}-\vec{W}\vec{x}||_2^2+\lambda||\vec{x}||_1}~,\quad \lambda>0$$

ISTA 执行以下迭代过程：

$$\vec{x}^{l+1}=\mathcal{S}_\lambda\{(\vec{I}-\cfrac{1}{\mu}~\vec{W}^T\vec{W})~\vec{x}^l+\cfrac{1}{\mu}~\vec{W}^T\vec{y}\}, \quad l=0,1,...$$

1.  $\vec{I}$ 是单位矩阵
2.  $\mu$ 是控制迭代部长的正参数
3.  $\mathcal{S}_\lambda(x)=sign(x)\cdot\max\{|x|-\lambda,0\}$ 是软阈值函数

算法展开：

![](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/7jP2lRj7REG6l8g5/img/a7593833-598e-4e60-a6c3-a0f0ba8380b0.png)

优化参数：$$\vec{W}_t=\vec{I}-\cfrac{1}{\mu}\vec{W}^T\vec{W}\quad and \quad \vec{W}_e=\cfrac{1}{\mu}\vec{W}^T$$

损失函数：$$\mathcal{L}(\vec{W}_t,\vec{W}_e,\lambda)=\cfrac{1}{N}\sum_{n=1}^N||\hat{\vec{x}}^n(\vec{y}^n;~\vec{W}_t,\vec{W}_e,\lambda)-\vec{{x}^{*}}^n||_2^2$$
