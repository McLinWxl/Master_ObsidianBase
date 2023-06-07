# 基本信息：

[[2022_Su et al_Deep Alternating Projection Networks for Gridless DOA Estimation With Nested.pdf]]

作者：Xiao-long Su, Pan-he Hu*, Zhen Liu, Jun-peng Shi, Xiang Li

单位：国防科技大学

期刊：IEEE Signal Processing Letters

年份：2022

中科院分区/影响因子：二区 / 3.201

标签： [[DUN]] [[DOA]]


# 摘要：

提出 DAPN（Deep Alternating Projection Network）网络实现嵌套阵列用于无网格 DOA 估计

1.  将协方差矩阵转换为原子范数形式的测量向量，以此降低矩阵维度
2.  训练 DAPN 网络交替获得半正定矩阵和相应的 Hermitian Toeplitz 矩阵，损失函数是通过网络输出的迹导出的
3.  使用不规则根的多信号分类（MUSIC，MUltiple SIgnal Classification）方法获得嵌套阵列的无网格 DOA 估计

所提出的方法可以加快收敛速度、降低计算成本

# 引言:

DOA 估计在相控阵雷达、无源声纳以及无线通信中起着非常重要的作用

## 研究现状

1.  以往基于网格的方法从有限的网格中进行 DOA 估计，但是，会受到网格不匹配问题的影响
2.  [5]-[6] 中使用离网格方法、[7]-[10] 中使用无网格方法实现了 DOA 估计
	1.  [7]中在有限快拍的条件下，应用确定性原子范数（Deterministic Atomic Norm）优化来实现无网格 DOA 估计，其约束与协方差矩阵有关
	2.  [9]中提出了交叉投影（Alternating Projection）法，通过一定的迭代次数实现了无网格的 DOA 估计，克服了插值法计算量大的问题
3. 结合稀疏线性阵列的几何性质，[10] 中利用对偶原子范数最小化方法提升了无网格 DOA 估计的精度
4. [11]-[13] 表明，与均匀线性阵列相比，稀疏线性阵列（如嵌套阵列和互质阵列）需要更少的传感器（孔径相同时），可以降低硬件成本
5. 同时，已经提出了深度展开网络来解决问题稀疏重构问题 [14]，具有模型驱动和数据驱动方法的优点 [15]-[17]
	1. [18] 提出投影梯度下降法（PGD，Projected Gradient Descent），在非相邻层之间的全连接层展开网络，可以提高稀疏重构的性能
	2. [19] 提出 ADMM-net 用于稀疏孔径 ISAR 成像，可以加快收敛速度并降低计算成本

## 创新点

1.  对于嵌套阵列，使用交叉投影方法实现无网格的 DOA 估计
2.  结合深度展开网络，提升稀疏重构的性能

# 模型推导：

## 嵌套阵列的信号模型

![](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/7jP2lRj7REG6l8g5/img/a3db03e3-88d8-4b00-bdaf-0e5fefe5d833.png)

二级嵌套麦克风阵列的位置排布结构

$$\xi_m=\begin{cases}md,\quad m=1,2,...,M/2,\quad First\ subarray\\(m-M/2)(M/2+1)d,\quad m=M/2+1,M/2+2,...,M,\quad Second\ subarray\end{cases}$$

对应第 n 快拍，阵列的输出为：

$$\vec{x}=[x_1(n),x_2(n),...,x_M(n)]^T=\sum_{k=1}^K\vec{a}(\theta_k)s_k(n)+\vec{w}(n)=\vec{A}\vec{s}(n)+\vec{w}(n)$$

1. $\vec{w}(n)$ 表示白噪声，与声源独立
2. $\vec{A}=[\vec{a}(\theta_1),...,\vec{a}(\theta_K)]$ 表示 Steering Matrix
3. $\vec{a}(\theta_k)=[a_1(\theta_k),...,a_M(\theta_k)]^T$ 表示 Steering Vector
4. 其中 $a_m(\theta_k)=e^{-j(2\pi\xi_msin\theta_k/\lambda)}$
5. $\lambda$ 为声源波长

输出信号的协方差矩阵可以表示为：

$$\vec{R}=E[\vec{x}(n)\vec{x}^H(n)]=\vec{A}\ diag([\sigma_1^1,\sigma_2^2,...,\sigma_K^2]^T)\vec{A}^H+\sigma_w^2\vec{I}M\\ \approx\cfrac{1}{N}\sum_{n=1}^N\vec{x}(n)\vec{x}^H(n)$$

## 利用原子范数减少维度

利用特征分解来构造原子范数形式的测量向量。原子范数是重建测量向量所需的最小“原子”数
首先对协方差矩阵进行特征分解：

$$\vec{R}=\vec{U}_S\Delta_S\vec{U}_S^H+\vec{U}_W\Delta_W\vec{U}_W^H$$

1. $\vec{\Delta}_S,\ \vec{\Delta}_W$ 分别表示 $K$ 个较大特征值和 $M-K$ 个较小特征值对应的对角矩阵
2. $\vec{U}_S,\ \vec{U}_W$ 表示 $M*K, M*(M-K)$ 维的信号子空间和噪声子空间

由于信号子空间和噪声子空间的正交性，协方差矩阵和信号子空间的乘积为：

$$\vec{R}\vec{U}_S=\vec{U}_S\Delta_S\vec{U}_S^H\vec{U}_S+\vec{U}_W\Delta_W\vec{U}_W^H\vec{U}_S=\vec{U}_S\Delta_S$$

同时有：

$$\vec{R}\vec{U}_S=\vec{A}\vec{\Delta}_S\vec{A}^H\vec{U}_S+\sigma_w^2\vec{I}_M\vec{U}_S$$

可以得出：

$$\vec{A}\vec{\Delta}_S\vec{A}^H\vec{U}_S=\vec{U}_S\vec{\Delta}_S-\sigma_w^2\vec{I}_M\vec{U}_S=\vec{U}_S(\vec{\Delta}_S-\sigma_w^2\vec{I}_M)$$

定义 $\vec{\Delta}_S\vec{A}^H\vec{U}_S=\vec{B}：$

$$\vec{A}\vec{\Delta}S\vec{A}^H\vec{U}S=\vec{A}\vec{B}\\=\begin{bmatrix} {\sum_{k=1}^Ka{1,k}b_{k,1}}&{\cdots}&{\sum_{k=1}^Ka_{1,k}b_{k,p}}&{\cdots}&{\sum_{k=1}^Ka_{1,k}b_{k,K}}\\ {\vdots}&{\ddots}&{\vdots}&{\ddots}&{\vdots}\\ {\sum_{k=1}^Ka_{M,k}b_{k,1}}&{\cdots}&{\sum_{k=1}^Ka_{M,k}b_{k,p}}&{\cdots}&{\sum_{k=1}^Ka_{M,k}b_{k,K}}\\ \end{bmatrix}$$

同时：

$$\vec{A}\vec{\Delta}_S\vec{A}^H\vec{U}_S=\vec{U}_S(\vec{\Delta}S-\sigma_w^2\vec{I}M)=\\\begin{bmatrix} {\mu_{1,1}(\sigma_1^2-\sigma_w^2)}&{\cdots}&{\mu_{1,p}(\sigma_p^2-\sigma_w^2)}&{\cdots}&{\mu_{1,K}(\sigma_K^2-\sigma_w^2)}\\ {\vdots}&{\ddots}&{\vdots}&{\ddots}&{\vdots}\\ {\mu_{M,1}(\sigma_1^2-\sigma_w^2)}&{\cdots}&{\mu_{M,p}(\sigma_p^2-\sigma_w^2)}&{\cdots}&{\mu_{M,K}(\sigma_K^2-\sigma_w^2)}\\ \end{bmatrix}$$

可以得出：

$$\mu_{m,p}=\cfrac{1}{\sigma_P^2-\sigma_w^2}\ \sum_{k=1}^Ka_{m,k}\ b_{k,p}$$

令其乘以对应的特征值 $\sigma_p^2：$

$$\sum_{P=1}^K\sigma_p^2\mu_{m,p}=\sum_{p=1}^K\cfrac{\sigma_p^2}{\sigma_p^2-\sigma_w^2}\ \ \sum_{k=1}^Ka_{m,k}\ b_{k,p}=\sum_{p=1}^Ka_{m,k}\ \ \sum_{k=1}^K\cfrac{\sigma_p^2}{\sigma_p^2-\sigma_w^2}\ b_{k,p}$$

定义 $\sum_{k=1}^K\cfrac{\sigma_p^2}{\sigma_p^2-\sigma_w^2}\ b_{k,p}=f_k：$

$$\vec{y}=\sum_{p=1}^{K}\sigma_p^2\vec{\mu_p}=\sum_{k=1}^Kf_k\vec{a}(\theta_k)$$

$\vec{\mu}_p=[\mu_{1,p},\mu_{2,p},...,\mu_{M,p}]^T$ 表示信号子空间的第 $p$ 列，表示协方差矩阵第 $p$ 个特征值 $\sigma_p^2$ 对应的第 $p$ 个特征向量

同时 $f_k$ 可以表示为 $|f_k|~e^{j\phi_k}$，其中 $\phi_k$ 表示 $f_k$ 的相位。原子集可以表示为：

$$\mathcal{A}=\{ e^{j\phi}~\vec{a}(\theta)~|~\phi\in[0,2\pi),~\theta\in[-\frac{\pi}{2},\frac{\pi}{2}] \}$$

利用线性组合，测量矩阵 $\vec{y}$ 的原子 $l_o$ 范数可以表示为：

$$||\vec{y}||_{\cal{A},0}=inf_{|f_k|\geq0,~e^{j\phi_k}\vec{a}(\theta_k)\in\cal{A}}\{K:\vec{y}=\sum_{k=1}^K|f_k|~e^{j\phi_k}\vec{a}(\theta_k)\}$$

## 深度交替投影网络

考虑到基于模型的交替投影方法[9]需要一定的迭代次数，本文应用深度展开网络来加快收敛速度。

![](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/7jP2lRj7REG6l8g5/img/65d0d23c-1bfe-4aa9-a53f-ef4cc64cc59d.png)

深度交替投影网络的结构图

如上图所示，提出了深度交替投影网络以实现无网格 DOA 估计。其中交替投影算法的迭代次数与网络层数对应。

投影到第 $l$ 层中的半正定（PSD，Positive semi-definite）集合可以表示为：

$$\vec{\Lambda}^{l}=P_D(\vec{Z}^{(l-1)})=\sum_{m=1}^Mmax(0,\eta_m^{(l-1)})\vec{\psi}_m^{(l-1)}(\vec{\psi}_m^{(l-1)})^H$$

其中，$\eta_m^{(l-1)},~\vec{\psi}_m^{(l-1)}$分别表示为$\vec{Z}^{(l-1)}$的第 $m$ 个特征值和特征向量，输入$\vec{Z}^{(0)}$可以表示为：

$$\vec{Z}^{(0)}=\begin{bmatrix}\vec{R}&\vec{y}\\\vec{y}^H&\vec{y}^H\vec{y}\end{bmatrix}$$

与均匀线阵（ULA）中的 Hermitian Toeplitz 集不同，嵌套数组在不规则 Hermitian Toeplitz 集上的投影为：

$$\vec{\Omega}^{(l)}=P_H(\vec{\Lambda}^{(l)})$$

$\vec{\Omega}^{(l)}$ 中第 $(\tau_1,~\tau_2)$ 个元素 $\vec{\Omega}_{\tau_1,~\tau_2}^{(l)}$ 可以表示为：

$$\vec{\Omega}{\tau_1,~\tau_2}^{(l)}=\begin{cases}\mathrm{mean}(\vec{\Lambda}{\tau_3,~\tau_4}^{(l-1)}+(\vec{\Lambda}{\tau_4,~\tau_3}^{(l-1)})^*),\quad \tau_1\leq\tau_2\\\mathrm{mean}((\vec{\Lambda}{\tau_3,~\tau_4}^{(l-1)})^*+\vec{\Lambda}_{\tau_4,~\tau_3}^{(l-1)}),\quad \tau_1>\tau_2\end{cases}$$

其中，$(\cdot)^*$ 表示复共轭操作。并且 $\tau_1,\tau_2\leq M,~\xi_{\tau_1}-\xi_{\tau_2}=\xi_{\tau_3}-\xi_{\tau_4}$

因此，$\vec{Z}^{(l)}$ 可以更新为：

$$\vec{Z}^{(l)}=\begin{bmatrix}\vec{\Omega}^{(l)}&\vec{y}\\\vec{y}^H&\vec{\Lambda}_{M+1,M+1}^{(l)}\end{bmatrix}$$

通过凸模拟代替非凸优化问题，网络的损失函数为：$$min~tr(\vec{Z}^{(L)})$$

其中 $tr(\cdot)$ 表示矩阵的迹。

实际上，对 $\vec{\Omega}^{(L)}$ 中第 L 层的输出利用[9]中的不规则 Vandermonde 分解和不规则 root-MUSIC 方法，通过嵌套阵列的无网格 DOA 估计可以表示为：

$$\min(\sum_{m_1=1}^M\sum_{m_2=1}^M\hat{\vec{{V}}}{m_1,m_2}\gamma^{\xi{m_1}-\xi_{m_2}})\quad s.t.~|\gamma|=1\\\hat{\theta}=-\arcsin(∠\gamma/\pi)$$

其中，$\hat{\vec{V}}_{m_1,m_2}$ 表示矩阵 $\hat{\vec{U}}_W\hat{\vec{U}}_W^H$ 中的第 $(m_1,~m_2)$ 个元素。$\hat{\vec{U}}_W$ 表示 $\vec{\Omega}^{(l)}$ 中的噪声子空间。

# 实验/仿真验证

### 有效性和泛化能力

表 1 为提出的改进算法与原始算法的对比。可以看出所提方法能够实现不同声源数量的 DOA 估计，验证了其泛化能力

![](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/7jP2lRj7REG6l8g5/img/ebb06f5d-7adf-4136-b38d-016a37a3f85d.png)

图 3-(a) 为 \gamma 的实部与虚部可视化图，图 3-(b) 为 DOA 估计谱图，验证了其有效性

![](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/7jP2lRj7REG6l8g5/img/ea3d5d97-d5c2-4346-bfef-e12cdfe39661.png)

### 计算复杂度和收敛率

深度展开网络中的计算复杂度与网络的程度成正比

每层中的网络中 $\vec{\Lambda},~\vec{\Omega}$ 的特征分解计算复杂度分别为：$\mathcal{O}((M+1)^3),~\mathcal{O}(M)$

对比：

1.  [9]中未使用原子范数降维方法的协方差矩阵交叉投影的计算复杂的为：$\mathcal{O}((M+N)^3)$
2.  [6]中通过稀疏贝叶斯学习（SBL）方法实现离网格 DOA 估计，每次迭代的计算复杂度为：$\mathcal{O}(QM^4)$，其中 $Q$ 为网络数
3.  [10]中通过原子范数最小化（ANM）方法实现无网格 DOA 估计，协方差矩阵的计算需要 $M^2N$ 次乘法和 $M^2(N-1)$ 次加法运算

当声源设定为 $-10.3°,~0.3°$ 时，收敛率的比较如图:

收敛率定义为：$\cfrac{||\vec{\Omega}^{(l)}-\vec{\Omega}^{(l-1)}||_2}{||\vec{\Omega}^{(l-1)}||_2}$

![](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/7jP2lRj7REG6l8g5/img/ffa67a01-9378-47a6-9621-063d6c480830.png)

### RMSE 比较

$$RMSE=\sqrt{\cfrac{1}{VK}\sum_{v=1}^V\sum_{k=1}^K(\hat{\theta}_k^{(v)}-\theta_k)^2}$$

![](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/7jP2lRj7REG6l8g5/img/8682491b-93ec-45e9-aa5e-dfd907d17546.png)

# 结论

1.  提出了一个深度交替网络来实现嵌套阵列的无网格 DOA 估计
2.  将协方差矩阵转化为原子范数形式的测量向量，减少了投影过程中的计算成本
3.  对提出网络进行训练，交替获得嵌套阵列的半正定矩阵和对应的不规则 Hermitian Toeplitz 矩阵，其中损失函数与网络输出的迹有关
4.  应用不规则根 MUSIC 方法实现无网格 DOA 估计
5.  仿真结果表明，所提出的网络能够加快收敛速度并降低计算复杂度