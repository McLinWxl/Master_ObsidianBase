# 基本信息：

[[2022_Chen et al_Fast Iteration Shrinkage Thresholding Unfolding Network for Acoustic Source.pdf]]

作者：Fangchao Chen，Youhong Xiao，Liang Yu，Lin Chen

单位：哈尔滨工程大学

会议：2022 5th International Conference on Information Communication and Signal Processing (ICICSP)

年份：2022

标签： [[深度展开]]  [[ISTA-based]] 

# 摘要：

提出了一种基于快速迭代收缩阈值展开（FISTA）算法的深度网络（FISTA-Net），它结合和模型驱动和数据驱动方法的优点。

该网络将FISTA算法的迭代步骤映射到深度神经网络中，通过端到端学习自适应确定模型的参数。

FISTA-Net在声源定位方面比经典的反卷积算法具有更高的空间分辨率和精度。

# 引言:

声源定位技术主要侧重于从麦克风阵列测量中估计声源位置。波束形成是最常用的声源定位算法。波束形成算法是从常规波束形成和反卷积两个阶段发展而来的。

然而，麦克风阵列的性能会影响传统波束形成结果的分辨率。反卷积方法虽然可以通过迭代点扩散函数来提高分辨率，但也带来了计算成本高、空间分辨率受限等问题。

基于数据驱动的深度学习方法具有更好的性能。

目前存在问题：

1.  基于模型驱动的传统方法具有良好的可解释性，但是分辨率有限，且迭代耗时
2.  基于数据驱动的深度学习方法具有计算速度快、分辨率高等优点，但是缺乏可解释性

# 模型推导：

### 数学模型

假设阵元数$M$，声源数$K，声源平面 $N\times N$

![](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/7jP2lRj7REG6l8g5/img/55984d64-8e7d-4622-a829-76c0d366310b.png)

波束形成算法为：$\mathrm{b}\left(\boldsymbol{r}_{s}, \omega\right)=\cfrac{1}{M^{2}} \boldsymbol{v}\left(\boldsymbol{r}_{s}, \omega\right) \boldsymbol{C}(\omega) \boldsymbol{v}^{\boldsymbol{H}}\left(\boldsymbol{r}_{s}, \omega\right)$

对于单声源，麦克风阵列的PSF为：$\operatorname{PSF}\left(\boldsymbol{r}_s \mid \boldsymbol{r}_i\right)=\frac{1}{M^2}\left|\boldsymbol{v}\left(\boldsymbol{r}_s\right)^H \boldsymbol{u}\left(\boldsymbol{r}_i\right)\right|^2$

CMS波束形成算法为：$\mathbf{b}\left(\boldsymbol{r}_s\right)=\sum_{i=1}^s q\left(\boldsymbol{r}_i\right)^2 \operatorname{PSF}\left(\boldsymbol{r}_s \mid \boldsymbol{r}_i\right)$

可以重写为：$\mathbf{b}=\mathbf{Ax }$

其中：$\mathbf{A} = \left[P S F\left(\boldsymbol{r}_1 \mid \boldsymbol{r}_i\right), \operatorname{PSF}\left(\boldsymbol{r}_2 \mid \boldsymbol{r}_i\right) \ldots P S F\left(\boldsymbol{r}_s \mid \boldsymbol{r}_i\right)\right]^T$

因此，其优化目标函数可以表示为：$\boldsymbol{x} = \underset{x}{\operatorname{minimize}}\|\boldsymbol{A} \boldsymbol{x}-\boldsymbol{b}\|_2^2+\lambda\|x\|_1$

可以使用FISTA算法来求解，迭代过程为：

$$\left\{\begin{array}{c} \boldsymbol{x}_k=S_\alpha\left(\boldsymbol{y}_k-\mu \boldsymbol{A}^T\left(\boldsymbol{A} \boldsymbol{y}_k-\boldsymbol{b}\right)\right) \\ t_{k+1}=\frac{1+\sqrt{1+4\left(t_k\right)^2}}{2} \\ \boldsymbol{y}_{k+1}=\boldsymbol{x}_k+\frac{t_k-1}{t_{k+1}}\left(\boldsymbol{x}_k-\boldsymbol{x}_{k-1}\right) \end{array}\right.$$

FISTA与ISTA的区别在于迭代点的选择不同。FISTA的收敛速度要快于ISTA

## 网络模型

![](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/7jP2lRj7REG6l8g5/img/4a217f77-00d2-4514-b8be-9e0a281e2666.png)

整体框架遵循了FISTA迭代算法的结构。特别在于在软阈值函数之前之后加入了卷积层+ReLU，其前卷积目的在于提取特征、去除弱信息；其后卷积目的在于还原目标维度。

损失函数定义为：$Loss=MSE(x, \hat{x})$，无需约束前后卷积的对称性

网络中可以学习的参数为

1.  $\mu_k$：迭代步长
2.  $\alpha_k$：软阈值函数的阈值
3.  $\rho_k=\frac{t_k-1}{t_{k+1}}$：FISTA计算迭代点时的权重

网络展开层数：13

启发：FISTA-Net重点在于对软阈值函数前后加入了卷积部分以提取特征、去除弱信息。但是其对$\boldsymbol{r}_k$模块未进一步展开，仅把迭代步长作为可学习的参数，其字典矩阵部分是否也应该考虑作为可以学习的参数（参考L-ISTA）。