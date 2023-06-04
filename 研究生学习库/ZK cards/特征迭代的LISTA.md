202306011007

Status: #idea   

Tags: 

# 特征迭代的LISTA

## 信号模型

$$
\boldsymbol{X}=\boldsymbol{A}\boldsymbol{S}+\boldsymbol{V}
$$

其中，$\boldsymbol{X}\in\mathbb{C}^{M\times N}$ ，$\boldsymbol{A}\in\mathbb{C}^{M\times L}$， $\boldsymbol{S}\in\mathbb{C}^{L\times N}$

## 优化算法

$$
\hat{\boldsymbol{S}}=\min||\boldsymbol{X}-\boldsymbol{A}\boldsymbol{S}||^2_2+\lambda||\Psi(\boldsymbol{S})||_1
$$

## ISTA算法求解

$$
\boldsymbol{r}^{k}=(\boldsymbol{I}-\frac{1}{\alpha}\boldsymbol{A}^H\boldsymbol{A})\boldsymbol{S}^k+\frac{1}{\alpha}\boldsymbol{A}^H\boldsymbol{X}=\boldsymbol{W}_t\boldsymbol{S}^k+\boldsymbol{W}_e\boldsymbol{X}
$$

$$
\boldsymbol{S^{k+1}}=\Psi^{-1}[Soft(\Psi(\boldsymbol{r}^{k}),\frac{\lambda}{\alpha})]
$$




---
# Reference
