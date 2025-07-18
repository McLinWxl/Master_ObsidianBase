# 文献参考：

标签： #ISTA_based 

[[SPARSE SIGNAL RESTORATION.pdf]]

迭代软阈值算法用于求解如下的稀疏信号重构问题：  
![\min_{\bf{x}} \|{\bf{y}} - {\bf{Hx}}\|_2^2 + \lambda \|{\bf{x}}\|_1.](https://math.jianshu.com/math?formula=%5Cmin_%7B%5Cbf%7Bx%7D%7D%20%5C%7C%7B%5Cbf%7By%7D%7D%20-%20%7B%5Cbf%7BHx%7D%7D%5C%7C_2%5E2%20%2B%20%5Clambda%20%5C%7C%7B%5Cbf%7Bx%7D%7D%5C%7C_1.)  
理解ISTA算法的由来，需要从几块内容着手：

-   Majorization-minimization
-   软阈值函数的由来
-   迭代软阈值算法

---

# Majorization-minimization (MM) 简介

## 核心思想

MM 是一个应对优化问题的思想框架，它通过迭代的方式求解一个无约束待优化问题 ![\min_{\boldsymbol{x}} J({\boldsymbol{x}})](https://math.jianshu.com/math?formula=%5Cmin_%7B%5Cboldsymbol%7Bx%7D%7D%20J(%7B%5Cboldsymbol%7Bx%7D%7D))。 其主要思路为，在第 ![k+1](https://math.jianshu.com/math?formula=k%2B1) 步迭代（假设上一步求得![{\boldsymbol{x}}_k](https://math.jianshu.com/math?formula=%7B%5Cboldsymbol%7Bx%7D%7D_k) ），通过寻找另一个容易优化求得全局最优（极小）的函数 ![G_k ({\mathbf{x}})](https://math.jianshu.com/math?formula=G_k%20(%7B%5Cmathbf%7Bx%7D%7D))，使其满足  
![G_k ({\bf{x}}) \ge J({\bf{x}}), \forall {\bf{x}} \\ G_k ({\bf{x}}_k) = J({\bf{x}}_k).](https://math.jianshu.com/math?formula=G_k%20(%7B%5Cbf%7Bx%7D%7D)%20%5Cge%20J(%7B%5Cbf%7Bx%7D%7D)%2C%20%5Cforall%20%7B%5Cbf%7Bx%7D%7D%20%5C%5C%20G_k%20(%7B%5Cbf%7Bx%7D%7D_k)%20%3D%20J(%7B%5Cbf%7Bx%7D%7D_k).) 然后，将 ![G_k ({\mathbf{x}})](https://math.jianshu.com/math?formula=G_k%20(%7B%5Cmathbf%7Bx%7D%7D)) 的极小值点作为新的迭代值 ![{\boldsymbol{x}}_{k+1}](https://math.jianshu.com/math?formula=%7B%5Cboldsymbol%7Bx%7D%7D_%7Bk%2B1%7D)。这样，可以保证  
![J({\boldsymbol{x}}_{k+1}) <= G({\boldsymbol{x}}_{k+1}) < G({\boldsymbol{x}}_{k}) = J({\boldsymbol{x}}_{k}),](https://math.jianshu.com/math?formula=J(%7B%5Cboldsymbol%7Bx%7D%7D_%7Bk%2B1%7D)%20%3C%3D%20G(%7B%5Cboldsymbol%7Bx%7D%7D_%7Bk%2B1%7D)%20%3C%20G(%7B%5Cboldsymbol%7Bx%7D%7D_%7Bk%7D)%20%3D%20J(%7B%5Cboldsymbol%7Bx%7D%7D_%7Bk%7D)%2C) 即保证每步迭代都能使目标函数 ![J({\boldsymbol{x}})](https://math.jianshu.com/math?formula=J(%7B%5Cboldsymbol%7Bx%7D%7D)) 的值下降。

#### 一种构造 ![G_k ({\mathbf{x}})](https://math.jianshu.com/math?formula=G_k%20(%7B%5Cmathbf%7Bx%7D%7D)) 的方法

一种很好用的构造 ![G_k ({\mathbf{x}})](https://math.jianshu.com/math?formula=G_k%20(%7B%5Cmathbf%7Bx%7D%7D)) 的方法，是在原始函数 ![J({\boldsymbol{x}})](https://math.jianshu.com/math?formula=J(%7B%5Cboldsymbol%7Bx%7D%7D)) 上加入一个半正定的 ![\mathbf{x}](https://math.jianshu.com/math?formula=%5Cmathbf%7Bx%7D) 的二次型，如下所示：  
![G_k ({\bf{x}}) = J({\bf{x}}) + ({\bf{x}} - {\bf{x}}_k)^\top (\alpha {\bf{I}} - {\bf{H}}^\top{\bf{H}}) ({\bf{x}} - {\bf{x}}_k),](https://math.jianshu.com/math?formula=G_k%20(%7B%5Cbf%7Bx%7D%7D)%20%3D%20J(%7B%5Cbf%7Bx%7D%7D)%20%2B%20(%7B%5Cbf%7Bx%7D%7D%20-%20%7B%5Cbf%7Bx%7D%7D_k)%5E%5Ctop%20(%5Calpha%20%7B%5Cbf%7BI%7D%7D%20-%20%7B%5Cbf%7BH%7D%7D%5E%5Ctop%7B%5Cbf%7BH%7D%7D)%20(%7B%5Cbf%7Bx%7D%7D%20-%20%7B%5Cbf%7Bx%7D%7D_k)%2C) 需要满足 ![\alpha \ge max {\rm{eig}}({\boldsymbol{H}}^\top{\boldsymbol{H}})](https://math.jianshu.com/math?formula=%5Calpha%20%5Cge%20max%20%7B%5Crm%7Beig%7D%7D(%7B%5Cboldsymbol%7BH%7D%7D%5E%5Ctop%7B%5Cboldsymbol%7BH%7D%7D))。  
还有一些其他的 ![G_k ({\mathbf{x}})](https://math.jianshu.com/math?formula=G_k%20(%7B%5Cmathbf%7Bx%7D%7D)) 的构造方式，对应额外的一些算法，不在此介绍了。

#### 举个例子：Landweber 迭代

考虑 ![J({\mathbf{x}}) = \|{\bf{y}} - {\bf{Hx}}\|_2^2](https://math.jianshu.com/math?formula=J(%7B%5Cmathbf%7Bx%7D%7D)%20%3D%20%5C%7C%7B%5Cbf%7By%7D%7D%20-%20%7B%5Cbf%7BHx%7D%7D%5C%7C_2%5E2)，并且采用上面的方法进行迭代式的优化求解，那么步骤如下：

1.  第 ![k+1](https://math.jianshu.com/math?formula=k%2B1) 次迭代时，构造  
    ![G_k ({\mathbf{x}}) = \|{\mathbf{y}} - {\mathbf{Hx}}\|_2^2 + ({\mathbf{x}} - {\mathbf{x}}_k)^\top (\alpha {\mathbf{I}} - {\mathbf{H}}^\top{\mathbf{H}}) ({\mathbf{x}} - {\mathbf{x}}_k), \tag{1} \label{eq1}](https://math.jianshu.com/math?formula=G_k%20(%7B%5Cmathbf%7Bx%7D%7D)%20%3D%20%5C%7C%7B%5Cmathbf%7By%7D%7D%20-%20%7B%5Cmathbf%7BHx%7D%7D%5C%7C_2%5E2%20%2B%20(%7B%5Cmathbf%7Bx%7D%7D%20-%20%7B%5Cmathbf%7Bx%7D%7D_k)%5E%5Ctop%20(%5Calpha%20%7B%5Cmathbf%7BI%7D%7D%20-%20%7B%5Cmathbf%7BH%7D%7D%5E%5Ctop%7B%5Cmathbf%7BH%7D%7D)%20(%7B%5Cmathbf%7Bx%7D%7D%20-%20%7B%5Cmathbf%7Bx%7D%7D_k)%2C%20%5Ctag%7B1%7D%20%5Clabel%7Beq1%7D)
2.  其全局极小值点通过一阶梯度为 ![0](https://math.jianshu.com/math?formula=0) 求得  
    ![\frac{\partial }{\partial {\boldsymbol{x}}} G_k ({\boldsymbol{x}}) = 2 \alpha {\boldsymbol{x}} - 2 {\boldsymbol{H}}^\top {\boldsymbol{y}} - 2 (\alpha {\boldsymbol{I}} - {\boldsymbol{H}}^\top {\boldsymbol{H}}) {\boldsymbol{x}}_k = 0, \\ \Rightarrow {\boldsymbol{x}}_{k+1} = {\boldsymbol{x}}_K + \frac{1}{\alpha} {\boldsymbol{H}}^\top({\boldsymbol{y}} - {\boldsymbol{Hx}}_k).](https://math.jianshu.com/math?formula=%5Cfrac%7B%5Cpartial%20%7D%7B%5Cpartial%20%7B%5Cboldsymbol%7Bx%7D%7D%7D%20G_k%20(%7B%5Cboldsymbol%7Bx%7D%7D)%20%3D%202%20%5Calpha%20%7B%5Cboldsymbol%7Bx%7D%7D%20-%202%20%7B%5Cboldsymbol%7BH%7D%7D%5E%5Ctop%20%7B%5Cboldsymbol%7By%7D%7D%20-%202%20(%5Calpha%20%7B%5Cboldsymbol%7BI%7D%7D%20-%20%7B%5Cboldsymbol%7BH%7D%7D%5E%5Ctop%20%7B%5Cboldsymbol%7BH%7D%7D)%20%7B%5Cboldsymbol%7Bx%7D%7D_k%20%3D%200%2C%20%5C%5C%20%5CRightarrow%20%7B%5Cboldsymbol%7Bx%7D%7D_%7Bk%2B1%7D%20%3D%20%7B%5Cboldsymbol%7Bx%7D%7D_K%20%2B%20%5Cfrac%7B1%7D%7B%5Calpha%7D%20%7B%5Cboldsymbol%7BH%7D%7D%5E%5Ctop(%7B%5Cboldsymbol%7By%7D%7D%20-%20%7B%5Cboldsymbol%7BHx%7D%7D_k).) 上式即是 Landweber 迭代公式。

-   矩阵知识比较好的人，也可以直接将式 (1) 改写成如下形式  
    ![G_k ({\mathbf{x}}) = \alpha \|{\mathbf{x}}_k + \frac{1}{\alpha} {\mathbf{H}}^\top ({\mathbf{y}} - {\mathbf{Hx}}_k) - {\bf{x}}\|_2^2 + C](https://math.jianshu.com/math?formula=G_k%20(%7B%5Cmathbf%7Bx%7D%7D)%20%3D%20%5Calpha%20%5C%7C%7B%5Cmathbf%7Bx%7D%7D_k%20%2B%20%5Cfrac%7B1%7D%7B%5Calpha%7D%20%7B%5Cmathbf%7BH%7D%7D%5E%5Ctop%20(%7B%5Cmathbf%7By%7D%7D%20-%20%7B%5Cmathbf%7BHx%7D%7D_k)%20-%20%7B%5Cbf%7Bx%7D%7D%5C%7C_2%5E2%20%2B%20C) 从而得到相同的迭代式。

---

# 软阈值函数的由来

考虑求解如下形式的问题 （类似于基于稀疏约束的去噪）  
![\min \|{\boldsymbol{y}} - {\boldsymbol{x}}\|_2^2 + \lambda \|{\boldsymbol{x}}\|_1, \tag{2}\label{eq2}](https://math.jianshu.com/math?formula=%5Cmin%20%5C%7C%7B%5Cboldsymbol%7By%7D%7D%20-%20%7B%5Cboldsymbol%7Bx%7D%7D%5C%7C_2%5E2%20%2B%20%5Clambda%20%5C%7C%7B%5Cboldsymbol%7Bx%7D%7D%5C%7C_1%2C%20%5Ctag%7B2%7D%5Clabel%7Beq2%7D) 可以对向量中的每个元素单独求解，即  
![\min (y_i - x_i)^2 + \lambda |x_i|](https://math.jianshu.com/math?formula=%5Cmin%20(y_i%20-%20x_i)%5E2%20%2B%20%5Clambda%20%7Cx_i%7C) 通过求梯度为 0 的点可得  
![y_i = x_i + \frac{\lambda}{2} {\rm{sign}}(x_i),](https://math.jianshu.com/math?formula=y_i%20%3D%20x_i%20%2B%20%5Cfrac%7B%5Clambda%7D%7B2%7D%20%7B%5Crm%7Bsign%7D%7D(x_i)%2C) 通过上式可以反求出  
![x_i = \begin{cases} y_i + \frac{\lambda}{2} & y_i < - \frac{\lambda}{2} \\ 0 & |y_i | \le \frac{\lambda}{2} \\ y_i - \frac{\lambda}{2} & y_i > \frac{\lambda}{2} \end{cases}](https://math.jianshu.com/math?formula=x_i%20%3D%20%5Cbegin%7Bcases%7D%20y_i%20%2B%20%5Cfrac%7B%5Clambda%7D%7B2%7D%20%26%20y_i%20%3C%20-%20%5Cfrac%7B%5Clambda%7D%7B2%7D%20%5C%5C%200%20%26%20%7Cy_i%20%7C%20%5Cle%20%5Cfrac%7B%5Clambda%7D%7B2%7D%20%5C%5C%20y_i%20-%20%5Cfrac%7B%5Clambda%7D%7B2%7D%20%26%20y_i%20%3E%20%5Cfrac%7B%5Clambda%7D%7B2%7D%20%5Cend%7Bcases%7D) 将对 ![y_i](https://math.jianshu.com/math?formula=y_i) 的这个操作叫做软阈值函数，记为 ![{\rm{soft}} (y_i, \frac{\lambda}{2})](https://math.jianshu.com/math?formula=%7B%5Crm%7Bsoft%7D%7D%20(y_i%2C%20%5Cfrac%7B%5Clambda%7D%7B2%7D))。  
即，问题 (2) 的解为 ![{\boldsymbol{x}} = {\rm{soft}}({\boldsymbol{y}}, \frac{\lambda}{2})](https://math.jianshu.com/math?formula=%7B%5Cboldsymbol%7Bx%7D%7D%20%3D%20%7B%5Crm%7Bsoft%7D%7D(%7B%5Cboldsymbol%7By%7D%7D%2C%20%5Cfrac%7B%5Clambda%7D%7B2%7D))。

---

# 迭代软阈值算法

求解如下问题  
![\min_{\boldsymbol{x}} \|{\boldsymbol{y}} - {\boldsymbol{Hx}}\|_2^2 + \lambda \|{\boldsymbol{x}}\|_1.](https://math.jianshu.com/math?formula=%5Cmin_%7B%5Cboldsymbol%7Bx%7D%7D%20%5C%7C%7B%5Cboldsymbol%7By%7D%7D%20-%20%7B%5Cboldsymbol%7BHx%7D%7D%5C%7C_2%5E2%20%2B%20%5Clambda%20%5C%7C%7B%5Cboldsymbol%7Bx%7D%7D%5C%7C_1.)

1.  根据 MM 的思想，在每一步迭代时，通过添加半正定二次型，构造  
    ![\begin{aligned} G_k({\mathbf{x}}) =& \|{\mathbf{y}} - {\bf{Hx}}\|_2^2 + \lambda \|{\bf{x}}\|_1 + ({\bf{x}} - {\bf{x}}_k)^\top (\alpha {\bf{I}} - {\bf{H}}^\top{\bf{H}}) ({\bf{x}} - {\bf{x}}_k) \\ =& \alpha \|{\bf{x}}_k + \frac{1}{\alpha} {\bf{H}}^\top ({\bf{y}} - {\bf{Hx}}_k) - {\bf{x}}\|_2^2 + C + \lambda \|{\bf{x}}\|_1 \end{aligned}](https://math.jianshu.com/math?formula=%5Cbegin%7Baligned%7D%20G_k(%7B%5Cmathbf%7Bx%7D%7D)%20%3D%26%20%5C%7C%7B%5Cmathbf%7By%7D%7D%20-%20%7B%5Cbf%7BHx%7D%7D%5C%7C_2%5E2%20%2B%20%5Clambda%20%5C%7C%7B%5Cbf%7Bx%7D%7D%5C%7C_1%20%2B%20(%7B%5Cbf%7Bx%7D%7D%20-%20%7B%5Cbf%7Bx%7D%7D_k)%5E%5Ctop%20(%5Calpha%20%7B%5Cbf%7BI%7D%7D%20-%20%7B%5Cbf%7BH%7D%7D%5E%5Ctop%7B%5Cbf%7BH%7D%7D)%20(%7B%5Cbf%7Bx%7D%7D%20-%20%7B%5Cbf%7Bx%7D%7D_k)%20%5C%5C%20%3D%26%20%5Calpha%20%5C%7C%7B%5Cbf%7Bx%7D%7D_k%20%2B%20%5Cfrac%7B1%7D%7B%5Calpha%7D%20%7B%5Cbf%7BH%7D%7D%5E%5Ctop%20(%7B%5Cbf%7By%7D%7D%20-%20%7B%5Cbf%7BHx%7D%7D_k)%20-%20%7B%5Cbf%7Bx%7D%7D%5C%7C_2%5E2%20%2B%20C%20%2B%20%5Clambda%20%5C%7C%7B%5Cbf%7Bx%7D%7D%5C%7C_1%20%5Cend%7Baligned%7D)
2.  需要求 ![{\boldsymbol{x}}_{k+1} = \min_{\boldsymbol{x}} G_k({\boldsymbol{x}})](https://math.jianshu.com/math?formula=%7B%5Cboldsymbol%7Bx%7D%7D_%7Bk%2B1%7D%20%3D%20%5Cmin_%7B%5Cboldsymbol%7Bx%7D%7D%20G_k(%7B%5Cboldsymbol%7Bx%7D%7D))，根据上一节中的软阈值方法，可轻易求得  
    ![{\mathbf{x}}_{k+1} = {\rm{soft}} ({\mathbf{x}}_k + \frac{1}{\alpha} {\mathbf{H}}^\top ({\mathbf{y}} - {\mathbf{Hx}}_k), \frac{\lambda}{2 \alpha})](https://math.jianshu.com/math?formula=%7B%5Cmathbf%7Bx%7D%7D_%7Bk%2B1%7D%20%3D%20%7B%5Crm%7Bsoft%7D%7D%20(%7B%5Cmathbf%7Bx%7D%7D_k%20%2B%20%5Cfrac%7B1%7D%7B%5Calpha%7D%20%7B%5Cmathbf%7BH%7D%7D%5E%5Ctop%20(%7B%5Cmathbf%7By%7D%7D%20-%20%7B%5Cmathbf%7BHx%7D%7D_k)%2C%20%5Cfrac%7B%5Clambda%7D%7B2%20%5Calpha%7D)) 即为迭代软阈值算法的步骤，其中参数需要满足 ![\alpha \ge max~{\rm{eig}}({\boldsymbol{H}}^\top{\boldsymbol{H}})](https://math.jianshu.com/math?formula=%5Calpha%20%5Cge%20max~%7B%5Crm%7Beig%7D%7D(%7B%5Cboldsymbol%7BH%7D%7D%5E%5Ctop%7B%5Cboldsymbol%7BH%7D%7D))。



