[[000e Deep Convolution Network for Direction of Arrival Estimation With Sparse Prior]]

[[000c 基于机器学习的阵列信号处理方法(4章)：卷积网络]]

[[2019_Wu et al_Deep Convolution Network for Direction of Arrival Estimation With Sparse Prior.pdf]]

[[2019_吴_基于机器学习的阵列信号处理方法.pdf]]

# 2305

### 对于低角度间隔下的耦合问题

使用TensorFlow1运行吴流丽的卷积网络DCNN-DOA论文中给出的原始代码，仅修改了训练epoch数（3->300）和测试数据集的比例（0.99->0.2）。运行结果和使用pytorch复现结果基本一致，都是在低角度间隔下出现耦合现象。

![[Pasted image 20230516153658.png|500]]

进一步检查数据集时发现，在英文的小论文中给出的测试集角度间隔为\[10.4, 15.2, 20.7, 25.9]，而代码中给出的测试集角度间隔为\[5.5, 13.5, 20.67, 50]（这与博士论文中的描述一致），而复现实验中出现耦合的部分正是角度间隔为 5.5 的部分。

可能原因：作者在发布完小论文之后，找到了解决耦合的办法，做进了毕业论文中，但是之前小论文中给的代码没有及时更新。


---

### 对于大概率无法训练的问题

将训练数据的输入进行归一化处理，使信号的伪谱分布在0～1之间：

$$x_{input}=\frac{x_{input}-\min(x_{input})}{\max(x_{input})-\min(x_{input})}$$

同时对于标签，不再采用将真实 DOA 位置置“1”的方法，而是根据模型公式推导，将其设置为训练时信号的功率平方。

![[Pasted image 20230519230756.png|500]]

### 修正复现结果

数据集：角度间隔集 $[10.4,15.2,20.7,25.9]$

#### Sample Index

![[Pasted image 20230519225659.png|500]]

![[Pasted image 20230519225629.png|500]]

#### RMSE vs SNR

数据集：1000组 $[-10.5, 4.5]\pm N~(0,1)$

![[Pasted image 20230517172628.png|500]]

![[Pasted image 20230517172635.png|500]]

![[Pasted image 20230519225952.png|500]]
