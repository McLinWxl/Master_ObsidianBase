[[000d 基于机器学习的阵列信号处理方法(5章)：深度展开方法]]

# 2305

## 数据预处理

目前策略：对输入数据进行归一化，标签为真实功率谱（无预处理）

![[Pasted image 20230523142230.png|500]]

![[Pasted image 20230523142242.png|500]]

## 学习率影响

$lr=0.001$：非常容易过拟合。

![[Pasted image 20230523141545.png|500]]

$lr=0.0001$ ：更难出现过拟合，整体RMSE也较低。

![[Pasted image 20230523142305.png]]

## 结果

采用 $lr=0.0001$，$epoch=1500$ 时保存的模型进行评价:
1. 在重构功率谱时，波峰处往往带有一个较高的旁瓣，导致第二个声源的位置估计错误
2. 估计精度也较低（相对于DCNN）
3. 检查模型中的参数（软阈值函数的阈值）时发现，一共堆叠二十层，前十五层的阈值几乎没有更新

![[Pasted image 20230523143128.png|500]]

### Sample Index

![[Pasted image 20230523142549.png|500]]

![[Pasted image 20230523142511.png|500]]

![[Pasted image 20230523142529.png|500]]

![[Pasted image 20230523142753.png|500]]

### RMSE vs SNR

![[Pasted image 20230523233534.png]]


# 0605

修改训练数据集：20000组随机位置二声源+随机\[-10，10\]SNR

DCNN on same train dataset: SAME as before

![[Pasted image 20230606093709.png|500]]

结果对比：

| Name                                                                       | Layers | Lr     | Weight Decay | Result       | OverFitting |
| -------------------------------------------------------------------------- | ------ | ------ | ------------ | ------------ | ----------- |
| BASELINE                                                                   | 5      | 0.0004 | 0            | 0.40(不稳定) |             |
| <span style="background:rgba(205, 244, 105, 0.55)">WeightDecay</span>      | 5      | 0.0004 | 0.0001       | 0.38~        | 无          |
| (<span style="background:rgba(255, 183, 139, 0.55)">InputNorm</span>)      | 5      | 0.0004 | 0.0001       | 0.5          | 100ep       |
| (Binary)Debug3                                                             | 5      | 0.0004 | 0.0001       | 4(不稳定)    | 无          |
| <span style="background:rgba(205, 244, 105, 0.55)">Denoise</span>          | 5      | 0.0004 | 0            | 0.33         | 150         |
| Denoise+WeightDecay                                                        | 5      | 0.0004 | 0.0001       |              |             |
| Tied                                                                       | 5      | 0.0004 | 0            | 全0          |             |
| <span style="background:rgba(255, 183, 139, 0.55)">Tied</span>+WeightDecay | 5      | 0.0004 | 0.0001       | 随机         |             |


## 01

### 1000ep 

![[Pasted image 20230605230143.png|500]]

![[Pasted image 20230605230252.png|500]]

### 100epc

