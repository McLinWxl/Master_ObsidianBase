202306061108

Status:   #knowledge 

Tags: [[网络基础]]

# OverFitting

## 出现原因

1. 训练集的数量级和模型的==复杂度不匹配==。训练集的数量级要小于模型的复杂度
2. 训练集和测试集特征==分布不一致==
3. 样本里的==噪音数据干扰过大==，大到模型过分记住了噪音特征，反而忽略了真实的输入输出间的关系
4. 权值学习迭代次数足够多(==Overtraining==)，拟合了训练数据中的噪声和训练样例中没有代表性的特征

## 解决方法

1. 增大训练数据集规模或减小网络复杂度
2. 减少训练数据集中的噪声
3. Early Stopping 策略
4. 参数正则化
	1. L2: Weight Decay
5. Dropout

---
# Reference

[过拟合（定义、出现的原因4种、解决方案7种）](https://blog.csdn.net/NIGHT_SILENT/article/details/80795640)