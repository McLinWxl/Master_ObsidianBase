202304062143

Status: #idea

Tags: 

# DOA 估计算法展开属于哪一类问题？

根据优化目标，算法展开又可以分为[1]： 
1. Objective-Based：最小化训练损失函数 
2. Inverse Problems：从测量信号恢复真实信号 

## Objective-Based

如果把 DOA 估计问题归类于 Objective-Based 问题，那么就应该设计优化一个损失函数，损失函数可以设计为优化目标于声源真实标签之间的距离。

## Inverse Problems

如果把 DOA 估计问题归类于 Inverse Problems，优化则是能够恢复声源信号功率谱图。

>[!Question]
>但是这会引出一个问题：如何获得声源信号的功率谱图？

在 DOA 估计中，信号的时域模型为信号的复包络形式[2]，那么==如何获得其能量，或者说是功率==？

---
# Reference

1. [[005 Learning to Optimize： A Primer and A Benchmark#Algorithm unrolling]]
2. [[复包络]]