202305082233

Status: #konwledge

Tags: #PyFuns 

# map() 函数

---

## 语法


```python
map(function, iterable, ...)
```

Python2.x 中返回列表
Python3.x 中返回迭代器

```python
>>> def square(x) :         # 计算平方数  
...     return x ** 2  
>>> map(square, [1,2,3,4,5])    # 计算列表各个元素的平方  
<map object at 0x100d3d550>     # 返回迭代器  
>>> list(map(square, [1,2,3,4,5]))   # 使用 list() 转换为列表  
[1, 4, 9, 16, 25]  
>>> list(map(lambda x: x ** 2, [1, 2, 3, 4, 5]))   # 使用 lambda 匿名函数  
[1, 4, 9, 16, 25]  
```



---
# Reference

[Python map() 函数](https://www.runoob.com/python/python-func-map.html)
