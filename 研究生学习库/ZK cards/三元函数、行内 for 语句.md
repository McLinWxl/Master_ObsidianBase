202305082311

Status:  #konwledge 

Tags: #PyFuns 

# 三元函数、行内 if、for 语句
---
## 三元函数

```python
i = 5 if a > 7 else 0
```

等价于：

```python
if a > 7: i = 5 else: i = 0
```
---

## 行内 for

```python
mylist = [6,2,8,3,1]

newlist = [x**2 for x in mylist]
print(newlist)   #[36, 4, 64, 9, 1]
```

```python
li = [1, 2, -4, 5, -6, 2]
newli = list(map(lambda x: x**2, [x for x in li if x > 0]))
print(newli) #[1, 4, 25, 4]
```


---
# Reference