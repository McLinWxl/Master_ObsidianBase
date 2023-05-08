202305082252

Status: #konwledge 

Tags: #PyFuns 

# lambda 匿名函数

## 语法

```python
lambda args: expression
```

## 实例

```python
my_list = [1, 2, 3, 4, 5, 6]
new_list = list(map(lambda x: x*2, my_list))
print(new_list) # [2, 4, 6, 8, 10, 12]
```

```python
my_list = [18, -3, 5, 0, -1, 12]
new_list = list(filter(lambda x: x > 0, my_list))
print(new_list) # [18, 5, 12]
```

```python
def muliplyBy (n):
  return lambda x: x*n
  
double = multiplyBy(2)
triple = muliplyBy(3)
times10 = multiplyBy(10)

double(6)
> 12
triple(5)
> 15
times10(12)
> 120
```


# Reference

[# Python 中的 Lambda 函数——示例语法](https://www.freecodecamp.org/chinese/news/lambda-function-in-python-example-syntax/)