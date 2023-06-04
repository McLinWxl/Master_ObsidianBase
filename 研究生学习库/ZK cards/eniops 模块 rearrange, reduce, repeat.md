202306011224

Status: #Python 

Tags: 

# eniops 模块

```python
from einops import rearrange, reduce, repeat

import torch

A = torch.randn(2, 3, 4, 5)

print(f"Oringinal shape: {A.shape}")

# Oringinal shape: torch.Size([2, 3, 4, 5])

B = rearrange(A, 'b c h w -> b c (h w)')

print(f"Rearranged shape: {B.shape}")

# Rearranged shape: torch.Size([2, 3, 20])

C = rearrange(A, 'b c h w -> (b c) (h w)')

print(f"Rearranged shape: {C.shape}")

# Rearranged shape: torch.Size([6, 20])

  

D = reduce(A, 'b c h w -> b c', 'mean') # or 'sum', 'max', 'min'

print(f"Reduced shape: {D.shape}")

# Reduced shape: torch.Size([2, 3])

print(D)

  

E = repeat(D, 'b c -> b c h w', h=4, w=5)

print(f"Repeated shape: {E.shape}")

# Repeated shape: torch.Size([2, 3, 4, 5])

  

F = repeat(D, 'b c -> (h b) (w c)', h=4, w=5)

print(f"Repeated shape: {F.shape}")

# Repeated shape: torch.Size([8, 15])
```




---
# Reference