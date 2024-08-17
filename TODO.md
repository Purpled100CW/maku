# MaKu

For something easier than pytorch and smaller than tensorflow

# Tensor is just wrapper of numpy.ndarray.And can do simple operations

# Example 
```py
# example.py

from tensor import Tensor
import numpy as np

x = Tensor([1, 2, 3], requires_grad=True)
y = Tensor([4, 5, 6], requires_grad=True)


z = x.mul(y)  
print(z)        # Tensor([ 4 10 18], requires_grad=True)
z.backward()
print(x.grad)   # [4 5 6]
print(y.grad)   # [1 2 3]

```
15:21