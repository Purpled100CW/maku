import numpy as np
from engine.tensor import Tensor
from engine.function import Add, Multiply

# Create tensors with requires_grad=True
a = Tensor(np.array([2.0, 3.0]), requires_grad=True)
b = Tensor(np.array([4.0, 5.0]), requires_grad=True)

# Perform operations
c = a + b
d = c * Tensor(np.array([2.0, 2.0]))

# Perform backward pass
d.backward()

# Check gradients
print("Gradients of a:", a.grad)
print("Gradients of b:", b.grad)
