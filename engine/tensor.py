from typing import List
from dataclasses import dataclass, field
import numpy as np
from function import Add, Multiply
@dataclass
class Tensor:
    data: np.ndarray
    grad: np.ndarray = field(default=None)
    requires_grad: bool = field(default=False)
    _backward_fn: 'Function' = field(default=None, repr=False)

    def __post_init__(self):
        if self.requires_grad:
            self.grad = np.zeros_like(self.data)
    
    def backward(self, gradient: np.ndarray = None):
        if gradient is None:
            gradient = np.ones_like(self.data)
        if self._backward_fn is not None:
            self._backward_fn.backward(gradient)
            self._backward_fn.input_tensor.grad += gradient

    def __add__(self, other: 'Tensor') -> 'Tensor':
        return Add(self, other).forward()

    def __mul__(self, other: 'Tensor') -> 'Tensor':
        return Multiply(self, other).forward()

    def zero_grad(self):
        self.grad = np.zeros_like(self.data)
    
    @property
    def shape(self):
        return self.data.shape

@dataclass
class Function:
    input_tensor: Tensor

    def forward(self) -> Tensor:
        raise NotImplementedError

    def backward(self, gradient: np.ndarray):
        raise NotImplementedError
