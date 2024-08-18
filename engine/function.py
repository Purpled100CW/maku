from dataclasses import dataclass
import numpy as np
from tensor import Tensor, Function

@dataclass
class Add(Function):
    def forward(self) -> Tensor:
        result_data = self.input_tensor.data + self.other.data
        result = Tensor(result_data, requires_grad=self.input_tensor.requires_grad or self.other.requires_grad)
        result._backward_fn = self
        return result

    def backward(self, gradient: np.ndarray):
        if self.input_tensor.requires_grad:
            self.input_tensor.grad += gradient
        if self.other.requires_grad:
            self.other.grad += gradient

@dataclass
class Multiply(Function):
    def forward(self) -> Tensor:
        result_data = self.input_tensor.data * self.other.data
        result = Tensor(result_data, requires_grad=self.input_tensor.requires_grad or self.other.requires_grad)
        result._backward_fn = self
        return result

    def backward(self, gradient: np.ndarray):
        if self.input_tensor.requires_grad:
            self.input_tensor.grad += gradient * self.other.data
        if self.other.requires_grad:
            self.other.grad += gradient * self.input_tensor.data
