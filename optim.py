from tensor import Tensor
from typing import List
import numpy as np
class Optimizer:
    _registry = {}  # Class-level dictionary for optimizer registry

    def __init__(self, tensors: List[Tensor], lr: float):
        self.tensors = tensors
        self.lr = lr

    @classmethod
    def register(cls, name: str, optim_cls: 'Optimizer'):
        """Registers an optimizer class under a given name."""
        cls._registry[name] = optim_cls

    @classmethod
    def get_optimizer(cls, name: str) -> 'Optimizer':
        """Retrieves an optimizer class by name."""
        if name not in cls._registry:
            raise ValueError(f"Optimizer '{name}' is not registered.")
        return cls._registry[name]

    def step(self):
        """Applies the optimizer update step on the initialized tensors."""
        raise NotImplementedError
    
def register(name: str, optim_cls: 'Optimizer'):
    """
    Registers an optimizer class under a given name.
    Adds a method to the Optimizer class to create and apply the optimizer.
    """
    def method(tensors: List[Tensor], lr: float):
        return optim_cls.apply(tensors, lr)
    
    setattr(Optimizer, name, method)

class SGD(Optimizer):
    def step(self):
        """Updates tensors using the SGD algorithm."""
        for t in self.tensors:
            if t.grad is not None:
                t.data -= self.lr * t.grad
                t.grad = np.zeros_like(t.grad)

register('sgd', SGD)