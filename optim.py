from tensor import Tensor

class SGD:
    def __init__(self, *tensors: Tensor, lr=0.01):
        self.tensors = tensors
        self.lr = lr

    def step(self):
        for t in self.tensors:
            t.data -= self.lr * t.grad 

    