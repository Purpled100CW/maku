# tensor.py

import numpy as np

class Tensor:
    def __init__(self, data, requires_grad=False):
        self.data = np.array(data)
        self.requires_grad = requires_grad
        self.grad = None
        self._backward = lambda: None
        self._prev = set()
        self._ctx = None  # To store the context object from Function classes

    def backward(self):
        if self.requires_grad:
            self.grad = np.ones_like(self.data) if self.grad is None else self.grad

            topo_order = []
            visited = set()

            def build_topo(tensor):
                if tensor not in visited:
                    visited.add(tensor)
                    for prev in tensor._prev:
                        build_topo(prev)
                    topo_order.append(tensor)

            build_topo(self)
            for tensor in reversed(topo_order):
                tensor._backward()

    def __repr__(self):
        return f"Tensor({self.data}, requires_grad={self.requires_grad})"

    def __getattr__(self, name):
        """
        Dynamically resolve method names to registered functions.
        For example, calling `x.mul(y)` will resolve to `Mul.apply(x, y)`
        """
        def method(*args):
            func_cls = Function.get_function(name)
            return func_cls.apply(self, *args)
        return method

class Function:
    _registry = {}  # Class-level dictionary for function registry

    @classmethod
    def register(cls, name, func_cls):
        """Registers a function class under a given name."""
        cls._registry[name] = func_cls

    @classmethod
    def get_function(cls, name):
        """Retrieves a function class by name."""
        if name not in cls._registry:
            raise ValueError(f"Function '{name}' is not registered.")
        return cls._registry[name]

    @classmethod
    def apply(cls, *tensors):
        """Applies tensors to context."""
        ctx = Context()
        output = cls.forward(ctx, *tensors)
        if isinstance(output, Tensor) and output.requires_grad:
            def backward_fn(grad_output):
                cls.backward(ctx, grad_output)
            output._backward = lambda: backward_fn(output.grad)
            output._prev = tensors
            output._ctx = ctx
        return output

    @staticmethod
    def forward(ctx, *tensors):
        """Function to be overridden."""
        raise NotImplementedError

    @staticmethod
    def backward(ctx, grad_output):
        """Function to be overridden."""
        raise NotImplementedError

class Context:
    def __init__(self):
        self.saved_tensors = None

    def save_for_backward(self, *tensors):
        self.saved_tensors = tensors

    def __repr__(self):
        return f"Context(saved_tensors={self.saved_tensors})"

def register(name, func_cls):
    """
    Registers a function class and adds it as a method to the Tensor class.
    Uses setattr to dynamically add the method.
    """
    def method(self, *args):
        return func_cls.apply(self, *args)
    
    setattr(Tensor, name, method)

# Function subclasses

class Add(Function):
    @staticmethod
    def forward(ctx, x, y):
        ctx.save_for_backward(x, y)
        return Tensor(x.data + y.data, requires_grad=x.requires_grad or y.requires_grad)
    
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, grad_output
    
register('add', Add)

class Sub(Function):
    @staticmethod
    def forward(ctx, x, y):
        ctx.save_for_backward(x, y)
        return Tensor(x.data - y.data, requires_grad=x.requires_grad or y.requires_grad)
    
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, grad_output
    
register('sub', Sub)

class Mul(Function):
    @staticmethod
    def forward(ctx, x, y):
        ctx.save_for_backward(x, y)
        return Tensor(x.data * y.data, requires_grad=x.requires_grad or y.requires_grad)

    @staticmethod
    def backward(ctx, grad_output):
        x, y = ctx.saved_tensors
        grad_x = y.data * grad_output if x.requires_grad else None
        grad_y = x.data * grad_output if y.requires_grad else None
        if x.requires_grad:
            x.grad = grad_x
        if y.requires_grad:
            y.grad = grad_y
        return grad_x, grad_y

register('mul', Mul)

class Dot(Function):
    @staticmethod
    def forward(ctx, x, y):
        ctx.save_for_backward(x, y)
        return Tensor(np.dot(x.data, y.data), requires_grad=x.requires_grad or y.requires_grad)
    
    @staticmethod
    def backward(ctx, grad_output):
        x, y = ctx.saved_tensors
        grad_x = np.dot(grad_output, y.data.T) if x.requires_grad else None
        grad_y = np.dot(x.data.T, grad_output) if y.requires_grad else None
        if x.requires_grad:
            x.grad = grad_x
        if y.requires_grad:
            y.grad = grad_y
        return grad_x, grad_y

register('dot', Dot)

class Max(Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return Tensor(np.max(x.data, axis=0), requires_grad=x.requires_grad)
    
    @staticmethod
    def backward(ctx, grad_output):
        x = ctx.saved_tensors[0]
        grad = np.zeros_like(x.data)
        max_indices = np.argmax(x.data, axis=0)
        grad[max_indices] = grad_output
        if x.requires_grad:
            x.grad = grad
        return grad

register('max', Max)

class LogSoftMax(Function):
    @staticmethod
    def forward(ctx, x):
        max_val = np.max(x.data, axis=0)
        shifted_x = x.data - max_val
        log_sum_exp = np.log(np.sum(np.exp(shifted_x), axis=0))
        log_softmax = shifted_x - log_sum_exp
        ctx.save_for_backward(x, shifted_x, log_sum_exp)
        return Tensor(log_softmax, requires_grad=x.requires_grad)
    
    @staticmethod
    def backward(ctx, grad_output):
        x, shifted_x, log_sum_exp = ctx.saved_tensors
        exp_shifted_x = np.exp(shifted_x)
        sum_exp = np.sum(exp_shifted_x, axis=0)
        grad = exp_shifted_x / sum_exp - np.exp(log_sum_exp)
        grad = grad_output - grad * np.sum(grad_output, axis=0)
        if x.requires_grad:
            x.grad = grad
        return grad

register('logsoftmax', LogSoftMax)

class ReLU(Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return Tensor(np.maximum(x.data, 0), requires_grad=x.requires_grad)
    
    @staticmethod
    def backward(ctx, grad_output):
        x = ctx.saved_tensors[0]
        grad = grad_output * (x.data > 0).astype(float)
        if x.requires_grad:
            x.grad = grad
        return grad

register('relu', ReLU)

class Sum(Function):
    @staticmethod
    def forward(ctx, x, axis=None):
        ctx.save_for_backward(x)
        return Tensor(np.sum(x.data, axis=axis), requires_grad=x.requires_grad)
    
    @staticmethod
    def backward(ctx, grad_output):
        x = ctx.saved_tensors[0]
        if x.requires_grad:
            grad = np.ones_like(x.data) * grad_output
            if x.data.ndim > 0:
                grad = np.reshape(grad, x.data.shape)
            x.grad = grad
        return grad

register('sum', Sum)
