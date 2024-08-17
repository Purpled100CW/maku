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
        def method(*args, **kwargs):
            func_cls = Function.get_function(name)
            return func_cls.apply(self, *args, **kwargs)
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
    def apply(cls, *tensors, **kwargs):
        """Applies tensors to context."""
        ctx = Context()
        output = cls.forward(ctx, *tensors, **kwargs)
        if isinstance(output, Tensor) and output.requires_grad:
            def backward_fn(grad_output):
                cls.backward(ctx, grad_output)
            output._backward = lambda: backward_fn(output.grad)
            output._prev = tensors
            output._ctx = ctx
        return output

    @staticmethod
    def forward(ctx, *tensors, **kwargs):
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

class Add(Function):
    @staticmethod
    def forward(ctx, x, y):
        ctx.save_for_backward(x, y)
        return Tensor(x.data + y.data, requires_grad=x.requires_grad or y.requires_grad)
    
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
    
register('add', Add)

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

class ReLU(Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return Tensor(np.maximum(input.data, 0), requires_grad=input.requires_grad)
    
    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = grad_output * (input.data > 0)
        if input.requires_grad:
            input.grad = grad_input
        return grad_input

register('relu', ReLU)

class Sum(Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return Tensor([input.sum], requires_grad=input.requires_grad)
    
    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors

        return grad_output * np.ones_like(input)
    
register('sum', Sum)

class LogSoftMax(Function):
    @staticmethod
    def forward(ctx, input):
        
        def logsumexp(x):
            c = x.max(axis=1)
            return c+np.log(np.exp(x-c.reshape((-1, 1))).sum(axis=1))
        output = input - logsumexp(input).reshape((-1, 1))
        ctx.save_for_backward(output)
        return Tensor(output, requires_grad=output.requires_grad)
    @staticmethod
    def backward(ctx, grad_output):
        output, = ctx.saved_tensors

        return grad_output - np.exp(output)*grad_output.sum(axis=1).reshape((-1, 1))
    
register('logsoftmax', LogSoftMax)

class Max(Function):
    @staticmethod
    def forward(ctx, x, axis=None):
        ctx.save_for_backward(x)
        if axis is not None:
            return Tensor(np.max(x.data, axis=axis), requires_grad=x.requires_grad)
        else:
            return Tensor(np.max(x.data), requires_grad=x.requires_grad)

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        # Create a boolean mask where input is equal to the maximum value
        mask = (input.data == np.max(input.data))
        grad_input = grad_output * mask
        if input.requires_grad:
            input.grad = grad_input
        return grad_input
register('max', Max)
