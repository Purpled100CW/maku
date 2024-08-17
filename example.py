from tensor import Tensor
from optim import SGD
import numpy as np
class SimpleModel:
    def __init__(self):
        self.weight = Tensor(np.random.randn(1), requires_grad=True)  
        self.bias = Tensor(0.0, requires_grad=True)  

    def forward(self, x):
        return x.mul(self.weight.add(self.bias))  

    def parameters(self):
        return [self.weight, self.bias]


def mse_loss(pred, target):
    return pred.sub(target).pow(Tensor(2)).mean()

def test_train():
    model = SimpleModel()

    x = Tensor([1.0, 2.0, 3.0])
    target = Tensor([2.0, 4.0, 6.0])

    epochs = 100
    learning_rate = 0.01
    optimizer = SGD(model.parameters(), lr=learning_rate)

    for epoch in range(epochs):
        pred = model.forward(x)

        loss = mse_loss(pred, target)
        
        for param in model.parameters():
            param.grad = None

        loss.backward()

        optimizer.step()

        if epoch % 10 == 0:
            print(f"Epoch [{epoch}/{epochs}], Loss: {loss.data}")

    print("Training complete.")
    print(f"Final weight: {model.weight.data}")
    print(f"Final bias: {model.bias.data}")

if __name__ == "__main__":
    test_train()
