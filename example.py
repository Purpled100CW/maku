# example.py

from tensor import Tensor

def main():
    x = Tensor([4.0, 5.0, 6.0], requires_grad=True)
    y = Tensor([1.0, 2.0, 3.0], requires_grad=True)

    z = x.dot(y)
    g = x.sub(y)

    z.backward()
    g.backward()
    # Print results
    print(f"x: {x}")
    print(f"y: {y}")
    print(f"z: {z}")
    print(f"g: {g}")
    print(f"x.grad: {x.grad}")  # Gradient of x
    print(f"y.grad: {y.grad}")  # Gradient of y

if __name__ == "__main__":
    main()
