# example.py

from tensor import Tensor

def main():
    # Create tensors with requires_grad=True to enable gradient computation
    x = Tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    y = Tensor([[5.0, 6.0], [7.0, 8.0]], requires_grad=True)

    # Perform operations using all functions
    # Step 1: Compute the dot product of x and y
    z = x.dot(y)  # z = x @ y (matrix multiplication)

    # Step 2: Apply ReLU activation
    z_relu = z.relu()  # z_relu = max(z, 0)

    # Step 3: Sum all elements of the ReLU output
    s = z_relu.sum()  # s = sum of all elements in z_relu

    # Step 4: Compute LogSoftMax
    log_softmax = z_relu.logsoftmax()  # log_softmax = LogSoftMax of z_relu

    # Step 5: Compute the maximum value from log_softmax
    max_val = log_softmax.max()  # max_val = max of log_softmax

    # Backward pass to compute gradients
    max_val.backward()

    # Print results
    print(f"x: {x}")
    print(f"y: {y}")
    print(f"z (x.dot(y)): {z}")
    print(f"z_relu (ReLU of z): {z_relu}")
    print(f"s (sum of z_relu): {s}")
    print(f"log_softmax (LogSoftMax of z_relu): {log_softmax}")
    print(f"max_val (max of log_softmax): {max_val}")

    print(f"x.grad: {x.grad}")  # Gradient of x
    print(f"y.grad: {y.grad}")  # Gradient of y

if __name__ == "__main__":
    main()
