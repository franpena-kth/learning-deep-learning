import time

import numpy
import torch


def tensor_initialization():

    data = [[1, 2], [3, 4]]
    x_data = torch.tensor(data)

    np_array = numpy.array(data)
    x_np = torch.from_numpy(np_array)

    print('data', data)
    print('torch data', x_data)
    print('numpy array', np_array)
    print('torch array', x_np)

    # From another tensor:
    x_ones = torch.ones_like(x_data)  # retains the properties of x_data
    print(f"Ones Tensor: \n {x_ones} \n")

    x_rand = torch.rand_like(x_data, dtype=torch.float)  # overrides the datatype of x_data
    print(f"Random Tensor: \n {x_rand} \n")

    # With random or constant values:
    shape = (2, 3,)
    rand_tensor = torch.rand(shape)
    ones_tensor = torch.ones(shape)
    zeros_tensor = torch.zeros(shape)

    print(f"Random Tensor: \n {rand_tensor} \n")
    print(f"Ones Tensor: \n {ones_tensor} \n")
    print(f"Zeros Tensor: \n {zeros_tensor}")


def tensor_attributes():
    tensor = torch.rand(3, 4)

    print(f"Shape of tensor: {tensor.shape}")
    print(f"Datatype of tensor: {tensor.dtype}")
    print(f"Device tensor is stored on: {tensor.device}")


def tensor_operations():
    tensor = torch.rand(3, 4)

    # We move our tensor to the GPU if available
    if torch.cuda.is_available():
        tensor = tensor.to('cuda')

    # Standard numpy-like indexing and slicing:
    tensor = torch.ones(4, 4)
    tensor[:, 1] = 0
    print(tensor)

    # Joining tensors:
    t1 = torch.cat([tensor, tensor, tensor], dim=1)
    print(t1)

    # Multiplying tensors:
    # This computes the element-wise product
    print(f"tensor.mul(tensor) \n {tensor.mul(tensor)} \n")
    # Alternative syntax
    print(f"tensor * tensor \n {tensor * tensor}")

    # This computes the matrix multiplication between two tensors
    print(f"tensor.matmul(tensor.T) \n {tensor.matmul(tensor.T)} \n")
    # Alternative syntax:
    print(f"tensor @ tensor.T \n {tensor @ tensor.T}")

    # In-place operations: Operations that have a _ suffix are in-place.
    print(tensor, "\n")
    tensor.add_(5)
    print(tensor)


def tensor_to_numpy_array():

    t = torch.ones(5)
    print(f"t: {t}")
    n = t.numpy()
    print(f"n: {n}")

    # A change in the tensor reflects in the NumPy array.
    t.add_(1)
    print(f"t: {t}")
    print(f"n: {n}")


def numpy_array_to_tensor():
    n = numpy.ones(5)
    t = torch.from_numpy(n)

    # Changes in the NumPy array reflects in the tensor.
    numpy.add(n, 1, out=n)
    print(f"t: {t}")
    print(f"n: {n}")


def main():
    tensor_initialization()
    tensor_attributes()
    tensor_operations()
    tensor_to_numpy_array()
    numpy_array_to_tensor()


start = time.time()
main()
end = time.time()
total_time = end - start
print("%s: Total time = %f seconds" % (time.strftime("%Y/%m/%d-%H:%M:%S"), total_time))
