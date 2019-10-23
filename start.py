import torch
import numpy as np

# tensor = n-dimensional array
# 1: [42], 2: [[1,2,3], [4,5,6]] 3: [[[1], [2], [3]], [[4], [5], [6]]]
# akin to np.array(array)
torch.tensor([2, 3, 5], [1, 2, 9])

# create random tensor
# torch.rand(size)
# akin to np.random.rand(size)
torch.rand((2, 2))

# get the shape of a tensor
foo = torch.rand((3, 5))
foo.shape

# multiply matricies (dot product)
# akin to np.dot(a,b)
a = torch.rand((2, 2))
b = torch.rand((2, 2))
torch.matmul(a, b)

# multiply
# akin to np.multiply(a, b)
multiply_output = a * b

# zeros and ones
# akin to np.zeroes((2,2))
zero_tensor = torch.zeroes(2,2)
# tensor([[0, 0], [0, 0]])

# akin to np.ones((2,2))
one_tensor = torch.ones(2,2)
# tensor([[1, 1], [1, 1]])

# akin to np.identity((2,2))
eye_tensor = torch.eye(2)
# tensor([[1, 0], [0, 1]])

# converting torch to np
source = np.random.rand((2, 2))
converted = torch.from_numpy(source)
back_to_np = converted.numpy()
