import torch

# Tensor Indexing refers to the process of accessing or modifying specific elements, slices, or subsets of a tensor.
# Tensors are multi-dimensional arrays (similar to NumPy arrays), and indexing allows you to work with specific parts of
# the tensor efficiently. PyTorch, TensorFlow, and other deep learning frameworks support tensor indexing, which is
# essential for data manipulation and model operations.

batch_size = 10
features = 25
x = torch.rand((batch_size, features))

print(x)
print(x[0])
print(x[0].shape)       # x[0,;]

print(x[:,0].shape)

print(x[2, 0:10])       # 0:10 --> [0, 1, 2, ..., 9]

x[0, 0] = 100

# Fancy indexing
x = torch.arange(10)     # [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
print(x)
indices = [2, 5, 8]
print(x[indices])       # only going to pick out the 3rd, 6th, 9th example in our batch

x = torch.rand((3,5))
print(x)
# tensor([[0.5080, 0.8332, 0.7540, 0.8519, 0.0304],
#         [0.5150, 0.6296, 0.3049, 0.0070, 0.1073],
#         [0.2487, 0.7797, 0.5494, 0.5811, 0.4695]])
rows = torch.tensor([1, 0])
cols = torch.tensor([4, 0])
print(x[rows, cols])        # print element of 2nd row, 5th column and 1st row, 1st column
print(x[rows, cols].shape)

# More advanced indexing
x = torch.arange(10)
print(x[(x < 2) | (x > 8)])
print(x[x.remainder(2) == 0])       # if element mod 2 == 0 print element in x

# Useful operations
print(torch.where(x > 5, x, x*2))      # if condition satisfies, process x in argument 2 (remains unchanged in this case)
                                       # if not, process x in argument 3 (multiply x by 2 in this case)
print(torch.tensor([0,0,1,2,2,3,4]).unique())      # only print out values for one time (duplicate values are ignored)
print(x.ndimension())                   # print the number of dimensions of the tensor
print(x.numel())                        # count the number of elements in x

