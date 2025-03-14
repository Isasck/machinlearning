import torch

# print(torch.__version__)

# Initializing tensor
device = "cuda" if torch.cuda.is_available() else "cpu"      # use "cuda" for better performance (gpu)
my_tensor = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.float32,
                         device=device, requires_grad=True)

print(my_tensor)
print(my_tensor.dtype)
print(my_tensor.device)
print(my_tensor.shape)
print(my_tensor.requires_grad)

# Other common initialization methods
x = torch.empty(size = (3,3))
a = torch.zeros((3,3))
b = torch.rand((3,3))
c = torch.ones((3,3))
d = torch.eye(5,5) # I, eye
e = torch.arange(start=0, end=5, step=1)
f = torch.linspace(start=0.1, end=1, steps=10)
g = torch.empty(size=(1,5)).normal_(mean=0, std=1)      # mean value = 0, standard deviation = 1
h = torch.empty(size=(1,5)).uniform_(0,1)
i = torch.diag(torch.ones(3))
print(x, a, b, c, d, e, f, g, h, i)

# How to initalize and convert tensors to otehr types (int, float, double)
tensor = torch.arange(4)
print(tensor.bool())    # boolean True/False
print(tensor.short())   # int16
print(tensor.long())    # int64 (Important)
print(tensor.half())    # float16
print(tensor.float())   # float32 (Important)
print(tensor.double())  # float64

# Array to Tensor conversion and vice-versa
import numpy as np
np_array = np.zeros((5,5))
tensor = torch.from_numpy(np_array)
np_array_back = tensor.numpy()

