import torch

# Tensor Math and Comparison Operations
x = torch.tensor([1, 2, 3])
y = torch.tensor([9, 8, 7])

# Addition
z1 = torch.empty(3)     # Method 1
torch.add(x, y, out=z1)

z2 = torch.add(x, y)    # Method 2
z = x + y               # Method 3

# Subtraction
z = x - y

# Division
z = torch.true_divide(x, y) # 1/9, 2/8, 3/7
print(z)

# Inplace operations
t = torch.zeros(3)
t.add_(x)
t += x  # t = t + x

# Exponentiation
z = x.pow(2)            # Method 1
z = x ** 2              # Method 2

# Simple comparison
z = x > 0               # [True, True, True]
z = x < 0
print(z)

# Matrix Multiplication
x1 = torch.rand((2,5))
x2 = torch.rand((5,3))
xy = torch.mm(x1, x2)   # 2x3   # Method 1
x3 = x1.mm(x2)                  # Method 2

# Matrix Exponentiation
matrix_exp = torch.rand(5,5)
print(matrix_exp.matrix_power(3))

# Element wise multiplication
z = x * y
print(z)            # [9, 16, 21]

# Dot product
z = torch.dot(x,y)
print(z)

# Batch Matrix Multiplication
batch = 32
n = 10
m = 20
p = 30

tensor1 = torch.rand([batch, n, m])
tensor2 = torch.rand([batch, m, p])
out_bmm = torch.bmm(tensor1, tensor2)           # (batch, b, p)
print(out_bmm)

# Example of Broadcasting
x1 = torch.rand((5,5))
x2 = torch.rand((1,5))

z = x1 - x2         # subtract a matrix by a vector; Method 1
z = x1 ** x2        # Method 2
print(z)

# Other useful tensor operations
x = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
sum_x = torch.sum(x, dim=0)
values, indices = torch.max(x, dim=0)   # x.max(dim=0); return maximum value (and its indices) in a tensor
values, indices = torch.min(x , dim=0)  # x.min(dim=0); return minimum value (and its indices) in a tensor
abs_x = torch.abs(x)                # take absolute value element wise
z = torch.argmax(x, dim=0)          # same as .max() function, but only returns value (no indices)
z = torch.argmin(x, dim=0)
mean_x = torch.mean(x.float(), dim=0)
z = torch.eq(x, y)      # check which elements are equal, return True if equal
sorted_y, indices = torch.sort(y, dim=0, descending=False)  # sort values; ascending order in default; dim = dimension to be sorted

z = torch.clamp(x, min=0, max=10)       # if value < 0, set that value to 0; value > 10, set that value to 10

x = torch.tensor([1,0,1,1,1], dtype=torch.bool)
z = torch.any(x)                        # check if any values are true (1); True
z = torch.all(x)                        # check if all values are true (1); False

mean_y = torch.tensor([1.0, 2.0, 3.0], requires_grad=True).grad
print(mean_y)