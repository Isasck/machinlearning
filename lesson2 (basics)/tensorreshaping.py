import torch

x = torch.arange(9)
print(x)

x_3x3 = x.view(3,3) # view only acts on continguous tensors
print(x_3x3)
x_3x3 = x.reshape(3,3)  # reshape works in all situation (at the expense of performance loss)
# tensor([[0, 1, 2],
#         [3, 4, 5],
#         [6, 7, 8]])

y = x_3x3.t()       # t = transpose
print(y)
# tensor([[0, 3, 6],
#         [1, 4, 7],
#         [2, 5, 8]])
print(y.contiguous().view(9))

x1 = torch.rand((2,5))
x2 = torch.rand((2,5))
print(x1, x2)
print(torch.cat((x1,x2), dim=0).shape)  # torch.Size([4, 5])
print(torch.cat((x1,x2), dim=1).shape)  # torch.Size([2, 10])
# cat = concatenate

z = x1.view(-1)     # flatten entire tensor, 2x5 --> 1x10
print(z.shape)  # torch.Size([10])

batch = 64
x = torch.rand((batch, 2, 5))   # change dimension of tensor (customized)
z = x.view(batch, -1)       # 64x2x5 --> 64x10 (keep batch dimension)
print(z.shape)      # torch.Size([64, 10])


z = x.permute(0,2,1)        # switch axis; 64x5x2 --> 64x2x5
                            # keep dimension 0 at 0, put dimension 2 (old) at 1 (new), put dimension 1 at 2
print(z.shape)
# torch.Size([64, 5, 2])

x = torch.arange(10)
print(x)
print(x.shape)
print(x.unsqueeze(0).shape)     # torch.Size([1, 10])
print(x.unsqueeze(0))
print(x.unsqueeze(1).shape)     # torch.Size([10, 1])
print(x.unsqueeze(1))

x = torch.arange(10).unsqueeze(0).unsqueeze(1) # 1x1x10

z = x.squeeze(1)    # remove 1 dimension in 1x1x10 so that it becomes 1x10
print(z.shape)      # torch.Size([1, 10])

