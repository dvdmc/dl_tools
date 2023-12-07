import torch


# a = torch.rand(5, 2, 3)
# print(a)
# t = torch.tensor([2, 1])
# b = torch.arange(a.shape[1]).type_as(t)

# # shape [2, 3, 1]
# result = a[:, b, t]

a = torch.tensor([[1, 2, 3], [4, 5, 6]])
print(torch.sum(a, dim=0))
