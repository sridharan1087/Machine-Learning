import torch

x = torch.randn(1,5)
x2 = torch.tensor([1.0,2.0])
y = torch.eq(x2,1.0)
print(y)
mask = torch.tensor([False,True,True,True,False])
mask = torch.tensor([[0,1,1,1,0]],dtype=torch.bool)
print(x)
print(mask)
print(x.masked_fill_(mask,0))
print(x)

a = torch.empty(3,3).uniform_(0,1)
print(a)
print(torch.bernoulli(a).bool())