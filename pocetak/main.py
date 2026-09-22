import torch

device = torch.device("cuda")

"""
x = torch.tensor([
    [1.0, 2.0],
    [3.0, 4.0]
])

w = torch.tensor([
    [5.0],
    [6.0]
])


y = x @ w

print(y)
print(y.shape)

device = torch.device("cuda")

x = torch.tensor([1.0, 2.0, 3.0], device=device)

print(x)
print(x.device)

x = torch.tensor([1.0, 2.0, 3.0])

x = x.to("cuda")

print(x.device)"""

x = torch.tensor([[1.,2.,3.],
                  [4.,5.,6.],
                  [7.,8.,9.]], device=device)

"""print(x[1,:])
print(x[:,1])
print(x[2,1])"""

y = torch.tensor([[9.,8.,7.],
                 [6.,5.,4.],
                 [3.,2.,1.]], device=device)

print(x+y)
print(x*y)
print(x @ y)
"""print(x)
print(x.shape)
print(x.dtype)
print(x.device)
x.to("cuda")
y.to("cpu")"""
#print(torch.cuda.is_available())