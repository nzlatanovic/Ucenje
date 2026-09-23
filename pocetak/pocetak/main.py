import torch

device = torch.device("cuda")

def describe_tensor(x):
    print(f"--- Tensor Pregled ---")
    print(f"Vrednosti:\n{x}")
    print(f"Dimenzije: {x.shape}")
    print(f"Tip podataka: {x.dtype}")
    print(f"Gde se nalazi: {x.device}")
    print(f"Broj dimenzija: {x.ndim}")
    print(f"Broj elemenata: {x.numel()}")
    print(f"Prosek: {x.mean()}")
    print(f"Najmanji element: {x.min()}")
    print(f"Najveci element: {x.max()}")
    print(f"----------------------\n")




"""
x = torch.tensor([
    [1.0, 2.0],
    [3.0, 4.0]
])

w = torch.tensor([
    [5.0],
    [6.0]
])


y = x @ w #Ovo je matricno mnozenje

print(y)
print(y.shape)

device = torch.device("cuda")

x = torch.tensor([1.0, 2.0, 3.0], device=device)

print(x)
print(x.device)

x = torch.tensor([1.0, 2.0, 3.0])

x = x.to("cuda")

print(x.device)"""


A = torch.tensor([[1.,2.,3.],
                  [4.,5.,6.]], device=device)



B = torch.tensor([[9.,8.],
                 [6.,5.],
                 [3.,2.]], device=device)

C = torch.tensor([[1.,2.],
                  [3.,4.]], device=device)
x = torch.tensor([10.,20.,30.], device=device)

D = torch.randn(32, 10, 64) #batch, sequence, embedding (ovo se koristi u Transformerima)
E = torch.randn(64, 128)
# A @ B - Ovo je matricno mnozenje 2x3 @ 3x2 = 2x2, takodje B @ A je 3x2 @ 2x3 = 3x3
# A.T ovo je transponovanje matrice i ono menja broj redova i kolona (vrednosti zamene mesta suprotno od glavne dijagonale tj 1.2 predje na 2.1 i vice versa)
F = D @ E 

describe_tensor(C)



"""
print(B.T+x)
print(A*B)
print(A@B)
print(x[:, 1])
#print(torch.mean(x[1:3]))
ravan_niz=x.flatten()
print(ravan_niz[3:7])
print(x[1,:])
print(x[:,1])
print(x[2,1])
print(x+y)
print(x*y)
print(x @ y)
print(x)
print(x.shape)
print(x.dtype)
print(x.device)
x.to("cuda")
y.to("cpu")"""
#print(torch.cuda.is_available())