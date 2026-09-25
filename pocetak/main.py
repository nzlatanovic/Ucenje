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


"""A = torch.tensor([[1.,2.,3.],
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

describe_tensor(C)"""



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


"""x = torch.tensor(4.0, requires_grad=True, device=device)

y = 3 * x ** 2 + 2*x + 5

#print(x)
#print(y)

y.backward() #zapocinje izracunavanja izvoda ali se ta vrendnost ne upisuje u y!!!

print(x.grad) #vrednost se uvek cuva u ime_promenljive.grad (ime promenljive se odnosi na promenljivu po kojoj radimo izvod)

#describe_tensor(x)


x = torch.tensor(2.0, requires_grad=True)
y = torch.tensor(3.0, requires_grad=True)

z = x**2 + 3*x*y + y**2

z.backward()

print(x.grad) # 2x + 3y = 2 * 2 + 3 * 3 = 13
print(y.grad) # 3x + 2y = 3 * 2 + 2 * 3 = 12


x = torch.tensor(2.0, requires_grad=True)

y = 3*x + 1

loss = (y - 10)**2

loss.backward()

print(y)
print(loss)
print(x.grad)

y = model(x) - Prolaz napred (izračunaj rezultat).
loss = criterion(y, target) - Izračunaj grešku.
loss.backward() - Izračunaj gradijente (izvode).
optimizer.step() - Popravi model (promeni težine).
"""

"""x = torch.tensor(20.0, requires_grad=True, device=device)

learning_rate=0.25

for step in range(30):
    loss = (x+3) ** 2

    loss.backward()

    with torch.no_grad(): #ovo sluzi da bi menjali vrednost promenljive ali ne i izvod
        x -= learning_rate * x.grad # ovo ce biti u prvom krugu: 0.0 - 0.1 * 2 * (0.0 - 5.0) = 0 - 1 * (-10) = 0 -(-1) = 1

    x.grad.zero_() # ovo sluzi za brisanje starih gradijenata (jer PyTorch dodaje nove gradijente na stare, a ako to dozvolimo program ce "poludeti")

    print(step, x.item(), loss.item()) # klasican ispis, ".item" sluzi da bi broj lepse izgledao tjt (izvlacimo iz tensora obican Python broj)"""


#w = torch.tensor(0.0, requires_grad=True, device=device)
#b = torch.tensor(0.0, requires_grad=True, device=device)

"""x = torch.tensor(2.0)

target = torch.tensor(10.0)




learning_rate = 0.01

for step in range(100):


    prediction = w * x + b

    loss = (prediction - target) ** 2

    loss.backward()

    with torch.no_grad():
        w-=learning_rate*w.grad
        b-=learning_rate*b.grad
    
    w.grad.zero_()
    b.grad.zero_()

    if step % 10 == 0:
        print(step, prediction.item(), loss.item())"""



"""x = torch.tensor([1., 2., 3., 4., 5.], device=device)
#y = torch.tensor([5., 8., 11., 14., 17.], device=device)
# Treba dobiti ovo: y= 3*x + 2
learning_rate = 0.04

y = torch.tensor([2., 7., 12., 17., 22.], device=device) # vrednosti y menjamo u odnosu na ono sta nam se trazi (i primenjujemo sa x)
# sada treba dobiti y = 5*x - 3

print(w)
print(b)

for step in range(1000):

    prediction = w * x + b #ovo je za pravu

        #kako se formula za prediction menja u zavisnosti od zadatka:
        #• Ako podaci prate pravu liniju, formula je:
        #prediction = w * x + b
        #• Ako podaci prate parabolu (kvadratnu funkciju), formula postaje:
        #prediction = a * (x**2) + b * x + c
        #• Ako imaš dva različita ulaza (npr. kvadratura x1 i broj soba x2), formula je:
        #prediction = w1 * x1 + w2 * x2 + b
        #Dakle, ti pišeš formulu koja odgovara "obliku" tvog problema.



    loss=((prediction-y)**2).mean()

    loss.backward()

    with torch.no_grad():
        w-=learning_rate*w.grad
        b-=learning_rate*b.grad

    w.grad.zero_() #ovo zero sluzi da resetuje gradijente (obavezno resetovati gradijente pre nego sto se ode dalje!)
    b.grad.zero_()
    if step % 10 == 0:
        print(step, prediction.data, loss.item())

print(w)
print(b)

x_novi = torch.tensor(10.0, device=device)
nova_predikcija = w * x_novi + b
print(nova_predikcija)"""





