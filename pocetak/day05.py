import torch
import torch.nn as nn

model = nn.Linear(2, 1) # 1 ulaz(prvi broj) i 1 izlaz(drugi broj) (y=wx+b) #### Posle sam promenio na 2 ulaza

"""print(model)
print(model.weight) # ovo je nase "w" u formuli y=wx+b
print(model.bias) # ovo je nase "b" u formuli y=wx+b

print("\n")
for name, param in model.named_parameters():
    print(name, param)


#x = torch.tensor([[2.0]])
prediction = model(x)

print(prediction)
"""


x = torch.tensor([
    [1.0, 2.0],  # Primer 1: x1=1, x2=2  -> y = 2(1) + 3(2) + 1 = 9
    [2.0, 1.0],  # Primer 2: x1=2, x2=1  -> y = 2(2) + 3(1) + 1 = 8
    [3.0, 4.0],  # Primer 3: x1=3, x2=4  -> y = 2(3) + 3(4) + 1 = 19
    [0.0, 0.0],  # Primer 4: x1=0, x2=0  -> y = 2(0) + 3(0) + 1 = 1
    [5.0, 5.0]   # Primer 5: x1=5, x2=5  -> y = 2(5) + 3(5) + 1 = 26
])

y = torch.tensor([
    [9.0],
    [8.0],
    [19.0],
    [1.0],
    [26.0]
])


loss_fn = nn.MSELoss()

optimizer = torch.optim.SGD(
    model.parameters(),
    lr=0.01
) # u sustini ovo radi isto kao i w-=learning_rate * w.grad 

for step in range(1000):
    # 1. FORWARD PASS: Model uzima X i računa predviđanje (koristi formulu x * w + b)
    prediction = model(x)

    # 2. RAČUNANJE GREŠKE: Poredimo predviđanje modela sa tačnim odgovorima (y)
    loss = loss_fn(prediction, y)

    # 3. RESET GRADIJENATA: Brišemo stare proračune grešaka da se ne bi sabirali sa novim
    optimizer.zero_grad()

    # 4. BACKWARD PASS: PyTorch računa koliko je svaki parametar (w i b) kriv za grešku (aka racuna izvode)
    loss.backward()

    # 5. KORAK OPTIMIZACIJE: Optimizator blago menja w i b da bi smanjio grešku u sledećem krugu
    optimizer.step()

    if step % 10 == 0:
        print(step, loss.item()) #ovo item znaci da pretvara u broj

print("weight:", model.weight.data)
print("bias:", model.bias.data)

print("\n")
test = torch.tensor([[10.0, 5.0]])

prediction = model(test)

print(prediction)