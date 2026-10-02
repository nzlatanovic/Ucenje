import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

def accuracy_fn(logits, y):
    probabilities=torch.sigmoid(logits)
    predictions=(probabilities>=0.5).float()
    accuracy=(predictions==y).float().mean()
    return accuracy.item()*100


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


# Dan 07: OVDE JE KOD ZA PROVERU PODATAKA TJ VERODOSTOJNOSTI ISTIH (Konkretno kruznica)!??!!?
"""x = torch.tensor([
    [-3.0,  2.0],  
    [-2.0,  4.0],  
    [-1.0, -0.5],  
    [ 1.0,  1.0],  
    [ 2.0, -3.0],  
    [ 3.0,  0.5]   
])

y = torch.tensor([
    [0.0],
    [1.0],
    [0.0],
    [1.0],
    [0.0],
    [1.0]
])

# 1. Generišemo 40 primera, gde svaki ima 2 kolone (x1 i x2)
# Množenjem i oduzimanjem dobijamo opseg brojeva od -2.0 do 2.0
x = torch.rand(40, 2) * 4 - 2

# 2. Automatski računamo tačne odgovore (y) na osnovu pravila x1 + x2 > 0
# x[:, 0] je prva kolona (x1), a x[:, 1] je druga kolona (x2)
y = (x[:, 0] + x[:, 1] > 0).float().unsqueeze(1)


# Generišemo 100 nasumičnih tačaka radi bolje preciznosti (2 kolone: x1 i x2)
# Opseg od -1.5 do 1.5 da bi krug lepo upao u centar
x = torch.rand(100, 2) * 3 - 1.5

# Izvlačimo x1 i x2
x1 = x[:, 0]
x2 = x[:, 1]

# USLOV ZA KRUG: Klasa 1 je ako je kvadrat udaljenosti manji od 1
uslov = (x1**2 + x2**2) < 1

# Pretvaramo u uspravnu kolonu (oblik)
y = uslov.float().unsqueeze(1)



model = nn.Sequential(
    nn.Linear(2,10),
    nn.ReLU(),
    nn.Linear(10,1)
)



model = model.to(device)
x = x.to(device)
y = y.to(device)


print(device)
print(next(model.parameters()).device)

output = model(x)
print(output)

sigmoid = nn.Sigmoid() #veliki negativan broj ce biti blizu 0, 0 ce biti 0.5, a veliki pozitivan broj ce biti blizu 1

test = torch.tensor([-5.0, -2.0, 0.0, 2.0, 5.0])

print(sigmoid(test))

loss_fn = nn.BCEWithLogitsLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.01
)

for step in range(2001):
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



with torch.no_grad():
    logits = model(x)
    probabilities = torch.sigmoid(logits)

#       Kada model radi binarnu klasifikaciju, na kraju izbaci sirov broj (logits). 
#       Da bi taj broj pretvorio u procente koje razumeš, propustiš ga kroz torch.sigmoid().
#       Sve to staviš unutar with torch.no_grad(): da bi kompjuter radio brže.

predictions = (probabilities >= 0.5).float()

print(probabilities)
print(predictions)
print(y)

accuracy = (predictions == y).float().mean()

print(accuracy.item())"""

# Dan 08: OVDE SMO RADILI SA BATCH-EVIMA (oni dele sa 32 i spori su za malo podataka ali su vrh za ogroman broj podataka)
"""torch.manual_seed(42)

x=torch.randn(1000, 2)

y=((x[:, 0]**2 + x[:, 1]**2)<1).float().unsqueeze(1) #unsqueeze od niza pravi matricu dimenzija [x,1]

x=x.to(device)
y=y.to(device)

print(x.shape)
print(y.shape)
print(y[:10])

#describe_tensor(y)

train_size = 800

x_train = x[:train_size]
y_train = y[:train_size]

x_test = x[train_size:]
y_test = y[train_size:]


x_train = x_train.to(device)
y_train = y_train.to(device)

x_test = x_test.to(device)
y_test = y_test.to(device)



model = nn.Sequential(
    nn.Linear(2,16),
    nn.ReLU(),
    nn.Linear(16,16),
    nn.ReLU(),
    nn.Linear(16,1)
)
model = model.to(device)

loss_fn=nn.BCEWithLogitsLoss()

optimizer=torch.optim.Adam(
    model.parameters(),
    lr=0.01
)

for step in range (3001):
    prediction = model(x)

    loss = loss_fn(prediction, y)

    optimizer.zero_grad()

    loss.backward()

    optimizer.step()

    if step % 100 == 0:
        with torch.no_grad():
            accuracy = accuracy_fn(prediction, y)

            print(f"step {step}")
            print(f"train loss: {loss.item():.4f}")
            print(f"test accuracy: {accuracy:.1f}%")
            print()



train_dataset = TensorDataset(x_train, y_train)

train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True
)


for step in range (3001):
    for batch_x, batch_y in train_loader:

        prediction = model(batch_x)

        loss = loss_fn(prediction, batch_y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    if step % 100 == 0:
        with torch.no_grad():
            accuracy = accuracy_fn(prediction, batch_y)

            print(f"step {step}")
            print(f"train loss: {loss.item():.4f}")
            print(f"test accuracy: {accuracy:.1f}%")
            print()"""











