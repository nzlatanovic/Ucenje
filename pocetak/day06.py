import torch
import torch.nn as nn

"""x = torch.tensor([
    [-3.0],
    [-2.0],
    [-1.0],
    [0.0],
    [1.0],
    [2.0],
    [3.0]
])

y = x ** 2

#print(x)
#print(y)

model = nn.Sequential(
    nn.Linear(1,10), # h = Wx + b
    nn.ReLU(),       # h'= max(0,h)
    nn.Linear(10,1)  # y = W2​ * h' + b2
) # y= W2​ * ReLU(W1​x+b1​) + b2​

#relu = nn.ReLU()

#test = torch.tensor([-3.0, -1.0, 0.0, 2.0, 5.0])

#print(relu(test))


loss_fn = nn.MSELoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.01
)

for step in range(5000):

    prediction = model(x)

    loss = loss_fn(prediction, y)

    optimizer.zero_grad()

    loss.backward()

    optimizer.step()

#    if step % 500 == 0:
#        print(step, loss.item())



test = torch.tensor([
    [-4.0],
    [-2.5],
    [0.5],
    [3.5]
])"""

#prediction = model(test)

#print(prediction)

#print(model)

x = torch.linspace(-3.14, 3.14, 100).reshape(-1, 1) # generise 100 podataka izmeedju -3.14 i 3.14 (tj izmedju -pi i +pi) (reshape menja strukturu "matrice" reshape(broj redova, broj kolona))
y = torch.sin(x)

model = nn.Sequential(
    nn.Linear(1,20), # h = Wx + b
    nn.ReLU(),       # h'= max(0,h)
    nn.Linear(20,1)  # y = W2​ * h' + b2
)

loss_fn = nn.MSELoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.01
)

for step in range(5001):

    prediction = model(x)

    loss = loss_fn(prediction, y)

    optimizer.zero_grad()

    loss.backward()

    optimizer.step()

    if step % 500 == 0:
        print(step, loss.item())

#neural network:
#niz matematičkih transformacija čiji se parametri uče pomoću gradient descent-a.


"""torch.manual_seed(42)

# 2. TVOJI ORIGINALNI PODACI (Samo 7 tačaka!)
x = torch.tensor([[-3.0], [-2.0], [-1.0], [0.0], [1.0], [2.0], [3.0]])
y = x ** 2

# 3. Model sa Tanh aktivacijom (mnogo stabilnija za male podatke)
model = nn.Sequential(
    nn.Linear(1, 10),
    nn.Tanh(),       # Promenjeno sa ReLU na Tanh
    nn.Linear(10, 1)
)

loss_fn = nn.MSELoss()
# Smanjen lr sa 0.01 na 0.005 radi preciznosti
optimizer = torch.optim.Adam(model.parameters(), lr=0.005)

# 4. Trening (5000 koraka)
for step in range(5000):
    prediction = model(x)
    loss = loss_fn(prediction, y)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# 5. TVOJ TEST
test = torch.tensor([
    [-4.0],
    [-2.5],
    [0.5],
    [3.5]
])

print("\n--- KONAČNI REZULTATI ---")
with torch.no_grad():
    prediction = model(test)

for i in range(len(test)):
    stvarno = test[i].item() ** 2
    predvidjeno = prediction[i].item()
    print(f"Za x = {test[i].item():5.1f} | Stvarno (x²): {stvarno:5.2f} | Model kaže: {predvidjeno:5.2f}")"""