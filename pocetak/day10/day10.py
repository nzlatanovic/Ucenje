import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

torch.manual_seed(42) #generise random brojeve (ali uvek iste random brojeve)

x = torch.randn(1200, 2) # generise matricu od 1200 redova i 2 kolone
y = (
    (x[:, 0] ** 2 + x[:, 1] ** 2) < 1
).float().unsqueeze(1) #float pretvara true u 1 i false u 0, dok unsqueeze pretvara red u kolonu

print("Device:", device)
print("X shape:", x.shape)
print("Y shape:", y.shape)


# 2. Kreiranje nasumično izmešanih indeksa
# torch.randperm(1200) generiše niz brojeva od 0 do 1199 u nasumičnom redosledu
shuffled_indices = torch.randperm(1200)

# 3. Podela izmešanih indeksa na osnovu zadatih dimenzija
# Training: prvih 840 indeksa
train_indices = shuffled_indices[:840]

# Validation: sledećih 180 indeksa (od 840 do 840 + 180 = 1020)
val_indices = shuffled_indices[840:1020]

# Test: poslednjih 180 indeksa (od 1020 do kraja)
test_indices = shuffled_indices[1020:]

# 4. Uzimanje podataka preko dobijenih indeksa i direktno slanje na device
x_train, y_train = x[train_indices].to(device), y[train_indices].to(device)
x_val, y_val = x[val_indices].to(device), y[val_indices].to(device)
x_test, y_test = x[test_indices].to(device), y[test_indices].to(device)


model = nn.Sequential(
    nn.Linear(2, 32),   # Prvi sloj: 2 ulaza -> 32 skrivena neurona
    nn.ReLU(),          # Aktivaciona funkcija
    nn.Linear(32, 16),  # Drugi sloj: 32 -> 16 skrivenih neurona
    nn.ReLU(),          # Aktivaciona funkcija
    nn.Linear(16, 1)    # Izlazni sloj: 16 -> 1 izlaz (logits)
)

# Slanje modela na uređaj (GPU) koji smo ranije definisali
model = model.to(device)

# 2. Definisanje funkcije gubitka (Loss) i optimizatora
# Koristimo BCEWithLogitsLoss jer na izlazu modela nemamo Sigmoid (model vraća "logits")
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# 3. Kreiranje Dataset-a i DataLoader-a za trening podatke
# Spajamo x_train i y_train u jedan objekat
train_dataset = TensorDataset(x_train, y_train)

# Pravimo DataLoader koji će automatski deliti podatke na batch-eve i mešati ih u svakoj epohi
train_loader = DataLoader(
    dataset=train_dataset, 
    batch_size=32, 
    shuffle=True
)






def calculate_accuracy(predictions, targets):
    # BCEWithLogitsLoss daje sirove izlaze (logits). 
    # Ako je logits >= 0, to znači da je verovatnoća >= 50% (klasa 1), inače je klasa 0.
    predicted_classes = (predictions >= 0).float()
    correct = (predicted_classes == targets).sum().item()
    return correct / targets.size(0)

# Glavna petlja - 100 epoha
num_epochs = 100

for epoch in range(num_epochs):
    # ================= KORAK 1: TRENING =================
    model.train()  # Aktivacija trening režima
    
    running_train_loss = 0.0
    
    for batch_x, batch_y in train_loader:
        # 1. Model računa predikcije
        outputs = model(batch_x)
        
        # 2. Računanje loss-a za trenutni batch
        loss = criterion(outputs, batch_y)
        
        # 3. Resetovanje prethodnih gradijenata
        optimizer.zero_grad()
        
        # 4. Backward pass (računanje novih gradijenata)
        loss.backward()
        
        # 5. Ažuriranje parametara modela
        optimizer.step()
        
        # Pratimo ukupni loss u ovoj epohi (množimo sa veličinom batch-a jer loss vraća prosek za batch)
        running_train_loss += loss.item() * batch_x.size(0)
        
    # Računamo prosečan loss za celu epohu (delimo sa ukupnim brojem primera u trening setu)
    epoch_train_loss = running_train_loss / len(train_loader.dataset)
    
    # ================= KORAK 2: VALIDACIJA =================
    model.eval()  # Aktivacija evaluacionog režima (isključuje dropout, batchnorm itd.)
    
    with torch.no_grad():  # Isključujemo računanje gradijenata radi uštede memorije i brzine
        # Pošto validacioni set nije u DataLoader-u, računamo odjednom za ceo x_val i y_val
        val_outputs = model(x_val)
        
        # Računanje validation loss-a i accuracy-ja
        epoch_val_loss = criterion(val_outputs, y_val).item()
        epoch_val_acc = calculate_accuracy(val_outputs, y_val)
        
    # Ispisujemo rezultate svake 10. epohe da ne zatrpavamo ekran, plus poslednju 100. epohu
    if (epoch + 1) % 10 == 0 or epoch == 0:
        print(f"Epoch [{epoch+1}/{num_epochs}] | "
              f"Train Loss: {epoch_train_loss:.4f} | "
              f"Val Loss: {epoch_val_loss:.4f} | "
              f"Val Acc: {epoch_val_acc * 100:.2f}%")









# Liste u koje ćemo čuvati rezultate kroz epohe
train_losses = []
val_losses = []
val_accuracies = []
# Dodajemo i praćenje trening accuracy-ja radi kompletnog grafikona
train_accuracies = [] 

num_epochs = 100

for epoch in range(num_epochs):
    # ================= TRENING =================
    model.train()
    running_train_loss = 0.0
    running_train_correct = 0
    
    for batch_x, batch_y in train_loader:
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        running_train_loss += loss.item() * batch_x.size(0)
        
        # Računamo tačne primere u trenutnom batch-u za trening accuracy
        predicted_classes = (outputs >= 0).float()
        running_train_correct += (predicted_classes == batch_y).sum().item()
        
    # Prosečne vrednosti za trening epohu
    epoch_train_loss = running_train_loss / len(train_loader.dataset)
    epoch_train_acc = running_train_correct / len(train_loader.dataset)
    
    # ================= VALIDACIJA =================
    model.eval()
    with torch.no_grad():
        val_outputs = model(x_val)
        epoch_val_loss = criterion(val_outputs, y_val).item()
        epoch_val_acc = calculate_accuracy(val_outputs, y_val)
        
    # Čuvanje vrednosti u liste za grafikon
    train_losses.append(epoch_train_loss)
    val_losses.append(epoch_val_loss)
    train_accuracies.append(epoch_train_acc)
    val_accuracies.append(epoch_val_acc)




# Kreiramo figuru sa dva grafikona jedan pored drugog (1 red, 2 kolone)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# --- PRVI GRAFIKON: Kriva gubitka (Loss) ---
ax1.plot(range(1, num_epochs + 1), train_losses, label='Training Loss', color='blue', linewidth=2)
ax1.plot(range(1, num_epochs + 1), val_losses, label='Validation Loss', color='orange', linewidth=2, linestyle='--')
ax1.set_title('Training i Validation Loss po epohama')
ax1.set_xlabel('Epohe')
ax1.set_ylabel('Loss')
ax1.grid(True, linestyle=':', alpha=0.6)
ax1.legend()

# --- DRUGI GRAFIKON: Kriva tačnosti (Accuracy) ---
# Množimo sa 100 da bismo na Y osi imali procente (0-100%)
ax2.plot(range(1, num_epochs + 1), [acc * 100 for acc in train_accuracies], label='Training Accuracy', color='green', linewidth=2)
ax2.plot(range(1, num_epochs + 1), [acc * 100 for acc in val_accuracies], label='Validation Accuracy', color='red', linewidth=2, linestyle='--')
ax2.set_title('Training i Validation Accuracy po epohama')
ax2.set_xlabel('Epohe')
ax2.set_ylabel('Accuracy (%)')
ax2.grid(True, linestyle=':', alpha=0.6)
ax2.legend()

# Automatsko podešavanje razmaka između grafikona i prikaz
plt.tight_layout()

# SNIMANJE GRAFIKONA U FOLDER:
# bbox_inches='tight' osigurava da ivice i tekst ne budu odsečeni na slici
plt.savefig('trening_rezultati.png', dpi=300, bbox_inches='tight')

# Prikaz na ekranu (uvek stavljaj OVO POSLE savefig, inače će slika biti prazna)
plt.show()




# 1. Prebacujemo model u evaluacioni režim
model.eval()

# 2. Isključujemo računanje gradijenata radi uštede memorije i brzine
with torch.no_grad():
    # Propuštamo ceo test set kroz model odjednom
    test_outputs = model(x_test)
    
    # Računamo konačan test loss
    final_test_loss = criterion(test_outputs, y_test).item()
    
    # Računamo konačan test accuracy koristeći našu pomoćnu funkciju
    final_test_acc = calculate_accuracy(test_outputs, y_test)

# 3. Ispisujemo konačne rezultate na ekranu
print("=" * 40)
print("       KONAČNI REZULTATI NA TEST SKUPU       ")
print("=" * 40)
print(f"Final Test Loss: {final_test_loss:.4f}")
print(f"Final Test Accuracy: {final_test_acc * 100:.2f}%")
print("=" * 40)