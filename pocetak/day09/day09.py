import torch
import torch.nn as nn
import os
import matplotlib.pyplot as plt

def accuracy_fn(logits, y):
    probabilities=torch.sigmoid(logits)
    predictions=(probabilities>=0.5).float()
    accuracy=(predictions==y).float().mean()
    return accuracy.item()*100


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(42)

x=torch.randn(1000, 2)

y=((x[:, 0]**2 + x[:, 1]**2)<1).float().unsqueeze(1) #unsqueeze od niza pravi matricu dimenzija [x,1]

x=x.to(device)
y=y.to(device)

print(x.shape)
print(y.shape)
print(y[:10])

train_size = 700

x_train = x[:train_size]
y_train = y[:train_size]

x_test = x[train_size:]
y_test = y[train_size:]


x_train = x_train.to(device)
y_train = y_train.to(device)

x_test = x_test.to(device)
y_test = y_test.to(device)



model = nn.Sequential(
    nn.Linear(2,32),
    nn.ReLU(),
    nn.Linear(32,32),
    nn.ReLU(),
    nn.Linear(32,1)
)
model = model.to(device)





loss_fn=nn.BCEWithLogitsLoss()

optimizer=torch.optim.Adam(
    model.parameters(),
    lr=0.01
)

# 1. Kreiramo liste za čuvanje istorije
epoch_count = []
train_loss_values = []
val_loss_values = []
train_acc_values = []
val_acc_values = []

for step in range (101):
    prediction = model(x)

    loss = loss_fn(prediction, y)

    optimizer.zero_grad()

    loss.backward()

    optimizer.step()

    if step % 5 == 0:
        model.eval()
        with torch.no_grad():
            train_logits = model(x_train)
            train_loss = loss_fn(train_logits, y_train).item()
            train_acc = accuracy_fn(train_logits, y_train)
            
            val_logits = model(x_test)
            val_loss = loss_fn(val_logits, y_test).item()
            val_acc = accuracy_fn(val_logits, y_test)

            epoch_count.append(step)
            train_loss_values.append(train_loss)
            val_loss_values.append(val_loss)
            train_acc_values.append(train_acc)
            val_acc_values.append(val_acc)


            print(f"Epoch {step}")
            print(f"train loss: {train_loss:.2f}")
            print(f"val loss: {val_loss:.2f}")
            print(f"train acc: {train_acc:.0f}%")
            print(f"val acc: {val_acc:.0f}%")
            print()

#import matplotlib.pyplot as plt

folder_name = "day09"
if not os.path.exists(folder_name):
    os.makedirs(folder_name)  # Kreira fasciklu 'day09' ako već ne postoji

# --- GRAFIK 1: LOSS CURVE ---
plt.figure(figsize=(6, 4))
plt.plot(epoch_count, train_loss_values, label="Train Loss", color="blue")
plt.plot(epoch_count, val_loss_values, label="Validation Loss", color="orange")
plt.title("Loss kroz epohe")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)
# Čuvanje direktno u day09 fasciklu
plt.savefig(os.path.join(folder_name, "loss_curve.png"), bbox_inches='tight')
plt.close()  # Zatvaramo sliku da oslobodimo memoriju

# --- GRAFIK 2: ACCURACY CURVE ---
plt.figure(figsize=(6, 4))
plt.plot(epoch_count, train_acc_values, label="Train Accuracy", color="blue")
plt.plot(epoch_count, val_acc_values, label="Validation Accuracy", color="orange")
plt.title("Accuracy kroz epohe")
plt.xlabel("Epoch")
plt.ylabel("Accuracy (%)")
plt.legend()
plt.grid(True)
# Čuvanje direktno u day09 fasciklu
plt.savefig(os.path.join(folder_name, "accuracy_curve.png"), bbox_inches='tight')
plt.close()

# --- AUTOMATSKO KREIRANJE README.md FAJLA ---
readme_path = os.path.join(folder_name, "README.md")
with open(readme_path, "w", encoding="utf-8") as f:
    f.write("# Dan 09: Rezultati Treninga\n\n")
    f.write("Uspešno je završeno treniranje modela kroz epohe.\n\n")
    f.write(f"### Krajnje Metrike (Epoch {epoch_count[-1]}):\n")
    f.write(f"- **Krajnji Train Loss:** {train_loss_values[-1]:.4f}\n")
    f.write(f"- **Krajnji Validation Loss:** {val_loss_values[-1]:.4f}\n")
    f.write(f"- **Krajnji Train Accuracy:** {train_acc_values[-1]:.1f}%\n")
    f.write(f"- **Krajnji Validation Accuracy:** {val_acc_values[-1]:.1f}%\n\n")
    f.write("### Grafikoni performansi:\n")
    f.write("Pogledajte generisane grafikone u ovoj fascikli:\n")
    f.write("- `loss_curve.png`\n")
    f.write("- `accuracy_curve.png`\n")

print(f"Struktura uspešno kreirana unutar fascikle '{folder_name}'!")