import os
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import itertools
from sklearn.metrics import f1_score

print("📥 Wczytywanie danych...")
df = pd.read_csv('archive/Folds.csv')
print(f"✔️ Załadowano {len(df)} rekordów.")

# Ekstrakcja etykiet
def extract_label(path):
    return 0 if "benign" in path.lower() else 1

df['label'] = df['filename'].apply(extract_label)
df['filepath'] = df['filename'].apply(lambda x: os.path.join('archive', x))

# Podział
print("🔀 Podział na zbiór treningowy i walidacyjny...")
train_df, val_df = train_test_split(df, test_size=0.2, stratify=df['label'], random_state=42)
print(f"🔹 Trening: {len(train_df)} próbek, 🔸 Walidacja: {len(val_df)} próbek")

# Dataset
class HistologyDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.dataframe = dataframe
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        img_path = self.dataframe.iloc[idx]['filepath']
        label = self.dataframe.iloc[idx]['label']
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label

# Transformacje
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])

print("🗃️ Przygotowywanie datasetów...")
train_dataset = HistologyDataset(train_df, transform=transform)
val_dataset = HistologyDataset(val_df, transform=transform)
print("✔️ Dataset gotowy.")

# Model głębokiej sieci konwolucyjnej
class DeepGridCNN(nn.Module):
    def __init__(self, dropout=0.0):
        super(DeepGridCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)   # 128x128
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)  # 64x64
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)  # 32x32
        self.pool = nn.MaxPool2d(2, 2)

        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(64 * 16 * 16, 128)
        self.fc2 = nn.Linear(128, 2)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # 128 → 64
        x = self.pool(F.relu(self.conv2(x)))  # 64 → 32
        x = self.pool(F.relu(self.conv3(x)))  # 32 → 16
        x = x.view(-1, 64 * 16 * 16)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


# Funkcja eksperymentalna
def run_experiment(lr, dropout, batch_size):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DeepGridCNN(dropout=dropout).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    model.train()
    for epoch in range(1):  # możesz zwiększyć liczbę epok
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

    # Walidacja
    model.eval()
    correct, total = 0, 0
    y_true, y_pred = [], []

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())

    accuracy = 100 * correct / total
    micro_f1 = f1_score(y_true, y_pred, average='micro')
    return accuracy, micro_f1

# Grid search
lrs = [0.01, 0.001]
dropouts = [0.0, 0.3, 0.5]
batch_sizes = [16, 32]

results = []

print("\n🚀 Start Grid Search...\n")
for lr, dropout, batch_size in itertools.product(lrs, dropouts, batch_sizes):
    print(f"🔍 Test: lr={lr}, dropout={dropout}, batch_size={batch_size}")
    acc, f1 = run_experiment(lr, dropout, batch_size)
    print(f"➡️  Accuracy: {acc:.2f}%, Micro F1: {f1:.4f}\n")
    results.append({
        'lr': lr,
        'dropout': dropout,
        'batch_size': batch_size,
        'accuracy': acc,
        'micro_f1': f1
    })

# Zapis wyników
results_df = pd.DataFrame(results)
results_df.to_csv("grid_search_results.csv", index=False)
print("📁 Wyniki zapisane do pliku: grid_search_results.csv")
