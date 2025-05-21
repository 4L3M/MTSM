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

import torch.nn.utils.prune as prune

import time
from sklearn.metrics import classification_report, confusion_matrix, precision_score, recall_score, f1_score


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

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
print("✔️ Loadery gotowe.")

# Model
class DeepPrunedCNN(nn.Module):
    def __init__(self):
        super(DeepPrunedCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)

        self.fc1 = nn.Linear(64 * 16 * 16, 128)
        self.fc2 = nn.Linear(128, 2)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # 128x128 → 64x64
        x = self.pool(F.relu(self.conv2(x)))  # 64x64 → 32x32
        x = self.pool(F.relu(self.conv3(x)))  # 32x32 → 16x16
        x = x.view(-1, 64 * 16 * 16)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# Trening
print("⚙️ Przygotowanie modelu i treningu...")
device = torch.device("cpu")  # ❌ brak GPU
model = DeepPrunedCNN().to(device)

print("🔍 Pruning modelu...")
prune.l1_unstructured(model.conv1, name="weight", amount=0.2)  # Pruning 20% wag w conv1
prune.l1_unstructured(model.conv2, name="weight", amount=0.2)  # Pruning 20% wag w conv2
prune.l1_unstructured(model.conv3, name="weight", amount=0.2)  # Pruning 20% wag w conv3
prune.l1_unstructured(model.fc1, name="weight", amount=0.2)  # Pruning 20% wag w fc1

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

print("🏋️‍♂️ Start treningu...")
start_time = time.time()
for epoch in range(5):
    model.train()
    running_loss = 0.0
    print(f"\n🔁 Epoch {epoch+1}/3")

    for i, (images, labels) in enumerate(train_loader):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        if (i + 1) % 5 == 0:
            avg_loss = running_loss / 5
            print(f"   📦 Batch {i+1}/{len(train_loader)}, Loss: {avg_loss:.4f}")
            running_loss = 0.0

end_time = time.time()
train_time = end_time - start_time
print(f"\n✅ Czas treningu: {train_time:.2f} sekund")


print("\n🧪 Walidacja...")
start_eval = time.time()
model.eval()
correct = 0
total = 0
y_true = []
y_pred = []

with torch.no_grad():
    for i, (images, labels) in enumerate(val_loader):
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        y_true.extend(labels.cpu().numpy())
        y_pred.extend(predicted.cpu().numpy())

        if (i + 1) % 5 == 0:
            acc = 100 * correct / total
            print(f"   📊 Batch {i+1}/{len(val_loader)} - Accuracy: {acc:.2f}%")

end_eval = time.time()
val_time = end_eval - start_eval
print(f"\n✅ Czas walidacji: {val_time:.2f} sekund")

final_acc = 100 * correct / total
print(f"\n✅ Final validation accuracy: {final_acc:.2f}%")

# Pozostałe metryki
print("\n 📊 Szczegółowe metryki: ")
print(classification_report(y_true, y_pred, target_names=['Benign', 'Malignant']))

print("🔍 Macierz pomyłek:")
print(confusion_matrix(y_true, y_pred))

# 📌 Micro average metryki
micro_precision = precision_score(y_true, y_pred, average='micro')
micro_recall = recall_score(y_true, y_pred, average='micro')
micro_f1 = f1_score(y_true, y_pred, average='micro')

print("\n📐 Micro average metryki:")
print(f"   🔹 Precision (micro): {micro_precision:.4f}")
print(f"   🔹 Recall    (micro): {micro_recall:.4f}")
print(f"   🔹 F1-score  (micro): {micro_f1:.4f}")