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

"""
# Klasyfikacja obrazów histopatologicznych BreakHis

Zwykle CNN: ✅ Final validation accuracy: 98.51%

"""

# 1. Wczytaj dane
df = pd.read_csv('archive/Folds.csv')

# Wydobądź etykietę z nazwy pliku
def extract_label(path):
    return 0 if "benign" in path.lower() else 1

df['label'] = df['filename'].apply(extract_label)

# Popraw ścieżki
df['filepath'] = df['filename'].apply(lambda x: os.path.join('archive', x))

# Podziel dane
train_df, val_df = train_test_split(df, test_size=0.2, stratify=df['label'], random_state=42)

# 2. Zdefiniuj dataset
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

# 3. Transformacje
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])

train_dataset = HistologyDataset(train_df, transform=transform)
val_dataset = HistologyDataset(val_df, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# 4. Prosta sieć CNN
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.fc1 = nn.Linear(32 * 32 * 32, 128)
        self.fc2 = nn.Linear(128, 2)  # 2 klasy: benign / malignant

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # 128x128 -> 64x64
        x = self.pool(F.relu(self.conv2(x)))  # 64x64 -> 32x32
        x = x.view(-1, 32 * 32 * 32)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# 5. Trening
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SimpleCNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
prune.1l_unstructured(model.conv1, name="weight", amount=0.2)

for epoch in range(5):  # np. 5 epok
    model.train()
    running_loss = 0.0
    print(f"Epoch {epoch+1}/{5}")


    for i, (images, labels) in enumerate(train_loader):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        if (i + 1) % 10 == 0:
            print(f"Batch {i+1}, Loss: {running_loss / 10:.4f}")
            running_loss = 0.0

    print(f"Epoch {epoch+1}, Loss: {running_loss / len(train_loader)}")

# 6. Walidacja
model.eval()
correct = 0
total = 0
print("Walidacja...")

with torch.no_grad():
    for i, (images, labels) in enumerate(val_loader):
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        if (i + 1) % 5 == 0:  # co 5 batchy
            acc = 100 * correct / total
            print(f"[Batch {i+1}/{len(val_loader)}] Partial Accuracy: {acc:.2f}%")

print(f"\n✅ Final validation accuracy: {100 * correct / total:.2f}%")
print(f"Dokładność walidacji: {100 * correct / total:.2f}%")
