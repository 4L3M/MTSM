import os
import pandas as pd
import numpy as np
from PIL import Image
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score

import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

# Dataset class
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
print("1")

# Updated model with BatchNorm after Conv1 and Conv2
class DeepCNN_BatchNorm(nn.Module):
    def __init__(self):
        super(DeepCNN_BatchNorm, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 16 * 16, 128)
        self.fc2 = nn.Linear(128, 2)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(-1, 64 * 16 * 16)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Sprawdzenie pliku wejściowego
assert os.path.exists("archive/Folds.csv"), "Brak pliku archive/Folds.csv!"

# Konfiguracja
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])

df = pd.read_csv('archive/Folds.csv')
df['label'] = df['filename'].apply(lambda x: 0 if "benign" in x.lower() else 1)
df['filepath'] = df['filename'].apply(lambda x: os.path.join('archive', x))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Używane urządzenie: {device}")

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

results = []
summary_lines = []
start_time = time.time()

for fold, (train_idx, val_idx) in enumerate(skf.split(df['filepath'], df['label']), 1):
    print(f"Rozpoczynam fold {fold}...")
    fold_start_time = time.time()

    train_df = df.iloc[train_idx].reset_index(drop=True)
    val_df = df.iloc[val_idx].reset_index(drop=True)

    train_dataset = HistologyDataset(train_df, transform=transform)
    val_dataset = HistologyDataset(val_df, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    model = DeepCNN_BatchNorm().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01)

    # Trening
    model.train()
    for epoch in range(3):
        print(f"Fold {fold}, Epoch {epoch + 1}/3")
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

    # Walidacja
    model.eval()
    y_true, y_pred, y_probs = [], [], []
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            probs = F.softmax(outputs, dim=1)[:, 1]
            _, predicted = torch.max(outputs, 1)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())
            y_probs.extend(probs.cpu().numpy())

    # Metryki
    acc = 100 * (np.array(y_true) == np.array(y_pred)).sum() / len(y_true)
    f1_micro = f1_score(y_true, y_pred, average='micro')
    f1_macro = f1_score(y_true, y_pred, average='macro')
    precision_micro = precision_score(y_true, y_pred, average='micro')
    precision_macro = precision_score(y_true, y_pred, average='macro')
    recall_micro = recall_score(y_true, y_pred, average='micro')
    recall_macro = recall_score(y_true, y_pred, average='macro')
    auc = roc_auc_score(y_true, y_probs)

    results.append({
        'accuracy': acc,
        'f1_micro': f1_micro,
        'f1_macro': f1_macro,
        'precision_micro': precision_micro,
        'precision_macro': precision_macro,
        'recall_micro': recall_micro,
        'recall_macro': recall_macro,
        'auc': auc
    })

    fold_duration = time.time() - fold_start_time
    summary_lines.append(f"Czas folda {fold}: {fold_duration:.2f} sekundy")

# Podsumowanie wyników
mean_results = {k: np.mean([r[k] for r in results]) for k in results[0]}
std_results = {k: np.std([r[k] for r in results]) for k in results[0]}

for key in mean_results:
    summary_lines.append(f"{key:<18}: {mean_results[key]:.4f} ± {std_results[key]:.4f}")

summary_lines.append(f"Czas całkowity         : {time.time() - start_time:.2f} sekundy")

# Zapis wyników
with open("batcht-archi.txt", "w") as f:
    for line in summary_lines:
        f.write(line + "\n")

print("✅ batcht-archi.txt został zapisany.")
