import os
import pandas as pd
import numpy as np
from PIL import Image
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, precision_score, recall_score, f1_score, roc_auc_score, roc_curve

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import optuna

print("\U0001F4C5 Wczytywanie danych...")
df = pd.read_csv('archive/Folds.csv')
df['label'] = df['filename'].apply(lambda x: 0 if "benign" in x.lower() else 1)
df['filepath'] = df['filename'].apply(lambda x: os.path.join('archive', x))
print(f"✔️ Załadowano {len(df)} rekordów.")

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

# Model
class DeepCNN(nn.Module):
    def __init__(self, dropout=0.0):
        super(DeepCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 16 * 16, 128)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(128, 2)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(-1, 64 * 16 * 16)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# Optuna objective

def objective(trial):
    lr = trial.suggest_loguniform('lr', 1e-4, 1e-2)
    dropout = trial.suggest_uniform('dropout', 0.0, 0.5)
    batch_size = trial.suggest_categorical('batch_size', [16, 32])

    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    device = torch.device("cpu")

    f1_scores = []

    for train_idx, val_idx in skf.split(df['filepath'], df['label']):
        train_df = df.iloc[train_idx].reset_index(drop=True)
        val_df = df.iloc[val_idx].reset_index(drop=True)

        train_dataset = HistologyDataset(train_df, transform=transform)
        val_dataset = HistologyDataset(val_df, transform=transform)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        model = DeepCNN(dropout=dropout).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(), lr=lr)

        model.train()
        for epoch in range(2):
            for images, labels in train_loader:
                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

        model.eval()
        y_true = []
        y_pred = []
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                y_true.extend(labels.cpu().numpy())
                y_pred.extend(predicted.cpu().numpy())

        f1 = f1_score(y_true, y_pred, average='macro')
        f1_scores.append(f1)

    return np.mean(f1_scores)

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=10)

print("\n✅ Najlepsze parametry:")
for key, value in study.best_params.items():
    print(f"   {key}: {value:.4f}")
print(f"🏆 Najlepszy wynik (macro F1): {study.best_value:.4f}")
