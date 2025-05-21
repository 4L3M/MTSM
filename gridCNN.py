import os
import pandas as pd
import numpy as np
from PIL import Image
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.model_selection import ParameterGrid

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

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
    def __init__(self, dropout):
        super(DeepCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(64 * 16 * 16, 128)
        self.fc2 = nn.Linear(128, 2)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(-1, 64 * 16 * 16)
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.fc2(x)
        return x

# Parametry do testowania
param_grid = {
    'lr': [0.01, 0.001],
    'dropout': [0.0, 0.3, 0.5],
    'batch_size': [16, 32]
}

device = torch.device("cpu")
results = []

for params in ParameterGrid(param_grid):
    print(f"\n🔍 Testowanie: {params}")
    
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    fold = 1
    fold_f1_scores = []

    for train_idx, val_idx in skf.split(df['filepath'], df['label']):
        train_df = df.iloc[train_idx].reset_index(drop=True)
        val_df = df.iloc[val_idx].reset_index(drop=True)

        train_dataset = HistologyDataset(train_df, transform=transform)
        val_dataset = HistologyDataset(val_df, transform=transform)

        train_loader = DataLoader(train_dataset, batch_size=params['batch_size'], shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=params['batch_size'], shuffle=False)

        model = DeepCNN(dropout=params['dropout']).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(), lr=params['lr'])

        model.train()
        for epoch in range(3):
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
        y_probs = []
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                probs = F.softmax(outputs, dim=1)[:, 1]
                _, predicted = torch.max(outputs.data, 1)
                y_true.extend(labels.cpu().numpy())
                y_pred.extend(predicted.cpu().numpy())
                y_probs.extend(probs.cpu().numpy())

        f1_macro = f1_score(y_true, y_pred, average='macro')
        fold_f1_scores.append(f1_macro)
        fold += 1

    avg_f1 = np.mean(fold_f1_scores)
    results.append({
        'lr': params['lr'],
        'dropout': params['dropout'],
        'batch_size': params['batch_size'],
        'f1_macro_avg': avg_f1
    })

# Zapisz wyniki
results_df = pd.DataFrame(results)
results_df.to_csv("gridsearch_deepcnn_results.csv", index=False)
print("\n📁 Wyniki zapisane do gridsearch_deepcnn_results.csv")
