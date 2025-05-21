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

import csv

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
    def __init__(self):
        super(DeepCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(64 * 16 * 16, 128)
        self.fc2 = nn.Linear(128, 2)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(-1, 64 * 16 * 16)
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Cross-validation
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
device = torch.device("cpu")

accuracies = []
f1_micro_scores = []
f1_macro_scores = []
precision_micro_scores = []
precision_macro_scores = []
recall_micro_scores = []
recall_macro_scores = []
auc_scores = []
results = []
fold = 1

for train_idx, val_idx in skf.split(df['filepath'], df['label']):
    print(f"\n🔁 Fold {fold}")

    train_df = df.iloc[train_idx].reset_index(drop=True)
    val_df = df.iloc[val_idx].reset_index(drop=True)

    train_dataset = HistologyDataset(train_df, transform=transform)
    val_dataset = HistologyDataset(val_df, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    model = DeepCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01)

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

    acc = 100 * (np.array(y_true) == np.array(y_pred)).sum() / len(y_true)
    f1_micro = f1_score(y_true, y_pred, average='micro')
    f1_macro = f1_score(y_true, y_pred, average='macro')
    precision_micro = precision_score(y_true, y_pred, average='micro')
    precision_macro = precision_score(y_true, y_pred, average='macro')
    recall_micro = recall_score(y_true, y_pred, average='micro')
    recall_macro = recall_score(y_true, y_pred, average='macro')
    auc = roc_auc_score(y_true, y_probs)
    cm = confusion_matrix(y_true, y_pred).tolist()

    print(f"✅ Fold {fold} — Accuracy: {acc:.2f}%, F1_micro: {f1_micro:.4f}, F1_macro: {f1_macro:.4f}, Precision_micro: {precision_micro:.4f}, Precision_macro: {precision_macro:.4f}, Recall_micro: {recall_micro:.4f}, Recall_macro: {recall_macro:.4f}, AUC: {auc:.4f}")

    results.append({
        'fold': fold,
        'accuracy': acc,
        'f1_micro': f1_micro,
        'f1_macro': f1_macro,
        'precision_micro': precision_micro,
        'precision_macro': precision_macro,
        'recall_micro': recall_micro,
        'recall_macro': recall_macro,
        'auc': auc,
        'confusion_matrix': cm
    })

    accuracies.append(acc)
    f1_micro_scores.append(f1_micro)
    f1_macro_scores.append(f1_macro)
    precision_micro_scores.append(precision_micro)
    precision_macro_scores.append(precision_macro)
    recall_micro_scores.append(recall_micro)
    recall_macro_scores.append(recall_macro)
    auc_scores.append(auc)
    fold += 1

# Zapis do CSV
results_df = pd.DataFrame(results)
results_df.to_csv("deepcnn_dropout_results.csv", index=False)

# Średnie metryki
print("\n📊 Średnie metryki po 5-fold CV:")
print(f"   🔹 Accuracy           : {np.mean(accuracies):.2f}% ± {np.std(accuracies):.2f}")
print(f"   🔹 Micro Precision    : {np.mean(precision_micro_scores):.4f} ± {np.std(precision_micro_scores):.4f}")
print(f"   🔹 Macro Precision    : {np.mean(precision_macro_scores):.4f} ± {np.std(precision_macro_scores):.4f}")
print(f"   🔹 Micro Recall       : {np.mean(recall_micro_scores):.4f} ± {np.std(recall_micro_scores):.4f}")
print(f"   🔹 Macro Recall       : {np.mean(recall_macro_scores):.4f} ± {np.std(recall_macro_scores):.4f}")
print(f"   🔹 Micro F1-score     : {np.mean(f1_micro_scores):.4f} ± {np.std(f1_micro_scores):.4f}")
print(f"   🔹 Macro F1-score     : {np.mean(f1_macro_scores):.4f} ± {np.std(f1_macro_scores):.4f}")
print(f"   🔹 AUC                : {np.mean(auc_scores):.4f} ± {np.std(auc_scores):.4f}")
