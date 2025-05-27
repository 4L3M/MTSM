import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
import time

# Dataset class
class BreakHisDataset(Dataset):
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

# Transforms
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor()
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# Load dataset
df = pd.read_csv('archive/Folds.csv')
df['label'] = df['filename'].apply(lambda x: 0 if "benign" in x.lower() else 1)
df['filepath'] = df['filename'].apply(lambda x: os.path.join('archive', x))

# Configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
results = []
train_times = []
start_time = time.time()

# Cross-validation
for fold, (train_idx, val_idx) in enumerate(skf.split(df['filepath'], df['label']), 1):
    fold_start_time = time.time()

    train_df = df.iloc[train_idx].reset_index(drop=True)
    val_df = df.iloc[val_idx].reset_index(drop=True)

    train_dataset = BreakHisDataset(train_df, transform=train_transform)
    val_dataset = BreakHisDataset(val_df, transform=val_transform)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    # DenseNet121 model
    model = models.densenet121(pretrained=False) # Bez transfer learning
    for param in model.parameters():
        param.requires_grad = False
    model.classifier = nn.Linear(model.classifier.in_features, 2)
    
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=0.0001)

    # Training
    model.train()
    for epoch in range(3):
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

    # Validation
    model.eval()
    y_true, y_pred, y_probs = [], [], []
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            probs = nn.functional.softmax(outputs, dim=1)[:, 1]
            _, predicted = torch.max(outputs, 1)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())
            y_probs.extend(probs.cpu().numpy())

    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_probs)

    results.append({'accuracy': acc, 'f1': f1, 'precision': precision, 'recall': recall, 'auc': auc})

    fold_duration = time.time() - fold_start_time
    train_times.append(fold_duration)

# Summary
total_duration = time.time() - start_time
mean_fold_time = np.mean(train_times)
std_fold_time = np.std(train_times)

mean_results = {k: np.mean([r[k] for r in results]) for k in results[0]}
std_results = {k: np.std([r[k] for r in results]) for k in results[0]}

# Save to file
with open("densenet121_transfer_summary.txt", "w") as f:
    f.write(" Średnie metryki po 5-fold CV:\n")
    f.write("=" * 50 + "\n")
    for key in mean_results:
        f.write(f"{key.capitalize():<10}: {mean_results[key]:.4f} ± {std_results[key]:.4f}\n")
    f.write("\n Czas wykonywania:\n")
    f.write("=" * 50 + "\n")
    f.write(f"Średni czas folda   : {mean_fold_time:.2f} sekundy ± {std_fold_time:.2f}\n")
    f.write(f"Całkowity czas runu: {total_duration:.2f} sekundy\n")

"densenet121_transfer_summary.txt"
