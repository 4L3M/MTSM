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


# Dane wejściowe
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

# Transformacje dla ResNet
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor()
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# Wczytaj dane
df = pd.read_csv('archive/Folds.csv')
df['label'] = df['filename'].apply(lambda x: 0 if "benign" in x.lower() else 1)
df['filepath'] = df['filename'].apply(lambda x: os.path.join('archive', x))

# Setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Metryki
results = []

train_times = []
start_time = time.time()

# Cross-validation
for fold, (train_idx, val_idx) in enumerate(skf.split(df['filepath'], df['label']), 1):
    print(f"Fold {fold}")
    fold_start_time = time.time()

    train_df = df.iloc[train_idx].reset_index(drop=True)
    val_df = df.iloc[val_idx].reset_index(drop=True)

    train_dataset = BreakHisDataset(train_df, transform=train_transform)
    val_dataset = BreakHisDataset(val_df, transform=val_transform)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    # Model z transfer learningiem
    resnet = models.resnet18(pretrained=True)
    for param in resnet.parameters():
        param.requires_grad = False
    num_ftrs = resnet.fc.in_features
    resnet.fc = nn.Linear(num_ftrs, 2)
    model = resnet.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.fc.parameters(), lr=0.0001)

    # Trening
    model.train()
    for epoch in range(3):
        print(f"Epoch {epoch+1}/3")
        running_loss = 0.0
        for i, (images, labels) in enumerate(train_loader):
            print(f"   Batch {i+1}/{len(train_loader)}", end='\r')
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

    print(f" Epoch {epoch+1} done — Avg loss: {running_loss/len(train_loader):.4f}")

    # Walidacja
    print(" Walidacja...")
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

    # Metryki
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_probs)

    results.append({
        'fold': fold,
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall,
        'auc': auc
    })

    fold_duration = time.time() - fold_start_time
    train_times.append(fold_duration)
    print(f"Fold {fold} done in {fold_duration:.2f} seconds\n")

# Podsumowanie
total_duration = time.time() - start_time
mean_fold_time = np.mean(train_times)
std_fold_time = np.std(train_times)


mean_results = {
    'accuracy': np.mean([result['accuracy'] for result in results]),
    'f1': np.mean([result['f1'] for result in results]),
    'precision': np.mean([result['precision'] for result in results]),
    'recall': np.mean([result['recall'] for result in results]),
    'auc': np.mean([result['auc'] for result in results])
}


std_results = {
    'accuracy': np.std([result['accuracy'] for result in results]),
    'f1': np.std([result['f1'] for result in results]),
    'precision': np.std([result['precision'] for result in results]),
    'recall': np.std([result['recall'] for result in results]),
    'auc': np.std([result['auc'] for result in results])
}


print(f"\n Fold {fold} Results:")
print(f"   Accuracy  : {acc:.4f}")
print(f"   Precision : {precision:.4f}")
print(f"   Recall    : {recall:.4f}")
print(f"   F1-Score  : {f1:.4f}")
print(f"   AUC       : {auc:.4f}")
print(f" Sredni czas trwania folda: {mean_fold_time:.2f} ± {std_fold_time:.2f} sekund")
print(f" Czas całkowity: {total_duration:.2f} sekund")

# Zapisz do pliku
with open("resnet18_transfer_summary.txt", "w") as f:
    f.write("\nŚrednie metryki po 5-fold CV:\n")
    f.write("=" * 50 + "\n")
    f.write(f"Accuracy   : {mean_results['accuracy']:.4f} ± {std_results['accuracy']:.4f}\n")
    f.write(f"Precision  : {mean_results['precision']:.4f} ± {std_results['precision']:.4f}\n")
    f.write(f"Recall     : {mean_results['recall']:.4f} ± {std_results['recall']:.4f}\n")
    f.write(f"F1-Score   : {mean_results['f1']:.4f} ± {std_results['f1']:.4f}\n")
    f.write(f"AUC        : {mean_results['auc']:.4f} ± {std_results['auc']:.4f}\n")
    
    f.write("\n⏱️ Czas wykonywania:\n")
    f.write("=" * 50 + "\n")
    f.write(f"Średni czas folda   : {mean_fold_time:.2f} sekundy ± {std_fold_time:.2f}\n")
    f.write(f"Całkowity czas runu: {total_duration:.2f} sekundy\n")

    # Koniec
    print("Wyniki zapisane do pliku resnet18_transfer_summary.txt")