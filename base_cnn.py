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
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import time
import csv

# 1. Dane
df = pd.read_csv('archive/Folds.csv')
df['label'] = df['filename'].apply(lambda x: 0 if "benign" in x.lower() else 1)
df['filepath'] = df['filename'].apply(lambda x: os.path.join('archive', x))

small_df = df.sample(n=100, random_state=42, stratify=df['label'])
train_df, val_df = train_test_split(small_df, test_size=0.2, stratify=small_df['label'], random_state=42)

# 2. Dataset
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

transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])

train_dataset = HistologyDataset(train_df, transform=transform)
val_dataset = HistologyDataset(val_df, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)

# 3. Model BEZ Dropoutu, BEZ Pruningu, BEZ DataParallel
class BaselineCNN(nn.Module):
    def __init__(self):
        super(BaselineCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.fc1 = nn.Linear(32 * 32 * 32, 128)
        self.fc2 = nn.Linear(128, 2)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 32 * 32 * 32)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = BaselineCNN().to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adagrad(model.parameters(), lr=0.01)

# 4. Zapis wyników
results_file = "wyniki_baseline.csv"
with open(results_file, mode='w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['Epoch', 'TrainTime_s', 'ValTime_s', 'Accuracy', 'Precision', 'Recall', 'F1'])

# 5. Trening i walidacja
for epoch in range(2):
    model.train()
    start_train = time.time()
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
    train_time = time.time() - start_train

    model.eval()
    y_true, y_pred = [], []
    start_val = time.time()
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())
    val_time = time.time() - start_val

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average='macro', zero_division=0)
    rec = recall_score(y_true, y_pred, average='macro', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)

    print(f"\n📊 Epoch {epoch+1} | Acc: {acc:.2f} | Prec: {prec:.2f} | Recall: {rec:.2f} | F1: {f1:.2f}")
    print(f"⏱️ Trening: {train_time:.2f}s | Walidacja: {val_time:.2f}s")

    with open(results_file, mode='a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([epoch+1, f"{train_time:.2f}", f"{val_time:.2f}", f"{acc:.4f}", f"{prec:.4f}", f"{rec:.4f}", f"{f1:.4f}"])

print("\n✅ Zakończono. Wyniki zapisano do:", results_file)
