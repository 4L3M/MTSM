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
from sklearn.metrics import f1_score

import pygad
import numpy as np

# 📥 Wczytywanie danych
df = pd.read_csv('archive/Folds.csv')
df['label'] = df['filename'].apply(lambda x: 0 if 'benign' in x.lower() else 1)
df['filepath'] = df['filename'].apply(lambda x: os.path.join('archive', x))

train_df, val_df = train_test_split(df, test_size=0.2, stratify=df['label'], random_state=42)

# 📁 Dataset
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
    transforms.ToTensor()
])

train_dataset = HistologyDataset(train_df, transform=transform)
val_dataset = HistologyDataset(val_df, transform=transform)

# 🔧 Model z dropoutem jako parametrem
class DeepEvoCNN(nn.Module):
    def __init__(self, dropout=0.0):
        super(DeepEvoCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)    # 128x128
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)   # 128x128
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)   # 64x64
        self.pool = nn.MaxPool2d(2, 2)                 # pooling x2

        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(64 * 16 * 16, 128)
        self.fc2 = nn.Linear(128, 2)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))   # 128 → 64
        x = self.pool(F.relu(self.conv2(x)))   # 64 → 32
        x = F.relu(self.conv3(x))              # bez poolingu – 32x32
        x = x.view(-1, 64 * 16 * 16)            # po 2 poolach: 128 → 64 → 32 → 16
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


# ⚙️ Funkcja oceny rozwiązania (fitness)
def fitness_func(solution, solution_idx):
    lr = solution[0]
    dropout = solution[1]
    batch_size = int(solution[2])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DeepEvoCNN(dropout=dropout).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adagrad(model.parameters(), lr=lr)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    model.train()
    for epoch in range(1):  # Tylko 1 epoka dla szybkości
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

    # Walidacja
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())

    micro_f1 = f1_score(y_true, y_pred, average='micro')
    print(f"🧬 Solution {solution_idx} — lr={lr:.5f}, dropout={dropout:.2f} → F1={micro_f1:.4f}")
    return micro_f1  # Wyższy = lepszy

# 🧬 Parametry GA
gene_space = [
    {'low': 0.0001, 'high': 0.01},   # learning rate
    {'low': 0.0, 'high': 0.5},       # dropout
    {'low': 16, 'high': 64, 'step': 16}   # batch size (16, 32, 48, 64)

]

ga_instance = pygad.GA(
    num_generations=10,
    num_parents_mating=4,
    fitness_func=fitness_func,
    sol_per_pop=6,
    num_genes=2,
    gene_space=gene_space,
    mutation_percent_genes=50,
    mutation_type="random",
    crossover_type="single_point",
    parent_selection_type="tournament"
)

# 🚀 Uruchom optymalizację
print("🚀 Start algorytmu genetycznego (PyGAD)...")
ga_instance.run()

# 📊 Najlepszy wynik
solution, solution_fitness, _ = ga_instance.best_solution()
print(f"\n✅ Najlepsze parametry: lr={solution[0]:.5f}, dropout={solution[1]:.2f}")
print(f"🏆 Najlepszy wynik (F1): {solution_fitness:.4f}")

# Opcjonalnie wykres
ga_instance.plot_fitness()
