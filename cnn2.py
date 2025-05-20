import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from PIL import Image
import random

# --- PARAMETRY ---
IMG_SIZE = (64, 64)
DATASET_DIR = "BreakHis_dataset"
CLASSES = {"benign": 0, "malignant": 1}

def load_images(data_dir):
    data = []
    labels = []
    for label_name, label_id in CLASSES.items():
        class_dir = os.path.join(data_dir, label_name)
        for root, _, files in os.walk(class_dir):
            for fname in files:
                if fname.endswith('.png'):
                    path = os.path.join(root, fname)
                    try:
                        img = Image.open(path).resize(IMG_SIZE).convert('RGB')
                        arr = np.asarray(img) / 255.0
                        data.append(arr)
                        labels.append(label_id)
                    except:
                        continue  # nieudane pliki pomijamy
    return np.array(data), np.array(labels)

# --- Wczytaj dane ---
X, y = load_images(DATASET_DIR)

# --- Przemieszaj dane ręcznie ---
indices = np.arange(len(X))
np.random.shuffle(indices)
X = X[indices]
y = y[indices]

# --- Podział na train/test bez sklearn ---
split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# --- Prosty model ---
model = models.Sequential([
    layers.Flatten(input_shape=(64, 64, 3)),
    layers.Dense(32, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# --- Trening ---
model.fit(X_train, y_train, epochs=5, batch_size=16, validation_data=(X_test, y_test))

# --- Ewaluacja ---
loss, acc = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {acc:.4f}")
