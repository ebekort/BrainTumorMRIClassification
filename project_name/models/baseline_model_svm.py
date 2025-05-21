#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 21 11:31:44 2025

@author: sennehollard
"""

# baseline_model_svm.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler
from torchvision.datasets import ImageFolder
from torchvision import transforms
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
from PIL import Image
import os

# Define a transformation to match your image preprocessing
transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((64, 64)),
    transforms.ToTensor()
])

# Load dataset from the same folder as used by your PyTorch Dataset
data_path = "./data"
dataset = ImageFolder(root=data_path, transform=transform)
loader = DataLoader(dataset, batch_size=1, shuffle=False)

# Extract features and labels
X = []
y = []

print("Extracting image features for SVM...")
for img, label in tqdm(loader):
    img_np = img.squeeze().numpy().flatten()  # flatten the image
    X.append(img_np)
    y.append(label)

X = np.array(X)
y = np.array([int(l.item()) for l in y], dtype=int)

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train SVM
print("Training SVM...")
svm = SVC(kernel='linear')  # you could also try 'rbf' or others
svm.fit(X_train, y_train)

# Evaluate
y_pred = svm.predict(X_test)
print("Classification Report:")
print(classification_report(y_test, y_pred))
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
