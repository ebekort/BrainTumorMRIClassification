from torch.utils.data import Dataset
from PIL import Image
import os
import torch
import numpy as np

class MRI_Dataset(Dataset):
  def __init__(self, data_dir, classes, transform=None):
    self.data_dir = data_dir
    self.classes = classes
    self.class_to_idx = {cls: idx for idx, cls in enumerate(classes)}
    self.transform = transform

    self.dataset = self.load_dataset(self.data_dir)
    self.dataset = self.dataset


  def load_dataset(self, data_dir):
    data = []
    for cls in self.classes:
      cls_dir = os.path.join(data_dir, cls)
      for fname in os.listdir(cls_dir):
        fpath = os.path.join(cls_dir, fname)
        data.append((fpath, self.class_to_idx[cls]))
    return data


  def __len__(self):
    return len(self.dataset)


  def __getitem__(self, idx):
    img_path, label = self.dataset[idx]
    img = Image.open(img_path).convert("L")
    if self.transform:
      img = self.transform(img)
    
    # Convert to one channel maybe since it is just black/white data?? but that might conflict with resnet, and normalize right away?
    onehot = np.zeros(len(self.classes))
    onehot[label] = 1 # maybe we need dtype=long instead?
    return img, onehot
