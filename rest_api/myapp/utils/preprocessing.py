from torchvision import transforms
from PIL import Image
import os
import torch
import numpy as np


transform = transforms.Compose([
        transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

def preprocess_image(image):
    
    image = Image.open(image)
    if image is None:
        return ValueError("Image does not exist or is empty")
    image = transform(image)
    return image.unsqueeze(0)