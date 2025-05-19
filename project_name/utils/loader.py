from torch.utils.data import DataLoader, Subset, DataLoader
from sklearn.model_selection import train_test_split
from torchvision import transforms
from .dataset import MRI_Dataset
import os


transform = transforms.Compose([
        transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

def get_dataloaders(data_dir, classes, batch_size=32, test_size=0.15, val_size=0.1765, random_state=44, num_workers=4):
    # Initialize dataset
    dataset = MRI_Dataset(data_dir, classes, transform=transform)

    # Split indices
    indices = list(range(len(dataset)))
    temp_indices, test_indices = train_test_split(indices, test_size=test_size, random_state=random_state)
    train_indices, val_indices = train_test_split(temp_indices, test_size=val_size, random_state=random_state)

    # Create subsets
    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)
    test_dataset = Subset(dataset, test_indices)

    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)

    return train_loader, val_loader, test_loader
