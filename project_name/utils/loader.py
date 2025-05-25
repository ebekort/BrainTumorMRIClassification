from torch.utils.data import DataLoader, Subset, DataLoader
from sklearn.model_selection import train_test_split
from torchvision import transforms
from .dataset import MRI_Dataset
import os


# Define training transform
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.9, 1.1)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2),
    transforms.ToTensor(),
])

# Define validation and test transform
val_test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

def get_dataloaders(data_dir, classes, batch_size=32, test_size=0.15, val_size=0.1765, random_state=44, num_workers=4):
    # Create datasets with respective transforms
    train_dataset_full = MRI_Dataset(data_dir, classes, transform=train_transform)
    val_dataset_master = MRI_Dataset(data_dir, classes, transform=val_test_transform)
    # test_dataset_master is not strictly needed if val_dataset_master uses the same val_test_transform
    # and we are careful with indices. However, for clarity and safety:
    test_dataset_master = MRI_Dataset(data_dir, classes, transform=val_test_transform)


    # Determine the total number of samples from one of the datasets (e.g., train_dataset_full, as all should have same length before subsetting)
    # It's better to get indices from a single source of truth for dataset size, like querying the file system or a manifest.
    # For this setup, we assume all MRI_Dataset instances will correctly report the full dataset length.
    # However, the original code split based on a single 'dataset' instance. Let's get the full length first.
    # A common practice is to list all files or use a pre-computed list of (image_path, label) pairs.
    # For now, let's assume MRI_Dataset without transform can give us the total count for splitting.
    # Or, more simply, get indices from one transformed dataset (e.g. train_dataset_full) as they all scan the same directory.
    
    total_samples_dataset = MRI_Dataset(data_dir, classes, transform=None) # Get total number of samples for splitting
    indices = list(range(len(total_samples_dataset)))
    all_labels = [total_samples_dataset.dataset[i][1] for i in indices]

    # Split indices using stratification
    temp_indices, test_indices = train_test_split(indices, test_size=test_size, stratify=all_labels, random_state=random_state)
    
    temp_labels = [all_labels[i] for i in temp_indices]
    train_indices, val_indices = train_test_split(temp_indices, test_size=val_size, stratify=temp_labels, random_state=random_state)

    # Create subsets using the specific datasets and their corresponding indices
    train_dataset = Subset(train_dataset_full, train_indices)
    val_dataset = Subset(val_dataset_master, val_indices)
    test_dataset = Subset(test_dataset_master, test_indices) # if test_dataset_master was created
    # If not creating test_dataset_master, then: val_dataset = Subset(val_dataset_master, val_indices)
    # test_dataset = Subset(val_dataset_master, test_indices) # Re-using val_dataset_master for test subset is fine

    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, test_loader
