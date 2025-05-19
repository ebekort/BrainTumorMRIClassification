import os
from project_name.utils.loader import get_dataloaders
from project_name.models.main_model import Model
from train_model import train
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F


def main():
    classes = [cls for cls in os.listdir('./data') if not cls.endswith('.csv')]
    print(f'classes: {classes}')
    train_loader, val_loader, test_loader = get_dataloaders('./data', classes)
    model = Model()
    optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
    criterion = nn.CrossEntropyLoss()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    train(model, train_loader, val_loader, criterion, optimizer, device, num_epochs=25)
    # Save the model
    torch.save(model.state_dict(), './models/model.pth')

    


if __name__ == '__main__':
    main()