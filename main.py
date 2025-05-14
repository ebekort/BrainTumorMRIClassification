import os
from project_name.utils.loader import get_dataloaders
from project_name.models.main_model import Model

def main():
    classes = [cls for cls in os.listdir('./data') if not cls.endswith('.csv')]
    print(f'classes: {classes}')
    train_loader, val_loader, test_loader = get_dataloaders('./data', classes)
    model = Model()
    for images, labels in train_loader:
        print(model.forward(images))


if __name__ == '__main__':
    main()
