import os
import sys
import torch
import torch.optim as optim
import torch.nn as nn
from project_name.utils.loader import get_dataloaders
from project_name.models.main_model import Model
from project_name.models.baseline_model import Baseline
from train_model import train_model, train_baseline

# Optional imports (these should have a `run()` function inside them)
from project_name.evaluations.evaluate import run as run_evaluate_main
from project_name.evaluations.evaluate_and_train_baseline import run as run_evaluate_baseline
from project_name.explainability.explainable import run as run_explainable
from project_name.loading.load_model import run as run_loader


def main():
    if len(sys.argv) < 2:
        print("Usage: python main.py [main_model|baseline_model|eval_main|eval_baseline|explain|load]")
        return

    command = sys.argv[1]
    classes = [cls for cls in os.listdir('./data') if not cls.endswith('.csv')]
    train_loader, val_loader, test_loader = get_dataloaders('./data', classes)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if command == 'main_model':
        model = Model()
        optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
        criterion = nn.CrossEntropyLoss()
        model.to(device)
        train_model(model, train_loader, val_loader, criterion, optimizer, device, num_epochs=25)
        torch.save(model.state_dict(), './models/model.pth')
    
    elif command == 'baseline_model':
        model = Baseline()
        train_baseline(model, train_loader)
    
    elif command == 'eval_main':
        run_evaluate_main()
    
    elif command == 'eval_baseline':
        run_evaluate_baseline()

    elif command == 'explain':
        run_explainable()

    elif command == 'load':
        run_loader()

    else:
        print("Unknown command. Valid options are: main_model, baseline_model, eval_main, eval_baseline, explain, load")

if __name__ == '__main__':
    main()
