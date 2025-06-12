import torch
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import os
from project_name.utils.loader import get_dataloaders
from project_name.models.baseline_model import Baseline
import numpy as np

import os
import torch
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, accuracy_score
import matplotlib.pyplot as plt

def evaluate(model, test_loader, device, class_names, is_baseline=False):
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in test_loader:
            if not is_baseline:
                images = images.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)
                true_labels = torch.argmax(labels, dim=1)
            else:
                predicted = model.predict(images)
                true_labels = torch.argmax(labels, dim=1).numpy()

            all_preds.extend(predicted.cpu().numpy() if not is_baseline else predicted)
            all_labels.extend(true_labels)

    accuracy = accuracy_score(all_labels, all_preds)
    print(f"\nAccuracy: {accuracy * 100:.2f}%")

    print("\nClassification Report:\n")
    print(classification_report(all_labels, all_preds, target_names=class_names))

    cm = confusion_matrix(all_labels, all_preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)

    output_dir = 'results'
    os.makedirs(output_dir, exist_ok=True)

    disp.plot(cmap=plt.cm.Blues)
    plt.title("Confusion Matrix Baseline Model")

    save_path = os.path.join(output_dir, 'confusion_matrix_baseline.png')
    plt.savefig(save_path)
    print(f"Confusion matrix saved to {save_path}")

    plt.show()

def run():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    class_names = [cls for cls in os.listdir('./data') if not cls.endswith('.csv')]
    train_loader, val_loader, test_loader = get_dataloaders('./data', class_names)

    model = Baseline()


    for images, labels in train_loader:
        model.partial_fit(images, labels, classes=np.arange(len(class_names)))

    evaluate(model, test_loader, device, class_names, is_baseline=True)
