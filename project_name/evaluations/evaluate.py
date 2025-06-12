import torch
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import os
from project_name.utils.loader import get_dataloaders
from project_name.models.main_model import Model

def evaluate(model, test_loader, device, class_names):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            true_labels = torch.argmax(labels, dim=1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(true_labels.cpu().numpy())

    print("\nClassification Report:\n")
    print(classification_report(all_labels, all_preds, target_names=class_names))

    cm = confusion_matrix(all_labels, all_preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.show()


def run():
    class_names = [cls for cls in os.listdir('./data') if not cls.endswith('.csv')]
    _, _, test_loader = get_dataloaders('./data', class_names)

    model = Model()
    model_path = os.path.join("project_name", "models", "model7.pth")
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    evaluate(model, test_loader, device, class_names)
