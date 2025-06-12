import torch
import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image
from project_name.models.main_model import Model
from project_name.utils.loader import get_dataloaders


def apply_gradcam(model, image_tensor, target_class=None, device='cpu'):
    model.eval()
    image_tensor = image_tensor.unsqueeze(0).to(device)

    gradients = []
    activations = []

    def save_gradient(grad):
        gradients.append(grad)


    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            last_conv_layer = module

    def forward_hook(module, input, output):
        activations.append(output)
        output.register_hook(save_gradient)

    hook = last_conv_layer.register_forward_hook(forward_hook)


    output = model(image_tensor)
    pred_class = output.argmax(dim=1).item() if target_class is None else target_class

    model.zero_grad()
    class_loss = output[0, pred_class]
    class_loss.backward()
    hook.remove()

    grad = gradients[0].cpu().data.numpy()[0]
    act = activations[0].cpu().data.numpy()[0]
    weights = np.mean(grad, axis=(1, 2))
    cam = np.zeros(act.shape[1:], dtype=np.float32)
    for i, w in enumerate(weights):
        cam += w * act[i]

    cam = np.maximum(cam, 0)
    cam = cv2.resize(cam, (224, 224))
    cam -= cam.min()
    cam /= cam.max()


    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    orig_image = image_tensor.squeeze().cpu().numpy().transpose(1, 2, 0)


    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    orig_image = std * orig_image + mean
    orig_image = np.clip(orig_image, 0, 1)
    orig_image = np.uint8(255 * orig_image)

    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    overlay = cv2.addWeighted(orig_image, 0.5, heatmap, 0.5, 0)
    return overlay



if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    model = Model()
    model_path = os.path.join("project_name", "models", "model7.pth")
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()


    #For taking the first image from the testset, delete the part below
    classes = [cls for cls in os.listdir('./data') if not cls.endswith('.csv')]
    train_loader, val_loader, test_loader = get_dataloaders('./data', classes)
    image_tensor, label = next(iter(test_loader))
    image = transforms.ToPILImage()(image_tensor[1])

    #If you want to take the first image
    image_path = "data/brain_glioma/brain_glioma_0001.jpg"
    image = Image.open(image_path).convert("RGB")

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    image_tensor = transform(image)

    gradcam_overlay = apply_gradcam(model, image_tensor, device=device)


    plt.imshow(gradcam_overlay)
    plt.title("Grad-CAM Visualization")
    plt.axis("off")
    plt.show()
    cv2.imwrite("gradcam_output.jpg", cv2.cvtColor(gradcam_overlay, cv2.COLOR_RGB2BGR))
    


