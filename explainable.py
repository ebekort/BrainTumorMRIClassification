import cv2
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms

def apply_gradcam(model, image_tensor, target_class=None, device='cpu'):
    model.eval()
    image_tensor = image_tensor.unsqueeze(0).to(device)

    #hook the gradients
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

    # Clean up hook
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
    orig_image -= orig_image.min()
    orig_image /= orig_image.max()
    orig_image = np.uint8(255 * orig_image)

    overlay = cv2.addWeighted(orig_image, 0.5, heatmap, 0.5, 0)
    return overlay
