import torch.nn as nn
import torchvision.models as models
import torch

class Model(nn.Module):
  def __init__(self, num_classes=3, hidden_layers=[512]):
    super(Model, self).__init__()
    self.backbone = models.resnet50(pretrained=True)
    
    layers = []
    size_in = self.backbone.fc.out_features
    for size in hidden_layers:
      layers.append(nn.Linear(size_in, size))
      layers.append(nn.ReLU())
      size_in = size

    layers.append(nn.Linear(size_in, num_classes))
    self.sequential = nn.Sequential(*layers)


  def forward(self, x):
    features = self.backbone(x)
    logits = self.sequential(features)
    return logits