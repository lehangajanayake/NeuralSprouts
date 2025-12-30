import torch
import torch.nn as nn
import torchvision.models as models


def _create_resnet18(pretrained: bool = True):
    """Create ResNet18 with torchvision-compatible pretrained loading.

    torchvision changed the API from `pretrained=True` to `weights=...`.
    This helper supports both without pinning a specific torchvision version.
    """
    if not pretrained:
        return models.resnet18(weights=None) if hasattr(models, 'ResNet18_Weights') else models.resnet18(pretrained=False)

    if hasattr(models, 'ResNet18_Weights'):
        return models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    return models.resnet18(pretrained=True)

class SimpleResNetModel(nn.Module):
    def __init__(self, num_classes=3):
        super(SimpleResNetModel, self).__init__()
        # Use a pretrained ResNet18 as feature extractor
        self.resnet = _create_resnet18(pretrained=True)
        # Replace the final fully connected layer
        in_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Identity()
        # Two heads: regression and classification
        self.reg_head = nn.Sequential(
            nn.Linear(in_features, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        self.class_head = nn.Sequential(
            nn.Linear(in_features, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )
    def forward(self, x):
        features = self.resnet(x)
        reg_out = self.reg_head(features)
        class_out = self.class_head(features)
        return reg_out, class_out
