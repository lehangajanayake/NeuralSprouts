import torch
import torch.nn as nn


def _create_resnet18(pretrained: bool = True):
    """Create ResNet18 with torchvision-compatible pretrained loading.

    torchvision changed the API from `pretrained=True` to `weights=...`.
    This helper supports both without pinning a specific torchvision version.
    """
    import torchvision.models as models

    if not pretrained:
        return models.resnet18(weights=None) if hasattr(models, 'ResNet18_Weights') else models.resnet18(pretrained=False)

    if hasattr(models, 'ResNet18_Weights'):
        return models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    return models.resnet18(pretrained=True)

class SimpleResNetModel(nn.Module):
    def __init__(self, num_classes=3):
        super(SimpleResNetModel, self).__init__()
        # Default: the intended baseline (ResNet18).
        # If torchvision can't be imported in a user's environment, we fall
        # back to a tiny CNN so the codebase stays runnable.
        self._using_fallback = False
        try:
            self.resnet = _create_resnet18(pretrained=True)
            feat_dim = self.resnet.fc.in_features
            self.resnet.fc = nn.Identity()
        except Exception as e:
            self._using_fallback = True
            self._torchvision_error = str(e)
            self.resnet = None
            self.backbone = nn.Sequential(
                nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),

                nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),

                nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),

                nn.AdaptiveAvgPool2d((1, 1)),
            )
            feat_dim = 128

        self.reg_head = nn.Sequential(
            nn.Linear(feat_dim, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 1),
        )
        self.class_head = nn.Sequential(
            nn.Linear(feat_dim, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, num_classes),
        )
    def forward(self, x):
        if self.resnet is not None:
            feats = self.resnet(x)
        else:
            feats = self.backbone(x)
            feats = torch.flatten(feats, 1)
        reg_out = self.reg_head(feats)
        class_out = self.class_head(feats)
        return reg_out, class_out
