import torch.nn as nn
from torchvision import models
try:
    import timm
except Exception:
    timm = None


def build_model(arch: str = 'resnet50', num_classes: int = 2, pretrained: bool = True):
    if timm is not None and arch not in {'resnet18','resnet34','resnet50','resnet101','resnet152'}:
        return timm.create_model(arch, pretrained=pretrained, num_classes=num_classes)
    if arch == 'resnet50':
        m = models.resnet50(weights=models.ResNet50_Weights.DEFAULT if pretrained else None)
        in_features = m.fc.in_features
        m.fc = nn.Linear(in_features, num_classes)
        return m
    if arch == 'resnet18':
        m = models.resnet18(weights=models.ResNet18_Weights.DEFAULT if pretrained else None)
        in_features = m.fc.in_features
        m.fc = nn.Linear(in_features, num_classes)
        return m
    raise ValueError(f'Unsupported arch: {arch}. Install timm for many more backbones.')