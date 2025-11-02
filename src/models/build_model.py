from typing import Tuple
import torch
import torch.nn as nn
from torchvision import models


def build_model(model_name: str = "resnet18") -> Tuple[nn.Module, int]:
    if model_name.lower() == "resnet18":
        m = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        in_feats = m.fc.in_features
        m.fc = nn.Linear(in_feats, 1)
    elif model_name.lower() == "mobilenetv3_small":
        m = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
        in_feats = m.classifier[-1].in_features
        m.classifier[-1] = nn.Linear(in_feats, 1)
    else:
        raise ValueError(f"Modelo no soportado: {model_name}")
    return m, 1
