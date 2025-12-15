from typing import Tuple
import torch
import torch.nn as nn
from torchvision import models


def build_model(model_name: str = "resnet18", finetune_strategy: str = "full") -> Tuple[nn.Module, int]:
    """Construye un modelo de clasificación binaria a partir de un backbone pre-entrenado.

    finetune_strategy:
        - "full": entrena todos los parámetros del backbone.
        - "head": congela el backbone y entrena solo la última capa.
    """
    model_name = model_name.lower()

    if model_name == "resnet18":
        m = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        in_feats = m.fc.in_features
        m.fc = nn.Linear(in_feats, 1)
        head_params = list(m.fc.parameters())
        backbone_params = [p for n, p in m.named_parameters() if not n.startswith("fc.")]
    elif model_name == "mobilenetv3_small":
        m = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
        in_feats = m.classifier[-1].in_features
        m.classifier[-1] = nn.Linear(in_feats, 1)
        head_params = list(m.classifier[-1].parameters())
        backbone_params = [p for n, p in m.named_parameters() if not n.startswith("classifier.")]
    else:
        raise ValueError(f"Modelo no soportado: {model_name}")

    if finetune_strategy == "head":
        for p in backbone_params:
            p.requires_grad = False
        for p in head_params:
            p.requires_grad = True
    else:
        # "full" u otra cosa: entrenar todo
        for p in m.parameters():
            p.requires_grad = True

    return m, 1
