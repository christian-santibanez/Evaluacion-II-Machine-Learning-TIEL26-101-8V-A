import os
import json
from pathlib import Path
import sys

import torch

# Asegurar que la RAÍZ del proyecto esté en el sys.path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.models.build_model import build_model
from src.models.export_onnx import export_to_onnx

def test_build_model_output_shape():
    model, _ = build_model("resnet18", finetune_strategy="head")
    model.eval()
    x = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        y = model(x)
    # salida binaria: [batch, 1]
    assert y.shape == (2, 1)


def test_export_onnx_runs(tmp_path):
    """Verifica que export_to_onnx se ejecuta sin error con un experimento mínimo simulado.

    No comprueba la calidad del modelo, solo que la función corre y genera un archivo .onnx.
    """
    # Crear estructura mínima de experimento
    exp_dir = tmp_path / "exp_dummy"
    exp_dir.mkdir(parents=True, exist_ok=True)

    # Crear un modelo pequeño y guardarlo como best_model.pt
    model, _ = build_model("resnet18", finetune_strategy="head")
    best_path = exp_dir / "best_model.pt"
    torch.save(model.state_dict(), best_path)

    # Config mínima
    cfg = {
        "training": {
            "model_name": "resnet18",
            "image_size": 224,
            "finetune_strategy": "head",
        }
    }
    cfg_path = exp_dir / "config_used.json"
    import json

    with cfg_path.open("w", encoding="utf-8") as f:
        json.dump(cfg, f)

    onnx_path = export_to_onnx(str(exp_dir), str(cfg_path))
    assert os.path.isfile(onnx_path)
