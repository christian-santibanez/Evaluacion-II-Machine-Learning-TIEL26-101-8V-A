import argparse
import os

import torch

from src.utils.config import load_config
from src.models.build_model import build_model


def export_to_onnx(exp_dir: str, config_path: str | None = None) -> str:
    """Carga el mejor modelo entrenado de un experimento y lo exporta a ONNX.

    exp_dir debe apuntar a una carpeta tipo experiments/exp_YYYYMMDD_HHMMSS
    que contenga best_model.pt y, preferentemente, config_used.json.
    """
    if not os.path.isdir(exp_dir):
        raise FileNotFoundError(f"Directorio de experimento no encontrado: {exp_dir}")

    best_weights = os.path.join(exp_dir, "best_model.pt")
    if not os.path.isfile(best_weights):
        raise FileNotFoundError(f"No se encontró best_model.pt en {exp_dir}")

    # Preferir el config usado en el entrenamiento si existe
    used_cfg_path = os.path.join(exp_dir, "config_used.json")
    if os.path.isfile(used_cfg_path):
        cfg = load_config(used_cfg_path)
    else:
        if config_path is None:
            raise FileNotFoundError(
                "No se encontró config_used.json y no se proporcionó --config."
            )
        cfg = load_config(config_path)

    model_name = cfg["training"].get("model_name", "resnet18")
    image_size = int(cfg["training"].get("image_size", 224))
    finetune_strategy = cfg["training"].get("finetune_strategy", "full")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model, _ = build_model(model_name, finetune_strategy=finetune_strategy)
    model.load_state_dict(torch.load(best_weights, map_location=device))
    model.to(device)
    model.eval()

    dummy_input = torch.randn(1, 3, image_size, image_size, device=device)

    onnx_path = os.path.join(exp_dir, "model.onnx")
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        input_names=["input"],
        output_names=["logits"],
        opset_version=17,
        dynamic_axes={
            "input": {0: "batch_size"},
            "logits": {0: "batch_size"},
        },
    )

    return onnx_path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp_dir", required=True, help="Directorio del experimento con best_model.pt")
    ap.add_argument("--config", default="config.yaml", help="Ruta al config base (fallback)")
    args = ap.parse_args()

    onnx_path = export_to_onnx(args.exp_dir, args.config)
    print(f"Modelo exportado a ONNX en: {onnx_path}")


if __name__ == "__main__":
    main()
